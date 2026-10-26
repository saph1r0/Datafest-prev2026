"""
PASO 3 — Modelos: ¿quién convierte? (clasificación, Gini) y ¿por cuánto? (regresión, R²/RMSE)

Orden de trabajo (el que usan los equipos de analytics y el que recomendó el finalista de 2025):
  1. Cargar con cuidado del costo computacional: DuckDB lee solo lo necesario, números en float32,
     y opcionalmente se guarda solo una fracción de los "no" (muestreo de negativos).
  2. Selección de variables: fuera casi vacías, constantes, IDs de texto; luego un LightGBM rápido
     ordena por importancia, se quedan las mejores y se eliminan copias casi idénticas (correlación > 0.98).
  3. Alarma de fuga: una variable que SOLA predice demasiado bien se excluye y se reporta.
  4. Baseline: regresión logística (necesita rellenar vacíos; LightGBM no).
  5. LightGBM: hiperparámetros con Optuna (búsqueda bayesiana TPE), castigando el sobreajuste.
  6. Elección del ganador por Gini de validación, penalizando si la brecha train-valid pasa de 5 puntos.
  7. Modelo de monto (regresión) sobre log(1 + monto), solo con los positivos.
  8. Reentrenar con todo y puntuar; razones por fila (contribuciones SHAP de LightGBM).
  Opcional: un modelo por segmento (segmentar_por), como los 6 modelos producto x canal de 2025.
"""
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from utils import cargar_config, conectar, cronometro, gini

NO_FEATURES = {"corte", "id", "y", "monto_objetivo", "particion"}

# Traducción de variables a lenguaje de negocio (para el "¿por qué?" del jurado y del ejecutivo).
# El día del evento agreguen aquí las variables reales más importantes.
FRASES = {
    "contagio_ponderado": "sus pagadores ya tienen otros proveedores con factoring",
    "contagio_max": "uno de sus pagadores ya trabaja factoring con otros proveedores",
    "n_pagadores_con_factoring": "varios de sus pagadores ya ven factoring",
    "n_ops_otro_banco_12m": "ya hace factoring en otro banco",
    "monto_otro_banco_12m": "adelanta montos importantes en otro banco",
    "dias_desde_ult_otro_banco": "usó factoring en otro banco hace poco",
    "share_plazo_90": "gran parte de sus ventas se cobra a 90 días o más",
    "plazo_prom_ponderado": "sus clientes le pagan a plazos largos",
    "share_ventas_grandes": "vende sobre todo a empresas grandes o al Estado",
    "monto_cobrado_12m": "factura volúmenes altos a crédito",
    "monto_cobrado_6m": "factura volúmenes altos a crédito",
    "monto_cobrado_3m": "facturó mucho a crédito en el último trimestre",
    "cambio_saldo_3m": "su saldo en cuenta viene cayendo",
    "tendencia_3m_vs_previo": "sus ventas a crédito están creciendo",
    "rubro": "su rubro vende a crédito con frecuencia",
    "segmento": "su segmento suele usar factoring",
    "es_cliente": "ya es cliente del banco",
    "tiene_lead": "ya tiene un lead comercial",
}


# ============================ CARGA ============================
def cargar(cfg, ruta, donde="TRUE", columnas=None, muestra_neg=1.0, max_filas=None):
    """Lee con DuckDB: filtra, muestrea y convierte DOUBLE -> FLOAT antes de llegar a pandas (mitad de RAM)."""
    con = conectar(cfg)
    tipos = {r[0]: r[1] for r in con.execute(f"DESCRIBE SELECT * FROM '{ruta}'").fetchall()}
    cols = [c for c in (columnas or tipos) if c in tipos]
    sel = ", ".join(f'CAST("{c}" AS FLOAT) AS "{c}"' if tipos[c] == "DOUBLE" and c not in ("monto_objetivo",)
                    else f'"{c}"' for c in cols)
    filtro = donde
    if muestra_neg < 1.0:
        # muestreo determinista por id (hash): las 5 laptops obtienen exactamente la misma muestra
        filtro += f" AND (y = 1 OR hash(CAST(id AS VARCHAR) || 'neg') % 10000 < {int(muestra_neg * 10000)})"
    sql = f"SELECT {sel} FROM '{ruta}' WHERE {filtro}"
    if max_filas:
        sql = f"SELECT * FROM ({sql}) USING SAMPLE reservoir({int(max_filas)} ROWS) REPEATABLE (42)"
    df = con.execute(sql).df()
    # tipos "nulables" de pandas (Int64, Float64, boolean) -> numpy: LightGBM y sklearn no los aceptan
    for c in df.columns:
        if isinstance(df[c].dtype, pd.api.extensions.ExtensionDtype) and pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].astype("float64" if c == "monto_objetivo" else "float32")
        elif str(df[c].dtype) == "boolean":
            df[c] = df[c].astype("float32")
    if "y" in df.columns and df["y"].notna().all():
        df["y"] = df["y"].astype(int)
    return df


def corregir(p, k):
    """Si entrenamos con menos negativos (o pesamos más a los positivos), el modelo infla las
    probabilidades por un factor k en las 'odds'. Se deshace: p_real = p / (p + k (1 - p))."""
    return p if k == 1 else p / (p + k * (1 - p))


# ============================ PREPARACIÓN ============================
def preparar(df, cfg, meta=None):
    """Tipos limpios. 'meta' guarda columnas y categorías de entrenamiento para repetirlas igual al puntuar."""
    prohib = set(cfg["modelo"].get("columnas_prohibidas") or [])
    if meta is None:
        cols = [c for c in df.columns if c not in NO_FEATURES and c not in prohib]
    else:
        cols = meta["columnas"]
    X = df.reindex(columns=cols).copy()
    for c in cfg["modelo"].get("forzar_categoricas") or []:
        if c in X:
            X[c] = X[c].astype("string")
    for c in X.columns:
        if X[c].dtype == bool or str(X[c].dtype) == "boolean":
            X[c] = X[c].astype("float32")
        elif X[c].dtype == object and X[c].isna().all():
            X[c] = X[c].astype("float32")
    for c in X.columns:  # montos muy sesgados: log con signo (ayuda a la logística; a los árboles no les afecta)
        if c.startswith(("monto_", "saldo_")) and pd.api.types.is_numeric_dtype(X[c]):
            X[c] = (np.sign(X[c]) * np.log1p(np.abs(X[c]))).astype("float32")
    cats = [c for c in X.columns
            if pd.api.types.is_string_dtype(X[c]) or isinstance(X[c].dtype, pd.CategoricalDtype)]
    if meta is None:
        meta = {"columnas": list(X.columns),
                "categorias": {c: sorted(X[c].dropna().astype(str).unique()) for c in cats}}
    for c in cats:
        X[c] = pd.Categorical(X[c].astype(str), categories=meta["categorias"].get(c, []))
    for c in X.columns:
        if c not in cats and X[c].dtype == np.float64:
            X[c] = X[c].astype("float32")
    return X, cats, meta


def seleccionar(X, y, cfg, semilla):
    """Selección de variables pensada para cientos de columnas.
    Por qué así y no PCA: PCA mezcla variables y se pierde el '¿por qué?' que pide el jurado.
    Lasso es buena idea para la logística; aquí usamos la importancia de un LightGBM rápido
    porque capta relaciones no lineales y funciona con vacíos."""
    m = cfg["modelo"]
    fuera = {}
    for c in X.columns:
        nulos = X[c].isna().mean()
        if nulos > m.get("max_nulos", 0.98):
            fuera[c] = f"{nulos:.0%} vacía"
        elif X[c].nunique(dropna=True) <= 1:
            fuera[c] = "constante"
        elif isinstance(X[c].dtype, pd.CategoricalDtype) and X[c].nunique() > 200:
            fuera[c] = "texto con demasiados valores (¿un ID?)"
    vivas = [c for c in X.columns if c not in fuera]
    tope = m.get("max_variables", 60)
    if len(vivas) <= tope:
        return vivas, fuera
    n = min(len(X), 300_000)
    idx = np.random.default_rng(semilla).choice(len(X), n, replace=False)
    Xs, ys = X.iloc[idx][vivas], y.iloc[idx]
    rapido = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.1, num_leaves=31, min_child_samples=100,
                                colsample_bytree=0.5, subsample=0.7, subsample_freq=1,
                                random_state=semilla, verbose=-1).fit(Xs, ys)
    imp = pd.Series(rapido.booster_.feature_importance("gain"), index=vivas).sort_values(ascending=False)
    # quitar copias casi idénticas: se recorre de más a menos importante
    nums = [c for c in imp.index if not isinstance(X[c].dtype, pd.CategoricalDtype)]
    muestra = Xs[nums].iloc[:50_000].astype("float64")
    corr = muestra.corr().abs()
    elegidas = []
    for c in imp.index:
        if imp[c] <= 0:
            fuera[c] = "importancia cero"
            continue
        if c in corr.index and any(corr.loc[c, e] > 0.98 for e in elegidas if e in corr.columns):
            fuera[c] = "copia de otra variable (correlación > 0.98)"
            continue
        if len(elegidas) < tope:
            elegidas.append(c)
        else:
            fuera[c] = "poca importancia"
    return elegidas, fuera


def alarma_fuga(X, y, umbral):
    """Gini de cada variable por separado. Muy alto = probablemente 've el futuro'."""
    filas = []
    for c in X.columns:
        col = X[c]
        if isinstance(col.dtype, pd.CategoricalDtype):
            col = col.astype(str).map(pd.Series(y.values, index=X.index).groupby(col.astype(str)).mean())
        col = col.astype(float)
        filas.append((c, abs(gini(y, col.fillna(col.median())))))
    res = pd.DataFrame(filas, columns=["variable", "gini_individual"]).sort_values("gini_individual", ascending=False)
    return res, res[res.gini_individual > umbral]["variable"].tolist()


# ============================ MODELOS ============================
def logistica(cats, nums):
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("esc", StandardScaler())]), nums),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20), cats),
    ])
    return Pipeline([("pre", pre), ("lr", LogisticRegression(max_iter=2000))])


PARAMS_BASE = dict(learning_rate=0.03, num_leaves=15, min_child_samples=100, reg_lambda=5.0,
                   subsample=0.8, subsample_freq=1, colsample_bytree=0.7)


def lgbm(semilla, params, n_estimators=2000, peso_clases=False):
    return lgb.LGBMClassifier(n_estimators=n_estimators, random_state=semilla, verbose=-1,
                              is_unbalance=peso_clases, **params)


def particion_interna(tr, semilla):
    """Para elegir hiperparámetros y número de árboles SIN tocar la validación final.
    Con varios periodos: el último de train hace de mini-validación (respeta el tiempo). Si no, 20% al azar."""
    if "corte" in tr.columns and tr["corte"].nunique() >= 2:
        return (tr["corte"] != tr["corte"].max()).values
    return np.random.default_rng(semilla).random(len(tr)) >= 0.2


def ajustar_lgbm(Xtr, ytr, interno, cfg, semilla, etiqueta):
    """Optuna busca hiperparámetros maximizando:  Gini interno - castigo por brecha mayor al tope.
    Así no elegimos un modelo que solo memoriza (el error más común según el finalista de 2025)."""
    m = cfg["modelo"]
    tope, peso = m.get("max_brecha_gini", 0.05), bool(m.get("peso_clases"))
    Xa, ya, Xb, yb = Xtr[interno], ytr[interno], Xtr[~interno], ytr[~interno]

    def probar(params):
        mod = lgbm(semilla, params, peso_clases=peso).fit(
            Xa, ya, eval_set=[(Xb, yb)], eval_metric="auc", callbacks=[lgb.early_stopping(100, verbose=False)])
        it = max(int(mod.best_iteration_ or 100), 30)
        g_a = gini(ya, mod.predict_proba(Xa, num_iteration=it)[:, 1])
        g_b = gini(yb, mod.predict_proba(Xb, num_iteration=it)[:, 1])
        return g_b - max(0.0, (g_a - g_b) - tope), it, g_a, g_b

    puntaje, mejor_it, g_a, g_b = probar(PARAMS_BASE)
    mejores = dict(PARAMS_BASE)
    n_trials = int(m.get("optuna_trials", 0) or 0)
    if n_trials > 0:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objetivo(trial):
            params = dict(
                learning_rate=trial.suggest_float("learning_rate", 0.02, 0.1, log=True),
                num_leaves=trial.suggest_int("num_leaves", 7, 63, log=True),
                min_child_samples=trial.suggest_int("min_child_samples", 20, 1000, log=True),
                reg_lambda=trial.suggest_float("reg_lambda", 1e-2, 50, log=True),
                subsample=trial.suggest_float("subsample", 0.5, 1.0), subsample_freq=1,
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0),
            )
            p, it, _, _ = probar(params)
            trial.set_user_attr("it", it)
            return p

        estudio = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=semilla))
        estudio.enqueue_trial({k: v for k, v in PARAMS_BASE.items() if k != "subsample_freq"})
        t0 = time.perf_counter()
        estudio.optimize(objetivo, n_trials=n_trials, timeout=60 * m.get("optuna_minutos", 3))
        if estudio.best_value > puntaje:
            puntaje = estudio.best_value
            mejores = dict(estudio.best_params, subsample_freq=1)
            mejor_it = estudio.best_trial.user_attrs["it"]
        print(f"    [{etiqueta}] Optuna: {len(estudio.trials)} pruebas en {time.perf_counter() - t0:.0f} s, "
              f"puntaje interno {puntaje:.3f}")
    # el modelo final se entrena con TODO train y con algo más de árboles (hay ~25% más datos)
    n_arb = int(mejor_it * 1.1) + 1
    return mejores, n_arb


def regresor(semilla):
    return lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=15, min_child_samples=20,
                             subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                             random_state=semilla, verbose=-1)


def razones_lgbm(modelo, X):
    contrib = modelo.booster_.predict(X, pred_contrib=True)[:, :-1]  # última columna = sesgo
    return contrib, list(X.columns)


def razones_logistica(modelo, X, cats):
    """En la logística, contribución = coeficiente × valor estandarizado (sumando las dummies)."""
    pre, lr = modelo.named_steps["pre"], modelo.named_steps["lr"]
    Z = pre.transform(X)
    Z = Z.toarray() if hasattr(Z, "toarray") else Z
    contrib_dummy = Z * lr.coef_[0]
    cols = list(X.columns)
    contrib = np.zeros((len(X), len(cols)))
    for j, nd in enumerate(pre.get_feature_names_out()):
        bloque, nombre = nd.split("__", 1)
        original = nombre if bloque == "num" else next(c for c in cats if nombre.startswith(c + "_"))
        contrib[:, cols.index(original)] += contrib_dummy[:, j]
    return contrib, cols


def texto_razones(contrib, nombres, k=3):
    top = np.argsort(-contrib, axis=1)[:, :k]
    return ["; ".join(FRASES.get(nombres[j], nombres[j]) for j in fila if contrib[i, j] > 0)
            for i, fila in enumerate(top)]


# ============================ UN SEGMENTO DE PRINCIPIO A FIN ============================
def modelar(tr, va, sc, cfg, etiqueta, k_muestreo):
    m = cfg["modelo"]
    semilla = m["semilla"]
    met = {"segmento": etiqueta, "filas_train": len(tr), "positivos_train": int(tr.y.sum()),
           "filas_valid": len(va), "positivos_valid": int(va.y.sum())}
    print(f"  [{etiqueta}] train {len(tr):,} ({int(tr.y.sum()):,} positivos) | valid {len(va):,} ({int(va.y.sum()):,})")

    Xtr0, _, _ = preparar(tr, cfg)
    with cronometro(f"{etiqueta}: selección de variables"):
        elegidas, fuera = seleccionar(Xtr0, tr.y, cfg, semilla)
    with cronometro(f"{etiqueta}: alarma de fuga"):
        fuga, sospechosas = alarma_fuga(Xtr0[elegidas], tr.y, m["alerta_fuga_gini"])
    if sospechosas:
        print(f"    ALERTA DE FUGA: se EXCLUYEN {sospechosas} (predicen demasiado solas). Revísenlas.")
        for c in sospechosas:
            fuera[c] = "posible fuga de datos"
        elegidas = [c for c in elegidas if c not in sospechosas]
    print(f"    variables: {len(elegidas)} elegidas de {Xtr0.shape[1]}")
    del Xtr0

    sub = lambda d: d[[c for c in d.columns if c in elegidas or c in NO_FEATURES]]
    Xtr, cats, meta = preparar(sub(tr), cfg)
    Xva, _, _ = preparar(va, cfg, meta)
    nums = [c for c in Xtr.columns if c not in cats]
    ytr, yva = tr.y.values, va.y.values
    k_peso = (len(ytr) - ytr.sum()) / max(ytr.sum(), 1) if m.get("peso_clases") else 1.0
    k = k_muestreo * k_peso

    with cronometro(f"{etiqueta}: logística"):
        n_lr = min(len(Xtr), 500_000)
        base = logistica(cats, nums).fit(Xtr.iloc[:n_lr], ytr[:n_lr])
        pl_tr, pl_va = base.predict_proba(Xtr)[:, 1], base.predict_proba(Xva)[:, 1]
    with cronometro(f"{etiqueta}: LightGBM (+ Optuna)"):
        interno = particion_interna(tr, semilla)
        params, n_arb = ajustar_lgbm(Xtr, pd.Series(ytr), interno, cfg, semilla, etiqueta)
        clf = lgbm(semilla, params, n_arb, bool(m.get("peso_clases"))).fit(Xtr, ytr)
        pg_tr, pg_va = clf.predict_proba(Xtr)[:, 1], clf.predict_proba(Xva)[:, 1]

    tope = m.get("max_brecha_gini", 0.05)
    res = {}
    for nombre, a, b in [("logistica", pl_tr, pl_va), ("lgbm", pg_tr, pg_va)]:
        g_tr, g_va = gini(ytr, a), gini(yva, b)
        brecha = g_tr - g_va
        res[nombre] = g_va - max(0.0, brecha - tope)
        met.update({f"gini_{nombre}_train": g_tr, f"gini_{nombre}_valid": g_va, f"brecha_{nombre}": brecha})
        marca = "  <- SOBREAJUSTE" if brecha > tope else ""
        print(f"    Gini {nombre:<9} train {g_tr:.3f} | valid {g_va:.3f} | brecha {brecha:+.3f}{marca}")
    ganador = max(res, key=res.get)
    p_va = corregir(pg_va if ganador == "lgbm" else pl_va, k if ganador == "lgbm" else k_muestreo)
    met.update({"modelo_ganador": ganador, "arboles_lgbm": n_arb, "params_lgbm": params,
                "gini_ganador_valid": gini(yva, p_va), "tasa_real_valid": float(yva.mean()),
                "prob_media_valid": float(p_va.mean())})
    print(f"    ganador: {ganador.upper()} | calibración: real {yva.mean():.2%} vs predicho {p_va.mean():.2%}")

    # ---- monto (regresión) ----
    hay_monto = tr.loc[tr.y == 1, "monto_objetivo"].notna().sum() >= 30
    met["hay_monto"] = bool(hay_monto)
    if hay_monto:
        with cronometro(f"{etiqueta}: regresión del monto"):
            pos_tr = (tr.y == 1).values & tr.monto_objetivo.notna().values
            pos_va = (va.y == 1).values & va.monto_objetivo.notna().values
            reg = regresor(semilla).fit(Xtr[pos_tr], np.log1p(tr.monto_objetivo.values[pos_tr]))
            log_va = reg.predict(Xva)
            real = np.log1p(va.monto_objetivo.values[pos_va])
            met["r2_monto_log_valid"] = float(r2_score(real, log_va[pos_va]))
            met["rmse_monto_log_valid"] = float(np.sqrt(mean_squared_error(real, log_va[pos_va])))
            met["r2_monto_soles_valid"] = float(r2_score(np.expm1(real), np.expm1(log_va[pos_va])))
        print(f"    monto: R² (log) {met['r2_monto_log_valid']:.3f} | RMSE (log) {met['rmse_monto_log_valid']:.3f}")
        monto_va = np.expm1(log_va)
    else:
        monto_va = np.ones(len(va))

    extra = [c for c in cfg["_cols_descriptivas"] if c in va.columns]
    pred_va = pd.DataFrame({"id": va.id.values, "y": yva, "monto_objetivo": va.monto_objetivo.values,
                            "p_logistica": corregir(pl_va, k_muestreo), "p_lgbm": corregir(pg_va, k),
                            "p": p_va, "monto_estimado": monto_va, "segmento_modelo": etiqueta})
    for c in extra:
        pred_va[c] = va[c].values

    # ---- reentrenar con train + valid y puntuar ----
    with cronometro(f"{etiqueta}: reentrenar y puntuar"):
        todo = pd.concat([tr, va], ignore_index=True)
        Xall, _, _ = preparar(todo, cfg, meta)
        Xs, _, _ = preparar(sc, cfg, meta)
        if ganador == "lgbm":
            final = lgbm(semilla, params, n_arb, bool(m.get("peso_clases"))).fit(Xall, todo.y.values)
            contrib, nombres = razones_lgbm(final, Xs)
            p_s = corregir(final.predict_proba(Xs)[:, 1], k)
        else:
            n_lr = min(len(Xall), 500_000)
            final = logistica(cats, nums).fit(Xall.iloc[:n_lr], todo.y.values[:n_lr])
            contrib, nombres = razones_logistica(final, Xs, cats)
            p_s = corregir(final.predict_proba(Xs)[:, 1], k_muestreo)
        if hay_monto:
            pos = (todo.y == 1).values & todo.monto_objetivo.notna().values
            reg_f = regresor(semilla).fit(Xall[pos], np.log1p(todo.monto_objetivo.values[pos]))
            monto_s = np.expm1(reg_f.predict(Xs))
        else:
            monto_s = np.ones(len(sc))
    scores = pd.DataFrame({"id": sc.id.values, "p": p_s, "monto_estimado": monto_s,
                           "razones": texto_razones(contrib, nombres), "segmento_modelo": etiqueta})
    for c in extra:
        if c in sc.columns:
            scores[c] = sc[c].values

    imp = pd.DataFrame({"variable": Xtr.columns, "ganancia": clf.booster_.feature_importance("gain"),
                        "segmento_modelo": etiqueta})
    imp["ganancia_%"] = 100 * imp.ganancia / max(imp.ganancia.sum(), 1)
    descartes = pd.DataFrame({"variable": list(fuera), "motivo": list(fuera.values()), "segmento_modelo": etiqueta})
    return met, pred_va, scores, imp.sort_values("ganancia", ascending=False), fuga, descartes


def main():
    cfg = cargar_config()
    m, sal = cfg["modelo"], cfg["rutas"]["salidas"]
    seg = m.get("segmentar_por")
    lead = cfg.get("columna_lead")
    cfg["_cols_descriptivas"] = [c for c in (cfg.get("perfil_columnas") or []) + [lead, seg, "es_cliente"] if c]
    r = float(m.get("muestreo_negativos", 1.0))
    ruta_m, ruta_s = sal / "tabla_modelo.parquet", sal / "tabla_scoring.parquet"

    with cronometro("carga (DuckDB, float32, muestreo)"):
        tr_all = cargar(cfg, ruta_m, "particion = 'train'", muestra_neg=r)
        va_all = cargar(cfg, ruta_m, "particion = 'valid'", max_filas=m.get("max_filas_validacion"))
        sc_all = cargar(cfg, ruta_s)
    if r < 1:
        print(f"  muestreo de negativos: se usa {r:.0%} de los 'no' para entrenar; probabilidades corregidas")
    if seg and seg not in tr_all.columns:
        raise SystemExit(f"segmentar_por = '{seg}' no existe en los datos")

    grupos = sorted(tr_all[seg].dropna().unique()) if seg else ["TODOS"]
    salidas = []
    for g in grupos:
        f = (lambda d: d[d[seg] == g].reset_index(drop=True)) if seg else (lambda d: d)
        salidas.append(modelar(f(tr_all), f(va_all), f(sc_all), cfg, str(g), 1.0 / r))

    mets = [s[0] for s in salidas]
    pred_va = pd.concat([s[1] for s in salidas], ignore_index=True)
    scores = pd.concat([s[2] for s in salidas], ignore_index=True)
    imp = pd.concat([s[3] for s in salidas], ignore_index=True)
    fuga = pd.concat([s[4].assign(segmento_modelo=s[0]["segmento"]) for s in salidas], ignore_index=True)
    descartes = pd.concat([s[5] for s in salidas], ignore_index=True)

    glob = {"gini_global_valid": gini(pred_va.y, pred_va.p), "segmentos": len(grupos),
            "muestreo_negativos": r, "hay_monto": all(x["hay_monto"] for x in mets)}
    if glob["hay_monto"]:
        pos = (pred_va.y == 1) & pred_va.monto_objetivo.notna()
        glob["r2_monto_log_valid"] = float(r2_score(np.log1p(pred_va.monto_objetivo[pos]),
                                                    np.log1p(pred_va.monto_estimado[pos])))
    print(f"  Gini GLOBAL en validación: {glob['gini_global_valid']:.3f}")

    pred_va.to_parquet(sal / "predicciones_validacion.parquet", index=False)
    scores.to_parquet(sal / "scores.parquet", index=False)
    imp.to_csv(sal / "importancia.csv", index=False)
    descartes.to_csv(sal / "variables_descartadas.csv", index=False)
    (sal / "metricas.json").write_text(json.dumps({"global": glob, "por_segmento": mets}, indent=2,
                                                  default=str, ensure_ascii=False), encoding="utf-8")

    rep = ["# Reporte del modelo\n", f"Gini global en validación: **{glob['gini_global_valid']:.3f}**\n",
           "| segmento | ganador | Gini log. train | Gini log. valid | Gini LGBM train | Gini LGBM valid | brecha LGBM |",
           "|---|---|---|---|---|---|---|"]
    for x in mets:
        rep.append(f"| {x['segmento']} | {x['modelo_ganador']} | {x['gini_logistica_train']:.3f} | "
                   f"{x['gini_logistica_valid']:.3f} | {x['gini_lgbm_train']:.3f} | {x['gini_lgbm_valid']:.3f} | "
                   f"{x['brecha_lgbm']:+.3f} |")
    if glob["hay_monto"]:
        rep.append(f"\nMonto: R² (log) global en validación {glob['r2_monto_log_valid']:.3f}")
    top = (imp.groupby("variable")["ganancia_%"].mean().sort_values(ascending=False).head(15).round(1))
    rep += ["\n## Importancia de variables (LightGBM, % de ganancia promedio)\n", top.to_frame().to_markdown()]
    rep += ["\n## Alarma de fuga: Gini de cada variable sola (top 10)\n",
            fuga.sort_values("gini_individual", ascending=False).head(10).round(3).to_markdown(index=False)]
    rep += ["\n## Variables descartadas (resumen)\n",
            descartes.groupby("motivo").size().to_frame("n").to_markdown()]
    (sal / "reporte_modelo.md").write_text("\n".join(rep), encoding="utf-8")
    print("  top 5 variables:", ", ".join(top.index[:5]))


if __name__ == "__main__":
    main()
