"""
PASO 3 — Modelos: ¿quién convierte? (Gini) y ¿por cuánto? (R²)

Orden de trabajo, el mismo que en la industria:
  1. Alarma de fuga: ¿alguna variable SOLA predice demasiado bien? -> sospechosa.
  2. Baseline: regresión logística. Es el piso; si el modelo complejo no la supera, algo anda mal.
  3. Modelo principal: LightGBM (gradient boosting), el estándar para datos tabulares.
  4. Validación OUT-OF-TIME: entrenar con cortes antiguos, evaluar con un corte posterior.
  5. Modelo de monto: LightGBM sobre log(1 + monto), solo con las empresas que convirtieron.
  6. Reentrenar con todo y puntuar el corte de scoring (el "hoy").
  7. Razones por empresa: contribuciones tipo SHAP con pred_contrib de LightGBM (sin librerías extra).
"""
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from utils import cargar_config, cronometro, gini

NO_FEATURES = {"corte", "id", "y", "monto_objetivo"}

# Traducción de variables a lenguaje de negocio (para el "¿por qué?" del jurado y del ejecutivo)
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


def preparar(df: pd.DataFrame, categorias: dict | None = None):
    """Tipos limpios: booleanos a 0/1, montos en log, categóricas como 'category'."""
    X = df[[c for c in df.columns if c not in NO_FEATURES]].copy()
    if categorias is None:
        # Columnas 100% vacías (una base o columna que no vino) no aportan: se descartan.
        X = X.dropna(axis=1, how="all")
    else:
        X = X.reindex(columns=categorias["__columnas__"])  # mismas columnas que en entrenamiento
    for c in X.columns:
        if X[c].dtype == object and X[c].isna().all():
            X[c] = X[c].astype(float)
    for c in X.columns:
        if X[c].dtype == bool or str(X[c].dtype) == "boolean":
            X[c] = X[c].astype(float)
    for c in X.columns:
        if c.startswith(("monto_", "saldo_")):
            X[c] = np.log1p(X[c].clip(lower=0))
    cats = [c for c in X.columns
            if pd.api.types.is_string_dtype(X[c]) or isinstance(X[c].dtype, pd.CategoricalDtype)]
    if categorias is None:
        categorias = {c: sorted(X[c].dropna().astype(str).unique()) for c in cats}
        categorias["__columnas__"] = list(X.columns)
    for c in cats:
        X[c] = pd.Categorical(X[c].astype(str), categories=categorias[c])
    return X, cats, categorias


def alarma_fuga(X, y, umbral):
    """Gini de cada variable por separado. Muy alto = probablemente 've el futuro'."""
    filas = []
    for c in X.columns:
        col = X[c]
        if isinstance(col.dtype, pd.CategoricalDtype):
            col = col.astype(str).map(pd.Series(y.values, index=X.index).groupby(col.astype(str)).mean())
        g = abs(gini(y, col.astype(float).fillna(col.astype(float).median())))
        filas.append((c, g))
    res = pd.DataFrame(filas, columns=["variable", "gini_individual"]).sort_values("gini_individual", ascending=False)
    sospechosas = res[res.gini_individual > umbral]
    if len(sospechosas):
        print("  ALERTA DE FUGA: estas variables predicen demasiado solas:")
        print(sospechosas.to_string(index=False))
    return res


def baseline_logistica(cats, nums):
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("esc", StandardScaler())]), nums),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cats),
    ])
    return Pipeline([("pre", pre), ("lr", LogisticRegression(max_iter=2000))])


def clasificador(semilla, n_estimators=2000):
    # Regularizado a propósito: hojas pequeñas, mínimo de 100 empresas por hoja y penalización L2.
    # Con pocas conversiones (cientos) un árbol grande MEMORIZA: Gini 1.0 en train y bajo en validación.
    return lgb.LGBMClassifier(n_estimators=n_estimators, learning_rate=0.03, num_leaves=15,
                              min_child_samples=100, reg_lambda=5.0, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.7, random_state=semilla, verbose=-1)


def regresor(semilla):
    return lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=20,
                             subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                             random_state=semilla, verbose=-1)


def ajustar_lgbm(tr, Xtr, ytr, semilla):
    """Early stopping SIN tocar la validación: el último corte de entrenamiento hace de 'mini validación'.
    Así elegimos cuántos árboles usar y la validación out-of-time sigue siendo un examen limpio."""
    cortes = sorted(tr["corte"].unique())
    n_arboles = 300
    if len(cortes) >= 2:
        interno = (tr["corte"] != cortes[-1]).values
        m = clasificador(semilla).fit(Xtr[interno], ytr[interno], eval_set=[(Xtr[~interno], ytr[~interno])],
                                      eval_metric="auc", callbacks=[lgb.early_stopping(100, verbose=False)])
        n_arboles = max(int(m.best_iteration_ or 300), 50)
    return clasificador(semilla, n_arboles).fit(Xtr, ytr), n_arboles


def razones_lgbm(modelo, X):
    """Contribución de cada variable a la predicción de cada empresa (valores SHAP de LightGBM)."""
    contrib = modelo.booster_.predict(X, pred_contrib=True)[:, :-1]  # la última columna es el sesgo
    return contrib, list(X.columns)


def razones_logistica(modelo, X, cats):
    """En la logística, la contribución es coeficiente × valor estandarizado (sumando las dummies)."""
    pre, lr = modelo.named_steps["pre"], modelo.named_steps["lr"]
    Z = pre.transform(X)
    Z = Z.toarray() if hasattr(Z, "toarray") else Z
    contrib_dummy = Z * lr.coef_[0]
    nombres_dummy = pre.get_feature_names_out()
    cols = list(X.columns)
    contrib = np.zeros((len(X), len(cols)))
    for j, nd in enumerate(nombres_dummy):
        bloque, nombre = nd.split("__", 1)
        if bloque == "num":
            contrib[:, cols.index(nombre)] += contrib_dummy[:, j]
        else:
            original = next(c for c in cats if nombre.startswith(c + "_"))
            contrib[:, cols.index(original)] += contrib_dummy[:, j]
    return contrib, cols


def texto_razones(contrib, nombres, k=3):
    """Las k variables que más EMPUJAN hacia arriba el puntaje de cada empresa, en lenguaje de negocio."""
    top = np.argsort(-contrib, axis=1)[:, :k]
    return ["; ".join(FRASES.get(nombres[j], nombres[j]) for j in fila if contrib[i, j] > 0)
            for i, fila in enumerate(top)]


def main():
    cfg = cargar_config()
    sal = cfg["rutas"]["salidas"]
    semilla = cfg["modelo"]["semilla"]
    df = pd.read_parquet(sal / "tabla_modelo.parquet")
    score_df = pd.read_parquet(sal / "tabla_scoring.parquet")
    corte_val = cfg["tiempo"]["corte_validacion"]

    tr, va = df[df.corte != corte_val], df[df.corte == corte_val]
    Xtr, cats, categorias = preparar(tr)
    Xva, _, _ = preparar(va, categorias)
    ytr, yva = tr["y"], va["y"]
    nums = [c for c in Xtr.columns if c not in cats]
    print(f"  entrenamiento: {len(tr):,} filas ({ytr.sum():,} conversiones) | validación: {len(va):,} ({yva.sum():,})")

    with cronometro("alarma de fuga"):
        fuga = alarma_fuga(Xtr, ytr, cfg["modelo"]["alerta_fuga_gini"])

    with cronometro("baseline logística"):
        base = baseline_logistica(cats, nums).fit(Xtr, ytr)
        p_base_tr, p_base_va = base.predict_proba(Xtr)[:, 1], base.predict_proba(Xva)[:, 1]

    with cronometro("LightGBM clasificador"):
        clf, n_arboles = ajustar_lgbm(tr, Xtr, ytr, semilla)
        p_tr, p_va = clf.predict_proba(Xtr)[:, 1], clf.predict_proba(Xva)[:, 1]

    m = {
        "gini_logistica_train": gini(ytr, p_base_tr), "gini_logistica_valid": gini(yva, p_base_va),
        "gini_lgbm_train": gini(ytr, p_tr), "gini_lgbm_valid": gini(yva, p_va),
        "arboles_lgbm": n_arboles,
    }
    ganador = "lgbm" if m["gini_lgbm_valid"] >= m["gini_logistica_valid"] else "logistica"
    p_gan = p_va if ganador == "lgbm" else p_base_va
    m["modelo_ganador"] = ganador
    m["gini_ganador_valid"] = gini(yva, p_gan)
    m["tasa_real_valid"], m["prob_media_valid"] = float(yva.mean()), float(p_gan.mean())
    print(f"  Gini logística: train {m['gini_logistica_train']:.3f} | valid {m['gini_logistica_valid']:.3f}")
    print(f"  Gini LightGBM:  train {m['gini_lgbm_train']:.3f} | valid {m['gini_lgbm_valid']:.3f} ({n_arboles} árboles)")
    brecha = m["gini_lgbm_train"] - m["gini_lgbm_valid"]
    if brecha > 0.25:
        print(f"  AVISO: brecha train-valid de {brecha:.2f} en LightGBM = sobreajuste; regularizar más")
    print(f"  ganador en validación: {ganador.upper()}")
    print(f"  calibración: tasa real {m['tasa_real_valid']:.2%} vs probabilidad media {m['prob_media_valid']:.2%}")

    # estabilidad: Gini por segmento en validación (del ganador)
    por_seg = va.assign(p=p_gan).groupby("segmento").apply(lambda g: gini(g["y"], g["p"]), include_groups=False)

    # ---- modelo de monto (solo positivos), en escala log ----
    with cronometro("LightGBM monto"):
        pos_tr, pos_va = tr["y"] == 1, va["y"] == 1
        reg = regresor(semilla).fit(Xtr[pos_tr], np.log1p(tr.loc[pos_tr, "monto_objetivo"]))
        log_pred_va = reg.predict(Xva)
        m["r2_monto_log_valid"] = float(r2_score(np.log1p(va.loc[pos_va, "monto_objetivo"]), log_pred_va[pos_va.values]))
    print(f"  R² monto (log) en validación: {m['r2_monto_log_valid']:.3f}")

    imp = (pd.DataFrame({"variable": Xtr.columns, "ganancia": clf.booster_.feature_importance("gain")})
           .sort_values("ganancia", ascending=False))
    imp["ganancia_%"] = 100 * imp.ganancia / imp.ganancia.sum()

    pd.DataFrame({
        "id": va["id"].values, "y": yva.values, "monto_objetivo": va["monto_objetivo"].values,
        "p_logistica": p_base_va, "p_lgbm": p_va, "p": p_gan, "monto_estimado": np.expm1(log_pred_va),
        "tiene_lead": va["tiene_lead"].values, "rubro": va["rubro"].values,
        "region": va["region"].values, "segmento": va["segmento"].values,
    }).to_parquet(sal / "predicciones_validacion.parquet", index=False)

    # ---- reentrenar el ganador con TODO (train + valid) y puntuar el "hoy" ----
    with cronometro("reentrenar y puntuar"):
        Xall, _, _ = preparar(df, categorias)
        Xs, _, _ = preparar(score_df, categorias)
        Xs = Xs[Xall.columns]
        if ganador == "lgbm":
            final = clasificador(semilla, n_arboles).fit(Xall, df["y"])
            contrib, nombres = razones_lgbm(final, Xs)
        else:
            final = baseline_logistica(cats, nums).fit(Xall, df["y"])
            contrib, nombres = razones_logistica(final, Xs, cats)
        pos = df["y"] == 1
        reg_f = regresor(semilla).fit(Xall[pos], np.log1p(df.loc[pos, "monto_objetivo"]))
        scores = score_df[["id", "rubro", "segmento", "region", "tiene_lead", "es_cliente"]].copy()
        scores["p"] = final.predict_proba(Xs)[:, 1]
        scores["monto_estimado"] = np.expm1(reg_f.predict(Xs))
        scores["razones"] = texto_razones(contrib, nombres)
        scores.to_parquet(sal / "scores.parquet", index=False)

    (sal / "metricas.json").write_text(json.dumps(m, indent=2, default=str), encoding="utf-8")
    rep = ["# Reporte del modelo\n", "| Métrica | Valor |", "|---|---|"]
    rep += [f"| {k} | {v:.3f} |" if isinstance(v, float) else f"| {k} | {v} |" for k, v in m.items()]
    rep += ["\n## Gini por segmento (validación, modelo ganador)\n", por_seg.round(3).to_frame("gini").to_markdown()]
    rep += ["\n## Importancia de variables (LightGBM, % de ganancia)\n",
            imp[["variable", "ganancia_%"]].head(15).round(1).to_markdown(index=False)]
    rep += ["\n## Alarma de fuga: Gini de cada variable sola (entrenamiento)\n",
            fuga.head(10).round(3).to_markdown(index=False)]
    (sal / "reporte_modelo.md").write_text("\n".join(rep), encoding="utf-8")
    print("  top 5 variables:", ", ".join(imp.variable.head(5)))


if __name__ == "__main__":
    main()
