"""
PASO 4 — Del modelo al negocio: ¿cuánto?, ¿quiénes?, ¿por qué?  + OPTIMIZACIÓN con restricciones

  1. Valor esperado por fila:  VE = P(convierte) × monto estimado × margen
  2. Curva de ganancia en validación (resultados reales): % de conversiones y de monto captado
     contactando al top X%, contra el azar y contra los leads actuales.
  3. Selección óptima bajo restricciones (programación lineal con scipy + HiGHS, y redondeo):
         maximizar   Σ VE_i · x_i                     (x_i = 1 si contactamos a la fila i)
         sujeto a    Σ x_i ≤ capacidad                (fuerza comercial)
                     Σ monto_i · x_i ≤ presupuesto    (opcional)
                     restricciones por grupo           (ej. máx. 600 contactos en LIMA, mín. 100 en AGRO)
                     una sola opción por id            (si un id aparece varias veces, ej. producto x canal)
     Sin restricciones extra, la solución óptima es simplemente ordenar por VE y cortar en la capacidad.
     Además se calcula cuánto ingreso "cuesta" cada restricción: un dato que el jurado de negocio entiende.
  4. Habilitar: filas SIN lead cuyo VE ≥ el de la fila mediana CON lead -> factor de ampliación (2x, 3x...).
  5. Perfil de la lista vs leads actuales -> candidatos a hallazgo.
"""
import json

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix

from utils import cargar_config, cronometro


def curva(df, orden, valor="monto_objetivo"):
    d = df.iloc[orden]
    n, tot_y, tot_m = len(d), max(d["y"].sum(), 1), max(d[valor].fillna(0).sum(), 1e-9)
    y_acum, m_acum = d["y"].cumsum().values, d[valor].fillna(0).cumsum().values
    filas = []
    for pct in range(5, 101, 5):
        k = int(np.ceil(n * pct / 100)) - 1
        filas.append({"pct_contactado": pct, "pct_conversiones": 100 * y_acum[k] / tot_y,
                      "pct_monto": 100 * m_acum[k] / tot_m})
    return pd.DataFrame(filas)


def candidatos(s, cap, restricciones):
    """Reduce el problema: solo pueden entrar las mejores filas (de todo y de cada grupo restringido).
    Con millones de filas esto baja el tiempo de segundos-minutos a milisegundos sin cambiar la solución."""
    keep = set(s.nlargest(min(len(s), 10 * cap), "valor_esperado").index)
    for r in restricciones:
        g = s[s[r["columna"]].astype(str) == str(r["valor"])]
        n = int(10 * r["limite"]) if r["medida"] == "contactos" else 10 * cap
        keep |= set(g.nlargest(min(len(g), n), "valor_esperado").index)
    return s.loc[sorted(keep)]


def resolver(s, cap, restricciones, presupuesto, una_por_id, limite_s=60):
    """Programación lineal + redondeo.
    Por qué no un solver entero directo: con pocas restricciones, la relajación lineal (LP) casi siempre
    sale ya entera, y entonces ES el óptimo. Medido aquí: LP 1.3 s vs entero > 60 s sin solución.
    Si quedan fracciones, se redondea hacia abajo y se completa con las mejores filas que no rompan nada.
    Se reporta la distancia al límite teórico (cota LP): si es 0.0%, la solución es óptima demostrada.
    Devuelve (máscara, valor total, estado)."""
    n = len(s)
    ve = s["valor_esperado"].values
    filas, lo, hi = [np.ones(n)], [0], [cap]
    if presupuesto:
        filas.append(s["monto_esperado"].values); lo.append(0); hi.append(presupuesto)
    for r in restricciones:
        en = (s[r["columna"]].astype(str) == str(r["valor"])).values.astype(float)
        coef = en if r["medida"] == "contactos" else en * s["monto_esperado"].values
        filas.append(coef)
        if r["sentido"] == "max":
            lo.append(0); hi.append(r["limite"])
        else:
            lo.append(r["limite"]); hi.append(np.inf)
    A = csr_matrix(np.vstack(filas))
    lo, hi = np.array(lo, float), np.array(hi, float)
    restr = [LinearConstraint(A, lo, hi)]
    codigos = None
    if una_por_id and s["id"].duplicated().any():
        codigos, _ = pd.factorize(s["id"])
        G = csr_matrix((np.ones(n), (codigos, np.arange(n))), shape=(codigos.max() + 1, n))
        restr.append(LinearConstraint(G, 0, 1))
    res = milp(-ve, constraints=restr, integrality=np.zeros(n), bounds=Bounds(0, 1),
               options={"time_limit": limite_s})
    if res.x is None:
        return None, 0.0, f"sin solución: {res.message}"
    cota = -res.fun
    x = res.x > 1 - 1e-6
    n_frac = int(((res.x > 1e-6) & (res.x < 1 - 1e-6)).sum())
    if n_frac:  # completar con las mejores filas que respeten todos los topes
        uso = A @ x.astype(float)
        usado_id = set(codigos[x]) if codigos is not None else set()
        Ad = A.toarray()
        for i in np.argsort(-ve):
            if x[i] or ve[i] <= 0 or (codigos is not None and codigos[i] in usado_id):
                continue
            if np.all(uso + Ad[:, i] <= hi + 1e-9):
                x[i] = True
                uso += Ad[:, i]
                if codigos is not None:
                    usado_id.add(codigos[i])
        if np.any(uso < lo - 1e-9):
            return None, 0.0, "el redondeo no logró cumplir los mínimos"
    total = float(ve[x].sum())
    brecha = 100 * (cota - total) / max(cota, 1e-12)
    return x, total, f"programación lineal ({n_frac} fracciones redondeadas; a {brecha:.2f}% del óptimo teórico)"


def main():
    cfg = cargar_config()
    sal = cfg["rutas"]["salidas"]
    margen = cfg["negocio"]["margen"]
    cap = int(cfg["negocio"]["capacidad_contactos"])
    opt = cfg.get("optimizacion") or {}
    restricciones = opt.get("restricciones") or []
    lead = cfg.get("columna_lead")
    rng = np.random.default_rng(cfg["modelo"]["semilla"])
    met = json.loads((sal / "metricas.json").read_text(encoding="utf-8"))["global"]
    hay_monto = met.get("hay_monto", True)
    unidad = "ingreso esperado (S/)" if hay_monto else "conversiones esperadas"

    # ================= Validación con resultados reales =================
    val = pd.read_parquet(sal / "predicciones_validacion.parquet")
    hay_lead_val = bool(lead) and lead in val.columns
    if not val["monto_objetivo"].notna().any():
        val["monto_objetivo"] = val["y"].astype(float)
    val["ve"] = val["p"] * val["monto_estimado"]
    estrategias = {"Modelo: valor esperado": np.argsort(-val["ve"].values),
                   "Modelo: probabilidad": np.argsort(-val["p"].values)}
    if hay_lead_val:
        estrategias["Leads actuales primero"] = np.lexsort((rng.random(len(val)), ~val[lead].astype(bool).values))
    estrategias["Azar"] = rng.permutation(len(val))
    curvas = pd.concat([curva(val, o).assign(estrategia=e) for e, o in estrategias.items()])
    curvas.to_csv(sal / "curva_ganancia.csv", index=False)

    # ================= La lista de "hoy" =================
    s = pd.read_parquet(sal / "scores.parquet").reset_index(drop=True)
    perfil_cols = [c for c in (cfg.get("perfil_columnas") or []) if c in s.columns]
    hay_lead = bool(lead) and lead in s.columns
    s["monto_esperado"] = s["p"] * s["monto_estimado"]
    s["valor_esperado"] = s["monto_esperado"] * margen if hay_monto else s["p"]
    n_ids = s["id"].nunique()
    frac_cap = min(cap / n_ids, 1.0)
    pct_ref = max(5, int(round(frac_cap * 100 / 5)) * 5)
    tabla_val = (curvas[curvas.pct_contactado == pct_ref]
                 .set_index("estrategia")[["pct_conversiones", "pct_monto"]].round(1))

    # ---- optimización ----
    duplicados = s["id"].duplicated().any() and opt.get("una_opcion_por_id", True)
    simple = not restricciones and not opt.get("presupuesto_monto") and not duplicados
    costos = []
    with cronometro("selección óptima"):
        if simple:
            s = s.sort_values("valor_esperado", ascending=False).reset_index(drop=True)
            s["seleccionada"] = s.index < cap
            total, estado = s.loc[s.seleccionada, "valor_esperado"].sum(), "orden por VE (óptimo sin restricciones)"
        else:
            cand = candidatos(s, cap, restricciones)
            x, total, estado = resolver(cand, cap, restricciones, opt.get("presupuesto_monto"),
                                        opt.get("una_opcion_por_id", True))
            if x is None:
                print(f"  OJO: el problema no tiene solución ({estado}). ¿Restricciones mínimas imposibles? "
                      "Se usa el orden simple por VE.")
                s = s.sort_values("valor_esperado", ascending=False).reset_index(drop=True)
                s["seleccionada"] = s.index < cap
                total = s.loc[s.seleccionada, "valor_esperado"].sum()
            else:
                s["seleccionada"] = False
                s.loc[cand.index[x], "seleccionada"] = True
                # ¿cuánto cuesta cada restricción? se resuelve SIN ella (manteniendo las demás) y se compara.
                # Una restricción que cuesta 0 "no aprieta": se podría negociar sin perder nada.
                upid, pres = opt.get("una_opcion_por_id", True), opt.get("presupuesto_monto")
                tol = 1e-4 * max(abs(total), 1e-9)

                def costo(otras, p):
                    _, v, _ = resolver(candidatos(s, cap, otras), cap, otras, p, upid)
                    return v - total if v - total > tol else 0.0

                costos.append(("todas a la vez", costo([], None)))
                if pres:
                    costos.append((f"presupuesto de monto ({pres:,.0f})", costo(restricciones, None)))
                for j, r in enumerate(restricciones[:8]):
                    costos.append((f"{r['sentido']} {r['medida']} {r['columna']}={r['valor']} ({r['limite']:,})",
                                   costo(restricciones[:j] + restricciones[j + 1:], pres)))
            s = s.sort_values(["seleccionada", "valor_esperado"], ascending=False).reset_index(drop=True)
    s.insert(0, "prioridad", np.arange(1, len(s) + 1))
    sel = s[s.seleccionada]
    print(f"  {estado}: {len(sel):,} seleccionados | {unidad}: {total:,.1f}")
    for nombre, c in costos:
        print(f"    costo de '{nombre}': {c:,.1f}")

    # ================= Comparación con la práctica actual y "habilitar" =================
    comparacion, factor, n_lead, nuevos, umbral, mejora = None, None, 0, s.iloc[:0], None, None
    if hay_lead:
        leads = s[s[lead].astype(bool)].drop_duplicates("id").sample(frac=1, random_state=1)
        actual = leads.head(cap)
        comparacion = pd.DataFrame({
            "Lista del modelo": [len(sel), sel.p.sum(), sel.monto_esperado.sum(), sel.valor_esperado.sum()],
            "Leads actuales": [len(actual), actual.p.sum(), actual.monto_esperado.sum(), actual.valor_esperado.sum()],
        }, index=["contactos", "conversiones esperadas", "monto esperado (S/)", unidad])
        mejora = comparacion.loc[unidad, "Lista del modelo"] / max(comparacion.loc[unidad, "Leads actuales"], 1e-9)
        n_lead = int(s.drop_duplicates("id")[lead].astype(bool).sum())
        umbral = s.loc[s[lead].astype(bool), "valor_esperado"].median()
        nuevos = s[(~s[lead].astype(bool)) & (s.valor_esperado >= umbral)].drop_duplicates("id")
        factor = (n_lead + len(nuevos)) / max(n_lead, 1)
    else:
        actual = None

    # ================= Perfil de la lista =================
    def mezcla(df, col):
        return df[col].astype(str).value_counts(normalize=True).mul(100)

    perfil = {}
    for col in perfil_cols:
        t = pd.DataFrame({"lista_modelo_%": mezcla(sel, col), "universo_%": mezcla(s, col)})
        if actual is not None:
            t["leads_actuales_%"] = mezcla(actual, col)
        t = t.fillna(0).round(1)
        t["diferencia_vs_universo"] = t["lista_modelo_%"] - t["universo_%"]
        perfil[col] = t.sort_values("lista_modelo_%", ascending=False)
    conc = 100 * sel.valor_esperado.sum() / max(s.drop_duplicates("id").valor_esperado.sum(), 1e-9)

    # ================= Salidas =================
    cols = ["prioridad", "id"] + perfil_cols + ([lead] if hay_lead else []) + \
           [c for c in ["segmento_modelo", "p", "monto_estimado", "monto_esperado", "valor_esperado",
                        "razones", "seleccionada"] if c in s.columns]
    s[cols].to_csv(sal / "lista_priorizada.csv", index=False)

    partes = []
    if factor:
        partes.append(f"Ampliamos la base con lead {factor:.1f}x ({n_lead:,} → {n_lead + len(nuevos):,}).")
    partes.append(f"Con {cap:,} contactos ({100 * frac_cap:.0f}% del universo) concentramos el {conc:.0f}% del "
                  f"{'ingreso esperado' if hay_monto else 'total de conversiones esperadas'}")
    if perfil_cols:
        partes[-1] += f"; destacan {perfil_cols[0]} {', '.join(perfil[perfil_cols[0]].head(3).index)}."
    frase = " ".join(partes)

    md = ["# Resumen de negocio\n", f"**Frase para el jurado:** {frase}\n"]
    if mejora:
        md.append(f"Con la misma capacidad, la lista del modelo genera **{mejora:.1f} veces** el {unidad} "
                  f"de llamar a los leads actuales.\n")
        md += ["## ¿Cuánto? Misma capacidad, dos listas\n", comparacion.to_markdown(floatfmt=",.1f")]
    md += [f"\n## Prueba con resultados reales (validación), contactando al {pct_ref}% de la base\n",
           tabla_val.to_markdown()]
    md += [f"\n## Selección óptima\n", f"- Método: {estado}", f"- Seleccionados: {len(sel):,}",
           f"- {unidad}: {total:,.1f}"]
    if costos:
        md += ["\n### Lo que cuesta cada restricción\n",
               pd.DataFrame(costos, columns=["restricción", f"{unidad} que se deja de ganar"])
               .to_markdown(index=False, floatfmt=",.1f")]
    if factor:
        md += ["\n## Habilitar\n", f"- Con lead hoy: {n_lead:,}",
               f"- Umbral: valor esperado de la fila mediana con lead = {umbral:,.2f}",
               f"- Sin lead que superan el umbral: {len(nuevos):,}", f"- Factor de ampliación: **{factor:.1f}x**"]
    for col, t in perfil.items():
        md += [f"\n## Perfil por {col}\n", t.to_markdown()]
        md.append(f"\nCandidato a hallazgo: **{t['diferencia_vs_universo'].idxmax()}** pesa más en la lista "
                  f"que en el universo.")
    md += ["\n## ¿Por qué? Razones de las 10 primeras\n",
           s.head(10)[[c for c in ["prioridad", "id", "p", "monto_estimado", "razones"] if c in s.columns]]
           .round(3).to_markdown(index=False)]
    (sal / "resumen_negocio.md").write_text("\n".join(md), encoding="utf-8")

    resumen = {"frase": frase, "factor_habilitar": factor, "n_lead": n_lead, "n_nuevos": len(nuevos),
               "capacidad": cap, "mejora_vs_leads": mejora, "concentracion_pct": conc,
               "hay_monto": hay_monto, "perfil_columnas": perfil_cols, "columna_lead": lead if hay_lead else None,
               "costos_restricciones": costos}
    (sal / "resumen_negocio.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False, default=float),
                                              encoding="utf-8")
    print("  " + frase)
    if mejora:
        print(f"  Mejora vs leads actuales con la misma capacidad: {mejora:.1f}x")
    print(tabla_val.to_string())


if __name__ == "__main__":
    main()
