"""
PASO 4 — Del modelo al negocio: ¿cuánto?, ¿quiénes?, ¿por qué?

  1. Valor esperado por empresa:  VE = P(convierte) × monto estimado × margen
  2. Curva de ganancia (en validación, con resultados reales): ¿qué % de conversiones y de monto
     capturamos contactando al top X%? Comparado con el azar y con los leads actuales del banco.
  3. Lista priorizada bajo la capacidad comercial (corte de scoring = "hoy").
  4. Habilitar: empresas SIN lead cuyo VE es al menos el de la empresa mediana que hoy SÍ tiene lead.
     -> factor de ampliación de la base (2x, 3x...).
  5. Perfil de la lista (rubros, regiones) vs leads actuales -> candidatos a "hallazgo inesperado".
"""
import json

import numpy as np
import pandas as pd

from utils import cargar_config


def curva(df, orden, valor="monto_objetivo"):
    """% acumulado de conversiones y de monto real al contactar en el orden dado."""
    d = df.loc[orden]
    n = len(d)
    filas = []
    for pct in range(5, 101, 5):
        k = int(np.ceil(n * pct / 100))
        top = d.iloc[:k]
        filas.append({"pct_contactado": pct,
                      "pct_conversiones": 100 * top["y"].sum() / max(d["y"].sum(), 1),
                      "pct_monto": 100 * top[valor].fillna(0).sum() / max(d[valor].fillna(0).sum(), 1)})
    return pd.DataFrame(filas)


def main():
    cfg = cargar_config()
    sal = cfg["rutas"]["salidas"]
    margen = cfg["negocio"]["margen"]
    cap = cfg["negocio"]["capacidad_contactos"]
    rng = np.random.default_rng(cfg["modelo"]["semilla"])

    # ================= Validación: ¿el orden del modelo funciona con resultados reales? =================
    val = pd.read_parquet(sal / "predicciones_validacion.parquet")
    val["ve"] = val["p"] * val["monto_estimado"]
    azar = rng.permutation(len(val))
    leads_primero = np.lexsort((rng.random(len(val)), ~val["tiene_lead"].astype(bool).values))
    estrategias = {
        "Modelo: valor esperado": np.argsort(-val["ve"].values),
        "Modelo: probabilidad": np.argsort(-val["p"].values),
        "Leads actuales primero": leads_primero,
        "Azar": azar,
    }
    curvas = pd.concat([curva(val, o).assign(estrategia=e) for e, o in estrategias.items()])
    curvas.to_csv(sal / "curva_ganancia.csv", index=False)

    scores = pd.read_parquet(sal / "scores.parquet")
    frac_cap = min(cap / len(scores), 1.0)
    pct_ref = max(5, int(round(frac_cap * 100 / 5)) * 5)  # punto de la curva más cercano a la capacidad
    tabla_val = (curvas[curvas.pct_contactado == pct_ref]
                 .set_index("estrategia")[["pct_conversiones", "pct_monto"]].round(1))

    # ================= Scoring: la lista de "hoy" =================
    s = scores.copy()
    s["valor_esperado"] = s["p"] * s["monto_estimado"] * margen  # ingreso esperado para el banco (S/)
    s["monto_esperado"] = s["p"] * s["monto_estimado"]
    s = s.sort_values("valor_esperado", ascending=False).reset_index(drop=True)
    s.insert(0, "prioridad", np.arange(1, len(s) + 1))
    s["seleccionada"] = s["prioridad"] <= cap
    sel = s[s.seleccionada]

    # práctica actual: llamar a los leads existentes (en orden aleatorio) hasta llenar la capacidad
    leads = s[s.tiene_lead.astype(bool)].sample(frac=1, random_state=1)
    actual = leads.head(cap)
    comparacion = pd.DataFrame({
        "Lista del modelo": [len(sel), sel.p.sum(), sel.monto_esperado.sum(), sel.valor_esperado.sum()],
        "Leads actuales": [len(actual), actual.p.sum(), actual.monto_esperado.sum(), actual.valor_esperado.sum()],
    }, index=["empresas contactadas", "conversiones esperadas", "monto esperado (S/)", "ingreso esperado (S/)"])

    # ================= Habilitar: ¿cuántas veces ampliamos la base con lead? =================
    n_lead = int(s.tiene_lead.astype(bool).sum())
    umbral = s.loc[s.tiene_lead.astype(bool), "valor_esperado"].median()
    nuevos = s[(~s.tiene_lead.astype(bool)) & (s.valor_esperado >= umbral)]
    factor = (n_lead + len(nuevos)) / max(n_lead, 1)

    # ================= Perfil de la lista: ¿dónde está el valor que hoy no vemos? =================
    def mezcla(df, col):
        return df[col].value_counts(normalize=True).mul(100)

    perfil = {}
    for col in ("rubro", "region", "segmento"):
        perfil[col] = pd.DataFrame({"lista_modelo_%": mezcla(sel, col), "leads_actuales_%": mezcla(actual, col),
                                    "universo_%": mezcla(s, col)}).fillna(0).round(1)
        perfil[col]["diferencia_vs_leads"] = perfil[col]["lista_modelo_%"] - perfil[col]["leads_actuales_%"]
        perfil[col] = perfil[col].sort_values("lista_modelo_%", ascending=False)

    top_rub = perfil["rubro"].head(3).index.tolist()
    mayor_dif = perfil["region"]["diferencia_vs_leads"].idxmax()
    conc = 100 * sel.valor_esperado.sum() / s.valor_esperado.sum()

    # ================= Salidas =================
    cols = ["prioridad", "id", "rubro", "segmento", "region", "tiene_lead", "p", "monto_estimado",
            "monto_esperado", "valor_esperado", "razones", "seleccionada"]
    s[cols].to_csv(sal / "lista_priorizada.csv", index=False)

    frase = (f"Ampliamos la base con lead {factor:.1f}x ({n_lead:,} → {n_lead + len(nuevos):,} empresas). "
             f"Con {cap:,} contactos ({100 * frac_cap:.0f}% del universo) concentramos el {conc:.0f}% del "
             f"ingreso esperado; los rubros principales son {', '.join(top_rub)}.")
    mejora = (comparacion.loc["ingreso esperado (S/)", "Lista del modelo"]
              / max(comparacion.loc["ingreso esperado (S/)", "Leads actuales"], 1))

    md = [
        "# Resumen de negocio\n",
        f"**Frase para el jurado:** {frase}\n",
        f"Con la misma capacidad, la lista del modelo genera **{mejora:.1f} veces** el ingreso esperado de llamar a los leads actuales.\n",
        "## ¿Cuánto? Misma capacidad, dos listas\n", comparacion.to_markdown(floatfmt=",.1f"),
        f"\n## Prueba con resultados reales (validación out-of-time), contactando al {pct_ref}% de la base\n",
        tabla_val.to_markdown(),
        "\n## Habilitar\n",
        f"- Empresas con lead hoy: {n_lead:,}",
        f"- Umbral: ingreso esperado de la empresa mediana con lead = S/ {umbral:,.1f}",
        f"- Empresas sin lead que superan el umbral: {len(nuevos):,}",
        f"- Factor de ampliación: **{factor:.1f}x**",
        "\n## ¿Quiénes? Perfil de la lista vs leads actuales\n",
        "### Por rubro\n", perfil["rubro"].to_markdown(),
        "\n### Por región\n", perfil["region"].to_markdown(),
        f"\nCandidato a hallazgo: la región **{mayor_dif}** gana más peso en la lista del modelo que en los leads actuales.",
        "\n## ¿Por qué? Razones de las 10 primeras empresas\n",
        s.head(10)[["prioridad", "id", "rubro", "p", "monto_estimado", "razones"]].round(3).to_markdown(index=False),
    ]
    (sal / "resumen_negocio.md").write_text("\n".join(md), encoding="utf-8")

    resumen = {"frase": frase, "factor_habilitar": factor, "n_lead": n_lead, "n_nuevos": len(nuevos),
               "capacidad": cap, "mejora_vs_leads": mejora, "concentracion_pct": conc}
    (sal / "resumen_negocio.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    print("  " + frase)
    print(f"  Mejora vs leads actuales con la misma capacidad: {mejora:.1f}x")
    print(tabla_val.to_string())


if __name__ == "__main__":
    main()
