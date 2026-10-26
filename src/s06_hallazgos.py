"""
PASO 6 — Hallazgos de negocio para la presentación.

El finalista de 2025 contó que la pregunta del jurado fue: "¿Qué fue lo más importante que encontraron?".
Al jurado de negocio le importa poco el algoritmo; quiere CARACTERÍSTICAS de los buenos clientes,
mostradas con gráficos simples. Este paso las calcula sin opinar:

  1. Tasa de conversión por grupo (rubro, región, segmento...) y cuántas veces el promedio ("lift").
  2. Para las variables más importantes del modelo: tasa de conversión por quintil (20% más bajo ... 20% más alto).
  3. Si hay monto: monto promedio de los que convirtieron, por grupo.
  4. Frases automáticas ordenadas por fuerza, para elegir 2 o 3 para el PPT.
  5. Gráficos PNG listos para pegar en las diapositivas (outputs/graficos/).

Ojo: estas son ASOCIACIONES, no causas. En la presentación digan "las empresas que X convierten más",
no "X hace que conviertan".
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from s03_modelo import cargar
from utils import cargar_config, cronometro

AZUL, GRIS, NARANJA = "#0b5cad", "#9a9a94", "#e07a1f"


def estilo(ax, titulo, sub):
    ax.set_title(titulo, loc="left", fontsize=15, fontweight="bold", pad=26)
    ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=10.5, color="#555")
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    ax.spines["left"].set_color("#bbb"); ax.spines["bottom"].set_color("#bbb")
    ax.tick_params(colors="#333")


def grafico_barras(tabla, base, titulo, archivo, horizontal=True):
    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=160)
    t = tabla.sort_values("tasa", ascending=True) if horizontal else tabla  # quintiles: orden natural
    colores = [AZUL if v >= base * 1.2 else GRIS for v in t["tasa"]]
    if horizontal:
        ax.barh(t.index.astype(str), 100 * t["tasa"], color=colores)
        ax.axvline(100 * base, color=NARANJA, ls="--", lw=1.2)
        ax.text(100 * base, len(t) - 0.4, f" promedio {100 * base:.1f}%", color=NARANJA, fontsize=10, va="bottom")
        ax.set_xlabel("% que convierte")
        for i, (v, l) in enumerate(zip(t["tasa"], t["lift"])):
            ax.text(100 * v, i, f"  {l:.1f}x", va="center", fontsize=10, color="#333")
    else:
        ax.bar(t.index.astype(str), 100 * t["tasa"], color=colores)
        ax.axhline(100 * base, color=NARANJA, ls="--", lw=1.2)
        ax.text(1.0, 100 * base, f" promedio {100 * base:.1f}%", color=NARANJA, fontsize=10,
                ha="left", va="center", transform=ax.get_yaxis_transform(), clip_on=False)
        ax.set_ylabel("% que convierte")
        for i, (v, l) in enumerate(zip(t["tasa"], t["lift"])):
            ax.text(i, 100 * v, f"{l:.1f}x", ha="center", va="bottom", fontsize=10)
    estilo(ax, titulo, "Azul: al menos 1.2 veces el promedio. Etiquetas: veces el promedio.")
    fig.tight_layout()
    fig.savefig(archivo)
    plt.close(fig)


def main():
    cfg = cargar_config()
    sal = cfg["rutas"]["salidas"]
    graf = sal / "graficos"
    graf.mkdir(exist_ok=True)
    with cronometro("carga (muestra)"):
        df = cargar(cfg, sal / "tabla_modelo.parquet", max_filas=500_000)
    base = df["y"].mean()
    met = json.loads((sal / "metricas.json").read_text(encoding="utf-8"))["global"]
    hay_monto = met.get("hay_monto", True) and df["monto_objetivo"].notna().any()
    hallazgos = []

    # ---- 1. por grupo ----
    grupos = [c for c in (cfg.get("perfil_columnas") or []) + [cfg.get("columna_lead"), cfg["modelo"].get("segmentar_por")]
              if c and c in df.columns]
    tablas_md = []
    for col in dict.fromkeys(grupos):
        t = df.groupby(df[col].astype(str))["y"].agg(tasa="mean", n="size")
        t = t[t.n >= 100]
        if len(t) < 2:
            continue
        t["lift"] = t.tasa / base
        if hay_monto:
            t["monto_prom_convertidos"] = df[df.y == 1].groupby(df[col].astype(str))["monto_objetivo"].mean()
        grafico_barras(t, base, f"Conversión por {col}", graf / f"conversion_por_{col}.png")
        tablas_md += [f"\n### Por {col}\n", t.sort_values("tasa", ascending=False)
                      .assign(tasa=lambda x: (100 * x.tasa).round(2)).round(2).to_markdown()]
        mejor, peor = t.tasa.idxmax(), t.tasa.idxmin()
        hallazgos.append((t.lift.max(), f"{col} = {mejor} convierte {t.lift.max():.1f} veces el promedio "
                          f"({100 * t.tasa.max():.1f}% vs {100 * base:.1f}%); {peor} solo {100 * t.tasa.min():.1f}%."))

    # ---- 2. variables más importantes por quintil ----
    imp = pd.read_csv(sal / "importancia.csv").groupby("variable")["ganancia_%"].mean().sort_values(ascending=False)
    nums = [v for v in imp.index if v in df.columns and pd.api.types.is_numeric_dtype(df[v])
            and df[v].nunique() > 5][:6]
    for v in nums:
        x = df[v]
        q = pd.qcut(x.rank(method="first"), 5, labels=["20% más bajo", "2", "3", "4", "20% más alto"])
        q = q.cat.add_categories("sin dato").fillna("sin dato") if x.isna().any() else q
        t = df.groupby(q, observed=True)["y"].agg(tasa="mean", n="size")
        t["lift"] = t.tasa / base
        t["desde"] = x.groupby(q, observed=True).min()
        grafico_barras(t, base, f"Conversión según {v}", graf / f"quintiles_{v}.png", horizontal=False)
        alto, bajo = t.loc["20% más alto", "tasa"], t.loc["20% más bajo", "tasa"]
        tasas = t.drop(index="sin dato", errors="ignore")["tasa"].values
        monot = "sube de forma constante" if np.all(np.diff(tasas) > 0) else (
            "baja de forma constante" if np.all(np.diff(tasas) < 0) else "no es lineal (ojo al explicarlo)")
        razon = max(alto, bajo) / max(min(alto, bajo), 1e-9)
        lado = "más alto" if alto > bajo else "más bajo"
        hallazgos.append((razon, f"{v}: el 20% {lado} convierte {razon:.1f} veces más que el extremo opuesto "
                          f"({100 * max(alto, bajo):.1f}% vs {100 * min(alto, bajo):.1f}%); la relación {monot}."))
        tablas_md += [f"\n### {v} por quintil\n", t.assign(tasa=lambda d: (100 * d.tasa).round(2)).round(3).to_markdown()]

    hallazgos.sort(key=lambda h: -h[0])
    md = ["# Hallazgos para la presentación\n",
          f"Tasa de conversión promedio: **{100 * base:.2f}%** (muestra de {len(df):,} filas).\n",
          "## Las frases más fuertes (elijan 2 o 3 para el PPT)\n"]
    md += [f"{i + 1}. {h[1]}" for i, h in enumerate(hallazgos[:8])]
    md += ["\nRecuerden: son asociaciones, no causas. Los gráficos están en outputs/graficos/.\n", "## Detalle"]
    md += tablas_md
    (sal / "hallazgos.md").write_text("\n".join(md), encoding="utf-8")
    print(f"  {len(hallazgos)} hallazgos | {len(list(graf.glob('*.png')))} gráficos en {graf}")
    for h in hallazgos[:3]:
        print("   - " + h[1])


if __name__ == "__main__":
    main()
