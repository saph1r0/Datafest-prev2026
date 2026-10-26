"""
PASO 0b — Datos de práctica tipo "tabla plana" (como el caso presencial de 2025).

Así describió el finalista los datos del año pasado:
  - millones de filas en CSV, ~200 variables, muchos vacíos (NA)
  - números que en realidad son categorías (ej. sexo 1/2, nivel socioeconómico 1..5)
  - valores extremos (outliers)
  - la etiqueta ya venía armada en train; en test había que llenarla (como Kaggle)
  - pocos positivos (desbalance) y un monto desembolsado para estimar con regresión
  - la estructura "se prestaba" a varios modelos (producto x canal)

Este generador imita todo eso para practicar el modo tabla_plana, incluida una variable trampa.

Uso:  python s00b_generar_plana.py --filas 200000            (rápido)
      python s00b_generar_plana.py --filas 2000000           (prueba de costo computacional)
Luego en config.yaml:  modo: tabla_plana,  tabla_plana.tiempo: periodo,  segmentar_por: producto (opcional)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv


def sigmoide(x):
    return 1 / (1 + np.exp(-x))


def generar(n: int, semilla: int, salida: Path, n_ruido: int = 170):
    rng = np.random.default_rng(semilla)
    n_test = n // 4
    N = n + n_test
    f32 = np.float32

    # ---------- variables con señal ----------
    ingreso = rng.lognormal(8.0, 0.7, N).astype(f32)
    ingreso[rng.random(N) < 0.002] *= 40                      # outliers: ingresos absurdos
    saldo = (ingreso * rng.uniform(0.2, 3, N)).astype(f32)
    antig = rng.integers(1, 240, N).astype(f32)
    n_prod = rng.poisson(2, N).astype(f32)
    deuda = (ingreso * rng.gamma(1.5, 2, N)).astype(f32)
    uso_linea = rng.beta(2, 3, N).astype(f32)
    contactos = rng.poisson(1.2, N).astype(f32)
    dias_ult = rng.exponential(60, N).astype(f32)
    edad = rng.integers(18, 80, N).astype(f32)
    buro = rng.normal(600, 90, N).clip(300, 900).astype(f32)
    nse = rng.integers(1, 6, N)                                # 1..5: número que es CATEGORÍA
    sexo = rng.integers(1, 3, N)                               # 1/2: número que es CATEGORÍA
    region = rng.choice(["LIMA", "AREQUIPA", "LA_LIBERTAD", "PIURA", "CUSCO", "OTRAS"], N,
                        p=[.45, .12, .1, .08, .07, .18])
    producto = rng.choice(["TARJETA", "VEHICULAR", "PERSONAL"], N, p=[.5, .2, .3])
    canal = rng.choice(["A", "B"], N, p=[.6, .4])
    periodo = rng.choice([int(f"{y}{m:02d}") for y, m in
                          [(2024, 10), (2024, 11), (2024, 12)] + [(2025, k) for k in range(1, 10)]], N)

    efecto_nse = np.array([0, -0.3, 0.4, 0.1, -0.2, 0.5])[nse]  # NO es monótono: tratarlo como número falla
    efecto_prod = {"TARJETA": 0.0, "VEHICULAR": -0.8, "PERSONAL": -0.3}
    ep = np.vectorize(efecto_prod.get)(producto)
    z = (-4.3 + 0.6 * np.log1p(ingreso / 3000) - 1.2 * (uso_linea - 0.4)
         + 0.5 * np.tanh((buro - 600) / 80) * (canal == "A")      # interacción: el buró pesa en el canal A
         + 0.25 * contactos - 0.004 * dias_ult + efecto_nse + ep
         + 0.3 * (region == "AREQUIPA") + rng.normal(0, 0.7, N))
    y = (rng.random(N) < sigmoide(z)).astype(np.int8)
    monto = np.where(y == 1, (ingreso * rng.uniform(2, 8, N) * np.where(producto == "VEHICULAR", 4, 1)).round(-1),
                     np.nan)

    datos = {
        "id": np.array([f"C{i:08d}" for i in range(N)]), "periodo": periodo,
        "producto": producto, "canal": canal, "region": region,
        "ingreso_mensual": ingreso, "saldo_promedio": saldo, "antiguedad_meses": antig,
        "n_productos": n_prod, "deuda_sistema": deuda, "ratio_uso_linea": uso_linea,
        "n_contactos_previos": contactos, "dias_ultima_compra": dias_ult, "edad": edad,
        "score_buro": buro, "nivel_socioeconomico": nse, "sexo": sexo,
    }
    # vacíos en variables con señal (5% a 30%)
    for c in ["ingreso_mensual", "saldo_promedio", "deuda_sistema", "score_buro", "dias_ultima_compra"]:
        v = datos[c].astype(f32)
        v[rng.random(N) < rng.uniform(0.05, 0.3)] = np.nan
        datos[c] = v
    # copias casi idénticas (correlación ~0.99): la selección de variables debería eliminarlas
    datos["ingreso_anual_aprox"] = (ingreso * 12 * rng.normal(1, 0.02, N)).astype(f32)
    datos["saldo_promedio_3m"] = (saldo * rng.normal(1, 0.03, N)).astype(f32)
    # columnas inútiles
    datos["columna_constante"] = np.ones(N, dtype=np.int8)
    v = rng.normal(0, 1, N).astype(f32); v[rng.random(N) < 0.995] = np.nan
    datos["casi_vacia"] = v
    # TRAMPA de fuga: estado posterior a la oferta (en la vida real no se conoce al momento de decidir)
    datos["flag_estado_post_oferta"] = np.where(rng.random(N) < 0.97, y, 1 - y).astype(np.int8)
    # ruido: variables sin relación con la etiqueta, con vacíos
    for j in range(n_ruido):
        tipo = j % 3
        if tipo == 0:
            v = rng.normal(0, 1, N).astype(f32)
        elif tipo == 1:
            v = rng.lognormal(3, 1, N).astype(f32)
        else:
            v = rng.integers(0, 50, N).astype(f32)
        v[rng.random(N) < rng.uniform(0, 0.6)] = np.nan
        datos[f"v{j:03d}"] = v

    df = pd.DataFrame(datos)
    tr, te = df.iloc[:n].copy(), df.iloc[n:].drop(columns=["flag_estado_post_oferta"]).copy()
    tr["target"], tr["monto"] = y[:n], monto[:n]
    salida.mkdir(parents=True, exist_ok=True)
    pacsv.write_csv(pa.Table.from_pandas(tr, preserve_index=False), salida / "train.csv")
    pacsv.write_csv(pa.Table.from_pandas(te, preserve_index=False), salida / "test.csv")
    # respuesta oculta del test (solo para que ustedes midan su Gini "ciego" al practicar)
    pd.DataFrame({"id": te["id"], "target": y[n:]}).to_csv(salida / "test_respuestas_OCULTAS.csv", index=False)
    print(f"  train: {len(tr):,} filas x {tr.shape[1]} columnas | positivos {tr.target.mean():.2%}")
    print(f"  test:  {len(te):,} filas (sin etiqueta)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--filas", type=int, default=200_000)
    ap.add_argument("--semilla", type=int, default=11)
    ap.add_argument("--salida", default="../data/raw")
    a = ap.parse_args()
    generar(a.filas, a.semilla, Path(a.salida))
