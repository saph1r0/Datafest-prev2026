"""
PASO 0 — Generador de datos sintéticos (solo para practicar).

Imita las 4 bases que anunció el BCP:
  empresas.csv        maestro de empresas (rubro, segmento, región, flags)
  transacciones.csv   facturas entre empresas: quién le paga a quién (es una RED)
  factoring.csv       operaciones de factoring (BCP y otros bancos vía SPI)
  perfil_mensual.csv  saldos y productos mes a mes

Qué "esconde" el generador (para que el modelo tenga algo que descubrir):
  - Necesidad de liquidez: vender mucho a pagadores grandes que pagan lento.
  - Contagio: si los otros proveedores de mi pagador ya hacen factoring, yo también tiendo a hacerlo.
  - Demanda probada: quien ya hace factoring en otro banco convierte más.
  - Tensión de caja: saldo que cae mes a mes.
  - Trampa de fuga de datos: 'flag_factoring_actual' en el maestro es una foto del FINAL del periodo.
  - Sesgo de los leads actuales: el banco hoy elige leads sobre todo en Lima y medianas.

Uso:  python src/s00_generar_datos.py --escala 1     (≈20 mil empresas, ≈2 millones de facturas)
      python src/s00_generar_datos.py --escala 0.2   (rápido, para probar)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RUBROS = ["CONSTRUCCION", "MINERIA_SERVICIOS", "COMERCIO", "MANUFACTURA", "TRANSPORTE",
          "AGRO", "TEXTIL", "TECNOLOGIA", "SALUD", "ALIMENTOS"]
# Efecto del rubro en la propensión (oculto): construcción y servicios mineros venden a crédito largo.
EFECTO_RUBRO = np.array([0.9, 1.0, -0.2, 0.4, 0.5, 0.6, 0.1, -0.3, 0.2, 0.0])
REGIONES = ["LIMA", "AREQUIPA", "LA_LIBERTAD", "PIURA", "CUSCO", "JUNIN", "OTRAS"]
P_REGION = [0.50, 0.10, 0.08, 0.07, 0.06, 0.06, 0.13]
SEGMENTOS = ["MIPE", "PYME", "MEDIANA", "GRANDE"]
P_SEGMENTO = [0.45, 0.35, 0.13, 0.07]
VENTA_MENSUAL = {"MIPE": 15_000, "PYME": 80_000, "MEDIANA": 400_000}  # soles, mediana


def sigmoide(x):
    return 1 / (1 + np.exp(-x))


def generar(escala: float, semilla: int, salida: Path):
    rng = np.random.default_rng(semilla)
    n = int(20_000 * escala)
    meses = pd.period_range("2024-10", "2026-09", freq="M")  # 24 meses
    T = len(meses)

    # ---------------- 1. Maestro de empresas ----------------
    seg = rng.choice(SEGMENTOS, size=n, p=P_SEGMENTO)
    rubro_idx = rng.integers(0, len(RUBROS), size=n)
    rubro = np.array(RUBROS, dtype=object)[rubro_idx]
    es_grande = seg == "GRANDE"
    # 20% de las grandes son entidades del Estado (pagan lento)
    estado = es_grande & (rng.random(n) < 0.20)
    rubro[estado] = "SECTOR_PUBLICO"
    region = rng.choice(REGIONES, size=n, p=P_REGION)
    es_cliente = rng.random(n) < np.where(es_grande, 0.8, 0.55)
    ids = np.array([f"E{i:06d}" for i in range(n)])

    # ---------------- 2. La red: proveedores -> pagadores ----------------
    pagadores = np.where(es_grande | ((seg == "MEDIANA") & (rng.random(n) < 0.3)))[0]
    proveedores = np.where(~es_grande)[0]
    # plazo de pago de cada pagador: el Estado paga a 90-120 días
    plazo_pagador = np.zeros(n)
    plazo_pagador[pagadores] = rng.choice([30, 60, 90, 120], size=len(pagadores), p=[0.3, 0.35, 0.25, 0.10])
    plazo_pagador[estado] = rng.choice([90, 120], size=estado.sum())
    # pagadores "populares" atraen más proveedores (ley de potencia)
    peso_pag = rng.pareto(1.5, size=len(pagadores)) + 1
    peso_pag /= peso_pag.sum()

    k = np.clip(rng.poisson(2.2, size=len(proveedores)) + 1, 1, 8)
    e_prov = np.repeat(proveedores, k)
    e_pag = rng.choice(pagadores, size=len(e_prov), p=peso_pag)
    # quitar autoaristas y duplicados
    aristas = pd.DataFrame({"prov": e_prov, "pag": e_pag})
    aristas = aristas[aristas.prov != aristas.pag].drop_duplicates()
    aristas["share"] = rng.gamma(1.0, 1.0, size=len(aristas))
    aristas["share"] /= aristas.groupby("prov")["share"].transform("sum")
    e_prov = aristas.prov.to_numpy()
    e_pag = aristas.pag.to_numpy()
    e_share = aristas.share.to_numpy()

    # volumen de ventas mensual de cada proveedor
    venta_base = np.zeros(n)
    for s, v in VENTA_MENSUAL.items():
        m = seg == s
        venta_base[m] = rng.lognormal(np.log(v), 0.6, size=m.sum())
    crecimiento = rng.normal(0.005, 0.02, size=n)  # tendencia mensual

    # % de ventas que va a pagadores lentos (>=90 días): la "necesidad de liquidez"
    lento = (plazo_pagador[e_pag] >= 90).astype(float)
    share_lento = np.bincount(e_prov, weights=e_share * lento, minlength=n)
    share_grandes = np.bincount(e_prov, weights=e_share * es_grande[e_pag], minlength=n)

    # ---------------- 3. Propensión latente (lo que el modelo debe descubrir) ----------------
    efecto_seg = np.select([seg == "MIPE", seg == "PYME", seg == "MEDIANA"], [-0.4, 0.3, 0.2], 0)
    efecto_rub = EFECTO_RUBRO[rubro_idx]
    tension = rng.normal(0, 1, size=n)  # tensión de caja: se verá como saldo cayendo
    z = (efecto_seg + efecto_rub + 1.6 * share_lento + 0.6 * share_grandes
         + 0.35 * tension + rng.normal(0, 0.6, size=n))

    # Leads actuales: el banco hoy elige por intuición (Lima, medianas y clientes), no por necesidad real
    puntaje_lead = (1.2 * (region == "LIMA") + 1.0 * (seg == "MEDIANA") + 0.8 * es_cliente
                    + 0.3 * z + rng.normal(0, 1, size=n))
    elegible = ~es_grande
    umbral = np.quantile(puntaje_lead[elegible], 0.90)
    tiene_lead = elegible & (puntaje_lead >= umbral)

    # Usuarios de factoring en otros bancos (visibles vía SPI): demanda ya probada
    usa_otro_banco = elegible & (rng.random(n) < sigmoide(z - 2.8))
    inicio_otro = rng.integers(0, T, size=n)

    # ---------------- 4. Simulación mes a mes: adopción con contagio ----------------
    adopto_mes = np.full(n, -1)
    for t in range(T):
        adoptado = adopto_mes >= 0
        # por pagador: fracción de sus proveedores que ya adoptaron
        tot = np.bincount(e_pag, minlength=n)
        ad = np.bincount(e_pag, weights=adoptado[e_prov], minlength=n)
        tasa_pag = np.divide(ad, tot, out=np.zeros(n), where=tot > 0)
        contagio = np.bincount(e_prov, weights=e_share * tasa_pag[e_pag], minlength=n)
        otro_activo = usa_otro_banco & (inicio_otro <= t)
        logit = (-7.6 + z + 3.0 * contagio + 1.0 * otro_activo + 0.5 * es_cliente
                 + 0.7 * tiene_lead)
        h = sigmoide(logit)
        nuevo = elegible & ~adoptado & (rng.random(n) < h)
        adopto_mes[nuevo] = t
    print(f"  adoptantes BCP en el periodo: {(adopto_mes >= 0).sum():,} de {elegible.sum():,} elegibles")

    # ---------------- 5. Transacciones (facturas) ----------------
    filas = []
    for t, per in enumerate(meses):
        activo = rng.random(len(e_prov)) < 0.75
        n_fac = rng.integers(1, 4, size=len(e_prov)) * activo
        idx = np.repeat(np.arange(len(e_prov)), n_fac)
        if len(idx) == 0:
            continue
        venta_mes = venta_base[e_prov[idx]] * (1 + crecimiento[e_prov[idx]]) ** t
        monto = venta_mes * e_share[idx] / n_fac[idx] * rng.lognormal(0, 0.3, size=len(idx))
        dia = rng.integers(1, per.days_in_month + 1, size=len(idx))
        plazo = np.clip(plazo_pagador[e_pag[idx]] + rng.normal(0, 8, size=len(idx)), 15, 180).round()
        filas.append(pd.DataFrame({
            "fecha": per.start_time + pd.to_timedelta(dia - 1, unit="D"),
            "id_pagador": ids[e_pag[idx]],
            "id_proveedor": ids[e_prov[idx]],
            "monto": monto.round(2),
            "plazo_dias": plazo.astype(int),
            "tipo": "FACTURA",
        }))
        # ruido: transferencias que no son facturas
        n_tr = int(0.1 * len(idx))
        filas.append(pd.DataFrame({
            "fecha": per.start_time + pd.to_timedelta(rng.integers(0, per.days_in_month, n_tr), unit="D"),
            "id_pagador": ids[rng.integers(0, n, n_tr)],
            "id_proveedor": ids[rng.integers(0, n, n_tr)],
            "monto": rng.lognormal(8, 1.2, n_tr).round(2),
            "plazo_dias": 0,
            "tipo": "TRANSFERENCIA",
        }))
    trx = pd.concat(filas, ignore_index=True)

    # ---------------- 6. Operaciones de factoring ----------------
    ops = []
    pag_principal = aristas.sort_values("share").groupby("prov").tail(1).set_index("prov")["pag"]
    for t, per in enumerate(meses):
        # BCP: primera operación en el mes de adopción y luego repeticiones
        en_bcp = (adopto_mes >= 0) & (adopto_mes <= t)
        opera = en_bcp & ((adopto_mes == t) | (rng.random(n) < 0.5))
        # otros bancos (SPI)
        opera_otro = usa_otro_banco & (inicio_otro <= t) & (rng.random(n) < 0.4)
        for mask, canal in [(opera, "BCP"), (opera_otro, "OTRO_BANCO")]:
            q = np.where(mask)[0]
            if len(q) == 0:
                continue
            venta_mes = venta_base[q] * (1 + crecimiento[q]) ** t
            monto = venta_mes * np.clip(share_lento[q] + 0.2, 0.2, 1) * rng.uniform(0.2, 0.6, len(q))
            ops.append(pd.DataFrame({
                "fecha_operacion": per.start_time + pd.to_timedelta(rng.integers(0, per.days_in_month, len(q)), unit="D"),
                "id_proveedor": ids[q],
                "id_pagador": ids[pag_principal.reindex(q).fillna(pagadores[0]).astype(int).to_numpy()],
                "monto_desembolsado": monto.round(2),
                "canal": canal,
            }))
    fact = pd.concat(ops, ignore_index=True)
    fact.insert(0, "id_operacion", [f"F{i:07d}" for i in range(len(fact))])

    # ---------------- 7. Perfil mensual ----------------
    saldo_base = venta_base.clip(min=50_000) * rng.uniform(0.3, 1.2, size=n)
    deriva = -0.025 * tension  # tensión alta -> saldo cae
    prods_base = rng.poisson(np.where(es_grande, 5, 2), size=n) + 1
    perfil = []
    for t, per in enumerate(meses):
        perfil.append(pd.DataFrame({
            "id_empresa": ids,
            "mes": int(per.strftime("%Y%m")),  # formato AAAAMM, como en el Coding Challenge
            "saldo_promedio": (saldo_base * (1 + deriva) ** t * rng.lognormal(0, 0.15, n)).round(2),
            "n_productos": prods_base + ((adopto_mes >= 0) & (adopto_mes <= t)),
            "tiene_credito": rng.random(n) < 0.35,
        }))
    perfil = pd.concat(perfil, ignore_index=True)
    perfil = perfil[es_cliente[np.tile(np.arange(n), T)]]  # el banco solo ve el perfil de sus clientes

    empresas = pd.DataFrame({
        "id_empresa": ids, "rubro": rubro, "segmento": seg, "region": region,
        "es_cliente": es_cliente, "tiene_lead": tiene_lead,
        # TRAMPA: foto al final del periodo. Usarla como variable es fuga de datos.
        "flag_factoring_actual": adopto_mes >= 0,
    })

    salida.mkdir(parents=True, exist_ok=True)
    empresas.to_csv(salida / "empresas.csv", index=False)
    trx.to_csv(salida / "transacciones.csv", index=False)
    fact.to_csv(salida / "factoring.csv", index=False)
    perfil.to_csv(salida / "perfil_mensual.csv", index=False)
    for nombre, df in [("empresas", empresas), ("transacciones", trx), ("factoring", fact), ("perfil_mensual", perfil)]:
        print(f"  {nombre:<16} {len(df):>12,} filas")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--escala", type=float, default=1.0)
    ap.add_argument("--semilla", type=int, default=7)
    ap.add_argument("--salida", default="data/raw")
    a = ap.parse_args()
    generar(a.escala, a.semilla, Path(a.salida))
