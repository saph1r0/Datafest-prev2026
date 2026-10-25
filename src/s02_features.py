"""
PASO 2 — Tabla maestra: UNA FILA POR EMPRESA Y FECHA DE CORTE.

La regla de oro (evita la fuga de datos):
    variables  = solo lo que pasó ANTES o EN la fecha de corte
    objetivo   = lo que pasa DESPUÉS del corte (próximos N meses)

Se construye una tabla por cada corte (entrenamiento, validación y scoring) y se apilan.
Apilar varios cortes multiplica los ejemplos de entrenamiento y enseña al modelo patrones
que se repiten en el tiempo, no la casualidad de un solo mes.

Todo se calcula en SQL con DuckDB directamente sobre Parquet: agrega millones de filas
sin traerlas a pandas. A Python solo llega la tabla final (una fila por empresa).

Población elegible en cada corte: empresas que NO son grandes, que vendieron a crédito
en los últimos 12 meses y que todavía NO hacen factoring con el banco.
"""
import duckdb
import pandas as pd

from utils import cargar_config, cronometro, parquet


def disponible(con, cfg, tabla, col=None) -> bool:
    ruta = cfg["rutas"]["parquet"] / f"{tabla}.parquet"
    if not ruta.exists():
        return False
    if col is None:
        return True
    return con.execute(f"SELECT COUNT({col}) FROM '{ruta}'").fetchone()[0] > 0


def lista_sql(valores) -> str:
    return ", ".join(f"'{v}'" for v in valores)


def construir_corte(con, cfg, corte: str, con_objetivo: bool) -> pd.DataFrame:
    v = cfg["valores"]
    N = cfg["tiempo"]["ventana_objetivo_meses"]
    C = f"DATE '{corte}'"
    E, T, F, P = (parquet(cfg, t) for t in ("empresas", "transacciones", "factoring", "perfil"))

    hay_tipo = disponible(con, cfg, "transacciones", "tipo")
    hay_plazo = disponible(con, cfg, "transacciones", "plazo")
    hay_canal = disponible(con, cfg, "factoring", "canal")
    hay_perfil = disponible(con, cfg, "perfil")
    filtro_tipo = f"AND tipo = '{v['tipo_factura']}'" if hay_tipo else ""
    cols_emp = {r[0] for r in con.execute(f"DESCRIBE SELECT * FROM '{E}'").fetchall()}
    prohibidas = [c for c in cfg["modelo"]["columnas_prohibidas"] if c in cols_emp]
    excluir = f"EXCLUDE ({', '.join(prohibidas)})" if prohibidas else ""
    es_banco = f"canal = '{v['canal_banco']}'" if hay_canal else "TRUE"

    # ---------- bloques de variables (CTEs) ----------
    ctes = f"""
    emp AS (SELECT * {excluir} FROM '{E}'),
    grandes AS (
        SELECT id FROM '{E}'
        WHERE segmento IN ({lista_sql(v['segmentos_grandes'])}) OR rubro IN ({lista_sql(v['rubros_estado'])})
    ),
    trx AS (
        SELECT * FROM '{T}'
        WHERE fecha <= {C} AND fecha > {C} - INTERVAL 12 MONTH {filtro_tipo}
    ),
    -- (a) comportamiento como PROVEEDOR: cuánto vende a crédito, a cuántos, con qué tendencia
    prov AS (
        SELECT t.proveedor AS id,
            SUM(monto) FILTER (WHERE fecha > {C} - INTERVAL 3 MONTH)                    AS monto_cobrado_3m,
            SUM(monto) FILTER (WHERE fecha > {C} - INTERVAL 6 MONTH)                    AS monto_cobrado_6m,
            SUM(monto)                                                                  AS monto_cobrado_12m,
            COUNT(*)   FILTER (WHERE fecha > {C} - INTERVAL 6 MONTH)                    AS n_facturas_6m,
            COUNT(DISTINCT pagador)                                                     AS n_pagadores_12m,
            COUNT(DISTINCT date_trunc('month', fecha))                                  AS meses_activos_12m,
            SUM(monto) FILTER (WHERE g.id IS NOT NULL) / NULLIF(SUM(monto), 0)          AS share_ventas_grandes,
            {"SUM(monto * plazo) / NULLIF(SUM(monto), 0)" if hay_plazo else "NULL"}     AS plazo_prom_ponderado,
            {"SUM(monto) FILTER (WHERE plazo >= 90) / NULLIF(SUM(monto), 0)" if hay_plazo else "NULL"} AS share_plazo_90,
            SUM(monto) FILTER (WHERE fecha > {C} - INTERVAL 3 MONTH)
              / NULLIF(SUM(monto) FILTER (WHERE fecha <= {C} - INTERVAL 3 MONTH
                                         AND fecha > {C} - INTERVAL 6 MONTH), 0)        AS tendencia_3m_vs_previo
        FROM trx t LEFT JOIN grandes g ON t.pagador = g.id
        GROUP BY t.proveedor
    ),
    concentracion AS (
        SELECT proveedor AS id, MAX(m) / NULLIF(SUM(m), 0) AS share_pagador_principal
        FROM (SELECT proveedor, pagador, SUM(monto) AS m FROM trx GROUP BY 1, 2)
        GROUP BY 1
    ),
    -- (b) quiénes ya hacen factoring con el banco ANTES del corte
    fact_prev AS (SELECT * FROM '{F}' WHERE fecha <= {C}),
    adoptantes AS (SELECT DISTINCT proveedor AS id FROM fact_prev WHERE {es_banco}),
    -- (c) RED DE PAGOS: ¿mis "hermanos" (otros proveedores de mi pagador) ya usan factoring?
    rel AS (SELECT proveedor, pagador, SUM(monto) AS m FROM trx GROUP BY 1, 2),
    tasa_pagador AS (
        SELECT r.pagador, AVG(CASE WHEN a.id IS NOT NULL THEN 1.0 ELSE 0.0 END) AS tasa
        FROM rel r LEFT JOIN adoptantes a ON r.proveedor = a.id
        GROUP BY 1
    ),
    red AS (
        SELECT r.proveedor AS id,
            SUM(r.m * tp.tasa) / NULLIF(SUM(r.m), 0)          AS contagio_ponderado,
            MAX(tp.tasa)                                      AS contagio_max,
            COUNT(*) FILTER (WHERE tp.tasa > 0)               AS n_pagadores_con_factoring
        FROM rel r JOIN tasa_pagador tp USING (pagador)
        GROUP BY 1
    ),
    -- (d) demanda ya probada: factoring en OTROS bancos (visible vía SPI)
    otro_banco AS (
        SELECT proveedor AS id,
            COUNT(*)                                 AS n_ops_otro_banco_12m,
            SUM(monto)                               AS monto_otro_banco_12m,
            date_diff('day', MAX(fecha), {C})        AS dias_desde_ult_otro_banco
        FROM fact_prev
        WHERE {"NOT (" + es_banco + ")" if hay_canal else "FALSE"} AND fecha > {C} - INTERVAL 12 MONTH
        GROUP BY 1
    )"""

    if hay_perfil:
        ctes += f""",
    perf AS (
        SELECT id,
            arg_max(saldo, mes)                                                 AS saldo_ultimo,
            arg_max(productos, mes)                                             AS n_productos,
            arg_max(credito, mes)                                               AS tiene_credito,
            AVG(saldo) FILTER (WHERE mes > {C} - INTERVAL 3 MONTH)
              / NULLIF(AVG(saldo) FILTER (WHERE mes <= {C} - INTERVAL 3 MONTH
                                          AND mes > {C} - INTERVAL 6 MONTH), 0) - 1   AS cambio_saldo_3m
        FROM '{P}' WHERE mes <= {C} AND mes > {C} - INTERVAL 6 MONTH
        GROUP BY id
    )"""
    else:
        ctes += """,
    perf AS (SELECT NULL AS id, NULL AS saldo_ultimo, NULL AS n_productos, NULL AS tiene_credito,
             NULL AS cambio_saldo_3m WHERE FALSE)"""

    if con_objetivo:
        ctes += f""",
    objetivo AS (
        SELECT proveedor AS id, 1 AS y, SUM(monto) AS monto_objetivo
        FROM '{F}'
        WHERE {es_banco} AND fecha > {C} AND fecha <= {C} + INTERVAL {N} MONTH
        GROUP BY 1
    )"""
        sel_obj = ", COALESCE(o.y, 0) AS y, o.monto_objetivo"
        join_obj = "LEFT JOIN objetivo o USING (id)"
    else:
        sel_obj, join_obj = "", ""

    sql = f"""
    WITH {ctes}
    SELECT '{corte}' AS corte, e.*,
        p.* EXCLUDE (id), c.share_pagador_principal,
        COALESCE(r.contagio_ponderado, 0) AS contagio_ponderado,
        COALESCE(r.contagio_max, 0) AS contagio_max,
        COALESCE(r.n_pagadores_con_factoring, 0) AS n_pagadores_con_factoring,
        COALESCE(ob.n_ops_otro_banco_12m, 0) AS n_ops_otro_banco_12m,
        COALESCE(ob.monto_otro_banco_12m, 0) AS monto_otro_banco_12m,
        ob.dias_desde_ult_otro_banco,
        pf.saldo_ultimo, pf.n_productos, pf.tiene_credito, pf.cambio_saldo_3m
        {sel_obj}
    FROM emp e
    JOIN prov p USING (id)                       -- solo quien vendió a crédito en 12 meses
    LEFT JOIN concentracion c USING (id)
    LEFT JOIN red r USING (id)
    LEFT JOIN otro_banco ob USING (id)
    LEFT JOIN perf pf USING (id)
    {join_obj}
    WHERE e.id NOT IN (SELECT id FROM grandes)
      AND e.id NOT IN (SELECT id FROM adoptantes)  -- ya clientes de factoring: no son "conversión"
    """
    return con.execute(sql).df()


def main():
    cfg = cargar_config()
    con = duckdb.connect()
    t = cfg["tiempo"]
    tablas = []
    for corte in t["cortes_entrenamiento"] + [t["corte_validacion"]]:
        with cronometro(f"features corte {corte}"):
            df = construir_corte(con, cfg, corte, con_objetivo=True)
        print(f"    {len(df):,} empresas | tasa de conversión {df['y'].mean():.2%}")
        tablas.append(df)
    modelo = pd.concat(tablas, ignore_index=True)
    modelo.to_parquet(cfg["rutas"]["salidas"] / "tabla_modelo.parquet", index=False)

    with cronometro(f"features scoring {t['corte_scoring']}"):
        score = construir_corte(con, cfg, t["corte_scoring"], con_objetivo=False)
    score.to_parquet(cfg["rutas"]["salidas"] / "tabla_scoring.parquet", index=False)
    print(f"    {len(score):,} empresas para priorizar | {modelo.shape[1]} columnas")


if __name__ == "__main__":
    main()
