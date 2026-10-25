"""
PASO 1 — Ingesta y perfil de datos (la "capa de staging").

Qué hace:
  1. Lee cada archivo crudo (CSV o Parquet) con DuckDB, SIN cargarlo entero en la memoria de Python.
  2. Renombra las columnas reales a nombres lógicos según config.yaml.
  3. Convierte fechas en cualquier formato común (2026-03-31, 31/03/2026, 202603) a DATE.
  4. Guarda todo en Parquet (columnar y comprimido) para que los pasos siguientes lean rápido.
  5. Escribe outputs/perfil_datos.md: filas, nulos, rangos de fechas, llaves duplicadas e integridad.

Por qué así: en la industria esto se llama capa de staging o normalización. Aísla el "desorden"
de los datos de entrada en un solo lugar; si el formato cambia, solo se toca esta capa (y el config).
"""
import duckdb

from utils import cargar_config, cronometro, parquet

COLS_FECHA = {("transacciones", "fecha"), ("factoring", "fecha"), ("perfil", "mes")}


def expr_fecha(col: str) -> str:
    """Prueba varios formatos y se queda con el primero que funcione (COALESCE de TRY_*)."""
    c = f'CAST("{col}" AS VARCHAR)'
    return (f"COALESCE(TRY_CAST({c} AS DATE), "
            f"TRY_STRPTIME({c}, '%d/%m/%Y')::DATE, "
            f"TRY_STRPTIME({c}, '%Y%m')::DATE, "
            f"TRY_STRPTIME({c}, '%Y-%m-%d %H:%M:%S')::DATE)")


def lector(ruta: str) -> str:
    if ruta.endswith(".parquet"):
        return f"read_parquet('{ruta}')"
    # sample_size=-1 evita adivinar mal los tipos mirando solo las primeras filas
    return f"read_csv('{ruta}', header=true, sample_size=-1)"


def main():
    cfg = cargar_config()
    con = duckdb.connect()
    reporte = ["# Perfil de los datos\n"]

    for tabla, archivo in cfg["tablas"].items():
        if not archivo:
            print(f"  {tabla}: no configurada, se omite")
            continue
        ruta = cfg["rutas"]["crudos"] / archivo
        if not ruta.exists():
            print(f"  {tabla}: NO EXISTE {ruta} — se omite (el pipeline seguirá sin ella)")
            continue
        with cronometro(f"ingesta {tabla}"):
            reales = {r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {lector(str(ruta))}").fetchall()}
            selects = []
            for logico, real in cfg["columnas"][tabla].items():
                if real is None or real not in reales:
                    if real is not None:
                        print(f"    aviso: columna '{real}' ({logico}) no está en {archivo}; queda NULL")
                    selects.append(f"NULL AS {logico}")
                elif (tabla, logico) in COLS_FECHA:
                    selects.append(f"{expr_fecha(real)} AS {logico}")
                else:
                    selects.append(f'"{real}" AS {logico}')
            con.execute(f"COPY (SELECT {', '.join(selects)} FROM {lector(str(ruta))}) "
                        f"TO '{parquet(cfg, tabla)}' (FORMAT parquet, COMPRESSION zstd)")

        # ---- perfil rápido: todo en SQL, sobre el Parquet ----
        p = parquet(cfg, tabla)
        n = con.execute(f"SELECT COUNT(*) FROM '{p}'").fetchone()[0]
        reporte.append(f"\n## {tabla} — {n:,} filas\n\n| columna | tipo | % nulos | distintos (aprox.) |\n|---|---|---|---|")
        for col, tipo, *_ in con.execute(f"DESCRIBE SELECT * FROM '{p}'").fetchall():
            nulos, dist = con.execute(
                f"SELECT AVG(CASE WHEN {col} IS NULL THEN 1 ELSE 0 END)*100, APPROX_COUNT_DISTINCT({col}) FROM '{p}'"
            ).fetchone()
            reporte.append(f"| {col} | {tipo} | {nulos:.1f} | {dist:,} |")
        for (t, c) in COLS_FECHA:
            if t == tabla:
                mn, mx, malas = con.execute(
                    f"SELECT MIN({c}), MAX({c}), SUM(CASE WHEN {c} IS NULL THEN 1 ELSE 0 END) FROM '{p}'").fetchone()
                reporte.append(f"\nRango de {c}: {mn} a {mx}. Fechas no interpretables: {malas:,}.")

    # ---- chequeos de llaves e integridad referencial ----
    def existe(t):
        return (cfg["rutas"]["parquet"] / f"{t}.parquet").exists()

    reporte.append("\n## Chequeos de llaves\n")
    if existe("empresas"):
        dup = con.execute(f"SELECT COUNT(*) - COUNT(DISTINCT id) FROM '{parquet(cfg, 'empresas')}'").fetchone()[0]
        reporte.append(f"- IDs duplicados en empresas: {dup:,}")
        for t, cols in [("transacciones", ["proveedor", "pagador"]), ("factoring", ["proveedor"])]:
            if existe(t):
                for c in cols:
                    huerf = con.execute(
                        f"SELECT AVG(CASE WHEN e.id IS NULL THEN 1 ELSE 0 END)*100 FROM '{parquet(cfg, t)}' x "
                        f"LEFT JOIN '{parquet(cfg, 'empresas')}' e ON x.{c} = e.id").fetchone()[0]
                    reporte.append(f"- % de {t}.{c} que NO existe en empresas: {huerf:.2f}%")
    if existe("factoring"):
        canales = con.execute(
            f"SELECT canal, COUNT(*) FROM '{parquet(cfg, 'factoring')}' GROUP BY 1 ORDER BY 2 DESC").fetchall()
        reporte.append(f"- Operaciones de factoring por canal: {canales}")

    salida = cfg["rutas"]["salidas"] / "perfil_datos.md"
    salida.write_text("\n".join(reporte), encoding="utf-8")
    print(f"  reporte: {salida}")


if __name__ == "__main__":
    main()
