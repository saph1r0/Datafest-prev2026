"""
PASO 1 — Ingesta y perfil de datos (la "capa de staging").

Qué hace:
  1. Lee cada archivo crudo (CSV o Parquet) con DuckDB, SIN cargarlo entero en la memoria de Python.
  2. Renombra las columnas reales a nombres lógicos según config.yaml.
     - modo construir: mapea columna por columna (y deja pasar las extra si columnas_extra = true)
     - modo tabla_plana: renombra solo id, objetivo, monto y tiempo; el resto pasa tal cual
  3. Convierte fechas en cualquier formato común (2026-03-31, 31/03/2026, 202603) a DATE.
  4. Guarda todo en Parquet (columnar y comprimido) para que los pasos siguientes lean rápido.
  5. Escribe outputs/perfil_datos.md con, por columna: tipo, % vacíos, valores distintos, y avisos:
       - "posible categórica": un número con pocos valores distintos (ej. sexo 1/2, nivel 1..5)
       - "casi vacía" o "constante": candidata a descartar
       - "valores extremos": máximo muy lejos del resto (outliers)
     + chequeos de llaves (duplicados, códigos que no cruzan entre tablas).

Por qué así: en la industria esto se llama capa de staging. Aísla el "desorden" de los datos
de entrada en un solo lugar; si el formato cambia, solo se toca esta capa (y el config).
"""
from utils import cargar_config, conectar, cronometro, parquet

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


def selects_construir(cfg, tabla, reales, archivo):
    selects, usados = [], set()
    for logico, real in cfg["columnas"][tabla].items():
        if real is None or real not in reales:
            if real is not None:
                print(f"    aviso: columna '{real}' ({logico}) no está en {archivo}; queda NULL")
            selects.append(f"NULL AS {logico}")
        elif (tabla, logico) in COLS_FECHA:
            selects.append(f"{expr_fecha(real)} AS {logico}")
            usados.add(real)
        else:
            selects.append(f'"{real}" AS {logico}')
            usados.add(real)
    if cfg.get("columnas_extra", {}).get(tabla):
        logicos = set(cfg["columnas"][tabla])
        extra = [c for c in reales if c not in usados and c not in logicos]
        selects += [f'"{c}"' for c in extra]
        if extra:
            print(f"    {len(extra)} columnas extra pasan tal cual")
    return selects


def selects_plana(cfg, reales, es_test):
    tp = cfg["tabla_plana"]
    ren = {tp["id"]: "id", tp.get("objetivo"): "y", tp.get("monto"): "monto_objetivo", tp.get("tiempo"): "corte"}
    ren = {k: v for k, v in ren.items() if k}
    faltan = [k for k, v in ren.items() if k not in reales and not (es_test and v in ("y", "monto_objetivo"))]
    if faltan:
        raise SystemExit(f"Faltan columnas en la tabla plana: {faltan}. Revisen tabla_plana en config.yaml")
    return [f'"{c}" AS {ren[c]}' if c in ren else f'"{c}"' for c in reales]


def perfilar(con, p, tabla, reporte):
    """Una sola pasada con SUMMARIZE: estadísticas de TODAS las columnas a la vez (rápido con 200 columnas)."""
    n = con.execute(f"SELECT COUNT(*) FROM '{p}'").fetchone()[0]
    res = con.execute(f"SUMMARIZE SELECT * FROM '{p}'").df()
    reporte.append(f"\n## {tabla} — {n:,} filas, {len(res)} columnas\n")
    reporte.append("| columna | tipo | % vacíos | distintos (aprox.) | aviso |\n|---|---|---|---|---|")
    avisos = {"posible categórica": [], "casi vacía": [], "constante": [], "valores extremos": []}
    for r in res.itertuples():
        nulos = float(r.null_percentage) if r.null_percentage is not None else 0.0
        distintos = int(r.approx_unique or 0)
        aviso = ""
        es_num = any(t in r.column_type for t in ("INT", "DOUBLE", "FLOAT", "DECIMAL"))
        if nulos > 98:
            aviso = "casi vacía"
        elif distintos <= 1:
            aviso = "constante"
        elif es_num and "INT" in r.column_type and 2 < distintos <= 12 and r.column_name not in ("y", "corte"):
            aviso = "posible categórica"
        elif es_num and r.q25 is not None and r.q75 is not None:
            try:
                q25, q75, mx = float(r.q25), float(r.q75), float(r.max)
                if q75 > q25 and mx > q75 + 20 * (q75 - q25):
                    aviso = "valores extremos"
            except (TypeError, ValueError):
                pass
        if aviso:
            avisos[aviso].append(r.column_name)
        reporte.append(f"| {r.column_name} | {r.column_type} | {nulos:.1f} | {distintos:,} | {aviso} |")
    for k, cols in avisos.items():
        if cols:
            reporte.append(f"\n**{k}** ({len(cols)}): {', '.join(cols[:40])}{' …' if len(cols) > 40 else ''}")
    if avisos["posible categórica"]:
        print(f"    {len(avisos['posible categórica'])} posibles categóricas codificadas como número: "
              "revisen y agréguenlas a forzar_categoricas")
    return n


def main():
    cfg = cargar_config()
    con = conectar(cfg)
    reporte = ["# Perfil de los datos\n"]
    plana = cfg.get("modo") == "tabla_plana"
    tablas = ({"train": cfg["tabla_plana"]["train"], "test": cfg["tabla_plana"]["test"]} if plana
              else cfg["tablas"])

    for tabla, archivo in tablas.items():
        if not archivo:
            print(f"  {tabla}: no configurada, se omite")
            continue
        ruta = cfg["rutas"]["crudos"] / archivo
        if not ruta.exists():
            print(f"  {tabla}: NO EXISTE {ruta} — se omite (el pipeline seguirá sin ella)")
            continue
        with cronometro(f"ingesta {tabla}"):
            reales = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {lector(str(ruta))}").fetchall()]
            selects = (selects_plana(cfg, reales, tabla == "test") if plana
                       else selects_construir(cfg, tabla, set(reales), archivo))
            con.execute(f"COPY (SELECT {', '.join(selects)} FROM {lector(str(ruta))}) "
                        f"TO '{parquet(cfg, tabla)}' (FORMAT parquet, COMPRESSION zstd)")
        with cronometro(f"perfil {tabla}"):
            perfilar(con, parquet(cfg, tabla), tabla, reporte)
        if not plana:
            for (t, c) in COLS_FECHA:
                if t == tabla:
                    mn, mx, malas = con.execute(
                        f"SELECT MIN({c}), MAX({c}), SUM(CASE WHEN {c} IS NULL THEN 1 ELSE 0 END) "
                        f"FROM '{parquet(cfg, tabla)}'").fetchone()
                    reporte.append(f"\nRango de {c}: {mn} a {mx}. Fechas no interpretables: {malas:,}.")

    # ---- chequeos de llaves e integridad referencial ----
    def existe(t):
        return (cfg["rutas"]["parquet"] / f"{t}.parquet").exists()

    reporte.append("\n## Chequeos de llaves\n")
    if plana:
        for t in ("train", "test"):
            if existe(t):
                dup = con.execute(f"SELECT COUNT(*) - COUNT(DISTINCT id) FROM '{parquet(cfg, t)}'").fetchone()[0]
                reporte.append(f"- IDs repetidos en {t}: {dup:,} (si hay, ¿es una fila por id y producto/canal?)")
        if existe("train"):
            tasa = con.execute(f"SELECT AVG(y) FROM '{parquet(cfg, 'train')}'").fetchone()[0]
            reporte.append(f"- Tasa de positivos en train: {100 * tasa:.2f}% (desbalance)")
            print(f"    tasa de positivos en train: {100 * tasa:.2f}%")
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
