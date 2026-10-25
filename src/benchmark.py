"""
BENCHMARK — ¿Por qué DuckDB + Parquet y no solo pandas + CSV?

Mide la MISMA tarea (monto total cobrado por proveedor) de tres maneras, cada una en un proceso
aparte para medir su pico de memoria por separado:
  A) pandas leyendo el CSV completo y agrupando
  B) DuckDB leyendo el CSV directo
  C) DuckDB leyendo el Parquet (columnar: solo lee las 2 columnas que necesita)

Úsenlo para defender la elección ante el jurado con números de SU laptop, no con opiniones.
"""
import json
import subprocess
import sys

from utils import cargar_config

TAREAS = {
    "A) pandas + CSV": """
import pandas as pd
df = pd.read_csv(r'{csv}')
r = df.groupby('{prov_real}')['{monto_real}'].sum()
""",
    "B) DuckDB + CSV": """
import duckdb
r = duckdb.sql("SELECT {prov_real}, SUM({monto_real}) FROM read_csv('{csv}') GROUP BY 1").df()
""",
    "C) DuckDB + Parquet": """
import duckdb
r = duckdb.sql("SELECT proveedor, SUM(monto) FROM '{pq}' GROUP BY 1").df()
""",
}

MEDIDOR = """
import resource, sys, time, json
t0 = time.perf_counter()
{codigo}
seg = time.perf_counter() - t0
kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
mb = kb / 1024 / 1024 if sys.platform == 'darwin' else kb / 1024
print(json.dumps({{"segundos": seg, "memoria_mb": mb}}))
"""


def main():
    cfg = cargar_config()
    csv = cfg["rutas"]["crudos"] / cfg["tablas"]["transacciones"]
    pq = cfg["rutas"]["parquet"] / "transacciones.parquet"
    col = cfg["columnas"]["transacciones"]
    print(f"  CSV: {csv.stat().st_size / 1e6:,.0f} MB | Parquet: {pq.stat().st_size / 1e6:,.0f} MB "
          f"({csv.stat().st_size / pq.stat().st_size:.1f}x más pequeño)")
    print(f"  {'método':<22}{'tiempo (s)':>12}{'pico RAM (MB)':>16}")
    for nombre, codigo in TAREAS.items():
        codigo = codigo.format(csv=csv, pq=pq, prov_real=col["proveedor"], monto_real=col["monto"])
        out = subprocess.run([sys.executable, "-c", MEDIDOR.format(codigo=codigo)],
                             capture_output=True, text=True)
        if out.returncode != 0:
            print(f"  {nombre:<22} falló: {out.stderr.strip().splitlines()[-1]}")
            continue
        r = json.loads(out.stdout.strip().splitlines()[-1])
        print(f"  {nombre:<22}{r['segundos']:>12.2f}{r['memoria_mb']:>16,.0f}")


if __name__ == "__main__":
    main()
