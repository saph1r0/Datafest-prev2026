"""
PASO 7 — Archivo de entrega (el "submission").

En 2025 se entregó por USB, al mediodía, un archivo con el ID y la predicción (y el monto), más el código
y el PPT. Según el finalista, de 20 equipos solo 11 o 12 lograron entregar. Por eso:
  - este paso se corre APENAS haya un primer modelo (hora 6), y se vuelve a correr con cada mejora
  - valida el archivo antes de guardarlo: ids completos, sin repetidos, sin vacíos, probabilidades entre 0 y 1
  - guarda una copia con hora en outputs/entrega/ para no perder la última versión buena

El formato exacto lo darán el día del evento: se ajusta en config.yaml -> entrega.columnas.
"""
import shutil
import sys
from datetime import datetime

import pandas as pd

from utils import cargar_config, conectar


def main():
    cfg = cargar_config()
    sal = cfg["rutas"]["salidas"]
    ent = cfg.get("entrega") or {}
    mapa = ent.get("columnas") or {"id": "id", "probabilidad": "p"}
    lista = pd.read_csv(sal / "lista_priorizada.csv")
    faltan = [v for v in mapa.values() if v not in lista.columns]
    if faltan:
        sys.exit(f"Columnas internas inexistentes en lista_priorizada.csv: {faltan}")
    out = lista[list(mapa.values())].copy()
    out.columns = list(mapa.keys())

    # ---- validaciones ----
    errores = []
    con = conectar(cfg)
    esperados = con.execute(f"SELECT COUNT(DISTINCT id) FROM '{sal / 'tabla_scoring.parquet'}'").fetchone()[0]
    col_id, col_p = list(mapa.keys())[0], next((k for k, v in mapa.items() if v == "p"), None)
    if out[col_id].nunique() != esperados:
        errores.append(f"ids distintos {out[col_id].nunique():,} != {esperados:,} esperados")
    if out.duplicated().any():
        errores.append(f"{out.duplicated().sum():,} filas repetidas")
    if out.isna().any().any():
        errores.append(f"vacíos en: {out.columns[out.isna().any()].tolist()}")
    if col_p and not out[col_p].between(0, 1).all():
        errores.append("probabilidades fuera de [0, 1]")
    if errores:
        sys.exit("ENTREGA INVÁLIDA:\n  - " + "\n  - ".join(errores))

    carpeta = sal / "entrega"
    carpeta.mkdir(exist_ok=True)
    archivo = carpeta / ent.get("archivo", "entrega.csv")
    out.to_csv(archivo, index=False)
    copia = carpeta / f"{archivo.stem}_{datetime.now():%Y%m%d_%H%M}{archivo.suffix}"
    shutil.copy(archivo, copia)
    print(f"  entrega válida: {archivo} ({len(out):,} filas, columnas {list(out.columns)})")
    print(f"  copia de respaldo: {copia.name}")


if __name__ == "__main__":
    main()
