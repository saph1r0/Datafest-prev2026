"""
Corre el pipeline completo de punta a punta:  python run_all.py
Para correr solo desde un paso:                python run_all.py --desde 3
Para generar datos de práctica primero:        python run_all.py --generar 1.0
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
PASOS = [
    ("1", "Ingesta y perfil de datos", "s01_ingesta.py"),
    ("2", "Tabla maestra con fecha de corte", "s02_features.py"),
    ("3", "Modelos y validación out-of-time", "s03_modelo.py"),
    ("4", "Priorización y valor de negocio", "s04_priorizar.py"),
    ("5", "Simulador HTML", "s05_simulador.py"),
]


def correr(script, *args):
    r = subprocess.run([sys.executable, script, *args], cwd=RAIZ / "src")
    if r.returncode != 0:
        sys.exit(f"\nFalló {script}. Arreglen ese paso y retomen con --desde.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--desde", default="1")
    ap.add_argument("--generar", type=float, help="escala de datos sintéticos a generar antes")
    a = ap.parse_args()
    t0 = time.perf_counter()
    if a.generar:
        print("\n== 0. Generar datos sintéticos ==")
        correr("s00_generar_datos.py", "--escala", str(a.generar), "--salida", str(RAIZ / "data/raw"))
    for num, nombre, script in PASOS:
        if num >= a.desde:
            print(f"\n== {num}. {nombre} ==")
            correr(script)
    print(f"\nListo en {time.perf_counter() - t0:.0f} s. Resultados en outputs/")
