"""
Corre el pipeline completo de punta a punta:   python run_all.py
Retomar desde un paso:                         python run_all.py --desde 3
Generar datos de práctica (modo construir):    python run_all.py --generar 1.0
Generar datos de práctica (modo tabla_plana):  python run_all.py --generar-plana 200000
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
PASOS = [
    ("1", "Ingesta y perfil de datos", "s01_ingesta.py"),
    ("2", "Tabla para modelar (construir o tabla plana)", "s02_features.py"),
    ("3", "Modelos y validación", "s03_modelo.py"),
    ("4", "Priorización y optimización", "s04_priorizar.py"),
    ("5", "Simulador HTML", "s05_simulador.py"),
    ("6", "Hallazgos y gráficos para el PPT", "s06_hallazgos.py"),
    ("7", "Archivo de entrega validado", "s07_entrega.py"),
]


def correr(script, *args):
    r = subprocess.run([sys.executable, script, *args], cwd=RAIZ / "src")
    if r.returncode != 0:
        detalle = ""
        if r.returncode < 0:  # el sistema mató el proceso: no hay traceback de Python
            import signal
            nombre = signal.Signals(-r.returncode).name
            detalle = (f"\nEl proceso murió por la señal {nombre} (código {r.returncode}), sin error de Python.\n"
                       "Causas típicas: SIGSEGV/SIGABRT = choque de librerías nativas (OpenMP/BLAS), muy común\n"
                       "en Mac con el entorno 'base' de conda; SIGKILL = se quedó sin memoria.\n"
                       f"Para ver todo el detalle: cd src && python {script}")
        sys.exit(f"\nFalló {script} (código {r.returncode}).{detalle}\nArreglen ese paso y retomen con --desde.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--desde", default="1")
    ap.add_argument("--generar", type=float, help="escala de datos sintéticos (modo construir)")
    ap.add_argument("--generar-plana", type=int, help="filas de datos sintéticos (modo tabla_plana)")
    a = ap.parse_args()
    t0 = time.perf_counter()
    if a.generar:
        print("\n== 0. Generar datos sintéticos (4 bases) ==")
        correr("s00_generar_datos.py", "--escala", str(a.generar), "--salida", str(RAIZ / "data/raw"))
    if a.generar_plana:
        print("\n== 0b. Generar datos sintéticos (tabla plana) ==")
        correr("s00b_generar_plana.py", "--filas", str(a.generar_plana), "--salida", str(RAIZ / "data/raw"))
    for num, nombre, script in PASOS:
        if num >= a.desde:
            t = time.perf_counter()
            print(f"\n== {num}. {nombre} ==")
            correr(script)
            print(f"   ({time.perf_counter() - t:.0f} s)")
    print(f"\nListo en {time.perf_counter() - t0:.0f} s. Resultados en outputs/ (entrega en outputs/entrega/)")