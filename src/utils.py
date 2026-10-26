"""Utilidades compartidas: configuración, cronómetro con memoria y métricas."""
import resource
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import roc_auc_score

RAIZ = Path(__file__).resolve().parents[1]


def cargar_config(ruta: str | None = None) -> dict:
    with open(ruta or RAIZ / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for k in ("crudos", "parquet", "salidas"):
        cfg["rutas"][k] = RAIZ / cfg["rutas"][k]
        cfg["rutas"][k].mkdir(parents=True, exist_ok=True)
    return cfg


def memoria_mb() -> float:
    """Pico de memoria del proceso (RSS) en MB. En Linux ru_maxrss viene en KB; en macOS en bytes."""
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1024 / 1024 if sys.platform == "darwin" else r / 1024


@contextmanager
def cronometro(etapa: str):
    """Mide tiempo y memoria de cada etapa: el BCP dijo que evaluará el manejo computacional."""
    t0 = time.perf_counter()
    yield
    print(f"  [{etapa}] {time.perf_counter() - t0:6.1f} s | pico de memoria {memoria_mb():,.0f} MB")


def gini(y_true, score) -> float:
    """Gini = 2*AUC - 1. Solo mide el ORDEN: 0 = azar, 1 = perfecto."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return 2 * roc_auc_score(y_true, score) - 1


def conectar(cfg: dict):
    """Conexión DuckDB con tope de memoria: lo que no cabe se escribe a disco en vez de colgar la laptop."""
    import duckdb
    con = duckdb.connect()
    r = cfg.get("recursos", {})
    if r.get("memoria_duckdb"):
        con.execute(f"SET memory_limit = '{r['memoria_duckdb']}'")
    if r.get("hilos"):
        con.execute(f"SET threads = {int(r['hilos'])}")
    tmp = cfg["rutas"]["parquet"] / "_tmp_duckdb"
    tmp.mkdir(exist_ok=True)
    con.execute(f"SET temp_directory = '{tmp}'")
    con.execute("SET preserve_insertion_order = false")  # menos memoria en COPY grandes
    return con


def parquet(cfg: dict, tabla: str) -> str:
    """Ruta como texto para usar dentro de SQL de DuckDB."""
    return str(cfg["rutas"]["parquet"] / f"{tabla}.parquet")
