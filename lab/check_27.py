"""
Corrector del reto del 27 de octubre:
    python lab/check_27.py lab/27_metricas_ana.py      (cada una con SU archivo)
Compara sus funciones contra sklearn con datos al azar y casos borde (empates, nadie llamado, etc.).
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

ruta = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("27_metricas.py")
print(f"Corrigiendo {ruta.name}\n")
spec = importlib.util.spec_from_file_location("m27", ruta)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

rng = np.random.default_rng(7)
casos = []
for n, tasa in [(50, 0.3), (2000, 0.03), (500, 0.5)]:
    y = (rng.random(n) < tasa).astype(int)
    score = np.clip(0.3 * y + rng.random(n), 0, 1).round(2)  # redondeo => hay empates a propósito
    casos.append((y, score, (score >= 0.6).astype(int)))
y0 = np.array([0, 1, 0, 1, 0])
casos.append((y0, np.array([0.1, 0.4, 0.4, 0.8, 0.2]), np.zeros(5, dtype=int)))   # nadie llamado

ok_total = 0


def probar(nombre, fn, esperado_fn, usa_score=False, pista=""):
    global ok_total
    try:
        for y, score, pred in casos:
            arg = score if usa_score else pred
            r, e = fn(y, arg), esperado_fn(y, arg)
            if not np.allclose(np.asarray(r, dtype=float), np.asarray(e, dtype=float), atol=1e-9):
                print(f"  ❌ {nombre}: dio {r}, se esperaba {e}. {pista}")
                return
        print(f"  ✅ {nombre}")
        ok_total += 1
    except NotImplementedError:
        print(f"  ⏳ {nombre}: todavía sin hacer")
    except Exception as ex:
        print(f"  ❌ {nombre}: error {type(ex).__name__}: {ex}")


def cm(y, p):
    vn, fp, fn, vp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    return (vp, fp, fn, vn)


print("Parte 1 — decisión sí/no")
probar("matriz_confusion", m.matriz_confusion, cm, pista="El orden es (VP, FP, FN, VN).")
probar("exactitud", m.exactitud, accuracy_score)
probar("precision", m.precision, lambda y, p: precision_score(y, p, zero_division=0),
       pista="¿Devuelven 0.0 cuando no llaman a nadie?")
probar("sensibilidad", m.sensibilidad, lambda y, p: recall_score(y, p, zero_division=0))
probar("f1", m.f1, lambda y, p: f1_score(y, p, zero_division=0))
print("Parte 2 — orden")
probar("auc_por_pares", m.auc_por_pares, roc_auc_score, usa_score=True,
       pista="¿Cuentan los empates como 0.5?")
probar("gini", m.gini, lambda y, s: 2 * roc_auc_score(y, s) - 1, usa_score=True)
print("Parte 3 — extra")
probar("auc_por_rangos", m.auc_por_rangos, roc_auc_score, usa_score=True,
       pista="Los empates deben recibir el rango promedio.")
print(f"\n{ok_total} de 8 correctas." + (f"  ¡Todo listo! Corran ahora: python {ruta}" if ok_total >= 7 else ""))
if __name__ == "__main__":
    gini(y, 1 - score)
    gini(y, np.sqrt(score)) y gini(y, score * 100)
    gini(y, np.random.default_rng(1).permutation(score))
    