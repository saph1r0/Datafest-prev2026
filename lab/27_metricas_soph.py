"""
RETO DEL 27 DE OCTUBRE — Métricas de clasificación A MANO (solo numpy)

ANTES DE EMPEZAR: copien este archivo con su nombre, para no chocar en Git:
    cp lab/27_metricas.py lab/27_metricas_<nombre>.py
y corrijan con:  python lab/check_27.py lab/27_metricas_<nombre>.py

Objetivo: entender qué mide cada número que el jurado puede preguntar, calculándolo ustedes.
Reglas:
  - Solo numpy dentro de las funciones (nada de sklearn aquí; sklearn se usa en el corrector).
  - Completen cada TODO. Después de cada función, corran:  python lab/check_27.py
  - Pueden pedirle a Copilot que EXPLIQUE una fórmula, no que escriba la función.

Convención:
  y      = respuesta real (1 = convirtió, 0 = no)
  pred   = decisión del modelo (1 = "llamar", 0 = "no llamar")
  score  = probabilidad que da el modelo (entre 0 y 1)
"""
import numpy as np
from scipy.stats import rankdata

# ---------------------------------------------------------------- PARTE 1: con una decisión sí/no
def matriz_confusion(y, pred):
    """Devuelve (VP, FP, FN, VN).
    VP: llamamos y convirtió.        FP: llamamos y NO convirtió (llamada perdida).
    FN: no llamamos y sí convertía.  VN: no llamamos y no convertía.
    Pista: usen comparaciones booleanas, ej. ((y == 1) & (pred == 1)).sum()
    """
    y, pred = np.asarray(y), np.asarray(pred)
    VP=((y==1)&(pred==1)).sum()
    FP=((y==0)&(pred==1)).sum()
    FN=((y==1)&(pred==0)).sum()
    VN=((y==0)&(pred==0)).sum()
    return VP, FP, FN, VN


def exactitud(y, pred):
    """% de aciertos: (VP + VN) / total."""
    VP=((y==1)&(pred==1)).sum()
    VN=((y==0)&(pred==0)).sum()
    return (VP+VN)/len(y)


def precision(y, pred):
    """De los que LLAMAMOS, ¿qué fracción convirtió?  VP / (VP + FP).
    Si no llamamos a nadie (VP + FP = 0), devuelvan 0.0 (igual que sklearn con zero_division=0)."""
    VP=((y==1)&(pred==1)).sum()
    FP=((y==0)&(pred==1)).sum()
    if (VP+FP)==0:
        return 0.0
    return VP/(VP+FP)


def sensibilidad(y, pred):
    """De los que SÍ convertían, ¿a qué fracción encontramos?  VP / (VP + FN). También se llama recall.
    Si no hay positivos reales, devuelvan 0.0."""
    vp=((y==1)&(pred==1)).sum()
    fn=((y==1)&(pred==0)).sum()
    if (vp+fn)==0:
        return 0.0
    return vp/(vp+fn)


def f1(y, pred):
    """Media armónica de precisión y sensibilidad: 2·P·S / (P + S). Si P + S = 0, devuelvan 0.0."""
    p = precision(y, pred)
    s = sensibilidad(y, pred)
    if (p + s) == 0:
        return 0.0
    return 2 * p * s / (p + s)


# ---------------------------------------------------------------- PARTE 2: con el puntaje (orden)
def auc_por_pares(y, score):
    """AUC = probabilidad de que un positivo al azar tenga MAYOR score que un negativo al azar.
    Método directo: comparar TODOS los pares (positivo, negativo).
        par bien ordenado  -> cuenta 1
        empate de scores   -> cuenta 0.5
        par mal ordenado   -> cuenta 0
    AUC = suma / (n_positivos × n_negativos)
    Pista: s_pos[:, None] > s_neg[None, :] crea la matriz de todas las comparaciones de una vez.
    (Con 410 positivos y ~16 mil negativos son ~6.5 millones de pares: numpy lo hace en un segundo.)"""
    y, score = np.asarray(y), np.asarray(score, dtype=float)
    s_pos = score[y == 1]
    s_neg = score[y == 0]
    comparaciones = s_pos[:, None] > s_neg[None, :]
    suma = comparaciones.sum()
    n_pos = len(s_pos)
    n_neg = len(s_neg)

    # Contar empates (score igual)
    empates = (s_pos[:, None] == s_neg[None, :]).sum()
    suma += 0.5 * empates  # cada empate cuenta como 0.5

    total_pares = n_pos * n_neg
    if total_pares == 0:
        return 0.0
    return suma / total_pares


def gini(y, score):
    """Gini = 2 × AUC − 1.  0 = orden al azar, 1 = orden perfecto."""
    return 2 * auc_por_pares(y, score) - 1


# ---------------------------------------------------------------- PARTE 3 (extra): AUC rápido
def auc_por_rangos(y, score):
    """EXTRA. La misma AUC sin comparar pares (fórmula de Mann-Whitney), sirve para millones de filas:
        1. rango de cada score de menor a mayor (1, 2, 3...), con empates = promedio de sus rangos
        2. R = suma de los rangos de los positivos
        3. AUC = (R − n_pos·(n_pos+1)/2) / (n_pos · n_neg)
    Pista para los rangos con empates: scipy.stats.rankdata (aquí sí se permite)."""
    y, score = np.asarray(y), np.asarray(score, dtype=float)
    n_pos = np.sum(y == 1)
    n_neg = np.sum(y == 0)

    # Obtener los rangos
    rangos = rankdata(score)
    R = np.sum(rangos[y == 1])

    # Calcular AUC
    auc = (R - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg) if n_pos > 0 and n_neg > 0 else 0.0
    return auc


# ---------------------------------------------------------------- PARTE 4: aplicarlo al caso
if __name__ == "__main__":
    import pandas as pd
    from pathlib import Path

    val = pd.read_parquet(Path(__file__).resolve().parents[1] / "outputs" / "predicciones_validacion.parquet")
    y, score = val["y"].values, val["p"].values
    print(f"Validación: {len(y):,} empresas, {y.sum():,} convirtieron ({y.mean():.2%})\n")

    # Tres maneras de decidir a quién llamar:
    umbral_top10 = np.quantile(score, 0.90)
    decisiones = {
        "umbral 0.5": (score >= 0.5).astype(int),
        "top 10% del puntaje": (score >= umbral_top10).astype(int),
        "modelo tonto: nadie convierte": np.zeros_like(y),
    }
    filas = []
    for nombre, pred in decisiones.items():
        vp, fp, fn, vn = matriz_confusion(y, pred)
        filas.append({"decisión": nombre, "llamadas": int(pred.sum()), "VP": vp, "FP": fp, "FN": fn, "VN": vn,
                      "exactitud": exactitud(y, pred), "precisión": precision(y, pred),
                      "sensibilidad": sensibilidad(y, pred), "F1": f1(y, pred)})
    print(pd.DataFrame(filas).round(3).to_string(index=False))

    print(f"\nGini del modelo:         {gini(y, score):.3f}")
    print(f"Gini de un puntaje azar: {gini(y, np.random.default_rng(0).random(len(y))):.3f}")
    print(f"Gini del modelo tonto:   {gini(y, np.zeros(len(y))):.3f}   (todos empatados)")

    # TODO PREGUNTAS (respóndanlas en lab/27_respuestas_<nombre>.md, con sus palabras):
    #  1. ¿Por qué el "modelo tonto" tiene una exactitud altísima? ¿Sirve para algo?
    #  2. ¿Por qué con umbral 0.5 casi nadie es "llamado"? ¿Qué umbral usaría el banco y por qué?
    #  3. En factoring, ¿qué es un FP y qué es un FN? ¿Cuál cuesta más? (pista: depende del monto)
    #  4. ¿Por qué el Gini no cambia si cambian el umbral, y la precisión sí?
    #  5. ¿Por qué el banco mide con Gini y no con exactitud
    
if __name__ == "__main__":
    print(f"Gini de 1 - score: {gini(y, 1 - score):.3f}")
    print(f"Gini de raíz cuadrada del score: {gini(y, np.sqrt(score)):.3f}")
    print(f"Gini de score * 100: {gini(y, score * 100):.3f}")
    print(f"Gini de score aleatorio: {gini(y, np.random.default_rng(1).permutation(score)):.3f}")