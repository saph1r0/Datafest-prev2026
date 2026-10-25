# Base de solución — Caso Factoring

Pipeline de punta a punta para priorizar a qué empresas contactar con una fuerza comercial limitada.
Corre en una laptop, sin nube, en menos de un minuto con ~2 millones de transacciones.

## Arranque rápido

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_all.py --generar 1.0  # genera datos de práctica y corre todo
```

Abran `outputs/simulador.html` con doble clic.

## El día del evento

1. Copien los archivos del USB a `data/raw/`.
2. Editen **solo** `config.yaml`: nombres de archivos, nombres de columnas, valores (canal del banco, segmentos), fechas de corte y capacidad comercial.
3. `python run_all.py` y revisen `outputs/perfil_datos.md` antes que nada.

## Los pasos

| Paso | Script | Qué produce | Herramienta y por qué |
|---|---|---|---|
| 0 | `s00_generar_datos.py` | 4 bases sintéticas para practicar | numpy: simula red de pagos, contagio y una trampa de fuga |
| 1 | `s01_ingesta.py` | Parquet normalizado + `perfil_datos.md` | DuckDB: lee CSV grandes sin cargarlos en memoria; normaliza nombres y fechas |
| 2 | `s02_features.py` | `tabla_modelo.parquet` (una fila por empresa y corte) | SQL en DuckDB: agrega millones de filas; variables solo con datos antes del corte |
| 3 | `s03_modelo.py` | Gini, R², `scores.parquet`, `reporte_modelo.md` | Logística (baseline) vs LightGBM, validación out-of-time, alarma de fuga, razones por empresa |
| 4 | `s04_priorizar.py` | `lista_priorizada.csv`, `resumen_negocio.md` | Valor esperado, curva de ganancia, criterio de habilitar, comparación con leads actuales |
| 5 | `s05_simulador.py` | `simulador.html` | HTML estático: funciona sin servidor ni internet |
| — | `benchmark.py` | Tiempos y memoria de pandas vs DuckDB | Para defender la elección con números |

## Resultados con los datos de práctica (escala 1.0)

- Gini en validación out-of-time: logística 0.557, LightGBM 0.561 (prácticamente empate; el pipeline elige el ganador solo).
- R² del monto (escala log): 0.84.
- Contactando al 10% de la base, la lista por valor esperado capta 75% del monto real en validación; los leads actuales, 27%; el azar, 6%.
- Ordenar por probabilidad capta más conversiones (40%) pero menos monto (54%): por eso se ordena por valor esperado.
- Sin las bases de plazo, perfil y canal, el Gini cae a 0.33: así se mide cuánto aporta cada base.
- Fuga de datos de demostración: si se quita `flag_factoring_actual` de las prohibidas, el Gini salta a 1.000 y la alarma la detecta.
- Benchmark: DuckDB + Parquet 0.47 s y 175 MB vs pandas + CSV 2.33 s y 579 MB.

## Estructura

```text
config.yaml          ← lo único que cambia el día del evento
run_all.py           ← corre todo; --desde N para retomar
src/                 ← un script por paso, comentado
.github/copilot-instructions.md  ← contexto que Copilot lee en cada consulta
docs/                ← skills y guías en .md para Copilot
data/raw/            ← datos crudos (no se suben a Git)
outputs/             ← resultados (no se suben a Git)
```
