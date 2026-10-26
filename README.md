# Base de solución — DataFest BCP 2026

Pipeline de punta a punta para decidir **a quién contactar** con una fuerza comercial limitada:
probabilidad × monto = valor esperado, selección óptima con restricciones, hallazgos para el PPT y archivo
de entrega validado. Corre en una laptop, sin nube.

## Arranque rápido

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_all.py --generar 1.0  # datos de práctica (4 bases) y todo el pipeline
```

## Dos modos (se elige en config.yaml)

| Modo | Cuándo | Datos de práctica |
|---|---|---|
| `construir` | Llegan bases crudas (empresas, transacciones, operaciones, perfil) y hay que armar la tabla y el objetivo. Así se anunció el caso factoring | `python run_all.py --generar 1.0` |
| `tabla_plana` | Llega un train con la etiqueta y un test para llenar. Así fue 2025: millones de filas, ~200 variables | `python src/s00b_generar_plana.py --salida data/raw` + `config_ejemplo_tabla_plana.yaml` |

## El día del evento

1. Copiar los archivos del USB a `data/raw/`.
2. Editar **solo** `config.yaml`: modo, nombres de archivos y columnas, capacidad, restricciones.
3. `python run_all.py` y leer `outputs/perfil_datos.md` antes que nada.
4. Correr el paso 7 (entrega) apenas exista un primer modelo, y repetirlo con cada mejora.

## Los pasos

| Paso | Script | Qué produce |
|---|---|---|
| 0 | `s00_generar_datos.py` / `s00b_generar_plana.py` | Datos de práctica (4 bases / tabla ancha con trampas) |
| 1 | `s01_ingesta.py` | Parquet + `perfil_datos.md` (vacíos, categorías codificadas, extremos, llaves) |
| 2 | `s02_features.py` | `tabla_modelo.parquet` con partición train/valid y `tabla_scoring.parquet` |
| 3 | `s03_modelo.py` | Selección de variables, alarma de fuga, logística vs LightGBM + Optuna, brecha máx. 5 pts, monto |
| 4 | `s04_priorizar.py` | Valor esperado, curva de ganancia, **optimización con restricciones** y su costo, factor de habilitación |
| 5 | `s05_simulador.py` | `simulador.html` (sin servidor ni internet) |
| 6 | `s06_hallazgos.py` | `hallazgos.md` + gráficos PNG para el PPT |
| 7 | `s07_entrega.py` | `outputs/entrega/entrega.csv` validado + copia con hora |
| — | `benchmark.py` | pandas vs DuckDB en tiempo y memoria |

## Pruebas realizadas (laptop virtual de 2 núcleos y 7 GB)

| Prueba | Resultado |
|---|---|
| Construir, 20 mil empresas, 2.3 M facturas | 66 s de punta a punta |
| Construir, 80 mil empresas, 9.4 M facturas | 4 min 14 s, pico de 0.9 GB de RAM |
| Tabla plana, 200 mil x 194 columnas | 2 min; variable trampa excluida sola; 59 de 190 variables elegidas |
| Tabla plana, 1.5 M x 194 columnas (CSV de 1.6 GB) | 11 min, pico de 2.7 GB; Gini 0.369 en validación |
| Gini "ciego" en el test vs validación (200 mil) | 0.356 vs 0.331: estable |
| Optimización con 3 restricciones | 1.3–7 s, a 0.00% del óptimo teórico; costo por restricción |
| Columna trampa activada a propósito | Gini 1.00 → la alarma la detecta y la excluye |

## Guías en docs/ (Copilot las puede leer)

- `chuleta-sql.md`: uniones, agregados, ventanas, deduplicar (para imprimir).
- `skill-adaptar-caso.md`, `skill-tabla-plana.md`, `skill-optimizacion.md`, `skill-fuga-de-datos.md`.
