# Skill: cuando llega una tabla ya armada (modo tabla_plana)

Así fue el caso presencial de 2025: CSV con millones de filas, ~200 variables, etiqueta lista en train
y un test sin etiqueta para llenar (como en Kaggle).

## Pasos (20 minutos)

1. Copiar `train.csv` y `test.csv` a `data/raw/`.
2. En `config.yaml`:
   - `modo: tabla_plana`
   - `tabla_plana`: nombres reales de `id`, `objetivo`, `monto` (o null), `tiempo` (o null)
   - `perfil_columnas`: 2 o 3 columnas para describir la lista (región, segmento, producto...)
   - `columna_lead: null` si no aplica
3. `python run_all.py --desde 1 ` y leer `outputs/perfil_datos.md`:
   - **posible categórica** → agregar a `modelo.forzar_categoricas` (ej. sexo 1/2, nivel 1..5)
   - **casi vacía / constante** → el modelo ya las descarta solo
   - **valores extremos** → a LightGBM no le afectan; a la logística sí (ya usa log en montos)
4. Revisar en la consola la línea **ALERTA DE FUGA**: esas columnas se excluyen solas. Confirmar con
   un mentor que de verdad son información "del futuro".

## Si la laptop no aguanta

| Síntoma | Cambio en config.yaml |
|---|---|
| La ingesta se cuelga | `recursos.memoria_duckdb` a la mitad de la RAM |
| La carga del paso 3 se cuelga | `modelo.muestreo_negativos: 0.2` (las probabilidades se corrigen solas) |
| La validación es enorme | `modelo.max_filas_validacion: 500000` |
| Optuna tarda demasiado | `modelo.optuna_minutos: 1` o `optuna_trials: 0` |
| Demasiadas variables | `modelo.max_variables: 40` |

Medido en una laptop de 2 núcleos y 7 GB: 1.5 millones de filas x 194 columnas (ver README).

## ¿Un modelo o varios?

Si los datos "se prestan" (ej. producto x canal), prueben `segmentar_por`. Pero midan: cada segmento
tiene menos datos y sobreajusta más. En nuestra prueba, 3 modelos por producto dieron Gini global 0.324
contra 0.331 de un solo modelo. Se decide con el número, no con la intuición, y se explica al jurado por qué.
