# Contexto del proyecto (plantilla: reescríbanla con sus propias palabras)

Caso de factoring del BCP: decidir a qué empresas contactar en la próxima campaña, con capacidad
comercial limitada. Se evalúa con Gini (clasificación) y posiblemente R² (monto).

## Arquitectura
- `config.yaml` mapea nombres reales de columnas a nombres lógicos. Nunca escribas nombres reales en el código.
- Pasos en `src/`: s01 ingesta (DuckDB → Parquet), s02 features (SQL por fecha de corte),
  s03 modelos (logística vs LightGBM), s04 priorización (valor esperado), s05 simulador HTML.

## Reglas que siempre debes respetar
- Unidad de análisis: una fila por empresa y fecha de corte.
- Variables solo con datos ANTES o EN la fecha de corte; el objetivo es DESPUÉS.
- Validación out-of-time. Nunca partición aleatoria entre meses.
- Gini = 2*AUC - 1. Monto en log(1 + monto).
- Tablas grandes: SQL en DuckDB sobre Parquet; a pandas solo llega la tabla agregada.
- Columnas en `columnas_prohibidas` nunca entran al modelo.

## Cómo responder
- Explica en español simple qué hace cada bloque y por qué.
- Si un cambio podría introducir fuga de datos, adviértelo antes de escribir el código.
- Propón la solución más simple primero.
- Cada variable nueva debe venir con una frase de negocio que la justifique.
