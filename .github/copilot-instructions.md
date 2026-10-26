# Contexto del proyecto (plantilla: reescríbanla con sus propias palabras)

Caso de factoring del BCP: decidir a qué empresas contactar con capacidad comercial limitada.
Clasificación (¿convierte?, Gini) + regresión (¿por cuánto?, R²/RMSE) + optimización con restricciones.

## Arquitectura
- `config.yaml` controla todo: modo (construir | tabla_plana), nombres reales de columnas, restricciones.
  Nunca escribas nombres reales de columnas dentro del código.
- Pasos en `src/`: s01 ingesta → s02 tabla para modelar → s03 modelos → s04 priorización y optimización
  → s05 simulador → s06 hallazgos → s07 entrega. `run_all.py --desde N` retoma desde un paso.

## Reglas que siempre debes respetar
- Variables solo con datos ANTES o EN la fecha de corte; el objetivo es DESPUÉS.
- Validar por tiempo si hay fechas; al azar solo si no hay.
- Brecha Gini train − valid de máximo 0.05; si se pasa, regularizar, no celebrar.
- Tablas grandes: SQL en DuckDB sobre Parquet; a pandas solo llega lo agregado o muestreado.
- Si se pondera clases o se muestrean negativos, las probabilidades se corrigen (función `corregir`).
- Columnas en `columnas_prohibidas` y las de la alarma de fuga nunca entran al modelo.

## Cómo responder
- Explica en español simple qué hace cada bloque y por qué.
- Si un cambio podría introducir fuga de datos, adviértelo antes de escribir el código.
- Propón la solución más simple primero y mide antes de complicarla.
- Cada variable o hallazgo nuevo debe venir con una frase de negocio que lo justifique.
- Para SQL, revisa `docs/chuleta-sql.md`; para restricciones, `docs/skill-optimizacion.md`.
