# Skill: revisar fuga de datos

Usar antes de reportar cualquier métrica.

## Señales
- Gini de validación mayor a 0.90.
- Una variable sola con Gini mayor a 0.80 (`alarma_fuga` en `s03_modelo.py` lo calcula).
- Gini de validación MAYOR que el de entrenamiento por mucho.

## Fuentes típicas
1. Columnas "foto" del maestro tomadas al final del periodo (estado actual, flags de producto).
2. Ventanas que pasan el corte: `fecha <= corte` en variables, `fecha > corte` en objetivo. Revisar cada filtro.
3. Variables que contienen el producto: "n_productos" que cuenta factoring, "segmento comercial = factoring".
4. Normalizar o imputar con estadísticas de toda la base (incluida validación).
5. La misma empresa en train y valid en el MISMO periodo.

## Prueba de demostración
En `config.yaml`, quitar `flag_factoring_actual` de `columnas_prohibidas` y correr desde el paso 2:
el Gini sube a ~1.0 y la alarma se dispara. Volver a ponerla.
