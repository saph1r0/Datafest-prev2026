# Skill: adaptar la base a un caso nuevo en 30 minutos

Usar cuando lleguen los datos reales y no coincidan con los de práctica.

## Paso 1 — Inventario (10 min)
Para cada archivo: filas, columnas, llaves, rango de fechas. Responder por escrito:
- ¿Cuál es la unidad de cada tabla? (empresa, transacción, operación, empresa-mes)
- ¿Qué columna une las tablas? ¿Hay IDs huérfanos?
- ¿Qué columna dice si una empresa ya usa el producto? ¿Desde cuándo?

## Paso 2 — Mapear en config.yaml (10 min)
- `tablas`: archivo por nombre lógico. Si falta una base, `null`.
- `columnas`: nombre real por nombre lógico. Columna inexistente = `null`.
- `valores`: canal del banco, segmentos grandes, tipo de factura.
- `tiempo`: cortes trimestrales que dejen 12 meses de historia antes y N meses de objetivo después.

## Paso 3 — Correr y leer el perfil (10 min)
`python run_all.py` y luego leer `outputs/perfil_datos.md`:
- ¿Fechas no interpretables? Ajustar formato en `s01_ingesta.py::expr_fecha`.
- ¿% de huérfanos alto? Revisar la llave.
- ¿Tasa de conversión por corte razonable (0.5% a 15%)? Si es 0% o 50%, el objetivo está mal definido.

## Árbol de decisión si el caso cambia
- El objetivo es un MONTO y no una conversión → usar solo el regresor; métrica R².
- No hay historia de factoring del banco → objetivo = uso en cualquier banco (SPI) o look-alike.
- Pocas conversiones (menos de 200) → logística primero, cortes más largos, más cortes apilados.
- Base enorme que no entra en memoria → todo en DuckDB, muestrear negativos (no positivos) para entrenar.
- Piden ranking por ejecutivo o región → agrupar `lista_priorizada.csv` y repartir capacidad.
