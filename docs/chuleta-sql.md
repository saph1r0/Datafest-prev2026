# Chuleta de SQL (DuckDB) — imprimir y llevar

El equipo finalista de 2025 casi no entrega porque, a minutos del cierre, nadie recordaba cómo hacer un
merge en SQL. Esta hoja existe para que eso no pase. Todo funciona en DuckDB desde Python:

```python
import duckdb
con = duckdb.connect()
df = con.execute("SELECT ... FROM 'data/parquet/tabla.parquet'").df()   # resultado a pandas
con.execute("COPY (SELECT ...) TO 'salida.parquet' (FORMAT parquet)")  # resultado a archivo, sin pasar por pandas
```

## 1. Leer archivos

```sql
SELECT * FROM read_csv('data/raw/train.csv', header = true, sample_size = -1);  -- CSV
SELECT * FROM 'data/parquet/train.parquet';                                       -- Parquet
DESCRIBE SELECT * FROM 'train.parquet';      -- columnas y tipos
SUMMARIZE SELECT * FROM 'train.parquet';     -- min, max, nulos, distintos de TODAS las columnas
```

## 2. Uniones (merge)

| Quiero... | SQL | pandas equivalente |
|---|---|---|
| Solo lo que está en ambas | `a JOIN b ON a.id = b.id` | `a.merge(b, on="id")` |
| Todo `a`, y de `b` lo que cruce | `a LEFT JOIN b ON a.id = b.id` | `a.merge(b, on="id", how="left")` |
| Llaves con nombre distinto | `a LEFT JOIN b ON a.id_empresa = b.ruc` | `left_on=..., right_on=...` |
| Lo de `a` que NO está en `b` | `a ANTI JOIN b ON a.id = b.id` | `a[~a.id.isin(b.id)]` |
| Lo de `a` que SÍ está en `b` (sin traer columnas) | `a SEMI JOIN b ON a.id = b.id` | `a[a.id.isin(b.id)]` |
| Unir por la misma columna | `a JOIN b USING (id)` | `on="id"` |

**Regla de oro:** antes y después de un JOIN, contar filas. Si después hay MÁS filas, la llave está repetida en `b`.

```sql
SELECT COUNT(*), COUNT(DISTINCT id) FROM b;   -- si difieren, b tiene ids repetidos: agregar antes de unir
```

## 3. Agregar (una fila por empresa)

```sql
SELECT id_proveedor AS id,
       COUNT(*)                                   AS n_facturas,
       SUM(monto)                                 AS monto_total,
       AVG(monto)                                 AS monto_prom,
       COUNT(DISTINCT id_pagador)                 AS n_pagadores,
       MAX(fecha)                                 AS ultima_fecha,
       SUM(monto) FILTER (WHERE plazo >= 90)      AS monto_plazo_largo   -- suma condicional
FROM transacciones
GROUP BY id_proveedor;
```

## 4. Ventanas de tiempo antes de una fecha de corte (anti fuga de datos)

```sql
SELECT id,
       SUM(monto) FILTER (WHERE fecha >  DATE '2026-06-30' - INTERVAL 3 MONTH) AS monto_3m,
       SUM(monto) FILTER (WHERE fecha >  DATE '2026-06-30' - INTERVAL 12 MONTH) AS monto_12m
FROM transacciones
WHERE fecha <= DATE '2026-06-30'          -- NADA posterior al corte entra en las variables
GROUP BY id;
```

## 5. Quitar duplicados: quedarse con el registro más reciente

```sql
SELECT * FROM empresas
QUALIFY ROW_NUMBER() OVER (PARTITION BY id ORDER BY fecha_actualizacion DESC) = 1;
```

## 6. Funciones de ventana útiles

```sql
LAG(saldo) OVER (PARTITION BY id ORDER BY mes)            -- valor del mes anterior
AVG(saldo) OVER (PARTITION BY id ORDER BY mes ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)  -- promedio móvil 3 meses
RANK() OVER (ORDER BY valor_esperado DESC)                -- ranking
SUM(monto) OVER (PARTITION BY rubro)                      -- total del grupo en cada fila
```

## 7. Pivotear (filas a columnas)

```sql
PIVOT ventas ON producto USING SUM(monto) GROUP BY id;    -- una columna por producto
```

## 8. Fechas en formatos raros

```sql
TRY_CAST(col AS DATE)                    -- 2026-03-31
TRY_STRPTIME(col, '%d/%m/%Y')::DATE      -- 31/03/2026
TRY_STRPTIME(CAST(col AS VARCHAR), '%Y%m')::DATE   -- 202603
date_trunc('month', fecha)               -- primer día del mes
date_diff('day', fecha_a, fecha_b)       -- días entre fechas
```

## 9. Muestras y control de memoria

```sql
SET memory_limit = '4GB';                -- lo que no cabe va a disco en vez de colgar la laptop
SELECT * FROM t USING SAMPLE 10%;        -- muestra al azar
SELECT * FROM t WHERE y = 1 OR hash(id) % 100 < 20;   -- todos los positivos + 20% de negativos (determinista)
```

## 10. Errores típicos

| Error | Causa | Arreglo |
|---|---|---|
| `Binder Error: column not found` | nombre mal escrito o con mayúsculas/espacios | usar comillas dobles: `"Monto Total"` |
| `Conversion Error` | texto que no es número | `TRY_CAST(col AS DOUBLE)` |
| Más filas después del JOIN | llave repetida | agregar o deduplicar antes |
| Todo NULL después del LEFT JOIN | tipos distintos en la llave (texto vs número) | `CAST(a.id AS VARCHAR) = CAST(b.id AS VARCHAR)` |
| Out of memory | traer todo a pandas | resolver en SQL y traer solo el resultado |
