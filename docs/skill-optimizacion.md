# Skill: agregar las restricciones que anuncien el día del evento

Según el finalista de 2025, "siempre" viene machine learning + optimización, y las restricciones se
anunciaron el mismo día (a las 4 p. m., antes de empezar). Ejemplo de 2025: "no más de 20,000 clientes
que acepten tarjeta en el canal A".

## La idea en una frase

Elegir a quién contactar para **maximizar el valor esperado total** sin pasarse de los topes.

```
maximizar   Σ VE_i · x_i                 x_i = 1 si contactamos la fila i, 0 si no
sujeto a    Σ x_i ≤ capacidad
            Σ monto_i · x_i ≤ presupuesto
            reglas por grupo (máximos y mínimos)
            a lo más una opción por cliente
```

## Cómo traducir una frase del jurado a config.yaml

| Lo que dicen | Cómo se escribe |
|---|---|
| "Máximo 600 contactos en Lima" | `{columna: region, valor: LIMA, medida: contactos, sentido: max, limite: 600}` |
| "No desembolsar más de S/ 5 millones a MIPE" | `{columna: segmento, valor: MIPE, medida: monto, sentido: max, limite: 5000000}` |
| "Al menos 100 empresas del agro" | `{columna: rubro, valor: AGRO, medida: contactos, sentido: min, limite: 100}` |
| "Presupuesto total de S/ 20 millones" | `presupuesto_monto: 20000000` |
| "Cada cliente recibe una sola oferta" | `una_opcion_por_id: true` (con filas id x producto x canal) |

Si la restricción usa una columna que no está en la lista, agréguenla a `perfil_columnas`
para que llegue hasta `scores.parquet`.

## Qué reportar al jurado

`resumen_negocio.md` trae la tabla **"Lo que cuesta cada restricción"**: cuánto valor esperado se deja
de ganar por cada regla. Una regla que cuesta 0 no aprieta. Una que cuesta mucho es una conversación de
negocio: "el tope de Lima nos cuesta S/ 8 mil de ingreso esperado; si se sube a 700, lo recuperamos".

## Por qué programación lineal y no un solver entero directo

Con pocas restricciones, la solución "relajada" (x entre 0 y 1) casi siempre sale entera, y entonces es
el óptimo exacto. En nuestras pruebas el solver lineal tardó 1.3 s y el entero no terminó en 60 s. El
código redondea si quedan fracciones y reporta la distancia al óptimo teórico (0.00% = óptimo demostrado).

## Caso 2025 (producto x canal)

Si los datos traen una fila por cliente y opción (3 productos x 2 canales = 6 filas por cliente):
1. `segmentar_por: producto_canal` → un modelo por combinación, como hizo el equipo finalista.
2. Cada fila tiene su probabilidad y su monto → su valor esperado.
3. `una_opcion_por_id: true` → la optimización elige la mejor opción de cada cliente respetando los topes.
