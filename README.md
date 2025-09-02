> *Branch creada para analizar el código y hacer comentarios.*

# Indicadores Técnicos - Trading ML Strategy

Este documento describe los indicadores técnicos utilizados en el proyecto para análisis de trading diario. Cada indicador se calcula a partir de datos OHLCV (`open`, `high`, `low`, `close`, `volume`) y se explica su propósito y parámetros estándar.

| Indicador                                              | Descripción                                                                       | Uso común                                                                             | Parámetros estándar                                                         |
| ------------------------------------------------------ | ---------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| **SMA (Simple Moving Average)**                  | Media móvil simple que suaviza los precios históricos.                           | Identificación de tendencias; cruces de medias para señales de compra/venta.         | Ventanas: 20 (corto plazo), 50 (medio plazo), 200 (largo plazo)               |
| **EMA (Exponential Moving Average)**             | Media móvil exponencial que da más peso a los precios recientes.                 | Seguimiento más sensible de la tendencia; se combina con SMA para confirmar señales. | Ventanas: 20, 50, 200                                                         |
| **Ichimoku Cloud**                               | Conjunto de líneas que indican soporte, resistencia y tendencia.                  | Determinar fuerza de la tendencia, soportes y resistencias, zonas de entrada/salida.   | Líneas: Conversion Line (Tenkan-sen), Base Line (Kijun-sen), Senkou Span A/B |
| **ADX (Average Directional Index)**              | Mide la fuerza de la tendencia sin indicar dirección.                             | Identificar si el mercado está en tendencia fuerte o débil.                          | Ventana: 14                                                                   |
| **RSI (Relative Strength Index)**                | Oscilador que mide la velocidad y cambio de los movimientos de precio.             | Detectar sobrecompra (>70) o sobreventa (<30).                                         | Ventana: 14                                                                   |
| **MACD (Moving Average Convergence Divergence)** | Diferencia entre EMA rápida y lenta; incluye línea de señal y histograma.       | Seguimiento de tendencia y momentum; identificar cruces de tendencia.                  | EMA rápida: 12, EMA lenta: 26, Señal: 9                                     |
| **Estocástico (Stochastic Oscillator)**         | Compara el precio de cierre con el rango de precios recientes.                     | Detectar sobrecompra/sobreventa; identificar giros de corto plazo.                     | %K: 14, %D: 3                                                                 |
| **Bollinger Bands**                              | Bandas superior e inferior alrededor de una media móvil que reflejan volatilidad. | Identificar precios extremos y posibles reversiones; medir volatilidad.                | Media móvil: 20, Desviación estándar: 2                                    |
| **ATR (Average True Range)**                     | Mide volatilidad absoluta del activo.                                              | Determinar niveles de stop-loss; medir riesgo y volatilidad.                           | Ventana: 14                                                                   |

## Diccionario de columnas del dataset

Este dataset contiene precios históricos y una serie de indicadores técnicos calculados para análisis y generación de señales de inversión a medio/largo plazo.

### Datos originales (OHLCV)

- **date** → Fecha del registro.
- **open** → Precio de apertura del día.
- **high** → Precio máximo del día.
- **low** → Precio mínimo del día.
- **close** → Precio de cierre del día.
- **volume** → Volumen total de operaciones del día.

### Indicadores técnicos

- **rsi** → *Relative Strength Index* (RSI), mide sobrecompra/sobreventa.
- **macd** → Línea MACD (12 EMA - 26 EMA).
- **macd_signal** → Línea de señal del MACD (9 EMA del MACD).
- **macd_diff** → Diferencia entre MACD y su señal (histograma).
- **ema_50** → Media móvil exponencial de 50 días (tendencia de medio plazo).
- **ema_200** → Media móvil exponencial de 200 días (tendencia de largo plazo).
- **ichimoku_a** → Línea Senkou Span A (uno de los límites de la nube Ichimoku).
- **ichimoku_b** → Línea Senkou Span B (otro límite de la nube Ichimoku).
- **ichimoku_base** → Kijun-sen o línea base (26 periodos).
- **ichimoku_conversion** → Tenkan-sen o línea de conversión (9 periodos).
- **adx** → *Average Directional Index*, mide la fuerza de tendencia.
- **volume_sma_20** → Media simple de volumen de 20 días.
- **volume_ratio** → Ratio entre volumen actual y la media de volumen de 20 días.
- **awesome_osc** → *Awesome Oscillator*, mide momentum a corto/medio plazo.

---

## Notas de uso

1. Todos los indicadores se calculan **diariamente** a partir de datos OHLCV.
2. Este conjunto está pensado para análisis de **tendencia y volatilidad** en horizonte medio-largo plazo.

## Señales

Este proyecto genera señales de trading diarias basadas en indicadores técnicos y divergencias.

| Indicador       | Señal                           | Tipo          | Descripción                                  |
| --------------- | -------------------------------- | ------------- | --------------------------------------------- |
| SMA/EMA         | Golden Cross                     | Compra        | SMA/EMA 50 cruza sobre SMA/EMA 200            |
| SMA/EMA         | Death Cross                      | Venta         | SMA/EMA 50 cruza bajo SMA/EMA 200             |
| SMA/EMA         | Cruce EMA/SMA 20-50              | Compra/Venta  | EMA/SMA 20 cruza EMA/SMA 50 según dirección |
| Ichimoku Cloud  | Precio sobre nube                | Compra        | Tendencia alcista                             |
| Ichimoku Cloud  | Precio bajo nube                 | Venta         | Tendencia bajista                             |
| Ichimoku Cloud  | Tenkan/Kijun cruz                | Compra/Venta  | Señal de cruce según dirección             |
| Ichimoku Cloud  | Chikou Span cruz                 | Confirmación | Confirmación de tendencia                    |
| ADX             | ADX > 25                         | Tendencia     | Tendencia fuerte detectada                    |
| ADX             | +DI cruza -DI                    | Compra        | Indica fuerza de compra                       |
| ADX             | -DI cruza +DI                    | Venta         | Indica fuerza de venta                        |
| RSI             | RSI > 70                         | Venta         | Sobrecompra, posible reversión               |
| RSI             | RSI < 30                         | Compra        | Sobreventa, posible reversión                |
| RSI             | Divergencia alcista              | Compra        | Precio baja mientras RSI sube                 |
| RSI             | Divergencia bajista              | Venta         | Precio sube mientras RSI baja                 |
| MACD            | MACD cruza señal hacia arriba   | Compra        | Señal de compra del MACD                     |
| MACD            | MACD cruza señal hacia abajo    | Venta         | Señal de venta del MACD                      |
| MACD            | Divergencia alcista              | Compra        | Precio baja mientras MACD sube                |
| MACD            | Divergencia bajista              | Venta         | Precio sube mientras MACD baja                |
| Estocástico    | %K/%D cruza en sobreventa (<20)  | Compra        | Señal de reversión alcista                  |
| Estocástico    | %K/%D cruza en sobrecompra (>80) | Venta         | Señal de reversión bajista                  |
| Estocástico    | Divergencia alcista              | Compra        | Precio baja mientras estocástico sube        |
| Estocástico    | Divergencia bajista              | Venta         | Precio sube mientras estocástico baja        |
| Bollinger Bands | Precio rompe banda superior      | Venta         | Posible reversión a la baja                  |
| Bollinger Bands | Precio rompe banda inferior      | Compra        | Posible reversión al alza                    |
| Bollinger Bands | Reversión hacia banda media     | Ajuste        | Posible ajuste de tendencia                   |
| ATR             | Cambio abrupto en volatilidad    | Precaución   | Ajustar stops o tamaño de posición          |
