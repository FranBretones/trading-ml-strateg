#Libraries

#Librerias sistema
import os
import sys
from pathlib import Path

# Manipuladcion de datos 
import pandas as pd 
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import matplotlib.pyplot as plt

#libreria para descargar datos de productos financieros
import yfinance as yf 

# Calculo de indicadores tecnicos 'ta package' 
import ta as ta  
from ta.momentum import RSIIndicator, StochasticOscillator,AwesomeOscillatorIndicator
from ta.trend import MACD, ADXIndicator,IchimokuIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.trend import SMAIndicator, EMAIndicator

# Librerias para modelos de ML
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import pickle

# -----------------------------------------------------------------------------
# FUNCIONES DE DESCARGA, LIMPIEZA DE DATOS Y CÁLCULO DE INDICADORES
# -----------------------------------------------------------------------------

def extract_clean_ticker(ticker: str, start=None, end=None, interval=None) -> pd.DataFrame:
    """
    Descarga y limpia datos históricos de un ticker con yfinance y genera todo el dataframe que se utilizara para analizar el stock. 
    
    Args:
        ticker (str): Símbolo del ticker (ej: 'VOO', '^GSPC')
        start (str): Fecha de inicio 'YYYY-MM-DD'
        end (str): Fecha de fin 'YYYY-MM-DD'
        interval (str): Intervalo temporal ('1d', '1wk', '1h', etc.)
        
    Returns:
        pd.DataFrame: DataFrame limpio con columnas estandarizadas.
    """
    df = yf.download(
        tickers=ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,
        progress=False
    )

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    df = df.reset_index()

    # Detectar automáticamente la columna de fecha
    date_col = None
    for col in df.columns:
        if 'date' in col.lower() or 'time' in col.lower():
            date_col = col
            break
    if date_col is None:
        raise ValueError(f"No se encontró columna de fecha en {ticker}")

    df.rename(columns={date_col: "date"}, inplace=True)
    df['date'] = pd.to_datetime(df['date'])
    df['date'] = df['date'].dt.tz_localize(None)
    df.columns = [col.lower() for col in df.columns]
    df = df.sort_values('date')
    df = df.round(3)

    cols_esperadas = {'open', 'high', 'low', 'close', 'volume'}
    if not cols_esperadas.issubset(df.columns):
        raise ValueError(f"Faltan columnas necesarias en {ticker}: {cols_esperadas - set(df.columns)}")

    return df



def añadir_indicadores_tecnicos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula e incorpora indicadores técnicos  y divergencias 
    al DataFrame de precios OHLCV para análisis diario.
    
    Parámetros:
    -----------
    df : pd.DataFrame
        Dataframe con columnas 'open', 'high', 'low', 'close', 'volume'.
        
    Indicadores incluidos:
    ---------------------
    - RSI (Relative Strength Index) : fuerza relativa de precios.
    - MACD : tendencia y momentum; columnas macd, macd_signal, macd_diff.
    - Estocástico (%K, %D) : sobrecompra/sobreventa.
    - Bandas de Bollinger : volatilidad y niveles de soporte/resistencia dinámicos.
    - ATR (Average True Range) : volatilidad real del mercado.
    - SMA y EMA : medias móviles simples y exponenciales.
    - ADX : fuerza de tendencia.
    - Ichimoku Cloud : soporte, resistencia y tendencia.
    - Awesome Oscillator : presión del mercado.
    - Retorno diario : cambio porcentual diario del precio de cierre.
    - Volumen relativo : ratio del volumen actual respecto a su SMA de 20 días.
    
    
        Retorna:
    --------
    pd.DataFrame
        DataFrame con columnas adicionales para cada indicador y divergencias
        calculadas. Filtrado para eliminar filas con NaN.
    Notas:
    -----
    - Asegúrese de que el DataFrame de entrada contiene las columnas necesarias.
    - La función no modifica el DataFrame original; devuelve una copia con los indicadores añadidos.
    - Los indicadores se calculan con ventanas estándar (14, 20, 50, 200 días) pero pueden ajustarse según necesidad.
    - Se redondean los valores a 3 decimales para facilitar la lectura.
    - Se eliminan filas con NaN resultantes del cálculo de indicadores.
    - La función asume que los datos están ordenados cronológicamente por la columna 'date'.
    - Si no se desea modificar el DataFrame original, pasar una copia (por ejemplo, df.copy()).
    """
    df = df.copy()
    df['daily_return'] = df['close'].pct_change()
    # RSI
    rsi = RSIIndicator(close=df['close'], window=14)
    df['rsi'] = rsi.rsi()  # Oscilador de fuerza relativa

    # MACD
    macd = MACD(close=df['close'])
    df['macd'] = macd.macd()  # Línea MACD
    df['macd_signal'] = macd.macd_signal()  # Línea de señal
    df['macd_diff'] = macd.macd_diff()  # Histograma MACD

    # Estocástico
    stoch = StochasticOscillator(high=df['high'], low=df['low'], close=df['close'], window=14, smooth_window=3)
    df['stochastic_k'] = stoch.stoch()  # Línea %K
    df['stochastic_d'] = stoch.stoch_signal()  # Línea %D o linea de confirmacion

    # Bandas de Bollinger
    bb = BollingerBands(close=df['close'], window=20, window_dev=2)
    df['bollinger_hband'] = bb.bollinger_hband()  # Banda superior
    df['bollinger_lband'] = bb.bollinger_lband()  # Banda inferior
    df['bollinger_mavg'] = bb.bollinger_mavg()    # Media central
    df['bollinger_width'] = bb.bollinger_wband()  # Ancho de banda : valores entre hband - lband

    # ATR
    atr = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14)
    df['atr'] = atr.average_true_range()  # Medida de volatilidad

    # EMA
    df['ema_20'] = EMAIndicator(close=df['close'], window=20).ema_indicator()
    df['ema_50'] = EMAIndicator(close=df['close'], window=50).ema_indicator()
    df['ema_200'] = EMAIndicator(close=df['close'], window=200).ema_indicator()
    df['ema_9'] = EMAIndicator(close=df['close'], window=9).ema_indicator()
    
    
    # Ichimoku Cloud: identifica soporte, resistencia y tendencia
    ichimoku = IchimokuIndicator(high=df['high'], low=df['low'], window1=9, window2=26, window3=52)
    df['ichimoku_a'] = ichimoku.ichimoku_a() #linea Senkou Span A 
    df['ichimoku_b'] = ichimoku.ichimoku_b() # linea Senkou Span B
    df['ichimoku_base'] = ichimoku.ichimoku_base_line() # linea Kijun-sen
    df['ichimoku_conversion'] = ichimoku.ichimoku_conversion_line() #linea Tenkan-Sen

    # ADX
    adx = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14)
    df['adx'] = adx.adx()  # Fuerza de tendencia

    # Volumen relativo
    df['volume_sma_20'] = SMAIndicator(close=df['volume'], window=20).sma_indicator()
    df['volume_ratio'] = df['volume'] / df['volume_sma_20']  # Para usar en filtrado de divergencias
    
    # Awesome Oscillator: mide la presión del mercado y puede usarse para divergencias con volumen
    ao = AwesomeOscillatorIndicator(high=df['high'], low=df['low'])
    df['awesome_osc'] = ao.awesome_oscillator()
    

    df = df.round(3)
    df = df.dropna().reset_index(drop=True)

    return df

# -----------------------------------------------------------------------------
# ESTRATEGIAS DE SEÑALES DE COMPRA
# -----------------------------------------------------------------------------

## Estrategia Cruce de Precio sobre Media Móvil (Tendencia): 
# Esta estrategia genera la señal cuando el precio cruza hacia arriba la ema rapida 
# ( esta puede ser la ema = 20, ema=9, etc .. segun le indiquemos) 

def ema_price_signal(df, ema_fast='ema_20', close='close', rsi='rsi'):
    """
    Genera una señal de trading basada en la relación entre el precio de cierre, una media móvil exponencial (EMA) y el Índice de Fuerza Relativa (RSI).
    Esta función calcula la diferencia entre el precio de cierre y una EMA especificada, y crea una columna de señal que indica oportunidades de compra cuando:
        - El precio cruza por encima de la EMA,
        - El RSI está dentro de un rango especificado (por defecto: entre 20 y 50),
        - Las condiciones de cruce se cumplen tanto en la fila actual como en la anterior.
    Parámetros
    ----------
    df : pandas.DataFrame
        DataFrame que contiene al menos las columnas de precio de cierre, EMA y RSI.
    ema_fast : str, opcional
        Nombre de la columna para la EMA rápida (por defecto 'ema_20').
    close : str, opcional
        Nombre de la columna del precio de cierre (por defecto 'close').
    rsi : str, opcional
        Nombre de la columna del RSI (por defecto 'rsi').
    Retorno
    -------
    pandas.DataFrame
        El DataFrame de entrada con dos columnas adicionales:
            - 'ema_price_diff': Diferencia entre precio de cierre y EMA.
            - 'signal_ema_price': Señal booleana que indica dónde se cumplen las condiciones de compra.
    """

    df['ema_price_diff'] = df[close] - df[ema_fast]

    df['signal_ema_price'] = (
        (df['ema_price_diff'] > 0) 
        & (df['ema_price_diff'].shift(1) <= 0) 
        & (df[close] > df[ema_fast]) 
        & (df[close].shift(1) <= df[ema_fast].shift(1)) 
        & (df[rsi].between(20, 50))
        & (df[rsi].shift(1).between(20, 50))
    )

    return df

##  Estrategia Cruce de MACD (Momentum) :  

def macd_signal(df, macd_line='macd', signal_line='macd_signal', ema_long='ema_50', close='close'):
    """
    Genera señales de compra basadas en el cruce del MACD sobre su línea de señal y condiciones adicionales.
    Parámetros:
        df (pandas.DataFrame): DataFrame que contiene las columnas necesarias para el cálculo.
        macd_line (str, opcional): Nombre de la columna que contiene la línea MACD. Por defecto 'macd'.
        signal_line (str, opcional): Nombre de la columna que contiene la línea de señal del MACD. Por defecto 'macd_signal'.
        ema_long (str, opcional): Nombre de la columna que contiene la media exponencial larga (usada como filtro de tendencia). Por defecto 'ema_50'.
        close (str, opcional): Nombre de la columna que contiene los precios de cierre. Por defecto 'close'.
    Comportamiento:
        - Calcula una columna 'macd_diff' = df[macd_line] - df[signal_line].
        - Crea una columna booleana 'signal_macd_buy' que será True cuando se cumplan todas las condiciones:
            1. macd_diff > 0 (MACD por encima de la señal).
            2. macd_diff en el periodo anterior <= 0 (cruce ascendente en el periodo actual).
            3. el valor de macd_line < 0 (el cruce ocurre por debajo de cero).
            4. el precio de cierre (close) está por encima de la EMA larga (ema_long).
        - Modifica el DataFrame de entrada añadiendo las columnas mencionadas y devuelve el mismo DataFrame (operación in-place).
    Retorno:
        pandas.DataFrame: El DataFrame original con las columnas añadidas 'macd_diff' y 'signal_macd_buy'.
    Notas:
        - El DataFrame debe contener las columnas indicadas por los parámetros; de lo contrario se lanzará un KeyError.
        - Si no se desea modificar el DataFrame original, pasar una copia (por ejemplo, df.copy()).
        - La señal 'signal_macd_buy' está diseñada para detectar cruces alcistas del MACD por debajo de cero y filtrar por tendencia positiva (precio > EMA larga).
    Ejemplo de uso:
        df = macd_signal(df, macd_line='macd', signal_line='macd_signal', ema_long='ema_50', close='close')
    """
    
    df['macd_diff'] = df[macd_line] - df[signal_line]
    df['signal_macd_buy'] = (
        (df['macd_diff'] > 0)
        & (df['macd_diff'].shift(1) <= 0)
        & (df[macd_line] < 0)
        & (df[close] > df[ema_long])
    )
    return df

##  Estrategia Oscilador Estocástico (Reversión):  
def stochastic_oversold_signal(df, k_line='stochastic_k', d_line='stochastic_d', ema_long='ema_50', close='close'):
    """
    Genera y añade una señal de compra basada en el estocástico cuando el mercado estuvo sobrevendido
    y se produce un cruce alcista del %K sobre el %D en un contexto de tendencia alcista.

    Parámetros
    ----------
    df : pandas.DataFrame
        DataFrame que debe contener al menos las columnas especificadas por los argumentos
        `k_line`, `d_line`, `ema_long` y `close`. La función añade una columna booleana
        llamada 'signal_stochastic_buy' al propio DataFrame.
    k_line : str, opcional
        Nombre de la columna que contiene la línea %K del estocástico. Por defecto 'stochastic_k'.
    d_line : str, opcional
        Nombre de la columna que contiene la línea %D del estocástico. Por defecto 'stochastic_d'.
    ema_long : str, opcional
        Nombre de la columna que contiene la media móvil exponencial de largo plazo (ej. EMA 50).
        Por defecto 'ema_50'.
    close : str, opcional
        Nombre de la columna de precios de cierre. Por defecto 'close'.

    Valor devuelto
    --------------
    pandas.DataFrame
        El mismo DataFrame de entrada con una nueva columna booleana 'signal_stochastic_buy'.
        True indica que en esa fila se cumplen simultáneamente:
        - En la barra anterior tanto %K como %D estaban por debajo de 20 (condición de sobreventa).
        - En la barra actual %K cruza por encima de %D (cruce alcista).
        - El precio de cierre actual está por encima de la EMA de largo plazo (tendencia alcista).
        La función modifica el DataFrame en sitio y lo devuelve (retorno por conveniencia).

    Excepciones
    ----------
    KeyError
        Si faltan en `df` las columnas especificadas por `k_line`, `d_line`, `ema_long` o `close`.
    """
    is_oversold = (df[k_line].shift(1) < 20) & (df[d_line].shift(1) < 20)
    k_crosses_d_up = (df[k_line] > df[d_line]) & (df[k_line].shift(1) <= df[d_line].shift(1))
    is_uptrend = (df[close] > df[ema_long])
    df['signal_stochastic_buy'] = is_oversold & k_crosses_d_up & is_uptrend
    return df

## Estrategias con Ichimoku (Equilibrio y Soportes) 

#  Ichimoku Conservadora:  
def ichimoku_buy_signals(df, tenkan='ichimoku_conversion', kijun='ichimoku_base',senkou_a='ichimoku_a', senkou_b='ichimoku_b',close='close', rsi='rsi', adx='adx'):
    """
    Genera una columna de señales de compra basadas en Ichimoku con filtros de RSI y ADX.

    Parámetros
    ----------
    df : pandas.DataFrame
        DataFrame que contiene las columnas necesarias para calcular la señal.
    tenkan : str, opcional
        Nombre de la columna correspondiente a la línea Tenkan/Conversión.
        Por defecto 'ichimoku_conversion'.
    kijun : str, opcional
        Nombre de la columna correspondiente a la línea Kijun/Base.
        Por defecto 'ichimoku_base'.
    senkou_a : str, opcional
        Nombre de la columna para Senkou Span A (parte de la nube).
        Por defecto 'ichimoku_a'.
    senkou_b : str, opcional
        Nombre de la columna para Senkou Span B (parte de la nube).
        Por defecto 'ichimoku_b'.
    close : str, opcional
        Nombre de la columna de precio de cierre.
        Por defecto 'close'.
    rsi : str, opcional
        Nombre de la columna del RSI.
        Por defecto 'rsi'.
    adx : str, opcional
        Nombre de la columna del ADX.
        Por defecto 'adx'.

    Retorna
    -------
    pandas.DataFrame
        El mismo DataFrame recibido con una nueva columna booleana
        'signal_ichimoku_buy' que vale True cuando se cumple la señal de compra.
        La columna se añade/modifica en el DataFrame proporcionado (operación in-place).

    Comportamiento y lógica
    -----------------------
    La señal 'signal_ichimoku_buy' se activa (True) cuando se cumplen simultáneamente:
    - El precio de cierre está por encima de Senkou Span A y Senkou Span B (precio por encima de la nube).
    - La línea Tenkan está por encima de la línea Kijun.
    - Ha ocurrido un cruce alcista en el período actual respecto al anterior:
    tenkan > kijun y tenkan.shift(1) <= kijun.shift(1).
    - El RSI es menor que 70 (evitar condiciones de sobrecompra).
    - El ADX es mayor que 20 (indica fuerza de la tendencia).

    Notas
    -----
    - Si falta alguna de las columnas especificadas en los argumentos, se producirá un KeyError.
    - La columna añadida es de tipo booleano.
    - Si no desea modificar el DataFrame original, pase una copia: df.copy().
    
    Ejemplo de uso
    -----
        df = ichimoku_buy_signals(df, tenkan='ichimoku_conversion', kijun='ichimoku_base',
                                senkou_a='ichimoku_a', senkou_b='ichimoku_b',
                                close='close', rsi='rsi', adx='adx')
    
    """
    
    df['signal_ichimoku_buy'] = (
        (df[close] > df[senkou_a])
        & (df[close] > df[senkou_b])
        & (df[tenkan] > df[kijun])
        & (df[tenkan].shift(1) <= df[kijun].shift(1))
        & (df[rsi] < 70)
        & (df[adx] > 20)
    )
    return df

# 4.2 Ichimoku Agresiva (Cruce dentro de la nube)
def ichimoku_signal_aggressive(
    df, tenkan='ichimoku_conversion', 
    kijun='ichimoku_base',
    senkou_a='ichimoku_a', senkou_b='ichimoku_b',
    close='close', adx='adx'):
    """
    Genera una señal de compra agresiva basada en el cruce del Tenkan sobre el Kijun dentro de la nube de Ichimoku,
    filtrada por un umbral de ADX para asegurar fuerza de tendencia.
    Parámetros:
    -----------
    df : pandas.DataFrame
        DataFrame que contiene las columnas necesarias para calcular la señal.
    tenkan : str, opcional
        Nombre de la columna correspondiente a la línea Tenkan/Conversión.
        Por defecto 'ichimoku_conversion'.
    kijun : str, opcional
        Nombre de la columna correspondiente a la línea Kijun/Base.
        Por defecto 'ichimoku_base'.
    senkou_a : str, opcional
        Nombre de la columna para Senkou Span A (parte de la nube).
        Por defecto 'ichimoku_a'.
    senkou_b : str, opcional
        Nombre de la columna para Senkou Span B (parte de la nube).
        Por defecto 'ichimoku_b'.
    close : str, opcional
        Nombre de la columna de precio de cierre.
        Por defecto 'close'.
    adx : str, opcional
        Nombre de la columna del ADX.
        Por defecto 'adx'.
    Retorna:
    --------
    pandas.DataFrame
        El mismo DataFrame recibido con una nueva columna booleana
        'signal_ichimoku_aggressive' que vale True cuando se cumple la señal de compra agresiva.
        La columna se añade/modifica en el DataFrame proporcionado (operación in-place).
    Comportamiento y lógica:
    La señal 'signal_ichimoku_aggressive' se activa (True) cuando se cumplen simultáneamente:
    - La línea Tenkan cruza por encima de la línea Kijun en el período actual respecto al anterior:
    tenkan > kijun y tenkan.shift(1) <= kijun.shift(1).
    - El precio de cierre está dentro de la nube de Ichimoku, es decir, entre Senkou Span A y Senkou Span B.
    - El ADX es mayor que 18, indicando una tendencia con suficiente fuerza.
    Notas:
    -----
    - Si falta alguna de las columnas especificadas en los argumentos, se producirá un KeyError.
    - La columna añadida es de tipo booleano.
    - Esta señal es más agresiva ya que busca entradas dentro de la nube, lo que puede implicar mayor riesgo.
    
    - Se recomienda usar esta señal en combinación con otras confirmaciones para mejorar la fiabilidad.
    """
    
    inside_cloud = (
        (df[close] > df[[senkou_a, senkou_b]].min(axis=1)) &
        (df[close] < df[[senkou_a, senkou_b]].max(axis=1))
    )
    
    df['signal_ichimoku_aggressive'] = (
        (df[tenkan] > df[kijun])
        & (df[tenkan].shift(1) <= df[kijun].shift(1))
        & inside_cloud
        & (df[adx] > 18)
    )
    return df

# 4.3 Ichimoku Cruce del Kijun (Señal temprana de equilibrio)

def ichimoku_signal_kijun_cross(df, kijun='ichimoku_base', senkou_a='ichimoku_a',close='close', rsi='rsi'):
    """
    Genera una señal de compra basada en el cruce del precio de cierre sobre la línea Kijun de Ichimoku,
    filtrada por la posición del precio respecto a Senkou Span A y un rango específico de RSI.
    Parámetros:
    -----------
    df : pandas.DataFrame
        DataFrame que contiene las columnas necesarias para calcular la señal.
    kijun : str, opcional
        Nombre de la columna correspondiente a la línea Kijun/Base.
        Por defecto 'ichimoku_base'.
    senkou_a : str, opcional
        Nombre de la columna para Senkou Span A (parte de la nube).
        Por defecto 'ichimoku_a'.
    close : str, opcional
        Nombre de la columna de precio de cierre.
        Por defecto 'close'.
    rsi : str, opcional
        Nombre de la columna del RSI.
        Por defecto 'rsi'.
    Retorna:
    --------
    pandas.DataFrame
        El mismo DataFrame recibido con una nueva columna booleana
        'signal_ichimoku_kijun_cross' que vale True cuando se cumple la señal de compra.
        La columna se añade/modifica en el DataFrame proporcionado (operación in-place).
    Comportamiento y lógica:
    La señal 'signal_ichimoku_kijun_cross' se activa (True) cuando se cumplen simultáneamente:
    - El precio de cierre cruza por encima de la línea Kijun en el período actual respecto al anterior:
    close > kijun y close.shift(1) <= kijun.shift(1).
    - El precio de cierre está por encima de Senkou Span A (indicando una tendencia potencialmente alcista).
    - El RSI está entre 40 y 60, sugiriendo un mercado equilibrado sin condiciones extremas de sobrecompra o sobreventa.
    Notas:
    -----
    - Si falta alguna de las columnas especificadas en los argumentos, se producirá un KeyError.
    - La columna añadida es de tipo booleano.
    - Esta señal es una indicación temprana de posible equilibrio y cambio de tendencia.
    - Se recomienda usar esta señal en combinación con otras confirmaciones para mejorar la fiabilidad.
    """
    
    df['signal_ichimoku_kijun_cross'] = (
        (df[close] > df[kijun])
        & (df[close].shift(1) <= df[kijun].shift(1))
        & (df[close] > df[senkou_a])
        & (df[rsi].between(40, 60))
    )
    return df

# 4.4 Ichimoku Ruptura del Chikou Span (Confirmación rápida)

def ichimoku_signal_chikou_break(df, tenkan='ichimoku_conversion', kijun='ichimoku_base',close='close'):
    
    """
    Genera una señal de compra basada en el cruce del Chikou Span (precio retrasado) sobre el precio pasado,
    filtrada por la posición del Tenkan y Kijun.
    
    Parámetros:
    -----------
    df : pandas.DataFrame
        DataFrame que contiene las columnas necesarias para calcular la señal.
    tenkan : str, opcional
        Nombre de la columna correspondiente a la línea Tenkan/Conversión.
        Por defecto 'ichimoku_conversion'.
    kijun : str, opcional
        Nombre de la columna correspondiente a la línea Kijun/Base.
        Por defecto 'ichimoku_base'.
    close : str, opcional
        Nombre de la columna de precio de cierre.
        Por defecto 'close'.
        
    Retorna:
    --------
    pandas.DataFrame
        El mismo DataFrame recibido con una nueva columna booleana
        'signal_ichimoku_chikou' que vale True cuando se cumple la señal de compra.
        La columna se añade/modifica en el DataFrame proporcionado (operación in-place).
    Comportamiento y lógica:
    La señal 'signal_ichimoku_chikou' se activa (True) cuando se cumplen simultáneamente:
    - El Chikou Span (precio de cierre desplazado 26 períodos hacia atrás)
    cruza por encima del precio de cierre pasado: chikou_span > past_price
    - La línea Tenkan está por encima de la línea Kijun.
    - Ha ocurrido un cruce alcista en el período actual respecto al anterior:
    tenkan > kijun y tenkan.shift(1) <= kijun.shift(1).
    
    Notas:
    -----
    - Si falta alguna de las columnas especificadas en los argumentos, se producirá un Key
    - La columna añadida es de tipo booleano.
    - Esta señal es una confirmación rápida basada en el Chikou Span.
    - Se recomienda usar esta señal en combinación con otras confirmaciones para mejorar la fiabilidad.
    """
    
    chikou_span = df[close].shift(-26)
    past_price = df[close]
    df['signal_ichimoku_chikou'] = (
        (chikou_span > past_price)
        & (chikou_span.shift(1) <= past_price.shift(1))
        & (df[close] > df[tenkan])
        & (df[tenkan] > df[kijun])
    )
    return df

def bollinger_reversion_signal(df, close_col='close', lband_col='bollinger_lband',ema_long='ema_20', rsi_col='rsi'):
    """
    Genera una señal de compra por reversión a la media usando Bandas de Bollinger.
    
    Parámetros:
    -----------
        df (pandas.DataFrame): DataFrame que contiene las columnas necesarias para el cálculo.
        close_col (str, opcional): Nombre de la columna que contiene los precios de cierre. Por defecto 'close'.
        lband_col (str, opcional): Nombre de la columna que contiene la banda inferior de Bollinger. Por defecto 'bollinger_lband'.
        ema_long (str, opcional): Nombre de la columna que contiene la media exponencial larga (usada como filtro de tendencia). Por defecto 'ema_20'.
        rsi_col (str, opcional): Nombre de la columna que contiene el RSI. Por defecto 'rsi'.

    Comportamiento:
    -----------
        - Calcula si el precio de cierre cruza por debajo de la banda inferior de Bollinger.
        - Crea una columna booleana 'signal_bollinger_buy' que será True cuando se cumplan todas las condiciones:
            1. El precio de cierre cruza por debajo de la banda inferior de Bollinger (de arriba hacia abajo).
            2. El precio de cierre está por encima de la EMA larga (tendencia alcista).
            3. El RSI está por debajo de 30 (condición de sobreventa).
        - Modifica el DataFrame de entrada añadiendo la columna mencionada y devuelve el mismo DataFrame (operación in-place).
        
    Retorno:
    -----------
        pandas.DataFrame: El DataFrame original con la columna añadida 'signal_bollinger_buy'.
    Notas:
    -----------
        - El DataFrame debe contener las columnas indicadas por los parámetros; de lo contrario se lanzará un KeyError.
        - Si no se desea modificar el DataFrame original, pasar una copia (por ejemplo, df.copy()).
        - La señal 'signal_bollinger_buy' está diseñada para detectar condiciones de sobreventa en una tendencia alcista.
    """
    
    price_crosses_lband = (df[close_col] < df[lband_col]) & (df[close_col].shift(1) >= df[lband_col].shift(1))
    is_uptrend = df[close_col] > df[ema_long]
    is_oversold = df[rsi_col] < 30
    
    df['signal_bollinger_buy'] = price_crosses_lband & is_uptrend & is_oversold
    
    return df




def create_combined_signal(df, signal1, signal2, window=3):
    """
    Esta funcion crea las señales a partir de una lista que le propocionaremos con las combinaciones de las señales que queremos que se cumplan 
    siendo TRUE ambas, por ejemplo que se cumpla la señal de compra del estocastico junto con la del macd y genera uuna nueva columna dentro del dataframe. 
    La nueva columna se nombra dinámicamente, ej: 'signal_macd_&_stochastic'.
    
    Parametros:
    -----------
    
        df (pd.DataFrame): El DataFrame que ya contiene las señales individuales.
        signal1 (str): El nombre de la primera señal (la principal).
        signal2 (str): El nombre de la segunda señal (la de confirmación).
        window (int): La ventana de días para la señal de confirmación.
        
    Retorna:
    -----------
        tuple: Un DataFrame actualizado con la nueva señal combinada y el nombre de la nueva señal.
        Si no se encuentran las columnas de señales, devuelve el DataFrame original y None.
    Notas:
    -----------
        - La función asume que las señales en el DataFrame son booleanas (True/False).
        - La nueva señal será True solo cuando 'signal1' sea True y 'signal2
            haya sido True al menos una vez en los últimos 'window' días.
        - Si no se encuentran las columnas de señales especificadas, se imprime un mensaje de error y se devuelve el DataFrame original.
    """
    
    name1 = signal1.replace('signal_', '').replace('_buy', '')
    name2 = signal2.replace('signal_', '').replace('_buy', '')
    new_signal_name = f"signal_{name1}_&_{name2}"
    
    
    if signal1 not in df.columns or signal2 not in df.columns:
        print(f"Error: No se encontraron las columnas de señales '{signal1}' o '{signal2}'.")
        return df, None

    
    confirmation_in_window = df[signal2].rolling(window=window, min_periods=1).sum() > 0
    df[new_signal_name] = (df[signal1] == True) & (confirmation_in_window == True)
    
    print(f"Señal combinada '{new_signal_name}' creada.")
    
    return df, new_signal_name


# -----------------------------------------------------------------------------
# FUNCIONES DE BACKTESTING
# -----------------------------------------------------------------------------

def generar_estrategias_combinadas(df: pd.DataFrame, combinaciones_a_crear: list, window: int = 3) -> tuple[pd.DataFrame, list]:
    """
    Funcion que actua como coordinador para crear las estrategias combinadas del dataframe con la funcion 'create_combined_signal'

    Args:
        df : El DataFrame que ya contiene las señales individuales.
        combinaciones_a_crear (list): Una lista de tuplas, donde cada tupla contiene dos nombres de señales a combinar (principal, confirmación).
        window (int): La ventana de días para la señal de confirmación, poniendole el nombre de window seguimos la nomenclatura de los prametros
                        de la libreria TA dentro para una mejor comprension del trabajo. 

    Returns:
        nos devolvera las columnas con las señales generadas dentro de nuestro dataframe. 
    """
    estrategias_combinadas_creadas = []
    
    print(" Creando Estrategias Combinadas")
    
    
    for signal_principal, signal_confirmacion in combinaciones_a_crear:
        
        
        df, nueva_senal = create_combined_signal(
            df,
            signal1=signal_principal,
            signal2=signal_confirmacion,
            window=window
        )
        
        
        if nueva_senal:
            estrategias_combinadas_creadas.append(nueva_senal)

    print("Se han añadido las siguientes columnas de señales combinadas al dataframe:")
    for nombre in estrategias_combinadas_creadas:
        print(f"- {nombre}")
        
    return df, estrategias_combinadas_creadas


def backtest_dca_pure(df, price_col='close', date_col='date', monthly_invest=200):
    """
    Calcula el rendimiento de una estrategia de inversión pura de
    Aportaciones Periódicas (Dollar-Cost Averaging), es decir seria la estrategia basica (benchmark) que utiliaremos para evaluar la efectividad del trabajo.
    Aqui simplemente el inversor automatiza una inversion periodica todos los meses del año, con el fin de mejorar su patrimonio.
    No se realizan operaciones de trading, solo se acumulan acciones con las aportaciones mensuales.
    
    Parámetros:
    -----------
    
        df (pandas.DataFrame): DataFrame que contiene las columnas necesarias para el cálculo.
        price_col (str, opcional): Nombre de la columna que contiene los precios de cierre
        date_col (str, opcional): Nombre de la columna que contiene las fechas.
        monthly_invest (float, opcional): Monto fijo a invertir cada mes. Por
        defecto 200.

    Comportamiento:
    -----------
        - Convierte la columna de fechas a formato datetime.
        - Itera sobre cada fila del DataFrame, realizando una aportación mensual fija
        cuando cambia el mes.
        - Acumula acciones compradas con cada aportación mensual.
        - Calcula el valor del portafolio en cada fecha y lo almacena en una lista.
        - Al final del período, calcula el valor final del portafolio, el total aportado,
        el retorno absoluto y el retorno porcentual.
        - Imprime los resultados del backtest.

    Retorno:
    -----------
        tuple: Un diccionario con los resultados del backtest y un DataFrame con el valor del portafolio a lo largo del tiempo.
    Notas:
        - El DataFrame debe contener las columnas indicadas por los parámetros; de lo contrario se lanzará un KeyError.
        - La función asume que las fechas en `date_col` están en formato datetime o pueden convertirse a datetime.
        - Si no se desea modificar el DataFrame original, pasar una copia (por ejemplo, df.copy()).   
    """
    
    df[date_col] = pd.to_datetime(df[date_col])
    
    
    dca_shares = 0
    total_contributions = 0
    last_month = None
    portfolio_values = []
    contributions_history = []

    
    for i, row in df.iterrows():
        current_price = row[price_col]
        current_date = row[date_col]
        
        # logica para la aportacion mensual
        if last_month is None or current_date.month != last_month:
            dca_shares += monthly_invest / current_price
            total_contributions += monthly_invest
            last_month = current_date.month
        
        portfolio_values.append(dca_shares * current_price)
        contributions_history.append(total_contributions)
            
    final_value = portfolio_values[-1]
    final_contributions = contributions_history[-1]
    absolute_return = final_value - final_contributions
    percentage_return = (absolute_return / final_contributions) * 100 if final_contributions > 0 else 0

    results = {
        'Strategy': 'DCA Puro (Benchmark)',
        'Final Portfolio Value': final_value,
        'Total Contributions': final_contributions,
        'Absolute Return': absolute_return,
        'Percentage Return': percentage_return,
        'Trading PnL': 0,
        'Trading Capital Used': 0,
        'Open Trades at End': 0,
        'Value of Open Trades': 0
    }
    
    print(" Resultados para la estrategia: DCA Puro ")
    print(f"Valor Final del Portafolio: ${final_value:,.2f}")
    print(f"Total Aportado: ${final_contributions:,.2f}")
    print(f"Retorno Porcentual: {percentage_return:.2f}%\n")

    return results, df.assign(portfolio_value=portfolio_values, contributions=contributions_history)



def backtest_dca_plus_trading_WITH_SL(df, price_col='close', 
                                    signal_col='signal_ema_price',monthly_invest=200, 
                                    trade_amount=50,take_profit_pct=0.125, stop_loss_pct=-0.05, 
                                    date_col='date', initial_trading_cash=1000, 
                                    verbose=True):
    """
    Ejecuta un backtest CON Stop-Loss y reporta las posiciones que quedan abiertas al final. 
    En esta funcion se incluye la funcionalidad del DCA puro pero agregandole la funcionalidad de crear un stop loss y take profit, 
    automatiza que el inversor cuando el sistema detecta una oportunidad de compra en un punto en el que se activan las señales generadas,
    compra estableciendo un umbral en el que le proteja de posibles caidas con el fin de no quedarse pillado en una inversion en el caso de una gran caida, 
    recuperando posteriormente en un punto mas alto con el fin de ir juntando capital. 
    
    Parte de una inversion inicial en la cual ira jugando con ese capital para hacer estas inversiones de modo que con el tiempo este capital se recupere 
    y el inversor solo trabaje con el dinero generado con la estrategia, de modo que el riesto sobre el capital propio se vea reducido a 0 y este sea el producido por el mismo sistema. 
    
    Parámetros:
    -----------
        df (pandas.DataFrame): DataFrame que contiene las columnas necesarias para el cálculo.
        price_col (str, opcional): Nombre de la columna que contiene los precios de cierre
        signal_col (str, opcional): Nombre de la columna que contiene las señales de compra. Por defecto 'signal_ema_price'.
        monthly_invest (float, opcional): Monto fijo a invertir cada mes. Por defecto 200.
        trade_amount (float, opcional): Monto fijo a invertir en cada operación de trading. Por defecto 50.
        take_profit_pct (float, opcional): Porcentaje de ganancia para tomar beneficios. Por defecto 0.125 (12.5%).
        stop_loss_pct (float, opcional): Porcentaje de pérdida para activar el stop loss. Por defecto -0.05 (-5%).
        date_col (str, opcional): Nombre de la columna que contiene las fechas.
        initial_trading_cash (float, opcional): Capital inicial disponible para trading. Por defecto 1000.
        verbose (bool, opcional): Si es True, imprime los resultados del backtest. Por defecto True.
    Comportamiento:
    -----------
        - Convierte la columna de fechas a formato datetime.
        - Itera sobre cada fila del DataFrame, realizando una aportación mensual fija
        cuando cambia el mes.
        - Acumula acciones compradas con cada aportación mensual.
        - Cuando se activa una señal de compra y hay suficiente efectivo, realiza una operación de trading.
        - Cada operación de trading tiene un take profit y un stop loss basados en los porcentajes especificados.
        - Calcula el valor del portafolio en cada fecha y lo almacena en una lista.
        - Al final del período, calcula el valor final del portafolio, el total aportado,
        el retorno absoluto y el retorno porcentual.
        - Imprime los resultados del backtest si verbose es True.
    Retorno:
    -----------
        dict: Un diccionario con los resultados del backtest.
    Notas:
        - El DataFrame debe contener las columnas indicadas por los parámetros; de lo contrario se lanzará un KeyError.
        - La función asume que las fechas en `date_col` están en formato datetime o pueden convertirse a datetime.
        - Si no se desea modificar el DataFrame original, pasar una copia (por ejemplo, df.copy()).
    """
    df[date_col] = pd.to_datetime(df[date_col])
    
    cash = initial_trading_cash
    dca_shares = 0
    open_trades = []
    total_dca_contributions = 0
    trading_contributions = 0
    trading_pnl = 0
    last_month = None
    portfolio_values = []
    daily_returns = []

    for i, row in df.iterrows():
        current_price = row[price_col]
        current_date = row[date_col]
        
        if last_month is None or current_date.month != last_month:
            dca_shares += monthly_invest / current_price
            total_dca_contributions += monthly_invest
            last_month = current_date.month
            
        remaining_trades = []
        for trade in open_trades:
            return_pct = (current_price - trade['buy_price']) / trade['buy_price']
            if return_pct >= take_profit_pct or return_pct <= stop_loss_pct:
                sell_value = trade['shares'] * current_price
                cash += sell_value
                profit_or_loss = sell_value - (trade['shares'] * trade['buy_price'])
                trading_pnl += profit_or_loss
            else:
                remaining_trades.append(trade)
        open_trades = remaining_trades

        if row[signal_col] and cash >= trade_amount:
            shares_bought = trade_amount / current_price
            cash -= trade_amount
            trading_contributions += trade_amount
            open_trades.append({'buy_price': current_price, 'shares': shares_bought})
        
        # Calculate daily portfolio value
        current_portfolio_value = cash + (dca_shares * current_price) + sum(t['shares'] * current_price for t in open_trades)
        portfolio_values.append(current_portfolio_value)
        
        # Calculate daily returns
        if len(portfolio_values) > 1:
            daily_return = (portfolio_values[-1] - portfolio_values[-2]) / portfolio_values[-2]
            daily_returns.append(daily_return)
    
    # Calculate Sharpe Ratio
    if len(daily_returns) > 0:
        risk_free_rate = 0.02  # 2% annual risk-free rate
        daily_rf_rate = (1 + risk_free_rate) ** (1/252) - 1
        excess_returns = [r - daily_rf_rate for r in daily_returns]
        sharpe_ratio = (np.mean(excess_returns) * np.sqrt(252)) / (np.std(excess_returns) if np.std(excess_returns) > 0 else float('inf'))
    else:
        sharpe_ratio = 0
            
    final_price = df[price_col].iloc[-1]
    dca_value = dca_shares * final_price
    
    open_trades_at_end = len(open_trades)
    value_of_open_trades = sum(trade['shares'] * final_price for trade in open_trades)
    
    final_value = cash + dca_value + value_of_open_trades
    final_contributions = total_dca_contributions + initial_trading_cash
    absolute_return = final_value - final_contributions
    percentage_return = (absolute_return / final_contributions) * 100 if final_contributions > 0 else 0

    results = {
        'Strategy': f"{signal_col} (With SL)",
        'Final Portfolio Value': final_value,
        'Total Contributions': final_contributions,
        'Absolute Return': absolute_return,
        'Percentage Return': percentage_return,
        'Trading PnL': trading_pnl,
        'Trading Capital Used': trading_contributions,
        'Open Trades at End': open_trades_at_end,
        'Value of Open Trades': value_of_open_trades,
        'Sharpe Ratio': sharpe_ratio,
        'Total DCA Shares':dca_shares 
    }
    
    if verbose:
        print(f"Resultados para: {signal_col} (Con SL)")
        print(f"Retorno Porcentual: {percentage_return:.2f}%")
        print(f"Posiciones Abiertas al Final: {open_trades_at_end}")
        print(f"Valor de Posiciones Abiertas: ${value_of_open_trades:,.2f}")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}\n")
        print(f"Valor Final del Portafolio: ${final_value:,.2f}")
        print(f"Total Aportado: ${final_contributions:,.2f}\n")
        print(f"Trading PnL: ${trading_pnl:,.2f}")
        print(f"Capital Usado en Trading: ${trading_contributions:,.2f}\n")
        print(f"Total DCA Shares: {dca_shares:,.2f}\n")
    return results


def backtest_with_atr_SL(df, price_col='close', signal_col='signal_ema_price',
                        monthly_invest=200, trade_amount=50,
                        atr_multiplier_tp=3.0,  # Take profit a 3 veces el ATR
                        atr_multiplier_sl=1.5,  # Stop loss a 1.5 veces el ATR
                        date_col='date', initial_trading_cash=1000, atr_col='atr', 
                        verbose=True):
    
    """
    Esta funcion trabaja de una forma muy similar a 'backtest_dca_plus_trading_WITH_SL' pero tomando el indicador Average True Range (ATR) 
    como medida para establecer el stoploss y takeprofit de forma dinamica, es decir, segun el indice de volatilidad dado por este indicador marcara un SL y TP 
    diferentes en cada nivel con el fin de maximizar el beneficio. 
    Parte de una inversion inicial en la cual ira jugando con ese capital para hacer estas inversiones de modo que con el tiempo este capital se recupere
    y el inversor solo trabaje con el dinero generado con la estrategia, de modo que el riesto sobre el capital propio se vea reducido a 0 y este sea el producido por el mismo sistema.
    Parámetros:
    -----------
        df (pandas.DataFrame): DataFrame que contiene las columnas necesarias para el cálculo.
        price_col (str, opcional): Nombre de la columna que contiene los precios de cierre
        signal_col (str, opcional): Nombre de la columna que contiene las señales de compra. Por defecto 'signal_ema_price'.
        monthly_invest (float, opcional): Monto fijo a invertir cada mes. Por defecto 200.
        trade_amount (float, opcional): Monto fijo a invertir en cada operación de trading. Por defecto 50.
        atr_multiplier_tp (float, opcional): Multiplicador del ATR para establecer el take profit. Por defecto 3.0.
        atr_multiplier_sl (float, opcional): Multiplicador del ATR para establecer el stop loss. Por defecto 1.5.
        date_col (str, opcional): Nombre de la columna que contiene las fechas.
        initial_trading_cash (float, opcional): Capital inicial disponible para trading. Por defecto 1000.
        atr_col (str, opcional): Nombre de la columna que contiene el ATR. Por defecto 'atr'.
        verbose (bool, opcional): Si es True, imprime los resultados del backtest. Por defecto True.
    Comportamiento:
    -----------
        - Convierte la columna de fechas a formato datetime.
        - Itera sobre cada fila del DataFrame, realizando una aportación mensual fija
        cuando cambia el mes.
        - Acumula acciones compradas con cada aportación mensual.
        - Cuando se activa una señal de compra y hay suficiente efectivo, realiza una operación de trading.
        - Cada operación de trading tiene un take profit y un stop loss basados en el ATR y los multiplicadores especificados.
        - Calcula el valor del portafolio en cada fecha y lo almacena en una lista.
        - Al final del período, calcula el valor final del portafolio, el total aportado,
        el retorno absoluto y el retorno porcentual.
        - Imprime los resultados del backtest si verbose es True.
    Retorno:
    -----------
        dict: Un diccionario con los resultados del backtest.
    Notas:
        - El DataFrame debe contener las columnas indicadas por los parámetros; de lo contrario se lanzará un KeyError.
        - La función asume que las fechas en `date_col` están en formato datetime o pueden convertirse a datetime.  
        - Si no se desea modificar el DataFrame original, pasar una copia (por ejemplo, df.copy()).
    """
    df[date_col] = pd.to_datetime(df[date_col])
    cash, dca_shares, open_trades, total_dca_contributions, trading_contributions, trading_pnl, last_month = initial_trading_cash, 0, [], 0, 0, 0, None
    portfolio_values = []
    daily_returns = []
    
    for i, row in df.iterrows():
        current_price, current_date = row[price_col], row[date_col]
        
        if last_month is None or current_date.month != last_month:
            dca_shares += monthly_invest / current_price
            total_dca_contributions += monthly_invest
            last_month = current_date.month
            
        remaining_trades = []
        for trade in open_trades:
            # Venda dinamica
            if current_price >= trade['take_profit_price'] or current_price <= trade['stop_loss_price']:
                sell_value = trade['shares'] * current_price
                cash += sell_value
                profit_or_loss = sell_value - (trade['shares'] * trade['buy_price'])
                trading_pnl += profit_or_loss
            else:
                remaining_trades.append(trade)
        open_trades = remaining_trades

        if row[signal_col] and cash >= trade_amount:
            shares_bought = trade_amount / current_price
            cash -= trade_amount
            trading_contributions += trade_amount
            
            # Compra con stops dinamicos
            atr_at_buy = row[atr_col]
            stop_loss_price = current_price - (atr_multiplier_sl * atr_at_buy)
            take_profit_price = current_price + (atr_multiplier_tp * atr_at_buy)
            
            open_trades.append({
                'buy_price': current_price, 
                'shares': shares_bought,
                'stop_loss_price': stop_loss_price,
                'take_profit_price': take_profit_price
            })
        
        # Calculate daily portfolio value
        current_portfolio_value = cash + (dca_shares * current_price) + sum(t['shares'] * current_price for t in open_trades)
        portfolio_values.append(current_portfolio_value)
        
        # Calculate daily returns
        if len(portfolio_values) > 1:
            daily_return = (portfolio_values[-1] - portfolio_values[-2]) / portfolio_values[-2]
            daily_returns.append(daily_return)
    
    # Calculate Sharpe Ratio
    if len(daily_returns) > 0:
        risk_free_rate = 0.02  # 2% annual risk-free rate
        daily_rf_rate = (1 + risk_free_rate) ** (1/252) - 1
        excess_returns = [r - daily_rf_rate for r in daily_returns]
        sharpe_ratio = (np.mean(excess_returns) * np.sqrt(252)) / (np.std(excess_returns) if np.std(excess_returns) > 0 else float('inf'))
    else:
        sharpe_ratio = 0
    
    final_value = cash + (dca_shares * df[price_col].iloc[-1]) + sum(t['shares'] * df[price_col].iloc[-1] for t in open_trades)
    final_contributions = total_dca_contributions + initial_trading_cash
    absolute_return = final_value - final_contributions
    percentage_return = (absolute_return / final_contributions) * 100 if final_contributions > 0 else 0
    open_trades_at_end = len(open_trades)
    value_of_open_trades = sum(trade['shares'] * df[price_col].iloc[-1] for trade in open_trades)
    
    result = {
        'Strategy': f"{signal_col} (ATR Stops)", 
        'Final Portfolio Value': final_value, 
        'Total Contributions': final_contributions, 
        'Absolute Return': absolute_return, 
        'Percentage Return': percentage_return, 
        'Trading PnL': trading_pnl, 
        'Trading Capital Used': trading_contributions, 
        'Open Trades at End': open_trades_at_end, 
        'Value of Open Trades': value_of_open_trades,
        'Sharpe Ratio': sharpe_ratio
    }
    
    if verbose: 
        print(f"Resultados para: {signal_col} (Con SL)")
        print(f"Retorno Porcentual: {percentage_return:.2f}%")
        print(f"Posiciones Abiertas al Final: {open_trades_at_end}")
        print(f"Valor de Posiciones Abiertas: ${value_of_open_trades:,.2f}")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}\n")
        
    return result


def backtest_dca_plus_trading_NO_SL(df, price_col='close', signal_col='signal_ema_price',
                                    monthly_invest=200, trade_amount=50,
                                    take_profit_pct=0.10,
                                    date_col='date', initial_trading_cash=1000, verbose=True):
    """
    Esta funcion ejecuta solo un takeprofit que establecemos previamente, se elimina el stop loss y partimos de la hipotesis en la que en un mercado a largo plazo y predominantemente alcista, 
    el precio tenderá a subir y evitaremos el punto en el que nos quedamos pillados dentro de la compra de una accion. 
    Esta funcion tiene el fin de que el aporte en comprar extraordinarias de acciones se comporte de manera parecida a la estrategia del dca puro pero recuperando lo invertido en un TAKe profit establecido.
    Parte de una inversion inicial en la cual ira jugando con ese capital para hacer estas inversiones de modo que con el tiempo este capital se recupere
    y el inversor solo trabaje con el dinero generado con la estrategia, de modo que el riesto sobre el capital propio se vea reducido a 0 y este sea el producido por el mismo sistema.
    Parámetros:
    -----------
        df (pandas.DataFrame): DataFrame que contiene las columnas necesarias para el cálculo.
        price_col (str, opcional): Nombre de la columna que contiene los precios de cierre
        signal_col (str, opcional): Nombre de la columna que contiene las señales de compra. Por defecto 'signal_ema_price'.
        monthly_invest (float, opcional): Monto fijo a invertir cada mes. Por defecto 200.
        trade_amount (float, opcional): Monto fijo a invertir en cada operación de trading. Por defecto 50.
        take_profit_pct (float, opcional): Porcentaje de ganancia para tomar beneficios. Por defecto 0.10 (10%).
        date_col (str, opcional): Nombre de la columna que contiene las fechas.
        initial_trading_cash (float, opcional): Capital inicial disponible para trading. Por defecto 1000.
        verbose (bool, opcional): Si es True, imprime los resultados del backtest. Por defecto True.
    Comportamiento:
    -----------
        - Convierte la columna de fechas a formato datetime.
        - Itera sobre cada fila del DataFrame, realizando una aportación mensual fija
        cuando cambia el mes.
        - Acumula acciones compradas con cada aportación mensual.
        - Cuando se activa una señal de compra y hay suficiente efectivo, realiza una operación de trading.
        - Cada operación de trading tiene un take profit basado en el porcentaje especificado.
        - Calcula el valor del portafolio en cada fecha y lo almacena en una lista.
        - Al final del período, calcula el valor final del portafolio, el total aportado,
        el retorno absoluto y el retorno porcentual.
        - Imprime los resultados del backtest si verbose es True.
    Retorno:
    -----------
        dict: Un diccionario con los resultados del backtest.
    Notas:
        - El DataFrame debe contener las columnas indicadas por los parámetros; de lo contrario se lanzará un KeyError.
        - La función asume que las fechas en `date_col` están en formato datetime o pueden convertirse a datetime.
        - Si no se desea modificar el DataFrame original, pasar una copia (por ejemplo, df.copy()).
    """
    df[date_col] = pd.to_datetime(df[date_col])
    
    cash = initial_trading_cash
    dca_shares = 0
    open_trades = []
    total_dca_contributions = 0
    trading_contributions = 0
    trading_pnl = 0
    last_month = None
    daily_returns = []
    portfolio_values = []

    for i, row in df.iterrows():
        current_price = row[price_col]
        current_date = row[date_col]
        
        if last_month is None or current_date.month != last_month:
            dca_shares += monthly_invest / current_price
            total_dca_contributions += monthly_invest
            last_month = current_date.month
            
        remaining_trades = []
        for trade in open_trades:
            return_pct = (current_price - trade['buy_price']) / trade['buy_price']
            if return_pct >= take_profit_pct:
                sell_value = trade['shares'] * current_price
                cash += sell_value
                profit = sell_value - (trade['shares'] * trade['buy_price'])
                trading_pnl += profit
            else:
                remaining_trades.append(trade)
        open_trades = remaining_trades

        if row[signal_col] and cash >= trade_amount:
            shares_bought = trade_amount / current_price
            cash -= trade_amount
            trading_contributions += trade_amount
            open_trades.append({'buy_price': current_price, 'shares': shares_bought})
        
        
        current_portfolio_value = cash + (dca_shares * current_price) + sum(trade['shares'] * current_price for trade in open_trades)
        portfolio_values.append(current_portfolio_value)
        
        
        if len(portfolio_values) > 1:
            daily_return = (portfolio_values[-1] - portfolio_values[-2]) / portfolio_values[-2]
            daily_returns.append(daily_return)

    final_price = df[price_col].iloc[-1]
    dca_value = dca_shares * final_price
    
    # Calculate Sharpe Ratio
    if len(daily_returns) > 0:
        risk_free_rate = 0.02 
        daily_rf_rate = (1 + risk_free_rate) ** (1/252) - 1
        excess_returns = [r - daily_rf_rate for r in daily_returns]
        sharpe_ratio = (np.mean(excess_returns) * np.sqrt(252)) / (np.std(excess_returns) if np.std(excess_returns) > 0 else float('inf'))
    else:
        sharpe_ratio = 0

    open_trades_at_end = len(open_trades)
    value_of_open_trades = sum(trade['shares'] * final_price for trade in open_trades)
    
    final_value = cash + dca_value + value_of_open_trades
    final_contributions = total_dca_contributions + initial_trading_cash
    absolute_return = final_value - final_contributions
    percentage_return = (absolute_return / final_contributions) * 100 if final_contributions > 0 else 0

    results = {
        'Strategy': f"{signal_col} (No SL)",
        'Final Portfolio Value': final_value,
        'Total Contributions': final_contributions,
        'Absolute Return': absolute_return,
        'Percentage Return': percentage_return,
        'Trading PnL': trading_pnl,
        'Trading Capital Used': trading_contributions,
        'Open Trades at End': open_trades_at_end,    
        'Value of Open Trades': value_of_open_trades,
        'Sharpe Ratio': sharpe_ratio
    }
    
    if verbose:
        print(f" Resultados para: {signal_col} (Sin SL)")
        print(f"Retorno Porcentual: {percentage_return:.2f}%")
        print(f"Posiciones Abiertas al Final: {open_trades_at_end}")
        print(f"Valor de Posiciones Abiertas: ${value_of_open_trades:,.2f}\n")
        print(f"Valor Final de la Cartera: ${final_value:,.2f}")
        print(f"Trading PnL: ${trading_pnl:,.2f}")
        print(f"Capital Usado en Trading: ${trading_contributions:,.2f}")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}\n")
        print(f"Total DCA Shares: {dca_shares:.4f}")
        print(f"Total DCA Contributions: ${total_dca_contributions + initial_trading_cash:,.2f}\n")

    return results


def evaluate_all_strategies(df_original, signal_columns, backtest_func, backtest_params):
    """
    Esta funcion ejecuta el backtest para una lista de estrategias que definimos con anterioridad y nos devuelve una df comparativo de los resultados de cada una, con el fin de evaluar cual es la estrategia 
    que mejor funciona para posteriormente probarla con los modelos de machine learning entrenados. 
    
    Parametros:
    -----------
    
        df_original (pd.DataFrame):  DataFrame con todos los indicadores calculados.
        signal_columns (list): Una lista con los nombres de las columnas de señales a probar.
        backtest_func (function): La función de backtest a utilizar (ej. backtest_with_sl).
        backtest_params (dict): Un diccionario con los parámetros para el backtest.
    Comportamiento:
    -----------
        - Itera sobre cada columna de señales en `signal_columns`.
        - Para cada señal, crea una copia del DataFrame original para evitar modificaciones no deseadas.
        - Llama a la función de backtest con la copia del DataFrame y los parámetros proporcionados.
        - Almacena los resultados de cada backtest en una lista.
        - Al final, convierte la lista de resultados en un DataFrame y lo ordena por el retorno porcentual.
    Returns:
    -----------
        pd.DataFrame: Un nuevo DataFrame con el resumen de rendimiento de cada estrategia.
    """
    results_list = []
    
    for signal in signal_columns:
        print(f"Evaluando: {signal}...")
        df_copy = df_original.copy()
        
        results = backtest_func(
            df_copy,
            signal_col=signal,
            **backtest_params
        )
        results_list.append(results)
        
    results_df = pd.DataFrame(results_list).sort_values(by='Percentage Return', ascending=False).reset_index(drop=True)
    
    return results_df