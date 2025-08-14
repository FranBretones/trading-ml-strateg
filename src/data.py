
# =============== Packages needed =============== #

import pandas as pd 
import numpy as np
#libreria para descargar datos de productos financieros
import yfinance as yf 
# Calculo de indicadores tecnicos 'ta package' 
import ta as ta  
from ta.momentum import RSIIndicator, StochasticOscillator,AwesomeOscillatorIndicator
from ta.trend import MACD, ADXIndicator,IchimokuIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.trend import SMAIndicator, EMAIndicator

# =============== DATA EXTRACTION =============== #

def extract_clean_ticker(ticker: str, start=None, end=None, interval='1d') -> pd.DataFrame:
    """
    Descarga y limpia datos históricos de un ticker con yfinance.
    
    Args:
        ticker (str): Símbolo del ticker (ej: 'VOO', '^GSPC')
        start (str): Fecha de inicio 'YYYY-MM-DD'
        end (str): Fecha de fin 'YYYY-MM-DD'
        interval (str): Intervalo temporal ('1d', '1wk', etc.)
        
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
    

    if isinstance(df.columns, pd.MultiIndex):             # Si las columnas del DataFrame son un MultiIndex,
        df.columns = df.columns.droplevel(1)              # se elimina el segundo nivel del MultiIndex para simplificar.
        df.columns = df.columns.get_level_values(0)       # se selecciona solo el primer nivel de nombres de columna para simplificar el DataFrame.
    
    df = df.reset_index()
    df.rename(columns={"Date": "date"}, inplace=True)
    df['date'] = pd.to_datetime(df['date'])
    df.columns = [col.lower() for col in df.columns]
    df = df.sort_values('date')                           # datos ordenados por fecha
    df = df.round(3)                                      # Redondear los valores float a 3 decimales
    
    # Asegurarse de que las columnas necesarias están presentes
    cols_esperadas = {'open', 'high', 'low', 'close', 'volume'}
    if not cols_esperadas.issubset(df.columns):
        raise ValueError(f"Faltan columnas necesarias en {ticker}: {cols_esperadas - set(df.columns)}")
    
    return df

# =============== Technical Indicators =============== #

def añadir_indicadores_tecnicos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula e incorpora indicadores técnicos 
    al DataFrame de precios OHLCV para análisis diario.
    
    Parámetros:
    -----------
    df : pd.DataFrame
        DataFrame con columnas 'open', 'high', 'low', 'close', 'volume'.
    
    Retorna:
    --------
    pd.DataFrame
        DataFrame con columnas adicionales para cada indicador y divergencias.
    
    Indicadores incluidos:
    ---------------------
    - RSI (Relative Strength Index) : fuerza relativa de precios.
    - MACD : tendencia y momentum; columnas macd, macd_signal, macd_diff.
    - Estocástico (%K, %D) : sobrecompra/sobreventa.
    - Bandas de Bollinger : volatilidad y niveles de soporte/resistencia dinámicos.
    - ATR (Average True Range) : volatilidad real del mercado.
    - SMA y EMA : medias móviles simples y exponenciales.
    - ADX : fuerza de tendencia.
    - Divergencias de RSI y MACD filtradas por volumen.
    """
    df = df.copy()

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
    df['stochastic_d'] = stoch.stoch_signal()  # Línea %D

    # Bandas de Bollinger
    bb = BollingerBands(close=df['close'], window=20, window_dev=2)
    df['bollinger_hband'] = bb.bollinger_hband()  # Banda superior
    df['bollinger_lband'] = bb.bollinger_lband()  # Banda inferior
    df['bollinger_mavg'] = bb.bollinger_mavg()    # Media central
    df['bollinger_width'] = bb.bollinger_wband()  # Ancho de banda : valores entre hband - lband

    # ATR
    atr = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14)
    df['atr'] = atr.average_true_range()  # Medida de volatilidad

    # SMA y EMA
    df['sma_20'] = SMAIndicator(close=df['close'], window=20).sma_indicator()
    df['sma_50'] = SMAIndicator(close=df['close'], window=50).sma_indicator()
    df['sma_200'] = SMAIndicator(close=df['close'], window=200).sma_indicator()
    df['ema_20'] = EMAIndicator(close=df['close'], window=20).ema_indicator()
    df['ema_50'] = EMAIndicator(close=df['close'], window=50).ema_indicator()
    df['ema_200'] = EMAIndicator(close=df['close'], window=200).ema_indicator()
    
    # Ichimoku Cloud: identifica soporte, resistencia y tendencia
    ichimoku = IchimokuIndicator(high=df['high'], low=df['low'], window1=9, window2=26, window3=52)
    df['ichimoku_a'] = ichimoku.ichimoku_a()
    df['ichimoku_b'] = ichimoku.ichimoku_b()
    df['ichimoku_base'] = ichimoku.ichimoku_base_line()
    df['ichimoku_conversion'] = ichimoku.ichimoku_conversion_line()

    # ADX
    adx = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14)
    df['adx'] = adx.adx()  # Fuerza de tendencia

    # Volumen relativo
    df['volume_sma_20'] = SMAIndicator(close=df['volume'], window=20).sma_indicator()
    df['volume_ratio'] = df['volume'] / df['volume_sma_20']  # Para usar en filtrado de divergencias
    
        # Awesome Oscillator: mide la presión del mercado y puede usarse para divergencias con volumen
    ao = AwesomeOscillatorIndicator(high=df['high'], low=df['low'])
    df['awesome_osc'] = ao.awesome_oscillator()
    
    # Redondear todas las columnas numéricas a 3 decimales
    df = df.round(3)
    # Eliminar filas con valores nulos por cálculos
    df = df.dropna().reset_index(drop=True)

    return df

# =============== Buy Signal data generation =============== #

