#  📊 Estrategias de inversion con Machine Learning y Backtesting en ETFs

## 1. Introducción

 El presente proyecto tiene como objetivo principal el diseño, desarrollo y validación de un sistema de inversión cuantitativo híbrido. Este sistema busca resolver el dilema del inversor a largo plazo: cómo superar los rendimientos de una estrategia puramente pasiva, como el Dollar-Cost Averaging (DCA), sin incurrir en los altos riesgos asociados al trading activo tradicional.

Por ello se proponen dentro del proyecto metodologias de inversion basadas en la generación de señales mediante indicadores tecnicos (RSI, Cruce del precio con medias moviles, Nube de Ichimoku, etc ... ) para crear estrategias combinadas de una inversion activa (Trading) y pasiva (DCA).  

El propósito final es entregar no solo una estrategia de trading, sino un marco de trabajo completo para la investigación cuantitativa que demuestra cómo la aplicación del Machine Learning puede crear sistemas de inversión más inteligentes, robustos y adaptados a los objetivos a largo plazo de un inversor moderno.

## 2. Objetivos Específicos

- **1. Validar la Viabilidad de Estrategias Basadas en Reglas**: Investigar y realizar un backtest exhaustivo de múltiples estrategias de trading basadas en indicadores técnicos (Ichimoku, MACD, etc.) para establecer un benchmark de rendimiento activo y comprender sus limitaciones.

- **2. Desarrollar un Modelo Predictivo Superior**: Utilizar un abanico de modelos de aprendizaje supervisado (desde Regresión Logística hasta XGBoost) para crear una señal de compra con una precisión estadísticamente superior a la de cualquier estrategia basada en reglas.

- **3.Gestionar el Riesgo de Forma Inteligente**: Demostrar, a través del modelo de Machine Learning, la capacidad de generar una estrategia rentable que funcione con mecanismos de gestión de riesgo (stop-loss), a diferencia de las estrategias de reglas que solo superaron al mercado al asumir un riesgo máximo.

- **4.Crear una Herramienta de Simulación Robusta**: Desarrollar una pipeline de análisis completa y un motor de backtesting que permita evaluar de forma consistente y realista el rendimiento de cualquier estrategia en diferentes activos y bajo distintas condiciones de mercado.

## 3. Estructura del repositorio 

```
├── data/                               # Datos históricos
├── models/                             # Modelos entrenados
|  ├──best_models/                      # Modelos con mejores resultados 
|  ├──models_num/                       # Entrenados con variables numericas
|  ├──models_numbol/                    # Numericas + features boleanas
|  ├──models_numadvan/                  # numericas + features avanzadas (velas japonesas)
|
├── notebooks/                          # Jupyter Notebooks
|  ├── 1.Extract_data_and_EDA.ipynb     # Análisis exploratorio y extracción de datos
|  ├── 2.Feature_engineering.ipynb      # Inginiería de caracteristicas 
|  ├── 3.Initial_Backtesting.ipynb      # Backtesting inicial
|  ├── 4.Machine_learning_models.ipynb  # Entrenamiento de modelos de ML
|  ├── 5.Final_BackTest.ipynb           # Backtesting con datos test
|  ├── 6.Testing_ETFs.ipynb             # Backtest en diferentes ETFs con señales ML.
|
├── reports                             # Graficos y tablas de resultados 
├── src/                                # Código fuente (pipeline, funciones, indicadores)
├── README.md            
└── requirements.txt                    # Dependencias del proyecto
 ```
## 4. Funcionammiento

- **Fase 1: Extraccion y exploracion de los datos obtenidos**:
  - Se extraen los datos 'OHCL' con ``` yfinance ```.
  - Añadimos los indicadores tecnicos que queremos estudiar.
  - Exploramos la distribición de los retornos diarios, datos estadisticos, volatilidad del producto y drawdown. 
- **Fase 2: Ingieneria de caracteristicas**:
  - Creamos las señales a partir de los indicadores tecnicos: 
    - Señales basadas en las diferentes estrategias combinando los indicadores tecnicos. 
    - Señales numericas. 
    - Señales avanzadas. 
    - Definimos un target para posteriormente pasarlo a los modelos de ML.
- **Fase 3: Backtest inicial**:
  - Evaluamos mediante un backtest inicial las diferentes estrategias de compra basadas en los indicadores tecnicos asi como en la combinacion de estos: 

  ` ema_singal_price `: momento en el que el precio cruza la media movil de forma ascendente y el rsi está en un rango especifico.\
  ` macd_signal `: Genera una señal en el momento en el que la linea del macd cruza en hacia arriba la linea de la señal del macd y el precio de cierr se situa por encima de estas.\
  ` stochastic_oversold_signal `: Genera la señal el momendo que detecta un estado de sobreventa en el mercado.\
  ` ichimoku_signal_kijun_cross `: Genera una señal de compra basada en el cruce del precio de cierre sobre la línea Kijun de Ichimoku,
    filtrada por la posición del precio respecto a Senkou Span A y un rango específico de RSI\

  - Evaluamos las diferentes estrategias de inversion: 

   - **Dca Pura** :  Solo se invierte una cantidad fija al mes proveniente de la cuenta bancaria del inversor. No tiene cuota de entrada. 
   - **Dca + StopLoss(SL) y Takeprofit Fijos(TP)**: 
      - Sigue teniendo una aportacion mensual(DCA) la cual sale de la cuenta externa del inversor.
      - Se incluye un TP y SL fijos definidos en los parametros de la funcion. 
      - Incluye una cuota de entrada inicial, de la cual el sistema utilizará el dinero para realizar las compras (trades) y un momento se llegue al TP o SL, se volvera a incluir en la misma variable con el fin de que sea el fondo para hacer trades.
   - **Dca + StopLoss y Takeprofit dinámicos**:
      - Mismo funcionamiento que la anterior.
      - Se diferencia que tanto el SL como el TP, varian en funcion del ATR(indicador de volatilidad) que actuara como multiplicador.
   - **Dca + Trades sin StopLoss**: 
      - Se hacen compras (trades) pero como asumimos que la inversion es a largo plazo asumimos el riesgo de retirar el SL. 
      - Se hacen comprar mensuales (DCA)
   - **DCA Selfsufficient**: 
      - Solo se incluye una cuota unica, superior a las anteriores. 
      - No se extrae dinero externo para las aportaciones de DCA. 
      - De la cuota de entrada se extrae: 
          - 1.Aportacion mensual DCA.
          - 2.Aportaciones para los trades: un momento que estos alcanzan TP o SL, el dinero vuelve a la cuenta.
      - Esta estrategia hace que el inversor ingrese una sola cantidad la cual gracias a los trades se va retroalimentando con la finalidad que pasado un tiempo  el incremento del capital supere a la cantidad inical de manera que el inversor juege solo con "dinero de casa". 
   - **Fase 4: Entrenamiendo de modelos de ML (Clasificación)**:
    - Entrenamiendo de modelos de aprendizaje supervisado: 

        - Regresión Logistica.
        - Random Forest.
        - Support Vector Machine
        - Gradient Boosting
        - XGBoost
    - Entrenamiento de modelos de DeepLearning:
        - Redes perceptron Multicapa. (MLP)\

    - Evaluacion de resultados para cada modelo. 

   - **Fase 5: Backtest final sobre datos test y pruebas en otros Productos**:
      - Con las señales generadas por el modelo con mejores resultados se crean nuevas señales de compra para comprobar los resultados de cada estrategia de inversion. (`5.Final_Backtest.ipynb`)
      - Comprobamos el comportamiento de las estrategias en diferentes productos. (`6.Testing_ETFs`)

## 4. Metricas de evaluación:

  - Para el Backtesting: 

    - Valor final del porfolio.
    - Retorno (%) de lo invertido. 
    - Sharpe Ratio 

  - Para modelos de Machine Learning: 

    - Precission
    - Recall
    - F1-Score

## 5. Resultados obtenidos: 

Como resultados nos fijaremos en el Benchmark fijado en el backtest inicial sobre la estrategia de Dolar Cost Average pura, en la cual obteniamos un 170% de retorno en 12 años con un valor final del portfolio $113,480.61.

En el backtest inicial las diferentes estrategias arrojaron los siguientes datos: 
  
  - Estrategia con un StopLoss del 5%, take profit del 12,5%: 
  
    - Valor final del portfolio: $115,382.18
    - Retorno: 168.33%
    - Sharpe Ratio: 1.62
  
  - Estrategia con Stoploss y takeprofit dinamicos basados en el atr: 
  
    - Valor final del portfolio: $115,259.411	
    - Retorno: 168 %
    - Sharpe Ratio: 1.62

  - Estrategia sin stoploss:

    - Valor final del portfolio: $116,532.304
    - Retorno: 171 %
    - Sharpe Ratio: 1.62  

  - Estrategia selfsuficient, aclarar que dentro de esta estrategia solo se hace una aportacion inicial superior de $5000, siendo las aportaciones DCA y los trades de la misma cuantia que las anteriores estrategias, $250 y $150 respecivamente: 

    - Valor final del portfolio: $27,144.625
    - Retorno: 171,44 %
    - Sharpe Ratio: 0.71 
--------------------------------------------------------------------------------------------------------
  En los backtest finales con las señales generadas por el modelo de aprendizaje supervisado: 

  - Estrategia con un StopLoss del 5%, take profit del 12,5%: 
  
    - Valor final del portfolio: $117,987.80
    - Retorno: 180.92%
    - Sharpe Ratio: 1.68
  
  - Estrategia con Stoploss y takeprofit dinamicos basados en el atr: 
  
    - Valor final del portfolio: $118,511.40
    - Retorno: 168 %
    - Sharpe Ratio: 1.69

  - Estrategia sin stoploss:

    - Valor final del portfolio: $118,763.23
    - Retorno: 182.77%
    - Sharpe Ratio: 1.67  

  - Estrategia selfsuficient, aclarar que dentro de esta estrategia solo se hace una aportacion inicial superior de $5000, siendo las aportaciones DCA y los trades de la misma cuantia que las anteriores estrategias, $250 y $150 respecivamente: 

    - Valor final del portfolio: $30,419.53
    - Retorno: 186.98%
    - Sharpe Ratio: 0.76 

# [📄Presentación (PDF)](../trading-ml-strateg/reports/presentacion%20proyecto%20final%20.pdf)



## 🚀 Instalación y Requisitos

Sigue estos pasos para clonar y llevar el proyecto a tu propio entonrno virtual.

```
# 1. Clonar el repositorio

  git clone https://github.com/FranBretones/trading-ml-strateg
  cd trading-ml-strateg``

# 2.Crea un entorno virtual (recomndado Python3.10 o superior.)

   python -m venv .venv

# 3.Activa el entorno virtual

# Mac/Linux:

source .venv/bin/activate

# Windows(GitBash/PoweShell)**:

 .venv\Scripts\activate

# Instalar dependencias
# Actualiza pip e instala desde **requirement.txt**:

pip install --upgrade pip
pip install -r requirements.txt

# Actualizar dependencias

pip freeze > requirements.txt
```

## 👨‍💻 Contribución

- Si deseas contribuir:

  **1**. Haz un fork del repositorio

  **2**. Crea una nueva rama ``` git checkout -b feature/nueva-funcionalidad ```

  **3**. Realiza tus cambios y haz commit ``` git commit -m 'Agrego nueva funcionalidad' ```

  **4**. Envía un pull request