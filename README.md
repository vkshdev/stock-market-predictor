# Stock Market Predictor - Live on Streamlit

## Live Deployment
Access the application here: **https://stock-market-predictor-india.streamlit.app/**

Powered by: Streamlit Cloud

---

## Project Overview

**Stock Market Predictor** is an AI-powered machine learning application for predicting Indian stock market prices using an ensemble hybrid algorithm. The system combines XGBoost gradient boosting and Random Forest ensemble methods to provide 1-5 day price forecasts for NIFTY 50, SENSEX, and individual Indian stocks.

The application features:
- Real-time data integration with Yahoo Finance
- 25+ advanced technical indicators
- Ensemble ML algorithm (70% XGBoost + 30% Random Forest)
- Multi-day price predictions with confidence scoring
- Sentiment analysis integration
- Interactive visualizations and performance metrics
- Time series cross-validation for model robustness

**Note**: Predictions take 1-2 minutes due to:
1. Real-time data fetching from Yahoo Finance
2. 50+ feature engineering calculations
3. Model training on multi-year historical data
4. Time series cross-validation (3-5 folds)
5. Streamlit Cloud free tier resource constraints

---

## Core Algorithm Architecture

### 1. Ensemble (XGB+RF) Hybrid Model

The prediction engine uses a sophisticated ensemble approach combining two complementary ML algorithms:

```
INPUT: Market Data (OHLCV)
  |
  V
FEATURE ENGINEERING (50+ Features)
  |
  +-----> XGBoost Regressor (70% weight)
  |       (Captures: Complex patterns, momentum, trends)
  |
  +-----> Random Forest Regressor (30% weight)
  |       (Captures: Robustness, outlier resistance)
  |
  V
WEIGHTED ENSEMBLE PREDICTION
  Prediction = (0.7 * XGBoost) + (0.3 * Random Forest)
  |
  V
OUTPUT: Price Forecast + Confidence Score
```

### 2. XGBoost Component (70% Weight)

**Purpose**: Capture complex, non-linear patterns in time series data

**Configuration**:
```python
XGBRegressor(
    n_estimators=400,          # 400 boosting rounds
    max_depth=6,               # Tree depth (prevents overfitting)
    learning_rate=0.02,        # Conservative learning (0.02)
    subsample=0.8,             # 80% random sample of rows
    colsample_bytree=0.7,      # 70% random sample of features
    min_child_weight=3,        # Leaf node size constraint
    gamma=0.2,                 # Minimum loss reduction
    reg_alpha=0.3,             # L1 regularization
    reg_lambda=2.0,            # L2 regularization
    objective='reg:squarederror'
)
```

**Strengths**:
- Handles gradient boosting for sequential error correction
- Native feature importance ranking
- Optimized C++ backend for speed
- Reduces bias in predictions
- Captures accelerating trends

### 3. Random Forest Component (30% Weight)

**Purpose**: Provide stability and reduce variance through ensemble averaging

**Configuration**:
```python
RandomForestRegressor(
    n_estimators=200,          # 200 decision trees
    max_depth=15,              # Deeper trees for pattern complexity
    min_samples_split=10,      # Split criteria
    min_samples_leaf=4,        # Leaf size constraint
    max_features='sqrt',       # Feature selection per split
    random_state=42,           # Reproducibility
    n_jobs=-1                  # Parallel processing
)
```

**Strengths**:
- Robust to outliers and market anomalies
- Reduces overfitting through averaging
- Handles non-linear relationships well
- Independent from XGBoost patterns
- Improves generalization to new market conditions

---

## Feature Engineering Pipeline

### Total Features: 50+

The model uses 50+ engineered features derived from raw OHLCV(open, high, low, close, volume) data:

### 1. Price Features (6 features)
```
Open, High, Low, Volume
Returns (daily % change)
Log_Returns (logarithmic returns)
```

### 2. Moving Averages (14 features)
```
Simple Moving Averages:
- SMA_5, SMA_10, SMA_20, SMA_50, SMA_100, SMA_200

Exponential Moving Averages:
- EMA_5, EMA_10, EMA_20, EMA_50, EMA_100

Price Ratios & Signals:
- Price_SMA20_Ratio (current vs 20-day MA)
- Price_SMA50_Ratio (current vs 50-day MA)
- Price_SMA200_Ratio (current vs 200-day MA)
- Golden_Cross (SMA_50 > SMA_200 indicator)
```

**Purpose**: Identify trend direction and support/resistance levels

### 3. Technical Indicators - Momentum (14 features)
```
RSI (Relative Strength Index):
- RSI (14-period momentum)
- RSI_Overbought (RSI > 70)
- RSI_Oversold (RSI < 30)

MACD (Moving Average Convergence Divergence):
- MACD (12-26 exponential MA difference)
- MACD_Signal (9-period signal line)
- MACD_Histogram (MACD - Signal)
- MACD_Bullish (MACD > Signal crossover)

Stochastic Oscillator:
- Stoch_K (14-period fast line)
- Stoch_D (3-period slow line)

ADX (Average Directional Index):
- ADX (trend strength 0-100)
- Trend_Strength (Bull=1, Bear=0)
```

**Purpose**: Identify overbought/oversold conditions and momentum shifts

### 4. Bollinger Bands (3 features)
```
- BB_Position (price position between bands 0-1)
- BB_Width (distance between upper/lower bands)
- BB_Squeeze (width < 50-day average, low volatility)
```

**Purpose**: Measure volatility and identify breakout opportunities

### 5. Volume Analysis (4 features)
```
- Volume_Ratio (current / 20-day average)
- Volume_ROC (10-period volume rate of change)
- OBV_EMA (On-Balance Volume with EMA smoothing)
- VPT (Volume Price Trend)
```

**Purpose**: Confirm price trends with volume participation

### 6. Volatility Measures (4 features)
```
- Volatility (20-day annualized std deviation)
- ATR (Average True Range for daily volatility)
- HV_10 (Historical Volatility 10-period)
- HV_30 (Historical Volatility 30-period)
```

**Purpose**: Quantify price fluctuation risk and adjust confidence

### 7. Momentum Indicators (6 features)
```
- Momentum_5 (5-day price momentum %)
- Momentum_10 (10-day price momentum %)
- Momentum_20 (20-day price momentum %)
- Momentum_50 (50-day price momentum %)
- ROC_10 (10-period rate of change %)
- ROC_20 (20-period rate of change %)
```

**Purpose**: Track acceleration/deceleration in price movement

### 8. Lag Features (9 features)
```
Price Lags:
- Close_Lag_1,2,3,5,10,20 (previous closing prices)
- Volume_Lag_1,2,5 (previous volumes)
- Returns_Lag_1,2,3 (previous returns)
```

**Purpose**: Capture temporal dependencies and autoregressive patterns

### 9. Pattern Recognition (5 features)
```
Price Patterns:
- Higher_High (consecutive higher peaks)
- Lower_Low (consecutive lower troughs)
- Gap_Up (opening above previous high)
- Gap_Down (opening below previous low)
- Doji (small body <10% of range, reversal signal)
```

**Purpose**: Identify chart patterns for reversal/continuation

### Feature Selection Process

```
50 Candidate Features
        |
        V
Remove correlated features (>0.95 correlation)
        |
        V
Scale using RobustScaler (resistant to outliers)
        |
        V
Clean: Handle NaN, inf, and extreme outliers
        |
        V
Outlier removal: Clip to 1st-99th percentile
        |
        V
Final: 30-40 selected features (varies by data size)
```

---

## Data Flow and Processing

### Step 1: Data Collection
```
Yahoo Finance API
    |
    V
1250 trading days (5 years of history)
    |
    V
OHLCV Data (Open, High, Low, Close, Volume)
```

### Step 2: Data Cleaning
```
Raw Data
    |
    +-- Remove null/NaN values
    +-- Handle missing dates (weekends/holidays)
    +-- Forward/backward fill any gaps
    +-- Remove duplicate timestamps
    +-- Sort by ascending date
    |
    V
Clean Time Series Data
```

### Step 3: Feature Engineering
```
Clean Data + 5 years history
    |
    V
Calculate 25+ Technical Indicators
    |
    V
Generate 50+ Features
    |
    V
Handle Infinities & Null Values
    |
    V
Outlier Clipping (1% to 99% percentile)
    |
    V
Scale Features (RobustScaler)
    |
    V
Feature Matrix Ready for Model
```

### Step 4: Model Training
```
Feature Matrix (X): [samples x 50 features]
Target Vector (y): [samples x 1] (next day close price)
    |
    V
Remove samples with null targets
    |
    V
Split into train/test sets
    |
    V
Train XGBoost (70% weight)
Train Random Forest (30% weight)
    |
    V
Generate Ensemble Predictions
    |
    V
Calculate Metrics (R2, MAE, MAPE, Accuracy)
```

### Step 5: Cross-Validation
```
Time Series Data
    |
    V
5-Fold Time Series Split (no future leakage):
    
    Fold 1: Train [0-256 days]     | Test [257-384 days]
    Fold 2: Train [0-384 days]     | Test [385-512 days]
    Fold 3: Train [0-512 days]     | Test [513-640 days]
    Fold 4: Train [0-640 days]     | Test [641-768 days]
    Fold 5: Train [0-768 days]     | Test [769-896 days]
    |
    V
Average Metrics Across 5 Folds
    |
    V
Final Performance Report with
    R2 Score: 
    MAE: 
    MAPE: 
    Directional Accuracy:
```

---

## Why Predictions Take 1-2 Minutes

### Time Breakdown

```
Task                          Time        Reason

1. Data Fetch (Yahoo)         15-30s      Network latency + 1250 days
2. Data Cleaning              5-10s       50 indicators calculation
3. Feature Engineering        20-30s      All 50+ features calculation
4. Model Training             30-40s      XGB (400 trees) + RF (200 trees)
5. Cross-Validation           30-45s      5-fold time series split
6. Prediction Generation      10-15s      Multi-day forecast loop
7. Metrics Calculation        5-10s       5 metrics computed
8. UI Rendering               10-20s      Streamlit rendering

TOTAL                         125-180s    (2-3 minutes worst case)
```

---

## Technical Stack

### Core Machine Learning
```
XGBoost 2.0.3        - Gradient boosting regressor
Scikit-learn 1.8.0   - Random Forest + preprocessing
NumPy 2.4.4          - Numerical computations
Pandas 3.0.2         - Data manipulation
```

### Web Framework & Visualization
```
Streamlit 1.56.0     - Web application framework
Plotly 6.6.0         - Interactive charts
Matplotlib 3.10.8    - Static visualizations
Seaborn 0.13.2       - Statistical plots
```

### Data Sources & APIs
```
yfinance 1.2.1       - Yahoo Finance data fetching
TextBlob 0.20.0      - Sentiment analysis
Requests 2.33.1      - HTTP requests
```

---


## Model Limitations & Considerations

### What the Model CANNOT Predict
```
1. Black swan events (unexpected shocks)
2. Regulatory announcements
3. Corporate actions (splits, mergers)
4. Geopolitical events
5. Market circuit breaker halts
6. Sudden liquidity crises
```

### Performance Degradation Under
```
1. High Volatility: Accuracy drops during market stress
2. New Trends: Model takes 50-100 days to adapt
3. Weekend Gaps: Large Monday opening gaps not predicted
4. Festival Seasons: Abnormal volume patterns
5. Low Liquidity: Penny stocks excluded from analysis
```

### Data Quality Issues
```
1. Yahoo Finance occasionally has stale data
2. Corporate actions (splits) may need adjustment
3. Weekend/holiday data points removed
4. Delisted stocks return no data
5. Extreme outlier days cause temporary inaccuracy
```


---

## License & Disclaimer

MIT License - Free for educational purpose  
Not responsible for trading losses or investment decisions

---

## Acknowledgments

- XGBoost Development Team
- Scikit-learn Contributors
- Streamlit Team
- Yahoo Finance API
- Indian Stock Market Community

---
