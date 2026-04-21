"""
 Technical indicators
"""

import pandas as pd
import numpy as np

class TechnicalIndicators:
    
    @staticmethod
    def add_moving_averages(df):
        # More MA periods
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
            df[f'EMA_{period}'] = df['Close'].ewm(span=period).mean()
        
        df['Price_SMA20_Ratio'] = df['Close'] / df['SMA_20'].replace(0, np.nan)
        df['Price_SMA50_Ratio'] = df['Close'] / df['SMA_50'].replace(0, np.nan)
        df['Price_SMA200_Ratio'] = df['Close'] / df['SMA_200'].replace(0, np.nan)
        
        # Golden Cross / Death Cross
        df['Golden_Cross'] = (df['SMA_50'] > df['SMA_200']).astype(int)
        
        return df
    
    @staticmethod
    def add_rsi(df, period=14):
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss.replace(0, np.nan)
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # RSI oversold/overbought signals
        df['RSI_Overbought'] = (df['RSI'] > 70).astype(int)
        df['RSI_Oversold'] = (df['RSI'] < 30).astype(int)
        
        return df
    
    @staticmethod
    def add_macd(df):
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # MACD crossover signals
        df['MACD_Bullish'] = (df['MACD'] > df['MACD_Signal']).astype(int)
        
        return df
    
    @staticmethod
    def add_bollinger_bands(df, period=20):
        df['BB_Middle'] = df['Close'].rolling(window=period).mean()
        bb_std = df['Close'].rolling(window=period).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
        
        denominator = df['BB_Width'].replace(0, np.nan)
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / denominator
        
        # BB squeeze indicator
        df['BB_Squeeze'] = (df['BB_Width'] < df['BB_Width'].rolling(50).mean()).astype(int)
        
        return df
    
    @staticmethod
    def add_stochastic(df, period=14):
        low_min = df['Low'].rolling(window=period).min()
        high_max = df['High'].rolling(window=period).max()
        
        denominator = (high_max - low_min).replace(0, np.nan)
        df['Stoch_K'] = 100 * (df['Close'] - low_min) / denominator
        df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
        
        return df
    
    @staticmethod
    def add_volume_indicators(df):
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA'].replace(0, np.nan)
        df['Volume_ROC'] = df['Volume'].pct_change(periods=10)
        
        # On Balance Volume
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        df['OBV_EMA'] = df['OBV'].ewm(span=20).mean()
        
        # Volume Price Trend
        df['VPT'] = (df['Volume'] * df['Close'].pct_change()).cumsum()
        
        return df
    
    @staticmethod
    def add_volatility(df):
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Volatility'] = df['Returns'].rolling(window=20).std() * np.sqrt(252)
        
        # ATR (Average True Range)
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['ATR'] = true_range.rolling(14).mean()
        
        # Historical Volatility
        df['HV_10'] = df['Returns'].rolling(10).std() * np.sqrt(252)
        df['HV_30'] = df['Returns'].rolling(30).std() * np.sqrt(252)
        
        return df
    
    @staticmethod
    def add_momentum(df):
        # Multiple momentum periods
        for period in [5, 10, 20, 50]:
            prev_close = df['Close'].shift(period).replace(0, np.nan)
            df[f'Momentum_{period}'] = (df['Close'] - df['Close'].shift(period)) / prev_close
        
        # Rate of Change
        df['ROC_10'] = df['Close'].pct_change(periods=10) * 100
        df['ROC_20'] = df['Close'].pct_change(periods=20) * 100
        
        return df
    
    @staticmethod
    def add_lag_features(df):
        # More lag features
        for lag in [1, 2, 3, 5, 10, 20]:
            df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
            df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag)
            df[f'Returns_Lag_{lag}'] = df['Returns'].shift(lag)
        
        return df
    
    @staticmethod
    def add_pattern_features(df):
        # Price patterns
        df['Higher_High'] = ((df['High'] > df['High'].shift(1)) & 
                            (df['High'].shift(1) > df['High'].shift(2))).astype(int)
        df['Lower_Low'] = ((df['Low'] < df['Low'].shift(1)) & 
                          (df['Low'].shift(1) < df['Low'].shift(2))).astype(int)
        
        # Gap detection
        df['Gap_Up'] = (df['Low'] > df['High'].shift(1)).astype(int)
        df['Gap_Down'] = (df['High'] < df['Low'].shift(1)).astype(int)
        
        # Doji pattern (simple version)
        body = abs(df['Close'] - df['Open'])
        range_val = df['High'] - df['Low']
        df['Doji'] = (body < (range_val * 0.1)).astype(int)
        
        return df
    
    @staticmethod
    def add_trend_features(df):
        # ADX (Average Directional Index) - simplified
        high_diff = df['High'].diff()
        low_diff = -df['Low'].diff()
        
        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
        
        atr = df['ATR']
        plus_di = 100 * (plus_dm.rolling(14).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(14).mean() / atr)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        df['ADX'] = dx.rolling(14).mean()
        
        # Trend strength
        df['Trend_Strength'] = (df['Close'] > df['SMA_50']).astype(int)
        
        return df
    
    @staticmethod
    def add_all_indicators(df):
        df = TechnicalIndicators.add_moving_averages(df)
        df = TechnicalIndicators.add_rsi(df)
        df = TechnicalIndicators.add_macd(df)
        df = TechnicalIndicators.add_bollinger_bands(df)
        df = TechnicalIndicators.add_stochastic(df)
        df = TechnicalIndicators.add_volume_indicators(df)
        df = TechnicalIndicators.add_volatility(df)
        df = TechnicalIndicators.add_momentum(df)
        df = TechnicalIndicators.add_lag_features(df)
        df = TechnicalIndicators.add_pattern_features(df)
        df = TechnicalIndicators.add_trend_features(df)
        
        # Remove infinity values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        return df