"""
Technical indicators calculation module
"""

import pandas as pd
import numpy as np

class TechnicalIndicators:
    
    @staticmethod
    def add_moving_averages(df):
        for period in [5, 10, 20, 50, 200]:
            df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
            df[f'EMA_{period}'] = df['Close'].ewm(span=period).mean()
        
        df['Price_SMA20_Ratio'] = df['Close'] / df['SMA_20'].replace(0, np.nan)
        df['Price_SMA50_Ratio'] = df['Close'] / df['SMA_50'].replace(0, np.nan)
        
        return df
    
    @staticmethod
    def add_rsi(df, period=14):
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss.replace(0, np.nan)
        df['RSI'] = 100 - (100 / (1 + rs))
        
        return df
    
    @staticmethod
    def add_macd(df):
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
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
        
        return df
    
    @staticmethod
    def add_volatility(df):
        df['Returns'] = df['Close'].pct_change()
        df['Volatility'] = df['Returns'].rolling(window=20).std() * np.sqrt(252)
        df['ATR'] = (df['High'] - df['Low']).rolling(window=14).mean()
        
        return df
    
    @staticmethod
    def add_momentum(df):
        for period in [5, 10, 20]:
            prev_close = df['Close'].shift(period).replace(0, np.nan)
            df[f'Momentum_{period}'] = (df['Close'] - df['Close'].shift(period)) / prev_close
        
        return df
    
    @staticmethod
    def add_lag_features(df):
        for lag in [1, 2, 3, 5, 10]:
            df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
            df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag)
        
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
        
        # Remove infinity values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        return df