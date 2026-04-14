"""
XGBoost Prediction Engine
"""

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime, timedelta

class StockPredictor:
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.is_trained = False
        
    def prepare_features(self, df):
        feature_columns = [
            'Open', 'High', 'Low', 'Volume', 'Returns',
            'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50',
            'EMA_5', 'EMA_10', 'EMA_20',
            'Price_SMA20_Ratio', 'Price_SMA50_Ratio',
            'RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram',
            'BB_Position', 'BB_Width', 'Stoch_K', 'Stoch_D',
            'Volume_Ratio', 'Volume_ROC',
            'Volatility', 'ATR',
            'Momentum_5', 'Momentum_10', 'Momentum_20',
            'Close_Lag_1', 'Close_Lag_2', 'Close_Lag_3',
            'Volume_Lag_1', 'Volume_Lag_2'
        ]
        
        available_features = [col for col in feature_columns if col in df.columns]
        X = df[available_features].copy()
        
        # Clean infinity and NaN
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.ffill().bfill()
        X = X.fillna(0)
        
        # Clip extreme values
        for col in X.columns:
            X[col] = X[col].clip(lower=-1e10, upper=1e10)
        
        self.feature_columns = available_features
        return X, available_features
    
    def create_target(self, df, days_ahead=1):
        return df['Close'].shift(-days_ahead)
    
    def train(self, X, y):
        valid_idx = ~y.isna()
        X = X[valid_idx]
        y = y[valid_idx]
        
        # Final cleaning
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        X_scaled = self.scaler.fit_transform(X)
        
        self.model = XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        y_pred = self.model.predict(X_scaled)
        
        metrics = {
            'mse': mean_squared_error(y, y_pred),
            'mae': mean_absolute_error(y, y_pred),
            'r2': r2_score(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred))
        }
        
        return metrics
    
    def cross_validate(self, X, y, n_splits=5):
        valid_idx = ~y.isna()
        X = X[valid_idx]
        y = y[valid_idx]
        
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        cv_scores = {
            'train_mse': [], 'test_mse': [],
            'train_mae': [], 'test_mae': [],
            'train_r2': [], 'test_r2': []
        }
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model = XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.05,
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X_train_scaled, y_train)
            
            y_train_pred = model.predict(X_train_scaled)
            y_test_pred = model.predict(X_test_scaled)
            
            cv_scores['train_mse'].append(mean_squared_error(y_train, y_train_pred))
            cv_scores['test_mse'].append(mean_squared_error(y_test, y_test_pred))
            cv_scores['train_mae'].append(mean_absolute_error(y_train, y_train_pred))
            cv_scores['test_mae'].append(mean_absolute_error(y_test, y_test_pred))
            cv_scores['train_r2'].append(r2_score(y_train, y_train_pred))
            cv_scores['test_r2'].append(r2_score(y_test, y_test_pred))
        
        avg_scores = {
            'avg_train_r2': np.mean(cv_scores['train_r2']),
            'avg_test_r2': np.mean(cv_scores['test_r2']),
            'avg_test_mse': np.mean(cv_scores['test_mse']),
            'avg_test_mae': np.mean(cv_scores['test_mae']),
            'std_test_r2': np.std(cv_scores['test_r2'])
        }
        
        return {**cv_scores, **avg_scores}
    
    def predict(self, X):
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        return predictions
    
    def predict_multi_day(self, df, days=3, include_confidence=True):
        predictions = []
        current_data = df.copy()
        
        for day in range(1, days + 1):
            X, _ = self.prepare_features(current_data)
            
            if len(X) == 0:
                break
            
            X_latest = X.iloc[-1:].copy()
            next_price = self.predict(X_latest)[0]
            current_price = current_data['Close'].iloc[-1]
            
            change = next_price - current_price
            change_pct = (change / current_price) * 100
            
            if hasattr(self, 'cv_results') and include_confidence:
                base_confidence = self.cv_results.get('avg_test_r2', 0.7) * 100
                confidence = max(50, base_confidence - (day - 1) * 5)
            else:
                confidence = 75 - (day - 1) * 5
            
            prediction_date = datetime.now() + timedelta(days=day)
            
            prediction = {
                'day': day,
                'date': prediction_date.strftime('%Y-%m-%d'),
                'predicted_price': round(next_price, 2),
                'current_price': round(current_price, 2),
                'change': round(change, 2),
                'change_pct': round(change_pct, 2),
                'confidence': round(confidence, 1),
                'direction': 'Bullish' if change > 0 else 'Bearish',
                'strength': 'Strong' if abs(change_pct) > 2 else 'Moderate' if abs(change_pct) > 1 else 'Weak'
            }
            
            predictions.append(prediction)
            
            new_row = current_data.iloc[-1:].copy()
            new_row['Close'] = next_price
            new_row['Open'] = next_price * (1 + np.random.normal(0, 0.005))
            new_row['High'] = next_price * (1 + abs(np.random.normal(0, 0.01)))
            new_row['Low'] = next_price * (1 - abs(np.random.normal(0, 0.01)))
            new_row['Volume'] = current_data['Volume'].iloc[-1] * (1 + np.random.normal(0, 0.2))
            
            current_data = pd.concat([current_data, new_row])
            current_data = current_data.iloc[-100:]
        
        return predictions
    
    def get_feature_importance(self, top_n=15):
        if not hasattr(self.model, 'feature_importances_'):
            return None
        
        importance_df = pd.DataFrame({
            'Feature': self.feature_columns,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        return importance_df.head(top_n)