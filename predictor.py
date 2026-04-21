"""
Ensemble (XGB+RF) Predictor
"""

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class StockPredictor:
    
    def __init__(self):
        self.xgb_model = None
        self.rf_model = None
        self.scaler = RobustScaler()
        self.feature_columns = None
        self.is_trained = False
        
    def prepare_features(self, df):
        feature_columns = [
            # Price features
            'Open', 'High', 'Low', 'Volume', 'Returns', 'Log_Returns',
            
            # Moving averages
            'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50', 'SMA_100', 'SMA_200',
            'EMA_5', 'EMA_10', 'EMA_20', 'EMA_50', 'EMA_100',
            'Price_SMA20_Ratio', 'Price_SMA50_Ratio', 'Price_SMA200_Ratio',
            'Golden_Cross',
            
            # Technical indicators
            'RSI', 'RSI_Overbought', 'RSI_Oversold',
            'MACD', 'MACD_Signal', 'MACD_Histogram', 'MACD_Bullish',
            'BB_Position', 'BB_Width', 'BB_Squeeze',
            'Stoch_K', 'Stoch_D',
            'ADX', 'Trend_Strength',
            
            # Volume indicators
            'Volume_Ratio', 'Volume_ROC', 'OBV_EMA', 'VPT',
            
            # Volatility
            'Volatility', 'ATR', 'HV_10', 'HV_30',
            
            # Momentum
            'Momentum_5', 'Momentum_10', 'Momentum_20', 'Momentum_50',
            'ROC_10', 'ROC_20',
            
            # Lag features
            'Close_Lag_1', 'Close_Lag_2', 'Close_Lag_3', 'Close_Lag_5', 'Close_Lag_10', 'Close_Lag_20',
            'Volume_Lag_1', 'Volume_Lag_2', 'Volume_Lag_5',
            'Returns_Lag_1', 'Returns_Lag_2', 'Returns_Lag_3',
            
            # Pattern features
            'Higher_High', 'Lower_Low', 'Gap_Up', 'Gap_Down', 'Doji'
        ]
        
        available_features = [col for col in feature_columns if col in df.columns]
        X = df[available_features].copy()
        
        # Aggressive cleaning
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.ffill().bfill()
        X = X.fillna(0)
        
        # Remove extreme outliers
        for col in X.columns:
            q1 = X[col].quantile(0.01)
            q99 = X[col].quantile(0.99)
            X[col] = X[col].clip(lower=q1, upper=q99)
        
        self.feature_columns = available_features
        return X, available_features
    
    def create_target(self, df, days_ahead=1):
        return df['Close'].shift(-days_ahead)
    
    def train(self, X, y):
        valid_idx = ~y.isna()
        X = X[valid_idx]
        y = y[valid_idx]
        
        if len(X) < 200:
            raise ValueError("Not enough data (need 200+ samples)")
        
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        X_scaled = self.scaler.fit_transform(X)
        
        # XGBoost model
        self.xgb_model = XGBRegressor(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.02,
            subsample=0.8,
            colsample_bytree=0.7,
            min_child_weight=3,
            gamma=0.2,
            reg_alpha=0.3,
            reg_lambda=2.0,
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
        
        # RandomForest model
        self.rf_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=4,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        
        # Train both models
        self.xgb_model.fit(X_scaled, y, verbose=False)
        self.rf_model.fit(X_scaled, y)
        self.is_trained = True
        
        # Ensemble predictions (70% XGBoost, 30% RandomForest)
        xgb_pred = self.xgb_model.predict(X_scaled)
        rf_pred = self.rf_model.predict(X_scaled)
        y_pred = 0.7 * xgb_pred + 0.3 * rf_pred
        
        # Metrics
        mse = mean_squared_error(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(y, y_pred) * 100
        
        # Directional accuracy
        actual_direction = np.diff(y) > 0
        pred_direction = np.diff(y_pred) > 0
        directional_accuracy = np.mean(actual_direction == pred_direction) * 100
        
        metrics = {
            'mse': mse,
            'mae': mae,
            'r2': max(r2, 0),
            'rmse': rmse,
            'mape': mape,
            'directional_accuracy': directional_accuracy
        }
        
        return metrics
    
    def cross_validate(self, X, y, n_splits=5):
        valid_idx = ~y.isna()
        X = X[valid_idx]
        y = y[valid_idx]
        
        if len(X) < 500:
            n_splits = 3
        
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        cv_scores = {
            'test_r2': [],
            'test_mae': [],
            'test_mape': [],
            'directional_accuracy': []
        }
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # XGBoost
            xgb = XGBRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.02,
                subsample=0.8,
                colsample_bytree=0.7,
                min_child_weight=3,
                gamma=0.2,
                reg_alpha=0.3,
                reg_lambda=2.0,
                random_state=42,
                n_jobs=-1,
                verbosity=0
            )
            
            # RandomForest
            rf = RandomForestRegressor(
                n_estimators=150,
                max_depth=12,
                min_samples_split=10,
                random_state=42,
                n_jobs=-1
            )
            
            xgb.fit(X_train_scaled, y_train, verbose=False)
            rf.fit(X_train_scaled, y_train)
            
            # Ensemble
            xgb_pred = xgb.predict(X_test_scaled)
            rf_pred = rf.predict(X_test_scaled)
            y_pred = 0.7 * xgb_pred + 0.3 * rf_pred
            
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            mape = mean_absolute_percentage_error(y_test, y_pred) * 100
            
            actual_dir = np.diff(y_test) > 0
            pred_dir = np.diff(y_pred) > 0
            dir_acc = np.mean(actual_dir == pred_dir) * 100
            
            cv_scores['test_r2'].append(max(r2, 0))
            cv_scores['test_mae'].append(mae)
            cv_scores['test_mape'].append(mape)
            cv_scores['directional_accuracy'].append(dir_acc)
        
        avg_scores = {
            'avg_test_r2': np.mean(cv_scores['test_r2']),
            'avg_test_mae': np.mean(cv_scores['test_mae']),
            'avg_test_mape': np.mean(cv_scores['test_mape']),
            'avg_directional_accuracy': np.mean(cv_scores['directional_accuracy']),
            'std_test_r2': np.std(cv_scores['test_r2'])
        }
        
        return {**cv_scores, **avg_scores}
    
    def predict(self, X):
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        X_scaled = self.scaler.transform(X)
        
        # Ensemble prediction
        xgb_pred = self.xgb_model.predict(X_scaled)
        rf_pred = self.rf_model.predict(X_scaled)
        predictions = 0.7 * xgb_pred + 0.3 * rf_pred
        
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
            
            if hasattr(self, 'cv_results'):
                r2_conf = self.cv_results.get('avg_test_r2', 0.5) * 50
                dir_conf = self.cv_results.get('avg_directional_accuracy', 70) * 0.5
                base_confidence = r2_conf + dir_conf
                confidence = max(50, min(95, base_confidence - (day - 1) * 5))
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
            new_row['Open'] = next_price * (1 + np.random.normal(0, 0.002))
            new_row['High'] = next_price * (1 + abs(np.random.normal(0, 0.006)))
            new_row['Low'] = next_price * (1 - abs(np.random.normal(0, 0.006)))
            new_row['Volume'] = current_data['Volume'].iloc[-1] * (1 + np.random.normal(0, 0.1))
            
            current_data = pd.concat([current_data, new_row])
            current_data = current_data.iloc[-200:]
        
        return predictions
    
    def get_feature_importance(self, top_n=15):
        if not hasattr(self.xgb_model, 'feature_importances_'):
            return None
        
        # Combine feature importance from both models
        xgb_importance = self.xgb_model.feature_importances_
        rf_importance = self.rf_model.feature_importances_
        combined_importance = 0.7 * xgb_importance + 0.3 * rf_importance
        
        importance_df = pd.DataFrame({
            'Feature': self.feature_columns,
            'Importance': combined_importance
        }).sort_values('Importance', ascending=False)
        
        return importance_df.head(top_n)