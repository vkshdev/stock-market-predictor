"""
Machine Learning Prediction Engine with NIFTY/SENSEX Support
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class StockPredictor:
    """
    Advanced ML predictor for Indian stocks and indices
    Supports NIFTY 50, SENSEX, and individual stocks
    """
    
    def __init__(self, model_type='random_forest'):
        """
        Initialize predictor
        
        Args:
            model_type (str): 'random_forest', 'gradient_boosting', 'linear', 'ridge'
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.is_trained = False
        
    def prepare_features(self, df):
        """
        Select and prepare features for prediction
        
        Args:
            df (pd.DataFrame): DataFrame with technical indicators
        
        Returns:
            pd.DataFrame: Feature matrix
            list: Feature column names
        """
        # Define feature columns
        feature_columns = [
            # Price features
            'Open', 'High', 'Low', 'Volume', 'Returns',
            
            # Moving averages
            'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50',
            'EMA_5', 'EMA_10', 'EMA_20',
            'Price_SMA20_Ratio', 'Price_SMA50_Ratio',
            
            # Technical indicators
            'RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram',
            'BB_Position', 'BB_Width', 'Stoch_K', 'Stoch_D',
            
            # Volume indicators
            'Volume_Ratio', 'Volume_ROC',
            
            # Volatility
            'Volatility', 'ATR',
            
            # Momentum
            'Momentum_5', 'Momentum_10', 'Momentum_20',
            
            # Lag features
            'Close_Lag_1', 'Close_Lag_2', 'Close_Lag_3',
            'Volume_Lag_1', 'Volume_Lag_2'
        ]
        
        # Filter available columns
        available_features = [col for col in feature_columns if col in df.columns]
        
        # Handle missing values
        X = df[available_features].copy()
        X = X.fillna(method='ffill').fillna(method='bfill')
        
        self.feature_columns = available_features
        
        return X, available_features
    
    def create_target(self, df, days_ahead=1):
        """
        Create target variable (future price)
        
        Args:
            df (pd.DataFrame): Stock data
            days_ahead (int): Days to predict ahead
        
        Returns:
            pd.Series: Target variable
        """
        return df['Close'].shift(-days_ahead)
    
    def train(self, X, y):
        """
        Train the prediction model
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
        
        Returns:
            dict: Training metrics
        """
        logger.info(f"Training {self.model_type} model...")
        
        # Remove rows with NaN in target
        valid_idx = ~y.isna()
        X = X[valid_idx]
        y = y[valid_idx]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Initialize model
        if self.model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=150,
                max_depth=8,
                learning_rate=0.1,
                random_state=42
            )
        elif self.model_type == 'linear':
            self.model = LinearRegression()
        elif self.model_type == 'ridge':
            self.model = Ridge(alpha=1.0, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Train model
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        # Calculate training metrics
        y_pred = self.model.predict(X_scaled)
        
        metrics = {
            'mse': mean_squared_error(y, y_pred),
            'mae': mean_absolute_error(y, y_pred),
            'r2': r2_score(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred))
        }
        
        logger.info(f"Model trained. R² Score: {metrics['r2']:.3f}")
        
        return metrics
    
    def cross_validate(self, X, y, n_splits=5):
        """
        Perform time series cross-validation
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            n_splits (int): Number of CV splits
        
        Returns:
            dict: Cross-validation results
        """
        logger.info(f"Performing {n_splits}-fold time series cross-validation...")
        
        # Remove rows with NaN in target
        valid_idx = ~y.isna()
        X = X[valid_idx]
        y = y[valid_idx]
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        cv_scores = {
            'train_mse': [], 'test_mse': [],
            'train_mae': [], 'test_mae': [],
            'train_r2': [], 'test_r2': []
        }
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            # Split data
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            if self.model_type == 'random_forest':
                model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
            elif self.model_type == 'gradient_boosting':
                model = GradientBoostingRegressor(n_estimators=100, max_depth=6, random_state=42)
            else:
                model = LinearRegression()
            
            model.fit(X_train_scaled, y_train)
            
            # Predict
            y_train_pred = model.predict(X_train_scaled)
            y_test_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            cv_scores['train_mse'].append(mean_squared_error(y_train, y_train_pred))
            cv_scores['test_mse'].append(mean_squared_error(y_test, y_test_pred))
            cv_scores['train_mae'].append(mean_absolute_error(y_train, y_train_pred))
            cv_scores['test_mae'].append(mean_absolute_error(y_test, y_test_pred))
            cv_scores['train_r2'].append(r2_score(y_train, y_train_pred))
            cv_scores['test_r2'].append(r2_score(y_test, y_test_pred))
        
        # Calculate averages
        avg_scores = {
            'avg_train_r2': np.mean(cv_scores['train_r2']),
            'avg_test_r2': np.mean(cv_scores['test_r2']),
            'avg_test_mse': np.mean(cv_scores['test_mse']),
            'avg_test_mae': np.mean(cv_scores['test_mae']),
            'std_test_r2': np.std(cv_scores['test_r2'])
        }
        
        logger.info(f"CV Complete. Avg Test R²: {avg_scores['avg_test_r2']:.3f}")
        
        return {**cv_scores, **avg_scores}
    
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X (pd.DataFrame): Feature matrix
        
        Returns:
            np.array: Predictions
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        return predictions
    
    def predict_multi_day(self, df, days=3, include_confidence=True):
        """
        Predict multiple days ahead
        
        Args:
            df (pd.DataFrame): Stock data with indicators
            days (int): Number of days to predict
            include_confidence (bool): Include confidence scores
        
        Returns:
            list: Predictions for each day
        """
        predictions = []
        current_data = df.copy()
        
        for day in range(1, days + 1):
            # Prepare features
            X, _ = self.prepare_features(current_data)
            
            if len(X) == 0:
                logger.warning(f"No valid data for day {day}")
                break
            
            # Predict next price
            X_latest = X.iloc[-1:].copy()
            next_price = self.predict(X_latest)[0]
            current_price = current_data['Close'].iloc[-1]
            
            # Calculate change
            change = next_price - current_price
            change_pct = (change / current_price) * 100
            
            # Determine confidence (based on model performance)
            if hasattr(self, 'cv_results') and include_confidence:
                base_confidence = self.cv_results.get('avg_test_r2', 0.7) * 100
                # Decrease confidence for further days
                confidence = max(50, base_confidence - (day - 1) * 5)
            else:
                confidence = 75 - (day - 1) * 5
            
            # Prediction date
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
            
            # Update data for next iteration (simulate next day)
            new_row = current_data.iloc[-1:].copy()
            new_row['Close'] = next_price
            new_row['Open'] = next_price * (1 + np.random.normal(0, 0.005))
            new_row['High'] = next_price * (1 + abs(np.random.normal(0, 0.01)))
            new_row['Low'] = next_price * (1 - abs(np.random.normal(0, 0.01)))
            new_row['Volume'] = current_data['Volume'].iloc[-1] * (1 + np.random.normal(0, 0.2))
            
            # Append to data (for next iteration)
            current_data = pd.concat([current_data, new_row])
            current_data = current_data.iloc[-100:]  # Keep last 100 rows
        
        return predictions
    
    def get_feature_importance(self, top_n=15):
        """
        Get feature importance (for tree-based models)
        
        Args:
            top_n (int): Number of top features to return
        
        Returns:
            pd.DataFrame: Feature importance
        """
        if not hasattr(self.model, 'feature_importances_'):
            logger.warning("Model doesn't support feature importance")
            return None
        
        importance_df = pd.DataFrame({
            'Feature': self.feature_columns,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        return importance_df.head(top_n)
    
    def save_model(self, filepath):
        """Save trained model to file"""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'model_type': self.model_type
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load trained model from file"""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.model_type = model_data['model_type']
        self.is_trained = True
        
        logger.info(f"Model loaded from {filepath}")
    
    def calculate_directional_accuracy(self, y_true, y_pred):
        """
        Calculate directional accuracy (% of correct up/down predictions)
        
        Args:
            y_true (array): True values
            y_pred (array): Predicted values
        
        Returns:
            float: Directional accuracy (0-1)
        """
        # Calculate price changes
        true_direction = np.diff(y_true) > 0
        pred_direction = np.diff(y_pred) > 0
        
        # Calculate accuracy
        accuracy = np.mean(true_direction == pred_direction)
        
        return accuracy