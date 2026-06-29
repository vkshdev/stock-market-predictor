"""Deterministic multi-horizon stock return forecasting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor


SUPPORTED_HORIZONS = (1, 3, 5)

FEATURE_COLUMNS = (
    "Open",
    "High",
    "Low",
    "Volume",
    "Returns",
    "Log_Returns",
    "SMA_5",
    "SMA_10",
    "SMA_20",
    "SMA_50",
    "SMA_100",
    "SMA_200",
    "EMA_5",
    "EMA_10",
    "EMA_20",
    "EMA_50",
    "EMA_100",
    "Price_SMA20_Ratio",
    "Price_SMA50_Ratio",
    "Price_SMA200_Ratio",
    "Golden_Cross",
    "RSI",
    "RSI_Overbought",
    "RSI_Oversold",
    "MACD",
    "MACD_Signal",
    "MACD_Histogram",
    "MACD_Bullish",
    "BB_Position",
    "BB_Width",
    "BB_Squeeze",
    "Stoch_K",
    "Stoch_D",
    "ADX",
    "Trend_Strength",
    "Volume_Ratio",
    "Volume_ROC",
    "OBV_EMA",
    "VPT",
    "Volatility",
    "ATR",
    "HV_10",
    "HV_30",
    "Momentum_5",
    "Momentum_10",
    "Momentum_20",
    "Momentum_50",
    "ROC_10",
    "ROC_20",
    "Close_Lag_1",
    "Close_Lag_2",
    "Close_Lag_3",
    "Close_Lag_5",
    "Close_Lag_10",
    "Close_Lag_20",
    "Volume_Lag_1",
    "Volume_Lag_2",
    "Volume_Lag_5",
    "Returns_Lag_1",
    "Returns_Lag_2",
    "Returns_Lag_3",
    "Higher_High",
    "Lower_Low",
    "Gap_Up",
    "Gap_Down",
    "Doji",
)


@dataclass
class HorizonModel:
    """Fitted artifacts and validation evidence for one forecast horizon."""

    horizon: int
    feature_columns: list[str]
    imputer: SimpleImputer
    xgb_model: XGBRegressor
    rf_model: RandomForestRegressor
    probability_calibrator: LogisticRegression | None
    constant_probability_up: float
    interval_radius: float
    validation_metrics: dict[str, float | int | bool | None]


class StockPredictor:
    """Train direct return models for independent 1, 3, and 5-day horizons."""

    def __init__(
        self,
        horizons: Iterable[int] = SUPPORTED_HORIZONS,
        n_splits: int = 4,
        interval_coverage: float = 0.80,
        random_state: int = 42,
        xgb_estimators: int = 240,
        rf_estimators: int = 180,
    ) -> None:
        requested_horizons = tuple(sorted(set(int(value) for value in horizons)))
        unsupported = set(requested_horizons) - set(SUPPORTED_HORIZONS)
        if not requested_horizons or unsupported:
            raise ValueError(
                f"Horizons must be selected from {SUPPORTED_HORIZONS}; got {requested_horizons}"
            )
        if not 0.5 < interval_coverage < 1.0:
            raise ValueError("interval_coverage must be between 0.5 and 1.0")
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")

        self.horizons = requested_horizons
        self.n_splits = n_splits
        self.interval_coverage = interval_coverage
        self.random_state = random_state
        self.xgb_estimators = xgb_estimators
        self.rf_estimators = rf_estimators

        self.horizon_models: dict[int, HorizonModel] = {}
        self.validation_results: dict[str, dict[str, float | int | bool | None]] = {}
        self.feature_columns: list[str] = []
        self.xgb_model: XGBRegressor | None = None
        self.rf_model: RandomForestRegressor | None = None
        self.is_trained = False

    def prepare_features(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """Select the stable feature schema without fitting transforms globally."""
        available_features = [column for column in FEATURE_COLUMNS if column in df.columns]
        if not available_features:
            raise ValueError("No supported feature columns were found")

        features = df.loc[:, available_features].copy()
        features = features.replace([np.inf, -np.inf], np.nan)
        self.feature_columns = available_features
        return features, available_features

    def create_targets(
        self,
        df: pd.DataFrame,
        horizons: Iterable[int] | None = None,
    ) -> pd.DataFrame:
        """Create scale-independent forward return and direction targets."""
        selected_horizons = self.horizons if horizons is None else tuple(horizons)
        unsupported = set(selected_horizons) - set(SUPPORTED_HORIZONS)
        if unsupported:
            raise ValueError(f"Unsupported target horizons: {sorted(unsupported)}")
        if "Close" not in df.columns:
            raise ValueError("Close column is required to create targets")

        current_close = df["Close"].replace(0, np.nan)
        targets = pd.DataFrame(index=df.index)

        for horizon in selected_horizons:
            forward_return = df["Close"].shift(-horizon) / current_close - 1.0
            direction = (forward_return > 0).astype(float)
            direction = direction.where(forward_return.notna())
            targets[f"return_{horizon}d"] = forward_return
            targets[f"direction_{horizon}d"] = direction

        return targets

    def create_target(self, df: pd.DataFrame, days_ahead: int = 1) -> pd.Series:
        """Compatibility helper returning a forward-return target."""
        return self.create_targets(df, horizons=(days_ahead,))[f"return_{days_ahead}d"]

    def fit(
        self,
        features: pd.DataFrame,
        targets: pd.DataFrame,
    ) -> dict[str, dict[str, float | int | bool | None]]:
        """Run walk-forward validation, calibrate, then fit final horizon models."""
        if not features.index.is_monotonic_increasing:
            features = features.sort_index()
        if features.index.has_duplicates:
            raise ValueError("Feature index must not contain duplicate timestamps")

        self.horizon_models.clear()
        self.validation_results.clear()

        for horizon in self.horizons:
            target_column = f"return_{horizon}d"
            if target_column not in targets.columns:
                raise ValueError(f"Missing target column: {target_column}")

            target = targets[target_column].replace([np.inf, -np.inf], np.nan)
            valid_index = features.index.intersection(target.dropna().index)
            horizon_features = features.loc[valid_index]
            horizon_target = target.loc[valid_index].astype(float)

            if len(horizon_features) < 120:
                raise ValueError(
                    f"Not enough data for {horizon}-day model: "
                    f"need at least 120 samples, got {len(horizon_features)}"
                )

            validation = self._walk_forward_validate(
                horizon_features,
                horizon_target,
                horizon,
            )
            out_of_fold_predictions = validation.pop("out_of_fold_predictions")
            out_of_fold_actuals = validation.pop("out_of_fold_actuals")
            historical_mean_predictions = validation.pop("historical_mean_predictions")

            probability_calibrator, constant_probability = self._fit_calibrator(
                out_of_fold_predictions,
                out_of_fold_actuals > 0,
            )
            probabilities_up = self._calibrated_probabilities(
                out_of_fold_predictions,
                probability_calibrator,
                constant_probability,
            )
            residuals = out_of_fold_actuals - out_of_fold_predictions
            interval_radius = float(
                np.quantile(np.abs(residuals), self.interval_coverage)
            )

            metrics = self._validation_metrics(
                actual=out_of_fold_actuals,
                predicted=out_of_fold_predictions,
                probability_up=probabilities_up,
                historical_mean_predicted=historical_mean_predictions,
                interval_radius=interval_radius,
                fold_count=int(validation["fold_count"]),
                total_samples=len(horizon_features),
            )
            metrics["calibration_holdout_brier"] = self._temporal_calibration_brier(
                out_of_fold_predictions,
                out_of_fold_actuals > 0,
            )

            imputer, xgb_model, rf_model = self._fit_model_pair(
                horizon_features,
                horizon_target,
            )
            horizon_model = HorizonModel(
                horizon=horizon,
                feature_columns=list(horizon_features.columns),
                imputer=imputer,
                xgb_model=xgb_model,
                rf_model=rf_model,
                probability_calibrator=probability_calibrator,
                constant_probability_up=constant_probability,
                interval_radius=interval_radius,
                validation_metrics=metrics,
            )
            self.horizon_models[horizon] = horizon_model
            self.validation_results[f"{horizon}d"] = metrics

        primary_horizon = 1 if 1 in self.horizon_models else self.horizons[0]
        primary_model = self.horizon_models[primary_horizon]
        self.xgb_model = primary_model.xgb_model
        self.rf_model = primary_model.rf_model
        self.is_trained = True
        return self.validation_results

    def train(
        self,
        features: pd.DataFrame,
        target: pd.Series | pd.DataFrame,
    ) -> dict[str, dict[str, float | int | bool | None]]:
        """Compatibility wrapper; new code should call fit with all horizon targets."""
        if isinstance(target, pd.Series):
            target = target.to_frame(name="return_1d")
            self.horizons = (1,)
            return self.fit(features, target)
        return self.fit(features, target)

    def predict(self, features: pd.DataFrame, horizon: int = 1) -> np.ndarray:
        """Predict forward returns for a fitted direct horizon model."""
        model = self._require_horizon_model(horizon)
        aligned_features = features.loc[:, model.feature_columns].replace(
            [np.inf, -np.inf], np.nan
        )
        transformed = model.imputer.transform(aligned_features)
        return self._ensemble_predict(model.xgb_model, model.rf_model, transformed)

    def predict_multi_day(
        self,
        df: pd.DataFrame,
        days: int = 5,
        include_confidence: bool = True,
    ) -> list[dict[str, object]]:
        """Return deterministic direct forecasts up to the requested horizon."""
        if not self.is_trained:
            raise ValueError("Model not trained")
        if days < 1:
            raise ValueError("days must be at least 1")

        features, _ = self.prepare_features(df)
        current_price = float(df["Close"].iloc[-1])
        latest_timestamp = pd.Timestamp(df.index[-1])
        predictions: list[dict[str, object]] = []

        for horizon in (value for value in self.horizons if value <= days):
            model = self._require_horizon_model(horizon)
            latest_features = features.loc[:, model.feature_columns].iloc[[-1]]
            expected_return = round(
                float(self.predict(latest_features, horizon=horizon)[0]),
                12,
            )
            probability_up = float(
                self._calibrated_probabilities(
                    np.array([expected_return]),
                    model.probability_calibrator,
                    model.constant_probability_up,
                )[0]
            )
            lower_return = round(
                max(-0.99, expected_return - model.interval_radius),
                12,
            )
            upper_return = round(expected_return + model.interval_radius, 12)
            predicted_price = current_price * (1.0 + expected_return)
            lower_price = current_price * (1.0 + lower_return)
            upper_price = current_price * (1.0 + upper_return)
            signal = self._signal(expected_return, probability_up)
            directional_probability = (
                probability_up if expected_return >= 0 else 1.0 - probability_up
            )
            confidence_label = self._confidence_label(
                directional_probability,
                lower_return,
                upper_return,
            )
            prediction_date = latest_timestamp + pd.offsets.BDay(horizon)

            predictions.append(
                {
                    "day": horizon,
                    "horizon": f"{horizon}d",
                    "date": prediction_date.strftime("%Y-%m-%d"),
                    "predicted_price": round(predicted_price, 2),
                    "current_price": round(current_price, 2),
                    "change": round(predicted_price - current_price, 2),
                    "change_pct": round(expected_return * 100.0, 2),
                    "expected_return": expected_return,
                    "probability_up": round(probability_up * 100.0, 1),
                    "confidence": (
                        round(directional_probability * 100.0, 1)
                        if include_confidence
                        else None
                    ),
                    "confidence_label": confidence_label,
                    "direction": signal,
                    "signal": signal,
                    "strength": confidence_label,
                    "lower_return": lower_return,
                    "upper_return": upper_return,
                    "lower_price": round(lower_price, 2),
                    "upper_price": round(upper_price, 2),
                    "interval_coverage": self.interval_coverage,
                    "beats_zero_baseline": model.validation_metrics[
                        "beats_zero_baseline"
                    ],
                }
            )

        if not predictions:
            raise ValueError(
                f"No fitted direct horizon is available at or below {days} days"
            )
        return predictions

    def get_feature_importance(
        self,
        top_n: int = 15,
        horizon: int = 1,
    ) -> pd.DataFrame:
        """Return weighted tree feature importance for one direct horizon."""
        model = self._require_horizon_model(horizon)
        xgb_importance = model.xgb_model.feature_importances_
        rf_importance = model.rf_model.feature_importances_
        combined_importance = 0.7 * xgb_importance + 0.3 * rf_importance

        return (
            pd.DataFrame(
                {
                    "Feature": model.feature_columns,
                    "Importance": combined_importance,
                }
            )
            .sort_values("Importance", ascending=False)
            .head(top_n)
        )

    def _walk_forward_validate(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        horizon: int,
    ) -> dict[str, object]:
        splitter = self._time_series_splitter(len(features), horizon)
        predictions: list[np.ndarray] = []
        actuals: list[np.ndarray] = []
        historical_means: list[np.ndarray] = []
        fold_count = 0

        for train_index, validation_index in splitter.split(features):
            train_features = features.iloc[train_index]
            validation_features = features.iloc[validation_index]
            train_target = target.iloc[train_index]
            validation_target = target.iloc[validation_index]

            imputer, xgb_model, rf_model = self._fit_model_pair(
                train_features,
                train_target,
            )
            transformed_validation = imputer.transform(validation_features)
            fold_predictions = self._ensemble_predict(
                xgb_model,
                rf_model,
                transformed_validation,
            )

            predictions.append(fold_predictions)
            actuals.append(validation_target.to_numpy(dtype=float))
            historical_means.append(
                np.full(len(validation_target), float(train_target.mean()))
            )
            fold_count += 1

        return {
            "out_of_fold_predictions": np.concatenate(predictions),
            "out_of_fold_actuals": np.concatenate(actuals),
            "historical_mean_predictions": np.concatenate(historical_means),
            "fold_count": fold_count,
        }

    def _time_series_splitter(
        self,
        sample_count: int,
        horizon: int,
    ) -> TimeSeriesSplit:
        minimum_train_size = 80
        maximum_splits = max(2, (sample_count - minimum_train_size - horizon) // 20)
        split_count = min(self.n_splits, maximum_splits)
        test_size = (sample_count - minimum_train_size - horizon) // split_count
        test_size = min(63, max(15, test_size))

        while sample_count - split_count * test_size - horizon < minimum_train_size:
            test_size -= 1
            if test_size < 10:
                raise ValueError(
                    "Not enough samples for purged walk-forward validation"
                )

        return TimeSeriesSplit(
            n_splits=split_count,
            test_size=test_size,
            gap=horizon,
        )

    def _fit_model_pair(
        self,
        features: pd.DataFrame,
        target: pd.Series,
    ) -> tuple[SimpleImputer, XGBRegressor, RandomForestRegressor]:
        imputer = SimpleImputer(strategy="median")
        transformed = imputer.fit_transform(features)
        xgb_model, rf_model = self._new_model_pair()
        xgb_model.fit(transformed, target)
        rf_model.fit(transformed, target)
        return imputer, xgb_model, rf_model

    def _new_model_pair(self) -> tuple[XGBRegressor, RandomForestRegressor]:
        xgb_model = XGBRegressor(
            n_estimators=self.xgb_estimators,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.7,
            min_child_weight=3,
            gamma=0.1,
            reg_alpha=0.3,
            reg_lambda=2.0,
            objective="reg:squarederror",
            random_state=self.random_state,
            n_jobs=-1,
            verbosity=0,
        )
        rf_model = RandomForestRegressor(
            n_estimators=self.rf_estimators,
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=4,
            max_features="sqrt",
            random_state=self.random_state,
            n_jobs=-1,
        )
        return xgb_model, rf_model

    @staticmethod
    def _ensemble_predict(
        xgb_model: XGBRegressor,
        rf_model: RandomForestRegressor,
        features: np.ndarray,
    ) -> np.ndarray:
        return 0.7 * xgb_model.predict(features) + 0.3 * rf_model.predict(features)

    @staticmethod
    def _fit_calibrator(
        predicted_returns: np.ndarray,
        actual_direction: np.ndarray,
    ) -> tuple[LogisticRegression | None, float]:
        direction = np.asarray(actual_direction, dtype=int)
        constant_probability = float(np.mean(direction))
        if np.unique(direction).size < 2 or np.unique(predicted_returns).size < 2:
            return None, constant_probability

        calibrator = LogisticRegression(random_state=42)
        calibration_feature = np.asarray(predicted_returns).reshape(-1, 1) * 100.0
        calibrator.fit(calibration_feature, direction)
        return calibrator, constant_probability

    @staticmethod
    def _calibrated_probabilities(
        predicted_returns: np.ndarray,
        calibrator: LogisticRegression | None,
        constant_probability: float,
    ) -> np.ndarray:
        values = np.asarray(predicted_returns, dtype=float)
        if calibrator is None:
            return np.full(values.shape, constant_probability, dtype=float)
        return calibrator.predict_proba(values.reshape(-1, 1) * 100.0)[:, 1]

    def _temporal_calibration_brier(
        self,
        predicted_returns: np.ndarray,
        actual_direction: np.ndarray,
    ) -> float | None:
        sample_count = len(predicted_returns)
        split_index = int(sample_count * 0.7)
        if split_index < 20 or sample_count - split_index < 10:
            return None

        train_direction = np.asarray(actual_direction[:split_index], dtype=int)
        if np.unique(train_direction).size < 2:
            return None

        calibrator, constant_probability = self._fit_calibrator(
            predicted_returns[:split_index],
            train_direction,
        )
        holdout_probability = self._calibrated_probabilities(
            predicted_returns[split_index:],
            calibrator,
            constant_probability,
        )
        return float(
            brier_score_loss(
                np.asarray(actual_direction[split_index:], dtype=int),
                holdout_probability,
            )
        )

    def _validation_metrics(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
        probability_up: np.ndarray,
        historical_mean_predicted: np.ndarray,
        interval_radius: float,
        fold_count: int,
        total_samples: int,
    ) -> dict[str, float | int | bool | None]:
        actual_direction = actual > 0
        predicted_direction = predicted > 0
        model_mae = float(mean_absolute_error(actual, predicted))
        zero_baseline_mae = float(mean_absolute_error(actual, np.zeros_like(actual)))
        historical_mean_mae = float(
            mean_absolute_error(actual, historical_mean_predicted)
        )
        improvement = (
            (zero_baseline_mae - model_mae) / zero_baseline_mae * 100.0
            if zero_baseline_mae > 0
            else 0.0
        )
        lower = predicted - interval_radius
        upper = predicted + interval_radius

        return {
            "sample_count": total_samples,
            "oof_sample_count": len(actual),
            "fold_count": fold_count,
            "mae": model_mae,
            "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
            "r2": float(r2_score(actual, predicted)),
            "directional_accuracy": float(
                np.mean(actual_direction == predicted_direction) * 100.0
            ),
            "balanced_accuracy": float(
                balanced_accuracy_score(actual_direction, predicted_direction) * 100.0
            ),
            "brier_score": float(brier_score_loss(actual_direction, probability_up)),
            "interval_coverage": float(np.mean((actual >= lower) & (actual <= upper))),
            "interval_target_coverage": self.interval_coverage,
            "interval_radius": interval_radius,
            "zero_return_baseline_mae": zero_baseline_mae,
            "historical_mean_baseline_mae": historical_mean_mae,
            "mae_improvement_vs_zero_pct": float(improvement),
            "beats_zero_baseline": bool(model_mae < zero_baseline_mae),
            "beats_historical_mean_baseline": bool(model_mae < historical_mean_mae),
        }

    @staticmethod
    def _signal(expected_return: float, probability_up: float) -> str:
        if expected_return > 0 and probability_up >= 0.58:
            return "Bullish"
        if expected_return < 0 and probability_up <= 0.42:
            return "Bearish"
        return "Neutral"

    @staticmethod
    def _confidence_label(
        directional_probability: float,
        lower_return: float,
        upper_return: float,
    ) -> str:
        interval_excludes_zero = lower_return > 0 or upper_return < 0
        if directional_probability >= 0.65 and interval_excludes_zero:
            return "High"
        if directional_probability >= 0.58:
            return "Medium"
        return "Low"

    def _require_horizon_model(self, horizon: int) -> HorizonModel:
        if not self.is_trained or horizon not in self.horizon_models:
            raise ValueError(f"No trained model is available for the {horizon}-day horizon")
        return self.horizon_models[horizon]
