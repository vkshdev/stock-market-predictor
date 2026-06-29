from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from predictor import StockPredictor


def _market_data(sample_count: int = 240) -> pd.DataFrame:
    random = np.random.default_rng(17)
    index = pd.bdate_range("2024-01-01", periods=sample_count)
    periodic_return = 0.006 * np.sin(np.arange(sample_count) / 5.0)
    returns = periodic_return + random.normal(0.0003, 0.004, sample_count)
    close = 100.0 * np.cumprod(1.0 + returns)
    open_price = close * (1.0 + random.normal(0.0, 0.001, sample_count))

    return pd.DataFrame(
        {
            "Open": open_price,
            "High": np.maximum(open_price, close) * 1.004,
            "Low": np.minimum(open_price, close) * 0.996,
            "Close": close,
            "Volume": random.integers(500_000, 2_000_000, sample_count),
            "Returns": pd.Series(close, index=index).pct_change().to_numpy(),
            "Log_Returns": np.r_[np.nan, np.diff(np.log(close))],
            "Momentum_5": pd.Series(close, index=index).pct_change(5).to_numpy(),
        },
        index=index,
    )


@pytest.fixture(scope="module")
def fitted_predictor() -> tuple[StockPredictor, pd.DataFrame]:
    data = _market_data()
    predictor = StockPredictor(
        n_splits=2,
        xgb_estimators=8,
        rf_estimators=12,
    )
    features, _ = predictor.prepare_features(data)
    predictor.fit(features, predictor.create_targets(data))
    return predictor, data


def test_targets_are_forward_returns_with_nan_tail() -> None:
    data = _market_data(20)
    predictor = StockPredictor()

    targets = predictor.create_targets(data)

    expected_1d = data["Close"].iloc[1] / data["Close"].iloc[0] - 1.0
    expected_5d = data["Close"].iloc[5] / data["Close"].iloc[0] - 1.0
    assert targets["return_1d"].iloc[0] == pytest.approx(expected_1d)
    assert targets["return_5d"].iloc[0] == pytest.approx(expected_5d)
    assert targets["direction_1d"].iloc[0] == float(expected_1d > 0)
    assert targets["return_1d"].tail(1).isna().all()
    assert targets["return_5d"].tail(5).isna().all()


def test_direct_horizon_forecasts_are_deterministic(
    fitted_predictor: tuple[StockPredictor, pd.DataFrame],
) -> None:
    predictor, data = fitted_predictor

    first = predictor.predict_multi_day(data, days=5)
    second = predictor.predict_multi_day(data, days=5)

    assert first == second
    assert [prediction["day"] for prediction in first] == [1, 3, 5]
    assert all(
        prediction["lower_price"]
        <= prediction["predicted_price"]
        <= prediction["upper_price"]
        for prediction in first
    )
    assert all(
        0.0 <= prediction["probability_up"] <= 100.0
        for prediction in first
    )


def test_validation_reports_out_of_fold_baselines(
    fitted_predictor: tuple[StockPredictor, pd.DataFrame],
) -> None:
    predictor, _ = fitted_predictor

    assert set(predictor.validation_results) == {"1d", "3d", "5d"}
    for metrics in predictor.validation_results.values():
        assert metrics["fold_count"] == 2
        assert metrics["oof_sample_count"] > 0
        assert "zero_return_baseline_mae" in metrics
        assert "historical_mean_baseline_mae" in metrics
        assert "beats_zero_baseline" in metrics
        assert 0.0 <= metrics["interval_coverage"] <= 1.0


def test_walk_forward_split_purges_target_horizon() -> None:
    predictor = StockPredictor(n_splits=3)
    splitter = predictor._time_series_splitter(sample_count=240, horizon=5)
    dummy = np.arange(240)

    for train_index, validation_index in splitter.split(dummy):
        assert validation_index[0] - train_index[-1] > 5
