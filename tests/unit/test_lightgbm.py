"""Tests for prediction.gradient_boosting (LightGBM classifiers).

These tests are skipped if lightgbm is not installed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

lgb = pytest.importorskip("lightgbm", reason="lightgbm not installed")

from trading_crab.prediction.gradient_boosting import (
    train_lightgbm_current_regime,
    train_lightgbm_forward,
)


def _make_synthetic_data(n: int = 100, n_regimes: int = 3):
    """Create synthetic features + labels for testing."""
    rng = np.random.RandomState(42)
    idx = pd.period_range("1990Q1", periods=n, freq="Q")
    X = pd.DataFrame(
        rng.randn(n, 10),
        index=idx,
        columns=[f"f{i}" for i in range(10)],
    )
    y = pd.Series(rng.randint(0, n_regimes, size=n), index=idx, name="regime")
    return X, y


def _cfg():
    return {
        "prediction": {
            "cv_splits": 3,
            "random_state": 42,
            "forward_horizons_quarters": [1, 2],
            "lgbm_n_estimators": 20,
            "lgbm_max_depth": 3,
            "lgbm_num_leaves": 8,
        }
    }


def test_train_lightgbm_current_regime_returns_fitted_model():
    X, y = _make_synthetic_data()
    model = train_lightgbm_current_regime(X, y, _cfg())

    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")
    preds = model.predict(X)
    assert len(preds) == len(X)


def test_train_lightgbm_forward_returns_nested_dict():
    X, y = _make_synthetic_data()
    result = train_lightgbm_forward(X, y, _cfg())

    assert isinstance(result, dict)
    assert set(result.keys()) == {1, 2}

    for h, regime_models in result.items():
        assert isinstance(regime_models, dict)
        for regime_id, model in regime_models.items():
            assert hasattr(model, "predict")


def test_lightgbm_predictions_are_valid_classes():
    X, y = _make_synthetic_data(n=80, n_regimes=3)
    model = train_lightgbm_current_regime(X, y, _cfg())
    preds = model.predict(X)
    assert set(preds).issubset({0, 1, 2})
