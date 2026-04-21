"""Tests for prediction.gradient_boosting (LightGBM classifiers).

These tests are skipped if lightgbm is not installed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm", reason="lightgbm not installed")

from trading_crab_lib.prediction.gradient_boosting import (  # pylint: disable=wrong-import-position
    train_lightgbm_current_regime,
    train_lightgbm_forward,
)


def _make_synthetic_data(n: int = 100, n_regimes: int = 3):
    rng = np.random.RandomState(42)
    idx = pd.period_range("1990Q1", periods=n, freq="Q")
    features = pd.DataFrame(
        rng.randn(n, 10),
        index=idx,
        columns=[f"f{i}" for i in range(10)],
    )
    y = pd.Series(rng.randint(0, n_regimes, size=n), index=idx, name="regime")
    return features, y


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
    features, y = _make_synthetic_data()
    model = train_lightgbm_current_regime(features, y, _cfg())

    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")
    preds = model.predict(features)
    assert len(preds) == len(features)


def test_train_lightgbm_forward_returns_nested_dict():
    features, y = _make_synthetic_data()
    result = train_lightgbm_forward(features, y, _cfg())

    assert isinstance(result, dict)
    assert set(result.keys()) == {1, 2}

    for _h, regime_models in result.items():
        assert isinstance(regime_models, dict)
        for _regime_id, model in regime_models.items():
            assert hasattr(model, "predict")


def test_lightgbm_predictions_are_valid_classes():
    features, y = _make_synthetic_data(n=80, n_regimes=3)
    model = train_lightgbm_current_regime(features, y, _cfg())
    preds = model.predict(features)
    assert set(preds).issubset({0, 1, 2})
