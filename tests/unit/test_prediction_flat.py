"""Unit tests for the flat prediction API (prediction/__init__.py).

These test the production API used by run_pipeline.py and pipelines/*.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from market_regime.prediction import (
    predict_current,
    train_classifier,
    train_current_regime,
    train_decision_tree,
    train_forward_classifiers,
)


@pytest.fixture
def synthetic_data():
    rng = np.random.default_rng(42)
    n, n_feat, n_regimes = 60, 8, 4
    idx = pd.date_range("2000-03-31", periods=n, freq="QE")
    X = pd.DataFrame(
        rng.normal(size=(n, n_feat)),
        index=idx,
        columns=[f"feat_{i}" for i in range(n_feat)],
    )
    y = pd.Series(
        np.tile(np.arange(n_regimes), n // n_regimes + 1)[:n],
        index=idx,
        name="regime",
    )
    return X, y


@pytest.fixture
def cfg():
    return {
        "prediction": {
            "cv_splits": 3,
            "n_estimators": 50,
            "rf_max_depth": 6,
            "dt_max_depth": 4,
            "random_state": 42,
            "forward_horizons_quarters": [1, 2],
        }
    }


def test_train_current_regime_returns_rf(synthetic_data, cfg):
    X, y = synthetic_data
    model = train_current_regime(X, y, cfg)
    assert isinstance(model, RandomForestClassifier)
    assert hasattr(model, "predict_proba")
    proba = model.predict_proba(X)
    assert proba.shape[0] == len(X)


def test_train_decision_tree_returns_dt(synthetic_data, cfg):
    X, y = synthetic_data
    model = train_decision_tree(X, y, cfg)
    assert isinstance(model, DecisionTreeClassifier)
    assert model.get_depth() <= cfg["prediction"]["dt_max_depth"]


def test_train_classifier_invalid_kind_raises(synthetic_data, cfg):
    X, y = synthetic_data
    with pytest.raises(ValueError, match="kind must be"):
        train_classifier(X, y, cfg, kind="xgb")


def test_predict_current_returns_regime_and_probs(synthetic_data, cfg):
    X, y = synthetic_data
    model = train_current_regime(X, y, cfg)
    result = predict_current(model, X)
    assert "regime" in result
    assert "probabilities" in result
    assert isinstance(result["regime"], int)
    assert result["regime"] in y.unique()
    probs = result["probabilities"]
    assert abs(sum(probs.values()) - 1.0) < 1e-6


def test_train_forward_classifiers_structure(synthetic_data, cfg):
    X, y = synthetic_data
    results = train_forward_classifiers(X, y, cfg)
    assert set(results.keys()) == {1, 2}
    for h, per_regime in results.items():
        for regime_id, model in per_regime.items():
            assert isinstance(model, RandomForestClassifier)
            # Binary classifier: classes should be [0, 1]
            assert len(model.classes_) == 2
