"""Tests for GradientBoosting support in the bundle API (classifier.py)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from trading_crab_lib.prediction.classifier import (
    train_current_regime,
    train_forward_classifiers,
)


def _make_synthetic_data(n_samples: int = 40, n_features: int = 5, n_regimes: int = 3):
    rng = np.random.default_rng(42)
    index = pd.RangeIndex(n_samples)
    features = pd.DataFrame(
        rng.normal(size=(n_samples, n_features)),
        index=index,
        columns=[f"f{i}" for i in range(n_features)],
    )
    y = pd.Series(
        np.tile(np.arange(n_regimes), n_samples // n_regimes + 1)[:n_samples],
        index=index,
        name="regime",
    )
    return features, y


def test_train_current_regime_includes_gb_when_enabled():
    features, y = _make_synthetic_data()
    result = train_current_regime(features, y, cv_splits=3, include_gb=True)

    assert "gb" in result["models"]
    assert "gb" in result["cv_reports"]

    gb_model = result["models"]["gb"]
    proba = gb_model.predict_proba(features)
    assert proba.shape[0] == len(features)
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(features)), rtol=1e-6)


def test_train_forward_classifiers_supports_gb_flag():
    features, y = _make_synthetic_data()
    results = train_forward_classifiers(
        features, y, horizons=[1], cv_splits=3, include_gb=True
    )

    h1 = results[1]
    assert "gb" in h1["models"]
    assert "gb" in h1["cv_reports"]

    gb_model = h1["models"]["gb"]
    y_future = y.shift(-1).dropna()
    feat_aligned = features.loc[y_future.index]
    preds = gb_model.predict(feat_aligned)
    assert len(preds) == len(feat_aligned)
