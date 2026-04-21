from __future__ import annotations

import numpy as np
import pandas as pd

from trading_crab_lib.prediction.classifier import (
    FoldReport,
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
    regimes = pd.Series(
        np.tile(np.arange(n_regimes), n_samples // n_regimes + 1)[:n_samples],
        index=index,
        name="regime",
    )
    return features, regimes


def test_current_regime_cv_respects_temporal_order():
    features, y = _make_synthetic_data()
    result = train_current_regime(features, y, cv_splits=4)

    cv_reports = result["cv_reports"]
    assert set(cv_reports.keys()) == {"dt", "rf"}

    for _model_name, folds in cv_reports.items():
        assert folds, f"no folds for model {_model_name}"
        for fr in folds:
            assert isinstance(fr, FoldReport)
            assert fr.train_indices
            assert fr.test_indices
            assert max(fr.train_indices) < min(fr.test_indices)


def test_current_regime_models_and_probabilities():
    features, y = _make_synthetic_data()
    result = train_current_regime(features, y, cv_splits=3)

    models = result["models"]
    labels = result["labels"]

    assert set(models.keys()) == {"dt", "rf"}
    assert sorted(labels) == sorted(pd.unique(y))

    for _name, model in models.items():
        proba = model.predict_proba(features)
        assert proba.shape[0] == len(features)
        assert proba.shape[1] == len(model.classes_)
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(features)), rtol=1e-6)


def test_forward_regime_horizon_one_shift_and_probabilities():
    features, regimes = _make_synthetic_data()
    horizons = [1]
    results = train_forward_classifiers(features, regimes, horizons=horizons, cv_splits=3)

    assert set(results.keys()) == {1}
    h1 = results[1]

    assert set(h1["models"].keys()) == {"dt", "rf"}
    class_order = h1["class_order"]
    assert sorted(class_order) == sorted(pd.unique(regimes.dropna()))

    y_future_manual = regimes.shift(-1).dropna()
    assert len(y_future_manual) == len(regimes) - 1

    for _model_name, folds in h1["cv_reports"].items():
        assert folds, f"no folds for model {_model_name}"
        for fr in folds:
            assert isinstance(fr, FoldReport)
            assert max(fr.train_indices) < min(fr.test_indices)

    mask = regimes.shift(-1).notna()
    feat_h = features.loc[mask]
    for _name, model in h1["models"].items():
        proba = model.predict_proba(feat_h)
        assert proba.shape[0] == len(feat_h)
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(feat_h)), rtol=1e-6)
