"""Tests for interpretability helpers in classifier.py."""
from __future__ import annotations

import numpy as np
import pandas as pd

from trading_crab_lib.prediction.classifier import (
    extract_top_features,
    train_interpretability_tree,
)


def _make_synthetic_data(n_samples: int = 60, n_features: int = 10, n_regimes: int = 3):
    rng = np.random.default_rng(42)
    index = pd.RangeIndex(n_samples)
    features = pd.DataFrame(
        rng.normal(size=(n_samples, n_features)),
        index=index,
        columns=[f"feat_{i}" for i in range(n_features)],
    )
    y = pd.Series(
        np.tile(np.arange(n_regimes), n_samples // n_regimes + 1)[:n_samples],
        index=index,
        name="regime",
    )
    return features, y


def test_extract_top_features_ranks_by_importance():
    from sklearn.ensemble import RandomForestClassifier

    features, y = _make_synthetic_data()
    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    rf.fit(features, y)

    top = extract_top_features(rf, list(features.columns), top_k=5)
    assert len(top) == 5
    importances = [imp for _, imp in top]
    assert importances == sorted(importances, reverse=True)
    for name, imp in top:
        assert name in features.columns
        assert imp >= 0.0


def test_train_interpretability_tree_uses_reduced_features():
    from sklearn.ensemble import RandomForestClassifier

    features, y = _make_synthetic_data()
    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    rf.fit(features, y)

    tree, selected = train_interpretability_tree(
        features, y, rf, top_k=4, max_depth=3
    )
    assert len(selected) == 4
    assert all(s in features.columns for s in selected)

    preds = tree.predict(features[selected])
    assert len(preds) == len(features)
    assert tree.get_depth() <= 3
