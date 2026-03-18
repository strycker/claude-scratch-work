"""Tests for interpretability helpers in classifier.py."""
from __future__ import annotations

import numpy as np
import pandas as pd

from market_regime.prediction.classifier import (
    extract_top_features,
    train_interpretability_tree,
)


def _make_synthetic_data(n_samples: int = 60, n_features: int = 10, n_regimes: int = 3):
    rng = np.random.default_rng(42)
    index = pd.RangeIndex(n_samples)
    X = pd.DataFrame(
        rng.normal(size=(n_samples, n_features)),
        index=index,
        columns=[f"feat_{i}" for i in range(n_features)],
    )
    y = pd.Series(
        np.tile(np.arange(n_regimes), n_samples // n_regimes + 1)[:n_samples],
        index=index,
        name="regime",
    )
    return X, y


def test_extract_top_features_ranks_by_importance():
    from sklearn.ensemble import RandomForestClassifier

    X, y = _make_synthetic_data()
    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    rf.fit(X, y)

    top = extract_top_features(rf, list(X.columns), top_k=5)
    assert len(top) == 5
    # Sorted descending by importance
    importances = [imp for _, imp in top]
    assert importances == sorted(importances, reverse=True)
    # Feature names should be from the original columns
    for name, imp in top:
        assert name in X.columns
        assert imp >= 0.0


def test_train_interpretability_tree_uses_reduced_features():
    from sklearn.ensemble import RandomForestClassifier

    X, y = _make_synthetic_data()
    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    rf.fit(X, y)

    tree, selected = train_interpretability_tree(
        X, y, rf, top_k=4, max_depth=3
    )
    assert len(selected) == 4
    assert all(s in X.columns for s in selected)

    # Tree should be fitted and able to predict on the reduced feature set
    preds = tree.predict(X[selected])
    assert len(preds) == len(X)
    assert tree.get_depth() <= 3
