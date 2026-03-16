from __future__ import annotations

import numpy as np
import pandas as pd

from market_regime.prediction import (
    make_behavior_labels,
    train_forward_behavior_models,
)


def test_make_behavior_labels_series_assigns_up_flat_down() -> None:
    idx = pd.period_range("2000Q1", periods=6, freq="Q")
    # Chosen so that, with thresholds 0 and 0, we see all three classes.
    s = pd.Series([-0.03, -0.01, 0.0, 0.01, 0.03, 0.02], index=idx)

    labels = make_behavior_labels(s, horizon=1, up_threshold=0.0, down_threshold=0.0)
    # Last horizon period should be dropped
    assert labels.index[-1] == idx[-2]

    # Check a few concrete points
    assert labels.loc[idx[0]] == "down"   # future return -0.01 <= 0.0
    assert labels.loc[idx[1]] == "flat"   # future return 0.0 between thresholds
    assert labels.loc[idx[2]] == "up"     # future return 0.01 >= 0.0


def test_make_behavior_labels_drops_trailing_periods() -> None:
    idx = pd.period_range("2000Q1", periods=8, freq="Q")
    s = pd.Series(np.linspace(-0.02, 0.02, len(idx)), index=idx)

    horizon = 2
    labels = make_behavior_labels(s, horizon=horizon, up_threshold=0.0, down_threshold=0.0)

    # We should lose exactly `horizon` trailing periods.
    assert len(labels) == len(idx) - horizon
    assert labels.index.min() == idx[0]
    assert labels.index.max() == idx[-(horizon + 1)]


def test_train_forward_behavior_models_trains_per_asset_and_horizon() -> None:
    n = 40
    idx = pd.period_range("2000Q1", periods=n, freq="Q")
    rng = np.random.default_rng(0)

    features = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
        },
        index=idx,
    )
    regimes = pd.Series(rng.integers(0, 3, size=n), index=idx)

    # Construct two synthetic ETF return series with gentle trends so that
    # multiple behavior classes are present.
    returns = pd.DataFrame(
        {
            "ETF1": np.linspace(-0.05, 0.08, n),
            "ETF2": np.linspace(0.08, -0.05, n),
        },
        index=idx,
    )

    horizons = [1]
    results = train_forward_behavior_models(features, regimes, returns, horizons=horizons)

    assert "models" in results and "cv_reports" in results
    assert set(results["models"].keys()) == {"ETF1", "ETF2"}

    for asset in ("ETF1", "ETF2"):
        model = results["models"][asset][1]

        # Build the joint feature matrix in the same way as the helper for a
        # simple predict_proba sanity check.
        labels = make_behavior_labels(
            returns[asset],
            horizon=1,
            up_threshold=0.0,
            down_threshold=0.0,
        )
        idx_joint = labels.index.intersection(features.index).intersection(regimes.index)
        X_joint = features.loc[idx_joint].copy()
        X_joint["regime"] = regimes.loc[idx_joint].astype(int)

        proba = model.predict_proba(X_joint)
        # Probabilities should sum to 1 along the class axis.
        assert np.allclose(proba.sum(axis=1), 1.0)

        # At least two behavior classes should be represented.
        classes = set(model.classes_)
        assert len(classes) >= 2

