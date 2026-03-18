"""Tests for tactics.py — tactical asset classification."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from market_regime.tactics import classify_tactics, compute_tactics_metrics


@pytest.fixture
def quarterly_returns():
    rng = np.random.default_rng(42)
    idx = pd.date_range("2000-03-31", periods=20, freq="QE")
    return pd.DataFrame(
        {
            "SPY": rng.normal(0.03, 0.04, 20),   # trending, moderate vol
            "GLD": rng.normal(0.01, 0.10, 20),    # trending, high vol
            "TLT": rng.normal(-0.01, 0.02, 20),   # downtrending, low vol
        },
        index=idx,
    )


def test_compute_tactics_metrics_columns(quarterly_returns):
    metrics = compute_tactics_metrics(quarterly_returns)
    assert isinstance(metrics, pd.DataFrame)
    assert set(metrics.columns) >= {"asset", "volatility", "trend_slope"}
    assert len(metrics) == 3


def test_compute_tactics_metrics_with_benchmark(quarterly_returns):
    benchmark = quarterly_returns["SPY"]
    assets = quarterly_returns[["GLD", "TLT"]]
    metrics = compute_tactics_metrics(assets, benchmark_returns=benchmark)
    assert "correlation" in metrics.columns
    # Correlation should be in [-1, 1] where not NaN
    valid = metrics["correlation"].dropna()
    assert (valid >= -1.0).all() and (valid <= 1.0).all()


def test_compute_tactics_metrics_volatility_positive(quarterly_returns):
    metrics = compute_tactics_metrics(quarterly_returns)
    assert (metrics["volatility"] >= 0).all()


def test_classify_tactics_all_categories(quarterly_returns):
    metrics = compute_tactics_metrics(quarterly_returns)
    classified = classify_tactics(metrics, vol_threshold=0.10, trend_threshold=0.0)
    assert "tactic" in classified.columns
    valid_tactics = {"buy_hold", "swing", "stand_aside"}
    for t in classified["tactic"]:
        assert t in valid_tactics


def test_classify_tactics_downtrend_is_stand_aside():
    """An asset with negative trend should always be stand_aside."""
    metrics = pd.DataFrame([{
        "asset": "DOWN",
        "volatility": 0.05,
        "trend_slope": -0.01,
        "correlation": 0.5,
    }])
    result = classify_tactics(metrics, vol_threshold=0.10, trend_threshold=0.0)
    assert result.iloc[0]["tactic"] == "stand_aside"


def test_classify_tactics_low_vol_uptrend_is_buy_hold():
    """Low vol + positive trend → buy_hold."""
    metrics = pd.DataFrame([{
        "asset": "STEADY",
        "volatility": 0.05,
        "trend_slope": 0.02,
        "correlation": 0.8,
    }])
    result = classify_tactics(metrics, vol_threshold=0.10, trend_threshold=0.0)
    assert result.iloc[0]["tactic"] == "buy_hold"


def test_classify_tactics_high_vol_uptrend_is_swing():
    """High vol + positive trend → swing."""
    metrics = pd.DataFrame([{
        "asset": "VOLATILE",
        "volatility": 0.25,
        "trend_slope": 0.02,
        "correlation": 0.5,
    }])
    result = classify_tactics(metrics, vol_threshold=0.10, trend_threshold=0.0)
    assert result.iloc[0]["tactic"] == "swing"
