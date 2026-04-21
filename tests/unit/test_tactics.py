"""Tests for tactics.py — tactical asset classification."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_crab_lib.tactics import (
    _realized_vol,
    _rolling_corr,
    _trend_slope,
    classify_tactics,
    compute_tactics_metrics,
)


@pytest.fixture
def quarterly_prices():
    """Synthetic quarterly prices (not returns) for 3 assets."""
    rng = np.random.default_rng(42)
    idx = pd.date_range("2000-03-31", periods=20, freq="QE")
    # SPY: smooth uptrend
    spy = pd.Series(np.linspace(100, 150, 20) + rng.normal(0, 1, 20), index=idx)
    # GLD: volatile uptrend
    gld = pd.Series(np.linspace(50, 80, 20) + rng.normal(0, 5, 20), index=idx)
    # TLT: downtrend
    tlt = pd.Series(np.linspace(120, 100, 20) + rng.normal(0, 0.5, 20), index=idx)
    return pd.DataFrame({"SPY": spy, "GLD": gld, "TLT": tlt})


@pytest.fixture
def tactics_cfg():
    return {
        "tactics": {
            "vol_windows": [5, 10],
            "vol_bands": {"low": 0.01, "high": 0.05},
            "trend_windows": [5, 10],
            "trend_min_slope": 0.0,
            "corr_lookback": 10,
        }
    }


# ── Private helper tests ────────────────────────────────────────────────────

def test_realized_vol_basic():
    rng = np.random.default_rng(0)
    s = pd.Series(rng.normal(0, 0.02, 50))
    vol = _realized_vol(s, 20)
    assert 0 < vol < 0.1


def test_realized_vol_empty():
    assert np.isnan(_realized_vol(pd.Series(dtype=float), 10))


def test_realized_vol_short_series():
    """When series < window, uses full series."""
    s = pd.Series([0.01, -0.01, 0.02])
    vol = _realized_vol(s, 100)
    assert vol > 0


def test_trend_slope_uptrend():
    prices = pd.Series(np.linspace(100, 200, 20))
    slope = _trend_slope(prices, 10)
    assert slope > 0


def test_trend_slope_downtrend():
    prices = pd.Series(np.linspace(200, 100, 20))
    slope = _trend_slope(prices, 10)
    assert slope < 0


def test_trend_slope_empty():
    assert _trend_slope(pd.Series(dtype=float), 10) == 0.0


def test_trend_slope_single_point():
    assert _trend_slope(pd.Series([100.0]), 10) == 0.0


def test_rolling_corr_identical():
    """Identical series → correlation = 1."""
    s = pd.Series(np.arange(20, dtype=float))
    assert _rolling_corr(s, s, 10) == pytest.approx(1.0)


def test_rolling_corr_empty():
    assert np.isnan(_rolling_corr(pd.Series(dtype=float), pd.Series(dtype=float), 10))


# ── compute_tactics_metrics tests ────────────────────────────────────────────

def test_compute_tactics_metrics_columns(quarterly_prices, tactics_cfg):
    metrics = compute_tactics_metrics(quarterly_prices, cfg=tactics_cfg)
    assert isinstance(metrics, pd.DataFrame)
    assert metrics.index.name == "asset"
    assert len(metrics) == 3
    # Should have vol_ and slope_ columns from config windows
    assert any(c.startswith("vol_") for c in metrics.columns)
    assert any(c.startswith("slope_") for c in metrics.columns)
    assert "corr_spy" in metrics.columns


def test_compute_tactics_metrics_no_config(quarterly_prices):
    """Works with default config when cfg=None."""
    metrics = compute_tactics_metrics(quarterly_prices)
    assert len(metrics) == 3
    assert any(c.startswith("vol_") for c in metrics.columns)


def test_compute_tactics_metrics_with_regimes(quarterly_prices, tactics_cfg):
    regimes = pd.Series([0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2, 0, 0],
                        index=quarterly_prices.index)
    metrics = compute_tactics_metrics(quarterly_prices, regimes=regimes, cfg=tactics_cfg)
    assert "current_regime" in metrics.columns
    assert metrics["current_regime"].iloc[0] == 0  # last regime value


def test_compute_tactics_metrics_volatility_positive(quarterly_prices, tactics_cfg):
    metrics = compute_tactics_metrics(quarterly_prices, cfg=tactics_cfg)
    vol_cols = [c for c in metrics.columns if c.startswith("vol_")]
    for c in vol_cols:
        valid = metrics[c].dropna()
        assert (valid >= 0).all()


# ── classify_tactics tests ───────────────────────────────────────────────────

def test_classify_tactics_all_categories(quarterly_prices, tactics_cfg):
    metrics = compute_tactics_metrics(quarterly_prices, cfg=tactics_cfg)
    classified = classify_tactics(metrics, cfg=tactics_cfg)
    assert "tactics_label" in classified.columns
    valid_tactics = {"buy_hold", "swing", "stand_aside"}
    for t in classified["tactics_label"]:
        assert t in valid_tactics


def test_classify_tactics_no_vol_cols():
    """When no vol/slope columns, everything is stand_aside."""
    df = pd.DataFrame({"x": [1, 2]}, index=["A", "B"])
    result = classify_tactics(df)
    assert (result["tactics_label"] == "stand_aside").all()


def test_classify_tactics_buy_hold():
    """Low vol + positive slope → buy_hold."""
    df = pd.DataFrame({"vol_20": [0.005], "slope_20": [0.01]}, index=["STEADY"])
    cfg = {"tactics": {"vol_bands": {"low": 0.01, "high": 0.05}, "trend_min_slope": 0.0}}
    result = classify_tactics(df, cfg=cfg)
    assert result.iloc[0]["tactics_label"] == "buy_hold"


def test_classify_tactics_swing():
    """Medium vol + positive slope → swing."""
    df = pd.DataFrame({"vol_20": [0.03], "slope_20": [0.01]}, index=["ACTIVE"])
    cfg = {"tactics": {"vol_bands": {"low": 0.01, "high": 0.05}, "trend_min_slope": 0.0}}
    result = classify_tactics(df, cfg=cfg)
    assert result.iloc[0]["tactics_label"] == "swing"


def test_classify_tactics_stand_aside_downtrend():
    """Negative slope → stand_aside regardless of vol."""
    df = pd.DataFrame({"vol_20": [0.005], "slope_20": [-0.01]}, index=["DOWN"])
    cfg = {"tactics": {"vol_bands": {"low": 0.01, "high": 0.05}, "trend_min_slope": 0.0}}
    result = classify_tactics(df, cfg=cfg)
    assert result.iloc[0]["tactics_label"] == "stand_aside"


def test_classify_tactics_stand_aside_high_vol():
    """Very high vol → stand_aside even with positive slope."""
    df = pd.DataFrame({"vol_20": [0.10], "slope_20": [0.01]}, index=["WILD"])
    cfg = {"tactics": {"vol_bands": {"low": 0.01, "high": 0.05}, "trend_min_slope": 0.0}}
    result = classify_tactics(df, cfg=cfg)
    assert result.iloc[0]["tactics_label"] == "stand_aside"
