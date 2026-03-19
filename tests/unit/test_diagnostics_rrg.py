"""Tests for diagnostics.py — RRG analysis and rolling statistics."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_crab_lib.diagnostics import (
    compute_rrg,
    normalize_100,
    percentile_rank,
    rolling_zscore,
)


@pytest.fixture
def quarterly_returns():
    """Synthetic quarterly returns for 3 assets + benchmark."""
    rng = np.random.default_rng(42)
    idx = pd.date_range("2000-03-31", periods=40, freq="QE")
    return pd.DataFrame(
        {
            "SPY": rng.normal(0.02, 0.05, 40),
            "GLD": rng.normal(0.01, 0.03, 40),
            "TLT": rng.normal(0.005, 0.02, 40),
        },
        index=idx,
    )


def test_rolling_zscore_constant_series():
    """A constant series should produce NaN z-scores (zero std)."""
    s = pd.Series([5.0] * 20)
    result = rolling_zscore(s, window=10)
    # Constant → std=0 → NaN
    assert result.isna().all()


def test_rolling_zscore_varying_series():
    """A linearly increasing series should have meaningful z-scores."""
    s = pd.Series(np.arange(20, dtype=float))
    result = rolling_zscore(s, window=5)
    # Should not be all NaN
    assert not result.isna().all()
    # Last value should be positive (above the rolling mean)
    assert result.iloc[-1] > 0


def test_percentile_rank_simple():
    """Percentile rank of a sorted ascending series should be ~1.0 at the end."""
    s = pd.Series(np.arange(20, dtype=float))
    result = percentile_rank(s, window=10)
    # The last value (19.0) is the max in its window → rank should be 1.0
    assert result.iloc[-1] == pytest.approx(1.0)
    # All ranks should be in [0, 1]
    assert (result >= 0).all() and (result <= 1).all()


def test_normalize_100_centering():
    """A constant series should normalize to exactly 100."""
    s = pd.Series([50.0] * 20)
    result = normalize_100(s, center_window=10)
    np.testing.assert_allclose(result.values, 100.0, atol=1e-10)


def test_normalize_100_above_and_below():
    """Values above rolling mean should normalize above 100."""
    # Create data where last values are clearly above the rolling mean
    s = pd.Series([10.0] * 15 + [20.0] * 5)
    result = normalize_100(s, center_window=10)
    # The last value (20.0) is well above the rolling mean → should be > 100
    assert result.iloc[-1] > 100


def test_compute_rrg_quadrants(quarterly_returns):
    """compute_rrg should classify each asset into a valid quadrant."""
    benchmark = quarterly_returns["SPY"]
    assets = quarterly_returns[["GLD", "TLT"]]

    result = compute_rrg(assets, benchmark, rs_window=8, rm_window=4)
    assert isinstance(result, pd.DataFrame)
    assert "asset" in result.columns
    assert "quadrant" in result.columns

    valid_quadrants = {"LEADING", "WEAKENING", "LAGGING", "IMPROVING"}
    for _, row in result.iterrows():
        assert row["quadrant"] in valid_quadrants
        assert isinstance(row["rs"], float)
        assert isinstance(row["rm"], float)


def test_compute_rrg_insufficient_data():
    """Assets with too little data should be skipped."""
    idx = pd.date_range("2020-03-31", periods=3, freq="QE")
    assets = pd.DataFrame({"X": [0.01, 0.02, 0.03]}, index=idx)
    bench = pd.Series([0.01, 0.02, 0.03], index=idx)

    result = compute_rrg(assets, bench, rs_window=8, rm_window=4)
    # Only 3 data points < rs_window=8 → should skip
    assert len(result) == 0
