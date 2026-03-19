"""Tests for trading_crab_lib.momentum — momentum and cross-asset features."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_crab_lib.momentum import (
    add_momentum_features,
    compute_inflation_acceleration,
    compute_relative_strength,
    compute_rolling_cross_correlation,
    compute_trailing_momentum,
)


@pytest.fixture
def quarterly_df() -> pd.DataFrame:
    """Create a simple quarterly DataFrame for testing."""
    idx = pd.date_range("2000-03-31", periods=20, freq="QE")
    rng = np.random.RandomState(42)
    return pd.DataFrame(
        {
            "sp500": 1000 + np.cumsum(rng.randn(20) * 50),
            "gold_spot": 300 + np.cumsum(rng.randn(20) * 10),
            "wti_crude": 30 + np.cumsum(rng.randn(20) * 5),
            "10yr_ustreas": 5.0 + np.cumsum(rng.randn(20) * 0.2),
            "credit_spread": 1.5 + rng.rand(20) * 0.5,
            "fred_vixcls": 20 + rng.randn(20) * 5,
            "cpi": 170 + np.arange(20) * 0.8,
        },
        index=idx,
    )


# ── Trailing momentum ──────────────────────────────────────────────────────


def test_trailing_momentum_default_windows(quarterly_df):
    result = compute_trailing_momentum(quarterly_df, ["sp500", "gold_spot"])
    for w in [2, 4, 8]:
        assert f"sp500_mom_{w}q" in result.columns
        assert f"gold_spot_mom_{w}q" in result.columns
    # First w rows should be NaN for window w
    assert pd.isna(result["sp500_mom_4q"].iloc[0])
    assert pd.notna(result["sp500_mom_4q"].iloc[4])


def test_trailing_momentum_custom_windows(quarterly_df):
    result = compute_trailing_momentum(quarterly_df, ["sp500"], windows=[1, 3])
    assert "sp500_mom_1q" in result.columns
    assert "sp500_mom_3q" in result.columns
    assert "sp500_mom_2q" not in result.columns


def test_trailing_momentum_missing_column(quarterly_df):
    """Gracefully skip columns not present in DataFrame."""
    result = compute_trailing_momentum(quarterly_df, ["nonexistent", "sp500"])
    assert "sp500_mom_2q" in result.columns
    assert "nonexistent_mom_2q" not in result.columns


def test_trailing_momentum_values(quarterly_df):
    """Verify pct_change calculation is correct."""
    result = compute_trailing_momentum(quarterly_df, ["sp500"], windows=[1])
    expected = quarterly_df["sp500"].pct_change(periods=1)
    pd.testing.assert_series_equal(result["sp500_mom_1q"], expected, check_names=False)


# ── Relative strength ──────────────────────────────────────────────────────


def test_relative_strength_default_pairs(quarterly_df):
    result = compute_relative_strength(quarterly_df)
    assert "sp500_in_gold" in result.columns
    assert "sp500_in_oil" in result.columns
    assert "gold_in_oil" in result.columns


def test_relative_strength_custom_pairs(quarterly_df):
    pairs = [("sp500", "cpi", "sp500_real")]
    result = compute_relative_strength(quarterly_df, pairs=pairs)
    assert "sp500_real" in result.columns
    expected = quarterly_df["sp500"] / quarterly_df["cpi"]
    pd.testing.assert_series_equal(result["sp500_real"], expected, check_names=False)


def test_relative_strength_missing_column(quarterly_df):
    pairs = [("sp500", "nonexistent", "test")]
    result = compute_relative_strength(quarterly_df, pairs=pairs)
    assert "test" not in result.columns


def test_relative_strength_zero_denom(quarterly_df):
    """Zero denominator should produce NaN, not error."""
    quarterly_df.loc[quarterly_df.index[0], "gold_spot"] = 0
    result = compute_relative_strength(quarterly_df)
    assert pd.isna(result["sp500_in_gold"].iloc[0])


# ── Rolling cross-correlation ──────────────────────────────────────────────


def test_rolling_cross_correlation_default(quarterly_df):
    result = compute_rolling_cross_correlation(quarterly_df)
    assert "corr_sp500_10yr_ustreas_8q" in result.columns
    assert "corr_sp500_gold_spot_8q" in result.columns


def test_rolling_cross_correlation_custom(quarterly_df):
    pairs = [("sp500", "cpi", 4)]
    result = compute_rolling_cross_correlation(quarterly_df, pairs=pairs)
    assert "corr_sp500_cpi_4q" in result.columns
    # First few values should be NaN (not enough data for window)
    assert pd.isna(result["corr_sp500_cpi_4q"].iloc[0])


def test_rolling_cross_correlation_range(quarterly_df):
    """Correlation values should be in [-1, 1]."""
    result = compute_rolling_cross_correlation(quarterly_df)
    col = "corr_sp500_10yr_ustreas_8q"
    valid = result[col].dropna()
    assert (valid >= -1.0).all()
    assert (valid <= 1.0).all()


def test_rolling_cross_correlation_missing(quarterly_df):
    pairs = [("nonexistent", "sp500", 4)]
    result = compute_rolling_cross_correlation(quarterly_df, pairs=pairs)
    assert "corr_nonexistent_sp500_4q" not in result.columns


# ── Inflation acceleration ─────────────────────────────────────────────────


def test_inflation_acceleration(quarterly_df):
    result = compute_inflation_acceleration(quarterly_df)
    assert "cpi_acceleration" in result.columns
    # Linear CPI → constant d1 → zero d2 → acceleration ≈ 0
    valid = result["cpi_acceleration"].dropna()
    assert np.allclose(valid, 0, atol=1e-6)


def test_inflation_acceleration_no_cpi():
    df = pd.DataFrame({"sp500": [1, 2, 3]})
    result = compute_inflation_acceleration(df)
    assert "cpi_acceleration" not in result.columns


# ── Master wrapper ─────────────────────────────────────────────────────────


def test_add_momentum_features_default_config(quarterly_df):
    cfg = {"features": {}}
    result = add_momentum_features(quarterly_df, cfg)
    # Should have momentum columns
    mom_cols = [c for c in result.columns if "_mom_" in c]
    assert len(mom_cols) > 0
    # Should have cpi_acceleration
    assert "cpi_acceleration" in result.columns


def test_add_momentum_features_with_config(quarterly_df):
    cfg = {
        "features": {
            "momentum": {
                "momentum_columns": ["sp500"],
                "momentum_windows": [4],
                "relative_strength_pairs": [["sp500", "cpi", "sp500_real"]],
                "correlation_pairs": [["sp500", "cpi", 4]],
            }
        }
    }
    result = add_momentum_features(quarterly_df, cfg)
    assert "sp500_mom_4q" in result.columns
    assert "sp500_real" in result.columns
    assert "corr_sp500_cpi_4q" in result.columns
    # No 2q or 8q momentum since we specified only [4]
    assert "sp500_mom_2q" not in result.columns


def test_add_momentum_features_preserves_original_columns(quarterly_df):
    cfg = {"features": {}}
    result = add_momentum_features(quarterly_df, cfg)
    for col in quarterly_df.columns:
        assert col in result.columns
