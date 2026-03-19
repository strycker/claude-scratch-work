"""Tests for trading_crab_lib.divergence — cross-asset divergence features."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_crab_lib.divergence import (
    DEFAULT_DIVERGENCE_PAIRS,
    add_divergence_features,
    compute_derivative_divergence,
    compute_divergence,
    compute_divergence_triggers,
    compute_rolling_correlation,
)


@pytest.fixture
def quarterly_df() -> pd.DataFrame:
    """Quarterly DataFrame with correlated and anti-correlated pairs."""
    idx = pd.date_range("1990-03-31", periods=60, freq="QE")
    rng = np.random.RandomState(42)

    # S&P500: trending up with noise
    sp500 = 1000 + np.cumsum(rng.randn(60) * 30)
    # 10Y yield: loosely inversely correlated with equities
    ustreas = 6.0 - np.cumsum(rng.randn(60) * 0.1)
    # Gold: loosely correlated with inflation proxy
    gold = 400 + np.cumsum(rng.randn(60) * 15)
    # Oil: correlated with gold in normal times
    oil = 30 + np.cumsum(rng.randn(60) * 3)
    # Credit spread: widens during stress
    credit = 1.5 + rng.rand(60) * 0.5
    # VIX: spikes during credit stress
    vix = 20 + rng.randn(60) * 5

    return pd.DataFrame(
        {
            "sp500": sp500,
            "10yr_ustreas": ustreas,
            "gold_spot": gold,
            "wti_crude": oil,
            "credit_spread": credit,
            "fred_vixcls": vix,
        },
        index=idx,
    )


@pytest.fixture
def quarterly_df_with_d1(quarterly_df) -> pd.DataFrame:
    """Same as quarterly_df but with d1 derivative columns."""
    df = quarterly_df.copy()
    for col in quarterly_df.columns:
        df[f"{col}_d1"] = df[col].diff()
    return df


# ── Layer 1: Rolling correlation ───────────────────────────────────────────


class TestRollingCorrelation:
    def test_returns_series(self, quarterly_df):
        result = compute_rolling_correlation(
            quarterly_df["sp500"], quarterly_df["gold_spot"], window=8
        )
        assert isinstance(result, pd.Series)
        assert len(result) == len(quarterly_df)

    def test_values_in_range(self, quarterly_df):
        result = compute_rolling_correlation(
            quarterly_df["sp500"], quarterly_df["gold_spot"], window=8
        )
        valid = result.dropna()
        assert (valid >= -1.0).all()
        assert (valid <= 1.0).all()

    def test_early_values_are_nan(self, quarterly_df):
        result = compute_rolling_correlation(
            quarterly_df["sp500"], quarterly_df["gold_spot"], window=8
        )
        # First few values should be NaN (insufficient data)
        assert pd.isna(result.iloc[0])

    def test_perfect_correlation(self):
        idx = pd.date_range("2000-03-31", periods=20, freq="QE")
        a = pd.Series(np.arange(20, dtype=float), index=idx)
        b = pd.Series(np.arange(20, dtype=float) * 2 + 1, index=idx)
        result = compute_rolling_correlation(a, b, window=8, use_returns=False)
        valid = result.dropna()
        assert np.allclose(valid, 1.0, atol=1e-10)

    def test_use_returns_false(self, quarterly_df):
        result = compute_rolling_correlation(
            quarterly_df["sp500"], quarterly_df["gold_spot"],
            window=8, use_returns=False,
        )
        valid = result.dropna()
        assert len(valid) > 0


# ── Layer 2: Divergence magnitude ──────────────────────────────────────────


class TestDivergence:
    def test_returns_expected_keys(self, quarterly_df):
        result = compute_divergence(
            quarterly_df["sp500"], quarterly_df["gold_spot"],
            short_window=4, long_window=12,
        )
        assert set(result.keys()) == {
            "corr_short", "corr_long", "div_raw", "div_abs", "div_zscore",
        }

    def test_all_values_are_series(self, quarterly_df):
        result = compute_divergence(
            quarterly_df["sp500"], quarterly_df["gold_spot"],
        )
        for key, val in result.items():
            assert isinstance(val, pd.Series), f"{key} is not a Series"
            assert len(val) == len(quarterly_df)

    def test_div_raw_is_short_minus_long(self, quarterly_df):
        result = compute_divergence(
            quarterly_df["sp500"], quarterly_df["gold_spot"],
            short_window=4, long_window=12,
        )
        expected = result["corr_short"] - result["corr_long"]
        pd.testing.assert_series_equal(result["div_raw"], expected, check_names=False)

    def test_div_abs_is_nonnegative(self, quarterly_df):
        result = compute_divergence(
            quarterly_df["sp500"], quarterly_df["gold_spot"],
        )
        valid = result["div_abs"].dropna()
        assert (valid >= 0).all()

    def test_div_abs_equals_abs_of_raw(self, quarterly_df):
        result = compute_divergence(
            quarterly_df["sp500"], quarterly_df["gold_spot"],
        )
        expected = result["div_raw"].abs()
        pd.testing.assert_series_equal(result["div_abs"], expected, check_names=False)

    def test_custom_windows(self, quarterly_df):
        result = compute_divergence(
            quarterly_df["sp500"], quarterly_df["gold_spot"],
            short_window=2, long_window=8,
        )
        # Short window should have more valid values than long window
        n_short = result["corr_short"].notna().sum()
        n_long = result["corr_long"].notna().sum()
        assert n_short >= n_long


# ── Layer 3: Triggers ──────────────────────────────────────────────────────


class TestDivergenceTriggers:
    def test_returns_expected_keys(self):
        zscore = pd.Series([0.5, -1.5, 2.5, -3.0, 0.1])
        result = compute_divergence_triggers(zscore)
        assert set(result.keys()) == {"trigger", "direction"}

    def test_trigger_binary(self):
        zscore = pd.Series([0.5, -1.5, 2.5, -3.0, 0.1])
        result = compute_divergence_triggers(zscore, threshold=2.0)
        assert set(result["trigger"].unique()) <= {0, 1}
        # 2.5 and -3.0 exceed threshold
        assert result["trigger"].iloc[2] == 1
        assert result["trigger"].iloc[3] == 1
        assert result["trigger"].iloc[0] == 0

    def test_direction_values(self):
        zscore = pd.Series([0.5, -1.5, 0.0, 2.5])
        result = compute_divergence_triggers(zscore)
        assert result["direction"].iloc[0] == 1   # positive
        assert result["direction"].iloc[1] == -1  # negative
        assert result["direction"].iloc[2] == 0   # zero
        assert result["direction"].iloc[3] == 1   # positive

    def test_custom_threshold(self):
        zscore = pd.Series([1.0, 1.5, 2.0, 2.5])
        low = compute_divergence_triggers(zscore, threshold=0.5)
        high = compute_divergence_triggers(zscore, threshold=3.0)
        # Lower threshold → more triggers
        assert low["trigger"].sum() >= high["trigger"].sum()

    def test_nan_handling(self):
        zscore = pd.Series([np.nan, 2.5, np.nan, -3.0])
        result = compute_divergence_triggers(zscore, threshold=2.0)
        # NaN z-score should produce trigger=0, direction=0
        assert result["trigger"].iloc[0] == 0
        assert result["direction"].iloc[0] == 0


# ── Derivative-space divergence ────────────────────────────────────────────


class TestDerivativeDivergence:
    def test_returns_none_without_d1_columns(self, quarterly_df):
        result = compute_derivative_divergence(quarterly_df, "sp500", "gold_spot")
        assert result is None

    def test_returns_dict_with_d1_columns(self, quarterly_df_with_d1):
        result = compute_derivative_divergence(
            quarterly_df_with_d1, "sp500", "gold_spot",
        )
        assert result is not None
        assert "div_raw" in result
        assert "div_zscore" in result

    def test_uses_levels_not_returns(self, quarterly_df_with_d1):
        """Derivative divergence correlates d1 levels, not d1 returns."""
        result = compute_derivative_divergence(
            quarterly_df_with_d1, "sp500", "gold_spot",
            short_window=4, long_window=12,
        )
        # Should have non-NaN values (derivatives are already rate-of-change)
        assert result is not None
        valid = result["div_raw"].dropna()
        assert len(valid) > 0


# ── Master wrapper ─────────────────────────────────────────────────────────


class TestAddDivergenceFeatures:
    def test_adds_level_divergence_columns(self, quarterly_df):
        cfg = {"features": {}}
        result = add_divergence_features(quarterly_df, cfg)
        # Should have columns for default pairs
        assert "div_spy_tlt_4q" in result.columns
        assert "div_spy_gld_4q" in result.columns
        assert "div_spy_tlt_trigger" in result.columns
        assert "div_spy_tlt_dir" in result.columns

    def test_preserves_original_columns(self, quarterly_df):
        cfg = {"features": {}}
        result = add_divergence_features(quarterly_df, cfg)
        for col in quarterly_df.columns:
            assert col in result.columns

    def test_adds_derivative_divergence_when_d1_exists(self, quarterly_df_with_d1):
        cfg = {"features": {}}
        result = add_divergence_features(quarterly_df_with_d1, cfg)
        # Should have d1 divergence columns
        assert "div_d1_spy_tlt_4q" in result.columns
        assert "div_d1_spy_tlt_trigger" in result.columns

    def test_no_derivative_divergence_without_d1(self, quarterly_df):
        cfg = {"features": {}}
        result = add_divergence_features(quarterly_df, cfg)
        d1_cols = [c for c in result.columns if c.startswith("div_d1_")]
        assert len(d1_cols) == 0

    def test_custom_config(self, quarterly_df):
        cfg = {
            "features": {
                "divergence": {
                    "pairs": [["sp500", "gold_spot", "test_pair"]],
                    "short_window": 3,
                    "long_window": 8,
                    "trigger_threshold": 1.5,
                }
            }
        }
        result = add_divergence_features(quarterly_df, cfg)
        assert "div_test_pair_3q" in result.columns
        assert "div_test_pair_trigger" in result.columns
        # Should NOT have default pairs when config overrides
        assert "div_spy_tlt_4q" not in result.columns

    def test_missing_pair_columns_skipped(self):
        df = pd.DataFrame({"sp500": [1, 2, 3]})
        cfg = {"features": {}}
        result = add_divergence_features(df, cfg)
        # All default pairs require columns not in df → no divergence columns added
        div_cols = [c for c in result.columns if c.startswith("div_")]
        assert len(div_cols) == 0

    def test_count_features_per_pair(self, quarterly_df):
        """Each level-space pair should produce exactly 5 columns."""
        cfg = {
            "features": {
                "divergence": {
                    "pairs": [["sp500", "gold_spot", "test"]],
                }
            }
        }
        result = add_divergence_features(quarterly_df, cfg)
        new_cols = [c for c in result.columns if c not in quarterly_df.columns]
        assert len(new_cols) == 5  # raw, abs, z, trigger, dir

    def test_count_features_with_d1(self, quarterly_df_with_d1):
        """Each pair with d1 columns should produce 5 + 3 = 8 columns."""
        cfg = {
            "features": {
                "divergence": {
                    "pairs": [["sp500", "gold_spot", "test"]],
                }
            }
        }
        result = add_divergence_features(quarterly_df_with_d1, cfg)
        new_cols = [c for c in result.columns if c not in quarterly_df_with_d1.columns]
        assert len(new_cols) == 8  # 5 level + 3 derivative

    def test_trigger_values_binary(self, quarterly_df):
        cfg = {"features": {}}
        result = add_divergence_features(quarterly_df, cfg)
        trigger_cols = [c for c in result.columns if c.endswith("_trigger")]
        for col in trigger_cols:
            valid = result[col].dropna()
            assert set(valid.unique()) <= {0, 1}

    def test_direction_values_ternary(self, quarterly_df):
        cfg = {"features": {}}
        result = add_divergence_features(quarterly_df, cfg)
        dir_cols = [c for c in result.columns if c.endswith("_dir")]
        for col in dir_cols:
            valid = result[col].dropna()
            assert set(valid.unique()) <= {-1, 0, 1}
