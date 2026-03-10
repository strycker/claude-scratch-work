"""Unit tests for src/market_regime/features/transforms.py"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from market_regime.transforms import (
    add_cross_ratios,
    apply_log_transforms,
    apply_gap_fill,
    apply_derivatives,
    select_features,
)


# ── add_cross_ratios ───────────────────────────────────────────────────────

class TestAddCrossRatios:
    def test_all_ten_columns_added(self, raw_macro_df):
        result = add_cross_ratios(raw_macro_df)
        expected = [
            "div_yield2", "price_div", "price_gdp", "price_gdp2", "price_gnp2",
            "div_minus_baa", "credit_spread", "real_price2", "real_price3",
            "real_price_gdp2",
        ]
        for col in expected:
            assert col in result.columns, f"Missing column: {col}"

    def test_input_columns_preserved(self, raw_macro_df):
        result = add_cross_ratios(raw_macro_df)
        for col in raw_macro_df.columns:
            assert col in result.columns

    def test_div_yield2_formula(self, raw_macro_df):
        result = add_cross_ratios(raw_macro_df)
        expected = raw_macro_df["dividend"] / raw_macro_df["sp500"]
        pd.testing.assert_series_equal(result["div_yield2"], expected, check_names=False)

    def test_credit_spread_formula(self, raw_macro_df):
        result = add_cross_ratios(raw_macro_df)
        expected = (raw_macro_df["fred_baa"] - raw_macro_df["fred_aaa"]) / 100.0
        pd.testing.assert_series_equal(result["credit_spread"], expected, check_names=False)

    def test_does_not_mutate_input(self, raw_macro_df):
        original_cols = list(raw_macro_df.columns)
        add_cross_ratios(raw_macro_df)
        assert list(raw_macro_df.columns) == original_cols


# ── apply_log_transforms ───────────────────────────────────────────────────

class TestApplyLogTransforms:
    def test_adds_log_columns(self, raw_macro_df):
        cols = ["sp500", "gdp"]
        result = apply_log_transforms(raw_macro_df, cols)
        assert "log_sp500" in result.columns
        assert "log_gdp" in result.columns

    def test_log_values_correct(self, raw_macro_df):
        result = apply_log_transforms(raw_macro_df, ["sp500"])
        expected = np.log(raw_macro_df["sp500"].clip(lower=1e-9))
        pd.testing.assert_series_equal(result["log_sp500"], expected, check_names=False)

    def test_clips_at_1e9(self, quarterly_index):
        df = pd.DataFrame({"x": [-5.0, 0.0, 1.0, 100.0]},
                          index=quarterly_index[:4])
        result = apply_log_transforms(df, ["x"])
        assert np.all(np.isfinite(result["log_x"].values))

    def test_skips_missing_column(self, raw_macro_df):
        # Should not raise; just skip
        result = apply_log_transforms(raw_macro_df, ["nonexistent_col"])
        assert "log_nonexistent_col" not in result.columns

    def test_does_not_mutate_input(self, raw_macro_df):
        original_cols = list(raw_macro_df.columns)
        apply_log_transforms(raw_macro_df, ["sp500"])
        assert list(raw_macro_df.columns) == original_cols


# ── select_features ────────────────────────────────────────────────────────

class TestSelectFeatures:
    def test_keeps_requested_columns(self, raw_macro_df):
        result = select_features(raw_macro_df, ["sp500", "gdp"])
        assert list(result.columns) == ["sp500", "gdp"]

    def test_keeps_market_code_if_present(self, raw_macro_df):
        df = raw_macro_df.copy()
        df["market_code"] = 0
        result = select_features(df, ["sp500"])
        assert "market_code" in result.columns

    def test_no_market_code_if_absent(self, raw_macro_df):
        result = select_features(raw_macro_df, ["sp500"])
        assert "market_code" not in result.columns

    def test_missing_cols_silently_skipped(self, raw_macro_df):
        result = select_features(raw_macro_df, ["sp500", "does_not_exist"])
        assert "sp500" in result.columns
        assert "does_not_exist" not in result.columns


# ── apply_gap_fill ─────────────────────────────────────────────────────────

class TestApplyGapFill:
    def test_interior_nans_filled(self, quarterly_index):
        vals = np.array([1.0, np.nan, np.nan, 4.0, 5.0,
                         6.0, 7.0, 8.0, 9.0, 10.0,
                         11.0, 12.0, 13.0, 14.0, 15.0,
                         16.0, 17.0, 18.0, 19.0, 20.0])
        df = pd.DataFrame({"x": vals}, index=quarterly_index)
        result = apply_gap_fill(df)
        assert result["x"].isna().sum() == 0

    def test_no_nans_unchanged(self, quarterly_index):
        vals = np.arange(20, dtype=float)
        df = pd.DataFrame({"x": vals}, index=quarterly_index)
        result = apply_gap_fill(df)
        pd.testing.assert_series_equal(result["x"], df["x"])

    def test_market_code_not_filled(self, quarterly_index):
        vals = np.array([1.0, np.nan, 3.0] + [4.0] * 17)
        df = pd.DataFrame(
            {"x": vals, "market_code": [0, np.nan, 1] + [0] * 17},
            index=quarterly_index,
        )
        result = apply_gap_fill(df)
        # market_code NaN at index 1 should NOT be touched by gap fill
        assert np.isnan(result["market_code"].iloc[1])

    def test_does_not_mutate_input(self, quarterly_index):
        vals = np.array([1.0, np.nan, 3.0] + [4.0] * 17)
        df = pd.DataFrame({"x": vals}, index=quarterly_index)
        original_vals = df["x"].copy()
        apply_gap_fill(df)
        pd.testing.assert_series_equal(df["x"], original_vals)


# ── apply_derivatives ──────────────────────────────────────────────────────

class TestApplyDerivatives:
    def test_three_derivative_columns_added(self, quarterly_index):
        df = pd.DataFrame(
            {"x": np.linspace(1, 20, 20)},
            index=quarterly_index,
        )
        result = apply_derivatives(df)
        assert "x_d1" in result.columns
        assert "x_d2" in result.columns
        assert "x_d3" in result.columns

    def test_market_code_has_no_derivatives(self, quarterly_index):
        df = pd.DataFrame(
            {"x": np.linspace(1, 20, 20), "market_code": np.zeros(20)},
            index=quarterly_index,
        )
        result = apply_derivatives(df)
        assert "market_code_d1" not in result.columns
        assert "market_code_d2" not in result.columns

    def test_linear_series_has_constant_d1(self, quarterly_index):
        """Derivative of a linear series should be roughly constant."""
        df = pd.DataFrame(
            {"x": np.linspace(0, 1, 20)},
            index=quarterly_index,
        )
        result = apply_derivatives(df, window=1)
        d1 = result["x_d1"].dropna()
        # All values should be nearly the same (constant slope)
        assert d1.std() < d1.mean() * 0.1
