"""Tests for feature_engineering module — derivatives and interpolation."""

import numpy as np
import pandas as pd
import pytest

from trading_crab.feature_engineering import (
    compute_smoothed_derivatives,
    interpolate_column,
)
from datetime import date


class TestComputeSmoothedDerivatives:
    """Test the smoothed derivative computation."""

    def _make_linear_series(self, n=20, slope=2.0):
        """Create a linear time series with known derivative = slope."""
        dates = [date(2020, 1, 1 + i) for i in range(n)]
        values = pd.Series([slope * i for i in range(n)], index=dates)
        return values, dates

    def test_returns_three_series(self):
        """Should return d1, d2, d3 as pandas Series."""
        values, dates = self._make_linear_series()
        d1, d2, d3 = compute_smoothed_derivatives(values, dates, window=3)
        assert isinstance(d1, pd.Series)
        assert isinstance(d2, pd.Series)
        assert isinstance(d3, pd.Series)
        assert len(d1) == len(values)

    def test_linear_has_constant_d1(self):
        """A linear series should have roughly constant 1st derivative."""
        values, dates = self._make_linear_series(n=30, slope=1.0)
        d1, d2, d3 = compute_smoothed_derivatives(values, dates, window=3)
        # Middle values (away from edges) should be close to constant
        mid = d1.values[5:-5]
        assert np.std(mid) < 0.1 * np.mean(np.abs(mid)), (
            f"d1 should be nearly constant for linear input, got std={np.std(mid):.4f}"
        )

    def test_linear_has_near_zero_d2(self):
        """A linear series should have near-zero 2nd derivative."""
        values, dates = self._make_linear_series(n=30, slope=1.0)
        d1, d2, d3 = compute_smoothed_derivatives(values, dates, window=3)
        mid = d2.values[5:-5]
        assert np.max(np.abs(mid)) < 0.5, (
            f"d2 should be ~0 for linear input, got max={np.max(np.abs(mid)):.4f}"
        )

    def test_custom_window(self):
        """Different window sizes should still return valid results."""
        values, dates = self._make_linear_series()
        for w in [3, 5, 7]:
            d1, d2, d3 = compute_smoothed_derivatives(values, dates, window=w)
            assert len(d1) == len(values)
            assert not d1.isna().all()


class TestInterpolateColumn:
    """Test Bernstein polynomial gap interpolation."""

    def _make_gapped_df(self):
        """Create a simple DataFrame with a known gap in the middle."""
        dates = pd.date_range("2020-01-01", periods=20, freq="QE")
        date_strs = dates.strftime("%Y-%m-%d")

        values = np.sin(np.linspace(0, 2 * np.pi, 20)) * 10
        values_with_gap = values.copy()
        values_with_gap[8:12] = np.nan  # gap in the middle

        df = pd.DataFrame({
            "test_col": values_with_gap,
            "market_code": [0] * 20,
        }, index=date_strs)

        return df, values

    def test_fills_interior_gap(self):
        """Interior NaN gaps should be filled."""
        df, original = self._make_gapped_df()
        assert df["test_col"].isna().any()  # gap exists

        filled = interpolate_column(df, "test_col")
        assert not filled.isna().any(), "All NaNs should be filled"

    def test_filled_values_are_reasonable(self):
        """Filled values should be in the ballpark of the original."""
        df, original = self._make_gapped_df()
        filled = interpolate_column(df, "test_col")

        # Check that filled values are within 5x the range of original
        data_range = np.ptp(original)
        for i in range(8, 12):
            assert abs(filled.iloc[i]) < data_range * 3, (
                f"Filled value at {i} = {filled.iloc[i]:.2f} seems unreasonable "
                f"(data range = {data_range:.2f})"
            )

    def test_no_gap_returns_unchanged(self):
        """If there are no NaNs, the column should be returned unchanged."""
        dates = pd.date_range("2020-01-01", periods=10, freq="QE")
        date_strs = dates.strftime("%Y-%m-%d")
        df = pd.DataFrame({
            "test_col": range(10),
            "market_code": [0] * 10,
        }, index=date_strs)

        filled = interpolate_column(df, "test_col")
        np.testing.assert_array_equal(filled.values, df["test_col"].values)
