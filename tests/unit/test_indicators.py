"""Tests for composite economic indicators (Phase D3)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_crab_lib.indicators import compute_lei_proxy, DEFAULT_LEI_COMPONENTS


@pytest.fixture
def fred_df():
    """20-quarter DataFrame with typical FRED columns."""
    idx = pd.date_range("2015-03-31", periods=20, freq="QE")
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "fred_unrate": 4.0 + rng.normal(0, 0.5, 20),
            "fred_t10y2y": 1.0 + rng.normal(0, 0.3, 20),
            "fred_m2sl": np.linspace(12000, 15000, 20) + rng.normal(0, 100, 20),
            "fred_indpro": np.linspace(100, 110, 20) + rng.normal(0, 1, 20),
            "fred_payems": np.linspace(140000, 150000, 20) + rng.normal(0, 100, 20),
        },
        index=idx,
    )


class TestComputeLeiProxy:
    def test_returns_series_named_lei_proxy(self, fred_df):
        result = compute_lei_proxy(fred_df)
        assert isinstance(result, pd.Series)
        assert result.name == "lei_proxy"
        assert len(result) == len(fred_df)

    def test_has_valid_values(self, fred_df):
        result = compute_lei_proxy(fred_df)
        # First few rows may be NaN due to diff/pct_change + rolling
        assert result.notna().sum() > 10

    def test_all_components_used(self, fred_df):
        result = compute_lei_proxy(fred_df)
        # Should produce non-NaN values when all 5 components present
        assert result.notna().any()

    def test_missing_columns_handled(self):
        idx = pd.date_range("2015-03-31", periods=20, freq="QE")
        df = pd.DataFrame(
            {"fred_unrate": np.linspace(4, 5, 20)},
            index=idx,
        )
        result = compute_lei_proxy(df)
        assert result.name == "lei_proxy"
        # Should still work with just 1 component
        assert result.notna().sum() > 0

    def test_no_components_returns_nan(self):
        idx = pd.date_range("2015-03-31", periods=10, freq="QE")
        df = pd.DataFrame({"unrelated_col": range(10)}, index=idx)
        result = compute_lei_proxy(df)
        assert result.isna().all()

    def test_custom_components(self, fred_df):
        custom = {"fred_unrate": -1.0, "fred_indpro": 1.0}
        result = compute_lei_proxy(fred_df, components=custom)
        assert result.notna().any()

    def test_window_parameter(self, fred_df):
        r4 = compute_lei_proxy(fred_df, window=4)
        r8 = compute_lei_proxy(fred_df, window=8)
        # Different windows should produce different values
        assert not r4.equals(r8)

    def test_unemployment_sign_is_negative(self):
        """Unemployment component should have negative sign in the default config."""
        assert DEFAULT_LEI_COMPONENTS["fred_unrate"] == -1.0

    def test_default_components_dict(self):
        assert "fred_unrate" in DEFAULT_LEI_COMPONENTS
        assert DEFAULT_LEI_COMPONENTS["fred_unrate"] == -1.0
        assert DEFAULT_LEI_COMPONENTS["fred_indpro"] == 1.0
        assert len(DEFAULT_LEI_COMPONENTS) == 5
