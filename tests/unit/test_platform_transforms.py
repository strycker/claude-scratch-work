"""Tests for trading_crab_lib.platform.transforms_monthly — monthly feature-table
assembly (DATA-01, DATA-03 runtime, DATA-04).

All ingestion is mocked (``macro_monthly.fetch_macro_monthly``,
``prices_daily.fetch_universe_prices``, ``alfred.fetch_all_vintages``) — no live
network calls. Checkpoint writes are redirected to a temp directory (autouse
fixture below) so no production checkpoint under ``data/checkpoints/`` is ever
touched by this test run.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from trading_crab_lib.platform import taxonomy
from trading_crab_lib.platform.checkpoints import get_platform_checkpoint_manager
from trading_crab_lib.platform.transforms_monthly import (
    build_monthly_spine,
    compute_lean_features,
    tag_feature_columns,
)

# ── Shared fixtures ──────────────────────────────────────────────────────────

# Mirrors config/platform_settings.yaml's `splice`/`taxonomy` blocks — kept
# inline (not loaded from disk) so this suite stays isolated from concurrent
# edits to platform_settings.yaml, matching test_platform_splice.py's convention.
SPLICE_CFG: dict = {
    "equities": {
        "research_name": "equities_tr",
        "method": "total_return_from_price_div",
        "price_col": "sp500",
        "div_yield_col": "div_yield",
        "tradable": "SPY",
    },
    "long_duration": {
        "research_name": "long_duration_tr",
        "method": "cmt_par_bond_repricing",
        "yield_col": "fred_gs10",
        "maturity_years": 10,
        "coupon_freq": 2,
        "tradable": "TLT",
    },
    "gold": {
        "research_name": "gold",
        "method": "single_source",
        "source_col": "gold_spot",
        "tradable": "IAU",
    },
    "oil": {
        "research_name": "oil",
        "method": "single_source",
        "source_col": "wti_crude",
        "tradable": "USO",
    },
    "cash": {
        "research_name": "cash",
        "method": "yield_as_return",
        "yield_col": "fred_tb3ms",
        "tradable": "FZFXX",
    },
}

TAXONOMY_CFG: dict = {
    "fast": [
        "curve_10y3m", "curve_10y2y", "credit_spread_baa_aaa", "fred_vix", "gold", "oil",
        "trailing_return_1m", "trailing_return_3m", "realized_vol_1m", "realized_vol_3m",
    ],
    "slow": ["cape_shiller", "div_yield", "buffett_indicator", "real_rate_level"],
    "agency": ["fred_gdp", "fred_cpi", "fred_unrate", "fred_indpro", "fred_payems"],
}


@pytest.fixture(autouse=True)
def _redirect_platform_checkpoints(tmp_path, monkeypatch):
    """Route all platform checkpoint I/O to a temp dir — production
    data/checkpoints/platform/ is never written to during this test run."""
    from trading_crab_lib.platform import checkpoints as platform_checkpoints

    monkeypatch.setattr(platform_checkpoints, "PLATFORM_CHECKPOINT_DIR", tmp_path / "platform")


def _make_monthly_index(start: str = "2015-01-31", periods: int = 72) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=periods, freq="ME")


def _make_synthetic_macro(idx: pd.DatetimeIndex) -> pd.DataFrame:
    n = len(idx)
    return pd.DataFrame(
        {
            "fred_gs10": np.linspace(0.02, 0.035, n),
            "fred_tb3ms": np.linspace(0.01, 0.018, n),
            "fred_baa": np.linspace(0.05, 0.06, n),
            "fred_aaa": np.linspace(0.04, 0.045, n),
            "fred_t10y2y": np.linspace(0.01, 0.015, n),
            "fred_vix": np.linspace(15.0, 22.0, n),
            "sp500": 100 * (1.01 ** np.arange(n)),
            "div_yield": [0.02] * n,
            "cape_shiller": [25.0] * n,
            "gold_spot": 1000.0 + np.arange(n, dtype=float),
            "wti_crude": 50.0 + np.arange(n, dtype=float) * 0.5,
        },
        index=idx,
    )


def _make_cfg(start: str = "2015-01-01", end: str = "2020-12-31") -> dict:
    return {
        "data": {"start_date": start, "end_date": end, "monthly_freq": "ME"},
        "fred_vintage": {
            "series": {
                "GDPC1": {"name": "fred_gdp"},
                "CPIAUCSL": {"name": "fred_cpi"},
            }
        },
        "splice": SPLICE_CFG,
        "taxonomy": TAXONOMY_CFG,
    }


def _make_vintages(idx: pd.DatetimeIndex, name: str = "fred_gdp", value: float = 1.0) -> dict:
    """A trivial single-vintage-per-period synthetic all-releases frame — every
    reference period known from the very first index date onward."""
    return {
        name: pd.DataFrame(
            {
                "realtime_start": pd.to_datetime([idx[0]] * len(idx)),
                "realtime_end": pd.to_datetime(["2099-12-31"] * len(idx)),
                "date": idx,
                "value": [value] * len(idx),
            }
        )
    }


# ── compute_lean_features (DATA-04) ─────────────────────────────────────────


class TestComputeLeanFeatures:
    def test_curve_10y3m_equals_gs10_minus_tb3ms(self):
        idx = _make_monthly_index(periods=12)
        macro = _make_synthetic_macro(idx)
        cfg = _make_cfg()

        lean = compute_lean_features(macro, cfg)

        expected = macro["fred_gs10"] - macro["fred_tb3ms"]
        pd.testing.assert_series_equal(lean["curve_10y3m"], expected, check_names=False)

    def test_credit_spread_baa_aaa_equals_baa_minus_aaa(self):
        idx = _make_monthly_index(periods=12)
        macro = _make_synthetic_macro(idx)
        cfg = _make_cfg()

        lean = compute_lean_features(macro, cfg)

        expected = macro["fred_baa"] - macro["fred_aaa"]
        pd.testing.assert_series_equal(lean["credit_spread_baa_aaa"], expected, check_names=False)

    def test_output_columns_are_superset_of_lean_feature_set_for_available_input(self):
        idx = _make_monthly_index(periods=12)
        macro = _make_synthetic_macro(idx)
        macro["equities_tr"] = 1.01 ** np.arange(len(idx))  # normally added by splice
        cfg = _make_cfg()

        lean = compute_lean_features(macro, cfg)

        lean_set = taxonomy.lean_feature_set(cfg)
        # Every column compute_lean_features produces is taxonomy-listed.
        assert set(lean.columns) <= lean_set
        # Every column derivable from the synthetic input was actually produced.
        expected_produced = {
            "curve_10y3m", "curve_10y2y", "credit_spread_baa_aaa", "fred_vix",
            "trailing_return_1m", "trailing_return_3m", "realized_vol_1m", "realized_vol_3m",
            "cape_shiller", "div_yield",
        }
        assert expected_produced <= set(lean.columns)

    def test_missing_source_columns_are_skipped_not_crashed(self):
        idx = _make_monthly_index(periods=6)
        macro = pd.DataFrame({"fred_gs10": [0.02] * 6}, index=idx)  # sparse input
        cfg = _make_cfg()

        lean = compute_lean_features(macro, cfg)

        assert "curve_10y3m" not in lean.columns  # fred_tb3ms missing — skipped, not an error
        assert isinstance(lean, pd.DataFrame)


# ── tag_feature_columns / check_columns_tagged (DATA-04) ────────────────────


class TestTagFeatureColumns:
    def test_every_lean_column_is_tagged(self):
        idx = _make_monthly_index(periods=12)
        macro = _make_synthetic_macro(idx)
        macro["equities_tr"] = 1.01 ** np.arange(len(idx))
        cfg = _make_cfg()

        lean = compute_lean_features(macro, cfg)
        untagged = taxonomy.check_columns_tagged(list(lean.columns), cfg)

        assert untagged == []

    def test_tag_feature_columns_maps_each_column_to_its_tier(self):
        idx = _make_monthly_index(periods=12)
        macro = _make_synthetic_macro(idx)
        macro["equities_tr"] = 1.01 ** np.arange(len(idx))
        cfg = _make_cfg()

        lean = compute_lean_features(macro, cfg)
        tags = tag_feature_columns(lean, cfg)

        assert tags["curve_10y3m"] == "fast"
        assert tags["cape_shiller"] == "slow"


# ── build_monthly_spine — lean features wired end-to-end (DATA-01/04) ───────


class TestBuildMonthlySpineLeanFeatures:
    @patch("trading_crab_lib.platform.ingestion.alfred.fetch_all_vintages")
    @patch("trading_crab_lib.platform.ingestion.prices_daily.fetch_universe_prices")
    @patch("trading_crab_lib.platform.ingestion.macro_monthly.fetch_macro_monthly")
    def test_persists_monthly_features_checkpoint(self, mock_macro, mock_prices, mock_vintages):
        idx = _make_monthly_index(start="2015-01-31", periods=24)
        macro = _make_synthetic_macro(idx)
        mock_macro.return_value = macro
        mock_prices.return_value = (pd.DataFrame(), pd.DataFrame())
        mock_vintages.return_value = _make_vintages(idx)

        cfg = _make_cfg(start="2015-01-01", end="2016-12-31")
        result = build_monthly_spine(cfg)

        cm = get_platform_checkpoint_manager()
        loaded = cm.load("monthly_features")
        assert not loaded.empty
        assert "curve_10y3m" in result.columns
        assert "curve_10y3m" in loaded.columns
        # Passthrough lean columns (e.g. gold/oil/fred_vix) are not duplicated.
        assert list(result.columns).count("fred_vix") == 1
