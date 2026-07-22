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
    "slow": ["cape_shiller", "div_yield", "real_rate_level"],
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


# ── real_rate_level (DATA-04 gap closure — 01-VERIFICATION.md criterion 4) ──


class TestComputeLeanFeaturesRealRateLevel:
    def test_real_rate_level_equals_gs10_minus_cpi_yoy(self):
        """real_rate_level = nominal 10Y yield minus trailing-12M CPI YoY
        inflation (percentage points). GS10=5.0 constant; CPI up exactly 3%
        year-over-year (flat within each 12-month block) → YoY=3.0 →
        real_rate_level=2.0 for every month once 12 trailing months exist."""
        idx = _make_monthly_index(periods=24)
        cpi = pd.Series([100.0] * 12 + [103.0] * 12, index=idx)
        macro = pd.DataFrame({"fred_gs10": [5.0] * len(idx), "fred_cpi": cpi}, index=idx)
        cfg = _make_cfg()

        lean = compute_lean_features(macro, cfg)

        expected = pd.Series([float("nan")] * 12 + [2.0] * 12, index=idx)
        pd.testing.assert_series_equal(lean["real_rate_level"], expected, check_names=False)

    def test_missing_fred_cpi_skips_real_rate_level_not_crashed(self):
        idx = _make_monthly_index(periods=6)
        macro = pd.DataFrame({"fred_gs10": [5.0] * 6}, index=idx)  # no fred_cpi
        cfg = _make_cfg()

        lean = compute_lean_features(macro, cfg)

        assert "real_rate_level" not in lean.columns
        assert isinstance(lean, pd.DataFrame)


# ── lean_feature_set() invariant (01-VERIFICATION.md gap closure) ───────────
#
# The verifier found that taxonomy.lean_feature_set() could declare column
# names compute_lean_features() never produces (buffett_indicator,
# real_rate_level), which would KeyError any downstream code indexing
# monthly_features by lean_feature_set()'s output. Direct regression test:
# every declared name must be an actual computed column when all required
# source data is present.


def _make_fully_populated_macro(idx: pd.DatetimeIndex) -> pd.DataFrame:
    """`_make_synthetic_macro` plus splice-output columns (`equities_tr`,
    `gold`, `oil`) and the agency `fred_cpi` column — enough source data for
    every fast+slow lean feature to be computable."""
    macro = _make_synthetic_macro(idx)
    macro["equities_tr"] = 1.01 ** np.arange(len(idx))
    macro["gold"] = macro["gold_spot"]
    macro["oil"] = macro["wti_crude"]
    macro["fred_cpi"] = 100.0 * (1.03 ** (np.arange(len(idx)) / 12))
    return macro


class TestLeanFeatureSetInvariant:
    def test_every_lean_feature_set_name_is_a_computed_column(self):
        idx = _make_monthly_index(periods=24)
        macro = _make_fully_populated_macro(idx)
        cfg = _make_cfg()

        lean = compute_lean_features(macro, cfg)

        assert taxonomy.lean_feature_set(cfg) <= set(lean.columns)

    def test_real_platform_config_lean_feature_set_is_fully_computable(self):
        """Same invariant against the real config/platform_settings.yaml —
        guards against future taxonomy.slow additions that outrun
        compute_lean_features()."""
        from trading_crab_lib.platform.config import load_platform_config

        idx = _make_monthly_index(periods=24)
        macro = _make_fully_populated_macro(idx)
        cfg = load_platform_config()

        lean = compute_lean_features(macro, cfg)

        assert taxonomy.lean_feature_set(cfg) <= set(lean.columns)


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


# ── build_monthly_spine — end-to-end cadence, alignment, coverage, NULL-tolerance ──


def _make_gdp_vintages_with_look_ahead_gap() -> dict:
    """A revision-free 2-quarter agency series exercising the look-ahead guard:
    Q4-2017 known from 2018-01-30 (value=100.0); Q1-2018 known only from
    2018-04-27 (value=200.0) — the Q1 value must be invisible before its own
    realtime_start."""
    return {
        "fred_gdp": pd.DataFrame(
            {
                "realtime_start": pd.to_datetime(["2018-01-30", "2018-04-27"]),
                "realtime_end": pd.to_datetime(["2099-12-31", "2099-12-31"]),
                "date": pd.to_datetime(["2017-10-01", "2018-01-01"]),
                "value": [100.0, 200.0],
            }
        )
    }


def _make_cpi_vintages_wide_history_late_cutover() -> dict:
    """A constant-valued series whose entire ALFRED vintage archive was
    bulk-recorded on one late date (2018-06-15) despite reference periods
    reaching back to 2010 — models the real-world characteristic that ALFRED
    vintage coverage starts well after the underlying economic history
    (RESEARCH Pitfall 4 / A2). Constant value makes the fallback-vs-vintage
    result trivially distinguishable regardless of resample/ffill mechanics.
    """
    dates = pd.date_range("2010-01-01", "2020-12-01", freq="MS")
    return {
        "fred_cpi": pd.DataFrame(
            {
                "realtime_start": pd.to_datetime(["2018-06-15"] * len(dates)),
                "realtime_end": pd.to_datetime(["2099-12-31"] * len(dates)),
                "date": dates,
                "value": [42.0] * len(dates),
            }
        )
    }


class TestBuildMonthlySpineEndToEnd:
    @patch("trading_crab_lib.platform.ingestion.alfred.fetch_all_vintages")
    @patch("trading_crab_lib.platform.ingestion.prices_daily.fetch_universe_prices")
    @patch("trading_crab_lib.platform.ingestion.macro_monthly.fetch_macro_monthly")
    def test_monthly_row_count(self, mock_macro, mock_prices, mock_vintages):
        """The assembled monthly frame has ~12 rows/year over its span —
        monthly, not quarterly (DATA-01)."""
        idx = _make_monthly_index(start="2015-01-31", periods=72)  # 6 years
        mock_macro.return_value = _make_synthetic_macro(idx)
        mock_prices.return_value = (pd.DataFrame(), pd.DataFrame())
        mock_vintages.return_value = {}

        cfg = _make_cfg(start="2015-01-01", end="2020-12-31")
        result = build_monthly_spine(cfg)

        quarterly_equivalent = len(pd.date_range("2015-01-01", "2020-12-31", freq="QE"))
        assert len(result) == 72
        assert len(result) > quarterly_equivalent * 2

    @patch("trading_crab_lib.platform.ingestion.alfred.fetch_all_vintages")
    @patch("trading_crab_lib.platform.ingestion.prices_daily.fetch_universe_prices")
    @patch("trading_crab_lib.platform.ingestion.macro_monthly.fetch_macro_monthly")
    def test_quarterly_series_alignment(self, mock_macro, mock_prices, mock_vintages):
        """A synthetic agency series whose vintage becomes known mid-quarter
        does NOT appear in the monthly frame in a month before its
        realtime_start (no revision/timing look-ahead — DATA-01/DATA-03)."""
        idx = _make_monthly_index(start="2017-01-31", periods=24)  # Jan2017..Dec2018
        mock_macro.return_value = _make_synthetic_macro(idx)
        mock_prices.return_value = (pd.DataFrame(), pd.DataFrame())
        mock_vintages.return_value = _make_gdp_vintages_with_look_ahead_gap()

        cfg = _make_cfg(start="2017-01-01", end="2018-12-31")
        result = build_monthly_spine(cfg)

        # March 2018: Q1-2018 GDP (200.0) not yet published (realtime_start
        # 2018-04-27) — only the older Q4-2017 value (100.0) is visible.
        assert result.loc[pd.Timestamp("2018-03-31"), "fred_gdp"] == pytest.approx(100.0)
        # April 2018 (>= 2018-04-27): the newly-published Q1-2018 value appears.
        assert result.loc[pd.Timestamp("2018-04-30"), "fred_gdp"] == pytest.approx(200.0)

    @patch("trading_crab_lib.platform.ingestion.alfred.fetch_all_vintages")
    @patch("trading_crab_lib.platform.ingestion.prices_daily.fetch_universe_prices")
    @patch("trading_crab_lib.platform.ingestion.macro_monthly.fetch_macro_monthly")
    def test_pre_vintage_fallback_applied(self, mock_macro, mock_prices, mock_vintages):
        """Pre-vintage-era months use the publication-lag shift fallback value
        — never NaN, never an error (D-06/DATA-03)."""
        idx = _make_monthly_index(start="2015-01-31", periods=72)  # 2015-01..2020-12
        mock_macro.return_value = _make_synthetic_macro(idx)
        mock_prices.return_value = (pd.DataFrame(), pd.DataFrame())
        mock_vintages.return_value = _make_cpi_vintages_wide_history_late_cutover()

        cfg = _make_cfg(start="2015-01-01", end="2020-12-31")
        result = build_monthly_spine(cfg)

        # 2015-06 predates the earliest recorded vintage (2018-06-15).
        val = result.loc[pd.Timestamp("2015-06-30"), "fred_cpi"]
        assert not pd.isna(val)
        assert val == pytest.approx(42.0)

    @patch("trading_crab_lib.platform.ingestion.alfred.fetch_all_vintages")
    @patch("trading_crab_lib.platform.ingestion.prices_daily.fetch_universe_prices")
    @patch("trading_crab_lib.platform.ingestion.macro_monthly.fetch_macro_monthly")
    def test_every_feature_tagged(self, mock_macro, mock_prices, mock_vintages):
        """check_columns_tagged on the produced lean feature columns of the
        persisted monthly_raw checkpoint returns empty (DATA-04)."""
        idx = _make_monthly_index(start="2015-01-31", periods=24)
        mock_macro.return_value = _make_synthetic_macro(idx)
        mock_prices.return_value = (pd.DataFrame(), pd.DataFrame())
        mock_vintages.return_value = {}

        cfg = _make_cfg(start="2015-01-01", end="2016-12-31")
        build_monthly_spine(cfg)

        cm = get_platform_checkpoint_manager()
        monthly_raw = cm.load("monthly_raw")
        lean = compute_lean_features(monthly_raw, cfg)

        assert taxonomy.check_columns_tagged(list(lean.columns), cfg) == []

    @patch("trading_crab_lib.platform.ingestion.alfred.fetch_all_vintages")
    @patch("trading_crab_lib.platform.ingestion.prices_daily.fetch_universe_prices")
    @patch("trading_crab_lib.platform.ingestion.macro_monthly.fetch_macro_monthly")
    def test_short_history_satellite_null_tolerant(self, mock_macro, mock_prices, mock_vintages):
        """A short-history price column stays NaN-padded in the merged frame
        — never dropped rows, never a crash (DATA-05, RESEARCH Pitfall 5)."""
        idx = _make_monthly_index(start="2015-01-31", periods=24)
        mock_macro.return_value = _make_synthetic_macro(idx)

        short_idx = idx[-6:]  # only the last 6 months have data
        monthly_prices = pd.DataFrame({"IAU": np.arange(6.0)}, index=short_idx)
        mock_prices.return_value = (pd.DataFrame(), monthly_prices)
        mock_vintages.return_value = {}

        cfg = _make_cfg(start="2015-01-01", end="2016-12-31")
        result = build_monthly_spine(cfg)

        assert len(result) == 24  # no dropped rows
        assert "IAU" in result.columns
        assert pd.isna(result.loc[idx[0], "IAU"])  # pre-inception: NaN, not dropped
        assert not pd.isna(result.loc[idx[-1], "IAU"])
