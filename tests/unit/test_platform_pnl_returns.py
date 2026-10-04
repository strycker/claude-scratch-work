"""Month-end P&L returns (phase 08.3; DECISIONS E-07, E-08, E-10).

Strategy, ablation and baseline P&L move to month-end-to-month-end returns.
Features and L1/L2 inputs do not (D-01, D-09). Two things must hold:

* **The P&L-only raw columns never reach ``monthly_features``.** L2 fits on
  every ``monthly_features`` column, so a month-end close sitting there would
  be a new model input (the L2 leak guard in 08.3-CONTEXT). The drop lives in
  one place, ``transforms_monthly.features_from_raw``, shared by the build and
  the offline recompute. The leak tests below go red when that drop is removed
  (mutation proof, recorded in 08.3-01-SUMMARY).
* **Turning P&L on is one config block (``pnl_splice``) and changes nothing
  else.** Without the block every series is the feature series, byte for byte.

All data here is synthetic. No test computes a month-end KPI on real data:
the D-03 core-mix rule was declared before any "after" number exists.
"""

from __future__ import annotations

import copy
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from test_platform_point_in_time import (  # tests/unit is on sys.path (prepend mode)
    _build,
    _pit_cfg,
    _synthetic_macro,
    _synthetic_prices,
    _synthetic_vintages,
)

from trading_crab_lib.platform import splice, transforms_monthly
from trading_crab_lib.platform.backtest import driver
from trading_crab_lib.platform.config import load_platform_config
from trading_crab_lib.platform.honesty.registry import NO_REGISTRY
from trading_crab_lib.platform.ingestion import macro_monthly

#: The ``pnl_splice`` overlays 08.3-02 copies into the live config verbatim, and
#: the live block is pinned against. Each value overrides the keys of the same
#: class under ``splice:``; ``splice:`` itself is never edited (pitfall 2).
PNL_SPLICE_OVERLAYS: dict[str, dict] = {
    "equities": {"price_col": "sp500_close_me"},
}

LEAN_COLS = [
    "curve_10y3m", "curve_10y2y", "credit_spread_baa_aaa", "fred_vix", "gold", "oil",
    "trailing_return_1m", "trailing_return_3m", "realized_vol_1m", "realized_vol_3m",
    "cape_shiller", "div_yield", "real_rate_level",
]

SPLICE_CFG: dict = {
    "equities": {
        "research_name": "equities_tr", "method": "total_return_from_price_div",
        "price_col": "sp500", "div_yield_col": "div_yield", "tradable": "SPY",
    },
    "long_duration": {
        "research_name": "long_duration_tr", "method": "cmt_par_bond_repricing",
        "yield_col": "fred_gs10", "yield_units": "percent", "maturity_years": 10, "coupon_freq": 2,
        "tradable": "TLT",
    },
    "oil": {
        "research_name": "oil", "method": "single_source",
        "source_col": ["wti_fred", "wti_crude"], "cross_check_col": "wti_fred", "tradable": "USO",
    },
    "cash": {
        "research_name": "cash", "method": "yield_as_return",
        "yield_col": "fred_tb3ms", "yield_units": "percent", "tradable": "FZFXX",
    },
}


def _cfg(*, pnl_splice: dict | None = None) -> dict:
    """A small platform-shaped config: splice + the P&L-only flags + a fast backtest."""
    cfg: dict = {
        "splice": copy.deepcopy(SPLICE_CFG),
        "index_monthly": {"^GSPC": {"name": "sp500_close_me", "pnl_only": True}},
        "fred_monthly": {"series": {"GS10": {"name": "fred_gs10"}}},
        "taxonomy": {"fast": LEAN_COLS[:10], "slow": LEAN_COLS[10:], "agency": []},
        "labeling": {"K": 2, "lambda": 5.0, "n_restarts": 2, "embargo_months": 3},
        "allocation": {
            "target_vol_annual": 0.10, "ewma_halflife_months": 6, "portfolio_vol_min_obs": 3,
            "hysteresis": {"act_threshold": 0.70, "unwind_threshold": 0.40},
        },
        "backtest": {
            "cost_bps": 10, "min_train_months": 24, "feature_min_history": 12,
            "skip_l1l2_for_ablation": True, "sixty_forty_rebalance": "monthly",
            "apply_cost_to_baselines": True, "crisis_windows": [],
        },
    }
    if pnl_splice is not None:
        cfg["pnl_splice"] = copy.deepcopy(pnl_splice)
    return cfg


def _walk(rng: np.random.Generator, level: float, n: int, sigma: float = 0.03) -> np.ndarray:
    return level * np.exp(np.cumsum(rng.normal(0.0, sigma, n)))


def _raw(n: int = 48, start: str = "2010-01-31", seed: int = 7) -> pd.DataFrame:
    """A synthetic monthly_raw: every lean source, the research series, and the
    P&L-only month-end columns. Yields in percent, div_yield decimal (production units)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq="ME", name="date")
    sp500 = _walk(rng, 1000.0, n)
    raw = pd.DataFrame(
        {
            "fred_gs10": 4.0 + np.cumsum(rng.normal(0, 0.1, n)),
            "fred_tb3ms": 2.0 + np.cumsum(rng.normal(0, 0.05, n)).clip(-1.5, None),
            "fred_t10y2y": rng.normal(1.0, 0.4, n),
            "fred_baa": 6.0 + rng.normal(0, 0.2, n),
            "fred_aaa": 5.0 + rng.normal(0, 0.1, n),
            "fred_vix": 18.0 + rng.normal(0, 4, n),
            "fred_cpi": 100.0 * 1.002 ** np.arange(n),
            "gold": _walk(rng, 1200.0, n),
            "oil": _walk(rng, 60.0, n, 0.08),
            "wti_fred": _walk(rng, 60.0, n, 0.08),
            "cape_shiller": 25.0 + rng.normal(0, 1, n),
            "div_yield": rng.uniform(0.015, 0.03, n),
            "sp500": sp500,
            "equities_tr": sp500 / sp500[0],
            "sp500_close_me": sp500 * np.exp(rng.normal(0, 0.02, n)),
        },
        index=idx,
    )
    return raw


def _asset_returns(raw: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    from trading_crab_lib.platform.assets.returns import compute_monthly_returns, tradable_asset_returns

    return tradable_asset_returns(compute_monthly_returns(splice.build_core_research_series(raw, cfg)), cfg["splice"])


# ── 1. Leak guard: P&L-only columns never reach monthly_features ────────────


class TestFeaturesNeverCarryPnlColumns:
    def test_pnl_only_columns_reads_the_flags_and_the_overlay(self):
        cfg = _cfg(pnl_splice=PNL_SPLICE_OVERLAYS)
        assert splice.pnl_only_columns(cfg) == {"sp500_close_me"}
        # Config-driven: holds with no pnl_splice block at all (the live state after 08.3-01).
        assert splice.pnl_only_columns(_cfg()) == {"sp500_close_me"}
        # A config with neither block (e.g. the recompute tests' taxonomy-only cfg) has none.
        assert splice.pnl_only_columns({"taxonomy": {}}) == set()

    def test_an_overlay_source_is_pnl_only_unless_the_feature_splice_reads_it(self):
        cfg = _cfg(pnl_splice={"equities": {"price_col": "x_me"}, "oil": {"source_col": "wti_fred"}})
        del cfg["index_monthly"]
        # x_me is P&L-only; wti_fred is a feature-side splice source and must stay a feature.
        assert splice.pnl_only_columns(cfg) == {"x_me"}

    def test_flagging_a_feature_side_splice_source_raises(self):
        cfg = _cfg()
        cfg["fred_monthly"]["series"]["GS10"]["pnl_only"] = True
        with pytest.raises(ValueError, match="fred_gs10"):
            splice.pnl_only_columns(cfg)

    def test_features_from_raw_is_identical_with_and_without_a_pnl_column(self):
        cfg = _cfg()
        raw = _raw()
        with_me = transforms_monthly.features_from_raw(raw, cfg)
        without_me = transforms_monthly.features_from_raw(raw.drop(columns=["sp500_close_me"]), cfg)

        assert "sp500_close_me" not in with_me.columns
        assert not (splice.pnl_only_columns(cfg) & set(with_me.columns))
        pd.testing.assert_frame_equal(with_me, without_me, check_exact=True)

    def test_build_monthly_spine_writes_identical_dev_and_holdout_features(self, tmp_path):
        """The real build (network fetchers mocked), with and without the month-end
        columns in the macro frame, across the holdout cutoff."""
        cfg = _pit_cfg()
        cfg["data"]["start_date"], cfg["data"]["end_date"] = "2014-01-01", "2022-12-31"
        idx = pd.date_range("2014-01-31", "2022-12-31", freq="ME")
        rng = np.random.default_rng(83)
        macro = _synthetic_macro(cfg, rng, idx)
        prices = _synthetic_prices(cfg, rng, idx)
        vintages = _synthetic_vintages(rng, "2022-12-31")
        pnl_cols = sorted(splice.pnl_only_columns(cfg))
        assert pnl_cols and set(pnl_cols) <= set(macro.columns), "harness must feed the P&L columns"

        _build(cfg, macro, prices, vintages, tmp_path / "with")
        _build(cfg, macro.drop(columns=pnl_cols), prices, vintages, tmp_path / "without")

        for tree in ("platform", "holdout"):
            with_me = pd.read_parquet(tmp_path / "with" / tree / "monthly_features.parquet")
            without_me = pd.read_parquet(tmp_path / "without" / tree / "monthly_features.parquet")
            assert len(with_me) > 0, tree
            assert not (set(pnl_cols) & set(with_me.columns)), tree
            pd.testing.assert_frame_equal(with_me, without_me, check_exact=True)
        # monthly_raw keeps them: P&L reads them there.
        raw_with = pd.read_parquet(tmp_path / "with" / "platform" / "monthly_raw.parquet")
        assert set(pnl_cols) <= set(raw_with.columns)

    def test_the_l2_fit_never_receives_a_pnl_column(self):
        cfg = _cfg()
        raw = _raw()
        features = transforms_monthly.features_from_raw(raw, cfg)
        seen: list[list[str]] = []
        real_fit = driver.fit_l2_nowcaster

        def spy(train_features, train_states, cfg_):
            seen.append(list(train_features.columns))
            return real_fit(train_features, train_states, cfg_)

        with patch.object(driver, "fit_l2_nowcaster", spy):
            driver.run_backtest(features, _asset_returns(raw, cfg), cfg, registry_path=NO_REGISTRY)

        assert seen, "the L2 fit never ran — the leak check would be vacuous"
        leaked = {c for cols in seen for c in cols} & splice.pnl_only_columns(cfg)
        assert not leaked, f"P&L-only column(s) reached the L2 fit: {sorted(leaked)}"

    def test_recompute_delegates_to_the_shared_assembly(self):
        from recompute_monthly_features import rebuild_monthly_features

        cfg = _cfg()
        raw = _raw()
        pd.testing.assert_frame_equal(
            rebuild_monthly_features(raw, cfg), transforms_monthly.features_from_raw(raw, cfg), check_exact=True
        )


# ── 2. The P&L builder ──────────────────────────────────────────────────────


class TestBuildPnlResearchSeries:
    def test_without_a_block_it_is_the_feature_series_exactly(self):
        cfg = _cfg()
        raw = _raw()
        pd.testing.assert_frame_equal(
            splice.build_pnl_research_series(raw, cfg), splice.build_core_research_series(raw, cfg), check_exact=True
        )

    def test_equities_overlay_reads_the_month_end_close_and_nothing_else_moves(self):
        cfg = _cfg(pnl_splice=PNL_SPLICE_OVERLAYS)
        raw = _raw()
        pnl = splice.build_pnl_research_series(raw, cfg)
        core = splice.build_core_research_series(raw, cfg)

        expected = splice.build_equity_total_return(raw["sp500_close_me"], raw["div_yield"], cfg)
        pd.testing.assert_series_equal(pnl["equities_tr"], expected, check_exact=True)
        assert not np.allclose(pnl["equities_tr"].to_numpy(), core["equities_tr"].to_numpy())
        for col in ("long_duration_tr", "oil", "cash"):
            pd.testing.assert_series_equal(pnl[col], core[col], check_exact=True)

    def test_the_feature_splice_block_is_never_edited(self):
        cfg = _cfg(pnl_splice=PNL_SPLICE_OVERLAYS)
        before = copy.deepcopy(cfg)
        splice.build_pnl_research_series(_raw(), cfg)
        assert cfg == before

    def test_a_missing_month_end_column_raises_naming_it(self):
        cfg = _cfg(pnl_splice=PNL_SPLICE_OVERLAYS)
        with pytest.raises(ValueError, match="sp500_close_me"):
            splice.build_pnl_research_series(_raw().drop(columns=["sp500_close_me"]), cfg)

    def test_an_unknown_class_raises(self):
        cfg = _cfg(pnl_splice={"bitcoin": {"source_col": "btc_me"}})
        with pytest.raises(ValueError, match="bitcoin"):
            splice.build_pnl_research_series(_raw(), cfg)

    def test_a_chained_overlay_source_raises(self):
        """P&L sources are scalars: a chain could silently fall back to an average (pitfall 5)."""
        cfg = _cfg(pnl_splice={"equities": {"price_col": ["sp500_close_me", "sp500"]}})
        with pytest.raises(ValueError, match="scalar"):
            splice.build_pnl_research_series(_raw(), cfg)


# ── 3. The wealth line: run_backtest(pnl_returns=...) ───────────────────────


class TestRunBacktestPnlReturns:
    def test_pnl_returns_equal_to_asset_returns_is_byte_identical(self):
        cfg = _cfg()
        raw = _raw()
        features = transforms_monthly.features_from_raw(raw, cfg)
        assets = _asset_returns(raw, cfg)

        default_curve, default_metrics = driver.run_backtest(features, assets, cfg, registry_path=NO_REGISTRY)
        pnl_curve, pnl_metrics = driver.run_backtest(
            features, assets, cfg, registry_path=NO_REGISTRY, pnl_returns=assets.copy()
        )
        pd.testing.assert_frame_equal(pnl_curve, default_curve, check_exact=True)
        assert pnl_metrics["dates"] == default_metrics["dates"]

    def test_mismatched_columns_raise(self):
        cfg = _cfg()
        raw = _raw()
        assets = _asset_returns(raw, cfg)
        with pytest.raises(ValueError, match="columns"):
            driver.run_backtest(
                transforms_monthly.features_from_raw(raw, cfg), assets, cfg,
                registry_path=NO_REGISTRY, pnl_returns=assets.drop(columns=["USO"]),
            )

    def test_a_nan_on_a_decision_date_where_asset_returns_has_a_value_raises(self):
        cfg = _cfg()
        raw = _raw()
        assets = _asset_returns(raw, cfg)
        pnl = assets.copy()
        pnl.iloc[30, 0] = np.nan  # a decision date (min_train 24)
        with pytest.raises(ValueError, match="NaN"):
            driver.run_backtest(
                transforms_monthly.features_from_raw(raw, cfg), assets, cfg,
                registry_path=NO_REGISTRY, pnl_returns=pnl,
            )


# ── 4. Ingestion: the index month-end fetch ─────────────────────────────────


class TestIndexMonthEndFetch:
    def test_no_block_is_a_no_op(self):
        with patch.object(macro_monthly.prices_daily, "fetch_yfinance_month_end") as fetch:
            assert macro_monthly._fetch_index_month_end({"data": {"start_date": "2020-01-01"}}) == {}
        fetch.assert_not_called()

    def test_tickers_are_renamed_to_their_configured_names(self):
        idx = pd.date_range("2020-01-31", periods=3, freq="ME")
        monthly = pd.DataFrame({"^GSPC": [1.0, 2.0, 3.0]}, index=idx)
        cfg = {**_cfg(), "data": {"start_date": "2020-01-01", "end_date": "2020-03-31", "monthly_freq": "ME"}}
        with patch.object(macro_monthly.prices_daily, "fetch_yfinance_month_end", return_value=monthly) as fetch:
            out = macro_monthly._fetch_index_month_end(cfg)
        fetch.assert_called_once_with(["^GSPC"], "2020-01-01", "2020-03-31", "ME")
        assert list(out) == ["sp500_close_me"]
        assert out["sp500_close_me"].name == "sp500_close_me"
        assert out["sp500_close_me"].tolist() == [1.0, 2.0, 3.0]

    def test_a_failed_fetch_warns_and_the_column_is_absent(self, caplog):
        cfg = {**_cfg(), "data": {"start_date": "2020-01-01", "end_date": "2020-03-31"}}
        with patch.object(macro_monthly.prices_daily, "fetch_yfinance_month_end", return_value=pd.DataFrame()):
            with caplog.at_level("WARNING"):
                assert macro_monthly._fetch_index_month_end(cfg) == {}
        assert "sp500_close_me" in caplog.text

    def test_live_config_has_the_fetch_and_lag_entries_but_no_pnl_splice_block(self):
        cfg = load_platform_config()
        assert cfg["index_monthly"]["^GSPC"] == {"name": "sp500_close_me", "pnl_only": True}
        assert cfg["publication_lags"]["sp500_close_me"] == 0
        assert "^GSPC" not in str(cfg["universe"])
        assert "pnl_splice" not in cfg  # 08.3-02 writes it after the before run and the migration

