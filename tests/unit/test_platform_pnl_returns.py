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

import ast
import copy
from pathlib import Path
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
from tracked_record import tracked_record_cfg

from trading_crab_lib.platform import splice, transforms_monthly
from trading_crab_lib.platform.assets.returns import compute_monthly_returns, tradable_asset_returns
from trading_crab_lib.platform.backtest import driver
from trading_crab_lib.platform.config import load_platform_config
from trading_crab_lib.platform.honesty.registry import NO_REGISTRY
from trading_crab_lib.platform.ingestion import macro_monthly

#: The ``pnl_splice`` overlays 08.3-02 copies into the live config verbatim, and
#: the live block is pinned against. Each value overrides the keys of the same
#: class under ``splice:``; ``splice:`` itself is never edited (pitfall 2).
PNL_SPLICE_OVERLAYS: dict[str, dict] = {
    # ^GSPC month-end close + the existing div-yield accrual (div_yield still lagged 3).
    "equities": {"price_col": "sp500_close_me"},
    # DGS10 month-end yield through the existing CMT par-bond repricing (yield_units percent, inherited).
    "long_duration": {"yield_col": "dgs10_me"},
    # D-02: WTISPLC monthly average through 1986-01, DCOILWTICO month-end close from 1986-02.
    # ratio_splice splices RETURNS (avg/avg, then close/close), never a mixed close/avg month.
    "oil": {"method": "ratio_splice", "old_col": "wti_fred", "new_col": "wti_me", "join_date": "1986-01-31"},
}
#: 2026-10-09: gold's P&L overlay, added when gold moved to the World Bank monthly average.
GOLD_PNL_OVERLAY = {"method": "ratio_splice", "old_col": "gold_wb", "new_col": "IAU", "join_date": "2005-01-31"}
EQUITIES_ONLY = {"equities": PNL_SPLICE_OVERLAYS["equities"]}
OIL_JOIN = pd.Timestamp(PNL_SPLICE_OVERLAYS["oil"]["join_date"])

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
    return tradable_asset_returns(compute_monthly_returns(splice.build_core_research_series(raw, cfg)), cfg["splice"])


# ── 1. Leak guard: P&L-only columns never reach monthly_features ────────────


class TestFeaturesNeverCarryPnlColumns:
    def test_pnl_only_columns_reads_the_flags_and_the_overlay(self):
        cfg = _cfg(pnl_splice=PNL_SPLICE_OVERLAYS)
        # wti_fred is the oil overlay's old leg but a feature-side source: it stays a feature.
        assert splice.pnl_only_columns(cfg) == {"sp500_close_me", "dgs10_me", "wti_me"}
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
        cfg = _cfg(pnl_splice=EQUITIES_ONLY)
        raw = _raw()
        pnl = splice.build_pnl_research_series(raw, cfg)
        core = splice.build_core_research_series(raw, cfg)

        expected = splice.build_equity_total_return(raw["sp500_close_me"], raw["div_yield"], cfg)
        pd.testing.assert_series_equal(pnl["equities_tr"], expected, check_exact=True)
        assert not np.allclose(pnl["equities_tr"].to_numpy(), core["equities_tr"].to_numpy())
        for col in ("long_duration_tr", "oil", "cash"):
            pd.testing.assert_series_equal(pnl[col], core[col], check_exact=True)

    def test_the_feature_splice_block_is_never_edited(self):
        cfg = _cfg(pnl_splice=EQUITIES_ONLY)
        before = copy.deepcopy(cfg)
        splice.build_pnl_research_series(_raw(), cfg)
        assert cfg == before

    def test_a_missing_month_end_column_raises_naming_it(self):
        cfg = _cfg(pnl_splice=EQUITIES_ONLY)
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


# ── 2b. All three overlays: long duration and oil (D-02) ────────────────────


def _raw_1980(seed: int = 11) -> pd.DataFrame:
    """1980-1995 raw spanning the oil join: wti_fred is a monthly AVERAGE level,
    wti_me a month-end CLOSE that only starts at the join month (as DCOILWTICO does)
    and differs from the average there."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("1980-01-31", "1995-12-31", freq="ME", name="date")
    n = len(idx)
    raw = _raw(n=n, start="1980-01-31", seed=seed)
    raw["dgs10_me"] = (raw["fred_gs10"] + rng.normal(0, 0.15, n)).to_numpy()
    wti_me = raw["wti_fred"] * np.exp(rng.normal(0, 0.05, n))
    raw["wti_me"] = wti_me.where(idx >= OIL_JOIN)
    return raw


class TestAllOverlays:
    def test_each_overlaid_class_reads_its_month_end_source_and_the_rest_are_unchanged(self):
        cfg = _cfg(pnl_splice=PNL_SPLICE_OVERLAYS)
        raw = _raw_1980()
        pnl = splice.build_pnl_research_series(raw, cfg)
        core = splice.build_core_research_series(raw, cfg)

        pd.testing.assert_series_equal(
            pnl["equities_tr"], splice.build_equity_total_return(raw["sp500_close_me"], raw["div_yield"], cfg),
            check_exact=True,
        )
        pd.testing.assert_series_equal(
            pnl["long_duration_tr"], splice.build_treasury_tr_synthetic(raw["dgs10_me"], cfg), check_exact=True
        )
        pd.testing.assert_series_equal(pnl["cash"], core["cash"], check_exact=True)
        for col in ("equities_tr", "long_duration_tr", "oil"):
            assert not pnl[col].dropna().equals(core[col].dropna()), col

    def test_a_percent_yield_misread_as_decimal_still_trips_the_units_guard(self):
        """The overlay inherits splice.long_duration's yield_units; DGS10 is percent."""
        cfg = _cfg(pnl_splice={"long_duration": {"yield_col": "dgs10_me", "yield_units": "decimal"}})
        with pytest.raises(ValueError, match="yield_units"):
            splice.build_pnl_research_series(_raw_1980(), cfg)


class TestOilSplice:
    """V6: avg/avg returns through the join month, close/close after, no mixed month, no gap."""

    @pytest.fixture
    def oil(self):
        cfg = _cfg(pnl_splice=PNL_SPLICE_OVERLAYS)
        raw = _raw_1980()
        returns = compute_monthly_returns(splice.build_pnl_research_series(raw, cfg))["oil"]
        return raw, returns

    def test_returns_through_the_join_month_are_average_over_average(self, oil):
        raw, returns = oil
        expected = raw["wti_fred"].pct_change(fill_method=None)
        before = returns.index <= OIL_JOIN
        np.testing.assert_allclose(returns[before].iloc[1:], expected[before].iloc[1:], rtol=1e-9, atol=0)

    def test_returns_after_the_join_month_are_close_over_close(self, oil):
        raw, returns = oil
        expected = raw["wti_me"].pct_change(fill_method=None)
        after = returns.index > OIL_JOIN
        assert after.sum() > 100
        np.testing.assert_allclose(returns[after], expected[after], rtol=1e-9, atol=0)

    def test_the_mixed_close_over_average_return_occurs_nowhere(self, oil):
        raw, returns = oil
        mixed = raw.loc[OIL_JOIN, "wti_me"] / raw.loc[OIL_JOIN - pd.offsets.MonthEnd(1), "wti_fred"] - 1.0
        assert not np.isclose(returns.dropna().to_numpy(), mixed, rtol=1e-9, atol=0).any()

    def test_no_nan_between_the_first_old_month_and_the_last_new_month(self, oil):
        raw, returns = oil
        span = returns.loc[raw["wti_fred"].first_valid_index():raw["wti_me"].last_valid_index()]
        assert span.iloc[1:].notna().all()  # the first month has no prior level, by definition


# ── 2c. SC1: the averaged series fails the no-leak check the month-end one passes ──


def _lead_corr(pnl_returns: pd.Series, known_close_returns: pd.Series) -> float:
    """corr(pnl_{t+1}, close_t): how much of next month's P&L return was already
    visible in the month-end move at t. The E-07 leak, as one number."""
    frame = pd.concat([pnl_returns.shift(-1), known_close_returns], axis=1).dropna()
    return float(frame.iloc[:, 0].corr(frame.iloc[:, 1]))


class TestAveragedSeriesIsCaught:
    N_MONTHS = 600

    @pytest.fixture(scope="class")
    def world(self):
        rng = np.random.default_rng(2026)
        days = pd.bdate_range("1960-01-01", periods=self.N_MONTHS * 22)
        daily = pd.Series(1000.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, len(days)))), index=days)
        closes, averages = daily.resample("ME").last(), daily.resample("ME").mean()
        closes, averages = closes.iloc[: self.N_MONTHS], averages.iloc[: self.N_MONTHS]
        n = len(closes)
        raw = _raw(n=n, start=str(closes.index[0].date()), seed=5)
        raw.index = closes.index.rename("date")
        raw["sp500_close_me"], raw["sp500_avg"] = closes.to_numpy(), averages.to_numpy()
        raw["div_yield"] = 0.02
        return raw

    def _pnl_equity_returns(self, raw: pd.DataFrame, price_col: str) -> pd.Series:
        cfg = _cfg(pnl_splice={"equities": {"price_col": price_col}})
        return compute_monthly_returns(splice.build_pnl_research_series(raw, cfg))["equities_tr"]

    def test_month_end_pnl_passes(self, world):
        known = world["sp500_close_me"].pct_change(fill_method=None)
        assert abs(_lead_corr(self._pnl_equity_returns(world, "sp500_close_me"), known)) < 0.15

    def test_the_same_builder_on_averages_fails(self, world):
        known = world["sp500_close_me"].pct_change(fill_method=None)
        assert _lead_corr(self._pnl_equity_returns(world, "sp500_avg"), known) > 0.3


# ── 2d. pandas 2/3 independence ─────────────────────────────────────────────


class TestPandasIndependence:
    def test_an_interior_gap_is_nan_never_forward_filled(self):
        """pandas 2's default pct_change pads the gap (a 0 return, then a 2-month one);
        fill_method=None keeps both NaN. Green on pandas 2 and 3."""
        cfg = _cfg(pnl_splice=EQUITIES_ONLY)
        raw = _raw()
        gap = raw.index[20]
        raw.loc[gap, "sp500_close_me"] = np.nan
        returns = compute_monthly_returns(splice.build_pnl_research_series(raw, cfg))["equities_tr"]
        after = raw.index[21]
        assert np.isnan(returns[gap]) and np.isnan(returns[after])
        assert returns.drop([gap, after]).iloc[1:].notna().all()

    def test_every_platform_pct_change_passes_fill_method(self):
        root = Path(splice.__file__).resolve().parent
        bad = [
            f"{path.relative_to(root)}:{node.lineno}"
            for path in sorted(root.rglob("*.py"))
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
            if isinstance(node, ast.Call)
            and getattr(node.func, "attr", "") == "pct_change"
            and not any(k.arg == "fill_method" for k in node.keywords)
        ]
        assert root.name == "platform" and bad == [], bad


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


# ── 3b. Measurement only (V3): P&L moves the return column and nothing else ─


DECISION_COLUMNS = ["turnover", "scale", "active_regime", "degraded"]


def _assert_measurement_only(pnl_curve: pd.DataFrame, default_curve: pd.DataFrame) -> None:
    pd.testing.assert_index_equal(pnl_curve.index, default_curve.index)
    pd.testing.assert_frame_equal(pnl_curve[DECISION_COLUMNS], default_curve[DECISION_COLUMNS], check_exact=True)
    # cost = gross - net = turnover * bps: the subtraction rounds differently per gross (rel 1e-9, abs 0).
    pd.testing.assert_series_equal(
        pnl_curve["cost"], default_curve["cost"], check_exact=False, rtol=1e-9, atol=0.0
    )
    assert not np.allclose(pnl_curve["return"].to_numpy(), default_curve["return"].to_numpy())


class TestMeasurementOnly:
    @pytest.fixture(scope="class")
    def world(self):
        cfg = _cfg()
        raw = _raw()
        features = transforms_monthly.features_from_raw(raw, cfg)
        assets = _asset_returns(raw, cfg)
        # A synthetic P&L frame far enough from asset_returns that routing it into the
        # decision inputs would move the weights (an affine copy keeps the tilt's ranking).
        noise = np.random.default_rng(99).normal(0.0, 0.05, assets.shape)
        pnl = (assets + noise).where(assets.notna())
        pnl["USO"] = -pnl["USO"]
        cash = compute_monthly_returns(splice.build_core_research_series(raw, cfg))["cash"]
        return cfg, features, assets, pnl, cash

    def test_strategy_moves_only_its_return(self, world):
        cfg, features, assets, pnl, cash = world
        default_curve, default_metrics = driver.run_backtest(
            features, assets, cfg, cash_returns=cash, registry_path=NO_REGISTRY
        )
        pnl_curve, pnl_metrics = driver.run_backtest(
            features, assets, cfg, cash_returns=cash, registry_path=NO_REGISTRY, pnl_returns=pnl
        )
        assert (~default_curve["degraded"]).sum() > 0, "every step degraded — nothing was measured"
        _assert_measurement_only(pnl_curve, default_curve)
        assert pnl_metrics["dates"] == default_metrics["dates"]
        assert pnl_metrics["classes"] == default_metrics["classes"]
        for a, b in zip(pnl_metrics["proba"], default_metrics["proba"]):
            np.testing.assert_array_equal(a, b)

    def test_ablation_moves_only_its_return(self, world):
        from trading_crab_lib.platform.backtest.baselines import no_regime_ablation

        cfg, features, assets, pnl, cash = world
        default_curve, _ = no_regime_ablation(features, assets, cfg, cash_returns=cash, registry_path=NO_REGISTRY)
        pnl_curve, _ = no_regime_ablation(
            features, assets, cfg, cash_returns=cash, registry_path=NO_REGISTRY, pnl_returns=pnl
        )
        _assert_measurement_only(pnl_curve, default_curve)


# ── 3c. Baselines and the report read the P&L series (D-08) ─────────────────


def _dev(series: pd.Series) -> pd.Series:
    from trading_crab_lib.platform.honesty.holdout import DEFAULT_HOLDOUT_CUTOFF, split_by_holdout_boundary

    return split_by_holdout_boundary(series, cutoff=DEFAULT_HOLDOUT_CUTOFF)[0]


class TestBaselinesReadPnl:
    def _expected(self, research: pd.DataFrame) -> dict[str, pd.Series]:
        from trading_crab_lib.platform.backtest.baselines import faber_sma, sixty_forty, spy_buy_hold

        ret = compute_monthly_returns(research)
        return {
            "spy_buy_hold": spy_buy_hold(_dev(ret["equities_tr"])),
            "sixty_forty": sixty_forty(
                _dev(ret["equities_tr"]), _dev(ret["long_duration_tr"]), rebalance="monthly", cost_bps=10
            ),
            "faber_sma": faber_sma(_dev(research["equities_tr"]), _dev(ret["cash"]), cost_bps=10),
        }

    def test_without_a_block_every_curve_is_unchanged(self):
        from trading_crab_lib.platform.backtest.baselines import baseline_curves

        cfg, raw = _cfg(), _raw()
        curves = baseline_curves(raw, cfg)
        for name, expected in self._expected(splice.build_core_research_series(raw, cfg)).items():
            pd.testing.assert_series_equal(curves[name], expected, check_exact=True)

    def test_with_a_block_spy_6040_and_faber_signal_and_returns_are_month_end(self):
        from trading_crab_lib.platform.backtest.baselines import baseline_curves

        cfg, raw = _cfg(pnl_splice=EQUITIES_ONLY), _raw()
        curves = baseline_curves(raw, cfg)
        pnl_expected = self._expected(splice.build_pnl_research_series(raw, cfg))
        core_expected = self._expected(splice.build_core_research_series(raw, cfg))
        for name, expected in pnl_expected.items():
            pd.testing.assert_series_equal(curves[name], expected, check_exact=True)
            assert not curves[name].equals(core_expected[name]), name


class TestReportWiring:
    def _run(self, cfg: dict, tmp_path: Path) -> tuple[dict, list, list]:
        from trading_crab_lib.platform.evaluation import report

        raw = _raw()
        features = transforms_monthly.features_from_raw(raw, cfg)
        strategy_calls, ablation_calls = [], []
        real_rb, real_abl = report.run_backtest, report.no_regime_ablation

        def rb(*args, **kwargs):
            strategy_calls.append((args, kwargs))
            return real_rb(*args, **kwargs)

        def abl(*args, **kwargs):
            ablation_calls.append((args, kwargs))
            return real_abl(*args, **kwargs)

        with patch.object(report, "run_backtest", rb), patch.object(report, "no_regime_ablation", abl):
            result = report.run_full_backtest_evaluation(
                features, raw, cfg, registry_path=NO_REGISTRY, output_dir=tmp_path
            )
        return result, strategy_calls, ablation_calls

    def test_both_legs_get_the_same_pnl_object_and_baselines_match_baseline_curves(self, tmp_path):
        from trading_crab_lib.platform.backtest.baselines import baseline_curves
        from trading_crab_lib.platform.evaluation.kpis import terminal_log_wealth

        cfg = _cfg(pnl_splice=EQUITIES_ONLY)
        raw = _raw()
        result, strategy_calls, ablation_calls = self._run(cfg, tmp_path)

        (s_args, s_kwargs), = strategy_calls
        (a_args, a_kwargs), = ablation_calls
        assert s_kwargs["pnl_returns"] is a_kwargs["pnl_returns"]
        expected_pnl = tradable_asset_returns(
            compute_monthly_returns(splice.build_pnl_research_series(raw, cfg)), cfg["splice"]
        )
        pd.testing.assert_frame_equal(s_kwargs["pnl_returns"], expected_pnl, check_exact=True)
        # The legs' decision inputs stay on the feature series (D-09).
        pd.testing.assert_frame_equal(s_args[1], _asset_returns(raw, cfg), check_exact=True)
        assert s_args[1] is a_args[1]
        # The scoreboard reconciles: the report's baselines are baseline_curves' (rel 1e-9).
        curves = baseline_curves(raw, cfg)
        for name, kpis in result["baseline_kpis"].items():
            assert np.isclose(kpis["terminal_log_wealth"], terminal_log_wealth(curves[name]), rtol=1e-9, atol=0), name

    def test_without_a_block_pnl_returns_equal_the_asset_returns(self, tmp_path):
        cfg = _cfg()
        _, strategy_calls, ablation_calls = self._run(cfg, tmp_path)
        (s_args, s_kwargs), = strategy_calls
        pd.testing.assert_frame_equal(s_kwargs["pnl_returns"], s_args[1], check_exact=True)
        assert ablation_calls[0][1]["pnl_returns"] is s_kwargs["pnl_returns"]

    def test_the_conventions_section_names_both_builders(self, tmp_path):
        result, _, _ = self._run(_cfg(pnl_splice=EQUITIES_ONLY), tmp_path)
        markdown = result["report_path"].read_text(encoding="utf-8")
        assert "build_pnl_research_series" in markdown and "build_core_research_series" in markdown
        assert "month-end prices" in markdown
        assert "E-08" in markdown and "E-10" in markdown

    def test_without_a_block_the_conventions_section_does_not_claim_month_end(self, tmp_path):
        # 08.3-02 checkpoint finding 1: the before run's report claimed month-end P&L with no block.
        result, _, _ = self._run(_cfg(), tmp_path)
        markdown = result["report_path"].read_text(encoding="utf-8")
        assert "month-end prices" not in markdown
        assert "no `pnl_splice` block is configured" in markdown and "E-07" in markdown


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

    def test_live_config_has_the_fetch_and_lag_entries_and_the_pnl_splice_block(self):
        # 08.3 (2026-10-04): `"pnl_splice" not in cfg` -> the live block equals PNL_SPLICE_OVERLAYS
        # (08.3-02 Task 3 wrote it after the before run, the migration and Glenn's "Run").
        cfg = load_platform_config()
        assert cfg["index_monthly"]["^GSPC"] == {"name": "sp500_close_me", "pnl_only": True}
        assert cfg["fred_monthly"]["series"]["DGS10"] == {"name": "dgs10_me", "pnl_only": True}
        assert cfg["fred_monthly"]["series"]["DCOILWTICO"] == {"name": "wti_me", "pnl_only": True}
        for name in ("sp500_close_me", "dgs10_me", "wti_me"):
            assert cfg["publication_lags"][name] == 0, name
        assert splice.pnl_only_columns(cfg) == {"sp500_close_me", "dgs10_me", "wti_me"}
        # The overlay's sources are exactly the live P&L-only columns plus feature-side wti_fred.
        assert splice.pnl_only_columns({**cfg, "pnl_splice": PNL_SPLICE_OVERLAYS}) == splice.pnl_only_columns(cfg)
        assert "^GSPC" not in str(cfg["universe"])
        # The live block is the constant, verbatim; join_date stays a string (quoted in the YAML).
        # 2026-10-09: plus gold's overlay (World Bank average through 2005-01, IAU close after).
        assert cfg["pnl_splice"] == {**PNL_SPLICE_OVERLAYS, "gold": GOLD_PNL_OVERLAY}
        assert isinstance(cfg["pnl_splice"]["oil"]["join_date"], str)
        # The feature-side splice block is untouched by the overlay (D-01).
        assert cfg["splice"]["equities"]["price_col"] == "sp500"
        assert cfg["splice"]["long_duration"]["yield_col"] == "fred_gs10"
        assert cfg["splice"]["oil"]["source_col"] == ["wti_fred", "wti_crude"]



# ── 5. The tracked migration (08.3-02 Task 1; V1, V4) ───────────────────────

REPO_ROOT = Path(__file__).resolve().parents[2]
TRACKED_RAW = REPO_ROOT / "data" / "checkpoints" / "platform" / "monthly_raw.parquet"
TRACKED_DEV_FEATURES = REPO_ROOT / "data" / "checkpoints" / "platform" / "monthly_features.parquet"
TRACKED_HOLDOUT_FEATURES = REPO_ROOT / "data" / "holdout" / "monthly_features.parquet"
TRACKED_MARKER = REPO_ROOT / "data" / "checkpoints" / "platform" / "publication_lags.json"
#: The pre-08.3 data commit: the tracked monthly_raw before the additive migration.
PRE_08_3_DATA_COMMIT = "b2907787694fdc04dacaa1e11cf82b8395923533"
MONTH_END_COLUMNS = ["sp500_close_me", "dgs10_me", "wti_me"]


def _pre_migration_raw() -> pd.DataFrame | None:
    """The tracked monthly_raw at the pre-08.3 data commit, or None if git/the blob is unreachable."""
    import io
    import subprocess

    try:
        out = subprocess.run(
            ["git", "show", f"{PRE_08_3_DATA_COMMIT}:data/checkpoints/platform/monthly_raw.parquet"],
            capture_output=True, check=True, cwd=REPO_ROOT,
        ).stdout
        return pd.read_parquet(io.BytesIO(out))
    except (subprocess.CalledProcessError, FileNotFoundError, OSError, ValueError):
        return None


class TestTrackedMigration:
    def test_the_47_existing_columns_equal_the_pre_08_3_blob_exactly(self):
        before = _pre_migration_raw()
        if before is None:
            pytest.skip(f"git blob {PRE_08_3_DATA_COMMIT[:8]} unreachable")
        raw = pd.read_parquet(TRACKED_RAW)
        assert before.shape == (776, 47)
        assert raw.shape == (776, 50)
        assert raw.index.equals(before.index)
        assert list(raw.columns) == list(before.columns) + MONTH_END_COLUMNS
        pd.testing.assert_frame_equal(raw[before.columns], before, check_exact=True)

    def test_coverage_of_the_month_end_columns(self):
        raw = pd.read_parquet(TRACKED_RAW)
        assert raw["sp500_close_me"].notna().all() and raw["dgs10_me"].notna().all()
        assert raw.loc[raw.index < OIL_JOIN, "wti_me"].isna().all()
        assert raw.loc[raw.index >= OIL_JOIN, "wti_me"].notna().all()

    def test_features_rebuilt_from_the_tracked_raw_equal_dev_and_holdout_exactly(self):
        """D-01 neutrality on real data: the migration moved no feature."""
        from trading_crab_lib.platform.honesty.holdout import DEFAULT_HOLDOUT_CUTOFF, split_by_holdout_boundary

        cfg = load_platform_config()
        raw = pd.read_parquet(TRACKED_RAW)
        assert set(MONTH_END_COLUMNS) <= set(raw.columns)
        rebuilt = transforms_monthly.features_from_raw(raw, cfg)
        assert not (set(MONTH_END_COLUMNS) & set(rebuilt.columns))
        dev, holdout = split_by_holdout_boundary(rebuilt, cutoff=DEFAULT_HOLDOUT_CUTOFF)
        pd.testing.assert_frame_equal(dev, pd.read_parquet(TRACKED_DEV_FEATURES), check_exact=True, check_freq=False)
        pd.testing.assert_frame_equal(
            holdout, pd.read_parquet(TRACKED_HOLDOUT_FEATURES), check_exact=True, check_freq=False
        )

    def test_the_tracked_marker_matches_the_live_lag_table(self):
        from trading_crab_lib.platform.ingestion.publication_lags import lag_marker_matches

        # 2026-10-09: the lag table the tracked data was built under (gold_spot, not gold_wb).
        assert lag_marker_matches(TRACKED_MARKER, tracked_record_cfg(load_platform_config()))

    @pytest.mark.parametrize("dropped", MONTH_END_COLUMNS)
    def test_dropping_a_lag_entry_makes_apply_publication_lags_raise(self, dropped):
        from trading_crab_lib.platform.ingestion.publication_lags import apply_publication_lags

        cfg = copy.deepcopy(load_platform_config())
        raw = pd.read_parquet(TRACKED_RAW)[MONTH_END_COLUMNS]
        apply_publication_lags(raw, cfg)  # the live table lists all three
        del cfg["publication_lags"][dropped]
        with pytest.raises(ValueError, match=dropped):
            apply_publication_lags(raw, cfg)


# ── 5b. The live P&L series on tracked data (08.3-02 Task 3; V2, V6) ───────


class TestTrackedPnlSeries:
    """The live config's P&L series on the tracked monthly_raw, 1972-2020 (return statistics only)."""

    @pytest.fixture(scope="class")
    def series(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        cfg = tracked_record_cfg(load_platform_config())  # 2026-10-09: built before the gold switch
        raw = pd.read_parquet(TRACKED_RAW)
        feature = compute_monthly_returns(splice.build_core_research_series(raw, cfg))
        pnl = compute_monthly_returns(splice.build_pnl_research_series(raw, cfg))
        return raw, feature, pnl

    @staticmethod
    def _passes(own: pd.Series, feature: pd.Series) -> bool:
        """V2: no own lag-1 persistence, and next month's AVERAGE return leads this month's return."""
        own, feature = own.loc["1972-01-31":"2020-12-31"], feature.loc["1972-01-31":"2020-12-31"]
        return abs(own.shift(-1).corr(own)) < 0.15 and feature.shift(-1).corr(own) > 0.3

    @pytest.mark.parametrize("col", ["equities_tr", "long_duration_tr", "oil"])
    def test_month_end_pnl_passes_and_the_averaged_series_fails(self, series, col):
        _, feature, pnl = series
        assert self._passes(pnl[col], feature[col]), col
        # SC1: the same check on the averaged series fails (its own lag-1 is the leak).
        assert not self._passes(feature[col], feature[col]), col

    def test_oil_is_average_over_average_through_the_join_and_close_over_close_after(self, series):
        raw, _, pnl = series
        jan = raw.loc["1986-01-31", "wti_fred"] / raw.loc["1985-12-31", "wti_fred"] - 1
        feb = raw.loc["1986-02-28", "wti_me"] / raw.loc["1986-01-31", "wti_me"] - 1
        # rel 1e-9, not exact: ratio_splice rescales the level (08.3-02 finding 3).
        assert pnl.loc["1986-01-31", "oil"] == pytest.approx(jan, rel=1e-9, abs=0)
        assert pnl.loc["1986-02-28", "oil"] == pytest.approx(feb, rel=1e-9, abs=0)
        assert np.isnan(raw.loc["1985-12-31", "wti_me"])


class TestMigrationScriptGuards:
    """``scripts/migrate_month_end_columns.py``: the checks it raises on before saving."""

    def _frames(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        raw = pd.read_parquet(TRACKED_RAW)
        return raw, raw.drop(columns=MONTH_END_COLUMNS)

    def test_the_tracked_raw_passes_every_check(self):
        from migrate_month_end_columns import check_migrated

        new, existing = self._frames()
        corrs = check_migrated(new, existing)
        assert all(v >= 0.98 for v in corrs.values()), corrs

    def test_a_changed_existing_cell_raises(self):
        from migrate_month_end_columns import check_migrated

        new, existing = self._frames()
        new = new.copy()
        new.iloc[100, new.columns.get_loc("sp500")] *= 1 + 1e-12
        with pytest.raises(AssertionError):
            check_migrated(new, existing)

    def test_merge_style_column_reordering_raises(self):
        from migrate_month_end_columns import check_migrated

        new, existing = self._frames()
        reordered = new[MONTH_END_COLUMNS + list(existing.columns)]  # what merge=True's ordered_cols does
        with pytest.raises(AssertionError, match="column order"):
            check_migrated(reordered, existing)

    def test_an_added_row_raises(self):
        from migrate_month_end_columns import check_migrated

        new, existing = self._frames()
        extra = pd.DataFrame(np.nan, index=[pd.Timestamp("2026-09-30")], columns=new.columns)
        with pytest.raises(AssertionError, match="index"):
            check_migrated(pd.concat([new, extra]), existing)

    def test_wti_values_before_the_join_raise(self):
        from migrate_month_end_columns import check_migrated

        new, existing = self._frames()
        new = new.copy()
        new.loc[pd.Timestamp("1985-12-31"), "wti_me"] = 26.0
        with pytest.raises(AssertionError, match="wti_me"):
            check_migrated(new, existing)

    def test_a_second_run_is_refused_before_any_fetch(self):
        import migrate_month_end_columns as mig

        with patch.object(mig, "fetch_month_end_columns") as fetch, pytest.raises(SystemExit, match="runs once"):
            mig.migrate(load_platform_config(), dry_run=True)
        fetch.assert_not_called()


# ── 6. The smoothed-vs-filtered gap's hindsight oracle earns the P&L series ──

PIT_08_3_BEFORE = REPO_ROOT / "outputs" / "reports" / "platform" / "pit_08.3" / "before"


def _oracle_inputs(raw: pd.DataFrame, cfg: dict) -> tuple[pd.Series, pd.DataFrame, pd.Series]:
    returns = compute_monthly_returns(splice.build_core_research_series(raw, cfg))
    states = pd.Series(np.arange(len(raw)) // 6 % 3, index=raw.index, name="state")
    return states, tradable_asset_returns(returns, cfg["splice"]), returns[cfg["splice"]["cash"]["research_name"]]


class TestHindsightOracleReadsPnl:
    """``report._smoothed_hindsight_perf``: decisions on ``asset_returns`` (E-10 / D-09), the
    realized return on ``pnl_returns`` — the same split ``run_backtest`` makes for the
    filtered leg the gap compares it with (carried forward from 08.3-01)."""

    ALLOCATION = {"target_vol_annual": 0.10, "ewma_halflife_months": 6, "portfolio_vol_min_obs": 12}

    def test_pnl_equal_to_the_asset_returns_is_byte_identical(self):
        from trading_crab_lib.platform.evaluation.report import _smoothed_hindsight_perf

        cfg = _cfg()
        states, assets, cash = _oracle_inputs(_raw(), cfg)
        dates = list(assets.index[24:])
        default = _smoothed_hindsight_perf(states, assets, cash, dates, self.ALLOCATION)
        same = _smoothed_hindsight_perf(states, assets, cash, dates, self.ALLOCATION, pnl_returns=assets.copy())
        assert np.isfinite(default) and same == default

    def test_only_the_realized_return_reads_pnl(self):
        from trading_crab_lib.platform.evaluation import report
        from trading_crab_lib.platform.evaluation.kpis import terminal_log_wealth

        cfg = _cfg()
        raw = _raw()
        states, assets, cash = _oracle_inputs(raw, cfg)
        pnl = assets + pd.DataFrame(
            np.random.default_rng(5).normal(0, 0.05, assets.shape), index=assets.index, columns=assets.columns
        )
        pnl[assets.columns[0]] = -pnl[assets.columns[0]]
        dates = list(assets.index[24:])
        tilts: dict[str, list] = {"default": [], "pnl": []}
        real_tilt = report.vol_targeted_tilt

        def spy(key):
            def _tilt(*args, **kwargs):
                out = real_tilt(*args, **kwargs)
                tilts[key].append(out)
                return out
            return _tilt

        with patch.object(report, "vol_targeted_tilt", spy("default")):
            default = report._smoothed_hindsight_perf(states, assets, cash, dates, self.ALLOCATION)
        with patch.object(report, "vol_targeted_tilt", spy("pnl")):
            got = report._smoothed_hindsight_perf(states, assets, cash, dates, self.ALLOCATION, pnl_returns=pnl)

        # Decision side unchanged: the same weights and cash at every date.
        for a, b in zip(tilts["default"], tilts["pnl"], strict=True):
            pd.testing.assert_series_equal(a["weights"], b["weights"], check_exact=True)
            assert a["cash"] == b["cash"]
        # Realized side: the pnl frame, booked with those weights.
        steps = [
            float((t["weights"] * pnl.loc[d, t["weights"].index]).sum()) + t["cash"] * float(cash.loc[d])
            for t, d in zip(tilts["pnl"], dates, strict=True)
        ]
        assert got == pytest.approx(terminal_log_wealth(pd.Series(steps)), rel=1e-12, abs=0)
        assert got != default

    def test_the_report_passes_the_same_pnl_object_as_the_legs(self, tmp_path):
        from trading_crab_lib.platform.evaluation import report

        cfg = _cfg(pnl_splice=EQUITIES_ONLY)
        raw = _raw()
        features = transforms_monthly.features_from_raw(raw, cfg)
        seen: dict[str, list] = {"oracle": [], "legs": []}
        real_oracle, real_rb = report._smoothed_hindsight_perf, report.run_backtest

        def oracle(*args, **kwargs):
            seen["oracle"].append(kwargs)
            return real_oracle(*args, **kwargs)

        def rb(*args, **kwargs):
            seen["legs"].append(kwargs)
            return real_rb(*args, **kwargs)

        with patch.object(report, "_smoothed_hindsight_perf", oracle), patch.object(report, "run_backtest", rb):
            report.run_full_backtest_evaluation(features, raw, cfg, registry_path=NO_REGISTRY, output_dir=tmp_path)
        (kwargs,) = seen["oracle"]
        assert kwargs["pnl_returns"] is seen["legs"][0]["pnl_returns"]

    def test_before_the_switch_on_the_tracked_gap_cannot_move(self):
        """Real data, no pnl_splice block: the P&L frame IS the asset frame, so the oracle (and the
        gap the 08.3 before run recorded) is unchanged by reading it."""
        import re

        from trading_crab_lib.platform.evaluation.report import _smoothed_hindsight_perf

        cfg = {k: v for k, v in tracked_record_cfg(load_platform_config()).items() if k != "pnl_splice"}
        raw = pd.read_parquet(TRACKED_RAW)
        returns = compute_monthly_returns(splice.build_core_research_series(raw, cfg))
        assets = tradable_asset_returns(returns, cfg["splice"])
        pnl = tradable_asset_returns(compute_monthly_returns(splice.build_pnl_research_series(raw, cfg)), cfg["splice"])
        pd.testing.assert_frame_equal(pnl, assets, check_exact=True)

        states = pd.read_parquet(PIT_08_3_BEFORE / "backtest_full_sample_states.parquet")["state"]
        # The oracle visits the driver's non-degraded DECISION dates (per_step_metrics["dates"]),
        # which are the persisted filtered_state_probs index, not the curve's booking dates.
        dates = list(pd.read_parquet(PIT_08_3_BEFORE / "backtest_filtered_state_probs.parquet").index)
        cash = returns[cfg["splice"]["cash"]["research_name"]]
        allocation = cfg.get("allocation", {})
        default = _smoothed_hindsight_perf(states, assets, cash, dates, allocation)
        assert _smoothed_hindsight_perf(states, assets, cash, dates, allocation, pnl_returns=pnl) == default

        kpi = pd.read_parquet(PIT_08_3_BEFORE / "backtest_kpi_table.parquet").set_index("leg")
        text = (PIT_08_3_BEFORE / "backtest_report.md").read_text(encoding="utf-8")
        recorded = float(re.search(r"real-time filtered performance\): (-?[0-9.]+)", text).group(1))
        assert round(default - float(kpi.loc["strategy", "terminal_log_wealth"]), 4) == recorded


# ── Gold on the World Bank average, P&L on month-end (2026-10-09, DECISIONS D-09) ──


def test_live_gold_pnl_is_average_through_the_join_and_iau_close_after():
    """Features read gold_wb (a monthly average); P&L reads it only through 2005-01 and IAU's
    month-end closes from 2005-02 (E-08). The live chain and overlay, on a small synthetic frame."""
    live = load_platform_config()
    cfg = {**live, "splice": {"gold": live["splice"]["gold"]}, "pnl_splice": {"gold": live["pnl_splice"]["gold"]}}
    idx = pd.date_range("2004-10-31", "2005-04-30", freq="ME", name="date")
    raw = pd.DataFrame(
        {
            "gold_wb": [420.0, 439.0, 442.0, 424.0, 423.0, 434.0, 429.0],
            "IAU": [np.nan, np.nan, np.nan, 42.3, 43.6, 42.8, 43.5],
        },
        index=idx,
    )

    feature = compute_monthly_returns(splice.build_core_research_series(raw, cfg))["gold"]
    pnl = compute_monthly_returns(splice.build_pnl_research_series(raw, cfg))["gold"]

    wb, iau = raw["gold_wb"], raw["IAU"]
    pd.testing.assert_series_equal(feature, wb.pct_change(fill_method=None), check_names=False, check_freq=False)
    assert pnl.loc["2005-01-31"] == pytest.approx(wb["2005-01-31"] / wb["2004-12-31"] - 1, rel=1e-9, abs=0)
    for month, prev in (("2005-02-28", "2005-01-31"), ("2005-04-30", "2005-03-31")):
        assert pnl.loc[month] == pytest.approx(iau[month] / iau[prev] - 1, rel=1e-9, abs=0), month
    assert not {"gold_wb", "IAU"} & splice.pnl_only_columns(cfg)  # both stay feature-side sources
