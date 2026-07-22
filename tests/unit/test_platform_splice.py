"""Tests for trading_crab_lib.platform.splice (Phase 1 Plan 01-02 Task 1).

All synthetic Series/DataFrames — no network calls, mirroring the incumbent's
test convention (tests/unit/test_transforms.py).
"""

from __future__ import annotations

import pandas as pd
import pytest

from trading_crab_lib.platform.splice import (
    bond_price,
    build_core_research_series,
    build_equity_total_return,
    build_treasury_tr_synthetic,
    monthly_total_return,
    ratio_splice,
)


def _monthly_index(start: str, periods: int) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=periods, freq="ME")


# Mirrors the schema of config/platform_settings.yaml's `splice` block —
# kept inline (not loaded from disk) so this test suite stays isolated from
# concurrent edits to platform_settings.yaml by sibling plans in this wave.
SPLICE_CFG: dict = {
    "splice": {
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
}


class TestRatioSplice:
    def test_splice_continuity_at_join(self):
        idx_old = _monthly_index("2000-01-31", 6)
        idx_new = _monthly_index("2000-04-30", 6)
        old = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0, 15.0], index=idx_old, name="old")
        new = pd.Series([200.0, 210.0, 220.0, 230.0, 240.0, 250.0], index=idx_new, name="new")
        join_date = idx_new[0]

        result = ratio_splice(old, new, join_date)

        assert abs(result.loc[join_date] - new.loc[join_date]) < 1e-9

    def test_pre_join_segment_scaled_to_match_new(self):
        idx_old = _monthly_index("2000-01-31", 6)
        idx_new = _monthly_index("2000-04-30", 6)
        old = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0, 15.0], index=idx_old, name="old")
        new = pd.Series([200.0, 210.0, 220.0, 230.0, 240.0, 250.0], index=idx_new, name="new")
        join_date = idx_new[0]
        scale = new.loc[join_date] / old.loc[join_date]

        result = ratio_splice(old, new, join_date)

        last_pre_join = old.index[old.index < join_date][-1]
        assert abs(result.loc[last_pre_join] - old.loc[last_pre_join] * scale) < 1e-9

    def test_result_named_after_new(self):
        idx = _monthly_index("2000-01-31", 3)
        old = pd.Series([1.0, 2.0, 3.0], index=idx, name="old")
        new = pd.Series([10.0, 20.0, 30.0], index=idx, name="new")

        result = ratio_splice(old, new, idx[1])

        assert result.name == "new"

    def test_raises_when_join_date_missing_from_old(self):
        idx_old = _monthly_index("2000-01-31", 6)
        idx_new = _monthly_index("2000-07-31", 6)
        old = pd.Series(range(6), index=idx_old, dtype=float)
        new = pd.Series(range(6), index=idx_new, dtype=float)

        with pytest.raises(ValueError, match="old"):
            ratio_splice(old, new, idx_new[0])

    def test_raises_when_join_date_missing_from_new(self):
        idx_old = _monthly_index("2000-01-31", 6)
        idx_new = _monthly_index("2000-07-31", 6)
        old = pd.Series(range(6), index=idx_old, dtype=float)
        new = pd.Series(range(6), index=idx_new, dtype=float)

        with pytest.raises(ValueError, match="new"):
            ratio_splice(old, new, idx_old[-1])

    def test_does_not_mutate_inputs(self):
        idx_old = _monthly_index("2000-01-31", 4)
        idx_new = _monthly_index("2000-03-31", 4)
        old = pd.Series([1.0, 2.0, 3.0, 4.0], index=idx_old)
        new = pd.Series([100.0, 200.0, 300.0, 400.0], index=idx_new)
        old_copy, new_copy = old.copy(), new.copy()

        ratio_splice(old, new, idx_new[0])

        pd.testing.assert_series_equal(old, old_copy)
        pd.testing.assert_series_equal(new, new_copy)


class TestBondPrice:
    def test_par_bond_prices_to_par(self):
        price = bond_price(0.05, 0.05, 10)
        assert abs(price - 1.0) < 1e-6

    def test_discount_bond_when_yield_above_coupon(self):
        price = bond_price(0.07, 0.05, 10)
        assert price < 1.0

    def test_premium_bond_when_yield_below_coupon(self):
        price = bond_price(0.03, 0.05, 10)
        assert price > 1.0


class TestMonthlyTotalReturn:
    def test_rising_yield_hurts_total_return(self):
        assert monthly_total_return(0.04, 0.05, 10) < 0.05 / 12

    def test_falling_yield_helps_total_return(self):
        assert monthly_total_return(0.05, 0.04, 10) > 0.05 / 12


class TestBuildTreasuryTrSynthetic:
    def test_builds_cumulative_index_from_yield_series(self):
        idx = _monthly_index("1962-01-31", 24)
        yields = pd.Series([0.04 + 0.0005 * i for i in range(24)], index=idx)

        result = build_treasury_tr_synthetic(yields, SPLICE_CFG)

        assert result.name == "long_duration_tr"
        assert len(result) == len(yields)
        assert result.iloc[0] == pytest.approx(1.0)
        assert not result.isna().any()

    def test_falling_yields_produce_rising_index(self):
        idx = _monthly_index("1962-01-31", 12)
        yields = pd.Series([0.06 - 0.002 * i for i in range(12)], index=idx)

        result = build_treasury_tr_synthetic(yields, SPLICE_CFG)

        assert result.iloc[-1] > result.iloc[0]


class TestBuildEquityTotalReturn:
    def test_builds_cumulative_index_starting_at_one(self):
        idx = _monthly_index("1962-01-31", 12)
        price = pd.Series([100 * (1.01**i) for i in range(12)], index=idx)
        div_yield = pd.Series([0.03] * 12, index=idx)

        result = build_equity_total_return(price, div_yield, SPLICE_CFG)

        assert result.name == "equities_tr"
        assert result.iloc[0] == pytest.approx(1.0)
        assert result.iloc[-1] > result.iloc[0]

    def test_does_not_mutate_inputs(self):
        idx = _monthly_index("1962-01-31", 6)
        price = pd.Series([100.0] * 6, index=idx)
        div_yield = pd.Series([0.02] * 6, index=idx)
        price_copy, div_copy = price.copy(), div_yield.copy()

        build_equity_total_return(price, div_yield, SPLICE_CFG)

        pd.testing.assert_series_equal(price, price_copy)
        pd.testing.assert_series_equal(div_yield, div_copy)


class TestBuildCoreResearchSeries:
    def test_returns_exactly_five_research_columns(self):
        idx = _monthly_index("1962-01-31", 24)
        raw = pd.DataFrame(
            {
                "sp500": [100 * (1.01**i) for i in range(24)],
                "div_yield": [0.03] * 24,
                "fred_gs10": [0.04 + 0.0005 * i for i in range(24)],
                "gold_spot": [35.0 + i for i in range(24)],
                "wti_crude": [3.0 + 0.1 * i for i in range(24)],
                "fred_tb3ms": [0.02] * 24,
            },
            index=idx,
        )

        result = build_core_research_series(raw, SPLICE_CFG)

        expected_columns = {"equities_tr", "long_duration_tr", "gold", "oil", "cash"}
        assert set(result.columns) == expected_columns

    def test_unknown_method_raises(self):
        idx = _monthly_index("1962-01-31", 3)
        raw = pd.DataFrame({"x": [1.0, 2.0, 3.0]}, index=idx)
        bad_cfg = {"splice": {"bogus": {"research_name": "bogus", "method": "not_a_real_method"}}}

        with pytest.raises(ValueError, match="Unknown splice method"):
            build_core_research_series(raw, bad_cfg)
