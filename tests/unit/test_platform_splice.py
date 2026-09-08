"""Tests for trading_crab_lib.platform.splice (Phase 1 Plan 01-02 Task 1).

All synthetic Series/DataFrames — no network calls, mirroring the incumbent's
test convention (tests/unit/test_transforms.py).
"""

from __future__ import annotations

import json
import logging

import pandas as pd
import pytest

from trading_crab_lib.platform.splice import (
    bond_price,
    build_core_research_series,
    build_equity_total_return,
    build_treasury_tr_synthetic,
    monthly_total_return,
    ratio_splice,
    resolve_class_sources,
    resolve_source_column,
    source_candidates,
    write_splice_provenance,
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

    def test_missing_source_column_raises_clear_error(self):
        # Simulate a failed macrotrends fetch: gold_spot / wti_crude never arrive.
        # The preflight must raise ONE actionable ValueError naming both missing
        # columns (and the likely cause) rather than a bare KeyError deep in the loop.
        idx = _monthly_index("1962-01-31", 24)
        raw = pd.DataFrame(
            {
                "sp500": [100 * (1.01**i) for i in range(24)],
                "div_yield": [0.03] * 24,
                "fred_gs10": [0.04 + 0.0005 * i for i in range(24)],
                "fred_tb3ms": [0.02] * 24,
                # gold_spot and wti_crude intentionally absent
            },
            index=idx,
        )

        with pytest.raises(ValueError, match="required source columns are missing") as exc:
            build_core_research_series(raw, SPLICE_CFG)

        message = str(exc.value)
        assert "gold_spot" in message
        assert "wti_crude" in message
        assert "macrotrends" in message  # actionable cause hint

    def test_single_source_falls_back_to_fallback_col(self):
        # oil's primary source (macrotrends wti_crude) is absent, but its
        # configured fallback_col (FRED wti_fred) is present — the class must be
        # assembled from the fallback with NO error, mirroring a macrotrends block.
        idx = _monthly_index("1962-01-31", 12)
        raw = pd.DataFrame(
            {
                "wti_fred": [20.0 + i for i in range(12)],  # FRED oil present
                # wti_crude (macrotrends) intentionally absent
            },
            index=idx,
        )
        cfg = {
            "splice": {
                "oil": {
                    "research_name": "oil",
                    "method": "single_source",
                    "source_col": "wti_crude",
                    "fallback_col": "wti_fred",
                },
            }
        }

        result = build_core_research_series(raw, cfg)

        assert "oil" in result.columns
        assert list(result["oil"].dropna().values) == [20.0 + i for i in range(12)]

    def test_optional_class_skipped_when_source_missing(self):
        # gold is optional and its source (gold_spot) is absent → the class is
        # skipped WITHOUT error; the other classes still assemble.
        idx = _monthly_index("1962-01-31", 12)
        raw = pd.DataFrame(
            {
                "sp500": [100 * (1.01**i) for i in range(12)],
                "div_yield": [0.03] * 12,
                "fred_tb3ms": [0.02] * 12,
                # gold_spot intentionally absent
            },
            index=idx,
        )
        cfg = {
            "splice": {
                "equities": {
                    "research_name": "equities_tr",
                    "method": "total_return_from_price_div",
                    "price_col": "sp500",
                    "div_yield_col": "div_yield",
                },
                "gold": {
                    "research_name": "gold",
                    "method": "single_source",
                    "source_col": "gold_spot",
                    "optional": True,
                },
                "cash": {
                    "research_name": "cash",
                    "method": "yield_as_return",
                    "yield_col": "fred_tb3ms",
                },
            }
        }

        result = build_core_research_series(raw, cfg)

        assert "gold" not in result.columns  # skipped, not errored
        assert "equities_tr" in result.columns
        assert "cash" in result.columns

    def test_optional_class_included_when_source_present(self):
        # When gold_spot IS present, the optional gold class is still assembled.
        idx = _monthly_index("1962-01-31", 6)
        raw = pd.DataFrame(
            {"gold_spot": [1800.0 + i for i in range(6)], "fred_tb3ms": [0.02] * 6},
            index=idx,
        )
        cfg = {
            "splice": {
                "gold": {
                    "research_name": "gold",
                    "method": "single_source",
                    "source_col": "gold_spot",
                    "optional": True,
                },
                "cash": {"research_name": "cash", "method": "yield_as_return", "yield_col": "fred_tb3ms"},
            }
        }

        result = build_core_research_series(raw, cfg)

        assert "gold" in result.columns
        assert list(result["gold"].dropna().values) == [1800.0 + i for i in range(6)]


class TestSourceChains:
    """source_candidates / resolve_source_column / resolve_class_sources —
    the config-driven multi-source fallback-chain generalization."""

    # ── source_candidates() ─────────────────────────────────────────────

    def test_scalar_value_becomes_one_element_list(self):
        assert source_candidates({"source_col": "gold_spot"}, "source_col") == ["gold_spot"]

    def test_list_value_taken_as_is(self):
        params = {"source_col": ["gold_spot", "IAU"]}
        assert source_candidates(params, "source_col") == ["gold_spot", "IAU"]

    def test_missing_key_is_empty_list(self):
        assert source_candidates({}, "source_col") == []

    def test_legacy_fallback_col_appended_for_source_col_key(self):
        params = {"source_col": "wti_crude", "fallback_col": "wti_fred"}
        assert source_candidates(params, "source_col") == ["wti_crude", "wti_fred"]

    def test_fallback_col_not_duplicated_if_already_listed(self):
        params = {"source_col": ["wti_crude", "wti_fred"], "fallback_col": "wti_fred"}
        assert source_candidates(params, "source_col") == ["wti_crude", "wti_fred"]

    def test_fallback_col_ignored_for_non_source_col_keys(self):
        params = {"yield_col": "a", "fallback_col": "b"}
        assert source_candidates(params, "yield_col") == ["a"]

    def test_deduplicates_preserving_order(self):
        assert source_candidates({"source_col": ["a", "b", "a"]}, "source_col") == ["a", "b"]

    # ── resolve_source_column() ─────────────────────────────────────────

    def test_resolves_first_candidate_when_present(self):
        params = {"source_col": ["gold_spot", "IAU"]}
        assert resolve_source_column(params, "source_col", {"gold_spot", "IAU"}) == "gold_spot"

    def test_resolves_second_candidate_when_first_absent(self):
        params = {"source_col": ["gold_spot", "IAU"]}
        assert resolve_source_column(params, "source_col", {"IAU"}) == "IAU"

    def test_none_when_no_candidate_present(self):
        params = {"source_col": ["gold_spot", "IAU"]}
        assert resolve_source_column(params, "source_col", {"SPY"}) is None

    # ── resolve_class_sources() ─────────────────────────────────────────

    def test_resolves_all_required_keys(self):
        params = {"method": "single_source", "source_col": "gold_spot"}
        assert resolve_class_sources(params, {"gold_spot"}) == {"source_col": "gold_spot"}

    def test_none_when_any_required_key_unresolvable(self):
        params = {"method": "single_source", "source_col": "gold_spot"}
        assert resolve_class_sources(params, {"SPY"}) is None

    def test_multi_key_method_resolves_both_keys(self):
        params = {
            "method": "total_return_from_price_div",
            "price_col": "sp500",
            "div_yield_col": "div_yield",
        }
        resolved = resolve_class_sources(params, {"sp500", "div_yield"})
        assert resolved == {"price_col": "sp500", "div_yield_col": "div_yield"}

    # ── chain resolution through build_core_research_series() ──────────

    def test_first_candidate_resolves_with_info_log_position_1_of_n(self, caplog):
        idx = pd.date_range("1962-01-31", periods=6, freq="ME")
        raw = pd.DataFrame({"gold_spot": [1000.0 + i for i in range(6)]}, index=idx)
        cfg = {
            "splice": {
                "gold": {
                    "research_name": "gold",
                    "method": "single_source",
                    "source_col": ["gold_spot", "IAU"],
                },
            }
        }

        with caplog.at_level(logging.INFO):
            result = build_core_research_series(raw, cfg)

        assert list(result["gold"].dropna().values) == [1000.0 + i for i in range(6)]
        info_logs = [r.message for r in caplog.records if r.levelno == logging.INFO]
        assert any("gold_spot" in m and "1 of 2" in m for m in info_logs)
        prov = result.attrs["splice_provenance"]["gold"]
        assert prov["status"] == "primary"
        assert prov["sources"]["source_col"] == {
            "candidates": ["gold_spot", "IAU"], "resolved": "gold_spot", "position": 1,
        }

    def test_second_candidate_resolves_with_info_log_position_2_of_n(self, caplog):
        idx = pd.date_range("1962-01-31", periods=6, freq="ME")
        raw = pd.DataFrame({"IAU": [30.0 + i for i in range(6)]}, index=idx)  # gold_spot absent
        cfg = {
            "splice": {
                "gold": {
                    "research_name": "gold",
                    "method": "single_source",
                    "source_col": ["gold_spot", "IAU"],
                },
            }
        }

        with caplog.at_level(logging.INFO):
            result = build_core_research_series(raw, cfg)

        assert "gold" in result.columns
        assert list(result["gold"].dropna().values) == [30.0 + i for i in range(6)]
        info_logs = [r.message for r in caplog.records if r.levelno == logging.INFO]
        assert any("IAU" in m and "2 of 2" in m for m in info_logs)
        prov = result.attrs["splice_provenance"]["gold"]
        assert prov["status"] == "fallback"
        assert prov["sources"]["source_col"]["resolved"] == "IAU"
        assert prov["sources"]["source_col"]["position"] == 2

    def test_optional_class_all_candidates_absent_skipped_with_warning_listing_all(self, caplog):
        idx = pd.date_range("1962-01-31", periods=6, freq="ME")
        raw = pd.DataFrame({"fred_tb3ms": [0.02] * 6}, index=idx)
        cfg = {
            "splice": {
                "gold": {
                    "research_name": "gold",
                    "method": "single_source",
                    "source_col": ["gold_spot", "IAU"],
                    "optional": True,
                },
                "cash": {"research_name": "cash", "method": "yield_as_return", "yield_col": "fred_tb3ms"},
            }
        }

        with caplog.at_level(logging.WARNING):
            result = build_core_research_series(raw, cfg)

        assert "gold" not in result.columns  # skipped, not errored
        assert "cash" in result.columns
        warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("gold_spot" in m and "IAU" in m for m in warnings)
        prov = result.attrs["splice_provenance"]["gold"]
        assert prov["status"] == "skipped"
        assert prov["sources"]["source_col"]["candidates"] == ["gold_spot", "IAU"]
        assert prov["sources"]["source_col"]["resolved"] is None

    def test_required_class_all_candidates_absent_raises_naming_every_candidate(self):
        idx = pd.date_range("1962-01-31", periods=6, freq="ME")
        raw = pd.DataFrame({"unrelated": [1.0] * 6}, index=idx)
        cfg = {
            "splice": {
                "gold": {
                    "research_name": "gold",
                    "method": "single_source",
                    "source_col": ["gold_spot", "IAU"],
                },
                "oil": {
                    "research_name": "oil",
                    "method": "single_source",
                    "source_col": ["wti_crude", "wti_fred"],
                },
            }
        }

        with pytest.raises(ValueError, match="required source columns are missing") as exc:
            build_core_research_series(raw, cfg)

        message = str(exc.value)
        assert "gold_spot" in message and "IAU" in message
        assert "wti_crude" in message and "wti_fred" in message

    def test_chain_works_on_non_single_source_key(self):
        idx = pd.date_range("1962-01-31", periods=12, freq="ME")
        raw = pd.DataFrame({"b": [0.02] * 12}, index=idx)  # "a" absent, "b" present
        cfg = {
            "splice": {
                "cash": {
                    "research_name": "cash",
                    "method": "yield_as_return",
                    "yield_col": ["a", "b"],
                },
            }
        }

        result = build_core_research_series(raw, cfg)

        assert list(result["cash"].dropna().values) == [0.02] * 12
        prov = result.attrs["splice_provenance"]["cash"]
        assert prov["sources"]["yield_col"]["resolved"] == "b"
        assert prov["sources"]["yield_col"]["position"] == 2
        assert prov["status"] == "fallback"

    # ── gold -> IAU wiring specifically ─────────────────────────────────

    def test_gold_resolves_to_iau_when_gold_spot_absent_no_gold_specific_python_branch(self):
        idx = pd.date_range("1962-01-31", periods=6, freq="ME")
        raw = pd.DataFrame({"IAU": [30.0 + i for i in range(6)]}, index=idx)
        cfg = {
            "splice": {
                "gold": {
                    "research_name": "gold",
                    "method": "single_source",
                    "source_col": ["gold_spot", "IAU"],
                    "optional": True,
                },
            }
        }

        result = build_core_research_series(raw, cfg)

        assert "gold" in result.columns
        assert list(result["gold"].dropna().values) == [30.0 + i for i in range(6)]
        assert result.attrs["splice_provenance"]["gold"]["status"] == "fallback"

    def test_gold_prefers_gold_spot_over_iau_when_both_present(self):
        idx = pd.date_range("1962-01-31", periods=6, freq="ME")
        raw = pd.DataFrame(
            {"gold_spot": [1000.0 + i for i in range(6)], "IAU": [30.0 + i for i in range(6)]},
            index=idx,
        )
        cfg = {
            "splice": {
                "gold": {
                    "research_name": "gold",
                    "method": "single_source",
                    "source_col": ["gold_spot", "IAU"],
                    "optional": True,
                },
            }
        }

        result = build_core_research_series(raw, cfg)

        assert list(result["gold"].dropna().values) == [1000.0 + i for i in range(6)]
        prov = result.attrs["splice_provenance"]["gold"]
        assert prov["status"] == "primary"
        assert prov["sources"]["source_col"]["resolved"] == "gold_spot"

    # ── write_splice_provenance() ───────────────────────────────────────

    def test_write_splice_provenance_round_trips_to_json(self, tmp_path):
        idx = pd.date_range("1962-01-31", periods=6, freq="ME")
        raw = pd.DataFrame({"gold_spot": [1000.0 + i for i in range(6)]}, index=idx)
        cfg = {
            "splice": {
                "gold": {
                    "research_name": "gold",
                    "method": "single_source",
                    "source_col": ["gold_spot", "IAU"],
                },
            }
        }
        result = build_core_research_series(raw, cfg)
        provenance = result.attrs["splice_provenance"]

        path = write_splice_provenance(provenance, tmp_path / "splice_provenance.json")
        loaded = json.loads(path.read_text())

        assert "captured_at" in loaded
        assert loaded["provenance"]["gold"]["status"] == "primary"
        assert loaded["provenance"]["gold"]["sources"]["source_col"]["resolved"] == "gold_spot"
