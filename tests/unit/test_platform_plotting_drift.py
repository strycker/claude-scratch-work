"""Tests for src/trading_crab_lib/platform/plotting/drift.py.

D-09 drift-against-baseline and D-11 plausibility bands, tested as pure
functions with an in-band and an out-of-band case per band. Includes the two
non-negotiable historical regression cases from 06-VALIDATION.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_crab_lib.platform.plotting import drift


# ── Terminal log wealth ──────────────────────────────────────────────────────

class TestAssertTerminalLogWealthPlausible:
    def test_in_band_passes(self):
        assert drift.assert_terminal_log_wealth_plausible(4.0265, leg="strategy") is None

    def test_historical_regression_111_raises(self):
        # Non-negotiable regression case (06-VALIDATION.md): terminal log
        # wealth of 111.06 (e^111 ~ 10^48) must fail the universal band.
        with pytest.raises(ValueError, match="universal"):
            drift.assert_terminal_log_wealth_plausible(111.06, leg="strategy")

    def test_out_of_domain_band_but_in_universal_raises(self):
        # -5.0 has abs(x) = 5 < 10 (passes universal) but is below the
        # domain floor of -3.0 (fails domain).
        with pytest.raises(ValueError, match="domain band"):
            drift.assert_terminal_log_wealth_plausible(-5.0, leg="strategy")


# ── Max drawdown ─────────────────────────────────────────────────────────────

class TestAssertMaxDrawdownPlausible:
    def test_in_band_passes(self):
        assert drift.assert_max_drawdown_plausible(-0.269625, leg="sixty_forty") is None

    def test_historical_regression_sixty_forty_raises(self):
        # Non-negotiable regression case (06-VALIDATION.md): a 60/40-shaped
        # leg at -2.27% max drawdown must fail its domain band, even though
        # it sits inside the universal [-1, 0] bound.
        with pytest.raises(ValueError, match="domain band"):
            drift.assert_max_drawdown_plausible(-0.0227, leg="sixty_forty")

    def test_universal_bound_raises(self):
        with pytest.raises(ValueError, match="universal"):
            drift.assert_max_drawdown_plausible(-1.5, leg="spy_buy_hold")

    def test_unknown_leg_warns_and_applies_universal_only(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING):
            drift.assert_max_drawdown_plausible(-0.02, leg="unknown_leg")
        assert any("no domain band configured" in r.message for r in caplog.records)


# ── Brier ────────────────────────────────────────────────────────────────────

class TestAssertBrierPlausible:
    def test_no_skill_brier_k5_is_016(self):
        assert abs(drift.no_skill_brier(5) - 0.16) < 1e-12

    def test_in_band_but_above_no_skill(self):
        result = drift.assert_brier_plausible(0.208746, n_classes=5)
        assert abs(result["no_skill"] - 0.16) < 1e-12
        assert result["beats_no_skill"] is False
        assert result["value"] == 0.208746

    def test_out_of_band_raises(self):
        # This codebase's Brier is mean((p-onehot)^2) over (n, K) — bound is
        # [0, 1], not the textbook [0, 2].
        with pytest.raises(ValueError):
            drift.assert_brier_plausible(1.7, n_classes=5)

    def test_beats_no_skill_true_when_below_baseline(self):
        result = drift.assert_brier_plausible(0.10, n_classes=5)
        assert result["beats_no_skill"] is True


# ── Turnover ─────────────────────────────────────────────────────────────────

class TestAssertTurnoverPlausible:
    def test_in_band_passes(self):
        assert drift.assert_turnover_plausible(0.0734) is None

    def test_out_of_band_raises(self):
        with pytest.raises(ValueError):
            drift.assert_turnover_plausible(2.5)

    def test_above_operational_ceiling_warns(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING):
            drift.assert_turnover_plausible(0.5)
        assert any("operational ceiling" in r.message for r in caplog.records)


# ── CVaR ─────────────────────────────────────────────────────────────────────

class TestAssertCvarPlausible:
    def test_in_band_passes(self):
        assert drift.assert_cvar_plausible(-0.0463) is None

    def test_out_of_band_raises(self):
        with pytest.raises(ValueError):
            drift.assert_cvar_plausible(-0.6)

    def test_outside_operational_range_warns(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING):
            drift.assert_cvar_plausible(-0.30)
        assert any("operational range" in r.message for r in caplog.records)


# ── Regime occupancy ─────────────────────────────────────────────────────────

class TestAssertRegimeOccupancyPlausible:
    def test_sums_to_one_returns_below_warn_states(self):
        occupancy = pd.Series({0: 0.016, 1: 0.14, 2: 0.319, 3: 0.406, 4: 0.119})
        below = drift.assert_regime_occupancy_plausible(occupancy)
        assert below == [0]

    def test_sums_to_point_eight_raises(self):
        occupancy = {0: 0.2, 1: 0.3, 2: 0.3}
        with pytest.raises(ValueError, match="sum"):
            drift.assert_regime_occupancy_plausible(occupancy)

    def test_share_outside_unit_interval_raises(self):
        with pytest.raises(ValueError):
            drift.assert_regime_occupancy_plausible({0: 1.5, 1: -0.5})


# ── Portfolio weights ────────────────────────────────────────────────────────

class TestAssertPortfolioWeightsPlausible:
    def test_valid_weights_pass(self):
        weights = pd.Series({"spy": 0.4, "tlt": 0.3, "gld": 0.2})
        assert drift.assert_portfolio_weights_plausible(weights, cash_weight=0.1) is None

    def test_negative_weight_raises(self):
        weights = pd.Series({"spy": -0.1, "tlt": 1.1})
        with pytest.raises(ValueError, match="long-only"):
            drift.assert_portfolio_weights_plausible(weights)

    def test_total_off_one_raises(self):
        weights = pd.Series({"spy": 0.5, "tlt": 0.3})
        with pytest.raises(ValueError, match="sum"):
            drift.assert_portfolio_weights_plausible(weights, cash_weight=0.0)


# ── Level discontinuity ──────────────────────────────────────────────────────

class TestAssertNoLevelDiscontinuity:
    def test_smooth_series_returns_empty(self):
        idx = pd.date_range("2000-01-31", periods=24, freq="ME")
        series = pd.Series(np.linspace(100, 110, 24), index=idx)
        assert drift.assert_no_level_discontinuity(series, name="smooth") == []

    def test_tripling_series_raises_naming_date(self):
        idx = pd.date_range("2000-01-31", periods=5, freq="ME")
        series = pd.Series([100.0, 101.0, 303.0, 305.0, 306.0], index=idx)
        with pytest.raises(ValueError, match="2000-03-31"):
            drift.assert_no_level_discontinuity(series, name="fred_cpi")

    def test_short_series_returns_empty(self):
        series = pd.Series([1.0], index=pd.date_range("2000-01-31", periods=1, freq="ME"))
        assert drift.assert_no_level_discontinuity(series, name="short") == []


# ── KPI table (Amendment 1 item A: both historical regressions together) ────

class TestAssertKpiTablePlausible:
    def test_live_kpi_table_passes_and_returns_verdict_frame(self):
        kpi_table = pd.DataFrame(
            {
                "leg": ["strategy", "no_regime_ablation", "spy_buy_hold", "sixty_forty", "faber_sma"],
                "terminal_log_wealth": [4.026505, 3.647238, 5.680455, 5.047073, 6.372592],
                "max_drawdown": [-0.212432, -0.198068, -0.489475, -0.269625, -0.189421],
            }
        )
        verdict = drift.assert_kpi_table_plausible(kpi_table)
        assert set(verdict.columns) == {"leg", "metric", "value", "universal_band", "domain_band", "verdict"}
        assert verdict["leg"].nunique() == 5
        assert (verdict["verdict"] == "pass").all()

    def test_historical_bad_kpi_table_raises_collecting_both_violations(self):
        kpi_table = pd.DataFrame(
            {
                "leg": ["strategy", "sixty_forty"],
                "terminal_log_wealth": [111.06, 5.047073],
                "max_drawdown": [-0.21, -0.0227],
            }
        )
        with pytest.raises(ValueError) as exc_info:
            drift.assert_kpi_table_plausible(kpi_table)
        message = str(exc_info.value)
        assert "111.06" in message or "111.0600" in message
        assert "sixty_forty" in message


# ── Drift (D-09) ─────────────────────────────────────────────────────────────

class TestComputeDrift:
    def test_same_distribution_shows_small_shift_and_no_flag(self):
        rng = np.random.default_rng(0)
        baseline = pd.Series(rng.normal(0, 1, 400))
        current = pd.Series(rng.normal(0, 1, 400))
        result = drift.compute_drift(current, baseline)
        assert abs(result["standardized_mean_shift"]) < 0.5
        assert result["flag"] is False

    def test_three_sigma_shift_is_flagged(self):
        rng = np.random.default_rng(0)
        baseline = pd.Series(rng.normal(0, 1, 400))
        current = pd.Series(rng.normal(0, 1, 400) + 3.0)
        result = drift.compute_drift(current, baseline)
        assert abs(result["standardized_mean_shift"] - 3.0) < 0.5
        assert result["flag"] is True

    def test_empty_inputs_do_not_raise(self):
        result = drift.compute_drift(pd.Series(dtype=float), pd.Series(dtype=float))
        assert result["n_baseline"] == 0
        assert result["n_current"] == 0
        assert np.isnan(result["standardized_mean_shift"])
        assert result["flag"] is False

    def test_zero_baseline_std_does_not_raise(self):
        baseline = pd.Series([1.0, 1.0, 1.0])
        current = pd.Series([1.0, 2.0, 3.0])
        result = drift.compute_drift(current, baseline)
        assert result["baseline_std"] == 0.0
        assert np.isnan(result["standardized_mean_shift"])

    def test_drift_is_scale_invariant_so_a_uniform_units_error_is_invisible(self):
        # The standing lesson (06-VALIDATION.md): drift is computed in
        # standardized units, so a uniform units error (e.g. every value
        # scaled by the same wrong constant, as in the percent-vs-decimal
        # defect that compounded long_duration_tr to 2.3e128) does not
        # change the standardized mean shift at all. Drift alone cannot
        # distinguish "scaled wrong" from "scaled right" — only a
        # plausibility check on the absolute value can.
        rng = np.random.default_rng(3)
        baseline = pd.Series(rng.normal(0, 1, 400))
        current = pd.Series(rng.normal(0.2, 1, 400))
        correct = drift.compute_drift(current, baseline)

        wrong_scale = 100.0
        scaled = drift.compute_drift(current * wrong_scale, baseline * wrong_scale)

        assert scaled["standardized_mean_shift"] == pytest.approx(correct["standardized_mean_shift"])
        assert scaled["flag"] == correct["flag"]


class TestDriftReport:
    def test_sorted_by_descending_absolute_shift(self):
        idx = pd.date_range("1962-01-31", periods=776, freq="ME")
        rng = np.random.default_rng(1)
        df = pd.DataFrame(
            {
                "stable": rng.normal(0, 1, 776),
                "drifted": np.concatenate([rng.normal(0, 1, 708), rng.normal(5, 1, 68)]),
            },
            index=idx,
        )
        report = drift.drift_report(df, ["stable", "drifted"])
        assert list(report["column"]) == ["drifted", "stable"]
        assert report.iloc[0]["flag"] is True or bool(report.iloc[0]["flag"])

    def test_empty_columns_does_not_raise(self):
        idx = pd.date_range("1962-01-31", periods=10, freq="ME")
        df = pd.DataFrame({"x": range(10)}, index=idx)
        report = drift.drift_report(df, [])
        assert isinstance(report, pd.DataFrame)
        assert report.empty

    def test_missing_column_does_not_raise(self):
        idx = pd.date_range("1962-01-31", periods=10, freq="ME")
        df = pd.DataFrame({"x": range(10)}, index=idx)
        report = drift.drift_report(df, ["does_not_exist"])
        assert len(report) == 1
        assert np.isnan(report.iloc[0]["standardized_mean_shift"])


class TestNoMatplotlibImport:
    def test_drift_module_has_no_matplotlib_dependency(self):
        assert "matplotlib" not in vars(drift)
        assert "pyplot" not in vars(drift)
