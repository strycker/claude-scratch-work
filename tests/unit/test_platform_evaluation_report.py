"""Unit tests for trading_crab_lib.platform.evaluation.report (EVAL-01..04
capstone). Synthetic precomputed inputs only — no network, no real
checkpoints, no real backtest run (the full wiring is exercised end-to-end by
tests/integration/test_mini_backtest.py instead).

Three behaviors under test (Task 1, RED):

- ``TestHeadlineOrdering``: ``assemble_backtest_report`` renders the
  sojourn/detection-lag ratio (D-01a) as the FIRST metrics section, before
  the Faber comparison and the no-regime-ablation delta.
- ``TestBaselineGauntletSection``: the markdown names SPY, 60/40, and Faber
  with both log wealth and max drawdown, plus a no-regime-ablation delta
  line.
- ``TestArtifactsWritten``: ``write_backtest_report`` writes the markdown to
  a tmp path and the KPI/equity-curve parquet artifacts round-trip via
  ``read_parquet``.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from trading_crab_lib.platform.assets.returns import returns_by_regime_stats
from trading_crab_lib.platform.evaluation import report
from trading_crab_lib.platform.evaluation.sojourn_lag import compute_sojourn_lag_headline


def _sojourn_lag() -> dict:
    return {"median_sojourn": 18.0, "median_lag": 2.0, "ratio": 9.0}


def _strategy_kpis() -> dict:
    return {
        "terminal_log_wealth": 0.55,
        "max_drawdown": -0.12,
        "duration_months": 8,
        "cvar": -0.08,
        "turnover": 0.15,
        "crisis_capture": {"2008-09_gfc": 0.6},
    }


def _ablation_kpis() -> dict:
    return {"terminal_log_wealth": 0.40, "max_drawdown": -0.18}


def _baseline_kpis() -> dict:
    return {
        "spy_buy_hold": {"terminal_log_wealth": 0.45, "max_drawdown": -0.30},
        "sixty_forty": {"terminal_log_wealth": 0.35, "max_drawdown": -0.15},
        "faber_sma": {"terminal_log_wealth": 0.42, "max_drawdown": -0.10},
    }


def _assemble() -> str:
    return report.assemble_backtest_report(
        sojourn_lag=_sojourn_lag(),
        strategy_kpis=_strategy_kpis(),
        ablation_kpis=_ablation_kpis(),
        baseline_kpis=_baseline_kpis(),
        gap=0.03,
    )


# ── TestHeadlineOrdering (D-01a) ──────────────────────────────────────────────


class TestHeadlineOrdering:
    def test_sojourn_lag_ratio_is_the_first_metrics_section(self):
        """The FIRST '## ' section header must be the sojourn/detection-lag
        headline — before the Faber comparison and the ablation delta
        (design §5.4, D-01a: this is the go/no-go number the report shows
        FIRST)."""
        markdown = _assemble()
        headers = [line for line in markdown.splitlines() if line.startswith("## ")]
        assert headers, "no '## ' section headers found in the assembled markdown"

        headline_idx = next(
            i for i, h in enumerate(headers) if "sojourn" in h.lower() or "detection-lag" in h.lower()
        )
        faber_idx = next(i for i, h in enumerate(headers) if "faber" in h.lower())
        ablation_idx = next(i for i, h in enumerate(headers) if "ablation" in h.lower())

        assert headline_idx == 0, (
            f"sojourn/detection-lag headline must be the FIRST metrics section, "
            f"found at index {headline_idx} of {headers}"
        )
        assert headline_idx < faber_idx
        assert headline_idx < ablation_idx

    def test_headline_section_reports_median_sojourn_lag_and_ratio(self):
        markdown = _assemble()
        assert "18.0" in markdown  # median_sojourn
        assert "2.0" in markdown  # median_lag
        assert "9.0" in markdown  # ratio


# ── TestBaselineGauntletSection ───────────────────────────────────────────────


class TestBaselineGauntletSection:
    def test_gauntlet_names_all_three_baselines_with_wealth_and_drawdown(self):
        markdown = _assemble()
        assert "SPY" in markdown
        assert "60/40" in markdown
        assert "Faber" in markdown
        # log wealth AND max drawdown values for each baseline appear.
        assert "0.4500" in markdown  # spy_buy_hold terminal log wealth
        assert "-30.00%" in markdown  # spy_buy_hold max drawdown
        assert "0.3500" in markdown  # sixty_forty terminal log wealth
        assert "-15.00%" in markdown  # sixty_forty max drawdown
        assert "0.4200" in markdown  # faber_sma terminal log wealth
        assert "-10.00%" in markdown  # faber_sma max drawdown

    def test_no_regime_ablation_delta_line_present(self):
        markdown = _assemble()
        assert "ablation" in markdown.lower()
        assert "pay rent" in markdown.lower() or "does the regime layer" in markdown.lower()
        # the delta itself (strategy 0.55 - ablation 0.40 = +0.15) is surfaced.
        assert "+0.1500" in markdown or "0.1500" in markdown


# ── TestArtifactsWritten ───────────────────────────────────────────────────────


class TestArtifactsWritten:
    def test_markdown_and_parquet_artifacts_round_trip(self, tmp_path):
        equity_curve = pd.DataFrame(
            {
                "return": [0.01, -0.02],
                "turnover": [0.1, 0.05],
                "cost": [0.001, 0.0005],
                "active_regime": [0, 1],
                "scale": [0.8, 0.6],
                "degraded": [False, False],
            },
            index=pd.date_range("2000-01-31", periods=2, freq="ME"),
        )
        kpi_table = pd.DataFrame(
            [
                {"leg": "strategy", "terminal_log_wealth": 0.55, "max_drawdown": -0.12},
                {"leg": "spy_buy_hold", "terminal_log_wealth": 0.45, "max_drawdown": -0.30},
            ]
        )
        markdown = "# Honest Backtest Report\n\n## Headline\n\nsome content\n"

        report_path = report.write_backtest_report(
            markdown,
            {"equity_curve_strategy": equity_curve, "kpi_table": kpi_table},
            output_dir=tmp_path,
        )

        assert report_path.exists()
        assert report_path.read_text(encoding="utf-8") == markdown

        roundtrip_equity = pd.read_parquet(tmp_path / "backtest_equity_curve_strategy.parquet")
        assert len(roundtrip_equity) == len(equity_curve)
        assert "return" in roundtrip_equity.columns
        assert "turnover" in roundtrip_equity.columns

        roundtrip_kpi = pd.read_parquet(tmp_path / "backtest_kpi_table.parquet")
        assert len(roundtrip_kpi) == 2
        assert set(roundtrip_kpi["leg"]) == {"strategy", "spy_buy_hold"}

    def test_creates_parent_dirs(self, tmp_path):
        nested = tmp_path / "reports" / "platform"
        report_path = report.write_backtest_report("# Report\n", {}, output_dir=nested)
        assert report_path.exists()


class TestNewLabelingArtifacts:
    """Amendment 3 item H — the additive persistence of `full_sample_states`
    and `filtered_state_probs` at the SAME `write_backtest_report` call site.
    Exercises `write_backtest_report` directly with a five-key artifacts
    dict shaped like the new caller-side dict (06-02-PLAN.md Task 1)."""

    def _five_key_artifacts(self):
        equity_curve = pd.DataFrame(
            {"return": [0.01, -0.02], "turnover": [0.1, 0.05]},
            index=pd.date_range("2000-01-31", periods=2, freq="ME"),
        )
        ablation_curve = pd.DataFrame(
            {"return": [0.005, -0.01], "turnover": [0.08, 0.04]},
            index=pd.date_range("2000-01-31", periods=2, freq="ME"),
        )
        kpi_table = pd.DataFrame(
            [{"leg": "strategy", "terminal_log_wealth": 0.55, "max_drawdown": -0.12}]
        )
        states_idx = pd.date_range("1963-01-31", periods=5, freq="ME")
        states_df = pd.DataFrame({"state": [0, 0, 1, 1, 2]}, index=states_idx)

        probs_idx = pd.date_range("1972-01-31", periods=4, freq="ME")
        raw = np.array(
            [
                [0.7, 0.1, 0.1, 0.05, 0.05],
                [0.2, 0.6, 0.1, 0.05, 0.05],
                [0.1, 0.1, 0.7, 0.05, 0.05],
                [0.05, 0.05, 0.1, 0.7, 0.10],
            ]
        )
        probs_df = pd.DataFrame(
            raw, index=probs_idx, columns=[f"state_{k}" for k in range(5)]
        )
        return {
            "equity_curve_strategy": equity_curve,
            "equity_curve_ablation": ablation_curve,
            "kpi_table": kpi_table,
            "full_sample_states": states_df,
            "filtered_state_probs": probs_df,
        }, states_df, probs_df

    def test_all_five_parquet_files_written_with_backtest_prefix(self, tmp_path):
        artifacts, _states_df, _probs_df = self._five_key_artifacts()
        report.write_backtest_report("# report\n", artifacts, output_dir=tmp_path)

        for name in artifacts:
            assert (tmp_path / f"backtest_{name}.parquet").exists()

    def test_states_frame_round_trips_with_datetime_index_and_int_states(self, tmp_path):
        artifacts, states_df, _probs_df = self._five_key_artifacts()
        report.write_backtest_report("# report\n", artifacts, output_dir=tmp_path)

        roundtrip = pd.read_parquet(tmp_path / "backtest_full_sample_states.parquet")
        assert isinstance(roundtrip.index, pd.DatetimeIndex)
        assert list(roundtrip.index) == list(states_df.index)
        assert list(roundtrip["state"].astype(int)) == list(states_df["state"])

    def test_probability_frame_round_trips_with_state_k_columns_summing_to_one(self, tmp_path):
        artifacts, _states_df, probs_df = self._five_key_artifacts()
        report.write_backtest_report("# report\n", artifacts, output_dir=tmp_path)

        roundtrip = pd.read_parquet(tmp_path / "backtest_filtered_state_probs.parquet")
        assert list(roundtrip.columns) == [f"state_{k}" for k in range(5)]
        assert np.allclose(roundtrip.sum(axis=1).to_numpy(), 1.0, atol=1e-9)
        assert float(roundtrip.to_numpy().min()) >= 0.0
        assert float(roundtrip.to_numpy().max()) <= 1.0
        pd.testing.assert_frame_equal(roundtrip, probs_df, check_dtype=False, check_freq=False)

    def test_round_tripped_pair_feeds_compute_sojourn_lag_headline(self, tmp_path):
        """The round trip P6 depends on: rename `state_{k}` columns back to
        integers, then feed the pair to `compute_sojourn_lag_headline`."""
        artifacts, _states_df, _probs_df = self._five_key_artifacts()
        report.write_backtest_report("# report\n", artifacts, output_dir=tmp_path)

        states_roundtrip = pd.read_parquet(tmp_path / "backtest_full_sample_states.parquet")
        probs_roundtrip = pd.read_parquet(tmp_path / "backtest_filtered_state_probs.parquet")

        states_series = states_roundtrip["state"].astype(int)
        rename_map = {
            col: int(col.rsplit("_", 1)[-1])
            for col in probs_roundtrip.columns
            if isinstance(col, str) and col.startswith("state_")
        }
        probs_int_cols = probs_roundtrip.rename(columns=rename_map)

        headline = compute_sojourn_lag_headline(states_series, probs_int_cols, act_threshold=0.70)

        for key in ("median_sojourn", "median_lag", "ratio", "n_transitions", "n_resolved", "act_threshold"):
            assert key in headline
        assert isinstance(headline["n_resolved"], int)
        assert 0 <= headline["n_resolved"] <= headline["n_transitions"]
        assert np.isnan(headline["ratio"]) or np.isfinite(headline["ratio"])


class TestExcludedAssets:
    def test_excluded_assets_flagged_in_report(self):
        md = report.assemble_backtest_report(
            sojourn_lag=_sojourn_lag(),
            strategy_kpis=_strategy_kpis(),
            ablation_kpis=_ablation_kpis(),
            baseline_kpis=_baseline_kpis(),
            gap=0.03,
            excluded_assets=["gold"],
        )
        assert "Excluded assets" in md
        assert "gold" in md

    def test_no_excluded_note_when_none(self):
        assert "Excluded assets" not in _assemble()


class TestHeadlineSampleTransparency:
    """The headline surfaces its resolved/total transition counts and warns
    when the resolved sample is too small to trust as a go/no-go number
    (the real long-history run resolves only ~2 of ~6 transitions)."""

    def test_reports_resolved_and_total_counts(self):
        md = report.assemble_backtest_report(
            sojourn_lag={
                "median_sojourn": 84.5,
                "median_lag": 161.5,
                "ratio": 0.52,
                "n_transitions": 6,
                "n_resolved": 2,
                "act_threshold": 0.70,
            },
            strategy_kpis=_strategy_kpis(),
            ablation_kpis=_ablation_kpis(),
            baseline_kpis=_baseline_kpis(),
            gap=0.03,
        )
        assert "2 resolved of 6 transitions" in md
        assert "70%" in md  # action threshold rendered as a percentage
        assert "Small sample" in md

    def test_no_small_sample_warning_when_ample(self):
        md = report.assemble_backtest_report(
            sojourn_lag={
                "median_sojourn": 18.0,
                "median_lag": 2.0,
                "ratio": 9.0,
                "n_transitions": 20,
                "n_resolved": 15,
                "act_threshold": 0.70,
            },
            strategy_kpis=_strategy_kpis(),
            ablation_kpis=_ablation_kpis(),
            baseline_kpis=_baseline_kpis(),
            gap=0.03,
        )
        assert "15 resolved of 20 transitions" in md
        assert "Small sample" not in md

    def test_counts_omitted_when_absent(self):
        # Legacy callers pass a 3-key sojourn_lag (no counts) — the render must
        # not crash and must not invent a sample line.
        assert "resolved of" not in _assemble()


class TestReferenceLabelColumns:
    """The full-sample smoothed reference must span EVERY decision date (the
    walk-forward now labels pre-1990 under approach ii), so it keeps long-history
    columns and drops structural late-starts."""

    def test_drops_late_start_keeps_warmup_and_complete(self):
        idx = pd.date_range("1962-01-31", periods=240, freq="ME")
        first_decision = idx[120]  # ~1972, like min_train=120
        df = pd.DataFrame(index=idx)
        df["complete"] = np.arange(240, dtype=float)
        df["warmup_only"] = np.arange(240, dtype=float)
        df.iloc[:3, df.columns.get_loc("warmup_only")] = np.nan       # NaN only pre-1962Q1 (< first_decision)
        df["late_start"] = np.arange(240, dtype=float)
        df.iloc[:180, df.columns.get_loc("late_start")] = np.nan       # NaN through ~1977 (> first_decision)

        ref = report._reference_label_columns(df, list(df.columns), first_decision)

        assert "complete" in ref        # present across decision range
        assert "warmup_only" in ref     # NaN only before the first decision → kept
        assert "late_start" not in ref  # NaN within the decision range → dropped

    def test_selected_columns_cover_all_decision_dates(self):
        idx = pd.date_range("1962-01-31", periods=240, freq="ME")
        first_decision = idx[120]
        df = pd.DataFrame(
            {"a": np.arange(240, dtype=float), "b": np.arange(240, dtype=float)}, index=idx
        )
        df.iloc[:3, df.columns.get_loc("a")] = np.nan  # warmup only
        ref = report._reference_label_columns(df, list(df.columns), first_decision)
        # dropna over the selected cols must retain every decision-date row
        covered = df[ref].dropna().index
        assert covered.max() == idx.max()
        assert (pd.DatetimeIndex(idx[120:]).isin(covered)).all()



# ── _smoothed_hindsight_perf: hindsight in the LABELS only, not the universe ─


class TestSmoothedHindsightUniverse:
    """The smoothed oracle is deliberately non-causal about regime LABELS.
    It must not also be non-causal about which assets EXIST: tilting into a
    ticker that had not been issued yet is a second, undocumented leak, and
    it crashed the run outright (IAU/USO had 0 observations at the traced
    1974-10-31 decision date, so the per-asset EWMA fallback indexed an empty
    series).
    """

    N_MONTHS = 40
    LATE_START = 30  # 'LATE' has no observations before this position

    def _frame(self, *, include_late: bool) -> pd.DataFrame:
        """Deliberately deterministic, not seeded RNG.

        Both assets must have a strictly POSITIVE full-sample Sharpe. A random
        draw can easily hand SPY a negative one, and ``_per_regime_tilt`` clips
        negative Sharpes to zero — which would zero out BOTH legs and let the
        comparison below pass vacuously, for a reason unrelated to the universe
        restriction under test.
        """
        idx = pd.date_range("2000-01-31", periods=self.N_MONTHS, freq="ME")
        # mean 0.01, non-zero std => Sharpe comfortably positive.
        data = {"SPY": np.tile([0.02, 0.0], self.N_MONTHS // 2)}
        if include_late:
            late = np.full(self.N_MONTHS, np.nan)
            # Near-riskless and far higher mean => dominant full-sample Sharpe,
            # so the oracle WILL tilt into it if the universe is not restricted.
            n_late = self.N_MONTHS - self.LATE_START
            late[self.LATE_START:] = np.tile([0.08, 0.079], n_late // 2)
            data["LATE"] = late
        return pd.DataFrame(data, index=idx)

    def test_fixture_gives_both_assets_positive_sharpe(self):
        """Guards the fixture itself: if SPY's Sharpe were <= 0 it would be
        clipped to zero weight and the comparison test would pass for the
        wrong reason."""
        frame = self._frame(include_late=True)
        stats = returns_by_regime_stats(frame, pd.Series(0, index=frame.index))
        sharpes = stats.set_index("asset")["sharpe_annualized"]
        assert sharpes["SPY"] > 0
        assert sharpes["LATE"] > sharpes["SPY"]

    def _args(self, frame: pd.DataFrame) -> tuple:
        states = pd.Series(0, index=frame.index)
        cash_ret = pd.Series(0.0, index=frame.index)
        # Every decision date precedes LATE's inception.
        decision_dates = list(frame.index[12:25])
        return states, cash_ret, decision_dates, {"portfolio_vol_min_obs": 12, "ewma_halflife_months": 6}

    def test_no_weight_given_to_an_asset_that_has_not_started(self):
        frame = self._frame(include_late=True)
        states, cash_ret, decision_dates, alloc_cfg = self._args(frame)

        seen_weights: list[pd.Series] = []
        real_tilt = report.vol_targeted_tilt

        def capturing_tilt(*args, **kwargs):
            result = real_tilt(*args, **kwargs)
            seen_weights.append(result["weights"])
            return result

        with patch.object(report, "vol_targeted_tilt", capturing_tilt):
            report._smoothed_hindsight_perf(states, frame, cash_ret, decision_dates, alloc_cfg)

        assert seen_weights, "the oracle never called the allocator"
        for weights in seen_weights:
            assert float(weights.get("LATE", 0.0)) == 0.0

    def test_result_unchanged_by_the_presence_of_a_not_yet_started_column(self):
        """Merely guarding the empty-series crash is not enough: with only that
        guard, LATE still draws full-sample stats, wins the Sharpe ranking, and
        takes weight that then earns nothing (its return is NaN) — so the two
        results diverge. They may only agree if the universe itself is
        restricted to assets that have started by the decision date."""
        with_late = self._frame(include_late=True)
        without_late = self._frame(include_late=False)
        states, cash_ret, decision_dates, alloc_cfg = self._args(with_late)

        perf_with = report._smoothed_hindsight_perf(
            states, with_late, cash_ret, decision_dates, alloc_cfg
        )
        perf_without = report._smoothed_hindsight_perf(
            states, without_late, cash_ret, decision_dates, alloc_cfg
        )

        assert perf_with == pytest.approx(perf_without)


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-q"])
