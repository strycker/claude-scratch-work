"""
Tests for the L4-02 weekly report assembly + trades-implied + opt-in email
delivery (04-CONTEXT.md D-02: markdown ALWAYS written, email opt-in behind
--send-email, reusing the incumbent email.py machinery read-only).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_crab_lib.platform.report import weekly

# ── trades_implied: signal rules against the flat no-trade band ─────────────


class TestTradesImplied:
    def test_hold_within_threshold(self):
        target = pd.Series({"SPY": 0.40})
        current = pd.Series({"SPY": 0.41})
        result = weekly.trades_implied(target, current, threshold=0.03)
        row = result.set_index("asset").loc["SPY"]
        assert row["signal"] == "HOLD"
        assert row["current_pct"] == pytest.approx(0.41)
        assert row["target_pct"] == pytest.approx(0.40)
        assert row["delta_pct"] == pytest.approx(-0.01)

    def test_buy_when_delta_at_or_above_threshold(self):
        target = pd.Series({"SPY": 0.45})
        current = pd.Series({"SPY": 0.40})
        result = weekly.trades_implied(target, current, threshold=0.03)
        assert result.set_index("asset").loc["SPY", "signal"] == "BUY"

    def test_sell_when_delta_at_or_below_negative_threshold(self):
        target = pd.Series({"SPY": 0.30})
        current = pd.Series({"SPY": 0.40})
        result = weekly.trades_implied(target, current, threshold=0.03)
        assert result.set_index("asset").loc["SPY", "signal"] == "SELL"

    def test_asset_in_target_not_in_current_is_buy_above_threshold(self):
        """delta = target - 0 -> BUY when above threshold."""
        target = pd.Series({"GLD": 0.10})
        current = pd.Series({"SPY": 0.50})
        result = weekly.trades_implied(target, current, threshold=0.03).set_index("asset")
        assert result.loc["GLD", "signal"] == "BUY"
        assert result.loc["GLD", "current_pct"] == pytest.approx(0.0)
        assert result.loc["GLD", "delta_pct"] == pytest.approx(0.10)

    def test_asset_held_not_targeted_is_sell_below_threshold(self):
        """delta = 0 - current -> SELL when below -threshold."""
        target = pd.Series({"SPY": 0.50})
        current = pd.Series({"TLT": 0.20, "SPY": 0.50})
        result = weekly.trades_implied(target, current, threshold=0.03).set_index("asset")
        assert result.loc["TLT", "signal"] == "SELL"
        assert result.loc["TLT", "target_pct"] == pytest.approx(0.0)
        assert result.loc["TLT", "delta_pct"] == pytest.approx(-0.20)

    def test_one_row_per_unioned_asset(self):
        target = pd.Series({"SPY": 0.5, "GLD": 0.1})
        current = pd.Series({"SPY": 0.5, "TLT": 0.2})
        result = weekly.trades_implied(target, current, threshold=0.03)
        assert set(result["asset"]) == {"SPY", "GLD", "TLT"}
        assert list(result.columns) == ["asset", "current_pct", "target_pct", "delta_pct", "signal"]


# ── assemble_weekly_report: markdown flags low-confidence returns cells ──────


class TestAssembleWeeklyReport:
    def _returns_by_regime(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "regime": 0, "asset": "SPY", "mean_monthly_return": 0.01,
                    "std_monthly_return": 0.03, "sharpe_annualized": 1.2,
                    "hit_rate": 0.6, "max_drawdown": -0.1, "n_obs": 30,
                },
                {
                    "regime": 0, "asset": "GLD", "mean_monthly_return": 0.005,
                    "std_monthly_return": 0.02, "sharpe_annualized": 0.3,
                    "hit_rate": 0.5, "max_drawdown": -0.05, "n_obs": 3,
                },
            ]
        )

    def test_flags_low_confidence_cells_below_min_obs_flag(self, tmp_path):
        (tmp_path / "acct1.yaml").write_text("weights:\n  SPY: 0.5\ncash: 0.5\n", encoding="utf-8")
        markdown = weekly.assemble_weekly_report(
            regime_probs={0: 0.8, 1: 0.2},
            transition_matrix=pd.DataFrame({0: [0.7, 0.3], 1: [0.4, 0.6]}, index=[0, 1]),
            returns_by_regime=self._returns_by_regime(),
            target_weights=pd.Series({"SPY": 0.5}),
            accounts=["acct1"],
            active_regime=0,
            accounts_dir=tmp_path,
            min_obs_flag=6,
        )
        assert "GLD" in markdown
        assert "LOW-CONFIDENCE" in markdown
        # SPY has n_obs=30 >= 6 -> not flagged low-confidence.
        spy_line = next(line for line in markdown.splitlines() if "SPY" in line and "mean=" in line)
        assert "LOW-CONFIDENCE" not in spy_line

    def test_includes_per_account_trades_implied(self, tmp_path):
        (tmp_path / "acct1.yaml").write_text("weights:\n  SPY: 0.2\ncash: 0.8\n", encoding="utf-8")
        markdown = weekly.assemble_weekly_report(
            regime_probs={0: 1.0},
            transition_matrix=pd.DataFrame({0: [1.0]}, index=[0]),
            returns_by_regime=self._returns_by_regime(),
            target_weights=pd.Series({"SPY": 0.5}),
            accounts=["acct1"],
            active_regime=0,
            accounts_dir=tmp_path,
        )
        assert "acct1" in markdown
        assert "BUY" in markdown  # SPY target 0.5 vs current 0.2 -> BUY


# ── stale_series: per-series staleness against the run date (08.1, A-12) ─────


class TestStaleSeries:
    def _frame(self) -> pd.DataFrame:
        idx = pd.date_range("2021-01-31", "2021-06-30", freq="ME")
        return pd.DataFrame(
            {
                "fresh": 1.0,
                "two_late": [1.0, 1.0, 1.0, 1.0, np.nan, np.nan],
                "never": np.nan,
                "gappy": [np.nan, 1.0, np.nan, 1.0, 1.0, 1.0],  # an interior gap is not staleness
            },
            index=idx,
        )

    def test_names_only_series_whose_last_value_precedes_the_expected_month(self):
        got = weekly.stale_series(
            self._frame(), ["fresh", "two_late", "never", "gappy"],
            run_date=pd.Timestamp("2021-07-08"), grace_days=7,
        )
        assert got == {"two_late": 2, "never": None}

    def test_the_grace_window_moves_the_expected_month(self):
        frame = self._frame().drop(columns=["never"])
        inside = weekly.stale_series(frame, ["two_late"], run_date=pd.Timestamp("2021-07-05"), grace_days=7)
        assert inside == {"two_late": 1}  # expected 05-31, last valid 04-30
        none_needed = weekly.stale_series(frame, ["fresh"], run_date=pd.Timestamp("2021-07-05"), grace_days=7)
        assert none_needed == {}

    def test_a_month_end_run_date_is_not_yet_due_for_that_month(self):
        frame = self._frame()[["fresh"]]
        # 2021-07-07 minus 7 days is 06-30: June is still inside its grace, so expected is 05-31.
        assert weekly.stale_series(frame.iloc[:-1], ["fresh"], run_date=pd.Timestamp("2021-07-07"),
                                   grace_days=7) == {}
        assert weekly.stale_series(frame.iloc[:-1], ["fresh"], run_date=pd.Timestamp("2021-07-08"),
                                   grace_days=7) == {"fresh": 1}

    def test_does_not_mutate_the_frame_and_ignores_columns_not_listed(self):
        frame = self._frame()
        before = frame.copy()
        assert weekly.stale_series(frame, [], run_date=pd.Timestamp("2021-07-08"), grace_days=7) == {}
        pd.testing.assert_frame_equal(frame, before)

    def test_the_banner_lines_render_months_and_never_observed(self):
        md = weekly.assemble_weekly_report(
            regime_probs={0: 1.0}, transition_matrix=pd.DataFrame(), returns_by_regime=pd.DataFrame(),
            target_weights=pd.Series(dtype=float), accounts=[], active_regime=0,
            stale_series={"div_yield": 2, "m2": 1, "gone": None},
            stale_expected_through=pd.Timestamp("2026-07-31"),
        )
        assert md.index("## STALE DATA") < md.index("## Current Regime Distribution")
        assert "- div_yield: 2 months late (last value 2026-05-31, expected through 2026-07-31)" in md
        assert "- m2: 1 month late (last value 2026-06-30, expected through 2026-07-31)" in md
        assert "- gone: no value ever observed (expected through 2026-07-31)" in md
        clean = weekly.assemble_weekly_report(
            regime_probs={0: 1.0}, transition_matrix=pd.DataFrame(), returns_by_regime=pd.DataFrame(),
            target_weights=pd.Series(dtype=float), accounts=[], active_regime=0, stale_series={},
        )
        assert "STALE DATA" not in clean


# ── write_weekly_report: markdown ALWAYS written (D-02) ──────────────────────


class TestWriteWeeklyReport:
    def test_writes_weekly_report_md_at_default_platform_path(self, tmp_path):
        path = weekly.write_weekly_report("# Report\n", output_dir=tmp_path)
        assert path == tmp_path / "weekly_report.md"
        assert path.exists()
        assert path.read_text(encoding="utf-8") == "# Report\n"

    def test_write_weekly_report_creates_parent_dirs(self, tmp_path):
        nested = tmp_path / "reports" / "platform"
        path = weekly.write_weekly_report("# Report\n", output_dir=nested)
        assert path.exists()


# ── main(): opt-in email path (D-02) ─────────────────────────────────────────


class TestMainOptInEmail:
    def _patch_pipeline(self, monkeypatch, tmp_path):
        monkeypatch.setattr(weekly, "OUTPUT_DIR", tmp_path)
        monkeypatch.setattr(weekly, "load_platform_config", lambda: {"report": {}, "universe": {}})
        monkeypatch.setattr(
            weekly,
            "_build_report_inputs",
            lambda cfg, cm=None: {
                "regime_probs": pd.Series({0: 1.0}),
                "active_regime": 0,
                "transition_matrix": pd.DataFrame({0: [1.0]}, index=[0]),
                "returns_by_regime": pd.DataFrame(
                    columns=[
                        "regime", "asset", "mean_monthly_return", "std_monthly_return",
                        "sharpe_annualized", "hit_rate", "max_drawdown", "n_obs",
                    ]
                ),
                "target_weights": pd.Series(dtype=float),
                "cash": 1.0,
            },
        )

    def test_main_no_args_writes_markdown_and_skips_email(self, monkeypatch, tmp_path):
        self._patch_pipeline(monkeypatch, tmp_path)
        send_calls = []
        monkeypatch.setattr(weekly, "send_weekly_email", lambda *a, **k: send_calls.append((a, k)))

        exit_code = weekly.main([])

        assert exit_code == 0
        assert (tmp_path / "reports" / "platform" / "weekly_report.md").exists()
        assert send_calls == []

    def test_main_send_email_calls_email_pipeline_once(self, monkeypatch, tmp_path):
        self._patch_pipeline(monkeypatch, tmp_path)
        build_calls = []
        load_cfg_calls = []
        send_calls = []
        monkeypatch.setattr(
            weekly,
            "build_weekly_email_body",
            lambda report_dir, subject_prefix=None: build_calls.append((report_dir, subject_prefix))
            or ("subject", "body"),
        )
        monkeypatch.setattr(
            weekly,
            "load_email_config",
            lambda: load_cfg_calls.append(True) or {"smtp_host": "x"},
        )
        monkeypatch.setattr(
            weekly,
            "send_weekly_email",
            lambda cfg, subject, body, **k: send_calls.append((cfg, subject, body)),
        )

        exit_code = weekly.main(["--send-email"])

        assert exit_code == 0
        assert len(build_calls) == 1
        assert len(load_cfg_calls) == 1
        assert len(send_calls) == 1


# ── security / reuse conventions ──────────────────────────────────────────────


class TestReuseConventions:
    def test_imports_email_functions_from_incumbent_module(self):
        import inspect

        source = inspect.getsource(weekly)
        assert "from trading_crab_lib.email import" in source
        assert "build_weekly_email_body" in source
        assert "load_email_config" in source
        assert "send_weekly_email" in source

    def test_does_not_import_incumbent_reporting_or_load_portfolio(self):
        import inspect

        source = inspect.getsource(weekly)
        assert "trading_crab_lib.reporting" not in source
        assert "import load_portfolio" not in source
        assert "from trading_crab_lib.config import" not in source


# ── the Bayes filter at serve (plan 08-08) ──────────────────────────────────


class _FakeNowcaster:
    classes_ = [0, 1, 2]
    feature_names_in_ = np.array(["x"])

    def predict_proba(self, row):
        return [[0.30, 0.45, 0.25]]


_SERVED_PRIOR = [0.2, 0.3, 0.5]


#: Sentinel for ``_serve_env(served_prior=...)``: a served training prior equal to the fixture
#: labels' own distribution (a served model whose training block had the label frequencies).
_LABEL_DISTRIBUTION = "label_distribution"


def _serve_env(monkeypatch, tmp_path, *, as_of: str = "2026-08-31", served_prior=None):
    """A real CheckpointManager in tmp_path plus in-memory inputs for _build_report_inputs.

    ``regime_labels`` is deliberately NON-uniform (state 0 dominates), so the filtered
    belief cannot coincide with the raw posterior by accident of a flat prior.

    ``served_prior`` is the ``nowcaster_class_prior`` artifact (CR-01): by default
    ``_SERVED_PRIOR``, deliberately NOT the label prior. ``_LABEL_DISTRIBUTION`` serves the
    labels' own distribution instead: used only by the 08-09 band/hysteresis scenario tests,
    whose constants were derived on that belief and which test the band, not the prior.
    """
    from trading_crab_lib.checkpoints import CheckpointManager

    cm = CheckpointManager(checkpoint_dir=tmp_path / "cp")
    idx = pd.date_range("2020-01-31", as_of, freq="ME")
    labels = pd.Series(([0] * 12 + [1] * 4 + [2] * 3) * (len(idx) // 19 + 1), dtype=int).iloc[: len(idx)]
    labels.index = idx
    frames = {
        "regime_labels": pd.DataFrame({"state": labels}),
        "returns_by_regime": pd.DataFrame(
            {
                "regime": [0, 0, 1, 1, 2, 2],
                "asset": ["SPY", "TLT"] * 3,
                "mean_monthly_return": [0.01, 0.0, -0.01, 0.01, 0.0, 0.005],
                "sharpe_annualized": [1.0, 0.2, -0.5, 0.8, 0.1, 0.6],
                "n_obs": [30] * 6,
            }
        ),
        "asset_returns": pd.DataFrame({"SPY": [0.01, -0.02] * 20, "TLT": [0.0, 0.01] * 20},
                                      index=pd.date_range("2023-01-31", periods=40, freq="ME")),
        # CR-01: the served model's training prior, built beside it. Deliberately NOT the
        # label prior (about .63 / .21 / .16), so dividing by the wrong one is visible.
        "nowcaster_class_prior": pd.DataFrame({"state": [0, 1, 2], "prior": _SERVED_PRIOR}),
    }
    if served_prior is not None:
        from trading_crab_lib.platform.prediction.regime_filter import unconditional_belief

        values = (
            list(unconditional_belief(labels, state_index=[0, 1, 2]))
            if served_prior == _LABEL_DISTRIBUTION else list(served_prior)
        )
        frames["nowcaster_class_prior"] = pd.DataFrame({"state": [0, 1, 2], "prior": values})
    real_load = cm.load

    def load(name):
        return frames[name] if name in frames else real_load(name)

    monkeypatch.setattr(cm, "load", load)
    monkeypatch.setattr(cm, "load_model", lambda name: _FakeNowcaster())
    monkeypatch.setattr(weekly, "load_full_span", lambda name: pd.DataFrame({"x": range(len(idx))}, index=idx))
    cfg = {"labeling": {"K": 3}, "allocation": {"ewma_halflife_months": 6, "portfolio_vol_min_obs": 3}}
    return cm, cfg, labels


class TestRegimeBeliefAtServe:
    def test_one_cold_start_rule_across_all_three_modules(self):
        """Asserted on the RESOLVED FUNCTION OBJECT, not by grep — a grep passes on a
        copied implementation. A divergent cold start is the only way train/serve skew
        can enter this design."""
        import trading_crab_lib.platform.backtest.driver as d
        import trading_crab_lib.platform.backtest.joint_driver as j
        from trading_crab_lib.platform.prediction import regime_filter

        for module in (d, j, weekly):
            assert getattr(module, "unconditional_belief", None) is regime_filter.unconditional_belief, module.__name__

    def test_cold_start_calls_the_shared_helper_and_the_tilt_gets_the_belief(self, monkeypatch, tmp_path):
        cm, cfg, labels = _serve_env(monkeypatch, tmp_path)
        cold, tilt, hyst = [], [], []
        real_ub, real_tilt, real_hyst = weekly.unconditional_belief, weekly.vol_targeted_tilt, weekly.update_active_regime
        monkeypatch.setattr(weekly, "unconditional_belief", lambda *a, **k: cold.append(a) or real_ub(*a, **k))
        monkeypatch.setattr(weekly, "vol_targeted_tilt", lambda p, *a, **k: tilt.append(p) or real_tilt(p, *a, **k))
        monkeypatch.setattr(weekly, "update_active_regime", lambda p, *a, **k: hyst.append(p) or real_hyst(p, *a, **k))

        out = weekly._build_report_inputs(cfg, cm)

        assert len(cold) == 1 and cold[0][0].equals(labels)
        belief, raw = out["regime_belief"], out["regime_probs"]
        from trading_crab_lib.platform.prediction.regime_filter import (
            filter_step,
            transition_matrix_for,
            unconditional_belief,
        )

        start = unconditional_belief(labels, state_index=[0, 1, 2])
        assert not start.round(9).eq(1 / 3).all(), "fixture prior is uniform; it cannot discriminate"
        # CR-01: pi_0 comes from the labels; L_t divides by the SERVED training prior.
        served_prior = pd.Series(_SERVED_PRIOR, index=[0, 1, 2])
        assert (served_prior - start).abs().max() > 0.1, "precondition: the served prior is not the label prior"
        expected = filter_step(start, transition_matrix_for(labels, state_index=[0, 1, 2]), raw, served_prior)
        pd.testing.assert_series_equal(belief, expected)
        assert (belief - raw.reindex(belief.index)).abs().max() > 1e-6, "belief equals the raw posterior"
        assert len(tilt) == 1 and tilt[0] is belief, "vol_targeted_tilt must receive the filtered belief"
        assert len(hyst) == 1 and hyst[0] is belief, "the hysteresis must see this run's belief"

    def test_load_before_save_the_run_filters_from_the_LOADED_belief(self, monkeypatch, tmp_path):
        cm, cfg, labels = _serve_env(monkeypatch, tmp_path, as_of="2026-08-31")
        loaded = pd.Series({0: 0.1, 1: 0.1, 2: 0.8})
        weekly.save_regime_belief(loaded, cm, as_of=pd.Timestamp("2026-07-31"))
        starts = []
        real_filter = weekly.filter_step
        monkeypatch.setattr(weekly, "filter_step", lambda b, *a: starts.append(b.copy()) or real_filter(b, *a))

        out = weekly._build_report_inputs(cfg, cm)

        assert len(starts) == 1
        pd.testing.assert_series_equal(starts[0], loaded, check_names=False)
        saved = weekly.load_regime_belief(cm)
        pd.testing.assert_series_equal(saved, out["regime_belief"], check_names=False)
        assert saved.name == pd.Timestamp("2026-08-31")
        assert not starts[0].equals(saved.rename(None)), "the start must be the loaded belief, not the one just saved"

    def test_a_weekly_rerun_in_the_same_month_does_not_double_count(self, monkeypatch, tmp_path):
        cm, cfg, _ = _serve_env(monkeypatch, tmp_path)
        first = weekly._build_report_inputs(cfg, cm)["regime_belief"]
        calls = []
        real_filter = weekly.filter_step
        monkeypatch.setattr(weekly, "filter_step", lambda *a: calls.append(a) or real_filter(*a))
        second = weekly._build_report_inputs(cfg, cm)["regime_belief"]
        assert calls == []
        pd.testing.assert_series_equal(first, second)

    def test_a_multi_month_gap_advances_by_predict_only_per_missing_month(self, monkeypatch, tmp_path):
        cm, cfg, labels = _serve_env(monkeypatch, tmp_path, as_of="2026-08-31")
        weekly.save_regime_belief(pd.Series({0: 0.1, 1: 0.1, 2: 0.8}), cm, as_of=pd.Timestamp("2026-05-31"))
        calls = []
        real_pos = weekly.predict_only_step
        monkeypatch.setattr(weekly, "predict_only_step", lambda *a: calls.append(a) or real_pos(*a))
        weekly._build_report_inputs(cfg, cm)
        assert len(calls) == 2  # June and July unobserved; August filtered

    def test_persisted_null_and_missing_checkpoint_are_cold_starts(self, tmp_path):
        from trading_crab_lib.checkpoints import CheckpointManager

        cm = CheckpointManager(checkpoint_dir=tmp_path / "cp")
        assert weekly.load_regime_belief(cm) is None
        weekly.save_regime_belief(None, cm)
        assert weekly.load_regime_belief(cm) is None

    def test_the_report_shows_the_belief_beside_the_raw_posterior(self, tmp_path):
        md = weekly.assemble_weekly_report(
            regime_probs={0: 0.5, 1: 0.5},
            regime_belief={0: 0.8, 1: 0.2},
            transition_matrix=pd.DataFrame(),
            returns_by_regime=pd.DataFrame(),
            target_weights=pd.Series(dtype=float),
            accounts=[],
            active_regime=0,
        )
        assert "## Filtered Regime Belief" in md
        assert "regime 0: 80.0%" in md


# ── plan 08-09: the report shows the hysteresis OUTPUT, and the band gates the book ──


def _banded(cfg: dict, band: float | None = 0.05) -> dict:
    out = {**cfg, "allocation": {**cfg["allocation"]}}
    out["allocation"]["no_trade_band"] = band
    return out


class TestActiveRegimeIsTheMachinesOutput:
    def _md(self, active_regime):
        rbr = pd.DataFrame(
            {
                "regime": [0, 1],
                "asset": ["SPY", "TLT"],
                "mean_monthly_return": [0.01, 0.002],
                "sharpe_annualized": [1.1, 0.4],
                "n_obs": [30, 30],
            }
        )
        return weekly.assemble_weekly_report(
            regime_probs={0: 0.55, 1: 0.45},     # argmax is regime 0
            transition_matrix=pd.DataFrame({0: [0.9, 0.2], 1: [0.1, 0.8]}, index=[0, 1]),
            returns_by_regime=rbr,
            target_weights=pd.Series(dtype=float),
            accounts=[],
            active_regime=active_regime,
        )

    def test_the_rendered_regime_is_the_hysteresis_output_not_the_argmax(self):
        """Hysteresis held regime 1 (its own P 0.45 >= unwind 0.40) while 0 is the argmax.
        Fails on the pre-08-09 behaviour, which recomputed probs.idxmax() = 0."""
        md = self._md(1)
        assert "- active regime: regime 1" in md
        assert "From regime 1, next-regime probabilities:" in md
        assert "From regime 0" not in md
        signals = md.split("## Per-Asset Signals")[1].split("##")[0]
        assert "TLT" in signals and "SPY" not in signals   # rows for the ACTIVE regime only

    def test_neutral_posture_is_rendered_as_such(self):
        md = self._md(None)
        assert "- active regime: none (neutral posture)" in md
        assert "(no trajectory available for the current regime)" in md

    def test_the_cold_start_paragraph_sits_directly_under_the_value_it_explains(self):
        lines = self._md(1).splitlines()
        value_at = lines.index("- active regime: regime 1")
        assert lines[value_at + 1] == ""
        assert lines[value_at + 2].startswith("Hysteresis cold-start rule (A1)")
        assert lines.index("## Active Regime (hysteresis state machine output)") == value_at - 2
        assert "gates no weight" in lines[value_at + 4]

    def test_no_internal_argmax_assignment_remains(self):
        import re
        from pathlib import Path

        src = Path(weekly.__file__).read_text()
        code = "\n".join(line for line in src.splitlines() if not line.lstrip().startswith("#"))
        assert not re.findall(r"^\s*active_regime\s*=\s*probs\.idxmax\(\)", code, re.M)

    def test_build_inputs_returns_the_hysteresis_output_and_main_renders_it(self, monkeypatch, tmp_path):
        """End to end on the serve fixture: the belief's max is 0.45 < 0.70, so the machine
        is neutral while the argmax is regime 1 — the markdown must say neutral."""
        cm, cfg, _ = _serve_env(monkeypatch, tmp_path, served_prior=_LABEL_DISTRIBUTION)
        seen = []
        real_hyst = weekly.update_active_regime
        monkeypatch.setattr(weekly, "update_active_regime", lambda *a, **k: seen.append(real_hyst(*a, **k)) or seen[-1])
        inputs = weekly._build_report_inputs(cfg, cm)
        assert inputs["regime_belief"].idxmax() == 1 and inputs["active_regime"] is None
        assert len(seen) == 1 and inputs["active_regime"] is seen[0]

        monkeypatch.setattr(weekly, "OUTPUT_DIR", tmp_path)
        monkeypatch.setattr(weekly, "load_platform_config", lambda: {**cfg, "report": {}})
        monkeypatch.setattr(weekly, "_build_report_inputs", lambda c, cm=None: inputs)
        assert weekly.main([]) == 0
        md = (tmp_path / "reports" / "platform" / "weekly_report.md").read_text(encoding="utf-8")
        assert "- active regime: none (neutral posture)" in md
        assert "From regime 1" not in md


# ── Per-asset rows follow the active regime (plan 08-15, verification human item 4) ──
#
# Before 08-15 a neutral posture (active_regime None) fell through to "no narrowing" and the
# section printed every (regime, asset) row of returns_by_regime with no regime id — on the
# 2026-06-30 real-data page, 24 rows, each asset six times.


def _per_asset_section(md: str) -> str:
    return md.split("## Per-Asset Signals")[1].split("\n## ")[0]


def _asset_rows(section: str) -> list[str]:
    return [line for line in section.splitlines() if line.startswith(("- SPY", "- TLT"))]


class TestPerAssetSignalsFollowTheActiveRegime:
    # Two regimes x two assets, distinct numbers everywhere so a leaked row is visible.
    RBR = pd.DataFrame(
        {
            "regime": [0, 0, 1, 1],
            "asset": ["SPY", "TLT", "SPY", "TLT"],
            "mean_monthly_return": [0.0123, 0.0045, -0.0321, 0.0167],
            "sharpe_annualized": [1.11, 0.44, -2.22, 1.66],
            "n_obs": [30, 30, 30, 30],
        }
    )

    def _md(self, active_regime):
        return weekly.assemble_weekly_report(
            regime_probs={0: 0.55, 1: 0.45},
            transition_matrix=pd.DataFrame({0: [0.9, 0.2], 1: [0.1, 0.8]}, index=[0, 1]),
            returns_by_regime=self.RBR,
            target_weights=pd.Series(dtype=float),
            accounts=[],
            active_regime=active_regime,
        )

    def test_neutral_posture_prints_no_per_asset_row_and_says_why(self):
        section = _per_asset_section(self._md(None))
        assert _asset_rows(section) == []
        sentence = weekly._NEUTRAL_PER_ASSET_SENTENCE
        assert "no regime is active" in sentence and "all regimes" in sentence
        assert sentence in section

    @pytest.mark.parametrize("active_regime", [None, 0, 1])
    def test_no_asset_appears_twice_in_any_posture(self, active_regime):
        rows = _asset_rows(_per_asset_section(self._md(active_regime)))
        for asset in ("SPY", "TLT"):
            assert sum(r.startswith(f"- {asset}") for r in rows) <= 1, rows

    def test_active_posture_rows_are_that_regimes_and_name_it(self):
        section = _per_asset_section(self._md(1))
        assert _asset_rows(section) == [
            "- SPY (regime 1): mean=-3.21% sharpe=-2.22 n_obs=30",
            "- TLT (regime 1): mean=1.67% sharpe=1.66 n_obs=30",
        ]
        for regime0_number in ("1.23%", "0.45%", "sharpe=1.11", "sharpe=0.44"):
            assert regime0_number not in section
        assert weekly._NEUTRAL_PER_ASSET_SENTENCE not in section

    def test_main_writes_the_neutral_sentence_to_the_page(self, monkeypatch, tmp_path):
        """Tracer: the WRITTEN page, via main(), carries the neutral-posture section."""
        monkeypatch.setattr(weekly, "OUTPUT_DIR", tmp_path)
        monkeypatch.setattr(weekly, "load_platform_config", lambda: {"report": {}, "universe": {}})
        monkeypatch.setattr(
            weekly,
            "_build_report_inputs",
            lambda cfg, cm=None: {
                "regime_probs": pd.Series({0: 0.55, 1: 0.45}),
                "active_regime": None,
                "transition_matrix": pd.DataFrame({0: [0.9, 0.2], 1: [0.1, 0.8]}, index=[0, 1]),
                "returns_by_regime": self.RBR,
                "target_weights": pd.Series(dtype=float),
                "cash": 1.0,
            },
        )
        assert weekly.main([]) == 0
        md = (tmp_path / "reports" / "platform" / "weekly_report.md").read_text(encoding="utf-8")
        section = _per_asset_section(md)
        assert _asset_rows(section) == []
        assert weekly._NEUTRAL_PER_ASSET_SENTENCE in section


class TestNoTradeBandAtServe:
    def test_the_band_is_the_shared_helper(self):
        from trading_crab_lib.platform.allocation import hysteresis

        assert weekly.execute_rebalance is hysteresis.execute_rebalance

    def test_first_run_trades_in_full_and_persists_the_executed_book(self, monkeypatch, tmp_path):
        cm, cfg, _ = _serve_env(monkeypatch, tmp_path)
        out = weekly._build_report_inputs(_banded(cfg), cm)
        pd.testing.assert_series_equal(out["target_weights"], out["pre_band_target_weights"])
        assert out["no_trade_band"] == 0.05
        saved = weekly.load_held_weights(cm, as_of=pd.Timestamp("2026-09-30"))
        pd.testing.assert_series_equal(saved, out["target_weights"], check_names=False)

    def test_the_band_suppresses_one_trade_and_allows_another_at_serve(self, monkeypatch, tmp_path):
        """Target is SPY 0.284838 / TLT 0.715162. Held SPY 0.26 (2.5pp away -> held) and
        TLT 0.60 (11.5pp -> traded). The report's weights must be the banded book."""
        cm, cfg, _ = _serve_env(monkeypatch, tmp_path, as_of="2026-08-31", served_prior=_LABEL_DISTRIBUTION)
        held = pd.Series({"SPY": 0.26, "TLT": 0.60})
        weekly.save_executed_weights(held, None, cm, as_of=pd.Timestamp("2026-07-31"))
        out = weekly._build_report_inputs(_banded(cfg), cm)
        target, executed = out["pre_band_target_weights"], out["target_weights"]
        assert abs(target["SPY"] - 0.26) <= 0.05 < abs(target["TLT"] - 0.60)   # precondition
        assert executed["SPY"] == 0.26 and executed["TLT"] == target["TLT"]
        assert not executed.equals(target), "the band did not change the served book"
        assert out["cash"] == pytest.approx(1.0 - 0.26 - float(target["TLT"]))

    def test_the_negative_residual_branch_fires_at_serve(self, monkeypatch, tmp_path):
        """Held SPY 0.32 (3.5pp -> held) + TLT's 0.715 target = 1.035: TLT alone is scaled."""
        cm, cfg, _ = _serve_env(monkeypatch, tmp_path, as_of="2026-08-31", served_prior=_LABEL_DISTRIBUTION)
        weekly.save_executed_weights(pd.Series({"SPY": 0.32, "TLT": 0.60}), None, cm, as_of=pd.Timestamp("2026-07-31"))
        out = weekly._build_report_inputs(_banded(cfg), cm)
        target, executed = out["pre_band_target_weights"], out["target_weights"]
        assert 0.32 + float(target["TLT"]) > 1.0                                # precondition
        assert executed["SPY"] == 0.32
        assert executed["TLT"] == pytest.approx(0.68) and executed["TLT"] < target["TLT"]
        assert float(executed.sum()) == pytest.approx(1.0) and out["cash"] == pytest.approx(0.0, abs=1e-12)

    def test_a_same_month_rerun_rebands_against_the_same_held_book(self, monkeypatch, tmp_path):
        cm, cfg, _ = _serve_env(monkeypatch, tmp_path, as_of="2026-08-31")
        held = pd.Series({"SPY": 0.26, "TLT": 0.60})
        weekly.save_executed_weights(held, None, cm, as_of=pd.Timestamp("2026-07-31"))
        first = weekly._build_report_inputs(_banded(cfg), cm)["target_weights"]
        # The held book for a re-run in August is July's execution, not August's.
        pd.testing.assert_series_equal(
            weekly.load_held_weights(cm, as_of=pd.Timestamp("2026-08-31")), held, check_names=False
        )
        second = weekly._build_report_inputs(_banded(cfg), cm)["target_weights"]
        pd.testing.assert_series_equal(first, second)
        # ...and the NEXT month's held book is August's execution.
        pd.testing.assert_series_equal(
            weekly.load_held_weights(cm, as_of=pd.Timestamp("2026-09-30")), first, check_names=False
        )

    def test_an_all_cash_book_is_a_held_book_not_a_cold_start(self, tmp_path):
        from trading_crab_lib.checkpoints import CheckpointManager

        cm = CheckpointManager(checkpoint_dir=tmp_path / "cp")
        assert weekly.load_held_weights(cm, as_of=pd.Timestamp("2026-08-31")) is None
        weekly.save_executed_weights(pd.Series(dtype=float), None, cm, as_of=pd.Timestamp("2026-07-31"))
        held = weekly.load_held_weights(cm, as_of=pd.Timestamp("2026-08-31"))
        assert held is not None and held.empty

    def test_band_off_serves_the_target_and_writes_no_held_book(self, monkeypatch, tmp_path):
        cm, cfg, _ = _serve_env(monkeypatch, tmp_path)
        out = weekly._build_report_inputs(_banded(cfg, None), cm)
        assert out["target_weights"] is out["pre_band_target_weights"]
        assert weekly.load_held_weights(cm, as_of=pd.Timestamp("2026-09-30")) is None

    def test_the_report_names_the_band(self):
        md = weekly.assemble_weekly_report(
            regime_probs={0: 1.0}, transition_matrix=pd.DataFrame(), returns_by_regime=pd.DataFrame(),
            target_weights=pd.Series(dtype=float), accounts=[], active_regime=0, no_trade_band=0.05,
        )
        assert "EXECUTED book after the 5.0% no-trade band" in md


# ── the always-printed target allocation table (08.1, D-09 / DECISIONS A-11) ─

_CLASSES = {"SPY": "equities", "TLT": "long_duration", "IAU": "gold", "USO": "oil", "FZFXX": "cash"}
_TARGET = pd.Series({"SPY": 0.40, "TLT": 0.20, "IAU": 0.10, "USO": 0.05})
_LAST_WEEK = pd.Series({"SPY": 0.35, "TLT": 0.20, "IAU": 0.10, "USO": 0.05})


def _allocation_md(**overrides) -> str:
    kwargs = dict(
        regime_probs={0: 1.0}, transition_matrix=pd.DataFrame(), returns_by_regime=pd.DataFrame(),
        target_weights=_TARGET, cash=0.25, accounts=[], active_regime=0, no_trade_band=0.05,
        last_week_weights=_LAST_WEEK, asset_classes=_CLASSES,
    )
    kwargs.update(overrides)
    return weekly.assemble_weekly_report(**kwargs)


def _table_rows(md: str) -> list[str]:
    start = md.index("### Target allocation")
    return [ln for ln in md[start:].splitlines() if ln.startswith("|")]


class TestAllocationTable:
    def test_prints_one_row_per_class_with_target_last_week_and_change(self):
        rows = _table_rows(_allocation_md())
        assert rows[0] == "| Class | Ticker | Target % | Last week % | Change |"
        assert "| equities | SPY | 40.0% | 35.0% | +5.0 pp |" in rows
        assert "| long_duration | TLT | 20.0% | 20.0% | +0.0 pp |" in rows
        assert "| gold | IAU | 10.0% | 10.0% | +0.0 pp |" in rows
        assert "| oil | USO | 5.0% | 5.0% | +0.0 pp |" in rows
        # cash: target is the residual passed in; last week's cash is 1 - sum(last week's risky).
        assert "| cash | FZFXX | 25.0% | 30.0% | -5.0 pp |" in rows
        classes = [r.split("|")[1].strip() for r in rows[2:]]
        assert classes == ["equities", "long_duration", "gold", "oil", "cash"], "asset_classes order"

    def test_no_last_week_book_reads_na(self):
        rows = _table_rows(_allocation_md(last_week_weights=None))
        assert "| equities | SPY | 40.0% | n/a | n/a |" in rows
        assert "| cash | FZFXX | 25.0% | n/a | n/a |" in rows

    def test_an_absent_target_weight_is_zero_and_an_all_cash_last_week_is_not_na(self):
        rows = _table_rows(_allocation_md(target_weights=pd.Series({"SPY": 0.5}), cash=0.5,
                                          last_week_weights=pd.Series(dtype=float)))
        assert "| long_duration | TLT | 0.0% | 0.0% | +0.0 pp |" in rows
        assert "| equities | SPY | 50.0% | 0.0% | +50.0 pp |" in rows
        assert "| cash | FZFXX | 50.0% | 100.0% | -50.0 pp |" in rows

    def test_a_ticker_outside_the_map_gets_its_own_unmapped_row(self):
        rows = _table_rows(_allocation_md(target_weights=pd.Series({"SPY": 0.4, "ZZZ": 0.1}), cash=0.5))
        assert "| unmapped | ZZZ | 10.0% | 0.0% | +10.0 pp |" in rows

    def test_a_zero_change_never_prints_a_negative_zero(self):
        same = pd.Series({"SPY": 0.1 + 0.2, "TLT": 0.7 - 0.4})
        rows = _table_rows(_allocation_md(target_weights=same, last_week_weights=pd.Series({"SPY": 0.3, "TLT": 0.3}),
                                          cash=0.4))
        assert not [r for r in rows if "-0.0 pp" in r]

    def test_prints_without_accounts_above_the_band_sentence_and_the_account_loop(self, tmp_path):
        (tmp_path / "acct1.yaml").write_text("weights:\n  SPY: 0.2\ncash: 0.8\n", encoding="utf-8")
        with_account = _allocation_md(accounts=["acct1"], accounts_dir=tmp_path)
        no_account = _allocation_md()
        for md in (with_account, no_account):
            assert md.index("## Target vs. Current — Trades Implied") < md.index("### Target allocation")
            assert md.index("### Target allocation") < md.index("EXECUTED book after the 5.0% no-trade band")
        assert with_account.index("### Target allocation") < with_account.index("### Account: acct1")
        assert "### Account:" not in no_account

    def test_an_empty_target_with_no_classes_still_renders_the_heading(self):
        md = _allocation_md(target_weights=pd.Series(dtype=float), cash=None, asset_classes=None,
                            last_week_weights=None)
        assert "### Target allocation" in md

    def test_the_class_map_is_the_reverse_of_the_splice_config_in_splice_order(self):
        from trading_crab_lib.platform.config import load_platform_config

        mapping = weekly._class_by_ticker(load_platform_config())
        assert list(mapping.items()) == [
            ("SPY", "equities"), ("TLT", "long_duration"), ("IAU", "gold"), ("USO", "oil"), ("FZFXX", "cash"),
        ]


class TestLastWeeksExecutedBook:
    def test_no_checkpoint_is_none(self, tmp_path):
        from trading_crab_lib.checkpoints import CheckpointManager

        assert weekly.load_last_executed_weights(CheckpointManager(checkpoint_dir=tmp_path / "cp")) is None

    def test_returns_the_executed_rows_never_the_held_in_rows(self, tmp_path):
        from trading_crab_lib.checkpoints import CheckpointManager

        cm = CheckpointManager(checkpoint_dir=tmp_path / "cp")
        weekly.save_executed_weights(
            pd.Series({"SPY": 0.26, "TLT": 0.60}), pd.Series({"SPY": 0.10}), cm, as_of=pd.Timestamp("2026-07-31")
        )
        got = weekly.load_last_executed_weights(cm)
        pd.testing.assert_series_equal(got, pd.Series({"SPY": 0.26, "TLT": 0.60}), check_names=False)
        # load_held_weights on a same-month re-run would return the held_in book: not this.
        held = weekly.load_held_weights(cm, as_of=pd.Timestamp("2026-07-31"))
        assert float(held["SPY"]) == 0.10

    def test_an_all_cash_book_is_an_empty_series_not_none(self, tmp_path):
        from trading_crab_lib.checkpoints import CheckpointManager

        cm = CheckpointManager(checkpoint_dir=tmp_path / "cp")
        weekly.save_executed_weights(pd.Series(dtype=float), None, cm, as_of=pd.Timestamp("2026-07-31"))
        got = weekly.load_last_executed_weights(cm)
        assert got is not None and got.empty

    def test_build_inputs_reads_last_weeks_book_before_it_saves_this_weeks(self, monkeypatch, tmp_path):
        cm, cfg, _ = _serve_env(monkeypatch, tmp_path, as_of="2026-08-31")
        last = pd.Series({"SPY": 0.26, "TLT": 0.60})
        weekly.save_executed_weights(last, None, cm, as_of=pd.Timestamp("2026-07-31"))
        out = weekly._build_report_inputs(_banded(cfg), cm)
        pd.testing.assert_series_equal(out["last_week_weights"], last, check_names=False)
        assert not out["last_week_weights"].equals(out["target_weights"]), "it must not be this run's own book"
        # ...and the checkpoint now holds this run's book, so a load AFTER the save would differ.
        pd.testing.assert_series_equal(
            weekly.load_last_executed_weights(cm), out["target_weights"], check_names=False
        )

    def test_band_off_has_no_last_week_book(self, monkeypatch, tmp_path):
        cm, cfg, _ = _serve_env(monkeypatch, tmp_path)
        assert weekly._build_report_inputs(_banded(cfg, None), cm)["last_week_weights"] is None

