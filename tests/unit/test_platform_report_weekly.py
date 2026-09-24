"""
Tests for the L4-02 weekly report assembly + trades-implied + opt-in email
delivery (04-CONTEXT.md D-02: markdown ALWAYS written, email opt-in behind
--send-email, reusing the incumbent email.py machinery read-only).
"""

from __future__ import annotations

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

    def predict_proba(self, row):
        return [[0.30, 0.45, 0.25]]


def _serve_env(monkeypatch, tmp_path, *, as_of: str = "2026-08-31"):
    """A real CheckpointManager in tmp_path plus in-memory inputs for _build_report_inputs.

    ``regime_labels`` is deliberately NON-uniform (state 0 dominates), so the filtered
    belief cannot coincide with the raw posterior by accident of a flat prior.
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
    }
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

        prior = unconditional_belief(labels, state_index=[0, 1, 2])
        assert not prior.round(9).eq(1 / 3).all(), "fixture prior is uniform; it cannot discriminate"
        expected = filter_step(prior, transition_matrix_for(labels, state_index=[0, 1, 2]), raw, prior)
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
        cm, cfg, _ = _serve_env(monkeypatch, tmp_path)
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
        cm, cfg, _ = _serve_env(monkeypatch, tmp_path, as_of="2026-08-31")
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
        cm, cfg, _ = _serve_env(monkeypatch, tmp_path, as_of="2026-08-31")
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
