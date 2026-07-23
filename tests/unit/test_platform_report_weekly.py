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
