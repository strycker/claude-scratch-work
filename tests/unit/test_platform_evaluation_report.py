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

import pandas as pd
import pytest

from trading_crab_lib.platform.evaluation import report


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


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-q"])
