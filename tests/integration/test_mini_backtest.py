"""Synthetic end-to-end integration test for the EVAL-01..04 capstone
(``trading_crab_lib.platform.evaluation.report::run_full_backtest_evaluation``).

No network, no real Phase 1-4 checkpoints — mirrors
``tests/integration/test_mini_pipeline.py``'s synthetic-frame convention
(``_make_synthetic_macro``-style factory, real fits on tiny data, no
monkeypatching of the layers under test). Exercises the REAL driver +
ablation + 3 baselines + multiclass sojourn/lag + date-joined y_true + kpis +
model-metrics + report wiring in one call.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_crab_lib.platform.evaluation import report
from trading_crab_lib.platform.honesty.registry import read_trials

N_MONTHS = 48
MIN_TRAIN = 24


def _cfg() -> dict:
    """A full platform-config-shaped dict (splice + taxonomy + labeling +
    allocation + backtest) — reduced sizes for a fast synthetic run.
    Mirrors test_platform_splice.py::SPLICE_CFG +
    test_platform_backtest_driver.py::_cfg's sizing."""
    lean_cols = _lean_cols()
    return {
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
        },
        "taxonomy": {"fast": lean_cols[:10], "slow": lean_cols[10:], "agency": []},
        "labeling": {"K": 2, "lambda": 5.0, "n_restarts": 2, "embargo_months": 3},
        "allocation": {
            "target_vol_annual": 0.10,
            "ewma_halflife_months": 6,
            "portfolio_vol_min_obs": 3,
            "hysteresis": {"act_threshold": 0.70, "unwind_threshold": 0.40},
        },
        "backtest": {
            "cost_bps": 10,
            "min_train_months": MIN_TRAIN,
            "skip_l1l2_for_ablation": True,
            "sixty_forty_rebalance": "monthly",
            "apply_cost_to_baselines": True,
            # Empty on purpose: this synthetic 2010-2013 window doesn't
            # overlap any real crisis window, and crisis_capture_ratio
            # requires config-driven windows within the holdout-bounded data.
            "crisis_windows": [],
        },
    }


def _lean_cols() -> list[str]:
    return [
        "curve_10y3m", "curve_10y2y", "credit_spread_baa_aaa", "fred_vix", "gold", "oil",
        "trailing_return_1m", "trailing_return_3m", "realized_vol_1m", "realized_vol_3m",
        "cape_shiller", "div_yield", "real_rate_level",
    ]


def _make_synthetic_frames(
    n_months: int = N_MONTHS, start: str = "2010-01-31", seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (monthly_features, monthly_raw) on a month-end index, well
    within the holdout cutoff (2020-12-31) by construction — no real Phase
    1 ingestion or checkpoint required.

    ``monthly_features`` mirrors the driver/baselines tests' lean-column
    synthetic frame. ``monthly_raw`` supplies the raw columns
    ``splice.build_core_research_series`` needs for all 5 core classes
    (sp500/div_yield/fred_gs10/gold_spot/wti_crude/fred_tb3ms), all positive
    and in the units ``test_platform_splice.py`` uses (decimal fraction
    yields, not percent).
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n_months, freq="ME")
    lean_cols = _lean_cols()

    monthly_features = pd.DataFrame({col: rng.normal(0, 1, n_months) for col in lean_cols}, index=idx)

    sp500 = 100 * (1.01 ** np.arange(n_months)) * (1 + rng.normal(0, 0.01, n_months))
    monthly_raw = pd.DataFrame(
        {
            "sp500": sp500,
            "div_yield": rng.uniform(0.015, 0.03, n_months),
            "fred_gs10": 0.04 + 0.001 * np.sin(np.arange(n_months) / 5) + rng.normal(0, 0.0005, n_months),
            "gold_spot": 1000 + np.cumsum(rng.normal(0, 5, n_months)),
            "wti_crude": 50 + np.cumsum(rng.normal(0, 1, n_months)),
            "fred_tb3ms": 0.01 + 0.001 * np.arange(n_months) / n_months + rng.normal(0, 0.0002, n_months),
        },
        index=idx,
    )
    return monthly_features, monthly_raw


class TestRunFullBacktestEvaluationEndToEnd:
    def test_produces_markdown_and_all_parquet_artifacts(self, tmp_path):
        monthly_features, monthly_raw = _make_synthetic_frames()
        cfg = _cfg()

        result = report.run_full_backtest_evaluation(
            monthly_features,
            monthly_raw,
            cfg,
            registry_path=tmp_path / "trials.jsonl",
            output_dir=tmp_path / "reports",
        )

        report_path = result["report_path"]
        assert report_path.exists()
        assert report_path.name == "backtest_report.md"
        markdown = report_path.read_text(encoding="utf-8")
        assert "Headline" in markdown
        assert "Sojourn" in markdown or "sojourn" in markdown

        reports_dir = tmp_path / "reports"
        for artifact_name in (
            "backtest_equity_curve_strategy.parquet",
            "backtest_equity_curve_ablation.parquet",
            "backtest_kpi_table.parquet",
            "model_metrics_brier.parquet",
            "model_metrics_calibration.parquet",
            "model_metrics_confusion.parquet",
        ):
            assert (reports_dir / artifact_name).exists(), f"missing artifact: {artifact_name}"

        kpi_table = pd.read_parquet(reports_dir / "backtest_kpi_table.parquet")
        assert set(kpi_table["leg"]) == {
            "strategy", "no_regime_ablation", "spy_buy_hold", "sixty_forty", "faber_sma",
        }

    def test_registry_gains_exactly_two_trials_strategy_and_ablation_only(self, tmp_path):
        """The 3 price baselines (spy_buy_hold/sixty_forty/faber_sma) log NO
        registry trial (review F7) — only the strategy and the no-regime
        ablation are genuinely evaluated configurations with a fitted model
        path."""
        monthly_features, monthly_raw = _make_synthetic_frames()
        cfg = _cfg()
        registry_path = tmp_path / "trials.jsonl"
        assert not registry_path.exists()

        report.run_full_backtest_evaluation(
            monthly_features, monthly_raw, cfg, registry_path=registry_path, output_dir=tmp_path / "reports",
        )

        trials = read_trials(path=registry_path)
        assert len(trials) == 2

    def test_smoothed_and_filtered_series_are_distinct_objects(self, tmp_path):
        """Pitfall 1: full_sample_states (smoothed reference) and
        per_step_metrics (walk-forward filtered) must be genuinely separate
        — not the same object, and not derived from re-slicing one another."""
        monthly_features, monthly_raw = _make_synthetic_frames()
        cfg = _cfg()

        result = report.run_full_backtest_evaluation(
            monthly_features, monthly_raw, cfg,
            registry_path=tmp_path / "trials.jsonl", output_dir=tmp_path / "reports",
        )

        full_sample_states = result["full_sample_states"]
        per_step_metrics = result["per_step_metrics"]

        assert full_sample_states is not per_step_metrics
        # full_sample_states spans the ENTIRE dev window; the walk-forward
        # per_step_metrics only visits post-min_train decision dates — the
        # two are not the same length, proving they were built independently.
        assert len(full_sample_states) > len(per_step_metrics["dates"])


class TestYTrueDateAlignment:
    """Review F2: the metrics y_true is joined by DATE to per_step_metrics
    ["dates"] from the smoothed reference — never loop-sourced."""

    def test_y_true_equals_smoothed_reference_state_at_each_decision_date(self, tmp_path):
        monthly_features, monthly_raw = _make_synthetic_frames()
        cfg = _cfg()

        result = report.run_full_backtest_evaluation(
            monthly_features, monthly_raw, cfg,
            registry_path=tmp_path / "trials.jsonl", output_dir=tmp_path / "reports",
        )

        per_step_metrics = result["per_step_metrics"]
        full_sample_states = result["full_sample_states"]

        assert "y_true" in per_step_metrics
        assert len(per_step_metrics["y_true"]) == len(per_step_metrics["dates"]) == len(per_step_metrics["proba"])

        for date, scored_label in zip(per_step_metrics["dates"], per_step_metrics["y_true"]):
            expected = int(full_sample_states.loc[date])
            assert scored_label == expected, (
                f"y_true at {date} ({scored_label}) must equal the smoothed reference "
                f"state at that decision month ({expected}) — review F2"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-q"])
