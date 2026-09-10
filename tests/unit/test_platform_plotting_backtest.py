"""Unit tests for ``platform/plotting/backtest.py`` (P6 — backtest evaluation).

Follows ``tests/unit/test_platform_plotting_allocation.py``'s shape: Agg
backend forced in the module header, seeded synthetic frames, one ``TestXxx``
class per public function with a does-not-crash case and an empty-input case.

No network, no real checkpoint reads. ``tests/conftest.py``'s session-scoped
checkpoint-isolation fixture deliberately does not seed real ``data/holdout/``
content, so any assertion about the real full span or the live persisted
artifacts belongs in the plan's own ``python -c`` ``<verify>`` scripts (which
run outside pytest), not here.

Source-discipline assertions over the module and over
``notebooks/platform/P6_backtest_evaluation.ipynb`` are AST-based wherever the
module's own docstring legitimately names the thing being forbidden — a naive
substring scan fails on the very documentation that records the constraint.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

# matplotlib.use("Agg") must precede pyplot import — import order is intentional.
# pylint: disable=wrong-import-position,wrong-import-order
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from trading_crab_lib.platform.plotting import backtest as pbacktest  # noqa: E402
from trading_crab_lib.platform.plotting import core as pcore  # noqa: E402

# pylint: enable=wrong-import-position,wrong-import-order


LEGS: list[str] = ["strategy", "no_regime_ablation", "spy_buy_hold", "sixty_forty", "faber_sma"]

# Mirrors config/platform_settings.yaml's `splice` + `backtest` blocks — kept
# inline (not loaded from disk) so this suite stays deterministic and isolated
# from concurrent edits to platform_settings.yaml.
CFG: dict = {
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
            "optional": True,
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
    "backtest": {
        "cost_bps": 10,
        "sixty_forty_rebalance": "monthly",
        "apply_cost_to_baselines": True,
    },
}


@pytest.fixture(autouse=True)
def _close_figures():
    """Every test here builds Figures; close them all so pyplot's registry
    never trips matplotlib's >20-open-figures warning mid-suite."""
    yield
    plt.close("all")


def _monthly_index(start: str, periods: int) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=periods, freq="ME")


@pytest.fixture
def synthetic_monthly_raw() -> pd.DataFrame:
    """A minimal monthly ingest frame carrying every CFG['splice'] source column.

    Spans 1962-2022 so the holdout split has something on BOTH sides of the
    2020-12-31 cutoff — otherwise the dev-bound assertion would pass trivially.
    """
    n = 730
    idx = _monthly_index("1962-01-31", n)
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "sp500": 100 * np.cumprod(1 + rng.normal(0.006, 0.03, n)),
            "div_yield": np.full(n, 0.03),
            "fred_gs10": 0.04 + 0.00002 * np.arange(n),
            "gold_spot": 35.0 + np.arange(n),
            "wti_crude": 3.0 + 0.01 * np.arange(n),
            "fred_tb3ms": np.full(n, 0.02),
        },
        index=idx,
    )


@pytest.fixture
def curves() -> dict[str, pd.Series]:
    """Five seeded monthly return series with deliberately ragged start dates."""
    rng = np.random.default_rng(42)
    out: dict[str, pd.Series] = {}
    for offset, leg in enumerate(LEGS):
        idx = _monthly_index("1972-01-31", 240 - 10 * offset)
        out[leg] = pd.Series(rng.normal(0.006, 0.03, len(idx)), index=idx, name=leg)
    return out


@pytest.fixture
def kpi_table() -> pd.DataFrame:
    """A 5-row KPI frame matching the real backtest_kpi_table.parquet schema.

    Values are synthetic-but-plausible; the live numbers are asserted in the
    plan's own ``python -c`` verify script, never hardcoded here.
    """
    return pd.DataFrame(
        {
            "leg": LEGS,
            "terminal_log_wealth": [4.0, 3.5, 5.5, 5.0, 6.0],
            "max_drawdown": [-0.21, -0.20, -0.49, -0.27, -0.19],
        }
    )


@pytest.fixture
def calibration_df() -> pd.DataFrame:
    """A calibration frame matching model_metrics_calibration.parquet's schema."""
    rng = np.random.default_rng(42)
    rows = []
    for class_label in range(5):
        for bin_id in range(4):
            low, high = bin_id * 0.25, (bin_id + 1) * 0.25
            rows.append(
                {
                    "class_label": class_label,
                    "bin": bin_id,
                    "bin_low": low,
                    "bin_high": high,
                    "predicted_prob_mean": (low + high) / 2,
                    "observed_freq": float(np.clip((low + high) / 2 + rng.normal(0, 0.08), 0, 1)),
                    "n_in_bin": int(rng.integers(1, 200)),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def headline() -> dict:
    """A sojourn/lag headline dict matching compute_sojourn_lag_headline's contract."""
    return {
        "median_sojourn": 97.0,
        "median_lag": 164.0,
        "ratio": 97.0 / 164.0,
        "n_transitions": 6,
        "n_resolved": 4,
        "act_threshold": 0.70,
    }


def _all_text(fig: plt.Figure) -> str:
    """Every string a figure renders: figure-level texts plus every axes artist."""
    chunks: list[str] = [t.get_text() for t in fig.texts]
    for ax in fig.axes:
        chunks.append(ax.get_title())
        chunks.append(ax.get_xlabel())
        chunks.append(ax.get_ylabel())
        chunks.extend(t.get_text() for t in ax.texts)
        chunks.extend(t.get_text() for t in ax.get_xticklabels())
        chunks.extend(t.get_text() for t in ax.get_yticklabels())
        legend = ax.get_legend()
        if legend is not None:
            chunks.extend(t.get_text() for t in legend.get_texts())
    return "\n".join(chunks)


# ── recompute_baseline_curves ────────────────────────────────────────────────


class TestRecomputeBaselineCurves:
    def test_returns_exactly_the_three_deterministic_legs(self, synthetic_monthly_raw):
        out = pbacktest.recompute_baseline_curves(synthetic_monthly_raw, CFG)

        assert set(out) == {"spy_buy_hold", "sixty_forty", "faber_sma"}
        assert all(isinstance(series, pd.Series) for series in out.values())

    def test_every_leg_is_holdout_bounded(self, synthetic_monthly_raw):
        """The 2021+ holdout is locked: no baseline leg may extend past the cutoff."""
        out = pbacktest.recompute_baseline_curves(synthetic_monthly_raw, CFG)

        for name, series in out.items():
            assert series.index.max() <= pd.Timestamp("2020-12-31"), (name, series.index.max())

    def test_input_frame_extends_past_the_cutoff_so_the_bound_is_not_trivial(self, synthetic_monthly_raw):
        assert synthetic_monthly_raw.index.max() > pd.Timestamp("2020-12-31")

    def test_costs_are_read_from_config_never_hardcoded(self, synthetic_monthly_raw):
        """apply_cost_to_baselines=False must produce a different (frictionless) 60/40 leg."""
        costed = pbacktest.recompute_baseline_curves(synthetic_monthly_raw, CFG)
        free_cfg = {**CFG, "backtest": {**CFG["backtest"], "apply_cost_to_baselines": False}}
        free = pbacktest.recompute_baseline_curves(synthetic_monthly_raw, free_cfg)

        assert not np.allclose(costed["sixty_forty"].to_numpy(), free["sixty_forty"].to_numpy())

    def test_spy_leg_is_the_untouched_equity_return_series(self, synthetic_monthly_raw):
        """spy_buy_hold is a single cost-free purchase — no haircut, ever."""
        costed = pbacktest.recompute_baseline_curves(synthetic_monthly_raw, CFG)
        free_cfg = {**CFG, "backtest": {**CFG["backtest"], "apply_cost_to_baselines": False}}
        free = pbacktest.recompute_baseline_curves(synthetic_monthly_raw, free_cfg)

        pd.testing.assert_series_equal(costed["spy_buy_hold"], free["spy_buy_hold"])


# ── plot_equity_curves ───────────────────────────────────────────────────────


class TestPlotEquityCurves:
    def test_five_legs_render_five_lines(self, curves):
        fig = pbacktest.plot_equity_curves(curves)

        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 1
        # One line per leg plus the zero reference line.
        assert len(fig.axes[0].lines) == len(curves) + 1

    def test_single_leg_renders(self, curves):
        fig = pbacktest.plot_equity_curves({"strategy": curves["strategy"]})

        assert isinstance(fig, plt.Figure)

    def test_empty_dict_returns_a_no_data_figure(self):
        fig = pbacktest.plot_equity_curves({})

        assert isinstance(fig, plt.Figure)
        assert "no data" in _all_text(fig)

    def test_all_nan_series_is_treated_as_no_data(self, curves):
        nan_curve = pd.Series(np.nan, index=curves["strategy"].index)
        fig = pbacktest.plot_equity_curves({"strategy": nan_curve})

        assert "no data" in _all_text(fig)

    def test_legend_reports_each_leg_terminal_log_wealth(self, curves):
        fig = pbacktest.plot_equity_curves(curves)
        text = _all_text(fig)

        for leg in curves:
            assert leg in text

    def test_every_leg_gets_a_distinct_color(self, curves):
        fig = pbacktest.plot_equity_curves(curves)
        # Skip the zero reference line, which is drawn first-to-last after data.
        colors = [line.get_color() for line in fig.axes[0].lines[: len(curves)]]

        assert len(set(colors)) == len(curves)

    def test_saves_to_disk_when_asked(self, curves, tmp_path):
        out = tmp_path / "nested" / "equity.png"
        pbacktest.plot_equity_curves(curves, save_path=out)

        assert out.exists()


# ── compute_ablation_delta ───────────────────────────────────────────────────


class TestComputeAblationDelta:
    def test_delta_is_strategy_minus_ablation_on_both_metrics(self, kpi_table):
        delta = pbacktest.compute_ablation_delta(kpi_table)

        assert delta["wealth_delta"] == pytest.approx(4.0 - 3.5)
        assert delta["dd_delta"] == pytest.approx(-0.21 - (-0.20))

    def test_missing_strategy_row_raises_naming_the_leg(self, kpi_table):
        without = kpi_table[kpi_table["leg"] != "strategy"]

        with pytest.raises(ValueError, match="strategy"):
            pbacktest.compute_ablation_delta(without)

    def test_missing_ablation_row_raises_naming_the_leg(self, kpi_table):
        without = kpi_table[kpi_table["leg"] != "no_regime_ablation"]

        with pytest.raises(ValueError, match="no_regime_ablation"):
            pbacktest.compute_ablation_delta(without)

    def test_returns_plain_floats(self, kpi_table):
        delta = pbacktest.compute_ablation_delta(kpi_table)

        assert all(isinstance(value, float) for value in delta.values())


# ── plot_kpi_table_bars ──────────────────────────────────────────────────────


class TestPlotKpiTableBars:
    def test_real_shaped_table_renders(self, kpi_table):
        fig = pbacktest.plot_kpi_table_bars(kpi_table)

        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 2  # one panel per KPI metric

    def test_every_leg_appears_as_a_tick_label(self, kpi_table):
        fig = pbacktest.plot_kpi_table_bars(kpi_table)
        text = _all_text(fig)

        for leg in kpi_table["leg"]:
            assert leg in text

    def test_metrics_are_not_drawn_on_one_shared_scale(self, kpi_table):
        """Drawdowns (~-0.2) would collapse onto zero beside log wealth (~5)."""
        fig = pbacktest.plot_kpi_table_bars(kpi_table)
        ylims = [ax.get_ylim() for ax in fig.axes]

        assert ylims[0] != ylims[1]

    def test_empty_table_returns_a_no_data_figure(self):
        fig = pbacktest.plot_kpi_table_bars(pd.DataFrame(columns=["leg", "terminal_log_wealth", "max_drawdown"]))

        assert isinstance(fig, plt.Figure)
        assert "no data" in _all_text(fig)


# ── plot_calibration_summary ─────────────────────────────────────────────────


class TestPlotCalibrationSummary:
    def test_real_shaped_calibration_frame_renders(self, calibration_df):
        fig = pbacktest.plot_calibration_summary(calibration_df)

        assert isinstance(fig, plt.Figure)
        assert len(fig.axes[0].collections) == calibration_df["class_label"].nunique()

    def test_a_forty_five_degree_reference_line_is_drawn(self, calibration_df):
        fig = pbacktest.plot_calibration_summary(calibration_df)
        lines = fig.axes[0].lines

        assert any(
            list(line.get_xdata()) == [0, 1] and list(line.get_ydata()) == [0, 1] for line in lines
        )

    def test_smallest_bin_is_still_visible(self, calibration_df):
        """A single-observation bin must never render at zero area."""
        small = calibration_df.copy()
        small.loc[0, "n_in_bin"] = 1
        small.loc[1, "n_in_bin"] = 100_000
        fig = pbacktest.plot_calibration_summary(small)
        areas = np.concatenate([coll.get_sizes() for coll in fig.axes[0].collections])

        assert areas.min() > 0

    def test_axes_pad_past_zero_and_one_so_edge_bins_are_not_clipped(self, calibration_df):
        fig = pbacktest.plot_calibration_summary(calibration_df)
        ax = fig.axes[0]

        assert ax.get_xlim()[0] < 0.0 and ax.get_xlim()[1] > 1.0
        assert ax.get_ylim()[0] < 0.0 and ax.get_ylim()[1] > 1.0

    def test_empty_frame_returns_a_no_data_figure(self):
        empty = pd.DataFrame(
            columns=["class_label", "bin", "bin_low", "bin_high", "predicted_prob_mean", "observed_freq", "n_in_bin"]
        )
        fig = pbacktest.plot_calibration_summary(empty)

        assert isinstance(fig, plt.Figure)
        assert "no data" in _all_text(fig)


# ── plot_sojourn_lag_headline (the A13 discipline) ───────────────────────────


class TestPlotSojournLagHeadline:
    def test_finite_ratio_renders(self, headline):
        fig = pbacktest.plot_sojourn_lag_headline(headline)

        assert isinstance(fig, plt.Figure)

    def test_figure_text_carries_the_verbatim_a13_caveat(self, headline):
        fig = pbacktest.plot_sojourn_lag_headline(headline)
        text = _all_text(fig)

        assert "A13" in text
        # The whole caveat is present modulo the wrapping newlines.
        assert " ".join(pcore.A13_CAVEAT.split()) in " ".join(text.split())

    def test_figure_text_carries_the_resolved_of_total_transition_count(self, headline):
        fig = pbacktest.plot_sojourn_lag_headline(headline)

        assert "4 of 6" in _all_text(fig)

    def test_nan_ratio_renders_without_raising_and_keeps_the_caveat(self, headline):
        degenerate = {**headline, "ratio": float("nan"), "median_lag": float("nan"), "n_resolved": 0}
        fig = pbacktest.plot_sojourn_lag_headline(degenerate)
        text = _all_text(fig)

        assert isinstance(fig, plt.Figure)
        assert "n/a (unresolved)" in text
        assert "A13" in text
        assert "0 of 6" in text

    def test_zero_valued_sojourn_is_not_coerced_to_nan(self, headline):
        """A real 0.0 must render as 0.0, not as 'n/a' (falsy-value trap)."""
        fig = pbacktest.plot_sojourn_lag_headline({**headline, "median_sojourn": 0.0})
        text = _all_text(fig)

        assert "0.0 mo" in text

    def test_the_caveat_is_never_conditional_on_the_ratio(self, headline):
        """Every ratio magnitude, including a 'reassuring' one, still carries A13."""
        for ratio in (0.0, 0.5914634146341463, 1.0, 25.0):
            fig = pbacktest.plot_sojourn_lag_headline({**headline, "ratio": ratio})
            assert "A13" in _all_text(fig), ratio

    def test_caveat_override_is_honored(self, headline):
        fig = pbacktest.plot_sojourn_lag_headline(headline, caveat="CUSTOM A13 NOTE")

        assert "CUSTOM A13 NOTE" in _all_text(fig)

    def test_saves_to_disk_when_asked(self, headline, tmp_path):
        out = tmp_path / "sojourn.png"
        pbacktest.plot_sojourn_lag_headline(headline, save_path=out)

        assert out.exists()


# ── D-01 fresh-package boundary + no-persistence proof ───────────────────────

_FORBIDDEN_LEGACY_MODULE = "trading_crab_lib.plotting"


def _local_imports(path: Path) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return {n for n in names if n.startswith("trading_crab_lib")}


def _transitive_closure(start_module: str) -> set[str]:
    seen: set[str] = set()
    stack = [start_module]
    while stack:
        mod_name = stack.pop()
        if mod_name in seen:
            continue
        seen.add(mod_name)
        spec = importlib.util.find_spec(mod_name)
        if spec is None or spec.origin is None:
            continue
        path = Path(spec.origin)
        if path.suffix != ".py":
            continue
        stack.extend(_local_imports(path))
    return seen


def _module_imports(module) -> set[str]:
    imported: set[str] = set()
    for node in ast.walk(ast.parse(Path(module.__file__).read_text())):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    return imported


def _called_names(module) -> set[str]:
    called: set[str] = set()
    for node in ast.walk(ast.parse(Path(module.__file__).read_text())):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                called.add(func.id)
            elif isinstance(func, ast.Attribute):
                called.add(func.attr)
    return called


class TestFreshPackageBoundary:
    def test_no_reachable_module_is_legacy_plotting(self):
        reachable = _transitive_closure("trading_crab_lib.platform.plotting.backtest")
        offending = {
            m
            for m in reachable
            if m == _FORBIDDEN_LEGACY_MODULE or m.startswith(_FORBIDDEN_LEGACY_MODULE + ".")
        }
        assert not offending, (
            f"platform.plotting.backtest transitively imports legacy plotting module(s), "
            f"which D-01 forbids: {offending}"
        )

    def test_matplotlib_is_reached_only_through_core(self):
        """backtest.py owns no plotting-library import of its own (core owns the Agg guard)."""
        offending = {name for name in _module_imports(pbacktest) if name.split(".")[0] in {"matplotlib", "seaborn", "pylab"}}

        assert not offending, f"backtest.py imports a plotting library directly: {offending}"

    def test_backtest_is_not_re_exported_from_the_package_barrel(self):
        """Wave-2 submodules are imported by path so the plans never contend for __init__.py."""
        from trading_crab_lib.platform import plotting as pplot

        assert not hasattr(pplot, "recompute_baseline_curves")


class TestNeverRunsTheWalkForward:
    """Opening P6 must never turn into a multi-minute 588-refit walk-forward."""

    def test_source_never_calls_the_full_walk_forward_entrypoints(self):
        """AST-level, not substring: the module docstring names both by design."""
        called = _called_names(pbacktest)

        assert "run_backtest" not in called
        assert "run_full_backtest_evaluation" not in called
        assert "no_regime_ablation" not in called

    def test_source_imports_no_checkpoint_manager_and_no_report_module(self):
        imported = _module_imports(pbacktest)
        imported_symbols: set[str] = set()
        for node in ast.walk(ast.parse(Path(pbacktest.__file__).read_text())):
            if isinstance(node, ast.ImportFrom):
                imported_symbols.update(alias.name for alias in node.names)

        assert "CheckpointManager" not in imported_symbols
        assert not any("evaluation.report" in name for name in imported)

    def test_source_never_saves_a_checkpoint_or_logs_a_trial(self):
        called = _called_names(pbacktest)

        assert "save" not in called
        assert "append_trial" not in called
