"""Unit tests for ``platform/plotting/nowcaster.py`` (P4 — nowcaster diagnostics).

Follows ``tests/unit/test_plotting.py``'s shape: Agg backend forced in the
module header, seeded synthetic frames, one ``TestXxx`` class per public
function with a does-not-crash case and an empty-input case. No network, no
real checkpoint reads — the plan's own ``<verify>`` scripts cover the live data
outside pytest, where the session-scoped checkpoint-isolation fixture in
``tests/conftest.py`` does not apply (it deliberately does not seed real
``data/holdout/`` content).
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

from trading_crab_lib.platform.plotting import nowcaster as pnowcaster  # noqa: E402

# pylint: enable=wrong-import-position,wrong-import-order


LEAN_COLUMNS: list[str] = [
    "credit_spread_baa_aaa",
    "curve_10y2y",
    "curve_10y3m",
    "fred_vix",
    "gold",
    "oil",
    "realized_vol_1m",
    "realized_vol_3m",
    "trailing_return_1m",
    "trailing_return_3m",
    "cape_shiller",
    "div_yield",
    "real_rate_level",
]

SYNTHETIC_CFG: dict = {
    "taxonomy": {
        "fast": LEAN_COLUMNS[:10],
        "slow": LEAN_COLUMNS[10:],
        "agency": ["fred_cpi"],
    },
    "labeling": {"embargo_months": 12},
    "cv": {"default_label_horizon_months": 3, "default_embargo_months": 1},
    "backtest": {"nowcaster_cv_splits": 3},
}


@pytest.fixture
def monthly_features() -> pd.DataFrame:
    """96 monthly rows over the 13 lean columns, seeded."""
    idx = pd.date_range("2010-01-31", periods=96, freq="ME")
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        rng.standard_normal((96, len(LEAN_COLUMNS))),
        index=idx,
        columns=LEAN_COLUMNS,
    )


@pytest.fixture
def labels(monthly_features: pd.DataFrame) -> pd.Series:
    """A synthetic 3-state label series with persistent runs (so transitions exist)."""
    rng = np.random.default_rng(7)
    raw = rng.integers(0, 3, len(monthly_features))
    # Blocked into runs of 8 so transition_window_accuracy has both transition
    # and steady-state rows to score, as real regime labels do.
    blocked = np.repeat(raw[:: 8][: (len(raw) + 7) // 8], 8)[: len(raw)]
    return pd.Series(blocked, index=monthly_features.index, name="state")


@pytest.fixture
def diagnostics(monthly_features: pd.DataFrame, labels: pd.Series) -> dict:
    return pnowcaster.fit_nowcaster_diagnostics(monthly_features, labels, SYNTHETIC_CFG)


# ── fit_nowcaster_diagnostics ────────────────────────────────────────────────


class TestFitNowcasterDiagnostics:
    def test_returns_the_documented_keys(self, diagnostics: dict):
        assert set(diagnostics) == {"model", "X", "y", "y_pred", "proba", "classes", "metrics"}

    def test_metrics_carry_all_three_accuracy_figures_in_unit_range(self, diagnostics: dict):
        metrics = diagnostics["metrics"]
        for key in ("overall_accuracy", "transition_accuracy", "steady_state_accuracy"):
            assert key in metrics, key
            value = metrics[key]
            assert np.isnan(value) or 0.0 <= value <= 1.0, (key, value)

    def test_proba_rows_align_with_returned_frames(self, diagnostics: dict):
        proba = diagnostics["proba"]
        assert proba.shape[0] == len(diagnostics["X"]) == len(diagnostics["y_pred"])
        assert proba.shape[1] == len(diagnostics["classes"])
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_trailing_embargo_months_are_excluded_from_the_training_target(
        self, diagnostics: dict, labels: pd.Series
    ):
        """D-01's structural label embargo is applied, not silently skipped."""
        cutoff = labels.index.max() - pd.DateOffset(months=12)
        assert diagnostics["y"].index.max() <= cutoff

    def test_features_are_narrowed_to_the_lean_taxonomy_set(
        self, monthly_features: pd.DataFrame, labels: pd.Series
    ):
        extended = monthly_features.copy()
        extended["SPY"] = np.nan  # an untagged ETF column that starts late
        result = pnowcaster.fit_nowcaster_diagnostics(extended, labels, SYNTHETIC_CFG)
        assert "SPY" not in result["X"].columns
        assert len(result["X"]) > 0, "narrowing to lean columns must not empty the scored set"


class TestFitNowcasterDiagnosticsNoSideEffects:
    """T-06-20: a notebook glance is not an evaluated configuration."""

    def test_calls_neither_append_trial_nor_save_model(
        self, monkeypatch: pytest.MonkeyPatch, monthly_features: pd.DataFrame, labels: pd.Series
    ):
        from trading_crab_lib import checkpoints as checkpoints_mod
        from trading_crab_lib.platform.honesty import registry as registry_mod

        def _boom(*args, **kwargs):  # pragma: no cover - must never run
            raise AssertionError("fit_nowcaster_diagnostics must not write here")

        monkeypatch.setattr(registry_mod, "append_trial", _boom)
        monkeypatch.setattr(checkpoints_mod.CheckpointManager, "save_model", _boom)

        result = pnowcaster.fit_nowcaster_diagnostics(monthly_features, labels, SYNTHETIC_CFG)
        assert result["metrics"]["overall_accuracy"] >= 0.0

    def test_source_never_references_the_two_side_effect_call_sites(self):
        source = Path(pnowcaster.__file__).read_text()
        tree = ast.parse(source)
        called = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        assert "append_trial" not in called
        assert "save_model" not in called

    def test_does_not_import_evaluate_nowcaster(self):
        """The one production entrypoint with the two side effects is never imported."""
        assert not hasattr(pnowcaster, "evaluate_nowcaster")


# ── plot_transition_window_accuracy ──────────────────────────────────────────


class TestPlotTransitionWindowAccuracy:
    def test_does_not_crash(self, diagnostics: dict):
        fig = pnowcaster.plot_transition_window_accuracy(diagnostics["metrics"])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_nan_transition_accuracy_renders_as_an_annotated_bar(self):
        metrics = {
            "overall_accuracy": 0.91,
            "transition_accuracy": float("nan"),
            "steady_state_accuracy": 0.91,
        }
        fig = pnowcaster.plot_transition_window_accuracy(metrics)
        texts = [t.get_text() for ax in fig.axes for t in ax.texts]
        assert "n/a" in texts, texts
        plt.close(fig)

    def test_all_three_figures_are_always_drawn(self):
        metrics = {"overall_accuracy": 0.9, "transition_accuracy": 0.4, "steady_state_accuracy": 0.95}
        fig = pnowcaster.plot_transition_window_accuracy(metrics)
        labels = [t.get_text() for t in fig.axes[0].get_xticklabels()]
        assert labels == ["overall accuracy", "transition accuracy", "steady state accuracy"]
        plt.close(fig)

    def test_empty_metrics_returns_a_no_data_figure(self):
        fig = pnowcaster.plot_transition_window_accuracy({})
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# ── plot_proba_over_time ─────────────────────────────────────────────────────


class TestPlotProbaOverTime:
    def test_does_not_crash(self, diagnostics: dict):
        fig = pnowcaster.plot_proba_over_time(
            diagnostics["X"].index, diagnostics["proba"], diagnostics["classes"]
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_empty_proba_returns_a_no_data_figure(self):
        fig = pnowcaster.plot_proba_over_time(pd.DatetimeIndex([]), np.zeros((0, 5)), [0, 1, 2, 3, 4])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_one_band_per_class(self):
        idx = pd.date_range("2020-01-31", periods=12, freq="ME")
        proba = np.full((12, 3), 1.0 / 3.0)
        fig = pnowcaster.plot_proba_over_time(idx, proba, [0, 1, 2])
        assert len(fig.axes[0].collections) == 3
        plt.close(fig)


# ── D-01 fresh-package boundary + module hygiene ─────────────────────────────

_FORBIDDEN_LEGACY_MODULE = "trading_crab_lib.plotting"


def _local_imports(py_file: Path) -> set[str]:
    tree = ast.parse(py_file.read_text(), filename=str(py_file))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
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


class TestFreshPackageBoundary:
    def test_no_reachable_module_is_legacy_plotting(self):
        reachable = _transitive_closure("trading_crab_lib.platform.plotting.nowcaster")
        offending = {
            m
            for m in reachable
            if m == _FORBIDDEN_LEGACY_MODULE or m.startswith(_FORBIDDEN_LEGACY_MODULE + ".")
        }
        assert not offending, (
            f"platform.plotting.nowcaster transitively imports legacy plotting module(s), "
            f"which D-01 forbids: {offending}"
        )

    def test_matplotlib_is_reached_only_through_core(self):
        """nowcaster.py owns no plotting-library import of its own (core owns the Agg guard)."""
        tree = ast.parse(Path(pnowcaster.__file__).read_text())
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        offending = {name for name in imported if name.split(".")[0] in {"matplotlib", "seaborn", "pylab"}}
        assert not offending, f"nowcaster.py imports a plotting library directly: {offending}"

    def test_nowcaster_is_not_re_exported_from_the_package_barrel(self):
        """Wave-2 submodules are imported by path so the five plans never contend for __init__.py."""
        from trading_crab_lib.platform import plotting as pplot

        assert not hasattr(pplot, "fit_nowcaster_diagnostics")

# ── P4 notebook source discipline (guarded until the notebook lands) ─────────

_P4_NOTEBOOK = Path("notebooks/platform/P4_nowcaster.ipynb")


def _p4_cells():
    import nbformat

    return nbformat.read(_P4_NOTEBOOK, as_version=4).cells


@pytest.mark.skipif(not _P4_NOTEBOOK.exists(), reason="P4 notebook not yet built")
class TestP4NotebookSource:
    def test_carries_no_sign_off_cell(self):
        """D-15: the sign-off cell is P3-exclusive."""
        combined = "\n".join(cell.source for cell in _p4_cells())
        assert "Sign-Off" not in combined
        assert "Sign-off" not in combined

    def test_never_invokes_the_full_walk_forward_entrypoints(self):
        code = "\n".join(c.source for c in _p4_cells() if c.cell_type == "code")
        for banned in ("run_full_backtest_evaluation", "run_backtest("):
            assert banned not in code, banned

    def test_never_appends_a_trial_or_saves_a_model(self):
        code = "\n".join(c.source for c in _p4_cells() if c.cell_type == "code")
        for banned in ("append_trial", "save_model", "evaluate_nowcaster"):
            assert banned not in code, banned

