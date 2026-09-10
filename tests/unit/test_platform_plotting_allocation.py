"""Unit tests for ``platform/plotting/allocation.py`` (P5 — assets & allocation).

Follows ``tests/unit/test_plotting.py``'s shape: Agg backend forced in the
module header, seeded synthetic frames, one ``TestXxx`` class per public
function with a does-not-crash case and an empty-input case.

No network, no real checkpoint reads. ``tests/conftest.py``'s session-scoped
checkpoint-isolation fixture deliberately does not seed real ``data/holdout/``
content, so any assertion about the real full span belongs in the plan's own
``python -c`` ``<verify>`` scripts (which run outside pytest), not here.
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

from trading_crab_lib.platform.assets.returns import returns_by_regime_stats  # noqa: E402
from trading_crab_lib.platform.plotting import allocation as pallocation  # noqa: E402

# pylint: enable=wrong-import-position,wrong-import-order


ASSETS: list[str] = ["SPY", "TLT", "IAU", "USO"]

# Mirrors the schema of config/platform_settings.yaml's `splice` block — kept
# inline (not loaded from disk) so this suite stays deterministic and isolated
# from concurrent edits to platform_settings.yaml.
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
    "allocation": {
        "target_vol_annual": 0.10,
        "ewma_halflife_months": 6,
        "portfolio_vol_min_obs": 12,
    },
}


def _monthly_index(start: str, periods: int) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=periods, freq="ME")


@pytest.fixture
def synthetic_monthly_raw() -> pd.DataFrame:
    """A minimal monthly ingest frame carrying every SPLICE_CFG source column."""
    idx = _monthly_index("1962-01-31", 60)
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "sp500": 100 * np.cumprod(1 + rng.normal(0.006, 0.03, 60)),
            "div_yield": np.full(60, 0.03),
            "fred_gs10": 0.04 + 0.0005 * np.arange(60),
            "gold_spot": 35.0 + np.arange(60),
            "wti_crude": 3.0 + 0.1 * np.arange(60),
            "fred_tb3ms": np.full(60, 0.02),
        },
        index=idx,
    )


@pytest.fixture
def asset_returns() -> pd.DataFrame:
    """60 seeded monthly return months over the four tradable tickers."""
    idx = _monthly_index("2015-01-31", 60)
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "SPY": rng.normal(0.008, 0.04, 60),
            "TLT": rng.normal(0.002, 0.02, 60),
            "IAU": rng.normal(0.003, 0.03, 60),
            "USO": rng.normal(0.001, 0.07, 60),
        },
        index=idx,
    )


@pytest.fixture
def cash_returns(asset_returns: pd.DataFrame) -> pd.Series:
    return pd.Series(0.002, index=asset_returns.index, name="cash")


@pytest.fixture
def states(asset_returns: pd.DataFrame) -> pd.Series:
    """A 2-state label series with persistent runs, as real regime labels have."""
    raw = np.repeat([0, 1, 0, 1, 0, 1], 10)[: len(asset_returns)]
    return pd.Series(raw, index=asset_returns.index, name="state")


@pytest.fixture
def stats_df(asset_returns: pd.DataFrame, states: pd.Series) -> pd.DataFrame:
    return returns_by_regime_stats(asset_returns, states)


class TestInvestableAssetReturns:
    def test_returns_the_four_tradable_tickers_and_a_cash_series(self, synthetic_monthly_raw):
        asset_returns, cash_ret = pallocation.investable_asset_returns(synthetic_monthly_raw, SPLICE_CFG)

        assert set(asset_returns.columns) == set(ASSETS)
        assert isinstance(cash_ret, pd.Series)
        assert len(cash_ret) == len(synthetic_monthly_raw)
        assert asset_returns.index.equals(cash_ret.index)

    def test_cash_is_never_a_tilted_asset_column(self, synthetic_monthly_raw):
        """FZFXX is the vol-target residual, not a risk position (report.py review F4)."""
        asset_returns, _ = pallocation.investable_asset_returns(synthetic_monthly_raw, SPLICE_CFG)

        assert "FZFXX" not in asset_returns.columns
        assert "cash" not in asset_returns.columns

    def test_unavailable_optional_class_is_skipped_with_a_warning(self, synthetic_monthly_raw, caplog):
        """An optional class whose source never resolved drops out, mirroring report.py's _excluded."""
        raw = synthetic_monthly_raw.drop(columns=["gold_spot"])

        with caplog.at_level("WARNING"):
            asset_returns, _ = pallocation.investable_asset_returns(raw, SPLICE_CFG)

        assert set(asset_returns.columns) == {"SPY", "TLT", "USO"}
        assert any("EXCLUDES" in record.message for record in caplog.records)

    def test_does_not_mutate_the_input_frame(self, synthetic_monthly_raw):
        before = synthetic_monthly_raw.copy()

        pallocation.investable_asset_returns(synthetic_monthly_raw, SPLICE_CFG)

        pd.testing.assert_frame_equal(synthetic_monthly_raw, before)


class TestPlotReturnsByRegimeHeatmap:
    def test_does_not_crash(self, stats_df):
        fig = pallocation.plot_returns_by_regime_heatmap(stats_df)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_alternate_metric(self, stats_df):
        fig = pallocation.plot_returns_by_regime_heatmap(stats_df, metric="mean_monthly_return")

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_empty_input(self):
        fig = pallocation.plot_returns_by_regime_heatmap(pd.DataFrame())

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_unknown_metric_degrades_to_no_data(self, stats_df):
        fig = pallocation.plot_returns_by_regime_heatmap(stats_df, metric="not_a_column")

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_save_path_writes_a_file(self, stats_df, tmp_path):
        target = tmp_path / "heatmap.png"

        pallocation.plot_returns_by_regime_heatmap(stats_df, save_path=target)

        assert target.exists()
        plt.close("all")


class TestPlotEwmaVolTimeline:
    def test_does_not_crash(self, asset_returns):
        fig = pallocation.plot_ewma_vol_timeline(asset_returns)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_single_column_frame(self, asset_returns):
        fig = pallocation.plot_ewma_vol_timeline(asset_returns[["SPY"]])

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_ragged_inception_column_is_still_drawn(self, asset_returns):
        """A late-inception ticker's leading NaNs must not blank the whole panel."""
        ragged = asset_returns.copy()
        ragged.loc[ragged.index[:40], "USO"] = np.nan

        fig = pallocation.plot_ewma_vol_timeline(ragged)

        assert isinstance(fig, plt.Figure)
        assert len(fig.axes[0].lines) == 4
        plt.close(fig)

    def test_empty_input(self):
        fig = pallocation.plot_ewma_vol_timeline(pd.DataFrame())

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_all_nan_input(self, asset_returns):
        fig = pallocation.plot_ewma_vol_timeline(asset_returns * np.nan)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# ── D-01 fresh-package boundary + T-06-24 no-persistence proof ───────────────

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


class TestFreshPackageBoundary:
    def test_no_reachable_module_is_legacy_plotting(self):
        reachable = _transitive_closure("trading_crab_lib.platform.plotting.allocation")
        offending = {
            m
            for m in reachable
            if m == _FORBIDDEN_LEGACY_MODULE or m.startswith(_FORBIDDEN_LEGACY_MODULE + ".")
        }
        assert not offending, (
            f"platform.plotting.allocation transitively imports legacy plotting module(s), "
            f"which D-01 forbids: {offending}"
        )

    def test_matplotlib_is_reached_only_through_core(self):
        """allocation.py owns no plotting-library import of its own (core owns the Agg guard)."""
        tree = ast.parse(Path(pallocation.__file__).read_text())
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        offending = {name for name in imported if name.split(".")[0] in {"matplotlib", "seaborn", "pylab"}}
        assert not offending, f"allocation.py imports a plotting library directly: {offending}"

    def test_allocation_is_not_re_exported_from_the_package_barrel(self):
        """Wave-2 submodules are imported by path so the five plans never contend for __init__.py."""
        from trading_crab_lib.platform import plotting as pplot

        assert not hasattr(pplot, "investable_asset_returns")


class TestNeverPersistsAnything:
    """T-06-24: opening P5 must not be able to write a production checkpoint."""

    def test_source_imports_no_checkpoint_manager(self):
        """AST-level, not substring: the module docstring names it by design."""
        tree = ast.parse(Path(pallocation.__file__).read_text())
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imported.add(node.module)
                imported.update(alias.name for alias in node.names)

        assert "CheckpointManager" not in imported
        assert "get_platform_checkpoint_manager" not in imported
        assert not any("checkpoints" in name for name in imported), imported

    def test_source_calls_dot_save_nowhere(self):
        source = Path(pallocation.__file__).read_text()

        for banned in (".save(", ".save_model(", "report_returns_by_regime", "save_active_regime"):
            assert banned not in source, banned

    def test_source_never_invokes_the_walk_forward_entrypoints(self):
        """AST-level, not substring: the module docstring names both by design."""
        tree = ast.parse(Path(pallocation.__file__).read_text())
        called: set[str] = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Name):
                called.add(func.id)
            elif isinstance(func, ast.Attribute):
                called.add(func.attr)
        for banned in ("run_backtest", "run_full_backtest_evaluation"):
            assert banned not in called, banned
