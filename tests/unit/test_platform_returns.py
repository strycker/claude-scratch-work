"""Unit tests for trading_crab_lib.platform.assets.returns (L3-01).

Follows the incumbent tests/unit/test_platform_gap_lag.py structure: a class
per behavior, docstring per test. Historical-conditioning stance (no trial
registry, no purged CV) per RESEARCH.md Pitfall 4/9 — see module docstring
in returns.py.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from trading_crab_lib.checkpoints import CheckpointManager
from trading_crab_lib.platform.assets.returns import (
    _MIN_OBS_FLAG,
    compute_monthly_returns,
    report_returns_by_regime,
    returns_by_regime_stats,
)

_EXPECTED_COLUMNS = {
    "regime",
    "asset",
    "mean_monthly_return",
    "std_monthly_return",
    "sharpe_annualized",
    "hit_rate",
    "max_drawdown",
    "n_obs",
}


def _synthetic_prices(n_months: int = 24) -> pd.DataFrame:
    idx = pd.date_range("2020-01-31", periods=n_months, freq="ME")
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        {
            "SPY": 100 * (1 + rng.normal(0.01, 0.03, n_months)).cumprod(),
            "TLT": 100 * (1 + rng.normal(0.0, 0.02, n_months)).cumprod(),
            "GLD": 100 * (1 + rng.normal(0.005, 0.02, n_months)).cumprod(),
        },
        index=idx,
    )


def _synthetic_states(idx: pd.DatetimeIndex) -> pd.Series:
    return pd.Series([0 if i % 2 == 0 else 1 for i in range(len(idx))], index=idx)


# ── compute_monthly_returns ──────────────────────────────────────────────────────


class TestComputeMonthlyReturns:
    def test_pct_change_on_month_end_levels(self):
        """Plain pct_change; first row is NaN (no prior observation)."""
        prices = _synthetic_prices(5)
        returns = compute_monthly_returns(prices)
        assert returns.iloc[1:].notna().all().all()
        assert returns.iloc[0].isna().all()

    def test_short_history_asset_produces_leading_nans_not_dropped_column(self):
        """D11: an asset that only has data starting mid-series keeps its column,
        with leading NaNs where price data was absent."""
        prices = _synthetic_prices(6)
        prices.loc[prices.index[:3], "GLD"] = np.nan
        returns = compute_monthly_returns(prices)
        assert "GLD" in returns.columns
        assert returns["GLD"].iloc[:4].isna().all()  # 3 NaN levels + 1 pct_change NaN


# ── returns_by_regime_stats ──────────────────────────────────────────────────────


class TestReturnsByRegimeStats:
    def test_one_row_per_regime_asset_with_exact_columns(self):
        """One row per (regime, asset) present; columns exactly the seven stats."""
        prices = _synthetic_prices(24)
        returns = compute_monthly_returns(prices).dropna(how="all")
        states = _synthetic_states(returns.index)

        stats = returns_by_regime_stats(returns, states)

        assert set(stats.columns) == _EXPECTED_COLUMNS
        n_regimes = states.loc[states.index.intersection(returns.index)].nunique()
        n_assets = returns.shape[1]
        assert len(stats) == n_regimes * n_assets
        assert set(stats["regime"].unique()) == set(states.unique())
        assert set(stats["asset"].unique()) == set(returns.columns)

    def test_short_history_asset_survives_with_reduced_n_obs(self):
        """D11: an all-NaN-until-M asset contributes a row per regime with n_obs
        equal to its observed month count — never a dropped column, never a crash."""
        prices = _synthetic_prices(24)
        prices.loc[prices.index[:20], "GLD"] = np.nan  # only 3 usable return months
        returns = compute_monthly_returns(prices)
        states = _synthetic_states(returns.index)

        stats = returns_by_regime_stats(returns, states)

        gld_rows = stats[stats["asset"] == "GLD"]
        assert not gld_rows.empty
        assert gld_rows["n_obs"].sum() == returns["GLD"].dropna().shape[0]
        assert (gld_rows["n_obs"] < _MIN_OBS_FLAG).any()  # low-confidence cells surfaced, not hidden

    def test_sharpe_annualized_formula_and_nan_on_zero_std(self):
        """sharpe_annualized == (mean/std)*sqrt(12) when std>0; NaN (not inf) when std==0."""
        idx = pd.date_range("2020-01-31", periods=6, freq="ME")
        returns = pd.DataFrame({"FLAT": [0.0] * 6, "VAR": [0.01, -0.02, 0.03, -0.01, 0.02, 0.0]}, index=idx)
        states = pd.Series([0] * 6, index=idx)

        stats = returns_by_regime_stats(returns, states)

        flat_row = stats[stats["asset"] == "FLAT"].iloc[0]
        assert math.isnan(flat_row["sharpe_annualized"])
        assert not math.isinf(flat_row["sharpe_annualized"])

        var_row = stats[stats["asset"] == "VAR"].iloc[0]
        col = returns["VAR"]
        expected = (col.mean() / col.std()) * math.sqrt(12)
        assert var_row["sharpe_annualized"] == pytest.approx(expected)

    def test_alignment_uses_index_intersection_not_iloc(self):
        """Extra label rows (before/after) or extra return rows do not misalign
        the joined regime column (mirrors Pitfall P3: no iloc slicing)."""
        prices = _synthetic_prices(10)
        returns = compute_monthly_returns(prices).dropna(how="all")

        # states has extra rows both before and after the returns index
        extended_idx = pd.date_range(
            returns.index.min() - pd.DateOffset(months=2), returns.index.max() + pd.DateOffset(months=2), freq="ME"
        )
        states = pd.Series([i % 2 for i in range(len(extended_idx))], index=extended_idx)

        stats = returns_by_regime_stats(returns, states)

        # every stats row's asset/regime combo must correspond to a real overlap
        common = returns.index.intersection(states.index)
        expected_regimes = set(states.loc[common].unique())
        assert set(stats["regime"].unique()).issubset(expected_regimes)

    def test_hit_rate_and_max_drawdown(self):
        """hit_rate == (col > 0).mean(); max_drawdown is the min of the cumulative
        drawdown curve (<= 0)."""
        idx = pd.date_range("2020-01-31", periods=5, freq="ME")
        col_values = [0.10, -0.20, 0.05, -0.05, 0.10]
        returns = pd.DataFrame({"X": col_values}, index=idx)
        states = pd.Series([0] * 5, index=idx)

        stats = returns_by_regime_stats(returns, states)
        row = stats.iloc[0]

        expected_hit_rate = (pd.Series(col_values) > 0).mean()
        cum = (1 + pd.Series(col_values)).cumprod()
        expected_dd = (cum / cum.cummax() - 1).min()

        assert row["hit_rate"] == pytest.approx(expected_hit_rate)
        assert row["max_drawdown"] == pytest.approx(expected_dd)
        assert row["max_drawdown"] <= 0


# ── report_returns_by_regime ─────────────────────────────────────────────────────


class TestReportReturnsByRegime:
    def test_persists_checkpoint_and_parquet(self, tmp_path):
        """Writes the returns_by_regime checkpoint AND a schema-stable parquet
        artifact under the given output_dir; round-trippable via read_parquet."""
        prices = _synthetic_prices(24)
        returns = compute_monthly_returns(prices).dropna(how="all")
        states = _synthetic_states(returns.index)
        cm = CheckpointManager(checkpoint_dir=tmp_path / "checkpoints")
        report_dir = tmp_path / "reports"

        path = report_returns_by_regime(returns, states, output_dir=report_dir, cm=cm)

        assert path.exists()
        df = pd.read_parquet(path)
        assert set(df.columns) == _EXPECTED_COLUMNS
        assert cm.load("returns_by_regime") is not None

    def test_no_incumbent_imports(self):
        """No import of trading_crab_lib.asset_returns or trading_crab_lib.reporting
        anywhere in the module (D-01 constraint)."""
        import inspect

        from trading_crab_lib.platform.assets import returns as returns_module

        source = inspect.getsource(returns_module)
        assert "trading_crab_lib.asset_returns" not in source
        assert "trading_crab_lib.reporting" not in source
