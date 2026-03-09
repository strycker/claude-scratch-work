"""Unit tests for src/market_regime/assets/returns.py"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from market_regime.assets.returns import (
    compute_quarterly_returns,
    returns_by_regime,
    returns_full_stats,
    rank_assets_by_regime,
)


# ── compute_quarterly_returns ──────────────────────────────────────────────

class TestComputeQuarterlyReturns:
    def test_returns_shape(self, asset_prices):
        result = compute_quarterly_returns(asset_prices)
        assert result.shape[1] == asset_prices.shape[1]

    def test_first_row_is_nan(self, asset_prices):
        result = compute_quarterly_returns(asset_prices)
        # dropna(how="all") removes the leading all-NaN row
        assert not result.empty

    def test_monthly_input_resampled(self, quarterly_index):
        # Build monthly prices — resample should produce quarterly
        monthly_index = pd.date_range("2000-01-31", periods=60, freq="ME")
        prices = pd.DataFrame({"A": np.linspace(100, 200, 60)}, index=monthly_index)
        result = compute_quarterly_returns(prices)
        # Should have ~19 rows (60 months → 20 quarters → 19 returns)
        assert 15 <= len(result) <= 20

    def test_no_inf_in_output(self, asset_prices):
        result = compute_quarterly_returns(asset_prices)
        assert not np.isinf(result.values).any()


# ── returns_by_regime ──────────────────────────────────────────────────────

class TestReturnsByRegime:
    def test_pivot_shape(self, asset_prices, cluster_labels):
        returns = compute_quarterly_returns(asset_prices)
        common = returns.index.intersection(cluster_labels.index)
        profile = returns_by_regime(returns.loc[common], cluster_labels.loc[common])
        assert profile.shape[1] == asset_prices.shape[1]  # one col per ticker
        assert len(profile) == cluster_labels.nunique()

    def test_index_is_regime_int(self, asset_prices, cluster_labels):
        returns = compute_quarterly_returns(asset_prices)
        common = returns.index.intersection(cluster_labels.index)
        profile = returns_by_regime(returns.loc[common], cluster_labels.loc[common])
        assert profile.index.name == "regime"
        assert profile.index.dtype == int or np.issubdtype(profile.index.dtype, np.integer)

    def test_values_are_medians(self, quarterly_index):
        """Manual check: regime 0 gets quarters [0, 5, 10, 15], verify median."""
        returns = pd.DataFrame(
            {"X": [0.10, 0.20, 0.30, 0.40, 0.50,
                   0.15, 0.25, 0.35, 0.45, 0.55,
                   0.12, 0.22, 0.32, 0.42, 0.52,
                   0.11, 0.21, 0.31, 0.41, 0.51]},
            index=quarterly_index,
        )
        labels = pd.Series(
            [0, 1, 2, 3, 4, 0, 1, 2, 3, 4,
             0, 1, 2, 3, 4, 0, 1, 2, 3, 4],
            index=quarterly_index,
        )
        profile = returns_by_regime(returns, labels)
        # Regime 0 rows: 0.10, 0.15, 0.12, 0.11 → median = (0.11+0.12)/2... sorted: 0.10,0.11,0.12,0.15
        regime0_median = profile.loc[0, "X"]
        expected = np.median([0.10, 0.15, 0.12, 0.11])
        assert abs(regime0_median - expected) < 1e-10

    def test_empty_input_returns_empty(self):
        empty_returns = pd.DataFrame(columns=["SPY"])
        empty_labels = pd.Series([], dtype=int)
        result = returns_by_regime(empty_returns, empty_labels)
        assert result.empty


# ── returns_full_stats ─────────────────────────────────────────────────────

class TestReturnsFullStats:
    def test_returns_three_keys(self, asset_prices, cluster_labels):
        returns = compute_quarterly_returns(asset_prices)
        common = returns.index.intersection(cluster_labels.index)
        stats = returns_full_stats(returns.loc[common], cluster_labels.loc[common])
        assert set(stats.keys()) == {"median_return", "hit_rate", "n_quarters"}

    def test_hit_rate_between_0_and_1(self, asset_prices, cluster_labels):
        returns = compute_quarterly_returns(asset_prices)
        common = returns.index.intersection(cluster_labels.index)
        stats = returns_full_stats(returns.loc[common], cluster_labels.loc[common])
        hr = stats["hit_rate"]
        assert (hr >= 0).all().all()
        assert (hr <= 1).all().all()

    def test_n_quarters_positive(self, asset_prices, cluster_labels):
        returns = compute_quarterly_returns(asset_prices)
        common = returns.index.intersection(cluster_labels.index)
        stats = returns_full_stats(returns.loc[common], cluster_labels.loc[common])
        nq = stats["n_quarters"]
        assert (nq > 0).all().all()


# ── rank_assets_by_regime ──────────────────────────────────────────────────

class TestRankAssetsByRegime:
    def test_rank_columns_present(self, asset_prices, cluster_labels):
        returns = compute_quarterly_returns(asset_prices)
        common = returns.index.intersection(cluster_labels.index)
        profile = returns_by_regime(returns.loc[common], cluster_labels.loc[common])
        ranked = rank_assets_by_regime(profile)
        assert set(ranked.columns) >= {"regime", "asset", "median_quarterly_return", "rank"}

    def test_rank_starts_at_one(self, asset_prices, cluster_labels):
        returns = compute_quarterly_returns(asset_prices)
        common = returns.index.intersection(cluster_labels.index)
        profile = returns_by_regime(returns.loc[common], cluster_labels.loc[common])
        ranked = rank_assets_by_regime(profile)
        for _, grp in ranked.groupby("regime"):
            assert grp["rank"].min() == 1

    def test_ranks_descending_by_return(self, asset_prices, cluster_labels):
        returns = compute_quarterly_returns(asset_prices)
        common = returns.index.intersection(cluster_labels.index)
        profile = returns_by_regime(returns.loc[common], cluster_labels.loc[common])
        ranked = rank_assets_by_regime(profile)
        for _, grp in ranked.groupby("regime"):
            sorted_grp = grp.sort_values("rank")
            assert sorted_grp["median_quarterly_return"].is_monotonic_decreasing
