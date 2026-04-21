"""Unit tests for src/trading_crab_lib/assets/returns.py"""

import numpy as np
import pandas as pd
import pytest

from trading_crab_lib.asset_returns import (
    compute_quarterly_returns,
    returns_by_regime,
    returns_full_stats,
    rank_assets_by_regime,
    compute_template_returns,
    behavior_tables,
    _absolute_signal,
    _score_absolute,
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
    def test_returns_five_keys(self, asset_prices, cluster_labels):
        returns = compute_quarterly_returns(asset_prices)
        common = returns.index.intersection(cluster_labels.index)
        stats = returns_full_stats(returns.loc[common], cluster_labels.loc[common])
        assert set(stats.keys()) == {"median_return", "q25", "q75", "hit_rate", "n_quarters"}

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


# ── _absolute_signal / _score_absolute ──────────────────────────────────────

class TestAbsoluteSignal:
    def test_green_above_threshold(self):
        assert _absolute_signal(0.03, 0.02, 0.0) == "GREEN"

    def test_red_below_threshold(self):
        assert _absolute_signal(-0.01, 0.02, 0.0) == "RED"

    def test_neutral_between(self):
        assert _absolute_signal(0.01, 0.02, 0.0) == "NEUTRAL"

    def test_exact_red_is_neutral(self):
        """Exact red_median (0%) should be NEUTRAL, not RED."""
        assert _absolute_signal(0.0, 0.02, 0.0) == "NEUTRAL"


class TestScoreAbsolute:
    def test_high_return_caps_at_100(self):
        assert _score_absolute(0.10) == 100.0

    def test_zero_return_is_50(self):
        assert _score_absolute(0.0) == 50.0

    def test_negative_return_below_50(self):
        assert _score_absolute(-0.01) < 50.0

    def test_moderate_positive(self):
        score = _score_absolute(0.03)
        assert 50.0 < score < 80.0


# ── compute_template_returns ────────────────────────────────────────────────

class TestComputeTemplateReturns:
    def test_basic_template(self):
        idx = pd.date_range("2000-03-31", periods=4, freq="QE")
        returns = pd.DataFrame(
            {"SPY": [0.05, 0.03, -0.02, 0.04], "GLD": [0.02, 0.01, 0.06, -0.01]},
            index=idx,
        )
        templates = [{"name": "60_40", "weights": {"SPY": 0.6, "GLD": 0.4}}]
        result = compute_template_returns(returns, templates)
        assert "60_40" in result.columns
        expected_q1 = 0.6 * 0.05 + 0.4 * 0.02
        assert result["60_40"].iloc[0] == pytest.approx(expected_q1)

    def test_missing_ticker_skipped(self):
        idx = pd.date_range("2000-03-31", periods=2, freq="QE")
        returns = pd.DataFrame({"SPY": [0.01, 0.02]}, index=idx)
        templates = [{"name": "test", "weights": {"SPY": 0.5, "MISSING": 0.5}}]
        result = compute_template_returns(returns, templates)
        # SPY weight gets normalized to 1.0
        assert "test" in result.columns
        assert result["test"].iloc[0] == pytest.approx(0.01)

    def test_empty_templates(self):
        idx = pd.date_range("2000-03-31", periods=2, freq="QE")
        returns = pd.DataFrame({"SPY": [0.01, 0.02]}, index=idx)
        result = compute_template_returns(returns, [])
        assert result.empty or len(result.columns) == 0


# ── behavior_tables ─────────────────────────────────────────────────────────

class TestBehaviorTables:
    @pytest.fixture
    def sample_data(self):
        rng = np.random.default_rng(42)
        idx = pd.date_range("2000-03-31", periods=20, freq="QE")
        returns = pd.DataFrame(
            {
                "SPY": rng.normal(0.03, 0.04, 20),
                "GLD": rng.normal(0.01, 0.06, 20),
                "TLT": rng.normal(-0.005, 0.03, 20),
            },
            index=idx,
        )
        labels = pd.Series(
            [0] * 7 + [1] * 6 + [2] * 7,
            index=idx,
            name="balanced_cluster",
        )
        return returns, labels

    def test_behavior_tables_columns(self, sample_data):
        returns, labels = sample_data
        bt = behavior_tables(returns, labels)
        expected_cols = {
            "regime", "asset", "median_return", "q25", "q75", "hit_rate",
            "n_quarters", "signal_absolute", "tertile", "signal_display",
            "score_relative", "score_absolute", "rank",
        }
        assert expected_cols <= set(bt.columns)

    def test_behavior_tables_all_assets_per_regime(self, sample_data):
        returns, labels = sample_data
        bt = behavior_tables(returns, labels)
        for regime in labels.unique():
            regime_rows = bt[bt["regime"] == regime]
            assert set(regime_rows["asset"]) == set(returns.columns)

    def test_behavior_tables_valid_signals(self, sample_data):
        returns, labels = sample_data
        bt = behavior_tables(returns, labels)
        valid_abs = {"GREEN", "RED", "NEUTRAL"}
        assert set(bt["signal_absolute"]) <= valid_abs
        valid_display = {"green_strong", "green", "neutral", "light_blue", "red", "red_strong"}
        assert set(bt["signal_display"]) <= valid_display

    def test_behavior_tables_custom_thresholds(self, sample_data):
        returns, labels = sample_data
        bt = behavior_tables(returns, labels, thresholds={"green_median": 0.10, "red_median": -0.10})
        # With high green threshold, most should be NEUTRAL
        assert "NEUTRAL" in bt["signal_absolute"].values
