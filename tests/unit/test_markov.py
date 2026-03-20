"""Tests for Markov regime-switching model module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_crab_lib.markov import (
    _STATSMODELS_AVAILABLE,
    compare_markov_kmeans,
    fit_markov_switching,
    markov_labels,
    markov_probabilities,
)

pytestmark = pytest.mark.skipif(
    not _STATSMODELS_AVAILABLE, reason="statsmodels not installed"
)


@pytest.fixture
def gdp_growth() -> pd.Series:
    """Synthetic GDP growth with clear 2-regime structure."""
    rng = np.random.RandomState(42)
    n = 120
    idx = pd.date_range("1990-01-01", periods=n, freq="QS")

    # Two regimes: expansion (mean=3.0) and recession (mean=-1.5)
    regime = np.array([0] * 40 + [1] * 20 + [0] * 40 + [1] * 20)
    means = {0: 3.0, 1: -1.5}
    data = np.array([means[r] for r in regime]) + rng.randn(n) * 0.5
    return pd.Series(data, index=idx, name="gdp_growth")


@pytest.fixture
def gdp_result(gdp_growth) -> dict:
    """Pre-fitted Markov switching result."""
    return fit_markov_switching(gdp_growth, k_regimes=2)


class TestFitMarkovSwitching:
    def test_returns_dict(self, gdp_growth):
        result = fit_markov_switching(gdp_growth)
        assert isinstance(result, dict)
        assert "model" in result
        assert "regime_means" in result
        assert "transition_matrix" in result
        assert "k_regimes" in result
        assert "series_name" in result

    def test_k_regimes(self, gdp_growth):
        result = fit_markov_switching(gdp_growth, k_regimes=2)
        assert result["k_regimes"] == 2
        assert len(result["regime_means"]) == 2

    def test_transition_matrix_shape(self, gdp_growth):
        result = fit_markov_switching(gdp_growth, k_regimes=2)
        tm = result["transition_matrix"]
        assert tm.shape == (2, 2)

    def test_regime_means_reasonable(self, gdp_growth):
        result = fit_markov_switching(gdp_growth, k_regimes=2)
        means = sorted(result["regime_means"].values())
        # Should find a low-mean regime and a high-mean regime
        assert means[0] < 0  # recession-like
        assert means[1] > 1  # expansion-like

    def test_series_name_preserved(self, gdp_growth):
        result = fit_markov_switching(gdp_growth)
        assert result["series_name"] == "gdp_growth"

    def test_too_few_observations_raises(self):
        short = pd.Series([1.0, 2.0, 3.0], name="short")
        with pytest.raises(ValueError, match="≥20"):
            fit_markov_switching(short)

    def test_nan_handling(self, gdp_growth):
        with_nans = gdp_growth.copy()
        with_nans.iloc[5] = np.nan
        with_nans.iloc[10] = np.nan
        result = fit_markov_switching(with_nans)
        assert result["k_regimes"] == 2

    def test_three_regimes(self, gdp_growth):
        result = fit_markov_switching(gdp_growth, k_regimes=3)
        assert result["k_regimes"] == 3
        assert len(result["regime_means"]) == 3


class TestMarkovLabels:
    def test_returns_series(self, gdp_result):
        labels = markov_labels(gdp_result)
        assert isinstance(labels, pd.Series)
        assert labels.name == "markov_regime"

    def test_label_values(self, gdp_result):
        labels = markov_labels(gdp_result)
        assert set(labels.unique()).issubset({0, 1})

    def test_canonicalization_regime0_lower_mean(self, gdp_result):
        labels = markov_labels(gdp_result)
        # Regime 0 should have lower mean → corresponds to recession
        means = gdp_result["regime_means"]
        sorted_means = sorted(means.values())
        # The lowest mean regime should be mapped to 0
        assert sorted_means[0] < sorted_means[1]

    def test_length_matches_input(self, gdp_result, gdp_growth):
        labels = markov_labels(gdp_result)
        # Length matches non-NaN observations
        assert len(labels) > 0


class TestMarkovProbabilities:
    def test_returns_dataframe(self, gdp_result):
        probs = markov_probabilities(gdp_result)
        assert isinstance(probs, pd.DataFrame)

    def test_columns(self, gdp_result):
        probs = markov_probabilities(gdp_result)
        assert list(probs.columns) == ["markov_prob_0", "markov_prob_1"]

    def test_rows_sum_to_one(self, gdp_result):
        probs = markov_probabilities(gdp_result)
        row_sums = probs.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-4)

    def test_values_between_0_and_1(self, gdp_result):
        probs = markov_probabilities(gdp_result)
        assert (probs >= -1e-10).all().all()
        assert (probs <= 1 + 1e-10).all().all()


class TestCompareMarkovKmeans:
    def test_returns_crosstab(self, gdp_result, gdp_growth):
        # Create fake KMeans labels aligned with gdp_growth
        rng = np.random.RandomState(0)
        labels = markov_labels(gdp_result)
        kmeans_labels = pd.Series(
            rng.randint(0, 5, size=len(labels)),
            index=labels.index,
            name="balanced_cluster",
        )
        ct = compare_markov_kmeans(gdp_result, kmeans_labels)
        assert isinstance(ct, pd.DataFrame)
        assert ct.shape[0] == 2  # 2 Markov regimes

    def test_no_overlap_raises(self, gdp_result):
        bad_idx = pd.date_range("2100-01-01", periods=10, freq="QS")
        kmeans_labels = pd.Series([0] * 10, index=bad_idx, name="cluster")
        with pytest.raises(ValueError, match="No overlapping"):
            compare_markov_kmeans(gdp_result, kmeans_labels)
