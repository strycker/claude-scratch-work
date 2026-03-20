"""Tests for HMM regime detection module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_crab_lib.hmm import (
    _HMM_AVAILABLE,
    fit_hmm,
    hmm_labels,
    hmm_probabilities,
    hmm_transition_matrix,
    select_hmm_k,
)

pytestmark = pytest.mark.skipif(not _HMM_AVAILABLE, reason="hmmlearn not installed")


@pytest.fixture
def pca_df() -> pd.DataFrame:
    """Synthetic PCA-like DataFrame with temporal structure (regime switches)."""
    rng = np.random.RandomState(42)
    n = 120
    idx = pd.date_range("1990-01-01", periods=n, freq="QS")

    # Two-regime synthetic data: regime switches every ~30 quarters
    regime = np.array([0] * 30 + [1] * 30 + [0] * 30 + [1] * 30)
    means = {0: np.array([1.0, -0.5, 0.2, 0.0, 0.3]),
             1: np.array([-1.0, 0.5, -0.2, 0.0, -0.3])}

    data = np.array([means[r] for r in regime]) + rng.randn(n, 5) * 0.3
    return pd.DataFrame(data, index=idx, columns=[f"PC{i+1}" for i in range(5)])


@pytest.fixture
def pca_df_small() -> pd.DataFrame:
    """Small PCA DataFrame for basic smoke tests."""
    rng = np.random.RandomState(123)
    idx = pd.date_range("2000-01-01", periods=40, freq="QS")
    data = rng.randn(40, 5)
    return pd.DataFrame(data, index=idx, columns=[f"PC{i+1}" for i in range(5)])


class TestFitHMM:
    def test_returns_three_items(self, pca_df):
        scores, models, scaler = fit_hmm(pca_df, k_range=range(2, 4), n_restarts=3)
        assert isinstance(scores, pd.DataFrame)
        assert isinstance(models, dict)
        assert scaler is not None

    def test_scores_columns(self, pca_df):
        scores, _, _ = fit_hmm(pca_df, k_range=range(2, 4), n_restarts=3)
        assert "k" in scores.columns
        assert "log_likelihood" in scores.columns
        assert "bic" in scores.columns
        assert "aic" in scores.columns
        assert "converged" in scores.columns

    def test_one_model_per_k(self, pca_df):
        scores, models, _ = fit_hmm(pca_df, k_range=range(2, 5), n_restarts=3)
        for k in range(2, 5):
            assert k in models

    def test_empty_raises(self):
        empty = pd.DataFrame(columns=["PC1", "PC2"])
        with pytest.raises(ValueError, match="empty"):
            fit_hmm(empty)

    def test_default_k_range(self, pca_df_small):
        scores, models, _ = fit_hmm(pca_df_small, n_restarts=2)
        ks = set(scores["k"].tolist())
        assert 2 in ks
        assert 7 in ks


class TestSelectHMMK:
    def test_returns_int(self, pca_df_small):
        scores, _, _ = fit_hmm(pca_df_small, k_range=range(2, 5), n_restarts=2)
        best_k = select_hmm_k(scores)
        assert isinstance(best_k, int)
        assert 2 <= best_k <= 4

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            select_hmm_k(pd.DataFrame())


class TestHMMLabels:
    def test_returns_series(self, pca_df):
        _, models, scaler = fit_hmm(pca_df, k_range=range(2, 3), n_restarts=3)
        labels = hmm_labels(pca_df, models[2], scaler=scaler)
        assert isinstance(labels, pd.Series)
        assert labels.name == "hmm_state"
        assert len(labels) == len(pca_df)

    def test_label_values(self, pca_df):
        _, models, scaler = fit_hmm(pca_df, k_range=range(2, 3), n_restarts=3)
        labels = hmm_labels(pca_df, models[2], scaler=scaler)
        assert set(labels.unique()) == {0, 1}

    def test_canonicalization_state0_has_smaller_pc1(self, pca_df):
        _, models, scaler = fit_hmm(pca_df, k_range=range(2, 3), n_restarts=3)
        labels = hmm_labels(pca_df, models[2], scaler=scaler)
        pc1 = pca_df.iloc[:, 0]
        mean_pc1_0 = pc1[labels == 0].mean()
        mean_pc1_1 = pc1[labels == 1].mean()
        assert mean_pc1_0 < mean_pc1_1

    def test_without_scaler_still_works(self, pca_df):
        _, models, _ = fit_hmm(pca_df, k_range=range(2, 3), n_restarts=3)
        labels = hmm_labels(pca_df, models[2])
        assert len(labels) == len(pca_df)
        assert set(labels.unique()) == {0, 1}

    def test_index_preserved(self, pca_df):
        _, models, scaler = fit_hmm(pca_df, k_range=range(2, 3), n_restarts=3)
        labels = hmm_labels(pca_df, models[2], scaler=scaler)
        assert labels.index.equals(pca_df.index)


class TestHMMProbabilities:
    def test_returns_dataframe(self, pca_df):
        _, models, scaler = fit_hmm(pca_df, k_range=range(2, 3), n_restarts=3)
        probs = hmm_probabilities(pca_df, models[2], scaler=scaler)
        assert isinstance(probs, pd.DataFrame)
        assert probs.shape == (len(pca_df), 2)

    def test_rows_sum_to_one(self, pca_df):
        _, models, scaler = fit_hmm(pca_df, k_range=range(2, 3), n_restarts=3)
        probs = hmm_probabilities(pca_df, models[2], scaler=scaler)
        row_sums = probs.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_column_names(self, pca_df):
        _, models, scaler = fit_hmm(pca_df, k_range=range(3, 4), n_restarts=3)
        probs = hmm_probabilities(pca_df, models[3], scaler=scaler)
        assert list(probs.columns) == ["hmm_prob_0", "hmm_prob_1", "hmm_prob_2"]

    def test_values_between_0_and_1(self, pca_df):
        _, models, scaler = fit_hmm(pca_df, k_range=range(2, 3), n_restarts=3)
        probs = hmm_probabilities(pca_df, models[2], scaler=scaler)
        assert (probs >= 0).all().all()
        assert (probs <= 1).all().all()


class TestHMMTransitionMatrix:
    def test_shape(self, pca_df):
        _, models, _ = fit_hmm(pca_df, k_range=range(3, 4), n_restarts=3)
        tm = hmm_transition_matrix(models[3])
        assert tm.shape == (3, 3)

    def test_rows_sum_to_one(self, pca_df):
        _, models, _ = fit_hmm(pca_df, k_range=range(2, 3), n_restarts=3)
        tm = hmm_transition_matrix(models[2])
        row_sums = tm.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_labels(self, pca_df):
        _, models, _ = fit_hmm(pca_df, k_range=range(2, 3), n_restarts=3)
        tm = hmm_transition_matrix(models[2])
        assert list(tm.index) == ["state_0", "state_1"]
        assert list(tm.columns) == ["state_0", "state_1"]
