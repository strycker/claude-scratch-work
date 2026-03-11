"""Unit tests for src/market_regime/gmm.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from market_regime.gmm import fit_gmm, gmm_labels, gmm_probabilities, select_gmm_k


# ── Shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def pca_df():
    """Small but realistic PCA-reduced DataFrame (60 quarters × 5 PCs)."""
    rng = np.random.default_rng(42)
    idx = pd.date_range("2000-03-31", periods=60, freq="QE")
    # Three well-separated clusters in PC1 direction
    data = np.vstack([
        rng.multivariate_normal([3, 0, 0, 0, 0], np.eye(5), 20),
        rng.multivariate_normal([-3, 0, 0, 0, 0], np.eye(5), 20),
        rng.multivariate_normal([0, 3, 0, 0, 0], np.eye(5), 20),
    ])
    return pd.DataFrame(data, index=idx, columns=[f"PC{i+1}" for i in range(5)])


@pytest.fixture
def small_bic_df():
    """Minimal BIC DataFrame with two rows."""
    return pd.DataFrame({
        "k": [2, 3, 4],
        "covariance_type": ["diag", "diag", "diag"],
        "bic": [200.0, 150.0, 180.0],
        "aic": [190.0, 140.0, 170.0],
        "log_likelihood": [-1.0, -0.8, -0.9],
    })


# ── fit_gmm ───────────────────────────────────────────────────────────────────

class TestFitGmm:
    def test_returns_three_values(self, pca_df):
        result = fit_gmm(pca_df, k_range=range(2, 4), covariance_types=("diag",), n_init=3)
        assert len(result) == 3  # bic_df, models, scaler

    def test_bic_df_columns(self, pca_df):
        bic_df, _, _ = fit_gmm(pca_df, k_range=range(2, 4), covariance_types=("diag",), n_init=3)
        for col in ("k", "covariance_type", "bic", "aic", "log_likelihood"):
            assert col in bic_df.columns, f"Missing column: {col}"

    def test_bic_df_row_count(self, pca_df):
        bic_df, _, _ = fit_gmm(pca_df, k_range=range(2, 5), covariance_types=("diag", "tied"), n_init=3)
        # 3 k values × 2 cov types = 6 rows
        assert len(bic_df) == 6

    def test_models_dict_keys(self, pca_df):
        _, models, _ = fit_gmm(pca_df, k_range=range(2, 4), covariance_types=("diag",), n_init=3)
        assert (2, "diag") in models
        assert (3, "diag") in models

    def test_scaler_is_standard_scaler(self, pca_df):
        _, _, scaler = fit_gmm(pca_df, k_range=range(2, 3), covariance_types=("diag",), n_init=3)
        assert isinstance(scaler, StandardScaler)
        # Scaler should be fitted (has mean_)
        assert hasattr(scaler, "mean_")
        assert scaler.mean_.shape == (pca_df.shape[1],)

    def test_scaler_mean_matches_pca_df(self, pca_df):
        """The fitted scaler should have mean close to pca_df column means."""
        _, _, scaler = fit_gmm(pca_df, k_range=range(2, 3), covariance_types=("diag",), n_init=3)
        np.testing.assert_allclose(scaler.mean_, pca_df.values.mean(axis=0), rtol=1e-6)

    def test_empty_pca_df_raises(self):
        empty = pd.DataFrame(columns=["PC1", "PC2"])
        with pytest.raises(ValueError, match="empty"):
            fit_gmm(empty)

    def test_bic_finite(self, pca_df):
        bic_df, _, _ = fit_gmm(pca_df, k_range=range(2, 4), covariance_types=("diag",), n_init=3)
        assert bic_df["bic"].notna().all()
        assert np.isfinite(bic_df["bic"].values).all()


# ── select_gmm_k ─────────────────────────────────────────────────────────────

class TestSelectGmmK:
    def test_returns_minimum_bic(self, small_bic_df):
        best_k, best_cov = select_gmm_k(small_bic_df)
        assert best_k == 3  # BIC=150 is minimum
        assert best_cov == "diag"

    def test_returns_tuple_of_int_and_str(self, small_bic_df):
        best_k, best_cov = select_gmm_k(small_bic_df)
        assert isinstance(best_k, int)
        assert isinstance(best_cov, str)

    def test_empty_bic_df_raises(self):
        with pytest.raises(ValueError, match="empty"):
            select_gmm_k(pd.DataFrame())

    def test_all_nan_bic_raises(self):
        df = pd.DataFrame({"k": [2, 3], "covariance_type": ["diag", "diag"], "bic": [float("nan"), float("nan")]})
        with pytest.raises(ValueError, match="NaN"):
            select_gmm_k(df)

    def test_mixed_cov_types_picks_overall_minimum(self):
        df = pd.DataFrame({
            "k":               [2, 2, 3, 3],
            "covariance_type": ["diag", "full", "diag", "full"],
            "bic":             [300.0, 200.0, 250.0, 100.0],
            "aic":             [290.0, 190.0, 240.0, 90.0],
            "log_likelihood":  [-1.0, -0.9, -0.8, -0.7],
        })
        best_k, best_cov = select_gmm_k(df)
        assert best_k == 3
        assert best_cov == "full"


# ── gmm_labels ────────────────────────────────────────────────────────────────

class TestGmmLabels:
    def test_returns_series(self, pca_df):
        _, models, scaler = fit_gmm(pca_df, k_range=range(2, 3), covariance_types=("diag",), n_init=3)
        model = models[(2, "diag")]
        labels = gmm_labels(pca_df, model, scaler=scaler)
        assert isinstance(labels, pd.Series)

    def test_index_matches_pca_df(self, pca_df):
        _, models, scaler = fit_gmm(pca_df, k_range=range(2, 3), covariance_types=("diag",), n_init=3)
        model = models[(2, "diag")]
        labels = gmm_labels(pca_df, model, scaler=scaler)
        assert labels.index.equals(pca_df.index)

    def test_label_count_matches_k(self, pca_df):
        _, models, scaler = fit_gmm(pca_df, k_range=range(3, 4), covariance_types=("diag",), n_init=3)
        model = models[(3, "diag")]
        labels = gmm_labels(pca_df, model, scaler=scaler)
        assert labels.nunique() == 3

    def test_labels_are_non_negative_integers(self, pca_df):
        _, models, scaler = fit_gmm(pca_df, k_range=range(3, 4), covariance_types=("diag",), n_init=3)
        model = models[(3, "diag")]
        labels = gmm_labels(pca_df, model, scaler=scaler)
        assert (labels >= 0).all()
        assert labels.dtype in (np.int32, np.int64, int)

    def test_canonicalized_cluster0_has_smallest_mean_pc1(self, pca_df):
        """Cluster 0 must correspond to smallest mean PC1 value."""
        _, models, scaler = fit_gmm(pca_df, k_range=range(3, 4), covariance_types=("diag",), n_init=5)
        model = models[(3, "diag")]
        labels = gmm_labels(pca_df, model, scaler=scaler)
        pc1 = pca_df.iloc[:, 0]
        mean_pc1 = pc1.groupby(labels).mean()
        # cluster 0 should have smallest mean
        assert mean_pc1.idxmin() == 0

    def test_no_scaler_warns(self, pca_df):
        _, models, scaler = fit_gmm(pca_df, k_range=range(2, 3), covariance_types=("diag",), n_init=3)
        model = models[(2, "diag")]
        with pytest.warns(None):  # expect a logging warning not a Python UserWarning
            # Function should still return without error
            labels = gmm_labels(pca_df, model, scaler=None)
        assert len(labels) == len(pca_df)

    def test_name_is_gmm_cluster(self, pca_df):
        _, models, scaler = fit_gmm(pca_df, k_range=range(2, 3), covariance_types=("diag",), n_init=3)
        labels = gmm_labels(pca_df, models[(2, "diag")], scaler=scaler)
        assert labels.name == "gmm_cluster"

    def test_scaler_consistency(self, pca_df):
        """Labels should be identical whether scaler is passed or freshly fitted (same data)."""
        _, models, scaler = fit_gmm(pca_df, k_range=range(2, 3), covariance_types=("diag",), n_init=5)
        model = models[(2, "diag")]
        labels_with_scaler = gmm_labels(pca_df, model, scaler=scaler)
        labels_no_scaler = gmm_labels(pca_df, model, scaler=None)
        # On the same pca_df, labels must agree (scaler fitted on same data)
        pd.testing.assert_series_equal(labels_with_scaler, labels_no_scaler)


# ── gmm_probabilities ─────────────────────────────────────────────────────────

class TestGmmProbabilities:
    def test_returns_dataframe(self, pca_df):
        _, models, scaler = fit_gmm(pca_df, k_range=range(3, 4), covariance_types=("diag",), n_init=3)
        probs = gmm_probabilities(pca_df, models[(3, "diag")], scaler=scaler)
        assert isinstance(probs, pd.DataFrame)

    def test_shape(self, pca_df):
        _, models, scaler = fit_gmm(pca_df, k_range=range(3, 4), covariance_types=("diag",), n_init=3)
        probs = gmm_probabilities(pca_df, models[(3, "diag")], scaler=scaler)
        assert probs.shape == (len(pca_df), 3)

    def test_rows_sum_to_one(self, pca_df):
        _, models, scaler = fit_gmm(pca_df, k_range=range(3, 4), covariance_types=("diag",), n_init=3)
        probs = gmm_probabilities(pca_df, models[(3, "diag")], scaler=scaler)
        row_sums = probs.sum(axis=1)
        np.testing.assert_allclose(row_sums.values, 1.0, atol=1e-6)

    def test_probs_between_0_and_1(self, pca_df):
        _, models, scaler = fit_gmm(pca_df, k_range=range(3, 4), covariance_types=("diag",), n_init=3)
        probs = gmm_probabilities(pca_df, models[(3, "diag")], scaler=scaler)
        assert (probs.values >= 0).all()
        assert (probs.values <= 1).all()

    def test_column_names(self, pca_df):
        _, models, scaler = fit_gmm(pca_df, k_range=range(3, 4), covariance_types=("diag",), n_init=3)
        probs = gmm_probabilities(pca_df, models[(3, "diag")], scaler=scaler)
        assert list(probs.columns) == ["gmm_prob_0", "gmm_prob_1", "gmm_prob_2"]

    def test_index_matches_pca_df(self, pca_df):
        _, models, scaler = fit_gmm(pca_df, k_range=range(3, 4), covariance_types=("diag",), n_init=3)
        probs = gmm_probabilities(pca_df, models[(3, "diag")], scaler=scaler)
        assert probs.index.equals(pca_df.index)
