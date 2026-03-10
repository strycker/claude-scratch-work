"""Unit tests for src/market_regime/clustering/kmeans.py"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from market_regime.clustering import (
    reduce_pca,
    evaluate_kmeans,
    pick_best_k,
    fit_clusters,
)


@pytest.fixture
def feature_df(quarterly_index):
    """70-quarter, 10-column feature matrix (no NaNs) for clustering tests."""
    rng = np.random.default_rng(42)
    n = 70
    index = pd.date_range("2000-03-31", periods=n, freq="QE")
    return pd.DataFrame(
        rng.standard_normal((n, 10)),
        index=index,
        columns=[f"f{i}" for i in range(10)],
    )


# ── reduce_pca ─────────────────────────────────────────────────────────────

class TestReducePca:
    def test_output_shape(self, feature_df):
        pca_df, pca, scaler = reduce_pca(feature_df, n_components=5)
        assert pca_df.shape == (len(feature_df), 5)

    def test_column_names(self, feature_df):
        pca_df, _, _ = reduce_pca(feature_df, n_components=3)
        assert list(pca_df.columns) == ["PC1", "PC2", "PC3"]

    def test_index_preserved(self, feature_df):
        pca_df, _, _ = reduce_pca(feature_df, n_components=5)
        pd.testing.assert_index_equal(pca_df.index, feature_df.index)

    def test_no_nans_in_output(self, feature_df):
        pca_df, _, _ = reduce_pca(feature_df, n_components=5)
        assert not pca_df.isna().any().any()

    def test_returns_fitted_objects(self, feature_df):
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        _, pca, scaler = reduce_pca(feature_df, n_components=5)
        assert isinstance(pca, PCA)
        assert isinstance(scaler, StandardScaler)


# ── evaluate_kmeans ────────────────────────────────────────────────────────

class TestEvaluateKmeans:
    def test_returns_dataframe_with_expected_cols(self, feature_df):
        pca_df, _, _ = reduce_pca(feature_df, n_components=3)
        scores = evaluate_kmeans(pca_df.values, k_range=range(2, 5), n_init=5)
        assert set(scores.columns) >= {"k", "silhouette", "calinski", "davies_bouldin", "inertia"}

    def test_one_row_per_k(self, feature_df):
        pca_df, _, _ = reduce_pca(feature_df, n_components=3)
        scores = evaluate_kmeans(pca_df.values, k_range=range(2, 6), n_init=5)
        assert len(scores) == 4

    def test_silhouette_between_neg1_and_1(self, feature_df):
        pca_df, _, _ = reduce_pca(feature_df, n_components=3)
        scores = evaluate_kmeans(pca_df.values, k_range=range(2, 5), n_init=5)
        assert (scores["silhouette"] >= -1).all()
        assert (scores["silhouette"] <= 1).all()


# ── pick_best_k ────────────────────────────────────────────────────────────

class TestPickBestK:
    def test_returns_highest_silhouette(self):
        scores = pd.DataFrame({
            "k":          [2, 3, 4, 5],
            "silhouette": [0.2, 0.5, 0.4, 0.3],
        })
        assert pick_best_k(scores, k_cap=10) == 3

    def test_cap_applied(self):
        scores = pd.DataFrame({
            "k":          [2, 3, 4, 5, 6, 7],
            "silhouette": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        })
        assert pick_best_k(scores, k_cap=5) == 5

    def test_cap_not_applied_when_best_below_cap(self):
        scores = pd.DataFrame({
            "k":          [2, 3, 4],
            "silhouette": [0.1, 0.5, 0.2],
        })
        assert pick_best_k(scores, k_cap=5) == 3


# ── fit_clusters ───────────────────────────────────────────────────────────

class TestFitClusters:
    def test_both_columns_present(self, feature_df):
        pca_df, _, _ = reduce_pca(feature_df, n_components=5)
        result = fit_clusters(pca_df, best_k=3, balanced_k=5, use_constrained=False)
        assert "cluster" in result.columns
        assert "balanced_cluster" in result.columns

    def test_cluster_values_are_integers(self, feature_df):
        pca_df, _, _ = reduce_pca(feature_df, n_components=5)
        result = fit_clusters(pca_df, best_k=3, balanced_k=5, use_constrained=False)
        assert np.issubdtype(result["cluster"].dtype, np.integer)
        assert np.issubdtype(result["balanced_cluster"].dtype, np.integer)

    def test_correct_number_of_unique_clusters(self, feature_df):
        pca_df, _, _ = reduce_pca(feature_df, n_components=5)
        result = fit_clusters(pca_df, best_k=3, balanced_k=4, use_constrained=False)
        assert result["cluster"].nunique() == 3
        assert result["balanced_cluster"].nunique() == 4

    def test_index_preserved(self, feature_df):
        pca_df, _, _ = reduce_pca(feature_df, n_components=5)
        result = fit_clusters(pca_df, best_k=3, balanced_k=5, use_constrained=False)
        pd.testing.assert_index_equal(result.index, pca_df.index)
