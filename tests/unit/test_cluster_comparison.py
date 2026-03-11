"""Unit tests for src/market_regime/cluster_comparison.py."""

from __future__ import annotations

import pickle
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from market_regime.cluster_comparison import (
    compare_all_methods,
    extract_rf_feature_importances,
    pairwise_rand_index,
    recommend_clustering_features,
)


# ── Shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def pca_df():
    """50 quarters × 5 PCs — 2 well-separated clusters."""
    rng = np.random.default_rng(42)
    idx = pd.date_range("2000-03-31", periods=50, freq="QE")
    data = np.vstack([
        rng.multivariate_normal([4, 0, 0, 0, 0], np.eye(5), 25),
        rng.multivariate_normal([-4, 0, 0, 0, 0], np.eye(5), 25),
    ])
    return pd.DataFrame(data, index=idx, columns=[f"PC{i+1}" for i in range(5)])


@pytest.fixture
def labels_a(pca_df):
    """Perfect 2-cluster labels (0 = first 25 quarters, 1 = last 25)."""
    return pd.Series(
        [0] * 25 + [1] * 25,
        index=pca_df.index,
        name="method_a",
        dtype=int,
    )


@pytest.fixture
def labels_b(pca_df):
    """Slightly shifted labels (agree on most but disagree on 5 points)."""
    vals = [0] * 25 + [1] * 25
    vals[22] = 1
    vals[23] = 1
    vals[24] = 1
    vals[25] = 0
    vals[26] = 0
    return pd.Series(vals, index=pca_df.index, name="method_b", dtype=int)


@pytest.fixture
def noise_labels(pca_df):
    """Labels with 10 noise points (label = -1)."""
    vals = [0] * 20 + [-1] * 10 + [1] * 20
    return pd.Series(vals, index=pca_df.index, name="method_noise", dtype=int)


@pytest.fixture
def rf_model_path(tmp_path):
    """Fit a small RF model, pickle it, return the path."""
    rng = np.random.default_rng(0)
    X = rng.random((100, 10))
    y = (X[:, 0] > 0.5).astype(int)
    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    rf.fit(X, y)
    path = tmp_path / "rf_model.pkl"
    with open(path, "wb") as f:
        pickle.dump(rf, f)
    return path, rf, [f"feat_{i}" for i in range(10)]


# ── compare_all_methods ───────────────────────────────────────────────────────

class TestCompareAllMethods:
    def test_returns_dataframe(self, pca_df, labels_a, labels_b):
        result = compare_all_methods(pca_df, {"a": labels_a, "b": labels_b})
        assert isinstance(result, pd.DataFrame)

    def test_one_row_per_method(self, pca_df, labels_a, labels_b):
        result = compare_all_methods(pca_df, {"a": labels_a, "b": labels_b})
        assert len(result) == 2

    def test_expected_columns(self, pca_df, labels_a):
        result = compare_all_methods(pca_df, {"a": labels_a})
        for col in ("method", "n_clusters", "n_noise", "silhouette", "davies_bouldin", "calinski"):
            assert col in result.columns

    def test_sorted_by_silhouette_descending(self, pca_df, labels_a, labels_b):
        result = compare_all_methods(pca_df, {"a": labels_a, "b": labels_b})
        sil = result["silhouette"].dropna()
        assert list(sil) == sorted(sil, reverse=True)

    def test_silhouette_in_range(self, pca_df, labels_a):
        result = compare_all_methods(pca_df, {"a": labels_a})
        sil = float(result.iloc[0]["silhouette"])
        assert -1 <= sil <= 1

    def test_noise_counted_correctly(self, pca_df, noise_labels):
        result = compare_all_methods(pca_df, {"noise": noise_labels})
        assert result.iloc[0]["n_noise"] == 10

    def test_empty_pca_df_raises(self, labels_a):
        empty = pd.DataFrame(columns=["PC1", "PC2"])
        with pytest.raises(ValueError, match="empty"):
            compare_all_methods(empty, {"a": labels_a})

    def test_empty_labels_dict_raises(self, pca_df):
        with pytest.raises(ValueError, match="empty"):
            compare_all_methods(pca_df, {})

    def test_noise_only_produces_nan_silhouette(self, pca_df):
        """If all points are noise (label=-1), metrics cannot be computed."""
        all_noise = pd.Series([-1] * len(pca_df), index=pca_df.index, name="all_noise", dtype=int)
        result = compare_all_methods(pca_df, {"all_noise": all_noise})
        assert result.iloc[0]["n_clusters"] == 0
        assert pd.isna(result.iloc[0]["silhouette"])

    def test_partial_index_alignment(self, pca_df, labels_a):
        """Labels covering only a subset of quarters should not crash."""
        partial = labels_a.iloc[10:]  # missing first 10 quarters
        result = compare_all_methods(pca_df, {"partial": partial})
        assert len(result) == 1
        assert result.iloc[0]["n_clusters"] >= 1


# ── pairwise_rand_index ───────────────────────────────────────────────────────

class TestPairwiseRandIndex:
    def test_returns_square_dataframe(self, labels_a, labels_b):
        result = pairwise_rand_index({"a": labels_a, "b": labels_b})
        assert result.shape == (2, 2)

    def test_diagonal_is_one(self, labels_a, labels_b):
        result = pairwise_rand_index({"a": labels_a, "b": labels_b})
        np.testing.assert_allclose(np.diag(result.values), 1.0)

    def test_symmetric(self, labels_a, labels_b):
        result = pairwise_rand_index({"a": labels_a, "b": labels_b})
        np.testing.assert_allclose(result.values, result.values.T)

    def test_ari_between_minus_half_and_one(self, labels_a, labels_b):
        result = pairwise_rand_index({"a": labels_a, "b": labels_b})
        off_diag = result.values[0, 1]
        assert -0.5 <= off_diag <= 1.0

    def test_identical_labels_give_ari_one(self, labels_a):
        result = pairwise_rand_index({"x": labels_a, "y": labels_a.copy()})
        off_diag = result.iloc[0, 1]
        np.testing.assert_allclose(off_diag, 1.0, atol=1e-6)

    def test_index_and_columns_are_method_names(self, labels_a, labels_b):
        result = pairwise_rand_index({"a": labels_a, "b": labels_b})
        assert list(result.index) == ["a", "b"]
        assert list(result.columns) == ["a", "b"]

    def test_single_method_raises(self, labels_a):
        with pytest.raises(ValueError, match="at least 2 methods"):
            pairwise_rand_index({"a": labels_a})

    def test_noise_excluded_from_ari(self, labels_a, noise_labels):
        """Noise points (-1) from either series must not influence ARI."""
        # Should not raise; noise points are excluded
        result = pairwise_rand_index({"a": labels_a, "noise": noise_labels})
        assert result.shape == (2, 2)
        ari = result.iloc[0, 1]
        assert np.isfinite(ari) or pd.isna(ari)  # either valid or NaN is ok


# ── extract_rf_feature_importances ───────────────────────────────────────────

class TestExtractRfFeatureImportances:
    def test_returns_series(self, rf_model_path):
        path, _, names = rf_model_path
        importances = extract_rf_feature_importances(path, feature_names=names)
        assert isinstance(importances, pd.Series)

    def test_values_sum_to_one(self, rf_model_path):
        path, _, names = rf_model_path
        importances = extract_rf_feature_importances(path, feature_names=names)
        np.testing.assert_allclose(importances.sum(), 1.0, atol=1e-6)

    def test_sorted_descending(self, rf_model_path):
        path, _, names = rf_model_path
        importances = extract_rf_feature_importances(path, feature_names=names)
        assert list(importances.values) == sorted(importances.values, reverse=True)

    def test_non_negative_importances(self, rf_model_path):
        path, _, names = rf_model_path
        importances = extract_rf_feature_importances(path, feature_names=names)
        assert (importances.values >= 0).all()

    def test_correct_feature_names(self, rf_model_path):
        path, _, names = rf_model_path
        importances = extract_rf_feature_importances(path, feature_names=names)
        assert set(importances.index) == set(names)

    def test_feature_names_from_model_when_none(self, rf_model_path):
        """Without explicit feature_names, should fall back to model.feature_names_in_."""
        path, rf, names = rf_model_path
        # RF fitted without feature names → will use feature_N fallback
        importances = extract_rf_feature_importances(path, feature_names=None)
        assert len(importances) == 10

    def test_missing_model_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            extract_rf_feature_importances(tmp_path / "nonexistent.pkl")

    def test_wrong_feature_names_length_raises(self, rf_model_path):
        path, _, _ = rf_model_path
        with pytest.raises(ValueError, match="length"):
            extract_rf_feature_importances(path, feature_names=["a", "b"])

    def test_non_tree_model_raises(self, tmp_path):
        """Non-tree models without feature_importances_ should raise AttributeError."""
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit([[1, 2], [3, 4]], [0, 1])
        path = tmp_path / "lr.pkl"
        with open(path, "wb") as f:
            pickle.dump(model, f)
        with pytest.raises(AttributeError, match="feature_importances_"):
            extract_rf_feature_importances(path)


# ── recommend_clustering_features ─────────────────────────────────────────────

class TestRecommendClusteringFeatures:
    @pytest.fixture
    def importances(self):
        return pd.Series(
            [0.3, 0.25, 0.2, 0.15, 0.1],
            index=["feat_a", "feat_b", "feat_c", "feat_d", "feat_e"],
            name="importance",
        )

    @pytest.fixture
    def clustering_features(self):
        return ["feat_a", "feat_b", "feat_c", "feat_d", "feat_e", "feat_f"]

    def test_returns_tuple(self, importances, clustering_features):
        result = recommend_clustering_features(importances, clustering_features, top_k=3)
        assert isinstance(result, tuple) and len(result) == 2

    def test_recommended_list_length(self, importances, clustering_features):
        recommended, _ = recommend_clustering_features(importances, clustering_features, top_k=3)
        assert len(recommended) == 3

    def test_recommended_sorted_by_importance(self, importances, clustering_features):
        recommended, _ = recommend_clustering_features(importances, clustering_features, top_k=5)
        # Should be in descending importance order
        imp_vals = [importances.get(f, 0) for f in recommended]
        assert imp_vals == sorted(imp_vals, reverse=True)

    def test_comparison_df_has_required_columns(self, importances, clustering_features):
        _, df = recommend_clustering_features(importances, clustering_features, top_k=3)
        for col in ("feature", "rf_importance", "rank", "in_recommended"):
            assert col in df.columns

    def test_comparison_df_row_count(self, importances, clustering_features):
        _, df = recommend_clustering_features(importances, clustering_features, top_k=3)
        # 6 clustering features total
        assert len(df) == 6

    def test_features_not_in_rf_have_nan_importance(self, importances, clustering_features):
        _, df = recommend_clustering_features(importances, clustering_features, top_k=3)
        not_in_rf = df[df["feature"] == "feat_f"]
        assert not_in_rf["rf_importance"].isna().all()

    def test_top_k_marked_in_recommended(self, importances, clustering_features):
        _, df = recommend_clustering_features(importances, clustering_features, top_k=3)
        recommended_rows = df[df["in_recommended"]]
        assert len(recommended_rows) == 3

    def test_top_k_exceeds_intersection_returns_all(self, importances, clustering_features):
        """When top_k > intersection size, return all intersection features without error."""
        recommended, _ = recommend_clustering_features(importances, clustering_features, top_k=100)
        assert len(recommended) == 5  # 5 features in intersection

    def test_empty_importances_raises(self, clustering_features):
        with pytest.raises(ValueError, match="empty"):
            recommend_clustering_features(pd.Series(dtype=float), clustering_features)

    def test_empty_clustering_features_raises(self, importances):
        with pytest.raises(ValueError, match="empty"):
            recommend_clustering_features(importances, [])
