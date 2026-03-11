"""
Unit tests for exploration/investigation functions added to clustering.py:
  - optimize_n_components
  - compare_svd_pca
  - compute_gap_statistic
  - find_knee_k
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from market_regime.clustering import (
    compare_svd_pca,
    compute_gap_statistic,
    evaluate_kmeans,
    find_knee_k,
    optimize_n_components,
    reduce_pca,
)


# ── Shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def feature_df():
    """Synthetic feature DataFrame — 60 quarters × 20 features, no NaNs."""
    rng = np.random.default_rng(0)
    idx = pd.date_range("2000-03-31", periods=60, freq="QE")
    data = rng.standard_normal((60, 20))
    return pd.DataFrame(data, index=idx, columns=[f"f{i}" for i in range(20)])


@pytest.fixture
def kmeans_scores(feature_df):
    """evaluate_kmeans result for k=2..6 — used to test find_knee_k."""
    pca_df, _, _ = reduce_pca(feature_df, n_components=5)
    return evaluate_kmeans(pca_df, k_range=range(2, 7))


# ── optimize_n_components ─────────────────────────────────────────────────────

class TestOptimizeNComponents:
    def test_returns_dataframe(self, feature_df):
        result = optimize_n_components(feature_df, n_range=range(3, 6), balanced_k=3, n_init=5)
        assert isinstance(result, pd.DataFrame)

    def test_one_row_per_valid_n(self, feature_df):
        result = optimize_n_components(feature_df, n_range=range(3, 6), balanced_k=3, n_init=5)
        assert len(result) == 3  # n=3,4,5

    def test_expected_columns(self, feature_df):
        result = optimize_n_components(feature_df, n_range=range(3, 5), balanced_k=3, n_init=5)
        for col in ("n_components", "explained_variance_pct", "silhouette", "davies_bouldin", "calinski"):
            assert col in result.columns

    def test_variance_pct_between_0_and_100(self, feature_df):
        result = optimize_n_components(feature_df, n_range=range(3, 6), balanced_k=3, n_init=5)
        assert (result["explained_variance_pct"] >= 0).all()
        assert (result["explained_variance_pct"] <= 100).all()

    def test_more_components_more_variance(self, feature_df):
        """Cumulative explained variance must be non-decreasing with n."""
        result = optimize_n_components(feature_df, n_range=range(3, 8), balanced_k=3, n_init=5)
        var = result.sort_values("n_components")["explained_variance_pct"].values
        assert all(var[i] <= var[i + 1] + 0.01 for i in range(len(var) - 1))

    def test_silhouette_in_valid_range(self, feature_df):
        result = optimize_n_components(feature_df, n_range=range(3, 5), balanced_k=3, n_init=5)
        sil = result["silhouette"]
        assert (sil >= -1).all() and (sil <= 1).all()

    def test_empty_df_raises(self):
        with pytest.raises(ValueError, match="empty"):
            optimize_n_components(pd.DataFrame())

    def test_balanced_k_below_2_raises(self, feature_df):
        with pytest.raises(ValueError, match="balanced_k"):
            optimize_n_components(feature_df, balanced_k=1)

    def test_n_range_exceeding_min_shape_is_skipped(self):
        """n values >= min(n_samples, n_features) should be skipped without error."""
        small_df = pd.DataFrame(np.random.randn(10, 5), columns=[f"f{i}" for i in range(5)])
        # n_range goes up to 9, but min(10, 5) = 5; so n=5..9 should be skipped
        result = optimize_n_components(small_df, n_range=range(2, 10), balanced_k=2, n_init=5)
        assert (result["n_components"] < 5).all()

    def test_returns_rows_sorted_by_n_components(self, feature_df):
        result = optimize_n_components(feature_df, n_range=range(3, 7), balanced_k=3, n_init=5)
        assert list(result["n_components"]) == sorted(result["n_components"])


# ── compare_svd_pca ───────────────────────────────────────────────────────────

class TestCompareSvdPca:
    def test_returns_three_dataframes(self, feature_df):
        result = compare_svd_pca(feature_df, n_components=3)
        assert len(result) == 3
        for df in result:
            assert isinstance(df, pd.DataFrame)

    def test_pca_df_shape(self, feature_df):
        pca_df, _, _ = compare_svd_pca(feature_df, n_components=4)
        assert pca_df.shape == (len(feature_df), 4)

    def test_svd_df_shape(self, feature_df):
        _, svd_df, _ = compare_svd_pca(feature_df, n_components=4)
        assert svd_df.shape == (len(feature_df), 4)

    def test_pca_column_names(self, feature_df):
        pca_df, _, _ = compare_svd_pca(feature_df, n_components=3)
        assert list(pca_df.columns) == ["PC1", "PC2", "PC3"]

    def test_svd_column_names(self, feature_df):
        _, svd_df, _ = compare_svd_pca(feature_df, n_components=3)
        assert list(svd_df.columns) == ["SV1", "SV2", "SV3"]

    def test_loadings_df_shape(self, feature_df):
        _, _, loadings_df = compare_svd_pca(feature_df, n_components=3)
        # 20 features × (3 PCA + 3 SVD) = 20 × 6
        assert loadings_df.shape == (20, 6)

    def test_loadings_are_non_negative(self, feature_df):
        """We store absolute values of loadings."""
        _, _, loadings_df = compare_svd_pca(feature_df, n_components=3)
        assert (loadings_df.values >= 0).all()

    def test_index_preserved_in_pca_svd(self, feature_df):
        pca_df, svd_df, _ = compare_svd_pca(feature_df, n_components=3)
        assert pca_df.index.equals(feature_df.index)
        assert svd_df.index.equals(feature_df.index)

    def test_empty_df_raises(self):
        with pytest.raises(ValueError, match="empty"):
            compare_svd_pca(pd.DataFrame())

    def test_n_components_too_large_raises(self):
        small = pd.DataFrame(np.random.randn(5, 3), columns=["a", "b", "c"])
        with pytest.raises(ValueError, match="n_components"):
            compare_svd_pca(small, n_components=5)  # must be < min(5,3) = 3

    def test_loadings_index_is_feature_names(self, feature_df):
        _, _, loadings_df = compare_svd_pca(feature_df, n_components=3)
        assert list(loadings_df.index) == list(feature_df.columns)

    def test_pca_svd_correlation_high(self, feature_df):
        """On StandardScaler-centred data, PCA and SVD should produce nearly identical embeddings."""
        pca_df, svd_df, _ = compare_svd_pca(feature_df, n_components=5)
        # Compare first component (up to sign flip)
        corr = abs(np.corrcoef(pca_df["PC1"].values, svd_df["SV1"].values)[0, 1])
        assert corr > 0.95, f"Expected high correlation between PC1 and SV1, got {corr:.4f}"


# ── compute_gap_statistic ─────────────────────────────────────────────────────

class TestComputeGapStatistic:
    @pytest.fixture
    def X(self):
        """Simple 2D data with 3 obvious clusters."""
        rng = np.random.default_rng(5)
        return np.vstack([
            rng.multivariate_normal([3, 0], 0.2 * np.eye(2), 30),
            rng.multivariate_normal([-3, 0], 0.2 * np.eye(2), 30),
            rng.multivariate_normal([0, 3], 0.2 * np.eye(2), 30),
        ])

    def test_returns_dataframe(self, X):
        result = compute_gap_statistic(X, k_range=range(2, 5), n_boots=3, n_init=3)
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self, X):
        result = compute_gap_statistic(X, k_range=range(2, 5), n_boots=3, n_init=3)
        for col in ("k", "gap", "gap_std", "gap_sk", "optimal"):
            assert col in result.columns

    def test_one_row_per_k(self, X):
        result = compute_gap_statistic(X, k_range=range(2, 6), n_boots=3, n_init=3)
        assert len(result) == 4

    def test_exactly_one_optimal_k(self, X):
        result = compute_gap_statistic(X, k_range=range(2, 6), n_boots=3, n_init=3)
        assert result["optimal"].sum() == 1

    def test_gap_sk_geq_gap_std(self, X):
        """gap_sk = gap_std * sqrt(1 + 1/B) must be >= gap_std."""
        result = compute_gap_statistic(X, k_range=range(2, 5), n_boots=5, n_init=3)
        assert (result["gap_sk"].values >= result["gap_std"].values - 1e-9).all()

    def test_gap_std_and_gap_sk_differ(self, X):
        """They should NOT be identical (gap_sk = std * sqrt(1 + 1/B) > std)."""
        result = compute_gap_statistic(X, k_range=range(2, 5), n_boots=5, n_init=3)
        assert not np.allclose(result["gap_std"].values, result["gap_sk"].values), (
            "gap_std and gap_sk are identical — the fix separating them may have been reverted"
        )

    def test_gap_values_are_finite(self, X):
        result = compute_gap_statistic(X, k_range=range(2, 5), n_boots=3, n_init=3)
        assert np.isfinite(result["gap"].values).all()

    def test_too_few_samples_raises(self):
        with pytest.raises(ValueError, match="samples"):
            compute_gap_statistic(np.array([[1.0]]), k_range=range(2, 4))

    def test_empty_k_range_raises(self, X):
        with pytest.raises(ValueError, match="empty"):
            compute_gap_statistic(X, k_range=range(0, 0))

    def test_reproducible_with_same_seed(self, X):
        r1 = compute_gap_statistic(X, k_range=range(2, 5), n_boots=3, n_init=3, random_state=42)
        r2 = compute_gap_statistic(X, k_range=range(2, 5), n_boots=3, n_init=3, random_state=42)
        pd.testing.assert_frame_equal(r1, r2)

    def test_k_column_matches_k_range(self, X):
        result = compute_gap_statistic(X, k_range=range(2, 5), n_boots=3, n_init=3)
        assert list(result["k"]) == [2, 3, 4]


# ── find_knee_k ───────────────────────────────────────────────────────────────

class TestFindKneeK:
    def test_returns_int(self, kmeans_scores):
        knee = find_knee_k(kmeans_scores)
        assert isinstance(knee, int)

    def test_knee_within_k_range(self, kmeans_scores):
        knee = find_knee_k(kmeans_scores)
        valid_ks = set(kmeans_scores["k"].tolist())
        assert knee in valid_ks

    def test_missing_k_column_raises(self, kmeans_scores):
        bad = kmeans_scores.drop(columns=["k"])
        with pytest.raises(ValueError, match="k"):
            find_knee_k(bad)

    def test_missing_inertia_column_raises(self, kmeans_scores):
        bad = kmeans_scores.drop(columns=["inertia"])
        with pytest.raises(ValueError, match="inertia"):
            find_knee_k(bad)

    def test_single_row_raises(self):
        single = pd.DataFrame({"k": [3], "inertia": [100.0]})
        with pytest.raises(ValueError, match="at least 2"):
            find_knee_k(single)

    def test_monotone_inertia_input(self, feature_df):
        """Inertia is monotonically decreasing — elbow detection should not crash."""
        pca_df, _, _ = reduce_pca(feature_df, n_components=5)
        scores = evaluate_kmeans(pca_df, k_range=range(2, 8))
        knee = find_knee_k(scores)
        assert 2 <= knee <= 7

    def test_kneed_fallback_gradient(self, kmeans_scores, monkeypatch):
        """When kneed is not installed, gradient fallback should still return a valid k."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "kneed":
                raise ImportError("No module named 'kneed'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        knee = find_knee_k(kmeans_scores)
        valid_ks = set(kmeans_scores["k"].tolist())
        assert knee in valid_ks
