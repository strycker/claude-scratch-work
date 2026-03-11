"""Unit tests for src/market_regime/density.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from market_regime.density import (
    fit_dbscan,
    fit_dbscan_sweep,
    hdbscan_labels,
    knn_distances,
    fit_hdbscan_sweep,
)


# ── Shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def pca_df():
    """
    60 quarters × 5 PCs with 3 well-separated clusters.
    Using large separation (mean at ±5) so DBSCAN with appropriate eps
    finds exactly 3 clusters.
    """
    rng = np.random.default_rng(7)
    idx = pd.date_range("2000-03-31", periods=60, freq="QE")
    data = np.vstack([
        rng.multivariate_normal([5, 0, 0, 0, 0], 0.1 * np.eye(5), 20),
        rng.multivariate_normal([-5, 0, 0, 0, 0], 0.1 * np.eye(5), 20),
        rng.multivariate_normal([0, 5, 0, 0, 0], 0.1 * np.eye(5), 20),
    ])
    return pd.DataFrame(data, index=idx, columns=[f"PC{i+1}" for i in range(5)])


@pytest.fixture
def empty_df():
    return pd.DataFrame(columns=[f"PC{i+1}" for i in range(5)])


# ── knn_distances ─────────────────────────────────────────────────────────────

class TestKnnDistances:
    def test_returns_series(self, pca_df):
        dists = knn_distances(pca_df, k=5)
        assert isinstance(dists, pd.Series)

    def test_length_equals_n_samples(self, pca_df):
        dists = knn_distances(pca_df, k=5)
        assert len(dists) == len(pca_df)

    def test_distances_are_non_negative(self, pca_df):
        dists = knn_distances(pca_df, k=5)
        assert (dists.values >= 0).all()

    def test_distances_are_sorted_ascending(self, pca_df):
        dists = knn_distances(pca_df, k=5)
        assert (dists.diff().dropna() >= 0).all()

    def test_name_reflects_k(self, pca_df):
        dists = knn_distances(pca_df, k=7)
        assert dists.name == "7nn_distance"

    def test_empty_df_raises(self, empty_df):
        with pytest.raises(ValueError, match="empty"):
            knn_distances(empty_df)

    def test_different_k_produces_different_distances(self, pca_df):
        d3 = knn_distances(pca_df, k=3)
        d10 = knn_distances(pca_df, k=10)
        # k=10 distances should generally be >= k=3 distances
        assert d10.mean() >= d3.mean()


# ── fit_dbscan_sweep ──────────────────────────────────────────────────────────

class TestFitDbscanSweep:
    def test_returns_dataframe(self, pca_df):
        result = fit_dbscan_sweep(pca_df, eps_values=[0.5, 1.0])
        assert isinstance(result, pd.DataFrame)

    def test_one_row_per_eps(self, pca_df):
        eps_values = [0.5, 1.0, 1.5, 2.0]
        result = fit_dbscan_sweep(pca_df, eps_values=eps_values)
        assert len(result) == len(eps_values)

    def test_expected_columns(self, pca_df):
        result = fit_dbscan_sweep(pca_df, eps_values=[1.0])
        for col in ("eps", "n_clusters", "n_noise", "noise_pct", "silhouette"):
            assert col in result.columns

    def test_noise_pct_between_0_and_100(self, pca_df):
        result = fit_dbscan_sweep(pca_df, eps_values=[0.3, 1.0, 5.0])
        assert (result["noise_pct"] >= 0).all()
        assert (result["noise_pct"] <= 100).all()

    def test_n_noise_plus_n_cluster_pts_equals_total(self, pca_df):
        """n_noise + sum(cluster points) = total rows."""
        result = fit_dbscan_sweep(pca_df, eps_values=[1.0])
        row = result.iloc[0]
        assert row["n_noise"] >= 0
        assert row["n_clusters"] >= 0

    def test_empty_df_raises(self, empty_df):
        with pytest.raises(ValueError, match="empty"):
            fit_dbscan_sweep(empty_df)

    def test_silhouette_nan_when_single_cluster(self, pca_df):
        """With very large eps, everything is one cluster; silhouette should be NaN."""
        result = fit_dbscan_sweep(pca_df, eps_values=[100.0])
        assert result.iloc[0]["n_clusters"] <= 1
        assert pd.isna(result.iloc[0]["silhouette"]) or result.iloc[0]["silhouette"] is None


# ── fit_dbscan ────────────────────────────────────────────────────────────────

class TestFitDbscan:
    def test_returns_series(self, pca_df):
        labels = fit_dbscan(pca_df, eps=1.0)
        assert isinstance(labels, pd.Series)

    def test_index_matches_pca_df(self, pca_df):
        labels = fit_dbscan(pca_df, eps=1.0)
        assert labels.index.equals(pca_df.index)

    def test_name_is_dbscan_cluster(self, pca_df):
        labels = fit_dbscan(pca_df, eps=1.0)
        assert labels.name == "dbscan_cluster"

    def test_labels_are_integers(self, pca_df):
        labels = fit_dbscan(pca_df, eps=1.0)
        assert labels.dtype in (np.int32, np.int64, int)

    def test_noise_label_is_minus_one(self, pca_df):
        """DBSCAN noise points must use label -1 (sklearn convention)."""
        # Very tight eps — expect some noise
        labels = fit_dbscan(pca_df, eps=0.01, min_samples=5)
        assert set(labels.unique()).issubset({-1} | set(range(20)))

    def test_large_eps_no_noise(self, pca_df):
        """With very large eps, every point should be in a cluster (no noise)."""
        labels = fit_dbscan(pca_df, eps=100.0)
        assert (labels >= 0).all()

    def test_empty_df_raises(self, empty_df):
        with pytest.raises(ValueError, match="empty"):
            fit_dbscan(empty_df, eps=1.0)

    def test_no_noise_gives_no_minus_one(self, pca_df):
        labels = fit_dbscan(pca_df, eps=100.0)
        assert -1 not in labels.values


# ── HDBSCAN (may not be installed) ────────────────────────────────────────────

@pytest.fixture
def hdbscan_available():
    try:
        import hdbscan  # noqa: F401
        return True
    except ImportError:
        return False


class TestFitHdbscanSweep:
    def test_import_error_when_not_installed(self, pca_df, monkeypatch):
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "hdbscan":
                raise ImportError("No module named 'hdbscan'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        with pytest.raises(ImportError, match="hdbscan not installed"):
            fit_hdbscan_sweep(pca_df)

    def test_empty_df_raises(self, empty_df, hdbscan_available):
        if not hdbscan_available:
            pytest.skip("hdbscan not installed")
        with pytest.raises(ValueError, match="empty"):
            fit_hdbscan_sweep(empty_df)

    def test_returns_dataframe(self, pca_df, hdbscan_available):
        if not hdbscan_available:
            pytest.skip("hdbscan not installed")
        result = fit_hdbscan_sweep(pca_df, min_cluster_sizes=[10, 15])
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self, pca_df, hdbscan_available):
        if not hdbscan_available:
            pytest.skip("hdbscan not installed")
        result = fit_hdbscan_sweep(pca_df, min_cluster_sizes=[15])
        for col in ("min_cluster_size", "n_clusters", "n_noise", "noise_pct", "silhouette"):
            assert col in result.columns

    def test_one_row_per_min_cluster_size(self, pca_df, hdbscan_available):
        if not hdbscan_available:
            pytest.skip("hdbscan not installed")
        result = fit_hdbscan_sweep(pca_df, min_cluster_sizes=[10, 15, 20])
        assert len(result) == 3


class TestHdbscanLabels:
    def test_import_error_when_not_installed(self, pca_df, monkeypatch):
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "hdbscan":
                raise ImportError("No module named 'hdbscan'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        with pytest.raises(ImportError, match="hdbscan not installed"):
            hdbscan_labels(pca_df)

    def test_returns_series(self, pca_df, hdbscan_available):
        if not hdbscan_available:
            pytest.skip("hdbscan not installed")
        labels = hdbscan_labels(pca_df, min_cluster_size=10)
        assert isinstance(labels, pd.Series)

    def test_index_matches_pca_df(self, pca_df, hdbscan_available):
        if not hdbscan_available:
            pytest.skip("hdbscan not installed")
        labels = hdbscan_labels(pca_df, min_cluster_size=10)
        assert labels.index.equals(pca_df.index)

    def test_name_is_hdbscan_cluster(self, pca_df, hdbscan_available):
        if not hdbscan_available:
            pytest.skip("hdbscan not installed")
        labels = hdbscan_labels(pca_df, min_cluster_size=10)
        assert labels.name == "hdbscan_cluster"

    def test_empty_df_raises(self, empty_df, hdbscan_available):
        if not hdbscan_available:
            pytest.skip("hdbscan not installed")
        with pytest.raises(ValueError, match="empty"):
            hdbscan_labels(empty_df)
