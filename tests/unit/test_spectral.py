"""Unit tests for src/market_regime/spectral.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from market_regime.spectral import fit_spectral_sweep, spectral_labels


# ── Shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def pca_df():
    """40 quarters × 5 PCs with 4 well-separated blobs."""
    rng = np.random.default_rng(99)
    idx = pd.date_range("2000-03-31", periods=40, freq="QE")
    data = np.vstack([
        rng.multivariate_normal([5, 0, 0, 0, 0], 0.2 * np.eye(5), 10),
        rng.multivariate_normal([-5, 0, 0, 0, 0], 0.2 * np.eye(5), 10),
        rng.multivariate_normal([0, 5, 0, 0, 0], 0.2 * np.eye(5), 10),
        rng.multivariate_normal([0, -5, 0, 0, 0], 0.2 * np.eye(5), 10),
    ])
    return pd.DataFrame(data, index=idx, columns=[f"PC{i+1}" for i in range(5)])


@pytest.fixture
def empty_df():
    return pd.DataFrame(columns=[f"PC{i+1}" for i in range(5)])


# ── fit_spectral_sweep ────────────────────────────────────────────────────────

class TestFitSpectralSweep:
    def test_returns_tuple(self, pca_df):
        result = fit_spectral_sweep(pca_df, k_range=range(2, 4))
        assert isinstance(result, tuple) and len(result) == 2

    def test_sweep_df_columns(self, pca_df):
        sweep_df, _ = fit_spectral_sweep(pca_df, k_range=range(2, 4))
        for col in ("k", "silhouette", "davies_bouldin", "calinski"):
            assert col in sweep_df.columns

    def test_one_row_per_k(self, pca_df):
        sweep_df, _ = fit_spectral_sweep(pca_df, k_range=range(2, 5))
        assert len(sweep_df) == 3

    def test_all_labels_dict_keys(self, pca_df):
        _, labels_dict = fit_spectral_sweep(pca_df, k_range=range(2, 5))
        assert set(labels_dict.keys()) == {2, 3, 4}

    def test_labels_series_index_matches_pca_df(self, pca_df):
        _, labels_dict = fit_spectral_sweep(pca_df, k_range=range(2, 4))
        for k, labels in labels_dict.items():
            assert labels.index.equals(pca_df.index), f"k={k}: index mismatch"

    def test_labels_correct_unique_count(self, pca_df):
        _, labels_dict = fit_spectral_sweep(pca_df, k_range=range(2, 5))
        for k, labels in labels_dict.items():
            assert labels.nunique() == k, f"k={k}: expected {k} unique labels, got {labels.nunique()}"

    def test_silhouette_in_valid_range(self, pca_df):
        sweep_df, _ = fit_spectral_sweep(pca_df, k_range=range(2, 4))
        sil = sweep_df["silhouette"].dropna()
        assert (sil >= -1).all() and (sil <= 1).all()

    def test_empty_df_raises(self, empty_df):
        with pytest.raises(ValueError, match="empty"):
            fit_spectral_sweep(empty_df)

    def test_precomputed_affinity_used_for_nearest_neighbors(self, pca_df):
        """Sweep should complete successfully — tests that precomputed affinity path works."""
        sweep_df, labels_dict = fit_spectral_sweep(
            pca_df, k_range=range(2, 4), affinity="nearest_neighbors", n_neighbors=5
        )
        # Both k values should produce results
        assert len(sweep_df) == 2
        assert 2 in labels_dict and 3 in labels_dict


# ── spectral_labels ───────────────────────────────────────────────────────────

class TestSpectralLabels:
    def test_returns_series(self, pca_df):
        labels = spectral_labels(pca_df, k=3)
        assert isinstance(labels, pd.Series)

    def test_index_matches_pca_df(self, pca_df):
        labels = spectral_labels(pca_df, k=3)
        assert labels.index.equals(pca_df.index)

    def test_name_is_spectral_cluster(self, pca_df):
        labels = spectral_labels(pca_df, k=3)
        assert labels.name == "spectral_cluster"

    def test_correct_number_of_clusters(self, pca_df):
        for k in (2, 3, 4):
            labels = spectral_labels(pca_df, k=k, random_state=42)
            assert labels.nunique() == k, f"k={k}: expected {k} clusters"

    def test_labels_are_non_negative(self, pca_df):
        labels = spectral_labels(pca_df, k=3)
        assert (labels >= 0).all()

    def test_canonicalized_cluster0_has_smallest_mean_pc1(self, pca_df):
        """Cluster 0 must correspond to smallest mean PC1 (canonicalization check)."""
        labels = spectral_labels(pca_df, k=4, random_state=42)
        pc1 = pca_df.iloc[:, 0]
        mean_pc1 = pc1.groupby(labels).mean()
        assert mean_pc1.idxmin() == 0, (
            f"Cluster 0 should have smallest mean PC1, got mean_pc1={mean_pc1.to_dict()}"
        )

    def test_empty_df_raises(self, empty_df):
        with pytest.raises(ValueError, match="empty"):
            spectral_labels(empty_df, k=3)

    def test_labels_cover_full_index(self, pca_df):
        """Every quarter should receive a label — no NaNs."""
        labels = spectral_labels(pca_df, k=3)
        assert labels.notna().all()
