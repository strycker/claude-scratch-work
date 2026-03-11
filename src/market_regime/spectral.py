"""
Spectral Clustering — graph-based regime detection.

Why Spectral Clustering?
- Operates on a similarity graph rather than raw distances, so it can find
  non-convex clusters (e.g. crescent or ring-shaped regime clouds in PCA space).
- Particularly useful when regimes are not well-separated in Euclidean space
  but are connected regions in the data manifold.
- Requires specifying k (same as KMeans); use optimal k from gap statistic or BIC.

Affinity options (sklearn)
---------------------------
- "nearest_neighbors" — connects each point to its k-NN; sparse and fast
- "rbf"              — Gaussian kernel similarity; dense, more sensitive to gamma
- precomputed matrix  — supply your own (not used here)

Computational cost: O(N³) eigendecomposition.  Feasible at N≈300 quarters.

Usage
------
    from market_regime.spectral import fit_spectral_sweep, spectral_labels

    sweep_df, all_labels = fit_spectral_sweep(pca_df, k_range=range(2, 8))
    labels = spectral_labels(pca_df, k=5)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.cluster import SpectralClustering
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)


def fit_spectral_sweep(
    pca_df: pd.DataFrame,
    k_range: range | None = None,
    affinity: str = "nearest_neighbors",
    n_neighbors: int = 10,
    n_init: int = 10,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[int, pd.Series]]:
    """
    Run SpectralClustering for each k in k_range and summarise results.

    Args:
        pca_df      — PCA-reduced feature matrix
        k_range     — k values to evaluate (default range(2, 8))
        affinity    — graph construction method ('nearest_neighbors' or 'rbf')
        n_neighbors — for affinity='nearest_neighbors'
        n_init      — k-means restarts in the spectral embedding step
        random_state

    Returns:
        sweep_df   — DataFrame with: k, silhouette, davies_bouldin, calinski
        all_labels — dict mapping k → pd.Series of cluster labels
    """
    if k_range is None:
        k_range = range(2, 8)

    X = StandardScaler().fit_transform(pca_df.values)
    rows: list[dict] = []
    all_labels: dict[int, pd.Series] = {}

    for k in k_range:
        try:
            sc = SpectralClustering(
                n_clusters=k,
                affinity=affinity,
                n_neighbors=n_neighbors,
                n_init=n_init,
                random_state=random_state,
            )
            labels_arr = sc.fit_predict(X)
            labels = pd.Series(labels_arr, index=pca_df.index, name=f"spectral_k{k}")
            all_labels[k] = labels

            sil = silhouette_score(X, labels_arr)
            db  = davies_bouldin_score(X, labels_arr)
            ch  = calinski_harabasz_score(X, labels_arr)
            rows.append({"k": k, "silhouette": sil, "davies_bouldin": db, "calinski": ch})
            log.info("Spectral k=%d  sil=%.4f  DB=%.4f  CH=%.1f", k, sil, db, ch)
        except Exception as exc:
            log.warning("Spectral k=%d failed: %s", k, exc)

    return pd.DataFrame(rows), all_labels


def spectral_labels(
    pca_df: pd.DataFrame,
    k: int,
    affinity: str = "nearest_neighbors",
    n_neighbors: int = 10,
    n_init: int = 20,
    random_state: int = 42,
) -> pd.Series:
    """
    Fit SpectralClustering with chosen k and return canonicalized labels.

    Labels are sorted so cluster 0 has the smallest mean PC1 value
    (consistent with KMeans canonicalization in clustering.py).

    Returns:
        Series indexed by quarter with integer cluster labels.
    """
    X = StandardScaler().fit_transform(pca_df.values)
    sc = SpectralClustering(
        n_clusters=k,
        affinity=affinity,
        n_neighbors=n_neighbors,
        n_init=n_init,
        random_state=random_state,
    )
    raw = pd.Series(sc.fit_predict(X), index=pca_df.index, name="spectral_cluster")

    # Canonicalize by mean PC1
    pc1 = pca_df.iloc[:, 0]
    mean_pc1 = raw.groupby(raw).apply(lambda g: pc1.loc[g.index].mean())
    label_map = {old: new for new, old in enumerate(mean_pc1.sort_values().index)}
    labels = raw.map(label_map).rename("spectral_cluster")

    log.info(
        "Spectral (k=%d, affinity=%s): %s",
        k, affinity,
        labels.value_counts().sort_index().to_dict(),
    )
    return labels
