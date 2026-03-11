"""
Density-based clustering — DBSCAN and HDBSCAN.

Why density-based clustering?
- Does NOT require specifying k ahead of time.
- Finds clusters of arbitrary shape (not restricted to convex Voronoi cells).
- Labels genuine outliers as noise (label = -1) rather than forcing every
  quarter into a regime.  Useful for identifying historically unique quarters.

DBSCAN (sklearn, always available)
------------------------------------
Key parameters:
  eps         — maximum distance between two samples to be considered neighbours.
                Tune via the k-distance plot: sort pairwise distances to the
                min_samples-th nearest neighbour; the elbow is a good eps.
  min_samples — minimum neighbours for a point to be a "core point".

HDBSCAN (optional — pip install hdbscan)
-----------------------------------------
Hierarchical variant; more robust to eps choice.  Produces a cluster hierarchy
and selects flat clusters from it automatically.  Recommended over DBSCAN when
eps is hard to tune (e.g. clusters of varying density).

Usage
------
    from market_regime.density import (
        knn_distances, fit_dbscan_sweep, fit_dbscan,
        fit_hdbscan_sweep, hdbscan_labels,
    )

    dists = knn_distances(pca_df, k=5)           # plot to pick eps
    sweep = fit_dbscan_sweep(pca_df, eps_values=[0.5, 1.0, 1.5, 2.0])
    labels = fit_dbscan(pca_df, eps=1.0, min_samples=5)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)


def knn_distances(pca_df: pd.DataFrame, k: int = 5) -> pd.Series:
    """
    Compute sorted k-NN distances for eps selection in DBSCAN.

    Plot the returned Series; the 'elbow' where the curve bends sharply upward
    is a good choice for eps.  k should equal the DBSCAN min_samples parameter.

    Returns:
        Series of sorted distances (ascending), index = integer rank.
    """
    if pca_df.empty:
        raise ValueError("pca_df is empty — cannot compute k-NN distances")
    X = StandardScaler().fit_transform(pca_df.values)
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    distances, _ = nbrs.kneighbors(X)
    kth_dist = np.sort(distances[:, -1])  # distance to k-th nearest neighbour
    return pd.Series(kth_dist, name=f"{k}nn_distance")


def fit_dbscan_sweep(
    pca_df: pd.DataFrame,
    eps_values: list[float] | None = None,
    min_samples: int = 5,
) -> pd.DataFrame:
    """
    Run DBSCAN for each eps in eps_values and summarise results.

    Returns:
        DataFrame with columns:
        eps, n_clusters (excl. noise), n_noise, noise_pct, silhouette
        (silhouette is NaN when fewer than 2 non-noise clusters are found).
    """
    if pca_df.empty:
        raise ValueError("pca_df is empty — cannot run DBSCAN sweep")
    if eps_values is None:
        eps_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    X = StandardScaler().fit_transform(pca_df.values)
    rows: list[dict] = []

    for eps in eps_values:
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
        n_noise = int((labels == -1).sum())
        unique_clusters = sorted(set(labels) - {-1})
        n_clusters = len(unique_clusters)

        if n_clusters == 0:
            log.warning(
                "DBSCAN eps=%.2f: all %d points are noise — eps may be too small or "
                "min_samples too large",
                eps, n_noise,
            )
        elif n_clusters == 1:
            log.warning(
                "DBSCAN eps=%.2f: only 1 cluster found — metrics require at least 2",
                eps,
            )

        sil: float | None = None
        if n_clusters >= 2:
            mask = labels != -1
            if mask.sum() >= n_clusters:
                try:
                    sil = silhouette_score(X[mask], labels[mask])
                except Exception as exc:
                    log.debug("silhouette_score failed for eps=%.2f: %s", eps, exc)

        rows.append({
            "eps":        eps,
            "n_clusters": n_clusters,
            "n_noise":    n_noise,
            "noise_pct":  round(100 * n_noise / len(labels), 1),
            "silhouette": sil,
        })
        log.info(
            "DBSCAN eps=%.2f  clusters=%d  noise=%d (%.1f%%)  sil=%s",
            eps, n_clusters, n_noise, 100 * n_noise / len(labels),
            f"{sil:.4f}" if sil is not None else "N/A",
        )

    return pd.DataFrame(rows)


def fit_dbscan(
    pca_df: pd.DataFrame,
    eps: float,
    min_samples: int = 5,
) -> pd.Series:
    """
    Fit DBSCAN with chosen eps and return labels as a Series.

    Noise points (label = -1) are preserved; downstream code should handle them
    (e.g. assign to nearest centroid, or exclude from supervised training).

    Warns if all points are noise or only 1 cluster is found.

    Returns:
        Series indexed by quarter, values = cluster id (or -1 for noise).
    """
    if pca_df.empty:
        raise ValueError("pca_df is empty — cannot run DBSCAN")
    X = StandardScaler().fit_transform(pca_df.values)
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
    series = pd.Series(labels, index=pca_df.index, name="dbscan_cluster")

    n_noise = int((series == -1).sum())
    n_clusters = int(series[series >= 0].nunique())

    if n_clusters == 0:
        log.warning(
            "DBSCAN (eps=%.2f, min_samples=%d): ALL %d points are noise — "
            "try increasing eps or decreasing min_samples",
            eps, min_samples, n_noise,
        )
    elif n_clusters == 1:
        log.warning(
            "DBSCAN (eps=%.2f, min_samples=%d): only 1 cluster found — "
            "clustering metrics require at least 2 clusters",
            eps, min_samples,
        )
    else:
        log.info(
            "DBSCAN (eps=%.2f, min_samples=%d): %d clusters, %d noise points (%.1f%%)",
            eps, min_samples, n_clusters, n_noise, 100 * n_noise / len(labels),
        )
    return series


def fit_hdbscan_sweep(
    pca_df: pd.DataFrame,
    min_cluster_sizes: list[int] | None = None,
) -> pd.DataFrame:
    """
    Run HDBSCAN for each min_cluster_size and summarise results.

    Requires: pip install hdbscan

    Returns:
        DataFrame with columns: min_cluster_size, n_clusters, n_noise, noise_pct, silhouette
    """
    try:
        import hdbscan as hdbscan_lib  # type: ignore[import]
    except ImportError:
        raise ImportError(
            "hdbscan not installed.  Run: pip install hdbscan\n"
            "Or use fit_dbscan_sweep() for the sklearn DBSCAN alternative."
        )

    if pca_df.empty:
        raise ValueError("pca_df is empty — cannot run HDBSCAN sweep")
    if min_cluster_sizes is None:
        min_cluster_sizes = [10, 15, 20, 25]

    X = StandardScaler().fit_transform(pca_df.values)
    rows: list[dict] = []

    for mcs in min_cluster_sizes:
        clusterer = hdbscan_lib.HDBSCAN(min_cluster_size=mcs)
        labels = clusterer.fit_predict(X)
        n_noise = int((labels == -1).sum())
        n_clusters = len(set(labels) - {-1})

        if n_clusters == 0:
            log.warning(
                "HDBSCAN min_cluster_size=%d: all %d points are noise — "
                "try decreasing min_cluster_size",
                mcs, n_noise,
            )
        elif n_clusters == 1:
            log.warning(
                "HDBSCAN min_cluster_size=%d: only 1 cluster found — "
                "clustering metrics require at least 2 clusters",
                mcs,
            )

        sil: float | None = None
        if n_clusters >= 2:
            mask = labels != -1
            if mask.sum() >= n_clusters:
                try:
                    sil = silhouette_score(X[mask], labels[mask])
                except Exception as exc:
                    log.debug("silhouette_score failed for min_cluster_size=%d: %s", mcs, exc)

        rows.append({
            "min_cluster_size": mcs,
            "n_clusters":       n_clusters,
            "n_noise":          n_noise,
            "noise_pct":        round(100 * n_noise / len(labels), 1),
            "silhouette":       sil,
        })
        log.info(
            "HDBSCAN min_cluster_size=%d  clusters=%d  noise=%d (%.1f%%)  sil=%s",
            mcs, n_clusters, n_noise, 100 * n_noise / len(labels),
            f"{sil:.4f}" if sil is not None else "N/A",
        )

    return pd.DataFrame(rows)


def hdbscan_labels(pca_df: pd.DataFrame, min_cluster_size: int = 15) -> pd.Series:
    """
    Fit HDBSCAN with chosen min_cluster_size and return labels as a Series.

    Requires: pip install hdbscan

    Warns if all points are noise or only 1 cluster is found.
    """
    try:
        import hdbscan as hdbscan_lib  # type: ignore[import]
    except ImportError:
        raise ImportError("hdbscan not installed.  Run: pip install hdbscan")

    if pca_df.empty:
        raise ValueError("pca_df is empty — cannot run HDBSCAN")

    X = StandardScaler().fit_transform(pca_df.values)
    labels = hdbscan_lib.HDBSCAN(min_cluster_size=min_cluster_size).fit_predict(X)
    series = pd.Series(labels, index=pca_df.index, name="hdbscan_cluster")

    n_noise = int((series == -1).sum())
    n_clusters = int(series[series >= 0].nunique())

    if n_clusters == 0:
        log.warning(
            "HDBSCAN (min_cluster_size=%d): ALL %d points are noise — "
            "try decreasing min_cluster_size",
            min_cluster_size, n_noise,
        )
    elif n_clusters == 1:
        log.warning(
            "HDBSCAN (min_cluster_size=%d): only 1 cluster found — "
            "metrics require at least 2 clusters",
            min_cluster_size,
        )
    else:
        log.info(
            "HDBSCAN (min_cluster_size=%d): %d clusters, %d noise (%.1f%%)",
            min_cluster_size, n_clusters, n_noise, 100 * n_noise / len(labels),
        )
    return series
