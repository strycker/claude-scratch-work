"""
PCA dimensionality reduction, KMeans cluster evaluation, and clustering.

Three public functions intended to be called in sequence from pipeline step 03:

  1. reduce_pca()       — StandardScale + PCA to N fixed components
  2. evaluate_kmeans()  — sweep k, score with silhouette/CH/DB, pick best k
  3. fit_clusters()     — standard KMeans at best_k + size-constrained KMeans
                          at balanced_k, both stored as columns on the output df
"""

import logging

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)


def _load_constrained_kmeans():
    try:
        from k_means_constrained import KMeansConstrained
        return KMeansConstrained
    except ImportError:
        log.warning(
            "k-means-constrained not installed — balanced clustering unavailable. "
            "Run: pip install k-means-constrained"
        )
        return None


# ── 1. PCA ─────────────────────────────────────────────────────────────────

def reduce_pca(
    df: pd.DataFrame,
    n_components: int = 5,
    random_state: int = 42,
) -> tuple[pd.DataFrame, PCA, StandardScaler]:
    """
    StandardScale the features then reduce to exactly n_components PCA axes.

    Returns:
        pca_df   — DataFrame of PC columns (PC1…PCn), same index as df
        pca_obj  — fitted PCA (kept for scoring new data later)
        scaler   — fitted StandardScaler (kept for the same reason)
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df.values)

    pca = PCA(n_components=n_components, random_state=random_state)
    X_reduced = pca.fit_transform(X_scaled)

    cumvar = np.cumsum(pca.explained_variance_ratio_)
    log.info(
        "PCA: %d components explain %.1f%% of variance  "
        "(ratios: %s)",
        n_components,
        cumvar[-1] * 100,
        np.round(pca.explained_variance_ratio_, 3),
    )

    col_names = [f"PC{i + 1}" for i in range(n_components)]
    pca_df = pd.DataFrame(X_reduced, index=df.index, columns=col_names)
    return pca_df, pca, scaler


# ── 2. K evaluation ────────────────────────────────────────────────────────

def evaluate_kmeans(
    X: np.ndarray,
    k_range: range,
    n_init: int = 50,
    random_state: int = 0,
) -> pd.DataFrame:
    """
    Run KMeans for each k in k_range and return a DataFrame of quality scores:
      inertia, silhouette, calinski_harabasz, davies_bouldin.

    Args:
        X            — scaled feature matrix (output of StandardScaler)
        k_range      — range of k values to evaluate, e.g. range(2, 13)
        n_init       — KMeans restarts per k (higher = more stable)
        random_state

    Returns:
        DataFrame with one row per k, sorted by silhouette descending.
    """
    results = []
    for k in k_range:
        model = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
        labels = model.fit_predict(X)
        results.append({
            "k":              k,
            "inertia":        model.inertia_,
            "silhouette":     silhouette_score(X, labels),
            "calinski":       calinski_harabasz_score(X, labels),
            "davies_bouldin": davies_bouldin_score(X, labels),
        })
        log.debug("k=%d  sil=%.4f  CH=%.1f  DB=%.4f",
                  k, results[-1]["silhouette"],
                  results[-1]["calinski"],
                  results[-1]["davies_bouldin"])

    scores = pd.DataFrame(results)
    log.info(
        "Best k by silhouette: %d (score=%.4f)",
        int(scores.loc[scores["silhouette"].idxmax(), "k"]),
        scores["silhouette"].max(),
    )
    return scores


def pick_best_k(scores: pd.DataFrame, k_cap: int = 5) -> int:
    """Return the k with the highest silhouette score, capped at k_cap."""
    best = int(scores.loc[scores["silhouette"].idxmax(), "k"])
    return min(best, k_cap)


# ── 3. Clustering ──────────────────────────────────────────────────────────

def fit_clusters(
    pca_df: pd.DataFrame,
    best_k: int,
    balanced_k: int,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Fit two clusterings on the PCA-reduced data:
      - "cluster"          — standard KMeans at best_k
      - "balanced_cluster" — size-constrained KMeans at balanced_k

    Args:
        pca_df      — output of reduce_pca()
        best_k      — k chosen by silhouette search (via pick_best_k)
        balanced_k  — k for equal-size clustering (from config)
        random_state

    Returns:
        pca_df with two new columns: cluster, balanced_cluster.
    """
    # Re-scale the PCA components before clustering
    X = StandardScaler().fit_transform(pca_df.values)
    result = pca_df.copy()

    # Standard KMeans
    result["cluster"] = KMeans(
        n_clusters=best_k, n_init=100, random_state=random_state
    ).fit_predict(X)
    log.info("Standard KMeans (k=%d): %s", best_k, _size_summary(result["cluster"]))

    # Size-constrained KMeans
    KMC = _load_constrained_kmeans()
    if KMC is not None:
        n = len(X)
        bucket = n // balanced_k
        model = KMC(
            n_clusters=balanced_k,
            size_min=bucket - 2,
            size_max=bucket + 2,
            random_state=random_state,
        )
        result["balanced_cluster"] = model.fit_predict(X)
        log.info(
            "Balanced KMeans (k=%d): %s",
            balanced_k, _size_summary(result["balanced_cluster"]),
        )
    else:
        # Fall back to plain KMeans so the column always exists
        result["balanced_cluster"] = KMeans(
            n_clusters=balanced_k, n_init=100, random_state=random_state
        ).fit_predict(X)
        log.warning("balanced_cluster uses plain KMeans (k-means-constrained unavailable)")

    return result


def _size_summary(labels: pd.Series) -> str:
    counts = labels.value_counts().sort_index()
    return ", ".join(f"{k}:{v}" for k, v in counts.items())
