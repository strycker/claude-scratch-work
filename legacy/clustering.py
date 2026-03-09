"""
Clustering: PCA dimensionality reduction, KMeans evaluation, regime labeling.

Takes a prepared quarterly DataFrame (all NaNs filled, derivative columns
added) and produces regime cluster labels via:
1. PCA to N_PCA_COMPONENTS
2. KMeans k-sweep with silhouette/Calinski-Harabasz/Davies-Bouldin scoring
3. Best-k standard KMeans
4. Size-constrained KMeans for balanced regime buckets
"""

import logging

import numpy as np
import pandas as pd
try:
    from k_means_constrained import KMeansConstrained
except ImportError:
    KMeansConstrained = None
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

from trading_crab.config import (
    BALANCED_K,
    CLUSTERING_FEATURES,
    K_CAP,
    MAX_K_SEARCH,
    N_PCA_COMPONENTS,
)

log = logging.getLogger(__name__)


def run_pca(df: pd.DataFrame) -> tuple[pd.DataFrame, PCA, np.ndarray]:
    """
    Standardize clustering features and reduce to N_PCA_COMPONENTS via PCA.

    Returns:
        reduced_df: DataFrame with columns PC1..PCn
        pca: fitted PCA object (for explained variance inspection)
        market_codes: aligned market_code array (same length as reduced_df)
    """
    log.info("Running PCA (%d components)...", N_PCA_COMPONENTS)

    # Drop rows with any NaN in clustering features or market_code
    clean = df[CLUSTERING_FEATURES + ["market_code"]].dropna()
    market_codes = clean["market_code"].values

    X = clean[CLUSTERING_FEATURES].values
    X_scaled = StandardScaler().fit_transform(X)

    pca = PCA(n_components=N_PCA_COMPONENTS)
    X_reduced = pca.fit_transform(X_scaled)

    reduced_df = pd.DataFrame(
        X_reduced,
        columns=[f"PC{i+1}" for i in range(N_PCA_COMPONENTS)],
    )

    log.info("  Explained variance ratios: %s",
             pca.explained_variance_ratio_.round(3))
    return reduced_df, pca, market_codes


def evaluate_kmeans(X: np.ndarray, k_range: range) -> pd.DataFrame:
    """
    Run KMeans for each k and return a DataFrame of cluster quality scores.

    Fits a single KMeans per k and extracts both labels and inertia from
    the same model (avoids the double-fit bug).
    """
    log.info("Evaluating KMeans for k in %s...", list(k_range))
    results = []
    for k in k_range:
        model = KMeans(n_clusters=k, n_init=50, random_state=0).fit(X)
        results.append({
            "k": k,
            "inertia": model.inertia_,
            "silhouette": silhouette_score(X, model.labels_),
            "calinski": calinski_harabasz_score(X, model.labels_),
            "davies_bouldin": davies_bouldin_score(X, model.labels_),
        })

    scores = pd.DataFrame(results)
    best = scores.loc[scores["silhouette"].idxmax()]
    log.info("  Best silhouette: %.3f at k=%d", best["silhouette"], int(best["k"]))
    return scores


def cluster_regimes(
    reduced_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """
    Apply standard and balanced KMeans clustering to PCA-reduced data.

    Returns:
        reduced_df: input DataFrame with added 'cluster' and 'balanced_cluster' columns
        scores: KMeans evaluation scores DataFrame
        best_k: the k chosen by silhouette (capped at K_CAP)
    """
    X = StandardScaler().fit_transform(reduced_df.values)

    # Evaluate across range of k
    scores = evaluate_kmeans(X, range(2, MAX_K_SEARCH))
    best_k = int(min(K_CAP, scores.loc[scores["silhouette"].idxmax(), "k"]))
    log.info("Using k=%d (capped at %d)", best_k, K_CAP)

    # Standard KMeans
    reduced_df["cluster"] = KMeans(
        n_clusters=best_k, n_init=100, random_state=42
    ).fit_predict(X)

    # Size-constrained KMeans for balanced buckets
    n = len(X)
    bucket_size = n // BALANCED_K
    if KMeansConstrained is not None:
        balanced = KMeansConstrained(
            n_clusters=BALANCED_K,
            size_min=bucket_size - 2,
            size_max=bucket_size + 2,
            random_state=0,
        )
        reduced_df["balanced_cluster"] = balanced.fit_predict(X)
    else:
        log.warning("k-means-constrained not installed; using standard KMeans "
                    "with k=%d as fallback for balanced clusters", BALANCED_K)
        reduced_df["balanced_cluster"] = KMeans(
            n_clusters=BALANCED_K, n_init=100, random_state=0
        ).fit_predict(X)

    log.info("  %d quarters → %d standard clusters, %d balanced clusters",
             n, best_k, BALANCED_K)
    return reduced_df, scores, best_k
