"""
PCA dimensionality reduction followed by KMeans clustering.

Returns a DataFrame with columns:
    pca_0, pca_1, …   — principal components retained
    cluster           — integer cluster label

Design note: we expose a plain KMeans path and an optional size-constrained
path (requires `pip install k-means-constrained`).  If the library is absent,
we fall back to standard KMeans with a warning.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

log = logging.getLogger(__name__)


def _load_constrained_kmeans():
    try:
        from k_means_constrained import KMeansConstrained
        return KMeansConstrained
    except ImportError:
        log.warning(
            "k-means-constrained not installed; "
            "falling back to standard KMeans. "
            "Run: pip install k-means-constrained"
        )
        return None


def reduce_pca(
    df: pd.DataFrame,
    variance_threshold: float = 0.90,
    random_state: int = 42,
) -> tuple[pd.DataFrame, PCA]:
    """
    Standardise features and reduce to the number of PCA components that
    explain >= variance_threshold of total variance.

    Returns:
        pca_df   — DataFrame of principal components, same index as df
        pca_obj  — fitted PCA (retained for later scoring / inspection)
    """
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df.dropna(axis=1))  # drop all-NaN columns

    pca = PCA(random_state=random_state)
    pca.fit(scaled)

    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_components = int(np.searchsorted(cumvar, variance_threshold)) + 1
    log.info(
        "PCA: retaining %d components (%.1f%% variance)",
        n_components, cumvar[n_components - 1] * 100,
    )

    pca_final = PCA(n_components=n_components, random_state=random_state)
    components = pca_final.fit_transform(scaled)

    col_names = [f"pca_{i}" for i in range(n_components)]
    pca_df = pd.DataFrame(components, index=df.dropna(axis=1).index, columns=col_names)
    return pca_df, pca_final


def fit_clusters(
    pca_df: pd.DataFrame,
    n_clusters: int = 6,
    size_constrained: bool = True,
    min_cluster_size: int = 8,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Cluster the PCA-reduced data.  Appends a 'cluster' column.

    Args:
        pca_df           — output of reduce_pca()
        n_clusters       — K
        size_constrained — attempt equal-size clusters if True
        min_cluster_size — minimum quarters per cluster (for constrained KMeans)
        random_state

    Returns:
        pca_df with an added 'cluster' integer column.
    """
    X = pca_df.values

    if size_constrained:
        KMC = _load_constrained_kmeans()
        if KMC is not None:
            size_max = len(X) // n_clusters + min_cluster_size
            model = KMC(
                n_clusters=n_clusters,
                size_min=min_cluster_size,
                size_max=size_max,
                random_state=random_state,
            )
            labels = model.fit_predict(X)
            log.info("Size-constrained KMeans: %d clusters", n_clusters)
        else:
            labels = _plain_kmeans(X, n_clusters, random_state)
    else:
        labels = _plain_kmeans(X, n_clusters, random_state)

    result = pca_df.copy()
    result["cluster"] = labels
    _log_cluster_sizes(result["cluster"])
    return result


def _plain_kmeans(X: np.ndarray, n_clusters: int, random_state: int) -> np.ndarray:
    model = KMeans(n_clusters=n_clusters, n_init="auto", random_state=random_state)
    labels = model.fit_predict(X)
    log.info("Standard KMeans: %d clusters", n_clusters)
    return labels


def _log_cluster_sizes(labels: pd.Series) -> None:
    counts = labels.value_counts().sort_index()
    sizes = ", ".join(f"{k}:{v}" for k, v in counts.items())
    log.info("Cluster sizes — %s", sizes)
