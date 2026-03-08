"""
Pipeline step 3 — Unsupervised Clustering

Reads features.parquet, runs PCA + KMeans, and writes:
  data/regimes/cluster_labels.parquet   — quarter → cluster_id
  data/regimes/pca_components.parquet   — quarter → pca_0, pca_1, …

Run:
    python pipelines/03_cluster.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from market_regime import DATA_DIR
from market_regime.config import load, setup_logging
from market_regime.clustering.kmeans import reduce_pca, fit_clusters

import pandas as pd


def main() -> None:
    setup_logging()
    cfg = load()
    clust_cfg = cfg["clustering"]

    features = pd.read_parquet(DATA_DIR / "processed" / "features.parquet")
    print(f"Loaded features: {features.shape}")

    pca_df, pca_model = reduce_pca(
        features,
        variance_threshold=clust_cfg["pca_variance_threshold"],
        random_state=clust_cfg["random_state"],
    )

    clustered = fit_clusters(
        pca_df,
        n_clusters=clust_cfg["n_clusters"],
        size_constrained=clust_cfg["size_constrained"],
        min_cluster_size=clust_cfg["min_cluster_size"],
        random_state=clust_cfg["random_state"],
    )

    out_dir = DATA_DIR / "regimes"
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = clustered[["cluster"]]
    labels.to_parquet(out_dir / "cluster_labels.parquet")
    clustered.drop(columns=["cluster"]).to_parquet(out_dir / "pca_components.parquet")

    print(f"Wrote cluster labels → {out_dir / 'cluster_labels.parquet'}")
    print(clustered["cluster"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
