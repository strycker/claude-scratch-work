"""
Pipeline step 3 — Unsupervised Clustering

Reads features.parquet, runs:
  1. PCA (fixed n_components from config)
  2. KMeans k-sweep (silhouette / CH / DB scoring)
  3. Standard KMeans at best_k
  4. Size-constrained KMeans at balanced_k

Writes:
  data/regimes/cluster_labels.parquet   — quarter → cluster, balanced_cluster
  data/regimes/pca_components.parquet   — quarter → PC1…PCn
  data/regimes/kmeans_scores.parquet    — k-sweep evaluation table

Run:
    python pipelines/03_cluster.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from market_regime import DATA_DIR
from market_regime.config import load, setup_logging
from market_regime.clustering import (
    reduce_pca,
    evaluate_kmeans,
    pick_best_k,
    fit_clusters,
)

import pandas as pd


def main() -> None:
    setup_logging()
    cfg = load()
    clust_cfg = cfg["clustering"]

    features = pd.read_parquet(DATA_DIR / "processed" / "features.parquet")
    X = features.drop(columns=["market_code"], errors="ignore")
    print(f"\nLoaded features: {X.shape}")

    # ── 1. PCA ─────────────────────────────────────────────────────────────
    # Library logs: "Running PCA... done." + variance ratios
    pca_df, pca_model, scaler = reduce_pca(
        X,
        n_components=clust_cfg["n_pca_components"],
        random_state=clust_cfg["random_state"],
    )

    # ── 2. Evaluate k values ────────────────────────────────────────────────
    # Library logs: "Evaluating cluster counts... done." + full sorted table
    from sklearn.preprocessing import StandardScaler
    X_scaled = StandardScaler().fit_transform(pca_df.values)
    scores = evaluate_kmeans(
        X_scaled,
        k_range=range(2, clust_cfg["n_clusters_search"] + 1),
        random_state=clust_cfg["random_state"],
    )
    best_k = pick_best_k(scores, k_cap=clust_cfg["k_cap"])
    print(f"\nChosen k={best_k}  (silhouette winner capped at k_cap={clust_cfg['k_cap']})")

    # ── 3 & 4. Fit both clusterings ─────────────────────────────────────────
    # Library logs: "N quarters clustered into K regimes (balanced into K)."
    clustered = fit_clusters(
        pca_df,
        best_k=best_k,
        balanced_k=clust_cfg["balanced_k"],
        random_state=clust_cfg["random_state"],
    )

    # Restore market_code for downstream steps
    if "market_code" in features.columns:
        clustered["market_code"] = features["market_code"]

    # ── Persist ─────────────────────────────────────────────────────────────
    out_dir = DATA_DIR / "regimes"
    out_dir.mkdir(parents=True, exist_ok=True)

    label_cols = ["cluster", "balanced_cluster"] + (
        ["market_code"] if "market_code" in clustered.columns else []
    )
    clustered[label_cols].to_parquet(out_dir / "cluster_labels.parquet")
    clustered.drop(columns=label_cols, errors="ignore").to_parquet(out_dir / "pca_components.parquet")
    scores.to_parquet(out_dir / "kmeans_scores.parquet", index=False)

    print(f"\nStandard clusters (k={best_k}):")
    print(clustered["cluster"].value_counts().sort_index().to_string())

    print(f"\nBalanced clusters (k={clust_cfg['balanced_k']}):")
    print(clustered["balanced_cluster"].value_counts().sort_index().to_string())

    print(f"\nOutputs written to {out_dir}")


if __name__ == "__main__":
    main()
