"""
Multi-clustering comparison and RF-driven feature selection for clustering.

Two responsibilities:
  1. compare_all_methods()         — score and rank all clustering methods side-by-side
  2. extract_rf_feature_importances() — read the step-5 RF model and rank features
  3. recommend_clustering_features()  — suggest a leaner feature set for re-clustering

Comparison metrics used
------------------------
- Silhouette score    — higher is better; [-1, 1]
- Davies-Bouldin      — lower is better; [0, ∞)
- Calinski-Harabasz  — higher is better; [0, ∞)
- Adjusted Rand Index — pairwise label agreement; [-0.5, 1]

CLI workflow after finding best clustering
-------------------------------------------
Once you identify the best method in the notebook:

  1. Save its labels as a named checkpoint:
         python run_pipeline.py --steps 3 --save-market-code
     (saves balanced_cluster as market_code_clustered)

  2. Propagate through downstream steps:
         python run_pipeline.py --steps 4,5,6,7 --market-code clustered --plots

  3. Update config/regime_labels.yaml with new regime names.

Usage
------
    from market_regime.cluster_comparison import (
        compare_all_methods, plot_comparison, plot_rand_heatmap,
        extract_rf_feature_importances, recommend_clustering_features,
    )

    summary = compare_all_methods(pca_df, {
        "kmeans_balanced": balanced_labels,
        "gmm_best":        gmm_labels,
        "spectral_5":      spectral_labels,
    })
    importances = extract_rf_feature_importances(model_path, feature_names)
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)


def compare_all_methods(
    pca_df: pd.DataFrame,
    labels_dict: dict[str, pd.Series],
) -> pd.DataFrame:
    """
    Compute clustering quality metrics for each method in labels_dict.

    Noise points (label = -1, e.g. from DBSCAN) are excluded from metric
    computation but reported in the 'n_noise' column.

    Args:
        pca_df      — PCA-reduced feature matrix (for computing metrics)
        labels_dict — {method_name: pd.Series of integer labels}

    Returns:
        DataFrame with one row per method, columns:
        method, n_clusters, n_noise, silhouette, davies_bouldin, calinski
        Sorted by silhouette descending.
    """
    X = StandardScaler().fit_transform(pca_df.values)
    rows: list[dict] = []

    for name, labels in labels_dict.items():
        # Align index to pca_df
        aligned = labels.reindex(pca_df.index).dropna().astype(int)
        n_noise = int((aligned == -1).sum())
        mask = aligned != -1
        X_clean = X[pca_df.index.isin(aligned[mask].index)]
        labels_clean = aligned[mask].values

        n_clusters = len(set(labels_clean))

        sil = db = ch = float("nan")
        if n_clusters >= 2 and len(labels_clean) >= n_clusters:
            try:
                sil = silhouette_score(X_clean, labels_clean)
                db  = davies_bouldin_score(X_clean, labels_clean)
                ch  = calinski_harabasz_score(X_clean, labels_clean)
            except Exception as exc:
                log.warning("%s: metric computation failed — %s", name, exc)

        rows.append({
            "method":        name,
            "n_clusters":    n_clusters,
            "n_noise":       n_noise,
            "silhouette":    sil,
            "davies_bouldin": db,
            "calinski":      ch,
        })
        log.info(
            "%-30s  k=%d  noise=%d  sil=%.4f  DB=%.4f  CH=%.1f",
            name, n_clusters, n_noise,
            sil if np.isfinite(sil) else -99,
            db  if np.isfinite(db)  else -99,
            ch  if np.isfinite(ch)  else -99,
        )

    df = pd.DataFrame(rows)
    return df.sort_values("silhouette", ascending=False).reset_index(drop=True)


def pairwise_rand_index(labels_dict: dict[str, pd.Series]) -> pd.DataFrame:
    """
    Compute adjusted Rand index for every pair of clustering methods.

    Returns:
        Square DataFrame with method names as index and columns.
        Diagonal = 1.0; off-diagonal = ARI(method_i, method_j).
    """
    names = list(labels_dict.keys())
    n = len(names)
    matrix = np.eye(n)

    for i in range(n):
        for j in range(i + 1, n):
            s1 = labels_dict[names[i]]
            s2 = labels_dict[names[j]]
            common = s1.index.intersection(s2.index)
            # Exclude noise points from both
            a = s1.loc[common]
            b = s2.loc[common]
            valid = (a >= 0) & (b >= 0)
            if valid.sum() < 2:
                ari = float("nan")
            else:
                ari = adjusted_rand_score(a[valid].values, b[valid].values)
            matrix[i, j] = matrix[j, i] = ari

    return pd.DataFrame(matrix, index=names, columns=names)


# ── RF feature importance for clustering feature selection ─────────────────────

def extract_rf_feature_importances(
    model_path: Path,
    feature_names: list[str] | None = None,
) -> pd.Series:
    """
    Load a pickled sklearn RF model and return feature importances as a Series.

    The model must have a .feature_importances_ attribute (RandomForestClassifier).
    If feature_names is None, tries model.feature_names_in_.

    Args:
        model_path    — path to pickled model file
        feature_names — optional list of feature names (overrides model's own)

    Returns:
        Series indexed by feature name, values = importance, sorted descending.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    if not hasattr(model, "feature_importances_"):
        raise AttributeError(f"{model_path.name} does not have feature_importances_ (not a tree-based model)")

    if feature_names is None:
        if hasattr(model, "feature_names_in_"):
            feature_names = list(model.feature_names_in_)
        else:
            feature_names = [f"feature_{i}" for i in range(len(model.feature_importances_))]

    importances = pd.Series(
        model.feature_importances_,
        index=feature_names,
        name="importance",
    ).sort_values(ascending=False)

    log.info("RF feature importances loaded: %d features from %s", len(importances), model_path.name)
    return importances


def recommend_clustering_features(
    importances: pd.Series,
    current_clustering_features: list[str],
    top_k: int = 35,
) -> tuple[list[str], pd.DataFrame]:
    """
    Suggest a leaner clustering feature set based on RF importances.

    Only considers features that are already in current_clustering_features
    (the RF may have been trained on a different feature set including
    non-clustering features — we only care about intersection).

    Args:
        importances               — output of extract_rf_feature_importances()
        current_clustering_features — list from config clustering_features
        top_k                     — number of features to recommend

    Returns:
        recommended_features — list of top_k features (intersection)
        comparison_df        — DataFrame showing all clustering features with
                               their RF importance rank and whether they are
                               in the recommended set
    """
    # Features that are in both the RF model and the clustering_features list
    in_both = [f for f in current_clustering_features if f in importances.index]
    not_in_rf = [f for f in current_clustering_features if f not in importances.index]

    if not_in_rf:
        log.warning(
            "%d clustering_features not found in RF model (may be derivatives "
            "not used in supervised step): %s",
            len(not_in_rf), not_in_rf[:5],
        )

    # Rank by importance (descending) among the intersection
    ranked = importances.loc[in_both].sort_values(ascending=False)
    recommended = list(ranked.head(top_k).index)

    # Build comparison table
    rows = []
    for rank, (feat, imp) in enumerate(ranked.items(), start=1):
        rows.append({
            "feature":          feat,
            "rf_importance":    round(float(imp), 6),
            "rank":             rank,
            "in_recommended":   feat in recommended,
        })
    for feat in not_in_rf:
        rows.append({
            "feature":        feat,
            "rf_importance":  float("nan"),
            "rank":           len(ranked) + 1,
            "in_recommended": False,
        })

    comparison_df = pd.DataFrame(rows).sort_values("rank").reset_index(drop=True)

    log.info(
        "Recommended %d/%d clustering features (top-%d by RF importance)",
        len(recommended), len(current_clustering_features), top_k,
    )
    return recommended, comparison_df
