"""
Phase D evaluation: impact of momentum features on regime classification.

Compares clustering quality and supervised model accuracy with and without
momentum features added to the feature pipeline.

Three evaluation axes:
  1. Clustering quality — silhouette, Calinski-Harabasz, Davies-Bouldin scores
  2. Supervised accuracy — TimeSeriesSplit RF CV accuracy (current-regime prediction)
  3. Transition detection — momentum feature behavior around known regime transitions

Usage:
    python scripts/evaluate_momentum.py

Requires macro_raw and cluster_labels checkpoints to exist (run pipeline steps 1-3 first).
"""

from __future__ import annotations

import copy
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trading_crab_lib.clustering import reduce_pca, evaluate_kmeans
from trading_crab_lib.config import load
from trading_crab_lib.transforms import engineer_all

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


# ── helpers ────────────────────────────────────────────────────────────────

MOMENTUM_PREFIXES = (
    "sp500_mom_", "sp500_adj_mom_", "10yr_ustreas_mom_",
    "credit_spread_mom_", "corr_sp500_", "cpi_acceleration",
)

# Features that only exist because of momentum (not in baseline)
MOMENTUM_INITIAL_FEATURES = [
    "sp500_mom_4q",
    "sp500_mom_8q",
    "10yr_ustreas_mom_4q",
    "credit_spread_mom_4q",
    "corr_sp500_10yr_ustreas_8q",
    "cpi_acceleration",
]

MOMENTUM_CLUSTERING_FEATURES = [
    "sp500_mom_4q",
    "sp500_mom_4q_d1",
    "sp500_mom_8q",
    "10yr_ustreas_mom_4q",
    "10yr_ustreas_mom_4q_d1",
    "credit_spread_mom_4q",
    "credit_spread_mom_4q_d1",
    "corr_sp500_10yr_ustreas_8q",
    "corr_sp500_10yr_ustreas_8q_d1",
    "cpi_acceleration",
    "cpi_acceleration_d1",
]


def _is_momentum_feature(name: str) -> bool:
    """Check if a feature name is a momentum feature."""
    return any(name.startswith(p) for p in MOMENTUM_PREFIXES)


def _build_features(raw: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Run engineer_all and drop NaN rows for clustering."""
    feat = engineer_all(raw.copy(), cfg, causal=False)
    feat_numeric = feat.drop(columns=["market_code"], errors="ignore")
    feat_clean = feat_numeric.dropna()
    return feat_clean


def _cluster_metrics(feat_clean: pd.DataFrame, n_pca: int, k: int) -> dict:
    """PCA → KMeans → quality metrics at a fixed k."""
    pca_df, pca_obj, scaler = reduce_pca(feat_clean, n_components=n_pca)
    X = StandardScaler().fit_transform(pca_df.values)

    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k, n_init=50, random_state=42)
    labels = km.fit_predict(X)

    return {
        "silhouette": silhouette_score(X, labels),
        "calinski_harabasz": calinski_harabasz_score(X, labels),
        "davies_bouldin": davies_bouldin_score(X, labels),
        "pca_variance_explained": float(np.sum(pca_obj.explained_variance_ratio_)),
        "n_samples": len(feat_clean),
        "n_features": feat_clean.shape[1],
    }


def _kmeans_sweep(feat_clean: pd.DataFrame, n_pca: int) -> pd.DataFrame:
    """PCA → KMeans sweep across k=2..8, return scores DataFrame."""
    pca_df, _, _ = reduce_pca(feat_clean, n_components=n_pca)
    X = StandardScaler().fit_transform(pca_df.values)
    scores = evaluate_kmeans(X, k_range=range(2, 9), n_init=50)
    return scores


def _supervised_cv(
    raw: pd.DataFrame,
    cfg: dict,
    labels: pd.Series,
    n_splits: int = 5,
) -> dict:
    """Build causal features → RF TimeSeriesSplit CV → return mean/std accuracy."""
    feat = engineer_all(raw.copy(), cfg, causal=True)
    feat_numeric = feat.drop(columns=["market_code"], errors="ignore")

    common = feat_numeric.index.intersection(labels.index)
    X = feat_numeric.loc[common].dropna()
    common2 = X.index.intersection(labels.index)
    X = X.loc[common2]
    y = labels.loc[common2]

    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    for train_idx, test_idx in tscv.split(X):
        rf = RandomForestClassifier(
            n_estimators=200, max_depth=12, random_state=42,
            n_jobs=-1, class_weight="balanced",
        )
        rf.fit(X.iloc[train_idx], y.iloc[train_idx])
        scores.append(rf.score(X.iloc[test_idx], y.iloc[test_idx]))

    # Feature importance from full model
    rf_full = RandomForestClassifier(
        n_estimators=200, max_depth=12, random_state=42,
        n_jobs=-1, class_weight="balanced",
    )
    rf_full.fit(X, y)
    importances = pd.Series(rf_full.feature_importances_, index=X.columns)

    return {
        "mean_accuracy": float(np.mean(scores)),
        "std_accuracy": float(np.std(scores)),
        "per_fold": scores,
        "n_features": X.shape[1],
        "n_samples": len(X),
        "top_features": importances.sort_values(ascending=False).head(20),
    }


def _strip_momentum_from_config(cfg: dict) -> dict:
    """Return a config copy with momentum features removed from feature lists."""
    cfg2 = copy.deepcopy(cfg)

    cfg2["features"]["initial_features"] = [
        f for f in cfg2["features"]["initial_features"]
        if not _is_momentum_feature(f)
    ]
    cfg2["features"]["clustering_features"] = [
        f for f in cfg2["features"]["clustering_features"]
        if not _is_momentum_feature(f)
    ]

    return cfg2


def _analyze_transitions(
    raw: pd.DataFrame,
    cfg: dict,
    labels: pd.Series,
) -> pd.DataFrame:
    """Check momentum feature values around regime transition points."""
    feat = engineer_all(raw.copy(), cfg, causal=True)
    feat_numeric = feat.drop(columns=["market_code"], errors="ignore")

    common = feat_numeric.index.intersection(labels.index)
    feat_aligned = feat_numeric.loc[common]
    labels_aligned = labels.loc[common]

    # Find transition quarters (where regime changes)
    transitions = labels_aligned.diff().fillna(0) != 0
    transition_idx = transitions[transitions].index

    mom_cols = [c for c in feat_aligned.columns if _is_momentum_feature(c)]

    if not mom_cols or len(transition_idx) == 0:
        return pd.DataFrame()

    # For each transition, collect momentum values at t-2, t-1, t, t+1
    rows = []
    for t in transition_idx:
        pos = feat_aligned.index.get_loc(t)
        for offset in [-2, -1, 0, 1]:
            idx = pos + offset
            if 0 <= idx < len(feat_aligned):
                row = {"transition_date": t, "offset_q": offset}
                for c in mom_cols:
                    row[c] = feat_aligned[c].iloc[idx]
                rows.append(row)

    return pd.DataFrame(rows)


# ── main ──────────────────────────────────────────────────────────────────


def main():
    cfg = load()
    k = cfg["clustering"]["balanced_k"]
    n_pca = cfg["clustering"]["n_pca_components"]

    # Load checkpoint data
    raw = pd.read_parquet("data/checkpoints/macro_raw.parquet")
    clust = pd.read_parquet("data/checkpoints/cluster_labels.parquet")
    labels = clust["balanced_cluster"]

    cfg_no_mom = _strip_momentum_from_config(cfg)

    print("=" * 72)
    print("PHASE D: Momentum Feature Impact Evaluation")
    print("=" * 72)

    # ── 1. Clustering quality ─────────────────────────────────────────────
    print("\n" + "─" * 72)
    print("1. CLUSTERING QUALITY COMPARISON (k={}, PCA={})".format(k, n_pca))
    print("─" * 72)

    print("\nBuilding features WITHOUT momentum...")
    feat_baseline = _build_features(raw, cfg_no_mom)
    metrics_baseline = _cluster_metrics(feat_baseline, n_pca, k)

    print("Building features WITH momentum...")
    feat_momentum = _build_features(raw, cfg)
    metrics_momentum = _cluster_metrics(feat_momentum, n_pca, k)

    print(f"\n{'Metric':<25s} {'Baseline':>12s} {'+ Momentum':>12s} {'Delta':>10s}")
    print("-" * 60)
    for metric in ["silhouette", "calinski_harabasz", "davies_bouldin",
                    "pca_variance_explained", "n_features", "n_samples"]:
        b = metrics_baseline[metric]
        d = metrics_momentum[metric]
        delta = d - b
        direction = ""
        if metric == "davies_bouldin":
            direction = " (lower is better)"
        elif metric in ("silhouette", "calinski_harabasz"):
            direction = " (higher is better)"
        print(f"  {metric:<23s} {b:>12.4f} {d:>12.4f} {delta:>+10.4f}{direction}")

    # ── 1b. K-sweep comparison ────────────────────────────────────────────
    print("\n" + "─" * 72)
    print("1b. K-SWEEP COMPARISON (k=2..8)")
    print("─" * 72)

    sweep_baseline = _kmeans_sweep(feat_baseline, n_pca)
    sweep_momentum = _kmeans_sweep(feat_momentum, n_pca)

    print(f"\n{'k':>3s}  {'Sil (base)':>10s} {'Sil (+mom)':>10s} {'Delta':>8s}  "
          f"{'CH (base)':>10s} {'CH (+mom)':>10s} {'Delta':>8s}")
    print("-" * 70)
    for _, row_b in sweep_baseline.iterrows():
        k_val = int(row_b["k"])
        row_d = sweep_momentum[sweep_momentum["k"] == k_val].iloc[0]
        sil_b, sil_d = row_b["silhouette"], row_d["silhouette"]
        ch_b, ch_d = row_b["calinski"], row_d["calinski"]
        print(f"  {k_val:>2d}  {sil_b:>10.4f} {sil_d:>10.4f} {sil_d-sil_b:>+8.4f}  "
              f"{ch_b:>10.1f} {ch_d:>10.1f} {ch_d-ch_b:>+8.1f}")

    # ── 2. Supervised accuracy ────────────────────────────────────────────
    print("\n" + "─" * 72)
    print("2. SUPERVISED MODEL ACCURACY (RF, 5-fold TimeSeriesSplit)")
    print("─" * 72)

    print("\nTraining RF WITHOUT momentum features...")
    sup_baseline = _supervised_cv(raw, cfg_no_mom, labels)

    print("Training RF WITH momentum features...")
    sup_momentum = _supervised_cv(raw, cfg, labels)

    print(f"\n{'':>30s} {'Baseline':>12s} {'+ Momentum':>12s} {'Delta':>10s}")
    print("-" * 65)
    print(f"  {'Mean CV accuracy':<28s} {sup_baseline['mean_accuracy']:>12.4f} "
          f"{sup_momentum['mean_accuracy']:>12.4f} "
          f"{sup_momentum['mean_accuracy']-sup_baseline['mean_accuracy']:>+10.4f}")
    print(f"  {'Std CV accuracy':<28s} {sup_baseline['std_accuracy']:>12.4f} "
          f"{sup_momentum['std_accuracy']:>12.4f} "
          f"{sup_momentum['std_accuracy']-sup_baseline['std_accuracy']:>+10.4f}")
    print(f"  {'N features':<28s} {sup_baseline['n_features']:>12d} "
          f"{sup_momentum['n_features']:>12d}")
    print(f"  {'N samples':<28s} {sup_baseline['n_samples']:>12d} "
          f"{sup_momentum['n_samples']:>12d}")

    print("\nPer-fold accuracy:")
    for i, (b, d) in enumerate(zip(sup_baseline["per_fold"], sup_momentum["per_fold"])):
        print(f"  Fold {i+1}: baseline={b:.4f}  +mom={d:.4f}  delta={d-b:+.4f}")

    # Feature importance comparison
    print("\nTop-15 features WITH momentum (RF importance):")
    for feat_name, imp in sup_momentum["top_features"].head(15).items():
        marker = " ** MOM **" if _is_momentum_feature(feat_name) else ""
        print(f"  {feat_name:<40s} {imp:.4f}{marker}")

    # ── 3. Transition detection ───────────────────────────────────────────
    print("\n" + "─" * 72)
    print("3. MOMENTUM BEHAVIOR AROUND REGIME TRANSITIONS")
    print("─" * 72)

    trans_df = _analyze_transitions(raw, cfg, labels)
    if trans_df.empty:
        print("\n  No momentum features or transitions found in aligned data.")
    else:
        mom_cols = [c for c in trans_df.columns if _is_momentum_feature(c)]
        n_transitions = trans_df["transition_date"].nunique()
        print(f"\n  {n_transitions} regime transitions found")
        print(f"  Momentum columns analyzed: {len(mom_cols)}")

        # Focus on key momentum columns
        key_cols = [c for c in mom_cols if c in [
            "sp500_mom_4q", "sp500_mom_8q",
            "10yr_ustreas_mom_4q", "credit_spread_mom_4q",
            "corr_sp500_10yr_ustreas_8q", "cpi_acceleration",
        ]]

        if key_cols:
            print(f"\n  Mean |momentum value| by quarter offset from transition:")
            print(f"  {'Offset':>8s}", end="")
            for c in key_cols:
                short = c.replace("_mom_", "M").replace("10yr_ustreas", "10y")
                short = short[:14]
                print(f"  {short:>14s}", end="")
            print()
            print("  " + "-" * (8 + 16 * len(key_cols)))
            for offset in [-2, -1, 0, 1]:
                subset = trans_df[trans_df["offset_q"] == offset]
                print(f"  {offset:>+8d}", end="")
                for c in key_cols:
                    mean_abs = subset[c].abs().mean()
                    print(f"  {mean_abs:>14.4f}", end="")
                print()

        # Compare magnitude at transition vs baseline
        print(f"\n  Transition vs baseline magnitude comparison:")
        for c in key_cols:
            t0 = trans_df[trans_df["offset_q"] == 0]
            t_2 = trans_df[trans_df["offset_q"] == -2]
            mag_at_t0 = t0[c].abs().mean()
            mag_at_t2 = t_2[c].abs().mean()
            short = c.replace("_mom_", "M").replace("10yr_ustreas", "10y")
            if mag_at_t0 > mag_at_t2 * 1.2:
                verdict = "SIGNAL — elevated at transitions"
            elif mag_at_t0 < mag_at_t2 * 0.8:
                verdict = "NO SIGNAL — lower at transitions"
            else:
                verdict = "INCONCLUSIVE — similar magnitude"
            print(f"  {short:<30s} t0={mag_at_t0:.4f} vs t-2={mag_at_t2:.4f} → {verdict}")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)

    sil_delta = metrics_momentum["silhouette"] - metrics_baseline["silhouette"]
    acc_delta = sup_momentum["mean_accuracy"] - sup_baseline["mean_accuracy"]

    print(f"\n  Clustering silhouette change:   {sil_delta:+.4f} "
          f"({'improved' if sil_delta > 0 else 'degraded' if sil_delta < 0 else 'unchanged'})")
    print(f"  Supervised CV accuracy change:  {acc_delta:+.4f} "
          f"({'improved' if acc_delta > 0 else 'degraded' if acc_delta < 0 else 'unchanged'})")

    # Momentum feature importance ranking
    mom_importance = {
        k: v for k, v in sup_momentum["top_features"].items()
        if _is_momentum_feature(k)
    }
    if mom_importance:
        best_mom = max(mom_importance, key=mom_importance.get)
        print(f"  Top momentum feature (RF importance): {best_mom} = {mom_importance[best_mom]:.4f}")
        print(f"  Momentum features in top-20: {len(mom_importance)}")
    else:
        print("  No momentum features appeared in top-20 RF importance")

    print()


if __name__ == "__main__":
    main()
