"""
Regime profiling — characterise each cluster using the original (pre-PCA)
features so the resulting statistics are interpretable.

Outputs:
  - A profile DataFrame (cluster × feature → mean/median/std)
  - Auto-generated regime name suggestions based on key indicator levels
  - A transition matrix (empirical probabilities of cluster-to-cluster moves)
"""

import logging
from pathlib import Path

import pandas as pd
import yaml

log = logging.getLogger(__name__)

# Key features and which direction = "high" for naming heuristics
# (feature, high_label, low_label)
NAMING_HEURISTICS = [
    ("cpi",          "High Inflation",   "Low Inflation"),
    ("unemployment", "High Unemployment","Low Unemployment"),
    ("gdp",          "Strong GDP",       "Weak GDP"),
    ("treasury_10y", "High Rates",       "Low Rates"),
    ("fed_funds",    "Tight Policy",     "Easy Policy"),
]


def build_profiles(
    features_df: pd.DataFrame,
    cluster_labels: pd.Series,
    stats: list[str] = ("mean", "median", "std"),
) -> pd.DataFrame:
    """
    Compute per-cluster descriptive statistics for all original features.

    Args:
        features_df    — wide DataFrame of (possibly engineered) features
        cluster_labels — Series aligned to features_df index, dtype int
        stats          — which aggregations to compute

    Returns:
        MultiIndex DataFrame: (cluster, stat) × feature
    """
    joined = features_df.copy()
    joined["cluster"] = cluster_labels
    profile = joined.groupby("cluster").agg(list(stats))
    log.info("Built profiles for %d clusters across %d features",
             profile.index.nunique(), len(features_df.columns) - 1)
    return profile


def suggest_names(profile: pd.DataFrame, median_df: pd.DataFrame) -> dict[int, str]:
    """
    Very simple heuristic name suggestion based on median values of key signals
    relative to their cross-cluster median.

    Returns dict mapping cluster_id → suggested_name string.
    """
    global_medians = median_df.median()
    names: dict[int, str] = {}

    for cluster_id in median_df.index:
        tags = []
        for feat, high_label, low_label in NAMING_HEURISTICS:
            if feat not in median_df.columns:
                continue
            val = median_df.loc[cluster_id, feat]
            gm = global_medians[feat]
            if val > gm * 1.10:
                tags.append(high_label)
            elif val < gm * 0.90:
                tags.append(low_label)

        names[cluster_id] = " / ".join(tags) if tags else f"Regime {cluster_id}"
        log.info("Cluster %d → %s", cluster_id, names[cluster_id])

    return names


def build_transition_matrix(cluster_labels: pd.Series) -> pd.DataFrame:
    """
    Compute empirical quarter-over-quarter regime transition probabilities.

    Returns a K×K DataFrame where entry [i, j] = P(next=j | current=i).
    """
    k = cluster_labels.nunique()
    labels = cluster_labels.values

    counts = pd.DataFrame(0, index=range(k), columns=range(k))
    for t in range(len(labels) - 1):
        counts.loc[labels[t], labels[t + 1]] += 1

    # Normalise rows to probabilities
    row_sums = counts.sum(axis=1).replace(0, 1)
    matrix = counts.div(row_sums, axis=0)
    matrix.index.name = "from_regime"
    matrix.columns.name = "to_regime"
    return matrix


def load_name_overrides(config_dir: Path) -> dict[int, str]:
    """Load any manually pinned regime names from regime_labels.yaml."""
    path = config_dir / "regime_labels.yaml"
    if not path.exists():
        return {}
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return {int(k): v for k, v in raw.items() if not str(k).startswith("#")}
