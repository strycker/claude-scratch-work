"""
Regime profiling — characterise each cluster using original (pre-PCA) features.

Three main outputs:
  build_profiles()          → per-cluster mean/median/std table
  suggest_names()           → heuristic human-readable labels
  build_transition_matrix() → empirical quarter-to-quarter transition probabilities

Naming heuristics use ACTUAL column names from the feature schema
(clustering_features in settings.yaml). A missing column silently skips that
heuristic rather than crashing.
"""

import logging
from pathlib import Path

import pandas as pd
import yaml

log = logging.getLogger(__name__)


# ── Naming heuristics ─────────────────────────────────────────────────────
#
# Each entry: (column, above-median label, below-median label, threshold)
# threshold: fractional deviation from global median required to fire.
#   0.20 = must be 20% above/below to label.
#
# Listed in priority order; only the top 3 tags are used per cluster.
# Uses actual column names from the clustering_features schema.
#
NAMING_HEURISTICS: list[tuple[str, str, str, float]] = [
    # Inflation
    ("us_infl",              "High Inflation",     "Low Inflation",       0.20),
    ("log_cpi_d1",           "Rising CPI",         "Falling CPI",         0.10),
    ("log_fred_cpi_d1",      "Rising CPI",         "Falling CPI",         0.10),
    # Growth
    ("gdp_growth",           "Strong Growth",      "Weak/Neg Growth",     0.20),
    ("real_gdp_growth",      "Strong Real Growth", "Weak Real Growth",    0.20),
    ("log_fred_gdp_d1",      "GDP Expanding",      "GDP Contracting",     0.10),
    # Rates / monetary
    ("10yr_ustreas",         "High Rates",         "Low Rates",           0.20),
    ("fred_gs10",            "High Rates",         "Low Rates",           0.20),
    ("fred_tb3ms",           "Tight Short Rates",  "Easy Short Rates",    0.20),
    ("10yr_ustreas_d1",      "Rates Rising",       "Rates Falling",       0.10),
    # Credit / risk
    ("credit_spread",        "Wide Credit Spread", "Tight Credit Spread", 0.20),
    ("div_minus_baa",        "High Div Premium",   "Low Div Premium",     0.10),
    # Equity valuation
    ("sp500_pe",             "High Valuations",    "Low Valuations",      0.20),
    ("log_cape_shiller_d1",  "Valuations Rising",  "Valuations Falling",  0.10),
    # Earnings / dividends
    ("log_earn_d1",          "Earnings Growing",   "Earnings Declining",  0.10),
    ("log_div_yield_d1",     "Yield Rising",       "Yield Falling",       0.10),
]


def build_profiles(
    features_df: pd.DataFrame,
    cluster_labels: pd.Series,
    stats: list[str] | None = None,
) -> pd.DataFrame:
    """
    Compute per-cluster descriptive statistics for all features.

    Args:
        features_df    — feature matrix (clustering_features or broader set)
        cluster_labels — integer Series aligned with features_df index
        stats          — agg functions; defaults to ["mean", "median", "std"]

    Returns:
        DataFrame with MultiIndex columns (stat, feature), rows = cluster IDs.
    """
    if stats is None:
        stats = ["mean", "median", "std"]

    joined = features_df.copy()
    joined["_cluster"] = cluster_labels.reindex(joined.index)
    joined = joined.dropna(subset=["_cluster"])

    profile = joined.groupby("_cluster").agg(stats)
    log.info(
        "Built profiles: %d clusters × %d features × %d stats",
        len(profile), len(features_df.columns), len(stats),
    )
    return profile


def suggest_names(
    features_df: pd.DataFrame,
    cluster_labels: pd.Series,
) -> dict[int, str]:
    """
    Heuristic regime names based on per-cluster medians vs global medians.

    Returns dict mapping cluster_id → suggested name string.
    Falls back to "Regime {id}" when no heuristics fire.
    """
    joined = features_df.copy()
    joined["_cluster"] = cluster_labels.reindex(joined.index)
    joined = joined.dropna(subset=["_cluster"])

    cluster_medians = joined.groupby("_cluster").median()
    global_medians = joined.drop(columns=["_cluster"]).median()

    names: dict[int, str] = {}
    for cid in sorted(cluster_medians.index):
        tags: list[str] = []
        for col, high_lbl, low_lbl, threshold in NAMING_HEURISTICS:
            if col not in cluster_medians.columns:
                continue
            gm = global_medians[col]
            if gm == 0:
                continue
            cm = cluster_medians.loc[cid, col]
            if cm > gm * (1 + threshold):
                tags.append(high_lbl)
            elif cm < gm * (1 - threshold):
                tags.append(low_lbl)

        # Deduplicate while preserving priority order, cap at 3 tags
        seen: set[str] = set()
        unique_tags: list[str] = []
        for t in tags:
            if t not in seen:
                seen.add(t)
                unique_tags.append(t)

        names[int(cid)] = " / ".join(unique_tags[:3]) if unique_tags else f"Regime {int(cid)}"
        log.info("Cluster %d → %s", int(cid), names[int(cid)])

    return names


def build_transition_matrix(cluster_labels: pd.Series) -> pd.DataFrame:
    """
    Compute empirical quarter-over-quarter regime transition probabilities.

    Args:
        cluster_labels — integer Series of cluster IDs, time-ordered

    Returns:
        DataFrame (k × k): entry [i, j] = P(next regime = j | current = i).
    """
    labels = cluster_labels.dropna().astype(int)
    k_vals = sorted(labels.unique())
    counts = pd.DataFrame(0, index=k_vals, columns=k_vals)

    vals = labels.values
    for t in range(len(vals) - 1):
        counts.loc[vals[t], vals[t + 1]] += 1

    row_sums = counts.sum(axis=1).replace(0, 1)
    matrix = counts.div(row_sums, axis=0)
    matrix.index.name = "from_regime"
    matrix.columns.name = "to_regime"

    log.info("Transition matrix built: %d regimes", len(k_vals))
    return matrix


def load_name_overrides(config_dir: Path) -> dict[int, str]:
    """
    Load manually pinned regime names from config/regime_labels.yaml.

    These take precedence over auto-suggested names from suggest_names().
    Returns empty dict if the file doesn't exist or has no valid entries.
    """
    path = config_dir / "regime_labels.yaml"
    if not path.exists():
        return {}
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    overrides = {int(k): v for k, v in raw.items() if not str(k).startswith("#")}
    if overrides:
        log.info("Loaded %d manual regime name overrides", len(overrides))
    return overrides
