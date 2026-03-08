"""
Pipeline step 4 — Regime Profiling & Labeling

Reads cluster_labels + raw features, computes per-cluster statistics,
suggests human-readable names, and writes:
  data/regimes/profiles.parquet
  data/regimes/transition_matrix.parquet
  data/regimes/regime_names.yaml   — auto-suggested (edit manually)

Run:
    python pipelines/04_regime_label.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from market_regime import DATA_DIR, CONFIG_DIR
from market_regime.config import load, setup_logging
from market_regime.regime.profiler import (
    build_profiles,
    suggest_names,
    build_transition_matrix,
    load_name_overrides,
)

import pandas as pd
import yaml


def main() -> None:
    setup_logging()
    cfg = load()

    features = pd.read_parquet(DATA_DIR / "processed" / "features.parquet")
    labels = pd.read_parquet(DATA_DIR / "regimes" / "cluster_labels.parquet")["cluster"]

    # Align index (features may have more rows if gap-filled beyond label dates)
    common = features.index.intersection(labels.index)
    features = features.loc[common]
    labels = labels.loc[common]

    # Profile
    profile = build_profiles(features, labels)
    profile.to_parquet(DATA_DIR / "regimes" / "profiles.parquet")

    # Use median stat for naming heuristics
    median_df = features.copy()
    median_df["cluster"] = labels
    median_per_cluster = median_df.groupby("cluster").median()

    # Auto-suggest names, then check for manual overrides
    auto_names = suggest_names(profile, median_per_cluster)
    overrides = load_name_overrides(CONFIG_DIR)
    regime_names = {**auto_names, **overrides}

    # Save auto-suggestions so user can review / edit config/regime_labels.yaml
    suggestions_path = DATA_DIR / "regimes" / "regime_names_suggested.yaml"
    with open(suggestions_path, "w") as f:
        yaml.dump(regime_names, f, default_flow_style=False)
    print(f"Regime name suggestions → {suggestions_path}")

    # Transition matrix
    tm = build_transition_matrix(labels)
    tm.to_parquet(DATA_DIR / "regimes" / "transition_matrix.parquet")

    print("\nRegime summary:")
    for rid, name in sorted(regime_names.items()):
        n = (labels == rid).sum()
        print(f"  Cluster {rid}: {name!r}  ({n} quarters)")

    print("\nTransition matrix (row=from, col=to):")
    print(tm.round(2).to_string())


if __name__ == "__main__":
    main()
