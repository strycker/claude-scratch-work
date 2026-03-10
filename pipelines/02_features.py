"""
Pipeline step 2 — Feature Engineering

Reads data/raw/macro_raw.parquet, applies log transforms, smoothed
derivatives, cross-ratios, and Bernstein gap filling.

Writes two feature files:
  data/processed/features.parquet            — centered rolling windows
                                               (for clustering in step 3-4)
  data/processed/features_supervised.parquet — causal/backward rolling windows
                                               (for supervised learning in step 5-7;
                                               no look-ahead bias)

Run:
    python pipelines/02_features.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from market_regime import DATA_DIR
from market_regime.config import load, setup_logging
from market_regime.features.transforms import engineer_all

import pandas as pd


def main() -> None:
    setup_logging()
    cfg = load()

    raw = pd.read_parquet(DATA_DIR / "raw" / "macro_raw.parquet")
    print(f"Loaded raw data: {raw.shape}")

    out_dir = DATA_DIR / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Centered features — for clustering (steps 3-4)
    features = engineer_all(raw, cfg, causal=False)
    out_path = out_dir / "features.parquet"
    features.to_parquet(out_path)
    print(f"Wrote {features.shape} → {out_path}  (centered)")

    # Causal features — for supervised learning and live scoring (steps 5-7)
    features_sup = engineer_all(raw, cfg, causal=True)
    out_path_sup = out_dir / "features_supervised.parquet"
    features_sup.to_parquet(out_path_sup)
    print(f"Wrote {features_sup.shape} → {out_path_sup}  (causal/backward)")


if __name__ == "__main__":
    main()
