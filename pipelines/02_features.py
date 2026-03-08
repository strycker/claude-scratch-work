"""
Pipeline step 2 — Feature Engineering

Reads data/raw/macro_raw.parquet, applies log transforms, smoothed
derivatives, cross-ratios, and Bernstein gap filling.
Writes data/processed/features.parquet.

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

    features = engineer_all(raw, cfg)

    out_path = DATA_DIR / "processed" / "features.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(out_path)
    print(f"Wrote {features.shape} → {out_path}")


if __name__ == "__main__":
    main()
