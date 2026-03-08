"""
Pipeline step 1 — Data Ingestion

Fetches macro data from FRED and multpl.com, merges into one wide DataFrame,
and writes data/raw/macro_raw.parquet.

Run:
    python pipelines/01_ingest.py
"""

import sys
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from market_regime import DATA_DIR
from market_regime.config import load, setup_logging
from market_regime.ingestion import fred as fred_module
from market_regime.ingestion import multpl as multpl_module

import pandas as pd


def main() -> None:
    setup_logging()
    cfg = load()

    # ── FRED ─────────────────────────────────────────────────────────────
    fred_df = fred_module.fetch_all(cfg)

    # ── multpl.com ────────────────────────────────────────────────────────
    multpl_df = multpl_module.fetch_all(cfg)

    # ── Merge ─────────────────────────────────────────────────────────────
    if not multpl_df.empty:
        combined = fred_df.join(multpl_df, how="outer")
    else:
        combined = fred_df

    # Filter to configured date range
    start = cfg["data"]["start_date"]
    combined = combined[combined.index >= start]

    # Persist
    out_path = DATA_DIR / "raw" / "macro_raw.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out_path)
    print(f"Wrote {len(combined)} rows × {len(combined.columns)} cols → {out_path}")


if __name__ == "__main__":
    main()
