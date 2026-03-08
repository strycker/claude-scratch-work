"""
Pipeline step 6 — Asset Returns by Regime

Placeholder: loads asset prices (from a CSV you supply), computes quarterly
returns, and profiles each regime.

To use: place a CSV at data/raw/asset_prices.csv with columns:
  date, SPY, GLD, TLT, USO, QQQ, IWM, VNQ, AGG
  (date in YYYY-MM-DD format, prices adjusted close)

Writes data/regimes/asset_return_profile.parquet

Run:
    python pipelines/06_asset_returns.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from market_regime import DATA_DIR
from market_regime.config import load, setup_logging
from market_regime.assets.returns import (
    compute_quarterly_returns,
    returns_by_regime,
    rank_assets_by_regime,
)

import pandas as pd


def main() -> None:
    setup_logging()
    cfg = load()

    prices_path = DATA_DIR / "raw" / "asset_prices.csv"
    if not prices_path.exists():
        print(f"No asset price file found at {prices_path}")
        print("Skipping step 6 — populate asset_prices.csv to continue.")
        return

    prices = pd.read_csv(prices_path, index_col="date", parse_dates=True)
    labels = pd.read_parquet(DATA_DIR / "regimes" / "cluster_labels.parquet")["cluster"]

    returns = compute_quarterly_returns(prices)

    common = returns.index.intersection(labels.index)
    profile = returns_by_regime(returns.loc[common], labels.loc[common])
    ranked = rank_assets_by_regime(profile)

    out = DATA_DIR / "regimes" / "asset_return_profile.parquet"
    profile.to_parquet(out)
    print(f"Wrote asset return profile → {out}")

    print("\nTop assets per regime (by median quarterly return):")
    print(ranked.to_string(index=False))


if __name__ == "__main__":
    main()
