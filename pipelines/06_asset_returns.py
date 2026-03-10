"""
Pipeline step 6 — Asset Returns by Regime

Fetches ETF price history via yfinance (SPY, GLD, TLT, USO, QQQ, IWM, VNQ, AGG),
computes quarterly returns, and profiles median return per regime.

Fallback: when yfinance is unavailable (SSL failure, network outage, or no cached
prices), derives proxy returns directly from macro columns already present in
data/raw/macro_raw.parquet (sp500, sp500_adj, 10yr_ustreas, gdp_growth, us_infl,
credit_spread).  Coverage extends back to ~1950, so every historical regime is
represented even without ETF data.

Priority order:
  1. yfinance (real ETF data — most accurate for recent periods)
     Only used when --refresh-assets is passed OR no cache exists.
  2. Cached asset_prices.parquet (if yfinance is temporarily unavailable)
  3. Macro-data proxy returns (fallback for SSL/network failures or back-history)

SSL note:
  Behind a corporate firewall with HTTPS inspection?  See the message printed
  by fetch_all() for remediation steps.  The simplest workaround:
    export CURL_CA_BUNDLE=""
    export REQUESTS_CA_BUNDLE=""
  Or run without --refresh-assets to skip yfinance and use the checkpoint.

Writes data/regimes/asset_return_profile.parquet

Run:
    python pipelines/06_asset_returns.py                  # use cache if available
    python pipelines/06_asset_returns.py --refresh-assets  # force re-fetch yfinance
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from market_regime import DATA_DIR
from market_regime.config import load, setup_logging
from market_regime.asset_returns import (
    compute_quarterly_returns,
    compute_proxy_returns,
    returns_by_regime,
    rank_assets_by_regime,
)

import pandas as pd

log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 6 — Asset returns by regime")
    parser.add_argument(
        "--refresh-assets", action="store_true",
        help=(
            "Force re-fetch ETF prices from yfinance even if a cached "
            "data/raw/asset_prices.parquet already exists.  "
            "Without this flag, the cached file is used (useful when behind a firewall)."
        ),
    )
    args = parser.parse_args()
    refresh_assets: bool = args.refresh_assets

    setup_logging()
    cfg = load()

    labels = pd.read_parquet(DATA_DIR / "regimes" / "cluster_labels.parquet")["balanced_cluster"]
    cache_path = DATA_DIR / "raw" / "asset_prices.parquet"

    # ── 1. Try yfinance (only if --refresh-assets or no cache exists) ──────────
    prices: pd.DataFrame | None = None
    if refresh_assets or not cache_path.exists():
        if not refresh_assets:
            print(f"No cached ETF prices found at {cache_path} — fetching from yfinance ...")
        try:
            from market_regime.ingestion.assets import fetch_all as fetch_prices
            prices = fetch_prices(cfg)
            if not prices.empty:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                prices.to_parquet(cache_path)
                print(f"Fetched {prices.shape} ETF prices via yfinance → cached to {cache_path}")
        except Exception as exc:
            log.warning("yfinance fetch failed: %s", exc)
            print(f"yfinance unavailable: {exc}")
    else:
        print(f"Using cached ETF prices (pass --refresh-assets to re-fetch from yfinance).")

    # ── 2. Fall back to cached parquet ─────────────────────────────────────────
    if (prices is None or prices.empty) and cache_path.exists():
        prices = pd.read_parquet(cache_path)
        print(f"Loaded cached ETF prices: {prices.shape}")

    # ── 3. Compute returns ─────────────────────────────────────────────────────
    returns: pd.DataFrame | None = None
    if prices is not None and not prices.empty:
        returns = compute_quarterly_returns(prices)
        print(f"ETF quarterly returns: {returns.shape}")
    else:
        print("No ETF price data — computing proxy returns from macro data …")
        macro_path = DATA_DIR / "raw" / "macro_raw.parquet"
        if macro_path.exists():
            macro_df = pd.read_parquet(macro_path)
            returns = compute_proxy_returns(macro_df)
            if returns.empty:
                print("Proxy returns also empty — skipping step 6.")
                return
            print(f"Proxy returns: {returns.shape}")
        else:
            print(f"macro_raw.parquet not found at {macro_path} — skipping step 6.")
            return

    common = returns.index.intersection(labels.index)
    profile = returns_by_regime(returns.loc[common], labels.loc[common])
    ranked = rank_assets_by_regime(profile)

    out = DATA_DIR / "regimes" / "asset_return_profile.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    profile.to_parquet(out)
    print(f"Wrote asset return profile → {out}")

    print("\nTop assets per regime (by median quarterly return):")
    print(ranked.to_string(index=False))


if __name__ == "__main__":
    main()
