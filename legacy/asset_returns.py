"""
Asset Returns by Regime: compute per-regime performance for major asset proxies.

Since we don't have ETF data in the current pipeline, we derive proxy
returns from the quarterly features already available (S&P500, treasuries,
CPI as an inflation proxy, GDP growth).  When real ETF data is added
later, this module's interface stays the same.
"""

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Asset proxies derivable from existing quarterly data.
# Each entry: (display_name, numerator_col, denominator_col_or_None)
# If denominator is None, we compute simple quarterly returns.
ASSET_PROXIES = [
    ("S&P 500",          "sp500",      None),
    ("S&P 500 Real",     "sp500_adj",  None),
    ("10Y Treasury",     "10yr_ustreas", None),
    ("GDP Growth",       "gdp_growth", None),
    ("Inflation (CPI)",  "us_infl",    None),
    ("Credit Spread",    "credit_spread", None),
]


def compute_quarterly_returns(
    quarterly_df: pd.DataFrame,
    price_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Compute simple quarterly returns (pct_change) for price-like columns.

    For rate/spread columns (already in level form), returns the level directly.
    """
    if price_cols is None:
        price_cols = ["sp500", "sp500_adj"]

    available_price = [c for c in price_cols if c in quarterly_df.columns]
    rate_cols = [c for c in quarterly_df.columns
                 if c in ("gdp_growth", "real_gdp_growth", "us_infl",
                          "credit_spread", "10yr_ustreas")]

    returns = pd.DataFrame(index=quarterly_df.index)

    # Price columns → percentage returns
    for col in available_price:
        numeric = pd.to_numeric(quarterly_df[col], errors="coerce")
        returns[f"{col}_return"] = numeric.pct_change()

    # Rate columns → keep as levels (they're already rates/spreads)
    for col in rate_cols:
        if col in quarterly_df.columns:
            returns[col] = pd.to_numeric(quarterly_df[col], errors="coerce")

    returns = returns.iloc[1:]  # drop first row (NaN from pct_change)
    log.info("Computed returns: %d rows × %d cols", *returns.shape)
    return returns


def returns_by_regime(
    returns_df: pd.DataFrame,
    regime_labels: np.ndarray,
) -> pd.DataFrame:
    """
    Compute median and mean returns per regime for each asset/column.

    Args:
        returns_df: quarterly returns (output of compute_quarterly_returns)
        regime_labels: integer regime labels aligned with returns_df

    Returns:
        DataFrame with multi-level columns: (asset, 'mean'|'median'|'std'|'count')
    """
    log.info("Computing per-regime returns...")

    # Align lengths
    n = min(len(returns_df), len(regime_labels))
    aligned = returns_df.iloc[:n].copy()
    aligned["regime"] = regime_labels[:n]

    summary = aligned.groupby("regime").agg(["mean", "median", "std", "count"])

    log.info("  Summarized %d assets across %d regimes",
             len(returns_df.columns), summary.index.nunique())
    return summary


def best_assets_per_regime(
    regime_returns: pd.DataFrame,
    metric: str = "median",
) -> pd.DataFrame:
    """
    Rank assets within each regime by the given metric.

    Returns a DataFrame showing the top assets for each regime.
    """
    log.info("Ranking assets per regime by %s return...", metric)

    # Extract just the metric column for each asset
    metric_cols = [col for col in regime_returns.columns if col[1] == metric]
    rankings = regime_returns[metric_cols].copy()
    rankings.columns = [col[0] for col in metric_cols]

    # Rank within each regime (row)
    ranked = rankings.rank(axis=1, ascending=False)

    result = pd.DataFrame(index=rankings.index)
    for regime in rankings.index:
        row = rankings.loc[regime].sort_values(ascending=False)
        log.info("  Regime %d best: %s (%.4f)", regime, row.index[0], row.iloc[0])

    return rankings
