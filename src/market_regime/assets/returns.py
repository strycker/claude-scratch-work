"""
Asset returns by regime.

Given a DataFrame of asset price history (one column per ETF/asset) and
the quarterly regime labels, compute:
  - Median quarterly return per regime
  - Hit rate (% of quarters with positive return) per regime
  - Ranking of assets within each regime

This module is deliberately data-source agnostic — prices can come from
yfinance, macrotrends, or a CSV.  The caller provides a prices DataFrame.
"""

import logging

import pandas as pd

log = logging.getLogger(__name__)


def compute_quarterly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a prices DataFrame (index=dates, columns=tickers) to
    quarterly percentage returns.  Resamples to QE if not already quarterly.
    """
    quarterly_prices = prices.resample("QE").last()
    returns = quarterly_prices.pct_change().dropna(how="all")
    return returns


def returns_by_regime(
    returns: pd.DataFrame,
    cluster_labels: pd.Series,
) -> pd.DataFrame:
    """
    Compute median return and hit rate for each (regime, asset) pair.

    Returns:
        DataFrame with MultiIndex (regime, stat) × ticker columns.
        stat ∈ {"median_return", "hit_rate", "n_quarters"}
    """
    joined = returns.copy()
    joined["regime"] = cluster_labels

    records = []
    for regime, group in joined.groupby("regime"):
        asset_data = group.drop(columns=["regime"])
        for ticker in asset_data.columns:
            col = asset_data[ticker].dropna()
            if col.empty:
                continue
            records.append({
                "regime": regime,
                "asset": ticker,
                "median_return": col.median(),
                "hit_rate": (col > 0).mean(),
                "n_quarters": len(col),
            })

    df = pd.DataFrame(records).set_index(["regime", "asset"])
    log.info("Asset return profile: %d regime-asset pairs", len(df))
    return df


def rank_assets_by_regime(profile: pd.DataFrame) -> pd.DataFrame:
    """
    Within each regime, rank assets by median_return (descending).
    Returns a clean DataFrame suitable for display / reporting.
    """
    ranked = (
        profile["median_return"]
        .groupby(level="regime", group_keys=False)
        .apply(lambda s: s.sort_values(ascending=False))
        .reset_index()
    )
    ranked.columns = ["regime", "asset", "median_quarterly_return"]
    ranked["rank"] = ranked.groupby("regime")["median_quarterly_return"].rank(
        ascending=False, method="min"
    ).astype(int)
    return ranked
