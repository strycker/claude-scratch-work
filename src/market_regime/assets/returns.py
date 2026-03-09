"""
Asset returns by regime.

Given a DataFrame of asset price history (one column per ETF/asset) and
the quarterly regime labels, compute per-regime return statistics.

Two public functions:

  returns_by_regime()    → pivoted DataFrame: index=regime, columns=tickers,
                           values=median quarterly return.
                           This is the format expected by all plotting helpers
                           and rank_assets_by_regime().

  returns_full_stats()   → pivoted DataFrames for median_return, hit_rate, and
                           n_quarters — returned as a dict keyed by stat name.
                           Useful for deeper analysis or custom reporting.

  rank_assets_by_regime() → flat DataFrame with columns [regime, asset,
                             median_quarterly_return, rank] suitable for the
                             dashboard asset_signals() function.

This module is deliberately data-source agnostic — prices can come from
yfinance, macrotrends, or a parquet file.  The caller provides a prices DataFrame.
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
    Compute median quarterly return for each (regime, asset) pair.

    Returns:
        DataFrame with index=regime (int), columns=tickers (str),
        values=median quarterly return (float).

        Shape: (n_regimes × n_tickers)

        This pivoted format is expected by:
          - plotting.plot_asset_returns_by_regime()
          - plotting.plot_asset_heatmap()
          - rank_assets_by_regime()
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
                "asset":  ticker,
                "median_return": col.median(),
            })

    if not records:
        return pd.DataFrame()

    flat = pd.DataFrame(records)
    pivot = flat.pivot(index="regime", columns="asset", values="median_return")
    pivot.index.name = "regime"
    pivot.columns.name = None  # drop the "asset" label from columns axis

    log.info(
        "Asset return profile: %d regimes × %d tickers",
        len(pivot), len(pivot.columns),
    )
    return pivot


def returns_full_stats(
    returns: pd.DataFrame,
    cluster_labels: pd.Series,
) -> dict[str, pd.DataFrame]:
    """
    Compute median return, hit rate, and n_quarters for each (regime, asset) pair.

    Returns:
        dict with keys "median_return", "hit_rate", "n_quarters", each mapping
        to a pivoted DataFrame: index=regime, columns=tickers.

    Use this when you need richer statistics than median_return alone (e.g.
    for detailed reporting or future plotting extensions).
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
                "regime":        regime,
                "asset":         ticker,
                "median_return": col.median(),
                "hit_rate":      (col > 0).mean(),
                "n_quarters":    len(col),
            })

    if not records:
        return {"median_return": pd.DataFrame(), "hit_rate": pd.DataFrame(), "n_quarters": pd.DataFrame()}

    flat = pd.DataFrame(records)
    result = {}
    for stat in ("median_return", "hit_rate", "n_quarters"):
        pivot = flat.pivot(index="regime", columns="asset", values=stat)
        pivot.index.name = "regime"
        pivot.columns.name = None
        result[stat] = pivot

    return result


def rank_assets_by_regime(profile: pd.DataFrame) -> pd.DataFrame:
    """
    Within each regime, rank assets by median_return (descending).

    Args:
        profile — pivoted DataFrame from returns_by_regime():
                  index=regime, columns=tickers, values=median return

    Returns:
        Flat DataFrame with columns: regime, asset, median_quarterly_return, rank.
        Suitable for passing to reporting.dashboard.asset_signals().
    """
    records = []
    for regime, row in profile.iterrows():
        sorted_assets = row.dropna().sort_values(ascending=False)
        for rank, (asset, ret) in enumerate(sorted_assets.items(), start=1):
            records.append({
                "regime":                  regime,
                "asset":                   asset,
                "median_quarterly_return": ret,
                "rank":                    rank,
            })
    return pd.DataFrame(records)
