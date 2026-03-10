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

compute_proxy_returns() provides a macro-data fallback when ETF price data is
unavailable (e.g. network/SSL failure in step 6).  It derives asset-class proxy
returns from columns already present in the raw macro DataFrame (sp500, sp500_adj,
10yr_ustreas, gdp_growth, us_infl, credit_spread).  Coverage goes back to ~1950,
so every historical regime is represented even without ETF data.
"""

from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger(__name__)


# Macro columns available in data/raw/macro_raw.parquet that serve as
# asset-class proxies when yfinance ETF data is unavailable.
# Each entry: (display_name, column, kind)
#   kind "price"  → compute quarterly pct_change
#   kind "rate"   → use level value directly (already a rate/spread/growth figure)
_PROXY_COLUMNS: list[tuple[str, str, str]] = [
    ("S&P 500",        "sp500",         "price"),
    ("S&P 500 Real",   "sp500_adj",     "price"),
    ("10Y Treasury",   "10yr_ustreas",  "rate"),
    ("GDP Growth",     "gdp_growth",    "rate"),
    ("Inflation",      "us_infl",       "rate"),
    ("Credit Spread",  "credit_spread", "rate"),
]


def compute_proxy_returns(macro_df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive asset-class proxy returns from macro columns in the raw data.

    Used as a fallback when yfinance ETF price data is unavailable.
    Price-like columns (sp500, sp500_adj) → quarterly pct_change.
    Rate/spread/growth columns           → quarterly level value.

    Args:
        macro_df: DataFrame from data/raw/macro_raw.parquet.  Must contain
                  at least one of the columns in _PROXY_COLUMNS.

    Returns:
        DataFrame indexed by quarter-end date, one column per proxy asset.
        Drops the first row (NaN from pct_change on price columns).
    """
    result = pd.DataFrame(index=macro_df.index)
    found: list[str] = []

    for display_name, col, kind in _PROXY_COLUMNS:
        if col not in macro_df.columns:
            continue
        series = pd.to_numeric(macro_df[col], errors="coerce")
        if kind == "price":
            result[display_name] = series.pct_change()
        else:
            result[display_name] = series
        found.append(display_name)

    if not found:
        log.warning("compute_proxy_returns: none of the expected macro columns found")
        return pd.DataFrame()

    # Drop rows that are entirely NaN (typically the first row after pct_change)
    result = result.dropna(how="all").iloc[1:]
    log.info(
        "Proxy returns computed: %d quarters × %d assets (%s)",
        len(result), len(found), ", ".join(found),
    )
    return result


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
    return returns_full_stats(returns, cluster_labels)["median_return"]


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
