"""
Asset returns by regime.

Given a DataFrame of asset price history (one column per ETF/asset) and
the quarterly regime labels, compute per-regime return statistics.

Two public functions:

  returns_by_regime()    → pivoted DataFrame: index=regime, columns=tickers,
                           values=median quarterly return.
                           This is the format expected by all plotting helpers
                           and rank_assets_by_regime().

  returns_full_stats()   → pivoted DataFrames for median_return, q25, q75, hit_rate,
                           and n_quarters — returned as a dict keyed by stat name.
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
    Compute median return, q25, q75, hit rate, and n_quarters for each (regime, asset) pair.

    Returns:
        dict with keys "median_return", "q25", "q75", "hit_rate", "n_quarters", each mapping
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
            q = col.quantile([0.25, 0.75])
            records.append({
                "regime":        regime,
                "asset":         ticker,
                "median_return": col.median(),
                "q25":           q.iloc[0],
                "q75":           q.iloc[1],
                "hit_rate":      (col > 0).mean(),
                "n_quarters":    len(col),
            })

    if not records:
        return {
            "median_return": pd.DataFrame(), "q25": pd.DataFrame(),
            "q75": pd.DataFrame(), "hit_rate": pd.DataFrame(),
            "n_quarters": pd.DataFrame(),
        }

    flat = pd.DataFrame(records)
    result = {}
    for stat in ("median_return", "q25", "q75", "hit_rate", "n_quarters"):
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


def compute_template_returns(
    returns: pd.DataFrame,
    templates: list[dict],
) -> pd.DataFrame:
    """
    Compute quarterly returns for portfolio templates (weighted sum of ETF returns).

    Args:
        returns: DataFrame index=quarter, columns=ETF tickers, values=quarterly return.
        templates: List of {"name": str, "weights": {ticker: weight, ...}}.
                   Weights are normalized per template; only tickers present in returns are used.

    Returns:
        DataFrame with same index as returns, columns=template names.
    """
    out = pd.DataFrame(index=returns.index)
    for tpl in templates:
        name = tpl.get("name", "unknown")
        weights = tpl.get("weights") or {}
        if not weights:
            continue
        common = [t for t in weights if t in returns.columns]
        if not common:
            log.warning("Template %s: no weight keys found in returns columns", name)
            continue
        w = {t: weights[t] for t in common}
        total = sum(w.values())
        if total <= 0:
            continue
        w = {t: w[t] / total for t in w}
        out[name] = sum(returns[t] * w[t] for t in w)
    return out


# Default behavior thresholds (overridden by config dashboard.behavior_thresholds).
_DEFAULT_BEHAVIOR = {"green_median": 0.02, "red_median": 0.0}


def _absolute_signal(median: float, green_median: float, red_median: float) -> str:
    if median >= green_median:
        return "GREEN"
    # Treat exact red_median (e.g. 0%) as neutral; strictly below is RED.
    if median < red_median:
        return "RED"
    return "NEUTRAL"


def _score_absolute(median: float) -> float:
    """Map median quarterly return to 0–100. 0% ≈ 50, +2% ≈ 70, +4% ≈ 85, -2% ≈ 30."""
    # Linear segments: -3% -> 15, 0% -> 50, 2% -> 70, 4% -> 85, 6%+ -> 100
    if median >= 0.06:
        return 100.0
    if median >= 0.04:
        return 50.0 + (median - 0.04) * (35.0 / 0.02)  # 85–100
    if median >= 0.02:
        return 50.0 + (median - 0.02) * (20.0 / 0.02)  # 70–85
    if median >= 0.0:
        return 50.0 + median * (20.0 / 0.02)            # 50–70
    if median >= -0.02:
        return 50.0 + median * (20.0 / 0.02)            # 30–50
    if median >= -0.03:
        return 30.0 + (median + 0.02) * (15.0 / 0.01)  # 15–30
    return max(0.0, 15.0 + (median + 0.03) * (15.0 / 0.03))


def behavior_tables(
    returns: pd.DataFrame,
    cluster_labels: pd.Series,
    thresholds: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Per-regime behavior: median, IQR, stoplight (absolute + tertile shading), and scores.

    Stoplight rules:
      - Absolute: green if median >= green_median (default 2%), red if median < red_median (0%), else neutral.
      - Tertile: within each regime, rank by median; top 1/3 → green_strong, middle → light_blue, bottom 1/3 → red_strong.

    Scores:
      - score_relative: 0–100, rank within regime (best median = 100).
      - score_absolute: 0–100 from absolute median return level.

    Returns:
        Flat DataFrame with columns: regime, asset, median_return, q25, q75, hit_rate,
        n_quarters, signal_absolute, tertile, signal_display, score_relative, score_absolute, rank.
    """
    th = {**_DEFAULT_BEHAVIOR, **(thresholds or {})}
    green_med = th["green_median"]
    red_med = th["red_median"]

    full = returns_full_stats(returns, cluster_labels)
    median_df = full["median_return"]
    q25_df = full["q25"]
    q75_df = full["q75"]
    hit_df = full["hit_rate"]
    n_df = full["n_quarters"]

    records = []
    for regime in median_df.index:
        med = median_df.loc[regime].dropna()
        if med.empty:
            continue
        # Tertile ranks (1 = top, 2 = mid, 3 = bottom)
        ranked = med.rank(ascending=False, method="first").astype(int)
        n = len(med)
        tertile = (ranked - 1) * 3 // n + 1  # 1-based tertile

        for asset in med.index:
            m = med[asset]
            abs_sig = _absolute_signal(m, green_med, red_med)
            tert = tertile[asset]
            # Combined display signal
            if abs_sig == "GREEN" and tert == 1:
                display = "green_strong"
            elif abs_sig == "GREEN":
                display = "green"
            elif abs_sig == "NEUTRAL" and tert == 2:
                display = "light_blue"
            elif abs_sig == "NEUTRAL":
                display = "neutral"
            elif abs_sig == "RED" and tert == 3:
                display = "red_strong"
            else:
                display = "red"

            # Relative score 0–100 (best in regime = 100)
            rrank = ranked[asset]
            score_rel = 100.0 * (1.0 - (rrank - 1) / max(1, n - 1)) if n > 1 else 50.0

            records.append({
                "regime":          regime,
                "asset":           asset,
                "median_return":   m,
                "q25":             q25_df.loc[regime, asset] if asset in q25_df.columns else None,
                "q75":             q75_df.loc[regime, asset] if asset in q75_df.columns else None,
                "hit_rate":        hit_df.loc[regime, asset] if asset in hit_df.columns else None,
                "n_quarters":      int(n_df.loc[regime, asset]) if asset in n_df.columns else None,
                "signal_absolute": abs_sig,
                "tertile":         int(tert),
                "signal_display":  display,
                "score_relative":  round(score_rel, 1),
                "score_absolute":  round(_score_absolute(m), 1),
                "rank":            int(ranked[asset]),
            })

    return pd.DataFrame(records)
