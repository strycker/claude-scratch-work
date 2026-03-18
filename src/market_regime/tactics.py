"""
Tactical asset classification.

Provides short-term (tactical) signals for asset allocation based on
volatility, trend, and correlation metrics.

  compute_tactics_metrics() — volatility, trend slope, and benchmark
                              correlation per asset
  classify_tactics()        — classify assets into buy_hold / swing /
                              stand_aside based on vol + trend thresholds

This complements the regime-based (strategic) allocation by adding
shorter-horizon signals.  The regime model says "we are in environment X,
historically these assets do well."  The tactics module says "right now,
this asset's price action is trending / mean-reverting / choppy."
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def compute_tactics_metrics(
    returns: pd.DataFrame,
    benchmark_returns: pd.Series | None = None,
    vol_window: int = 8,
    trend_window: int = 4,
) -> pd.DataFrame:
    """
    Compute tactical metrics for each asset.

    For each asset column in returns:
      - volatility:   annualized rolling std (quarterly → ×2 for sqrt(4))
      - trend_slope:  OLS slope of price over the trailing window (positive = uptrend)
      - correlation:  rolling correlation with benchmark (if provided)

    Args:
        returns:            DataFrame of quarterly returns (columns = tickers)
        benchmark_returns:  optional benchmark Series for correlation calc
        vol_window:         quarters for volatility calculation
        trend_window:       quarters for trend slope calculation

    Returns:
        DataFrame with columns: asset, volatility, trend_slope, correlation
    """
    records = []

    for ticker in returns.columns:
        series = returns[ticker].dropna()
        if len(series) < max(vol_window, trend_window):
            continue

        # Annualized volatility (quarterly data → multiply by sqrt(4) = 2)
        vol = float(series.tail(vol_window).std() * 2)

        # Trend slope: simple linear regression slope over trailing window
        tail = series.tail(trend_window).values
        x = np.arange(len(tail), dtype=float)
        if len(tail) >= 2:
            slope = float(np.polyfit(x, tail, 1)[0])
        else:
            slope = 0.0

        # Correlation with benchmark
        corr = np.nan
        if benchmark_returns is not None:
            common = series.index.intersection(benchmark_returns.index)
            if len(common) >= vol_window:
                corr = float(
                    series.loc[common].tail(vol_window).corr(
                        benchmark_returns.loc[common].tail(vol_window)
                    )
                )

        records.append({
            "asset": ticker,
            "volatility": round(vol, 4),
            "trend_slope": round(slope, 6),
            "correlation": round(corr, 4) if not np.isnan(corr) else np.nan,
        })

    return pd.DataFrame(records)


def classify_tactics(
    metrics: pd.DataFrame,
    vol_threshold: float = 0.20,
    trend_threshold: float = 0.0,
) -> pd.DataFrame:
    """
    Classify each asset into a tactical bucket based on vol + trend.

    Classification logic:
      - **buy_hold**: low volatility AND positive trend
        (steady uptrend — hold for regime duration)
      - **swing**: high volatility AND positive trend
        (trending but volatile — take profits actively)
      - **stand_aside**: negative trend (regardless of vol)
        (downtrend — avoid until trend reverses)

    Args:
        metrics:         DataFrame from compute_tactics_metrics()
        vol_threshold:   annualized vol above which asset is "high vol"
        trend_threshold: slope above which trend is considered positive

    Returns:
        Copy of metrics with added 'tactic' column.
    """
    result = metrics.copy()

    def _classify(row):
        if row["trend_slope"] <= trend_threshold:
            return "stand_aside"
        if row["volatility"] > vol_threshold:
            return "swing"
        return "buy_hold"

    result["tactic"] = result.apply(_classify, axis=1)
    return result
