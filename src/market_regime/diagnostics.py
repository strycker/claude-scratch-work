"""
Relative Rotation Graph (RRG) diagnostics and rolling statistics.

Provides tools for tactical asset rotation analysis:

  rolling_zscore()      — rolling z-score normalization for any time series
  percentile_rank()     — rolling percentile rank of a series within its history
  normalize_100()       — re-center a series around 100 (RRG convention)
  compute_rrg()         — classify assets into RRG quadrants:
                          LEADING / WEAKENING / LAGGING / IMPROVING

RRG analysis compares each asset's relative strength (RS) and relative momentum
(RM) against a benchmark.  Assets rotate clockwise through four quadrants as
they cycle from improving → leading → weakening → lagging.

References:
  - Julius de Kempenaer, "Relative Rotation Graphs" (2013)
  - RRG Research: https://www.relativerotationgraphs.com
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def rolling_zscore(
    series: pd.Series,
    window: int = 20,
) -> pd.Series:
    """
    Compute the rolling z-score of a time series.

    For each point, calculates (value - rolling_mean) / rolling_std.
    Returns NaN where rolling_std is zero (constant series).
    """
    rolling_mean = series.rolling(window=window, min_periods=1).mean()
    rolling_std = series.rolling(window=window, min_periods=1).std()
    result = (series - rolling_mean) / rolling_std
    # Replace inf/-inf from zero-std with NaN
    result = result.replace([np.inf, -np.inf], np.nan)
    return result


def percentile_rank(
    series: pd.Series,
    window: int = 20,
) -> pd.Series:
    """
    Compute the rolling percentile rank of each value within its trailing window.

    Returns values in [0, 1] where 1.0 means the current value is the maximum
    within the window.
    """
    def _rank_in_window(arr):
        if len(arr) < 2:
            return 0.5
        current = arr[-1]
        return float((arr[:-1] <= current).sum()) / (len(arr) - 1)

    return series.rolling(window=window, min_periods=1).apply(_rank_in_window, raw=True)


def normalize_100(
    series: pd.Series,
    center_window: int = 20,
) -> pd.Series:
    """
    Normalize a series to oscillate around 100.

    Uses the rolling mean as the center, then rescales:
        normalized = 100 + (value - rolling_mean) / rolling_mean * 100

    This is the standard RRG convention where 100 = benchmark-equivalent.
    """
    rolling_mean = series.rolling(window=center_window, min_periods=1).mean()
    # Avoid division by zero
    safe_mean = rolling_mean.replace(0, np.nan)
    normalized = 100 + (series - rolling_mean) / safe_mean * 100
    return normalized


def compute_rrg(
    asset_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    rs_window: int = 12,
    rm_window: int = 4,
) -> pd.DataFrame:
    """
    Classify assets into RRG quadrants based on relative strength and momentum.

    For each asset:
      RS = normalize_100(asset_return / benchmark_return, rs_window)
      RM = normalize_100(RS.diff(), rm_window)

    Quadrant classification (based on final value):
      - LEADING:    RS > 100 and RM > 100  (strong and accelerating)
      - WEAKENING:  RS > 100 and RM <= 100 (strong but decelerating)
      - LAGGING:    RS <= 100 and RM <= 100 (weak and decelerating)
      - IMPROVING:  RS <= 100 and RM > 100  (weak but accelerating)

    Args:
        asset_returns:     DataFrame of quarterly returns (columns = tickers)
        benchmark_returns: Series of benchmark returns (e.g., SPY)
        rs_window:         window for relative strength normalization
        rm_window:         window for relative momentum normalization

    Returns:
        DataFrame with columns: asset, rs, rm, quadrant
    """
    records = []

    for ticker in asset_returns.columns:
        asset = asset_returns[ticker].dropna()
        bench = benchmark_returns.reindex(asset.index).dropna()
        common = asset.index.intersection(bench.index)

        if len(common) < max(rs_window, rm_window) + 1:
            log.debug("Skipping %s: insufficient data (%d quarters)", ticker, len(common))
            continue

        asset_c = asset.loc[common]
        bench_c = bench.loc[common]

        # Relative strength: cumulative ratio normalized around 100
        # Use cumulative return ratio to get RS line
        cum_asset = (1 + asset_c).cumprod()
        cum_bench = (1 + bench_c).cumprod()
        rs_raw = cum_asset / cum_bench
        rs = normalize_100(rs_raw, center_window=rs_window)

        # Relative momentum: rate of change of RS, normalized around 100
        rs_change = rs.diff()
        rm = normalize_100(rs_change.dropna(), center_window=rm_window)

        # Use the most recent values for classification
        rs_last = rs.iloc[-1]
        rm_last = rm.iloc[-1] if len(rm) > 0 else np.nan

        if pd.isna(rs_last) or pd.isna(rm_last):
            continue

        if rs_last > 100 and rm_last > 100:
            quadrant = "LEADING"
        elif rs_last > 100 and rm_last <= 100:
            quadrant = "WEAKENING"
        elif rs_last <= 100 and rm_last <= 100:
            quadrant = "LAGGING"
        else:
            quadrant = "IMPROVING"

        records.append({
            "asset": ticker,
            "rs": round(rs_last, 2),
            "rm": round(rm_last, 2),
            "quadrant": quadrant,
        })

    return pd.DataFrame(records)


def rrg_for_benchmark(
    prices: pd.DataFrame,
    benchmark: str,
    lookback: int = 52,
) -> pd.DataFrame:
    """
    Compute RS-Ratio and RS-Momentum for each asset vs a benchmark (price-based).

    This is the price-level counterpart to :func:`compute_rrg` (which takes
    returns).  Pipeline step 08 uses this variant because it operates directly
    on ``asset_prices.parquet`` without pre-computing returns.

    Args:
        prices:    DataFrame of prices (columns = tickers, index = dates).
        benchmark: column name in *prices* to use as benchmark.
        lookback:  number of rows (periods) to use.

    Returns:
        DataFrame with columns: as_of, asset, benchmark, rs_ratio, rs_momentum,
        quadrant (LEADING / WEAKENING / LAGGING / IMPROVING).
    """
    if benchmark not in prices.columns:
        return pd.DataFrame()
    if len(prices) < lookback:
        lookback = len(prices)
    window_prices = prices.iloc[-lookback:]
    rs = window_prices.divide(window_prices[benchmark], axis=0)
    rs_smooth = rs.rolling(window=min(13, lookback), min_periods=1).mean()
    rs_ratio = rs_smooth.apply(normalize_100, axis=0)
    rs_mom_raw = rs_ratio.diff()
    rs_momentum = rs_mom_raw.apply(normalize_100, axis=0)

    as_of = window_prices.index[-1]
    records: list[dict] = []
    for col in prices.columns:
        if col == benchmark:
            continue
        rr = rs_ratio[col].dropna()
        mm = rs_momentum[col].dropna()
        if rr.empty or mm.empty:
            continue
        rr_last = rr.iloc[-1]
        mm_last = mm.iloc[-1]
        if rr_last >= 100 and mm_last >= 100:
            quadrant = "LEADING"
        elif rr_last >= 100 and mm_last < 100:
            quadrant = "WEAKENING"
        elif rr_last < 100 and mm_last < 100:
            quadrant = "LAGGING"
        else:
            quadrant = "IMPROVING"
        records.append(
            {
                "as_of": as_of,
                "asset": col,
                "benchmark": benchmark,
                "rs_ratio": rr_last,
                "rs_momentum": mm_last,
                "quadrant": quadrant,
            }
        )
    return pd.DataFrame.from_records(records)
