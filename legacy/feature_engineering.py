"""
Feature Engineering: log transforms, smoothed derivatives, gap interpolation.

Pure functions that transform a quarterly DataFrame in well-defined stages:
1. `apply_log_transforms()`  — stabilize variance on exponential columns
2. `interpolate_gaps()`      — fill NaN gaps with Bernstein polynomial fitting
3. `compute_all_derivatives()` — add d1/d2/d3 columns for every feature
"""

import logging
from datetime import datetime

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from scipy.interpolate import BPoly

from trading_crab.config import (
    CLUSTERING_FEATURES,
    INITIAL_FEATURES,
    LOG_COLUMNS,
    SMOOTHING_WINDOW,
)

log = logging.getLogger(__name__)


# ===================================================================
# Log transforms
# ===================================================================

def apply_log_transforms(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add log-transformed columns and keep only INITIAL_FEATURES + market_code.

    Exponential columns (S&P500, GDP, CPI, etc.) are log-transformed to
    stabilize their variance and make derivatives more meaningful.
    """
    log.info("Applying log transforms to %d columns...", len(LOG_COLUMNS))
    result = df.copy()
    for col in LOG_COLUMNS:
        result[f"log_{col}"] = np.log(result[col])

    result = result[INITIAL_FEATURES + ["market_code"]].copy()
    log.info("  Result: %d rows × %d cols", *result.shape)
    return result


# ===================================================================
# Smoothed derivatives
# ===================================================================

def _parse_dates(date_strings) -> list:
    """Convert string date index values to datetime.date objects."""
    return [datetime.strptime(str(d), "%Y-%m-%d").date() for d in date_strings]


def compute_smoothed_derivatives(
    series: pd.Series, dates: list, window: int = SMOOTHING_WINDOW
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Compute smoothed 1st, 2nd, and 3rd derivatives of a time series.

    Uses centered rolling means to suppress noise, and np.gradient for
    numeric differentiation against a day-number x-axis.
    """
    smoothed = series.rolling(window=window, min_periods=1, center=True).mean()
    x = mdates.date2num(dates) - mdates.date2num(dates).min()

    d1 = pd.Series(np.gradient(smoothed, x), index=dates).rolling(
        window=window, min_periods=1, center=True
    ).mean()
    d2 = pd.Series(np.gradient(d1, x), index=dates).rolling(
        window=window, min_periods=1, center=True
    ).mean()
    d3 = pd.Series(np.gradient(d2, x), index=dates).rolling(
        window=window, min_periods=1, center=True
    ).mean()

    return d1, d2, d3


# ===================================================================
# Bernstein polynomial gap interpolation
# ===================================================================

def interpolate_column(df: pd.DataFrame, col_name: str) -> pd.Series:
    """
    Fill NaN gaps in a single column using Bernstein polynomial interpolation.

    Interior gaps: BPoly through boundary values and their 1st–3rd derivatives.
    Leading gaps:  backward Taylor expansion from the first valid point.
    Trailing gaps: forward Taylor expansion from the last valid point.
    """
    all_dates_str = df.index.values
    all_dates = _parse_dates(all_dates_str)

    valid = df[[col_name, "market_code"]].dropna()
    valid_dates = _parse_dates(valid.index.values)

    d1, d2, d3 = compute_smoothed_derivatives(valid[col_name], valid_dates)

    # Numeric time axis for polynomial evaluation
    t = (mdates.date2num(all_dates_str) - mdates.date2num(all_dates_str).min()
         ).astype("int64")

    y = df[col_name].reindex(all_dates_str).copy()
    d1_full = d1.reindex(all_dates).copy()
    d2_full = d2.reindex(all_dates).copy()
    d3_full = d3.reindex(all_dates).copy()

    # Locate contiguous NaN blocks as (start, end) index pairs
    mask = y.isna().values
    edges = np.flatnonzero(np.diff(np.r_[0, mask, 0])).reshape(-1, 2)

    for start, end in edges:
        left, right = start - 1, end

        if left >= 0 and right < len(y):
            # Interior gap — Bernstein polynomial through both boundaries
            poly = BPoly.from_derivatives(
                [t[left], t[right]],
                [
                    [y.iloc[left],  d1_full.iloc[left],  d2_full.iloc[left],  d3_full.iloc[left]],
                    [y.iloc[right], d1_full.iloc[right], d2_full.iloc[right], d3_full.iloc[right]],
                ],
            )
            y.iloc[start:right] = poly(t[start:right])

        elif right < len(y):
            # Leading gap — backward Taylor expansion from right boundary
            dt = t[start:right] - t[right]
            y.iloc[start:right] = (
                y.iloc[right]
                + d1_full.iloc[right] * dt
                + d2_full.iloc[right] * dt**2 / 2
                + d3_full.iloc[right] * dt**3 / 6
            )
        else:
            # Trailing gap — forward Taylor expansion from left boundary
            dt = t[left + 1:] - t[left]
            y.iloc[left + 1:] = (
                y.iloc[left]
                + d1_full.iloc[left] * dt
                + d2_full.iloc[left] * dt**2 / 2
                + d3_full.iloc[left] * dt**3 / 6
            )

    return y


def interpolate_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """Fill all NaN gaps in feature columns using Bernstein interpolation."""
    log.info("Interpolating NaN gaps...")
    result = df.copy()
    feature_cols = [c for c in result.columns if c != "market_code"]

    filled_count = 0
    for col in feature_cols:
        if result[col].isna().any():
            n_nans = int(result[col].isna().sum())
            result[col] = interpolate_column(result, col)
            filled_count += n_nans
            log.debug("  %s: filled %d NaNs", col, n_nans)

    remaining = int(result[feature_cols].isna().sum().sum())
    log.info("  Filled %d total NaN values (%d remaining)", filled_count, remaining)
    return result


# ===================================================================
# Add derivative columns
# ===================================================================

def compute_all_derivatives(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute smoothed d1/d2/d3 for every feature column and append them.

    Then select only CLUSTERING_FEATURES + market_code for the final output.
    """
    log.info("Computing smoothed derivatives for %d features...",
             len([c for c in df.columns if c != "market_code"]))
    result = df.copy()
    feature_cols = [c for c in result.columns if c != "market_code"]

    for col in feature_cols:
        valid = result[[col, "market_code"]].dropna()
        dates = _parse_dates(valid.index.values)
        d1, d2, d3 = compute_smoothed_derivatives(valid[col], dates)

        result[f"{col}_d1"] = d1.values
        result[f"{col}_d2"] = d2.values
        result[f"{col}_d3"] = d3.values

        # Defragment periodically (pandas performance workaround)
        result = result.copy()

    result = result[CLUSTERING_FEATURES + ["market_code"]].copy()
    log.info("  Result: %d rows × %d cols", *result.shape)
    return result
