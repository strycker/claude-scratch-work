"""
Feature engineering transforms applied to the raw quarterly DataFrame.

Three categories:
  1. Log transforms    — for exponential-looking series (GDP, CPI, M2, …)
  2. Smoothed derivatives — rolling-windowed 1st/2nd/3rd differences of log-series
  3. Cross-asset ratios   — e.g. yield curve slope (10y - 2y)
  4. Bernstein gap fill   — polynomial interpolation for sparse columns

All transforms are additive (new columns appended), the originals are kept.
"""

import logging

import numpy as np
import pandas as pd
from scipy.special import comb  # for Bernstein basis

log = logging.getLogger(__name__)


# ── 1. Log transforms ──────────────────────────────────────────────────────

def apply_log_transforms(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Add log_{col} columns for each col in `columns` that exists in df."""
    for col in columns:
        if col not in df.columns:
            log.debug("Log transform skipped — column not found: %s", col)
            continue
        df[f"log_{col}"] = np.log(df[col].clip(lower=1e-9))
    return df


# ── 2. Smoothed derivatives ────────────────────────────────────────────────

def _rolling_diff(series: pd.Series, window: int) -> pd.Series:
    """Smoothed first difference: mean(x[t-w+1:t]) - mean(x[t-2w+1:t-w])."""
    smoothed = series.rolling(window, min_periods=1).mean()
    return smoothed.diff(window)


def apply_derivatives(
    df: pd.DataFrame,
    orders: list[int],
    window: int,
) -> pd.DataFrame:
    """
    For every log_{col} column, compute rolling-smoothed derivatives up to
    the requested orders and append as log_{col}_d1, _d2, _d3, etc.
    """
    log_cols = [c for c in df.columns if c.startswith("log_")]
    for col in log_cols:
        series = df[col].copy()
        for order in orders:
            for _ in range(order):
                series = _rolling_diff(series, window)
            df[f"{col}_d{order}"] = series
            series = df[col].copy()  # restart from original for each order
    return df


# ── 3. Cross-asset ratios ──────────────────────────────────────────────────

def apply_ratios(df: pd.DataFrame, pairs: list[list[str]]) -> pd.DataFrame:
    """
    For each [a, b] pair, add column "{a}_minus_{b}" (difference, not ratio,
    since both are already in log-space or are rates).
    If both are raw rates (e.g. yields), difference is the natural operation.
    """
    for pair in pairs:
        a, b = pair
        if a not in df.columns or b not in df.columns:
            log.debug("Ratio skipped — missing column(s): %s, %s", a, b)
            continue
        col_name = f"{a}_minus_{b}"
        df[col_name] = df[a] - df[b]
        log.debug("Added ratio column %s", col_name)
    return df


# ── 4. Bernstein polynomial gap fill ──────────────────────────────────────

def _bernstein_basis(t: np.ndarray, n: int, i: int) -> np.ndarray:
    return comb(n, i) * (t ** i) * ((1 - t) ** (n - i))


def bernstein_fill(series: pd.Series, degree: int = 5) -> pd.Series:
    """
    Fill NaN gaps in a Series using a Bernstein polynomial fitted to the
    known (non-NaN) points.  Only fills interior gaps — leading/trailing
    NaNs are left as-is.
    """
    known_mask = series.notna()
    if known_mask.sum() < degree + 1:
        return series  # not enough data to fit

    x_all = np.linspace(0, 1, len(series))
    x_known = x_all[known_mask]
    y_known = series.values[known_mask]

    # Build Vandermonde-style matrix for Bernstein basis
    B = np.column_stack([_bernstein_basis(x_known, degree, i) for i in range(degree + 1)])
    coeffs, *_ = np.linalg.lstsq(B, y_known, rcond=None)

    # Evaluate at all points
    B_all = np.column_stack([_bernstein_basis(x_all, degree, i) for i in range(degree + 1)])
    y_filled = B_all @ coeffs

    result = series.copy()
    result[~known_mask] = y_filled[~known_mask]
    return result


def apply_gap_fill(df: pd.DataFrame, degree: int = 5) -> pd.DataFrame:
    """Apply Bernstein gap filling to all columns that have interior NaNs."""
    for col in df.columns:
        if df[col].isna().any():
            df[col] = bernstein_fill(df[col], degree=degree)
    return df


# ── Convenience wrapper ────────────────────────────────────────────────────

def engineer_all(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Apply all feature engineering steps in order."""
    feat_cfg = cfg["features"]

    df = apply_log_transforms(df, feat_cfg.get("log_transform", []))
    df = apply_derivatives(
        df,
        orders=feat_cfg.get("derivative_orders", [1, 2, 3]),
        window=feat_cfg.get("derivative_window", 4),
    )
    df = apply_ratios(df, feat_cfg.get("cross_ratios", []))
    df = apply_gap_fill(df, degree=feat_cfg.get("bernstein_degree", 5))

    log.info(
        "Feature engineering complete: %d rows × %d features",
        len(df), len(df.columns),
    )
    return df
