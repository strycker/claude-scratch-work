"""
Feature engineering transforms applied to the merged quarterly DataFrame.

Pipeline order (matches legacy workflow exactly):
  1. Cross-asset ratios   — computed from raw columns before log transform
  2. Log transforms       — stabilise variance on exponential-looking series
  3. Feature selection    — drop to initial_features + market_code
  4. Gap filling          — Bernstein polynomial interpolation per column
  5. Smoothed derivatives — np.gradient on a real day-number time axis,
                            smoothed with a centered rolling mean (d1, d2, d3)
  6. Clustering selection — drop to clustering_features + market_code

All transforms append new columns; originals are retained until the
explicit selection steps.
"""

import logging

import numpy as np
import pandas as pd
import matplotlib.dates as mdates
from scipy.interpolate import BPoly

log = logging.getLogger(__name__)


# ── 1. Cross-asset ratios ──────────────────────────────────────────────────

def add_cross_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the ten derived cross-asset ratio columns used throughout the
    pipeline.  These are defined analytically (not configurable) because
    each ratio has a specific financial interpretation.
    """
    df = df.copy()
    df["div_yield2"]      = df["dividend"]  / df["sp500"]
    df["price_div"]       = df["sp500"]     / df["dividend"]
    df["price_gdp"]       = df["sp500"]     / df["gdp"]
    df["price_gdp2"]      = df["sp500"]     / df["fred_gdp"]
    df["price_gnp2"]      = df["sp500"]     / df["fred_gnp"]
    df["div_minus_baa"]   = df["div_yield"] - df["fred_baa"] / 100.0
    df["credit_spread"]   = (df["fred_baa"] - df["fred_aaa"]) / 100.0
    df["real_price2"]     = df["sp500"]     / df["cpi"]
    df["real_price3"]     = df["sp500"]     / df["fred_cpi"]
    df["real_price_gdp2"] = df["sp500_adj"] / df["gdp"]
    return df


# ── 2. Log transforms ──────────────────────────────────────────────────────

def apply_log_transforms(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Add log_{col} columns for each col in `columns` that exists in df."""
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            log.debug("Log transform skipped — column not found: %s", col)
            continue
        df[f"log_{col}"] = np.log(df[col].clip(lower=1e-9))
    return df


# ── 3. Feature selection ───────────────────────────────────────────────────

def select_features(df: pd.DataFrame, feature_list: list[str]) -> pd.DataFrame:
    """Keep only the columns in feature_list (plus market_code if present)."""
    keep = [c for c in feature_list if c in df.columns]
    missing = set(feature_list) - set(df.columns)
    if missing:
        log.warning("Feature selection: %d columns not found: %s", len(missing), sorted(missing))
    extra = ["market_code"] if "market_code" in df.columns else []
    return df[keep + extra].copy()


# ── 4. Bernstein polynomial gap filling ───────────────────────────────────

def _dates_to_daynum(index) -> np.ndarray:
    """Convert a DatetimeIndex or string index to matplotlib day-numbers."""
    return mdates.date2num(index.values.astype(str))


def _compute_derivatives(
    series: pd.Series, window: int
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Smoothed 1st, 2nd, 3rd derivatives using centered rolling mean + np.gradient.

    The time axis is in actual matplotlib day-numbers so magnitudes are
    physically meaningful (units = value/day, value/day², value/day³).
    """
    smoothed = series.rolling(window=window, min_periods=1, center=True).mean()
    x = _dates_to_daynum(series.index) - _dates_to_daynum(series.index).min()

    d1 = pd.Series(np.gradient(smoothed, x), index=series.index).rolling(
        window=window, min_periods=1, center=True
    ).mean()
    d2 = pd.Series(np.gradient(d1, x), index=series.index).rolling(
        window=window, min_periods=1, center=True
    ).mean()
    d3 = pd.Series(np.gradient(d2, x), index=series.index).rolling(
        window=window, min_periods=1, center=True
    ).mean()
    return d1, d2, d3


def _fill_column(df: pd.DataFrame, col: str, window: int) -> pd.Series:
    """
    Fill interior NaN gaps in one column using BPoly.from_derivatives().

    For each contiguous NaN block bounded by valid data on both sides, a
    Bernstein polynomial is fitted through the boundary values and their
    smoothed derivatives.  Leading/trailing gaps use a Taylor expansion.

    market_code is optional: when present, valid rows are those where both
    the feature column and market_code are non-NaN.  When absent, valid rows
    are those where the feature column itself is non-NaN.
    """
    if "market_code" in df.columns:
        valid = df[[col, "market_code"]].dropna()
    else:
        valid = df[[col]].dropna()
    if valid.empty:
        return df[col]

    d1, d2, d3 = _compute_derivatives(valid[col], window=window)

    t = _dates_to_daynum(df.index) - _dates_to_daynum(df.index).min()
    t = t.astype("int64")

    y = df[col].copy()
    d1_full = d1.reindex(df.index)
    d2_full = d2.reindex(df.index)
    d3_full = d3.reindex(df.index)

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
                + d2_full.iloc[right] * dt ** 2 / 2
                + d3_full.iloc[right] * dt ** 3 / 6
            )
        else:
            # Trailing gap — forward Taylor expansion from left boundary
            dt = t[left + 1:] - t[left]
            y.iloc[left + 1:] = (
                y.iloc[left]
                + d1_full.iloc[left] * dt
                + d2_full.iloc[left] * dt ** 2 / 2
                + d3_full.iloc[left] * dt ** 3 / 6
            )
    return y


def apply_gap_fill(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Apply Bernstein gap filling to every feature column that has NaNs."""
    df = df.copy()
    feature_cols = [c for c in df.columns if c != "market_code"]
    for col in feature_cols:
        if df[col].isna().any():
            df[col] = _fill_column(df, col, window=window)
            log.debug("Gap-filled: %s", col)
    return df


# ── 5. Smoothed derivatives ────────────────────────────────────────────────

def apply_derivatives(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    For every feature column (excluding market_code), compute d1, d2, d3
    and append as {col}_d1, {col}_d2, {col}_d3.
    """
    df = df.copy()
    feature_cols = [c for c in df.columns if c != "market_code"]
    for col in feature_cols:
        if "market_code" in df.columns:
            valid = df[[col, "market_code"]].dropna()
        else:
            valid = df[[col]].dropna()
        if valid.empty:
            continue
        d1, d2, d3 = _compute_derivatives(valid[col], window=window)
        df[f"{col}_d1"] = d1.reindex(df.index)
        df[f"{col}_d2"] = d2.reindex(df.index)
        df[f"{col}_d3"] = d3.reindex(df.index)
        df = df.copy()  # defragment — avoids pandas PerformanceWarning
    return df


# ── Master wrapper ─────────────────────────────────────────────────────────

def engineer_all(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Run the full feature engineering pipeline in order.

    Steps:
      1. Cross-asset ratios
      2. Log transforms
      3. Narrow to initial_features
      4. Gap fill (Bernstein + Taylor)
      5. Smoothed derivatives (d1, d2, d3)
      6. Narrow to clustering_features

    Returns the ML-ready DataFrame (no NaNs, clustering_features + market_code).
    """
    feat_cfg = cfg["features"]
    window = feat_cfg.get("derivative_window", 5)

    log.info("Step 1/6 — cross-asset ratios")
    df = add_cross_ratios(df)

    log.info("Step 2/6 — log transforms (%d columns)", len(feat_cfg["log_columns"]))
    df = apply_log_transforms(df, feat_cfg["log_columns"])

    log.info("Step 3/6 — initial feature selection (%d features)", len(feat_cfg["initial_features"]))
    df = select_features(df, feat_cfg["initial_features"])

    log.info("Step 4/6 — Bernstein gap filling")
    df = apply_gap_fill(df, window=window)

    log.info("Step 5/6 — smoothed derivatives (window=%d)", window)
    df = apply_derivatives(df, window=window)

    log.info("Step 6/6 — clustering feature selection (%d features)", len(feat_cfg["clustering_features"]))
    df = select_features(df, feat_cfg["clustering_features"])

    nan_count = df.drop(columns=["market_code"], errors="ignore").isna().sum().sum()
    log.info(
        "Feature engineering complete: %d rows × %d features, %d NaNs remaining",
        len(df), len(df.columns), nan_count,
    )
    return df
