"""
Momentum and cross-asset ratio features.

Computes trailing returns, relative strength ratios, rolling cross-asset
correlations, and acceleration features from macro time series.

These features capture trend, mean-reversion, and inter-market dynamics
that complement the level-based features already in the pipeline.

Usage:
    from trading_crab_lib.momentum import add_momentum_features
    df = add_momentum_features(df, cfg)

Hooked into ``engineer_all()`` in ``transforms.py`` after cross-ratios.
"""

from __future__ import annotations

from typing import Any
import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ── Trailing momentum (percentage change over N quarters) ──────────────────

def compute_trailing_momentum(
    df: pd.DataFrame,
    columns: list[str],
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """
    Compute trailing N-quarter percentage returns for specified columns.

    For each column and each window, creates ``{col}_mom_{window}q``.
    E.g., ``sp500_mom_4q`` = (sp500[t] - sp500[t-4]) / sp500[t-4].

    Args:
        df: DataFrame with quarterly time-series columns.
        columns: Column names to compute momentum for.
        windows: Lookback windows in quarters. Default [2, 4, 8].

    Returns:
        DataFrame with new momentum columns appended.
    """
    if windows is None:
        windows = [2, 4, 8]

    result = df.copy()
    added = 0
    for col in columns:
        if col not in result.columns:
            continue
        for w in windows:
            name = f"{col}_mom_{w}q"
            result[name] = result[col].pct_change(periods=w)
            added += 1

    if added > 0:
        log.info("Added %d trailing momentum features", added)
    return result


# ── Relative strength ratios ───────────────────────────────────────────────

# Default pairs: (numerator, denominator, output_name)
DEFAULT_RELATIVE_STRENGTH_PAIRS: list[tuple[str, str, str]] = [
    ("sp500", "gold_spot", "sp500_in_gold"),
    ("sp500", "wti_crude", "sp500_in_oil"),
    ("gold_spot", "wti_crude", "gold_in_oil"),
]


def compute_relative_strength(
    df: pd.DataFrame,
    pairs: list[tuple[str, str, str]] | None = None,
) -> pd.DataFrame:
    """
    Compute cross-asset relative strength ratios.

    Each pair (num, denom, name) produces ``name = num / denom``.
    Only computes if both columns exist. Skips gracefully otherwise.

    Args:
        df: DataFrame with quarterly time-series columns.
        pairs: List of (numerator_col, denominator_col, output_name).
               Default: S&P-in-Gold, S&P-in-Oil, Gold-in-Oil.

    Returns:
        DataFrame with relative strength columns appended.
    """
    if pairs is None:
        pairs = DEFAULT_RELATIVE_STRENGTH_PAIRS

    result = df.copy()
    added = 0
    for num, denom, name in pairs:
        if num not in result.columns or denom not in result.columns:
            log.debug("Skipping relative strength %s: missing %s or %s", name, num, denom)
            continue
        denom_safe = result[denom].replace(0, np.nan)
        result[name] = result[num] / denom_safe
        added += 1

    if added > 0:
        log.info("Added %d relative strength ratios", added)
    return result


# ── Rolling cross-asset correlation ────────────────────────────────────────

# Default correlation pairs: (col_a, col_b, window_quarters)
DEFAULT_CORRELATION_PAIRS: list[tuple[str, str, int]] = [
    ("sp500", "10yr_ustreas", 8),
    ("sp500", "gold_spot", 8),
    ("credit_spread", "fred_vix", 8),
]


def compute_rolling_cross_correlation(
    df: pd.DataFrame,
    pairs: list[tuple[str, str, int]] | None = None,
) -> pd.DataFrame:
    """
    Compute rolling Pearson correlation between signal pairs.

    For each (col_a, col_b, window), produces ``corr_{a}_{b}_{window}q``.
    Uses pct_change of each series (returns) to avoid spurious trend correlation.

    Args:
        df: DataFrame with quarterly time-series columns.
        pairs: List of (col_a, col_b, window_quarters).

    Returns:
        DataFrame with rolling correlation columns appended.
    """
    if pairs is None:
        pairs = DEFAULT_CORRELATION_PAIRS

    result = df.copy()
    added = 0
    for col_a, col_b, window in pairs:
        if col_a not in result.columns or col_b not in result.columns:
            log.debug("Skipping correlation %s/%s: column missing", col_a, col_b)
            continue
        ret_a = result[col_a].pct_change()
        ret_b = result[col_b].pct_change()
        name = f"corr_{col_a}_{col_b}_{window}q"
        result[name] = ret_a.rolling(window=window, min_periods=max(4, window // 2)).corr(ret_b)
        added += 1

    if added > 0:
        log.info("Added %d rolling cross-correlation features", added)
    return result


# ── Inflation acceleration (2nd derivative of CPI) ─────────────────────────

def compute_inflation_acceleration(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute inflation acceleration: the 2nd derivative of CPI.

    This captures whether inflation is accelerating (positive) or decelerating
    (negative), which is more predictive of regime changes than the CPI level.

    Creates ``cpi_acceleration`` = diff(diff(CPI)) / CPI.
    """
    result = df.copy()
    for col in ["cpi", "fred_cpi"]:
        if col in result.columns:
            d1 = result[col].diff()
            d2 = d1.diff()
            # Normalize by level to get percentage acceleration
            result["cpi_acceleration"] = d2 / result[col].replace(0, np.nan)
            log.info("Added cpi_acceleration from %s", col)
            break
    return result


# ── Master wrapper ─────────────────────────────────────────────────────────

def add_momentum_features(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """
    Add all momentum and cross-asset ratio features to the DataFrame.

    Called from ``engineer_all()`` in ``transforms.py`` after cross-ratios
    and yield curve features, before log transforms.

    Config keys (optional, all under ``features.momentum``):
        momentum_columns: list of columns to compute momentum for
        momentum_windows: list of lookback windows [2, 4, 8]
        relative_strength_pairs: list of [num, denom, name] triples
        correlation_pairs: list of [col_a, col_b, window] triples

    Args:
        df: DataFrame with raw + cross-ratio columns.
        cfg: Pipeline config dict.

    Returns:
        DataFrame with momentum features appended.
    """
    feat_cfg = cfg.get("features", {})
    mom_cfg = feat_cfg.get("momentum", {})

    # Momentum columns: default to key price/macro series present in df
    default_mom_cols = [
        c for c in ["sp500", "gold_spot", "wti_crude", "10yr_ustreas",
                     "sp500_adj", "credit_spread", "fred_vix"]
        if c in df.columns
    ]
    mom_cols = mom_cfg.get("momentum_columns", default_mom_cols)
    mom_windows = mom_cfg.get("momentum_windows", [2, 4, 8])

    df = compute_trailing_momentum(df, mom_cols, mom_windows)

    # Relative strength
    rs_pairs_raw = mom_cfg.get("relative_strength_pairs")
    rs_pairs = None
    if rs_pairs_raw:
        rs_pairs = [(p[0], p[1], p[2]) for p in rs_pairs_raw]
    df = compute_relative_strength(df, rs_pairs)

    # Rolling correlations
    corr_pairs_raw = mom_cfg.get("correlation_pairs")
    corr_pairs = None
    if corr_pairs_raw:
        corr_pairs = [(p[0], p[1], int(p[2])) for p in corr_pairs_raw]
    df = compute_rolling_cross_correlation(df, corr_pairs)

    # Inflation acceleration
    df = compute_inflation_acceleration(df)

    return df
