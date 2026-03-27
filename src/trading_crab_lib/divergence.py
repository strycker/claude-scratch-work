"""
Cross-asset divergence features for regime change detection.

Detects breakdowns in historically correlated signal pairs and converts
them into supervised learning features.  When correlations that have held
for years suddenly break (e.g., S&P500 and bonds move together instead of
inversely), these divergences signal regime transitions.

Three layers of features:

1. **Rolling correlation baseline** — trailing Pearson correlation between
   returns (or derivatives) of each signal pair.
2. **Divergence magnitude** — how far the current short-window correlation
   departs from its longer-term baseline.  Measured as both raw difference
   and z-score relative to historical divergence volatility.
3. **Divergence triggers** — binary flags when divergence exceeds a
   threshold (default 2σ), plus direction (+1 / −1).

Usage:
    from trading_crab_lib.divergence import add_divergence_features
    df = add_divergence_features(df, cfg)

Hooked into ``engineer_all()`` in ``transforms.py`` after momentum features,
before log transforms.
"""

from __future__ import annotations

from typing import Any
import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ── Default signal pairs ───────────────────────────────────────────────────

# Each tuple: (col_a, col_b, short_name)
# These are high-macro-significance pairs whose correlation breakdowns
# historically coincide with regime transitions.
DEFAULT_DIVERGENCE_PAIRS: list[tuple[str, str, str]] = [
    ("sp500", "10yr_ustreas", "spy_tlt"),       # equity-bond
    ("sp500", "gold_spot", "spy_gld"),           # equity-gold
    ("gold_spot", "wti_crude", "gld_oil"),       # gold-oil
    ("credit_spread", "fred_vix", "cred_vix"),  # credit risk vs equity vol
]


# ── Layer 1: Rolling correlation baseline ──────────────────────────────────

def compute_rolling_correlation(
    series_a: pd.Series,
    series_b: pd.Series,
    window: int,
    use_returns: bool = True,
) -> pd.Series:
    """
    Compute rolling Pearson correlation between two series.

    Args:
        series_a: First time series.
        series_b: Second time series (same index).
        window: Rolling window in quarters.
        use_returns: If True, correlate pct_change (returns) rather than
                     levels.  Avoids spurious trending correlation.

    Returns:
        Series of rolling correlations, NaN where insufficient data.
    """
    if use_returns:
        a = series_a.pct_change()
        b = series_b.pct_change()
    else:
        a = series_a
        b = series_b

    min_periods = min(window, max(3, window // 2))
    return a.rolling(window=window, min_periods=min_periods).corr(b)


# ── Layer 2: Divergence magnitude ──────────────────────────────────────────

def compute_divergence(
    series_a: pd.Series,
    series_b: pd.Series,
    short_window: int = 4,
    long_window: int = 12,
    use_returns: bool = True,
) -> dict[str, pd.Series]:
    """
    Measure how the short-window correlation departs from the long-window baseline.

    Returns a dict of named Series:
        - ``corr_short``: rolling correlation over the short window
        - ``corr_long``:  rolling correlation over the long window (baseline)
        - ``div_raw``:    corr_short − corr_long (signed divergence)
        - ``div_abs``:    |corr_short − corr_long| (magnitude)
        - ``div_zscore``: divergence normalized by its own rolling std
                          (how many σ is the current divergence?)

    Args:
        series_a: First time series.
        series_b: Second time series.
        short_window: Short-term correlation window (quarters).
        long_window: Long-term baseline window (quarters).
        use_returns: Correlate returns rather than levels.
    """
    corr_short = compute_rolling_correlation(series_a, series_b, short_window, use_returns)
    corr_long = compute_rolling_correlation(series_a, series_b, long_window, use_returns)

    div_raw = corr_short - corr_long
    div_abs = div_raw.abs()

    # Z-score: normalize by rolling std of the divergence itself
    # Use the long window for the std baseline so it's stable
    div_std = div_raw.rolling(window=long_window, min_periods=max(4, long_window // 2)).std()
    div_zscore = div_raw / div_std.replace(0, np.nan)

    return {
        "corr_short": corr_short,
        "corr_long": corr_long,
        "div_raw": div_raw,
        "div_abs": div_abs,
        "div_zscore": div_zscore,
    }


# ── Layer 3: Divergence triggers ──────────────────────────────────────────

def compute_divergence_triggers(
    div_zscore: pd.Series,
    threshold: float = 2.0,
) -> dict[str, pd.Series]:
    """
    Generate binary trigger and direction features from a divergence z-score.

    Returns:
        - ``trigger``: 1 if |z-score| > threshold, 0 otherwise
        - ``direction``: +1 if z-score > 0 (correlation increased), −1 if < 0

    Args:
        div_zscore: Z-scored divergence series.
        threshold: Number of standard deviations for trigger activation.
    """
    trigger = (div_zscore.abs() > threshold).astype(int)
    direction = np.sign(div_zscore).fillna(0).astype(int)
    return {
        "trigger": trigger,
        "direction": direction,
    }


# ── Derivative-space divergence ────────────────────────────────────────────

def compute_derivative_divergence(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
    short_window: int = 4,
    long_window: int = 12,
) -> dict[str, pd.Series] | None:
    """
    Compute divergence in derivative space (d1 columns).

    If ``{col_a}_d1`` and ``{col_b}_d1`` exist in df, compute their
    rolling correlation divergence.  Derivative divergences can lead
    level-based divergences by 1-2 quarters.

    Returns None if derivative columns don't exist (e.g., called before
    derivatives are computed in the pipeline).
    """
    d1_a = f"{col_a}_d1"
    d1_b = f"{col_b}_d1"

    if d1_a not in df.columns or d1_b not in df.columns:
        return None

    # Derivatives are already rate-of-change, so correlate levels (not returns)
    return compute_divergence(
        df[d1_a], df[d1_b],
        short_window=short_window,
        long_window=long_window,
        use_returns=False,
    )


# ── Master wrapper ─────────────────────────────────────────────────────────

def add_divergence_features(
    df: pd.DataFrame,
    cfg: dict,
) -> pd.DataFrame:
    """
    Add all divergence features to the DataFrame.

    Called from ``engineer_all()`` in ``transforms.py`` after momentum
    features, before log transforms.

    For each configured signal pair, produces:
        - ``div_{name}_{short}q``: raw divergence (continuous)
        - ``div_{name}_abs_{short}q``: absolute divergence magnitude
        - ``div_{name}_z_{short}q``: z-scored divergence
        - ``div_{name}_trigger``: binary (1 if |z| > threshold)
        - ``div_{name}_dir``: direction (+1/−1)
        - ``div_d1_{name}_{short}q``: derivative-space divergence (if d1 columns exist)
        - ``div_d1_{name}_z_{short}q``: derivative-space z-score
        - ``div_d1_{name}_trigger``: derivative-space trigger

    Config keys (optional, under ``features.divergence``):
        pairs: list of [col_a, col_b, short_name] triples
        short_window: short correlation window (default 4)
        long_window: long baseline window (default 12)
        trigger_threshold: z-score threshold for triggers (default 2.0)

    Args:
        df: DataFrame with raw + cross-ratio + momentum columns.
        cfg: Pipeline config dict.

    Returns:
        DataFrame with divergence features appended.
    """
    feat_cfg = cfg.get("features", {})
    div_cfg = feat_cfg.get("divergence", {})

    short_window = div_cfg.get("short_window", 4)
    long_window = div_cfg.get("long_window", 12)
    threshold = div_cfg.get("trigger_threshold", 2.0)

    # Parse pairs from config or use defaults
    pairs_raw = div_cfg.get("pairs")
    if pairs_raw:
        pairs = [(p[0], p[1], p[2]) for p in pairs_raw]
    else:
        pairs = DEFAULT_DIVERGENCE_PAIRS

    result = df.copy()
    level_added = 0
    deriv_added = 0

    for col_a, col_b, name in pairs:
        if col_a not in result.columns or col_b not in result.columns:
            log.debug(
                "Skipping divergence pair %s: missing %s or %s",
                name, col_a, col_b,
            )
            continue

        # Level-space divergence
        div = compute_divergence(
            result[col_a], result[col_b],
            short_window=short_window,
            long_window=long_window,
            use_returns=True,
        )
        result[f"div_{name}_{short_window}q"] = div["div_raw"]
        result[f"div_{name}_abs_{short_window}q"] = div["div_abs"]
        result[f"div_{name}_z_{short_window}q"] = div["div_zscore"]

        triggers = compute_divergence_triggers(div["div_zscore"], threshold)
        result[f"div_{name}_trigger"] = triggers["trigger"]
        result[f"div_{name}_dir"] = triggers["direction"]
        level_added += 5

        # Derivative-space divergence (only if d1 columns exist)
        d1_div = compute_derivative_divergence(
            result, col_a, col_b,
            short_window=short_window,
            long_window=long_window,
        )
        if d1_div is not None:
            result[f"div_d1_{name}_{short_window}q"] = d1_div["div_raw"]
            result[f"div_d1_{name}_z_{short_window}q"] = d1_div["div_zscore"]

            d1_triggers = compute_divergence_triggers(d1_div["div_zscore"], threshold)
            result[f"div_d1_{name}_trigger"] = d1_triggers["trigger"]
            deriv_added += 3

    if level_added > 0:
        log.info(
            "Added %d level-space + %d derivative-space divergence features "
            "(%d pairs, short=%dQ, long=%dQ, threshold=%.1fσ)",
            level_added, deriv_added, len(pairs), short_window, long_window, threshold,
        )
    return result
