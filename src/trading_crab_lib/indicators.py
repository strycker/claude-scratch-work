"""
Composite economic indicators derived from FRED series.

Phase D3: Conference Board LEI proxy — a simplified leading indicator
built from widely-available FRED series that approximates the
Conference Board's proprietary Leading Economic Index.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Components of the LEI proxy and their directional sign.
# Positive sign = higher value indicates expansion.
# Negative sign = higher value indicates contraction (inverted).
DEFAULT_LEI_COMPONENTS: dict[str, float] = {
    "fred_unrate": -1.0,     # unemployment rate (inverted — lower is better)
    "fred_t10y2y": 1.0,      # 10Y-2Y yield spread (positive = steeper curve)
    "fred_m2sl": 1.0,        # M2 money supply (expansion = liquidity)
    "fred_indpro": 1.0,      # industrial production index
    "fred_payems": 1.0,      # total nonfarm payrolls
}


def compute_lei_proxy(
    df: pd.DataFrame,
    *,
    components: dict[str, float] | None = None,
    window: int = 4,
) -> pd.Series:
    """Compute a Conference Board LEI proxy as a z-score composite.

    Each component is:
    1. Converted to quarterly percent change (or first difference for rates)
    2. Standardized to zero mean and unit variance over a rolling window
    3. Multiplied by its directional sign
    4. Equal-weight averaged across available components

    Parameters
    ----------
    df :
        DataFrame with FRED columns (e.g. fred_unrate, fred_t10y2y, etc.).
    components :
        ``{column_name: sign}`` mapping. Defaults to ``DEFAULT_LEI_COMPONENTS``.
    window :
        Rolling window (quarters) for z-score standardization. Default 4 (1 year).

    Returns
    -------
    pd.Series named ``lei_proxy`` aligned to *df*'s index.
    """
    if components is None:
        components = DEFAULT_LEI_COMPONENTS

    available = {col: sign for col, sign in components.items() if col in df.columns}

    if not available:
        log.warning("LEI proxy: no component columns found in DataFrame")
        return pd.Series(np.nan, index=df.index, name="lei_proxy")

    missing = set(components) - set(available)
    if missing:
        log.info("LEI proxy: %d/%d components available (%s missing)",
                 len(available), len(components), ", ".join(sorted(missing)))

    z_scores = []
    for col, sign in available.items():
        series = df[col].astype(float)

        # Rate-like series (yield spreads, unemployment) → first difference
        # Level series (M2, INDPRO, PAYEMS) → percent change
        if col in ("fred_unrate", "fred_t10y2y", "fred_t10y3m"):
            change = series.diff()
        else:
            change = series.pct_change()

        # Rolling z-score
        rolling_mean = change.rolling(window=window, min_periods=max(1, window // 2)).mean()
        rolling_std = change.rolling(window=window, min_periods=max(1, window // 2)).std()
        # Avoid division by zero
        rolling_std = rolling_std.replace(0, np.nan)
        z = (change - rolling_mean) / rolling_std
        z_scores.append(z * sign)

    # Equal-weight average across available components
    composite = pd.concat(z_scores, axis=1).mean(axis=1)
    composite.name = "lei_proxy"

    n_valid = composite.notna().sum()
    log.info("LEI proxy computed: %d valid quarters from %d components",
             n_valid, len(available))

    return composite
