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

from typing import Any
import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ── Private helpers ──────────────────────────────────────────────────────────

def _realized_vol(returns: pd.Series, window: int) -> float:
    """Realised volatility over a rolling window (std of returns)."""
    r = returns.dropna()
    if r.empty:
        return float("nan")
    if len(r) < window:
        return float(r.std(ddof=0))
    return float(r.rolling(window).std(ddof=0).iloc[-1])


def _trend_slope(prices: pd.Series, window: int) -> float:
    """Simple linear regression slope on log-prices over the last `window` points."""
    p = prices.dropna()
    if p.empty:
        return 0.0
    if len(p) < window:
        window = len(p)
    if window < 2:
        return 0.0
    y = np.log(p.iloc[-window:])
    x = np.arange(len(y), dtype=float)
    x = x - x.mean()
    y = y - y.mean()
    denom = (x ** 2).sum()
    if denom == 0:
        return 0.0
    return float((x * y).sum() / denom)


def _rolling_corr(a: pd.Series, b: pd.Series, window: int) -> float:
    """Rolling correlation over the last `window` points."""
    df = pd.concat([a, b], axis=1).dropna()
    if df.empty:
        return float("nan")
    if len(df) < window:
        return float(df.iloc[:, 0].corr(df.iloc[:, 1]))
    return float(df.iloc[-window:, 0].corr(df.iloc[-window:, 1]))


# ── Public API ───────────────────────────────────────────────────────────────

def compute_tactics_metrics(
    prices: pd.DataFrame,
    regimes: pd.Series | None = None,
    cfg: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Compute per-asset volatility, trend, correlation, and current regime metrics.

    Config shape (from settings.yaml → tactics):
        tactics:
          vol_windows: [5, 20, 60]
          trend_windows: [20, 60]
          corr_lookback: 60

    Args:
        prices:   DataFrame of prices (columns = tickers). If returns are passed
                  instead (no positive-only guarantee), trend_slope uses raw values.
        regimes:  optional Series of regime labels (used to tag current regime).
        cfg:      full config dict — reads cfg["tactics"] for window sizes.
                  If None, uses sensible defaults.

    Returns:
        DataFrame indexed by asset with vol/slope/correlation columns.
    """
    tac_cfg = (cfg or {}).get("tactics", {})
    vol_windows = tac_cfg.get("vol_windows", [5, 20, 60])
    trend_windows = tac_cfg.get("trend_windows", [20, 60])
    corr_lookback = int(tac_cfg.get("corr_lookback", 60))

    current_regime: int | None = None
    if regimes is not None and not regimes.dropna().empty:
        current_regime = int(regimes.dropna().iloc[-1])

    records: list[dict] = []
    for col in prices.columns:
        s = prices[col]
        ret = s.pct_change()
        row: dict = {"asset": col}
        if current_regime is not None:
            row["current_regime"] = current_regime

        for w in vol_windows:
            row[f"vol_{w}"] = _realized_vol(ret, w)

        for w in trend_windows:
            row[f"slope_{w}"] = _trend_slope(s, w)

        spy = prices["SPY"] if "SPY" in prices.columns else s
        row["corr_spy"] = _rolling_corr(s, spy, corr_lookback)

        records.append(row)

    return pd.DataFrame.from_records(records).set_index("asset")


def classify_tactics(metrics: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """
    Classify each asset as buy_hold / swing / stand_aside based on metrics.

    Config shape (from settings.yaml → tactics):
        tactics:
          vol_bands:
            low: 0.05
            high: 0.25
          trend_min_slope: 0.0

    Classification logic:
      - **buy_hold**: low volatility AND positive trend
      - **swing**: medium volatility AND positive trend
      - **stand_aside**: negative trend OR very high volatility
    """
    tac_cfg = (cfg or {}).get("tactics", {})
    vol_bands = tac_cfg.get("vol_bands", {"low": 0.05, "high": 0.25})
    low = float(vol_bands.get("low", 0.05))
    high = float(vol_bands.get("high", 0.25))
    trend_min = float(tac_cfg.get("trend_min_slope", 0.0))

    df = metrics.copy()
    vol_cols = sorted([c for c in df.columns if c.startswith("vol_")])
    slope_cols = sorted([c for c in df.columns if c.startswith("slope_")])
    if not vol_cols or not slope_cols:
        df["tactics_label"] = "stand_aside"
        return df

    # Use middle vol window as proxy when possible
    mid_idx = len(vol_cols) // 2
    vol_col = vol_cols[mid_idx]
    slope_col = slope_cols[0]

    labels: list[str] = []
    for _, row in df.iterrows():
        vol = row[vol_col]
        slope = row[slope_col]

        if np.isnan(vol) or np.isnan(slope):
            label = "stand_aside"
        elif vol <= low and slope >= trend_min:
            label = "buy_hold"
        elif low < vol < high and slope >= trend_min:
            label = "swing"
        else:
            label = "stand_aside"

        labels.append(label)

    df["tactics_label"] = labels
    return df
