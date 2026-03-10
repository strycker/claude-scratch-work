"""
Portfolio construction: regime-conditional allocation and trade recommendations.

Three public functions:

  simple_regime_portfolio()    — equal-weight the top-N assets for the current
                                 regime based on historical median returns.

  blended_regime_portfolio()   — probability-weighted allocation across ALL
                                 regimes, where each regime's weight is its
                                 predicted probability.  More stable than
                                 simple when regime confidence is moderate.

  generate_recommendation()    — compare a target portfolio to current holdings
                                 and emit BUY / SELL / HOLD signals per asset.

Usage in pipelines/07_dashboard.py:
    from market_regime.reporting.portfolio import (
        simple_regime_portfolio, blended_regime_portfolio, generate_recommendation,
    )
"""

from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger(__name__)


def simple_regime_portfolio(
    regime_returns: pd.DataFrame,
    current_regime: int,
    top_n: int = 3,
) -> pd.Series:
    """
    Equal-weight the top-N assets for the current regime.

    Args:
        regime_returns: pivoted DataFrame from returns_by_regime()
                        index=regime (int), columns=tickers (str),
                        values=median quarterly return.
        current_regime: integer label of the current predicted regime.
        top_n:          number of top assets to include.

    Returns:
        Series of portfolio weights indexed by ticker, summing to 1.0.
        Empty Series if current_regime not in regime_returns.
    """
    if current_regime not in regime_returns.index:
        log.warning("simple_portfolio: regime %d not in return history", current_regime)
        return pd.Series(dtype=float)

    row = regime_returns.loc[current_regime].dropna().sort_values(ascending=False)
    top = row.head(top_n)
    weights = pd.Series(1.0 / len(top), index=top.index)

    for asset, w in weights.items():
        log.info(
            "  simple  %-12s  %.1f%%  (hist. median: %+.2f%%/qtr)",
            asset, w * 100, row[asset] * 100,
        )
    return weights


def blended_regime_portfolio(
    regime_returns: pd.DataFrame,
    regime_probabilities: dict[int, float] | pd.Series,
    top_n: int = 3,
) -> pd.Series:
    """
    Probability-weighted allocation across all regimes.

    For each regime, identify its top-N assets and give them equal weight
    within that regime.  Then blend these per-regime allocations using the
    predicted regime probabilities as mixing weights.

    This is more robust than simple_regime_portfolio when the model assigns
    meaningful probability mass to multiple regimes simultaneously.

    Args:
        regime_returns:       pivoted DataFrame from returns_by_regime().
        regime_probabilities: dict or Series mapping regime_id → probability.
                              Values need not sum exactly to 1 (normalised
                              internally after removing zero-weight regimes).
        top_n:                number of assets to consider per regime.

    Returns:
        Series of blended weights indexed by ticker, summing to 1.0.
    """
    if isinstance(regime_probabilities, pd.Series):
        probs = regime_probabilities.to_dict()
    else:
        probs = dict(regime_probabilities)

    all_tickers = regime_returns.columns.tolist()
    blended = pd.Series(0.0, index=all_tickers)

    for regime, prob in probs.items():
        if regime not in regime_returns.index or prob <= 0:
            continue

        row = regime_returns.loc[regime].dropna().sort_values(ascending=False)
        top = row.head(top_n)
        if top.empty:
            continue

        # Equal weight within this regime, scaled by the regime's probability
        per_asset = prob / len(top)
        for asset in top.index:
            if asset in blended.index:
                blended[asset] += per_asset

        log.debug(
            "  blended regime %d (p=%.2f): top assets = %s",
            regime, prob, list(top.index),
        )

    total = blended.sum()
    if total <= 0:
        log.warning("blended_portfolio: all weights are zero — returning empty")
        return pd.Series(dtype=float)

    blended = (blended / total).sort_values(ascending=False)
    blended = blended[blended > 0]

    for asset, w in blended.items():
        log.info("  blended %-12s  %.1f%%", asset, w * 100)
    return blended


def generate_recommendation(
    target_weights: pd.Series,
    current_weights: pd.Series | None = None,
    threshold: float = 0.05,
) -> pd.DataFrame:
    """
    Compare current portfolio to target and emit BUY / SELL / HOLD signals.

    Args:
        target_weights:  recommended weights (e.g. from blended_regime_portfolio).
        current_weights: current portfolio weights.  Pass None to assume all-cash
                         (every target position is a BUY).
        threshold:       minimum absolute weight change (as a fraction) to trigger
                         a trade.  Smaller changes get HOLD.  Default 5%.

    Returns:
        DataFrame with columns:
            asset          — ticker / asset name
            current_pct    — current allocation (%)
            target_pct     — recommended allocation (%)
            delta_pct      — change needed (%)
            signal         — "BUY", "SELL", or "HOLD"
    """
    all_assets = sorted(
        set(target_weights.index)
        | (set(current_weights.index) if current_weights is not None else set())
    )

    records = []
    for asset in all_assets:
        current = (
            float(current_weights.get(asset, 0.0))
            if current_weights is not None
            else 0.0
        )
        target = float(target_weights.get(asset, 0.0))
        delta = target - current

        if delta > threshold:
            signal = "BUY"
        elif delta < -threshold:
            signal = "SELL"
        else:
            signal = "HOLD"

        records.append({
            "asset":       asset,
            "current_pct": round(current * 100, 1),
            "target_pct":  round(target * 100, 1),
            "delta_pct":   round(delta * 100, 1),
            "signal":      signal,
        })

    result = pd.DataFrame(records).set_index("asset")

    buys  = (result["signal"] == "BUY").sum()
    sells = (result["signal"] == "SELL").sum()
    holds = (result["signal"] == "HOLD").sum()
    log.info(
        "Trade signals: %d BUY, %d SELL, %d HOLD  (threshold=%.0f%%)",
        buys, sells, holds, threshold * 100,
    )
    return result
