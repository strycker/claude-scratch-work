"""
Portfolio Construction: regime-conditional allocation recommendations.

Given predicted regime probabilities and per-regime asset performance,
construct portfolio weight recommendations.

This module implements two approaches:
1. Simple "best asset per regime" allocation
2. Probability-weighted blended allocation across regimes
"""

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def simple_regime_portfolio(
    regime_returns: pd.DataFrame,
    current_regime: int,
    metric: str = "median",
    top_n: int = 3,
) -> pd.Series:
    """
    Allocate equally among the top N assets for the current regime.

    This is the simplest approach: look at which assets performed best
    historically in the current regime, and split allocation evenly.

    Args:
        regime_returns: output of returns_by_regime()
        current_regime: integer label of the predicted current regime
        metric: 'mean' or 'median' returns to rank by
        top_n: number of top assets to include

    Returns:
        Series of portfolio weights (sums to 1.0)
    """
    log.info("Building simple portfolio for regime %d (top %d by %s)...",
             current_regime, top_n, metric)

    # Extract metric values for the current regime
    metric_cols = [col for col in regime_returns.columns if col[1] == metric]
    if current_regime not in regime_returns.index:
        log.warning("  Regime %d not found in returns data", current_regime)
        return pd.Series(dtype=float)

    row = regime_returns.loc[current_regime, metric_cols]
    row.index = [col[0] for col in metric_cols]

    # Pick top N and allocate equally
    top = row.nlargest(top_n)
    weights = pd.Series(1.0 / len(top), index=top.index)

    for asset, weight in weights.items():
        log.info("  %s: %.1f%% (historical %s: %.4f)",
                 asset, weight * 100, metric, row[asset])
    return weights


def blended_regime_portfolio(
    regime_returns: pd.DataFrame,
    regime_probabilities: pd.Series,
    metric: str = "median",
    top_n: int = 3,
) -> pd.Series:
    """
    Blend portfolio weights across regimes by their predicted probabilities.

    More sophisticated than simple allocation: if we're 60% likely in
    regime A and 40% likely in regime B, we blend the best assets from
    both regimes proportionally.

    Args:
        regime_returns: output of returns_by_regime()
        regime_probabilities: Series mapping regime → probability (sums to ~1)
        metric: return metric to rank by
        top_n: number of assets per regime to consider

    Returns:
        Series of blended portfolio weights (sums to 1.0)
    """
    log.info("Building probability-blended portfolio...")

    metric_cols = [col for col in regime_returns.columns if col[1] == metric]
    asset_names = [col[0] for col in metric_cols]

    blended = pd.Series(0.0, index=asset_names)

    for regime, prob in regime_probabilities.items():
        if regime not in regime_returns.index or prob <= 0:
            continue

        row = regime_returns.loc[regime, metric_cols]
        row.index = asset_names

        # Rank assets, give weight only to top N
        top = row.nlargest(top_n)
        regime_weights = pd.Series(0.0, index=asset_names)
        regime_weights[top.index] = 1.0 / len(top)

        blended += prob * regime_weights
        log.debug("  Regime %d (p=%.2f): top assets = %s",
                  regime, prob, list(top.index))

    # Normalize to sum to 1
    total = blended.sum()
    if total > 0:
        blended /= total

    # Drop zero-weight assets for clean output
    blended = blended[blended > 0].sort_values(ascending=False)

    for asset, weight in blended.items():
        log.info("  %s: %.1f%%", asset, weight * 100)
    return blended


def generate_recommendation(
    current_weights: pd.Series | None,
    target_weights: pd.Series,
    threshold: float = 0.05,
) -> pd.DataFrame:
    """
    Compare current portfolio to target and generate buy/sell/hold signals.

    Args:
        current_weights: current portfolio weights (None = all cash)
        target_weights: recommended weights from portfolio construction
        threshold: minimum weight change to trigger a trade signal

    Returns:
        DataFrame with columns: asset, current, target, delta, signal
    """
    log.info("Generating trade recommendations (threshold=%.0f%%)...",
             threshold * 100)

    all_assets = set(target_weights.index)
    if current_weights is not None:
        all_assets |= set(current_weights.index)
    all_assets = sorted(all_assets)

    records = []
    for asset in all_assets:
        current = current_weights.get(asset, 0.0) if current_weights is not None else 0.0
        target = target_weights.get(asset, 0.0)
        delta = target - current

        if delta > threshold:
            signal = "BUY"
        elif delta < -threshold:
            signal = "SELL"
        else:
            signal = "HOLD"

        records.append({
            "asset": asset,
            "current_pct": round(current * 100, 1),
            "target_pct": round(target * 100, 1),
            "delta_pct": round(delta * 100, 1),
            "signal": signal,
        })

    result = pd.DataFrame(records).set_index("asset")

    buys = (result["signal"] == "BUY").sum()
    sells = (result["signal"] == "SELL").sum()
    holds = (result["signal"] == "HOLD").sum()
    log.info("  Signals: %d BUY, %d SELL, %d HOLD", buys, sells, holds)

    return result
