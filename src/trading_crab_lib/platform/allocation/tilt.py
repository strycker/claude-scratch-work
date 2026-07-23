"""
Naive vol-targeted regime-tilt allocation math (L4-01, design §6.2/§7, D-03).

The tilt comes from historical regime-conditional Sharpe ranking (Plan 02's
returns-by-regime table, blended by the nowcaster's probabilities) — never
from raw point-forecast optimization (design §7 "shrink brutally"). Position
scale is ``min(1.0, target_vol / sigma_hat_port)``: long-only, capped at 1.0
(never levers), cash absorbs the ``1 - scale`` residual.

Usage::

    from trading_crab_lib.platform.allocation.tilt import vol_targeted_tilt

    result = vol_targeted_tilt(
        probs, returns_by_regime, asset_returns,
        target_vol_annual=0.10, halflife=6,
    )
    # result == {"weights": pd.Series, "cash": float, "scale": float, "portfolio_vol": float}
"""

from __future__ import annotations

import logging

import pandas as pd

from trading_crab_lib.platform.assets.vol import ewma_vol

log = logging.getLogger(__name__)


def vol_target_scale(target_vol_annual: float, portfolio_vol_annual: float) -> float:
    """Long-only, no-leverage scale factor. Cash absorbs 1 - scale.

    Source: design §6.2 "position size ~ 1/sigma_hat" + §7 "shrink return
    forecasts brutally" — combined into the naive v1 scale rule.
    """
    if portfolio_vol_annual <= 0:
        return 0.0  # no signal — degrade to all-cash rather than divide by zero
    return min(1.0, target_vol_annual / portfolio_vol_annual)


# ponytail: linear-sum-of-vols fallback deliberately ignores diversification —
# it can only make the tilt MORE conservative than the true portfolio vol, never
# less. Upgrade path: regime-conditional Ledoit-Wolf covariance (L3-V2-01, v2).
def portfolio_vol(
    weights: pd.Series,
    asset_returns: pd.DataFrame,
    *,
    halflife: float,
    min_obs: int = 12,
) -> float:
    """Naive no-covariance portfolio vol estimate (design §6.2, RESEARCH Pattern 5).

    With >= min_obs overlapping non-NaN months, EWMA-vols the blended
    weighted-return series directly (implicitly captures diversification via
    realized co-movement). Otherwise falls back to the conservative linear
    sum sum(w_i * sigma_hat_i), which over-estimates true portfolio vol
    (ignores diversification) and so under-levers rather than over-levers —
    safe-by-construction given D-03's defensive framing.
    """
    common_assets = weights.index.intersection(asset_returns.columns)
    aligned = asset_returns[common_assets].dropna()
    if len(aligned) >= min_obs:
        port_returns = (aligned * weights[common_assets]).sum(axis=1)
        return float(ewma_vol(port_returns, halflife=halflife, annualization_factor=12).iloc[-1])
    per_asset_vol = asset_returns[common_assets].apply(
        lambda s: ewma_vol(s.dropna(), halflife=halflife, annualization_factor=12).iloc[-1]
    )
    return float((weights[common_assets].abs() * per_asset_vol).sum())


def _per_regime_tilt(regime_id, returns_by_regime: pd.DataFrame, min_obs_flag: int) -> pd.Series:
    """Rank one regime's assets by clipped, normalized annualized Sharpe."""
    sub = returns_by_regime[returns_by_regime["regime"] == regime_id]
    if sub.empty:
        return pd.Series(dtype=float)

    low_conf = sub[sub["n_obs"] < min_obs_flag]
    for row in low_conf.itertuples():
        log.warning(
            "regime %s asset %s low-confidence: n_obs=%d < min_obs_flag=%d (D11 short history)",
            regime_id, row.asset, row.n_obs, min_obs_flag,
        )

    sharpe = sub.set_index("asset")["sharpe_annualized"].clip(lower=0).fillna(0.0)
    total = sharpe.sum()
    if total <= 0:
        return pd.Series(0.0, index=sharpe.index)
    return sharpe / total


def regime_tilt_weights(
    regime,
    returns_by_regime: pd.DataFrame,
    probs: pd.Series | dict,
    *,
    min_obs_flag: int = 6,
) -> pd.Series:
    """Naive, transparent regime-conditional tilt (Claude's discretion, D-03/§7).

    For every regime present in `probs`, ranks assets by within-regime
    annualized Sharpe (`returns_by_regime`, the long-format table: columns
    regime/asset/sharpe_annualized/n_obs), clips negative Sharpes to zero (no
    short positions — long-only), and normalizes to sum to 1 within that
    regime. The final tilt is the nowcaster-probability-weighted average of
    these per-regime tilts — never a single point-forecast regime bet.

    `regime` is the currently-held/active regime id (hysteresis output, may
    be None). If `probs` is empty/all-zero, the tilt falls back to a single
    point bet on `regime`'s per-regime weights (or an empty Series if
    `regime` is also None — fully neutral posture).

    Any (regime, asset) cell with n_obs < min_obs_flag is logged at WARNING
    as low-confidence (D11 short-history assets) but still included.
    """
    probs_series = pd.Series(probs, dtype=float) if probs is not None else pd.Series(dtype=float)
    probs_series = probs_series[probs_series > 0]

    if probs_series.empty:
        if regime is None:
            return pd.Series(dtype=float)
        return _per_regime_tilt(regime, returns_by_regime, min_obs_flag)

    blended = pd.Series(dtype=float)
    for regime_id, p in probs_series.items():
        tilt = _per_regime_tilt(regime_id, returns_by_regime, min_obs_flag)
        blended = blended.add(tilt * p, fill_value=0.0)

    total = blended.sum()
    return blended if total <= 0 else blended / total


def vol_targeted_tilt(
    regime_or_probs,
    returns_by_regime: pd.DataFrame,
    asset_returns: pd.DataFrame,
    *,
    target_vol_annual: float = 0.10,
    halflife: float,
    min_obs: int = 12,
) -> dict:
    """Assemble base tilt -> compute portfolio_vol -> scale -> long-only result.

    `regime_or_probs` accepts either a nowcaster probability mapping
    (pd.Series/dict, blended tilt) or a bare active-regime id (single-regime
    point bet — e.g. for a cold-start/degenerate case where only one regime
    is known).

    Returns:
        dict with keys "weights" (pd.Series, >= 0, summing to "scale"),
        "cash" (1 - scale), "scale" (leverage-capped position scale), and
        "portfolio_vol" (the annualized sigma_hat used to compute scale).
    """
    if isinstance(regime_or_probs, (pd.Series, dict)):
        probs = pd.Series(regime_or_probs, dtype=float)
        regime = probs.idxmax() if not probs.empty else None
    else:
        regime = regime_or_probs
        probs = pd.Series({regime: 1.0}, dtype=float) if regime is not None else pd.Series(dtype=float)

    base_weights = regime_tilt_weights(regime, returns_by_regime, probs)
    if base_weights.empty or base_weights.sum() <= 0:
        return {"weights": pd.Series(dtype=float), "cash": 1.0, "scale": 0.0, "portfolio_vol": float("nan")}

    port_vol = portfolio_vol(base_weights, asset_returns, halflife=halflife, min_obs=min_obs)
    scale = vol_target_scale(target_vol_annual, port_vol)
    return {
        "weights": base_weights * scale,
        "cash": 1.0 - scale,
        "scale": scale,
        "portfolio_vol": port_vol,
    }
