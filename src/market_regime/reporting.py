"""
Dashboard and portfolio construction.

Dashboard (stoplight summary):
  asset_signals()       — RED/YELLOW/GREEN per asset for the current regime
  print_dashboard()     — concise textual summary to stdout
  save_dashboard_csv()  — write asset signals to CSV

Portfolio construction (regime-conditional allocation):
  simple_regime_portfolio()   — equal-weight top-N assets for the current regime
  blended_regime_portfolio()  — probability-weighted allocation across all regimes
  generate_recommendation()   — BUY / SELL / HOLD signals vs current holdings
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


# ── Dashboard ──────────────────────────────────────────────────────────────────

# Default thresholds — overridden at call time by values from settings.yaml.
# Do not tune these constants here; edit config/settings.yaml instead.
_DEFAULT_SIGNAL_THRESHOLDS = {
    "green":  0.05,   # median return > +5%/quarter
    "yellow": 0.00,   # median return > 0%
    # below 0% → red
}


def asset_signals(
    ranked_returns: pd.DataFrame,
    current_regime: int,
    thresholds: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    For the current regime, assign stoplight signal to each asset.

    Returns:
        DataFrame with columns: asset, median_quarterly_return, signal
    """
    t = thresholds if thresholds is not None else _DEFAULT_SIGNAL_THRESHOLDS
    subset = ranked_returns[ranked_returns["regime"] == current_regime].copy()

    def _signal(ret: float) -> str:
        if ret >= t["green"]:
            return "GREEN"
        elif ret >= t["yellow"]:
            return "YELLOW"
        return "RED"

    subset["signal"] = subset["median_quarterly_return"].apply(_signal)
    return subset[["asset", "median_quarterly_return", "signal", "rank"]]


def print_dashboard(
    current_prediction: dict,
    regime_names: dict[int, str],
    asset_signals_df: pd.DataFrame,
    transition_matrix: pd.DataFrame,
) -> None:
    """Print a concise dashboard to stdout / log."""
    regime = current_prediction["regime"]
    proba = current_prediction["probabilities"]

    print("\n" + "=" * 60)
    print(f"  CURRENT REGIME:  {regime} — {regime_names.get(regime, 'Unknown')}")
    print(f"  CONFIDENCE:      {proba[regime]:.1%}")
    print("=" * 60)

    print("\nRegime Probabilities:")
    for r, p in sorted(proba.items(), key=lambda x: -x[1]):
        bar = "█" * int(p * 30)
        print(f"  Regime {r} ({regime_names.get(r, '?'):<30s})  {p:5.1%}  {bar}")

    print("\nAsset Signals (current regime):")
    for _, row in asset_signals_df.iterrows():
        icon = {"GREEN": "●", "YELLOW": "◑", "RED": "○"}.get(row["signal"], "?")
        print(f"  {icon} {row['asset']:<8s}  {row['median_quarterly_return']:+.1%}  [{row['signal']}]")

    print("\nForward Transition Probabilities (from current regime):")
    if regime in transition_matrix.index:
        fwd = transition_matrix.loc[regime].sort_values(ascending=False)
        for to_r, p in fwd.items():
            print(f"  → Regime {to_r} ({regime_names.get(int(to_r), '?'):<30s})  {p:.1%}")
    print()


def save_dashboard_csv(
    asset_signals_df: pd.DataFrame,
    output_dir: Path,
    filename: str = "dashboard.csv",
) -> Path:
    out = output_dir / filename
    asset_signals_df.to_csv(out, index=False)
    log.info("Dashboard saved to %s", out)
    return out


# ── Portfolio construction ─────────────────────────────────────────────────────

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
