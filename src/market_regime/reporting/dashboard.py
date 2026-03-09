"""
Stoplight dashboard.

Produces a simple textual/CSV summary of:
  - Current predicted regime + confidence
  - Forward-looking regime probabilities
  - Asset performance ranking for the predicted regime
  - Red/Yellow/Green signal per asset

Extend this module to render HTML or push to a web dashboard later.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

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
