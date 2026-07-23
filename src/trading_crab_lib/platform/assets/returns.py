"""Returns-by-regime tables (L3-01, design §14 Phase 1 / §7).

``returns_by_regime_stats(returns, states)`` conditions monthly asset returns
on the Phase-3 regime labels and returns one row per (regime, asset) with the
mean/std/annualized-Sharpe/hit-rate/max-drawdown/n_obs stats that drive the
naive regime-conditional tilt (RESEARCH.md Pattern 4) instead of raw point
forecasts.

Historical conditioning, not model selection: this table is descriptive
statistics over the FULL smoothed label history, NOT an evaluated model
configuration or a supervised-learning target. Per RESEARCH.md Pitfall 4/9
it is explicitly exempt from the trial registry and from purged CV in v1,
and reading full history (including any post-2021 rows once real data
exists) is acceptable here — see 04-02-PLAN.md threat register T-04-04/06.

Usage::

    from trading_crab_lib.platform.assets.returns import (
        compute_monthly_returns,
        returns_by_regime_stats,
        report_returns_by_regime,
    )
    returns = compute_monthly_returns(monthly_raw[asset_cols])
    stats = returns_by_regime_stats(returns, regime_labels["state"])
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from trading_crab_lib import OUTPUT_DIR
from trading_crab_lib.checkpoints import CheckpointManager
from trading_crab_lib.platform.checkpoints import get_platform_checkpoint_manager

log = logging.getLogger(__name__)

# A (regime, asset) cell with fewer than this many observed months is
# low-confidence (D11) — the report layer (Plan 05) flags rather than hides it.
_MIN_OBS_FLAG = 6

# Schema-stable columns for the persisted returns-by-regime artifact — always
# present in the written parquet, even when the frame is empty (empty-safe
# idiom ported from honesty/gap_lag.py).
_ARTIFACT_COLUMNS = [
    "regime",
    "asset",
    "mean_monthly_return",
    "std_monthly_return",
    "sharpe_annualized",
    "hit_rate",
    "max_drawdown",
    "n_obs",
]


def compute_monthly_returns(monthly_prices: pd.DataFrame) -> pd.DataFrame:
    """Month-over-month simple returns from a month-end level/price frame.

    NULL-tolerant (D11): a short-history asset simply produces leading NaNs
    (before its first observed level, and the pct_change of its first
    observation) — its column is never dropped.
    """
    return monthly_prices.pct_change()


def returns_by_regime_stats(returns: pd.DataFrame, states: pd.Series) -> pd.DataFrame:
    """One row per (regime, asset): mean/std/ann.Sharpe/hit_rate/max_dd/n_obs.

    Aligns ``returns`` and ``states`` via index intersection only (never
    positional slicing — Pitfall P3 analog: a returns/labels length
    mismatch must not silently misalign the conditioning).

    n_obs surfaces D11 short-history assets explicitly — a cell with
    ``n_obs < _MIN_OBS_FLAG`` is low-confidence, not hidden or dropped.
    """
    common = returns.index.intersection(states.index)
    joined = returns.loc[common].copy()
    joined["regime"] = states.loc[common]

    rows = []
    for regime, group in joined.groupby("regime"):
        asset_data = group.drop(columns=["regime"])
        for ticker in asset_data.columns:
            col = asset_data[ticker].dropna()
            if col.empty:
                continue
            cum = (1 + col).cumprod()
            rows.append(
                {
                    "regime": regime,
                    "asset": ticker,
                    "mean_monthly_return": col.mean(),
                    "std_monthly_return": col.std(),
                    "sharpe_annualized": (col.mean() / col.std()) * np.sqrt(12) if col.std() > 0 else np.nan,
                    "hit_rate": (col > 0).mean(),
                    "max_drawdown": (cum / cum.cummax() - 1).min(),
                    "n_obs": len(col),
                }
            )
    return pd.DataFrame(rows, columns=_ARTIFACT_COLUMNS)


def report_returns_by_regime(
    returns: pd.DataFrame,
    states: pd.Series,
    *,
    output_dir: Path | None = None,
    cm: CheckpointManager | None = None,
) -> Path:
    """Compute the returns-by-regime table, persist it, print a summary.

    Persists BOTH the ``returns_by_regime`` platform checkpoint (via
    ``cm`` or the default platform checkpoint namespace) AND a
    schema-stable parquet artifact under ``OUTPUT_DIR/reports/platform/``
    (``output_dir`` overrides the target directory for tests).

    Returns:
        Path: the written parquet artifact path.
    """
    stats = returns_by_regime_stats(returns, states)

    cm = cm or get_platform_checkpoint_manager()
    cm.save(stats, "returns_by_regime")

    target_dir = Path(output_dir) if output_dir is not None else OUTPUT_DIR / "reports" / "platform"
    target_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = target_dir / "returns_by_regime.parquet"
    df = stats if not stats.empty else pd.DataFrame(columns=_ARTIFACT_COLUMNS)
    df.to_parquet(artifact_path, index=False)

    low_confidence = int((stats["n_obs"] < _MIN_OBS_FLAG).sum()) if not stats.empty else 0
    summary = (
        "Returns-by-Regime Table (design §14 Phase 1 / §7)\n"
        f"  rows (regime x asset):        {len(stats)}\n"
        f"  low-confidence cells (n_obs<{_MIN_OBS_FLAG}): {low_confidence}\n"
        f"  artifact:                     {artifact_path}"
    )
    log.info(summary)
    print(summary)  # noqa: T201 — first-class CLI run output, not debug noise

    return artifact_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Synthetic self-check — no network, no checkpoint dependency.
    rng = np.random.default_rng(42)
    demo_idx = pd.date_range("2020-01-31", periods=36, freq="ME")
    demo_prices = pd.DataFrame(
        {
            "SPY": 100 * (1 + rng.normal(0.008, 0.03, 36)).cumprod(),
            "TLT": 100 * (1 + rng.normal(0.0, 0.02, 36)).cumprod(),
            "GLD": 100 * (1 + rng.normal(0.004, 0.02, 36)).cumprod(),
        },
        index=demo_idx,
    )
    demo_returns = compute_monthly_returns(demo_prices)
    demo_states = pd.Series(rng.integers(0, 3, 36), index=demo_idx)
    demo_stats = returns_by_regime_stats(demo_returns, demo_states)
    print(demo_stats.head())  # noqa: T201 — first-class self-check output
