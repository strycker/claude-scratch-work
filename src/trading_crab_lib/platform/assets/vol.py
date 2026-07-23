"""EWMA volatility forecasts (L3-02, design §6.2).

``ewma_vol()`` is the single RiskMetrics-style EWMA volatility implementation
reused by the allocation tilt (Plan 03, position sizing ∝ 1/σ̂) and the daily
tripwire (Plan 04, vol-spike signal) — do not duplicate the decay math
elsewhere; import this function.

Usage::

    from trading_crab_lib.platform.assets.vol import ewma_vol, DAILY_ANNUALIZATION
    vol = ewma_vol(daily_returns, halflife=11.2, annualization_factor=DAILY_ANNUALIZATION)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from trading_crab_lib import OUTPUT_DIR

log = logging.getLogger(__name__)

# RiskMetrics (1994) daily convention: lambda=0.94 => halflife ~= 11.2 trading days.
DAILY_ANNUALIZATION = 252
MONTHLY_ANNUALIZATION = 12

_ARTIFACT_COLUMNS = ["asset", "ewma_vol_annualized", "halflife"]


def ewma_vol(returns: pd.Series, *, halflife: float, annualization_factor: float) -> pd.Series:
    """Annualized EWMA volatility of a return series.

    ``halflife`` is in the SAME units as ``returns``' index frequency (days
    for daily returns, months for monthly returns) — RiskMetrics convention:
    lambda=0.94 daily => halflife~=11.2 trading days.

    Do not re-implement the decay loop — ``.ewm()`` is the single source of
    truth for the exponential weighting math.
    """
    return returns.ewm(halflife=halflife, min_periods=2).std() * np.sqrt(annualization_factor)


def latest_ewma_vol(returns: pd.Series, *, halflife: float, annualization_factor: float) -> float:
    """The last EWMA vol value as a plain float — the scalar sizing/tripwire
    code needs. NaN-safe: a length-1 (or otherwise all-NaN) series returns
    NaN rather than raising (min_periods=2 semantics)."""
    vol = ewma_vol(returns, halflife=halflife, annualization_factor=annualization_factor)
    return float(vol.iloc[-1])


def report_vol(vols: dict[str, float], *, halflife: float, output_dir: Path | None = None) -> Path:
    """Persist a schema-stable EWMA-vol parquet artifact and print a summary.

    Args:
        vols: mapping of asset ticker -> latest annualized EWMA vol.
        halflife: the halflife used to compute ``vols`` (recorded per row).
        output_dir: overrides the default artifact directory (for tests).

    Returns:
        Path: the written parquet artifact path.
    """
    target_dir = Path(output_dir) if output_dir is not None else OUTPUT_DIR / "reports" / "platform"
    target_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = target_dir / "ewma_vol.parquet"

    rows = [{"asset": asset, "ewma_vol_annualized": vol, "halflife": halflife} for asset, vol in vols.items()]
    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=_ARTIFACT_COLUMNS)
    df.to_parquet(artifact_path, index=False)

    summary = (
        "EWMA Volatility Forecasts (design §6.2)\n"
        f"  assets:    {len(vols)}\n"
        f"  halflife:  {halflife}\n"
        f"  artifact:  {artifact_path}"
    )
    log.info(summary)
    print(summary)  # noqa: T201 — first-class CLI run output, not debug noise

    return artifact_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Synthetic self-check — no network, mirrors gap_lag.py's __main__ footer.
    rng = np.random.default_rng(42)
    synthetic_daily_returns = pd.Series(rng.normal(0, 0.01, 500))
    vol_series = ewma_vol(synthetic_daily_returns, halflife=11.2, annualization_factor=DAILY_ANNUALIZATION)
    print(f"self-check: annualized EWMA vol = {vol_series.iloc[-1]:.2%}")  # noqa: T201
