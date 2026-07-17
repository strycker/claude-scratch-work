"""
Smoothed-vs-filtered gap and detection-lag metrics (HON-05, design §5.4).

- ``compute_gap(smoothed_perf, filtered_perf)``: performance of a
  regime-driven decision computed with smoothed labels minus the same with
  real-time (filtered/nowcast) probabilities — "the measured hindsight
  content of the strategy" (§5.4).
- ``compute_detection_lag(transitions, filtered_probs, threshold)``: periods
  from each ex-post transition until the real-time probability first
  crosses the action threshold at or after that transition.
- ``sojourn_lag_ratio(median_sojourn_months, median_detection_lag_months)``:
  the ratio that "largely determines whether regime timing can work" (§5.4).

RESEARCH.md Pitfall 4: the real smoothed/filtered regime series do not
exist until Phase 3 (jump-model labeler + logistic nowcaster). This module
delivers the GENERIC, tested compute functions plus the CLI + artifact
reporting surface (D-05), proven end-to-end on synthetic series here. Phase
3 plugs in real series against this exact interface, unchanged.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from trading_crab_lib import OUTPUT_DIR

log = logging.getLogger(__name__)

# Schema-stable columns for the persisted gap/lag artifact (D-05) — always
# present in the written parquet, even when metrics is empty (empty-safe
# DataFrame idiom, ported from model_metrics_artifacts.py).
_ARTIFACT_COLUMNS = ["gap", "detection_lag_median", "sojourn_lag_ratio"]


def compute_gap(smoothed_perf: float, filtered_perf: float) -> float:
    """Smoothed-vs-filtered gap: ``smoothed_perf - filtered_perf`` (§5.4)."""
    return float(smoothed_perf) - float(filtered_perf)


def compute_detection_lag(
    transitions,
    filtered_probs: pd.Series,
    threshold: float,
) -> dict:
    """Per-transition detection lag: periods from each transition position
    until ``filtered_probs`` first reaches >= ``threshold`` at or after that
    position.

    Args:
        transitions: iterable of integer positions into ``filtered_probs``
            (0-based, aligned to its position, not its label index).
        filtered_probs: real-time (filtered) probability series.
        threshold: action threshold (design §5.4: typically 0.7).

    Returns:
        dict with keys:
            - ``"lags"``: list[float] — per-transition lag; ``nan`` for a
              transition whose probability never crosses the threshold at
              or after its position (documented "unresolved" rule — not a
              silent 0 or dropped entry).
            - ``"median"``: float — median of the resolved (non-NaN) lags;
              ``nan`` if none resolve.

    Raises:
        ValueError: if ``transitions`` or ``filtered_probs`` is empty, or
            any transition position is out of bounds for ``filtered_probs``.
    """
    transitions = list(transitions)
    if len(transitions) == 0:
        raise ValueError("transitions must be non-empty")
    if len(filtered_probs) == 0:
        raise ValueError("filtered_probs must be non-empty")

    probs = pd.Series(filtered_probs).reset_index(drop=True)
    n = len(probs)

    lags: list[float] = []
    for t in transitions:
        t = int(t)
        if t < 0 or t >= n:
            raise ValueError(f"transition position {t} out of bounds for filtered_probs of length {n}")
        tail = probs.iloc[t:]
        crossing = tail[tail >= threshold]
        lags.append(float(crossing.index[0] - t) if not crossing.empty else float("nan"))

    resolved = [lag for lag in lags if not np.isnan(lag)]
    median = float(np.median(resolved)) if resolved else float("nan")
    return {"lags": lags, "median": median}


def sojourn_lag_ratio(median_sojourn_months: float, median_detection_lag_months: float) -> float:
    """§5.4: ratio largely determines whether regime timing can work.

    e.g. sojourn≈18m, lag≈2m -> 9.0 (most of the regime captured).
    e.g. sojourn≈5m, lag≈2m -> 2.5 (lag eats the trade).

    Raises:
        ValueError: if ``median_detection_lag_months`` is not strictly positive.
    """
    if median_detection_lag_months <= 0:
        raise ValueError("detection lag must be positive")
    return median_sojourn_months / median_detection_lag_months


def report_gap_lag(metrics: dict, *, output_dir: Path | None = None) -> Path:
    """Print a human-readable summary and persist the gap/lag artifact (D-05).

    Writes a parquet artifact under ``OUTPUT_DIR/reports/model_metrics/``
    (default; ``output_dir`` overrides the full target directory for
    tests). Only keys in :data:`_ARTIFACT_COLUMNS` are written — schema is
    stable even when ``metrics`` is empty (empty-safe DataFrame + to_parquet
    idiom, ported from ``model_metrics_artifacts.py``).

    Args:
        metrics: dict of computed gap/lag metrics, any subset of
            ``{"gap", "detection_lag_median", "sojourn_lag_ratio"}``.
        output_dir: overrides the default artifact directory (for tests).

    Returns:
        Path: the written parquet artifact path.
    """
    target_dir = Path(output_dir) if output_dir is not None else OUTPUT_DIR / "reports" / "model_metrics"
    target_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = target_dir / "gap_lag_metrics.parquet"

    rows = [{col: metrics.get(col) for col in _ARTIFACT_COLUMNS}] if metrics else []
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=_ARTIFACT_COLUMNS)
    df.to_parquet(artifact_path, index=False)

    summary = (
        "Gap/Lag Honesty Metrics (design §5.4)\n"
        f"  smoothed-vs-filtered gap:      {metrics.get('gap')}\n"
        f"  median detection lag (months): {metrics.get('detection_lag_median')}\n"
        f"  sojourn/lag ratio:             {metrics.get('sojourn_lag_ratio')}\n"
        f"  artifact:                      {artifact_path}"
    )
    log.info(summary)
    print(summary)  # noqa: T201 — first-class CLI run output (D-05), not debug noise

    return artifact_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Synthetic smoothed/filtered example — proves the CLI + artifact path
    # end-to-end with no network and no Phase 3 dependency (RESEARCH.md
    # Pitfall 4). Phase 3 plugs real jump-model/nowcaster series into these
    # same three compute functions with no interface change.
    smoothed_perf = 0.12
    filtered_perf = 0.09
    gap = compute_gap(smoothed_perf, filtered_perf)

    filtered_probs = pd.Series([0.1, 0.2, 0.3, 0.55, 0.7, 0.75, 0.2, 0.3, 0.6, 0.8])
    transitions = [3, 6]
    lag_result = compute_detection_lag(transitions, filtered_probs, threshold=0.5)

    median_sojourn_months = 18.0
    ratio = sojourn_lag_ratio(median_sojourn_months, lag_result["median"])

    report_gap_lag(
        {
            "gap": gap,
            "detection_lag_median": lag_result["median"],
            "sojourn_lag_ratio": ratio,
        }
    )
