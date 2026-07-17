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

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


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
