"""
EVAL-03 headline orchestration — median sojourn / detection-lag ratio
(design §5.4). This is the go/no-go number the report shows FIRST (D-01a).

``sojourn_lag.py`` invents ZERO new math. It wires together three already
frozen, tested functions:

- ``labeling/diagnostics.py::occupancy_and_sojourns`` — the full-sample
  smoothed regime labeling's median run-length (sojourn) in months.
- ``honesty/gap_lag.py::compute_detection_lag`` — per-transition periods
  from a smoothed change-point until a real-time (filtered/walk-forward)
  probability series first crosses the action threshold.
- ``honesty/gap_lag.py::sojourn_lag_ratio`` — median sojourn / median lag.

Two distinct inputs are REQUIRED and must never be conflated (Pitfall 1):

1. ``full_sample_states`` — the full-sample smoothed L1 (jump-model) fit,
   indexed by date. This is what ``occupancy_and_sojourns`` measures, and
   its change points (positions where ``state[i] != state[i-1]``) are the
   ex-post regime transitions.
2. ``filtered_probs_matrix`` — the walk-forward, per-step MULTICLASS
   probability matrix (``build_filtered_probs_matrix`` below), one column
   per canonical state, indexed by decision date. This is what a real-time
   nowcaster actually knew at the time.

Review F1 fix: a transition INTO target state ``s`` is checked against
column ``s`` of ``filtered_probs_matrix`` — never a class-agnostic
max-across-classes probability. Using the row-max would systematically
understate detection lag, because an unrelated class's probability can
spike before the real target state's own probability crosses the
threshold — exactly the "fooled by its own backtest" failure this honesty
metric exists to prevent. Per-target-state lags are pooled across all
transitions (grouped by their own target state) before taking the median.

Usage::

    from trading_crab_lib.platform.evaluation.sojourn_lag import (
        build_filtered_probs_matrix, compute_sojourn_lag_headline,
    )

    filtered_probs_matrix = build_filtered_probs_matrix(per_step_metrics)
    headline = compute_sojourn_lag_headline(full_sample_states, filtered_probs_matrix)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trading_crab_lib.platform.honesty.gap_lag import compute_detection_lag, sojourn_lag_ratio
from trading_crab_lib.platform.labeling.diagnostics import occupancy_and_sojourns


def build_filtered_probs_matrix(per_step_metrics: dict) -> pd.DataFrame:
    """Stack the driver's per-step ``(proba, classes)`` rows into a
    ``(n_steps, K)`` multiclass probability matrix (review F1/F3).

    ``per_step_metrics`` is the walk-forward driver's return contract
    (``backtest/driver.py::run_backtest``) — keys ``"dates"``, ``"proba"``,
    ``"classes"``, with NO loop-sourced ``y_true`` (review F2). A given
    step's ``classes`` list may be a strict subset of the union of all
    states ever observed across every step (e.g. an early, small train
    window whose nowcaster only ever saw 3 of 5 canonical states) — every
    state not present in a step is padded with ``0.0``, never dropped or
    left as ``NaN`` (union-of-classes reconciliation).

    Args:
        per_step_metrics: dict with keys ``"dates"`` (list of decision
            dates), ``"proba"`` (list of 1-D probability arrays), and
            ``"classes"`` (list of lists of canonical state ids matching
            each step's ``proba`` array, in order).

    Returns:
        pd.DataFrame of shape ``(n_steps, K)``, indexed by
        ``per_step_metrics["dates"]``, with one column per state in the
        union of all observed states across every step (sorted ascending;
        fixed K=5 by ``labeling/jump_model.py::canonicalize_states`` in
        production, but this helper makes no assumption about K itself —
        it simply unions whatever appears).
    """
    dates = per_step_metrics["dates"]
    proba_list = per_step_metrics["proba"]
    classes_list = per_step_metrics["classes"]

    all_states = sorted({int(c) for classes in classes_list for c in classes})

    rows = []
    for proba, classes in zip(proba_list, classes_list):
        row = dict.fromkeys(all_states, 0.0)
        for state, p in zip(classes, proba):
            row[int(state)] = float(p)
        rows.append(row)

    return pd.DataFrame(rows, index=pd.DatetimeIndex(dates), columns=all_states)


def compute_sojourn_lag_headline(
    full_sample_states: pd.Series,
    filtered_probs_matrix: pd.DataFrame,
    *,
    act_threshold: float = 0.70,
) -> dict:
    """The EVAL-03 headline: median sojourn / median detection lag.

    Thin orchestration only — reimplements none of the underlying math:

    1. ``occupancy_and_sojourns(full_sample_states)`` gives
       ``overall_median_sojourn_months``.
    2. Ex-post transitions are derived from ``full_sample_states``' own
       change points: position ``i`` is a transition into
       ``target_state = full_sample_states[i]`` whenever
       ``full_sample_states[i] != full_sample_states[i-1]``.
    3. Transition positions are GROUPED by their own ``target_state``. For
       each target state ``s`` present as a column in
       ``filtered_probs_matrix``, that column is reindexed onto
       ``full_sample_states.index`` (so integer transition positions line
       up positionally; any date in ``full_sample_states`` with no
       corresponding walk-forward decision — e.g. the pre-warmup months —
       carries ``NaN`` and therefore never crosses the threshold) and
       ``compute_detection_lag`` is called with ONLY that state's own
       transitions against ONLY that state's own probability column
       (review F1 — never a class-agnostic max-across-classes series).
       Transitions into a state with no observed column at all are treated
       as fully unresolved (``NaN``, consistent with
       ``compute_detection_lag``'s own "unresolved" convention).
    4. Every transition's lag (across every target state) is pooled into
       one list and ``gap_lag.sojourn_lag_ratio`` is called on the median
       sojourn and the pooled median lag.

    Args:
        full_sample_states: the full-sample smoothed L1 state series,
            indexed by date (DatetimeIndex) — DISTINCT from
            ``filtered_probs_matrix`` (Pitfall 1); never the same object or
            derived from the same fit.
        filtered_probs_matrix: the walk-forward per-step multiclass
            probability matrix (``build_filtered_probs_matrix`` output),
            indexed by decision date, one column per canonical state.
        act_threshold: action threshold (design §5.4 default: 0.70).

    Returns:
        dict with keys ``median_sojourn``, ``median_lag``, ``ratio``.
    """
    states_arr = np.asarray(full_sample_states)
    occ = occupancy_and_sojourns(states_arr)
    median_sojourn = occ["overall_median_sojourn_months"]

    # Derive (position, target_state) pairs from the smoothed states' own
    # change points, then GROUP positions by their target_state — never by
    # any property of filtered_probs_matrix (Pitfall 1).
    transitions_by_state: dict[int, list[int]] = {}
    for i in range(1, len(states_arr)):
        if states_arr[i] != states_arr[i - 1]:
            target_state = int(states_arr[i])
            transitions_by_state.setdefault(target_state, []).append(i)

    pooled_lags: list[float] = []
    for target_state, positions in transitions_by_state.items():
        if target_state not in filtered_probs_matrix.columns:
            # No probability column was ever observed for this target state
            # across the whole walk-forward run — every transition into it
            # is unresolved, per compute_detection_lag's own NaN convention.
            pooled_lags.extend([float("nan")] * len(positions))
            continue

        # Reindex THIS state's own column onto the smoothed states'
        # positional index (review F1: never any other state's column, and
        # never a max-across-classes series). Dates in full_sample_states
        # with no corresponding filtered-probs row (pre-decision / warmup
        # months) carry NaN, which never crosses act_threshold.
        own_column = filtered_probs_matrix[target_state].reindex(full_sample_states.index)
        result = compute_detection_lag(positions, own_column, threshold=act_threshold)
        pooled_lags.extend(result["lags"])

    resolved = [lag for lag in pooled_lags if not np.isnan(lag)]
    median_lag = float(np.median(resolved)) if resolved else float("nan")

    ratio = sojourn_lag_ratio(median_sojourn, median_lag)

    return {"median_sojourn": median_sojourn, "median_lag": median_lag, "ratio": ratio}


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)

    # Synthetic self-check — no network, no Phase-1..4 dependency (mirrors
    # gap_lag.py's __main__ footer). full_sample_states and
    # filtered_probs_matrix are built independently (Pitfall 1): the former
    # is a hand-built smoothed run-length series, the latter simulates a
    # walk-forward driver's per_step_metrics via build_filtered_probs_matrix
    # with each target state's own probability ramping up 2 periods after
    # its ex-post transition (a deterministic, always-resolves construction
    # — random per-step probabilities risk an all-NaN/zero-lag degenerate
    # median on any given RNG draw, which is not a meaningful self-check).
    demo_index = pd.date_range("1972-01-31", periods=20, freq="ME")
    demo_states = pd.Series(
        [0] * 6 + [1] * 5 + [0] * 4 + [2] * 5,
        index=demo_index,
    )

    demo_dates = list(demo_index[6:20])
    demo_per_step_metrics: dict = {"dates": demo_dates, "proba": [], "classes": []}
    # Transitions land at position 6 (-> state 1), 11 (-> state 0), 15 (->
    # state 2); each own-state column ramps to a 0.8 crossing 2 periods
    # after its transition, so every transition resolves with lag == 2.
    ramp_starts = {1: 6, 0: 11, 2: 15}
    for step_idx in range(len(demo_dates)):
        position = step_idx + 6  # demo_dates start at full-index position 6
        proba = [0.8 if position >= ramp_starts[state] + 2 else 0.2 for state in (0, 1, 2)]
        demo_per_step_metrics["proba"].append(np.array(proba))
        demo_per_step_metrics["classes"].append([0, 1, 2])

    demo_matrix = build_filtered_probs_matrix(demo_per_step_metrics)
    headline = compute_sojourn_lag_headline(demo_states, demo_matrix)

    print(  # noqa: T201 — first-class self-check output
        "Sojourn/Lag Headline self-check (design §5.4)\n"
        f"  median sojourn (months): {headline['median_sojourn']}\n"
        f"  median detection lag:    {headline['median_lag']}\n"
        f"  ratio:                   {headline['ratio']}"
    )
