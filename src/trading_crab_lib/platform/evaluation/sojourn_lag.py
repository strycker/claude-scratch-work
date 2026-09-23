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

Plan 08-06 adds ONE piece of new arithmetic, and says so:
``compute_signed_detection_offsets``. ``compute_detection_lag`` searches
forward only and floors at zero; the signed offset calls it for every
transition the belief has not already crossed, and walks BACKWARDS through
the contiguous at-or-above-threshold run only where it has. Both functions
derive transitions from the same ``_transitions_by_state`` rule.

Usage::

    from trading_crab_lib.platform.evaluation.sojourn_lag import (
        build_filtered_probs_matrix, compute_signed_detection_offsets,
        compute_sojourn_lag_headline,
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


def _transitions_by_state(states_arr: np.ndarray) -> dict[int, list[int]]:
    """The ONE transition-derivation rule (plan 08-06): change points of the smoothed states.

    Position ``i`` is a transition into ``int(states_arr[i])`` whenever
    ``states_arr[i] != states_arr[i - 1]``. Positions are GROUPED by their own
    target state, in first-seen order — never by any property of the filtered
    probability matrix (Pitfall 1). Extracted from
    :func:`compute_sojourn_lag_headline` so that every consumer of transitions in
    this module (the headline, and plan 08-06's signed offset) uses one rule and
    none can disagree about what a transition is.
    """
    transitions_by_state: dict[int, list[int]] = {}
    for i in range(1, len(states_arr)):
        if states_arr[i] != states_arr[i - 1]:
            target_state = int(states_arr[i])
            transitions_by_state.setdefault(target_state, []).append(i)
    return transitions_by_state


def _require_integer_state_columns(filtered_probs_matrix: pd.DataFrame, *, caller: str) -> None:
    """The T0.12 guard, shared: refuse a matrix whose columns cannot denote integer states."""
    # ── T0.12 guard: a wrong column shape must not read as "no detections" ──
    #
    # target_state is an int. When filtered_probs_matrix carries "state_{k}"
    # STRING columns instead of canonical integer labels, `0 not in
    # ["state_0", ...]` is True for EVERY state, so every transition fell into
    # the caller's no-column branch and the function returned n_resolved = 0,
    # median_lag = NaN, ratio = NaN — with no exception and no warning. That
    # reads as the substantive finding "real-time detection never happened" when
    # the cause is that the caller passed the wrong matrix shape. Plan 07-11's
    # first draft reported 0 of 25 transitions resolved for exactly this reason,
    # and the wave-2 validation audit confirmed the trap was still live.
    #
    # The discriminator is the column TYPE, not overlap with the observed target
    # states. Overlap is the wrong test: a labeling whose only transition targets
    # a state that genuinely never appears as a column has zero overlap and is
    # still perfectly legitimate — that transition is unresolved and keeps the
    # NaN convention. What is never legitimate is a column that cannot denote a
    # canonical integer state at all.
    non_integer_columns = [
        col for col in filtered_probs_matrix.columns
        if not (isinstance(col, (int, np.integer)) and not isinstance(col, bool))
    ]
    if non_integer_columns:
        raise ValueError(
            f"{caller}: filtered_probs_matrix must be keyed by "
            "CANONICAL INTEGER state labels (build_filtered_probs_matrix's own "
            f"output). Got {len(non_integer_columns)} non-integer column(s): "
            f"{[repr(c) for c in non_integer_columns[:5]]}"
            f"{' ...' if len(non_integer_columns) > 5 else ''}. The usual culprit is "
            "a 'state_{k}'-string matrix, against which every transition scores "
            "unresolved and this function returns n_resolved=0, median_lag=NaN — "
            "indistinguishable from 'real-time detection never happened'. Refusing "
            "to report a zero that means a shape error."
        )


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

    Raises:
        ValueError: if ``filtered_probs_matrix`` carries any non-integer column
            (T0.12). Such a matrix — the ``state_{k}``-string shape is the usual
            culprit — scores every transition unresolved and would return
            ``n_resolved=0, median_lag=NaN``, indistinguishable from "real-time
            detection never happened". A target state that is simply ABSENT from
            an otherwise integer-keyed matrix is NOT an error: that transition is
            genuinely unresolved and keeps the NaN convention.
    """
    states_arr = np.asarray(full_sample_states)
    occ = occupancy_and_sojourns(states_arr)
    median_sojourn = occ["overall_median_sojourn_months"]

    transitions_by_state = _transitions_by_state(states_arr)

    _require_integer_state_columns(filtered_probs_matrix, caller="compute_sojourn_lag_headline")

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

    # Sample-size transparency: the ratio is a median over `n_resolved`
    # transitions (those whose target-state probability ever reached the
    # threshold) out of `n_transitions` total. A long-history reference yields
    # few regimes, and the 0.70 action threshold rarely resolves for a calibrated
    # 5-class model — so a small n_resolved means the headline is indicative, not
    # robust. Surfaced so the number is never mistaken for a large-sample estimate.
    n_transitions = sum(len(p) for p in transitions_by_state.values())
    n_resolved = len(resolved)

    return {
        "median_sojourn": median_sojourn,
        "median_lag": median_lag,
        "ratio": ratio,
        "n_transitions": n_transitions,
        "n_resolved": n_resolved,
        "act_threshold": act_threshold,
    }


def compute_signed_detection_offsets(
    full_sample_states: pd.Series,
    filtered_probs_matrix: pd.DataFrame,
    *,
    act_threshold: float = 0.70,
) -> dict:
    """Per-transition SIGNED detection offset — a lag that is allowed to be negative.

    **Why this exists beside** ``compute_detection_lag``: that function searches
    ``probs.iloc[t:]`` only (``honesty/gap_lag.py:86``), so it floors at zero. A
    belief that already sat above the threshold *before* the reference transition
    reads as lag **0** there — indistinguishable from a same-month detection. A
    causal filtered belief cannot know a transition before the data that reveal
    it; a smoothed (two-sided) label can and does. The leakage guard
    (``tests/unit/test_platform_nowcaster_recursion.py``) has to tell those two
    apart, so it needs a quantity that can go below zero. ``compute_detection_lag``
    is not changed: its forward-only convention is right for a detection *lag*.
    This is a different quantity with a different name.

    **Definition.** Transitions come from :func:`_transitions_by_state` — the same
    rule the headline uses. For a transition at position ``i`` into state ``s``,
    let ``c`` be ``filtered_probs_matrix[s]`` reindexed onto
    ``full_sample_states.index`` (dates with no filtered row carry NaN, which never
    counts as at-or-above the threshold):

    - ``c[i] >= act_threshold``: walk **backwards** to the start ``j`` of the
      contiguous at-or-above-threshold run containing ``i``; the offset is
      ``j - i <= 0``. Only the run containing ``i`` counts — an earlier, separate
      excursion above the threshold is not a lead on this transition. A run that
      reaches the first observed row is truncated there, so the offset is then the
      most negative value the data can show.
    - otherwise: the offset is ``compute_detection_lag``'s own answer for that
      transition (called, not reimplemented), which is ``>= 1`` or NaN.

    Unresolved (NaN) transitions — the column never crosses at or after ``i``, or
    ``s`` has no column at all — are counted in ``n_transitions`` and excluded from
    ``median_offset``/``min_offset``, the same convention ``compute_detection_lag``
    documents.

    Returns:
        dict with ``offsets`` (list, chronological order), ``positions`` and
        ``target_states`` (parallel lists), ``per_state`` (state -> offsets),
        ``n_transitions``, ``n_resolved``, ``min_offset``, ``median_offset``,
        ``n_negative`` (offset < 0), ``n_zero_or_negative`` (offset <= 0),
        ``act_threshold``.

    Raises:
        ValueError: on a non-integer-keyed matrix (the shared T0.12 guard).
    """
    _require_integer_state_columns(filtered_probs_matrix, caller="compute_signed_detection_offsets")
    states_arr = np.asarray(full_sample_states)
    transitions_by_state = _transitions_by_state(states_arr)

    by_position: dict[int, tuple[int, float]] = {}
    per_state: dict[int, list[float]] = {}
    for target_state, positions in transitions_by_state.items():
        if target_state not in filtered_probs_matrix.columns:
            offsets = [float("nan")] * len(positions)
        else:
            own_column = filtered_probs_matrix[target_state].reindex(full_sample_states.index)
            forward = compute_detection_lag(positions, own_column, threshold=act_threshold)["lags"]
            values = own_column.to_numpy(dtype=float)
            offsets = []
            for i, lag in zip(positions, forward):
                if values[i] >= act_threshold:
                    # The backward walk. compute_detection_lag (gap_lag.py:86) would
                    # report 0 here whether the belief crossed this month or crossed
                    # months ago; the guard needs to know which. Walk back to the
                    # start of the contiguous at-or-above run that contains i.
                    j = i
                    while j - 1 >= 0 and values[j - 1] >= act_threshold:
                        j -= 1
                    offsets.append(float(j - i))
                else:
                    offsets.append(float(lag))
        per_state[target_state] = offsets
        for i, off in zip(positions, offsets):
            by_position[i] = (target_state, off)

    ordered = sorted(by_position)
    all_offsets = [by_position[i][1] for i in ordered]
    resolved = [o for o in all_offsets if not np.isnan(o)]
    return {
        "offsets": all_offsets,
        "positions": ordered,
        "target_states": [by_position[i][0] for i in ordered],
        "per_state": per_state,
        "n_transitions": len(all_offsets),
        "n_resolved": len(resolved),
        "min_offset": float(min(resolved)) if resolved else float("nan"),
        "median_offset": float(np.median(resolved)) if resolved else float("nan"),
        "n_negative": sum(1 for o in resolved if o < 0),
        "n_zero_or_negative": sum(1 for o in resolved if o <= 0),
        "act_threshold": act_threshold,
    }


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
