"""Empirical forward-transition matrix diagnostic (L2-02, design §5.1/R3).

``empirical_transition_matrix(states)`` returns the row-normalized K x K
crosstab of consecutive label pairs implied by the labeler's full smoothed
state history — this is the v1 transition machinery. The feature-conditional
time-varying-transition-probability (TVTP) model is explicitly deferred to v2
(L2-V2-02).

This is a descriptive diagnostic over the FULL (non-embargoed) smoothed
label sequence, not a training target — no honesty embargo applies to it
(see 03-04-PLAN.md threat register T-03-13).

Usage::

    from trading_crab_lib.platform.prediction.transition_matrix import (
        empirical_transition_matrix,
    )
    matrix = empirical_transition_matrix(regime_labels)
"""

from __future__ import annotations

import pandas as pd


def empirical_transition_matrix(states: pd.Series) -> pd.DataFrame:
    """Row-normalized K x K count table: P(next state = j | current state = i).

    Args:
        states: full label history (e.g. the labeler's smoothed state
            sequence), ordered by time.

    Returns:
        pd.DataFrame indexed by 'from' state, columned by 'to' state, with
        each occupied row summing to 1.0. A state that never appears as a
        'from' state (e.g. only the final label) produces no row for it —
        no ZeroDivision/NaN crash.
    """
    pairs = pd.DataFrame({"from": states.iloc[:-1].values, "to": states.iloc[1:].values})
    counts = pd.crosstab(pairs["from"], pairs["to"])
    return counts.div(counts.sum(axis=1), axis=0)


if __name__ == "__main__":
    # Synthetic self-check — no network, no checkpoint (mirrors gap_lag.py footer).
    demo_states = pd.Series([0, 1, 0, 1, 1, 0, 2, 2, 1])
    demo_matrix = empirical_transition_matrix(demo_states)
    print(demo_matrix)  # noqa: T201 — first-class self-check output
