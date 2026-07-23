"""
Hysteresis state machine — the L4-01 anti-flicker Schmitt trigger (design §5.3).

``update_active_regime()`` tracks P(the CURRENTLY-HELD regime's OWN
probability), never argmax(P) — a competitor regime spiking above the act
threshold must NOT steal the tilt while the held regime's own confidence is
still above the unwind floor (RESEARCH Pitfall 3). This is what keeps the
report from churning Glenn's Fidelity account on every noisy month.

State is persisted load-BEFORE-save via ``load_active_regime()`` /
``save_active_regime()``, mirroring ``labeling/diagnostics.py``'s
``_churn_against_previous()`` ordering (same Pitfall 3 concern: reading the
checkpoint after writing it would silently return the just-written value).

Usage::

    from trading_crab_lib.platform.allocation.hysteresis import (
        load_active_regime, save_active_regime, update_active_regime,
    )

    prev_active = load_active_regime()  # load BEFORE any save this run
    new_active = update_active_regime(probs, prev_active)
    save_active_regime(new_active)
"""

from __future__ import annotations

import logging

import pandas as pd

from trading_crab_lib.checkpoints import CheckpointManager
from trading_crab_lib.platform.checkpoints import get_platform_checkpoint_manager

log = logging.getLogger(__name__)


def load_active_regime(cm: CheckpointManager | None = None) -> int | None:
    """Load the previously-acted-on regime. Cold start (no checkpoint yet,
    or a persisted null) returns None."""
    cm = cm or get_platform_checkpoint_manager()
    try:
        state = cm.load("hysteresis_state")
    except FileNotFoundError:
        return None
    value = state["active_regime"].iloc[0]
    return None if pd.isna(value) else int(value)


def save_active_regime(active_regime: int | None, cm: CheckpointManager | None = None) -> None:
    """Persist the acted-on regime (or None for the neutral/cash-heavy posture)."""
    cm = cm or get_platform_checkpoint_manager()
    cm.save(pd.DataFrame([{"active_regime": active_regime}]), "hysteresis_state")


def update_active_regime(
    probs: pd.Series | dict,
    prev_active,
    *,
    act_threshold: float = 0.70,
    unwind_threshold: float = 0.40,
):
    """Schmitt-trigger regime switch — the pure state-transition function.

    Args:
        probs: regime id -> probability (pd.Series or dict).
        prev_active: the previously-held regime id, or None (cold start).
        act_threshold: probability a regime must reach to become/remain active (design §5.3, ~0.7).
        unwind_threshold: probability floor below which the held regime is
            considered to have collapsed (design §5.3, ~0.4).

    Returns:
        The new active regime id, or None (neutral/cash-heavy posture).

    Cold-start rule [ASSUMED — A1, not explicitly specified in design/CONTEXT,
    flagged for the weekly report]: with no prior state, act immediately on
    argmax(probs) if it already clears act_threshold; otherwise stay neutral
    until some regime first crosses the act threshold.
    """
    probs = pd.Series(probs)

    if prev_active is None:
        top_regime = probs.idxmax()
        return top_regime if probs[top_regime] >= act_threshold else None

    # Branch 2: the held regime's own probability is still high enough — HOLD,
    # regardless of any competitor's probability (Pitfall 3: never argmax(P)).
    if probs.get(prev_active, 0.0) >= unwind_threshold:
        return prev_active

    # Branch 3: the held regime has collapsed — switch only to a competitor
    # that has itself crossed the act threshold; otherwise go neutral.
    qualifying = probs[probs >= act_threshold]
    return qualifying.idxmax() if not qualifying.empty else None
