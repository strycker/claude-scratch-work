"""
Tests for the L4-01 hysteresis state machine (Schmitt trigger on P(active regime)).

Headline invariant: a probability path oscillating 0.65<->0.72 for the HELD
regime must NEVER flip ``active_regime`` — this is a Schmitt trigger on the
held regime's OWN probability, not argmax(P) (RESEARCH Pitfall 3).
"""

from __future__ import annotations

import pandas as pd
import pytest

from trading_crab_lib.checkpoints import CheckpointManager
from trading_crab_lib.platform.allocation.hysteresis import (
    load_active_regime,
    save_active_regime,
    update_active_regime,
)

# ── headline no-flip invariant ───────────────────────────────────────────────


class TestNoFlipInvariant:
    def test_oscillating_probability_never_flips_active_regime(self):
        """P[A] oscillates 0.65<->0.72 (always >= unwind_threshold 0.40); a
        competitor B sits at ~0.30/0.28. active_regime must stay A on every step."""
        prev_active = "A"
        sequence = [
            {"A": 0.72, "B": 0.28},
            {"A": 0.65, "B": 0.30},
            {"A": 0.72, "B": 0.28},
            {"A": 0.65, "B": 0.30},
            {"A": 0.72, "B": 0.28},
        ]
        active = prev_active
        for probs in sequence:
            active = update_active_regime(probs, active)
            assert active == "A"

    def test_competitor_spike_does_not_steal_active_regime(self):
        """P[A]=0.45 (still >= 0.40 unwind floor) while P[B]=0.72 (>= act
        threshold) -> must return A, NOT B (Pitfall 3: only unwind once the
        HELD regime's own probability collapses)."""
        probs = {"A": 0.45, "B": 0.72}
        assert update_active_regime(probs, "A") == "A"


# ── unwind + switch branches ─────────────────────────────────────────────────


class TestUnwindAndSwitch:
    def test_unwind_and_switch_to_qualifying_competitor(self):
        """P[A]=0.35 (< 0.40 unwind) AND P[B]=0.75 (>= 0.70 act) -> switches to B."""
        probs = {"A": 0.35, "B": 0.75}
        assert update_active_regime(probs, "A") == "B"

    def test_unwind_to_neutral_when_no_competitor_qualifies(self):
        """P[A]=0.35 (< 0.40) and no competitor >= 0.70 -> returns None
        (cash-heavy neutral posture)."""
        probs = {"A": 0.35, "B": 0.50, "C": 0.15}
        assert update_active_regime(probs, "A") is None


# ── cold start (A1 assumption) ───────────────────────────────────────────────


class TestColdStart:
    def test_cold_start_returns_argmax_when_above_act_threshold(self):
        probs = {"A": 0.20, "B": 0.75, "C": 0.05}
        assert update_active_regime(probs, None) == "B"

    def test_cold_start_returns_none_when_below_act_threshold(self):
        probs = {"A": 0.20, "B": 0.60, "C": 0.20}
        assert update_active_regime(probs, None) is None


# ── update_active_regime is a pure function ──────────────────────────────────


class TestPureFunction:
    def test_no_checkpoint_io_in_body(self):
        import inspect

        source = inspect.getsource(update_active_regime)
        assert "cm.load" not in source
        assert "cm.save" not in source

    def test_accepts_pandas_series(self):
        import pandas as pd

        probs = pd.Series({"A": 0.72, "B": 0.28})
        assert update_active_regime(probs, "A") == "A"


# ── load-before-save persistence round-trip ──────────────────────────────────


class TestPersistence:
    def test_load_missing_checkpoint_returns_none(self, tmp_path):
        cm = CheckpointManager(checkpoint_dir=tmp_path)
        assert load_active_regime(cm) is None

    def test_save_then_load_round_trips(self, tmp_path):
        cm = CheckpointManager(checkpoint_dir=tmp_path)
        save_active_regime(2, cm)
        assert load_active_regime(cm) == 2

    def test_save_none_then_load_round_trips(self, tmp_path):
        cm = CheckpointManager(checkpoint_dir=tmp_path)
        save_active_regime(None, cm)
        assert load_active_regime(cm) is None


# ── F-2's one-hot identity, pinned as a permanent property (plan 08-09) ──────
#
# Under the decision-bearing ROUTING_L1_ONLY the hysteresis receives
# ``_last_state_one_hot(states_1)``. With a one-hot input and any threshold pair with
# ``0 < unwind <= act <= 1.0``, branch 2 fires iff ``prev_active`` is the hot state and
# branch 3 otherwise selects the single state at p=1.0 — so ``update_active_regime`` is
# the identity on argmax. Measured on the tracked l1only curves: ``active_regime`` equals
# ``state_1`` in all 588 months (1972-01-31 -> 2020-12-31), 246 changes, 0 None. This pin
# exists so no future reader takes that series as evidence the hysteresis does work.

_ADMISSIBLE_PAIRS = [
    (act, unwind)
    for act in (1e-9, 0.05, 0.1667, 0.40, 0.50, 0.70, 0.90, 1.0)
    for unwind in (1e-9, 0.05, 0.1667, 0.40, 0.50, 0.70, 0.90, 1.0)
    if 0.0 < unwind <= act <= 1.0
]


def _one_hot(hot: int, k: int) -> dict[int, float]:
    return {s: (1.0 if s == hot else 0.0) for s in range(k)}


class TestOneHotIdentity:
    def test_the_grid_is_large_and_covers_both_boundaries(self):
        """A grid that silently shrank to one pair would make the pin a single example."""
        assert len(_ADMISSIBLE_PAIRS) == 36
        assert (1.0, 1.0) in _ADMISSIBLE_PAIRS          # act == unwind == 1.0
        assert (0.70, 0.40) in _ADMISSIBLE_PAIRS        # the ruled pair (Task 2, keep-absolute)
        assert (1e-9, 1e-9) in _ADMISSIBLE_PAIRS        # act == unwind near 0

    @pytest.mark.parametrize(("act", "unwind"), _ADMISSIBLE_PAIRS)
    def test_one_hot_input_returns_argmax_for_every_prev_active(self, act, unwind):
        """K in {5, 6} (classifier #2's and #1's), every hot state, cold start AND every
        possible previously-held state."""
        n_cases = 0
        for k in (5, 6):
            for hot in range(k):
                probs = pd.Series(_one_hot(hot, k))
                for prev in [None, *range(k)]:
                    got = update_active_regime(probs, prev, act_threshold=act, unwind_threshold=unwind)
                    assert got == probs.idxmax() == hot, (k, hot, prev, act, unwind, got)
                    n_cases += 1
        assert n_cases == 5 * 6 + 6 * 7

    def test_the_identity_check_can_fail_on_a_non_one_hot_input(self):
        """Evidence the assertion above discriminates: fed a real posterior, the function
        is NOT the identity on argmax — branch 2 holds regime 1 while 0 is the argmax."""
        probs = pd.Series({0: 0.55, 1: 0.45})
        assert update_active_regime(probs, 1, act_threshold=0.70, unwind_threshold=0.40) == 1
        assert probs.idxmax() == 0

    def test_the_identity_needs_the_admissibility_bound(self):
        """Outside ``act <= 1.0`` a one-hot fails to clear the act threshold on a cold
        start and the function returns None — so the admissibility condition is what the
        identity rests on, not a formality."""
        probs = pd.Series(_one_hot(3, 6))
        assert update_active_regime(probs, None, act_threshold=1.01, unwind_threshold=0.40) is None
