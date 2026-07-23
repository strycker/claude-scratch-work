"""
Tests for the L4-01 hysteresis state machine (Schmitt trigger on P(active regime)).

Headline invariant: a probability path oscillating 0.65<->0.72 for the HELD
regime must NEVER flip ``active_regime`` — this is a Schmitt trigger on the
held regime's OWN probability, not argmax(P) (RESEARCH Pitfall 3).
"""

from __future__ import annotations

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
