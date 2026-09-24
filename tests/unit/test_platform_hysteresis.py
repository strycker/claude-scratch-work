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
    execute_rebalance,
    hysteresis_thresholds,
    load_active_regime,
    no_trade_band_from_config,
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


# ── the 5pp no-trade band (plan 08-09 Task 1 ruling: b-bounded-turnover) ────
#
# Every arm below exhibits BOTH outcomes the band can produce — a suppressed trade and
# an allowed one — or pins a boundary exactly. A band test that only ever sees
# suppressed (or only allowed) trades proves nothing about the comparison.

BAND = 0.05


def _w(**kw: float) -> pd.Series:
    return pd.Series(kw, dtype=float)


class TestNoTradeBand:
    def test_suppresses_one_trade_and_allows_another_in_the_same_step(self):
        held = _w(SPY=0.50, TLT=0.30)
        target = _w(SPY=0.53, TLT=0.20)          # SPY moves 3pp, TLT moves 10pp
        out = execute_rebalance(target, 0.27, held, band=BAND)
        assert out["held_assets"] == ["SPY"] and out["traded_assets"] == ["TLT"]
        assert out["weights"]["SPY"] == 0.50      # suppressed: last executed weight kept
        assert out["weights"]["TLT"] == 0.20      # allowed: traded to target
        assert out["cash"] == pytest.approx(0.30)
        assert out["scale_down"] == 1.0

    def test_a_move_of_exactly_the_band_is_not_traded_and_just_above_is(self):
        """``<=``: exactly 5pp holds. 0.05 - 0.0 and 0.05 - 0.10 are exact in binary."""
        assert (0.05 - 0.0) == BAND and abs(0.05 - 0.10) == BAND
        up = execute_rebalance(_w(A=0.05), 0.95, _w(B=0.0), band=BAND)
        assert up["weights"]["A"] == 0.0 and "A" in up["held_assets"]
        down = execute_rebalance(_w(A=0.05), 0.95, _w(A=0.10), band=BAND)
        assert down["weights"]["A"] == 0.10 and down["held_assets"] == ["A"]
        above = execute_rebalance(_w(A=0.0500001), 0.9499999, _w(A=0.0), band=BAND)
        assert above["weights"]["A"] == 0.0500001 and above["traded_assets"] == ["A"]

    def test_a_zero_target_is_sold_only_above_the_band(self):
        held = _w(GLD=0.04, TLT=0.20, SPY=0.60)
        target = _w(SPY=0.62)                    # GLD and TLT absent from the target
        out = execute_rebalance(target, 0.38, held, band=BAND)
        assert out["weights"]["GLD"] == 0.04     # 4pp away from 0 -> held
        assert out["weights"]["TLT"] == 0.0      # 20pp away from 0 -> sold to zero
        assert out["weights"]["SPY"] == 0.60     # 2pp -> held
        assert sorted(out["held_assets"]) == ["GLD", "SPY"] and out["traded_assets"] == ["TLT"]
        assert out["cash"] == pytest.approx(0.36)

    def test_negative_residual_scales_only_the_traded_assets_pro_rata(self):
        held = _w(A=0.64, B=0.36)
        target = _w(A=0.60, B=0.20, C=0.20)      # A held at 0.64; B, C traded; 0.64+0.40 = 1.04
        # Precondition: without the scale-down the book would be over-invested.
        assert float(held["A"] + target["B"] + target["C"]) > 1.0
        out = execute_rebalance(target, 0.0, held, band=BAND)
        w = out["weights"]
        assert out["held_assets"] == ["A"] and out["traded_assets"] == ["B", "C"]
        assert w["A"] == 0.64                    # held asset untouched, exactly
        assert out["scale_down"] == pytest.approx(0.36 / 0.40)
        assert w["B"] == pytest.approx(0.18) and w["C"] == pytest.approx(0.18)
        assert w["B"] / w["C"] == pytest.approx(1.0)   # pro rata: the traded ratio survives
        assert float(w.sum()) == pytest.approx(1.0)
        assert out["cash"] == pytest.approx(0.0, abs=1e-12)

    def test_no_scale_down_when_the_residual_is_non_negative(self):
        out = execute_rebalance(_w(A=0.60, B=0.20, C=0.10), 0.10, _w(A=0.64, B=0.36), band=BAND)
        assert out["scale_down"] == 1.0 and out["weights"]["B"] == 0.20 and out["weights"]["C"] == 0.10

    def test_cash_is_the_residual_and_is_never_banded(self):
        """Target cash moves 2pp (0.50 -> 0.48); the band never compares cash — it is
        whatever the risky book leaves."""
        out = execute_rebalance(_w(A=0.52), 0.48, _w(A=0.50), band=BAND)
        assert out["weights"]["A"] == 0.50 and out["cash"] == 0.50 and out["cash"] != 0.48

    def test_held_is_the_last_executed_weight_not_the_last_target(self):
        """Two 3pp steps: 0.50 -> target 0.53 (held), then target 0.56. Against the last
        EXECUTED 0.50 that is 6pp and trades; against the last TARGET 0.53 it would be 3pp
        and hold. Fails if the caller carried the target instead of the executed book."""
        step1 = execute_rebalance(_w(A=0.53), 0.47, _w(A=0.50), band=BAND)
        assert step1["weights"]["A"] == 0.50
        step2 = execute_rebalance(_w(A=0.56), 0.44, step1["weights"], band=BAND)
        assert step2["weights"]["A"] == 0.56 and step2["traded_assets"] == ["A"]
        wrong = execute_rebalance(_w(A=0.56), 0.44, _w(A=0.53), band=BAND)
        assert wrong["weights"]["A"] == 0.53

    def test_first_step_with_no_held_trades_in_full(self):
        target = _w(SPY=0.03, TLT=0.40)          # SPY is inside the band from 0 — still bought
        out = execute_rebalance(target, 0.57, None, band=BAND)
        pd.testing.assert_series_equal(out["weights"], target)
        assert out["cash"] == pytest.approx(0.57) and out["held_assets"] == []

    def test_band_none_returns_the_target_objects_unchanged(self):
        target = _w(SPY=0.53, TLT=0.20)
        out = execute_rebalance(target, 0.27, _w(SPY=0.50, TLT=0.30), band=None)
        assert out["weights"] is target and out["cash"] == 0.27


class TestBandAndThresholdConfig:
    def test_live_config_carries_the_ruled_values(self):
        from trading_crab_lib.platform.config import load_platform_config

        cfg = load_platform_config()
        assert no_trade_band_from_config(cfg) == 0.05
        assert hysteresis_thresholds(cfg) == (0.70, 0.40)

    def test_absent_or_null_band_is_disabled(self):
        assert no_trade_band_from_config({}) is None
        assert no_trade_band_from_config({"allocation": {"no_trade_band": None}}) is None

    @pytest.mark.parametrize("bad", [0, 0.0, 1.0, -0.05, 1.5, True, "0.05"])
    def test_an_invalid_band_raises(self, bad):
        with pytest.raises(ValueError, match="no_trade_band"):
            no_trade_band_from_config({"allocation": {"no_trade_band": bad}})

    def test_absent_thresholds_are_the_ruled_pair(self):
        assert hysteresis_thresholds({}) == (0.70, 0.40)

    @pytest.mark.parametrize(
        ("act", "unwind"), [(0.40, 0.70), (1.01, 0.40), (0.70, 0.0), (0.70, -0.1), (True, 0.4), ("0.7", 0.4)]
    )
    def test_thresholds_that_break_the_invariant_raise(self, act, unwind):
        cfg = {"allocation": {"hysteresis": {"act_threshold": act, "unwind_threshold": unwind}}}
        with pytest.raises(ValueError, match="act_threshold|unwind|thresholds"):
            hysteresis_thresholds(cfg)
