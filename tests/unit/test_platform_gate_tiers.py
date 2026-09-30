"""Pin of the A11 ruling: **`b-promote-dsr`, decided by Glenn on 2026-09-21**.

**What this module pins, and why it can fail.** A11 asked whether any gate in this
project should fail on a *bad-but-working* model rather than only on a broken
measurement. It was left **open by deliberate choice** on 2026-09-18
(``.planning/phases/07-regime-representation/07-BANDS.md`` §8), reopened by Glenn on
2026-09-21, and **answered YES** — promoting the deflated-Sharpe hurdle to a
governing quality gate. This module is that answer pinned; the reversal itself is
recorded in ``.planning/phases/08-regime-persistence-stability/08-A11.md`` and in
``platform_design/adr/0003-quality-gate-tier.md``.

**The gate, as a predicate.** A decision-bearing criterion-7 leg passes the quality
tier iff::

    observed_sharpe > expected_max_sharpe(total_trial_count(), sharpe_variance)

which is *identically* ``deflated_sharpe_ratio(...) > 0.5``, because the DSR is
``norm.cdf`` of a z whose numerator is exactly that difference.
:data:`~trading_crab_lib.platform.evaluation.deflated_sharpe._VERDICT_HURDLE` is
imported rather than re-declared so the threshold has a single source.

**The two values this module names.**

- **REJECTED** — the recorded criterion-7 legs, 588 steps 1972-01-31 -> 2020-12-31,
  routing ``L1_ONLY_LAST_FILTERED_STATE``: observed Sharpe **0.9170725133308871**
  (#1-alone) and **0.914903185594245** (joint), giving DSR **2.28151091802503e-12**
  and **1.4690427074211624e-11** against a hurdle of **2.2086935028832686**.
- **ACCEPTED** — observed Sharpe **2.60** on the same moments and track length,
  giving DSR **0.752687478118391**. The gate is therefore not a constant ``False``.

**The retroactive consequence, pinned deliberately.** Criterion 7's recorded
``MET 2026-09-21`` (ROADMAP) becomes **FAILED** on **both** legs under this ruling.
That consequence was accepted in advance, at the moment of the ruling, before any
number produced by phase 8 existed — not discovered at plan 08-10.

**The firewalled leg, named but not asserted on.** The L2-observational routing's
two legs report DSR 3.254192e-30 and 6.144320e-33 over the same window. That routing
carries ``NO_REGISTRY`` and is observational and firewalled per ADR-0002 decision
(e); no assertion here rests on it.

**The standing assumption, not papered over.** ``sharpe_variance`` is
:data:`DEGENERATE_SHARPE_VARIANCE` = 1.0, a **declared placeholder**, until 20
independent Sharpe-bearing trials exist. The 2.2086935028832686 hurdle therefore
rests on an assumption, carried forward by ADR-0002 open item 8 and now load-bearing
on a gate. ``test_the_variance_inside_the_hurdle_is_a_declared_placeholder`` pins
that it is still a placeholder, so the day it stops being one this pin goes red and
the hurdle is re-dated on purpose.
"""

from __future__ import annotations

import pathlib

import pytest

from trading_crab_lib.platform.evaluation.deflated_sharpe import (
    _VERDICT_HURDLE,
    DEGENERATE_SHARPE_VARIANCE,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    registry_sharpe_variance,
)
from trading_crab_lib.platform.honesty.registry import total_trial_count

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

# ── The ruling's own identifiers ─────────────────────────────────────────────────

SELECTED_OPTION_ID = "b-promote-dsr"
RULING_DATE = "2026-09-21"
REVERSES_DECISION_DATED = "2026-09-18"
A11_RECORD = REPO_ROOT / ".planning" / "phases" / "08-regime-persistence-stability" / "08-A11.md"
RULING_ADR = REPO_ROOT / "platform_design" / "adr" / "0003-quality-gate-tier.md"
PRIOR_ADR = REPO_ROOT / "platform_design" / "adr" / "0002-l1-second-classifier.md"

# ── The recorded criterion-7 measurement (07-JOINT-LIFT.md / measurement_l1only.json) ──

RECORDED_N_TRIALS = 42
RECORDED_TRIAL_CEILING = 44
RECORDED_HURDLE = 2.2086935028832686

RECORDED_N_OBS = 588
BASELINE_SHARPE = 0.9170725133308871
BASELINE_SKEW = 7.395819434497793
BASELINE_KURTOSIS = 125.80638949780558
BASELINE_RECORDED_DSR = 2.28151091802503e-12

JOINT_SHARPE = 0.914903185594245
JOINT_SKEW = 7.584287869738228
JOINT_KURTOSIS = 135.56908322096996
JOINT_RECORDED_DSR = 1.4690427074211624e-11

#: A Sharpe above the hurdle. Not a measurement — an exhibit, so the gate is shown to
#: have a passing side. Evaluated on the #1-alone leg's own moments and track length,
#: so the only thing that differs from the rejected arm is the Sharpe itself.
ACCEPTED_SHARPE = 2.60
ACCEPTED_DSR = 0.752687478118391


def quality_tier_passes(dsr: float) -> bool:
    """The A11 gate, as promoted. ``_VERDICT_HURDLE`` is imported, never re-declared."""
    return dsr > _VERDICT_HURDLE


def _dsr(observed_sharpe: float, *, skew: float, kurtosis: float, n_trials: int = RECORDED_N_TRIALS) -> float:
    return deflated_sharpe_ratio(
        observed_sharpe=observed_sharpe,
        n_trials=n_trials,
        sharpe_variance=DEGENERATE_SHARPE_VARIANCE,
        skew=skew,
        kurtosis=kurtosis,
        n_obs=RECORDED_N_OBS,
    )


# ── The gate rejects: the recorded criterion-7 legs ──────────────────────────────


class TestTheGateRejectsTheRecordedResult:
    """Both decision-bearing legs FAIL. This is the retroactive consequence, pinned."""

    @pytest.mark.parametrize(
        ("leg", "sharpe", "skew", "kurtosis", "recorded_dsr"),
        [
            ("#1-alone", BASELINE_SHARPE, BASELINE_SKEW, BASELINE_KURTOSIS, BASELINE_RECORDED_DSR),
            ("joint", JOINT_SHARPE, JOINT_SKEW, JOINT_KURTOSIS, JOINT_RECORDED_DSR),
        ],
    )
    def test_recorded_leg_fails_the_quality_tier(self, leg, sharpe, skew, kurtosis, recorded_dsr):
        dsr = _dsr(sharpe, skew=skew, kurtosis=kurtosis)
        assert dsr == pytest.approx(recorded_dsr, rel=1e-9), (
            f"the {leg} leg's DSR no longer reproduces the value recorded in "
            f"07-JOINT-LIFT.md ({recorded_dsr!r}); the estimator changed under a pin "
            "that exists to hold a verdict fixed"
        )
        assert not quality_tier_passes(dsr), (
            f"the {leg} leg (observed Sharpe {sharpe!r}, DSR {dsr!r}) now PASSES the "
            f"A11 quality tier. Under ruling {SELECTED_OPTION_ID} ({RULING_DATE}) it must "
            f"FAIL: {sharpe!r} does not exceed the hurdle {RECORDED_HURDLE!r}. If this is "
            "green, either the hurdle moved or the measurement did — say which, in writing, "
            "before moving this pin."
        )

    def test_the_rejected_sharpe_is_below_the_hurdle_by_more_than_a_factor_of_two(self):
        """The failure is not marginal, and the record does not get to call it close."""
        hurdle = expected_max_sharpe(RECORDED_N_TRIALS, DEGENERATE_SHARPE_VARIANCE)
        assert hurdle == pytest.approx(RECORDED_HURDLE, rel=1e-12)
        assert max(BASELINE_SHARPE, JOINT_SHARPE) * 2 < hurdle


# ── The gate accepts: a Sharpe above the hurdle ──────────────────────────────────


class TestTheGateAcceptsAValueAboveTheHurdle:
    """Without this arm the gate could be a constant ``False`` and every test above
    would still be green."""

    def test_a_sharpe_of_two_point_six_passes(self):
        dsr = _dsr(ACCEPTED_SHARPE, skew=BASELINE_SKEW, kurtosis=BASELINE_KURTOSIS)
        assert dsr == pytest.approx(ACCEPTED_DSR, rel=1e-9)
        assert quality_tier_passes(dsr), (
            f"observed Sharpe {ACCEPTED_SHARPE!r} exceeds the hurdle "
            f"{RECORDED_HURDLE!r} and must PASS; a gate that rejects everything is not "
            "a gate, it is a refusal"
        )

    def test_the_gate_turns_over_exactly_at_the_hurdle(self):
        """``dsr > 0.5`` and ``observed_sharpe > expected_max_sharpe(...)`` are the
        same statement — at the hurdle the DSR is exactly 0.5, not approximately."""
        at_hurdle = _dsr(RECORDED_HURDLE, skew=BASELINE_SKEW, kurtosis=BASELINE_KURTOSIS)
        assert at_hurdle == pytest.approx(0.5, abs=1e-12)
        assert not quality_tier_passes(at_hurdle), "the boundary is exclusive: equalling the bar is not clearing it"


# ── Provenance: the hurdle is derived, not a literal ─────────────────────────────


class TestTheHurdleIsDerivedFromTheTrialCount:
    """``b-promote-dsr`` was chosen over the three alternatives precisely because this
    threshold is arithmetic on the trial count rather than a number written by analogy.
    If the hurdle ever becomes a literal, that reason evaporates."""

    def test_more_trials_raise_the_bar(self):
        at_42 = expected_max_sharpe(RECORDED_N_TRIALS, DEGENERATE_SHARPE_VARIANCE)
        at_ceiling = expected_max_sharpe(RECORDED_TRIAL_CEILING, DEGENERATE_SHARPE_VARIANCE)
        assert at_ceiling > at_42, (
            "spending a trial must raise the bar that trial's own result has to clear; "
            "a hurdle insensitive to the trial count is not a multiple-testing correction"
        )
        assert at_ceiling == pytest.approx(2.2268911497604993, rel=1e-12)

    def test_the_live_trial_count_is_what_feeds_the_hurdle(self):
        """Read live, never copied. The count is a reading of a moment; what is pinned
        is that the hurdle tracks it and that the ADR-0002 ceiling still binds."""
        live = total_trial_count()
        assert live <= RECORDED_TRIAL_CEILING, (
            f"live trial count {live} exceeds ADR-0002's ceiling {RECORDED_TRIAL_CEILING}; "
            "exceeding it requires an explicit amendment to that ADR"
        )
        assert expected_max_sharpe(live, DEGENERATE_SHARPE_VARIANCE) >= RECORDED_HURDLE

    def test_the_recorded_legs_still_fail_at_the_ceiling(self):
        """The consequence does not depend on the count being exactly 42."""
        dsr = _dsr(
            BASELINE_SHARPE,
            skew=BASELINE_SKEW,
            kurtosis=BASELINE_KURTOSIS,
            n_trials=RECORDED_TRIAL_CEILING,
        )
        assert not quality_tier_passes(dsr)

    def test_the_variance_inside_the_hurdle_is_a_declared_placeholder(self, tmp_path):
        """07-BANDS §8's objection survives in narrowed form, and this pins the part
        that survives: the 1.0 is an assumption, not a measurement. When 20 independent
        Sharpe-bearing trials exist this goes red and the hurdle is re-dated on purpose."""
        assert DEGENERATE_SHARPE_VARIANCE == 1.0
        empty_ledger = tmp_path / "trials.jsonl"
        empty_ledger.write_text("", encoding="utf-8")
        assert registry_sharpe_variance(empty_ledger) == DEGENERATE_SHARPE_VARIANCE


# ── The ruling reached the record ────────────────────────────────────────────────


class TestTheRulingIsOnTheRecord:
    """A gate nobody wrote down is not a decision. These read the artifacts."""

    def test_the_adr_exists_and_names_the_selected_option(self):
        assert RULING_ADR.is_file(), f"{RULING_ADR} does not exist; the A11 ruling was not recorded as an ADR"
        text = RULING_ADR.read_text(encoding="utf-8")
        for token in (SELECTED_OPTION_ID, "A11", RULING_DATE, REVERSES_DECISION_DATED, "2.208694"):
            assert token in text, f"{RULING_ADR.name} does not name {token!r}"

    def test_the_prior_adr_points_at_the_ruling(self):
        text = PRIOR_ADR.read_text(encoding="utf-8")
        assert "0003-quality-gate-tier.md" in text, (
            "ADR-0002 carries A11 as open item 10; it must cross-reference the ADR that closes it"
        )

    def test_the_reversal_record_states_its_registry_cost_and_its_consequence(self):
        assert A11_RECORD.is_file(), f"{A11_RECORD} does not exist"
        text = A11_RECORD.read_text(encoding="utf-8")
        assert "registry rows spent: 0" in text, (
            "08-09 and 08-10 parse this exact phrase to reconcile the registry budget"
        )
        assert SELECTED_OPTION_ID in text
        assert "FAILED" in text, "the retroactive consequence for criterion 7 must be stated, not implied"
