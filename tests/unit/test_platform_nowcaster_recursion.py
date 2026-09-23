"""The leakage guard for the Bayes-filter recursion (plan 08-06 Task 3, S-1 and S-3).

08-CONTEXT.md D-03, verbatim: *"A guard test must FAIL if the smoothed label is
substituted. A test that merely checks the feature exists is the evidence-shape
failure this project has now recorded five times."*

**Arm 2 is the reason this file exists.** Arm 1 establishes what an honest,
causal belief path looks like on the signed-offset readout; Arm 2 substitutes the
smoothed label — the D-03 trap, 08-RESEARCH.md §2.2 Option C, the one-liner a
hurried reader reaches for — and shows Arm 1's own assertion *failing* on it,
inside ``pytest.raises(AssertionError)``. A guard that is only ever run on the
honest path cannot be told apart from a guard that cannot fail. Arm 3 covers the
other success-shaped failure (S-3): an honest filter that collapses into "repeat
the prior state".

**Why S-2 is deliberately not shipped here:** S-2
(``accuracy(argmax(prior_col), y.shift(1)) < 1.0``) was designed to catch a
*trained* prior-state column set to ``y.shift(1)``; under the Bayes filter there
is no training column, so written against the filter's output it would be a test
that can only pass — the exact defect this file exists to avoid. The signed
offset replaces it and is stronger: S-2 cannot tell a smoothed/filtered *blend*
(Option D, agreement strictly below 1.0) from an honest column, whereas the
signed offset reads timing, which a blend inherits from its smoothed part
whenever that part carries the belief over the action threshold before the turn.
(A blend too light ever to cross early is invisible to both; that limit is
stated, not hidden.)

**``honesty/gating.py::assert_causal_features`` gives zero protection here.** It
is a name-suffix scan over ``FORBIDDEN_CENTERED_SUFFIXES`` (``gating.py:33,48``),
so any name passes it trivially. Its presence on ``fit_nowcaster``'s first line
must not create the impression that this recursion is checked. These arms are the
only guard.

**The synthetic world** (no network, no checkpoint, no RNG — every number below is
deterministic). A latent regime path ``z`` over K=3 states, ten runs, nine
reference transitions: ``z`` is the full-sample labeling the reference transitions
are read from. Each month carries an evidence vector ``e_t`` — the class-
conditional likelihood of that month's data: 4.0 on the current state and 1.0
elsewhere in steady months. Three of the nine transitions are *led*: in the three
months before them the data are ambiguous with a mild tilt toward the incoming
state (``e = 1.6`` on it, 1.0 elsewhere). The per-month nowcaster posterior is
``c * e_t`` normalized, where ``c`` is the in-window class prior — a function of
month ``t``'s data only, i.e. strictly causal. The smoothed label is the DP
(Viterbi) decode of the whole evidence sequence under the same ``A``: it places
each switch using evidence from both sides, which is the property that makes the
jump model's labels non-causal within their window (``labeling/jump_model.py:15-18``).

A fourth, non-arm test pins the signed offset's measured **false-positive mode**:
an honest filter with a much stickier ``A`` that never registers a short
intervening regime reads the return to the state it never left as a 14-month
"lead". 08-RESEARCH.md §3 S-1's *"even one strictly negative offset is proof"* is
therefore not true as stated; a negative offset is proof of a lead only when the
belief registered the run it spans.

No arm asserts a target for churn: none is pre-declared (08-CONTEXT.md), and an
arm that asserted one would turn a measurement into a gate after the fact. The
real-data arm (``joint_lift_probs_1_l2.parquet``) is plan 08-08's, which owns the
wiring and therefore the number.

**Plan 08-08's real-data arm** (bottom of this file) runs the signed offset over the
REAL filtered belief path (``joint_lift_belief_{1,2}_l2.parquet``) against the real
full-sample reference labels, and classifies every strictly negative offset with
``classify_negative_offsets`` — the held-through-return rule pre-registered in
08-08-PLAN.md (AMENDED 2026-09-23, before any real 08-08 number existed) and pinned
against this file's synthetic arms in ``test_platform_evaluation_sojourn_lag.py``
first. ``n_lead == 0`` is the gate: one LEAD is proof that post-*t* information
reached the belief. Held-through misses are reported with their positions, never
dropped.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trading_crab_lib.platform.evaluation.churn import read_probability_matrix
from trading_crab_lib.platform.evaluation.sojourn_lag import (
    classify_negative_offsets,
    compute_signed_detection_offsets,
    compute_sojourn_lag_headline,
)
from trading_crab_lib.platform.prediction.regime_filter import (
    filter_step,
    transition_matrix_for,
    unconditional_belief,
)

IDX = [0, 1, 2]
ACT = 0.70
#: (state, run length). Reference transitions are at the start of runs 1..9.
RUNS = [(0, 14), (1, 10), (2, 12), (0, 9), (2, 11), (1, 13), (0, 10), (1, 9), (2, 12), (0, 11)]
#: Runs whose START is preceded by three ambiguous, mildly-leading months.
LED_RUNS = (1, 4, 6)
LEAD_MONTHS = 3
STEADY_EVIDENCE = 4.0
LEAD_EVIDENCE = 1.6


def _world() -> tuple[pd.Series, np.ndarray, list[int], list[int]]:
    """Return (reference labels z, evidence e [T x K], all transition positions, led positions)."""
    z = np.concatenate([[s] * n for s, n in RUNS])
    index = pd.date_range("1975-01-31", periods=len(z), freq="ME")
    evidence = np.ones((len(z), len(IDX)))
    evidence[np.arange(len(z)), z] = STEADY_EVIDENCE
    starts = np.cumsum([0] + [n for _, n in RUNS])[:-1]
    led = []
    for run in LED_RUNS:
        i = int(starts[run])
        incoming = RUNS[run][0]
        evidence[i - LEAD_MONTHS:i] = 1.0
        evidence[i - LEAD_MONTHS:i, incoming] = LEAD_EVIDENCE
        led.append(i)
    return pd.Series(z, index=index, name="state"), evidence, [int(s) for s in starts[1:]], led


def _causal_posteriors(evidence: np.ndarray, class_prior: pd.Series) -> list[pd.Series]:
    """The nowcaster's calibrated posterior for month t: class prior x month-t likelihood."""
    out = []
    for row in evidence:
        unnorm = class_prior.to_numpy() * row
        out.append(pd.Series(unnorm / unnorm.sum(), index=IDX))
    return out


def _run_filter(states: pd.Series, evidence: np.ndarray, a: pd.DataFrame) -> pd.DataFrame:
    """Cold start -> filter_step every month. Returns the belief path, int columns."""
    prior = unconditional_belief(states, state_index=IDX)
    belief = prior.copy()  # cold start: the SAME call as the class prior
    rows = []
    for posterior in _causal_posteriors(evidence, prior):
        belief = filter_step(belief, a, posterior, prior)
        rows.append(belief.to_numpy())
    return pd.DataFrame(rows, index=states.index, columns=IDX)


def _viterbi(evidence: np.ndarray, a: pd.DataFrame, start: pd.Series) -> np.ndarray:
    """MAP state path under A and the per-month likelihoods — a two-sided DP decode."""
    with np.errstate(divide="ignore"):
        log_a = np.log(a.loc[IDX, IDX].to_numpy())
        log_e = np.log(evidence)
        score = np.log(start.to_numpy()) + log_e[0]
    back = np.zeros(evidence.shape, dtype=int)
    for t in range(1, len(evidence)):
        cand = score[:, None] + log_a
        back[t] = cand.argmax(axis=0)
        score = cand.max(axis=0) + log_e[t]
    path = np.zeros(len(evidence), dtype=int)
    path[-1] = int(score.argmax())
    for t in range(len(evidence) - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def _one_hot(path: np.ndarray, index: pd.Index) -> pd.DataFrame:
    return pd.DataFrame({k: (path == k).astype(float) for k in IDX}, index=index)


def _assert_honest_expectation(result: dict) -> None:
    """Arm 1's expectation, as a callable so Arm 2 can show it FAILING.

    Under the window's own A a causal belief does not reach the action threshold
    on the incoming state before the turn on this world, so no offset may be <= 0.
    That is a property of this world, not a theorem: offset 0 is excluded because
    the incoming-state belief measures 0.14-0.63 at the turn (a single decisive
    month would be a legitimate same-month detection, not a leak), and a negative
    offset is excluded only because the filter never clings to a state it should
    have left — see the caveat test at the bottom for an honest filter that does.
    That is why the number Arm 2 discriminates on is n_negative, and why a real
    negative offset must be read against the run it spans.
    """
    assert result["min_offset"] >= 1, result
    assert result["n_zero_or_negative"] == 0, result


# ── Arm 1 — honest ───────────────────────────────────────────────────────────────


def test_arm1_honest_filtered_belief_never_leads_a_transition():
    states, evidence, transitions, led = _world()
    a = transition_matrix_for(states, state_index=IDX)
    belief = _run_filter(states, evidence, a)

    result = compute_signed_detection_offsets(states, belief, act_threshold=ACT)

    assert result["positions"] == transitions
    assert result["n_transitions"] == result["n_resolved"] == 9
    _assert_honest_expectation(result)
    # Measured on this world: the three led turns are detected one month late, the
    # other six two months late. The filter DOES use the ambiguous months (its belief
    # on the incoming state reaches 0.63 / 0.49 / 0.57 at the led turns, against
    # 0.14-0.26 at the others), but it cannot commit before the turn on them.
    by_pos = dict(zip(result["positions"], result["offsets"]))
    assert {p: by_pos[p] for p in led} == dict.fromkeys(led, 1.0)
    assert all(by_pos[p] == 2.0 for p in transitions if p not in led)


# ── Arm 2 — the smoothed substitution: the arm that makes the guard a guard ──────


def test_arm2_smoothed_label_substitution_leads_and_breaks_the_honest_expectation():
    states, evidence, transitions, led = _world()
    a = transition_matrix_for(states, state_index=IDX)
    prior = unconditional_belief(states, state_index=IDX)

    smoothed = _viterbi(evidence, a, prior)          # the two-sided DP decode
    substituted = _one_hot(smoothed, states.index)   # D-03 / Option C: belief := one-hot(smoothed label)

    result = compute_signed_detection_offsets(states, substituted, act_threshold=ACT)

    assert result["n_negative"] >= 1
    assert result["min_offset"] < 0
    # The guard DISCRIMINATES: the very assertion Arm 1 passes fails here.
    with pytest.raises(AssertionError):
        _assert_honest_expectation(result)

    # Where the lead comes from, shown rather than asserted: exactly the three led
    # turns read -LEAD_MONTHS, and every other turn reads 0.
    by_pos = dict(zip(result["positions"], result["offsets"]))
    assert {p: by_pos[p] for p in led} == dict.fromkeys(led, -float(LEAD_MONTHS))
    assert all(by_pos[p] == 0.0 for p in transitions if p not in led)
    # ...and it comes from the FUTURE: decode only the months up to the one before
    # each led turn (the terminal-month view a causal consumer has) and the
    # ambiguous months stay in the outgoing state.
    for i in led:
        truncated = _viterbi(evidence[:i], a, prior)
        assert (truncated[i - LEAD_MONTHS:i] == states.iloc[i - 1]).all(), i


# ── Arm 3 — S-3: honest but useless (collapse into a persistence classifier) ─────


def test_arm3_near_diagonal_transition_matrix_moves_the_readout_the_collapse_way():
    """A near-diagonal A makes the filter lean on "repeat the prior state". That
    would cut churn and look like a triumph; the §5.4 readout must move the way that
    exposes it: median offset UP toward the median sojourn, and the
    median_sojourn / median_lag ratio DOWN toward 1 (today, on real l1only data,
    9.5 / 4.0 = 2.375). Directions only — no threshold is asserted."""
    states, evidence, _, _ = _world()
    a_emp = transition_matrix_for(states, state_index=IDX)
    a_near = pd.DataFrame(0.995 * np.eye(len(IDX)), index=IDX, columns=IDX) + 0.005 * a_emp

    honest = compute_signed_detection_offsets(states, _run_filter(states, evidence, a_emp), act_threshold=ACT)
    sticky_belief = _run_filter(states, evidence, a_near)
    sticky = compute_signed_detection_offsets(states, sticky_belief, act_threshold=ACT)

    # Same transitions, all resolved in both, so the two medians compare like with like.
    assert honest["n_resolved"] == sticky["n_resolved"] == honest["n_transitions"]
    # The collapse is still causal — no strict lead — which is what makes it
    # dangerous: nothing here is a leak for the S-1 guard to catch. (It does reach
    # offset 0 once, a same-month crossing; see the caveat test below for what a
    # stickier A does.)
    assert sticky["n_negative"] == 0

    assert sticky["median_offset"] > honest["median_offset"]

    head_honest = compute_sojourn_lag_headline(states, _run_filter(states, evidence, a_emp), act_threshold=ACT)
    head_sticky = compute_sojourn_lag_headline(states, sticky_belief, act_threshold=ACT)
    assert head_sticky["median_sojourn"] == head_honest["median_sojourn"]
    assert head_sticky["ratio"] < head_honest["ratio"]
    assert abs(head_sticky["ratio"] - 1.0) < abs(head_honest["ratio"] - 1.0)


# ── Caveat, measured: the signed offset's false-positive mode ────────────────────


def test_caveat_an_honest_filter_that_MISSES_a_short_regime_reads_as_a_lead():
    """08-RESEARCH.md §3 S-1 says *"Even one strictly negative offset is proof."*
    Measured here: it is not. An honest, causal filter with a much stickier A (0.999
    on the diagonal) never registers the 9-month state-0 run at positions 36-44, so
    its belief on state 2 stays above threshold straight through it; the return to
    state 2 at position 45 then reads as a 14-month "lead". Nothing leaked — the
    filter simply missed a regime. A real-data n_negative must therefore be read
    against the preceding reference run: a negative offset whose run spans a whole
    reference run the belief never registered is a MISS, not a lead. Pinned so that
    reading is never skipped (plan 08-08 owns the real number)."""
    states, evidence, _, _ = _world()
    a_emp = transition_matrix_for(states, state_index=IDX)
    a_stickier = pd.DataFrame(0.999 * np.eye(len(IDX)), index=IDX, columns=IDX) + 0.001 * a_emp

    belief = _run_filter(states, evidence, a_stickier)
    result = compute_signed_detection_offsets(states, belief, act_threshold=ACT)

    by_pos = dict(zip(result["positions"], result["offsets"]))
    assert by_pos[45] == -14.0
    assert result["n_negative"] == 1
    # The discriminating feature: the intervening reference run (state 0 at 36-44)
    # was never registered by the belief at all.
    assert (states.iloc[36:45] == 0).all()
    assert belief[0].iloc[36:45].max() < ACT


# ── Plan 08-08: the REAL-DATA arm ────────────────────────────────────────────────

_JOINT_LIFT = Path(__file__).resolve().parents[2] / "outputs" / "reports" / "platform" / "joint_lift"


def _real_reference(checkpoint: str) -> pd.Series:
    from trading_crab_lib.platform.checkpoints import get_platform_checkpoint_manager
    from trading_crab_lib.platform.honesty.holdout import DEFAULT_HOLDOUT_CUTOFF, split_by_holdout_boundary

    full = get_platform_checkpoint_manager().load(checkpoint)["state"]
    dev, _ = split_by_holdout_boundary(full.to_frame("state"), cutoff=DEFAULT_HOLDOUT_CUTOFF)
    return dev["state"]


def _real_act_threshold() -> float:
    from trading_crab_lib.platform.config import load_platform_config

    return float(load_platform_config()["allocation"]["hysteresis"]["act_threshold"])


#: Measured 2026-09-23 on the real belief paths (plan 08-08), reported, never dropped.
#: Classifier #1: position 300 (1988-02-29), a return into state 4 after the reference's
#: 4-month state-0 run 1987-10..1988-01 (the October 1987 crash) that the belief held
#: state 4 straight through (belief[4] 0.81-0.96, belief[0] <= 0.01).
_MEASURED_HELD_THROUGH_MISSES = {1: [300], 2: []}


@pytest.mark.parametrize("clf,checkpoint", [(1, "regime_labels"), (2, "regime_labels_2")])
def test_real_data_the_filtered_belief_never_LEADS_a_reference_transition(clf, checkpoint):
    reference = _real_reference(checkpoint)
    belief = read_probability_matrix(_JOINT_LIFT / f"joint_lift_belief_{clf}_l2.parquet")
    assert reference.index.max() <= pd.Timestamp("2020-12-31")  # holdout never read
    assert belief.index.max() <= pd.Timestamp("2020-12-31")
    act = _real_act_threshold()

    offsets = compute_signed_detection_offsets(reference, belief, act_threshold=act)
    classified = classify_negative_offsets(reference, belief, offsets, act)

    assert classified["n_negative"] == offsets["n_negative"]
    assert classified["n_lead"] == 0, (
        f"classifier #{clf}: {classified['n_lead']} strictly negative offset(s) are LEADS under the "
        f"pre-registered held-through-return rule — proof that post-t information reached the belief. "
        f"HALT. Details: {[d for d in classified['details'] if d['verdict'] == 'lead']}"
    )
    assert classified["n_held_through_miss"] + classified["n_lead"] == offsets["n_negative"]
    assert classified["held_through_miss_positions"] == _MEASURED_HELD_THROUGH_MISSES[clf], (
        "a held-through miss appeared or disappeared — report it, never drop it"
    )
