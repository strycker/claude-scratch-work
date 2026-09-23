"""Explicit Bayes filter over the L2 nowcaster's posterior (plan 08-06, ROADMAP criterion 1).

**The recursion.** Design §4.2's forward recursion, verbatim in structure:

    π_t(j)  ∝  [ Σ_i π_{t−1}(i) · A_ij ] · L_t(j)

``A`` is ``prediction/transition_matrix.py::empirical_transition_matrix`` over the
step's own in-window L1 labels, completed to a full K x K by
:func:`transition_matrix_for`. ``L_t(j)`` is the nowcaster's calibrated posterior
divided by the in-window class prior (:func:`likelihood_ratio`) — a Bayes inversion
from a posterior back to a class-conditional likelihood ratio, so the class prior is
not counted twice (once inside the posterior, once again through ``A``). ``π_0`` is
the in-window unconditional class distribution (:func:`unconditional_belief`), and
the same call supplies the class prior: one rule, used identically wherever a
filtered run begins, because a cold-start rule that differed between train and
serve would itself be train/serve skew. There is no training-time analogue of any
of this, so there is no train/serve skew to have, no leakage surface through a
feature column, and no free parameter — no λ, no ε, no smoothing constant, nothing
to register as a trial. Cost: one K x K multiply per step. Design §5.1's first
sentence names the object: *"A discriminative replacement for the HMM filter."*

**The second catch, stated rather than glossed.** ``A`` is estimated from the
*smoothed* in-window L1 labels, and those labels are non-causal within their window
by design (``labeling/jump_model.py:15-18``: *"the DP jointly optimizes over the
full time axis, so a label at month t is influenced by the global fit including
months after t"*). So ``A`` encodes a transition structure that was itself fitted
non-causally inside the window — a weaker cousin of the teacher-forcing problem
(08-RESEARCH.md §2.2 Option C). It is defensible: those same in-window labels are
already the nowcaster's training target, so no new *class* of information enters,
and none of it lies at or after the decision date. It is not nothing, and it is
recorded here rather than in a footnote.

**There is no prior-state feature column, and four research pitfalls are therefore
structurally void.** A reader who arrives expecting one should stop looking:
08-RESEARCH.md §6 Pitfall 1 (``_cv_safe_active_features`` at ``driver.py:166``
drops the newest-starting column first, which a prior-state column would always
be) cannot fire; §6 Pitfall 3 (K prior-state columns are exactly collinear) has
nothing to be collinear; §6 Pitfall 4 (a third concept named "embargo") introduces
nothing; and §2.5's whole ``PurgedEmbargoedKFold`` interaction — the widened
embargo, the meaningless CV-accuracy rise, the calibration-fold mismatch — is moot,
because the CV never sees a recursive column. **§6 Pitfall 6 stays live:** 100 of
588 steps degrade under ``l2`` routing, and a degraded step has no ``L_t``. The
rule for that step is :func:`predict_only_step`, and every number this machinery
produces must be quoted with its degraded count. Likewise
``honesty/gating.py::assert_causal_features`` gives **zero** protection here — it
is a name-suffix scan over ``FORBIDDEN_CENTERED_SUFFIXES`` (``gating.py:33,48``) —
and its presence at the top of ``fit_nowcaster`` must not be read as a check on
this recursion. The guard is ``tests/unit/test_platform_nowcaster_recursion.py``.

**The missing-observation rule.** On a degraded step there is no posterior.
Holding ``π`` unchanged would assert "the world did not move", a different and
unwarranted claim. Advancing by the transition step alone — ``π_{t−1} A``,
normalized — is what a filter does when an observation is missing.

This module does not touch ``prediction/nowcaster.py``: that module's
``build_nowcaster_training_set`` owns the D-01 structural label embargo, and its
docstring says the two embargo concepts there *"must never be merged"*. Wiring the
filter into the drivers is plan 08-08's; the caller composes the two.

Usage::

    from trading_crab_lib.platform.prediction.regime_filter import (
        filter_step, predict_only_step, transition_matrix_for, unconditional_belief,
    )

    idx = list(range(K))
    prior = unconditional_belief(states_1, state_index=idx)   # class prior
    belief = prior.copy()                                     # cold start: the SAME call
    A = transition_matrix_for(states_1, state_index=idx)
    belief = filter_step(belief, A, posterior, prior)         # or predict_only_step(belief, A)
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd

from trading_crab_lib.platform.prediction.transition_matrix import empirical_transition_matrix

log = logging.getLogger(__name__)


def _as_state_list(state_index: Sequence[int]) -> list[int]:
    states = [int(s) for s in state_index]
    if not states:
        raise ValueError("state_index must be non-empty")
    if len(set(states)) != len(states):
        raise ValueError(f"state_index contains duplicates: {states}")
    return states


def unconditional_belief(states: pd.Series, *, state_index: Sequence[int]) -> pd.Series:
    """The in-window class distribution — BOTH the cold start π_0 and the class prior.

    One helper for both roles, by design: a cold-start rule that differs from the
    prior rule is a second rule to keep in step, and a rule that differs between
    train and serve is train/serve skew.

    Args:
        states: the window's L1 labels (integer state ids; NaN rows are ignored).
        state_index: the full canonical state set, e.g. ``range(K)``.

    Returns:
        pd.Series over ``state_index`` summing to 1.0, zero for states the window
        never visited.

    Raises:
        ValueError: if ``states`` has no non-NaN label, or carries a label that is
            not in ``state_index`` (a wiring bug, not a distribution).
    """
    idx = _as_state_list(state_index)
    clean = pd.Series(states).dropna().astype(int)
    if clean.empty:
        raise ValueError("unconditional_belief: states has no non-NaN label")
    unknown = sorted({int(v) for v in clean.unique()} - set(idx))
    if unknown:
        raise ValueError(f"unconditional_belief: labels {unknown} are not in state_index {idx}")
    counts = clean.value_counts().reindex(idx, fill_value=0).astype(float)
    return counts / counts.sum()


def likelihood_ratio(
    posterior: pd.Series,
    class_prior: pd.Series,
    *,
    state_index: Sequence[int],
) -> pd.Series:
    """``L_t(j) = posterior(j) / class_prior(j)``: the posterior inverted back to a likelihood ratio.

    A state absent from ``posterior.index`` gets **1.0** — "no evidence either way".
    ``model.classes_`` can be a strict subset of the canonical states when an early
    training window never saw some state (``driver.py:302-303``); assigning 0.0 to
    such a state would assert evidence *against* it that the nowcaster never produced.

    Raises:
        ValueError: if ``class_prior(j)`` is 0 (or missing) for a state that IS in
            ``posterior`` — a state the window never saw cannot have produced a
            calibrated posterior, so that combination is a wiring bug; also if the
            posterior names a state outside ``state_index``.
    """
    idx = _as_state_list(state_index)
    post = pd.Series(posterior, dtype=float)
    post.index = [int(s) for s in post.index]
    foreign = sorted(set(post.index) - set(idx))
    if foreign:
        raise ValueError(f"likelihood_ratio: posterior states {foreign} are not in state_index {idx}")
    prior = pd.Series(class_prior, dtype=float)
    prior.index = [int(s) for s in prior.index]
    prior = prior.reindex(idx, fill_value=0.0)

    ratio = pd.Series(1.0, index=idx)
    for state in post.index:
        p_prior = float(prior.loc[state])
        if not p_prior > 0.0:
            raise ValueError(
                f"likelihood_ratio: state {state} has class prior {p_prior} but a posterior of "
                f"{float(post.loc[state])} — a state the window never saw cannot have produced a "
                "calibrated posterior. This is a wiring bug (prior and posterior from different windows?)."
            )
        ratio.loc[state] = float(post.loc[state]) / p_prior
    return ratio


def _check_transition_matrix(transition_matrix: pd.DataFrame, idx: list[int]) -> np.ndarray:
    rows = [int(r) for r in transition_matrix.index]
    cols = [int(c) for c in transition_matrix.columns]
    if sorted(rows) != sorted(idx) or sorted(cols) != sorted(idx):
        raise ValueError(
            f"transition_matrix must cover exactly the belief's states {idx} as rows and columns; "
            f"got rows {rows}, columns {cols}. Build it with transition_matrix_for()."
        )
    frame = transition_matrix.copy()
    frame.index, frame.columns = rows, cols
    values = frame.loc[idx, idx].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("transition_matrix contains non-finite entries")
    return values


def _normalize(values: np.ndarray, idx: list[int], *, what: str) -> pd.Series:
    mass = float(values.sum())
    if not (np.isfinite(mass) and mass > 0.0):
        raise ValueError(
            f"{what}: pre-normalization mass is {mass}; refusing to return a NaN belief. "
            "Every state with belief mass received zero likelihood."
        )
    return pd.Series(values / mass, index=idx)


def predict_only_step(prior_belief: pd.Series, transition_matrix: pd.DataFrame) -> pd.Series:
    """``π_{t−1} A``, normalized — the missing-observation rule.

    Used on a degraded step, where the nowcaster produced no posterior. Holding the
    belief would claim the world did not move; a filter with no observation advances
    by the transition step alone.
    """
    idx = [int(s) for s in prior_belief.index]
    a = _check_transition_matrix(transition_matrix, idx)
    predicted = prior_belief.to_numpy(dtype=float) @ a
    return _normalize(predicted, idx, what="predict_only_step")


def filter_step(
    prior_belief: pd.Series,
    transition_matrix: pd.DataFrame,
    posterior: pd.Series,
    class_prior: pd.Series,
) -> pd.Series:
    """One filter step: predict (``π_{t−1} A``), update (× ``L_t``), normalize.

    Pure: no argument is mutated. The returned Series is indexed like
    ``prior_belief``.

    Raises:
        ValueError: if the pre-normalization mass is 0 (never returns NaN), plus
            everything :func:`likelihood_ratio` raises.
    """
    idx = [int(s) for s in prior_belief.index]
    a = _check_transition_matrix(transition_matrix, idx)
    predicted = prior_belief.to_numpy(dtype=float) @ a
    ratio = likelihood_ratio(posterior, class_prior, state_index=idx).to_numpy(dtype=float)
    return _normalize(predicted * ratio, idx, what="filter_step")


def transition_matrix_for(states: pd.Series, *, state_index: Sequence[int]) -> pd.DataFrame:
    """``empirical_transition_matrix(states)`` completed to a full K x K over ``state_index``.

    ``empirical_transition_matrix`` omits the row of any state that never appears as a
    'from' state (the window's final label only, or never visited). Such a row is
    filled with :func:`unconditional_belief` — the maximum-entropy choice consistent
    with the window — and a WARNING names the state. Never a uniform row (that would
    invent transitions to states the window never visited) and never NaN. A 'to'
    column absent from the crosstab is a genuinely observed zero and is filled with 0.

    Raises:
        ValueError: if fewer than two non-NaN labels are given, or a label is not in
            ``state_index``.
    """
    idx = _as_state_list(state_index)
    clean = pd.Series(states).dropna().astype(int)
    if len(clean) < 2:
        raise ValueError("transition_matrix_for: need at least two labels to observe a transition")
    fallback = unconditional_belief(clean, state_index=idx)  # also validates labels

    counts = empirical_transition_matrix(clean)
    counts.index = [int(r) for r in counts.index]
    counts.columns = [int(c) for c in counts.columns]
    matrix = counts.reindex(index=idx, columns=idx)
    observed_rows = set(counts.index)
    for state in idx:
        if state in observed_rows:
            matrix.loc[state] = matrix.loc[state].fillna(0.0)
        else:
            seen = {int(v) for v in clean.unique()}
            reason = "is only the window's final label" if state in seen else "never occurs in the window"
            log.warning(
                "transition_matrix_for: state %d %s, so it has no observed 'from' row; "
                "filling that row with the window's unconditional distribution %s",
                state, reason, fallback.round(6).to_dict(),
            )
            matrix.loc[state] = fallback.to_numpy()
    matrix = matrix.astype(float)
    matrix.index.name, matrix.columns.name = "from", "to"
    return matrix


if __name__ == "__main__":
    # Synthetic self-check — no network, no checkpoint, deterministic (mirrors the
    # transition_matrix.py / gap_lag.py footers). A 12-step run over a hand-built
    # sticky A and a hand-built posterior stream; step 7 is a degraded step and
    # takes the missing-observation rule.
    demo_idx = [0, 1, 2]
    demo_labels = pd.Series([0] * 5 + [1] * 4 + [2] * 3)
    demo_prior = unconditional_belief(demo_labels, state_index=demo_idx)
    demo_a = pd.DataFrame(
        [[0.8, 0.15, 0.05], [0.05, 0.8, 0.15], [0.15, 0.05, 0.8]], index=demo_idx, columns=demo_idx
    )
    demo_posteriors: list[pd.Series | None] = [
        pd.Series({0: 0.7, 1: 0.2, 2: 0.1}),
        pd.Series({0: 0.7, 1: 0.2, 2: 0.1}),
        pd.Series({0: 0.6, 1: 0.3, 2: 0.1}),
        pd.Series({0: 0.4, 1: 0.5, 2: 0.1}),
        pd.Series({0: 0.2, 1: 0.7, 2: 0.1}),
        pd.Series({0: 0.1, 1: 0.8, 2: 0.1}),
        None,  # degraded step: no posterior
        pd.Series({0: 0.1, 1: 0.6, 2: 0.3}),
        pd.Series({0: 0.1, 1: 0.3, 2: 0.6}),
        pd.Series({0: 0.1, 1: 0.2, 2: 0.7}),
        pd.Series({0: 0.1, 2: 0.9}),  # a strict-subset posterior: state 1 gets ratio 1.0
        pd.Series({0: 0.1, 1: 0.1, 2: 0.8}),
    ]
    belief = demo_prior.copy()  # cold start: the same call as the class prior
    for step, post in enumerate(demo_posteriors):
        belief = predict_only_step(belief, demo_a) if post is None else filter_step(belief, demo_a, post, demo_prior)
        tag = "predict-only" if post is None else "filter"
        print(  # noqa: T201 — first-class self-check output
            f"step {step:2d} [{tag:12s}] belief={np.round(belief.to_numpy(), 4)} argmax={int(belief.idxmax())}"
        )
