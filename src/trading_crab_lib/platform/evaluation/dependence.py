"""
Criterion-6 measurement — statistical dependence between classifier #1's and
classifier #2's labelings (07-CONTEXT.md D-15, D-14, D-11; ADR-0002).

**What this measures, in criterion 6's own terms.** Classifier #2 was built to
add an *axis* — a leadership/relative dimension classifier #1's crisis/stress
lean set does not describe. This module measures whether it actually did. High
dependence between the two labelings is a **failure to add an axis** and is
recorded as such; it is never explained away, softened into a partial success,
or reported as "dependent but still informative".

**No pre-declared threshold (D-15).** Adjusted Rand, normalized mutual
information and Cramer's V are reported side by side with the
cross-tabulation, and a human reads them. A threshold was offered at the
wave-2 decision checkpoint and declined on the grounds that it would have no
empirical basis in this project yet. Consequently **nothing in this module
branches on a statistic**: no function returns a pass/fail verdict, raises on
a high value, or filters on one. ``DEPENDENCE_FLAG_ARI`` and its two siblings
are prose flag levels used only to word the rendered report.

**Why three statistics rather than one.** ARI is pairwise and
chance-corrected; NMI is information-theoretic and robust to the two
classifiers having different K (5 and 5 today, but not by construction);
Cramer's V is a classical effect size giving one at-a-glance unit-interval
number. Each can be fooled differently, so they are reported together. A
directional statistic (Theil's U) is deliberately not added.

**Both extremes are wiring suspicions, not findings**
(``07-VALIDATION.md`` § "The 'suspiciously clean' class, extended to
dependence"):

- ARI or NMI at exactly 1.0 almost certainly means both inputs resolved to the
  same labeling — the same failure class ``disagreement.py``'s
  ``suspicious``/``suspicious_reason`` fields already guard for criterion 3.
- All three statistics at exactly 0.0 simultaneously is unusually clean for
  real financial data and warrants the same "confirm this is not an alignment
  bug" scrutiny.

Neither is a gate. Both set ``suspicious`` and a reason, and both are a prompt
to check the wiring before any reading is taken.

**One alignment, never two.** The alignment and the cross-tabulation are
delegated to ``platform/plotting/regime.py::label_disagreement`` — exactly as
``evaluation/disagreement.py`` does — so criterion 3's disagreement number and
criterion 6's dependence numbers are computed over the same common index by
construction. A second alignment implementation is precisely the divergence
criterion 1 exists to prevent.

**The silent-zero trap, carried over from ``disagreement.py``.**
``label_disagreement`` coerces its inputs with
``pd.to_numeric(..., errors="coerce")`` and then drops NaNs, so handing it raw
``"state_3"`` STRING labels empties the frame and returns
``{"n_compared": 0, "pct_disagree": 0.0}`` with no exception — which reads as
"totally resolved" while having compared nothing. This module distinguishes
the two causes of a zero overlap and warns differently for each: a genuinely
disjoint span, versus a full index overlap that the coercion emptied.

Usage::

    from trading_crab_lib.platform.checkpoints import get_platform_checkpoint_manager
    from trading_crab_lib.platform.evaluation.dependence import (
        format_dependence_report, measure_labeling_dependence,
    )

    cm = get_platform_checkpoint_manager()
    s1 = cm.load("regime_labels")["state"]
    s2 = cm.load("regime_labels_2")["state"]
    print(format_dependence_report(measure_labeling_dependence(s1, s2)))
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats.contingency import association
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from trading_crab_lib.platform.plotting.regime import label_disagreement

log = logging.getLogger(__name__)

# ── prose flag levels — NOT thresholds (D-15) ─────────────────────────────
#
# These three constants are the levels at which ``format_dependence_report``
# WORDS a statistic as flagged in the human-readable record. They are not
# thresholds: nothing branches on them to change a returned value, nothing
# raises when one is exceeded, nothing is filtered or selected by them, and
# no caller may treat an exceeded level as a failed gate. D-15 declined a
# pass/fail line for criterion 6 because it would have no empirical basis in
# this project yet; the judgement is a human's and it is recorded.
#
# All three are [ASSUMED] per ``07-VALIDATION.md`` § "Plausibility Bands" —
# they are not derived from anything this project has measured. A test
# (``test_flag_constants_are_referenced_only_inside_the_report_formatter``)
# asserts they appear in no other function in this module.
DEPENDENCE_FLAG_ARI: float = 0.7
DEPENDENCE_FLAG_NMI: float = 0.5
DEPENDENCE_FLAG_CRAMERS_V: float = 0.5

_SUSPICIOUS_REASON_PERFECT = (
    "adjusted_rand and/or nmi is EXACTLY 1.0. Both statistics are "
    "permutation-invariant, so an exact 1.0 means the two inputs describe the "
    "identical partition of the identical months. The most likely cause is a "
    "wiring bug — both classifiers reading the same underlying labels (e.g. "
    "the same checkpoint loaded twice, or classifier #2's fit silently "
    "persisting over classifier #1's) — not a decisive finding about the "
    "market. This is the same failure class disagreement.py's "
    "suspicious_reason already guards for criterion 3. Confirm the wiring "
    "before reading this as dependence. This flag is advisory (D-15) — it "
    "does not gate anything."
)

_SUSPICIOUS_REASON_ALL_ZERO = (
    "adjusted_rand, nmi and cramers_v are ALL exactly 0.0 simultaneously. "
    "That is unusually clean for real financial data: two labelings of the "
    "same five decades of macro history would be expected to share at least "
    "some structure by accident. The most likely cause is an alignment bug — "
    "a degenerate labeling (one classifier collapsed to a single state), or "
    "two series aligned onto an index that carries no shared information. "
    "Confirm this is not an alignment bug before reading it as independence. "
    "This flag is advisory (D-15) — it does not gate anything."
)

_ZERO_OVERLAP_DISJOINT = (
    "measure_labeling_dependence: n_compared == 0 — the two labelings' date "
    "indexes do not overlap. A disjoint span is a FINDING, never an error, "
    "and must never be reported as agreement or as independence: all three "
    "statistics are returned as NaN rather than 0.0, because 0.0 would read "
    "as 'the two labelings are independent' when in truth nothing was "
    "compared."
)

_ZERO_OVERLAP_COERCION = (
    "measure_labeling_dependence: n_compared == 0 even though the two "
    "labelings share %d date(s). The indexes overlap, so this is NOT a "
    "disjoint span — it is the coercion trap 07-PREFIX-EVIDENCE.md "
    "documents: label_disagreement runs pd.to_numeric(..., errors='coerce') "
    "and then dropna(), so raw non-numeric labels such as the 'state_3' "
    "STRINGS persisted in the parquet artifacts are silently NaN-ed out and "
    "dropped. Strip the 'state_' prefix and cast to int before calling. All "
    "three statistics are returned as NaN, never 0.0."
)


def _nan_result(base: dict[str, Any]) -> dict[str, Any]:
    """The zero-overlap return shape: NaN statistics, never zeros."""
    return {
        **base,
        "adjusted_rand": float("nan"),
        "nmi": float("nan"),
        "cramers_v": float("nan"),
        "suspicious": False,
        "suspicious_reason": "",
    }


def _cramers_v(a: np.ndarray, b: np.ndarray) -> float:
    """Cramer's V over the two aligned labelings' contingency table.

    ``scipy.stats.contingency.association`` divides by
    ``min(n_rows - 1, n_cols - 1)``, which is zero when either labeling takes
    a single value — it returns NaN there with a RuntimeWarning. That
    degenerate case is handled explicitly and reported as **0.0**, not NaN,
    for a deliberate reason: a labeling collapsed to one state carries no
    association with anything, and reporting 0.0 lets the all-three-zero
    suspicion fire on exactly the wiring bug that produced it. A NaN would
    propagate silently past that check.
    """
    table = pd.crosstab(a, b).to_numpy()
    if table.shape[0] < 2 or table.shape[1] < 2:
        log.warning(
            "measure_labeling_dependence: the contingency table is %dx%d — at least "
            "one labeling takes a single value over the common window. Cramer's V is "
            "undefined there and is reported as 0.0; a labeling collapsed to one state "
            "is itself the wiring signature the all-zero suspicion flags.",
            table.shape[0], table.shape[1],
        )
        return 0.0
    return float(association(table, method="cramer"))


def measure_labeling_dependence(states_1: pd.Series, states_2: pd.Series) -> dict[str, Any]:
    """Criterion 6's dependence measurement — three statistics, one crosstab, no gate.

    Delegates alignment and the cross-tabulation to
    ``plotting/regime.py::label_disagreement`` so this measurement and
    criterion 3's disagreement measurement share one alignment implementation
    and stay comparable by construction.

    Args:
        states_1: classifier #1's labeling (the ``regime_labels`` checkpoint's
            ``state`` column), indexed by month-end decision date.
        states_2: classifier #2's labeling (the ``regime_labels_2``
            checkpoint's ``state`` column), same index convention. D-11 froze
            both at the same 1972+ decision window so they are month-for-month
            comparable.

    Returns:
        dict — every key ``label_disagreement`` returns (``n_compared``,
        ``n_disagree``, ``pct_disagree``, ``first_common_date``,
        ``last_common_date``, ``per_state_confusion``), plus
        ``adjusted_rand``, ``nmi``, ``cramers_v``, ``suspicious`` (bool) and
        ``suspicious_reason`` (str, empty when not suspicious). A caller can
        never read a statistic without its ``n_compared`` denominator sitting
        beside it in the same mapping.

        On a zero overlap all three statistics are ``float("nan")`` — never
        ``0.0``, which would read as independence — and a WARNING names which
        of the two causes applies.

    Never raises on a high statistic and never returns a verdict (D-15).
    """
    base: dict[str, Any] = dict(label_disagreement(states_1, states_2))

    if base["n_compared"] == 0:
        shared = 0
        if states_1 is not None and states_2 is not None and len(states_1) and len(states_2):
            shared = len(states_1.index.intersection(states_2.index))
        if shared > 0:
            log.warning(_ZERO_OVERLAP_COERCION, shared)
        else:
            log.warning(_ZERO_OVERLAP_DISJOINT)
        return _nan_result(base)

    common = states_1.index.intersection(states_2.index)
    aligned = pd.DataFrame(
        {
            "a": pd.to_numeric(states_1.loc[common], errors="coerce"),
            "b": pd.to_numeric(states_2.loc[common], errors="coerce"),
        }
    ).dropna().astype(int)
    a = aligned["a"].to_numpy()
    b = aligned["b"].to_numpy()

    result: dict[str, Any] = {
        **base,
        "adjusted_rand": float(adjusted_rand_score(a, b)),
        "nmi": float(normalized_mutual_info_score(a, b)),
        "cramers_v": _cramers_v(a, b),
    }

    perfect = result["adjusted_rand"] == 1.0 or result["nmi"] == 1.0
    all_zero = result["adjusted_rand"] == 0.0 and result["nmi"] == 0.0 and result["cramers_v"] == 0.0

    reasons = []
    if perfect:
        reasons.append(_SUSPICIOUS_REASON_PERFECT)
    if all_zero:
        reasons.append(_SUSPICIOUS_REASON_ALL_ZERO)

    result["suspicious"] = bool(reasons)
    result["suspicious_reason"] = " ".join(reasons)
    if reasons:
        log.warning(
            "measure_labeling_dependence: SUSPICIOUS — adjusted_rand=%r nmi=%r "
            "cramers_v=%r over n_compared=%d. %s",
            result["adjusted_rand"], result["nmi"], result["cramers_v"],
            result["n_compared"], result["suspicious_reason"],
        )
    return result


def _flag_prose(value: float, level: float) -> str:
    """Prose flag status for one statistic. Renders text; decides nothing."""
    if value != value:  # NaN
        return "not computed (nothing was compared)"
    if value > level:
        return (
            f"ABOVE the {level} flag level — this indicates a FAILURE TO ADD AN AXIS "
            "(criterion 6)"
        )
    return f"at or below the {level} flag level"


def format_dependence_report(result: dict[str, Any]) -> str:
    """Render *result* as the human-readable criterion-6 record.

    Every statistic is rendered on one line together with its ``n_compared``
    denominator and its first/last common date — the binding condition wave
    1's UAT attached to criterion 3, carried forward here so a reader cannot
    see a number without seeing what it was computed over.

    When a flag level is exceeded the rendered text states that this
    indicates a failure to add an axis, in those words, and offers **no**
    mitigating explanation: criterion 6 says such a result is recorded as a
    failure, and the recording is the deliverable.

    The flag levels are prose only (D-15). This function words them; it does
    not gate on them, and its caller must not either.
    """
    n = result["n_compared"]
    first = result.get("first_common_date")
    last = result.get("last_common_date")
    window = (
        f"{pd.Timestamp(first).date()}..{pd.Timestamp(last).date()}"
        if first is not None and last is not None
        else "no common window"
    )
    denom = f"n_compared={n}, window {window}"

    lines = [
        "Labeling dependence — classifier #1 vs classifier #2 (criterion 6, D-15)",
        "",
        f"  Adjusted Rand        : {result['adjusted_rand']!r}  [{denom}]  "
        f"{_flag_prose(result['adjusted_rand'], DEPENDENCE_FLAG_ARI)}",
        f"  Normalized mutual info: {result['nmi']!r}  [{denom}]  "
        f"{_flag_prose(result['nmi'], DEPENDENCE_FLAG_NMI)}",
        f"  Cramer's V           : {result['cramers_v']!r}  [{denom}]  "
        f"{_flag_prose(result['cramers_v'], DEPENDENCE_FLAG_CRAMERS_V)}",
        "",
        "  No pass/fail threshold is declared or applied (D-15): these levels word the",
        "  report, they gate nothing. A human reads the numbers and the judgement is",
        "  recorded.",
        "",
    ]

    if n == 0:
        lines.append(
            "  n_compared == 0: nothing was compared. All three statistics are NaN, "
            "not 0.0 —\n  a zero would read as independence. This is a finding about "
            "the two inputs' spans\n  or their dtypes, never a result about the market."
        )
        return "\n".join(lines)

    if result.get("suspicious"):
        lines += [
            "  SUSPICION FLAGGED (advisory, not a gate):",
            f"    {result['suspicious_reason']}",
            "",
        ]

    lines += ["  Cross-tabulation (classifier #1 state x classifier #2 state):", ""]
    ct = result["per_state_confusion"]
    if isinstance(ct, pd.DataFrame) and not ct.empty:
        with_margins = ct.copy()
        with_margins["ALL"] = ct.sum(axis=1)
        with_margins.loc["ALL"] = with_margins.sum(axis=0)
        lines += ["    " + line for line in with_margins.to_string().splitlines()]
    else:
        lines.append("    (empty)")
    lines += ["", f"  Disagreement over the same alignment: pct_disagree="
              f"{result['pct_disagree']!r} ({result['n_disagree']}/{n}). "
              "Reported for reference only;\n  the two classifiers number their states "
              "independently, so a raw label mismatch is not\n  itself a dependence "
              "statistic."]
    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Synthetic self-check — no network, no checkpoint dependency (mirrors
    # disagreement.py's __main__ footer). The six-element 3x3 oracle.
    demo_index = pd.date_range("1972-01-31", periods=6, freq="ME")
    demo_1 = pd.Series([0, 0, 1, 1, 2, 2], index=demo_index, name="state")
    demo_2 = pd.Series([0, 0, 1, 2, 1, 2], index=demo_index, name="state")

    print(format_dependence_report(measure_labeling_dependence(demo_1, demo_2)))  # noqa: T201


# ── Block-permutation null (pre-registered control, 07-09 Task 3) ────────────
#
# ARI / NMI / Cramer's V assume exchangeable observations. Regime labelings are
# step functions on a shared time axis: classifier #1 has 7 contiguous blocks
# over 695 months, #2 has 13. Two block partitions of one timeline are
# associated BEFORE any shared economics, purely from temporal contiguity, so
# the raw statistics cannot separate "both see the same market structure" from
# "both are slow".
#
# This null holds each labeling's block-length multiset and occupancy EXACTLY
# and randomises only the ARRANGEMENT of blocks, which destroys alignment
# between the two labelings while preserving everything else. The observed
# statistic is then read against that distribution.
#
# The decision rule is PRE-REGISTERED in 07-DEPENDENCE.md and committed before
# this code existed. Nothing here branches on the outcome.


def _blocks(states: np.ndarray) -> list[tuple[int, int]]:
    """Decompose a label sequence into (state, run_length) contiguous blocks."""
    out: list[tuple[int, int]] = []
    cur = states[0]
    n = 0
    for v in states:
        if v != cur:
            out.append((int(cur), n))
            cur = v
            n = 1
        else:
            n += 1
    out.append((int(cur), n))
    return out


def _shuffle_blocks(states: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Randomly re-order a sequence's own blocks. Occupancy is preserved EXACTLY.

    Known and deliberate imprecision: when the shuffle places two blocks of the
    same state adjacent, they merge, so the shuffled series can carry fewer and
    longer runs than the original. That makes each resample BLOCKIER than the
    input, which raises association by chance and shifts the null UP. The bias
    is therefore conservative in a specific direction: it makes clearing the
    null harder, so it can only push a verdict toward (b) "failure to add an
    axis" and never manufacture an (a). Recorded rather than corrected, because
    correcting it (rejection-sampling arrangements with no same-state
    adjacency) would bias the arrangement distribution itself.
    """
    blocks = _blocks(states)
    order = rng.permutation(len(blocks))
    return np.concatenate([np.full(blocks[i][1], blocks[i][0]) for i in order])


def block_permutation_null(
    states_1: pd.Series,
    states_2: pd.Series,
    *,
    n_resamples: int = 2000,
    random_state: int = 20260918,
) -> dict[str, Any]:
    """Null distribution of the three dependence statistics under block shuffling.

    Both labelings are block-shuffled independently each resample. Returns the
    observed values, the null percentiles used by the pre-registered rule, and
    the raw null arrays for plotting.

    This function makes NO verdict. It reports a distribution; the rule that
    reads it lives in 07-DEPENDENCE.md and predates this code.
    """
    observed = measure_labeling_dependence(states_1, states_2)
    common = states_1.index.intersection(states_2.index)
    a = states_1.loc[common].to_numpy()
    b = states_2.loc[common].to_numpy()

    rng = np.random.default_rng(random_state)
    null = {
        "adjusted_rand": np.empty(n_resamples),
        "nmi": np.empty(n_resamples),
        "cramers_v": np.empty(n_resamples),
    }
    for i in range(n_resamples):
        sa = _shuffle_blocks(a, rng)
        sb = _shuffle_blocks(b, rng)
        null["adjusted_rand"][i] = adjusted_rand_score(sa, sb)
        null["nmi"][i] = normalized_mutual_info_score(sa, sb)
        null["cramers_v"][i] = _cramers_v(sa, sb)

    pct = {
        k: {
            "p50": float(np.percentile(v, 50)),
            "p95": float(np.percentile(v, 95)),
            "p99": float(np.percentile(v, 99)),
            "max": float(v.max()),
            "observed_percentile": float((v < observed[k]).mean() * 100.0),
        }
        for k, v in null.items()
    }
    return {
        "observed": {k: observed[k] for k in ("adjusted_rand", "nmi", "cramers_v")},
        "n_compared": observed["n_compared"],
        "n_resamples": n_resamples,
        "random_state": random_state,
        "n_blocks_1": len(_blocks(a)),
        "n_blocks_2": len(_blocks(b)),
        "null": pct,
        "_raw": null,
    }
