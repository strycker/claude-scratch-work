"""
Criterion-3 measurement — post-fix labeling disagreement (07-CONTEXT.md D-05,
07-VALIDATION.md criterion 3).

``measure_label_disagreement()`` is a thin wrapper, never a reimplementation,
around ``platform/plotting/regime.py::label_disagreement`` — the LOCATED
provenance of the 389/470 = 82.8% pre-fix baseline
(``.planning/phases/07-regime-representation/07-PREFIX-EVIDENCE.md``).
Delegation is mandatory: a second implementation of the same comparison would
make the post-fix number methodologically incomparable to the baseline,
which is the exact risk criterion 3 exists to guard against.

The two persisted artifacts this module accepts are stored in their raw,
on-disk shapes (``evaluation/report.py`` steps (h) / lines 754-766):

- ``full_sample_states`` — ``backtest_full_sample_states.parquet``, a
  one-column ``state`` frame (or, for callers that already extracted it, a
  bare Series) holding the ONE full-sample smoothed jump-model fit.
- ``filtered_state_probs`` — ``backtest_filtered_state_probs.parquet``, the
  walk-forward's per-step multiclass probability matrix with columns renamed
  to ``state_{k}`` STRINGS (parquet column names must be strings).

**The silent-zero trap (independently reproduced at plan time,
``07-PREFIX-EVIDENCE.md`` section 1):** passing ``filtered_state_probs``'s
raw ``idxmax(axis=1)`` output (the STRING column label, e.g. ``"state_3"``)
straight into ``label_disagreement`` makes its internal
``pd.to_numeric(..., errors="coerce")`` coerce every value to NaN; the
subsequent ``dropna()`` then empties the frame and the function returns
``{"n_compared": 0, ...}`` SILENTLY — no exception. This reads as "0%
disagreement, fully resolved" while having compared nothing. Stripping the
``state_`` prefix and casting to int (``_comparison_from_state_probs``
below) is what this module exists to make load-bearing rather than a trap a
caller has to remember.

**A near-zero disagreement is itself suspicious (D-07, advisory only, never
a gate).** A hindsight full-sample fit and a per-step walk-forward fit
SHOULD still disagree somewhat even under one shared feature space, because
one sees the whole history and the other does not. Near-zero disagreement
most plausibly means the two paths are reading the same fitted object rather
than genuinely converging. ``suspicious`` / ``suspicious_reason`` surface
this as a flag on the result — they never gate, filter, or raise (D-04: no
measured number here is used to select anything).

Usage::

    from trading_crab_lib.platform.evaluation.disagreement import measure_label_disagreement

    full = pd.read_parquet("outputs/reports/platform/backtest_full_sample_states.parquet")
    probs = pd.read_parquet("outputs/reports/platform/backtest_filtered_state_probs.parquet")
    result = measure_label_disagreement(full, probs)
    assert result["n_compared"] > 0, "a percentage without its denominator is not a result"
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from trading_crab_lib.platform.plotting.regime import label_disagreement

log = logging.getLogger(__name__)

DEFAULT_SUSPICIOUS_THRESHOLD = 0.02

_SUSPICIOUS_REASON = (
    "pct_disagree is below the suspicious_threshold with n_compared > 0. A "
    "hindsight full-sample fit and a per-step walk-forward fit should still "
    "disagree somewhat even under one feature space, because one sees the "
    "whole history and the other does not. Near-zero disagreement most "
    "likely means the walk-forward path and the hindsight path are reading "
    "the same fitted object rather than genuinely converging. This flag is "
    "advisory (D-07) — it does not gate anything."
)


def _as_reference_series(full_sample_states: pd.DataFrame | pd.Series) -> pd.Series:
    """Normalize the persisted ``full_sample_states`` artifact to a Series.

    The on-disk artifact is a one-column ``state`` frame
    (``report.py``:754); a caller may also hand this a bare Series directly.
    """
    if isinstance(full_sample_states, pd.DataFrame):
        if "state" in full_sample_states.columns:
            return full_sample_states["state"]
        if full_sample_states.shape[1] == 1:
            return full_sample_states.iloc[:, 0]
        raise ValueError(
            "full_sample_states DataFrame has no 'state' column and is not "
            f"single-column (columns={list(full_sample_states.columns)}) — "
            "cannot unambiguously derive the reference labeling."
        )
    return full_sample_states


def _comparison_from_state_probs(filtered_state_probs: pd.DataFrame) -> pd.Series:
    """Derive the row-wise argmax comparison labeling from the persisted
    ``state_{k}``-string-columned probability matrix — the load-bearing
    coercion this module exists for (see module docstring, silent-zero trap).
    """
    if filtered_state_probs is None or filtered_state_probs.empty:
        return pd.Series(dtype="int64")

    argmax_labels = filtered_state_probs.idxmax(axis=1)
    # Columns are persisted as "state_{k}" strings (report.py:755-757). Strip
    # the prefix and cast to int BEFORE label_disagreement ever sees it —
    # label_disagreement's own pd.to_numeric(..., errors="coerce") would
    # silently NaN-out (and then drop) a raw "state_3" string, returning
    # n_compared == 0 with no error.
    stripped = argmax_labels.astype(str).str.replace("state_", "", regex=False)
    return stripped.astype(int)


def measure_label_disagreement(
    full_sample_states: pd.DataFrame | pd.Series,
    filtered_state_probs: pd.DataFrame,
    *,
    suspicious_threshold: float = DEFAULT_SUSPICIOUS_THRESHOLD,
) -> dict[str, Any]:
    """Criterion-3's disagreement measurement, delegating to the located
    ``label_disagreement`` methodology so the result is comparable to the
    389/470 = 82.8% pre-fix baseline by construction.

    Args:
        full_sample_states: the persisted ``backtest_full_sample_states``
            artifact (one-column ``state`` frame) or an equivalent Series —
            the ONE full-sample smoothed reference labeling.
        filtered_state_probs: the persisted ``backtest_filtered_state_probs``
            artifact — the walk-forward's per-step multiclass probability
            matrix, columns ``state_{k}`` strings.
        suspicious_threshold: below this ``pct_disagree`` (with
            ``n_compared > 0``), the result is flagged ``suspicious``
            (default 0.02, ``07-VALIDATION.md``'s ``[ASSUMED]`` band).

    Returns:
        dict — every key ``label_disagreement`` returns (``n_compared``,
        ``n_disagree``, ``pct_disagree``, ``first_common_date``,
        ``last_common_date``, ``per_state_confusion``), plus ``suspicious``
        (bool) and ``suspicious_reason`` (str, empty when not suspicious). A
        caller can never read ``pct_disagree`` without its ``n_compared``
        denominator sitting right beside it in the same mapping.
    """
    reference = _as_reference_series(full_sample_states)
    comparison = _comparison_from_state_probs(filtered_state_probs)

    result: dict[str, Any] = dict(label_disagreement(reference, comparison))

    if result["n_compared"] == 0:
        log.warning(
            "measure_label_disagreement: n_compared == 0 — the two labelings' "
            "date indexes do not overlap. This is a disjoint-span finding, "
            "never an error, and must never be reported as pct_disagree == "
            "0.0 meaning perfect agreement."
        )
        result["suspicious"] = False
        result["suspicious_reason"] = ""
        return result

    suspicious = bool(result["pct_disagree"] < suspicious_threshold)
    result["suspicious"] = suspicious
    result["suspicious_reason"] = _SUSPICIOUS_REASON if suspicious else ""
    if suspicious:
        log.warning(
            "measure_label_disagreement: SUSPICIOUS — pct_disagree=%.6f is "
            "below suspicious_threshold=%.4f with n_compared=%d. %s",
            result["pct_disagree"], suspicious_threshold, result["n_compared"], _SUSPICIOUS_REASON,
        )
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Synthetic self-check — no network, no checkpoint dependency (mirrors
    # sojourn_lag.py's __main__ footer). Reference and comparison are built
    # independently: reference disagrees with the probs-derived comparison on
    # a known fraction of months.
    demo_index = pd.date_range("1972-01-31", periods=20, freq="ME")
    demo_reference = pd.Series([0] * 10 + [1] * 10, index=demo_index, name="state")

    # Comparison predicts state 0 for the first 15 months and state 1 for the
    # last 5. Reference is state 0 for months 0-9 and state 1 for 10-19, so
    # months 10-14 disagree (reference=1, comparison=0) and 15-19 agree ->
    # pct_disagree == 5/20 == 0.25.
    demo_probs = pd.DataFrame(
        {
            "state_0": [0.9] * 15 + [0.1] * 5,
            "state_1": [0.1] * 15 + [0.9] * 5,
        },
        index=demo_index,
    )

    demo_result = measure_label_disagreement(demo_reference, demo_probs)

    print(  # noqa: T201 — first-class self-check output
        "Label Disagreement self-check (criterion 3)\n"
        f"  n_compared:  {demo_result['n_compared']}\n"
        f"  n_disagree:  {demo_result['n_disagree']}\n"
        f"  pct_disagree: {demo_result['pct_disagree']}\n"
        f"  suspicious:  {demo_result['suspicious']}"
    )
