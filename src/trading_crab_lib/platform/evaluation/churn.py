"""Two churn series, one module, and no way to mistake one for the other (plan 08-01).

**There are two filtered "churn" quantities in this system and they are not
interchangeable.** Every function below belongs to exactly one of them and says
which:

- **Track A — the L1 jump model's terminal-month label.** ``joint_driver.py:502``
  records ``state_1 = states_1.iloc[-1]``: the label the DP assigns to the last
  month of the training window, re-read at every walk-forward step. This is the
  series the recorded 246/587 = 41.91% describes. :func:`state_change_count` and
  :func:`churn_rate` serve it (and serve Track B's arithmetic too — the *count*
  and the *rate* are the same operations; only the series differs).
- **Track B — the argmax of the L2 nowcaster's calibrated posterior.** This is
  the object design §5.1 changes, and until this plan it was never persisted and
  therefore never measured. :func:`write_probability_matrix`,
  :func:`read_probability_matrix` and :func:`argmax_churn` serve it.

Why the separation is structural rather than a naming convention: under the
decision-bearing ``ROUTING_L1_ONLY``, ``joint_driver.py:431`` sets
``probs_1 = _last_state_one_hot(states_1)`` (the one-hot is constructed at
``joint_driver.py:241-252``) and ``_refit_l2`` is never called. The two tracks
are then elementwise IDENTICAL — not similar, identical — so a measurement of
Track A taken after a change to L2 reports the same number whether the change
worked perfectly or not at all. That degeneracy is real and is pinned as such in
``scripts/joint_lift_diagnostics.py``'s ``series_identity`` block. It is not a
reason to collapse the two into one function; it is the reason not to.

Two denominators, stated once here. A churn *count* is over adjacent PAIRS, so a
588-row window carries 587 pairs and :func:`churn_rate` divides by ``n_rows - 1``.
Dividing by rows is F-4's off-by-one, which recorded 246/588 = 41.84% where
246/587 = 41.91% is the rate.

Usage::

    from trading_crab_lib.platform.evaluation.churn import (
        argmax_churn, read_probability_matrix, write_probability_matrix,
    )

    info = write_probability_matrix(meta["per_step_metrics_1"], path)
    track_b = argmax_churn(read_probability_matrix(path))
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd

from trading_crab_lib.platform.evaluation.sojourn_lag import build_filtered_probs_matrix

log = logging.getLogger(__name__)

#: The on-disk column convention, matching ``evaluation/report.py:1038``'s
#: ``filtered_state_probs`` artifact exactly. Parquet column names must be
#: strings; canonical states are integers. One convention, two representations.
_STATE_COLUMN_RE = re.compile(r"^state_(-?\d+)$")


def state_change_count(series: pd.Series) -> int:
    """Adjacent-pair state changes over a series' own non-null run.

    **One definition of "a state change" for this phase**, used by Track A and
    Track B alike so that a difference between the two numbers is a difference
    between the two *series* and never between two counting conventions.

    NaN handling follows ``scripts/run_joint_lift.py::_n_transitions``: drop
    nulls first, then compare adjacent survivors. ``[0, NaN, 0]`` is therefore
    one run with zero changes, not two changes through a gap.

    Args:
        series: any state-valued series (Track A's ``state_{k}`` column, or
            Track B's ``idxmax`` over a probability matrix).

    Returns:
        int: the number of adjacent pairs whose values differ. ``0`` for an
        empty, all-null or single-valued series — each of which genuinely has
        no differing pair.
    """
    clean = series.dropna()
    if len(clean) == 0:
        return 0
    return int((clean != clean.shift()).sum() - 1)


def churn_rate(n_changes: int, n_rows: int) -> float:
    """Changes per adjacent PAIR — ``n_changes / (n_rows - 1)``.

    A window of ``n_rows`` months contains ``n_rows - 1`` adjacent pairs, and a
    change is a property of a pair. Dividing by ``n_rows`` is F-4's recorded
    off-by-one (246/588 = 0.418367 where 246/587 = 0.418980 is the rate).

    Args:
        n_changes: the count from :func:`state_change_count`.
        n_rows: the number of rows the count was taken over.

    Returns:
        float: the pair-denominated rate.

    Raises:
        ValueError: if ``n_rows < 2``. A rate over fewer than two rows is not a
            rate, and returning ``0.0`` there would be a number where no
            quantity exists — the shape this project has been misled by before.
    """
    if n_rows < 2:
        raise ValueError(
            f"churn_rate needs at least two rows to form one adjacent pair; got n_rows={n_rows}. "
            "Returning 0.0 here would report a rate where none is defined."
        )
    return n_changes / (n_rows - 1)


def write_probability_matrix(per_step_metrics: dict, path: str | Path) -> dict[str, Any]:
    """Persist Track B's per-step probability matrix (ROADMAP criterion 0).

    Delegates the stacking to
    ``sojourn_lag.build_filtered_probs_matrix`` — union-of-classes
    reconciliation and zero-padding included — and does exactly one further
    thing: renames the integer state columns to ``state_{k}`` strings, because
    parquet column names must be strings. **No second stacker exists**; a
    divergent second implementation of this shape is the class ADR-0002's D-11
    was written to prevent.

    The driver accumulates ``per_step_metrics`` only on NON-degraded steps
    (``joint_driver.py:508-510``), so the row count is ``n_steps - n_degraded``,
    not ``n_steps``. Callers should log that difference: a churn number quoted
    without its degraded count is not quotable.

    Args:
        per_step_metrics: the driver's bucket — ``{"dates", "proba", "classes"}``.
        path: destination parquet path; parent directories are created.

    Returns:
        dict with ``path`` (str), ``n_rows`` (int) and ``states`` (the canonical
        INTEGER state ids, ascending).
    """
    matrix = build_filtered_probs_matrix(per_step_metrics)
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    on_disk = matrix.rename(columns={col: f"state_{col}" for col in matrix.columns})
    on_disk.to_parquet(out_path)
    return {
        "path": str(out_path),
        "n_rows": int(len(matrix)),
        "states": [int(col) for col in matrix.columns],
    }


def read_probability_matrix(path: str | Path) -> pd.DataFrame:
    """Read a persisted Track B matrix back with **integer** columns restored.

    The inverse of :func:`write_probability_matrix`'s one rename, and the reason
    the round trip is closed rather than one-way:
    ``sojourn_lag.compute_sojourn_lag_headline`` RAISES on non-integer columns
    (T0.12) precisely because a ``state_{k}``-string matrix once scored every
    transition unresolved and returned ``n_resolved = 0`` that read as the
    finding "real-time detection never happened". Handing that function a frame
    straight off disk must not be possible.

    Args:
        path: a parquet written by :func:`write_probability_matrix`.

    Returns:
        pd.DataFrame indexed by decision date with an int64 column index.

    Raises:
        ValueError: naming the offending column, if any column does not parse as
            ``state_{int}``. Never silently passes a string-keyed frame onward.
    """
    frame = pd.read_parquet(Path(path))
    parsed: list[int] = []
    for col in frame.columns:
        match = _STATE_COLUMN_RE.match(str(col)) if not isinstance(col, bool) else None
        if match is None:
            raise ValueError(
                f"read_probability_matrix({path}): column {col!r} does not parse as "
                "'state_{int}'. The persisted matrix must carry canonical integer state "
                "ids behind the string prefix; a frame keyed any other way scores every "
                "transition unresolved inside compute_sojourn_lag_headline and returns "
                "n_resolved=0, which reads as a finding rather than as a shape error."
            )
        parsed.append(int(match.group(1)))

    restored = frame.copy()
    restored.columns = pd.Index(parsed, dtype="int64")
    return restored


def argmax_churn(probs_matrix: pd.DataFrame) -> dict[str, Any]:
    """**Track B's churn**: how often ``argmax`` of the posterior moves.

    This is the series design §5.1 can actually move, and the one that had never
    been measured anywhere before this plan. It is NOT Track A: see this module's
    docstring for why they coincide under ``ROUTING_L1_ONLY`` and why that
    coincidence is a degeneracy to be recorded rather than a licence to merge
    them.

    Args:
        probs_matrix: an integer-column probability matrix, as returned by
            :func:`read_probability_matrix` or ``build_filtered_probs_matrix``.

    Returns:
        dict with ``n_changes``, ``n_rows``, ``n_pairs``, ``rate`` (pair
        denominated), ``first_date`` and ``last_date`` (ISO date strings) — the
        rate never travels without the window it was measured on.

    Raises:
        ValueError: via :func:`churn_rate` if fewer than two rows are present.
    """
    labels = probs_matrix.idxmax(axis=1)
    n_rows = int(len(probs_matrix))
    n_changes = state_change_count(labels)
    return {
        "n_changes": n_changes,
        "n_rows": n_rows,
        "n_pairs": n_rows - 1,
        "rate": churn_rate(n_changes, n_rows),
        "first_date": str(pd.Timestamp(probs_matrix.index.min()).date()) if n_rows else None,
        "last_date": str(pd.Timestamp(probs_matrix.index.max()).date()) if n_rows else None,
    }


if __name__ == "__main__":
    import tempfile

    import numpy as np

    logging.basicConfig(level=logging.INFO)

    # Synthetic self-check — no network, no checkpoint, no Phase-1..4 dependency
    # (mirrors sojourn_lag.py's footer). A hand-built driver-shaped bucket goes
    # to disk as state_{k} strings and comes back as integers; Track B's churn is
    # then measured off the restored frame, which is the whole round trip this
    # module exists to close.
    demo_index = pd.date_range("1972-01-31", periods=10, freq="ME")
    demo_argmax = [0, 0, 1, 1, 0, 2, 2, 2, 1, 1]  # 4 adjacent-pair changes by hand
    demo_metrics: dict = {"dates": list(demo_index), "proba": [], "classes": []}
    for winner in demo_argmax:
        row = np.full(3, 0.15)
        row[winner] = 0.70
        demo_metrics["proba"].append(row)
        demo_metrics["classes"].append([0, 1, 2])

    with tempfile.TemporaryDirectory() as tmp:
        demo_path = Path(tmp) / "demo_probs.parquet"
        demo_info = write_probability_matrix(demo_metrics, demo_path)
        demo_matrix = read_probability_matrix(demo_path)
        demo_churn = argmax_churn(demo_matrix)

    print(  # noqa: T201 — first-class self-check output
        "Track B churn self-check (plan 08-01)\n"
        f"  rows written / states:   {demo_info['n_rows']} / {demo_info['states']}\n"
        f"  restored column dtype:   {demo_matrix.columns.dtype} (must be integer)\n"
        f"  argmax changes:          {demo_churn['n_changes']} (hand-built expectation: 4)\n"
        f"  rate over {demo_churn['n_pairs']} pairs:     {demo_churn['rate']:.6f}\n"
        f"  window:                  {demo_churn['first_date']} -> {demo_churn['last_date']}"
    )
