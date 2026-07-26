"""
Per-run model-metrics artifacts — multiclass Brier score, calibration bins,
and confusion-table counts (design §8.8, EVAL-04). This is the forecast-
quality audit trail: how well-calibrated was the walk-forward nowcaster's
probability output, not a strategy-performance metric (see ``kpis.py`` for
that).

The three pure metric functions below are adapted math: the exact formulas
were read directly from the Phase-3 quarterly pipeline's retired
model-metrics helpers this session and re-expressed as plain functions over
``(y_true: list, proba: np.ndarray, classes: list)`` inputs — never imported.
This module's import list is numpy/pandas/stdlib only; it does not depend on
the incumbent pipeline's bundle-report shape (per-fold objects, per-horizon/
per-asset grouping), which the platform's walk-forward driver does not
produce.

Usage::

    from trading_crab_lib.platform.evaluation.model_metrics import (
        compute_brier_multiclass, calibration_bins, confusion_tidy,
        report_model_metrics,
    )
    artifacts = report_model_metrics(per_step_metrics, output_dir=None)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_crab_lib import OUTPUT_DIR

log = logging.getLogger(__name__)

# 5 fixed calibration bins (design §8.8) — the top bin is inclusive of 1.0.
BIN_EDGES = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], dtype=float)

# Schema-stable columns for the three persisted artifacts — always present in
# the written parquet, even when there are no predictions (empty-safe
# DataFrame + to_parquet idiom, ported from honesty/gap_lag.py).
_BRIER_COLUMNS = ["brier"]
_CALIBRATION_COLUMNS = [
    "class_label",
    "bin",
    "bin_low",
    "bin_high",
    "predicted_prob_mean",
    "observed_freq",
    "n_in_bin",
]
_CONFUSION_COLUMNS = ["true_label", "pred_label", "count"]


def compute_brier_multiclass(y_true: list[Any], proba: np.ndarray, classes: list[Any]) -> float:
    """Multiclass Brier score: mean over samples/classes of ``(p_k -
    onehot_k)^2`` — verbatim adapted math.

    Args:
        y_true: length-``n`` list of observed class labels.
        proba: ``(n, K)`` array of predicted probabilities, column order
            matching ``classes``.
        classes: length-``K`` list of class labels, in the column order of
            ``proba``.

    Returns:
        float: mean squared error of ``proba`` against the one-hot encoding
        of ``y_true`` (0.0 for perfectly one-hot-correct predictions).
    """
    proba = np.asarray(proba, dtype=float)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    onehot = np.zeros_like(proba, dtype=float)
    for i, yt in enumerate(y_true):
        idx = class_to_idx.get(yt)
        if idx is not None:
            onehot[i, idx] = 1.0
    diff = proba - onehot
    return float(np.mean(diff * diff))


def calibration_bins(y_true: list[Any], proba: np.ndarray, classes: list[Any]) -> list[dict]:
    """Tidy per-class calibration bins over 5 fixed bin edges (``BIN_EDGES``).

    The top bin (``[0.8, 1.0]``) is inclusive of 1.0; every other bin is
    half-open (``[low, high)``). Empty bins are skipped (never written as a
    zero-row placeholder — the per-run artifact's empty-safety is handled
    one level up, at ``report_model_metrics``).

    Args:
        y_true: length-``n`` list of observed class labels.
        proba: ``(n, K)`` array of predicted probabilities, column order
            matching ``classes``.
        classes: length-``K`` list of class labels, in the column order of
            ``proba``.

    Returns:
        list[dict]: one row per (class, non-empty bin) with keys
        ``class_label`` (forced to ``str`` so int regime ids and str
        behavior labels can coexist in one parquet), ``bin`` (1-indexed),
        ``bin_low``, ``bin_high``, ``predicted_prob_mean``,
        ``observed_freq``, ``n_in_bin``.
    """
    proba = np.asarray(proba, dtype=float)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_arr = np.array(y_true, dtype=object)
    rows: list[dict] = []

    for c in classes:
        c_idx = class_to_idx[c]
        p = proba[:, c_idx]
        true_mask = y_arr == c

        for b in range(len(BIN_EDGES) - 1):
            low = float(BIN_EDGES[b])
            high = float(BIN_EDGES[b + 1])
            if b == len(BIN_EDGES) - 2:
                bin_mask = (p >= low) & (p <= high)
            else:
                bin_mask = (p >= low) & (p < high)
            n = int(bin_mask.sum())
            if n == 0:
                continue

            rows.append(
                {
                    "class_label": str(c),
                    "bin": b + 1,
                    "bin_low": low,
                    "bin_high": high,
                    "predicted_prob_mean": float(p[bin_mask].mean()),
                    "observed_freq": float(true_mask[bin_mask].mean()),
                    "n_in_bin": n,
                }
            )

    return rows


def confusion_tidy(y_true: list[Any], y_pred: list[Any], classes: list[Any]) -> list[dict]:
    """Tidy confusion-matrix counts: one row per (true, pred) cell with a
    nonzero count.

    Args:
        y_true: length-``n`` list of observed class labels.
        y_pred: length-``n`` list of predicted class labels.
        classes: list of class labels to enumerate over (both axes).

    Returns:
        list[dict]: rows with keys ``true_label``, ``pred_label`` (both
        forced to ``str``), ``count``. Counts sum to ``len(y_true)``.
    """
    y_true_arr = np.array(y_true, dtype=object)
    y_pred_arr = np.array(y_pred, dtype=object)
    rows: list[dict] = []

    for t in classes:
        for p in classes:
            cnt = int(np.sum((y_true_arr == t) & (y_pred_arr == p)))
            if cnt <= 0:
                continue
            rows.append({"true_label": str(t), "pred_label": str(p), "count": cnt})

    return rows


def _reconcile_and_stack_proba(
    proba_rows: list[np.ndarray], classes_rows: list[list[Any]]
) -> tuple[np.ndarray, list[Any]]:
    """Union-of-classes reconciliation (review F3): pad every per-step proba
    row to a single rectangular ``(n_steps, K)`` array before scoring.

    A given step's ``classes`` may be a strict subset of the union of every
    state ever observed across all steps (e.g. an early, small train window
    whose nowcaster only saw 3 of 5 canonical states, fixed K=5 by
    ``labeling/jump_model.py::canonicalize_states`` in production). Any state
    missing from a step's own ``classes`` list is padded with ``0.0`` in that
    step's row — never dropped, never left to corrupt the stack shape.

    Args:
        proba_rows: list of 1-D probability arrays, one per walk-forward
            step, in the SAME order as ``classes_rows``.
        classes_rows: list of per-step class-label lists, matching each
            ``proba_rows`` entry's column order.

    Returns:
        tuple[np.ndarray, list]: ``(stacked, classes)`` — ``stacked`` is a
        rectangular ``(n_steps, K)`` array; ``classes`` is the reconciled,
        ascending-sorted union of every state observed across all steps
        (the single column order shared by every row of ``stacked``).
    """
    all_states = sorted({state for classes in classes_rows for state in classes})
    state_to_col = {state: i for i, state in enumerate(all_states)}

    stacked = np.zeros((len(proba_rows), len(all_states)), dtype=float)
    for row_i, (proba, classes) in enumerate(zip(proba_rows, classes_rows)):
        for state, p in zip(classes, proba):
            stacked[row_i, state_to_col[state]] = float(p)

    return stacked, all_states


def report_model_metrics(per_step_metrics: dict, *, output_dir: Path | None = None) -> dict[str, Path]:
    """Score and persist the EVAL-04 model-metrics artifacts for one full
    walk-forward run.

    ``per_step_metrics`` is the walk-forward driver's return contract
    (``backtest/driver.py::run_backtest``) — keys ``"dates"``, ``"proba"``,
    ``"classes"`` — plus a ``"y_true"`` list that the REPORT LAYER (not this
    function, not the walk-forward loop) has already joined by date from the
    full-sample smoothed reference labeling (review F2). This function
    asserts ``len(y_true) == len(dates) == len(proba)`` and raises
    ``ValueError`` on any mismatch — a silent length mismatch would corrupt
    every downstream Brier/calibration/confusion number without ever raising.

    Per-step proba rows are reconciled to a rectangular ``(n_steps, K)``
    array via ``_reconcile_and_stack_proba`` (review F3) before scoring, so a
    ragged early-history step never corrupts or misaligns the stack.
    ``y_pred`` is derived as the argmax reconciled class per step.

    Writes three parquet files under ``output_dir`` (or, by default,
    ``OUTPUT_DIR/reports/platform/``): ``model_metrics_brier.parquet``,
    ``model_metrics_calibration.parquet``, ``model_metrics_confusion.parquet``.
    Each is schema-stable (declared columns always present) even when there
    are no predictions at all (empty-safe DataFrame + to_parquet idiom,
    ported from ``honesty/gap_lag.py``).

    Args:
        per_step_metrics: dict with keys ``"dates"``, ``"proba"``,
            ``"classes"``, ``"y_true"`` (see above).
        output_dir: overrides the default artifact directory (for tests).

    Returns:
        dict[str, Path]: ``{"brier": ..., "calibration": ..., "confusion": ...}``
        — the three written parquet paths.

    Raises:
        ValueError: if ``len(y_true)`` does not equal ``len(dates)`` and
            ``len(proba)``.
    """
    dates = per_step_metrics["dates"]
    proba_rows = per_step_metrics["proba"]
    classes_rows = per_step_metrics["classes"]
    y_true = per_step_metrics["y_true"]

    if not (len(y_true) == len(dates) == len(proba_rows)):
        raise ValueError(
            f"y_true/dates/proba length mismatch: len(y_true)={len(y_true)}, "
            f"len(dates)={len(dates)}, len(proba)={len(proba_rows)} — y_true "
            "must be joined by date from the full-sample smoothed reference "
            "labeling by the report layer (review F2), never sourced from "
            "the in-progress walk-forward loop."
        )

    target_dir = Path(output_dir) if output_dir is not None else OUTPUT_DIR / "reports" / "platform"
    target_dir.mkdir(parents=True, exist_ok=True)
    brier_path = target_dir / "model_metrics_brier.parquet"
    calibration_path = target_dir / "model_metrics_calibration.parquet"
    confusion_path = target_dir / "model_metrics_confusion.parquet"

    brier_value: float | None = None
    if not dates:
        brier_df = pd.DataFrame(columns=_BRIER_COLUMNS)
        calibration_df = pd.DataFrame(columns=_CALIBRATION_COLUMNS)
        confusion_df = pd.DataFrame(columns=_CONFUSION_COLUMNS)
    else:
        stacked_proba, classes = _reconcile_and_stack_proba(proba_rows, classes_rows)
        y_pred = [classes[i] for i in np.argmax(stacked_proba, axis=1)]

        brier_value = compute_brier_multiclass(y_true, stacked_proba, classes)
        calibration_rows = calibration_bins(y_true, stacked_proba, classes)
        confusion_rows = confusion_tidy(y_true, y_pred, classes)

        brier_df = pd.DataFrame([{"brier": brier_value}])
        calibration_df = (
            pd.DataFrame(calibration_rows) if calibration_rows else pd.DataFrame(columns=_CALIBRATION_COLUMNS)
        )
        confusion_df = (
            pd.DataFrame(confusion_rows) if confusion_rows else pd.DataFrame(columns=_CONFUSION_COLUMNS)
        )

    brier_df.to_parquet(brier_path, index=False)
    calibration_df.to_parquet(calibration_path, index=False)
    confusion_df.to_parquet(confusion_path, index=False)

    summary = (
        "Model Metrics Artifacts (design §8.8)\n"
        f"  n_steps:              {len(dates)}\n"
        f"  brier:                {brier_value}\n"
        f"  brier artifact:       {brier_path}\n"
        f"  calibration artifact: {calibration_path}\n"
        f"  confusion artifact:   {confusion_path}"
    )
    log.info(summary)
    print(summary)  # noqa: T201 — first-class CLI run output, not debug noise

    return {"brier": brier_path, "calibration": calibration_path, "confusion": confusion_path}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Synthetic self-check — no network, no checkpoint dependency, no
    # Phase-1..4 execution required. Builds a tiny per_step_metrics dict
    # (dates + ragged proba/classes + a date-joined y_true) and writes the
    # three artifacts to a temp dir.
    import tempfile

    _demo_dates = list(pd.date_range("1972-01-31", periods=4, freq="ME"))
    _demo_per_step_metrics = {
        "dates": _demo_dates,
        # Step 0 only observed 3 of 5 canonical states (early small-sample
        # nowcaster window) — review F3 reconciliation pads the rest.
        "proba": [
            np.array([0.6, 0.3, 0.1]),
            np.array([0.1, 0.1, 0.1, 0.6, 0.1]),
            np.array([0.2, 0.2, 0.2, 0.2, 0.2]),
            np.array([0.05, 0.05, 0.1, 0.1, 0.7]),
        ],
        "classes": [
            [0, 1, 2],
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
        ],
        # A date-joined y_true (review F2) — in production this comes from
        # the report layer's full-sample smoothed reference labeling.
        "y_true": [0, 3, 2, 4],
    }

    with tempfile.TemporaryDirectory() as _tmp:
        _artifacts = report_model_metrics(_demo_per_step_metrics, output_dir=Path(_tmp))
        print(f"Self-check artifacts written to: {_artifacts}")  # noqa: T201
