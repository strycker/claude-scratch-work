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

from typing import Any

import numpy as np

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
