"""Unit tests for trading_crab_lib.platform.evaluation.model_metrics (EVAL-04).

Follows the incumbent test_platform_gap_lag.py / test_platform_evaluation_kpis.py
structure: a class per behavior, docstring per test. All computation runs
against hand-constructed synthetic (y_true, proba, classes) inputs — no
network, no checkpoint dependency, no incumbent-pipeline FoldReport-shaped
object anywhere in this file.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_crab_lib.platform.evaluation.model_metrics import (
    calibration_bins,
    compute_brier_multiclass,
    confusion_tidy,
)

# ── compute_brier_multiclass ─────────────────────────────────────────────────────


class TestBrierKnownAnswer:
    def test_perfect_one_hot_predictions_is_zero(self):
        """Brier score is exactly 0.0 when every predicted probability vector is a
        perfect one-hot match for y_true (mean squared error vs one-hot)."""
        classes = [0, 1, 2]
        y_true = [0, 1, 2, 0]
        proba = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
            ]
        )

        result = compute_brier_multiclass(y_true, proba, classes)

        assert result == pytest.approx(0.0)

    def test_miscalibrated_predictions_is_strictly_positive(self):
        """A deliberately miscalibrated (flat) probability vector against a known
        y_true produces a Brier score strictly greater than 0.0."""
        classes = [0, 1, 2]
        y_true = [0, 1, 2]
        proba = np.array(
            [
                [1 / 3, 1 / 3, 1 / 3],
                [1 / 3, 1 / 3, 1 / 3],
                [1 / 3, 1 / 3, 1 / 3],
            ]
        )

        result = compute_brier_multiclass(y_true, proba, classes)

        assert result > 0.0


# ── calibration_bins ──────────────────────────────────────────────────────────────


class TestCalibrationBinEdges:
    def test_uses_five_fixed_bin_edges(self):
        """calibration_bins partitions predicted probabilities into the 5 fixed
        bins [0, 0.2, 0.4, 0.6, 0.8, 1.0] — every returned bin id is in 1..5."""
        classes = [0, 1]
        y_true = [0, 0, 1, 1]
        proba = np.array(
            [
                [0.1, 0.9],
                [0.3, 0.7],
                [0.5, 0.5],
                [0.9, 0.1],
            ]
        )

        rows = calibration_bins(y_true, proba, classes)

        bins_seen = {row["bin"] for row in rows}
        assert bins_seen.issubset({1, 2, 3, 4, 5})

    def test_probability_of_exactly_one_falls_in_top_inclusive_bin(self):
        """A predicted probability of exactly 1.0 falls in bin 5 ([0.8, 1.0],
        inclusive of 1.0) — never dropped as out-of-range."""
        classes = [0, 1]
        y_true = [1, 0]
        proba = np.array(
            [
                [0.0, 1.0],
                [1.0, 0.0],
            ]
        )

        rows = calibration_bins(y_true, proba, classes)

        class1_top_bin_rows = [
            row for row in rows if row["class_label"] == "1" and row["bin"] == 5
        ]
        assert len(class1_top_bin_rows) == 1
        assert class1_top_bin_rows[0]["n_in_bin"] == 1
        assert class1_top_bin_rows[0]["predicted_prob_mean"] == pytest.approx(1.0)

    def test_each_row_carries_declared_schema_columns(self):
        """Each returned row carries class_label, bin, bin_low, bin_high,
        predicted_prob_mean, observed_freq, n_in_bin — nothing more, nothing less."""
        classes = [0, 1]
        y_true = [0, 1]
        proba = np.array([[0.9, 0.1], [0.1, 0.9]])

        rows = calibration_bins(y_true, proba, classes)

        expected_keys = {
            "class_label",
            "bin",
            "bin_low",
            "bin_high",
            "predicted_prob_mean",
            "observed_freq",
            "n_in_bin",
        }
        assert all(set(row.keys()) == expected_keys for row in rows)

    def test_class_label_forced_to_string(self):
        """class_label is forced to str so int regime ids and str behavior labels
        can coexist in the same parquet without a pyarrow type conflict."""
        classes = [0, 1]
        y_true = [0, 1]
        proba = np.array([[0.9, 0.1], [0.1, 0.9]])

        rows = calibration_bins(y_true, proba, classes)

        assert all(isinstance(row["class_label"], str) for row in rows)


# ── confusion_tidy ────────────────────────────────────────────────────────────────


class TestConfusionSumsToN:
    def test_counts_sum_to_n(self):
        """confusion_tidy's counts sum to len(y_true) for a synthetic
        (y_true, y_pred) pair — the tidy-count invariant."""
        classes = [0, 1, 2]
        y_true = [0, 1, 2, 0, 1, 2, 0]
        y_pred = [0, 1, 1, 0, 2, 2, 1]

        rows = confusion_tidy(y_true, y_pred, classes)

        assert sum(row["count"] for row in rows) == len(y_true)

    def test_class_labels_forced_to_string(self):
        """true_label/pred_label are forced to str so int regime ids and str
        behavior labels can coexist in the same parquet without a pyarrow type
        conflict."""
        classes = [0, 1]
        y_true = [0, 1]
        y_pred = [0, 1]

        rows = confusion_tidy(y_true, y_pred, classes)

        assert all(
            isinstance(row["true_label"], str) and isinstance(row["pred_label"], str)
            for row in rows
        )
