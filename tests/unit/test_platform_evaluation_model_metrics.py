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
    _reconcile_and_stack_proba,
    calibration_bins,
    compute_brier_multiclass,
    confusion_tidy,
    report_model_metrics,
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


# ── _reconcile_and_stack_proba (review F3) ───────────────────────────────────────


class TestRaggedClassesReconciliation:
    def test_pads_missing_states_with_zero_and_stacks_rectangular(self):
        """An early step observing only 3 of 5 canonical states (proba row width
        3, classes [0, 1, 2]) is padded with 0.0 for the missing states 3 and 4,
        producing a rectangular (n_steps, 5) array with a single reconciled
        class list [0, 1, 2, 3, 4] (review F3)."""
        proba_rows = [
            np.array([0.5, 0.3, 0.2]),
            np.array([0.1, 0.2, 0.2, 0.3, 0.2]),
        ]
        classes_rows = [
            [0, 1, 2],
            [0, 1, 2, 3, 4],
        ]

        stacked, classes = _reconcile_and_stack_proba(proba_rows, classes_rows)

        assert classes == [0, 1, 2, 3, 4]
        assert stacked.shape == (2, 5)
        np.testing.assert_allclose(stacked[0], [0.5, 0.3, 0.2, 0.0, 0.0])
        np.testing.assert_allclose(stacked[1], [0.1, 0.2, 0.2, 0.3, 0.2])

    def test_reconciled_stack_scores_without_shape_error(self):
        """compute_brier_multiclass/calibration_bins run on the reconciled,
        padded stack without a shape error, even though the raw per-step rows
        were ragged (a mismatched-width stack would previously raise)."""
        proba_rows = [np.array([0.5, 0.3, 0.2]), np.array([0.1, 0.2, 0.2, 0.3, 0.2])]
        classes_rows = [[0, 1, 2], [0, 1, 2, 3, 4]]
        y_true = [0, 3]

        stacked, classes = _reconcile_and_stack_proba(proba_rows, classes_rows)
        brier = compute_brier_multiclass(y_true, stacked, classes)
        rows = calibration_bins(y_true, stacked, classes)

        assert brier >= 0.0
        assert isinstance(rows, list)


# ── report_model_metrics: y_true date-alignment guard (review F2) ────────────────


class TestYTrueDateAlignment:
    def test_matching_length_y_true_scores_correctly(self, tmp_path):
        """A y_true list whose length matches dates/proba scores correctly —
        report_model_metrics succeeds and returns the written artifact paths."""
        dates = list(pd.date_range("2020-01-31", periods=3, freq="ME"))
        per_step_metrics = {
            "dates": dates,
            "proba": [np.array([0.8, 0.2]), np.array([0.2, 0.8]), np.array([0.6, 0.4])],
            "classes": [[0, 1], [0, 1], [0, 1]],
            "y_true": [0, 1, 0],
        }

        paths = report_model_metrics(per_step_metrics, output_dir=tmp_path)

        assert paths["brier"].exists()

    def test_mismatched_length_y_true_raises_value_error(self, tmp_path):
        """A y_true whose length differs from dates/proba raises ValueError —
        the alignment guard fires, proving y_true is joined-by-date from the
        report layer and never sourced from the in-progress walk-forward loop
        (review F2)."""
        dates = list(pd.date_range("2020-01-31", periods=3, freq="ME"))
        per_step_metrics = {
            "dates": dates,
            "proba": [np.array([0.8, 0.2]), np.array([0.2, 0.8]), np.array([0.6, 0.4])],
            "classes": [[0, 1], [0, 1], [0, 1]],
            "y_true": [0, 1],  # length 2, not 3 — deliberately misaligned
        }

        with pytest.raises(ValueError):
            report_model_metrics(per_step_metrics, output_dir=tmp_path)


# ── report_model_metrics: schema-stable parquet artifacts ────────────────────────


class TestReportArtifacts:
    def test_writes_three_round_trippable_parquet_files(self, tmp_path):
        """report_model_metrics writes brier/calibration/confusion parquet
        files, each round-trippable via read_parquet with the declared
        columns."""
        dates = list(pd.date_range("2020-01-31", periods=4, freq="ME"))
        per_step_metrics = {
            "dates": dates,
            "proba": [
                np.array([0.7, 0.3]),
                np.array([0.4, 0.6]),
                np.array([0.9, 0.1]),
                np.array([0.2, 0.8]),
            ],
            "classes": [[0, 1]] * 4,
            "y_true": [0, 1, 0, 1],
        }

        paths = report_model_metrics(per_step_metrics, output_dir=tmp_path)

        brier_df = pd.read_parquet(paths["brier"])
        calibration_df = pd.read_parquet(paths["calibration"])
        confusion_df = pd.read_parquet(paths["confusion"])

        assert "brier" in brier_df.columns
        assert {
            "class_label",
            "bin",
            "bin_low",
            "bin_high",
            "predicted_prob_mean",
            "observed_freq",
            "n_in_bin",
        }.issubset(calibration_df.columns)
        assert {"true_label", "pred_label", "count"}.issubset(confusion_df.columns)

    def test_empty_per_step_metrics_writes_schema_stable_zero_row_parquets(self, tmp_path):
        """An empty per_step_metrics (no predictions at all) still writes
        schema-stable, zero-row parquets for all three artifacts — never a
        missing file or a schema that varies with input size."""
        per_step_metrics = {"dates": [], "proba": [], "classes": [], "y_true": []}

        paths = report_model_metrics(per_step_metrics, output_dir=tmp_path)

        brier_df = pd.read_parquet(paths["brier"])
        calibration_df = pd.read_parquet(paths["calibration"])
        confusion_df = pd.read_parquet(paths["confusion"])

        assert len(brier_df) == 0
        assert len(calibration_df) == 0
        assert len(confusion_df) == 0
        assert set(brier_df.columns) == {"brier"}
        assert set(calibration_df.columns) == {
            "class_label",
            "bin",
            "bin_low",
            "bin_high",
            "predicted_prob_mean",
            "observed_freq",
            "n_in_bin",
        }
        assert set(confusion_df.columns) == {"true_label", "pred_label", "count"}
