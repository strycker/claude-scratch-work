"""Persist calibration and accuracy metrics from :class:`~.classifier.FoldReport` to disk.

Writes JSON/Parquet artifacts under ``outputs/reports/model_metrics/`` for dashboards
and regression checks (multiclass Brier score, per-bin calibration, fold summaries).
"""

# Offline audit trail: same models as step 5, extra tables for calibration — not used to train.

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from .classifier import FoldReport

log = logging.getLogger(__name__)

BIN_EDGES = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], dtype=float)


def _compute_brier_multiclass(y_true: list[Any], proba: np.ndarray, classes: list[Any]) -> float:
    """
    Multiclass Brier: mean over samples/classes of (p_k - onehot_k)^2.
    """
    if proba.ndim != 2:
        raise ValueError("proba must be a 2D array [n_samples, n_classes]")

    class_to_idx = {c: i for i, c in enumerate(classes)}
    onehot = np.zeros_like(proba, dtype=float)
    for i, yt in enumerate(y_true):
        idx = class_to_idx.get(yt)
        if idx is None:
            continue
        onehot[i, idx] = 1.0
    diff = proba - onehot
    return float(np.mean(diff * diff))


def _calibration_bins(
    y_true: list[Any],
    proba: np.ndarray,
    classes: list[Any],
) -> list[dict]:
    """
    Tidy per-class calibration bins.
    """
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
                    # Force to string so regime (ints) and behavior (up/flat/down)
                    # can coexist in the same parquet without pyarrow type conflicts.
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


def _confusion_tidy(
    y_true: list[Any],
    y_pred: list[Any],
    classes: list[Any],
    fold: int | str,
) -> list[dict]:
    """
    Build tidy confusion matrix counts for a fold.
    """
    y_true_arr = np.array(y_true, dtype=object)
    y_pred_arr = np.array(y_pred, dtype=object)
    rows: list[dict] = []

    for t in classes:
        for p in classes:
            cnt = int(np.sum((y_true_arr == t) & (y_pred_arr == p)))
            if cnt <= 0:
                continue
            rows.append(
                {
                    "fold": fold,
                    # Force to string so regime (ints) and behavior (up/flat/down)
                    # can coexist in the same parquet without pyarrow type conflicts.
                    "true_label": str(t),
                    "pred_label": str(p),
                    "count": cnt,
                }
            )
    return rows


def write_model_metrics_artifacts(
    *,
    output_dir: Path,
    feature_source: Literal["features_supervised", "features"],
    noncausal_used: bool,
    regime_current_bundle: dict,
    forward_models: dict,
    behavior_bundle: dict,
    binning_log: bool = False,
) -> None:
    """
    Persist structured metrics artifacts for Phase 3 plan-04.

    Writes:
      - cv_summary.parquet
      - per_fold.jsonl
      - confusion_matrices.parquet
      - calibration.parquet
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    cv_summary_path = output_dir / "cv_summary.parquet"
    per_fold_path = output_dir / "per_fold.jsonl"
    confusion_path = output_dir / "confusion_matrices.parquet"
    calibration_path = output_dir / "calibration.parquet"

    cv_summary_rows: list[dict] = []
    per_fold_rows: list[dict] = []
    confusion_rows: list[dict] = []
    calibration_rows: list[dict] = []

    def add_agg_metrics(
        *,
        family: str,
        model: str,
        horizon: int | None,
        asset: str | None,
        fold_reports: list[FoldReport],
    ) -> None:
        if not fold_reports:
            return

        n_splits = len(fold_reports)
        accuracy = float(np.mean([fr.report.get("accuracy", 0.0) for fr in fold_reports]))
        macro_f1 = float(
            np.mean([fr.report.get("macro avg", {}).get("f1-score", 0.0) for fr in fold_reports])
        )
        weighted_f1 = float(
            np.mean([fr.report.get("weighted avg", {}).get("f1-score", 0.0) for fr in fold_reports])
        )

        briers: list[float] = []
        for fr in fold_reports:
            if fr.y_true_test is None or fr.proba_test is None or fr.class_order is None:
                continue
            briers.append(
                _compute_brier_multiclass(fr.y_true_test, np.asarray(fr.proba_test), fr.class_order)
            )
        brier_mean = float(np.mean(briers)) if briers else 0.0

        for metric, value in (
            ("accuracy", accuracy),
            ("macro_f1", macro_f1),
            ("weighted_f1", weighted_f1),
            ("brier", brier_mean),
        ):
            cv_summary_rows.append(
                {
                    "family": family,
                    "model": model,
                    "horizon": horizon,
                    "asset": asset,
                    "metric": metric,
                    "value": float(value),
                    "n_splits": int(n_splits),
                    "feature_source": feature_source,
                    "noncausal_used": bool(noncausal_used),
                }
            )

    # ── Regime: current (horizon=None) ───────────────────────────────────────
    for model_name, bundle in regime_current_bundle.get("cv_scores", {}).items():
        reports: list[FoldReport] = bundle.get("reports", []) or []
        add_agg_metrics(
            family="regime",
            model=model_name,
            horizon=None,
            asset=None,
            fold_reports=reports,
        )

        for fold_i, fr in enumerate(reports, start=1):
            if (
                fr.y_true_test is None
                or fr.y_pred_test is None
                or fr.proba_test is None
                or fr.class_order is None
            ):
                continue
            brier = _compute_brier_multiclass(
                fr.y_true_test, np.asarray(fr.proba_test), fr.class_order
            )
            calib_bins = _calibration_bins(
                fr.y_true_test, np.asarray(fr.proba_test), fr.class_order
            )
            conf_tidy = _confusion_tidy(fr.y_true_test, fr.y_pred_test, fr.class_order, fold=fold_i)

            per_fold_rows.append(
                {
                    "family": "regime",
                    "model": model_name,
                    "horizon": None,
                    "asset": None,
                    "fold": fold_i,
                    "train_indices": fr.train_indices,
                    "test_indices": fr.test_indices,
                    "classification_report": fr.report,
                    "brier": float(brier),
                    "feature_source": feature_source,
                    "noncausal_used": bool(noncausal_used),
                }
            )

            for r in conf_tidy:
                confusion_rows.append(
                    {
                        "family": "regime",
                        "model": model_name,
                        "horizon": None,
                        "asset": None,
                        "fold": r["fold"],
                        "true_label": r["true_label"],
                        "pred_label": r["pred_label"],
                        "count": int(r["count"]),
                        "feature_source": feature_source,
                    }
                )

            for c_row in calib_bins:
                calibration_rows.append(
                    {
                        "family": "regime",
                        "model": model_name,
                        "horizon": None,
                        "asset": None,
                        "fold": fold_i,
                        **c_row,
                        "brier": float(brier),
                        "feature_source": feature_source,
                    }
                )

    # ── Regime: forward models ───────────────────────────────────────────────
    for horizon, bundle_h in forward_models.items():
        cv_scores = bundle_h.get("cv_scores", {}) or {}
        for model_name, model_blob in cv_scores.items():
            reports = model_blob.get("reports", []) or []
            add_agg_metrics(
                family="regime",
                model=model_name,
                horizon=int(horizon),
                asset=None,
                fold_reports=reports,
            )

            for fold_i, fr in enumerate(reports, start=1):
                if (
                    fr.y_true_test is None
                    or fr.y_pred_test is None
                    or fr.proba_test is None
                    or fr.class_order is None
                ):
                    continue
                brier = _compute_brier_multiclass(
                    fr.y_true_test, np.asarray(fr.proba_test), fr.class_order
                )
                calib_bins = _calibration_bins(
                    fr.y_true_test, np.asarray(fr.proba_test), fr.class_order
                )
                conf_tidy = _confusion_tidy(
                    fr.y_true_test, fr.y_pred_test, fr.class_order, fold=fold_i
                )

                per_fold_rows.append(
                    {
                        "family": "regime",
                        "model": model_name,
                        "horizon": int(horizon),
                        "asset": None,
                        "fold": fold_i,
                        "train_indices": fr.train_indices,
                        "test_indices": fr.test_indices,
                        "classification_report": fr.report,
                        "brier": float(brier),
                        "feature_source": feature_source,
                        "noncausal_used": bool(noncausal_used),
                    }
                )

                for r in conf_tidy:
                    confusion_rows.append(
                        {
                            "family": "regime",
                            "model": model_name,
                            "horizon": int(horizon),
                            "asset": None,
                            "fold": r["fold"],
                            "true_label": r["true_label"],
                            "pred_label": r["pred_label"],
                            "count": int(r["count"]),
                            "feature_source": feature_source,
                        }
                    )

                for c_row in calib_bins:
                    calibration_rows.append(
                        {
                            "family": "regime",
                            "model": model_name,
                            "horizon": int(horizon),
                            "asset": None,
                            "fold": fold_i,
                            **c_row,
                            "brier": float(brier),
                            "feature_source": feature_source,
                        }
                    )

    # ── Behavior models ──────────────────────────────────────────────────────
    # behavior_bundle is expected to be {"models": ..., "cv_scores": ...}
    for asset, asset_blob in (behavior_bundle.get("cv_scores") or {}).items():
        for horizon, horizon_blob in (asset_blob or {}).items():
            reports = horizon_blob.get("reports", []) or []
            add_agg_metrics(
                family="behavior",
                model="behavior-rf",
                horizon=int(horizon),
                asset=str(asset),
                fold_reports=reports,
            )

            for fold_i, fr in enumerate(reports, start=1):
                if (
                    fr.y_true_test is None
                    or fr.y_pred_test is None
                    or fr.proba_test is None
                    or fr.class_order is None
                ):
                    continue
                brier = _compute_brier_multiclass(
                    fr.y_true_test, np.asarray(fr.proba_test), fr.class_order
                )
                calib_bins = _calibration_bins(
                    fr.y_true_test, np.asarray(fr.proba_test), fr.class_order
                )
                conf_tidy = _confusion_tidy(
                    fr.y_true_test, fr.y_pred_test, fr.class_order, fold=fold_i
                )

                per_fold_rows.append(
                    {
                        "family": "behavior",
                        "model": "behavior-rf",
                        "horizon": int(horizon),
                        "asset": str(asset),
                        "fold": fold_i,
                        "train_indices": fr.train_indices,
                        "test_indices": fr.test_indices,
                        "classification_report": fr.report,
                        "brier": float(brier),
                        "feature_source": feature_source,
                        "noncausal_used": bool(noncausal_used),
                    }
                )

                for r in conf_tidy:
                    confusion_rows.append(
                        {
                            "family": "behavior",
                            "model": "behavior-rf",
                            "horizon": int(horizon),
                            "asset": str(asset),
                            "fold": r["fold"],
                            "true_label": r["true_label"],
                            "pred_label": r["pred_label"],
                            "count": int(r["count"]),
                            "feature_source": feature_source,
                        }
                    )

                for c_row in calib_bins:
                    calibration_rows.append(
                        {
                            "family": "behavior",
                            "model": "behavior-rf",
                            "horizon": int(horizon),
                            "asset": str(asset),
                            "fold": fold_i,
                            **c_row,
                            "brier": float(brier),
                            "feature_source": feature_source,
                        }
                    )

    # ── Write artifacts ───────────────────────────────────────────────────────
    cv_df = pd.DataFrame(cv_summary_rows)
    if cv_df.empty:
        cv_df = pd.DataFrame(
            columns=[
                "family",
                "model",
                "horizon",
                "asset",
                "metric",
                "value",
                "n_splits",
                "feature_source",
                "noncausal_used",
            ]
        )
    cv_df.to_parquet(cv_summary_path, index=False)

    # One JSON object per line for per-fold records
    with per_fold_path.open("w", encoding="utf-8") as f:
        for row in per_fold_rows:
            json.dump(row, f, default=str)
            f.write("\n")

    conf_df = pd.DataFrame(confusion_rows)
    if conf_df.empty:
        conf_df = pd.DataFrame(
            columns=[
                "family",
                "model",
                "horizon",
                "asset",
                "fold",
                "true_label",
                "pred_label",
                "count",
                "feature_source",
            ]
        )
    conf_df.to_parquet(confusion_path, index=False)

    cal_df = pd.DataFrame(calibration_rows)
    if cal_df.empty:
        cal_df = pd.DataFrame(
            columns=[
                "family",
                "model",
                "horizon",
                "asset",
                "fold",
                "class_label",
                "bin",
                "bin_low",
                "bin_high",
                "predicted_prob_mean",
                "observed_freq",
                "n_in_bin",
                "brier",
                "feature_source",
            ]
        )
    cal_df.to_parquet(calibration_path, index=False)

    if binning_log:
        log.info("Wrote metrics artifacts to %s", output_dir)
