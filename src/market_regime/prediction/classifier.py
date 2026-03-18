"""
Backwards-compatible bundle API for regime classifiers.

The public functions in this module mirror the original API used by pipelines
and tests that pre-date the refactor.  They return rich bundle dicts with
FoldReport objects, dual (RF + DT) models, and class-order metadata.

New code should import from ``market_regime.prediction`` directly; this module
exists for backwards compatibility and for tests that assert on fold-level CV
metadata.
"""

from __future__ import annotations

import logging
from collections import namedtuple
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import TimeSeriesSplit
from sklearn.tree import DecisionTreeClassifier

log = logging.getLogger(__name__)

# ── public types ───────────────────────────────────────────────────────────────

FoldReport = namedtuple("FoldReport", ["train_indices", "test_indices", "accuracy"])


# ── internal helpers ───────────────────────────────────────────────────────────

def _rf_factory(n_estimators: int = 200, random_state: int = 42) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced",
    )


def _dt_factory(max_depth: int = 8, random_state: int = 42) -> DecisionTreeClassifier:
    return DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)


def _gb_factory(
    n_estimators: int = 200,
    max_depth: int = 5,
    random_state: int = 42,
) -> GradientBoostingClassifier:
    return GradientBoostingClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
    )


def _run_tscv(
    factory,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int,
) -> list[FoldReport]:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    reports: list[FoldReport] = []
    for train_idx, test_idx in tscv.split(X):
        m = factory()
        m.fit(X.iloc[train_idx], y.iloc[train_idx])
        acc = float(m.score(X.iloc[test_idx], y.iloc[test_idx]))
        reports.append(FoldReport(
            train_indices=train_idx.tolist(),
            test_indices=test_idx.tolist(),
            accuracy=acc,
        ))
    return reports


# ── public training functions ──────────────────────────────────────────────────

def train_current_regime(
    X: pd.DataFrame,
    y: pd.Series,
    cv_splits: int = 5,
    n_estimators: int = 200,
    random_state: int = 42,
    dt_max_depth: int = 8,
    include_gb: bool = False,
    gb_max_depth: int = 5,
) -> dict:
    """
    Train RF and DT classifiers to predict today's regime.

    Returns a bundle dict::

        {
            "models":     {"rf": fitted_rf, "dt": fitted_dt},
            "cv_reports": {"rf": [FoldReport, ...], "dt": [FoldReport, ...]},
            "labels":     [regime_id, ...],   # sorted unique labels
        }

    If ``include_gb=True``, a GradientBoostingClassifier is also trained and
    added to "models" and "cv_reports" under the "gb" key.

    TimeSeriesSplit is used so every test fold only sees future data relative
    to its training fold (walk-forward CV, no look-ahead).
    """
    rf_reports = _run_tscv(
        lambda: _rf_factory(n_estimators, random_state), X, y, cv_splits
    )
    dt_reports = _run_tscv(
        lambda: _dt_factory(dt_max_depth, random_state), X, y, cv_splits
    )

    rf_final = _rf_factory(n_estimators, random_state)
    rf_final.fit(X, y)

    dt_final = _dt_factory(dt_max_depth, random_state)
    dt_final.fit(X, y)

    models: dict = {"rf": rf_final, "dt": dt_final}
    cv_reports: dict = {"rf": rf_reports, "dt": dt_reports}

    if include_gb:
        gb_reports = _run_tscv(
            lambda: _gb_factory(n_estimators, gb_max_depth, random_state), X, y, cv_splits
        )
        gb_final = _gb_factory(n_estimators, gb_max_depth, random_state)
        gb_final.fit(X, y)
        models["gb"] = gb_final
        cv_reports["gb"] = gb_reports

    labels = sorted(pd.unique(y).tolist())

    return {
        "models": models,
        "cv_reports": cv_reports,
        "labels": labels,
    }


def train_forward_classifiers(
    X: pd.DataFrame,
    y: pd.Series,
    horizons: list[int] | None = None,
    cv_splits: int = 5,
    n_estimators: int = 200,
    random_state: int = 42,
    dt_max_depth: int = 8,
    include_gb: bool = False,
    gb_max_depth: int = 5,
) -> dict[int, dict]:
    """
    Train RF and DT classifiers to predict regime H quarters forward.

    If ``include_gb=True``, a GradientBoostingClassifier is also trained per horizon.

    Returns::

        {
            horizon: {
                "models":     {"rf": fitted_rf, "dt": fitted_dt},
                "cv_reports": {"rf": [FoldReport, ...], "dt": [FoldReport, ...]},
                "class_order": [regime_id, ...],   # sorted unique regimes
            },
            ...
        }
    """
    if horizons is None:
        horizons = [1, 2, 4, 8]

    class_order = sorted(pd.unique(y.dropna()).tolist())
    results: dict[int, dict] = {}

    for h in horizons:
        y_future = y.shift(-h).dropna().astype(int)
        X_aligned = X.loc[y_future.index]

        rf_reports = _run_tscv(
            lambda: _rf_factory(n_estimators, random_state), X_aligned, y_future, cv_splits
        )
        dt_reports = _run_tscv(
            lambda: _dt_factory(dt_max_depth, random_state), X_aligned, y_future, cv_splits
        )

        rf_final = _rf_factory(n_estimators, random_state)
        rf_final.fit(X_aligned, y_future)

        dt_final = _dt_factory(dt_max_depth, random_state)
        dt_final.fit(X_aligned, y_future)

        models: dict = {"rf": rf_final, "dt": dt_final}
        cv_reports: dict = {"rf": rf_reports, "dt": dt_reports}

        if include_gb:
            gb_reports = _run_tscv(
                lambda: _gb_factory(n_estimators, gb_max_depth, random_state),
                X_aligned, y_future, cv_splits,
            )
            gb_final = _gb_factory(n_estimators, gb_max_depth, random_state)
            gb_final.fit(X_aligned, y_future)
            models["gb"] = gb_final
            cv_reports["gb"] = gb_reports

        results[h] = {
            "models": models,
            "cv_reports": cv_reports,
            "class_order": class_order,
        }

    return results


# ── interpretability helpers ───────────────────────────────────────────────────


def extract_top_features(
    model,
    feature_names: list[str],
    top_k: int = 10,
) -> list[tuple[str, float]]:
    """
    Rank features by importance from a fitted tree-based model.

    Returns a list of (feature_name, importance) tuples sorted descending.
    """
    importances = model.feature_importances_
    pairs = sorted(
        zip(feature_names, importances),
        key=lambda p: p[1],
        reverse=True,
    )
    return pairs[:top_k]


def train_interpretability_tree(
    X: pd.DataFrame,
    y: pd.Series,
    model,
    top_k: int = 10,
    max_depth: int = 4,
    random_state: int = 42,
) -> tuple[DecisionTreeClassifier, list[str]]:
    """
    Train a shallow DecisionTree on only the top-k features from ``model``.

    This produces a human-readable tree that approximates the complex model's
    decisions using only the most important features.

    Returns:
        (fitted_tree, selected_feature_names)
    """
    top = extract_top_features(model, list(X.columns), top_k=top_k)
    selected = [name for name, _ in top]
    X_reduced = X[selected]
    tree = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    tree.fit(X_reduced, y)
    return tree, selected


# ── metrics summary ────────────────────────────────────────────────────────────

def _aggregate_classification_reports(reports: list[dict]) -> dict:
    """
    Aggregate a list of sklearn classification_report dicts into overall + per-class stats.

    Each report is the dict returned by
    ``sklearn.metrics.classification_report(..., output_dict=True)``.
    """
    if not reports:
        return {"overall": {"accuracy": 0.0, "macro_f1": 0.0, "weighted_f1": 0.0}, "per_class": {}}

    accuracies = [r["accuracy"] for r in reports if "accuracy" in r]
    macro_f1s = [r["macro avg"]["f1-score"] for r in reports if "macro avg" in r]
    weighted_f1s = [r["weighted avg"]["f1-score"] for r in reports if "weighted avg" in r]

    skip_keys = {"accuracy", "macro avg", "weighted avg"}
    class_keys = [k for k in reports[0] if k not in skip_keys]
    per_class: dict[str, dict] = {}
    for cls in class_keys:
        per_class[cls] = {
            "precision": float(np.mean([r[cls]["precision"] for r in reports if cls in r])),
            "recall": float(np.mean([r[cls]["recall"] for r in reports if cls in r])),
            "f1": float(np.mean([r[cls]["f1-score"] for r in reports if cls in r])),
            "support": float(np.mean([r[cls]["support"] for r in reports if cls in r])),
        }

    return {
        "overall": {
            "accuracy": float(np.mean(accuracies)) if accuracies else 0.0,
            "macro_f1": float(np.mean(macro_f1s)) if macro_f1s else 0.0,
            "weighted_f1": float(np.mean(weighted_f1s)) if weighted_f1s else 0.0,
        },
        "per_class": per_class,
    }


def model_metrics_summary(results: dict) -> dict:
    """
    Flatten model metrics into a JSON-serialisable summary dict.

    Accepts three input shapes:

    1. **Current-regime bundle** — ``{"models": {...}, "cv_reports": {"dt": [report_dicts], "rf": [...]}}``
       → ``{"current": {"dt": {"overall": {...}, "per_class": {...}}, "rf": {...}}}``

    2. **Forward-horizon bundle** — ``{horizon_int: {"cv_reports": {"rf": [...]}}, ...}``
       → ``{horizon_int: {"rf": {"overall": {...}}, ...}, ...}``

    3. **Flat family rows** — ``{"regime": [row_dicts], "behavior": [row_dicts]}``
       → ``{"rows": [{...}, ...]}`` (each row gets a ``"family"`` tag added)

    The input dict is never mutated.
    """
    # --- detect shape ---

    # Shape 1: current-regime bundle
    if "models" in results and "cv_reports" in results:
        cv = results["cv_reports"]
        current: dict[str, Any] = {}
        for model_name, fold_reports in cv.items():
            if fold_reports and isinstance(fold_reports[0], FoldReport):
                # FoldReport objects from this module — build stub report dicts
                report_dicts: list[dict] = []
                for fr in fold_reports:
                    report_dicts.append({"accuracy": fr.accuracy, "macro avg": {"f1-score": fr.accuracy}, "weighted avg": {"f1-score": fr.accuracy}})
                current[model_name] = _aggregate_classification_reports(report_dicts)
            else:
                # Already classification_report dicts
                current[model_name] = _aggregate_classification_reports(list(fold_reports))
        return {"current": current}

    # Shape 2: forward-horizon bundle (all integer keys)
    if results and all(isinstance(k, int) for k in results):
        out: dict[int, dict] = {}
        for h, h_data in results.items():
            out[h] = {}
            for model_name, fold_reports in h_data.get("cv_reports", {}).items():
                if fold_reports and isinstance(fold_reports[0], FoldReport):
                    report_dicts = [{"accuracy": fr.accuracy, "macro avg": {"f1-score": fr.accuracy}, "weighted avg": {"f1-score": fr.accuracy}} for fr in fold_reports]
                    out[h][model_name] = _aggregate_classification_reports(report_dicts)
                else:
                    out[h][model_name] = _aggregate_classification_reports(list(fold_reports))
        return out

    # Shape 3: flat family rows
    rows: list[dict] = []
    for family, family_rows in results.items():
        for row in family_rows:
            merged = {"family": family}
            merged.update(row)
            rows.append(merged)
    return {"rows": rows}
