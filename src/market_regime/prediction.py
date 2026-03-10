"""
Supervised regime classifier.

Trains two model types per pipeline run:

1. RandomForestClassifier  — high accuracy, ensemble; used for production predictions.
2. DecisionTreeClassifier  — shallow (max_depth=8), single tree; human-readable rules
   and fast feature-importance inspection before committing to the forest.

Both models use TimeSeriesSplit cross-validation so CV accuracy estimates reflect
genuine walk-forward performance — no data from the future leaks into any fold.

Also trains forward-looking binary classifiers for each (horizon, regime) pair:
    "Will we be in regime R exactly H quarters from now?"

Design note: ALL features fed to these classifiers must come from
data/processed/features_supervised.parquet, which is built with causal
(backward/right-aligned) rolling windows.  This guarantees that no future
information is present in any feature value used for training or scoring.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import TimeSeriesSplit
from sklearn.tree import DecisionTreeClassifier

log = logging.getLogger(__name__)


# ── helpers ────────────────────────────────────────────────────────────────────

def _log_feature_importance(model, feature_names, top_n: int = 15) -> None:
    importances = pd.Series(model.feature_importances_, index=feature_names)
    top = importances.sort_values(ascending=False).head(top_n)
    lines = "\n".join(f"  {f:<40s} {v:.4f}" for f, v in top.items())
    log.info("Top-%d feature importances:\n%s", top_n, lines)


def _tscv_scores(
    model_factory,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int,
    label: str,
) -> list[float]:
    """Run TimeSeriesSplit CV and return per-fold accuracy scores."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        m = model_factory()
        m.fit(X.iloc[train_idx], y.iloc[train_idx])
        acc = m.score(X.iloc[test_idx], y.iloc[test_idx])
        scores.append(acc)
        log.debug("%s fold %d/%d: accuracy=%.3f", label, fold, n_splits, acc)
    log.info(
        "%s CV accuracy: %.3f ± %.3f  (n_splits=%d)",
        label, np.mean(scores), np.std(scores), n_splits,
    )
    return scores


# ── public training functions ──────────────────────────────────────────────────

def train_classifier(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: dict,
    kind: str = "rf",
) -> RandomForestClassifier | DecisionTreeClassifier:
    """
    Train a classifier to predict today's regime label.

    Args:
        X    — feature matrix (rows = quarters, causal features only)
        y    — integer cluster labels aligned to X
        cfg  — pipeline config dict
        kind — "rf" for RandomForestClassifier, "dt" for DecisionTreeClassifier

    Returns:
        Fitted classifier (trained on all data).

    Uses TimeSeriesSplit for CV so every evaluation fold only looks at data
    that was available at that point in time.  The final model is re-fitted
    on ALL available data for maximum accuracy in production.
    """
    pcfg = cfg["prediction"]
    n_splits = pcfg.get("cv_splits", 5)
    rs = pcfg.get("random_state", 42)

    if kind == "rf":
        def _factory():
            return RandomForestClassifier(
                n_estimators=pcfg.get("n_estimators", 200),
                max_depth=pcfg.get("rf_max_depth", 12),
                random_state=rs,
                n_jobs=-1,
                class_weight="balanced",
            )
        label = "RF current-regime"
    elif kind == "dt":
        def _factory():
            return DecisionTreeClassifier(
                max_depth=pcfg.get("dt_max_depth", 8),
                random_state=rs,
            )
        label = "DT current-regime"
    else:
        raise ValueError(f"kind must be 'rf' or 'dt', got {kind!r}")

    _tscv_scores(_factory, X, y, n_splits, label)

    final = _factory()
    final.fit(X, y)

    log.info(
        "%s — in-sample report:\n%s",
        label, classification_report(y, final.predict(X), zero_division=0),
    )
    _log_feature_importance(final, X.columns)
    return final


# Convenience aliases kept for call-site readability
def train_current_regime(X: pd.DataFrame, y: pd.Series, cfg: dict) -> RandomForestClassifier:
    """Train a RandomForest to predict today's regime. See train_classifier()."""
    return train_classifier(X, y, cfg, kind="rf")


def train_decision_tree(X: pd.DataFrame, y: pd.Series, cfg: dict) -> DecisionTreeClassifier:
    """Train a shallow DecisionTree to predict today's regime. See train_classifier()."""
    return train_classifier(X, y, cfg, kind="dt")


def train_forward_classifiers(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: dict,
) -> dict[int, dict[int, RandomForestClassifier]]:
    """
    For each (horizon, target_regime) pair, train a binary RandomForest:
        "Will we be in regime R exactly H quarters from now?"

    Uses TimeSeriesSplit CV for evaluation; final model is fitted on all data.

    Returns:
        {horizon: {regime_id: fitted_model}}
    """
    pcfg = cfg["prediction"]
    horizons: list[int] = pcfg.get("forward_horizons_quarters", [1, 2, 4, 8])
    n_splits = pcfg.get("cv_splits", 5)
    n_estimators = pcfg.get("n_estimators", 200)
    rs = pcfg.get("random_state", 42)

    results: dict[int, dict[int, RandomForestClassifier]] = {}

    for h in horizons:
        results[h] = {}
        # Shift labels back by h so X[t] predicts y[t+h]
        y_future = y.shift(-h).dropna().astype(int)
        X_aligned = X.loc[y_future.index]

        for regime in sorted(y.unique()):
            y_binary = (y_future == regime).astype(int)

            def _factory():
                return RandomForestClassifier(
                    n_estimators=n_estimators,
                    random_state=rs,
                    n_jobs=-1,
                    class_weight="balanced",
                )

            scores = _tscv_scores(
                _factory, X_aligned, y_binary, n_splits,
                f"RF h={h}Q regime={regime}",
            )
            log.info(
                "Forward h=%dQ regime=%d — mean CV accuracy=%.3f",
                h, regime, np.mean(scores),
            )

            final = _factory()
            final.fit(X_aligned, y_binary)
            results[h][regime] = final

    return results


# ── inference ──────────────────────────────────────────────────────────────────

def predict_current(model: RandomForestClassifier, X_now: pd.DataFrame) -> dict:
    """
    Score the most recent quarter.

    Returns:
        {"regime": int, "probabilities": {regime_id: prob, …}}
    """
    proba = model.predict_proba(X_now)[-1]
    regime = int(model.classes_[np.argmax(proba)])
    return {
        "regime": regime,
        "probabilities": dict(zip(model.classes_.tolist(), proba.tolist())),
    }
