"""
Supervised regime classifier.

Trains a RandomForestClassifier to predict the current quarter's regime
from features available at that moment (no look-ahead).

Also trains forward-looking binary classifiers for each horizon defined in
cfg["prediction"]["forward_horizons_quarters"].

Design note: feature importance is printed after each fit — use this to guide
manual feature selection before adding more complex models.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.inspection import permutation_importance

log = logging.getLogger(__name__)


def train_current_regime(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: dict,
) -> RandomForestClassifier:
    """
    Train a classifier to predict today's regime label.

    Args:
        X   — feature matrix (rows = quarters, no future data)
        y   — integer cluster labels aligned to X
        cfg — pipeline config dict

    Returns:
        Fitted RandomForestClassifier.
    """
    pcfg = cfg["prediction"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y,
        test_size=pcfg["test_size"],
        shuffle=False,          # preserve time order — no leakage
        random_state=pcfg["random_state"],
    )

    model = RandomForestClassifier(
        n_estimators=pcfg["n_estimators"],
        random_state=pcfg["random_state"],
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(X_tr, y_tr)

    y_pred = model.predict(X_te)
    log.info("Current-regime classifier — test set report:\n%s",
             classification_report(y_te, y_pred))

    _log_feature_importance(model, X.columns)
    return model


def train_forward_classifiers(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: dict,
) -> dict[int, dict[int, RandomForestClassifier]]:
    """
    For each (horizon, target_regime) pair, train a binary classifier:
    "Will we be in regime R exactly H quarters from now?"

    Returns:
        {horizon: {regime_id: fitted_model}}
    """
    pcfg = cfg["prediction"]
    horizons = pcfg.get("forward_horizons_quarters", [1, 2, 4, 8])
    results: dict[int, dict] = {}

    for h in horizons:
        results[h] = {}
        # Shift labels back by h so that X[t] predicts y[t+h]
        y_future = y.shift(-h).dropna().astype(int)
        X_aligned = X.loc[y_future.index]

        for regime in sorted(y.unique()):
            y_binary = (y_future == regime).astype(int)
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_aligned, y_binary,
                test_size=pcfg["test_size"],
                shuffle=False,
                random_state=pcfg["random_state"],
            )
            model = RandomForestClassifier(
                n_estimators=pcfg["n_estimators"],
                random_state=pcfg["random_state"],
                n_jobs=-1,
                class_weight="balanced",
            )
            model.fit(X_tr, y_tr)
            acc = model.score(X_te, y_te)
            log.info("Forward classifier h=%dQ regime=%d  accuracy=%.3f", h, regime, acc)
            results[h][regime] = model

    return results


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


def _log_feature_importance(model: RandomForestClassifier, feature_names) -> None:
    importances = pd.Series(model.feature_importances_, index=feature_names)
    top = importances.sort_values(ascending=False).head(15)
    lines = "\n".join(f"  {f:<40s} {v:.4f}" for f, v in top.items())
    log.info("Top-15 feature importances:\n%s", lines)
