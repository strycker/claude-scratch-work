"""
Supervised Regime Prediction: train classifiers to predict the current
quarter's regime from features available at that moment.

Key design principle: uses TimeSeriesSplit for cross-validation to
prevent temporal leakage — the #1 risk in financial ML.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.tree import DecisionTreeClassifier

from trading_crab.config import CLUSTERING_FEATURES

log = logging.getLogger(__name__)


def prepare_supervised_data(
    quarterly_df: pd.DataFrame,
    cluster_labels: np.ndarray,
    features: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Prepare feature matrix X and target vector y for supervised learning.

    Args:
        quarterly_df: prepared DataFrame with all features
        cluster_labels: regime labels from clustering (aligned after dropna)
        features: columns to use as predictors (default: CLUSTERING_FEATURES)

    Returns:
        X: feature matrix (n_samples × n_features)
        y: target vector (n_samples,)
        feature_names: list of feature column names
    """
    features = features or CLUSTERING_FEATURES
    available = [f for f in features if f in quarterly_df.columns]

    clean = quarterly_df[available].dropna()
    clean = clean.iloc[:len(cluster_labels)]

    X = clean.values
    y = cluster_labels[:len(X)]

    log.info("Supervised data: %d samples × %d features, %d classes",
             X.shape[0], X.shape[1], len(set(y)))
    return X, y, available


def train_decision_tree(
    X: np.ndarray, y: np.ndarray,
    feature_names: list[str],
    n_splits: int = 5,
) -> tuple[DecisionTreeClassifier, dict]:
    """
    Train a Decision Tree with TimeSeriesSplit cross-validation.

    Starts with a simple tree for interpretability. Returns the model
    fitted on all data plus cross-validation metrics.
    """
    log.info("Training Decision Tree (max_depth=8, %d CV splits)...", n_splits)
    tscv = TimeSeriesSplit(n_splits=n_splits)

    cv_accuracies = []
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        model = DecisionTreeClassifier(max_depth=8, random_state=42)
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[test_idx])
        acc = accuracy_score(y[test_idx], preds)
        cv_accuracies.append(acc)
        log.debug("  Fold %d: accuracy=%.3f", fold + 1, acc)

    # Final model on all data
    final_model = DecisionTreeClassifier(max_depth=8, random_state=42)
    final_model.fit(X, y)

    # Feature importance ranking
    importances = pd.Series(
        final_model.feature_importances_, index=feature_names
    ).sort_values(ascending=False)

    metrics = {
        "cv_accuracies": cv_accuracies,
        "mean_cv_accuracy": np.mean(cv_accuracies),
        "std_cv_accuracy": np.std(cv_accuracies),
        "feature_importances": importances,
    }

    log.info("  CV accuracy: %.3f ± %.3f",
             metrics["mean_cv_accuracy"], metrics["std_cv_accuracy"])
    log.info("  Top 5 features: %s", list(importances.head().index))
    return final_model, metrics


def train_random_forest(
    X: np.ndarray, y: np.ndarray,
    feature_names: list[str],
    n_splits: int = 5,
    n_estimators: int = 200,
) -> tuple[RandomForestClassifier, dict]:
    """
    Train a Random Forest with TimeSeriesSplit cross-validation.

    More powerful than a single tree but less interpretable.
    """
    log.info("Training Random Forest (%d trees, %d CV splits)...",
             n_estimators, n_splits)
    tscv = TimeSeriesSplit(n_splits=n_splits)

    cv_accuracies = []
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=12, random_state=42, n_jobs=-1
        )
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[test_idx])
        acc = accuracy_score(y[test_idx], preds)
        cv_accuracies.append(acc)
        log.debug("  Fold %d: accuracy=%.3f", fold + 1, acc)

    # Final model on all data
    final_model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=12, random_state=42, n_jobs=-1
    )
    final_model.fit(X, y)

    importances = pd.Series(
        final_model.feature_importances_, index=feature_names
    ).sort_values(ascending=False)

    metrics = {
        "cv_accuracies": cv_accuracies,
        "mean_cv_accuracy": np.mean(cv_accuracies),
        "std_cv_accuracy": np.std(cv_accuracies),
        "feature_importances": importances,
    }

    log.info("  CV accuracy: %.3f ± %.3f",
             metrics["mean_cv_accuracy"], metrics["std_cv_accuracy"])
    log.info("  Top 5 features: %s", list(importances.head().index))
    return final_model, metrics


def generate_classification_report(
    model, X: np.ndarray, y: np.ndarray,
    regime_names: dict[int, str] | None = None,
) -> str:
    """
    Generate a full classification report and confusion matrix.

    Args:
        model: fitted classifier
        X, y: data to evaluate on
        regime_names: optional mapping of regime int → human name

    Returns:
        Formatted report string
    """
    preds = model.predict(X)
    labels = sorted(set(y))
    target_names = [regime_names.get(l, f"Regime {l}") for l in labels] if regime_names else None

    report = classification_report(y, preds, target_names=target_names, zero_division=0)
    cm = confusion_matrix(y, preds, labels=labels)

    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.index.name = "actual"
    cm_df.columns.name = "predicted"

    output = f"Classification Report:\n{report}\nConfusion Matrix:\n{cm_df}\n"
    log.info("\n%s", output)
    return output
