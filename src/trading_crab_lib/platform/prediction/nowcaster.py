"""
L2 calibrated nowcaster (L2-01, design §5.1, §14 Phase 1).

Trains a calibrated multinomial classifier on causal ``monthly_features``
columns against the L1 labeler's output, with the trailing 12 months of
labels structurally excluded from ever being a training target (D-01/L1-02).
Reports transition-window accuracy alongside overall accuracy — the
persistence-trap warning from design §5.1 (a trivial persistence classifier
scores ~90% overall and hides at exactly the moments that matter).

Two distinct "embargo" concepts appear in this module and must never be
merged (RESEARCH.md Pitfall 4):

- ``build_nowcaster_training_set``'s ``embargo_months`` (D-01): applied
  ONCE, when building (X, y), physically excludes the trailing N months of
  labels from ever becoming a training target.
- ``fit_nowcaster``'s ``label_horizon``/``embargo`` (Phase 2,
  ``PurgedEmbargoedKFold``): applied EVERY fold, inside calibration CV, to
  purge/embargo potential feature-window overlap within the
  already-embargoed training set.

Usage::

    from trading_crab_lib.platform.prediction.nowcaster import (
        build_nowcaster_training_set,
        fit_nowcaster,
        transition_window_accuracy,
        evaluate_nowcaster,
    )

    X, y = build_nowcaster_training_set(monthly_features, regime_labels)
    model = fit_nowcaster(X, y)
    proba = model.predict_proba(X.iloc[[-1]])  # distribution, never argmax
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

from trading_crab_lib.platform.checkpoints import get_platform_checkpoint_manager
from trading_crab_lib.platform.honesty import registry
from trading_crab_lib.platform.honesty.cv import PurgedEmbargoedKFold
from trading_crab_lib.platform.honesty.gating import assert_causal_features


def build_nowcaster_training_set(
    features_df: pd.DataFrame,
    labels: pd.Series,
    *,
    embargo_months: int = 12,
) -> tuple[pd.DataFrame, pd.Series]:
    """Physically exclude the trailing *embargo_months* of labels (D-01).

    Fresh trailing labels are revision-prone; training on them is look-ahead
    leakage (T-03-04). ``embargo_months`` is applied ONCE here, not per CV
    fold — see module docstring for the distinction from
    :func:`fit_nowcaster`'s per-fold ``PurgedEmbargoedKFold`` embargo.

    Returns:
        tuple[pd.DataFrame, pd.Series]: index-aligned ``(X, y)`` — the
        intersection of ``features_df.index`` and the eligible (post-embargo)
        label index.

    Raises:
        ValueError: if ``embargo_months < 0`` — raised before any slicing
            (T-03-07: prevents a silently empty/inverted training set).
    """
    if embargo_months < 0:
        raise ValueError("embargo_months must be >= 0")
    cutoff = labels.index.max() - pd.DateOffset(months=embargo_months)
    eligible_labels = labels.loc[labels.index <= cutoff]
    common = features_df.index.intersection(eligible_labels.index)
    X = features_df.loc[common]
    y = eligible_labels.loc[common]
    return X, y


def fit_nowcaster(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    label_horizon: int = 12,
    embargo: int = 1,
    n_splits: int = 5,
    random_state: int = 42,
) -> CalibratedClassifierCV:
    """Fit a calibrated multinomial LogisticRegression through PurgedEmbargoedKFold.

    ``label_horizon``/``embargo`` are the PER-FOLD calibration-CV purge/embargo
    widths (Phase 2's ``cv.default_label_horizon_months`` /
    ``cv.default_embargo_months``) — a DIFFERENT concern from D-01's
    once-applied structural label embargo already handled by
    :func:`build_nowcaster_training_set` (see module docstring).

    The base estimator is a plain ``LogisticRegression`` — NO ``multi_class``
    kwarg (removed from the installed sklearn 1.9.0 API; multinomial
    behavior is automatic for the default ``lbfgs`` solver). Calibration uses
    ``method="sigmoid"`` (isotonic overfits the calibration map at this
    sample size).

    Returns:
        CalibratedClassifierCV: fitted model. ``predict_proba(X)`` returns a
        ``(len(X), n_classes)`` distribution — callers must use
        ``predict_proba``, never ``predict()``/argmax alone (L2-01).
    """
    assert_causal_features(X.columns)  # guards the nowcaster's INPUTS —
    # distinct from build_nowcaster_training_set's label embargo above.
    cv = PurgedEmbargoedKFold(n_splits=n_splits, label_horizon=label_horizon, embargo=embargo)
    model = CalibratedClassifierCV(
        LogisticRegression(max_iter=1000, random_state=random_state),
        method="sigmoid",
        cv=cv,
    )
    model.fit(X, y)
    return model


def evaluate_nowcaster(
    features_df: pd.DataFrame,
    labels: pd.Series,
    *,
    embargo_months: int = 12,
    min_train: int = 60,
    registry_path: Path | str | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Run the honest nowcaster evaluation and log exactly one registry trial.

    Builds the embargoed training set (D-01), fits the calibrated nowcaster,
    and reports transition-window + overall accuracy. Every call logs
    exactly one row to the trial registry (T-03-05), and persists the fitted
    model via joblib (never raw pickle — P27).

    ``min_train`` is recorded in the logged trial config for traceability;
    v1 evaluates against the full embargoed set rather than re-fitting at
    every step of an expanding walk-forward window (impractical here — five
    calibration-CV folds need enough per-class rows per fold, which an early
    small expanding-window slice cannot guarantee at K=5 classes). The
    embargo (D-01) and per-fold purge/embargo (PurgedEmbargoedKFold) still
    provide the leakage guarantees; a full walk-forward refit loop is future
    work, not required by this phase's must_haves.
    """
    X, y = build_nowcaster_training_set(features_df, labels, embargo_months=embargo_months)
    model = fit_nowcaster(X, y)
    y_pred = pd.Series(model.predict(X), index=y.index)
    # Overall accuracy only for now; transition_window_accuracy() (added in
    # Task 3) replaces this with the full overall/transition/steady-state
    # dict so the persistence-trap is never presented alone.
    metrics = {"overall_accuracy": float(y.eq(y_pred).mean())}

    trial_config = dict(config or {})
    trial_config.update(
        {
            "model_class": "CalibratedClassifierCV",
            "base_estimator": "LogisticRegression",
            "embargo_months": embargo_months,
            "min_train": min_train,
        }
    )
    registry.append_trial(
        config=trial_config,
        features=list(X.columns),
        metrics=metrics,
        path=registry_path,
    )

    get_platform_checkpoint_manager().save_model(model, "nowcaster")
    return metrics
