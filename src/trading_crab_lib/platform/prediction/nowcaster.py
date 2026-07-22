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

import pandas as pd


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
