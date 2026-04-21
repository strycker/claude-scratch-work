"""
LightGBM-based regime classifiers.

Provides the same interface as the flat API in ``prediction/__init__.py``
but uses LightGBM gradient-boosted trees instead of scikit-learn RandomForest.

LightGBM is preferred over XGBoost for this dataset (~300 observations):
faster training, lower memory, and comparable accuracy on small tabular data.

Usage:
    from trading_crab_lib.prediction.gradient_boosting import (
        train_lightgbm_current_regime,
        train_lightgbm_forward,
    )

Both functions return bare fitted ``lgb.LGBMClassifier`` objects, consistent
with the flat-API convention that pickled models are bare estimators.

Requires: ``pip install lightgbm>=4.0``  (declared as optional extra).
"""

from __future__ import annotations

from typing import Any
import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

try:
    import lightgbm as lgb  # type: ignore[import]
    _LGBM_AVAILABLE = True
except ImportError:
    lgb = None  # type: ignore[assignment]
    _LGBM_AVAILABLE = False

log = logging.getLogger(__name__)


def _get_lgbm_params(cfg: dict) -> dict:
    """Build LightGBM parameters from pipeline config with sensible defaults."""
    pcfg = cfg.get("prediction", {})
    return {
        "num_leaves": pcfg.get("lgbm_num_leaves", 15),
        "max_depth": pcfg.get("lgbm_max_depth", 5),
        "min_child_samples": pcfg.get("lgbm_min_child_samples", 5),
        "learning_rate": pcfg.get("lgbm_learning_rate", 0.05),
        "n_estimators": pcfg.get("lgbm_n_estimators", 300),
        "subsample": pcfg.get("lgbm_bagging_fraction", 0.8),
        "colsample_bytree": pcfg.get("lgbm_feature_fraction", 0.8),
        "reg_lambda": pcfg.get("lgbm_lambda_l2", 1.0),
        "class_weight": "balanced",
        "random_state": pcfg.get("random_state", 42),
        "verbose": -1,
    }


def _tscv_scores_lgbm(
    params: dict,
    features_df: pd.DataFrame,
    y: pd.Series,
    n_splits: int,
    label: str,
) -> list[float]:
    """Run TimeSeriesSplit CV with LightGBM and return per-fold accuracy scores."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores: list[float] = []
    for fold, (train_idx, test_idx) in enumerate(tscv.split(features_df), start=1):
        m = lgb.LGBMClassifier(**params)
        m.fit(features_df.iloc[train_idx], y.iloc[train_idx])
        acc = m.score(features_df.iloc[test_idx], y.iloc[test_idx])
        scores.append(acc)
        log.debug("%s fold %d/%d: accuracy=%.3f", label, fold, n_splits, acc)
    log.info(
        "%s CV accuracy: %.3f ± %.3f  (n_splits=%d)",
        label, np.mean(scores), np.std(scores), n_splits,
    )
    return scores


def train_lightgbm_current_regime(
    features_df: pd.DataFrame,
    y: pd.Series,
    cfg: dict[str, Any],
) -> Any:
    """
    Train a LightGBM classifier to predict the current regime label.

    Args:
        features_df — feature matrix (rows = quarters, causal features only)
        y           — integer cluster labels aligned to features_df
        cfg         — pipeline config dict

    Returns:
        Fitted ``lgb.LGBMClassifier`` (trained on all data).
    """
    if not _LGBM_AVAILABLE:
        raise ImportError("lightgbm is not installed. Run: pip install lightgbm>=4.0")

    params = _get_lgbm_params(cfg)
    n_splits = cfg.get("prediction", {}).get("cv_splits", 5)

    _tscv_scores_lgbm(params, features_df, y, n_splits, "LGBM current-regime")

    final = lgb.LGBMClassifier(**params)
    final.fit(features_df, y)

    importances = pd.Series(final.feature_importances_, index=features_df.columns)
    top = importances.sort_values(ascending=False).head(15)
    lines = "\n".join(f"  {f:<40s} {v:.0f}" for f, v in top.items())
    log.info("LGBM top-15 feature importances:\n%s", lines)

    return final


def train_lightgbm_forward(
    features_df: pd.DataFrame,
    y: pd.Series,
    cfg: dict[str, Any],
) -> dict[int, dict[int, object]]:
    """
    For each (horizon, target_regime) pair, train a binary LightGBM classifier:
        "Will we be in regime R exactly H quarters from now?"

    Returns:
        {horizon: {regime_id: fitted_lgb.LGBMClassifier}}
    """
    if not _LGBM_AVAILABLE:
        raise ImportError("lightgbm is not installed. Run: pip install lightgbm>=4.0")

    params = _get_lgbm_params(cfg)
    pcfg = cfg.get("prediction", {})
    horizons: list[int] = pcfg.get("forward_horizons_quarters", [1, 2, 4, 8])
    n_splits = pcfg.get("cv_splits", 5)

    results: dict[int, dict[int, object]] = {}

    for h in horizons:
        results[h] = {}
        y_future = y.shift(-h).dropna().astype(int)
        features_aligned = features_df.loc[y_future.index]

        for regime in sorted(y.unique()):
            y_binary = (y_future == regime).astype(int)

            binary_params = {**params}
            # Binary classification doesn't need balanced multiclass weights
            binary_params.pop("class_weight", None)

            scores = _tscv_scores_lgbm(
                binary_params, features_aligned, y_binary, n_splits,
                f"LGBM h={h}Q regime={regime}",
            )
            log.info(
                "LGBM forward h=%dQ regime=%d — mean CV accuracy=%.3f",
                h, regime, np.mean(scores),
            )

            final = lgb.LGBMClassifier(**binary_params)
            final.fit(features_aligned, y_binary)
            results[h][regime] = final

    return results
