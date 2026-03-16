"""
Pipeline step 5 — Supervised Regime Prediction

Trains:
  1. RandomForestClassifier  — high accuracy; used for production predictions.
  2. DecisionTreeClassifier  — shallow (max_depth=8); human-readable rules and
                               fast feature-importance inspection.
  3. Forward-looking binary classifiers for each regime × horizon pair.

All models use TimeSeriesSplit walk-forward cross-validation so CV accuracy
estimates reflect genuine out-of-sample performance.

Features are read from features_supervised.parquet (causal/backward rolling
windows — no future data leaks into any feature value).

Saves fitted models to outputs/models/.

Run:
    python pipelines/05_predict.py
"""

import sys
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd

from market_regime import DATA_DIR, OUTPUT_DIR
from market_regime.config import load, setup_logging
from market_regime.prediction.classifier import (
    train_current_regime,
    train_forward_classifiers,
)
from market_regime.transforms import trim_incomplete_tail


def main() -> None:
    setup_logging()
    cfg = load()

    # Use causal features — no look-ahead bias for supervised learning.
    # We now require features_supervised.parquet to exist; this avoids
    # accidentally training on non-causal features.
    sup_path = DATA_DIR / "processed" / "features_supervised.parquet"
    if not sup_path.exists():
        raise RuntimeError(
            "features_supervised.parquet not found in data/processed.\n"
            "Run step 2 (pipelines/02_features.py or run_pipeline.py --steps 2,3,4,5)"
            " to generate causal features before running step 5."
        )
    features = pd.read_parquet(sup_path)
    labels = pd.read_parquet(DATA_DIR / "regimes" / "cluster_labels.parquet")["balanced_cluster"]

    common = features.index.intersection(labels.index)
    drop_tail: bool = cfg.get("data", {}).get("drop_incomplete_tail", True)
    X_raw = features.loc[common].drop(columns=["market_code"], errors="ignore")
    # trim_incomplete_tail removes the trailing quarter(s) where centered
    # np.gradient leaves NaN in derivative columns (edge effect).
    # dropna(axis=0) removes any remaining rows with interior NaN.
    X = trim_incomplete_tail(X_raw, enabled=drop_tail).dropna(axis=0, how="any")
    y = labels.loc[X.index]

    # ── Current-regime classifiers (DT + RF bundle) ───────────────────────
    cv_splits = cfg.get("prediction", {}).get("cv_splits", 5)
    current_bundle = train_current_regime(X, y, cv_splits=cv_splits)

    models = current_bundle["models"]
    labels = current_bundle["labels"]

    # Use the RandomForest as the primary production model.
    rf = models["rf"]
    latest_X = X.iloc[[-1]]
    proba = rf.predict_proba(latest_X)[0]
    classes = rf.classes_

    # Map class → probability and pick argmax as current regime.
    prob_by_class = dict(zip(classes, proba))
    best_regime = max(prob_by_class.items(), key=lambda kv: kv[1])[0]

    print(f"\nLatest quarter prediction: regime {best_regime}")
    for r, p in sorted(prob_by_class.items(), key=lambda x: -x[1]):
        print(f"  Regime {r}: {p:.1%}")

    # ── Forward regime classifiers ────────────────────────────────────────
    horizons = cfg.get("prediction", {}).get("forward_horizons_quarters", [1, 2, 4, 8])
    forward_models = train_forward_classifiers(X, y, horizons=horizons, cv_splits=cv_splits)

    # ── Persist models ────────────────────────────────────────────────────
    model_dir = OUTPUT_DIR / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    with open(model_dir / "current_regime.pkl", "wb") as f:
        pickle.dump(current_bundle, f)
    with open(model_dir / "forward_classifiers.pkl", "wb") as f:
        pickle.dump(forward_models, f)

    print(f"\nModels saved to {model_dir}")


if __name__ == "__main__":
    main()
