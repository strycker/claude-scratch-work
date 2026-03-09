"""
Pipeline step 5 — Supervised Regime Prediction

Trains:
  1. Current-regime classifier (predict today's regime from real-time features)
  2. Forward-looking binary classifiers for each regime × horizon pair

Prints feature importances and test-set accuracy.
Saves fitted models to outputs/models/.

Run:
    python pipelines/05_predict.py
"""

import sys
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from market_regime import DATA_DIR, OUTPUT_DIR
from market_regime.config import load, setup_logging
from market_regime.prediction.classifier import (
    train_current_regime,
    train_forward_classifiers,
    predict_current,
)

import pandas as pd


def main() -> None:
    setup_logging()
    cfg = load()

    features = pd.read_parquet(DATA_DIR / "processed" / "features.parquet")
    labels = pd.read_parquet(DATA_DIR / "regimes" / "cluster_labels.parquet")["balanced_cluster"]

    common = features.index.intersection(labels.index)
    X = features.loc[common].dropna(axis=1, how="any")  # drop cols still NaN after gap fill
    y = labels.loc[common]

    # ── Current-regime classifier ──────────────────────────────────────────
    current_model = train_current_regime(X, y, cfg)

    # Score on the most recent available quarter
    latest = predict_current(current_model, X)
    print(f"\nLatest quarter prediction: regime {latest['regime']}")
    for r, p in sorted(latest["probabilities"].items(), key=lambda x: -x[1]):
        print(f"  Regime {r}: {p:.1%}")

    # ── Forward classifiers ───────────────────────────────────────────────
    forward_models = train_forward_classifiers(X, y, cfg)

    # ── Persist models ────────────────────────────────────────────────────
    model_dir = OUTPUT_DIR / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    with open(model_dir / "current_regime.pkl", "wb") as f:
        pickle.dump(current_model, f)
    with open(model_dir / "forward_classifiers.pkl", "wb") as f:
        pickle.dump(forward_models, f)

    print(f"\nModels saved to {model_dir}")


if __name__ == "__main__":
    main()
