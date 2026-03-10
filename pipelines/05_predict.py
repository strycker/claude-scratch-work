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

from market_regime import DATA_DIR, OUTPUT_DIR
from market_regime.config import load, setup_logging
from market_regime.prediction import (
    train_current_regime,
    train_decision_tree,
    train_forward_classifiers,
    predict_current,
)
from market_regime.transforms import trim_incomplete_tail

import pandas as pd


def main() -> None:
    setup_logging()
    cfg = load()

    # Use causal features — no look-ahead bias for supervised learning
    sup_path = DATA_DIR / "processed" / "features_supervised.parquet"
    feat_path = sup_path if sup_path.exists() else DATA_DIR / "processed" / "features.parquet"
    if not sup_path.exists():
        print(
            "WARNING: features_supervised.parquet not found — falling back to features.parquet.\n"
            "Re-run step 2 to generate causal features."
        )
    features = pd.read_parquet(feat_path)
    labels = pd.read_parquet(DATA_DIR / "regimes" / "cluster_labels.parquet")["balanced_cluster"]

    common = features.index.intersection(labels.index)
    drop_tail: bool = cfg.get("data", {}).get("drop_incomplete_tail", True)
    X_raw = features.loc[common].drop(columns=["market_code"], errors="ignore")
    # trim_incomplete_tail removes the trailing quarter(s) where centered
    # np.gradient leaves NaN in derivative columns (edge effect).
    # dropna(axis=0) removes any remaining rows with interior NaN.
    X = trim_incomplete_tail(X_raw, enabled=drop_tail).dropna(axis=0, how="any")
    y = labels.loc[X.index]

    # ── Current-regime classifiers ─────────────────────────────────────────
    current_model = train_current_regime(X, y, cfg)
    dt_model = train_decision_tree(X, y, cfg)

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
    with open(model_dir / "decision_tree.pkl", "wb") as f:
        pickle.dump(dt_model, f)
    with open(model_dir / "forward_classifiers.pkl", "wb") as f:
        pickle.dump(forward_models, f)

    print(f"\nModels saved to {model_dir}")


if __name__ == "__main__":
    main()
