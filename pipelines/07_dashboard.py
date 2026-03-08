"""
Pipeline step 7 — Stoplight Dashboard

Loads all previously computed artifacts and prints a concise summary:
  - Current predicted regime
  - Asset stoplight signals
  - Forward transition probabilities

Run:
    python pipelines/07_dashboard.py
"""

import sys
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from market_regime import DATA_DIR, CONFIG_DIR, OUTPUT_DIR
from market_regime.config import load, setup_logging
from market_regime.prediction.classifier import predict_current
from market_regime.assets.returns import rank_assets_by_regime
from market_regime.reporting.dashboard import (
    asset_signals,
    print_dashboard,
    save_dashboard_csv,
)

import pandas as pd
import yaml


def load_regime_names() -> dict[int, str]:
    # Prefer manually edited config/regime_labels.yaml, fall back to auto-suggestions
    override_path = CONFIG_DIR / "regime_labels.yaml"
    suggested_path = DATA_DIR / "regimes" / "regime_names_suggested.yaml"

    for path in [override_path, suggested_path]:
        if path.exists():
            with open(path) as f:
                raw = yaml.safe_load(f) or {}
            names = {int(k): v for k, v in raw.items() if not str(k).startswith("#")}
            if names:
                return names
    return {}


def main() -> None:
    setup_logging()
    cfg = load()

    # Load models
    model_dir = OUTPUT_DIR / "models"
    with open(model_dir / "current_regime.pkl", "rb") as f:
        current_model = pickle.load(f)

    # Load features and predict
    features = pd.read_parquet(DATA_DIR / "processed" / "features.parquet")
    X = features.dropna(axis=1, how="any")
    prediction = predict_current(current_model, X)

    # Load supporting data
    tm = pd.read_parquet(DATA_DIR / "regimes" / "transition_matrix.parquet")
    regime_names = load_regime_names()

    # Asset signals (optional — gracefully skip if not yet computed)
    asset_signals_df = pd.DataFrame()
    profile_path = DATA_DIR / "regimes" / "asset_return_profile.parquet"
    if profile_path.exists():
        profile = pd.read_parquet(profile_path)
        # rebuild ranked from profile index
        ranked = rank_assets_by_regime(profile.reset_index())
        asset_signals_df = asset_signals(ranked, prediction["regime"])

    print_dashboard(prediction, regime_names, asset_signals_df, tm)

    # Save CSV snapshot
    if not asset_signals_df.empty:
        save_dashboard_csv(asset_signals_df, OUTPUT_DIR / "reports")


if __name__ == "__main__":
    main()
