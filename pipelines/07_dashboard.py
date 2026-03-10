"""
Pipeline step 7 — Stoplight Dashboard + Portfolio Recommendations

Loads all previously computed artifacts and prints a concise summary:
  - Current predicted regime
  - Asset stoplight signals (GREEN / YELLOW / RED)
  - Forward transition probabilities
  - Portfolio weights (simple + blended)
  - BUY / SELL / HOLD trade recommendations vs all-cash baseline

Features are read from features_supervised.parquet (causal/backward rolling
windows — consistent with how the model was trained in step 5).

Saves to outputs/reports/:
  dashboard.csv              — asset signals
  portfolio_simple.csv       — equal-weight top-3 assets for current regime
  portfolio_blended.csv      — probability-weighted allocation across all regimes
  trade_recommendations.csv  — BUY/SELL/HOLD signals vs all-cash

Run:
    python pipelines/07_dashboard.py
"""

import sys
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from market_regime import DATA_DIR, CONFIG_DIR, OUTPUT_DIR
from market_regime.config import load, setup_logging
from market_regime.prediction import predict_current
from market_regime.asset_returns import rank_assets_by_regime
from market_regime.reporting import (
    asset_signals,
    print_dashboard,
    save_dashboard_csv,
    simple_regime_portfolio,
    blended_regime_portfolio,
    generate_recommendation,
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

    # Load current-regime model
    model_dir = OUTPUT_DIR / "models"
    with open(model_dir / "current_regime.pkl", "rb") as f:
        current_model = pickle.load(f)

    # Use causal features for live scoring — same as training in step 5
    sup_path = DATA_DIR / "processed" / "features_supervised.parquet"
    feat_path = sup_path if sup_path.exists() else DATA_DIR / "processed" / "features.parquet"
    if not sup_path.exists():
        print(
            "WARNING: features_supervised.parquet not found — falling back to features.parquet.\n"
            "Re-run step 2 to generate causal features."
        )
    features = pd.read_parquet(feat_path)
    X = features.drop(columns=["market_code"], errors="ignore")
    if hasattr(current_model, "feature_names_in_"):
        X = X[current_model.feature_names_in_]
    else:
        X = X.dropna(axis=1, how="any")
    prediction = predict_current(current_model, X)

    # Load supporting data
    tm = pd.read_parquet(DATA_DIR / "regimes" / "transition_matrix.parquet")
    regime_names = load_regime_names()
    thresholds = cfg.get("dashboard", {}).get("signal_thresholds", None)

    # ── Asset signals ──────────────────────────────────────────────────────
    asset_signals_df = pd.DataFrame()
    profile_path = DATA_DIR / "regimes" / "asset_return_profile.parquet"
    profile: pd.DataFrame | None = None
    if profile_path.exists():
        profile = pd.read_parquet(profile_path)
        ranked = rank_assets_by_regime(profile)
        asset_signals_df = asset_signals(ranked, prediction["regime"], thresholds=thresholds)

    print_dashboard(prediction, regime_names, asset_signals_df, tm)

    report_dir = OUTPUT_DIR / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    if not asset_signals_df.empty:
        save_dashboard_csv(asset_signals_df, report_dir)

    # ── Portfolio construction ─────────────────────────────────────────────
    if profile is not None and not profile.empty:
        current_regime = prediction["regime"]
        probs = prediction["probabilities"]

        simple_weights = simple_regime_portfolio(profile, current_regime, top_n=3)
        blended_weights = blended_regime_portfolio(profile, probs, top_n=3)
        recommendations = generate_recommendation(blended_weights)

        print("\n── Simple portfolio (top-3 for current regime) ──")
        for asset, w in simple_weights.items():
            print(f"  {asset:<12s}  {w:.1%}")

        print("\n── Blended portfolio (probability-weighted) ──")
        for asset, w in blended_weights.items():
            print(f"  {asset:<12s}  {w:.1%}")

        print("\n── Trade recommendations (blended vs all-cash) ──")
        print(recommendations.to_string())

        if not simple_weights.empty:
            simple_weights.to_frame("weight").to_csv(report_dir / "portfolio_simple.csv")
        if not blended_weights.empty:
            blended_weights.to_frame("weight").to_csv(report_dir / "portfolio_blended.csv")
        if not recommendations.empty:
            recommendations.to_csv(report_dir / "trade_recommendations.csv")
            print(f"\nReports saved to {report_dir}")


if __name__ == "__main__":
    main()
