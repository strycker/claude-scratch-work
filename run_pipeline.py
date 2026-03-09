"""
run_pipeline.py — Master entry point for the Trading-Crab pipeline.

Runs all 7 pipeline steps in order (or a selected subset) with consistent
runtime flags passed through to every module via RunConfig.

Usage:
    python run_pipeline.py --refresh --recompute --plots
    python run_pipeline.py --steps 3,4,5 --plots --verbose
    python run_pipeline.py --steps 1,2 --refresh
    python run_pipeline.py  # load checkpoints, run all, no plots

CLI flags:
    --refresh       Re-scrape multpl.com + re-hit FRED API (~10 min)
    --recompute     Recompute features from cached raw data
    --plots         Generate matplotlib figures → outputs/plots/
    --verbose       Set logging level to DEBUG
    --steps 1,3,5   Run only these step numbers (comma-separated)
    --no-constrained Skip k-means-constrained (if package not installed)
    --show-plots    Call plt.show() after each figure (off by default)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running from repo root without `pip install -e .`
sys.path.insert(0, str(Path(__file__).parent / "src"))

from market_regime import DATA_DIR, OUTPUT_DIR, CONFIG_DIR
from market_regime.config import load, setup_logging
from market_regime.runtime import RunConfig

log = logging.getLogger(__name__)


# ── Step registry ──────────────────────────────────────────────────────────────

def step1_ingest(cfg: dict, run_cfg: RunConfig) -> None:
    """Scrape multpl.com + FRED → data/raw/macro_raw.parquet"""
    from market_regime.ingestion import fred as fred_module
    from market_regime.ingestion import multpl as multpl_module
    from market_regime.io.checkpoints import CheckpointManager
    import pandas as pd

    cm = CheckpointManager()

    if not run_cfg.refresh_source_datasets and cm.is_fresh("macro_raw", max_age_days=7):
        log.info("Step 1: using cached macro_raw checkpoint")
        return

    log.info("Step 1: fetching FRED data …")
    fred_df = fred_module.fetch_all(cfg)

    log.info("Step 1: scraping multpl.com (%d series) …",
             len(cfg["multpl"]["datasets"]))
    multpl_df = multpl_module.fetch_all(cfg)

    combined = fred_df.join(multpl_df, how="outer") if not multpl_df.empty else fred_df
    start = cfg["data"]["start_date"]
    combined = combined[combined.index >= start]

    raw_dir = DATA_DIR / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(raw_dir / "macro_raw.parquet")
    cm.save(combined, "macro_raw")

    log.info("Step 1 done: %d rows × %d cols", len(combined), len(combined.columns))


def step2_features(cfg: dict, run_cfg: RunConfig) -> None:
    """Engineer features from macro_raw → data/processed/features.parquet"""
    from market_regime.features.transforms import engineer_all
    from market_regime.io.checkpoints import CheckpointManager
    import pandas as pd

    cm = CheckpointManager()

    if not run_cfg.recompute_derived_datasets and cm.is_fresh("features", max_age_days=7):
        log.info("Step 2: using cached features checkpoint")
        return

    raw_path = DATA_DIR / "raw" / "macro_raw.parquet"
    if not raw_path.exists():
        raise FileNotFoundError(
            f"macro_raw.parquet not found — run step 1 first: {raw_path}"
        )
    raw = pd.read_parquet(raw_path)

    log.info("Step 2: engineering features from %d × %d raw data …",
             len(raw), len(raw.columns))
    features = engineer_all(raw, cfg)

    out_dir = DATA_DIR / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    features.to_parquet(out_dir / "features.parquet")
    cm.save(features, "features")

    log.info("Step 2 done: %d rows × %d feature cols", len(features), len(features.columns))


def step3_cluster(cfg: dict, run_cfg: RunConfig) -> None:
    """PCA + KMeans clustering → data/regimes/cluster_labels.parquet"""
    from market_regime.clustering.kmeans import (
        reduce_pca, evaluate_kmeans, pick_best_k, fit_clusters,
    )
    from market_regime.io.checkpoints import CheckpointManager
    from market_regime import plotting
    from sklearn.preprocessing import StandardScaler
    import pandas as pd

    cm = CheckpointManager()
    clust_cfg = cfg["clustering"]

    if (not run_cfg.recompute_derived_datasets
            and cm.is_fresh("cluster_labels", max_age_days=7)):
        log.info("Step 3: using cached cluster_labels checkpoint")
        return

    features = pd.read_parquet(DATA_DIR / "processed" / "features.parquet")
    X = features.drop(columns=["market_code"], errors="ignore")

    pca_df, pca_model, scaler = reduce_pca(
        X,
        n_components=clust_cfg["n_pca_components"],
        random_state=clust_cfg["random_state"],
    )

    X_scaled = StandardScaler().fit_transform(pca_df.values)
    scores = evaluate_kmeans(
        X_scaled,
        k_range=range(2, clust_cfg["n_clusters_search"] + 1),
        random_state=clust_cfg["random_state"],
    )
    best_k = pick_best_k(scores, k_cap=clust_cfg["k_cap"])

    log.info("K-sweep: chose k=%d  (cap=%d)", best_k, clust_cfg["k_cap"])

    clustered = fit_clusters(
        pca_df,
        best_k=best_k,
        balanced_k=clust_cfg["balanced_k"],
        random_state=clust_cfg["random_state"],
        use_constrained=run_cfg.use_constrained_kmeans,
    )

    if "market_code" in features.columns:
        clustered["market_code"] = features["market_code"]

    out_dir = DATA_DIR / "regimes"
    out_dir.mkdir(parents=True, exist_ok=True)

    label_cols = ["cluster", "balanced_cluster"] + (
        ["market_code"] if "market_code" in clustered.columns else []
    )
    clustered[label_cols].to_parquet(out_dir / "cluster_labels.parquet")
    clustered.drop(columns=label_cols, errors="ignore").to_parquet(
        out_dir / "pca_components.parquet"
    )
    scores.to_parquet(out_dir / "kmeans_scores.parquet", index=False)

    cm.save(clustered[label_cols], "cluster_labels")
    cm.save(pca_df, "pca_components")

    if run_cfg.generate_plots:
        regime_names: dict[int, str] = {}  # populated in step 4; use IDs for now
        plotting.plot_pca_scatter(pca_df, clustered["balanced_cluster"], regime_names, run_cfg)
        plotting.plot_elbow_curve(scores, best_k, run_cfg)

    log.info("Step 3 done: balanced_k=%d", clust_cfg["balanced_k"])


def step4_regime_label(cfg: dict, run_cfg: RunConfig) -> None:
    """Profile clusters → data/regimes/profiles.parquet + transition_matrix.parquet"""
    from market_regime.regime.profiler import (
        build_profiles, suggest_names, build_transition_matrix, load_name_overrides,
    )
    from market_regime.io.checkpoints import CheckpointManager
    from market_regime import plotting
    import pandas as pd
    import yaml

    cm = CheckpointManager()

    features = pd.read_parquet(DATA_DIR / "processed" / "features.parquet")
    labels = pd.read_parquet(DATA_DIR / "regimes" / "cluster_labels.parquet")["balanced_cluster"]

    common = features.index.intersection(labels.index)
    features = features.loc[common]
    labels = labels.loc[common]

    profile = build_profiles(features, labels)
    profile.to_parquet(DATA_DIR / "regimes" / "profiles.parquet")

    auto_names = suggest_names(features, labels)
    overrides = load_name_overrides(CONFIG_DIR)
    regime_names = {**auto_names, **overrides}

    suggestions_path = DATA_DIR / "regimes" / "regime_names_suggested.yaml"
    with open(suggestions_path, "w") as f:
        yaml.dump(regime_names, f, default_flow_style=False)

    tm = build_transition_matrix(labels)
    tm.to_parquet(DATA_DIR / "regimes" / "transition_matrix.parquet")

    if run_cfg.generate_plots:
        plotting.plot_transition_matrix(tm, regime_names, run_cfg)
        plotting.plot_regime_timeline(labels, regime_names, run_cfg)

    for rid, name in sorted(regime_names.items()):
        n = (labels == rid).sum()
        log.info("Cluster %d: %r  (%d quarters)", rid, name, n)

    log.info("Step 4 done")


def step5_predict(cfg: dict, run_cfg: RunConfig) -> None:
    """Train supervised classifiers → outputs/models/"""
    from market_regime.prediction.classifier import (
        train_current_regime, train_forward_classifiers, predict_current,
    )
    from market_regime import plotting
    import pandas as pd
    import pickle

    features = pd.read_parquet(DATA_DIR / "processed" / "features.parquet")
    labels = pd.read_parquet(DATA_DIR / "regimes" / "cluster_labels.parquet")["balanced_cluster"]

    common = features.index.intersection(labels.index)
    X = features.loc[common].dropna(axis=1, how="any")
    y = labels.loc[common]

    current_model = train_current_regime(X, y, cfg)

    latest = predict_current(current_model, X)
    log.info("Latest quarter → regime %d", latest["regime"])
    for r, p in sorted(latest["probabilities"].items(), key=lambda x: -x[1]):
        log.info("  Regime %d: %.1f%%", r, p * 100)

    forward_models = train_forward_classifiers(X, y, cfg)

    model_dir = OUTPUT_DIR / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    with open(model_dir / "current_regime.pkl", "wb") as f:
        pickle.dump(current_model, f)
    with open(model_dir / "forward_classifiers.pkl", "wb") as f:
        pickle.dump(forward_models, f)

    if run_cfg.generate_plots:
        try:
            regime_names_path = DATA_DIR / "regimes" / "regime_names_suggested.yaml"
            import yaml
            regime_names = {}
            if regime_names_path.exists():
                with open(regime_names_path) as f:
                    regime_names = yaml.safe_load(f) or {}
                regime_names = {int(k): v for k, v in regime_names.items()}
            plotting.plot_feature_importance(current_model, X.columns.tolist(), run_cfg)
            plotting.plot_forward_probabilities(latest, regime_names, run_cfg)
        except Exception as exc:
            log.warning("Could not generate prediction plots: %s", exc)

    log.info("Step 5 done — models saved to %s", model_dir)


def step6_asset_returns(cfg: dict, run_cfg: RunConfig) -> None:
    """Fetch ETF prices via yfinance → data/regimes/asset_return_profile.parquet"""
    from market_regime.ingestion.assets import fetch_all as fetch_prices
    from market_regime.ingestion.assets import compute_quarterly_returns
    from market_regime.assets.returns import returns_by_regime, rank_assets_by_regime
    from market_regime.io.checkpoints import CheckpointManager
    from market_regime import plotting
    import pandas as pd

    cm = CheckpointManager()

    labels = pd.read_parquet(DATA_DIR / "regimes" / "cluster_labels.parquet")["balanced_cluster"]

    # Try yfinance first; fall back to cached CSV if present
    prices: pd.DataFrame | None = None
    cache_path = DATA_DIR / "raw" / "asset_prices.parquet"
    csv_path = DATA_DIR / "raw" / "asset_prices.csv"

    if run_cfg.refresh_source_datasets or not cache_path.exists():
        try:
            prices = fetch_prices(cfg)
            if not prices.empty:
                raw_dir = DATA_DIR / "raw"
                raw_dir.mkdir(parents=True, exist_ok=True)
                prices.to_parquet(cache_path)
                cm.save(prices, "asset_prices")
        except Exception as exc:
            log.warning("yfinance fetch failed: %s", exc)

    if (prices is None or prices.empty) and cache_path.exists():
        prices = pd.read_parquet(cache_path)
    elif (prices is None or prices.empty) and csv_path.exists():
        prices = pd.read_csv(csv_path, index_col="date", parse_dates=True)

    if prices is None or prices.empty:
        log.warning("Step 6: no asset price data — skipping")
        return

    returns = compute_quarterly_returns(prices)
    common = returns.index.intersection(labels.index)
    profile = returns_by_regime(returns.loc[common], labels.loc[common])
    ranked = rank_assets_by_regime(profile)

    out_dir = DATA_DIR / "regimes"
    out_dir.mkdir(parents=True, exist_ok=True)
    profile.to_parquet(out_dir / "asset_return_profile.parquet")

    if run_cfg.generate_plots:
        try:
            regime_names_path = DATA_DIR / "regimes" / "regime_names_suggested.yaml"
            import yaml
            regime_names = {}
            if regime_names_path.exists():
                with open(regime_names_path) as f:
                    regime_names = yaml.safe_load(f) or {}
                regime_names = {int(k): v for k, v in regime_names.items()}
            plotting.plot_asset_returns_by_regime(profile, regime_names, run_cfg)
            plotting.plot_asset_heatmap(profile, regime_names, run_cfg)
        except Exception as exc:
            log.warning("Could not generate asset plots: %s", exc)

    log.info("Step 6 done — asset return profile written")


def step7_dashboard(cfg: dict, run_cfg: RunConfig) -> None:
    """Print + save stoplight dashboard → outputs/reports/dashboard.csv"""
    from market_regime.prediction.classifier import predict_current
    from market_regime.assets.returns import rank_assets_by_regime
    from market_regime.reporting.dashboard import (
        asset_signals, print_dashboard, save_dashboard_csv,
    )
    import pandas as pd
    import pickle
    import yaml

    model_dir = OUTPUT_DIR / "models"
    current_model_path = model_dir / "current_regime.pkl"
    if not current_model_path.exists():
        log.warning("Step 7: current_regime.pkl not found — run step 5 first")
        return

    with open(current_model_path, "rb") as f:
        current_model = pickle.load(f)

    features = pd.read_parquet(DATA_DIR / "processed" / "features.parquet")
    X = features.dropna(axis=1, how="any")
    prediction = predict_current(current_model, X)

    tm = pd.read_parquet(DATA_DIR / "regimes" / "transition_matrix.parquet")

    # Load regime names
    override_path = CONFIG_DIR / "regime_labels.yaml"
    suggested_path = DATA_DIR / "regimes" / "regime_names_suggested.yaml"
    regime_names: dict[int, str] = {}
    for path in [override_path, suggested_path]:
        if path.exists():
            with open(path) as f:
                raw = yaml.safe_load(f) or {}
            names = {int(k): v for k, v in raw.items() if not str(k).startswith("#")}
            if names:
                regime_names = names
                break

    asset_signals_df = pd.DataFrame()
    profile_path = DATA_DIR / "regimes" / "asset_return_profile.parquet"
    if profile_path.exists():
        profile = pd.read_parquet(profile_path)
        ranked = rank_assets_by_regime(profile.reset_index())
        asset_signals_df = asset_signals(ranked, prediction["regime"])

    print_dashboard(prediction, regime_names, asset_signals_df, tm)

    if not asset_signals_df.empty:
        save_dashboard_csv(asset_signals_df, OUTPUT_DIR / "reports")

    log.info("Step 7 done")


# ── Step dispatch table ────────────────────────────────────────────────────────

STEPS: dict[int, tuple[str, callable]] = {
    1: ("Ingest macro data",            step1_ingest),
    2: ("Engineer features",            step2_features),
    3: ("PCA + clustering",             step3_cluster),
    4: ("Regime profiling + labeling",  step4_regime_label),
    5: ("Supervised prediction",        step5_predict),
    6: ("Asset returns",                step6_asset_returns),
    7: ("Dashboard",                    step7_dashboard),
}


# ── CLI ────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Trading-Crab market regime pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--refresh", action="store_true",
                   help="Re-scrape multpl.com + re-hit FRED API")
    p.add_argument("--recompute", action="store_true",
                   help="Recompute features from cached raw data")
    p.add_argument("--plots", action="store_true",
                   help="Generate and save matplotlib figures")
    p.add_argument("--show-plots", action="store_true",
                   help="Call plt.show() after each figure")
    p.add_argument("--pairplot", action="store_true",
                   help="Generate seaborn pairplot (slow)")
    p.add_argument("--verbose", action="store_true",
                   help="Set logging to DEBUG")
    p.add_argument("--steps", type=str, default=None,
                   help="Comma-separated step numbers to run, e.g. 1,3,5")
    p.add_argument("--no-constrained", action="store_true",
                   help="Skip k-means-constrained (if package not installed)")
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    setup_logging()
    run_cfg = RunConfig.from_args(args)
    run_cfg.apply_logging()

    cfg = load()

    # Determine which steps to run
    if args.steps:
        try:
            requested = {int(s.strip()) for s in args.steps.split(",")}
        except ValueError:
            parser.error("--steps must be comma-separated integers, e.g. 1,3,5")
    else:
        requested = set(STEPS.keys())

    invalid = requested - set(STEPS.keys())
    if invalid:
        parser.error(f"Unknown step numbers: {invalid}. Valid: {sorted(STEPS.keys())}")

    print(f"\nTrading-Crab pipeline  [{run_cfg}]")
    print(f"Steps to run: {sorted(requested)}")
    print()

    # Ensure output dirs exist
    (OUTPUT_DIR / "plots").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "models").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "reports").mkdir(parents=True, exist_ok=True)

    for step_num in sorted(requested):
        label, fn = STEPS[step_num]
        print(f"── Step {step_num}: {label} ──")
        try:
            fn(cfg, run_cfg)
            print(f"   ✓ done\n")
        except Exception as exc:
            log.exception("Step %d failed: %s", step_num, exc)
            print(f"   ✗ FAILED: {exc}\n")
            sys.exit(1)

    print("Pipeline complete.")


if __name__ == "__main__":
    main()
