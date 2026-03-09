"""
Pipeline: end-to-end orchestration of the Trading-Crab pipeline.

Run with: python -m trading_crab.pipeline [--refresh] [--recompute] [--plots] [-v]

Stages:
  1. Data ingestion (scrape or load from checkpoint)
  2. Feature engineering (log transforms, interpolation, derivatives)
  3. Clustering (PCA + KMeans)
  4. Regime analysis (profiling, naming, transitions)
  5. Supervised prediction (Decision Tree + Random Forest)
  6. Asset returns by regime
  7. Portfolio construction
"""

import logging
import sys

import numpy as np
import pandas as pd

from trading_crab.config import (
    CLUSTERING_FEATURES,
    SNAPSHOT_DATE_FRED,
    SNAPSHOT_DATE_GROK,
    SNAPSHOT_DATE_MULTPL,
    SNAPSHOT_DATE_PREPARED,
    configure_logging,
    data_path,
    parse_args,
)
from trading_crab.data_ingestion import (
    fetch_fred_data,
    fetch_multpl_data,
    load_checkpoint,
    load_grok_labels,
    merge_all_sources,
    save_checkpoint,
)
from trading_crab.feature_engineering import (
    apply_log_transforms,
    compute_all_derivatives,
    interpolate_gaps,
)
from trading_crab.clustering import cluster_regimes, run_pca
from trading_crab.regime_analysis import (
    compute_forward_probabilities,
    compute_transition_matrix,
    name_regimes,
    profile_regimes,
)
from trading_crab.supervised import (
    generate_classification_report,
    prepare_supervised_data,
    train_decision_tree,
    train_random_forest,
)
from trading_crab.asset_returns import (
    best_assets_per_regime,
    compute_quarterly_returns,
    returns_by_regime,
)
from trading_crab.portfolio import (
    blended_regime_portfolio,
    generate_recommendation,
    simple_regime_portfolio,
)

log = logging.getLogger(__name__)


def validate_no_nans(df: pd.DataFrame, stage: str, columns: list[str] = None) -> None:
    """Assert that the given columns have no NaN values. Raises on failure."""
    cols = columns or [c for c in df.columns if c != "market_code"]
    nan_count = int(df[cols].isna().sum().sum())
    assert nan_count == 0, (
        f"Validation FAILED at '{stage}': {nan_count} NaN values remain"
    )
    log.info("  ✓ Validation passed: '%s' — 0 NaNs in %d columns", stage, len(cols))


def validate_shape(df: pd.DataFrame, stage: str, min_rows: int = 10) -> None:
    """Assert the DataFrame has a reasonable number of rows."""
    assert len(df) >= min_rows, (
        f"Validation FAILED at '{stage}': only {len(df)} rows (min {min_rows})"
    )
    log.info("  ✓ Validation passed: '%s' — %d rows × %d cols",
             stage, *df.shape)


def validate_labels_aligned(labels: np.ndarray, df: pd.DataFrame, stage: str) -> None:
    """Assert cluster labels match the DataFrame length after dropna."""
    clean_len = len(df[CLUSTERING_FEATURES].dropna())
    assert len(labels) == clean_len, (
        f"Validation FAILED at '{stage}': labels length {len(labels)} "
        f"!= clean data length {clean_len}"
    )
    log.info("  ✓ Validation passed: '%s' — %d labels aligned", stage, len(labels))


def main(argv: list[str] | None = None) -> None:
    """Run the full Trading-Crab pipeline."""
    args = parse_args(argv)
    configure_logging(args.verbose)
    dp = data_path(args.data_dir)

    log.info("=" * 60)
    log.info("Trading-Crab Pipeline")
    log.info("  --refresh=%s  --recompute=%s  --plots=%s",
             args.refresh, args.recompute, args.plots)
    log.info("  data_dir=%s", dp)
    log.info("=" * 60)

    # =================================================================
    # Stage 1: Data ingestion
    # =================================================================
    if args.refresh and args.recompute:
        multpl_df = fetch_multpl_data()
        save_checkpoint(multpl_df, dp, "multpl_datasets_snapshot")

        fred_df = fetch_fred_data()
        save_checkpoint(fred_df, dp, "fred_api_datasets_snapshot")

        grok_df = load_grok_labels(dp)
        save_checkpoint(grok_df, dp, "grok_quarter_classifications")

    elif args.recompute:
        multpl_df = load_checkpoint(dp, "multpl_datasets_snapshot", SNAPSHOT_DATE_MULTPL)
        fred_df = load_checkpoint(dp, "fred_api_datasets_snapshot", SNAPSHOT_DATE_FRED)
        grok_df = load_checkpoint(dp, "grok_quarter_classifications", SNAPSHOT_DATE_GROK)

    # =================================================================
    # Stage 2: Feature engineering
    # =================================================================
    if args.recompute:
        quarterly_df = merge_all_sources(multpl_df, fred_df, grok_df)
        validate_shape(quarterly_df, "merge")

        quarterly_df = apply_log_transforms(quarterly_df)
        validate_shape(quarterly_df, "log_transforms")

        quarterly_df = interpolate_gaps(quarterly_df)
        validate_no_nans(quarterly_df, "interpolation")

        quarterly_df = compute_all_derivatives(quarterly_df)
        validate_shape(quarterly_df, "derivatives")
        validate_no_nans(quarterly_df, "derivatives", CLUSTERING_FEATURES)

        save_checkpoint(quarterly_df, dp, "prepared_quarterly_data_smoothed")
    else:
        quarterly_df = load_checkpoint(
            dp, "prepared_quarterly_data_smoothed", SNAPSHOT_DATE_PREPARED
        )
        validate_shape(quarterly_df, "loaded_checkpoint")

    # =================================================================
    # Stage 3: Clustering
    # =================================================================
    log.info("")
    log.info("--- Stage 3: Clustering ---")
    reduced_df, pca, market_codes = run_pca(quarterly_df)
    reduced_df, scores, best_k = cluster_regimes(reduced_df)
    validate_labels_aligned(market_codes, quarterly_df, "pca_alignment")

    if args.plots:
        from trading_crab.plotting import plot_kmeans_scores, plot_pc_pairs
        plot_kmeans_scores(scores)
        plot_pc_pairs(reduced_df, market_codes, title_prefix="Grok regimes: ")
        plot_pc_pairs(reduced_df, reduced_df["balanced_cluster"].values,
                      title_prefix="Balanced clusters: ")

    # =================================================================
    # Stage 4: Regime analysis
    # =================================================================
    log.info("")
    log.info("--- Stage 4: Regime Analysis ---")
    cluster_labels = reduced_df["balanced_cluster"].values

    profile = profile_regimes(quarterly_df, cluster_labels)
    regime_names = name_regimes(profile)

    transition = compute_transition_matrix(cluster_labels)
    log.info("Transition matrix:\n%s", transition.round(2))

    forward_probs = compute_forward_probabilities(cluster_labels, horizons=[1, 4, 8])
    for h, probs in forward_probs.items():
        log.info("Forward probabilities (horizon=%d quarters):\n%s", h, probs.round(2))

    if args.plots:
        from trading_crab.plotting import (
            plot_regime_profile_heatmap,
            plot_transition_matrix,
        )
        plot_regime_profile_heatmap(profile, regime_names)
        plot_transition_matrix(transition, regime_names)

    # =================================================================
    # Stage 5: Supervised prediction
    # =================================================================
    log.info("")
    log.info("--- Stage 5: Supervised Prediction ---")
    X, y, feature_names = prepare_supervised_data(quarterly_df, cluster_labels)

    dt_model, dt_metrics = train_decision_tree(X, y, feature_names)
    rf_model, rf_metrics = train_random_forest(X, y, feature_names)

    log.info("")
    generate_classification_report(dt_model, X, y, regime_names)
    generate_classification_report(rf_model, X, y, regime_names)

    if args.plots:
        from trading_crab.plotting import plot_feature_importances
        plot_feature_importances(rf_metrics["feature_importances"], top_n=15)

    # =================================================================
    # Stage 6: Asset returns by regime
    # =================================================================
    log.info("")
    log.info("--- Stage 6: Asset Returns by Regime ---")

    # We need the full quarterly_df (with price columns) for returns.
    # The prepared df only has clustering features, so reload if needed.
    if args.recompute:
        full_df = merge_all_sources(multpl_df, fred_df, grok_df)
    else:
        try:
            full_df = load_checkpoint(dp, "multpl_datasets_snapshot", SNAPSHOT_DATE_MULTPL)
            fred_df_full = load_checkpoint(dp, "fred_api_datasets_snapshot", SNAPSHOT_DATE_FRED)
            grok_df_full = load_checkpoint(dp, "grok_quarter_classifications", SNAPSHOT_DATE_GROK)
            full_df = merge_all_sources(full_df, fred_df_full, grok_df_full)
        except FileNotFoundError:
            log.warning("Cannot load full data for asset returns — skipping stage 6–7")
            log.info("Pipeline complete (stages 1–5).")
            return

    returns = compute_quarterly_returns(full_df)
    # Align returns with cluster labels (skip first row lost to pct_change)
    aligned_labels = cluster_labels[1:len(returns) + 1]
    regime_ret = returns_by_regime(returns, aligned_labels)
    rankings = best_assets_per_regime(regime_ret)

    log.info("Per-regime median returns:\n%s",
             rankings.round(4) if not rankings.empty else "(empty)")

    # =================================================================
    # Stage 7: Portfolio construction
    # =================================================================
    log.info("")
    log.info("--- Stage 7: Portfolio Construction ---")

    # Use the last quarter's cluster as "current regime"
    current_regime = int(cluster_labels[-1])
    log.info("Current regime: %d (%s)", current_regime,
             regime_names.get(current_regime, "Unknown"))

    simple_weights = simple_regime_portfolio(regime_ret, current_regime)

    # Blended: use forward probabilities at 1-quarter horizon as regime probs
    if 1 in forward_probs and current_regime in forward_probs[1].index:
        regime_probs = forward_probs[1].loc[current_regime]
        blended_weights = blended_regime_portfolio(regime_ret, regime_probs)
    else:
        blended_weights = simple_weights

    # Generate trade recommendation (assume starting from cash)
    recommendation = generate_recommendation(
        current_weights=None,
        target_weights=blended_weights,
    )
    log.info("Trade recommendation:\n%s", recommendation)

    if args.plots:
        from trading_crab.plotting import plot_portfolio_recommendation
        plot_portfolio_recommendation(recommendation)

    # =================================================================
    # Done
    # =================================================================
    log.info("")
    log.info("=" * 60)
    log.info("Pipeline complete.")
    log.info("  %d quarters processed", len(quarterly_df))
    log.info("  %d regimes identified", len(regime_names))
    log.info("  Decision Tree CV accuracy: %.3f ± %.3f",
             dt_metrics["mean_cv_accuracy"], dt_metrics["std_cv_accuracy"])
    log.info("  Random Forest CV accuracy: %.3f ± %.3f",
             rf_metrics["mean_cv_accuracy"], rf_metrics["std_cv_accuracy"])
    log.info("=" * 60)


if __name__ == "__main__":
    main()
