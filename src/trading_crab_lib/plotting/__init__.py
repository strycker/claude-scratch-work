"""
plotting — Shared visualization helpers for all pipeline stages.

All plot functions:
  - Accept run_cfg: RunConfig and honour save_plots / show_plots
  - Save to outputs/plots/{step}_{description}.png when save_plots=True
  - Are importable by notebooks without side-effects

Custom 5-regime color palette (from legacy/unified_script.py):
    CUSTOM_COLORS = ["#0000d0","#d00000","#f48c06","#8338ec","#50a000"]

Usage:
    from trading_crab_lib import plotting
    from trading_crab_lib.runtime import RunConfig
    run_cfg = RunConfig(generate_plots=True, save_plots=True)
    plotting.plot_pca_scatter(pca_df, labels, regime_names, run_cfg)
"""

from __future__ import annotations

# ── Step 06: Asset Returns ────────────────────────────────────────────────────
from trading_crab_lib.plotting.assets import (
    plot_asset_heatmap,
    plot_asset_return_distributions,
    plot_asset_returns_by_regime,
)

# ── Step 03: Clustering ──────────────────────────────────────────────────────
from trading_crab_lib.plotting.clustering import (
    plot_cluster_sizes,
    plot_elbow_curve,
    plot_gmm_bic_surface,
    plot_method_comparison_table,
    plot_pca_loadings,
    plot_pca_scatter,
    plot_regime_colored_pca_3d,
    plot_scree,
    plot_silhouette_samples,
)

# ── Core helpers and constants ────────────────────────────────────────────────
from trading_crab_lib.plotting.core import (
    CUSTOM_COLORS,
    PLOT_DIR,
    REGIME_CMAP,
    list_available_plots,
    load_or_generate,
)
from trading_crab_lib.plotting.core import (
    _in_jupyter as _in_jupyter,
)
from trading_crab_lib.plotting.core import (
    _plot_is_fresh as _plot_is_fresh,
)
from trading_crab_lib.plotting.core import (
    _regime_color as _regime_color,
)
from trading_crab_lib.plotting.core import (
    _save_or_show as _save_or_show,
)

# ── Diagnostics ───────────────────────────────────────────────────────────────
from trading_crab_lib.plotting.diagnostics import (
    plot_divergence_timeseries,
    plot_momentum_dashboard,
    plot_rrg_scatter,
)

# ── Step 02: Features ─────────────────────────────────────────────────────────
from trading_crab_lib.plotting.features import (
    plot_centered_vs_causal_comparison,
    plot_feature_correlations,
    plot_feature_distributions,
    plot_feature_variance_ranking,
    plot_gap_fill_before_after,
    plot_nan_heatmap,
    plot_pairplot,
)

# ── Step 01: Ingestion ────────────────────────────────────────────────────────
from trading_crab_lib.plotting.ingestion import (
    plot_raw_series_coverage,
    plot_raw_series_sample,
)

# ── Step 05: Prediction ──────────────────────────────────────────────────────
from trading_crab_lib.plotting.prediction import (
    plot_calibration_curve,
    plot_confusion_matrix,
    plot_cv_fold_accuracy,
    plot_decision_tree,
    plot_feature_importance,
    plot_feature_importance_comparison,
    plot_feature_selection_curve,
    plot_forward_probabilities,
    plot_learning_curve,
    plot_model_comparison_bar,
    plot_predicted_vs_actual,
)

# ── Step 04: Regime Profiling ─────────────────────────────────────────────────
from trading_crab_lib.plotting.regime import (
    plot_correlation_change_heatmap,
    plot_feature_regime_overlay,
    plot_forward_prob_evolution,
    plot_regime_duration_histogram,
    plot_regime_profiles,
    plot_regime_timeline,
    plot_soft_probabilities,
    plot_transition_matrix,
)

__all__ = [
    # Constants
    "CUSTOM_COLORS",
    "PLOT_DIR",
    "REGIME_CMAP",
    # Core helpers
    "load_or_generate",
    "list_available_plots",
    # Ingestion
    "plot_raw_series_coverage",
    "plot_raw_series_sample",
    # Features
    "plot_feature_correlations",
    "plot_feature_distributions",
    "plot_pairplot",
    "plot_gap_fill_before_after",
    "plot_feature_variance_ranking",
    "plot_nan_heatmap",
    "plot_centered_vs_causal_comparison",
    # Clustering
    "plot_elbow_curve",
    "plot_pca_scatter",
    "plot_cluster_sizes",
    "plot_scree",
    "plot_pca_loadings",
    "plot_silhouette_samples",
    "plot_gmm_bic_surface",
    "plot_method_comparison_table",
    "plot_regime_colored_pca_3d",
    # Regime
    "plot_regime_timeline",
    "plot_transition_matrix",
    "plot_regime_profiles",
    "plot_soft_probabilities",
    "plot_feature_regime_overlay",
    "plot_forward_prob_evolution",
    "plot_regime_duration_histogram",
    "plot_correlation_change_heatmap",
    # Prediction
    "plot_feature_importance",
    "plot_forward_probabilities",
    "plot_confusion_matrix",
    "plot_predicted_vs_actual",
    "plot_decision_tree",
    "plot_cv_fold_accuracy",
    "plot_model_comparison_bar",
    "plot_calibration_curve",
    "plot_learning_curve",
    "plot_feature_importance_comparison",
    "plot_feature_selection_curve",
    # Assets
    "plot_asset_returns_by_regime",
    "plot_asset_heatmap",
    "plot_asset_return_distributions",
    # Diagnostics
    "plot_rrg_scatter",
    "plot_divergence_timeseries",
    "plot_momentum_dashboard",
]
