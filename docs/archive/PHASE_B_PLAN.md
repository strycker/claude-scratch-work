# Phase B — Decompose & Document

> **⚠️ SUPERSEDED (July 2026).** Historical plan — kept for context, not active work.
> The project's target and execution plan are now `platform_design/platform_design.md`
> (v1.7) and `ROADMAP.md` Tier 0. Do not treat items below as current.

**Branch:** `claude/phase-b-decompose`
**Status:** In progress
**Depends on:** Phase A (complete)

---

## Overview

Split the two largest modules (`plotting.py` at 2,022 lines and `monitoring.py`
at 652 lines) into sub-packages, then add docstrings to all public functions.

---

## B1 — Decompose `plotting.py` into `plotting/` package (M)

**Goal:** Split 2,022-line monolith into ~6 focused submodules. Re-export
everything from `plotting/__init__.py` so all existing imports continue to work.

### Target layout

```
src/trading_crab_lib/plotting/
├── __init__.py          ← re-exports everything; holds CUSTOM_COLORS, REGIME_CMAP
├── core.py              ← _in_jupyter, _save_or_show, _regime_color, _plot_is_fresh,
│                           load_or_generate, list_available_plots
├── ingestion.py         ← plot_raw_series_coverage, plot_raw_series_sample (step 01)
├── features.py          ← plot_feature_correlations, plot_feature_distributions,
│                           plot_pairplot, plot_gap_fill_before_after,
│                           plot_feature_variance_ranking, plot_nan_heatmap,
│                           plot_centered_vs_causal_comparison (steps 02 / A5)
├── clustering.py        ← plot_elbow_curve, plot_pca_scatter, plot_cluster_sizes,
│                           plot_scree, plot_pca_loadings, plot_silhouette_samples,
│                           plot_gmm_bic_surface, plot_method_comparison_table,
│                           plot_regime_colored_pca_3d (step 03 / A2)
├── regime.py            ← plot_regime_timeline, plot_transition_matrix,
│                           plot_regime_profiles, plot_soft_probabilities,
│                           plot_feature_regime_overlay, plot_forward_prob_evolution,
│                           plot_regime_duration_histogram,
│                           plot_correlation_change_heatmap (step 04 / A3-A4)
├── prediction.py        ← plot_feature_importance, plot_forward_probabilities,
│                           plot_confusion_matrix, plot_predicted_vs_actual,
│                           plot_decision_tree, plot_cv_fold_accuracy,
│                           plot_model_comparison_bar, plot_calibration_curve,
│                           plot_learning_curve, plot_feature_importance_comparison,
│                           plot_feature_selection_curve (step 05 / A1-A4)
├── assets.py            ← plot_asset_returns_by_regime, plot_asset_heatmap,
│                           plot_asset_return_distributions (step 06)
└── diagnostics.py       ← plot_rrg_scatter, plot_divergence_timeseries,
                            plot_momentum_dashboard (steps 08-09 / A4-A5)
```

### Steps
1. Create `plotting/` directory
2. Create `core.py` with shared helpers + matplotlib setup
3. Move functions group-by-group into submodules (imports from `core`)
4. Create `__init__.py` with wildcard re-exports
5. Run tests after each submodule to catch breakage
6. Delete old `plotting.py`
7. Verify `from trading_crab_lib.plotting import plot_pca_scatter` still works
8. Verify `from trading_crab_lib import plotting; plotting.REGIME_CMAP` still works

### Backward compat contract
All of these must keep working:
- `from trading_crab_lib.plotting import plot_pca_scatter`
- `from trading_crab_lib import plotting`
- `plotting.CUSTOM_COLORS`, `plotting.REGIME_CMAP`
- `plotting.PLOT_DIR`

---

## B2 — Decompose `monitoring.py` into `monitoring/` package (S)

**Goal:** Split 652-line module into ~4 focused submodules.

### Target layout

```
src/trading_crab_lib/monitoring/
├── __init__.py          ← re-exports everything
├── ingestion.py         ← format_completeness_table, DateRangeReport,
│                           validate_date_range, SourceRowCounts,
│                           count_source_columns (C1 items)
├── features.py          ← FeatureQualityReport, compute_feature_quality (C1.4)
├── clustering.py        ← format_method_comparison, RegimeStabilityReport,
│                           compute_regime_stability (C2 items)
├── prediction.py        ← CVFoldReport, compute_cv_fold_scores,
│                           check_regime_probabilities (C3 items)
└── pipeline.py          ← format_tactics_summary, StepValidation,
                            validate_step_output, PipelineHealthSummary (C4 items)
```

### Steps
1. Create `monitoring/` directory
2. Move each section into its submodule
3. Create `__init__.py` with re-exports
4. Run tests
5. Delete old `monitoring.py`

---

## B3 — Docstring pass for public API (M)

**Goal:** Add one-line docstrings + parameter/return docs to all public functions
that currently lack them. Skip `_`-prefixed internals.

### Rules
- Modules to cover: everything under `src/trading_crab_lib/`
- Only add docstrings where missing or incomplete
- Include `Args:` and `Returns:` sections in numpydoc-ish format
- Don't modify functions that already have good docstrings
- Run tests after completion

---

## Execution Order

```
B1 (plotting decomposition) → B2 (monitoring decomposition) → B3 (docstrings)
```

Each task = one commit. All three on the same branch/PR.
