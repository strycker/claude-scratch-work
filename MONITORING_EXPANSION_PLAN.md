# Pipeline Monitoring & Notebook Expansion Plan

**Created:** 2026-03-23  |  **Updated:** 2026-03-24
**Status:** Planning — awaiting owner approval on phase ordering

Each phase has **at most 5 items** so it can be completed in a single Claude Code
session without timeout. Phases are ordered by dependency.

**Total: 85 work items across 17 phases.**

---

## Goals

1. **Monitoring** — every pipeline step validates outputs before the next step runs
2. **Plots** — 25+ new visualization functions in `plotting.py`
3. **Notebooks** — every step (01–10) gets a comprehensive diagnostic notebook
4. **Plot reuse** — notebooks load pre-generated PNGs; only regenerate on demand
5. **Feature engineering tools** — notebooks for exploring what features to add/drop
6. **Model selection tools** — notebooks for comparing classifiers and tuning hyperparams
7. **Email attachments** — key plots embedded in the weekly email
8. **QA gates** — pipeline prints pass/fail validation at each step boundary

---

## Phase A1 — Core Model Evaluation Plot Functions (5 items)

New functions in `plotting.py` + smoke tests in `test_plotting.py`.

| Item | Function | Purpose |
|------|----------|---------|
| A1.1 | `plot_decision_tree(tree, feature_names, regime_names, run_cfg)` | Render sklearn DT via `plot_tree` (readable tree diagram) |
| A1.2 | `plot_cv_fold_accuracy(fold_reports, run_cfg)` | Bar chart of per-fold accuracy from TimeSeriesSplit |
| A1.3 | `plot_model_comparison_bar(metrics_dict, run_cfg)` | Grouped bar: RF vs DT vs GB accuracy/F1 side-by-side |
| A1.4 | `plot_calibration_curve(model, X, y, regime_names, run_cfg)` | Reliability diagram: predicted prob vs actual frequency |
| A1.5 | `plot_learning_curve(model, X, y, run_cfg)` | Train/test score vs training set size (detects overfitting) |

---

## Phase A2 — PCA & Clustering Plot Functions (5 items)

| Item | Function | Purpose |
|------|----------|---------|
| A2.1 | `plot_scree(pca_obj, run_cfg)` | Scree plot: individual + cumulative explained variance |
| A2.2 | `plot_pca_loadings(pca_obj, feature_names, run_cfg, top_n=15)` | Heatmap: top features × PCA components (absolute loadings) |
| A2.3 | `plot_silhouette_samples(X, labels, run_cfg)` | Per-sample silhouette width grouped by cluster |
| A2.4 | `plot_gmm_bic_surface(bic_df, run_cfg)` | Heatmap: (k, covariance_type) → BIC from `fit_gmm()` |
| A2.5 | `plot_method_comparison_table(comparison_df, run_cfg)` | Table-as-figure for `compare_all_methods()` output |

---

## Phase A3 — Time-Series & Regime Plot Functions (5 items)

| Item | Function | Purpose |
|------|----------|---------|
| A3.1 | `plot_soft_probabilities(probs_df, regime_names, run_cfg)` | Stacked area: GMM/HMM posterior probabilities over time |
| A3.2 | `plot_feature_regime_overlay(series, labels, regime_names, run_cfg)` | Line plot with regime-colored background bands |
| A3.3 | `plot_forward_prob_evolution(forward_probs, regime_names, run_cfg)` | Heatmap: regime × horizon P(transition) for 1Q/2Q/4Q/8Q |
| A3.4 | `plot_gap_fill_before_after(raw_col, filled_col, run_cfg)` | Overlay: raw (with gaps) vs filled, gaps highlighted |
| A3.5 | `plot_regime_colored_pca_3d(pca_df, labels, regime_names, run_cfg)` | 3D scatter PC1×PC2×PC3 with regime colors |

---

## Phase A4 — Specialty Diagnostic Plot Functions (5 items)

| Item | Function | Purpose |
|------|----------|---------|
| A4.1 | `plot_rrg_scatter(rrg_df, run_cfg)` | RRG 4-quadrant scatter with asset labels and arrows |
| A4.2 | `plot_feature_importance_comparison(models_dict, feature_names, run_cfg)` | Side-by-side importance: RF vs DT vs GB in one figure |
| A4.3 | `plot_regime_duration_histogram(labels, regime_names, run_cfg)` | How many consecutive quarters each regime persists |
| A4.4 | `plot_correlation_change_heatmap(features, labels, run_cfg)` | Feature correlation matrix per-regime (shows structure changes) |
| A4.5 | `plot_feature_variance_ranking(features, run_cfg, top_n=30)` | Horizontal bar: features ranked by variance (find dead features) |

---

## Phase A5 — Feature Engineering & Selection Plot Functions (5 items)

| Item | Function | Purpose |
|------|----------|---------|
| A5.1 | `plot_feature_selection_curve(importances, run_cfg)` | Cumulative importance vs # features (find diminishing returns) |
| A5.2 | `plot_divergence_timeseries(div_features, labels, run_cfg)` | Z-score divergence over time with regime-transition markers |
| A5.3 | `plot_momentum_dashboard(momentum_features, labels, run_cfg)` | Grid: trailing momentum + relative strength for key series |
| A5.4 | `plot_nan_heatmap(df, run_cfg)` | Binary heatmap: which cells are NaN (data coverage map) |
| A5.5 | `plot_centered_vs_causal_comparison(feat_c, feat_s, cols, run_cfg)` | Side-by-side: centered vs causal for same feature (shows look-ahead) |

---

## Phase B — Plot Reuse Infrastructure (3 items)

| Item | What |
|------|------|
| B.1 | Add `load_or_generate(plot_func, *args, filename, run_cfg)` helper to `plotting.py` |
| B.2 | Helper checks `outputs/plots/{filename}` freshness vs checkpoint; shows PNG via `IPython.display.Image` if fresh, else regenerates |
| B.3 | Add `list_available_plots()` utility: print table of all PNGs in `outputs/plots/` with timestamps |

---

## Phase C1 — Pipeline Monitoring: Steps 1–2 (5 items) ✅ DONE

| Item | Step | Enhancement | Status |
|------|------|-------------|--------|
| C1.1 | Step 1 | Wire `ingestion_completeness_report()` into logging; print table of missing/extra/high-NaN columns | ✅ |
| C1.2 | Step 1 | Add date-range validation: warn if data doesn't extend to current quarter | ✅ |
| C1.3 | Step 1 | Add per-source row count summary: multpl (N cols), FRED (N cols), macrotrends (N cols), ETFs (N cols) | ✅ |
| C1.4 | Step 2 | Print feature quality report: NaN count per col, top-5 highest-variance, top-5 highest-correlation pairs | ✅ |
| C1.5 | Step 2 | Generate `plot_gap_fill_before_after()` for 3 sample columns when `--plots` | ✅ |

**Implementation:** `src/trading_crab_lib/monitoring.py` (new module, ~250 lines).
Wired into `run_pipeline.py` steps 1-2 and `pipelines/01_ingest.py`, `pipelines/02_features.py`.
23 tests in `tests/unit/test_monitoring.py`.

---

## Phase C2 — Pipeline Monitoring: Steps 3–4 (5 items) ✅ DONE

| Item | Step | Enhancement | Status |
|------|------|-------------|--------|
| C2.1 | Step 3 | Generate `plot_scree()` and `plot_pca_loadings()` when `--plots` | ✅ Done |
| C2.2 | Step 3 | Generate `plot_silhouette_samples()` when `--plots` | ✅ Done |
| C2.3 | Step 3 | Print method comparison table: KMeans vs balanced (silhouette, CH, DB) | ✅ Done |
| C2.4 | Step 4 | Print regime stability summary: persistence (diagonal), most/least stable regime | ✅ Done |
| C2.5 | Step 4 | Generate `plot_feature_regime_overlay()` for 4 key indicators when `--plots` | ✅ Done |

---

## Phase C3 — Pipeline Monitoring: Steps 5–7 (5 items) ✅ DONE

| Item | Step | Enhancement | Status |
|------|------|-------------|--------|
| C3.1 | Step 5 | Print per-fold CV accuracy table + mean ± std for RF, DT, LGBM | ✅ Done |
| C3.2 | Step 5 | Generate `plot_cv_fold_accuracy()` and `plot_decision_tree()` when `--plots` | ✅ Done |
| C3.3 | Step 5 | Generate `plot_calibration_curve()` and `plot_model_comparison_bar()` when `--plots` | ✅ Done |
| C3.4 | Step 7 | Generate `plot_forward_prob_evolution()` when `--plots` | ✅ Done |
| C3.5 | Step 7 | Print QA gate: warn if any regime has <5% probability (suspiciously low) | ✅ Done |

---

## Phase C4 — Pipeline Monitoring: Steps 8–9 + QA Gates (5 items) ✅ DONE

| Item | Step | Enhancement | Status |
|------|------|-------------|--------|
| C4.1 | Step 8 | Generate `plot_rrg_scatter()` when `--plots` | ✅ Done |
| C4.2 | Step 9 | Print tactics summary: count of buy_hold/swing/stand_aside per asset | ✅ Done |
| C4.3 | All | Add `validate_step_output(step_num, outputs)` function that checks shape, NaN%, dtype | ✅ Done |
| C4.4 | All | Print step timing: elapsed seconds per step in final summary | ✅ Done |
| C4.5 | All | Print pipeline health summary at end: steps run, plots generated, warnings raised | ✅ Done |

---

## Phase D1 — Notebook 01: Ingestion Diagnostics (5 items)

Add cells to `notebooks/01_ingestion.ipynb`.

| Item | New Cell(s) |
|------|-------------|
| D1.1 | Ingestion completeness table: expected vs actual columns, missing highlighted |
| D1.2 | Per-source breakdown: multpl row count, FRED row count, macrotrends row count, ETF row count |
| D1.3 | Date range validation: first/last date per series, flag series ending before current quarter |
| D1.4 | NaN heatmap (`plot_nan_heatmap`) showing coverage gaps across all series |
| D1.5 | Summary statistics table: per-column min/max/mean/std/NaN% (sorted by coverage) |

---

## Phase D2 — Notebook 02: Feature Engineering Diagnostics (5 items) ✅ DONE

Add cells to `notebooks/02_features.ipynb`.

| Item | New Cell(s) | Status |
|------|-------------|--------|
| D2.1 | Gap-fill before/after overlays for 3 columns (sp500, us_cpi, 10yr_ustreas) | ✅ Done |
| D2.2 | Feature variance ranking bar chart (`plot_feature_variance_ranking`) | ✅ Done |
| D2.3 | Centered vs causal comparison panel (`plot_centered_vs_causal_comparison`) for 3 features | ✅ Done |
| D2.4 | Derivative magnitude distributions: histograms of d1/d2/d3 for key features | ✅ Done |
| D2.5 | Correlation heatmap limited to divergence + momentum features (are they redundant?) | ✅ Done |

**Implementation:** 10 new cells (5 markdown + 5 code) appended to `notebooks/02_features.ipynb`.
Uses existing `plotting.py` functions: `plot_gap_fill_before_after()`, `plot_feature_variance_ranking()`,
`plot_centered_vs_causal_comparison()`. D2.4 and D2.5 use inline matplotlib/seaborn with
`_save_or_show()` for consistent plot output. All cells guarded with `if features is not None`.

---

## Phase D3a — Notebook 03: PCA Diagnostics (5 items) ✅ DONE

Add cells to `notebooks/03_clustering.ipynb` — PCA-focused section.

| Item | New Cell(s) | Status |
|------|-------------|--------|
| D3a.1 | Scree plot with cumulative variance threshold line at 90% | ✅ Done (new cell using `plot_scree()`) |
| D3a.2 | PCA loadings heatmap: top-15 features × 5 components | ✅ Done (new cell using `plot_pca_loadings()`) |
| D3a.3 | PCA component sweep: silhouette score for n=3,4,5,6,7,8 (from `optimize_n_components()`) | ✅ Already existed (cells 17-18) |
| D3a.4 | SVD vs PCA loadings side-by-side (from `compare_svd_pca()`) | ✅ Already existed (cells 20-21) |
| D3a.5 | PC1 vs PC2 scatter with regime colors + marginal distributions (rugplot or KDE) | ✅ Done (new cell using seaborn `jointplot` with marginal KDE) |

---

## Phase D3b — Notebook 03: Alternative Clustering Methods (5 items) ✅ DONE

Add cells to `notebooks/03_clustering.ipynb` — alternative methods section.

| Item | New Cell(s) | Status |
|------|-------------|--------|
| D3b.1 | GMM BIC surface heatmap (`plot_gmm_bic_surface`) | ✅ Done (new cell using `plot_gmm_bic_surface()`) |
| D3b.2 | DBSCAN eps-sweep curve (from `fit_dbscan_sweep()`) + k-NN distance elbow plot | ✅ Already existed (cells 30-31) |
| D3b.3 | Spectral k-sweep: silhouette/CH/DB curves (from `fit_spectral_sweep()`) | ✅ Already existed (cell 34) |
| D3b.4 | Multi-method comparison table (`plot_method_comparison_table`) | ✅ Done (new cell using `plot_method_comparison_table()`) |
| D3b.5 | Gap statistic curve with error bars + optimal k marker | ✅ Already existed (cell 24) |

---

## Phase D3c — Notebook 03: Cluster Quality Deep-Dive (4 items) ✅ DONE

Add cells to `notebooks/03_clustering.ipynb` — quality analysis section.

| Item | New Cell(s) | Status |
|------|-------------|--------|
| D3c.1 | Per-sample silhouette plot (`plot_silhouette_samples`) | ✅ Done (new cell) |
| D3c.2 | 3D PCA scatter (`plot_regime_colored_pca_3d`) | ✅ Done (new cell) |
| D3c.3 | Regime duration histogram (`plot_regime_duration_histogram`) — how long regimes persist | ✅ Done (new cell + run-length summary stats) |
| D3c.4 | Pairwise ARI matrix heatmap (from `pairwise_rand_index()`) — method agreement | ✅ Already existed (cell 38) |

**Implementation:** 16 new cells (8 markdown + 8 code) appended to `notebooks/03_clustering.ipynb`.
D3a.3, D3a.4, D3b.2, D3b.3, D3b.5, and D3c.4 already existed as inline implementations in the
notebook's 28 investigation cells. New cells call the official `plotting.py` functions for
consistent, reusable output. D3a.5 uses seaborn `jointplot` for marginal KDE distributions.
D3c.3 includes run-length summary statistics per regime.

---

## Phase D4 — Notebook 04: Regime Profiling Diagnostics (5 items) ✅ DONE

Add cells to `notebooks/04_regimes.ipynb`.

| Item | New Cell(s) | Status |
|------|-------------|--------|
| D4.1 | Feature-regime overlay for 4 key indicators (`plot_feature_regime_overlay`) | ✅ Done |
| D4.2 | Regime stability metrics: persistence %, avg consecutive quarters, stability ranking | ✅ Done |
| D4.3 | Forward probability heatmap (`plot_forward_prob_evolution`) | ✅ Done |
| D4.4 | Per-regime correlation heatmap (`plot_correlation_change_heatmap`) — structure changes | ✅ Done |
| D4.5 | Empirical vs learned transition matrix comparison (if HMM available) | ✅ Done |

**Implementation:** 10 new cells (5 markdown + 5 code) appended to `notebooks/04_regimes.ipynb`.
D4.1 overlays `log_sp500_d1`, `log_us_cpi_d1`, `credit_spread`, `10yr_ustreas_d1` with regime
bands. D4.2 uses `compute_regime_stability()` from monitoring.py + dual bar chart (persistence
probability and average duration). D4.3 uses `compute_forward_probabilities()` for 1Q/4Q/8Q
horizons. D4.4 uses `plot_correlation_change_heatmap(top_n=12)`. D4.5 fits GaussianHMM (if
hmmlearn installed) and shows side-by-side empirical vs learned transition matrices with
absolute difference heatmap.

---

## Phase D5a — Notebook 05: CV Diagnostics (5 items) ✅ DONE

Add cells to `notebooks/05_prediction.ipynb` — cross-validation section.

| Item | New Cell(s) | Status |
|------|-------------|--------|
| D5a.1 | CV fold accuracy bar chart (`plot_cv_fold_accuracy`) for RF | ✅ Done |
| D5a.2 | Per-fold confusion matrix grid (small 5×5 heatmaps, one per fold) | ✅ Done |
| D5a.3 | Learning curve (`plot_learning_curve`) — train vs test accuracy vs N | ✅ Done |
| D5a.4 | Per-fold class distribution table (check for imbalanced folds) | ✅ Done |
| D5a.5 | Temporal accuracy plot: accuracy by decade (1950s, 1960s, ... 2020s) | ✅ Done |

---

## Phase D5b — Notebook 05: Model Comparison & Interpretability (5 items) ✅ DONE

Add cells to `notebooks/05_prediction.ipynb` — model comparison section.

| Item | New Cell(s) | Status |
|------|-------------|--------|
| D5b.1 | Decision tree rendering (`plot_decision_tree`) for DT model | ✅ Done |
| D5b.2 | Interpretability tree: shallow DT on top-10 RF features, print rules as text | ✅ Done |
| D5b.3 | Calibration curve (`plot_calibration_curve`) for RF | ✅ Done |
| D5b.4 | RF vs DT vs GB accuracy comparison (`plot_model_comparison_bar`) | ✅ Done |
| D5b.5 | Feature importance comparison (`plot_feature_importance_comparison`) — RF vs DT vs GB | ✅ Done |

**Implementation:** 20 new cells (10 markdown + 10 code) appended to `notebooks/05_prediction.ipynb`.
D5a cells use `TimeSeriesSplit` + `clone()` for proper temporal CV. D5a.2 builds a per-fold
confusion matrix grid with seaborn heatmaps. D5a.4 pivots class counts per fold and flags
folds with missing regimes. D5a.5 computes per-decade accuracy with bar chart. D5b.1 trains
a DT via flat API and renders with `plot_decision_tree()`. D5b.2 trains a shallow DT on
top-10 RF features and prints `export_text()` rules. D5b.4 trains RF+DT+LGBM (optional),
evaluates via CV, and plots grouped bar comparison. Also removed trailing empty cell.

---

## Phase D6 — Notebook 06: Asset Return Analysis (5 items) ✅ DONE

Add cells to `notebooks/06_assets.ipynb`.

| Item | New Cell(s) | Status |
|------|-------------|--------|
| D6.1 | Per-regime box/violin plots of asset returns (not just medians) | ✅ Done |
| D6.2 | Regime-conditional Sharpe ratio table (mean/std per regime per asset) | ✅ Done |
| D6.3 | Best/worst asset per regime summary table | ✅ Done |
| D6.4 | Return correlation matrix per regime (do assets diversify within each regime?) | ✅ Done |
| D6.5 | ETF coverage timeline: which ETFs have data in each quarter (sparse before 2000) | ✅ Done |

**Implementation:** 10 new cells (5 markdown + 5 code) appended to `notebooks/06_assets.ipynb`.
D6.1 uses seaborn violin plots for 6 key ETFs (SPY, TLT, GLD, QQQ, VNQ, AGG). D6.2 computes
annualized Sharpe ratios (mean/std × sqrt(4)) with styled DataFrame (RdYlGn gradient). D6.3
shows best/worst asset per regime with spread. D6.4 plots per-regime correlation matrices for
top-10 ETFs by coverage. D6.5 creates binary heatmap of ETF data availability plus first-date
summary table.

---

## Phase D7 — New Notebook 09: Diagnostics & RRG (5 items) ✅ DONE

Created `notebooks/09_diagnostics.ipynb` (13 cells).

| Item | New Cell(s) | Status |
|------|-------------|--------|
| D7.1 | Setup + load ratio diagnostics and RRG data from `outputs/reports/diagnostics/` | ✅ 3 cells: setup, run_step_if_needed helper, data loading with fallback |
| D7.2 | RRG 4-quadrant scatter plot (`plot_rrg_scatter`) | ✅ Column rename rs_ratio→rs, rs_momentum→rm; fallback on-the-fly computation from prices |
| D7.3 | Rolling z-score time-series for 4 key ratios | ✅ Config-driven ratios with ±2σ bands and shaded extremes |
| D7.4 | Quadrant rotation history: how often each asset moves between quadrants | ✅ Stacked horizontal bar chart with quadrant frequency per asset |
| D7.5 | Ratio percentile rank dashboard: current values vs historical distribution | ✅ Per-ratio histogram with current value marked + summary table with HIGH/LOW/NORMAL signal |

---

## Phase D8a — New Notebook 10: Model Comparison — Clustering (5 items) ✅ DONE

Created `notebooks/10_model_comparison.ipynb` — clustering comparison section.

| Item | New Cell(s) | Status |
|------|-------------|--------|
| D8a.1 | Setup + load PCA data + fit KMeans/GMM/HMM/Spectral (or load from cache) | ✅ 4 cells: imports, run_step_if_needed, PCA+labels loading, fit all methods |
| D8a.2 | Side-by-side PCA scatter: 4 panels (KMeans, GMM, HMM, Spectral) same color scale | ✅ Dynamic N-panel scatter with consistent palette |
| D8a.3 | ARI pairwise matrix heatmap — which methods agree most? | ✅ Uses pairwise_rand_index() + seaborn heatmap |
| D8a.4 | Temporal label agreement: % of quarters where KMeans == GMM == HMM | ✅ Rolling unique-label diversity + pairwise ARI summary |
| D8a.5 | Regime timeline comparison: 4 stacked horizontal timelines (one per method) | ✅ N stacked timelines with per-method legend |

---

## Phase D8b — New Notebook 10: Model Comparison — Soft Probabilities (4 items) ✅ DONE

Continue `notebooks/10_model_comparison.ipynb` — soft probability section.

| Item | New Cell(s) | Status |
|------|-------------|--------|
| D8b.1 | GMM soft probabilities stacked area (`plot_soft_probabilities`) | ✅ Uses gmm_probabilities() + plot_soft_probabilities() with custom title/filename |
| D8b.2 | HMM soft probabilities stacked area (`plot_soft_probabilities`) | ✅ Uses hmm_probabilities() + plot_soft_probabilities(), graceful skip if hmmlearn missing |
| D8b.3 | Side-by-side: GMM vs HMM — which gives sharper/more uncertain assignments? | ✅ Entropy time-series + max-probability distribution + summary table |
| D8b.4 | Markov 2-state recession probability overlay on regime timeline | ✅ fit_markov_switching on GDP/CPI derivative, recession prob overlay on KMeans timeline, cross-tabulation |

---

## Phase D9 — New Notebook 11: Feature Selection Workbench (5 items) ✅ DONE

Created `notebooks/11_feature_selection.ipynb` (12 cells).

| Item | New Cell(s) | Status |
|------|-------------|--------|
| D9.1 | Setup + load RF model + feature importances | ✅ Loads from current_regime.pkl via extract_rf_feature_importances(), features checkpoint, KMeans labels |
| D9.2 | Feature importance cumulative curve (`plot_feature_selection_curve`) | ✅ Uses plot_feature_selection_curve() + prints N features for 90%/95% thresholds |
| D9.3 | Recommended feature subset from `recommend_clustering_features()` — table | ✅ Shows recommended vs dropped features with comparison DataFrame |
| D9.4 | What-if: re-cluster with top-35 features only, compare silhouette vs full set | ✅ Full PCA+KMeans pipeline on both sets, bar chart comparison |
| D9.5 | Dead feature detector: features with <0.5% importance flagged for removal | ✅ Horizontal bar chart with dead threshold line, lists features not in RF model |

---

## Phase D10 — New Notebook 12: Divergence & Momentum Workbench (5 items)

Create `notebooks/12_divergence_momentum.ipynb`.

| Item | New Cell(s) |
|------|-------------|
| D10.1 | Setup + load features with divergence + momentum columns |
| D10.2 | Divergence z-score time-series with regime-transition markers (`plot_divergence_timeseries`) |
| D10.3 | Momentum dashboard: trailing returns + relative strength (`plot_momentum_dashboard`) |
| D10.4 | Divergence trigger analysis: % of transitions preceded by trigger (leading indicator test) |
| D10.5 | Correlation of divergence features with each other (redundancy check) |

---

## Phase E — Email Plot Attachments (3 items)

| Item | What |
|------|------|
| E.1 | Update `email.py`: `build_weekly_email_body()` returns `MIMEMultipart` with inline PNG attachments |
| E.2 | Add `config/email.yaml` key `attach_plots: [list of plot filenames]` |
| E.3 | Update `scripts/run_weekly_report.py` to pass plot paths to email builder |

---

## Execution Order (Dependency Graph)

```
A1 → A2 → A3 → A4 → A5     (plot functions, independent batches)
         ↓
         B                    (plot reuse infrastructure)
         ↓
C1 → C2 → C3 → C4            (pipeline monitoring, sequential by step)
C5                             (bug fixes + email config — independent)
C6                             (env var paths + convenience imports)
C7                             (preservation checkpoints)
         ↓
D1 → D2                      (data notebooks)
D3a → D3b → D3c              (clustering notebook, 3 parts)
D4                            (regime notebook)
D5a → D5b                    (prediction notebook, 2 parts)
D6                            (assets notebook)
D7                            (diagnostics notebook)
D8a → D8b                    (model comparison notebook, 2 parts)
D9                            (feature selection notebook)
D10                           (divergence/momentum notebook)
         ↓
         E                    (email attachments, after plots exist)
```

**Suggested session order:**
1. A1, A2, A3 (plot functions — 3 sessions)
2. A4, A5 (more plot functions — 2 sessions)
3. B (infrastructure — 1 session)
4. C1, C2, C3, C4 (pipeline monitoring — 4 sessions)
4b. C5 (email bug fixes — 1 session), C6 (env vars — 1 session), C7 (preservation — 1 session)
5. D1, D2 (data notebooks — 2 sessions)
6. D3a, D3b, D3c (clustering notebook — 3 sessions)
7. D4, D5a, D5b (regime + prediction — 3 sessions)
8. D6, D7 (assets + diagnostics — 2 sessions)
9. D8a, D8b, D9, D10 (new notebooks — 4 sessions)
10. E (email — 1 session)

**Total: ~28 sessions, each completing one phase.**

---

## Phase C5 — Bug Fixes + Email Config Alignment (5 items) ✅ DONE

| Item | Description | Status |
|------|-------------|--------|
| C5.1 | Align `email.py` to use `from_address`/`to_address` (GSD convention), matching `email.example.yaml`; add `portfolio.local.yaml` to `.gitignore`; add `trading-crab-lib-repo-copy` to `MANIFEST.in` prune list | ✅ |
| C5.2 | Add env var fallback for email config: `TC_SMTP_HOST`, `TC_SMTP_PORT`, `TC_SMTP_USER`, `TC_SMTP_PASSWORD`, `TC_EMAIL_FROM`, `TC_EMAIL_TO`, `TC_EMAIL_USE_TLS`, `TC_EMAIL_USE_SSL` — env vars override YAML values | ✅ |
| C5.3 | Guard weekly report email flow: skip `send_weekly_email()` when `weekly_report.md` is missing (don't attempt email at all) | ✅ |
| C5.4 | Add strict validation to `load_email_config()` — fail-fast at load time when required keys are missing, return empty dict with clear error | ✅ |
| C5.5 | Update `.env.example` with all `TC_*` email env vars + comments; add email scaffolding step to `setup.sh` (copies `email.example.yaml` → `email.local.yaml`) | ✅ |

**Implementation:** Rewrote `src/trading_crab_lib/email.py` (~210 lines). Updated
`scripts/run_weekly_report.py`, `.env.example`, `scripts/setup.sh`, `.gitignore`,
`MANIFEST.in`. 21 tests in `tests/test_email_weekly.py` (8 new, 13 updated), all passing.

---

## Phase C6 — Env Var Path Overrides + Convenience Imports (5 items) ✅ DONE

| Item | Description | Status |
|------|-------------|--------|
| C6.1 | Add env var overrides for paths in `__init__.py`: `TC_ROOT_DIR`, `TC_CONFIG_DIR`, `TC_DATA_DIR`, `TC_OUTPUT_DIR` (env var wins if set, else relative path) | ✅ |
| C6.2 | Add GSD-style convenience re-exports to `__init__.py`: `load`, `load_portfolio`, `RunConfig`, `CheckpointManager` — enables `import trading_crab_lib as tcl; tcl.load()` | ✅ |
| C6.3 | Enrich `pyproject.toml` metadata: License + OS classifiers, Changelog URL | ✅ |
| C6.4 | *(Nice-to-have, deferred)*: `[project.scripts]` entry point for a `trading-crab` CLI command. For now `python run_pipeline.py` is sufficient | — |
| C6.5 | Tests for env var path overrides + convenience import aliases | ✅ |

**Implementation:** Updated `src/trading_crab_lib/__init__.py` (~55 lines). `_resolve_dir()`
helper checks `TC_*` env vars at import time. Convenience re-exports use lazy `__getattr__`
for `RunConfig` and `CheckpointManager` to avoid circular imports. `load()` and `load_portfolio()`
are thin wrappers with deferred imports. 15 tests in `tests/unit/test_init_module.py`
(1 skipped without joblib). `.env.example` updated with `TC_*` path vars.
`pyproject.toml` enriched with License + OS classifiers and Changelog URL.

---

## Phase C7 — Preservation Checkpoints (`--refresh-preservation`) (5 items) ✅ DONE

Ported from GSD submodule. Preservation checkpoints are wide parquet snapshots
(`macro_raw_secondary`, `features_secondary`, `features_supervised_secondary`) that
survive `clear_all()`. Purpose: downstream steps that drop sparse columns via
`dropna(axis=1)` don't erase the full column audit trail.

| Item | Description | Status |
|------|-------------|--------|
| C7.1 | Add `PRESERVATION_CHECKPOINT_NAMES` frozenset and `preservation_checkpoint_should_write()` decision function to `checkpoints.py` | ✅ |
| C7.2 | Add `refresh_preservation_checkpoints: bool` field to `RunConfig` + `--refresh-preservation` argparse flag in `run_pipeline.py` | ✅ |
| C7.3 | Wire preservation saves into step 1: save `macro_raw_secondary` after ingestion | ✅ |
| C7.4 | Wire preservation saves into step 2: save `features_secondary` and `features_supervised_secondary` after feature engineering | ✅ |
| C7.5 | Update `clear_all()` to skip preservation files; add tests for all preservation logic | ✅ |

**Implementation:** `PRESERVATION_CHECKPOINT_NAMES` frozenset in `checkpoints.py` with
`preservation_checkpoint_should_write()` decision function (write-once unless `--refresh-preservation`).
`clear_all()` updated with `include_preservation` kwarg (defaults False). `RunConfig` gains
`refresh_preservation_checkpoints` field mapped to `--refresh-preservation` argparse flag.
Steps 1 and 2 in `run_pipeline.py` save `*_secondary` checkpoints after primary saves.
10 new tests across `test_checkpoints.py` (7) and `test_runtime.py` (3), all passing.

---

## Open Questions for Owner

1. **3D plots (A3.5):** matplotlib 3D is clunky. Skip, or use plotly (adds dep)?
2. **Email format (E.1):** HTML with inline images, or plain text with attachments?
3. **HMM/Markov in notebooks:** D8a-D8b assume HMM is available. Skip if hmmlearn not installed, or require it?
4. **Priority:** Start with plot functions (A1-A5) or jump to the notebook you use most?
