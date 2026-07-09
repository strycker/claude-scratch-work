# Codebase Structure

**Analysis Date:** 2026-07-09

## Directory Layout

```
trading-crab/
├── .planning/                       ← GSD planning outputs (this repo's docs)
├── .claude/                         ← Claude Code tooling (keybindings, settings)
├── CLAUDE.md                        ← Project guide (read first)
├── README.md                        ← User-facing overview
├── ROADMAP.md                       ← Prioritized backlog
├── STATE.md                         ← Pipeline status + test counts
├── platform_design/                 ← 5-layer L0-L4 architecture vision (design phase)
│   └── platform_design.md
├── .env.example                     ← Copy to .env, fill FRED_API_KEY
├── pyproject.toml                   ← Root workspace + app package
├── Makefile                         ← Dev shortcuts
│
├── config/                          ← All tuneable parameters
│   ├── settings.yaml                ← Master config (data ranges, FRED series, features, clustering k, model depths)
│   └── regime_labels.yaml           ← Manual regime name curation (edit after step 3/4)
│
├── data/                            ← Runtime output (gitignored)
│   ├── raw/                         ← Cached raw data
│   │   ├── macro_raw.parquet        ← Step 1 output (FRED + multpl + macrotrends + ETF prices merged)
│   │   └── asset_prices.parquet     ← ETF prices cached from step 1, reused by step 6
│   ├── processed/                   ← Derived datasets
│   │   ├── features.parquet         ← Step 2 output (centered derivatives, for clustering)
│   │   └── features_supervised.parquet  ← Step 2 output (causal derivatives, for supervised learning)
│   ├── regimes/                     ← Step 3-4 outputs
│   │   ├── cluster_labels.parquet   ← Step 3 output (both `cluster` and `balanced_cluster` columns)
│   │   ├── kmeans_scores.parquet    ← Silhouette/CH/DB scores from k-sweep
│   │   └── profiles.parquet         ← Step 4 output (per-regime means, stds, transition matrix)
│   └── checkpoints/                 ← CheckpointManager directory (parquet + .meta.json pairs)
│       ├── {name}.parquet
│       ├── {name}.meta.json
│       ├── macro_raw_secondary.parquet  ← Preservation checkpoint: full columns pre-dropna
│       ├── features_secondary.parquet   ← Preservation checkpoint: all engineered features
│       ├── features_supervised_secondary.parquet  ← Preservation checkpoint: causal versions
│       └── market_code_*.parquet    ← External labels (grok, clustered, predicted)
│
├── outputs/                         ← Final outputs (gitignored)
│   ├── models/                      ← Pickled sklearn models
│   │   ├── current_regime.pkl       ← Step 5: RandomForestClassifier for nowcasting
│   │   ├── dt_current.pkl           ← Step 5: DecisionTreeClassifier (optional)
│   │   ├── lgbm_current.pkl         ← Step 5: LightGBM (optional, requires lightgbm)
│   │   └── forward_*.pkl            ← Step 5: Forward classifiers (1Q/2Q/4Q/8Q)
│   ├── plots/                       ← PNG/PDF figures (per-step)
│   │   ├── 01_*.png                 ← Step 1 plots (raw series coverage)
│   │   ├── 02_*.png                 ← Step 2 plots (gap-fill, variance, centered vs causal)
│   │   ├── 03_*.png                 ← Step 3 plots (PCA, elbow, silhouette, method comparison)
│   │   ├── 04_*.png                 ← Step 4 plots (regime timeline, transition matrix, forward probs)
│   │   ├── 05_*.png                 ← Step 5 plots (feature importance, CV accuracy, calibration)
│   │   ├── 06_*.png                 ← Step 6 plots (asset returns by regime, heatmap)
│   │   ├── 08_*.png                 ← Step 8 plots (RRG scatter, rolling z-scores)
│   │   └── 09_*.png                 ← Step 9 plots (tactics summary)
│   └── reports/                     ← CSV/text outputs
│       ├── dashboard.csv            ← Step 7: regime + asset signals (machine-readable)
│       ├── weekly_report.md         ← Weekly automation: regime + portfolio + email draft
│       ├── email_body.txt           ← Step 7 (optional): HTML/plain-text email body
│       ├── diagnostics/
│       │   ├── rrg_quadrants.csv    ← Step 8: RRG classification
│       │   └── rolling_ratios.csv   ← Step 8: rolling z-scores
│       └── tactics.csv              ← Step 9: buy_hold/swing/stand_aside per asset
│
├── legacy/                          ← Reference (DO NOT MODIFY)
│   └── unified_script.py            ← Original 1249-line monolith; algorithm ground truth
│
├── pipelines/                       ← Simplified step entry points (for minimal use cases)
│   ├── 01_ingest.py                 ← Runs step1_ingest() directly (no RunConfig)
│   ├── 02_features.py               ← Runs step2_features() directly
│   ├── 03_cluster.py
│   ├── 04_regime_label.py
│   ├── 05_predict.py
│   ├── 06_asset_returns.py
│   ├── 07_dashboard.py
│   ├── 08_diagnostics.py
│   └── 09_tactics.py
│
├── notebooks/                       ← Jupyter exploration (one per pipeline stage + comparisons)
│   ├── 01_ingestion.ipynb           ← Raw series inspection
│   ├── 02_features.ipynb            ← Gap-fill diagnostics, variance ranking, centered vs causal
│   ├── 03_clustering.ipynb          ← PCA, silhouette, GMM/DBSCAN/Spectral comparison
│   ├── 04_regimes.ipynb             ← Regime profiles, transition matrix, HMM comparison
│   ├── 05_prediction.ipynb          ← CV diagnostics, model comparison, calibration
│   ├── 06_assets.ipynb              ← Per-regime violin plots, Sharpe table, ETF coverage
│   ├── 07_pairplot.ipynb            ← Triple-colored pairplots (unsupervised/grok/RF)
│   ├── 08_raw_series.ipynb          ← Raw series exploration
│   ├── 09_diagnostics.ipynb         ← RRG scatter, rolling z-scores, quadrant rotation
│   ├── 10_model_comparison.ipynb    ← KMeans vs GMM vs HMM vs Spectral; soft probs
│   ├── 11_feature_selection.ipynb   ← RF importance curves, dead-feature detector, what-if
│   └── 12_divergence_momentum.ipynb ← Divergence z-scores, momentum dashboard
│
├── scripts/                         ← Automation + setup
│   ├── setup.sh                     ← Automated environment setup
│   ├── jupyter_notebook_local.sh    ← Local notebook launcher helper
│   └── run_weekly_report.py         ← Weekly report automation (pipeline + archive + email)
│
├── tests/                           ← pytest test suite (~769 tests)
│   ├── conftest.py                  ← Shared fixtures (quarterly_index, raw_macro_df, etc.)
│   ├── fixtures/                    ← Test data (currently empty)
│   ├── integration/
│   │   └── test_mini_pipeline.py    ← Synthetic end-to-end: determinism regression
│   ├── test_pipeline_smoke.py       ← Pipeline dispatch + step registry tests
│   ├── test_cli_smoke.py            ← CLI entry-point tests
│   ├── test_pipelines_ingest_features.py  ← Steps 1-2 smoke tests
│   ├── test_models_regime.py        ← Bundle API regime tests
│   ├── test_models_boosting.py      ← GradientBoosting tests
│   ├── test_models_interpret_tree.py ← Interpretability helpers
│   ├── test_models_behavior.py      ← Behavior model tests
│   ├── test_models_reporting.py     ← Metrics aggregation
│   ├── test_email_weekly.py         ← Email + weekly report tests
│   ├── test_scripts_weekly_report.py ← run_weekly_report.py tests
│   ├── test_constraints_etf_universe.py   ← ETF universe validation
│   ├── test_constraints_frequency.py      ← Data frequency validation
│   └── unit/                        ← Unit tests for src/trading_crab_lib modules (50+ files)
│       ├── test_transforms.py       ← engineer_all, gap-fill, derivatives
│       ├── test_clustering.py       ← KMeans, model selection
│       ├── test_clustering_exploration.py ← GMM sweep, gap stat, knee detection
│       ├── test_cluster_comparison.py     ← ARI, feature importance
│       ├── test_gmm.py
│       ├── test_hmm.py              ← GaussianHMM (optional)
│       ├── test_markov.py           ← MarkovRegression (optional)
│       ├── test_density.py          ← DBSCAN/HDBSCAN
│       ├── test_spectral.py
│       ├── test_checkpoints.py      ← CheckpointManager
│       ├── test_returns.py
│       ├── test_prediction_flat.py  ← Flat prediction API
│       ├── test_lightgbm.py         ← LightGBM (optional)
│       ├── test_ingestion.py        ← HTTP-mocked FRED/multpl/assets
│       ├── test_macrotrends.py      ← macrotrends scraper (mocked)
│       ├── test_diagnostics_rrg.py  ← RRG analysis
│       ├── test_tactics.py          ← Tactical classification
│       ├── test_config.py           ← Config loading + validation
│       ├── test_regime.py           ← Regime profiles + transitions
│       ├── test_yield_curve_features.py
│       ├── test_divergence.py       ← Cross-asset divergence
│       ├── test_momentum.py         ← Momentum features
│       ├── test_indicators.py       ← LEI proxy
│       ├── test_monitoring.py       ← Pipeline health checks
│       ├── test_plotting.py         ← All plot functions
│       ├── test_reporting.py        ← Dashboard, portfolio helpers
│       ├── test_runtime.py          ← RunConfig
│       ├── test_init_module.py      ← Env var overrides, convenience imports
│       └── 20+ more unit test files
│
├── ideas/                           ← Salvaged code + explorations (do not use in production)
│   └── gsd-salvage/                 ← Code extracted from submodules for reference
│
├── src/                             ← Two-package workspace
│   ├── trading_crab/                ← App package (pip name: trading-crab)
│   │   ├── __init__.py              ← Version metadata
│   │   ├── cli.py                   ← CLI entry point
│   │   └── pipeline.py              ← 9-step pipeline orchestration (1200+ lines)
│   │
│   └── trading_crab_lib/            ← Library package (pip name: trading-crab-lib)
│       ├── pyproject.toml           ← Independent lib pyproject.toml + extras
│       ├── __init__.py              ← Path resolution (TC_*_DIR env vars), version
│       ├── config.py                ← load(), validate_config(), load_portfolio()
│       ├── runtime.py               ← RunConfig dataclass
│       ├── checkpoints.py           ← CheckpointManager
│       ├── transforms.py            ← engineer_all(): cross-ratios → log → gap-fill → deriv
│       ├── clustering.py            ← PCA, KMeans, model selection, gap statistic
│       ├── gmm.py                   ← Gaussian Mixture Model
│       ├── hmm.py                   ← Hidden Markov Model (optional)
│       ├── markov.py                ← Markov regime-switching (optional)
│       ├── density.py               ← DBSCAN, HDBSCAN
│       ├── spectral.py              ← Spectral clustering
│       ├── cluster_comparison.py    ← Pairwise ARI, RF feature importance
│       ├── regime.py                ← Regime profiling, naming, transitions
│       ├── asset_returns.py         ← Compute quarterly returns by regime
│       ├── reporting.py             ← Dashboard signals, portfolio construction
│       ├── diagnostics.py           ← RRG, rolling z-scores
│       ├── tactics.py               ← Tactical classification
│       ├── email.py                 ← Weekly email composition + SMTP
│       ├── divergence.py            ← Cross-asset divergence features
│       ├── momentum.py              ← Momentum features
│       ├── indicators.py            ← Composite indicators (LEI proxy)
│       ├── yield_curve_features.py  ← Yield curve spreads
│       ├── ingestion/               ← Data source fetchers
│       │   ├── __init__.py          ← ingestion_completeness_report()
│       │   ├── fred.py              ← FRED API (with publication-lag shifts)
│       │   ├── multpl.py            ← multpl.com scraper (lxml)
│       │   ├── assets.py            ← yfinance ETF prices
│       │   ├── macrotrends.py       ← macrotrends.net commodity prices
│       │   └── grok.py              ← Load external LLM labels
│       ├── prediction/              ← Regime classifiers
│       │   ├── __init__.py          ← Flat API (production): RandomForest, DecisionTree, LightGBM
│       │   ├── classifier.py        ← Bundle API (test-only): per-fold reports, GradientBoosting
│       │   └── gradient_boosting.py ← GradientBoostingClassifier helpers
│       ├── plotting/                ← Visualization subpackage
│       │   ├── __init__.py          ← Re-exports all plot functions + color constants
│       │   ├── core.py              ← _save_or_show(), _regime_color(), _in_jupyter()
│       │   ├── ingestion.py         ← Step 1 plots (coverage, sample)
│       │   ├── features.py          ← Step 2 plots (gap-fill, variance, centered vs causal)
│       │   ├── clustering.py        ← Step 3 plots (elbow, PCA, silhouette, GMM BIC)
│       │   ├── regime.py            ← Step 4 plots (timeline, transition matrix, forward probs)
│       │   ├── prediction.py        ← Step 5 plots (importance, tree, calibration, learning curve)
│       │   ├── assets.py            ← Step 6 plots (returns by regime, heatmap)
│       │   └── diagnostics.py       ← Step 8-9 plots (RRG, divergence, momentum)
│       └── monitoring/              ← Pipeline health + QA checks
│           ├── __init__.py          ← Re-exports all monitoring functions
│           ├── ingestion.py         ← Completeness checks, date range validation
│           ├── features.py          ← Feature quality metrics
│           ├── clustering.py        ← Regime stability, method comparison
│           ├── prediction.py        ← CV fold scores, calibration checks
│           └── pipeline.py          ← Step output validation, health summary
│
├── gsd-scratch-work/                ← READ-ONLY submodule (earlier GSD checkpoint)
├── trading-crab/                    ← READ-ONLY submodule (public/PyPI repo)
├── trading-crab-lib/                ← READ-ONLY submodule (library repo)
│
├── run_pipeline.py                  ← Backward-compat shim (python run_pipeline.py --help)
├── requirements.txt                 ← Legacy pinned deps (prefer pyproject.toml)
├── requirements-dev.txt             ← Legacy dev deps (prefer pyproject.toml)
│
└── .gitignore                       ← Excludes .env, data/, outputs/, *.pyc
```

## Directory Purposes

**config/**
- Purpose: Master parameters for the entire pipeline
- Contains: YAML files (all hand-editable, version-controlled)
- Key files: `settings.yaml` (data ranges, series lists, feature lists, clustering k, model depths), `regime_labels.yaml` (manual semantic names after clustering)

**data/**
- Purpose: Runtime intermediate data (gitignored)
- Contains: Raw, processed, regime, and checkpoint parquet files
- Special: `checkpoints/` is the CheckpointManager directory (all steps save/load here)

**outputs/**
- Purpose: Final outputs (gitignored)
- Contains: Pickled models, PNG/PDF plots, CSV/text reports

**legacy/**
- Purpose: Algorithm ground truth (DO NOT MODIFY or PUSH)
- Contains: `unified_script.py` — original 1249-line monolith; reference for every formula, parameter, pipeline order

**pipelines/**
- Purpose: Simplified entry points (legacy compatibility; minimal use cases)
- Contains: One script per pipeline step; no RunConfig, hardcoded flags
- Use: Only if you need to run a single step without the full CLI

**notebooks/**
- Purpose: Exploration and visualization (Jupyter)
- Contains: 12 notebooks, one per pipeline stage plus comparisons
- Pattern: Each notebook imports from `trading_crab_lib` and calls `CheckpointManager` to load checkpoints

**scripts/**
- Purpose: Automation (setup, weekly reports, helper launchers)
- Contains: `setup.sh` (environment), `run_weekly_report.py` (email automation)

**tests/**
- Purpose: pytest test suite
- Layout: `test_*.py` files in root; `unit/` subdirectory for library tests; `integration/` for end-to-end
- Fixture data: `fixtures/` (currently empty; populated as needed)

**src/trading_crab/**
- Purpose: CLI app package (pip name: trading-crab)
- Contents: `cli.py` (entry point), `pipeline.py` (9-step orchestration + step functions)
- Dependency: Requires `trading-crab-lib>=0.1.2`

**src/trading_crab_lib/**
- Purpose: Reusable library package (pip name: trading-crab-lib)
- Independent: No dependency on `trading-crab` (app depends on lib, not vice versa)
- Extras: `[ingestion]`, `[plotting]`, `[hmm]`, `[clustering-extras]`, `[boosting]`, `[all]`
- Subpackages: `ingestion/`, `prediction/`, `plotting/`, `monitoring/`

**ideas/gsd-salvage/**
- Purpose: Reference code extracted from submodules (do not use in production)
- Contains: Earlier implementations of features, tactics, prediction
- Use: For design inspiration or historical reference only

## Naming Conventions

**Files:**
- `test_*.py` — pytest test file (in tests/ or tests/unit/)
- `*_test.py` — alternate pytest naming (rare; use test_* prefix)
- `*.ipynb` — Jupyter notebook (numbered 01–12 in notebooks/)
- `.meta.json` — Checkpoint metadata (alongside {name}.parquet in checkpoints/)

**Directories:**
- `trading_crab_lib` — package (pip: trading-crab-lib, hyphenated)
- `trading_crab` — package (pip: trading-crab, hyphenated)
- `src/`, `data/`, `outputs/` — lowercase, underscore for multi-word (underscore not hyphen)
- `raw`, `processed`, `regimes`, `checkpoints` — data subdirs, lowercase
- `ingestion`, `prediction`, `plotting`, `monitoring` — subpackages, lowercase, noun-based

**Functions:**
- `add_X()` — adds columns to DataFrame (e.g., `add_cross_ratios()`)
- `apply_X()` — transforms existing columns (e.g., `apply_log_transforms()`)
- `compute_X()` — derives new data (e.g., `compute_regime_stability()`)
- `fit_X()` — trains a model (e.g., `fit_clusters()`)
- `X_labels()` — extract hard labels from a fitted model (e.g., `gmm_labels()`)
- `X_probabilities()` — extract soft probabilities (e.g., `hmm_probabilities()`)
- `plot_X()` — matplotlib visualization (e.g., `plot_elbow_curve()`)
- `predict_X()` — inference function (e.g., `predict_current()`)
- `build_X()` — construct complex output (e.g., `build_profiles()`)
- `save_X()` / `load_X()` — I/O (e.g., `save_dashboard_csv()`)
- `step1_X()` / `step2_X()` — pipeline steps (e.g., `step1_ingest()`)
- `_X()` — private helper (leading underscore, not exported)

**Variables:**
- `df` — pandas DataFrame (raw data)
- `features`, `X` — feature DataFrame for modeling
- `labels`, `y` — target labels (regime, returns, etc.)
- `cfg` — configuration dict (from `config.load()`)
- `run_cfg` — RunConfig instance
- `cm` — CheckpointManager instance
- `pca_obj`, `model`, `clf` — fitted sklearn objects
- `log` — logger instance (per-module: `log = logging.getLogger(__name__)`)

**Types:**
- `RegimeProfile` — dict with per-regime statistics (if dataclass, use snake_case)
- `FoldReport` — namedtuple or dataclass with per-fold CV results
- `StepValidation` — dataclass with validation results

**Checkpoint names:**
- `macro_raw` — step 1 output (raw merged data)
- `features` — step 2 output (centered derivatives)
- `features_supervised` — step 2 output (causal derivatives)
- `cluster_labels` — step 3 output
- `profiles` — step 4 output
- `*_secondary` — preservation checkpoints (wide versions, `macro_raw_secondary`, etc.)
- `market_code_grok` — external labels (grok)
- `market_code_clustered` — saved via `--save-market-code`
- `market_code_predicted` — auto-saved by step 5
- `asset_prices` — cached ETF prices (fetched in step 1, reused by step 6)

## Where to Add New Code

**New Feature for Feature Engineering:**
- Primary code: `src/trading_crab_lib/transforms.py` (if simple ratios/transforms) or new module `src/trading_crab_lib/{feature_name}.py` (if complex, e.g., `divergence.py`, `momentum.py`)
- Hook it into `engineer_all()` at the appropriate step (order: cross-ratios → yield-curve → divergence → momentum → log → select → gap-fill → derivatives → select)
- Tests: `tests/unit/test_{feature_name}.py`
- Add to `config/settings.yaml` feature lists (`initial_features` or `clustering_features`) to enable

**New Clustering Algorithm:**
- Location: `src/trading_crab_lib/{algorithm}.py` (e.g., `gmm.py`, `hmm.py`)
- API: Match existing pattern — `fit_{algorithm}()`, `{algorithm}_labels()`, `{algorithm}_probabilities()` (if applicable)
- Also add comparison: `cluster_comparison.py` function `compare_X_vs_kmeans()` if valuable for notebooks
- Tests: `tests/unit/test_{algorithm}.py`

**New Prediction Model Type:**
- Flat API (production): Add to `src/trading_crab_lib/prediction/__init__.py` as `train_X()` and call from `step5_predict()`
- Bundle API (test-only): Add to `src/trading_crab_lib/prediction/classifier.py` if tests need per-fold reports
- Tests: `tests/unit/test_prediction_*.py` (flat) + `tests/test_models_regime.py` (bundle)
- Model persistence: Save to `outputs/models/{model_name}.pkl` via `joblib.dump()`

**New Plot Function:**
- Location: `src/trading_crab_lib/plotting/{step_name}.py` (e.g., `features.py`, `clustering.py`)
- Signature: `plot_X(df, ..., run_cfg: RunConfig) -> None` (mutates matplotlib state)
- Helper: Use `_save_or_show(fname, run_cfg.save_plots, run_cfg.show_plots)` for consistent save/show
- Color: Use `CUSTOM_COLORS` (5 regimes) or `REGIME_CMAP` (ListedColormap)
- Tests: `tests/unit/test_plotting.py`
- Wire into step: Call from `step{N}_*()` function in `src/trading_crab/pipeline.py` if `run_cfg.generate_plots`

**New Monitoring Check:**
- Location: `src/trading_crab_lib/monitoring/{aspect}.py` (e.g., `features.py`, `prediction.py`)
- Pattern: Function returns dataclass (or dict) with check results; logged at INFO
- Example: `check_regime_probabilities()` in `prediction.py` warns if any regime has <5% predicted probability
- Wire into step: Call from `step{N}_*()` function before/after key computation
- Tests: `tests/unit/test_monitoring.py`

**New CLI Flag:**
- Location: `src/trading_crab/pipeline.py` (argparse setup) + `src/trading_crab_lib/runtime.py` (RunConfig field)
- Pattern: Add argparse argument in `build_parser()`, add field to `RunConfig` dataclass, populate in `from_args()` factory
- Example: `--refresh`, `--plots`, `--market-code`
- Pass `run_cfg` to steps that need the flag

**New Test:**
- Location: `tests/unit/test_{module}.py` for library code, `tests/test_{aspect}.py` for integration/pipeline
- Fixtures: Reuse from `tests/conftest.py` (e.g., `quarterly_index`, `raw_macro_df`) or define locally
- Mocking: Use `monkeypatch` (pytest) for environment + file I/O redirection; mock HTTP with `responses` or `unittest.mock`
- Determinism: Set seed at test start: `np.random.seed(42)` + `random.seed(42)`
- Run: `pytest tests/ -v` or `pytest tests/unit/test_X.py::test_Y -v`

**New Notebook:**
- Location: `notebooks/{step_num}_{description}.ipynb` (e.g., `09_diagnostics.ipynb`)
- Pattern: Load checkpoints via `CheckpointManager`, call library functions, plot results via `plotting` module
- Never inline plotting logic — call `plotting.plot_X()` functions
- Example imports:
  ```python
  from trading_crab_lib.checkpoints import CheckpointManager
  from trading_crab_lib import plotting
  cm = CheckpointManager()
  features = cm.load("features")
  plotting.plot_feature_correlations(features)
  ```

**New Pipeline Step:**
- When step count grows beyond 9, add to `src/trading_crab/pipeline.py`:
  1. Define `step{N}_description(cfg, run_cfg) -> None`
  2. Add to `STEPS` dict: `{N: ("description", step{N}_description)}`
  3. Add to pipeline loop in `main()` (after step 9 guard)
  4. Add CLI flag if needed (e.g., `--steps 1,2,10` includes new step 10)
  5. Create `pipelines/{N:02d}_description.py` shim (if backward-compat needed)

**New Configuration Parameter:**
- Edit: `config/settings.yaml` (all parameters here, never hardcode in Python)
- Validate: Add to `validate_config()` check in `config.py` if required
- Reference: Access via `cfg["section"]["key"]` (e.g., `cfg["clustering"]["n_pca_components"]`)
- Document: Add comment in YAML explaining the parameter and its impact

## Special Directories

**data/checkpoints/**
- Purpose: Parquet checkpoints + JSON metadata (managed by CheckpointManager)
- Generated: Yes (at runtime, one per pipeline step)
- Committed: No (gitignored)
- Lifecycle: `CheckpointManager.save()` writes both `{name}.parquet` and `{name}.meta.json`; `load()` validates metadata before loading

**outputs/models/**
- Purpose: Pickled sklearn models
- Generated: Yes (step 5)
- Committed: No (gitignored)
- Lifecycle: `joblib.dump(model, path)` saves; `joblib.load(path)` or `pickle.load()` loads

**outputs/plots/**
- Purpose: PNG/PDF figures (one per plot function, per step)
- Generated: Yes (if `--plots` flag)
- Committed: No (gitignored)
- Naming: `{step}_{description}.png` (e.g., `03_regime_pca_scatter.png`)

**outputs/reports/**
- Purpose: Dashboard, diagnostics, tactics outputs
- Generated: Yes (steps 7-9)
- Committed: No (gitignored)
- Key files: `dashboard.csv` (step 7), `weekly_report.md` (weekly), `diagnostics/` (step 8), `tactics.csv` (step 9)

**tests/fixtures/**
- Purpose: Test data files (fixture data, mock responses, etc.)
- Generated: No (manually curated or generated by test setup)
- Committed: Yes (small, representative examples)
- Current: Empty (tests generate synthetic data inline)

---

*Structure analysis: 2026-07-09*
