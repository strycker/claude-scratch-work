# Architecture

**Analysis Date:** 2026-03-16

## Pattern Overview

**Overall:** Configuration-driven, stepwise batch pipeline (ETL → feature engineering → unsupervised clustering → regime profiling → supervised prediction → reporting) with checkpointed artifacts (parquet/pickle) and CLI orchestration.

**Key Characteristics:**
- Deterministic, numbered pipeline stages with stable intermediate artifacts under `data/` and `outputs/`
- A thin orchestration layer (`run_pipeline.py`, `pipelines/*.py`) calling a reusable library package (`src/market_regime/`)
- Config-first: all tunable parameters live in `config/settings.yaml`; runtime behavior lives in `src/market_regime/runtime.py`

## Layers

**Orchestration / CLI entry points:**
- Purpose: Dispatch pipeline stages, build runtime config, and ensure output directories exist.
- Location: `run_pipeline.py`, `pipelines/01_ingest.py` … `pipelines/07_dashboard.py`
- Contains: argparse parsing, step registry, top-level I/O wiring, minimal printing.
- Depends on: `src/market_regime/*`
- Used by: humans running the pipeline locally / CI scripts (via `python run_pipeline.py ...`).

**Configuration:**
- Purpose: Load YAML settings and inject environment-derived secrets (FRED API key).
- Location: `src/market_regime/config.py`, `config/settings.yaml`, `config/regime_labels.yaml`
- Contains: `load()` and `setup_logging()` plus YAML configuration.
- Depends on: `python-dotenv`, `pyyaml`; environment variables.
- Used by: `run_pipeline.py` and all `pipelines/*.py`.

**Runtime flags:**
- Purpose: Centralize runtime toggles (plots/verbose/refresh/recompute) and apply them consistently.
- Location: `src/market_regime/runtime.py`
- Contains: `RunConfig` dataclass with `from_args()` and `apply_logging()`.
- Depends on: stdlib only.
- Used by: `run_pipeline.py` (master path); plot helpers honor `RunConfig`.

**Checkpointing / Artifact management:**
- Purpose: Persist/reuse intermediate DataFrames and models between runs; avoid re-scraping and expensive recomputation.
- Location: `src/market_regime/checkpoints.py`
- Contains: `CheckpointManager.save/load/is_fresh/list`, plus config hashing via `config/settings.yaml`.
- Depends on: `pandas`, `pyarrow`, `yaml`.
- Used by: `run_pipeline.py` (master flow); library modules indirectly through orchestrator.

**Ingestion (macro + assets + label overlay):**
- Purpose: Fetch quarterly macro series and optional overlay labels.
- Location: `src/market_regime/ingestion/fred.py`, `src/market_regime/ingestion/multpl.py`, `src/market_regime/ingestion/assets.py`, `src/market_regime/ingestion/grok.py`
- Contains:
  - FRED: threaded fetch, quarterly resample, publication-lag shift (`shift: true`) for select series
  - multpl.com: lxml CSS-selector scraper, rate-limited, quarterly resample
  - assets: yfinance with multi-source fallback chain, resample to quarterly
  - grok: load external labels and convert to integer `market_code`
- Depends on: `fredapi`, `requests`, `lxml`, `yfinance` (+ optional fallbacks), `pandas`.
- Used by: step 1 (macro), step 6 (assets), and optional label overlay in `run_pipeline.py` (`--market-code`).

**Feature engineering:**
- Purpose: Convert raw macro panel into ML-ready feature matrix for clustering and supervised learning.
- Location: `src/market_regime/transforms.py`
- Contains:
  - A fixed transform order (`engineer_all`) with config-selected feature lists
  - Dual-mode derivative computation: centered (for clustering) vs causal/right-aligned (for supervised)
  - Bernstein polynomial gap-filling (`BPoly.from_derivatives`) + Taylor extrapolation
- Depends on: `numpy`, `pandas`, `scipy`, `matplotlib.dates`.
- Used by: step 2 (`pipelines/02_features.py`) and `run_pipeline.py` step 2.

**Unsupervised learning (clustering + investigation utilities):**
- Purpose: Reduce features with PCA, evaluate k, fit KMeans variants, and produce stable regime IDs.
- Location: `src/market_regime/clustering.py` (+ exploration modules: `src/market_regime/gmm.py`, `src/market_regime/density.py`, `src/market_regime/spectral.py`, `src/market_regime/cluster_comparison.py`)
- Contains:
  - `reduce_pca()`, `evaluate_kmeans()`, `pick_best_k()`, `fit_clusters()`
  - `balanced_cluster` via `k-means-constrained` when available; fallback to plain KMeans
  - canonicalization of cluster IDs (ordered by mean PC1) for label stability
- Depends on: `scikit-learn` (and optional extras).
- Used by: step 3 (`pipelines/03_cluster.py`) and `run_pipeline.py` step 3.

**Regime profiling & labeling:**
- Purpose: Produce descriptive statistics and human-readable labels per cluster; compute empirical transition matrix.
- Location: `src/market_regime/regime.py`
- Contains: `build_profiles()`, `suggest_names()`, `build_transition_matrix()`, `load_name_overrides()`.
- Depends on: `pandas`, `pyyaml`.
- Used by: step 4 (`pipelines/04_regime_label.py`) and `run_pipeline.py` step 4.

**Supervised prediction:**
- Purpose: Train regime classifiers (current regime + forward regime membership) using walk-forward CV and causal features.
- Location: `src/market_regime/prediction.py`
- Contains:
  - `train_current_regime()` (RF), `train_decision_tree()` (DT), `train_forward_classifiers()` (binary RF per horizon/regime)
  - `predict_current()` for scoring most recent quarter
- Depends on: `scikit-learn`, `numpy`, `pandas`.
- Used by: step 5 (`pipelines/05_predict.py`) and `run_pipeline.py` step 5/7.

**Asset returns analysis:**
- Purpose: Convert ETF prices (or macro proxies) into quarterly returns and summarize by regime.
- Location: `src/market_regime/asset_returns.py`
- Contains: `compute_quarterly_returns()`, `compute_proxy_returns()`, `returns_by_regime()`, `rank_assets_by_regime()`.
- Depends on: `pandas`.
- Used by: step 6 (`pipelines/06_asset_returns.py`) and `run_pipeline.py` step 6/7.

**Reporting / dashboard & portfolio:**
- Purpose: Generate stoplight dashboard, save report CSVs, and compute portfolio weights and trade recs.
- Location: `src/market_regime/reporting.py`
- Contains: `asset_signals()`, `print_dashboard()`, `save_dashboard_csv()`, `simple_regime_portfolio()`, `blended_regime_portfolio()`, `generate_recommendation()`.
- Depends on: `pandas`.
- Used by: step 7 (`pipelines/07_dashboard.py`) and `run_pipeline.py` step 7.

**Plotting (cross-cutting):**
- Purpose: Centralize all visualization helpers used by pipeline steps and notebooks.
- Location: `src/market_regime/plotting.py`
- Contains: plot functions per step; output to `outputs/plots/` honoring `RunConfig`.
- Depends on: `matplotlib`, `seaborn` (optional).
- Used by: `run_pipeline.py` when `--plots` is enabled and by notebooks in `notebooks/`.

## Data Flow

**End-to-end pipeline (master):**

1. **Step 1 ingest** (`run_pipeline.py:step1_ingest`, `pipelines/01_ingest.py`)
   - Inputs: `config/settings.yaml`, FRED API, multpl.com
   - Outputs:
     - Canonical: `data/raw/macro_raw.parquet`
     - Checkpoint: `data/checkpoints/macro_raw.parquet` (+ `.meta.json`)
   - Optional overlay: `market_code` loaded via `--market-code` from `src/market_regime/ingestion/grok.py` or checkpoints.

2. **Step 2 features** (`run_pipeline.py:step2_features`, `pipelines/02_features.py`)
   - Inputs: `data/raw/macro_raw.parquet`
   - Outputs:
     - `data/processed/features.parquet` (centered windows; used for steps 3–4)
     - `data/processed/features_supervised.parquet` (causal windows; used for steps 5–7)
     - Checkpoints: `features`, `features_supervised`

3. **Step 3 cluster** (`run_pipeline.py:step3_cluster`, `pipelines/03_cluster.py`)
   - Inputs: `data/processed/features.parquet`
   - Outputs:
     - `data/regimes/cluster_labels.parquet` (columns include `cluster`, `balanced_cluster`, optional `market_code`)
     - `data/regimes/pca_components.parquet`
     - `data/regimes/kmeans_scores.parquet`
     - Checkpoints: `cluster_labels`, `pca_components`
   - Optional: `--save-market-code` saves `balanced_cluster` as checkpoint `market_code_clustered`.

4. **Step 4 regime_label** (`pipelines/04_regime_label.py`, `run_pipeline.py:step4_regime_label`)
   - Inputs: `data/processed/features.parquet`, `data/regimes/cluster_labels.parquet`
   - Outputs:
     - `data/regimes/profiles.parquet`
     - `data/regimes/transition_matrix.parquet`
     - `data/regimes/regime_names_suggested.yaml`
   - Manual override path: `config/regime_labels.yaml` (preferred by step 7).

5. **Step 5 predict** (`pipelines/05_predict.py`, `run_pipeline.py:step5_predict`)
   - Inputs: `data/processed/features_supervised.parquet` (preferred) and `data/regimes/cluster_labels.parquet`
   - Outputs:
     - Models: `outputs/models/current_regime.pkl`, `outputs/models/decision_tree.pkl`, `outputs/models/forward_classifiers.pkl`
     - Checkpoint side-effect (master path): `market_code_predicted` saved via `_save_market_code()` in `run_pipeline.py`.

6. **Step 6 asset_returns** (`pipelines/06_asset_returns.py`, `run_pipeline.py:step6_asset_returns`)
   - Inputs: `data/raw/asset_prices.parquet` (cache; optional), `data/raw/macro_raw.parquet` (proxy fallback), `data/regimes/cluster_labels.parquet`
   - Outputs:
     - `data/regimes/asset_return_profile.parquet`
     - Optional cache/checkpoint: `data/raw/asset_prices.parquet`, checkpoint `asset_prices`

7. **Step 7 dashboard** (`pipelines/07_dashboard.py`, `run_pipeline.py:step7_dashboard`)
   - Inputs: `outputs/models/current_regime.pkl`, `data/processed/features_supervised.parquet`, `data/regimes/transition_matrix.parquet`, `data/regimes/asset_return_profile.parquet`
   - Outputs:
     - `outputs/reports/dashboard.csv`
     - `outputs/reports/portfolio_simple.csv`
     - `outputs/reports/portfolio_blended.csv`
     - `outputs/reports/trade_recommendations.csv`

**State Management:**
- Pipeline state is file-based. “Current state” is whatever is in `data/` and `outputs/`.
- Reuse is controlled by `src/market_regime/checkpoints.py` freshness checks and CLI flags (`--refresh`, `--recompute`, `--refresh-assets`) in `run_pipeline.py`.

## Key Abstractions

**`RunConfig` (runtime behavior carrier):**
- Purpose: Single object to propagate flags through steps and plotting.
- Examples: `src/market_regime/runtime.py`, used in `run_pipeline.py` and `src/market_regime/plotting.py`.
- Pattern: Dataclass + `from_args()` factory.

**Checkpoint system (`CheckpointManager`):**
- Purpose: Consistent parquet/pickle persistence with metadata and freshness checks.
- Examples: `src/market_regime/checkpoints.py`; used in `run_pipeline.py` step functions.
- Pattern: Directory-backed manager with `{name}.parquet` + `{name}.meta.json`.

**Canonical inter-step artifacts:**
- Purpose: Stable file paths for step-to-step dependencies, independent of checkpoint storage.
- Examples: `data/raw/macro_raw.parquet`, `data/processed/features.parquet`, `data/regimes/cluster_labels.parquet`, `outputs/models/*.pkl`.
- Pattern: Steps write to canonical paths and may also save checkpoints.

## Entry Points

**Master entry point:**
- Location: `run_pipeline.py`
- Triggers: `python run_pipeline.py [flags]`
- Responsibilities:
  - Parse CLI args → `RunConfig` (`src/market_regime/runtime.py`)
  - Load config (`src/market_regime/config.py`)
  - Dispatch selected steps via `STEPS` registry
  - Enforce canonical I/O locations and optional checkpoint fallback (`_load_parquet()`)

**Per-step entry points (standalone scripts):**
- Locations:
  - `pipelines/01_ingest.py`
  - `pipelines/02_features.py`
  - `pipelines/03_cluster.py`
  - `pipelines/04_regime_label.py`
  - `pipelines/05_predict.py`
  - `pipelines/06_asset_returns.py`
  - `pipelines/07_dashboard.py`
- Triggers: `python pipelines/0X_*.py`
- Responsibilities: Minimal; call into `src/market_regime/*` and write canonical artifacts.

## Error Handling

**Strategy:** Fail-fast for orchestration; warn-and-skip for missing optional components and partial-data scenarios.

**Patterns:**
- `run_pipeline.py` wraps each step invocation with `try/except`, logs exception, and exits non-zero (`sys.exit(1)`).
- Library modules log warnings and return empty data when appropriate:
  - Missing multpl datasets → empty DataFrame (`src/market_regime/ingestion/multpl.py`)
  - Missing grok label file → returns `None` (`src/market_regime/ingestion/grok.py`)
  - Missing optional clustering backends → log warning and proceed (`src/market_regime/clustering.py`)
  - Asset ingestion fallbacks progress through multiple sources before returning empty (`src/market_regime/ingestion/assets.py`)

## Cross-Cutting Concerns

**Logging:** `logging` module everywhere; initialized via `src/market_regime/config.py:setup_logging()` and adjusted by `RunConfig.apply_logging()`.
**Validation:** Light runtime validation (e.g., missing features lists, missing columns) via warnings in `src/market_regime/transforms.py`.
**Authentication:** Environment-based only (FRED) via `FRED_API_KEY` injected by `src/market_regime/config.py` into the loaded config dict.

---

*Architecture analysis: 2026-03-16*
