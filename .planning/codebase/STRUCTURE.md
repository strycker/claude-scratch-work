# Codebase Structure

**Analysis Date:** 2026-03-16

## Directory Layout

```
[project-root]/
├── config/                  # YAML configuration (settings + regime name overrides)
├── data/                    # runtime data artifacts (raw/processed/regimes + checkpoints)
├── legacy/                  # reference implementation (do not modify)
├── notebooks/               # exploratory Jupyter notebooks per pipeline step
├── outputs/                 # runtime outputs (models, plots, reports)
├── pipelines/               # standalone runnable step scripts (01–07)
├── scratch/                 # extended design notes / scratchpad docs
├── scripts/                 # helper scripts (setup, launch notebooks)
├── src/                     # installable python package (src-layout)
│   └── market_regime/        # core library modules used by pipeline steps
├── tests/                   # pytest tests (unit + integration)
├── run_pipeline.py          # master CLI entry point (runs selected steps)
├── pyproject.toml           # package metadata + dependencies + pytest config
├── requirements.txt         # base requirements (alternative to pyproject)
├── requirements-dev.txt     # dev requirements (alternative to pyproject extras)
└── README.md                # user-facing overview
```

## Directory Purposes

**`config/`:**
- Purpose: Central configuration (tunable parameters and regime label overrides).
- Contains: YAML files only.
- Key files:
  - `config/settings.yaml`
  - `config/regime_labels.yaml`

**`src/market_regime/`:**
- Purpose: Core library code (imported by orchestration scripts and notebooks).
- Contains: Modules grouped by pipeline concern (ingestion, transforms, clustering, prediction, reporting).
- Key files:
  - `src/market_regime/__init__.py` (defines `ROOT`, `CONFIG_DIR`, `DATA_DIR`, `OUTPUT_DIR`)
  - `src/market_regime/config.py` (config loading + logging setup)
  - `src/market_regime/runtime.py` (`RunConfig`)
  - `src/market_regime/checkpoints.py` (`CheckpointManager`)
  - `src/market_regime/ingestion/fred.py`, `src/market_regime/ingestion/multpl.py`, `src/market_regime/ingestion/assets.py`, `src/market_regime/ingestion/grok.py`
  - `src/market_regime/transforms.py` (feature engineering)
  - `src/market_regime/clustering.py` (PCA + KMeans + utilities)
  - `src/market_regime/regime.py` (profiling + naming + transitions)
  - `src/market_regime/prediction.py` (supervised models)
  - `src/market_regime/asset_returns.py` (returns + regime profiling)
  - `src/market_regime/reporting.py` (dashboard + portfolio)
  - `src/market_regime/plotting.py` (all plots; used by notebooks and `run_pipeline.py`)

**`pipelines/`:**
- Purpose: One-file runnable scripts for each pipeline stage.
- Contains: `01_ingest.py` … `07_dashboard.py` (thin wrappers around `src/market_regime/*`).
- Key files:
  - `pipelines/01_ingest.py` (macro ingestion)
  - `pipelines/02_features.py` (feature engineering)
  - `pipelines/03_cluster.py` (PCA + clustering)
  - `pipelines/04_regime_label.py` (profiling + suggested names)
  - `pipelines/05_predict.py` (train models)
  - `pipelines/06_asset_returns.py` (ETF/proxy returns by regime)
  - `pipelines/07_dashboard.py` (print + save dashboard and portfolio outputs)

**`data/`:**
- Purpose: File-based pipeline state (inputs and intermediates).
- Contains: parquet/pickle artifacts and checkpoints.
- Key subdirectories:
  - `data/raw/` (canonical raw artifacts like `macro_raw.parquet`, `asset_prices.parquet`)
  - `data/processed/` (derived feature matrices like `features.parquet`, `features_supervised.parquet`)
  - `data/regimes/` (cluster labels, PCA components, transition matrix, profiles, return profiles)
  - `data/checkpoints/` (checkpoint parquet + metadata JSON)

**`outputs/`:**
- Purpose: Generated outputs for users/consumers.
- Key subdirectories:
  - `outputs/models/` (pickled sklearn models, e.g. `current_regime.pkl`)
  - `outputs/plots/` (PNG plots named by step, produced by `src/market_regime/plotting.py`)
  - `outputs/reports/` (CSV reports: dashboard and portfolio allocations)

**`notebooks/`:**
- Purpose: Exploration and plotting notebooks; expected to import plotting helpers from `src/market_regime/plotting.py`.
- Contains: numbered notebooks matching pipeline stages (e.g. `notebooks/03_clustering.ipynb`).

**`legacy/`:**
- Purpose: Ground-truth/reference implementation; not part of the active package.
- Contains: scripts and notebooks (including `legacy/unified_script.py`) used as algorithmic reference.

**`tests/`:**
- Purpose: Automated tests run via `pytest` (see `pyproject.toml` pytest config).
- Contains: `tests/unit/`, `tests/integration/`, plus shared fixtures in `tests/conftest.py`.

## Key File Locations

**Entry Points:**
- `run_pipeline.py`: master CLI runner (dispatches steps 1–7)
- `pipelines/01_ingest.py` … `pipelines/07_dashboard.py`: per-step runnable scripts

**Configuration:**
- `config/settings.yaml`: all tunable parameters (data range, feature lists, clustering/prediction settings, assets list, dashboard thresholds)
- `config/regime_labels.yaml`: pinned cluster ID → name overrides (preferred by dashboard)
- `src/market_regime/config.py`: loads YAML and injects `FRED_API_KEY`

**Core Logic:**
- `src/market_regime/transforms.py`: feature engineering pipeline (`engineer_all`, derivatives, gap fill)
- `src/market_regime/clustering.py`: PCA and k-means clustering
- `src/market_regime/regime.py`: regime profiling + naming + transition matrix
- `src/market_regime/prediction.py`: supervised training + inference helpers
- `src/market_regime/asset_returns.py`: compute quarterly returns + profile per regime
- `src/market_regime/reporting.py`: dashboard output + portfolio recommendations
- `src/market_regime/checkpoints.py`: artifact checkpointing

**Testing:**
- `tests/unit/`: unit tests for library modules in `src/market_regime/`
- `tests/integration/`: integration tests that exercise multi-module flows
- `tests/conftest.py`: shared fixtures

## Naming Conventions

**Files:**
- Pipeline scripts use a numeric prefix to define ordering: `pipelines/01_ingest.py`, `pipelines/02_features.py`, …, `pipelines/07_dashboard.py`.
- Library modules are snake_case and domain-named: `src/market_regime/asset_returns.py`, `src/market_regime/transforms.py`, `src/market_regime/checkpoints.py`.
- Ingestion submodules are grouped under `src/market_regime/ingestion/` with one module per source (`fred.py`, `multpl.py`, `assets.py`, `grok.py`).

**Directories:**
- `data/processed/` and `data/regimes/` reflect pipeline stage boundaries.
- `outputs/models/`, `outputs/plots/`, `outputs/reports/` reflect consumer output types.

## Where to Add New Code

**New pipeline stage (step N):**
- Primary code: add reusable logic under `src/market_regime/` (new module or extend existing one)
- Orchestration: add a new `pipelines/0N_new_step.py` and (if applicable) wire into `run_pipeline.py` step registry (`STEPS` in `run_pipeline.py`)
- Artifacts: write canonical files under `data/...` and/or `outputs/...` following existing step patterns in `run_pipeline.py`
- Tests: add unit tests under `tests/unit/` and an integration test under `tests/integration/` if it spans multiple modules

**New ingestion source:**
- Implementation: `src/market_regime/ingestion/<source>.py`
- Configuration: add source metadata under `config/settings.yaml` (avoid hardcoding URLs/series IDs in Python)
- Orchestration: call it from `run_pipeline.py:step1_ingest` (macro) or step 6 (assets), depending on source type

**New feature engineering transform:**
- Implementation: `src/market_regime/transforms.py` (integrate into `engineer_all` to preserve ordering)
- Configuration: expose tunables via `config/settings.yaml` (feature lists, window sizes)
- Tests: `tests/unit/` covering transform behavior on small DataFrames

**New plotting helper:**
- Implementation: `src/market_regime/plotting.py` (all plots centralized)
- Output naming: `outputs/plots/{step}_{description}.png` via `_save_or_show()`

## Special Directories

**`.planning/`:**
- Purpose: AI-generated planning and codebase mapping documents for GSD workflows.
- Key location: `.planning/codebase/` (this mapping output)
- Generated: Yes
- Committed: Typically yes (planning artifacts)

**`data/` and `outputs/`:**
- Purpose: runtime-generated artifacts and outputs
- Generated: Yes
- Committed: No (expected to be gitignored; see `.gitignore`)

---

*Structure analysis: 2026-03-16*
