<!-- refreshed: 2026-07-09 -->
# Architecture

**Analysis Date:** 2026-07-09

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    CLI Entry Point (tradingcrab)                        │
│                 src/trading_crab/cli.py, pipeline.py                    │
│                           (App Layer)                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                          Pipeline Orchestration                         │
│   Steps 1-9: Ingest → Features → Cluster → Regime → Predict → Report   │
│            src/trading_crab/pipeline.py + pipelines/*.py shims          │
├─────────────────────────────────────────────────────────────────────────┤
│                  Core Library (trading-crab-lib)                        │
│                  src/trading_crab_lib/                                  │
├────────────────────────────────────────────────────────────────────────┤
│  Ingestion  │  Features   │  Clustering  │  Prediction  │  Reporting   │
│  Layer      │  Layer      │  Layer       │  Layer       │  Layer       │
│  ─────────  │  ───────    │  ───────     │  ──────────  │  ──────────  │
│ multpl.py   │transforms.py│clustering.py │prediction/   │reporting.py  │
│ fred.py     │divergence.py│gmm.py        │__init__.py   │asset_returns.│
│ assets.py   │momentum.py   │hmm.py        │classifier.py │py            │
│ macrotrends │indicators.py│markov.py     │gradient_     │regime.py     │
│ .py         │yield_curve_ │density.py    │boosting.py   │diagnostics.py│
│ grok.py     │features.py  │spectral.py   │              │tactics.py    │
│             │             │cluster_      │              │email.py      │
│             │             │comparison.py │              │              │
└────────────────────────────────────────────────────────────────────────┘
│                        Horizontal Utilities                             │
│  RuntimeConfig (runtime.py) · CheckpointManager (checkpoints.py)        │
│  Config (config.py) · Plotting (plotting/) · Monitoring (monitoring/)   │
└─────────────────────────────────────────────────────────────────────────┘
```

## Component Responsibilities

| Component | Responsibility | File |
|-----------|----------------|------|
| **CLI & Pipeline Orchestration** | Parse args, dispatch 9 steps, manage flow | `src/trading_crab/cli.py`, `src/trading_crab/pipeline.py` |
| **Ingestion** | Fetch FRED, multpl.com, macrotrends, yfinance; merge into `macro_raw.parquet` | `src/trading_crab_lib/ingestion/{fred,multpl,assets,macrotrends,grok}.py` |
| **Feature Engineering** | Cross-ratios, log transforms, gap-fill, derivatives → `features.parquet` | `src/trading_crab_lib/transforms.py`, `divergence.py`, `momentum.py`, `indicators.py` |
| **Clustering** | PCA reduction, KMeans/GMM/HMM/DBSCAN/Spectral, model comparison → `cluster_labels.parquet` | `src/trading_crab_lib/clustering.py`, `gmm.py`, `hmm.py`, `markov.py`, `density.py`, `spectral.py`, `cluster_comparison.py` |
| **Regime Profiling** | Build regime statistics, suggest names, compute transitions → `profiles.parquet` | `src/trading_crab_lib/regime.py` |
| **Supervised Prediction** | Train current-regime + forward classifiers, save models | `src/trading_crab_lib/prediction/__init__.py`, `classifier.py`, `gradient_boosting.py` |
| **Asset Analysis** | Compute quarterly returns by regime, rank assets | `src/trading_crab_lib/asset_returns.py` |
| **Reporting** | Dashboard signals, portfolio construction, email templates | `src/trading_crab_lib/reporting.py`, `email.py` |
| **Diagnostics** | RRG analysis, rolling statistics, ratio tracking | `src/trading_crab_lib/diagnostics.py` |
| **Tactical Classification** | Assign buy_hold/swing/stand_aside per asset | `src/trading_crab_lib/tactics.py` |
| **Visualization** | All matplotlib figures (per-step plotting submodule) | `src/trading_crab_lib/plotting/` (9 files) |
| **Pipeline Monitoring** | Step output validation, CV diagnostics, health summaries | `src/trading_crab_lib/monitoring/` (5 files) |
| **State Persistence** | Checkpoint save/load with parquet + JSON metadata | `src/trading_crab_lib/checkpoints.py` |
| **Configuration** | Load settings.yaml, validate, supply to all steps | `src/trading_crab_lib/config.py` |
| **Runtime Flags** | Dataclass for CLI args (refresh, plots, steps, verbose, etc.) | `src/trading_crab_lib/runtime.py` |

## Pattern Overview

**Overall:** Data-driven regime-switching prediction pipeline with modular steps, checkpoint-based state persistence, and structured reporting.

**Key Characteristics:**
- **Two-stage learning:** unsupervised regime discovery (steps 1-4) → supervised nowcasting/forecasting (steps 5-9)
- **Look-ahead bias guards:** centered features for clustering; causal features for supervised learning (separate checkpoints: `features.parquet` vs `features_supervised.parquet`)
- **Configuration-driven:** all parameters (FRED series, feature lists, clustering k, model depths) live in `config/settings.yaml`, not hardcoded
- **Deterministic:** global random seed at pipeline start; `market_code` column excluded from gap-fill row-validity logic to prevent label-pattern leakage
- **Checkpoint-everything:** every step saves intermediate parquet + metadata; `is_fresh()` checks skip recomputation for data < N days old
- **Three regime sources:** `grok` (AI labels), `clustered` (step 3 output), `predicted` (step 5 output), or none (fully data-driven)

## Layers

**Layer 1: Ingestion (Step 1)**
- Purpose: Fetch raw time-series data from external sources (FRED API, web scrapers, yfinance)
- Location: `src/trading_crab_lib/ingestion/`
- Contains: HTTP fetchers for FRED, multpl.com, macrotrends.net, yfinance; optional grok label loader
- Depends on: config (series list + URLs), requests/fredapi/lxml/yfinance (optional extras)
- Used by: Pipeline step 1; saved to `data/raw/macro_raw.parquet`
- Key innovation: publication-lag shifts for GDP/GNP (+1 quarter) to prevent look-ahead bias

**Layer 2: Feature Engineering (Step 2)**
- Purpose: Transform raw columns into engineered features (ratios, log transforms, gap-fill, derivatives)
- Location: `src/trading_crab_lib/transforms.py`, plus `divergence.py`, `momentum.py`, `indicators.py`, `yield_curve_features.py`
- Contains: `engineer_all()` master function orchestrating 6-step pipeline (see ADR #4 in CLAUDE.md for order)
- Depends on: scipy (BPoly gap-fill), numpy (gradient), raw macro_raw DataFrame
- Used by: Steps 2-3-4-5 (clustering and supervised models)
- Key abstractions: `add_cross_ratios()`, `apply_log_transforms()`, `_fill_column()`, `apply_derivatives()`
- Outputs: Two versions for two purposes:
  - `features.parquet` (centered derivatives) — for steps 3-4 (regime discovery, no look-ahead bias acceptable)
  - `features_supervised.parquet` (causal derivatives) — for step 5 (supervised learning, strict look-ahead guard)

**Layer 3: Unsupervised Regime Discovery (Steps 3-4)**
- Purpose: Find regime structure via clustering; profile and name each regime
- Location: `src/trading_crab_lib/clustering.py`, plus `gmm.py`, `hmm.py`, `markov.py`, `density.py`, `spectral.py`, `cluster_comparison.py`, `regime.py`
- Contains: PCA, KMeans (standard + constrained), GMM, HMM, Markov, DBSCAN, HDBSCAN, Spectral, gap statistic, knee detection
- Key decisions:
  - **PCA fixed at 5 components** (ADR #2) — clustering geometry is sensitive; not variance-threshold
  - **Two clusterings always produced** — `cluster` (best-k from silhouette) + `balanced_cluster` (size-constrained). Downstream defaults to `balanced_cluster` for robust per-regime stats (ADR #3)
  - **Regime naming heuristics** — map cluster centroids to hand-written semantic templates (e.g., "Growth Boom", "Stagflation")
- Outputs: `cluster_labels.parquet`, `profiles.parquet`, `regime_labels.yaml` (manual curation after auto-suggest)

**Layer 4: Supervised Prediction (Step 5)**
- Purpose: Train classifiers to predict current regime and forward transitions from causal features only
- Location: `src/trading_crab_lib/prediction/`
- Contains: Two APIs (ADR #12):
  - **Flat API** (`__init__.py`): Production-grade — single `RandomForestClassifier`, returns dict with regime + probabilities. Used by pipeline, saved to `outputs/models/current_regime.pkl`
  - **Bundle API** (`classifier.py`): Test-friendly — returns dict with models + per-fold CV reports + interpretability helpers. Used by tests only
- Also includes: LightGBM wrapper, GradientBoosting, decision-tree training, forward classifiers (h ∈ {1,2,4,8} quarters)
- Key features: TimeSeriesSplit CV (never shuffle), label confidence weighting (γ from regime smoother), calibration curve output
- Outputs: `outputs/models/{current_regime,dt_current,lgbm_current,forward_*}.pkl`

**Layer 5: Reporting & Tactics (Steps 6-9)**
- Purpose: Summarize results for human decision-making (current regime + assets, buy/sell signals, portfolio weighting, diagnostics)
- Location: `src/trading_crab_lib/reporting.py`, `asset_returns.py`, `regime.py`, `diagnostics.py`, `tactics.py`, `email.py`
- Contains:
  - `asset_returns.py`: Compute quarterly returns by regime, rank assets (Sharpe/vol/mean per regime)
  - `reporting.py`: Build portfolio signals (green/red/yellow per asset), print/save dashboard, portfolio construction helpers
  - `regime.py`: Regime profiling (means/stds), transition matrix, forward probabilities
  - `diagnostics.py`: RRG (Relative Rotation Graph) classification, rolling z-scores, percentile ranks
  - `tactics.py`: Tactical classification (volatility + trend → buy_hold/swing/stand_aside)
  - `email.py`: Weekly report composition (markdown + optional HTML) and SMTP delivery
- Outputs: `outputs/reports/dashboard.csv`, `outputs/reports/weekly_report.md`, `outputs/reports/diagnostics/`

**Layer 6: Horizontal Infrastructure**
- **RuntimeConfig** (`runtime.py`): Dataclass holding all CLI flags (refresh, plots, verbose, steps, etc.). Passed through every step.
- **CheckpointManager** (`checkpoints.py`): Save/load DataFrames as parquet + JSON metadata; validity checks; clear operations. Prevents re-scraping and re-computing.
- **Config** (`config.py`): Load and validate `config/settings.yaml`; convenience loaders for FRED series, multpl URLs, ETF tickers, etc.
- **Plotting** (`plotting/` package): Visualization package with submodules per pipeline stage. Re-exports all plot functions + color constants. Honors `save_plots` / `show_plots` flags.
- **Monitoring** (`monitoring/` package): Pipeline health checks (step output validation, CV fold accuracy, ingestion completeness, regime stability). Logged at INFO; wired into main loop.

## Data Flow

### Primary Request Path (Full Pipeline)

1. **Step 1 — Ingestion** (`src/trading_crab/pipeline.py:step1_ingest()`)
   - Fetch FRED, multpl, macrotrends, yfinance in parallel where possible
   - Merge into quarterly index DataFrame
   - Optionally inject `market_code` column from configured source (grok/clustered/predicted)
   - Save to `data/raw/macro_raw.parquet`
   - Also save `asset_prices.parquet` for step 6

2. **Step 2 — Feature Engineering** (`src/trading_crab/pipeline.py:step2_features()`)
   - Load `macro_raw`, apply `engineer_all(df, causal=False)` → `features.parquet` (centered derivatives)
   - Load same, apply `engineer_all(df, causal=True)` → `features_supervised.parquet` (causal derivatives)
   - Drop trailing row if incomplete (centered rolling derivative lacks edge values)

3. **Step 3 — Clustering** (`src/trading_crab/pipeline.py:step3_cluster()`)
   - Load `features`, apply PCA(5) + StandardScaler
   - Run KMeans k-sweep (k ∈ 2–12), select best k from silhouette + Calinski-Harabasz + Davies-Bouldin
   - Also fit `KMeansConstrained(n_clusters=balanced_k, size_min/max)` for balanced labels
   - Save both: `cluster` (best-k) + `balanced_cluster` (constrained) as separate columns

4. **Step 4 — Regime Labeling** (`src/trading_crab/pipeline.py:step4_regime_label()`)
   - Load `cluster_labels`, build per-regime profiles (mean, std, Sharpe, etc. of key features)
   - Suggest semantic names via heuristic (e.g., "Growth Boom" if high SP500, low VIX)
   - Compute transition matrix + forward probabilities
   - Save to `profiles.parquet`

5. **Step 5 — Prediction** (`src/trading_crab/pipeline.py:step5_predict()`)
   - Load `features_supervised` + `cluster_labels` (causal features + regime labels from full-info clustering)
   - Train RF/DT/LGBM (optional) with TimeSeriesSplit CV on current regime classification
   - Also train forward classifiers (P of regime at t+1, t+2, t+4, t+8)
   - Save models to `outputs/models/`, also auto-save predicted labels as `market_code_predicted` checkpoint

6. **Step 6 — Asset Returns** (`src/trading_crab/pipeline.py:step6_asset_returns()`)
   - Load `asset_prices` (cached from step 1) and regime labels
   - Compute quarterly returns for each ETF
   - Rank by per-regime Sharpe/return/volatility
   - Produce regime-asset heatmap

7. **Step 7 — Dashboard** (`src/trading_crab/pipeline.py:step7_dashboard()`)
   - Load all regime + prediction outputs
   - Call `print_dashboard()` (formatted table), `save_dashboard_csv()`
   - Generate regime-colored timeline, forward probability heatmaps

8. **Step 8 — Diagnostics** (`src/trading_crab/pipeline.py:step8_diagnostics()`)
   - Compute RRG quadrants (relative strength + momentum vs benchmark)
   - Rolling z-scores for configured ratios (Oil:Gold, Bonds:Stocks, etc.)
   - Save summary table to `outputs/reports/diagnostics/`

9. **Step 9 — Tactics** (`src/trading_crab/pipeline.py:step9_tactics()`)
   - Classify each asset as buy_hold / swing / stand_aside based on volatility + trend slope
   - Save to `outputs/reports/tactics.csv`

### Conditional Flows

**Market-Code Injection** (any step with `--market-code NAME`):
- If `NAME == "grok"`: Load from `data/grok_*.pickle` (external LLM labels, static)
- If `NAME == "clustered"`: Load from checkpoint `market_code_clustered` (step 3 output, if saved with `--save-market-code`)
- If `NAME == "predicted"`: Load from checkpoint `market_code_predicted` (step 5 auto-save, latest prediction)
- Otherwise: No external label; clustering fully data-driven

**Checkpointing & Skipping**:
- Each step checks `CheckpointManager.is_fresh(name)` before recomputing
- `--refresh` forces step 1-2 re-scraping/recompute; other steps load cached checkpoints if < 7 days old
- `--recompute` forces step 2; steps 3+ still load cached checkpoints unless they depend on step 2
- `--steps 3,4,5` skips steps 1-2, loads their checkpoints; runs only listed steps

**State Management:**
- **Global state:** None. RunConfig passed explicitly to every function.
- **Mutable shared state:** DataFrame columns (e.g., market_code) are not used to determine gap-fill validity; only feature columns determine row validity (prevents label-pattern leakage into gap-fill boundaries)
- **Model persistence:** sklearn objects pickled to `outputs/models/`, not kept in memory across steps

## Key Abstractions

**RunConfig dataclass** (`src/trading_crab_lib/runtime.py`)
- Purpose: Encapsulate all CLI flags (refresh, plots, verbose, steps, market_code, etc.)
- Created once at entry point, passed to every step
- `from_args(Namespace)` factory converts argparse → RunConfig
- Example usage: `if run_cfg.generate_plots: plot_(..., run_cfg.save_plots, run_cfg.show_plots)`

**CheckpointManager** (`src/trading_crab_lib/checkpoints.py`)
- Purpose: Persistent state — DataFrames as parquet + JSON metadata
- API: `save(df, name)`, `load(name)`, `is_fresh(name, max_age_days)`, `clear(name)`, `clear_all()`
- Metadata tracked: creation timestamp, config hash (settings.yaml MD5), row/col counts
- Special: Preservation checkpoints (`*_secondary`) survive `clear_all()` to audit pre-selection columns
- Example: `if cm.is_fresh("macro_raw", max_age_days=7): df = cm.load(...) else: df = fetch_fresh(...)`

**feature_engineer_all()** (`src/trading_crab_lib/transforms.py`)
- Purpose: Orchestrate the complete feature pipeline (6 steps in fixed order)
- Takes: raw DataFrame + config + `causal` boolean
- Returns: fully engineered DataFrame (or two versions if called twice with causal=True/False)
- Order guaranteed by design: cross-ratios → log → select → gap-fill → derivatives → select (ADR #4)
- Key invariant: gap-fill happens AFTER log transform, BEFORE derivatives

**predict_current()** (`src/trading_crab_lib/prediction/__init__.py`)
- Purpose: Load trained model from pickle, score current quarter
- Returns: dict with `{"regime": int, "probabilities": {0: 0.15, 1: 0.70, ...}}`
- Used by: Dashboard (step 7), notebooks, live scoring scripts
- Example: `pred = predict_current(features_row, model_path="outputs/models/current_regime.pkl")`

**regime.build_profiles()** (`src/trading_crab_lib/regime.py`)
- Purpose: Compute per-regime statistics (mean, std, Sharpe, transition matrix)
- Takes: features DataFrame + regime labels
- Returns: dict with regime_id → dict of statistics
- Used by: Regime naming heuristics, reporting, diagnostics

**plotting module hierarchy** (`src/trading_crab_lib/plotting/`)
- Purpose: All matplotlib figures in one package; consistent palette and save/show behavior
- Submodules: `core.py` (helpers), `ingestion.py`, `features.py`, `clustering.py`, `regime.py`, `prediction.py`, `assets.py`, `diagnostics.py`
- All functions: `plot_X(..., run_cfg: RunConfig)` — honor `save_plots` / `show_plots`
- Color palette: `CUSTOM_COLORS` (5 regimes), `REGIME_CMAP` (matplotlib ListedColormap)
- Example: `from trading_crab_lib import plotting; plotting.plot_regime_timeline(..., run_cfg)`

## Entry Points

**CLI Entry Point**
- Location: `src/trading_crab/cli.py:run_pipeline()`
- Triggers: `tradingcrab` command (pip-installed entry point)
- Responsibilities: Parse argparse, construct RunConfig, call `trading_crab.pipeline.main()`
- Example: `tradingcrab --refresh --recompute --plots --steps 3,4,5`

**Direct Python Entry Point**
- Location: `src/trading_crab/pipeline.py:main()` (can be called as `from trading_crab.pipeline import main; main()`)
- Behavior: Parse sys.argv, construct RunConfig, dispatch step functions

**Backward-Compatibility Shim**
- Location: `run_pipeline.py` (repo root)
- Behavior: Adds repo root to sys.path, imports and calls `trading_crab.pipeline.main()`
- Purpose: Allows `python run_pipeline.py ...` without pip install
- Example: `python run_pipeline.py --refresh --plots`

**Notebook Entry Points**
- Each notebook (01–12) in `notebooks/` imports from `trading_crab_lib` and checkpoint manager
- Example: `from trading_crab_lib.checkpoints import CheckpointManager; cm = CheckpointManager(); features = cm.load("features")`

**Standalone Pipeline Scripts** (for minimal use cases)
- Location: `pipelines/{01..09}_*.py`
- Behavior: Simplified entry points (no RunConfig, no checkpoint management, often hardcoded flags)
- Purpose: Legacy compatibility + examples for minimal runs
- Example: `python pipelines/03_cluster.py`

## Architectural Constraints

- **Threading:** Single-threaded event loop. No asyncio or multiprocessing. Network fetches (FRED, yfinance) are serial (rate-limited to 2 sec/request).
- **Global state:** None. All state is passed via function args (RunConfig, config dict, DataFrames).
- **Circular imports:** Prevented by lazy imports in `__getattr__` for convenience re-exports (`trading_crab_lib.RunConfig`, `trading_crab_lib.CheckpointManager`).
- **Module-level code:** Runs at import time for path resolution (`_resolve_dir()` in `__init__.py`), config validation (`validate_config()` called at end of `load()`). All data loads are lazy (inside step functions).
- **Determinism:** Controlled by `cfg["pipeline"]["random_state"]` (default 42). Set once in `main()` via `np.random.seed()` and `random.seed()`. Gap-fill row validity excludes `market_code` to prevent label-pattern interference.

## Anti-Patterns

### Look-Ahead Bias (the #1 trap)

**What happens:** Clustered regimes are labeled using centered rolling windows (both past and future). Supervised models trained on those same centered features then naively use "future" information to predict current regime.

**Why it's wrong:** The model learns patterns that cannot be reproduced at inference time with only historical data available.

**Do this instead:** Always use `features_supervised.parquet` (causal=True, one-sided rolling windows) for steps 5-9. Use `features.parquet` (causal=False, centered) only for steps 3-4 (regime discovery, where two-sided smoothing is acceptable). See ADR #1 in CLAUDE.md for the full rationale.

### Ignoring Publication Lag

**What happens:** GDP/GNP data are fetched and used as-is without considering that the BEA releases "advance estimate" ~30 days after quarter end. A model trained on Q1 GDP at the end of Q1 has look-ahead bias.

**Why it's wrong:** At the end of Q1, Q1 GDP is unknown. Models trained this way cannot score real-time data accurately.

**Do this instead:** Use `shift: true` in `config/settings.yaml` for any FRED series with significant publication lag. GDP and GNP are shifted +1 quarter automatically in `ingestion/fred.py`. See ADR #7 for the design.

### Changing Clustering Geometry Without Re-Labeling

**What happens:** A developer edits `clustering_features` list in settings.yaml (or changes `n_pca_components`) and re-runs step 3. The new clustering produces different regime assignments. Downstream regime names in `config/regime_labels.yaml` are now misaligned.

**Why it's wrong:** Manual regime labels become incorrect. Supervised models trained on old labels score new data against wrong regime definitions.

**Do this instead:** After changing feature lists or PCA components, delete the old checkpoints and re-run steps 3-7. Update `regime_labels.yaml` after examining new regime profiles. Commit the new YAML. See ADR #2 and #10 for the design.

### Using Standard KMeans Instead of Balanced KMeans

**What happens:** Step 3 produces `cluster` (from standard KMeans best-k) with highly imbalanced sizes (e.g., 200 quarters in regime 0, 30 in regime 4). Regime statistics are unreliable for small regimes.

**Why it's wrong:** With only ~300 quarters, a 30-quarter regime has very noisy mean/std estimates. Portfolio decisions conditioned on this regime's statistics are less stable.

**Do this instead:** Always use `balanced_cluster` (from `KMeansConstrained`) for downstream steps. It ensures each regime has ~60 quarters at k=5. See ADR #3 for the tradeoff rationale.

## Error Handling

**Strategy:** Fail fast on data/config errors; gracefully degrade on network/optional-dependency errors.

**Patterns:**
- **Config validation:** `validate_config()` (called at end of `load()`) checks all required sections and critical keys. Single `ValueError` lists every issue at once.
- **Missing optional dependencies:** Clustering modules (gmm.py, hmm.py, markov.py, density.py) catch `ImportError` and provide helpful messages with install instructions.
- **Network failures:** Ingestion modules (fred.py, multpl.py, assets.py) catch exceptions, log at WARNING, and return empty DataFrame. Pipeline continues with whatever data was fetched (partial ingestion is acceptable; raises flag in monitoring output).
- **Checkpoint misses:** `CheckpointManager.load()` raises `FileNotFoundError` if checkpoint missing (caller must decide: re-compute or exit).
- **No re-raising:** No bare `except:`. All exception handlers are explicit by type.

## Cross-Cutting Concerns

**Logging:** All library modules use `log = logging.getLogger(__name__)` at module level. CLI sets root logger to DEBUG if `--verbose`. Each step logs start/end with timing.

**Validation:** 
- Config: `validate_config()` checks settings.yaml at load time
- Data: `validate_step_output()` in monitoring checks DataFrame shape, NaN fractions, dtypes
- Features: `compute_feature_quality()` warns if NaN > 50% in any column

**Authentication:** 
- FRED: API key from `.env` (required; fails if missing)
- multpl.com: Rate-limited by 2-second delays (no auth)
- yfinance: No auth required
- Email: Optional; SMTP config from `config/email.yaml` or `TC_SMTP_*` env vars

**Caching/Performance:**
- CheckpointManager memoizes via parquet + metadata validation (MD5 of settings.yaml)
- Feature engineering applies `dropna()` on final feature columns (shrinks from 100+ to 70 columns typically)
- PCA is fixed at 5 components (not variance-threshold, which would vary across runs)
- Plotting writes to disk only if `save_plots=True`; `show_plots` controls plt.show() (false in CI)

---

*Architecture analysis: 2026-07-09*
