# CLAUDE.md — Project Guide for Claude Code

This file is read automatically by Claude Code at the start of every session.
It explains what this project is, how to work in it, and what conventions to follow.
Architecture decisions, pitfalls, and development history are all in this file —
no separate ARCHITECTURE.md, DECISIONS.md, or PITFALLS.md exists.

---

## What This Project Is

**Trading-Crab** is a market regime classification and prediction pipeline written in Python.

The core idea: macro-economic time series (quarterly, ~1950–present) are used to
label each calendar quarter with a "market regime" (e.g. Stagflation, Growth Boom,
Rising-Rate Slowdown) using unsupervised clustering. Those labels then feed supervised
models that (a) predict today's regime from currently-available data, (b) predict
regime transitions 1–8 quarters forward, and (c) rank asset-class performance within
each regime to produce portfolio recommendations.

**End goal:** a weekly automated report that says "current regime is X, these assets
are green, hold / buy / sell."

The algorithm reference lives in `legacy/unified_script.py` — the original 1249-line
monolith that is ground truth for every formula, parameter choice, and pipeline order.
**Do not modify any file in `legacy/`.**

The modular pipeline in `src/` and `pipelines/` implements everything that script does,
organized more cleanly, with checkpointing, CLI flags, and dedicated plotting notebooks.

**Reference submodules** — This repo contains two Git submodules used as **read-only
references**. You may `git pull` / `git submodule update` to keep them current, but
never modify or push to them. Use them only to compare implementations and inform
changes to the main repo:
- `gsd-scratch-work-repo-copy/` — GSD framework version of the project (earlier checkpoint)
- `trading-crab-lib-repo-copy/` — Separate trading-crab library repo

---

## Repository Layout

```
trading-crab/
├── CLAUDE.md                      ← you are here (all dev docs in one place)
├── README.md                      ← project overview (user-facing)
├── gsd-scratch-work-repo-copy/    ← READ-ONLY submodule (GSD framework version)
├── trading-crab-lib-repo-copy/    ← READ-ONLY submodule (trading-crab library repo)
├── ROADMAP.md                     ← prioritized feature backlog
├── STATE.md                       ← current pipeline status and known gaps
├── .env.example                   ← copy to .env, fill in FRED_API_KEY
├── pyproject.toml                 ← pip-installable package (src layout)
├── Makefile                       ← common dev shortcuts
│
├── config/
│   ├── settings.yaml              ← ALL tuneable parameters live here
│   └── regime_labels.yaml         ← manually-pinned regime names (edit after clustering)
│
├── data/                          ← gitignored; created at runtime
│   ├── raw/                       ← macro_raw.parquet, asset_prices.parquet
│   ├── processed/                 ← features.parquet (after step 02)
│   ├── regimes/                   ← cluster_labels.parquet, profiles.parquet, …
│   └── checkpoints/               ← timestamped parquet checkpoints (see CheckpointManager)
│
├── legacy/                        ← reference implementation; do not modify
│   └── unified_script.py          ← THE reference — all logic must be reachable here
│
├── notebooks/                     ← plotting/exploration notebooks (one per pipeline stage)
│   ├── 01_ingestion.ipynb
│   ├── 02_features.ipynb
│   ├── 03_clustering.ipynb        ← expanded: 28 investigation cells (GMM, DBSCAN, Spectral, gap stat)
│   ├── 04_regimes.ipynb
│   ├── 05_prediction.ipynb
│   ├── 06_assets.ipynb
│   ├── 07_pairplot.ipynb          ← triple-colored pairplots (unsupervised / grok / RF)
│   └── 08_raw_series.ipynb        ← raw series inspection
│
├── pipelines/                     ← runnable pipeline steps
│   ├── 01_ingest.py
│   ├── 02_features.py
│   ├── 03_cluster.py
│   ├── 04_regime_label.py
│   ├── 05_predict.py
│   ├── 06_asset_returns.py
│   ├── 07_dashboard.py
│   ├── 08_diagnostics.py          ← ratio diagnostics + RRG rotation view
│   └── 09_tactics.py              ← per-asset buy_hold / swing / stand_aside
│
├── run_pipeline.py                ← master entry point with --steps / --refresh / --plots
│
├── requirements.txt               ← pinned runtime dependencies
├── requirements-dev.txt           ← runtime + dev extras (pytest, jupyterlab)
│
├── scripts/
│   ├── setup.sh                   ← automated environment setup
│   ├── jupyter_notebook_local.sh  ← local notebook launcher helper
│   └── run_weekly_report.py       ← weekly report automation (pipeline + archive + email)
│
├── tests/                         ← pytest test suite (496 tests)
│   ├── conftest.py                ← shared fixtures (quarterly_index, raw_macro_df, etc.)
│   ├── fixtures/                  ← test fixture data (currently empty)
│   ├── integration/               ← integration tests (currently empty)
│   ├── test_pipelines_ingest_features.py  ← pipeline steps 1-2 smoke tests
│   ├── test_models_regime.py      ← regime classifier tests (bundle API)
│   ├── test_models_boosting.py    ← GradientBoosting in bundle API
│   ├── test_models_interpret_tree.py ← interpretability helpers (feature ranking + reduced tree)
│   ├── test_models_behavior.py    ← behavior model tests
│   ├── test_models_reporting.py   ← metrics aggregation tests
│   ├── test_email_weekly.py       ← email delivery + weekly report automation
│   ├── test_scripts_weekly_report.py      ← weekly report script (archive, CLI, email)
│   ├── test_constraints_etf_universe.py   ← ETF universe validation
│   ├── test_constraints_frequency.py      ← data frequency validation
│   └── unit/                      ← unit tests for src/ modules
│       ├── test_transforms.py
│       ├── test_clustering.py
│       ├── test_clustering_exploration.py
│       ├── test_cluster_comparison.py
│       ├── test_gmm.py
│       ├── test_density.py
│       ├── test_spectral.py
│       ├── test_checkpoints.py
│       ├── test_returns.py
│       ├── test_prediction_flat.py    ← flat prediction API (production)
│       ├── test_ingestion.py          ← HTTP-mocked tests for multpl, FRED, assets
│       ├── test_diagnostics_rrg.py    ← RRG analysis + rolling statistics
│       ├── test_tactics.py            ← tactical asset classification
│       ├── test_config.py             ← config.load_portfolio()
│       ├── test_regime.py             ← regime profiling + transition matrix
│       ├── test_fred_series_config.py ← FRED settings.yaml validation
│       ├── test_yield_curve_features.py ← yield curve spread features
│       ├── test_reporting.py          ← dashboard signals, portfolio, recommendations
│       ├── test_plotting.py           ← all plot functions (steps 01–06)
│       ├── test_runtime.py            ← RunConfig defaults, from_args, str, logging
│       └── test_ingestion_completeness.py ← ingestion completeness report (P23)
│
├── outputs/                       ← gitignored; created at runtime
│   ├── models/                    ← pickled sklearn models
│   ├── plots/                     ← saved figures (PNG/PDF)
│   └── reports/                   ← dashboard.csv, weekly summaries
│
└── src/trading_crab_lib/             ← installable Python package
    ├── __init__.py                ← defines ROOT, CONFIG_DIR, DATA_DIR, OUTPUT_DIR
    ├── config.py                  ← load(), load_portfolio(), setup_logging()
    ├── runtime.py                 ← RunConfig dataclass (verbose, plots, refresh flags)
    ├── checkpoints.py             ← CheckpointManager (save/load/is_fresh/clear)
    ├── transforms.py              ← ratios, log, select, gap-fill, derivatives, engineer_all
    ├── clustering.py              ← reduce_pca, evaluate_kmeans, pick_best_k, fit_clusters
    │                                 + optimize_n_components, compare_svd_pca,
    │                                 + compute_gap_statistic, find_knee_k
    ├── gmm.py                     ← fit_gmm (returns scaler), select_gmm_k, gmm_labels, gmm_probabilities
    ├── density.py                 ← knn_distances, fit_dbscan_sweep, fit_dbscan, fit_hdbscan_sweep, hdbscan_labels
    ├── spectral.py                ← fit_spectral_sweep (affinity cached), spectral_labels
    ├── cluster_comparison.py      ← compare_all_methods, pairwise_rand_index,
    │                                 extract_rf_feature_importances, recommend_clustering_features
    ├── regime.py                  ← build_profiles, suggest_names, build_transition_matrix
    ├── asset_returns.py           ← compute_quarterly_returns, returns_by_regime, rank_assets_by_regime
    ├── reporting.py               ← asset_signals, print_dashboard, save_dashboard_csv, portfolio helpers
    ├── plotting.py                ← ALL visualization helpers (used by notebooks + pipelines)
    ├── diagnostics.py             ← RRG analysis: rolling_zscore, percentile_rank, normalize_100, compute_rrg
    ├── tactics.py                 ← tactical classification: compute_tactics_metrics, classify_tactics
    ├── email.py                   ← weekly email: load_email_config, build_weekly_email_body, send_weekly_email
    ├── ingestion/
    │   ├── multpl.py              ← lxml scraper for 46 multpl.com series
    │   ├── fred.py                ← FRED API fetcher with publication-lag shift
    │   ├── assets.py              ← yfinance ETF price fetcher (16 ETFs, 3-phase fallback)
    │   └── grok.py               ← load external LLM-assisted quarter classifications
    └── prediction/
        ├── __init__.py            ← FLAT API: train_current_regime(X,y,cfg), train_decision_tree,
        │                             train_forward_classifiers, predict_current — used by run_pipeline.py
        └── classifier.py          ← BUNDLE API with FoldReport + GradientBoosting + interpretability
                                      helpers; backwards-compat layer for tests (see ADR #12 below)
```

---

## How to Run

### Full pipeline (scrape fresh data, recompute everything, generate plots)
```bash
python run_pipeline.py --refresh --recompute --plots
```

### Load from checkpoints, skip re-scraping and re-computing, only re-cluster
```bash
python run_pipeline.py --steps 3,4,5,6,7 --plots
```

### Run individual steps
```bash
python pipelines/01_ingest.py
python pipelines/02_features.py
python pipelines/03_cluster.py
python pipelines/04_regime_label.py
python pipelines/05_predict.py
python pipelines/06_asset_returns.py
python pipelines/07_dashboard.py
python pipelines/08_diagnostics.py
python pipelines/09_tactics.py
```

### CLI flag reference (run_pipeline.py)
| Flag | Effect |
|---|---|
| `--refresh` | Re-scrape multpl.com + re-hit FRED API (slow, ~10 min) |
| `--recompute` | Recompute features from cached raw data (skips scraping) |
| `--plots` | Generate all matplotlib figures and save to `outputs/plots/` |
| `--verbose` | Set logging level to DEBUG |
| `--steps 1,3,5` | Run only the listed step numbers |
| `--no-constrained` | Skip k-means-constrained (if not installed) |
| `--market-code NAME` | Load market_code from `grok`, `clustered`, `predicted`, or any saved checkpoint |
| `--save-market-code` | After step 3, save `balanced_cluster` as `market_code_clustered` checkpoint |
| `--show-plots` | Call `plt.show()` in addition to saving (avoid in headless/CI) |
| `--weekly-report` | Archive weekly_report.md to dated copy + email_body.txt |
| `--send-email` | Send weekly report via SMTP (requires config/email.local.yaml) |

### Jupyter notebooks (exploration / plotting)
```bash
pip install -e ".[dev]"
jupyter lab notebooks/
```

---

## Environment Setup

```bash
# 1. Install package + dev extras
pip install -e ".[dev]"

# 2. Optional but recommended for balanced clustering
pip install k-means-constrained

# 3. Set FRED API key (free at fred.stlouisfed.org/docs/api/api_key.html)
cp .env.example .env
# edit .env: FRED_API_KEY=your_key_here

# 4. Verify
python -c "from trading_crab_lib.config import load; print(load()['data'])"
```

### Key dependencies
| Package | Purpose |
|---|---|
| `fredapi` | FRED macroeconomic data |
| `lxml` | Fast HTML parsing for multpl.com scraper |
| `yfinance` | ETF/equity price history |
| `scipy` | `BPoly.from_derivatives` for gap filling |
| `scikit-learn` | PCA, KMeans, RandomForest |
| `k-means-constrained` | Balanced-size clustering (optional) |
| `matplotlib` / `seaborn` | All visualization |
| `pyarrow` | Parquet checkpoint I/O |

---

## Key Design Decisions

### Checkpoint system
Every pipeline step checks `CheckpointManager.is_fresh(name)` before recomputing.
Checkpoints are stored as parquet files under `data/checkpoints/` with a manifest
tracking creation timestamp and config hash. Pass `--refresh` or `--recompute` to
force regeneration. This is the most important usability feature for day-to-day
development — scraping 46 URLs every run is ~10 minutes.

### Global runtime flags (`RunConfig`)
All runtime behaviour is controlled by a `RunConfig` dataclass (not hardcoded in
modules). Construct it once in `run_pipeline.py` or any pipeline step, and pass it
through. Key flags mirror the legacy script:

```python
@dataclass
class RunConfig:
    verbose: bool = False
    generate_plots: bool = False
    generate_pairplot: bool = False          # seaborn pairplot (slow)
    generate_scatter_matrix: bool = False    # pandas scatter_matrix (slow)
    refresh_source_datasets: bool = False    # re-scrape multpl + FRED
    recompute_derived_datasets: bool = False # recompute features from cached raw
    save_plots: bool = True                  # save figures to outputs/plots/
    show_plots: bool = False                 # plt.show() (use False in CI/headless)
```

### Publication-lag shift
GDP (`fred_gdp`) and GNP (`fred_gnp`) are shifted +1 quarter in `fred.py` to prevent
look-ahead bias. The raw BEA release comes ~30 days after quarter end, so at the end
of Q1 you cannot know Q1 GDP. This is set per-series in `config/settings.yaml`
(`shift: true`).

### Feature pipeline order (transforms.py — `engineer_all`)
1. Cross-asset ratios (10 derived columns: div_yield2, price_gdp, credit_spread, etc.)
2. Log transforms (23 columns → log_{col})
3. Narrow to `initial_features` (36 columns + market_code)
4. Bernstein polynomial gap filling (interior NaNs) + Taylor extrapolation (edges)
5. Smoothed derivatives via `np.gradient` on day-number time axis (d1, d2, d3 per column)
6. Narrow to `clustering_features` (69 columns + market_code)

Steps 3 and 6 are controlled by `initial_features` and `clustering_features` lists in
`config/settings.yaml`. Edit those lists there — not in the Python code.

### PCA is fixed at 5 components
The legacy analysis established 5 PCA components as the working baseline.
`n_pca_components: 5` in settings.yaml. Do not switch to variance-threshold
PCA without benchmarking first — it changes the cluster geometry.

### Two clusterings are always produced
`fit_clusters()` always returns both `cluster` (best-k from silhouette, capped at
`k_cap`) and `balanced_cluster` (size-constrained at `balanced_k`). Downstream
steps default to `balanced_cluster` for regime labeling because equal-size clusters
are better for per-regime statistics with limited data.

### Plotting convention
All visualization helpers live in `src/trading_crab_lib/plotting.py`. Notebooks import
from there — they do not define plotting logic inline. Every plot function accepts
`run_cfg: RunConfig` and honours `save_plots` / `show_plots`. Output filenames are
standardized as `outputs/plots/{step}_{description}.png`.

### Custom color palette
Five-regime color palette from the legacy script:
```python
CUSTOM_COLORS = ["#0000d0", "#d00000", "#f48c06", "#8338ec", "#50a000"]
```
Use `plotting.REGIME_CMAP` everywhere for consistency.

### `prediction/` package — two APIs, two consumers

The `prediction/` subpackage has two modules with deliberately different APIs:

- **`prediction/__init__.py` — flat API** (production): `train_current_regime(X, y, cfg)` returns a
  single fitted `RandomForestClassifier`; `train_decision_tree()` returns a `DecisionTreeClassifier`;
  `predict_current()` returns `{"regime": int, "probabilities": {...}}`. Used by `run_pipeline.py`
  and `pipelines/05_predict.py`. The `outputs/models/current_regime.pkl` file contains a plain RF.

- **`prediction/classifier.py` — bundle API** (backwards-compat): `train_current_regime(X, y, cv_splits=N)`
  returns `{"models": {"rf": ..., "dt": ...}, "cv_reports": {"rf": [FoldReport, ...], ...}, "labels": [...]}`.
  Used only by `tests/test_models_regime.py` and `tests/test_models_reporting.py` which assert on
  per-fold CV indices and aggregate classification-report metrics. **Do not use from pipeline code.**

See ADR #12 below for the full rationale.

---

## Data Sources

### multpl.com (46 series)
Scraped via lxml cssselect from `#datatable`. All URLs and `value_type` metadata
are in `config/settings.yaml` under `multpl.datasets`. Do not hardcode URLs in Python.
Rate-limited to 2 seconds per request (`RATE_LIMIT_SECONDS`).

### FRED API (14 series)
Current: GDP (shifted +1Q), GNP (shifted +1Q), BAA, AAA, CPI (CPIAUCSL), GS10, TB3MS,
VIXCLS, UNRATE, M2SL, M2NS, GS2, T10Y2Y, T10Y3M.

Planned additions (see `ROADMAP.md` Tier 1):
- HOUST (housing starts), UMCSENT (consumer sentiment)

Requires `FRED_API_KEY` in `.env`. Free registration at fred.stlouisfed.org.

### macrotrends.net (planned — not yet implemented)
Gold spot price back to 1915, WTI crude oil back to 1946, silver, copper.
See `ROADMAP.md` Tier 1 item 1.5 and `src/trading_crab_lib/ingestion/macrotrends.py` (to be created).
Scraping approach: extract embedded JSON from `<script>var rawData={...}</script>` tags.

### ETF price history (yfinance)
SPY, GLD, TLT, USO, QQQ, IWM, VNQ, AGG — monthly adjusted close, resampled to
quarterly. Fetched in `ingestion/assets.py`. No API key required.

### Grok baseline labels
`data/grok_quarter_classifications_20260216.pickle` — an external LLM-assisted
classification of quarters used as a visual reference overlay in notebooks. Not used
for model training. Loaded via `ingestion/grok.py` (or directly in notebooks).

---

## Config Reference (settings.yaml)

All tuneable parameters are in `config/settings.yaml`. Key sections:

| Section | Key parameters |
|---|---|
| `data` | `start_date`, `end_date`, `frequency` |
| `fred.series` | per-series `name` + `shift` flag |
| `multpl.datasets` | list of `[name, description, url, value_type]` |
| `features.log_columns` | columns to log-transform |
| `features.initial_features` | columns retained before gap fill |
| `features.clustering_features` | final columns fed to PCA |
| `features.derivative_window` | rolling mean window for np.gradient smoothing |
| `clustering.n_pca_components` | fixed at 5 |
| `clustering.n_clusters_search` | upper bound for k-sweep (default 12) |
| `clustering.k_cap` | max k accepted from silhouette (default 5) |
| `clustering.balanced_k` | k for size-constrained KMeans (default 5) |
| `prediction.forward_horizons_quarters` | [1, 2, 4, 8] |
| `prediction.cv_splits` | 5 (TimeSeriesSplit folds) |
| `prediction.dt_max_depth` | 8 (DecisionTree depth) |
| `prediction.rf_max_depth` | 12 (RandomForest max depth) |

---

## What Must NOT Change Without Discussion

- **The feature pipeline order** — cross-ratios → log → select → gap-fill → deriv → select.
  The Bernstein gap fill must happen AFTER log transform so it interpolates in log space.
- **Publication-lag shifts** — GDP and GNP must always be shifted. Do not remove without
  explicit approval.
- **`clustering_features` list** — this is analytically determined. Changes here change
  the clustering geometry and invalidate any manually pinned `regime_labels.yaml`.
- **`n_pca_components = 5`** — changing this changes which regimes you find. Benchmark first.
- **Saving to `.env` or committing API keys** — never. Use `.env.example` only.
- **`prediction/__init__.py` flat API** — `run_pipeline.py`, `pipelines/05_predict.py`, and
  `pipelines/07_dashboard.py` all expect `current_regime.pkl` to be a bare `RandomForestClassifier`.
  Do not change to the bundle-dict API without updating all three consumers. See ADR #12 below.
- **Reference submodules — no modifications, no pushes** — `gsd-scratch-work-repo-copy/`
  and `trading-crab-lib-repo-copy/` are Git submodules for reference only. Pulling updates
  (`git pull` / `git submodule update`) is fine, but never modify files inside them or push
  to their remotes. Use them to compare implementations and inform changes to the main repo.

---

## What the Legacy Code Does That Must Be Matched

Cross-reference `legacy/unified_script.py` for ground truth on all algorithms.
All items are verified as matching in `src/`; see `STATE.md` for known gaps.

### Algorithms (all ✓ — fully matched in src/)

1. **Scraping** — lxml `cssselect("#datatable tr")`, user-agent string, 2s rate limit
2. **FRED** — per-series `shift`, quarterly resample with `.last()`
3. **Cross-ratios** — exact 10 formulas (div_yield2, price_div, price_gdp, price_gdp2,
   price_gnp2, div_minus_baa, credit_spread, real_price2, real_price3, real_price_gdp2)
4. **Log transform** — `np.log(col.clip(lower=1e-9))`
5. **Gap filling** — `BPoly.from_derivatives` with 4 boundary conditions per side
   (value + d1 + d2 + d3); Taylor extrapolation for leading/trailing edges
6. **Derivatives** — `np.gradient` on matplotlib day-number axis + centered rolling
   mean of window=5 before and after each gradient call
7. **PCA** — `StandardScaler` → `PCA(n_components=5)` → re-`StandardScaler` before KMeans
8. **K-sweep** — `range(2, 13)` with `n_init=50`, silhouette + CH + DB
9. **Balanced clustering** — `KMeansConstrained(size_min=bucket-2, size_max=bucket+2)`
10. **Color palette** — `["#0000d0", "#d00000", "#f48c06", "#8338ec", "#50a000"]`

### Things src/ does better than legacy (do not regress)

- ✓ Real ETF price data via yfinance (16 ETFs) instead of macro-data proxies
- ✓ `CheckpointManager` with parquet + manifest (vs. ad-hoc pickle/CSV)
- ✓ `RunConfig` dataclass for clean flag management
- ✓ All config in `settings.yaml` (vs. hardcoded Python constants)
- ✓ Full CLI in `run_pipeline.py` with `--steps`, `--refresh`, `--recompute`, etc.
- ✓ Dedicated exploration notebooks (01–08)
- ✓ Clustering investigation suite (GMM, DBSCAN, HDBSCAN, Spectral, gap statistic, SVD)

---

## Conventions

### General Python style
- Python 3.10+ — use `match`, `|` union types, `X | None` not `Optional[X]`
- Type hints on all public functions
- `logging` everywhere, no `print()` in library code (only in `pipelines/` and `run_pipeline.py`)
- No bare `except:` — always catch specific exception types
- All file paths via `pathlib.Path`, never string concatenation

### Naming
- DataFrames: noun describing contents (`features`, `pca_df`, `clustered`, `returns`)
- Series: noun describing the single variable (`labels`, `cluster`)
- Functions: verb_noun (`fetch_all`, `apply_log_transforms`, `build_profiles`)
- Config keys: `snake_case` throughout YAML

### Checkpoint files
- Stored under `data/checkpoints/{name}.parquet` (DataFrames) or `{name}.pkl` (models)
- Always prefer parquet over pickle for DataFrames (smaller, typed, readable)
- Pickle only for sklearn models (no parquet-serializable alternative)
- Never commit data files — `data/` and `outputs/` are in `.gitignore`

### Testing
```bash
pytest tests/ -v
```
Tests live under `tests/`. Unit tests should not require network access — mock
`requests.get` for scraping tests and FRED API calls. Use fixtures from `tests/conftest.py`.

### Commits
- Conventional format: `feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `chore:`
- Example: `feat: add yfinance asset price ingestion (step 06)`
- Branch: always `claude/description-sessionID` — never push directly to `main`

---

## Current Status (as of March 2026)

See `STATE.md` for a full breakdown of what runs, what's tested, and what output
files are produced. See `ROADMAP.md` for prioritized feature backlog.

**Summary:** all 9 pipeline steps run end-to-end on real data. **496 tests collected**
(10 skipped: HDBSCAN + cssselect optional). All 5 legacy alignment gaps closed.
Clustering investigation suite (GMM, DBSCAN, Spectral, gap statistic, SVD) fully
implemented. Phase 3 supervised models (RF + DT + GB + forward classifiers) implemented.
New modules: diagnostics (RRG), tactics, email/weekly report.  FRED expanded from 7
to 14 series; yield curve features added.  ETF universe expanded from 16 to 38.
Diagnostics and tactics integrated as pipeline steps 8-9.  Weekly report flow with
`--weekly-report` + `--send-email` CLI flags.  Interpretability tree in step 5.

### Known Limitations
- `regime.py` naming heuristics silently skip 4 features (`10yr_ustreas`, `fred_gs10`,
  `fred_tb3ms`, `div_minus_baa`) because only their derivatives are in `clustering_features`.
  Graceful fallback is intentional.
- ETF data starts 1993-2006; pre-1993 gold and oil regime analysis uses proxy columns only.
  macrotrends.net backfill would extend coverage to 1915+ for gold.
- Clustering uses KMeans which treats each quarter independently; HMM would model
  temporal autocorrelation natively (Tier 2 roadmap item).
- Standalone `pipelines/*.py` scripts do not use `RunConfig` or `CheckpointManager` —
  they are simplified entry points without plot generation or checkpoint management.
  Use `run_pipeline.py --steps N` for full-featured single-step execution.
- `diagnostics.py` and `tactics.py` are not yet integrated into `run_pipeline.py` steps;
  they are available as library modules for notebooks and custom scripts.
- `email.py` requires `config/email.yaml` (not committed; add to `.env.example` pattern).

---

## Frequently Needed Commands

```bash
# Check what checkpoints exist
ls data/checkpoints/

# Run just the clustering step with plots
python run_pipeline.py --steps 3 --plots --verbose

# Reload raw data from pickles (skip re-scraping) and recompute everything
python run_pipeline.py --recompute --plots

# Start fresh (re-scrape multpl + FRED, recompute all)
python run_pipeline.py --refresh --recompute --plots

# Launch notebooks
jupyter lab notebooks/

# Quick sanity check (no network, loads a checkpoint)
python -c "
from trading_crab_lib.checkpoints import CheckpointManager
cm = CheckpointManager()
print(cm.list())
"

# Print current dashboard (requires steps 01-06 to have run)
python pipelines/07_dashboard.py
```

---

## Architecture Decision Records

Documents the "why" behind key design decisions so future contributors
don't accidentally break invariants that look arbitrary.

### ADR #1. Two Feature Files: `features.parquet` and `features_supervised.parquet`

Step 2 produces two separate parquet files from the same raw data.

- **Centered smoothing** (`causal=False`) uses both past and future neighbors in each rolling
  window. Superior for interpolating genuinely missing historical data and characterizing what
  a regime "looks like" across its full span. Used for: clustering (step 3), regime profiling (step 4).
- **Causal smoothing** (`causal=True`) uses only past data in every rolling window — exactly
  what you could compute at the end of a quarter with only information available at that moment.
  Used for: supervised learning (step 5), live scoring (steps 5-7).
- **Critical invariant:** training a supervised model on centered features and then scoring
  "today's" data is look-ahead bias — the model learned patterns that cannot be reproduced in real-time.

**Column names are identical** in both files (intentional). The checkpoint manager uses
`"features"` vs `"features_supervised"` keys to distinguish them.

**Rejected alternative:** single file with a flag column — leads to accidental mixing of
centered and causal features when steps share files.

---

### ADR #2. Five PCA Components (Fixed, Not Variance-Threshold)

`n_pca_components: 5` in `settings.yaml`. Not "keep 90% variance".

- The legacy script established 5 components as the working baseline after experimenting with
  scree plots on the actual 69-column feature matrix.
- Changing the number of PCA components changes the clustering geometry, which changes cluster
  assignments, which invalidates any manually pinned regime names in `config/regime_labels.yaml`.
- Variance-threshold PCA is non-deterministic across data updates (as more data arrives the
  cumulative variance curve shifts). Fixed components are reproducible.

**When to revisit:** if the feature set changes substantially, re-run the scree plot and
benchmark silhouette scores for 3, 5, 7, 10 components. Document the new choice here.

---

### ADR #3. Balanced KMeans as the Primary Regime Assignment

We use `balanced_cluster` (from `KMeansConstrained`) for all downstream steps, not `cluster`
(from standard KMeans with best-k from silhouette).

- Per-regime statistics require sufficient samples to be meaningful. Standard KMeans often
  produces clusters of wildly different sizes (e.g., 70% in one cluster).
- With only ~300 quarters, a cluster of 10 quarters has unreliable mean/std estimates.
- `KMeansConstrained(size_min=bucket-2, size_max=bucket+2)` ensures each regime has ~60
  quarters at k=5, giving reliable statistics for all downstream computations.

**Tradeoff:** balanced clustering slightly distorts cluster geometry — some quarters near a
boundary get assigned to a less-natural regime to meet the size constraint. Acceptable: the
goal is interpretable regimes with robust statistics, not geometrically pure clusters.

**Rejected alternative:** hierarchical clustering — doesn't produce equal-size clusters and
has no clear stopping rule for k.

---

### ADR #4. Bernstein Polynomial Gap Fill in Log Space

Gap fill happens AFTER log transform.

- Raw series (e.g., S&P 500, GDP) are exponential-looking. Interpolating between 1000 and 2000
  in linear space overshoots. In log space, the midpoint of [log(1000), log(2000)] = log(1414).
- Bernstein polynomials require 4 boundary conditions per side (value, d1, d2, d3). All three
  derivatives must also be computed in log space for consistency.
- **Invariant:** the order is always: cross-ratios → log → select → gap-fill → derivatives → select.
  Do not move gap fill before log transform.

**Why Bernstein (not cubic spline)?** `BPoly.from_derivatives` exactly matches value + first 3
derivatives at both endpoints — smooth and compatible with derivative features computed afterward.
Cubic splines minimize curvature globally; Bernstein interpolates boundary conditions locally.
For gap filling (usually 1-4 quarters), local is better.

---

### ADR #5. Taylor Extrapolation for Edge Gaps

Use Taylor expansion (not Bernstein) for leading and trailing edge gaps.

Bernstein requires boundary conditions on both sides. For edge gaps (missing data at the start
or end of the time series), one side has no neighbors. Taylor extrapolation uses value + d1 + d2
+ d3 at the known edge to project outward: `f(x+h) ≈ f(x) + h·f'(x) + (h²/2)·f''(x) + ...`
This is mathematically consistent with the interior Bernstein approach.

---

### ADR #6. CheckpointManager: Parquet for DataFrames, Pickle for Models

**Parquet for DataFrames:** smaller files (columnar compression), typed (dtypes preserved),
human-inspectable (duckdb/pandas/parquet-viewer), no Python version lock-in.

**Pickle for sklearn models:** sklearn's serialization format is pickle; no parquet-serializable
alternative exists for a fitted `RandomForestClassifier`. Risk: pickle files are Python-version-
sensitive. Mitigation: use `joblib.dump` which is slightly more stable. (TODO: migrate from
`pickle.dump` to `joblib.dump` in `pipelines/05_predict.py`)

---

### ADR #7. Publication-Lag Shifts for GDP and GNP

`fred_gdp` and `fred_gnp` are shifted +1 quarter.

The BEA releases the "advance estimate" of GDP approximately 30 days after quarter end; the
"third estimate" (most revised) comes ~90 days later. At the end of Q1 you cannot know Q1 GDP.
Not shifting introduces look-ahead bias. This is set in `config/settings.yaml` as `shift: true`
per series. **Invariant:** all FRED series with significant revision history and a publication
lag longer than one quarter should be shifted.

---

### ADR #8. Runtime Flags via `RunConfig` Dataclass

All runtime behavior is controlled by a single `RunConfig` object passed through the pipeline,
not by global variables or config file values.

- Avoids action-at-a-distance bugs where a deeply nested module checks a global flag set elsewhere.
- Makes the pipeline deterministic and testable: pass `RunConfig(generate_plots=False)` in tests
  to skip all matplotlib code without monkeypatching globals.
- The dataclass `from_args()` factory converts argparse `Namespace` to `RunConfig` in
  `run_pipeline.py` — the only place argparse is used.

---

### ADR #9. Two-Clustering Architecture (Standard + Constrained)

Always produce both `cluster` and `balanced_cluster`, even though only `balanced_cluster`
is used downstream.

- `cluster` (unconstrained, best-k from silhouette) serves as a geometric reference: if
  `balanced_cluster` looks very different, the size constraint is distorting natural clusters.
- Having both lets you visually compare in notebooks without re-running clustering.
- The k-sweep silhouette scores that determine `best_k` are saved (`data/regimes/kmeans_scores.parquet`)
  for elbow-curve visualization.

---

### ADR #10. Config-Driven Feature Lists

`initial_features` and `clustering_features` lists live in `config/settings.yaml`, not
hardcoded in Python. These were analytically determined by examining which series have coverage
back to ~1950 and which derivatives are informative for clustering. Putting them in YAML lets
you experiment without touching Python source code. **Invariant:** changing `clustering_features`
changes clustering geometry and invalidates `regime_labels.yaml`. Delete the old checkpoint
and re-run steps 3-7 before committing.

---

### ADR #11. All Visualization in `plotting.py` — Never Inline in Notebooks

Notebooks call functions from `src/trading_crab_lib/plotting.py`; they do not define plotting logic
inline. Reasons: reusability (same plot needed in notebook AND CLI `--plots` mode), testability
(plotting functions can be tested by mocking matplotlib), consistency (same palette and naming),
DRY (prevents three slightly-different versions of the same chart drifting apart). If you need
a new plot, add it to `plotting.py` first, then call it from the notebook.

---

### ADR #12. `prediction/` Package: Two APIs, One Consumer Each

`prediction/__init__.py` (flat API) and `prediction/classifier.py` (bundle API) coexist in the
same package but serve different consumers and must not be conflated.

**Context:** During a GSD-assisted development session (March 2026), an alternative
`pipelines/05_predict.py` was generated using a "bundle" API returning a dict
`{"models": {"rf": ..., "dt": ...}, "cv_reports": {...}}`. This made it easy to write tests
asserting on per-fold CV metadata. However, adopting it as the production API would have required
simultaneous changes to:
- `run_pipeline.py` (`step5_predict`) — imports and uses the flat API
- `pipelines/07_dashboard.py` — loads `current_regime.pkl` assuming a bare `RandomForestClassifier`

**Decision:** keep the flat API in `prediction/__init__.py` as production. Create
`prediction/classifier.py` as a backwards-compatible layer for tests that need to inspect
per-fold `FoldReport` objects or aggregate classification-report dicts across folds.

**Rules that must hold:**
- `run_pipeline.py` and all `pipelines/*.py` scripts import from `trading_crab_lib.prediction` (flat API).
- `tests/test_models_regime.py` and `tests/test_models_reporting.py` import from
  `trading_crab_lib.prediction.classifier` (bundle API).
- `outputs/models/current_regime.pkl` always contains a bare `RandomForestClassifier`.
- Do not "simplify" by merging the two modules — the bundle dict cannot be pickled as
  `current_regime.pkl` without breaking `07_dashboard.py`.
- If you add a new classifier, add it to the flat API first. Only add bundle-API support in
  `classifier.py` if a test specifically needs per-fold CV metadata for the new model type.

---

## Known Pitfalls and Gotchas

A collection of traps, anti-patterns, and non-obvious failures discovered during development.
Read before making changes.

### Look-ahead Bias (the #1 Financial ML Sin)

**P1. Using centered rolling windows for supervised learning**

Symptom: model accuracy looks great but real-time predictions are wrong. Cause:
`rolling(window=5, center=True)` uses 2 future quarters in every window — a model trained on
centered features can only be scored on centered features, which requires knowing the future.
Fix: always use `features_supervised.parquet` (causal=True) for steps 5-7. `features.parquet`
(causal=False) is for clustering steps 3-4 only. Never swap them.

**P2. Not applying publication-lag shifts to GDP/GNP**

Symptom: model learns to use Q1 GDP to predict Q1 regime label. Fix: `shift: true` in
`config/settings.yaml` for `fred_gdp` and `fred_gnp`. Any FRED series with significant revision
history and a release lag longer than one quarter must be shifted. Check BEA release calendar.

**P3. Using clustering labels as supervised training targets without alignment**

Symptom: `X` and `y` have different lengths; `.dropna()` removes extra rows silently. Cause:
clustering runs on `features.dropna()` which may drop leading rows. Fix — always use index intersection:
```python
common = features.index.intersection(labels.index)
X = features.loc[common].drop(columns=["market_code"], errors="ignore").dropna(axis=1, how="any")
y = labels.loc[common]
```
Never use `iloc[:len(labels)]` — this silently misaligns if any rows were dropped.

---

### Temporal Leakage in Cross-Validation

**P4. Using `train_test_split` (shuffled) for time-series data**

Symptom: CV accuracy is 95%; production accuracy is 60%. Fix: always use
`TimeSeriesSplit(n_splits=5)`. `shuffle=False` is not enough — you need `TimeSeriesSplit`
which ensures all training data precedes all test data in each fold.

**P5. Forward-looking binary classifiers: label alignment**

`y_future = y.shift(-h)` introduces NaN at the end. Current code does
`y_future = y.shift(-h).dropna()` and then `X_aligned = X.loc[y_future.index]`. This is correct.
Do not simplify to `X.iloc[:len(y_future)]`.

---

### SSL and Network Issues

**P6. yfinance "self signed certificate in chain" error**

`assets.py` sets `CURL_CA_BUNDLE` and `SSL_CERT_FILE` to `certifi.where()` at module load.
Do not remove those lines.

**P7. multpl.com rate limiting**

Never reduce `RATE_LIMIT_SECONDS` below 2. The `--refresh` flag should only be used when
genuinely needed. Use checkpoints for development iteration.

---

### Python Version and Dependency Issues

**P8. `X | Y` union type syntax on Python < 3.10**

Add `from __future__ import annotations` at the top of every module that uses `X | Y` syntax.
All `src/trading_crab_lib/` files should have this.

**P9. `contourpy` and other transitive deps failing on Python 3.10**

`requirements.txt` uses `>=` minimum bounds (not exact pins) for direct dependencies only.
Never regenerate with `pip-compile --generate-hashes`.

**P10. `k-means-constrained` compilation on some platforms**

Use the `--no-constrained` flag which falls back to standard KMeans. The `setup.sh` script
prompts before attempting installation.

---

### Data and Config Pitfalls

**P11. Changing `clustering_features` invalidates `regime_labels.yaml`**

After any change: (1) delete `data/checkpoints/cluster_labels*` and
`data/regimes/cluster_labels.parquet`, (2) re-run steps 3-4, (3) inspect new regime profiles
and update `config/regime_labels.yaml`, (4) commit the new YAML.

**P12. `end_date: "2025-09-30"` in settings.yaml is hardcoded**

Pipeline silently ignores data after that date. Fix: change to `null` and handle in
`ingestion/fred.py` and `ingestion/multpl.py` using `datetime.today()`.

**P13. Checkpoint freshness check uses wall-clock time, not data time**

`cm.is_fresh("macro_raw", max_age_days=7)` returns True even if FRED released new data
yesterday. For production, always run with `--refresh` on Fridays. The weekly cron job
(Tier 3 roadmap) should always pass `--refresh`.

---

### Clustering Pitfalls

**P14. Silhouette score selects k=2 when data is bimodal**

Real macro data often has two dominant modes (growth vs recession) that score highest at k=2.
`k_cap: 5` in `settings.yaml` caps the accepted k at 5. `balanced_k: 5` forces 5 balanced
clusters regardless of silhouette result.

**P15. PCA re-scaling before KMeans**

PCA components are not unit-variance. `StandardScaler` must be applied AFTER PCA and BEFORE
KMeans. **Invariant:** `features → StandardScaler → PCA(5) → StandardScaler → KMeans`.

---

### Plotting Pitfalls

**P16. `plt.show()` in headless environments**

`run_cfg.show_plots = False` by default. Only set `True` via `--show-plots` locally.
CI/CD pipelines should never pass `--show-plots`.

**P17. Seaborn pairplot is very slow on large feature sets**

Pairplot with 69 features generates 69×69 = 4761 subplots. Disabled by default
(`generate_pairplot: False` in RunConfig). Enable only when specifically investigating
feature relationships.

---

### Portfolio Construction Pitfalls

**P18. `generate_recommendation()` parameter order differs from legacy**

Always call with keyword arguments:
```python
generate_recommendation(target_weights=blended, current_weights=None)
```
Never rely on positional argument order for this function.

**P19. `blended_regime_portfolio()` probabilities must sum to ~1.0**

Only use `prediction["probabilities"]` (from the multi-class RF) as input to
`blended_regime_portfolio()`. Forward classifier probabilities (binary, one per regime) are
independent binary classifiers that do NOT sum to 1.0 — they are not valid blending inputs.

---

### Test Suite Pitfalls

**P20. Running `pytest` no longer corrupts the `macro_raw` checkpoint** — FIXED

`tests/test_pipelines_ingest_features.py` uses `monkeypatch.setattr(step, "DATA_DIR", tmp_path)`
to redirect all file I/O to pytest's temporary directory. No production checkpoint files are
touched during `pytest`.

---

### Behavior Model Pitfalls

**P21. `make_behavior_labels` uses strict inequalities — exactly-at-threshold is "flat"**

`r > up_threshold` and `r < down_threshold` (strict). With both thresholds at `0.0`:
- `r > 0`: "up" | `r < 0`: "down" | `r == 0`: "flat"

This is intentional. Do not change to `>=` / `<=` — the test suite verifies strict behavior.

---

### Tech Debt and Security Pitfalls

**P22. SSL verification is disabled in `ingestion/assets.py`**

Uses `curl_cffi.requests.Session(verify=False)` unconditionally — susceptible to MITM on price
data. Planned fix: add a `RunConfig` / settings flag to control SSL verification, defaulting
to secure.

**P23. Partial ingestion silently produces plausible-looking outputs**

Ingestion failures are caught and logged at WARNING level but the pipeline continues with
whatever data was successfully fetched. Check `macro_raw.parquet` column count after ingestion;
should be ~53 columns. Planned fix: add ingestion completeness report.

**P24. `CheckpointManager.list()` silently ignores corrupt metadata files**

Catches all JSON parse errors without logging which file failed. Fix: log at WARNING which file
failed to parse before continuing.

**P25. Committed data artifacts in `data/` can create stale-data bugs**

`data/fred_api_datasets_snapshot_20260216.pickle`, `data/multpl_datasets_snapshot_20260216.pickle`,
`data/grok_quarter_classifications_*.pickle` — if the pipeline accidentally loads these instead
of freshly-fetched data, results are silently based on Feb 2026 snapshots. Planned fix: move to
`data/archives/` with explicit documentation, or move small fixtures to `tests/fixtures/`.

**P26. FRED ingestion hard-fails when `FRED_API_KEY` is missing**

`fred.py`'s `fetch_all()` calls `fredapi.Fred(api_key=...)` which raises if the key is None.
Fix: copy `.env.example` to `.env` and add your free key from fred.stlouisfed.org.

**P27. Pickle files are an arbitrary-code-execution risk**

`outputs/models/current_regime.pkl` and all other pickles execute arbitrary code on load.
Never load a pickle file whose provenance you cannot verify. Planned fix: migrate sklearn model
serialization to `joblib.dump` / `joblib.load`.

---

## Development Decisions Log

A chronological log of judgment calls that don't rise to the level of a formal ADR but are
important for future contributors to know about.

### D1. GSD `pipelines_from_gsd_version/05_predict.py` — NOT adopted (2026-03-16)

The GSD-generated `05_predict.py` (bundle API) was reviewed and rejected. The existing
`pipelines/05_predict.py` (flat API) is canonical. Adopting the GSD version would have required
simultaneous changes to `run_pipeline.py` and `pipelines/07_dashboard.py` with no immediate
benefit. **What WAS adopted:** the monkeypatch fix in `pipelines/02_features.py` — changed direct
import of `engineer_all` to a module-level reference so `monkeypatch.setattr` works in tests.

### D2. `prediction/` converted from flat module to package (2026-03-16)

`src/trading_crab_lib/prediction.py` was converted to a package so new test files could import from
`trading_crab_lib.prediction.classifier`. Split: existing flat-API content moved intact to
`__init__.py`; new `classifier.py` created with bundle API. See ADR #12.

### D3. `make_behavior_labels` changed to strict inequalities (2026-03-16)

Changed `r >= up_threshold` / `r <= down_threshold` to `r > up_threshold` / `r < down_threshold`.
With both thresholds at `0.0`, a return of exactly `0.0` was incorrectly classified as "up".
Impact: extremely rare on real price data; only affects synthetic test data.

### D4. GSD `01_ingest.py` and `02_features.py` wrappers — NOT adopted (deferred) (2026-03-16)

GSD wrappers adding `--refresh`, `--verbose`, `--market-code` CLI flags to standalone pipeline
scripts were reviewed but not applied. `run_pipeline.py --steps 1,2` already provides the same
functionality. Revisit if `step1_ingest()` and `step2_features()` are significantly changed.

### D5. Pipeline smoke tests use `tmp_path` — checkpoint contamination eliminated (2026-03-16)

`tests/test_pipelines_ingest_features.py` redirects all file I/O to pytest's `tmp_path` fixture
using `monkeypatch.setattr(step, "DATA_DIR", tmp_path)`. No production checkpoint files are
written during `pytest`. The `--recompute` workaround after test runs is no longer needed.

### D6. `pipelines_from_gsd_version/` removed from repo (2026-03-16)

These scripts represented an alternative pipeline design explored via the GSD framework.
They were deleted in commit `bc3bc1b` as they cluttered the repo. The decisions about which
changes were adopted are documented in D1 and D4 above.

### D7. `legacy/` kept in repo (per owner decision) (2026-03-16)

`legacy/unified_script.py` is the algorithm ground truth. When implementing a remaining gap,
refer to `unified_script.py`, not any modular legacy files, to avoid inconsistencies.

### D8. FRED expanded from 7 to 14 series (2026-03-18)

Added VIXCLS, UNRATE, M2SL, M2NS, GS2, T10Y2Y, T10Y3M to `config/settings.yaml`.
All new series use `shift: false` (no publication-lag shift needed — these are released
with minimal delay). `end_date` changed from hardcoded `"2025-09-30"` to `null` (P12 fix).

### D9. Yield curve features added to transforms pipeline (2026-03-18)

New `src/trading_crab_lib/yield_curve_features.py` module with `add_yield_curve_features()`.
Computes 10Y-2Y and 10Y-3M spreads from multpl.com treasury columns and/or FRED columns
(GS10-GS2, T10Y2Y, T10Y3M). Hooked into `engineer_all()` in `transforms.py` after
cross-ratios step. Does not affect `clustering_features` list — spreads are available for
analysis but must be explicitly added to the feature lists to influence clustering.

### D10. GradientBoosting added to bundle API (2026-03-18)

`classifier.py` now supports `include_gb=True` on `train_current_regime()` and
`train_forward_classifiers()`. Uses `GradientBoostingClassifier` (sklearn, not LightGBM)
for zero-dependency convenience. The flat API in `prediction/__init__.py` is NOT changed —
production still uses bare RF. GB is bundle-API-only for comparative testing.

### D11. Interpretability helpers added to classifier.py (2026-03-18)

`extract_top_features(model, feature_names, top_k)` ranks features by importance.
`train_interpretability_tree(X, y, model, top_k, max_depth)` trains a shallow DT on
only the most important features for human-readable decision rules. Both are in
`classifier.py` (bundle API side) since they're analysis tools, not production inference.

### D12. New modules: diagnostics, tactics, email (2026-03-18)

Three new library modules created from GSD Phase 6-8 designs:

- **`diagnostics.py`** — Relative Rotation Graph (RRG) analysis. `compute_rrg()` classifies
  assets into LEADING/WEAKENING/LAGGING/IMPROVING quadrants based on relative strength and
  momentum vs benchmark. Also provides `rolling_zscore()`, `percentile_rank()`, `normalize_100()`.

- **`tactics.py`** — Tactical asset classification. `compute_tactics_metrics()` computes
  volatility, trend slope, and benchmark correlation. `classify_tactics()` assigns buy_hold /
  swing / stand_aside based on vol + trend thresholds.

- **`email.py`** — Weekly email delivery. `load_email_config()` reads `config/email.yaml`,
  `build_weekly_email_body()` composes from report files, `send_weekly_email()` sends via
  SMTP (TLS or SSL). Paired with `scripts/run_weekly_report.py` for full automation.

These are library modules only — not yet integrated as pipeline steps. Use from notebooks
or `scripts/run_weekly_report.py`.

### D13. Test suite expanded from 238 to 294 tests (2026-03-18)

Added 56 new tests covering previously untested modules:
- `config.load_portfolio()` (4 tests), `regime.py` (5 tests), FRED config validation (1 test)
- Flat prediction API (5 tests), GradientBoosting (2 tests), interpretability (2 tests)
- Ingestion HTTP-mocked tests: multpl (6), FRED (5), assets (4)
- Diagnostics/RRG (8 tests), tactics (7 tests), email/weekly report (15 tests)
- Yield curve features (2 tests)

Coverage gaps closed: `prediction/__init__.py`, `config.load_portfolio()`, `regime.py`,
all three ingestion modules, and three new modules.

### D14. Tier 2 improvements: test coverage, joblib migration, P23/P24 fixes (2026-03-18)

**Test coverage** for previously untested modules — 68 new tests:
- `reporting.py` (15 tests): dashboard signals, portfolio construction, recommendations,
  recommendation digest, weekly report
- `plotting.py` (20 tests): all plot functions (steps 01–06), `_save_or_show`, `_regime_color`,
  constants, empty-input edge cases
- `runtime.py` (25 tests): defaults, `from_args()` with all flag combinations, `apply_logging()`,
  `__str__()` representation
- Ingestion completeness report (8 tests): missing columns, high-NaN detection, summary formatting

**P27 fix — pickle → joblib migration** across 7 files:
- `checkpoints.py`: `save_model()` / `load_model()` now use `joblib.dump` / `joblib.load`
- `pipelines/05_predict.py`, `pipelines/07_dashboard.py`, `run_pipeline.py`: all model
  serialization switched from `pickle` to `joblib`
- `cluster_comparison.py`: RF feature importance loading via `joblib.load`
- `tests/unit/test_cluster_comparison.py`: test fixture uses `joblib.dump`
- `requirements.txt`: added `joblib>=1.3`

**P24 fix — CheckpointManager corrupt metadata logging:**
- `is_fresh()`: catches `json.JSONDecodeError` / `KeyError` / `ValueError` and logs WARNING
  with the specific file and error before returning `False`
- `list()`: catches all metadata parse errors and logs WARNING with file name

**P23 fix — Ingestion completeness report:**
- New `ingestion_completeness_report()` in `src/trading_crab_lib/ingestion/__init__.py`
- Returns `CompletenessReport` dataclass with missing columns, extra columns, high-NaN columns
- Integrated into `pipelines/01_ingest.py` and `run_pipeline.py` step 1
- Builds expected column list from config (FRED + multpl + macrotrends)

Test count: 301 → 428 collected (11 skipped: HDBSCAN + cssselect optional).
All previously untested modules now have test coverage.

### D15. Package renamed from `market_regime` to `trading_crab_lib` (2026-03-19)

Atomic rename of the Python package directory `src/market_regime/` → `src/trading_crab_lib/`
plus ~438 import references across 89 files. pip package name: `trading-crab-lib`.
See `RENAME_PLAN.md` for the full rename strategy. `market_code` (a DataFrame column name)
was NOT renamed — it is a data concept, not a package reference.

### D16. Submodule comparison: main repo is authoritative (2026-03-19)

Compared `gsd-scratch-work-repo-copy/` and `trading-crab-lib-repo-copy/` against the main
repo. Both submodules are earlier snapshots — the main repo is strictly ahead. No GSD-only
functionality needs porting. Key differences:
- GSD has 7 FRED series; main has 14
- GSD has 28 test files; main has 35+
- GSD lacks: LightGBM, macrotrends scraper, ingestion completeness report, forward
  probabilities, confusion matrix plot, diagnostics/tactics/email modules
Submodules remain as read-only references only.

### D17. Cross-asset divergence features implemented (2026-03-19)

ROADMAP item 2.15 — Phases A+B complete. New `src/trading_crab_lib/divergence.py` module:
- `compute_rolling_correlation()`: trailing Pearson correlation between signal pairs
- `compute_divergence()`: short-window vs long-window correlation departure (raw, abs, z-score)
- `compute_divergence_triggers()`: binary triggers when |z-score| > threshold, plus direction
- `compute_derivative_divergence()`: divergence in d1 (derivative) space for leading indicators
- `add_divergence_features()`: master wrapper, config-driven signal pairs and windows

Hooked into `engineer_all()` in two places: (1) level-space divergence after momentum features
(before log transforms), (2) derivative-space divergence after derivatives are computed.
Default pairs: SPY/TLT, SPY/GLD, GLD/Oil, CreditSpread/VIX. Config: `features.divergence`.
Per pair: 5 level columns + 3 derivative columns = 8 features.
29 tests in `tests/unit/test_divergence.py`.

Phases C (add to clustering/supervised feature lists) and D (evaluate impact) deferred.

### D18. Momentum and cross-asset ratio features (2026-03-19)

ROADMAP item 2.12 implementation. New `src/trading_crab_lib/momentum.py` module with:
- `compute_trailing_momentum()`: 2Q, 4Q, 8Q trailing returns for major series
- `compute_relative_strength()`: S&P-in-Gold, S&P-in-Oil, Gold-in-Oil ratios
- `compute_rolling_cross_correlation()`: rolling 8Q correlation between signal pairs
- `compute_inflation_acceleration()`: 2nd derivative of CPI
Hooked into `engineer_all()` in `transforms.py`. Features available for analysis;
must be explicitly added to feature lists in `settings.yaml` to influence clustering.

### D19. Divergence features Phase C+D: feature list integration and evaluation (2026-03-19)

**Phase C** — Added divergence features to `settings.yaml` feature lists:
- `initial_features`: added `sp500` (raw level for derivative-space), `div_spy_tlt_z_4q`,
  `div_spy_tlt_trigger`, `div_cred_vix_z_4q`, `div_cred_vix_trigger`
- `clustering_features`: added 10 divergence columns (level z-scores + d1 derivatives +
  triggers + derivative-space z-scores + triggers for spy_tlt and cred_vix pairs)
- Fixed `fred_vixcls` → `fred_vix` column name in DEFAULT_DIVERGENCE_PAIRS

**Phase D** — Evaluated impact via `scripts/evaluate_divergence.py`:
- **Clustering improved**: silhouette +0.032 (0.189→0.221), CH +6.8, DB −0.10. Improvement
  consistent across k=2–6 in sweep; strongest at k=5 (+0.046 silhouette)
- **Supervised accuracy**: −0.018 mean CV accuracy (within noise, 36.4%→34.6%). However,
  `div_cred_vix_z_4q_d1` ranks 5th/80 in RF feature importance — the signal is there but
  may need feature selection or more data to improve CV generalization
- **Transition detection**: SPY-TLT z-score is 36% higher at regime transitions (0.92) vs
  baseline (0.67) — confirmed as a leading indicator. Other pairs inconclusive with current data
- **Recommendation**: keep in clustering (clear win). For supervised, defer until gold/oil data
  activates additional pairs, or apply feature selection to prune noisy divergence columns

### D20. ETF prices moved to step 1; macrotrends + asset derivatives added (2026-03-19)

**Architectural change**: ETF price ingestion moved from step 6 to step 1. Prices are now
fetched alongside FRED/multpl/macrotrends data and cached as the `asset_prices` checkpoint.
Step 6 reuses cached prices instead of re-fetching (unless `--refresh-assets` is passed).

**Why**: ETF price derivatives (d1, d2) of major asset classes are informative for regime
classification. Moving ingestion to step 1 makes ETF data available for step 2 feature
engineering alongside macro data.

**New data flow**:
- `_fetch_and_cache_asset_prices()`: extracted from step 6, now called in step 1
- `_merge_asset_prices_into_raw()`: merges a curated ETF subset (SPY, TLT, GLD, QQQ, VNQ)
  into `macro_raw` as `etf_spy`, `etf_tlt`, etc. columns
- Config: `features.asset_price_columns` controls which tickers merge into macro_raw

**Feature list additions** (`config/settings.yaml`):
- `log_columns`: added `gold_spot`, `wti_crude`, `etf_spy`, `etf_tlt`, `etf_gld`, `etf_qqq`, `etf_vnq`
- `initial_features`: added `log_gold_spot`, `log_wti_crude`, `log_etf_*` (available for supervised learning)
- `clustering_features`: added `log_gold_spot_d1`, `log_gold_spot_d2`, `log_wti_crude_d1`, `log_wti_crude_d2`

**Key decision**: ETF derivatives (e.g. `log_etf_spy_d1`) are intentionally NOT in
`clustering_features`. ETFs start 1993–2004 (~80 quarters), while clustering uses 305 quarters
back to 1950. Adding ETF derivatives to clustering features would force `dropna()` to discard
all pre-1993 rows, losing 55 years of regime history. Gold (1915+) and oil (1946+) from
macrotrends have enough history for clustering. ETF derivatives remain available for supervised
learning via `features_supervised.parquet`.

**Divergence auto-activation**: `spy_gld` and `gld_oil` divergence pairs (in
`DEFAULT_DIVERGENCE_PAIRS`) will now auto-activate once macrotrends data populates
`gold_spot` and `wti_crude` columns in `macro_raw`.

### D21. Momentum features Phase C+D: feature list integration and evaluation (2026-03-20)

**Phase C** — Added momentum features to `settings.yaml` feature lists:
- `initial_features`: added `sp500_mom_4q`, `sp500_mom_8q`, `10yr_ustreas_mom_4q`,
  `credit_spread_mom_4q`, `corr_sp500_10yr_ustreas_8q`, `cpi_acceleration`
- `clustering_features`: added 11 momentum columns (raw rate-like values + d1 derivatives):
  `sp500_mom_4q`, `sp500_mom_4q_d1`, `sp500_mom_8q`, `10yr_ustreas_mom_4q`,
  `10yr_ustreas_mom_4q_d1`, `credit_spread_mom_4q`, `credit_spread_mom_4q_d1`,
  `corr_sp500_10yr_ustreas_8q`, `corr_sp500_10yr_ustreas_8q_d1`,
  `cpi_acceleration`, `cpi_acceleration_d1`
- Fixed `fred_vixcls` → `fred_vix` column name in `default_mom_cols` (line 210)

**Phase D** — Evaluation script `scripts/evaluate_momentum.py`:
- Same A/B methodology as divergence evaluation: compare clustering quality,
  supervised accuracy, and transition detection with/without momentum features
- 20 tests in `tests/unit/test_evaluate_momentum.py`
- Requires checkpoint data from pipeline steps 1-3 to run evaluation

All momentum features have deep history (1950+), so they are safe for
`clustering_features` without dropping pre-1993 rows.
