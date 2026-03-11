# CLAUDE.md — Project Guide for Claude Code

This file is read automatically by Claude Code at the start of every session.
It explains what this project is, how to work in it, and what conventions to follow.

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

The reference implementation lives in `legacy/`.  Two layers of reference exist:

- `legacy/unified_script.py` — the original 1249-line monolith; ground truth for
  every algorithm, formula, and parameter choice.
- `legacy/*.py` modular scripts — a refactored version of the monolith organized
  into: `config.py`, `data_ingestion.py`, `feature_engineering.py`, `clustering.py`,
  `regime_analysis.py`, `supervised.py`, `asset_returns.py`, `portfolio.py`,
  `plotting.py`, `pipeline.py`.  These are used as the design reference for the
  `src/market_regime/` package.  **Do not modify legacy files.**

The modular pipeline in `src/` and `pipelines/` should do everything
that script does, organized more cleanly, with checkpointing, CLI flags, and
dedicated plotting notebooks.

---

## Repository Layout

```
trading-crab/
├── CLAUDE.md                      ← you are here
├── README.md                      ← project overview (user-facing)
├── scratch/README.md              ← extended design notes
├── .env.example                   ← copy to .env, fill in FRED_API_KEY
├── pyproject.toml                 ← pip-installable package (src layout)
│
├── config/
│   ├── settings.yaml              ← ALL tuneable parameters live here
│   └── regime_labels.yaml         ← manually-pinned regime names (edit after clustering)
│
├── data/                          ← gitignored; created at runtime
│   ├── raw/                       ← macro_raw.parquet, asset_prices.parquet
│   ├── processed/                 ← features.parquet (after step 02)
│   ├── regimes/                   ← cluster_labels.parquet, profiles.parquet, …
│   └── checkpoints/               ← timestamped pickle checkpoints (see CheckpointManager)
│
├── legacy/                        ← reference implementation; do not modify
│   ├── unified_script.py          ← THE reference — all logic must be reachable here
│   └── step{1-5}_*.ipynb          ← original Jupyter notebooks
│
├── notebooks/                     ← plotting/exploration notebooks (one per pipeline stage)
│   ├── 01_ingestion.ipynb
│   ├── 02_features.ipynb
│   ├── 03_clustering.ipynb
│   ├── 04_regimes.ipynb
│   ├── 05_prediction.ipynb
│   └── 06_assets.ipynb
│
├── pipelines/                     ← runnable pipeline steps
│   ├── 01_ingest.py
│   ├── 02_features.py
│   ├── 03_cluster.py
│   ├── 04_regime_label.py
│   ├── 05_predict.py
│   ├── 06_asset_returns.py
│   └── 07_dashboard.py
│
├── run_pipeline.py                ← master entry point with --steps / --refresh / --plots
│
├── outputs/                       ← gitignored; created at runtime
│   ├── models/                    ← pickled sklearn models
│   ├── plots/                     ← saved figures (PNG/PDF)
│   └── reports/                   ← dashboard.csv, weekly summaries
│
└── src/market_regime/             ← installable Python package
    ├── __init__.py                ← defines ROOT, CONFIG_DIR, DATA_DIR, OUTPUT_DIR
    ├── config.py                  ← load(), setup_logging()
    ├── runtime.py                 ← RunConfig dataclass (verbose, plots, refresh flags)
    ├── io/
    │   └── checkpoints.py         ← CheckpointManager (save/load/is_fresh/clear)
    ├── ingestion/
    │   ├── multpl.py              ← lxml scraper for 46 multpl.com series
    │   ├── fred.py                ← FRED API fetcher with publication-lag shift
    │   └── assets.py              ← yfinance ETF price fetcher
    ├── features/
    │   └── transforms.py          ← ratios, log, select, gap-fill, derivatives, engineer_all
    ├── clustering.py              ← reduce_pca, evaluate_kmeans, pick_best_k, fit_clusters
    │                                 + optimize_n_components, compare_svd_pca,
    │                                 + compute_gap_statistic, find_knee_k
    ├── gmm.py                     ← fit_gmm (returns scaler), select_gmm_k, gmm_labels, gmm_probabilities
    ├── density.py                 ← knn_distances, fit_dbscan_sweep, fit_dbscan, fit_hdbscan_sweep, hdbscan_labels
    ├── spectral.py                ← fit_spectral_sweep (affinity cached), spectral_labels
    ├── cluster_comparison.py      ← compare_all_methods, pairwise_rand_index,
    │                                 extract_rf_feature_importances, recommend_clustering_features
    ├── regime/
    │   └── profiler.py            ← build_profiles, suggest_names, build_transition_matrix
    ├── prediction/
    │   └── classifier.py          ← train_current_regime, train_forward_classifiers
    ├── assets/
    │   └── returns.py             ← compute_quarterly_returns, returns_by_regime, rank_assets_by_regime
    ├── reporting/
    │   └── dashboard.py           ← asset_signals, print_dashboard, save_dashboard_csv
    └── plotting.py                ← ALL visualization helpers (used by notebooks + pipelines)
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
python -c "from market_regime.config import load; print(load()['data'])"
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
All visualization helpers live in `src/market_regime/plotting.py`. Notebooks import
from there — they do not define plotting logic inline. Every plot function accepts
`run_cfg: RunConfig` and honours `save_plots` / `show_plots`. Output filenames are
standardized as `outputs/plots/{step}_{description}.png`.

### Custom color palette
Five-regime color palette from the legacy script:
```python
CUSTOM_COLORS = ["#0000d0", "#d00000", "#f48c06", "#8338ec", "#50a000"]
```
Use `plotting.REGIME_CMAP` everywhere for consistency.

---

## Data Sources

### multpl.com (46 series)
Scraped via lxml cssselect from `#datatable`. All URLs and `value_type` metadata
are in `config/settings.yaml` under `multpl.datasets`. Do not hardcode URLs in Python.
Rate-limited to 2 seconds per request (`RATE_LIMIT_SECONDS`).

### FRED API (7 series, more planned)
Current: GDP (shifted +1Q), GNP (shifted +1Q), BAA, AAA, CPI (CPIAUCSL), GS10, TB3MS.

Planned additions (see `ROADMAP.md` Tier 1):
- VIXCLS (VIX, 1990+), UNRATE (unemployment, 1948+), M2NS (money supply, 1959+)
- T10Y2Y (10Y-2Y spread), GS2 (2Y Treasury), HOUST (housing starts), UMCSENT

Requires `FRED_API_KEY` in `.env`. Free registration at fred.stlouisfed.org.

### macrotrends.net (planned — not yet implemented)
Gold spot price back to 1915, WTI crude oil back to 1946, silver, copper.
See `ROADMAP.md` Tier 1 item 1.5 and `src/market_regime/ingestion/macrotrends.py` (to be created).
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

---

## What the Legacy Code Does That Must Be Matched

Cross-reference `legacy/unified_script.py` and the `legacy/*.py` modules for
ground truth.  Items marked ✓ are verified as matching in `src/`.  Items marked
✗ are known gaps that still need to be implemented or aligned.

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

### Features in legacy NOT yet in src/ (see Legacy Alignment Gaps section)

- ✓ `DecisionTreeClassifier` training for interpretability — implemented in `classifier.py`
- ✓ `TimeSeriesSplit` cross-validation — implemented via `_tscv_scores()` helper
- ✓ Portfolio construction — `src/market_regime/reporting/portfolio.py` (new file)
- ✓ Macro-data fallback for asset returns — `compute_proxy_returns()` in `assets/returns.py`
- ✗ Empirical forward probabilities — `compute_forward_probabilities()` from `legacy/regime_analysis.py`
- ✗ Confusion matrix in classification report — `generate_classification_report()` from `legacy/supervised.py`

### Things src/ does better than legacy (do not regress)

- ✓ Real ETF price data via yfinance (SPY, GLD, TLT, USO, QQQ, IWM, VNQ, AGG)
  instead of macro-data proxies
- ✓ `CheckpointManager` with parquet + manifest (vs. ad-hoc pickle/CSV)
- ✓ `RunConfig` dataclass for clean flag management
- ✓ All config in `settings.yaml` (vs. hardcoded Python constants)
- ✓ Full CLI in `run_pipeline.py` with `--steps`, `--refresh`, `--recompute`, etc.
- ✓ Dedicated exploration notebooks (01–08)

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
files are produced.  See `ROADMAP.md` for prioritized feature backlog.
See `ARCHITECTURE.md` for design decisions.  See `PITFALLS.md` for known gotchas.

### Complete ✓
- Steps 01–07 run end-to-end; pipeline verified on real data (all 7 steps)
- `CheckpointManager` — fully implemented; parquet + manifest
- `RunConfig` — fully implemented, including `from_args()` factory
- `run_pipeline.py` — master runner with full CLI (all flags implemented)
- `ingestion/assets.py` — yfinance ETF price fetcher + 3-phase fallback chain (stooq → OpenBB → macro proxy)
- `plotting.py` — 17 visualization helpers covering all 7 pipeline steps
- `notebooks/01–08` — all notebooks present; 03_clustering expanded with 28 investigation cells
- Requirements — minimum-bound strategy, Python 3.10+ compatible
- `from __future__ import annotations` — present in all source files using `X | Y` syntax
- Unit tests — **213 passing tests** (8 skipped: HDBSCAN) covering all modules including clustering investigation suite
- **Gap 1** — `TimeSeriesSplit` CV in `classifier.py` (5-fold walk-forward)
- **Gap 2** — `DecisionTreeClassifier` in `classifier.py` (max_depth=8)
- **Gap 3** — `reporting/portfolio.py` — simple + blended portfolio + BUY/SELL/HOLD
- **Gap 4** — `compute_proxy_returns()` fallback in `assets/returns.py`
- **Causal smoothing** — `engineer_all(causal=True/False)` + dual parquet outputs from step 2
- **Clustering investigation suite** — GMM (`gmm.py`), DBSCAN/HDBSCAN (`density.py`), Spectral (`spectral.py`), multi-method comparison (`cluster_comparison.py`), gap statistic + SVD + PCA sweep in `clustering.py`
- **Config-driven var lists** — `plotting.sample_series`, `plotting.key_indicators`, `assets.etfs` all in `settings.yaml`
- **16 ETFs** — expanded from 8 to 16 (added HYG, XLK, XLP, XLE, GDX, TIP, BIL, EDV)
- **`pythonpath = ["src"]`** in `pyproject.toml` pytest config — tests run without `pip install -e .`

### Next Priority (implement in upcoming sessions)
1. **Additional FRED series** — VIX (VIXCLS), unemployment (UNRATE), M2 (M2NS),
   yield spreads (T10Y2Y, T10Y3M, GS2), housing starts (HOUST), consumer sentiment (UMCSENT)
2. **Yield curve derived features** — 10Y-2Y, 10Y-3M spread computed in `transforms.py`
3. **Empirical forward probabilities** — `compute_forward_probabilities()` from legacy
4. **Confusion matrix plot** — `plot_confusion_matrix()` in `plotting.py`
5. **macrotrends.net scraper** — gold/oil spot prices back to 1915/1946
6. **LightGBM classifier** — alongside RF + DT in `classifier.py`
7. **Expand test suite** — classifier, portfolio, dashboard, profiler
8. **`end_date: null`** in settings.yaml → use today's date at runtime
9. **Per-asset regime probability models** ("Putting it all together — Part I")
10. **Weekly automated report** with AI-written narrative via Claude API

### Known Limitations
- `profiler.py` naming heuristics silently skip 4 features (`10yr_ustreas`, `fred_gs10`,
  `fred_tb3ms`, `div_minus_baa`) because only their derivatives are in `clustering_features`.
  Graceful fallback is intentional.
- Only 7 FRED series currently ingested; many high-value series (VIX, unemployment,
  M2, yield curve) are not yet fetched.
- ETF data starts 1993-2006; pre-1993 gold and oil regime analysis uses proxy columns only.
  macrotrends.net backfill would extend coverage to 1915+ for gold.
- Clustering uses KMeans which treats each quarter independently; HMM would model
  temporal autocorrelation natively (Tier 2 roadmap item).

---

## Legacy Alignment Gaps

Full comparison of `legacy/*.py` vs `src/market_regime/` completed March 2026.

### Closed Gaps (all implemented)
- ✓ **Gap 1** — TimeSeriesSplit CV (`legacy/supervised.py` → `classifier.py`)
- ✓ **Gap 2** — DecisionTreeClassifier (`legacy/supervised.py` → `classifier.py`)
- ✓ **Gap 3** — Portfolio construction (`legacy/portfolio.py` → `reporting/portfolio.py`)
- ✓ **Gap 4** — Macro-data proxy returns fallback (`legacy/asset_returns.py` → `assets/returns.py`)
- ✓ **Gap 5** — Causal/backward rolling windows for supervised learning (`transforms.py`)

### Remaining Gaps

#### Empirical forward probabilities (`legacy/regime_analysis.py` → `regime/profiler.py`)
`compute_forward_probabilities()` computes count-based empirical P(reach regime j
within N quarters | currently in regime i).  Useful as a sanity check alongside the
model-based binary RF forward classifiers.  Low effort.  Status: not implemented.

#### Confusion matrix report (`legacy/supervised.py` → `plotting.py`)
`generate_classification_report()` prints a per-class confusion matrix.  Currently
we only log the in-sample `classification_report()` string but do not visualize
the confusion matrix.  Status: not implemented.

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
from market_regime.io.checkpoints import CheckpointManager
cm = CheckpointManager()
print(cm.list())
"

# Print current dashboard (requires steps 01-06 to have run)
python pipelines/07_dashboard.py
```
