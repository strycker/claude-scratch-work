# Trading-Crab — Market Regime Pipeline

![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-769%20passing-brightgreen)
![CI](https://img.shields.io/badge/CI-GitHub%20Actions-blue)

Market regime classification and prediction pipeline. Predict market conditions, best portfolios, and stock picks.

## Overview

Predict market conditions, optimal portfolios, and stock picks by:

1. **Data Ingestion** — Scrapes macro financial data from multpl.com and the FRED API (quarterly resolution, ~1950–present).
2. **Feature Engineering** — Log transforms, smoothed derivatives (1st–3rd order), cross-asset ratios, Bernstein-polynomial gap filling.
3. **Clustering** — PCA dimensionality reduction, KMeans + size-constrained KMeans to label each quarter with a market regime. Investigation suite compares GMM, DBSCAN, Spectral, and gap-statistic optimal-k selection.
4. **Regime Interpretation** — Statistical profiling of each cluster to assign human-readable names (e.g. "Stagflation", "Growth Boom").
5. **Supervised Prediction** — Classifiers to predict today's regime from currently-available features (no look-ahead).
6. **Transition Probabilities** — Empirical regime transition matrices and forward-looking probability models.
7. **Asset Returns by Regime** — Per-regime median returns for major asset classes.
8. **Portfolio Construction** — Regime-conditional portfolio recommendations.

## Concepts / Main Approach Outline:
- Scrape public datasets and use free APIs to obtain macro financial data over a 50-year period, ensuring these metrics are still available today if I had to score a model now
- Assumption: one of the most predictive features in any financial model will be the market conditions... are we in a recession?  A market boom?  A bubble?  A slowly forming top?  High/Low inflation?  Stagflation?  Therefore we want to CLASSIFY (apply unsupervised learning) to our time series datasets on the order of quarters.  Idea would be to get roughly equally-sized clusters that have distinct behaviors
- Once we have the time-series classified according to variance techniques, we want to PREDICT today's classification using data available to us TODAY.  This means we want to construct a SUPERVISED learning model that, given features known only at that time &mdash; nothing forward-looking or revised &mdash; we have a notion of what market condition regime we are in
- Even more powerfully, we can also construct supervised learning models to predict whether certain classifications will occur in the next quarter, next year, next 2 years, etc.  For example, if we are in a boom period, what are the chances that we'll experience a recession in the next 2 years?
- Once we have good predictions for market conditions and some rough models for predicting future conditions, we can then try to predict the value of various asset classes (or ETFs), either each relative to cash (USD) or relative to each other (e.g. S&P500 priced in $Gold, or TLT bonds priced in USO oil prices).  This will give us an idea of what assets do best in each PREDICTED market regime (that is, you should be able to rank the assets according to which out-perform or under-perform the others, including cash).  We can use these relative performance models to construct rough portfolio mixes.
- Putting it all together (Part I):  modeling individual asset performance
  - Using predicted market current market conditions, future market conditions, and all historic data and derived data (e.g., smoothed first derivative of oil prices measured in gold, etc.), predict the likelihood of whether a given ETF will be +X% at Y quarters in the future.
  - For example, we might be interested in the likelihood that the S&P will at some point in the next 2 years crash 20%, or separately, be 20% higher.
  - Note that these models are somewhat independent, particularly in volatile markets.  Models need not sum up to 100% &mdash; you could simultaneously predict that the S&P500 will crash with 80% probability AND with 80% probability rebound to +20% (actually you won't know the order... it might have a blow-off top and THEN crash).
  - Use these models to build a "stoplight" dashboard... for every asset, what are the probabilities of the asset going up or down as measured in dollars (or relative to another asset)
- Putting it all together (Part II):  Final project conclusion = actual trading recommendations
  - Given a portfolio of X assets at Y percentages, the market condition regime, the recommended portfolio mix, the projected performance of each asset (which indicators have recently turned on warning lights), should you buy, sell, or hold that asset?
  - Send a weekly email (can use AI for this part!) with the final recommendations on portfolio changes &mdash; what assets need traded, bought, or sold THIS WEEK?

## Two Packages

This monorepo contains two independent Python packages:

| Package | pip name | What it provides |
|---|---|---|
| `src/trading_crab_lib/` | **`trading-crab-lib`** | Library: transforms, clustering, prediction, reporting, plotting, ingestion |
| `src/trading_crab/` | **`trading-crab`** | Application: CLI (`tradingcrab`) + pipeline orchestration |

`trading-crab` depends on `trading-crab-lib`. The library can be used standalone.

## Installation

### Prerequisites

- **Python 3.10+** — check with `python3 --version`
- **Git** — to clone the repo
- **FRED API key** — free at [fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html)

### Quick Start (automated)

```bash
# 1. Clone the repo
git clone <repo-url>
cd claude-scratch-work

# 2. Run the setup script (creates .venv, installs deps, scaffolds directories)
bash scripts/setup.sh

# For testing + JupyterLab support:
bash scripts/setup.sh --dev

# 3. Activate the virtual environment
source .venv/bin/activate

# 4. Add your FRED API key
#    Edit .env and replace the placeholder:
#    FRED_API_KEY=your_key_here

# 5. Run the pipeline
tradingcrab --refresh --recompute --plots --market-code grok --save-market-code
```

### Manual Installation

```bash
# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Install both packages (editable mode, all extras)
pip install -e "src/trading_crab_lib/[all,dev]"
pip install -e ".[dev]"

# Optional but recommended — enables balanced-size clustering
pip install k-means-constrained

# Set up .env
cp .env.example .env
# Edit .env and set: FRED_API_KEY=your_key_here
```

### Library-Only Installation

If you only need the library (no CLI, no pipeline scripts):

```bash
pip install trading-crab-lib                # core only (transforms, clustering, prediction)
pip install "trading-crab-lib[ingestion]"   # + FRED/multpl/yfinance fetchers
pip install "trading-crab-lib[plotting]"    # + matplotlib/seaborn
pip install "trading-crab-lib[all]"         # everything
```

### Docker

If you prefer not to manage a Python environment locally, Docker provides a
fully self-contained setup:

```bash
# Build the image (first time, ~5 min)
docker build -t trading-crab .

# Run the weekly report (requires FRED_API_KEY and SMTP vars in .env)
docker compose run --rm weekly-report

# Run an arbitrary pipeline subset
docker compose run --rm pipeline --steps 3,4,5 --plots

# Launch JupyterLab at http://localhost:8888
docker compose up notebook
```

See [`Dockerfile`](Dockerfile) and [`docker-compose.yml`](docker-compose.yml)
for the full configuration.  All secrets are passed via environment variables
or a `.env` file — never baked into the image.

### Common Commands (via Makefile)

```bash
make setup          # Automated setup (runs scripts/setup.sh)
make setup-dev      # Setup with testing + notebook extras
make run            # Steps 3-7 from cached checkpoints (fast)
make run-full       # Full pipeline — re-scrape + recompute + plots
make test           # Run the test suite
make dashboard      # Print current regime dashboard
make notebooks      # Launch JupyterLab
make help           # Show all available targets
```

### Running Tests

```bash
# Core test run (requires only core deps; ~46 tests skipped for optional deps)
pytest tests/ -v

# Full test run — zero skips, zero warnings (requires requirements-dev.txt)
pip install -r requirements-dev.txt
pytest tests/ -v

# Run a specific module's tests
pytest tests/unit/test_transforms.py -v

# Run with coverage report
pytest tests/ --cov=src/trading_crab_lib --cov-report=term-missing
```

**Optional dependencies and which tests they unlock:**

| Package | Tests | Install |
|---------|-------|---------|
| `hmmlearn>=0.3` | `test_hmm.py` (~19 tests) | `pip install hmmlearn` |
| `statsmodels>=0.14` | `test_markov.py` (~18 tests) | `pip install statsmodels` |
| `hdbscan>=0.8` | `test_density.py` hdbscan block | `pip install hdbscan` |
| `lightgbm>=4.0` | `test_lightgbm.py` (~5 tests) | `pip install lightgbm` |
| `lxml>=4.9` + `cssselect>=1.2` | `test_ingestion.py` multpl scraper | `pip install lxml cssselect` |
| `kneed>=0.8` | `test_clustering_exploration.py` knee detection | `pip install kneed` |

All warnings from `statsmodels` optimization on synthetic data are suppressed via
`[tool.pytest.ini_options] filterwarnings` in `pyproject.toml` — they are harmless
numerical artefacts from short synthetic series and do not affect correctness.

### Running the Pipeline

#### All CLI Flags

| Flag | Description |
|------|-------------|
| `--refresh` | Re-scrape multpl.com + re-hit FRED API (~10 min). Without this flag, steps 1-2 load from cached checkpoints if less than 7 days old. |
| `--recompute` | Recompute derived features (step 2) from cached raw data. Use after editing `settings.yaml` or `transforms.py` without wanting to re-scrape. |
| `--refresh-assets` | Re-fetch ETF prices via yfinance (step 6 only). Without this flag, step 6 reuses `data/raw/asset_prices.parquet` if it exists. |
| `--plots` | Generate and save matplotlib figures to `outputs/plots/`. |
| `--show-plots` | Also call `plt.show()` after each figure. Off by default; do **not** use in CI or headless environments. |
| `--verbose` | Set logging to DEBUG. |
| `--steps 1,3,5` | Run only the listed step numbers (comma-separated). Valid: `1 2 3 4 5 6 7`. |
| `--no-constrained` | Skip the `k-means-constrained` package even if installed. Falls back to plain KMeans. |
| `--no-drop-tail` | Include the most-recent (potentially incomplete) quarter. By default the trailing row is dropped when it contains NaN in any feature column. |
| `--market-code NAME` | Inject a market_code label column. `NAME` = `grok` \| `clustered` \| `predicted` \| any checkpoint name. Omit for a fully data-driven run. |
| `--save-market-code` | After step 3, save `balanced_cluster` labels as the `market_code_clustered` checkpoint for use with `--market-code clustered`. |

> **Auto-saved checkpoint:** Step 5 automatically saves predicted current-regime labels as
> `market_code_predicted` every time it runs. No flag needed — use with `--market-code predicted`.

#### Common Workflows

All commands below use `tradingcrab` (installed CLI). You can substitute
`python run_pipeline.py` if you prefer the backward-compatible entry point.

```bash
# ① FRESH START — scrape everything, seed with Grok labels (recommended first run)
tradingcrab --refresh --recompute --plots \
    --market-code grok --save-market-code

# ② FULLY DATA-DRIVEN — no label seed, cluster purely from data
tradingcrab --refresh --recompute --plots --save-market-code

# ③ FAST RE-RUN — skip scraping, use cached checkpoints, regenerate plots
tradingcrab --steps 3,4,5,6,7 --plots

# ④ RE-CLUSTER ONLY — update cluster assignments and save for downstream
tradingcrab --steps 3 --save-market-code --plots

# ⑤ DOWNSTREAM WITH NEW CLUSTER LABELS — use labels saved in ④
tradingcrab --steps 4,5,6,7 --market-code clustered --plots

# ⑥ DOWNSTREAM WITH GROK SEED — overlay original AI labels
tradingcrab --steps 4,5,6,7 --market-code grok --plots

# ⑦ DOWNSTREAM WITH PREDICTED LABELS — use last step-5 predictions as seed
tradingcrab --steps 4,5,6,7 --market-code predicted --plots

# ⑧ RECOMPUTE FEATURES WITHOUT RE-SCRAPING (e.g. after editing settings.yaml)
tradingcrab --recompute --steps 2,3,4,5,6,7 --plots

# ⑨ ETF DATA REFRESH ONLY (no macro re-scrape)
tradingcrab --steps 6,7 --refresh-assets --plots

# ⑩ DEBUG A SINGLE STEP
tradingcrab --steps 3 --verbose --plots --show-plots
```

#### Individual Step Scripts

```bash
python pipelines/01_ingest.py
python pipelines/02_features.py
python pipelines/03_cluster.py
python pipelines/04_regime_label.py
python pipelines/05_predict.py
python pipelines/06_asset_returns.py
python pipelines/07_dashboard.py
```

```bash
# Launch the exploration notebooks
jupyter lab notebooks/
```

### Feature Artifacts & Contracts

Step 2 produces two feature variants from the `macro_raw` checkpoint:

- **Centered features (non-causal)**: written to `data/processed/features.parquet` and checkpointed as `features` / `features_noncausal`. These use centered smoothing windows and are intended for unsupervised clustering and regime profiling (steps 3–4).
- **Causal features**: written to `data/processed/features_supervised.parquet` and checkpointed as `features_supervised` / `features_causal`. These use backward-only smoothing windows so no future information leaks into derivatives; they are used for supervised models and live scoring (steps 5–7).

To (re)generate both artifacts from the latest `macro_raw` checkpoint:

```bash
python pipelines/02_features.py
```

Downstream code and notebooks can load the non-causal or causal variants unambiguously via the corresponding parquet paths or `CheckpointManager` names.

### Market Code — Label Seeding Workflows

The `market_code` is a per-quarter integer label (0–4) that serves as the reference
regime assignment, attached to `macro_raw` in step 1 and propagated through all
downstream steps as an overlay/reference column.

| Source (`--market-code NAME`) | Description |
|-------------------------------|-------------|
| `grok` | Original AI-assisted labels (stable reference, never changes). Loaded from `data/grok_*.pickle`; cached automatically on first use. |
| `clustered` | Labels from the most recent `--save-market-code` run. Updated every time you run step 3 with `--save-market-code`. |
| `predicted` | Labels from the most recent step 5 run. Reflects the trained classifier's best guess for historical quarters. Saved automatically. |
| *(omitted)* | Fully data-driven run — no market_code column is injected. |
| *`<custom>`* | Load checkpoint `market_code_<custom>` — any name you previously saved. |

**Typical label-seeding workflow:**
1. **First run** — establish a stable baseline from Grok labels:
   ```bash
   python run_pipeline.py --refresh --recompute --plots \
       --market-code grok --save-market-code
   ```
2. **Re-cluster** — explore a different `balanced_k` or clustering algorithm in the
   notebook, then persist the preferred assignments:
   ```bash
   python run_pipeline.py --steps 3 --save-market-code --plots
   ```
3. **Pin regime names** — inspect `notebooks/03_clustering.ipynb`, then edit
   `config/regime_labels.yaml` to assign human-readable names to each cluster ID.
4. **Re-run downstream** with the new labels:
   ```bash
   python run_pipeline.py --steps 4,5,6,7 --market-code clustered --plots
   ```
5. **Use predicted labels** for subsequent runs once the classifier is trained:
   ```bash
   python run_pipeline.py --steps 4,5,6,7 --market-code predicted --plots
   ```

To list all available `market_code` checkpoints:
```bash
python -c "
from trading_crab_lib.checkpoints import CheckpointManager
cm = CheckpointManager()
mc = [e for e in cm.list() if e['name'].startswith('market_code_')]
for e in mc:
    print(e['name'], '—', e.get('rows', '?'), 'rows')
"
```

### Clustering Investigation Extras

`notebooks/03_clustering.ipynb` contains a full investigation suite — gap statistic,
GMM, DBSCAN/HDBSCAN, Spectral, SVD, and multi-method comparison. Most of it works
with just the core dependencies, but two optional extras unlock additional features:

```bash
# Automated elbow/knee detection for KMeans inertia curve
pip install kneed

# Hierarchical DBSCAN — more robust than DBSCAN for varying cluster densities
pip install hdbscan

# Or install both at once via the pyproject.toml extra:
pip install -e ".[clustering-extras]"
```

### Alternative ETF Data Sources

When yfinance is unavailable, the pipeline falls back automatically through:
1. **yfinance** (primary) — standard ETF OHLCV data
2. **stooq** via `pandas-datareader` — free, daily data, no API key needed
3. **OpenBB** — multi-provider fallback (cboe free; others need API keys)
4. **Macro proxy** — synthetic returns computed from macro_raw.parquet

Install the data extras to enable stooq + OpenBB phases:
```bash
pip install -e ".[data-extras]"
# or individually:
pip install pandas-datareader openbb
```

### Dependency Notes

Dependencies are managed via `pyproject.toml` optional extras (not `requirements.txt`).

| Extra group | Packages | Purpose |
|---|---|---|
| *(core)* | pandas, numpy, scikit-learn, scipy, pyarrow | Always installed with `trading-crab-lib` |
| `[ingestion]` | fredapi, lxml, yfinance, requests | Data fetching from FRED, multpl, yfinance |
| `[plotting]` | matplotlib, seaborn | Visualization |
| `[hmm]` | hmmlearn, statsmodels | Hidden Markov Model + Markov switching |
| `[clustering-extras]` | hdbscan, kneed | HDBSCAN + elbow detection |
| `[boosting]` | lightgbm | LightGBM classifier |
| `[all]` | All of the above | Everything |
| `[dev]` | pytest, flake8, pytest-cov | Testing and linting |

Additional optional packages (not in extras):

| Package | Purpose |
|---|---|
| `k-means-constrained` | Balanced-size clustering; falls back to plain KMeans if absent |
| `pandas-datareader` | Stooq ETF fallback |
| `openbb` | Multi-provider ETF fallback |

Legacy `requirements.txt` / `requirements-dev.txt` files are still present for
backward compatibility but `pyproject.toml` extras are the preferred install method.

---

## Reference Submodules

This repo contains two Git submodules that are **read-only references**. Never modify
files inside them — use them only to compare implementations and inform changes to the
main repo.

| Submodule | Purpose |
|-----------|---------|
| `gsd-scratch-work/` | GSD framework version of the project (earlier development checkpoint) |
| `trading-crab-lib/` | Separate trading-crab library repo |

To initialize submodules after cloning:
```bash
git submodule update --init --recursive
```

---

## Project Documentation

| File | Contents |
|------|----------|
| `CLAUDE.md` | Code conventions, design rules, ADRs, pitfalls, session instructions for AI |
| `ROADMAP.md` | Prioritized feature backlog with effort estimates |
| `STATE.md` | Current implementation status and test coverage |
| `LESSONS_LEARNED.md` | Pitfalls discovered during development and what we'd do differently |
| `DISTRIBUTION.md` | How to distribute/deploy: PyPI, Docker, Maven/npm context |
| `REBUILD-FROM-SCRATCH-GUIDE.md` | Step-by-step guide to rebuilding the project from zero |

---

## To Do

See **`ROADMAP.md`** for the full prioritized backlog with effort estimates.
Short summary:

### Next Up (Tier 1 — remaining)
- [ ] Add FRED series: INDPRO (industrial production), PAYEMS (nonfarm payrolls), DPCERA3Q086SBEA (real PCE)
- [ ] macrotrends.net scraper for gold (1915+) and oil (1946+) price backfill
- [ ] LightGBM integration into flat production API (module exists, needs wiring)

### Medium Term (Tier 2)
- [ ] Per-asset probability models — "will ETF be +X% at Y quarters?" (Part I vision)
- [ ] Conference Board LEI proxy from FRED components
- [ ] Finviz Elite sector/stock signals for within-regime stock picking
- [ ] Docker image for reproducible weekly runs (`Dockerfile` + `docker-compose.yml`)

### Long Term (Tier 3)
- [ ] Backtest framework — walk-forward validation of full strategy
- [ ] Weekly automated report with AI-written narrative via Claude API
- [ ] Streamlit interactive dashboard
- [ ] Factor model for asset returns within regimes (LASSO/Ridge per regime)

### Completed ✓

- ✓ Full 9-step pipeline runs end-to-end on real data
- ✓ Data ingestion: multpl.com (46 series), FRED (14 series), yfinance (38 ETFs)
- ✓ Feature engineering: log transforms, Bernstein gap fill, smoothed derivatives
- ✓ Yield curve features: 10Y-2Y and 10Y-3M spreads
- ✓ Causal + centered smoothing — two separate feature files prevent look-ahead bias
- ✓ PCA + KMeans clustering (standard + size-constrained)
- ✓ Clustering investigation suite — gap statistic, GMM, DBSCAN/HDBSCAN, Spectral, SVD, RF feature selection
- ✓ Regime profiling, naming heuristics, transition matrix, forward probabilities
- ✓ RandomForest + DecisionTree + GradientBoosting with TimeSeriesSplit 5-fold CV
- ✓ Forward binary classifiers for each (horizon, regime) pair
- ✓ Interpretability tree (shallow DT on top-k features for human-readable rules)
- ✓ Asset returns by regime (yfinance ETFs + macro proxy fallback)
- ✓ Portfolio construction: simple + blended weights + BUY/SELL/HOLD recommendations
- ✓ Text + CSV dashboard with GREEN/YELLOW/RED asset signals
- ✓ Diagnostics: RRG analysis (relative rotation graphs), tactical classification
- ✓ Weekly email report pipeline (`--weekly-report` + `--send-email`)
- ✓ CheckpointManager (parquet + manifest, joblib for models, corrupt metadata logging)
- ✓ Ingestion completeness report (validates column count + NaN coverage)
- ✓ Full CLI (`run_pipeline.py --steps --refresh --recompute --plots …`)
- ✓ Confusion matrix visualization in `plotting.py`
- ✓ Package renamed: `market_regime` → `trading_crab_lib` (pip name: `trading-crab-lib`)
- ✓ yfinance fallback chain: stooq → OpenBB → macro proxy
- ✓ Momentum features: trailing returns, S&P-in-Gold/Oil, rolling cross-asset correlation, CPI acceleration
- ✓ Cross-asset divergence features: SPY/TLT, SPY/GLD, GLD/Oil, CreditSpread/VIX pairs (z-scores + triggers)
- ✓ Hidden Markov Model regime detection (`hmm.py` + `markov.py`)
- ✓ ~635 tests (unit + integration), all passing
- ✓ Exploration notebooks (01–12)

---

## Audit Discoveries (2026-03-17)

Full codebase audit comparing documentation, disk state, and code quality.

### Documentation vs. Disk Alignment

**98%+ match** — nearly everything in `CLAUDE.md`'s layout tree exists on disk in the correct location. Specific findings:

| Finding | Severity | Notes |
|---------|----------|-------|
| `data/raw/`, `data/processed/`, `data/regimes/` referenced in `.gitignore` and CLAUDE.md layout but don't exist on disk | Low | All data lives under `data/checkpoints/` instead. Dirs would be created on-demand if pipeline wrote there, but current code doesn't. |
| `outputs/models/` and `outputs/reports/` don't exist yet | Low | Referenced in Makefile `clean-*` targets. Only `outputs/plots/` (26 PNGs) exists. Created on-demand when steps 5/7 save artifacts. |
| `src/trading_crab_lib/ingestion/macrotrends.py` documented in ROADMAP Tier 1 but not created | Expected | Planned feature, not a gap. |
| Undocumented data snapshots on disk | Info | `prepared_quarterly_data_smoothed_20260301.pickle`, `standardized_quarterly_data_20260216.pickle`, `grok_quarter_classifications_20260201.xlsx` — legitimate runtime artifacts, correctly gitignored. |
| Checkpoint aliases not enumerated in docs | Info | `features_causal.parquet` = `features_supervised`, `features_noncausal.parquet` = `features` — consistent with ADR #1, just not listed in the layout tree. |

### Runtime State

- **22 checkpoint files** (11 parquet datasets + 11 `.meta.json` manifests) in `data/checkpoints/`
- **26 PNG plots** in `outputs/plots/` covering all 8 pipeline stages
- **238 tests collected**, 8 skipped (HDBSCAN optional dependency)
- All 7 pipeline steps produce expected outputs end-to-end

### Code Quality Observations

1. **Prediction package dual-API is clean** — `prediction/__init__.py` (flat/production) and `prediction/classifier.py` (bundle/test) are correctly separated per ADR #12. No cross-contamination.
2. **Checkpoint system works as designed** — parquet + manifest pairs, freshness checks, config hash tracking.
3. **No stale or orphaned code detected** — the `pipelines_from_gsd_version/` directory was already cleaned up (D6).
4. **SSL workaround in `assets.py`** is a known tech debt item (P22) but functional.

---

## Prioritized Next Steps

See **`ROADMAP.md`** for full details and effort estimates. Quick summary of what's next:

| # | Item | Effort | Notes |
|---|------|--------|-------|
| 1 | Add FRED: INDPRO, PAYEMS, DPCERA3Q086SBEA | S | Config-only |
| 2 | macrotrends.net live verification | S | Code exists, needs real-data test |
| 3 | LightGBM flat-API wiring | M | `gradient_boosting.py` exists |
| 4 | Momentum & cross-asset ratio features | M | 6M/12M momentum, relative strength |
| 5 | Cross-asset divergence features | S | Phases A+B done; C+D remaining (ROADMAP 2.15) |
| 6 | HMM regime detection | M | Temporal alternative to KMeans |
| 7 | Backtest framework | XL | Walk-forward strategy validation |

### Already completed (former priority items)
- ✅ P12 fix (`end_date: null`) — done
- ✅ FRED expansion (7 → 14 series) — done
- ✅ Yield curve features — done
- ✅ Confusion matrix plot — done
- ✅ Pickle → joblib migration (P27) — done
- ✅ Ingestion completeness report (P23) — done
- ✅ CheckpointManager corrupt metadata logging (P24) — done
- ✅ Package rename `market_regime` → `trading_crab_lib` — done
