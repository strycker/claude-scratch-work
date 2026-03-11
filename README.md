# claude-scratch-work
Repository for playing around with Claude Code

First project:  market regime classification and prediction pipeline.  Predict market conditions, best portfolios, and stock picks.

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
python run_pipeline.py --refresh --recompute --plots --market-code grok --save-market-code
```

### Manual Installation

If you prefer to manage your own environment:

```bash
# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Install runtime dependencies (pinned)
pip install -r requirements.txt

# OR install dev extras (adds pytest + JupyterLab)
pip install -r requirements-dev.txt

# Optional but recommended — enables balanced-size clustering
pip install k-means-constrained

# Set up .env
cp .env.example .env
# Edit .env and set: FRED_API_KEY=your_key_here

# Create runtime directories
mkdir -p data/{raw,processed,regimes,checkpoints}
mkdir -p outputs/{plots,models,reports}
```

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

```bash
# ① FRESH START — scrape everything, seed with Grok labels (recommended first run)
python run_pipeline.py --refresh --recompute --plots \
    --market-code grok --save-market-code

# ② FULLY DATA-DRIVEN — no label seed, cluster purely from data
python run_pipeline.py --refresh --recompute --plots --save-market-code

# ③ FAST RE-RUN — skip scraping, use cached checkpoints, regenerate plots
python run_pipeline.py --steps 3,4,5,6,7 --plots

# ④ RE-CLUSTER ONLY — update cluster assignments and save for downstream
python run_pipeline.py --steps 3 --save-market-code --plots

# ⑤ DOWNSTREAM WITH NEW CLUSTER LABELS — use labels saved in ④
python run_pipeline.py --steps 4,5,6,7 --market-code clustered --plots

# ⑥ DOWNSTREAM WITH GROK SEED — overlay original AI labels
python run_pipeline.py --steps 4,5,6,7 --market-code grok --plots

# ⑦ DOWNSTREAM WITH PREDICTED LABELS — use last step-5 predictions as seed
python run_pipeline.py --steps 4,5,6,7 --market-code predicted --plots

# ⑧ RECOMPUTE FEATURES WITHOUT RE-SCRAPING (e.g. after editing settings.yaml)
python run_pipeline.py --recompute --steps 2,3,4,5,6,7 --plots

# ⑨ ETF DATA REFRESH ONLY (no macro re-scrape)
python run_pipeline.py --steps 6,7 --refresh-assets --plots

# ⑩ DEBUG A SINGLE STEP
python run_pipeline.py --steps 3 --verbose --plots --show-plots
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
from market_regime.io.checkpoints import CheckpointManager
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

| Package | Required? | Purpose |
|---|---|---|
| All in `requirements.txt` | Yes | Core pipeline |
| `k-means-constrained` | Recommended | Balanced-size clustering; falls back to plain KMeans if absent |
| `kneed` | Optional | Automated elbow detection for KMeans (via `[clustering-extras]`) |
| `hdbscan` | Optional | Hierarchical DBSCAN (via `[clustering-extras]`) |
| `pandas-datareader` | Optional | Stooq ETF fallback (via `[data-extras]`) |
| `openbb` | Optional | Multi-provider ETF fallback (via `[data-extras]`) |
| `requirements-dev.txt` extras | Dev only | pytest, JupyterLab, IPython kernel |

To upgrade all pinned dependencies to their latest compatible versions:

```bash
pip install pip-tools
pip-compile pyproject.toml --upgrade --output-file requirements.txt
pip-compile pyproject.toml --extra dev --upgrade --output-file requirements-dev.txt
```

---

## Project Documentation

| File | Contents |
|------|----------|
| `CLAUDE.md` | Code conventions, design rules, session instructions for AI |
| `ROADMAP.md` | Prioritized feature backlog with effort estimates |
| `ARCHITECTURE.md` | Why key design decisions were made (ADR format) |
| `PITFALLS.md` | Known gotchas, anti-patterns, and things not to break |
| `STATE.md` | Current implementation status and test coverage |

---

## To Do

See **`ROADMAP.md`** for the full prioritized backlog with effort estimates.
Short summary:

### Next Up (Tier 1)
- [ ] Add FRED series: VIX, unemployment, M2, yield spreads (10Y-2Y, 10Y-3M), housing starts
- [ ] Add yield curve derived features in `transforms.py`
- [ ] macrotrends.net scraper for gold (1915+) and oil (1946+) price backfill
- [ ] LightGBM classifier alongside RandomForest + Decision Tree
- [ ] Empirical forward probabilities in `profiler.py` (small remaining legacy gap)
- [ ] Confusion matrix visualization in `plotting.py`
- [ ] `end_date: null` → use today in `settings.yaml`
- [ ] Expand test suite (classifier, portfolio, dashboard, profiler)

### Medium Term (Tier 2)
- [ ] Hidden Markov Model regime detection (`clustering/hmm.py`)
- [ ] Per-asset probability models — "will ETF be +X% at Y quarters?" (Part I vision)
- [ ] Momentum + cross-asset ratio features (6M/12M momentum, gold-in-oil, etc.)
- [ ] Finviz Elite sector/stock signals for within-regime stock picking

### Long Term (Tier 3)
- [ ] Individual asset-return predictors per regime (binary classifiers per ETF)
- [ ] Regime-conditional portfolio optimization (mean-variance, risk-parity)
- [ ] Weekly automated report with AI-written narrative via Claude API
- [ ] Streamlit interactive dashboard

### Completed ✓

- ✓ Full 7-step pipeline runs end-to-end on real data
- ✓ Data ingestion: multpl.com (46 series), FRED (7 series), yfinance (16 ETFs)
- ✓ Feature engineering: log transforms, Bernstein gap fill, smoothed derivatives
- ✓ Causal + centered smoothing — two separate feature files prevent look-ahead bias
- ✓ PCA + KMeans clustering (standard + size-constrained)
- ✓ Regime profiling, naming heuristics, transition matrix
- ✓ RandomForest + DecisionTree with TimeSeriesSplit 5-fold walk-forward CV
- ✓ Forward binary classifiers for each (horizon, regime) pair
- ✓ Asset returns by regime (yfinance ETFs + macro proxy fallback)
- ✓ Portfolio construction: simple + blended weights + BUY/SELL/HOLD recommendations
- ✓ Text + CSV dashboard with GREEN/YELLOW/RED asset signals
- ✓ CheckpointManager (parquet + manifest; avoids re-scraping)
- ✓ Full CLI (`run_pipeline.py --steps --refresh --recompute --plots …`)
- ✓ Exploration notebooks (01–08)
- ✓ Installation setup (`requirements.txt`, `setup.sh`, `Makefile`)
- ✓ Python 3.10+ compatibility; SSL fix for yfinance curl_cffi
- ✓ **Clustering investigation suite** — gap statistic, GMM, DBSCAN/HDBSCAN, Spectral,
  SVD vs PCA, PCA component sweep, multi-method comparison + ARI heatmap, RF feature selection
- ✓ **yfinance fallback chain** — stooq → OpenBB → macro proxy
- ✓ **213 unit tests** covering all core modules and new clustering investigation suite
