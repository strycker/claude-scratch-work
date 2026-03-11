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

```bash
# Full run from scratch (slow — scrapes ~46 URLs + FRED API)
python run_pipeline.py --refresh --recompute --plots --market-code grok --save-market-code

# Fast run using cached checkpoints (no re-scraping)
python run_pipeline.py --steps 3,4,5,6,7 --plots --market-code grok

# Individual pipeline steps
python pipelines/01_ingest.py
python pipelines/02_features.py
python pipelines/03_cluster.py
python pipelines/04_regime_label.py
python pipelines/05_predict.py
python pipelines/06_asset_returns.py
python pipelines/07_dashboard.py

# Launch the exploration notebooks
jupyter lab notebooks/
```

### Using New Cluster Labels as a Pipeline Starting Point

After experimenting with a different clustering method (GMM, DBSCAN, Spectral, or a
different `balanced_k`), you can save the new cluster labels as a named checkpoint
and use them as the starting point for all downstream steps instead of the Grok seed labels.

```bash
# 1. Run clustering and save the balanced_cluster column as a named checkpoint
python run_pipeline.py --steps 3 --save-market-code
#    This saves 'balanced_cluster' to data/checkpoints/market_code_clustered.parquet

# 2. Run downstream steps (regime labeling → supervised → assets → dashboard)
#    using the newly-saved clustered labels as the market_code source
python run_pipeline.py --steps 4,5,6,7 --market-code clustered --plots

# 3. Optional: if you've manually pinned regime names in config/regime_labels.yaml
#    after inspecting the new clusters, reload from there:
python run_pipeline.py --steps 4,5,6,7 --market-code clustered --plots
#    (regime_labels.yaml is loaded automatically in step 4)

# To compare multiple clustering runs side-by-side, save each to a custom checkpoint:
python -c "
from market_regime.io.checkpoints import CheckpointManager
import pandas as pd
cm = CheckpointManager()
# List available checkpoints
print(cm.list())
"

# To start a fresh end-to-end run with the new labels (no re-scraping):
python run_pipeline.py --recompute --steps 3,4,5,6,7 --save-market-code --plots
```

**Workflow summary:**
1. Re-cluster (`--steps 3 --save-market-code`)
2. Inspect `notebooks/03_clustering.ipynb` and `notebooks/07_pairplot.ipynb`
3. Edit `config/regime_labels.yaml` to pin human-readable names to the new cluster IDs
4. Re-run downstream steps (`--steps 4,5,6,7 --market-code clustered`)
5. Inspect `notebooks/04_regimes.ipynb` through `notebooks/06_assets.ipynb`

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
