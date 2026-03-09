# claude-scratch-work
Repository for playing around with Claude Code

First project:  market regime classification and prediction pipeline.  Predict market conditions, best portfolios, and stock picks.

## Overview

Predict market conditions, optimal portfolios, and stock picks by:

1. **Data Ingestion** — Scrapes macro financial data from multpl.com and the FRED API (quarterly resolution, ~1950–present).
2. **Feature Engineering** — Log transforms, smoothed derivatives (1st–3rd order), cross-asset ratios, Bernstein-polynomial gap filling.
3. **Clustering** — PCA dimensionality reduction, KMeans + size-constrained KMeans to label each quarter with a market regime.
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

### Dependency Notes

| Package | Required? | Purpose |
|---|---|---|
| All in `requirements.txt` | Yes | Core pipeline |
| `k-means-constrained` | Recommended | Balanced-size clustering; falls back to plain KMeans if absent |
| `requirements-dev.txt` extras | Dev only | pytest, JupyterLab, IPython kernel |

To upgrade all pinned dependencies to their latest compatible versions:

```bash
pip install pip-tools
pip-compile pyproject.toml --upgrade --output-file requirements.txt
pip-compile pyproject.toml --extra dev --upgrade --output-file requirements-dev.txt
```

---

## To Do

The items below are ordered by priority.  Items marked ✓ are done; open items are
the next suggested steps.  See `CLAUDE.md` "Legacy Alignment Gaps" for deeper
technical notes on each open item.

### Priority 1 — Complete the core pipeline (code alignment)

- [ ] **`TimeSeriesSplit` CV in classifier** (`src/market_regime/prediction/classifier.py`) —
  Replace the single `train_test_split(shuffle=False)` with 5-fold rolling walk-forward
  cross-validation.  This is the methodologically correct approach for financial time series
  and matches the legacy design in `legacy/supervised.py`.

- [ ] **Decision Tree classifier** (`src/market_regime/prediction/classifier.py`) —
  Add `train_decision_tree()` alongside the existing RandomForest.  A shallow DT
  (`max_depth=8`) gives interpretable rules and fast feature importance before the RF.
  Legacy design principle: "run a single Decision Tree just to get most of the
  explanatory power before running a Random Forest / XGBoost."

- [ ] **Portfolio construction module** (`src/market_regime/reporting/portfolio.py`) —
  Implement `simple_regime_portfolio()` (top-N assets, equal weight) and
  `blended_regime_portfolio()` (probability-weighted across all regimes), plus
  `generate_recommendation()` that compares current portfolio weights to targets and
  outputs BUY / SELL / HOLD signals.  Wire into `pipelines/07_dashboard.py`.
  Reference: `legacy/portfolio.py`.

- [ ] **Macro-data proxy fallback for step 6** (`src/market_regime/assets/returns.py`) —
  When yfinance ETF data is unavailable (network/SSL failure), fall back to quarterly
  returns derived from macro columns already in the features DataFrame (sp500, sp500_adj,
  10yr_ustreas, gdp_growth, us_infl, credit_spread).  This keeps the dashboard useful
  even without network access.  Reference: `legacy/asset_returns.py` `ASSET_PROXIES`.

### Priority 2 — Data coverage and model improvements

- [ ] **Macrotrends historical price data** — Backfill gold, oil, and long-bond price
  history pre-1993 from https://www.macrotrends.net/ so that ETF-era proxies (GLD,
  USO, TLT) have full 1950–present coverage for per-regime statistics.

- [ ] **`settings.yaml` `end_date` → null** — The date is currently hardcoded as
  `"2025-09-30"`.  Changing to `null` makes every fresh run fetch through today.
  Note: changing this invalidates existing checkpoints and triggers a re-scrape.

- [ ] **XGBoost / model comparison in step 5** — Benchmark XGBoost against the
  RandomForest and Decision Tree.  Keep the best model(s) for production use.
  Add a `train_xgboost()` function if XGBoost wins.

- [ ] **Expand test suite** — `tests/unit/` currently covers checkpoints, clustering,
  transforms, and basic asset returns (68 tests).  Add tests for:
  step 5 classifiers (mock sklearn), step 7 dashboard signals, step 6 proxy fallback,
  and portfolio construction functions.

### Priority 3 — "Putting it all together" (full end-to-end product)

These items realize the full vision described in the Overview above.

- [ ] **Individual asset-return predictors** — For each ETF (SPY, GLD, TLT, …) and
  each regime, train a binary classifier: "will this asset be +X% at Y quarters?"
  Use regime predictions + macro features as inputs.  Output: probability distribution
  over future performance for each asset, displayed as a stoplight dashboard.

- [ ] **Regime-conditional portfolio optimizer** — Given asset-return probability
  distributions and the blended regime portfolio weights, optimize allocation (e.g.
  mean-variance, risk-parity) subject to user-specified constraints.

- [ ] **Weekly automated report** — Schedule the pipeline to run weekly, generate a
  dashboard CSV + HTML, and send an AI-written narrative email summarizing the current
  regime, asset signals, and recommended portfolio changes.

### Completed ✓

- ✓ Data ingestion (multpl.com scraper, FRED API, yfinance ETF prices)
- ✓ Feature engineering (log transforms, Bernstein gap fill, smoothed derivatives)
- ✓ PCA + KMeans clustering (standard + size-constrained)
- ✓ Regime profiling, naming heuristics, transition matrix
- ✓ RandomForest regime classifier + forward binary classifiers
- ✓ Asset returns by regime (yfinance: SPY, GLD, TLT, USO, QQQ, IWM, VNQ, AGG)
  — **Note:** all 8 tickers are valid; earlier "possibly delisted" errors were SSL
  cascade failures now resolved via `certifi` CA bundle fix in `assets.py`
- ✓ Text + CSV dashboard with GREEN/YELLOW/RED asset signals
- ✓ CheckpointManager (parquet + manifest; avoids re-scraping)
- ✓ Full CLI (`run_pipeline.py --steps --refresh --recompute --plots …`)
- ✓ Exploration notebooks (01–08)
- ✓ Installation setup (`requirements.txt`, `setup.sh`, `Makefile`)
- ✓ Python 3.10+ compatibility (`from __future__ import annotations`, `certifi` dep,
  loose version bounds in requirements files)
