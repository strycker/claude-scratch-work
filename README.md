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

## To Do:
- Add historic Gold, Oil, TLT, etc. to datasets &mdash; see https://www.macrotrends.net/
- Standardize the time range (1950-2025), infer missing data, throw away or fix anything looking odd
- Change all variables that are exponential-looking into something normalized and predictive of regime, like taking a logrithm and/or using the 1st, 2nd, and 3rd derivative... likely CHANGE in a variable, or change RELATIVE to another variable is what will be predictive.  For example, the S&P500 itself is not a good signal of regime, but the S&P priced in gold or oil might be.
  - Consider using smoothed variables / polynomial fits / or other kinds of parameterized versions of the variable features as needed, as some might be too volatile over even quarterly time series
- For the initial unsupervised clustering phase of the project, can consider using adjusted / revised since I'm only trying to get regimes, so backward-looking features MIGHT be ok
- Once we have all quarters CLASSIFIED according to k-means or whatever, NOW we can look into
  - For each regime, find what assets CONSISTENTLY grew, e.g., for every quarter labeled for class X, asset Y always grew each quarter, no negative quarters.  There will likely be noise, but see if you can relax the criteria or find a cleaner signal, then find the right ETFs for the right asset classes or sectors (e.g., stagflation might mean gold is best, growth might mean tech and small caps best, etc.).
  - SUPERVISED predictive modeling using other features.  At this point you CANNOT use any variable that was revised or otherwise had forward-knowledge of the current or future state.  All features must be values known at the moment we would have been choosing a portfolio.
- During the supervised learning phase, good to first find feature importance, then reduce the number of features, then try running a single Decision Tree just to get most of the explanatory power before running a Random Forest / XGBoost or whatever is the best final model
