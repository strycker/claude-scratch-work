# Trading-Crab

Market regime classification and prediction pipeline.

Predict market conditions, best portfolios, and stock picks.

<br>

![Glenn with crab 2025](/images/glenn_with_crab_2025_300x400.png)
![Glenn with Wei-Xuen and Wei-Tong 2025](/images/glenn_weixuen_weitong_with_crab_300x400.png)

<br>

___

<br>

## Overview

Predicts market conditions, optimal portfolios, and stock picks by:

1. **Data Ingestion** — Scrapes macro financial data from multpl.com and the FRED API (quarterly resolution, ~1950–present).
2. **Feature Engineering** — Log transforms, smoothed derivatives (1st–3rd order), cross-asset ratios, Bernstein-polynomial gap filling.
3. **Clustering** — PCA dimensionality reduction, KMeans + size-constrained KMeans to label each quarter with a market regime.
4. **Regime Interpretation** — Statistical profiling of each cluster to assign human-readable names (e.g. "Stagflation", "Growth Boom").
5. **Supervised Prediction** — Classifiers to predict today's regime from currently-available features (no look-ahead).
6. **Transition Probabilities** — Empirical regime transition matrices and forward-looking probability models.
7. **Asset Returns by Regime** — Per-regime median returns for major asset classes.
8. **Portfolio Construction** — Regime-conditional portfolio recommendations.

<br>

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

<br>
<br>

## To Do:
- Reduce number of rows in the initial dataset, as many do not span the right time range
- Add historic Gold, Oil, TLT, etc. to datasets &mdash; see https://www.macrotrends.net/
- Standardize the time range (1950-2025?), infer missing data, throw away or fix anything looking odd
- Change all variables that are exponential-looking into something normalized and predictive of regime, like taking a logrithm and/or using the 1st, 2nd, and 3rd derivative... likely CHANGE in a variable, or change RELATIVE to another variable is what will be predictive.  For example, the S&P500 itself is not a good signal of regime, but the S&P priced in gold or oil might be.
  - Consider using smoothed variables / polynomial fits / or other kinds of parameterized versions of the variable features as needed, as some might be too volatile over even quarterly time series
- For the initial unsupervised clustering phase of the project, can consider using adjusted / revised since I'm only trying to get regimes, so backward-looking features MIGHT be ok
- Once we have all quarters CLASSIFIED according to k-means or whatever, NOW we can look into
  - For each regime, find what assets CONSISTENTLY grew, e.g., for every quarter labeled for class X, asset Y always grew each quarter, no negative quarters.  There will likely be noise, but see if you can relax the criteria or find a cleaner signal, then find the right ETFs for the right asset classes or sectors (e.g., stagflation might mean gold is best, growth might mean tech and small caps best, etc.).
  - SUPERVISED predictive modeling using other features.  At this point you CANNOT use any variable that was revised or otherwise had forward-knowledge of the current or future state.  All features must be values known at the moment we would have been choosing a portfolio.
- During the supervised learning phase, good to first find feature importance, then reduce the number of features, then try running a single Decision Tree just to get most of the explanatory power before running a Random Forest / XGBoost or whatever is the best final model

<br>
<br>

## Repository Structure

```
trading_crab/
├── README.md
├── requirements.txt
├── setup.py
├── src/python           # Python package
│   ├── __init__.py
│   ├── config.py           # All constants, feature lists, CLI parsing
│   ├── data_ingestion.py   # Scraping multpl.com, FRED API, Grok labels
│   ├── feature_engineering.py  # Log transforms, derivatives, interpolation
│   ├── clustering.py       # PCA, KMeans, regime labeling
│   ├── regime_analysis.py  # Interpretation, transitions, naming
│   ├── supervised.py       # Regime prediction models
│   ├── asset_returns.py    # Per-regime asset class performance
│   ├── portfolio.py        # Portfolio construction
│   ├── plotting.py         # All visualization helpers
│   └── pipeline.py         # End-to-end orchestration
├── tests/
│   ├── __init__.py
│   ├── test_data_ingestion.py
│   ├── test_feature_engineering.py
│   ├── test_clustering.py
│   └── test_regime_analysis.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_clustering.ipynb
│   └── 03_regime_analysis.ipynb
└── data/                   # Pickle/CSV checkpoints (gitignored)
```

## Usage

```bash
# Full pipeline (scrape + compute + cluster + analyze)
python -m trading_crab.pipeline --refresh --recompute --plots

# Load from checkpoints, just re-cluster
python -m trading_crab.pipeline --plots

# Recompute derived features but don't re-scrape
python -m trading_crab.pipeline --recompute

# Run tests
pytest tests/ -v
```

## Requirements

See `requirements.txt`. Key dependencies: pandas, numpy, scikit-learn,
scipy, fredapi, k-means-constrained, lxml, requests, python-dotenv, matplotlib.
