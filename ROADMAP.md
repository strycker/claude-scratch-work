# Trading-Crab — Product Roadmap

Prioritized backlog of features, data sources, and improvements.
Updated: March 2026.

---

## How to Read This

Each item has an effort estimate (S/M/L/XL) and a dependency note.
Items within a tier are roughly priority-ordered top → bottom.

---

## Tier 1 — High Impact, Achievable Soon

### 1.1  LightGBM supervised classifier  `M`
Add gradient-boosted classifier alongside RF + DT in `classifier.py`.
**Prefer LightGBM over XGBoost** for this dataset: at ~300 observations, LightGBM
is faster, more memory-efficient, and performs comparably.

Recommended hyperparameters for small-sample regime classification:
```python
lgb_params = {
    "num_leaves": 15,         # restrict to prevent overfitting
    "max_depth": 5,           # shallow trees = lower variance at N~300
    "min_child_samples": 5,   # higher leaf occupancy
    "learning_rate": 0.05,    # conservative; pair with more rounds
    "num_boost_round": 300,
    "feature_fraction": 0.8,  # column subsampling
    "bagging_fraction": 0.8,  # row subsampling
    "lambda_l2": 1.0,         # L2 regularization
    "class_weight": "balanced",
}
```
- New file: `src/market_regime/prediction/gradient_boosting.py`
- Functions: `train_lightgbm_current_regime()`, `train_lightgbm_forward()`
- Use same `_tscv_scores()` helper as RF + DT
- Do NOT over-tune hyperparameters with 300 obs (fixed grid, max 50 combos)
- Add `lightgbm>=4.0` as optional extra in `pyproject.toml`
- **Files**: `src/market_regime/prediction/gradient_boosting.py` (new), `pipelines/05_predict.py`

### 1.2  Additional FRED macro series  `S`
Several high-signal FRED series are free and require no new scraping infrastructure:

| Series ID | Description | Back to | Why useful |
|-----------|-------------|---------|------------|
| `VIXCLS` | CBOE VIX daily close | 1990 | Fear/volatility regime signal |
| `UNRATE` | Unemployment rate | 1948 | Recession leading indicator |
| `M2NS` | M2 money supply | 1959 | Inflation / liquidity regime |
| `T10Y2Y` | 10Y-2Y Treasury spread | 1976 | Inversion = recession predictor |
| `T10Y3M` | 10Y-3M Treasury spread | 1982 | Strongest recession signal |
| `HOUST` | Housing starts | 1959 | Cycle leading indicator |
| `UMCSENT` | U Michigan Consumer Sentiment | 1952 | Demand signal |
| `INDPRO` | Industrial Production Index | 1919 | Broad economic output |
| `PAYEMS` | Nonfarm payrolls | 1939 | Employment health |
| `DPCERA3Q086SBEA` | Real PCE quarterly | 1947 | Consumer spending |

- Add each to `config/settings.yaml` under `fred.series`
- Apply appropriate `shift` lag (VIX: none; payrolls: +1Q; PCE: +1Q)
- Rerun PCA + clustering after adding — expect silhouette improvement
- **Files**: `config/settings.yaml`, `src/market_regime/ingestion/fred.py`

### 1.3  Yield curve features  `S`
Compute derived yield-curve features in `transforms.py`:
- `yield_spread_10y2y` = GS10 − GS2 (add GS2 to FRED series)
- `yield_spread_10y3m` = GS10 − TB3MS (already have both)
- `yield_curve_slope` = (GS10 − TB3MS) / 10
- These are among the strongest empirical recession predictors in the literature
- **Files**: `src/market_regime/features/transforms.py`, `config/settings.yaml`

### 1.4  Empirical forward probabilities  `S`
Implement `compute_forward_probabilities()` from `legacy/regime_analysis.py`.
Computes empirical P(reach regime j within N quarters | currently in regime i)
as a diagnostic alongside model-based forward classifiers.
- Already spec'd in `CLAUDE.md` as Low Priority gap 5
- Output: `data/regimes/forward_probs_{N}q.parquet` for N in [1, 4, 8]
- **Files**: `src/market_regime/regime/profiler.py`, `pipelines/04_regime_label.py`

### 1.5  macrotrends.net historical price backfill  `M`
Extends commodity and asset data before 1993 (ETF inception dates):
- **Gold price**: monthly back to 1915 (`https://www.macrotrends.net/1333/historical-gold-prices-100-year-chart`)
- **WTI Crude Oil**: monthly back to 1946
- **Silver**: back to 1960
- **10Y Treasury yield**: back to 1962 (to cross-check FRED)
- macrotrends uses **static HTML tables** (NOT JavaScript-rendered) — confirmed via research.
- Parse approach: `pandas.read_html()` with CSS selector `table.historical_data_table`,
  OR `requests` + `BeautifulSoup` with `.select("table.historical_data_table")`.
  No Selenium or Playwright needed.
- Rate-limit to 2-3s between requests
- After resampling to quarterly, resample with `.mean()` (price) or `.last()` (rate)
- Merge into `macro_raw.parquet` alongside FRED + multpl series
- **Files**: `src/market_regime/ingestion/macrotrends.py` (new), `config/settings.yaml`

### 1.6  Expand asset universe and move ticker lists to config  `S`
Add ETFs that cover a wider range of regime-relevant categories:
- `HYG` — high-yield / junk bonds (credit risk / spread regime signal)
- `XLK` — Technology sector (growth-regime outperformer)
- `XLP` — Consumer staples (defensive / low-growth regime)
- `XLE` — Energy sector (stagflation / commodity regime)
- `GDX` — Gold miners (amplified gold / inflation hedge)
- `TIP` — TIPS / inflation-linked bonds (real yield signal)
- `BIL` — T-bills / cash equivalent (rising-rate / defensive)
- `EDV` — Extended-duration Treasuries 25+ yr (duration risk)

All ticker lists now live in `config/settings.yaml` under `assets.etfs`.
Notebooks read from `cfg["assets"]["etfs"]` — no hardcoded lists in notebook code.
`plotting.sample_series` and `plotting.key_indicators` also moved to config.
- **Files**: `config/settings.yaml`, `notebooks/01_ingestion.ipynb`, `notebooks/04_regimes.ipynb`,
  `notebooks/06_assets.ipynb`, `src/market_regime/plotting.py`
- **Status**: ✓ Done (settings.yaml + notebooks updated; ETF data fetched on next step 1 run)

### 1.7  Confusion matrix and classification report in plots  `S`
`legacy/supervised.py` has `generate_classification_report()` that produces a
confusion matrix; this is not exposed in `src/` plotting or logs.
- Add `plot_confusion_matrix(model, X, y, regime_names, run_cfg)` to `plotting.py`
- Call from `pipelines/05_predict.py` when `--plots` is set
- **Files**: `src/market_regime/plotting.py`, `pipelines/05_predict.py`

---

## Tier 2 — High Value, More Effort

### 2.1  Optimal k investigation — beyond silhouette  `S`
Silhouette score alone is insufficient to determine optimal k for regime detection.
Add a multi-metric k-selection panel to `notebooks/03_clustering.ipynb`:
- **Gap statistic** (Tibshirani et al.) — compares within-cluster dispersion to a
  null reference distribution; often picks a higher k than silhouette
- **BIC via GMM** — fit a `GaussianMixture(n_components=k)` for k in [2, 12];
  select k minimizing BIC; directly comparable to KMeans as a model-selection criterion
- **Elbow on within-cluster variance (inertia)** — classic, but use the "knee" heuristic
  (`kneed` library: `KneeLocator`) for automated detection
- **Davies-Bouldin + Calinski-Harabasz** — already computed; include in combined panel

Display all four metrics side-by-side; let user compare before committing to `balanced_k`.
- **Files**: `notebooks/03_clustering.ipynb`, `src/market_regime/clustering/kmeans.py`

### 2.2  Gaussian Mixture Models (GMM) as KMeans alternative  `M`
`sklearn.mixture.GaussianMixture` is a principled Bayesian alternative to KMeans:
- Soft cluster assignments (regime probabilities, not hard labels)
- BIC/AIC for model selection → no need to pre-specify k
- Can model elliptical (non-spherical) clusters — KMeans assumes equal-radius spheres
- Covariance types: try `"full"`, `"tied"`, `"diag"` — `"diag"` best for small N
- Risk: with N≈300 quarters and 69 features, GMM can overfit — use PCA components as input

Implementation:
- Add `fit_gmm(pca_df, k_range, covariance_type, random_state)` to `clustering/kmeans.py`
  (or new `clustering/gmm.py`)
- Returns both hard labels (argmax of responsibilities) and soft probabilities
- Compare silhouette and BIC vs KMeans; present in notebook 03
- Downstream: soft probabilities can replace hard labels as supervised model input
- **Files**: `src/market_regime/clustering/gmm.py` (new), `pipelines/03_cluster.py`,
  `notebooks/03_clustering.ipynb`

### 2.3  DBSCAN / HDBSCAN density-based clustering  `M`
Density-based clustering does not require specifying k; finds clusters of arbitrary shape
and treats outlier quarters as noise (label = -1):
- `sklearn.cluster.DBSCAN(eps, min_samples)` — requires tuning `eps` (distance threshold)
  Use k-distance plot to select `eps` (elbow in sorted k-NN distances)
- `hdbscan.HDBSCAN(min_cluster_size)` — hierarchical, more robust to `eps` choice;
  `pip install hdbscan` (optional extra)
- Key question: are some quarters genuinely outliers (not belonging to any regime)?
  DBSCAN answers this; KMeans forces every point into a cluster
- Limitation: noise-labeled quarters break downstream supervised training — need a strategy
  (e.g., assign noise to nearest cluster centroid, or drop from supervised training)
- **Files**: `src/market_regime/clustering/density.py` (new), `notebooks/03_clustering.ipynb`

### 2.4  Spectral Clustering  `M`
Graph-based clustering; constructs a similarity graph and clusters its eigenvectors:
- `sklearn.cluster.SpectralClustering(n_clusters, affinity)`
- Better than KMeans for non-convex cluster shapes (e.g., crescent or ring shapes in PCA space)
- More expensive: O(N³) eigendecomposition — feasible at N≈300
- Must specify k (same as KMeans), so use optimal-k from 2.1 as input
- Compare resulting labels to KMeans via adjusted Rand index and NMI
- **Files**: `src/market_regime/clustering/spectral.py` (new), `notebooks/03_clustering.ipynb`

### 2.5  SVD as complement / alternative to PCA  `S`
`sklearn.decomposition.TruncatedSVD` is mathematically equivalent to PCA on centered data
but worth examining explicitly:
- Compute SVD of the feature matrix directly; compare singular values to PCA explained variance
- Interpretability: SVD right singular vectors (components) have the same meaning as PCA loadings;
  visualize which raw features load most heavily on each component
- Use case: if features contain many near-zero rows (pre-1970 NaN-filled quarters), SVD on
  the non-centered matrix may give more stable components than centered PCA
- Add a comparison cell in `notebooks/03_clustering.ipynb`: PCA vs TruncatedSVD component
  loadings, variance explained, and resulting cluster quality
- **Files**: `notebooks/03_clustering.ipynb`, `src/market_regime/clustering/kmeans.py`

### 2.6  Feature selection for clustering using RF importances  `M`
The current `clustering_features` list (69 columns) was manually curated from the legacy
analysis. RF importances from step 5 provide a data-driven alternative:
- After step 5 runs, extract `feature_importances_` from the current-regime RF classifier
- Rank features by importance; identify which of the 69 clustering features are most
  predictive of the regime labels
- Re-run PCA + clustering using only the top-K features (try K in [20, 35, 50])
- Compare silhouette score, regime separation, and step-5 CV accuracy vs the full 69-feature set
- Key question: does a leaner feature set produce more stable, more predictable regimes?
- **Important**: any change to `clustering_features` invalidates `regime_labels.yaml` — re-pin
  regime names after re-clustering
- **Files**: `notebooks/03_clustering.ipynb`, `config/settings.yaml`

### 2.7  Multi-clustering model selection strategy  `S`
When running multiple clustering methods (KMeans, GMM, DBSCAN, Spectral), a principled
selection strategy is needed:
- **Quantitative**: compare silhouette, Davies-Bouldin, and (for GMM) BIC across methods;
  compute adjusted Rand index between pairs to measure label agreement
- **Economic**: for each candidate clustering, run step 5 (regime classifier) and compare
  5-fold TSCV accuracy — the "best" clustering should be the most predictive
- **Visual**: overlay each method's labels on the PCA scatter (notebook 07 pairplot);
  prefer methods where regimes form compact, well-separated clouds
- **CLI workflow**: after selecting a preferred clustering, save its labels as a named
  checkpoint (see README "Using new cluster labels" section) and propagate through steps 4–7
- **Files**: `notebooks/03_clustering.ipynb`, `run_pipeline.py`

### 2.8  Finviz Elite integration for sector/stock signals  `M`
With a Finviz Elite subscription:
- Use `finvizfinance` Python library (`pip install finvizfinance`)
- Screener API: pull all S&P 500 stocks filtered by sector, market cap, momentum
- Quarterly sector aggregation: for each regime, which sectors (XLK, XLF, XLE, etc.) outperform?
- Useful for "within-regime" stock picking after portfolio ETF allocation is set
- **Note**: Finviz data is point-in-time; historical screener data requires Elite API
- Separate from regime detection (which is macro-driven); feeds into a "stock signal" layer
- **Files**: `src/market_regime/ingestion/finviz.py` (new), `pipelines/08_stock_signals.py` (new)

### 2.9  Hidden Markov Model regime detection (alternative to KMeans)  `M`
`hmmlearn.hmm.GaussianHMM` is a principled alternative to KMeans for regime detection:
- Handles temporal autocorrelation natively (KMeans treats each quarter independently)
- Produces soft probabilities rather than hard cluster assignments
- Compare: does HMM agree with KMeans regimes? Does it produce cleaner transitions?
- Risk: HMM requires EM fitting which is sensitive to initialization on small datasets
- Implementation: add `fit_hmm()` to `src/market_regime/clustering/hmm.py` (new file)
- Use identical PCA features as input for fair comparison with KMeans
- **Files**: `src/market_regime/clustering/hmm.py` (new), `pipelines/03_cluster.py`

### 2.10  SMOTE / class-weight tuning for imbalanced regimes  `S`
With 5 balanced clusters, sizes should be equal, but temporal distribution may still
cause class imbalance in train/test splits of the TSCV folds.
- RF already uses `class_weight="balanced"` — log per-fold class counts to verify
- Consider `imbalanced-learn` SMOTE for XGBoost (which doesn't have class_weight)
- Add to `pyproject.toml` as optional extra: `imbalanced-learn>=0.11`
- **Files**: `src/market_regime/prediction/classifier.py`

### 2.11  Per-asset regime probability models  `L`
For each ETF (SPY, GLD, TLT, USO, QQQ, IWM, VNQ, AGG), train per-asset models:
- Binary: "Will this ETF be +X% in Y quarters?" for X in [5, 10, 20] and Y in [1, 2, 4, 8]
- Features: regime probabilities + causal macro features + asset momentum
- Output: per-asset stoplight probability matrix → feeds dashboard signal layer
- This is "Putting it all together — Part I" from the original design doc
- **Files**: `src/market_regime/prediction/asset_classifier.py` (new), `pipelines/05b_asset_predict.py` (new)

### 2.12  Momentum and cross-asset ratio features  `M`
Additional derived features for clustering and supervised models:
- 6M and 12M momentum (trailing return) for each major series
- Relative strength: S&P priced in Gold, S&P priced in Oil, Gold priced in Oil
- Cross-asset correlation (rolling 8Q window) between SP500 and 10Y yield
- Inflation acceleration: 2nd derivative of CPI (d/dt of d/dt)
- PMI-equivalent proxy from FRED INDPRO momentum
- **Files**: `src/market_regime/features/transforms.py`, `config/settings.yaml`

### 2.13  Markov regime-switching model (statsmodels)  `M`
`statsmodels.tsa.regime_switching.markov_regression.MarkovRegression` fits a model
where parameters switch between discrete states via a Markov chain:
- Interprets GDP growth as a switching-mean process (growth vs recession states)
- Useful as a 2-state sanity check: does our 5-regime KMeans align with the
  statsmodels recession/expansion signal?
- Not a replacement for KMeans; more of a diagnostic and feature generator
- **Files**: `src/market_regime/clustering/markov.py` (new)

### 2.14  Conference Board LEI proxy from FRED  `S`
The Conference Board LEI is the gold standard for recession prediction but is not
freely available. Construct a proxy from FRED components:
- `PERMIT` (building permits) + `AWHMAN` (avg weekly hours) + `AMDMNO` (new orders)
  + `ISM manufacturing` + `UMCSENT` + spread measures = 6-component LEI approximation
- Validate against NBER recession dates (`USREC` on FRED — binary recession indicator)
- **Files**: `src/market_regime/features/transforms.py`, `config/settings.yaml`

---

## Tier 3 — Longer-term Vision

### 3.1  Weekly automated report with AI narrative  `XL`
Full automation of the pipeline from cron job to email:
- `cron` or GitHub Actions: run every Friday at market close
- Pull latest data (FRED releases, multpl.com, yfinance)
- Run steps 2–7 (features → dashboard)
- Draft AI narrative using Claude API: "This week the regime probability shifted..."
- Send via SendGrid / AWS SES / Gmail SMTP
- **Files**: `scripts/weekly_report.py` (new), `.github/workflows/weekly.yml` (new)

### 3.2  Interactive Streamlit dashboard  `L`
Replace the terminal `print_dashboard()` with a Streamlit web app:
- Tabs: Regime Overview / Asset Signals / Portfolio / History
- Live regime probability gauge chart
- Regime timeline (colored scatter) back to 1950
- Asset heatmap and stoplight table
- Trade recommendations with current vs target weight sliders
- **Files**: `app/dashboard.py` (new)

### 3.3  Macrotrends deep history backfill  `M`
Additional macrotrends series for pre-1970 data:
- Gold-to-S&P ratio (1915–present)
- Silver price
- Copper price (industrial demand proxy)
- Dow Jones (pre-S&P 500 era)
- Fed Funds Rate historical (FRED already has back to 1954; macrotrends back to 1800s)

### 3.4  Factor model for asset returns within regimes  `L`
LASSO regression / Ridge regression per regime:
- Dependent variable: next-quarter ETF return
- Independent variables: causal macro features for that regime
- Gives coefficient insights: "in stagflation regimes, credit spread and gold momentum
  are the dominant predictors of GLD outperformance"
- **Files**: `src/market_regime/prediction/factor_model.py` (new)

### 3.5  Backtest framework  `XL`
Walk-forward backtest of the full pipeline:
- At each quarter T, train on [T-N, T], predict regime and portfolio for T+1
- Compare strategy vs S&P 500 benchmark: returns, Sharpe, max drawdown
- Avoids look-ahead by construction (causal features + TSCV)
- Requires ~50 walk-forward steps (1975–2025 at quarterly resolution)
- **Files**: `src/market_regime/backtest/` (new module)

### 3.6  StockCharts.com — historical data scraping  `M`
StockCharts.com (subscription already active) has historical OHLCV chart data
but no public JSON/CSV export API.  Potential approaches:
- **Symbol lookup + CSV export**: StockCharts renders chart data as an embedded
  JavaScript array in its `SharpCharts` pages.  Scraping with `requests` +
  regex/json extraction may work for daily close data.
- **`/def/` page scraping**: the `stockcharts.com/h-sc/ui?s={SYMBOL}&type=BAR`
  endpoint returns chart HTML; inspect for embedded `chartData` JSON objects.
- **Use case**: primary value is as a yfinance fallback for historical close prices
  (Phase 5 before macro proxy), and for technical indicators (RSI, MACD, etc.)
  that are rendered on the charts.
- **Risk**: ToS review required; rate-limit to ≥3s/request; no guaranteed format stability.
- **Alternative**: compute the same technical indicators from yfinance/stooq OHLCV
  using the `ta` or `pandas-ta` library — avoids scraping entirely.
- **Files**: `src/market_regime/ingestion/stockcharts.py` (new)

### 3.7  Finviz Elite — sector/fundamental overlays  `M`
Finviz Elite (subscription already active) is a **stock screener**, not a
historical price data source.  It is NOT suitable as a yfinance price fallback.

What Finviz IS good for:
- Current fundamental data (P/E, EPS, sector, market cap) per ticker
- Sector-level performance views (1W, 1M, 3M, YTD heatmaps)
- Screener for within-regime stock picking (which stocks in XLK outperform in growth regimes?)
- News sentiment per ticker

Implementation approach (when ready):
- Use `finvizfinance` Python library: `pip install finvizfinance`
- `finvizfinance.main.finvizfinance('SPY').ticker_fundament()` → current fundamentals
- `finvizfinance.group.performance.Performance().screener_view(...)` → sector perf
- **Files**: `src/market_regime/ingestion/finviz.py` (new), `pipelines/08_stock_signals.py` (new)
- **Note**: historical screener data requires Finviz Elite API; current data is available
  via the `finvizfinance` library without authentication for many fields

---

## Data Sources Master Table

| Source | Library/Approach | What We Get | Back to | In Pipeline? | Priority |
|--------|-----------------|-------------|---------|-------------|----------|
| multpl.com | lxml scraper | 46 Shiller series | varies | ✓ Step 1 | Done |
| FRED API | `fredapi` | GDP, CPI, BAA, AAA, GS10, TB3MS, GNP | varies | ✓ Step 1 | Done |
| yfinance | `yfinance` | ETF OHLCV (SPY, GLD, TLT, USO, QQQ, IWM, VNQ, AGG) | 1993+ | ✓ Step 6 | Done |
| FRED — VIX | `fredapi` | VIXCLS daily volatility index | 1990 | ✗ | **Tier 1** |
| FRED — unemployment | `fredapi` | UNRATE monthly | 1948 | ✗ | **Tier 1** |
| FRED — M2 | `fredapi` | M2NS money supply | 1959 | ✗ | **Tier 1** |
| FRED — yield spreads | `fredapi` | T10Y2Y, T10Y3M, GS2 | varies | ✗ | **Tier 1** |
| FRED — housing | `fredapi` | HOUST, PERMIT | 1959 | ✗ | **Tier 1** |
| FRED — consumer | `fredapi` | UMCSENT, DPCERA3Q086SBEA | 1952 | ✗ | **Tier 1** |
| macrotrends.net | custom scraper | Gold, oil, silver prices | 1915+ | ✗ | **Tier 1** |
| stooq.pl | `pandas-datareader` | Free ETF/stock OHLCV (Phase 3 yfinance fallback) | ~1993 | ✓ Phase 3 | Done (optional install) |
| OpenBB | `openbb` | Multi-provider ETF prices (Phase 4 yfinance fallback) | varies | ✓ Phase 4 | Done (optional install) |
| Finviz Elite | `finvizfinance` | Sector screener + fundamentals (NOT historical prices) | recent | ✗ | Tier 3 (3.7) |
| StockCharts.com | custom scraper | Chart data + technical indicators | varies | ✗ | Tier 3 (3.6) |
| hmmlearn | Python lib | HMM regime states | n/a | ✗ | Tier 2 (2.9) |
| statsmodels | Python lib | Markov regime-switching | n/a | ✗ | Tier 2 (2.13) |
| sklearn GMM | Python lib | Gaussian Mixture Models (soft clusters) | n/a | ✗ | Tier 2 (2.2) |
| sklearn SpectralClustering | Python lib | Spectral / graph clustering | n/a | ✗ | Tier 2 (2.4) |
| hdbscan | Python lib | Density-based clustering (HDBSCAN) | n/a | ✗ | Tier 2 (2.3) |
| Streamlit | Python lib | Interactive dashboard | n/a | ✗ | Tier 3 |
| Claude API | `anthropic` | AI weekly narrative | n/a | ✗ | Tier 3 |
| StockCharts | scrape | Historical OHLCV + technical indicators | varies | ✗ | Tier 3 (3.6) |

---

## What to Do This Session (Suggested Starting Points)

1. Add FRED series (VIX, unemployment, M2, yield spreads) — very low effort, high signal
2. Add yield curve features in `transforms.py` — computed from existing FRED data
3. Add `compute_forward_probabilities()` to `profiler.py` — small gap, legacy already has it
4. Add `plot_confusion_matrix()` to `plotting.py` — small visualization gap
5. Start `macrotrends.py` scraper — extends gold/oil back to 1915/1946

Items 1-4 can be done in a single session. Item 5 needs care with scraping.
