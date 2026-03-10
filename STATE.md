# Trading-Crab — Current State

Snapshot of what is implemented, what runs, and what doesn't.
Updated: March 2026.

---

## Pipeline Steps — Status

| Step | Script | Status | Notes |
|------|--------|--------|-------|
| 1 — Ingest | `pipelines/01_ingest.py` | ✅ Working | multpl.com + FRED API |
| 2 — Features | `pipelines/02_features.py` | ✅ Working | Produces both centered + causal parquets |
| 3 — Cluster | `pipelines/03_cluster.py` | ✅ Working | KMeans + KMeansConstrained |
| 4 — Label | `pipelines/04_regime_label.py` | ✅ Working | Profiles + transition matrix |
| 5 — Predict | `pipelines/05_predict.py` | ✅ Working | RF + DT + TSCV + forward classifiers |
| 6 — Assets | `pipelines/06_asset_returns.py` | ✅ Working | yfinance + macro proxy fallback |
| 7 — Dashboard | `pipelines/07_dashboard.py` | ✅ Working | Signals + portfolio + BUY/SELL/HOLD |
| Master runner | `run_pipeline.py` | ✅ Working | All flags implemented |

---

## Unit Tests

```
tests/unit/test_checkpoints.py    18 tests — ✅ all passing
tests/unit/test_clustering.py     15 tests — ✅ all passing
tests/unit/test_returns.py        14 tests — ✅ all passing
tests/unit/test_transforms.py     21 tests — ✅ all passing
─────────────────────────────────────────────────────────
Total: 68 tests — ✅ all passing (Python 3.11)
```

**Coverage gaps** (no tests for):
- `src/market_regime/prediction/classifier.py` — classifier training + TSCV
- `src/market_regime/reporting/portfolio.py` — portfolio construction
- `src/market_regime/reporting/dashboard.py` — dashboard signals
- `src/market_regime/ingestion/` — all ingestion (mocked network access needed)
- `src/market_regime/regime/profiler.py` — regime naming heuristics
- `src/market_regime/plotting.py` — plotting functions

---

## Implemented Features

### Data Ingestion
- ✅ multpl.com scraper: 46 quarterly series via lxml
- ✅ FRED API: GDP, GNP, BAA, AAA, CPI, GS10, TB3MS (7 series)
- ✅ yfinance: SPY, GLD, TLT, USO, QQQ, IWM, VNQ, AGG (8 ETFs)
- ✅ Grok baseline labels: `data/grok_quarter_classifications_20260216.pickle`
- ✅ SSL fix for curl_cffi (macOS/proxy environments)
- ✅ Publication-lag shift for GDP (+1Q) and GNP (+1Q)

### Feature Engineering
- ✅ Cross-asset ratios: 10 derived columns (div_yield2, price_gdp, credit_spread, etc.)
- ✅ Log transforms: 23 columns
- ✅ Column selection: `initial_features` (36 cols) and `clustering_features` (69 cols)
- ✅ Bernstein polynomial gap fill (interior) with Taylor extrapolation (edges)
- ✅ Smoothed derivatives: d1, d2, d3 per column via `np.gradient`
- ✅ Centered smoothing for clustering (`causal=False`)
- ✅ Causal/backward smoothing for supervised learning (`causal=True`)

### Clustering
- ✅ StandardScaler → PCA(5) → StandardScaler → KMeans
- ✅ K-sweep: k=2..12, silhouette + CH + DB scores
- ✅ Best-k selection with `k_cap=5`
- ✅ KMeansConstrained balanced clustering with `balanced_k=5`
- ✅ Optional `--no-constrained` fallback for environments without the package

### Regime Profiling
- ✅ `build_profiles()`: mean/std of features per regime
- ✅ `suggest_names()`: heuristic regime naming (5 rules)
- ✅ `build_transition_matrix()`: empirical 1-step transition probabilities
- ✅ `load_name_overrides()`: reads `config/regime_labels.yaml`

### Supervised Prediction
- ✅ `train_current_regime()`: RandomForest with TSCV (gap 1 — done)
- ✅ `train_decision_tree()`: shallow DecisionTree with TSCV (gap 2 — done)
- ✅ `train_forward_classifiers()`: binary RF per (horizon, regime) pair
- ✅ `predict_current()`: returns regime + probabilities for most recent quarter

### Asset Returns
- ✅ `compute_quarterly_returns()`: pct_change from yfinance ETF prices
- ✅ `compute_proxy_returns()`: fallback from macro_raw.parquet columns (gap 4 — done)
- ✅ `returns_by_regime()`: median/mean/std per regime per asset
- ✅ `rank_assets_by_regime()`: ranked flat form for dashboard

### Portfolio Construction (gap 3 — done)
- ✅ `simple_regime_portfolio()`: equal-weight top-3 for current regime
- ✅ `blended_regime_portfolio()`: probability-weighted across all regimes
- ✅ `generate_recommendation()`: BUY/SELL/HOLD vs current holdings

### Dashboard and Reporting
- ✅ GREEN/YELLOW/RED asset signals per regime
- ✅ `print_dashboard()`: terminal output with regime + signals + transitions
- ✅ `save_dashboard_csv()`: timestamped CSV to `outputs/reports/`
- ✅ `portfolio_simple.csv`, `portfolio_blended.csv`, `trade_recommendations.csv`

### Infrastructure
- ✅ `CheckpointManager`: parquet + manifest, freshness check, list/clear
- ✅ `RunConfig`: dataclass with `from_args()` factory
- ✅ Full CLI: `--refresh`, `--recompute`, `--plots`, `--steps`, `--market-code`, etc.
- ✅ `config/settings.yaml`: all tunable parameters

---

## Known Gaps (Not Yet Implemented)

### Priority 1 (implement next)
| Gap | Where | Effort |
|-----|-------|--------|
| XGBoost/LightGBM classifiers | `classifier.py` | S |
| Additional FRED series (VIX, unemployment, M2, yield spreads, housing) | `settings.yaml` + FRED ingestion | S |
| Yield curve derived features (10Y-2Y, 10Y-3M spreads) | `transforms.py` | S |
| Empirical forward probabilities | `profiler.py` | S |
| macrotrends.net scraper (gold, oil pre-1993) | `ingestion/macrotrends.py` (new) | M |
| Confusion matrix plot | `plotting.py` | S |

### Priority 2
| Gap | Where | Effort |
|-----|-------|--------|
| Hidden Markov Model regime detection | `clustering/hmm.py` (new) | M |
| SMOTE for class imbalance in XGB training | `classifier.py` | S |
| Per-asset regime probability models | `prediction/asset_classifier.py` (new) | L |
| Momentum + cross-asset ratio features | `transforms.py` | M |
| Finviz Elite sector signals | `ingestion/finviz.py` (new) | M |

### Priority 3
| Gap | Where | Effort |
|-----|-------|--------|
| Weekly automated report | `scripts/weekly_report.py` (new) | XL |
| Streamlit dashboard | `app/dashboard.py` (new) | L |
| Backtest framework | `src/market_regime/backtest/` (new) | XL |
| `joblib.dump` for sklearn model serialization | `pipelines/05_predict.py` | S |
| `end_date: null` → use today | `settings.yaml` + ingestion | S |

---

## Data Coverage

| Series | Source | Start Date | Frequency |
|--------|--------|-----------|-----------|
| S&P 500 price | multpl.com | 1871 | Quarterly |
| S&P 500 PE (CAPE) | multpl.com | 1881 | Quarterly |
| US Inflation (CPI) | multpl.com + FRED | 1950 | Quarterly |
| 10Y Treasury yield | multpl.com + FRED | 1950 | Quarterly |
| Dividend yield | multpl.com | 1871 | Quarterly |
| BAA/AAA corporate yields | FRED | 1919/1919 | Quarterly |
| GDP | FRED (shifted +1Q) | 1947 | Quarterly |
| GNP | FRED (shifted +1Q) | 1947 | Quarterly |
| Gold (ETF GLD) | yfinance | 2004 | Quarterly |
| Oil (ETF USO) | yfinance | 2006 | Quarterly |
| Bonds (ETF TLT) | yfinance | 2002 | Quarterly |
| SPY / QQQ / IWM / VNQ / AGG | yfinance | 1993-2003 | Quarterly |
| Gold (spot price proxy) | macrotrends.net | **Not yet** | Monthly |
| WTI Crude (spot) | macrotrends.net | **Not yet** | Monthly |
| VIX | FRED | **Not yet** | Daily |
| Unemployment | FRED | **Not yet** | Monthly |
| M2 Money Supply | FRED | **Not yet** | Monthly |
| 10Y-2Y Spread | FRED | **Not yet** | Daily |

---

## Output Files (when pipeline runs successfully)

```
data/raw/
  macro_raw.parquet          — ~300 rows × ~50 cols (FRED + multpl combined)
  asset_prices.parquet       — quarterly ETF prices from yfinance

data/processed/
  features.parquet           — ~300 rows × ~70 cols (centered smoothing)
  features_supervised.parquet — same shape (causal/backward smoothing)

data/regimes/
  cluster_labels.parquet     — cluster + balanced_cluster columns
  pca_components.parquet     — 5 PCA components
  kmeans_scores.parquet      — silhouette/CH/DB vs k
  profiles.parquet           — mean/std per regime per feature
  asset_return_profile.parquet — median return per regime per ETF
  transition_matrix.parquet  — 5×5 regime transition probabilities
  regime_names_suggested.yaml — auto-generated regime name suggestions

outputs/models/
  current_regime.pkl         — fitted RandomForestClassifier
  decision_tree.pkl          — fitted DecisionTreeClassifier (interpretable)
  forward_classifiers.pkl    — {horizon: {regime: binary RF}}

outputs/reports/
  dashboard.csv              — timestamped asset signals
  portfolio_simple.csv       — equal-weight top-3 for current regime
  portfolio_blended.csv      — probability-weighted blended weights
  trade_recommendations.csv  — BUY/SELL/HOLD vs all-cash baseline

outputs/plots/               — PNG figures from --plots flag
```

---

## Environment

| Item | Value |
|------|-------|
| Python | 3.10+ (tested on 3.11) |
| Key deps | scikit-learn, pandas, numpy, scipy, fredapi, lxml, yfinance, certifi |
| Optional | k-means-constrained (balanced clustering) |
| API keys | FRED_API_KEY (free at fred.stlouisfed.org) |
| Finviz | Elite subscription (for future sector/stock signals) |

---

## Last Verified End-to-End Run

- Date: March 2026
- Python: 3.11
- All 7 steps ran successfully
- 68 unit tests pass
- Regime labels saved in `data/regimes/`; models in `outputs/models/`
- All 4 legacy alignment gaps (TSCV, DT, portfolio, proxy returns) closed
- Causal vs centered smoothing split implemented
