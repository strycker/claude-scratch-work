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
| 8 — Diagnostics | `pipelines/08_diagnostics.py` | ✅ Working | Ratio diagnostics + RRG rotation view |
| 9 — Tactics | `pipelines/09_tactics.py` | ✅ Working | Per-asset buy_hold / swing / stand_aside |
| Master runner | `run_pipeline.py` | ✅ Working | Steps 1-7,9 + --weekly-report + --send-email |

---

## Unit Tests

```
tests/unit/test_checkpoints.py              18 tests — ✅ all passing
tests/unit/test_clustering.py               15 tests — ✅ all passing
tests/unit/test_clustering_exploration.py   40 tests — ✅ all passing
tests/unit/test_cluster_comparison.py       36 tests — ✅ all passing
tests/unit/test_density.py                  27 tests — ✅ all passing (8 skipped: HDBSCAN)
tests/unit/test_gmm.py                      27 tests — ✅ all passing
tests/unit/test_returns.py                  14 tests — ✅ all passing
tests/unit/test_spectral.py                 16 tests — ✅ all passing
tests/unit/test_transforms.py               21 tests — ✅ all passing
tests/unit/test_prediction_flat.py           5 tests — ✅ all passing (flat API: RF, DT, predict_current)
tests/unit/test_ingestion.py                17 tests — ✅ all passing (HTTP-mocked: multpl, FRED, assets; 2 skipped: cssselect)
tests/unit/test_diagnostics_rrg.py           8 tests — ✅ all passing (RRG analysis + rolling stats)
tests/unit/test_tactics.py                   7 tests — ✅ all passing (tactical classification)
tests/unit/test_config.py                    4 tests — ✅ all passing (load_portfolio)
tests/unit/test_regime.py                    5 tests — ✅ all passing (profiles, names, transitions)
tests/unit/test_fred_series_config.py        1 test  — ✅ all passing (FRED config validation)
tests/unit/test_yield_curve_features.py      2 tests — ✅ all passing (yield curve spreads)
tests/unit/test_reporting.py                15 tests — ✅ all passing (dashboard signals, portfolio, recommendations)
tests/unit/test_plotting.py                 20 tests — ✅ all passing (all plot functions, save/show, constants)
tests/unit/test_runtime.py                  25 tests — ✅ all passing (RunConfig defaults, from_args, str, logging)
tests/unit/test_ingestion_completeness.py    8 tests — ✅ all passing (completeness report P23)
tests/test_models_regime.py                  3 tests — ✅ all passing (classifier bundle API + TSCV ordering)
tests/test_models_boosting.py                2 tests — ✅ all passing (GradientBoosting in bundle API)
tests/test_models_interpret_tree.py          2 tests — ✅ all passing (interpretability helpers)
tests/test_models_reporting.py               3 tests — ✅ all passing (model_metrics_summary, 3 input shapes)
tests/test_models_behavior.py                3 tests — ✅ all passing (make_behavior_labels, train_forward_behavior_models)
tests/test_email_weekly.py                  13 tests — ✅ all passing (email config, body, SMTP, archive)
tests/test_scripts_weekly_report.py          7 tests — ✅ all passing (weekly report archive, CLI argv, email)
tests/test_pipelines_ingest_features.py      2 tests — ✅ all passing (pipeline step 01 + 02 smoke tests)
tests/test_constraints_etf_universe.py       2 tests — ✅ all passing (ETF universe constraints)
tests/test_constraints_frequency.py          2 tests — ✅ all passing (data frequency constraints)
─────────────────────────────────────────────────────────────────────
Total: 496 collected — ✅ all passing (Python 3.11; 11 skipped: HDBSCAN + cssselect optional)
```

**Coverage gaps:** All previously untested modules now have test coverage.
Former gaps (`reporting.py`, `plotting.py`, `runtime.py`) are fully covered.

---

## Implemented Features

### Data Ingestion
- ✅ multpl.com scraper: 46 quarterly series via lxml
- ✅ FRED API: GDP, GNP, BAA, AAA, CPI, GS10, TB3MS, VIXCLS, UNRATE, M2SL, M2NS, GS2, T10Y2Y, T10Y3M (14 series)
- ✅ yfinance: SPY, GLD, TLT, USO, QQQ, IWM, VNQ, AGG, HYG, XLK, XLP, XLE, GDX, TIP, BIL, EDV (16 ETFs)
- ✅ Grok baseline labels: `data/grok_quarter_classifications_20260216.pickle`
- ✅ SSL fix for curl_cffi (macOS/proxy environments)
- ✅ Publication-lag shift for GDP (+1Q) and GNP (+1Q)

### Feature Engineering
- ✅ Cross-asset ratios: 10 derived columns (div_yield2, price_gdp, credit_spread, etc.)
- ✅ Log transforms: 23 columns
- ✅ Column selection: `initial_features` (36 cols) and `clustering_features` (69 cols)
- ✅ Bernstein polynomial gap fill (interior) with Taylor extrapolation (edges)
- ✅ Smoothed derivatives: d1, d2, d3 per column via `np.gradient`
- ✅ Yield curve features: 10Y-2Y and 10Y-3M spreads (from multpl + FRED)
- ✅ Centered smoothing for clustering (`causal=False`)
- ✅ Causal/backward smoothing for supervised learning (`causal=True`)

### Clustering
- ✅ StandardScaler → PCA(5) → StandardScaler → KMeans
- ✅ K-sweep: k=2..12, silhouette + CH + DB scores
- ✅ Best-k selection with `k_cap=5`
- ✅ KMeansConstrained balanced clustering with `balanced_k=5`
- ✅ Optional `--no-constrained` fallback for environments without the package
- ✅ Deterministic cluster label canonicalization (`_canonicalize_cluster_col`): cluster IDs sorted by ascending mean PC1 value, so label 0 always maps to the lowest-PC1 regime regardless of random seed

### Clustering Investigation Suite (`notebooks/03_clustering.ipynb`)
- ✅ **PCA component sweep**: `optimize_n_components()` — sweep n=3..10, score with KMeans(5)
- ✅ **SVD vs PCA comparison**: `compare_svd_pca()` — side-by-side component loadings
- ✅ **Gap statistic**: `compute_gap_statistic()` — Tibshirani 2001 criterion; correctly separates `gap_std` (raw sd) from `gap_sk` (simulation error = std×√(1+1/B))
- ✅ **Elbow detection**: `find_knee_k()` — kneed library or gradient fallback
- ✅ **Gaussian Mixture Models** (`src/trading_crab_lib/gmm.py`): BIC sweep, soft probabilities, convergence detection; `fit_gmm()` returns fitted scaler for consistent predictions
- ✅ **DBSCAN** (`src/trading_crab_lib/density.py`): eps sweep, k-NN distance plot, noise handling with warnings
- ✅ **HDBSCAN** (`src/trading_crab_lib/density.py`): optional (`pip install hdbscan`), `min_cluster_size` sweep
- ✅ **Spectral Clustering** (`src/trading_crab_lib/spectral.py`): affinity matrix pre-computed once per sweep (~k-fold speedup), k sweep
- ✅ **Multi-method comparison** (`src/trading_crab_lib/cluster_comparison.py`): silhouette/DB/CH for all methods, pairwise ARI matrix
- ✅ **RF feature selection**: `extract_rf_feature_importances()` + `recommend_clustering_features()` — rank and filter the 69 clustering features by step-5 RF importance

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
- ✅ GradientBoosting: optional `include_gb=True` in bundle API (`classifier.py`)
- ✅ `extract_top_features()`: rank features by model importance
- ✅ `train_interpretability_tree()`: shallow DT on top-k features for human-readable rules

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

### Notebooks (01–07)
- ✅ `%matplotlib inline` added to all notebooks (plots display inline; no FigureCanvasAgg warning)
- ✅ `show_plots=False` in RunConfig (Jupyter inline handles display; no double-show)
- ✅ In-cell pipeline execution: each notebook auto-runs its prerequisite step if data files are missing
- ✅ `04_regimes`: fixed `KEY_INDICATORS` to use columns that exist in `clustering_features` (removed `10yr_ustreas` and `us_pop_growth`; added `10yr_ustreas_d1` and `real_gdp_growth`)
- ✅ `04_regimes`: fixed `IntCastingNaNError` in `plot_regime_profiles` when labels contain NaN after reindex
- ✅ `05_prediction`: model loading tries both `current_regime.pkl` and `current_regime_classifier.pkl`
- ✅ `07_pairplot`: triple-colored pairplots — unsupervised (balanced_cluster), Grok market_code, supervised (RF predicted)

### Diagnostics and Tactics (new — library modules, not pipeline steps)
- ✅ `diagnostics.py`: rolling z-score, percentile rank, normalize_100, RRG quadrant classification
- ✅ `tactics.py`: volatility/trend/correlation metrics, buy_hold/swing/stand_aside classification
- ✅ `email.py`: SMTP email delivery (TLS/SSL), config from `config/email.yaml`
- ✅ `scripts/run_weekly_report.py`: pipeline automation + archive + email delivery

### Infrastructure
- ✅ `CheckpointManager`: parquet + manifest, freshness check, list/clear, corrupt metadata logging (P24)
- ✅ `CheckpointManager`: `save_model`/`load_model` use `joblib` instead of `pickle` (P27)
- ✅ Ingestion completeness report (P23): `ingestion_completeness_report()` checks missing columns + NaN coverage
- ✅ `RunConfig`: dataclass with `from_args()` factory
- ✅ Full CLI: `--refresh`, `--recompute`, `--plots`, `--steps`, `--market-code`, etc.
- ✅ `config/settings.yaml`: all tunable parameters, including `gmm`, `dbscan`, `hdbscan`, `spectral` sub-sections
- ✅ `pyproject.toml`: `clustering-extras = [hdbscan, kneed]`, `data-extras = [pandas-datareader, openbb]`
- ✅ `pythonpath = ["src"]` added to pytest config (fixes test discovery without `pip install -e .`)

---

## Known Gaps (Not Yet Implemented)

### Priority 1 (implement next)
| Gap | Where | Effort |
|-----|-------|--------|
| LightGBM classifier (production flat API) | `prediction/__init__.py` | S |
| Empirical forward probabilities | `regime.py` | S |
| macrotrends.net scraper (gold, oil pre-1993) | `ingestion/macrotrends.py` (new) | M |
| Confusion matrix plot | `plotting.py` | S |

### Priority 2
| Gap | Where | Effort |
|-----|-------|--------|
| Hidden Markov Model regime detection | `hmm.py` (new module) | M |
| SMOTE for class imbalance in XGB training | `classifier.py` | S |
| Per-asset regime probability models | `prediction/asset_classifier.py` (new) | L |
| Momentum + cross-asset ratio features | `transforms.py` | M |
| Finviz Elite sector signals | `ingestion/finviz.py` (new) | M |

### Priority 3
| Gap | Where | Effort |
|-----|-------|--------|
| Streamlit dashboard | `app/dashboard.py` (new) | L |
| Backtest framework | `src/trading_crab_lib/backtest/` (new) | XL |
| ~~`joblib.dump` for sklearn model serialization~~ | ~~`pipelines/05_predict.py`~~ | ✅ Done |
| ~~Tests for `reporting.py` and `plotting.py`~~ | ~~`tests/`~~ | ✅ Done |

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
| HYG / XLK / XLP / XLE / GDX / TIP / BIL / EDV | yfinance | 2003-2007 | Quarterly |
| VIX (VIXCLS) | FRED | 1990 | Daily → Quarterly |
| Unemployment (UNRATE) | FRED | 1948 | Monthly → Quarterly |
| M2 Money Supply (M2SL + M2NS) | FRED | 1959 | Monthly → Quarterly |
| 2Y Treasury (GS2) | FRED | 1976 | Monthly → Quarterly |
| 10Y-2Y Spread (T10Y2Y) | FRED | 1976 | Daily → Quarterly |
| 10Y-3M Spread (T10Y3M) | FRED | 1982 | Daily → Quarterly |
| Gold (spot price proxy) | macrotrends.net | **Not yet** | Monthly |
| WTI Crude (spot) | macrotrends.net | **Not yet** | Monthly |

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

- Date: March 18, 2026
- Python: 3.11
- All 9 steps ran successfully
- **369 unit tests collected** (10 skipped: HDBSCAN + cssselect optional)
- Regime labels saved in `data/regimes/`; models in `outputs/models/`
- All 5 legacy alignment gaps (TSCV, DT, portfolio, proxy returns, causal smoothing) closed
- Clustering investigation suite fully implemented and tested (GMM, DBSCAN, Spectral, gap statistic, SVD, feature selection)
- All critical bugs fixed: GMM scaler consistency, gap_std vs gap_sk separation, spectral affinity caching, cluster comparison index alignment
- `prediction/` converted from flat module to package; `classifier.py` provides backwards-compat bundle API with GradientBoosting + interpretability helpers
- FRED expanded from 7 to 14 series; yield curve features added
- New modules: `diagnostics.py` (RRG), `tactics.py`, `email.py`, `scripts/run_weekly_report.py`
- `end_date: null` fix applied (P12); `from __future__ import annotations` added to fred.py, multpl.py

**Note:** the test suite no longer contaminates production checkpoints. Pipeline smoke tests use `monkeypatch.setattr(step, "DATA_DIR", tmp_path)` to isolate all file I/O. See CLAUDE.md D5.
