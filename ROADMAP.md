# Trading-Crab — Product Roadmap

Prioritized backlog of features, data sources, and improvements.
Updated: July 2026.

---

## Tier 0 — Platform Redesign (north star)

**`platform_design/platform_design.md` (v1.7) is now the authoritative target** and supersedes
the ad-hoc design behind Tiers 1–3 below. It reframes the project from "improve the quarterly
clustering pipeline" into a monthly, walk-forward-honest, 5-layer regime-conditional allocation
platform (L0 data → L1 labeling → L2 prediction → L3 asset MoE → L4 allocation). Verdict from
§13: **enhance, don't restart** — keep the chassis (config, checkpoints, lib/pipeline split,
tests, email report); rebuild the modeling core.

Execution backbone lives in the design doc, not here — do not re-transcribe:
- **Phases 0–6** — design §14 (tracer-bullet skeleton first, honesty framework before any tuning).
- **R1–R15** — design §11 re-eval checklist; re-build order R6/R13 → R1 → R2/R11/R14 → R10/R12 → R8 → R9 → L3 → L4.

Tiers 1–3 below are re-scoped against this: items become **near-term (still valid)**,
**warm-start / baseline / diagnostic** (kept but demoted from the spine), or **superseded**.
Full reconciliation is deferred to the GSD planning pass (see "GSD planning" note at bottom).

**Salvaged modules to fold in** (from `ideas/gsd-salvage/`, extracted from `gsd-scratch-work`):
- `feature_gating.py` → **Phase 0/1**: enforces causal (`features_supervised`) selection for L2 training; operationalizes P1 / design R5 / §8.2. Highest value.
- `model_metrics_artifacts.py` → **Phase 1+**: Brier / calibration / confusion artifacts = design §8 honesty KPIs.
- `dashboard_model.py` → **L4/reporting**: RF-vs-GB model-path resolution; minor, fold in when a GB dashboard model is saved.

---

## Tier 0 — Platform Build Backlog

The concrete items behind Tier 0, in the design's §11 re-build order (dependency-correct).
Each maps to a design section/R-number and a §14 phase. This is the backlog GSD planning will
turn into `.planning/` phases; it is not yet decomposed into per-file plans.

### T0.1  Phase-0 honesty infrastructure  `L`  (§8, §14 Phase 0 — **do first**)
The design's stated prerequisite to *any* modeling — none of it exists today.
- **2021+ holdout carve** — physically separate files/paths the pipeline cannot read by default; evaluated once at design freeze (`S`).
- **Trial registry** — flat-file/SQLite `config-hash → metrics` store; the multiple-testing denominator for a deflated Sharpe (`S`/`M`).
- **Walk-forward runner** — core infra: at each rebalance refit L1 labels + L2/L3 on data ≤ t, record decisions, step forward. Report the smoothed-vs-filtered gap (`L`).
- `model_metrics_artifacts.py` (salvaged) = down payment on the KPI half (Brier / calibration / confusion).

### T0.2  Monthly data spine  `L`  (R1, §9, §14 Phase 0)
Move ingestion + transforms from quarterly to **monthly**; keep quarterly agency series with proper alignment. Quarterly starves the labeler and ~4× the detection lag in calendar time. Foundational — everything downstream inherits the frequency.

### T0.3  1962+ spliced histories + ALFRED vintages  `XL`  (R13/R6, §9)
- Splice ETF prices with index/futures/spot back to ~1962 (SPY←S&P TR, GLD←gold spot, TLT←CMT synthetics…); tradability applies to the present, not the history.
- Replace revised FRED with **ALFRED point-in-time vintages** + publication-lag alignment; reclassify features into fast/slow/agency taxonomy (market-observed preferred for labeling).

### T0.4  Jump-model labeler (L1)  `L`  (R2/R11/R14, §4)
k-means **+ per-jump penalty λ** (Bemporad–Boyd), exact DP decode; existing balanced k-means = warm start, HMM (`hmm.py`) = the Student-t benchmark. Occupancy floor/cap + sojourn as **acceptance criteria** (§4.4), replacing forced-balance clustering. Recast naming heuristics as skeleton sign constraints.

### T0.5  Purged & embargoed CV  `M`  (R7, §6.5)
Add purging + embargo to all supervised CV; current `TimeSeriesSplit(5)` leaks through overlapping h-month labels. Report transition-window metrics (±3m) separately from overall accuracy.

### T0.6  L2 split: nowcaster + transition model  `M`  (R8, §5)
Split the single classifier into a **nowcaster** (recursive prior-state feature, γ sample weights, calibration) and a **transition model** (spreads/vol/**regime age** features). Keep the RF/DT/GBM zoo; the problem framing splits, not the models.

### T0.7  Volatility & covariance layer (L3 half)  `L`  (R9, §6.2, §7)
GARCH(1,1)/EWMA per asset; **vol targeting** overlay (size ∝ 1/σ̂); regime-conditional covariance with Ledoit–Wolf shrinkage. Replaces `tactics.py`'s fixed vol/trend thresholds; feeds model-driven vol-scaled stops (§27 policy stack).

### T0.8  Wire in `feature_gating.py` (causal-feature guard)  `S`  (R5, §8.2, salvaged)
Enforce `features_supervised.parquet` (causal) for L2 training; `--allow-noncausal-features` opt-in falls back with a loud warning. Cheap, and it locks in the L1-may-see-future / L2-may-not invariant the whole design rests on. Do early alongside T0.1.

---

## Phase Progress

| Phase | Name                                          | Plans Complete | Status      | Completed  |
|-------|-----------------------------------------------|----------------|-------------|------------|
| 1     | Data and Constraints Foundations              | 3/3            | Complete    | -          |
| 2     | Regime Clustering and Interpretation          | 2/2            | Complete    | 2026-03-16 |
| 3     | Supervised Regime and Behavior Models         | 3/3            | Complete    | 2026-03-18 |
| 4     | Pipeline Monitoring & Notebook Expansion      | All done (A-E) | Complete    | 2026-03-27 |
| 5     | Audit & Hardening (META_PLAN P1–P5)           | P1–P5 done     | In progress | 2026-03-31 |
| 6     | Test Hardening + CI/CD (META_PLAN H, J)       | —              | Not started | -          |
| 7     | Migration Prep + Advanced Features (K, F)     | —              | Not started | -          |

### Phase 1 Plans (3 plans — all complete)
- [x] `01-null-01-PLAN.md` — Data ingestion foundations and checkpoint system
- [x] `01-null-02-PLAN.md` — Feature engineering pipeline (transforms, gap fill, derivatives)
- [x] `01-null-03-PLAN.md` — Clustering investigation suite (GMM, DBSCAN, Spectral, gap statistic)

### Phase 2 Plans (2 plans — all complete)
- [x] `02-regime-clustering-interpretation-01-PLAN.md` — Regime profiling and transition matrix
- [x] `02-regime-clustering-interpretation-02-PLAN.md` — Regime naming heuristics and label overrides

### Phase 3 Plans (3 plans — all complete)
- [x] `03-supervised-regime-behavior-models-01-PLAN.md` — Classifier module scaffold + test modules (functionally complete; APIs in flat + bundle API)
- [x] `03-supervised-regime-behavior-models-02-PLAN.md` — Supervised current-regime classifier (RF + DT)
- [x] `03-supervised-regime-behavior-models-03-PLAN.md` — Forward binary classifiers and behavior models

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
- New file: `src/trading_crab_lib/prediction/gradient_boosting.py`
- Functions: `train_lightgbm_current_regime()`, `train_lightgbm_forward()`
- Use same `_tscv_scores()` helper as RF + DT
- Do NOT over-tune hyperparameters with 300 obs (fixed grid, max 50 combos)
- Add `lightgbm>=4.0` as optional extra in `pyproject.toml`
- **Files**: `src/trading_crab_lib/prediction/gradient_boosting.py` (new), `pipelines/05_predict.py`

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
- **Files**: `config/settings.yaml`, `src/trading_crab_lib/ingestion/fred.py`

### 1.3  Yield curve features  `S`
Compute derived yield-curve features in `transforms.py`:
- `yield_spread_10y2y` = GS10 − GS2 (add GS2 to FRED series)
- `yield_spread_10y3m` = GS10 − TB3MS (already have both)
- `yield_curve_slope` = (GS10 − TB3MS) / 10
- These are among the strongest empirical recession predictors in the literature
- **Files**: `src/trading_crab_lib/transforms.py`, `config/settings.yaml`

### 1.4  Empirical forward probabilities  `S`
Implement `compute_forward_probabilities()` from `legacy/unified_script.py`.
Computes empirical P(reach regime j within N quarters | currently in regime i)
as a diagnostic alongside model-based forward classifiers.
- Output: `data/regimes/forward_probs_{N}q.parquet` for N in [1, 4, 8]
- **Files**: `src/trading_crab_lib/regime.py`, `pipelines/04_regime_label.py`

### 1.5  macrotrends.net historical price backfill  `M`  ✅ DONE
Extends commodity and asset data before 1993 (ETF inception dates):
- **Gold price**: monthly back to 1915 (`https://www.macrotrends.net/1333/historical-gold-prices-100-year-chart`)
- **WTI Crude Oil**: monthly back to 1946
- ✅ Scraper implemented in `src/trading_crab_lib/ingestion/macrotrends.py` (242 lines, 9 tests)
- ✅ Wired into step 1 via `config/settings.yaml` macrotrends section
- ✅ `log_gold_spot_d1/d2` and `log_wti_crude_d1/d2` added to `clustering_features`
- ✅ Gold/oil divergence pairs (`spy_gld`, `gld_oil`) auto-activate when data present
- **Files**: `src/trading_crab_lib/ingestion/macrotrends.py`, `config/settings.yaml`

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
  `notebooks/06_assets.ipynb`, `src/trading_crab_lib/plotting.py`
- **Status**: ✓ Done (settings.yaml + notebooks updated; ETF data fetched on next step 1 run)
- **Update (D20)**: ETF price ingestion moved from step 6 to step 1. A curated subset
  (SPY, TLT, GLD, QQQ, VNQ) is merged into `macro_raw` as `etf_{ticker}` columns.
  Their log-price derivatives are available in `initial_features` for supervised learning.
  Gold/oil derivatives from macrotrends are in `clustering_features` (deep enough history).

### 1.7  Confusion matrix and classification report in plots  `S`
`legacy/supervised.py` has `generate_classification_report()` that produces a
confusion matrix; this is not exposed in `src/` plotting or logs.
- Add `plot_confusion_matrix(model, X, y, regime_names, run_cfg)` to `plotting.py`
- Call from `pipelines/05_predict.py` when `--plots` is set
- **Files**: `src/trading_crab_lib/plotting.py`, `pipelines/05_predict.py`

---

## Tier 2 — High Value, More Effort

### 2.8  Finviz Elite integration for sector/stock signals  `M`
With a Finviz Elite subscription:
- Use `finvizfinance` Python library (`pip install finvizfinance`)
- Screener API: pull all S&P 500 stocks filtered by sector, market cap, momentum
- Quarterly sector aggregation: for each regime, which sectors (XLK, XLF, XLE, etc.) outperform?
- Useful for "within-regime" stock picking after portfolio ETF allocation is set
- **Note**: Finviz data is point-in-time; historical screener data requires Elite API
- Separate from regime detection (which is macro-driven); feeds into a "stock signal" layer
- **Files**: `src/trading_crab_lib/ingestion/finviz.py` (new), `pipelines/08_stock_signals.py` (new)

### 2.9  Hidden Markov Model regime detection (alternative to KMeans)  `M` ✅ DONE
**Implementation**: `src/trading_crab_lib/hmm.py` (D22).
- `fit_hmm()`: GaussianHMM sweep across k with best-of-N restarts, BIC/AIC scoring
- `select_hmm_k()`: BIC-based model selection
- `hmm_labels()`: Viterbi-decoded hard state assignments (canonicalized)
- `hmm_probabilities()`: forward-backward posterior probabilities
- `hmm_transition_matrix()`: learned state transition matrix
- 19 tests in `tests/unit/test_hmm.py`. Library module only — not yet in pipeline.
- **Next**: integrate into `pipelines/03_cluster.py` and compare with KMeans

### 2.10  SMOTE / class-weight tuning for imbalanced regimes  `S`
With 5 balanced clusters, sizes should be equal, but temporal distribution may still
cause class imbalance in train/test splits of the TSCV folds.
- RF already uses `class_weight="balanced"` — log per-fold class counts to verify
- Consider `imbalanced-learn` SMOTE for XGBoost (which doesn't have class_weight)
- Add to `pyproject.toml` as optional extra: `imbalanced-learn>=0.11`
- **Files**: `src/trading_crab_lib/prediction/classifier.py`

### 2.11  Per-asset regime probability models  `L`
For each ETF (SPY, GLD, TLT, USO, QQQ, IWM, VNQ, AGG), train per-asset models:
- Binary: "Will this ETF be +X% in Y quarters?" for X in [5, 10, 20] and Y in [1, 2, 4, 8]
- Features: regime probabilities + causal macro features + asset momentum
- Output: per-asset stoplight probability matrix → feeds dashboard signal layer
- This is "Putting it all together — Part I" from the original design doc
- **Files**: `src/trading_crab_lib/prediction/asset_classifier.py` (new), `pipelines/05b_asset_predict.py` (new)

### 2.12  Momentum and cross-asset ratio features  `M` ✅ DONE
**Implementation**: `src/trading_crab_lib/momentum.py` (Phases A+B, D18).
- Trailing 2Q/4Q/8Q returns for sp500, sp500_adj, 10yr_ustreas, credit_spread
- Relative strength: S&P-in-Gold, S&P-in-Oil, Gold-in-Oil (activate when macrotrends data available)
- Rolling 8Q equity-bond correlation (corr_sp500_10yr_ustreas_8q)
- CPI acceleration (2nd derivative)

**Phase C+D** (D21): Added 6 momentum features to `initial_features`, 11 to
`clustering_features` (all 1950+, safe for clustering). Evaluation script:
`scripts/evaluate_momentum.py`. 20 tests in `tests/unit/test_evaluate_momentum.py`.
Fixed `fred_vixcls` → `fred_vix` column name bug.

### 2.13  Markov regime-switching model (statsmodels)  `M` ✅ DONE
**Implementation**: `src/trading_crab_lib/markov.py` (D22).
- `fit_markov_switching()`: switching-mean model on univariate series (GDP growth)
- `markov_labels()`: hard 2-state assignments (regime 0 = lower mean = recession)
- `markov_probabilities()`: smoothed marginal probabilities
- `compare_markov_kmeans()`: cross-tabulate Markov vs KMeans regimes
- 18 tests in `tests/unit/test_markov.py`. Library module — diagnostic use from notebooks.
- **Next**: use recession probability as a supervised feature; compare with KMeans regimes

### 2.15  Cross-asset divergence features for regime change detection  `L`
Derive new supervised learning features from **divergences between historically correlated
signals** (or their derivatives). When correlations that have held for years suddenly break
(e.g., S&P500 and TLT move together instead of inversely; gold and oil decouple), these
divergences can signal regime transitions and trading opportunities.

**Core idea — three layers:**

1. **Rolling correlation baseline** — For each signal pair (SPY/TLT, GLD/USO, SPY/GLD,
   10Y yield/S&P, credit_spread/VIX, etc.), compute rolling Pearson or Spearman correlation
   over a trailing window (8Q, 12Q). This is the "expected" relationship.

2. **Divergence magnitude** — At each quarter, measure how far the current correlation
   departs from its trailing mean. Also measure divergence in **derivative space**: if
   d1(SPY) and d1(TLT) normally move inversely (corr ~ −0.5), and suddenly both spike up
   (corr ~ +0.5), the **divergence magnitude = |actual_corr − expected_corr|** is ~1.0.
   Differential equations angle: if dS/dt and dT/dt satisfy a quasi-linear relationship
   in normal times, deviations from that relationship are regime-change evidence.

3. **Divergence trigger features** — Binary or continuous features for the supervised model:
   - `div_spy_tlt_8q`: rolling correlation divergence magnitude (continuous)
   - `div_spy_tlt_trigger`: 1 if divergence exceeds 2σ of its own history, 0 otherwise (binary)
   - `div_spy_tlt_direction`: +1 if correlation increased, −1 if decreased (categorical)
   - `div_d1_spy_d1_gld_8q`: divergence in derivative space (captures leading indicators)

**Signal pairs to start with (high macro significance):**
- SPY vs TLT (equity-bond: breaks during stagflation and liquidity crises)
- SPY vs GLD (equity-gold: breaks during inflation fears)
- GLD vs USO (gold-oil: breaks when gold is safe-haven vs commodity)
- 10Y yield vs S&P 500 (rate-equity: breaks during Fed pivots)
- Credit spread (BAA-AAA) vs VIX (credit risk vs equity vol)
- CPI derivative vs M2 derivative (money supply → inflation pipeline)

**Why this is valuable:**
- Current clustering treats each quarter independently; divergence features inject
  **relational dynamics** that capture "something is changing" before the change manifests
  in level data.
- Leading indicator potential: derivative divergences may lead level changes by 1-2 quarters.
- Natural fit for the existing pipeline: compute in `transforms.py` after cross-ratios,
  add to `clustering_features` or `supervised_features` lists in settings.yaml.

**Implementation phases:**
- ✅ Phase A: `src/trading_crab_lib/divergence.py` — `compute_rolling_correlation()`,
  `compute_divergence()`, `compute_divergence_triggers()`, `compute_derivative_divergence()`,
  `add_divergence_features()` (~250 lines, 29 tests)
- ✅ Phase B: Hooked into `transforms.py` `engineer_all()` in two places: level-space
  divergence after momentum features, derivative-space after derivatives computed
- ✅ Phase C: Added spy_tlt + cred_vix divergence features (z-scores, triggers, d1 derivatives,
  derivative-space z-scores + triggers) to `initial_features` and `clustering_features` in
  `settings.yaml`. Fixed `fred_vixcls` → `fred_vix` column name bug. Added `sp500` raw level
  to `initial_features` for derivative-space divergence support.
- ✅ Phase D: Evaluation complete (`scripts/evaluate_divergence.py`). Results:
  - **Clustering quality improved**: silhouette +0.032 (0.189→0.221), CH +6.8, DB −0.10
  - **K-sweep**: divergence features improve silhouette at k=2–6; largest gain at k=5 (+0.046)
  - **Supervised accuracy**: slight degradation (−0.018 mean CV, within noise). RF feature
    importance ranks `div_cred_vix_z_4q_d1` 5th out of 80 features — divergence features
    carry signal but more samples or feature selection may be needed to improve CV scores
  - **Transition detection**: `spy_tlt` z-score is 36% higher at transitions (0.92) vs
    baseline (0.67) — confirmed as a leading indicator of regime change. Other pairs inconclusive
  - **Recommendation**: keep divergence features in clustering pipeline (clear improvement).
    For supervised pipeline, consider (a) feature selection to prune noisy divergence columns,
    (b) waiting for gold/oil data (macrotrends) to activate spy_gld and gld_oil pairs
- **Files**: `src/trading_crab_lib/divergence.py`, `src/trading_crab_lib/transforms.py`,
  `config/settings.yaml`, `scripts/evaluate_divergence.py`

### 2.14  Conference Board LEI proxy from FRED  `S`
The Conference Board LEI is the gold standard for recession prediction but is not
freely available. Construct a proxy from FRED components:
- `PERMIT` (building permits) + `AWHMAN` (avg weekly hours) + `AMDMNO` (new orders)
  + `ISM manufacturing` + `UMCSENT` + spread measures = 6-component LEI approximation
- Validate against NBER recession dates (`USREC` on FRED — binary recession indicator)
- **Files**: `src/trading_crab_lib/transforms.py`, `config/settings.yaml`

---

## Phase 4 — Pipeline Monitoring & Notebook Expansion

**Plan document:** `MONITORING_EXPANSION_PLAN.md` (85 work items across 17 phases)

| Phase | Items | Summary |
|-------|-------|---------|
| A1–A5 | 25 | New plotting functions: model eval, PCA, time-series, diagnostics, feature engineering |
| B | 3 | Plot reuse infrastructure — notebooks load pre-generated PNGs |
| C1–C4 | 20 | Pipeline monitoring — validation summaries + diagnostic plots + QA gates |
| D1–D10 | 34 | Notebook expansion — 8 existing + 4 new notebooks (diagnostics, model comparison, feature selection, divergence/momentum) |
| E | 3 | Email plot attachments — key plots embedded in weekly email |

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
- **Files**: `src/trading_crab_lib/prediction/factor_model.py` (new)

### 3.5  Backtest framework  `XL`
Walk-forward backtest of the full pipeline:
- At each quarter T, train on [T-N, T], predict regime and portfolio for T+1
- Compare strategy vs S&P 500 benchmark: returns, Sharpe, max drawdown
- Avoids look-ahead by construction (causal features + TSCV)
- Requires ~50 walk-forward steps (1975–2025 at quarterly resolution)
- **Files**: `src/trading_crab_lib/backtest/` (new module)

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
- **Files**: `src/trading_crab_lib/ingestion/stockcharts.py` (new)

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
- **Files**: `src/trading_crab_lib/ingestion/finviz.py` (new), `pipelines/08_stock_signals.py` (new)
- **Note**: historical screener data requires Finviz Elite API; current data is available
  via the `finvizfinance` library without authentication for many fields

---

## Data Sources Master Table

| Source | Library/Approach | What We Get | Back to | In Pipeline? | Priority |
|--------|-----------------|-------------|---------|-------------|----------|
| multpl.com | lxml scraper | 46 Shiller series | varies | ✓ Step 1 | Done |
| FRED API | `fredapi` | GDP, CPI, BAA, AAA, GS10, TB3MS, GNP | varies | ✓ Step 1 | Done |
| yfinance | `yfinance` | ETF OHLCV (SPY, GLD, TLT, USO, QQQ, IWM, VNQ, AGG) | 1993+ | ✓ Step 1+6 | Done |
| FRED — VIX | `fredapi` | VIXCLS daily volatility index | 1990 | ✓ Step 1 | Done |
| FRED — unemployment | `fredapi` | UNRATE monthly | 1948 | ✓ Step 1 | Done |
| FRED — M2 | `fredapi` | M2SL + M2NS money supply | 1959 | ✓ Step 1 | Done |
| FRED — yield spreads | `fredapi` | T10Y2Y, T10Y3M, GS2 | varies | ✓ Step 1 | Done |
| FRED — housing | `fredapi` | HOUST | 1959 | ✓ Step 1 | Done |
| FRED — consumer | `fredapi` | UMCSENT | 1952 | ✓ Step 1 | Done |
| FRED — industrial | `fredapi` | INDPRO, PAYEMS, DPCERA3Q086SBEA | varies | ✗ | **Tier 1** |
| macrotrends.net | custom scraper | Gold, oil, silver prices | 1915+ | ✓ Step 1 | Done |
| stooq.pl | `pandas-datareader` | Free ETF/stock OHLCV (Phase 3 yfinance fallback) | ~1993 | ✓ Phase 3 | Done (optional install) |
| OpenBB | `openbb` | Multi-provider ETF prices (Phase 4 yfinance fallback) | varies | ✓ Phase 4 | Done (optional install) |
| Finviz Elite | `finvizfinance` | Sector screener + fundamentals (NOT historical prices) | recent | ✗ | Tier 3 (3.7) |
| StockCharts.com | custom scraper | Chart data + technical indicators | varies | ✗ | Tier 3 (3.6) |
| hmmlearn | Python lib | HMM regime states | n/a | ✗ | Tier 2 (2.9) |
| statsmodels | Python lib | Markov regime-switching | n/a | ✗ | Tier 2 (2.13) |
| sklearn GMM | Python lib | Gaussian Mixture Models (soft clusters) | n/a | ✓ Investigation suite | Done |
| sklearn SpectralClustering | Python lib | Spectral / graph clustering | n/a | ✓ Investigation suite | Done |
| hdbscan | Python lib | Density-based clustering (HDBSCAN) | n/a | ✓ Investigation suite (optional) | Done |
| Streamlit | Python lib | Interactive dashboard | n/a | ✗ | Tier 3 |
| Claude API | `anthropic` | AI weekly narrative | n/a | ✗ | Tier 3 |
| StockCharts | scrape | Historical OHLCV + technical indicators | varies | ✗ | Tier 3 (3.6) |

---

## What to Do This Session (Suggested Starting Points)

**Already completed (prior sessions):**
- ~~Add FRED series (VIX, unemployment, M2, yield spreads)~~ → ✅ 14 FRED series now
- ~~Add yield curve features~~ → ✅ 10Y-2Y and 10Y-3M spreads
- ~~Add `compute_forward_probabilities()`~~ → ✅ in `regime.py`
- ~~Add `plot_confusion_matrix()`~~ → ✅ in `plotting.py`
- ~~Package rename to `trading_crab_lib`~~ → ✅ complete
- ~~Expand ETF universe to 38~~ → ✅ in settings.yaml
- ~~HOUST + UMCSENT FRED series~~ → ✅ added
- ~~Non-determinism fix (market_code in gap-fill/derivatives)~~ → ✅ META_PLAN P2 done
- ~~Pytest warnings (statsmodels RuntimeWarning)~~ → ✅ META_PLAN P3 done
- ~~Email Diagnostics section + HTML rendering~~ → ✅ META_PLAN P4 done

**Next priorities (April 2026):**
1. **Phase H — Test hardening** — smoke tests for `trading_crab.pipeline` + `trading_crab.cli`; integration test in `tests/integration/`
2. **Phase J — CI/CD cleanup** — deduplicate 6 workflow files → 3; add mypy; add pre-commit hooks
3. **Phase K — Migration prep** — config independence for library; Dockerfile; settings.yaml schema validation
4. Add remaining FRED series (INDPRO, PAYEMS, DPCERA3Q086SBEA) — `S`, quick config additions
5. Backtest framework (item 3.5) — `XL`, validates the full strategy end-to-end

---

## GSD planning (how the long-term plan gets built)

This ROADMAP is the interim plan-of-record. The durable long-term plan will be generated
**fresh** with GSD (installed in `.claude/`) — no `.planning/` copied from `gsd-scratch-work`.

Recommended kickoff (each step is a Claude Code session on Fable):
1. `/gsd-map-codebase` — analyze the existing repo into `.planning/codebase/` (tech, arch, quality).
2. `/gsd-new-project` — deep context gathering → `.planning/PROJECT.md` + roadmap, feeding it
   `platform_design/platform_design.md` as the design input. Adopt design §14 Phases 0–6 as the
   milestone spine; map §11 R1–R15 and the Tier-0 salvage items into phases.
3. Per phase: `/gsd-plan-phase` → `/gsd-execute-phase` → `/gsd-verify-work` → `/gsd-ship`.

Phase 0 (honesty framework: holdout carve, trial registry, walk-forward runner) comes **before**
any modeling — that ordering is the whole point of the design.
