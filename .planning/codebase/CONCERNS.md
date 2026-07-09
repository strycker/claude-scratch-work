<!-- refreshed: 2026-07-09 -->
# Codebase Concerns

**Analysis Date:** 2026-07-09

---

## Tech Debt

### Architecture & Design Gaps (from platform_design.md §11, R1-R15)

**R1 — Quarterly data spine instead of monthly** (BLOCKING REDESIGN ITEM)
- **Issue:** Data frequency is hardcoded to `frequency: "Q"` in `config/settings.yaml`. Design calls for monthly spine (D10) with weekly scoring overlay.
- **Files:** `config/settings.yaml`, `src/trading_crab_lib/ingestion/fred.py`, `src/trading_crab_lib/ingestion/multpl.py`, `src/trading_crab_lib/ingestion/assets.py`
- **Impact:** ~260 quarterly observations from 1950 vs ~800 monthly observations. Quadruples detection lag in calendar time for regime transitions. Labeler (L1) starved of signal.
- **Fix approach:** Migrate ingestion to monthly; keep quarterly agency series (GDP/GNP) with proper alignment via publication-lag shift; resample ETF prices to monthly. Estimated effort: 2 phases (data layer + feature engineering).

**R2 — Forced-balance clustering instead of jump-model labeling** (MODELING DEBT)
- **Issue:** Current: `KMeansConstrained(size_min=budget-2, size_max=budget+2)` produces `balanced_k=5` clusters of artificially equal occupancy. Design prefers: jump-model (k-means + jump penalty with DP solve) using occupancy floor/cap as acceptance criteria (not forcing), plus temporal persistence via penalty term.
- **Files:** `src/trading_crab_lib/clustering.py` (lines 380-440 for `fit_clusters()`)
- **Impact:** Distorts cluster geometry near occupancy-constrained boundaries. No native temporal persistence — regimes appear i.i.d. rather than temporally sticky. Current k-means serves as warm start for future jump-model implementation.
- **Fix approach:** Implement jump-model solver as drop-in replacement; keep k-means for initialization and baseline diagnostics. No code change in downstream steps. Effort: 1 phase (labeler upgrade).

**R3 — Constant transition matrix instead of feature-conditional transitions** (MODELING DEBT)
- **Issue:** Current: `build_transition_matrix()` computes empirical 1-step transition probabilities (constant across time). Design requires: TVTP-style transitions P(S_{t+1}=j | S_t=i, z_t, age_t) with features (credit spreads, vol, etc.) + regime age as predictors.
- **Files:** `src/trading_crab_lib/regime.py` (lines 80-120 for `build_transition_matrix()`)
- **Impact:** Transition model (L2) cannot distinguish pre-crisis vs post-crisis likelihood. Nowcast + transition model are conflated in step-5 supervisedmodels.
- **Fix approach:** Add transition-model training in prediction/__init__.py; keep empirical matrix as diagnostic baseline.

**R4 — PCA(5) before clustering obscures semantics** (ANALYSIS DEBT)
- **Issue:** Current: `reduce_pca()` projects 69 clustering features onto 5 PCA components before KMeans. This loses interpretability and mixes signal dimensions (e.g., PC1 is a soup of all series' leading edges).
- **Files:** `src/trading_crab_lib/clustering.py` (lines 50-80 for `reduce_pca()`)
- **Impact:** Regime profiles are in PCA space, not original feature space — makes interpretation harder. Semantic skeleton constraints (D3) cannot be applied directly.
- **Fix approach:** Keep PCA as diagnostic (notebook 03); use standardized curated features directly, or remove PCA pre-clustering. Will require cluster geometry re-validation.

**R5 — Non-stationary levels in clustering feature set** (VERIFICATION GAP)
- **Issue:** Design §9 forbids non-stationary raw levels (rates, log-prices, etc.) from reaching the labeler. Current `clustering_features` in `config/settings.yaml` may include harmful levels. Audit required.
- **Files:** `config/settings.yaml` (lines 140-180), `src/trading_crab_lib/transforms.py` (engineer_all function)
- **Impact:** If raw levels leak into clustering, cointegration / spurious regression can appear as regimes.
- **Fix approach:** Audit final clustering matrix after step 2; enforce stationarity check (ADF test or ACF decay) before step 3. Add pre-clustering validation step.

**R6 — FRED data revised (no ALFRED vintage alignment)** (DATA INTEGRITY DEBT)
- **Issue:** Current ingestion fetches live FRED series via `fredapi.Fred()` without vintage tracking. FRED data revises continuously. Supervised models trained on Q1 data retrain in Q2 with Q1 "final" revision, leaking revision surprises into labels.
- **Files:** `src/trading_crab_lib/ingestion/fred.py` (entire fetch_all function)
- **Impact:** Revised data leakage violates D12 (point-in-time discipline). Impacts model reproducibility and real-time scoring.
- **Fix approach:** Migrate to ALFRED (Archival FRED) API which preserves vintage dates. Reclassify features into fast/slow/agency taxonomy (D12). Estimated effort: 1 phase (data layer).

**R7 — TimeSeriesSplit CV without purging/embargo** (LOOK-AHEAD BIAS)
- **Issue:** Current: `TimeSeriesSplit(n_splits=5)` is used (good), but no purging of overlapping-label boundaries or embargo of trailing-uncertainty label windows. Forward classifiers (h ∈ {1,2,4,8} quarters) create 92%+ overlapping labels; CV folds can leak future information.
- **Files:** `src/trading_crab_lib/prediction/__init__.py` (lines 150-200 for train_current_regime), `src/trading_crab_lib/monitoring/prediction.py` (compute_cv_fold_scores)
- **Impact:** CV accuracy inflated; real-time scoring worse than backtest. Transition-window detection metrics are optimistic.
- **Fix approach:** Add purged CV (López de Prado ch.7–8): remove training data within h periods of each test fold start; embargo last 6–12 months of labels (unstable due to two-sided smoothing). Report transition-window metrics separately from overall accuracy.

**R8 — Conflated nowcast and transition models** (MODELING DEBT)
- **Issue:** Current: Single RF/DT trained on regime labels predicts both "what regime now" and transitions implicitly. Design requires separate nowcaster (recursive state feature, γ sample weights from L1 confidence) and transition model (features: spreads, vol, age).
- **Files:** `src/trading_crab_lib/prediction/__init__.py` (train_current_regime, train_forward_classifiers)
- **Impact:** Transition signals (spreads, vol) and level signals (means) compete in one model. Harder to diagnose and tune separately.
- **Fix approach:** Split into two models; nowcaster includes prior P(S_t | z_{1:t-1}) as a feature (recursive). Add recursive = True parameter to prediction API.

**R9 — No volatility forecasting or regime-conditional stops** (PORTFOLIO DEBT)
- **Issue:** Current: `tactics.py` uses fixed vol/trend thresholds → buy_hold/swing/stand_aside. No GARCH/EWMA vol forecasting. Stops are not regime-conditional or vol-scaled.
- **Files:** `src/trading_crab_lib/tactics.py` (entire module), `src/trading_crab_lib/portfolio construction` logic
- **Impact:** Portfolio allocation static vs vol; regime crashes not de-risked automatically. Missing §6.2 and §7 (design).
- **Fix approach:** Add GARCH(1,1) or EWMA vol layer; implement vol targeting; add regime-conditional stop multipliers m_k·σ̂.

**R10 — No walk-forward harness; full-sample fit used** (EVALUATION DEBT)
- **Issue:** Current: Clustering fit on full data → labels → CV. No refit loop for each CV fold. Parameter lookahead: labels are filtered with future-informed parameters (PCA, gap-fill thresholds computed on 1950–2026 data).
- **Files:** `src/trading_crab/pipeline.py` (main function, lines 1-1374), `src/trading_crab_lib/clustering.py`
- **Impact:** Prevents honest walk-forward testing. Smoothed-vs-filtered gap (hindsight content) not measurable. Design §8.1 requires walk-forward as core infrastructure.
- **Fix approach:** Build walk-forward runner with refit at each time point; report smoothed-vs-filtered gap and detection lag. Report transition-window metrics separately. Implement before next model iteration.

**R11 — Gaussian mixtures instead of Student-t** (STATISTICAL DEBT)
- **Issue:** Current: `gmm.py` uses GaussianMixture. Design D7 mandates Student-t emissions everywhere (fat tails are real; extra Gaussian states waste capacity absorbing outliers like 2008/2020).
- **Files:** `src/trading_crab_lib/gmm.py` (lines 30-80 for fit_gmm)
- **Impact:** GMM over-fits with extra states to model tails. Interpretability degraded.
- **Fix approach:** Replace `GaussianMixture` with `sklearn_mixture.BayesianGaussianMixture` with Wishart priors, or use statsmodels Student-t HMM. Effort: 1 component upgrade.

**R12 — No trial registry or holdout discipline** (EVALUATION DEBT)
- **Issue:** Current: No log of which configs were evaluated, in what order. No locked holdout period (design D13: 2021-01-01+ untouched until design freeze).
- **Files:** Not applicable (missing infrastructure)
- **Impact:** Cannot compute deflated Sharpe or report multiple-testing corrections. Enables unintentional p-hacking.
- **Fix approach:** Implement trial registry (flat SQLite or JSON file) with config hash → metrics. Carve 2021+ holdout as separate data partition before next modeling iteration. Add to Phase 0 (Foundations).

**R13 — Short ETF histories, no spliced long asset prices** (DATA DEBT)
- **Issue:** Current: Labeling (L1, ~260 observations) needs 1962+ history (~30+ regime transitions). ETFs start 1993–2006. Missing spliced index/futures/spot histories (S&P TR, gold, WTI, constant-maturity Treasuries).
- **Files:** `src/trading_crab_lib/ingestion/assets.py` (entire module), config lacks splice definitions
- **Impact:** Asset-returns regression (L3) blind for 1962–1993 (~25 years, ~10 regimes). Pre-1993 regime profiles use macro proxies (log_sp500 proxy) only.
- **Fix approach:** Build spliced long histories per asset (SPY←S&P 500 TR, GLD←gold spot/futures, TLT←constant-maturity synthetic, etc.). Estimated effort: 1 phase (R13). Design §9 has source references.

**R14 — Regime naming post-hoc instead of skeleton-constrained** (DESIGN PATTERN DEBT)
- **Issue:** Current: `suggest_names()` uses median-deviation heuristics to name clusters after clustering. Design prefers: hand-specified semantic skeleton (growth/inflation 2×2 grid from D3) as *constraints* during L1 solve (or at least applied to interpret outputs).
- **Files:** `src/trading_crab_lib/regime.py` (lines 40-80 for suggest_names)
- **Impact:** Regime names can drift across runs. Skeleton enforces economic interpretability without overfitting.
- **Fix approach:** Recast suggest_names() output as skeleton seeds; add skeleton-constraint layer in L1 (could be soft or hard depending on jump-model variant used). Keeps existing code as warm start.

**R15 — Checkpointing and config infrastructure, no test coverage for pipeline.py** (TESTING DEBT)
- **Issue:** Positives: Functions-only library, caller-driven config, parquet checkpoints, CLI, ~769 tests. Negative: `src/trading_crab/pipeline.py` (1374 lines) has zero direct test coverage. Full-stack integration tests exist but no unit tests of step dispatch, flag handling, or error recovery.
- **Files:** `src/trading_crab/pipeline.py`, no corresponding `tests/test_pipeline_unit.py`
- **Impact:** Regressions in flag handling (--steps, --market-code, etc.) not caught. Pipeline changes land untested.
- **Fix approach:** Add unit test module `tests/test_pipeline_unit.py` mocking checkpoints and internal steps. Estimated effort: 0.5 phase (testing, not modeling).

---

## Known Bugs

### P22 — SSL verification disabled for price ingestion
- **What happens:** `src/trading_crab_lib/ingestion/assets.py` line 145: `curl_requests.Session(verify=False)`. Certificate verification is permanently disabled for yfinance calls.
- **Why it's wrong:** Susceptible to MITM attacks on price data. While unlikely for yfinance, an attacker could inject false prices.
- **Workaround:** Use VPN or trusted network for production runs. Verify data sanity after each ingest (step 1 logs column counts).
- **Fix approach:** Add `RunConfig.ssl_verify` flag; default False (current behavior); add doc + warning. Planned: migrate to system trust store or cert pinning.

### P23 — Partial ingestion produces plausible-looking outputs (SILENT FAILURE)
- **What happens:** Ingestion failures (network, API down) are caught and logged at WARNING level; pipeline continues with whatever data was successfully fetched. Macro_raw.parquet column count is diagnostic only (should be ~53 cols).
- **Why it's wrong:** User won't notice a missing series until downstream metrics drift. Adds stale-data risk.
- **Workaround:** Check `macro_raw.parquet` column count after step 1. Run `python -c "import pandas as pd; df=pd.read_parquet('data/raw/macro_raw.parquet'); print(len(df.columns), df.columns[:10])"`.
- **Fix approach:** Already partially fixed (P23 in D14: added `ingestion_completeness_report()`). Call `validate_ingestion_completeness()` after step 1 and FAIL if columns missing. Wired in `run_pipeline.py` — verify still active.

### P24 — CheckpointManager.list() silently ignores corrupt metadata
- **What happens:** `CheckpointManager.list()` catches `JSONDecodeError` and `OSError` without logging which file failed.
- **Why it's wrong:** Silent failures make debugging metadata corruption hard.
- **Workaround:** Manually inspect `data/checkpoints/*.meta.json` for syntax errors.
- **Fix approach:** Already fixed in D14 — log at WARNING with filename before skipping. Verify in `checkpoints.py` line 245.

### P25 — Committed data artifacts can create stale-data bugs
- **What happens:** `data/grok_quarter_classifications_20260216.pickle`, `data/fred_api_datasets_snapshot_20260216.pickle`, `data/multpl_datasets_snapshot_20260216.pickle` are checked in. If pipeline accidentally loads these instead of fresh-fetched data, results silently based on Feb 2026 snapshots.
- **Why it's wrong:** Stale-data gap to current date undetected. Produces misleading regime classifications.
- **Workaround:** Ensure `--refresh` flag is used in weekly/monthly automation.
- **Fix approach:** Move snapshots to `data/archives/` with explicit "do not load by default" documentation. Or add timestamp check (warn if snapshot >30 days old) in ingestion/__init__.py.

### P26 — FRED ingestion hard-fails when FRED_API_KEY missing
- **What happens:** `fred.py`'s `fetch_all()` calls `fredapi.Fred(api_key=...)` which raises ValueError if key is None.
- **Why it's wrong:** Error message doesn't guide user to solution.
- **Workaround:** Copy `.env.example` to `.env` and add free key from fred.stlouisfed.org.
- **Fix approach:** Catch KeyError in __main__, print helpful message: "FRED_API_KEY not set in .env. Get free key at ... and run: cp .env.example .env && edit .env". Estimated effort: 1 line + docstring.

### P13 — Checkpoint freshness check uses wall-clock time, not data time
- **What happens:** `CheckpointManager.is_fresh(name, max_age_days=7)` rejects checkpoints older than 7 wall-clock days. If FRED releases new data on Monday but pipeline hasn't run, checkpoint remains "fresh" until Thursday, stale data flows downstream.
- **Why it's wrong:** Data staleness and checkpoint staleness are decoupled.
- **Workaround:** Always run with `--refresh` on production schedules (Friday or Monday).
- **Fix approach:** Add optional `data_lag_quarters=1` parameter to `is_fresh()`. Check checkpoint metadata's index_end date against today's quarter; reject if more than `data_lag_quarters` behind. Estimated effort: 0.5 component.

---

## Security Considerations

### SSL/TLS: Disabled verification in yfinance fallback
- **Risk:** MITM on ETF price data (unlikely but possible in compromised networks).
- **Files:** `src/trading_crab_lib/ingestion/assets.py` lines 118-148
- **Current mitigation:** Limited to Phase 2 fallback (only triggered if batch download fails); SSL warning logged.
- **Recommendations:** 
  1. Add `RunConfig.ssl_verify` toggle (default False for backward compatibility).
  2. Document in README: "SSL verification is disabled for yfinance to work around cert-store issues. Use on trusted networks."
  3. Plan: Use `certifi.where()` or system trust store if curl_cffi upgrades bundled libcurl.

### Secrets management
- **Risk:** `.env` file with FRED_API_KEY could be accidentally committed.
- **Current mitigation:** `.env` in `.gitignore`; `.env.example` provided as template; setup.sh prompts user.
- **Status:** ✅ Good. Email config also guarded (`config/email.yaml`, `config/email.local.yaml` in `.gitignore`).

### Pickle/Joblib serialization
- **Risk:** `joblib.dump()` / `joblib.load()` execute arbitrary code on load.
- **Current mitigation:** Models stored in `outputs/models/*.pkl` (not committed); users control access.
- **Files:** `src/trading_crab_lib/checkpoints.py` lines 210-230 (save_model/load_model).
- **Recommendations:** Document: "Never load a .pkl file whose provenance you cannot verify." No code change needed.

---

## Performance Bottlenecks

### Large feature matrix and derivatives computation
- **Problem:** 69 clustering features × 3 derivatives (d1/d2/d3) = 207 columns before final feature selection. `np.gradient()` + rolling mean computation is O(n·m).
- **Files:** `src/trading_crab_lib/transforms.py` (apply_derivatives, lines 240-300)
- **Symptom:** Step 2 (features) takes ~30 seconds for full history.
- **Mitigation:** Checkpoint-driven caching; `--recompute` flag skips redundant derivative calculations. Efficient enough for quarterly pipeline; would be stressed at monthly frequency (R1).
- **Long-term:** Consider vectorized derivative via rolling_apply or NumPy broadcasting if monthly spine adopted.

### PCA + KMeans sweep across k=2..12
- **Problem:** Each k requires full PCA fit + KMeans fit with n_init=50. Step 3 runs ~11 models × 50 restarts each.
- **Files:** `src/trading_crab_lib/clustering.py` (fit_clusters, lines 380-440)
- **Symptom:** Step 3 takes ~5–10 seconds.
- **Mitigation:** Checkpoint-driven; single PCA fit reused across k-sweep; parallelization not implemented but feasible.
- **Impact:** Negligible for quarterly pipeline; at monthly frequency (R1) would warrant parallel sweep.

### RRG (Relative Rotation Graph) computation
- **Problem:** `compute_rrg()` in `src/trading_crab_lib/diagnostics.py` computes rolling percentile ranks and z-scores for all assets and ratios. Quadratic in asset count.
- **Files:** `src/trading_crab_lib/diagnostics.py` (lines 60–150 for compute_rrg, percentile_rank)
- **Symptom:** Step 8 (diagnostics) can take 30+ seconds with 20+ ETFs.
- **Mitigation:** Only run step 8 if --plots or --diagnostics flag set. Cacheable with checkpoint.
- **Impact:** Acceptable for weekly reports; would stall at daily frequency.

### Seaborn pairplot for 69 features
- **Problem:** `notebooks/03_clustering.ipynb` can generate 69×69 = 4761 subplots. Very slow.
- **Files:** `notebooks/03_clustering.ipynb` (cell for pairplot)
- **Mitigation:** Disabled by default (`generate_pairplot: False` in RunConfig). Only generated on explicit request.

---

## Fragile Areas

### Gap-fill algorithm (Bernstein polynomial + Taylor extrapolation)
- **Files:** `src/trading_crab_lib/transforms.py` (lines 130-200 for apply_gap_fill and _fill_column)
- **Why fragile:** 
  - Boundary conditions require 4 derivatives per side (value + d1 + d2 + d3). Any NaN in d2 or d3 near gap causes failure.
  - Edge gaps (leading/trailing NaNs) use Taylor extrapolation; small data → unreliable derivatives.
  - Silent fallback to forward-fill if BPoly creation fails (line 175: `except Exception`).
- **Safe modification:** Test on synthetic gap patterns before production changes. Add assertions: "Taylor extrapolation degree ≤ min(available_rows, 3)".
- **Test coverage:** `tests/unit/test_transforms.py` has 5 gap-fill tests. Not comprehensive for all edge cases.

### Cluster label canonicalization
- **Files:** `src/trading_crab_lib/clustering.py` (_canonicalize_cluster_col, lines 50-80)
- **Why fragile:** Assumes cluster labels are sortable and PC1 is computed. If PCA fails or cluster IDs are non-integer, function breaks silently.
- **Safe modification:** Add assertions: `assert np.issubdtype(labels.dtype, np.integer)`, `assert len(pca_obj.components_) > 0`.
- **Test coverage:** `tests/unit/test_clustering.py` has 2 canonicalization tests.

### Regime profile computation and naming
- **Files:** `src/trading_crab_lib/regime.py` (build_profiles, suggest_names, lines 20-80)
- **Why fragile:** 
  - `suggest_names()` uses heuristics (median-deviation ratios) that silently skip 4 features (`10yr_ustreas`, `fred_gs10`, `fred_tb3ms`, `div_minus_baa`) if only derivatives are in clustering_features.
  - No validation that regime has ≥ 3 observations for profile estimation.
- **Safe modification:** Assert `len(regime_data) >= 3` before computing std.
- **Test coverage:** `tests/test_models_reporting.py` has 3 profile tests, but limited edge cases.

### Forward classifiers (multi-horizon predictions)
- **Files:** `src/trading_crab_lib/prediction/__init__.py` (train_forward_classifiers, lines 370-420)
- **Why fragile:** 
  - Uses `y.shift(-h).dropna()` which silently drops last h rows. If h ≥ remaining rows, y_future is empty, fitting fails.
  - No check for minimum data per class; imbalanced horizons can produce degenerate trees.
- **Safe modification:** Assert `len(y_future) > 2*n_classes` before fitting.
- **Test coverage:** `tests/unit/test_prediction_flat.py` has 2 forward-classifier tests, but no imbalance/edge cases.

### Email body construction and SMTP
- **Files:** `src/trading_crab_lib/email.py` (build_weekly_email_body, send_weekly_email, lines 150-280)
- **Why fragile:** 
  - SMTP SSL/TLS negotiation can fail silently if port/use_tls mismatch.
  - Report content is markdown-like; no validation that plots exist before embedding.
- **Safe modification:** Test SMTP connection at config load time (connection pool check). Validate plot paths exist before building HTML.
- **Test coverage:** `tests/test_email_weekly.py` has 30 tests; good coverage but no real SMTP test (uses mock).

---

## Scaling Limits

### Quarterly frequency bottleneck (R1)
- **Current capacity:** ~260 observations (1950–2026, quarterly).
- **Limit:** Labeler needs ~15–30 regime transitions to be stable. With quarterly data at ~5-year sojourn, only ~10 transitions in history. Monthly would give ~40 transitions.
- **Scaling path:** Migrate to monthly frequency (R1). Estimated effort: 2 phases.

### ETF price history coverage (R13)
- **Current capacity:** SPY/GLD/TLT/QQQ/etc. start 1993–2006; ~30 years of asset data.
- **Limit:** Asset-returns regression (L3) is blind 1962–1993. Pre-1993 regimes have no asset price data.
- **Scaling path:** Build spliced long index/futures histories (S&P 500 TR, gold spot, WTI, Treasury synthetics). Design §9 has source references. Estimated effort: 1 phase.

### Feature set size
- **Current capacity:** 69 clustering features fit comfortably in `features.parquet` (300 rows × 70 cols).
- **Limit:** At monthly frequency (R1, ~850 rows), memory is not a constraint, but derivative computation becomes noticeable.
- **Scaling path:** Vectorize derivatives via NumPy broadcasting or Numba. No architectural change needed.

### Model CV evaluation
- **Current capacity:** `TimeSeriesSplit(n_splits=5)` with ~260 rows gives ~50-row test folds. Sufficient for 5-class regime prediction.
- **Limit:** At lower frequency (more data per quarter) or with additional prediction targets (L3 per-asset models), CV become fragmented.
- **Scaling path:** Implement walk-forward harness (R10) with configurable refit frequency. Estimated effort: 1 phase.

---

## Dependencies at Risk

### Optional: `k-means-constrained`
- **Risk:** Not on PyPI maintained anymore; unmaintained since 2020. Vendor dependency (relies on C++ binding).
- **Impact:** Balanced clustering falls back to plain KMeans if package unavailable (line 100 in clustering.py). Cluster sizes become imbalanced.
- **Mitigation:** `--no-constrained` flag allows users to skip if missing.
- **Recommendation:** Lock version in `requirements.txt` (currently pinned). Monitor for updates; consider fork if upstream dies. No immediate action needed.

### Optional: `hmmlearn`
- **Risk:** Smaller maintainer community than scikit-learn. Unstable release cycle (~1 release/year).
- **Impact:** HMM-based labeling (hmm.py) skips gracefully if missing; GMM + KMeans continue working.
- **Mitigation:** Tests skip with `pytest.mark.skipif(not HAS_HMMLEARN)`.
- **Recommendation:** Safe; conditional dependency. No action needed.

### Optional: `lxml` and `cssselect`
- **Risk:** External C++ binding for XML parsing. Large attack surface.
- **Impact:** multpl.com scraping (step 1) fails if missing. Fallback to... none (hard requirement for step 1).
- **Mitigation:** Core requirement in `requirements.txt` and `pyproject.toml[ingestion]`.
- **Recommendation:** Good. Mark as required, not optional.

### Optional: `joblib` (now required)
- **Risk:** Migrated from pickle to joblib for model serialization (D14, 2026-03-31). Joblib versioning is stable but worth monitoring.
- **Impact:** Model pickle/unpickle depends on joblib compatibility.
- **Mitigation:** Pinned in `requirements.txt` (joblib>=1.3).
- **Recommendation:** Good. Keep pin current.

---

## Missing Critical Features

### Walk-forward evaluation harness (R10, BLOCKING)
- **Problem:** No refit loop for honest backtesting. All labels computed on full data.
- **Blocks:** Can't measure hindsight content or detection lag. Can't validate strategy under real-time constraints.
- **Implementation notes:** Requires refit loop at each time point t: fit L1/L2/L3 on data ≤ t, score data at t, step forward.
- **Priority:** HIGH — must implement before next modeling iteration to avoid inflated metrics.

### Purged and embargoed CV (R7, HIGH PRIORITY)
- **Problem:** No purging of overlapping-label windows. No embargo of trailing labels (unstable due to two-sided smoothing).
- **Blocks:** Forward-classifier evaluation likely inflated. Real-time detection lag not measured.
- **Implementation notes:** López de Prado ch.7–8 shows purging mechanics. For h-step-ahead labels, remove training data within h periods of test fold start. Embargo last 6–12 months.
- **Priority:** HIGH — affects supervised (L2/L3) accuracy metrics.

### Trial registry and holdout discipline (R12, MEDIUM PRIORITY)
- **Problem:** No log of which configs were evaluated. No locked holdout (2021+ untouched).
- **Blocks:** Can't compute deflated Sharpe. Multiple-testing inflation unmeasured.
- **Implementation notes:** SQLite or flat JSON file: {config_hash → {params, metrics, timestamp}}. Carve 2021+ as read-only partition.
- **Priority:** MEDIUM — needed for honest strategy evaluation in Phase 6 (design freeze).

### Monthly data spine (R1, RESHAPING)
- **Problem:** Quarterly frequency limits regime detection (10 transitions in history vs 40 at monthly).
- **Blocks:** Labeler starved. Detection lag 4× longer in calendar time.
- **Implementation notes:** Rearchitect ingestion to monthly; keep quarterly agency series with alignment. Resample ETF prices to monthly.
- **Priority:** HIGH for redesign; deferred for current codebase (too invasive).

### Spliced long asset price histories (R13, DATA ENGINEERING)
- **Problem:** ETF prices start 1993–2006; asset-returns regression (L3) blind 1962–1993.
- **Blocks:** Can't train regime-conditional asset models on full history.
- **Implementation notes:** Build spliced SPY←S&P 500 TR, GLD←gold spot/futures, etc. Design §9 lists sources.
- **Priority:** MEDIUM for redesign; low for current use case (30-year period has ~25% of available regime data).

### Volatility forecasting and regime-conditional stops (R9, PORTFOLIO)
- **Problem:** No GARCH/EWMA vol layer. No vol-scaled stops or regime-conditional multipliers.
- **Blocks:** Portfolio exposure not de-risked into volatility spikes or crashes.
- **Implementation notes:** Add GARCH(1,1) or EWMA vol forecast. Implement vol targeting and regime-conditional stop multipliers.
- **Priority:** MEDIUM — nice-to-have; current tactics module functional but crude.

---

## Test Coverage Gaps

### `src/trading_crab/pipeline.py` (1374 lines, ZERO unit test coverage)
- **What's not tested:** Step dispatch, flag handling (--steps, --market-code, --refresh, etc.), checkpoint fallback logic, error recovery.
- **Files:** `src/trading_crab/pipeline.py` (no corresponding unit test file).
- **Why it matters:** Pipeline CLI is the primary user interface; regressions in flag handling silently break workflows.
- **Test coverage:** Integration tests exist (test_pipeline_smoke.py, test_cli_smoke.py), but no unit tests of step dispatch or flag combinations.
- **Fix approach:** Create `tests/test_pipeline_unit.py` with mocked checkpoints and steps. Estimated effort: 0.5 phase (testing).

### TimeSeriesSplit CV edge cases
- **What's not tested:** Behavior when n_splits ≥ number of available samples. Behavior when n_samples < 2*n_splits.
- **Files:** `src/trading_crab_lib/prediction/__init__.py` (lines 145-160 for TSCV usage), `src/trading_crab_lib/monitoring/prediction.py` (lines 30-50 for compute_cv_fold_scores).
- **Why it matters:** Small datasets (early time steps) can produce empty test folds, causing silent failures.
- **Fix approach:** Add assertions: `n_samples >= 2*n_splits` before TSCV initialization. Add test cases for edge n_samples.

### Forward classifier label alignment
- **What's not tested:** Behavior when shifted label has all NaN (h ≥ remaining rows). Behavior when class imbalance is extreme.
- **Files:** `src/trading_crab_lib/prediction/__init__.py` (lines 370-420 for train_forward_classifiers).
- **Why it matters:** Silent fitting failures (empty y_future) produce degenerate models.
- **Fix approach:** Add guard: `if len(y_future) < 2: return None` with warning. Add test cases.

### Regime naming heuristics
- **What's not tested:** Behavior when clustering_features lacks certain series (e.g., only derivatives retained). Behavior when regime has <3 observations.
- **Files:** `src/trading_crab_lib/regime.py` (suggest_names, lines 40-80).
- **Why it matters:** Silent skipping of features makes naming inconsistent across runs.
- **Fix approach:** Log skipped features at INFO level. Add test cases for all skip scenarios.

### Email SMTP configuration
- **What's not tested:** Real SMTP connection (all tests mock). TLS/SSL negotiation. Credential validation.
- **Files:** `src/trading_crab_lib/email.py` (send_weekly_email, lines 200-250).
- **Why it matters:** Email failures only discovered at deployment time.
- **Workaround:** Test locally with `--send-email` flag + mock SMTP server (`smtp4dev` or similar) before production.
- **Fix approach:** Add optional integration test with real SMTP credentials (gated by env var, e.g., TEST_SMTP=1).

---

## Summary: Risk Triage

| Issue | Severity | Category | Blocks | Effort |
|-------|----------|----------|--------|--------|
| R1 (Quarterly spine) | HIGH | Architecture | Redesign | 2 phases |
| R10 (Walk-forward harness) | HIGH | Evaluation | Honest metrics | 1 phase |
| R7 (Purged CV) | HIGH | Bias | L2/L3 eval | 0.5 phase |
| R2 (Forced balance) | MEDIUM | Modeling | Temporal logic | 1 phase |
| R6 (FRED revised data) | MEDIUM | Data | Reproducibility | 1 phase |
| R13 (Short asset histories) | MEDIUM | Data | L3 training | 1 phase |
| P22 (SSL disabled) | MEDIUM | Security | MITM risk | 0.2 phase |
| P13 (Checkpoint staleness) | MEDIUM | Operations | Stale data | 0.5 phase |
| R12 (No trial registry) | MEDIUM | Eval | Strategy val | 0.5 phase |
| P23 (Partial ingestion) | LOW | Debugging | Silent failure | Already fixed |
| P24 (Metadata corruption) | LOW | Debugging | Silent failure | Already fixed |
| R3 (Constant transitions) | LOW | Modeling | Transition model | 1 phase |
| R4 (PCA interpretation) | LOW | Analysis | Readability | 0.5 phase |
| R9 (No vol forecasting) | LOW | Portfolio | Vol de-risking | 1 phase |

---

*Concerns audit: 2026-07-09*
