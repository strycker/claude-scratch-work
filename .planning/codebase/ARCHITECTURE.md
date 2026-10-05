<!-- refreshed: 2026-10-05 -->
# Architecture

**Analysis Date:** 2026-10-05

## System Overview

```text
┌──────────────────────────────────────────────────────────────────┐
│ L0  DATA LAYER         Raw daily USD prices (spliced long       │
│ (platform/)            histories), market-observed features,     │
│                        ALFRED-vintage macro, slow anchors        │
│ Files: ingestion/*,                                              │
│ splice.py, transforms_monthly.py                                 │
├──────────────────────────────────────────────────────────────────┤
│ L1  LABELING LAYER     Two-sided regime labeling: jump model    │
│ (platform/labeling/)   (preferred) / t-HMM benchmark            │
│ Files: jump_model.py,  → hard labels s_t, soft confidences γ_t  │
│ diagnostics.py                                                   │
├──────────────────────────────────────────────────────────────────┤
│ L2  REGIME PREDICTION  Causal features only (one-sided)         │
│ (platform/prediction/) • Nowcaster: P(S_t | features_t)         │
│ Files: nowcaster.py,   • Transition: P(S_{t+1}=j | S_t=i, age)  │
│ regime_filter.py       → posterior probabilities, filtered belief│
│ transition_matrix.py                                             │
├──────────────────────────────────────────────────────────────────┤
│ L3  ASSET PREDICTION   Regime-conditional returns + volatility  │
│ (platform/assets/)     per-asset, per-horizon forecasts,        │
│ Files: returns.py,     covariance + stop levels                 │
│ vol.py                                                           │
├──────────────────────────────────────────────────────────────────┤
│ L4  ALLOCATION & REPORT Target mix, buy/sell/hold per holding,  │
│ (platform/allocation/, target weights → traded book via        │
│ platform/report/)      hysteresis (no-trade band), weekly       │
│ Files: tilt.py,        markdown report with email opt-in        │
│ hysteresis.py,                                                  │
│ weekly.py, scoreboard.py                                         │
├──────────────────────────────────────────────────────────────────┤
│ M8 HONESTY LAYER       Walk-forward CV, registry budgets, DSR  │
│ (platform/honesty/)    gating, HOLDOUT (2021+, locked)         │
│ Files: cv.py, registry.py,  → deflated Sharpe, quality tier    │
│ holdout.py, walkforward.py                                      │
├──────────────────────────────────────────────────────────────────┤
│ M9+ MONITORING         Weekly tripwire (credit spreads),        │
│ (platform/tripwire/)   stability (parked), classifier #2 (parked)│
│ Files: monitor.py,     parked/classifier2.py, parked/stability.py
│ parked/                                                          │
├──────────────────────────────────────────────────────────────────┤
│ BACKBONE: STATE         Checkpoints, config, snapshots,         │
│ MANAGEMENT              CLI entry, validation                    │
│ (platform/)                                                     │
│ Files: checkpoints.py, config.py, snapshots.py, taxonomy.py,   │
│ splice.py                                                       │
├──────────────────────────────────────────────────────────────────┤
│ EVALUATION & PLOTTING   Model diagnostics, plotting subpackage, │
│ (platform/evaluation/,  comparison metrics, churn analysis      │
│ platform/plotting/)                                             │
│ Files: deflated_sharpe.py, kpis.py, report.py,                 │
│ plotting/*.py                                                   │
└──────────────────────────────────────────────────────────────────┘
```

## Component Responsibilities

| Component | Responsibility | File |
|-----------|----------------|------|
| **M0 Config & Checkpoints** | Load `config/platform_settings.yaml`, validate schema, manage parquet checkpoints with metadata | `config.py`, `checkpoints.py` |
| **M1 Data Spine** | Fetch FRED/ALFRED, splice daily ETF prices, handle publication lags, build monthly `monthly_raw` checkpoint | `ingestion/alfred.py`, `ingestion/macro_monthly.py`, `splice.py`, `scripts/build_platform_data.py` |
| **M2 Feature Taxonomy** | Define fast/slow/agency tiers, validate taxonomy, compute causal-only features → `monthly_features` checkpoint | `taxonomy.py`, `features/invariants.py`, `transforms_monthly.py` |
| **M3 Baseline + Backtest** | Vol-targeted no-regime baseline, walk-forward backtest driver, cost accounting, Faber comparison | `assets/returns.py`, `assets/vol.py`, `backtest/driver.py`, `backtest/baselines.py` |
| **M4 Weekly Report (Serve)** | Nowcaster → hysteresis → weighted tilt → markdown + email; no regime gating for MVP-1 baseline | `report/weekly.py`, `report/serving.py`, `allocation/hysteresis.py` |
| **MVP-1** | Usable weekly guidance from baseline (no regime classifier); Glenn executes manually in Fidelity | (M0–M4 only) |
| **M5 Regime Labeler** | Jump model or HMM, grid search λ/K, occupancy/sojourn/stability validation, label alignment across refits | `labeling/jump_model.py`, `labeling/diagnostics.py` |
| **M6 Nowcaster + Filter** | Causal-only P(regime now) classifier + regime-age feature; Bayesian filter for tracked belief over time | `prediction/nowcaster.py`, `prediction/regime_filter.py`, `prediction/transition_matrix.py` |
| **M7 Regime Tilt + Ablation** | Regime-conditional covariance + per-regime weights; backtest compares tilt vs no-regime baseline | `allocation/tilt.py`, `backtest/driver.py` (tilt-off ablation leg) |
| **MVP-2** | Regime-aware guidance (M5+M6+M7); regime tilt beats baseline or stays advisory | (M0–M7 only) |
| **M8 Honesty** | CV embargo + gap-lag, DSR as quality gate, registry trial budgets, HOLDOUT (2021+) evaluation | `honesty/{cv,gap_lag,holdout,registry,walkforward}.py`, `evaluation/deflated_sharpe.py` |
| **M9+ Tripwire** | Weekly credit spread alerts (DAAA/DBAA) for manual crisis override; advisory post-08.2-03 | `tripwire/monitor.py` |
| **Parked: Classifier #2** | Leadership/relative classifier (rejected: no lift, independence unproven); un-park by `git mv` | `parked/classifier2.py` |
| **Parked: Joint Driver** | Two-classifier backtest (joint lift not met); un-park by `git mv` | `parked/joint_driver.py` |
| **Parked: Stability** | Subsample stability suite (out of scope for weekly MVP); un-park by `git mv` | `parked/stability.py` |
| **Evaluation Metrics** | KPIs (Sharpe, Calmar, max DD), deflated Sharpe, model comparison, churn analysis | `evaluation/{kpis,deflated_sharpe,model_metrics}.py` |
| **Plotting** | Diagnostics + visuals per layer (data, features, regimes, nowcaster, backtest, allocation) | `plotting/{core,data,features,regime,nowcaster,backtest,allocation}.py` |

## Pattern Overview

**Regime-switching mixture of experts:** Offline two-sided regime discovery (L1) feeds online causal nowcaster (L2) that gates per-regime asset models (L3) → allocation weights (L4). Information-flow firewall: L1 sees future; L2–L4 do not. Monthly refits; weekly scoring. Walk-forward validation only; no standard CV shuffle (temporal structure preserved).

**Layered decision gates:** MVP-1 runs M0–M4 (baseline, no regimes) for manual trading. MVP-2 adds M5–M7 (regimes). M8–M9+ are monitoring/honesty/monitoring (production hardening).

**Trial registry + deflated Sharpe:** Every model variant (K, λ, feature set, hyperparams) recorded as a trial row; DSR gates selection; post-2021 HOLDOUT locked and evaluated exactly once at design freeze.

**Data lineage:** platform-specific checkpoints under `data/checkpoints/platform/` (tracked in git, safe to refresh); frozen incumbent's `data/` and `outputs/` are separate (D-02, avoids collision).

## Layers

### L0: Data Spine (M1)

- **Purpose:** Fetch, validate, splice long-history price and macro series.
- **Location:** `src/trading_crab_lib/platform/ingestion/`, `splice.py`, `transforms_monthly.py`
- **Key Files:**
  - `ingestion/alfred.py` — ALFRED vintage-corrected macro (publication-lag applied per D-07)
  - `ingestion/macro_monthly.py` — FRED monthly series (GDP, inflation, yields)
  - `ingestion/prices_daily.py`, others — Asset prices (spliced to long histories)
  - `splice.py` — Ratio splice, treasury TR synthesis (bond-price repricing), equity TR from price+div
  - `transforms_monthly.py` — Resample to monthly, apply publication lags, build `monthly_raw` checkpoint
- **Outputs:** `monthly_raw.parquet` (time × features), `daily_raw.parquet` (for tripwire), `fred_daily_raw.parquet` (DAAA/DBAA)
- **Key Constraints:** Publication-lag shifts for any series with non-trivial revision lag; no forward-looking data in causal-downstream layers (L2–L4).

### L1: Regime Labeling (M5)

- **Purpose:** Ground-truth regime discovery via two-sided information.
- **Location:** `src/trading_crab_lib/platform/labeling/`
- **Key Files:**
  - `jump_model.py` — Preferred: k-means + jump-penalty (λ), Viterbi DP solver, multi-restart
  - `diagnostics.py` — Occupancy/sojourn/stability validation, label alignment (Hungarian algorithm)
- **Acceptance Criteria:** (1) Occupancy ≥~8% and ≤~35% per state (recurrence exemption for crisis); (2) Median sojourn ≥3 months; (3) Subsample stability (hold under decade drop); (4) Economic interpretation; (5) Decision-relevant in asset returns; (6) Effective-sample honesty (≤ ~30 independent transitions).
- **Outputs:** `regime_labels.parquet` (time × regime_id), `regime_profiles.parquet` (regime_id → statistics), `transition_matrix_empirical.parquet`
- **Gate:** Timeline vs recessions, occupancy table, profiles make economic sense (P3_regime_labeling.ipynb).

### L2: Nowcaster + Filter (M6)

- **Purpose:** Real-time causal classification into current regime; persist belief across time.
- **Location:** `src/trading_crab_lib/platform/prediction/`
- **Key Files:**
  - `nowcaster.py` — Fit RandomForest on full history (causal features only), TimeSeriesSplit CV, returns classifier object
  - `regime_filter.py` — Bayesian filter: predict → update belief via likelihood ratio (uses training class prior)
  - `transition_matrix.py` — Build transition matrix from L1 labels, regime-age feature
- **Scoring:** Latest complete month in model's `feature_names_in_` (no imputation; staleness per series); distinct posterior count across holdout.
- **Outputs:** `nowcaster_model.pkl` (fitted RandomForest), `regime_belief.parquet` (current belief over states), `nowcaster_class_prior.parquet`
- **Gate:** Distinct-output count > 1 (M6 currently count=1, hence advisory per A-14; A-14 says view suspended till fix).

### L3: Asset Prediction

- **Purpose:** Regime-conditional quarterly return and volatility forecasts per ETF.
- **Location:** `src/trading_crab_lib/platform/assets/`
- **Key Files:**
  - `returns.py` — Per-asset, per-horizon models (returns predicted by regime)
  - `vol.py` — Realized vol, GARCH/EWMA forecasts, regime-conditional covariance
- **Outputs:** Per-asset returns table (regime × horizon), vol matrix, covariance
- **Not Yet Implemented:** L3 per-asset models deferred; M3 baseline uses fixed vol target, M7 uses empirical covariance per regime.

### L4: Allocation & Report (M4, M7)

- **Purpose:** Weights → execution → markdown guidance.
- **Location:** `src/trading_crab_lib/platform/allocation/`, `src/trading_crab_lib/platform/report/`
- **Key Files:**
  - `allocation/tilt.py` — Regime-conditional target weights (covariance-first + shrunk returns + vol-scaling + Kelly)
  - `allocation/hysteresis.py::execute_rebalance` — 5pp no-trade band, load-before-save Pitfall 3 ordering
  - `report/weekly.py` — Assemble markdown: regime distribution, trajectory, per-asset signals, target vs held
  - `report/serving.py` — Cold-start belief, filter step, holds hysteresis output
  - `report/scoreboard.py` — Asset performance heatmap (regime × ETF), added by 08.2-03
- **Outputs:** `executed_weights.parquet` (traded book, post-hysteresis), `regime_belief.parquet`, `weekly_report.md` (sent to Glenn)
- **Email:** Opt-in via `--send-email`, reuses incumbent `email.py` read-only (not forked).

### M8: Honesty Framework

- **Purpose:** Guard against overfitting: CV embargo, deflated Sharpe as gate, trial registry, HOLDOUT discipline.
- **Location:** `src/trading_crab_lib/platform/honesty/`
- **Key Files:**
  - `cv.py` — TimeSeriesSplit, purged/embargoed fold boundaries
  - `gap_lag.py` — Embargo last 6–12 months of labels (refresh churn)
  - `holdout.py` — Locked 2021-01-01 onward; evaluated once at design freeze
  - `registry.py` — Trial rows (K, λ, features, hyperparams, results); per-phase budgets (ADR-0004)
  - `deflated_sharpe.py` — Sharpe adjusted for multiple testing / selection bias
- **Rule:** Every model configuration is a trial row; DSR gates selection; post-2021 is firewall (no tuning, only evaluation).

### M9+: Monitoring & Tripwire

- **Purpose:** Runtime safety: weekly credit alerts, stability (parked), classifier #2 (parked).
- **Location:** `src/trading_crab_lib/platform/tripwire/`, `src/trading_crab_lib/platform/parked/`
- **Key Files:**
  - `tripwire/monitor.py` — Weekly DAAA/DBAA spreads, advisory flag when > threshold (post-08.2-03)
  - `parked/classifier2.py` — Leadership/relative; rejected (L1-04 DEFER)
  - `parked/stability.py` — Subsample stability (out of scope for MVP)

## Data Flow

### Primary Path: Weekly Report Generation

1. **Load state** (load-before-save Pitfall 3 ordering):
   - `regime_belief` checkpoint (previous week's posterior) — cold start if absent
   - `executed_weights` checkpoint (previous month's post-hysteresis weights) — cold start if absent
   - `nowcaster_model.pkl`, `transition_matrix_empirical.parquet`, `regime_labels.parquet`

2. **Score & filter** (monthly granularity):
   - Latest complete month in nowcaster's `feature_names_in_` → `predict_proba()` → posterior
   - Regime filter: advance belief by unobserved months (predict-only), absorb latest month (filter step)
   - Return: belief `prob_dist` over regimes, as-of date

3. **Allocate** (if month changed):
   - `vol_targeted_tilt(belief, covariance, shrunk_returns)` → target weights
   - `execute_rebalance(target, held, band=5pp)` → executed (post-hysteresis) weights
   - Persist `executed_weights`, `regime_belief`

4. **Report**:
   - Assemble markdown: regime distribution (belief), nowcaster trajectory (history), per-asset returns (from L1 profiles), target vs held, implied trades
   - Write `outputs/reports/platform/weekly_report.md`
   - Optionally email via incumbent `email.py` (read-only)

### Training Path: Monthly Refit (Ad-hoc via Scripts)

- `scripts/build_platform_data.py` → ingestion → `monthly_raw`, `monthly_features`
- Labeling (ad-hoc, full-info): Jump model grid sweep (K, λ) → labels + profiles + transition matrix
- Nowcaster fit: TimeSeriesSplit CV on `monthly_features` (causal, gap-lag embargo) → `nowcaster_model.pkl`, `nowcaster_class_prior.parquet`
- Backtest: Drive monthly history through L4 allocation loop; log walk-forward results
- Evaluation: Registry trial (K, λ, etc.) → DSR gate → accept/reject
- Serve: New model pickled, live report runs against it

## Key Abstractions

### CheckpointManager
- **Purpose:** Persist and load DataFrames as parquet + metadata.
- **Files:** `checkpoints.py`
- **API:** `save(df, name)`, `load(name)`, `is_fresh(name)`, `clear(name)`, `clear_all()`
- **Metadata:** Creation timestamp, config hash, row/col counts
- **Platform namespace:** `data/checkpoints/platform/` (separate from frozen incumbent's namespace; safe to re-run)
- **Example:** `cm.save(monthly_raw, "monthly_raw")` → `data/checkpoints/platform/monthly_raw.parquet`

### Config Schema (platform_settings.yaml)
- **Purpose:** Centralize all tunable parameters (no hardcoded constants).
- **Sections:** `data`, `fred_monthly`, `fred_vintage`, `splice`, `universe`, `taxonomy` (required; D-02 independent of incumbent `settings.yaml`)
- **Validation:** Collect-all-errors-then-raise-once (mirrors incumbent `validate_config()`)
- **Usage:** Loaded once at startup via `load_platform_config()`; passed to every layer as a single dict

### Taxonomy (fast/slow/agency)
- **Purpose:** Classify features by revision history and availability.
- **Files:** `taxonomy.py`, configured in `platform_settings.yaml`
- **Tiers:**
  - `fast`: Market-observed, unrevised (equity prices, yields)
  - `slow`: Strategic anchors (CAPE, Buffett Indicator, spot gold/oil via macrotrends)
  - `agency`: ALFRED vintage-corrected (GDP, employment — revised monthly)
- **Guard:** Lean labeling uses fast+slow only (1962+ unbroken history); agency for strategic only
- **Validation:** Every feature in exactly one tier (errors at load time)

### Regime Labels & Alignment
- **Purpose:** Consistent state IDs across refits.
- **Challenge:** Jump model + k-means are non-identifiable (label switching); Viterbi-decoded HMM same issue.
- **Current Workaround (CR-02 open):** Ad-hoc manual alignment by economic description; Hungarian algorithm in parked stability.py.
- **Future Fix:** Semantic skeleton (D3) + label-switching diagnostics (see labeling/diagnostics.py roadmap).

### Regime Filter (Bayesian Recursion)
- **Purpose:** Persist belief state across weekly runs without re-fitting.
- **Implementation:** `regime_filter.py::predict_only_step()` (advance by unobserved months), `filter_step()` (absorb latest posterior).
- **Prior:** Training class prior from `nowcaster_class_prior.parquet`; coldstart from `unconditional_belief()` on L1 labels.
- **Persistence:** Belief (prob dist + as-of date) saved to checkpoint after each run.
- **Invariant:** One month per filter step (same as allocation band), so repeated weekly runs within a month re-band against the same held weights (no drift).

### Hysteresis (No-Trade Band)
- **Purpose:** Suppress noise-chasing weight churn (stability, not cost modeling).
- **Implementation:** `allocation/hysteresis.py::execute_rebalance(target, held, band=5pp)` → weights only rebalance if drift > band
- **Persistence:** Held weights saved after each rebalance; loaded before saved (Pitfall 3 ordering).
- **Monthly granularity:** One band step per month; same-month re-run re-bands against the same held book (no compound drift).

## Entry Points

### Weekly Report (Production Serve)
- **CLI:** `python -m trading_crab_lib.platform.evaluation.report` (part of `tradingcrab` CLI)
- **Triggered by:** `scripts/run_weekly_report.py` or cron job (Phase 9, not yet automated)
- **Responsibility:** Load state → score → filter → allocate (if month changed) → report → email (opt-in)
- **Output:** `outputs/reports/platform/weekly_report.md` + email

### Monthly Refit (Development/Tuning)
- **Script:** `scripts/build_platform_data.py` (M1 fetch), then ad-hoc notebook or script for labeling/nowcaster/backtest
- **Notebook:** `P1_data_spine.ipynb` (M1), `P3_regime_labeling.ipynb` (M5), `P4_nowcaster.ipynb` (M6), `P6_backtest_evaluation.ipynb` (M3+M7)
- **Responsibility:** Grid search, evaluation, registry logging, trial selection
- **Output:** Checkpoint files, trial rows in `registry/trials.jsonl`

### Trial Registry & Evaluation
- **Script:** `scripts/run_policy_trials.py` (phase tuning per ADR-0004 budgets)
- **Responsibility:** Execute trial configuration, log results
- **Output:** `registry/trials.jsonl` (one row per config)

### Diagnostics & Plotting
- **Plotting module:** `src/trading_crab_lib/platform/plotting/`
- **Used by:** All notebooks (P1–P9), reporting (scoreboard)
- **Responsibility:** Consistent visuals + diagnostics across layers

## Architectural Constraints

- **Single-threaded:** No async, no multiprocessing. Network fetches serial.
- **Monthly-discrete:** All logic works on monthly periods; weekly runs re-use monthly belief (no re-filtering within a month).
- **Causal-only in L2–L4:** No forward-looking features below L1; embargo last 6–12 months of labels from CV.
- **Checkpoint boundary:** Platform namespace (`data/checkpoints/platform/`) separate from frozen incumbent; no cross-imports of platform code into incumbent's legacy pipeline.
- **Parked boundary:** Active code under `report/` and `tripwire/` (serving path) cannot import `parked/` or `allocation/joint_tilt`, `evaluation/{churn,dependence}` (enforced by `test_platform_parked_boundary.py` AST scan + fresh-interpreter load test).
- **Module non-re-exports:** Explicit imports only (e.g., `from trading_crab_lib.platform.config import load_platform_config`), no top-level `__init__.py` re-exports (D-02, avoids accidental circular imports as subpackage grows).

## Anti-Patterns

### Look-Ahead Bias

**What happens:** Using L1 labels or future features to score L2–L4.
**Why it's wrong:** L1 sees the entire history (two-sided smoother); L2–L4 refits must use only data available at that date (causal). Training nowcaster on centered features and scoring on causal data is training/serve skew.
**Do this instead:** Keep `monthly_features` (causal, derived from features computed without future) separate from any label-dependent preprocessing. Every L2–L4 model trains only on its `feature_names_in_` columns as-of each train date.

### Publishing L1 Labels to Training Data Without Embargo

**What happens:** Fit nowcaster on last 12 months' labels, which were computed with forward smoothing; refresh labels, labels change, nowcaster breaks.
**Why it's wrong:** L1 labels are unstable at the sample edge (backward window missing). Embedding them in training without embargo means the model trains on data that changes every refresh.
**Do this instead:** Embargo (exclude) last 6–12 months of L1 labels from CV; use only older, stable labels. This is the `gap_lag.py` guard.

### Confusing Monthly Belief State with Immediate Nowcast

**What happens:** Re-filter within the same month; the filter counts the same month's evidence again, inflating confidence.
**Why it's wrong:** One month = one filter step. Re-running within a month should re-use the same belief (only allocation band gets re-computed against the same held weights).
**Do this instead:** Check `regime_belief`'s as-of date. If current month ≤ belief date, return belief unchanged (no re-filter). Only advance and filter on a new month.

### Trading Without Recording (No Hysteresis Persistence)

**What happens:** Compute target weights, execute, forget held weights; next week compute target from scratch. Random week-to-week noise can cause churn.
**Why it's wrong:** Hysteresis (no-trade band) only works if the held book persists. Without it, the band is useless.
**Do this instead:** Persist `executed_weights` after every rebalance (save-after-load ordering, Pitfall 3). Load it before computing this week's band.

## Error Handling

- **Config validation:** Collect all errors (missing sections, bad types), raise once with full list (collect-all-errors-then-raise-once, matching incumbent pattern).
- **Missing checkpoints:** `load()` raises `FileNotFoundError` if checkpoint missing (caller decides: recompute or exit).
- **Network ingestion failures:** Caught, logged at WARNING; pipeline continues with empty/partial data. Monitoring via completeness report.
- **Stale data:** Per-series staleness vs `publication_lags` + grace days; warning banner in report (never refuses).
- **No silent failures:** All exceptions explicit by type; no bare `except:` clauses.

## Cross-Cutting Concerns

- **Logging:** Each module `log = logging.getLogger(__name__)` at top; root configured once at startup.
- **Data lineage:** Checkpoints tagged with config hash, row/col counts, timestamp; provenance logged (M1 splice provenance → JSON on disk).
- **Trial registry:** Every hyperparameter combination is a trial row in `registry/trials.jsonl`; DSR gates acceptance.
- **Honesty discipline:** Walk-forward only; purged CV with embargo; HOLDOUT (2021+) locked and evaluated once.

---

*Architecture analysis: 2026-10-05*
