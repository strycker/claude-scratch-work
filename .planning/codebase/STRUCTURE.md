# Codebase Structure

**Analysis Date:** 2026-10-05

> **Update 2026-10-05 (after this map was written; DECISIONS G-13 and P-07).** These were **deleted**:
> - `platform/parked/` (classifier #2, the joint driver, the stability suite);
> - the parked-only helpers `allocation/joint_tilt.py`, `evaluation/dependence.py` and the `features/` package;
> - the `labeling_2` config block;
> - the research scripts `run_joint_lift`, `run_subsample_stability`, `terminal_month_diagnostic`,
>   `joint_lift_diagnostics` and `diagnose_s1_truncation`;
> - the one-off scripts `diagnose_yahoo_tls`, `diagnose_yfinance`, `diagnose_cpi_handoff`, `run_policy_trials`,
>   `smoke_step5.sh` and `egress_test.sh`;
> - the parked-boundary and doc-count tests.
>
> Root `CLAUDE.md` was trimmed; the old text is in `docs/archive/LEGACY-CLAUDE.md`. Wherever this map mentions any
> of these, read it as history.


## Directory Layout

```
trading-crab/ (repo root)
│
├── .claude/, .codex/, .cursor/          # Editor configs (ignore)
├── .github/                              # GitHub Actions CI/CD workflows
├── gsd-scratch-work/                     # Git submodule: GSD-framework version (read-only reference)
├── trading-crab/                         # Git submodule: public PyPI version (read-only reference)
├── trading-crab-lib/                     # Git submodule: library PyPI version (read-only reference)
│
├── config/                               # Configuration files
│   ├── settings.yaml                     # Frozen incumbent quarterly pipeline config (D-02: read-only)
│   ├── platform_settings.yaml            # Platform monthly pipeline config (M0, D-02)
│   ├── email.example.yaml                # Email template (optional, read-only)
│   ├── regime_labels.yaml                # Manually pinned regime names (D-02: not platform)
│   ├── portfolio.yaml                    # ETF allocation blueprint (D-02: not platform)
│   └── accounts/                         # Account-specific settings (Glenn's Fidelity holdings)
│
├── data/                                 # Data storage (gitignored except checkpoints)
│   ├── raw/                              # Raw ingestion (gitignored; sourced live)
│   ├── checkpoints/
│   │   ├── <incumbent-quarterly>/        # Frozen incumbent step outputs (read-only)
│   │   └── platform/                     # Platform phase outputs (tracked in git, safe to refresh)
│   │       ├── monthly_raw.parquet       # M1 output: raw monthly series
│   │       ├── monthly_features.parquet  # M2 output: causal features
│   │       ├── regime_labels.parquet     # M5 output: hard regime assignments
│   │       ├── nowcaster_model.pkl       # M6 output: fitted RandomForest
│   │       ├── regime_belief.parquet     # Weekly state: current posterior belief
│   │       ├── executed_weights.parquet  # Weekly state: post-hysteresis book
│   │       └── <other>_*.parquet         # Intermediate checkpoints (profiles, matrices, etc.)
│   ├── holdout/                          # 2021-01-01 onward (M8, locked, evaluated once)
│   └── snapshots/                        # Daily snapshots (price updates)
│
├── legacy/                               # Frozen incumbent quarterly pipeline (do NOT modify)
│   └── unified_script.py                 # THE reference: all legacy logic must reach this
│
├── src/trading_crab/                     # App package (pip name: trading-crab)
│   ├── __init__.py
│   ├── cli.py                            # Entry point: tradingcrab CLI
│   └── pipeline.py                       # Incumbent quarterly pipeline orchestration
│
├── src/trading_crab_lib/                 # Library package (pip name: trading-crab-lib)
│   ├── __init__.py                       # Path resolution, convenience imports
│   ├── config.py                         # Incumbent quarterly config (frozen)
│   ├── runtime.py                        # RunConfig dataclass
│   ├── checkpoints.py                    # CheckpointManager (frozen incumbent's)
│   ├── <incumbent modules>.py            # Frozen: transforms, clustering, prediction, reporting, etc.
│   ├── ingestion/                        # Frozen incumbent fetchers (fred, multpl, macrotrends, assets)
│   ├── prediction/                       # Frozen incumbent classifiers + bundle API
│   ├── plotting/                         # Frozen incumbent visualization
│   └── platform/                         # NEW: Platform monthly pipeline (active development)
│       ├── __init__.py
│       ├── config.py                     # Platform config loader (independent schema)
│       ├── checkpoints.py                # Platform-namespace checkpoint manager
│       ├── snapshots.py                  # Daily snapshot persistence
│       ├── taxonomy.py                   # Feature tier classification (fast/slow/agency)
│       ├── splice.py                     # Data splicing engine (ratio_splice, TR synthesis)
│       ├── transforms_monthly.py         # Resample, publication lags, monthly features
│       ├── ingestion/                    # M1: Data sources
│       │   ├── __init__.py
│       │   ├── alfred.py                 # ALFRED macro (vintage-corrected)
│       │   ├── macro_monthly.py          # FRED monthly aggregates
│       │   ├── macro_daily.py            # FRED daily rates
│       │   ├── prices_daily.py           # Equity/ETF prices (yfinance)
│       │   ├── eodhd.py, tiingo.py       # Alternative price sources
│       │   ├── norgate.py                # Norgate price data
│       │   └── publication_lags.py       # Apply D-07 publication-lag shifts
│       ├── labeling/                     # M5: Regime labeling (L1)
│       │   ├── __init__.py
│       │   ├── jump_model.py             # Preferred: k-means + jump-penalty
│       │   └── diagnostics.py            # Occupancy, sojourn, stability validation
│       ├── features/                     # Feature engineering (M2 details)
│       │   ├── __init__.py
│       │   ├── invariants.py             # Causal feature pipeline
│       │   └── relative.py               # Relative ratios (parked-helper)
│       ├── prediction/                   # M6: Nowcaster + filter (L2)
│       │   ├── __init__.py
│       │   ├── nowcaster.py              # P(regime | causal features)
│       │   ├── regime_filter.py          # Bayesian filter, belief persistence
│       │   └── transition_matrix.py      # Empirical transition matrix
│       ├── assets/                       # L3: Asset returns + vol (M3, M7)
│       │   ├── __init__.py
│       │   ├── returns.py                # Quarterly returns by regime
│       │   └── vol.py                    # Vol, covariance, GARCH/EWMA
│       ├── allocation/                   # L4: Weights (M4, M7)
│       │   ├── __init__.py
│       │   ├── tilt.py                   # Vol-targeted regime-conditional weights
│       │   ├── hysteresis.py             # No-trade band execution
│       │   └── joint_tilt.py             # Parked-helper (joint classifier)
│       ├── backtest/                     # Walk-forward backtest + baselines
│       │   ├── __init__.py
│       │   ├── driver.py                 # Monthly rebalance loop, ablations
│       │   ├── baselines.py              # 60/40, Faber, vol-parity
│       │   ├── costs.py                  # Rebalance costs
│       │   └── joint_driver.py           # Parked: joint-classifier backtest
│       ├── report/                       # L4: Output & serve (M4)
│       │   ├── __init__.py
│       │   ├── weekly.py                 # Markdown report assembly + email
│       │   ├── serving.py                # Live model serving, cold-start belief
│       │   ├── holdings.py               # Account-specific holdings
│       │   ├── scoreboard.py             # Asset performance heatmap (added 08.2-03)
│       │   └── deduce_live.py            # Infer current holdings (dev tool)
│       ├── evaluation/                   # Metrics, DSR, model comparison
│       │   ├── __init__.py
│       │   ├── kpis.py                   # Sharpe, Calmar, max DD
│       │   ├── deflated_sharpe.py        # Multiple-testing adjustment + quality tier (M8)
│       │   ├── model_metrics.py          # Per-model CV diagnostics
│       │   ├── report.py                 # High-level evaluation report
│       │   ├── sojourn_lag.py            # Regime duration analysis
│       │   ├── churn.py                  # Label refresh churn (parked-helper)
│       │   └── dependence.py             # State dependence analysis (parked-helper)
│       ├── honesty/                      # M8: Trial registry, CV, holdout
│       │   ├── __init__.py
│       │   ├── cv.py                     # TimeSeriesSplit, embargo
│       │   ├── gap_lag.py                # 6–12 month embargo for L1 labels
│       │   ├── holdout.py                # 2021-01-01 onward lockdown
│       │   ├── registry.py               # Trial row persistence + DSR check
│       │   ├── walkforward.py            # Walk-forward split definitions
│       │   └── gating.py                 # Quality tier decision logic
│       ├── tripwire/                     # M9: Monitoring (advisory post-08.2-03)
│       │   ├── __init__.py
│       │   └── monitor.py                # Weekly credit spread alerts (DAAA/DBAA)
│       ├── plotting/                     # Diagnostics + visuals per layer
│       │   ├── __init__.py
│       │   ├── core.py                   # Base plotting utilities (colors, layout)
│       │   ├── data.py                   # M1 ingestion visuals
│       │   ├── features.py               # M2 feature diagnostics
│       │   ├── regime.py                 # M5 labeling visuals
│       │   ├── nowcaster.py              # M6 nowcaster diagnostics
│       │   ├── backtest.py               # M3/M7 backtest equity curves
│       │   ├── allocation.py             # M4/M7 weight heatmaps
│       │   ├── history.py                # Historical comparison plots
│       │   ├── drift.py                  # Model drift analysis
│       │   └── loaders.py                # Checkpoint → plotting data
│       └── parked/                       # Deferred (out of weekly path)
│           ├── __init__.py
│           ├── classifier2.py            # Parked: Leadership/relative classifier (L1-04)
│           ├── joint_driver.py           # Parked: Two-classifier backtest (E-02/E-03)
│           └── stability.py              # Parked: Subsample stability suite (G-07)
│
├── notebooks/                            # Exploration & diagnostics
│   ├── platform/                         # Platform pipeline diagnostics (P1–P9)
│   │   ├── P1_data_spine.ipynb           # M1: Ingestion, splice validation
│   │   ├── P2_features_taxonomy.ipynb    # M2: Feature engineering, tier validation
│   │   ├── P3_regime_labeling.ipynb      # M5: Jump model grid search, profiles
│   │   ├── P4_nowcaster.ipynb            # M6: Classifier fit, CV diagnostics
│   │   ├── P5_assets_allocation.ipynb    # M3: Baseline returns, vol targeting
│   │   ├── P6_backtest_evaluation.ipynb  # M3/M7: Walk-forward curves vs SPY
│   │   ├── P7_serving_report.ipynb       # M4: Weekly report assembly (added 08.2-03)
│   │   ├── P8_filtered_belief.ipynb      # M6: Bayesian filter, belief evolution (added 08.2-03)
│   │   └── P9_does_regime_pay.ipynb      # M7: Regime tilt vs baseline ablation (added 08.2-03)
│   └── <incumbent>/                      # Frozen quarterly pipeline notebooks (read-only)
│
├── scripts/                              # Automation & utilities
│   ├── build_platform_data.py            # M1: Fetch all sources → checkpoints (main entry)
│   ├── run_weekly_report.py              # Serve: Load state → report → email (cron target)
│   ├── run_policy_trials.py              # Execute trial configuration (phase budget sweep)
│   ├── recompute_monthly_features.py     # M2: Re-engineer features (dev)
│   ├── migrate_publication_lags.py       # D-07: Apply publication lags to raw (one-time)
│   ├── diagnose_*.py                     # Various ingestion diagnostics
│   ├── run_joint_lift.py                 # Parked: Two-classifier comparison (dev)
│   ├── run_subsample_stability.py        # Parked: Stability analysis (dev)
│   ├── terminal_month_diagnostic.py      # Parked: Month-end edge diagnostics (dev)
│   ├── joint_lift_diagnostics.py         # Parked: Joint lift analysis (dev)
│   └── platform_snapshot.py              # Checkpoint → snapshot (archive)
│
├── registry/                             # Trial registry (M8, tracked in git)
│   ├── trials.jsonl                      # One row per trial (K, λ, features, results, DSR)
│   └── archive/                          # Old trial records (reference)
│
├── outputs/                              # Runtime outputs (gitignored)
│   ├── reports/platform/                 # Weekly report markdown
│   │   └── weekly_report.md              # Glenn reads this before trading
│   ├── models/                           # Pickled model files
│   └── plots/                            # Diagnostics figures
│
├── platform_design/                      # Design documents (read-only reference)
│   ├── platform_design.md                # v1.8 design (math + architecture)
│   ├── MODULE-MAP.md                     # M0–M9+ module map + parked list (D-07)
│   ├── DECISIONS.md                      # Phase 1–8 ruling index (D-08+)
│   └── adr/                              # Architecture decision records
│       ├── 0001-l1-feature-policy.md     # L1 feature classification (D-04)
│       ├── 0002-l1-second-classifier.md  # Why classifier #2 was deferred (L1-04)
│       ├── 0003-quality-gate-tier.md     # DSR as the one quality gate (M8)
│       └── 0004-trial-budgeting-policy.md # Per-phase trial budgets (ADR-0004)
│
├── docs/                                 # Reference documentation
│   ├── splicing_rules.md                 # D-04: Splice methods per core asset (data lineage)
│   ├── archive/STATE.md                  # Historical state snapshots (legacy)
│   └── archive/                          # Frozen docs (reference only)
│
├── tests/                                # Test suite (pytest)
│   ├── unit/
│   │   ├── platform/                     # Platform unit tests (100+ test modules)
│   │   │   ├── test_platform_config.py
│   │   │   ├── test_platform_checkpoints.py
│   │   │   ├── test_taxonomy.py
│   │   │   ├── test_ingestion_*.py       # M1 ingestion tests
│   │   │   ├── test_labeling_*.py        # M5 labeling tests
│   │   │   ├── test_prediction_*.py      # M6 nowcaster tests
│   │   │   ├── test_allocation_*.py      # M4/M7 allocation tests
│   │   │   ├── test_backtest_*.py        # Backtest tests
│   │   │   ├── test_evaluation_*.py      # Metrics tests
│   │   │   ├── test_honesty_*.py         # M8 registry/DSR tests
│   │   │   ├── test_report_*.py          # M4 report tests
│   │   │   ├── test_plotting_*.py        # Plotting tests
│   │   │   ├── test_monitoring_*.py      # Tripwire tests
│   │   │   ├── test_platform_parked_boundary.py  # M9+: Active code cannot import parked
│   │   │   └── <incumbent tests>         # Frozen quarterly pipeline tests (read-only)
│   │   └── <incumbent unit tests>        # Quarterly pipeline unit tests
│   ├── integration/
│   │   ├── test_mini_pipeline.py         # End-to-end platform flow (synthetic data)
│   │   └── <incumbent integration tests>
│   └── conftest.py                       # Shared fixtures (pytest)
│
├── .gitignore                            # Untracked: .env, data/raw/, outputs/
├── pyproject.toml                        # Build config, dependencies, version
├── setup.cfg                             # setuptools config (legacy)
├── setup.py                              # Legacy entry point
├── MANIFEST.in                           # Package data files
├── Dockerfile                            # Multi-stage build (base + pipeline)
├── docker-compose.yml                    # Orchestrated services
├── CLAUDE.md                             # Developer guide (comprehensive)
├── README.md                             # User guide
├── ROADMAP.md                            # Feature backlog (prioritized)
├── run_pipeline.py                       # Backward-compat shim (quarterly pipeline CLI)
├── Makefile                              # Common dev shortcuts
├── requirements.txt                      # Pinned dependencies (legacy incumbent)
└── requirements-dev.txt                  # Dev extras (legacy)
```

## Key File Locations

**Configuration:**
- `config/platform_settings.yaml` — All platform tuneable parameters
- `config/settings.yaml` — Frozen incumbent settings (do not edit)
- `config/portfolio.yaml` — ETF allocation blueprint
- `.env` (gitignored) — Secrets (FRED_API_KEY, SMTP credentials)

**Data Checkpoints:**
- `data/checkpoints/platform/monthly_raw.parquet` — M1 output (raw monthly)
- `data/checkpoints/platform/monthly_features.parquet` — M2 output (causal features)
- `data/checkpoints/platform/regime_labels.parquet` — M5 output (hard labels)
- `data/checkpoints/platform/nowcaster_model.pkl` — M6 output (fitted classifier)
- `data/checkpoints/platform/regime_belief.parquet` — Weekly state (posterior)
- `data/checkpoints/platform/executed_weights.parquet` — Weekly state (held book)

**Design & Decisions:**
- `platform_design/platform_design.md` — Design document (L0–L4, math + architecture)
- `platform_design/MODULE-MAP.md` — Module map (M0–M9+), parked list, boundary test
- `platform_design/DECISIONS.md` — Phase 1–8 decisions, all rulings indexed
- `platform_design/adr/` — ADRs 0001–0004 (policy + exceptions)

**Core Libraries:**
- `src/trading_crab_lib/platform/config.py` — Config loader + validation
- `src/trading_crab_lib/platform/checkpoints.py` — Checkpoint manager
- `src/trading_crab_lib/platform/taxonomy.py` — Feature tier classification
- `src/trading_crab_lib/platform/splice.py` — Data splicing engine

**Layer Implementations:**
- M1: `src/trading_crab_lib/platform/ingestion/`, `transforms_monthly.py`
- M2: `src/trading_crab_lib/platform/features/`
- M3/M7: `src/trading_crab_lib/platform/assets/`, `allocation/`, `backtest/`
- M4: `src/trading_crab_lib/platform/report/`
- M5: `src/trading_crab_lib/platform/labeling/`
- M6: `src/trading_crab_lib/platform/prediction/`
- M8: `src/trading_crab_lib/platform/honesty/`, `evaluation/`
- M9+: `src/trading_crab_lib/platform/tripwire/`, `parked/`

**Scripts:**
- `scripts/build_platform_data.py` — M1 data fetch (main entry)
- `scripts/run_weekly_report.py` — M4 serve (production)
- `scripts/run_policy_trials.py` — Trial execution (tuning)

**Tests:**
- `tests/unit/platform/` — Platform unit tests (100+ modules)
- `tests/integration/` — End-to-end flow (synthetic data)
- `tests/unit/test_platform_parked_boundary.py` — Parked code guard (critical)

**Notebooks:**
- `notebooks/platform/P1_data_spine.ipynb` — M1 diagnostics
- `notebooks/platform/P2_features_taxonomy.ipynb` — M2 diagnostics
- `notebooks/platform/P3_regime_labeling.ipynb` — M5 diagnostics
- `notebooks/platform/P4_nowcaster.ipynb` — M6 diagnostics
- `notebooks/platform/P5_assets_allocation.ipynb` — M3 diagnostics
- `notebooks/platform/P6_backtest_evaluation.ipynb` — M3/M7 diagnostics
- `notebooks/platform/P7_serving_report.ipynb` — M4 serve (added 08.2-03)
- `notebooks/platform/P8_filtered_belief.ipynb` — M6 filter (added 08.2-03)
- `notebooks/platform/P9_does_regime_pay.ipynb` — M7 ablation (added 08.2-03)

**Output:**
- `outputs/reports/platform/weekly_report.md` — Weekly report for Glenn (M4)
- `registry/trials.jsonl` — Trial registry (M8)

## Naming Conventions

### Files

**Platform modules:**
- Subpackage: lowercase with underscores (`ingestion/`, `features/`, `labeling/`, `prediction/`, `allocation/`, `backtest/`, `report/`, `evaluation/`, `honesty/`, `tripwire/`, `plotting/`, `parked/`)
- Module: lowercase with underscores (`config.py`, `checkpoints.py`, `taxonomy.py`, `splice.py`, `transforms_monthly.py`)
- Private/parked: same convention; `parked/` subdirectory denotes deferred code

**Test files:**
- Pattern: `test_<module>.py` (e.g., `test_taxonomy.py`, `test_nowcaster.py`)
- Platform tests: `tests/unit/platform/test_<module>.py`
- Boundary test: `test_platform_parked_boundary.py` (critical: enforces AST scan + fresh-import)

**Data files:**
- Checkpoints: `<checkpoint_name>.parquet` or `.pkl` (e.g., `monthly_raw.parquet`, `nowcaster_model.pkl`)
- Registry: `trials.jsonl` (one row per trial)
- Report: `weekly_report.md`

**Notebooks:**
- Pattern: `P<stage>_<description>.ipynb` (e.g., `P1_data_spine.ipynb`, `P3_regime_labeling.ipynb`)
- Stage numbering: P1–P9 mapped to M0–M8

**Scripts:**
- Pattern: `<verb>_<noun>.py` (e.g., `build_platform_data.py`, `run_weekly_report.py`, `run_policy_trials.py`)
- Diagnostic: `diagnose_<issue>.py` (e.g., `diagnose_s1_truncation.py`)

### Code Style

**Modules:**
- Private (internal): `_<name>` prefix (e.g., `_TIERS`, `_fill_column()`)
- Public (API): no prefix (e.g., `load_platform_config()`, `fit_jump_model()`)

**Functions:**
- Verb + noun pattern: `fetch_all()`, `build_core_research_series()`, `fit_nowcaster()`
- Predicates: `is_fresh()`, `should_filter()`, `check_columns_tagged()`
- Main entry: `main()` (for scripts)

**Classes:**
- CamelCase: `CheckpointManager`, `RegimeFilter`, `TiltAllocator`

**Variables:**
- DataFrames: noun (e.g., `monthly_raw`, `regime_labels`, `executed_weights`)
- Series: noun (e.g., `weights`, `returns`, `belief`)
- Config: `cfg` (dict)
- Checkpoint manager: `cm` (CheckpointManager)

**Constants:**
- UPPER_CASE: `_REQUIRED_SECTIONS`, `_TIERS`, `_BAND_WIDTH`

## Where to Add New Code

### New Feature (within active M0–M7 path)

1. **Feature engineering (M2 enhancement):**
   - Add to `src/trading_crab_lib/platform/features/invariants.py`
   - Update `config/platform_settings.yaml` `taxonomy:` blocks (classify into fast/slow/agency)
   - Add unit test in `tests/unit/platform/test_features_invariants.py`
   - Update `notebooks/platform/P2_features_taxonomy.ipynb` to visualize

2. **New labeling method (M5 alternative):**
   - Create `src/trading_crab_lib/platform/labeling/<method>.py`
   - Implement grid search, acceptance criteria (occupancy, sojourn, stability)
   - Add diagnostics in `src/trading_crab_lib/platform/labeling/diagnostics.py`
   - Add unit tests: `tests/unit/platform/test_labeling_<method>.py`
   - Update `notebooks/platform/P3_regime_labeling.ipynb` to compare methods

3. **New nowcaster model (M6 alternative):**
   - Create `src/trading_crab_lib/platform/prediction/<model>.py`
   - Implement `fit_<model>(X, y, cfg)` → fitted model
   - Add TimeSeriesSplit CV wrapper
   - Add unit tests: `tests/unit/platform/test_prediction_<model>.py`
   - Update `notebooks/platform/P4_nowcaster.ipynb` to compare

4. **New allocation method (M7 enhancement):**
   - Create `src/trading_crab_lib/platform/allocation/<method>.py`
   - Implement weight computation (tilt, no-regime baseline, etc.)
   - Add unit tests: `tests/unit/platform/test_allocation_<method>.py`
   - Update backtest driver to dispatch on config switch
   - Update `notebooks/platform/P5_assets_allocation.ipynb`

5. **New evaluation metric (M8 enhancement):**
   - Create `src/trading_crab_lib/platform/evaluation/<metric>.py`
   - Implement walk-forward computation + gating logic
   - Add unit tests: `tests/unit/platform/test_evaluation_<metric>.py`
   - Update backtest + registry to log the new metric

### New Script (Automation)

1. **Data refresh or re-export:**
   - Create `scripts/<verb>_<noun>.py`
   - Import `build_platform_data` patterns if fetching sources
   - Checkpoint manager for I/O
   - Add `if __name__ == "__main__": main()` entry

2. **Diagnostics:**
   - Create `scripts/diagnose_<issue>.py`
   - Import relevant checkpoint loaders + plotting
   - Output to `outputs/` (no git tracking)

3. **Tuning/trials:**
   - Create `scripts/run_<experiment>.py`
   - Use `honesty/registry.py` to log trials
   - Gate on DSR, record results

### New Test

**Unit test for a module:**
- Create `tests/unit/platform/test_<module>.py`
- Use `pytest` fixtures from `conftest.py`
- Mocks for external I/O (ingestion, checkpoints)
- Example: `tests/unit/platform/test_taxonmy.py`

**Boundary/integration test:**
- Update `tests/unit/test_platform_parked_boundary.py` if parked boundary changes
- Update `tests/integration/test_mini_pipeline.py` if end-to-end flow changes

### New Notebook (Exploration)

1. **Stage-specific diagnostics (P<N>):**
   - Create `notebooks/platform/P<N>_<description>.ipynb`
   - Load checkpoints via `CheckpointManager`
   - Leverage `src/trading_crab_lib/platform/plotting/` for consistent visuals
   - Link to MODULE-MAP.md for context

### New Parked Feature (Out of Scope)

1. **Deferral (like classifier #2, stability, joint driver):**
   - Create `src/trading_crab_lib/platform/parked/<feature>.py`
   - Add to `PARKED_MODULES` in `tests/unit/test_platform_parked_boundary.py`
   - Update MODULE-MAP.md `Parked` table with reason + un-park instructions
   - Do NOT import `parked/` from active code; enforced by AST scan

## Special Directories

**data/checkpoints/platform/:**
- Purpose: Tracked checkpoints (safe to refresh via `build_platform_data.py`; unlike incumbent's gitignored raw)
- Created by: M1 ingestion, M2 features, M5 labeling, M6 nowcaster, etc.
- Loaded by: Every downstream layer (CheckpointManager)
- Git strategy: Tracked for reproducibility; refresh-safe via re-running scripts

**registry/:**
- Purpose: Trial registry (one row per hyperparameter configuration)
- Format: `trials.jsonl` (one trial per line, JSON)
- Consumed by: DSR gate (honesty/registry.py)
- Git strategy: Tracked; append-only (historical record)

**outputs/**
- Purpose: Runtime outputs (weekly report, plots, models)
- Gitignored: Not tracked
- Lifecycle: Ephemeral (overwritten weekly)

**notebooks/platform/:**
- Purpose: Exploration per stage (P1–P9)
- Linked to: MODULE-MAP.md notebook column (verification of module implementation)
- Not code: Executed by humans; checkpoint-based (safe to re-run)

## Special Files

**platform_settings.yaml:**
- Purpose: Single source of truth for platform tuneables (M0, D-02)
- Independent of incumbent `settings.yaml` (no collision risk)
- Schema validated at load: `load_platform_config()` → `validate_platform_config()`
- Sections: `data`, `fred_monthly`, `fred_vintage`, `splice`, `universe`, `taxonomy`

**MODULE-MAP.md:**
- Purpose: Module-to-file mapping (M0–M9+), parked list, acceptance criteria
- Maintained by: Phase lead after each milestone
- Consulted by: Developers implementing features (know which file to edit)
- Parked boundary: Lists all deferred code + un-park instructions

**DECISIONS.md:**
- Purpose: Index of all Phase 1–8 decisions (rulings, policy, exceptions)
- Format: One row per decision (phase, who, when, what, why, impact)
- Consulted by: Code reviewers, future phases (understand context)

**ADR/:**
- Purpose: Architectural Decision Records (policy + exceptions to design.md)
- Example: ADR-0003 "DSR is the one quality gate"; ADR-0004 "Per-phase trial budgets"
- Amends: Specific sections of platform_design.md (§4.3/§4.4, §8, etc.)

**test_platform_parked_boundary.py:**
- Purpose: Guard against accidental active → parked imports
- Mechanism: AST scan + fresh-interpreter load test
- Enforced by: CI (must pass before merge)
- Modifies when: New code enters/exits `parked/` subdirectory

---

*Structure analysis: 2026-10-05*
