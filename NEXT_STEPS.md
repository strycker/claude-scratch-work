# NEXT_STEPS.md — Phase 5+ Planning Document

Created: 2026-03-27
Branch: `claude/analyze-and-plan-5JtZM`

---

## Current State Summary

**What's done (Phases 1–4):**
- 9,899 lines of library code across 31 modules in `src/trading_crab_lib/`
- 573 tests (all passing, 11 skipped for optional deps)
- 9 pipeline steps running end-to-end
- 12 Jupyter notebooks with full diagnostic cells
- 85 monitoring/plotting items delivered (Phase 4 complete)
- Package renamed to `trading_crab_lib`, pyproject.toml ready
- HMM, Markov, GMM, DBSCAN, Spectral clustering all implemented
- Divergence + momentum features integrated
- Email + weekly report automation
- Preservation checkpoints, env var overrides, convenience imports

**Three repos in play:**
| Repo | Purpose | State |
|------|---------|-------|
| `claude-scratch-work` (this) | Development sandbox | 9.9K lines, 573 tests, fully featured |
| `trading-crab` (submodule: `trading-crab-repo-copy/`) | Human-validated production repo | Notebooks + data only; `src/python/` deleted in cleanup |
| `trading-crab-lib` (submodule: `trading-crab-lib-repo-copy/`) | PyPI library package | Earlier snapshot of `src/trading_crab_lib/` (missing: HMM, Markov, momentum, divergence, monitoring, macrotrends, gradient_boosting) |

---

## Big Picture: What We're Building Toward

```
┌─────────────────────────────────────────────────────────────┐
│  trading-crab-lib (PyPI package)                            │
│  pip install trading-crab-lib                               │
│  import trading_crab_lib as crab                            │
│                                                             │
│  Pure library: transforms, clustering, prediction,          │
│  regime detection, asset returns, reporting, diagnostics    │
│  NO pipelines, NO notebooks, NO scripts, NO data files     │
└──────────────────────┬──────────────────────────────────────┘
                       │ depends on
┌──────────────────────┴──────────────────────────────────────┐
│  trading-crab (application repo)                            │
│  git clone strycker/trading-crab                            │
│                                                             │
│  Pipelines, notebooks, config, scripts, run_pipeline.py     │
│  Does: pip install trading-crab-lib                         │
│  Then: import trading_crab_lib as crab                      │
│  Data lives here. Config lives here. Plots live here.       │
└─────────────────────────────────────────────────────────────┘
```

**Migration strategy:** Incrementally move validated code from `claude-scratch-work`
→ `trading-crab-lib` (library) and `trading-crab` (app), keeping both working at
each step. `claude-scratch-work` remains the development sandbox.

---

## Phase 5: Library Simplification & PyPI Prep

**Goal:** Get `trading_crab_lib` into a clean, publishable state.

### Phase 5A: Code Audit & Cleanup (prerequisite for everything)

Each task is a standalone PR-sized unit of work.

#### 5A.1 — Dead code removal scan (S)
- Grep for unused imports, unreachable functions, commented-out blocks
- Check which public functions have zero callers outside tests
- Remove dead code; run tests to confirm nothing breaks
- **Output:** Leaner codebase, list of removed items

#### 5A.2 — plotting.py decomposition (M)
- `plotting.py` is 2,018 lines — 20% of the entire library
- Split into logical submodules: `plotting/pca.py`, `plotting/regime.py`,
  `plotting/prediction.py`, `plotting/assets.py`, `plotting/diagnostics.py`,
  `plotting/core.py` (shared helpers, palette, `_save_or_show`)
- Re-export everything from `plotting/__init__.py` for backward compat
- Run full test suite after each file move
- **Output:** `src/trading_crab_lib/plotting/` package with ~6 files

#### 5A.3 — monitoring.py decomposition (S)
- 648 lines covering steps 1–9 validation, dataclasses, formatters
- Split into `monitoring/ingestion.py`, `monitoring/clustering.py`,
  `monitoring/prediction.py`, `monitoring/pipeline.py`
- Re-export from `monitoring/__init__.py`
- **Output:** `src/trading_crab_lib/monitoring/` package

#### 5A.4 — Dependency audit (S)
- Classify deps as core vs optional
- Core: pandas, numpy, scikit-learn, scipy, pyyaml, pyarrow
- Optional: hmmlearn, statsmodels, lightgbm, hdbscan, k-means-constrained,
  matplotlib, seaborn, yfinance, fredapi, lxml
- Add proper `extras_require` groups in pyproject.toml:
  - `pip install trading-crab-lib` → core only
  - `pip install trading-crab-lib[plotting]` → + matplotlib, seaborn
  - `pip install trading-crab-lib[ingestion]` → + fredapi, lxml, yfinance, requests
  - `pip install trading-crab-lib[hmm]` → + hmmlearn, statsmodels
  - `pip install trading-crab-lib[all]` → everything
- **Output:** Updated pyproject.toml with optional dependency groups

#### 5A.5 — Type hint completeness pass (S)
- Ensure all public functions have full type annotations
- Add `py.typed` marker (already exists)
- Fix any `Any` types that should be specific
- **Output:** Cleaner API for IDE autocomplete

#### 5A.6 — Docstring pass for public API (M)
- Add one-line docstrings to all public functions that lack them
- Focus on parameters and return types
- Skip private/internal functions
- **Output:** Usable `help(crab.clustering.fit_clusters)` etc.

### Phase 5B: PyPI Publishing Infrastructure

#### 5B.1 — Version & metadata finalization (S)
- Bump version to 0.2.0 (or 1.0.0-alpha)
- Add proper classifiers, keywords, project URLs
- Add `[project.urls]` for Homepage, Documentation, Issues
- Verify `MANIFEST.in` excludes tests, notebooks, data, legacy, submodules
- **Output:** pyproject.toml ready for `python -m build`

#### 5B.2 — Build & test locally (S)
- `python -m build` → check wheel and sdist
- `pip install dist/trading_crab_lib-*.whl` in a clean venv
- `python -c "import trading_crab_lib as crab; print(crab.load)"` → verify
- **Output:** Confirmed installable wheel

#### 5B.3 — CI/CD for library (S)
- GitHub Actions workflow: lint + test on Python 3.10, 3.11, 3.12
- Publish to TestPyPI on push to `main`
- Publish to PyPI on tagged release
- **Output:** `.github/workflows/ci.yml` and `publish.yml`

---

## Phase 6: Incremental Migration to `trading-crab-lib` Repo

**Strategy:** Copy validated modules one-at-a-time from `claude-scratch-work`
into the `trading-crab-lib` repo. Each migration is a single PR with tests.

### Phase 6A: Core modules first (no external deps)

These modules have zero network/API dependencies and form the foundation.

#### 6A.1 — Migrate core infrastructure (S)
- `__init__.py`, `config.py`, `runtime.py`, `checkpoints.py`
- These are the foundation everything else imports
- Include `config/settings.yaml` and `config/regime_labels.yaml`

#### 6A.2 — Migrate transforms pipeline (S)
- `transforms.py` + `momentum.py` + `divergence.py`
- All pure pandas/numpy operations, no network
- Include relevant tests

#### 6A.3 — Migrate clustering suite (M)
- `clustering.py`, `gmm.py`, `density.py`, `spectral.py`,
  `cluster_comparison.py`
- Optional deps: hdbscan, kneed
- Include tests

#### 6A.4 — Migrate regime + prediction (M)
- `regime.py`, `prediction/__init__.py`, `prediction/classifier.py`,
  `prediction/gradient_boosting.py`
- Optional dep: lightgbm
- Include tests

#### 6A.5 — Migrate analysis modules (S)
- `asset_returns.py`, `reporting.py`, `diagnostics.py`, `tactics.py`
- Include tests

#### 6A.6 — Migrate HMM + Markov (S)
- `hmm.py`, `markov.py`
- Optional deps: hmmlearn, statsmodels
- Include tests

### Phase 6B: Ingestion modules (network-dependent)

#### 6B.1 — Migrate ingestion package (M)
- `ingestion/__init__.py`, `fred.py`, `multpl.py`, `macrotrends.py`,
  `assets.py`, `grok.py`
- Include HTTP-mocked tests
- Requires: fredapi, lxml, yfinance, requests, certifi

### Phase 6C: Visualization modules

#### 6C.1 — Migrate plotting package (M)
- After 5A.2 decomposition, migrate the `plotting/` package
- Include `monitoring/` package
- Requires: matplotlib, seaborn

### Phase 6D: Email & automation

#### 6D.1 — Migrate email module (S)
- `email.py`
- Include tests

---

## Phase 7: Migration to `trading-crab` Application Repo

**Goal:** The app repo (`trading-crab`) does `pip install trading-crab-lib` and
contains only: pipelines, notebooks, config, scripts, data, run_pipeline.py.

### Phase 7A: Scaffold the app repo

#### 7A.1 — Create app structure (S)
- Add `pyproject.toml` or `requirements.txt` with `trading-crab-lib` dependency
- Migrate `config/settings.yaml` and `config/regime_labels.yaml`
- Migrate `run_pipeline.py`
- Migrate `pipelines/01_ingest.py` through `pipelines/09_tactics.py`
- Update all imports: `from trading_crab_lib import ...`

#### 7A.2 — Migrate notebooks (M)
- Copy notebooks 01–12
- Update imports to use installed `trading_crab_lib`
- Verify each notebook runs

#### 7A.3 — Migrate scripts (S)
- `scripts/run_weekly_report.py`, `scripts/setup.sh`
- `config/email.example.yaml`

#### 7A.4 — Migrate legacy reference (S)
- `legacy/unified_script.py` stays in app repo (not library)

---

## Phase 8: New Features (Post-Migration)

These can be done in `claude-scratch-work` first, then migrated.

### Tier 1 — High impact, achievable soon

#### 8.1 — LightGBM flat-API integration (M)
- Currently: LightGBM exists in `prediction/gradient_boosting.py` and
  bundle API (`classifier.py`) but NOT in flat API (`prediction/__init__.py`)
- Task: Add `train_lightgbm()` to flat API, wire into `pipelines/05_predict.py`
  and `run_pipeline.py`, save as `outputs/models/lightgbm_regime.pkl`
- Include in model comparison bar chart
- **Prerequisite:** None

#### 8.2 — Additional FRED series: INDPRO, PAYEMS, DPCERA3Q086SBEA (S)
- INDPRO = Industrial Production Index (monthly → quarterly)
- PAYEMS = Total Nonfarm Payrolls (monthly → quarterly)
- DPCERA3Q086SBEA = Real PCE (quarterly, already aligned)
- Add to `config/settings.yaml` under `fred.series`
- Add to `initial_features` list (not clustering until evaluated)
- Update ingestion completeness expected columns
- **Prerequisite:** None

#### 8.3 — Conference Board LEI proxy from FRED (S)
- Construct a composite leading indicator from existing FRED series:
  UNRATE (inverted), T10Y2Y (yield curve), M2SL (money supply), INDPRO, PAYEMS
- Standardize each, equal-weight average → `lei_proxy`
- Add to `transforms.py` or new `indicators.py`
- **Prerequisite:** 8.2 (needs INDPRO, PAYEMS)

### Tier 2 — High value, more effort

#### 8.4 — SMOTE / class-weight tuning for imbalanced regimes (S)
- Balanced clustering helps, but supervised models still see unequal class
  distribution in TimeSeriesSplit folds
- Options: (a) `class_weight="balanced"` in RF/DT/LGBM, (b) SMOTE via
  `imblearn.over_sampling.SMOTE`, (c) both
- Evaluate via CV accuracy per fold and per class
- Add `prediction.class_balance_method` to settings.yaml
- **Prerequisite:** None

#### 8.5 — Per-asset regime probability models (L)
- Currently: one RF predicts the regime, then a lookup table maps regime → asset ranking
- Proposed: per-asset binary classifiers predicting P(asset outperforms | features)
- Train one model per ETF using same features as regime classifier
- Blend with regime-based ranking for final signal
- **Prerequisite:** 8.1 (LightGBM for better per-asset models)

### Tier 3 — Longer-term

#### 8.6 — Backtest framework (XL)
- Walk-forward backtest: at each quarter, retrain model on all prior data,
  predict regime, construct portfolio, measure returns
- Compare vs buy-and-hold SPY, 60/40, equal-weight
- Sharpe ratio, max drawdown, Calmar ratio
- **Prerequisite:** 8.1, 8.4

#### 8.7 — Interactive Streamlit dashboard (L)
- Replace static CSV/terminal dashboard with browser-based UI
- Tabs: Current Regime, Regime History, Asset Rankings, Portfolio, Diagnostics
- Use `trading_crab_lib` as the backend
- **Prerequisite:** Phase 6 (library on PyPI)

#### 8.8 — Weekly automated report with AI narrative (XL)
- Extend existing email module
- Use Claude API to generate natural-language market commentary
- Input: current regime, transition probabilities, asset signals
- Output: 2-paragraph narrative + table + inline plots
- **Prerequisite:** Phase 7 (app repo), 8.7 (dashboard for data)

---

## Recommended Execution Order

The key insight: **Claude Code works best with small, focused tasks** that each
produce a testable result. Large tasks time out. Here's the optimal sequencing:

### Wave 1: Simplify (can be done in parallel)
```
5A.1  Dead code removal scan                    [S, standalone]
5A.4  Dependency audit + optional groups         [S, standalone]
5A.5  Type hint completeness                     [S, standalone]
```

### Wave 2: Decompose large modules
```
5A.2  plotting.py → plotting/ package            [M, after 5A.1]
5A.3  monitoring.py → monitoring/ package         [S, after 5A.1]
```

### Wave 3: PyPI prep
```
5A.6  Docstring pass                             [M, after 5A.2/5A.3]
5B.1  Version & metadata finalization            [S, after 5A.4]
5B.2  Build & test locally                       [S, after 5B.1]
5B.3  CI/CD for library                          [S, after 5B.2]
```

### Wave 4: New features (can start during Wave 2-3)
```
8.2   Additional FRED series                     [S, standalone]
8.1   LightGBM flat-API integration              [M, standalone]
8.4   SMOTE / class-weight tuning                [S, standalone]
8.3   Conference Board LEI proxy                 [S, after 8.2]
```

### Wave 5: Migration (after Wave 3)
```
6A.1  Core infrastructure → trading-crab-lib     [S]
6A.2  Transforms pipeline → trading-crab-lib     [S]
6A.3  Clustering suite → trading-crab-lib        [M]
6A.4  Regime + prediction → trading-crab-lib     [M]
6A.5  Analysis modules → trading-crab-lib        [S]
6A.6  HMM + Markov → trading-crab-lib            [S]
6B.1  Ingestion package → trading-crab-lib       [M]
6C.1  Plotting package → trading-crab-lib        [M]
6D.1  Email module → trading-crab-lib            [S]
```

### Wave 6: App repo (after Wave 5)
```
7A.1  Scaffold trading-crab app                  [S]
7A.2  Migrate notebooks                          [M]
7A.3  Migrate scripts                            [S]
7A.4  Migrate legacy reference                   [S]
```

### Wave 7: Advanced features (after Wave 4-5)
```
8.5   Per-asset regime probability models        [L]
8.6   Backtest framework                         [XL]
8.7   Streamlit dashboard                        [L]
8.8   AI narrative weekly report                 [XL]
```

---

## Size Guide

| Size | Meaning | Claude Code sessions | Risk of timeout |
|------|---------|---------------------|-----------------|
| S | Small: 1-3 files, <200 lines changed | 1 session | Low |
| M | Medium: 3-8 files, 200-600 lines | 1-2 sessions | Medium |
| L | Large: 8-15 files, 600-1500 lines | 2-4 sessions | High — split further |
| XL | Extra large: 15+ files, 1500+ lines | 4+ sessions | Must be split into S/M pieces |

**For Claude Code reliability:** Always break L/XL tasks into S/M subtasks before
starting implementation. Each subtask should be independently committable and testable.

---

## Critical Constraints (Do Not Forget)

1. **Submodules are read-only** — never modify files in `*-repo-copy/` directories
2. **`legacy/` is read-only** — the algorithm ground truth
3. **Feature pipeline order is sacred** — cross-ratios → log → select → gap-fill → deriv → select
4. **`balanced_cluster` is the primary regime assignment** — not `cluster`
5. **Two prediction APIs must coexist** — flat (production) vs bundle (tests)
6. **`current_regime.pkl` is always a bare RandomForestClassifier**
7. **GDP and GNP are always shifted +1Q**
8. **PCA = 5 components** — do not change without benchmarking
9. **No committed secrets** — `.env`, API keys, email configs stay local
10. **Tests must pass after every change** — `pytest tests/ -v`

---

## Open Questions for Owner Decision

1. **Library vs monorepo:** Should `trading-crab-lib` be a separate Git repo
   (current setup) or a subdirectory of `trading-crab`? Separate repo = cleaner
   PyPI publishing. Monorepo = simpler development workflow.

2. **Version numbering:** Start at 0.2.0 (incremental from current 0.1.0) or
   jump to 1.0.0-alpha to signal "feature complete for core pipeline"?

3. **Migration order priority:** Start with library cleanup (Phase 5) or new
   features (Phase 8) first? Recommendation: Phase 5 first — a clean library
   makes features easier to add.

4. **Notebook migration:** Keep notebooks in `trading-crab` app repo (current plan)
   or create a separate `trading-crab-notebooks` repo? Notebooks are large (some
   >1MB with output cells) and don't belong in a PyPI package.

5. **What to do with `claude-scratch-work` after migration?** Archive it? Keep as
   development sandbox? It has 573 tests and full history.

6. **Finviz data in `trading-crab-repo-copy/data/`:** 25 monthly CSVs of Finviz
   snapshots. Should these be incorporated into the pipeline? They're stock-level
   data (not macro), which is a different use case.
