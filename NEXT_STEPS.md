# NEXT_STEPS.md — Phase 5+ Planning Document

Created: 2026-03-27
Updated: 2026-03-27 (revised: monorepo 2-package architecture)
Branch: `claude/refresh-submodule-analysis-1Icoo`

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

**Three submodules in play (all read-only references):**
| Submodule path | Remote repo | Purpose |
|---|---|---|
| `gsd-scratch-work/` | strycker/gsd-scratch-work | GSD framework workspace — earlier canonical checkpoint with extensive planning docs |
| `trading-crab-lib/` | strycker/trading-crab-lib | Earlier PyPI library snapshot (missing: HMM, Markov, momentum, divergence, monitoring, macrotrends) |
| `trading-crab/` | strycker/trading-crab | Historical notebook-era reference; no Python package |

**What's missing from `trading-crab-lib` submodule vs main repo:**
`divergence.py`, `hmm.py`, `markov.py`, `momentum.py`, `monitoring.py` — all 5 are in the
main repo but not yet in the submodule's snapshot.

---

## Big Picture: What We're Building Toward

### Architecture: One Repo, Two PyPI Packages

```
claude-scratch-work/  (this git repo)
│
├── pyproject.toml                  ← "trading-crab" (APPLICATION package)
│                                      pip install trading-crab
│                                      → installs `tradingcrab` CLI
│                                      → depends on trading-crab-lib
│
├── src/
│   ├── trading_crab/               ← thin Python app layer (CLI, setup, publish)
│   │   ├── __init__.py
│   │   └── cli.py
│   │
│   └── trading_crab_lib/           ← pure library
│       ├── pyproject.toml          ← "trading-crab-lib" (LIBRARY package)
│       │                              pip install trading-crab-lib
│       │                              → no executables, pure import API
│       ├── __init__.py
│       └── ... (31 library modules)
│
├── pipelines/                      ← live here; NOT installed to site-packages
├── notebooks/                      ← live here; NOT installed to site-packages
├── scripts/                        ← live here; NOT installed to site-packages
├── config/                         ← live here; some copied by `tradingcrab setup`
├── run_pipeline.py                 ← used directly (git clone workflow); called by CLI
├── tests/                          ← tests for both packages
└── ...
```

**Why this layout?**

| Concern | Answer |
|---|---|
| AI visibility | Single repo = Claude Code sees lib + app + docs in one context window |
| Single source of truth | One CLAUDE.md, one ROADMAP.md, one test suite, no sync drift |
| Atomic changes | Change lib API and update all app callers in one commit |
| Independent installability | `pip install trading-crab-lib` pulls only library deps; no scripts, no notebooks |
| CLI entry points | `pip install trading-crab` installs `tradingcrab`, `tradingcrab-setup`, `tradingcrab-publish` |
| Industry precedent | Standard monorepo pattern (uv workspaces, Pants, many Google/Meta projects) |

**The key technical trick:** the library's `pyproject.toml` lives nested inside
`src/trading_crab_lib/` and uses `where = [".."]` to tell setuptools "packages are
in the parent directory (`src/`)". This means `pip install ./src/trading_crab_lib/`
builds and installs only the library. The root `pyproject.toml` for the app uses
`include = ["trading_crab"]` to pick up only `src/trading_crab/`, excluding the
library which manages itself.

---

## Exact Target Directory Layout

```
claude-scratch-work/
├── pyproject.toml                    ← "trading-crab" app package (root level)
├── src/
│   ├── trading_crab/                 ← thin app Python package (NEW)
│   │   ├── __init__.py
│   │   └── cli.py                    ← tradingcrab / tradingcrab-setup / tradingcrab-publish
│   └── trading_crab_lib/             ← pure library Python package (EXISTS)
│       ├── pyproject.toml            ← "trading-crab-lib" lib package (NEW — nested)
│       ├── py.typed
│       ├── __init__.py
│       ├── config.py
│       ├── runtime.py
│       ├── checkpoints.py
│       ├── transforms.py
│       ├── momentum.py
│       ├── divergence.py
│       ├── clustering.py
│       ├── cluster_comparison.py
│       ├── gmm.py
│       ├── hmm.py
│       ├── markov.py
│       ├── density.py
│       ├── spectral.py
│       ├── regime.py
│       ├── asset_returns.py
│       ├── reporting.py
│       ├── plotting.py               ← or plotting/ package after Phase 5A.2
│       ├── monitoring.py             ← or monitoring/ package after Phase 5A.3
│       ├── diagnostics.py
│       ├── tactics.py
│       ├── email.py
│       ├── ingestion/
│       │   ├── __init__.py
│       │   ├── fred.py
│       │   ├── multpl.py
│       │   ├── macrotrends.py
│       │   ├── assets.py
│       │   └── grok.py
│       └── prediction/
│           ├── __init__.py           ← flat API (production)
│           ├── classifier.py         ← bundle API (test/analysis)
│           └── gradient_boosting.py  ← LightGBM (optional)
├── pipelines/
├── notebooks/
├── scripts/
├── config/
├── run_pipeline.py
├── tests/
├── legacy/
├── data/                             ← gitignored
├── outputs/                          ← gitignored
├── gsd-scratch-work/                 ← read-only submodule
├── trading-crab-lib/                 ← read-only submodule
└── trading-crab/                     ← read-only submodule
```

---

## Exact pyproject.toml Files

### Root `pyproject.toml` — `trading-crab` (application)

```toml
[build-system]
requires = ["setuptools>=68,<76", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "trading-crab"
version = "0.1.0"
description = "Market regime classification pipeline — CLI and pipeline orchestration"
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.10"
authors = [{ name = "Strycker" }]
keywords = ["finance", "trading-crab", "macroeconomics", "portfolio"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
]
dependencies = [
    "trading-crab-lib>=0.2.0",   # the library this app wraps
    "pyyaml>=6.0",
    "python-dotenv>=1.0",
]

[project.scripts]
tradingcrab         = "trading_crab.cli:run_pipeline"
tradingcrab-setup   = "trading_crab.cli:setup"
tradingcrab-publish = "trading_crab.cli:publish_notebooks"

[project.urls]
Homepage   = "https://github.com/strycker/trading-crab"
Repository = "https://github.com/strycker/trading-crab"
Issues     = "https://github.com/strycker/trading-crab/issues"

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov",
    "ipykernel",
    "jupyterlab",
    "flake8",
]

# Find only the app package; the library has its own pyproject.toml
[tool.setuptools.packages.find]
where   = ["src"]
include = ["trading_crab"]

[tool.pytest.ini_options]
testpaths  = ["tests"]
pythonpath = ["src"]

# uv workspace: editable-installs both packages with a single `uv sync`
[tool.uv.workspace]
members = ["src/trading_crab_lib"]
```

### `src/trading_crab_lib/pyproject.toml` — `trading-crab-lib` (library)

```toml
[build-system]
requires = ["setuptools>=68,<76", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "trading-crab-lib"
version = "0.2.0"
description = "Market regime classification library — transforms, clustering, prediction, reporting"
readme = "../../README.md"
license = {text = "MIT"}
requires-python = ">=3.10"
authors = [{ name = "Strycker" }]
keywords = ["finance", "clustering", "macroeconomics", "machine-learning", "time-series"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Financial and Insurance Industry",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Topic :: Office/Business :: Financial :: Investment",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Typing :: Typed",
]

# Core deps: pure data science + ML only — no network/API/viz
dependencies = [
    "pandas>=2.0",
    "numpy>=1.25",
    "pyarrow>=14.0",
    "scikit-learn>=1.4",
    "scipy>=1.11",
    "pyyaml>=6.0",
    "joblib>=1.3",
    "python-dotenv>=1.0",
]

[project.urls]
Homepage   = "https://github.com/strycker/trading-crab"
Repository = "https://github.com/strycker/trading-crab"
Issues     = "https://github.com/strycker/trading-crab/issues"
Changelog  = "https://github.com/strycker/trading-crab/blob/main/STATE.md"

[project.optional-dependencies]
# Data ingestion from external APIs/scrapers
ingestion = [
    "fredapi>=0.5",
    "requests>=2.31",
    "lxml>=4.9",
    "beautifulsoup4>=4.12",
    "yfinance>=0.2",
    "certifi>=2024.0",
]
# Visualization
plotting = [
    "matplotlib>=3.8",
    "seaborn>=0.13",
]
# Probabilistic / state-space models
hmm = [
    "hmmlearn>=0.3",
    "statsmodels>=0.14",
]
# Optional clustering backends
clustering-extras = [
    "hdbscan>=0.8",
    "kneed>=0.8",
]
# Gradient boosting
boosting = [
    "lightgbm>=4.0",
]
# Install everything
all = [
    "trading-crab-lib[ingestion,plotting,hmm,clustering-extras,boosting]",
]
# Development tools
dev = [
    "pytest>=8.0",
    "pytest-cov",
    "flake8",
]

# pyproject.toml lives in src/trading_crab_lib/ — tell setuptools to look
# one level up (in src/) for the trading_crab_lib package directory.
[tool.setuptools.packages.find]
where   = [".."]
include = ["trading_crab_lib*"]

[tool.setuptools.package-data]
trading_crab_lib = ["py.typed"]
```

### How the two packages install

**For development (edit both packages simultaneously):**
```bash
# Option A — standard pip (two commands)
pip install -e src/trading_crab_lib/       # install lib in editable mode first
pip install -e .                            # install app; already satisfies lib dep

# Option B — uv workspaces (one command, recommended)
pip install uv
uv sync                                     # reads [tool.uv.workspace] members
```

**For end users:**
```bash
# Just the library (data scientists, algorithm researchers)
pip install trading-crab-lib
pip install "trading-crab-lib[ingestion,plotting]"   # with extras

# Full application (pipeline runners, weekly report users)
pip install trading-crab
tradingcrab --help
tradingcrab --steps 1,2,3 --refresh --plots
tradingcrab-setup                                     # interactive env setup
```

**For CI:**
```yaml
- run: pip install -e src/trading_crab_lib/[dev] && pip install -e .[dev]
- run: pytest tests/ -v
```

---

## Phase 5: Library Simplification & PyPI Prep

**Goal:** Get `trading_crab_lib` into a clean, publishable state before the structural split.

### Phase 5A: Code Audit & Cleanup

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

#### 5A.4 — Dependency audit & optional groups (S)
- Move all optional imports to lazy try/except guards in each module:
  - `matplotlib` / `seaborn` → only imported inside plotting functions
  - `fredapi` / `lxml` / `yfinance` → only imported in `ingestion/`
  - `hmmlearn` / `statsmodels` → only imported in `hmm.py` / `markov.py`
  - `lightgbm` → only imported in `prediction/gradient_boosting.py`
- Confirm: `python -c "from trading_crab_lib.transforms import engineer_all"` works with
  only the core deps installed (pandas, numpy, scikit-learn, scipy, pyyaml, pyarrow, joblib)
- **Output:** Library installable as a lean core; extras genuinely optional

#### 5A.5 — Type hint completeness pass (S)
- Ensure all public functions have full type annotations
- `py.typed` marker already exists
- Fix any `Any` types that should be more specific
- **Output:** Cleaner API for IDE autocomplete

#### 5A.6 — Docstring pass for public API (M)
- Add one-line docstrings to all public functions that lack them
- Include parameters and return type in the docstring body
- Skip private/internal functions (`_` prefix)
- **Output:** Usable `help(trading_crab_lib.clustering.fit_clusters)` etc.

### Phase 5B: Two-Package Infrastructure

#### 5B.1 — Add `src/trading_crab_lib/pyproject.toml` (S)
- Create using the exact template above
- Verify `pip install -e src/trading_crab_lib/` works from repo root
- Verify `python -c "import trading_crab_lib; print(trading_crab_lib.__version__)"` works
- Bump version in `__init__.py` to `0.2.0`
- **Output:** Library independently installable

#### 5B.2 — Create `src/trading_crab/` thin app package (S)
- Create `src/trading_crab/__init__.py` (version: `0.1.0`, short description)
- Create `src/trading_crab/cli.py` with three entry point functions:
  - `run_pipeline()` — thin wrapper that calls `run_pipeline.py:main()` via importlib
  - `setup()` — copies `.env.example` → `.env`, `config/email.example.yaml` → `config/email.local.yaml`, checks optional deps
  - `publish_notebooks()` — stub with `print("Coming soon: notebook → GitHub markdown dashboard")`
- Update root `pyproject.toml` per the template above (rename from `trading-crab-lib` to `trading-crab`, add scripts, restrict packages to `trading_crab`)
- Verify `pip install -e .` installs `tradingcrab` command and `tradingcrab --help` works
- **Output:** `tradingcrab` CLI functional

#### 5B.3 — Build & test both packages locally (S)
- `python -m build src/trading_crab_lib/` → check wheel and sdist contain only lib files
- `python -m build` at root → check `trading-crab` wheel contains only `src/trading_crab/`
- Install each wheel in a clean venv, verify imports and CLI
- **Output:** Both wheels confirmed installable and clean

#### 5B.4 — CI/CD update (S)
- Update flake8 `--exclude` in both workflow files to add `trading-crab,trading-crab-lib,gsd-scratch-work` (submodule dirs)
- Add `publish-lib.yml`: triggers on tag `lib-v*`, builds `src/trading_crab_lib/`, uploads to PyPI
- Add `publish-app.yml`: triggers on tag `v*`, builds root, uploads to PyPI
- **Output:** Both packages publish independently on their own tag patterns

---

## Phase 6: Monorepo Structural Cleanup

**Goal:** After Phase 5B the two-package structure works. This phase cleans up ancillary files.

#### 6A — Update MANIFEST.in for two packages (S)
- Current root `MANIFEST.in` prunes the right things for the library but not the app
- For the library build (from `src/trading_crab_lib/`): create a minimal
  `src/trading_crab_lib/MANIFEST.in` that excludes tests, notebooks, pipelines
- Root `MANIFEST.in`: revise to reflect app package (include `src/trading_crab/`, not `src/trading_crab_lib/`)
- **Output:** Clean sdists for both packages

#### 6B — Update CLAUDE.md layout section (S)
- Add `src/trading_crab/` entry to repository layout tree
- Expand "How to Run" section with `tradingcrab` CLI examples
- Update "Environment Setup" to show `uv sync` or two-step pip install
- Add note explaining the two-package architecture and `where = [".."]` trick

#### 6C — Update README.md (S)
- Add "Installation" section with `pip install trading-crab` vs `pip install trading-crab-lib`
- Update repo layout diagram
- Add PyPI version badges for both packages once published

---

## Phase 7: New Features

### Tier 1 — High impact, achievable soon

#### 7.1 — LightGBM flat-API integration (M)
- LightGBM exists in `prediction/gradient_boosting.py` and bundle API but NOT flat API
- Add `train_lightgbm()` to `prediction/__init__.py`, wire into `pipelines/05_predict.py`
  and `run_pipeline.py`, save as `outputs/models/lightgbm_regime.pkl`
- Include in model comparison bar chart
- **Prerequisite:** None

#### 7.2 — Additional FRED series: INDPRO, PAYEMS, DPCERA3Q086SBEA (S)
- INDPRO = Industrial Production Index; PAYEMS = Total Nonfarm Payrolls; DPCERA3Q = Real PCE
- Add to `config/settings.yaml` under `fred.series`
- Add to `initial_features` list (not clustering until evaluated)
- **Prerequisite:** None

#### 7.3 — Conference Board LEI proxy from FRED (S)
- Composite leading indicator from UNRATE (inverted), T10Y2Y, M2SL, INDPRO, PAYEMS
- Equal-weight average of standardized series → `lei_proxy` column
- **Prerequisite:** 7.2

### Tier 2 — High value, more effort

#### 7.4 — SMOTE / class-weight tuning (S)
- `class_weight="balanced"` in RF/DT/LGBM, optionally SMOTE
- Evaluate via per-class CV accuracy
- **Prerequisite:** None

#### 7.5 — Per-asset regime probability models (L)
- Per-asset binary classifiers: P(asset outperforms | features)
- Blend with regime-based ranking for final signal
- **Prerequisite:** 7.1

### Tier 3 — Longer-term

#### 7.6 — Backtest framework (XL)
- Walk-forward: retrain per quarter, predict regime, measure portfolio returns
- Compare vs SPY buy-and-hold, 60/40, equal-weight
- **Prerequisite:** 7.1, 7.4

#### 7.7 — Interactive Streamlit dashboard (L)
- Browser-based UI; `tradingcrab-publish` CLI entry point
- Uses `trading_crab_lib` as backend
- **Prerequisite:** Phase 5B complete

#### 7.8 — Weekly automated report with AI narrative (XL)
- Claude API generates natural-language market commentary
- Input: current regime, transition probs, asset signals
- **Prerequisite:** Phase 5B complete, 7.7

---

## Recommended Execution Order

```
Wave 1 (parallel, S-sized):   5A.1 + 5A.4 + 5A.5
Wave 2 (sequential, M-sized): 5A.2 → 5A.3 → 5A.6
Wave 3 (parallel, S-sized):   5B.1 + 5B.2 → 5B.3 + 5B.4
Wave 4 (features):             7.1 + 7.2 (parallel) → 7.3
Wave 5 (structural cleanup):   6A → 6B → 6C
Wave 6 (advanced features):    7.4 → 7.5 → 7.6 → 7.7 → 7.8
```

Each wave becomes its own branch + PR into `main`.

---

## Critical Constraints (unchanged)

1. Feature pipeline order is sacred: cross-ratios → log → gap-fill → derivatives → select
2. `balanced_cluster` is primary regime assignment for all downstream steps
3. Two prediction APIs must coexist: flat (`prediction/__init__.py`) + bundle (`classifier.py`)
4. `current_regime.pkl` always contains a bare `RandomForestClassifier`
5. GDP/GNP always shifted +1Q (publication-lag bias prevention)
6. PCA = 5 components — don't change without benchmarking
7. No committed secrets — never commit `.env`, API keys, or `email.yaml`
8. Submodules are read-only — never modify or push to `gsd-scratch-work/`, `trading-crab-lib/`, `trading-crab/`
9. Tests must pass after every change: `pytest tests/ -v`
