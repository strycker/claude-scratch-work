# NEXT_STEPS.md — Unified Phase Plan

Created: 2026-03-27
Updated: 2026-03-27 (unified with Phase letter notation; monorepo 2-package architecture)
Branch: `claude/refresh-submodule-analysis-1Icoo`

---

## Current State Summary

**What's done (prior Phases 1–4):**
- 9,899 lines of library code across 31 modules in `src/trading_crab_lib/`
- 573 tests (all passing, 11 skipped for optional deps)
- 9 pipeline steps running end-to-end on real data
- 12 Jupyter notebooks with full diagnostic cells
- 85 monitoring/plotting items delivered
- Package renamed to `trading_crab_lib`, pyproject.toml ready
- HMM, Markov, GMM, DBSCAN, Spectral clustering all implemented
- Divergence + momentum features integrated
- Email + weekly report automation
- Preservation checkpoints, env var overrides, convenience imports

**Three submodules (all read-only references):**
| Submodule | Remote | Purpose |
|---|---|---|
| `gsd-scratch-work/` | strycker/gsd-scratch-work | GSD framework workspace — earlier canonical checkpoint |
| `trading-crab-lib/` | strycker/trading-crab-lib | Earlier PyPI library snapshot |
| `trading-crab/` | strycker/trading-crab | Historical notebook-era reference |

**What `claude-scratch-work` is:**
This repo is the AI-assisted sandbox. Code validated here gets hand-copied by the
owner to `strycker/trading-crab` (the human-validated production repo). This sandbox
continues to exist for AI coding, experimentation, and feature development.

---

## Architecture: One Repo, Two PyPI Packages

```
claude-scratch-work/  (this git repo)
│
├── pyproject.toml                  ← "trading-crab" v0.1.2 (APPLICATION package)
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
│       ├── pyproject.toml          ← "trading-crab-lib" v0.1.2 (LIBRARY package)
│       │                              pip install trading-crab-lib
│       │                              → no executables, pure import API
│       ├── __init__.py
│       └── ... (31 library modules)
│
├── pipelines/                      ← NOT installed to site-packages
├── notebooks/                      ← NOT installed to site-packages
├── scripts/                        ← NOT installed to site-packages
├── config/                         ← some files copied by `tradingcrab-setup`
├── run_pipeline.py                 ← used directly (git clone) or via `tradingcrab` CLI
├── tests/                          ← tests for both packages
└── ...
```

**Why this layout?**

| Concern | Answer |
|---|---|
| AI visibility | Single repo = Claude Code sees lib + app + docs in one context |
| Single source of truth | One CLAUDE.md, one test suite, no sync drift |
| Atomic changes | Change lib API + update callers in one commit |
| Independent installability | `pip install trading-crab-lib` = lean library only |
| CLI entry points | `pip install trading-crab` = `tradingcrab`, `tradingcrab-setup`, `tradingcrab-publish` |

---

## Exact pyproject.toml Files

### Root `pyproject.toml` — `trading-crab` (application)

```toml
[build-system]
requires = ["setuptools>=68,<76", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "trading-crab"
version = "0.1.2"
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
    "trading-crab-lib>=0.1.2",   # the library this app wraps
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
version = "0.1.2"
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

# Core deps: pure data science + ML — no network/API/viz
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
ingestion = [
    "fredapi>=0.5",
    "requests>=2.31",
    "lxml>=4.9",
    "beautifulsoup4>=4.12",
    "yfinance>=0.2",
    "certifi>=2024.0",
]
plotting = [
    "matplotlib>=3.8",
    "seaborn>=0.13",
]
hmm = [
    "hmmlearn>=0.3",
    "statsmodels>=0.14",
]
clustering-extras = [
    "hdbscan>=0.8",
    "kneed>=0.8",
]
boosting = [
    "lightgbm>=4.0",
]
all = [
    "trading-crab-lib[ingestion,plotting,hmm,clustering-extras,boosting]",
]
dev = [
    "pytest>=8.0",
    "pytest-cov",
    "flake8",
]

# pyproject.toml is at src/trading_crab_lib/ — look one level up (src/) for packages
[tool.setuptools.packages.find]
where   = [".."]
include = ["trading_crab_lib*"]

[tool.setuptools.package-data]
trading_crab_lib = ["py.typed"]
```

### Install workflows

```bash
# Development (edit both packages)
pip install -e src/trading_crab_lib/       # lib editable
pip install -e .                            # app editable (lib already satisfied)
# OR: uv sync                              # both at once via workspace

# End users — library only
pip install trading-crab-lib
pip install "trading-crab-lib[ingestion,plotting]"

# End users — full pipeline
pip install trading-crab
tradingcrab --steps 1,2,3 --refresh --plots
tradingcrab-setup

# CI
pip install -e src/trading_crab_lib/[dev] && pip install -e .[dev]
pytest tests/ -v
```

---

## Phase Plan

### Phase A — Simplify (3 parallel S-sized tasks)

Clean up the library codebase to make everything downstream easier.

| Task | Description | Size | Depends on |
|---|---|---|---|
| **A1** | Dead code removal scan — grep for unused imports, unreachable functions, commented-out blocks; remove and verify tests pass | S | — |
| **A2** | Dependency audit — classify deps as core vs optional; add try/except ImportError guards to all optional imports (matplotlib, fredapi, lxml, yfinance, hmmlearn, statsmodels, lightgbm); confirm `from trading_crab_lib.transforms import engineer_all` works with core-only deps | S | — |
| **A3** | Type hint completeness — ensure all public functions have full annotations; fix `Any` types that should be specific | S | — |

**A1, A2, A3 can all run in parallel.** Each is a standalone PR.

---

### Phase B — Decompose & Document (3 sequential M/S-sized tasks)

Split oversized modules and add documentation for the public API.

| Task | Description | Size | Depends on |
|---|---|---|---|
| **B1** | Decompose `plotting.py` (2,018 lines) → `plotting/` package with `core.py`, `pca.py`, `regime.py`, `prediction.py`, `assets.py`, `diagnostics.py`; re-export from `__init__.py`; run tests after each move | M | A1 |
| **B2** | Decompose `monitoring.py` (648 lines) → `monitoring/` package with `ingestion.py`, `clustering.py`, `prediction.py`, `pipeline.py`; re-export from `__init__.py` | S | A1 |
| **B3** | Docstring pass — add one-line docstrings + parameter/return docs to all public functions; skip `_`-prefixed internals | M | B1, B2 |

**B1 → B2 → B3** are sequential (each builds on the prior).

---

### Phase C — Two-Package Infrastructure (4 tasks, partially parallel)

Create the monorepo 2-package structure. This is the structural change.

| Task | Description | Size | Depends on |
|---|---|---|---|
| **C1** | Add `src/trading_crab_lib/pyproject.toml` (from template above, version 0.1.2); verify `pip install -e src/trading_crab_lib/` works; update `__init__.py` version | S | A2 (dep audit informs which deps are core vs optional) |
| **C2** | Create `src/trading_crab/` app package — `__init__.py` (v0.1.2), `cli.py` with `run_pipeline()` (thin import wrapping `run_pipeline.py` as-is), `setup()` (interactive config generation from example `settings.yaml`), `publish_notebooks()` (stub); ship `config/settings.example.yaml` as package data in `trading-crab` (NOT in `trading-crab-lib`); update root `pyproject.toml` (rename package to `trading-crab`, add `[project.scripts]`, restrict to `include = ["trading_crab"]`); verify `tradingcrab --help` works | S | C1 |
| **C3** | Build & test — `python -m build src/trading_crab_lib/` + `python -m build` at root; install both wheels in clean venv; verify imports and CLI | S | C1, C2 |
| **C4** | CI/CD update — update flake8 excludes; add `publish-lib.yml` (tag `lib-v*`) and `publish-app.yml` (tag `v*`) workflows | S | C3 |

**C1 → C2 → C3 + C4** (C3 and C4 can be parallel after C2).

---

### Phase D — New Features, Tier 1 (can start during Phase B)

Feature work that is independent of the packaging restructure.

| Task | Description | Size | Depends on |
|---|---|---|---|
| **D1** | Additional FRED series — add INDPRO, PAYEMS, DPCERA3Q086SBEA to `config/settings.yaml`; add to `initial_features`; update ingestion completeness expected columns | S | — |
| **D2** | LightGBM flat-API integration — add `train_lightgbm()` to `prediction/__init__.py`; wire into `pipelines/05_predict.py` and `run_pipeline.py`; save as `outputs/models/lightgbm_regime.pkl`; add to model comparison bar chart | M | — |
| **D3** | Conference Board LEI proxy — composite indicator from UNRATE (inverted), T10Y2Y, M2SL, INDPRO, PAYEMS; add to `transforms.py` or new `indicators.py` | S | D1 |
| **D4** | SMOTE / class-weight tuning — `class_weight="balanced"` in RF/DT/LGBM; optionally SMOTE via imblearn; evaluate per-class CV accuracy; add `prediction.class_balance_method` to settings | S | — |

**D1 + D2 + D4 can run in parallel** (D3 waits for D1).
Phase D can start as early as Phase B — no dependency on C.

---

### Phase E — Structural Cleanup & Refactor (4 tasks)

Polish docs and build artifacts after the 2-package split, then refactor the pipeline.

| Task | Description | Size | Depends on |
|---|---|---|---|
| **E1** | Update MANIFEST.in — create `src/trading_crab_lib/MANIFEST.in` for lib sdist; revise root MANIFEST.in for app sdist | S | C2 |
| **E2** | Update CLAUDE.md — add `src/trading_crab/` to layout tree; add 2-package architecture note; update "How to Run" with `tradingcrab` CLI examples; update "Environment Setup" with `uv sync` | S | C2 |
| **E3** | Update README.md — add Installation section (`pip install trading-crab` vs `trading-crab-lib`); update layout diagram; add PyPI badges placeholder | S | C2 |
| **E4** | Refactor `run_pipeline.py` (62KB monolith at repo root) into `src/trading_crab/pipeline.py` — move pipeline logic into the app package proper; `run_pipeline.py` becomes a thin shim (`from trading_crab.pipeline import main; main()`) for backward compat; `cli.py:run_pipeline()` calls the same entry point directly | M | C2 |

**E1, E2, E3 can all run in parallel** after Phase C. **E4** is independent and can be done any time after C2.

---

### Phase F — Advanced Features (long-term backlog)

Larger efforts for after the foundation is solid. Ordered by priority.
F1–F3 are Tier 1 (high priority); F4–F6 are Tier 2 (deferred).

| Task | Description | Size | Depends on | Tier |
|---|---|---|---|---|
| **F1** | Per-asset regime probability models — per-ETF binary classifiers predicting P(outperform \| features); blend with regime-based ranking | L | D2 | 1 |
| **F2** | Backtest framework — walk-forward: retrain per quarter, predict regime, construct portfolio, measure returns vs SPY/60-40/equal-weight; Sharpe, max drawdown, Calmar | XL | D2, D4 | 1 |
| **F3** | Interactive Streamlit dashboard — browser-based UI with tabs for regime history, asset rankings, portfolio, diagnostics; wire into `tradingcrab-publish` | L | Phase C | 1 |
| **F4** | Weekly automated report with AI narrative — Claude API generates market commentary from current regime, transition probs, asset signals | XL | Phase C, F3 | 2 |
| **F5** | Finviz Elite integration — sector rotation, institutional flow data; gated behind API key / optional dep | L | Phase C | 2 |
| **F6** | `trading-crab-lib` config independence — allow library to accept a config dict or path at runtime rather than requiring `config/settings.yaml` on disk; enables clean `pip install` usage without git clone | S | Phase C | 2 |

---

## Execution Schedule

```
                A1 ─┐
                A2 ─┤── (parallel) ── Phase A
                A3 ─┘
                     │
                B1 → B2 → B3        Phase B
                     │
          D1 ─┐     │               Phase D (features — can start
          D2 ─┤     │                during Phase B, parallel with it)
          D4 ─┘     │
           │        │
           D3       │
                     │
                C1 → C2 → C3 ─┐     Phase C (2-package infra)
                          C4 ─┘
                     │
              E1 ─┐  │              Phase E (structural cleanup)
              E2 ─┤                  (parallel, after C)
              E3 ─┘
                     │
       F1, F2, F3, F4, F5           Phase F (long-term, sequential)
```

**Concrete order for branches/PRs:**
1. **Branch 1:** A1 + A2 + A3 (parallel — one PR with 3 commits, or 3 small PRs)
2. **Branch 2:** B1 → B2 (decomposition)
3. **Branch 3:** D1 + D2 + D4 (features — can overlap with Branch 2)
4. **Branch 4:** B3 (docstrings — after decomposition settles)
5. **Branch 5:** C1 → C2 → C3 + C4 (2-package infrastructure)
6. **Branch 6:** D3 (LEI proxy — after D1 merges)
7. **Branch 7:** E1 + E2 + E3 (doc cleanup — after C merges)
8. **Branch 8:** E4 (refactor run_pipeline.py → src/trading_crab/pipeline.py)
9. **Branch 9+:** F1–F6 (long-term, one per branch)

---

## Design Decisions (resolved)

**Version:** Both packages start at `0.1.2`. We are still early — don't increment minor
version yet. Patch bumps (0.1.3, 0.1.4, ...) for each meaningful release.

**`claude-scratch-work` role:** Continues as the AI-assisted sandbox. Owner hand-copies
validated code to `strycker/trading-crab` (production). This repo is not going away.

**Finviz:** Deferred to Phase F5. Not incorporated now; listed as long-term backlog.

**Submodules:** Remain read-only references. Never modify or push to them.

**Config ownership:** `trading-crab-lib` does NOT ship `settings.yaml`. The library
expects config to be provided at runtime (path or dict). `trading-crab` (app) ships
`settings.example.yaml` as package data and `tradingcrab-setup` generates the user's
`config/settings.yaml` from it interactively. Long-term (F6): library should accept
config dict/path at init for standalone `pip install` usage without git clone.

**`run_pipeline.py` refactor:** Phase C2 wraps `run_pipeline.py` as-is via thin import
in `cli.py`. Phase E4 refactors the 62KB monolith into `src/trading_crab/pipeline.py`.
`run_pipeline.py` at repo root becomes a backward-compat shim for `python run_pipeline.py`.

**Phase D overlaps with B:** Acceptable. Feature work (D) can start during Phase B
decomposition since they touch different files. In practice, tasks will likely execute
in order (A → B → C → D → E → F), but the dependency graph allows parallelism.

**B1 dependency:** B1 (plotting decomposition) waits for A1 (dead code scan) only,
not all of Phase A. A2 and A3 can merge independently.

---

## Critical Constraints

1. Feature pipeline order is sacred: cross-ratios → log → gap-fill → derivatives → select
2. `balanced_cluster` is primary regime assignment for all downstream steps
3. Two prediction APIs must coexist: flat (`prediction/__init__.py`) + bundle (`classifier.py`)
4. `current_regime.pkl` always contains a bare `RandomForestClassifier`
5. GDP/GNP always shifted +1Q (publication-lag bias prevention)
6. PCA = 5 components — don't change without benchmarking
7. No committed secrets — never commit `.env`, API keys, or `email.yaml`
8. Submodules are read-only — never modify or push to `gsd-scratch-work/`, `trading-crab-lib/`, `trading-crab/`
9. Tests must pass after every change: `pytest tests/ -v`
