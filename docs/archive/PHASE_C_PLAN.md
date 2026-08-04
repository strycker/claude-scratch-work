# PHASE_C_PLAN.md — Two-Package Infrastructure

> **⚠️ SUPERSEDED (July 2026).** Historical plan — kept for context, not active work.
> The project's target and execution plan are now `platform_design/platform_design.md`
> (v1.7) and `ROADMAP.md` Tier 0. Do not treat items below as current.

Created: 2026-03-28
Branch: `claude/implement-phase-a2-SmHEA`

---

## Overview

Phase C creates the monorepo two-package structure described in `NEXT_STEPS.md`:

1. **`trading-crab-lib`** — pure library at `src/trading_crab_lib/` with its own `pyproject.toml`
2. **`trading-crab`** — thin application layer at `src/trading_crab/` (CLI, setup, publish)

The root `pyproject.toml` becomes the `trading-crab` app package.

---

## Tasks

### C1 — Library `pyproject.toml`

Create `src/trading_crab_lib/pyproject.toml` from the template in `NEXT_STEPS.md`:
- Core deps: pandas, numpy, pyarrow, scikit-learn, scipy, pyyaml, joblib, python-dotenv
- Optional dep groups: `[ingestion]`, `[plotting]`, `[hmm]`, `[clustering-extras]`, `[boosting]`, `[all]`, `[dev]`
- Version: 0.1.2
- Add `__version__ = "0.1.2"` to `__init__.py`
- Verify: `pip install -e src/trading_crab_lib/` works

### C2 — App Package

Create `src/trading_crab/` with:
- `__init__.py` — version, description
- `cli.py` — `run_pipeline()`, `setup()`, `publish_notebooks()` entry points
  - `run_pipeline()` wraps existing `run_pipeline.py` logic
  - `setup()` stub for interactive config generation
  - `publish_notebooks()` stub for notebook publishing

### C3 — Root `pyproject.toml` Update

Transform root `pyproject.toml`:
- Rename package from `trading-crab-lib` to `trading-crab`
- Version: 0.1.2
- Dependencies: `trading-crab-lib>=0.1.2`, `pyyaml`, `python-dotenv`
- Add `[project.scripts]` for CLI entry points
- Restrict `[tool.setuptools.packages.find]` to `include = ["trading_crab"]`
- Add `[tool.uv.workspace]` for uv
- Verify both packages install and tests pass

### C4 — CI/CD Update

- Update flake8 exclude in both workflow files to also exclude `trading-crab-lib/` submodule
- Install both packages in CI: `pip install -e src/trading_crab_lib/[dev]` then `pip install -e .[dev]`
- Add `publish-lib.yml` (triggered on `lib-v*` tags)
- Add `publish-app.yml` (triggered on `v*` tags)

---

## Remaining Phases (D–F)

See `NEXT_STEPS.md` for full details. Summary:

### Phase D — New Features (Tier 1)
| Task | Description |
|------|-------------|
| D1 | Additional FRED series (INDPRO, PAYEMS, DPCERA3Q086SBEA) |
| D2 | LightGBM flat-API integration |
| D3 | Conference Board LEI proxy composite indicator |
| D4 | SMOTE / class-weight tuning for balanced training |

### Phase E — Structural Cleanup
| Task | Description |
|------|-------------|
| E1 | Update MANIFEST.in for both packages |
| E2 | Update CLAUDE.md with 2-package architecture |
| E3 | Update README.md with install instructions |
| E4 | Refactor run_pipeline.py into src/trading_crab/pipeline.py |

### Phase F — Advanced Features (long-term)
| Task | Description |
|------|-------------|
| F1 | Per-asset regime probability models |
| F2 | Walk-forward backtest framework |
| F3 | Interactive Streamlit dashboard |
| F4 | Weekly automated report with AI narrative |
| F5 | Finviz Elite integration |
| F6 | trading-crab-lib config independence |
