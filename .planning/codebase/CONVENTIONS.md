# Coding Conventions

**Analysis Date:** 2026-03-16

## Naming Patterns

**Files:**
- Python modules use **snake_case** filenames.
  - Example: `src/market_regime/transforms.py`, `src/market_regime/checkpoints.py`
- Tests use `test_*.py`.
  - Example: `tests/unit/test_transforms.py`

**Functions:**
- **snake_case** function names.
  - Example: `load()` / `setup_logging()` in `src/market_regime/config.py`
  - Example: `apply_derivatives()` / `engineer_all()` in `src/market_regime/transforms.py`

**Variables:**
- **snake_case** for locals and module-level values.
  - Example: `fred_key`, `settings_path` in `src/market_regime/config.py`
- Module logger is consistently named `log`.
  - Example: `log = logging.getLogger(__name__)` in `src/market_regime/transforms.py`

**Types:**
- Python 3.10+ typing is used, with `from __future__ import annotations` + PEP 604 unions (`X | None`) in many modules.
  - Example: `RunConfig.from_args()` in `src/market_regime/runtime.py`
  - Example: `load(settings_path: Path | None = None) -> dict` in `src/market_regime/config.py`

## Code Style

**Formatting:**
- No formatter configuration detected (no `ruff.toml`, `.ruff.toml`, `pyproject.toml` tool sections for ruff/black/isort, `setup.cfg`, or `.pre-commit-config.yaml`).
  - Relevant file: `pyproject.toml` (only config present is pytest options).
- Code follows standard PEP 8 conventions in practice: 4-space indentation, double quotes, and docstrings on modules/functions/classes.
  - Example: module + function docstrings and consistent indentation in `src/market_regime/transforms.py`

**Linting:**
- No linter configuration detected (ruff/flake8/pylint/mypy not configured via repo config files).
  - Evidence: repository root lacks `ruff.toml`, `.flake8`, `mypy.ini`, `pyproject.toml` tool sections.

## Import Organization

**Order:**
1. Standard library imports
2. Third-party imports
3. Local/package imports

Examples:
- `src/market_regime/checkpoints.py`: stdlib (`hashlib`, `json`, `pickle`, `datetime`, `pathlib`, `typing`) → third-party (`pandas`, `yaml`) → local (`from market_regime import DATA_DIR, CONFIG_DIR`)
- `src/market_regime/ingestion/fred.py`: stdlib (`concurrent.futures`, `datetime`) → third-party (`pandas`, `fredapi`) → local logger usage

**Path Aliases:**
- Not applicable (Python repo; no TS/JS aliasing observed).

## Error Handling

**Patterns:**
- Prefer explicit exception types with informative messages.
  - Missing env config: `raise EnvironmentError("FRED_API_KEY is not set")` in `src/market_regime/ingestion/fred.py`
  - Missing checkpoint: `raise FileNotFoundError(...)` in `src/market_regime/checkpoints.py`
  - Validation/guardrails in numerical utilities are tested via `ValueError` + regex `match` in `tests/unit/test_density.py` and `tests/unit/test_clustering_exploration.py`
- Broad exception handling is used only around external I/O boundaries, with warnings logged and the pipeline continuing with partial results.
  - Example: `_fetch_task()` catches `Exception` and logs a warning in `src/market_regime/ingestion/fred.py`

## Logging

**Framework:** Python stdlib `logging`

**Patterns:**
- Per-module logger named `log` with `logging.getLogger(__name__)`.
  - Example: `src/market_regime/config.py`, `src/market_regime/transforms.py`, `src/market_regime/checkpoints.py`
- Libraries log (no `print()` in library code) and pipeline/CLI entry points are expected to configure logging.
  - Example: `setup_logging()` in `src/market_regime/config.py`
  - Example: `RunConfig.apply_logging()` toggles root logger level in `src/market_regime/runtime.py`
- Logging levels:
  - `info` for step progress and checkpoint lifecycle (`src/market_regime/transforms.py`, `src/market_regime/checkpoints.py`)
  - `warning` for missing config/optional failures (`src/market_regime/config.py`, `src/market_regime/ingestion/fred.py`)
  - `debug` for verbose details (e.g., skipped transforms, checkpoint freshness, per-column gap fill)

## Comments

**When to Comment:**
- Use docstrings to document intent, ordering constraints, and rationale (not just restating code).
  - Example: detailed pipeline ordering and causal-vs-centered rationale in `src/market_regime/transforms.py`
- Use section dividers to organize long modules.
  - Example: `# ── ... ─────────────────` blocks in `src/market_regime/transforms.py` and `src/market_regime/checkpoints.py`

**JSDoc/TSDoc:**
- Not applicable (no JS/TS code detected).

## Function Design

**Size:** Functions are generally cohesive and grouped by pipeline stage; larger modules are organized with explicit section headers.
- Example: `src/market_regime/transforms.py` defines small, testable transforms and a single orchestration function `engineer_all()`.

**Parameters:** Explicit parameters with sensible defaults; runtime variability is driven by config dicts and `RunConfig`.
- Example: `engineer_all(df, cfg, causal=False)` in `src/market_regime/transforms.py`
- Example: `CheckpointManager.is_fresh(..., max_age_days=7.0, require_config_match=False)` in `src/market_regime/checkpoints.py`

**Return Values:** Prefer returning new DataFrames/Series (copying inputs to avoid mutation).
- Example: `add_cross_ratios()`, `apply_log_transforms()`, `apply_gap_fill()`, `apply_derivatives()` all `df = df.copy()` in `src/market_regime/transforms.py`

## Module Design

**Exports:** Modules expose functional APIs (top-level functions/classes) imported directly by pipeline code and tests.
- Example: `tests/unit/test_clustering.py` imports `reduce_pca`, `evaluate_kmeans`, `pick_best_k`, `fit_clusters` from `src/market_regime/clustering.py`

**Barrel Files:** Not applicable (Python).

---

*Convention analysis: 2026-03-16*
