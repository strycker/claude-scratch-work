# Coding Conventions

**Analysis Date:** 2026-07-09

## Naming Patterns

**Files:**
- **Library modules** (in `src/trading_crab_lib/`): lowercase with underscores (`transforms.py`, `checkpoints.py`, `regime.py`)
- **Pipeline steps** (in `pipelines/`): numbered prefix `NN_descriptive_name.py` (e.g., `01_ingest.py`, `02_features.py`)
- **Test files**: `test_<module>.py` for unit tests in `tests/unit/`; `test_<feature>.py` for integration tests in `tests/integration/`
- **CLI and app modules** (in `src/trading_crab/`): lowercase with underscores (`cli.py`, `pipeline.py`)

**Functions:**
- Verb + noun pattern: `fetch_all()`, `apply_log_transforms()`, `build_profiles()`, `fit_clusters()`
- Helper functions prefixed with underscore: `_fetch_one()`, `_config_hash()`, `_get_nested()`
- Boolean predicates: `is_fresh()`, `should_write()`, `preservation_checkpoint_should_write()`
- Test functions: `test_<behavior>()` with explicit behavior description (e.g., `test_gap_fill_idempotent()`, `test_market_code_not_filled()`)

**Variables:**
- DataFrame variables: noun describing contents (`features`, `pca_df`, `clustered`, `returns`, `profiles`)
- Series variables: noun for single values (`labels`, `cluster`, `sp500_prices`, `transitions`)
- Config/dict variables: `cfg`, `meta`, `frames`, `results`
- Loop indices: single letters accepted only for math/ML contexts (`i`, `j`, `k`, `n`, `q`, `v`, `t`, `h`) — never used for meaningful data (see `.pylintrc` `good-names`)
- Temporary/intermediate: typically short-lived (`tmp_path`, `session_dir`, `result`, `df`)

**Types:**
- Type aliases: avoided; use explicit union types (`X | None` instead of `Optional[X]`)
- Type hints on all public functions (enforced by ruff + mypy)
- `cls` in classmethods left unannotated per Python convention
- Generic objects from optional dependencies: `model: object`, `pca_obj: object` (used when type cannot be imported)

**Constants:**
- Module-level: `UPPER_CASE` (e.g., `_MAX_WORKERS`, `PRESERVATION_CHECKPOINT_NAMES`, `CHECKPOINT_DIR`)
- Availability flags: `_HMM_AVAILABLE`, `_STATSMODELS_AVAILABLE`, `HAS_LIGHTGBM` assigned in try/except blocks
- Private availability flags allowed by pylint via `.pylintrc` `variable-rgx = (_[A-Z][A-Z0-9_]*$)|...`

## Code Style

**Formatting:**
- Tool: ruff (linter + auto-fixer) + flake8 (syntax only in pre-commit)
- Line length: 127 characters (set in `pyproject.toml` and `.pylintrc`)
- Imports: ruff's isort integration (handled automatically via `ruff --fix`)
- No trailing whitespace (enforced by pre-commit `trailing-whitespace` hook)
- EOF fixers: all files end with single newline (enforced by pre-commit `end-of-file-fixer` hook)

**Linting:**
- Tool: ruff (primary) with selective pylint (secondary, via `.pylintrc`)
- ruff rules: `["E", "F", "W", "I", "UP"]` (pycodestyle errors/warnings, pyflakes, isort, pyupgrade)
- Ruff ignores: `["E741"]` (ambiguous variable names like `l`, `O`, `I` — common in ML code)
- Per-file ignores for tests: `"tests/**/*.py" = ["F811"]` (re-imported/shadowed fixtures are expected)
- Pre-commit hooks: ruff (`--fix` auto-correct), flake8 (syntax only: `E9,F63,F7,F82`), mypy (informational only)
- `.pylintrc` disables false-positives common in ML/data science: documentation checks (C0114/115/116), complexity metrics (R0914/912/915/913), duplication (R0801), broad exception catching (W0718)

**Python version:**
- Target: Python 3.10+
- Required: `from __future__ import annotations` at the top of all source files (enables PEP 563 postponed evaluation for 3.10 compatibility)
- Union types: `X | Y` not `Union[X, Y]`; optional: `X | None` not `Optional[X]`
- Pattern matching: `match` statements acceptable but not required

## Import Organization

**Order:**
1. `from __future__ import annotations` (always first if present)
2. Standard library imports (stdlib)
3. Third-party imports (pandas, numpy, scipy, sklearn, yaml, etc.)
4. Relative imports from the same package (`from trading_crab_lib.config import load`)
5. Grouped by functional purpose within each section (optional, but encouraged)

**Pattern examples:**
```python
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from dotenv import load_dotenv

from trading_crab_lib import CONFIG_DIR, DATA_DIR
from trading_crab_lib.checkpoints import CheckpointManager
```

**Path Aliases:**
- None defined at project level (ruff isort does not require aliases)
- Absolute imports preferred for clarity (e.g., `from trading_crab_lib.transforms import engineer_all`)

**Conditional/Optional Imports:**
- Gracefully handled in try/except for optional dependencies (`hmmlearn`, `statsmodels`, `hdbscan`, `lightgbm`, `k-means-constrained`)
- Error message provides install instructions: `"Install with: pip install 'trading-crab-lib[ingestion]'"`
- Availability flags set at module load: `_HMM_AVAILABLE = False` assigned in except block, later checked with `if _HMM_AVAILABLE:` before use
- Tests skip via `pytest.mark.skipif` when optional deps unavailable

## Error Handling

**Patterns:**
- Specific exception types always caught (no bare `except:`)
- Broad exception catching used only for network ingestion code (marked with `# noqa: BLE001`)
- Fail-fast for config issues: `ValueError` raised immediately with complete error list via `validate_config()`
- Missing/invalid files: `FileNotFoundError` raised with full path included
- Network failures: caught and logged at WARNING, pipeline continues with empty/partial data (graceful degradation)

**Examples:**
```python
# Config validation — collect all errors before raising
if errors:
    bullet_list = "\n".join(f"  • {e}" for e in errors)
    raise ValueError(f"settings.yaml has {len(errors)} validation error(s):\n{bullet_list}")

# Checkpoint loading — specific error
if not parquet_path.exists():
    raise FileNotFoundError(f"Checkpoint not found: {parquet_path}")

# Network code — broad catch with noqa, log warning
try:
    s = _fetch_one(fred, series_id, start, end, shift)
except Exception as exc:  # noqa: BLE001 — fredapi raises various types
    log.warning("Failed to fetch %s (%s): %s", friendly_name, series_id, exc)
    return friendly_name, None
```

## Logging

**Framework:** Standard library `logging` module (no external frameworks)

**Setup:**
- Each module: `log = logging.getLogger(__name__)` at top (after imports)
- Root logger configured in `config.py`: `setup_logging(level: str = "INFO")`
- Format: `"%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"` with `datefmt="%Y-%m-%d %H:%M:%S"`
- Verbosity control: `RunConfig.apply_logging()` sets root to DEBUG if `verbose=True`

**Levels:**
- **DEBUG**: checkpoint freshness checks, detailed step progression (only when `--verbose`)
- **INFO**: normal pipeline progress, checkpoint saves/loads, ingestion completion counts, named regime outputs
- **WARNING**: missing/invalid config, corrupt checkpoint metadata, network failures, stale data
- **ERROR**: not used; critical failures raise exceptions instead

**No print() in library code:** All library output via `logging`; only `pipelines/` scripts and `run_pipeline.py` may use `print()` for user-facing messages

**Examples:**
```python
log.debug("Checkpoint missing: %s", name)
log.info("Fetching FRED %-10s → %s%s", series_id, friendly_name, lag_note)
log.warning("Failed to fetch %s (%s): %s", friendly_name, series_id, exc)
log.info("FRED fetch complete: %d quarters, %d series", len(df), len(df.columns))
```

## Comments

**When to Comment:**
- Algorithm intent: explain *why* a non-obvious approach is chosen (e.g., Bernstein gap fill in log space, not linear)
- Complex math: document the formula or reference (e.g., "Derivative of a linear series should be roughly constant")
- Publication-lag shifts: why GDP/GNP are shifted, not other FRED series
- Intentional simplifications (ponytail style): mark with `# ponytail: explanation` comment naming the simplification and upgrade path

**JSDoc/TSDoc style:**
- Triple-quoted docstrings on all public functions (enforced at code-review, not lint time)
- Format: brief one-liner, then longer description if needed, Args/Returns/Raises sections
- Example from `regime.py`:
  ```python
  def build_profiles(
      features_df: pd.DataFrame,
      cluster_labels: pd.Series,
      stats: list[str] | None = None,
  ) -> pd.DataFrame:
      """
      Compute per-cluster descriptive statistics for all features.

      Args:
          features_df    — feature matrix (clustering_features or broader set)
          cluster_labels — integer Series aligned with features_df index
          stats          — agg functions; defaults to ["mean", "median", "std"]

      Returns:
          DataFrame with MultiIndex columns (stat, feature), rows = cluster IDs.
      """
  ```

**Module-level docstrings:**
- Always present: explain module purpose, usage example, key concepts
- Rendered as the first triple-quoted string in the file (before imports of `from __future__`)
- Example from `checkpoints.py`: lists usage pattern for save/load/is_fresh/clear

**Section dividers:**
- Horizontal lines using `# ── Name ──` (en-dashes, exactly 2 dashes before and after)
- Used to organize long modules into logical sections
- No other divider styles

## Function Design

**Size:**
- Target: short enough to fit on one screen without scrolling (rarely exceed ~50 lines)
- Longer functions acceptable only for pipelines (step functions inherently complex) and exploratory notebooks
- Library functions broken into helpers when they exceed 40 lines

**Parameters:**
- Keyword-only arguments (`*`) for optional flags to prevent accidental positional misuse
- Config objects passed whole (`cfg: dict[str, Any]`) rather than unpacked
- RunConfig always passed as positional parameter when needed: `def step_func(df: pd.DataFrame, run_cfg: RunConfig)`
- Examples from registry:
  ```python
  def apply_gap_fill(df: pd.DataFrame) -> pd.DataFrame:  # No RunConfig needed
  def fit_clusters(pca_df: pd.DataFrame, best_k: int, balanced_k: int, *, use_constrained: bool) -> pd.Series:  # Keyword-only flag
  def plot_rrg_scatter(..., run_cfg: RunConfig) -> None:  # RunConfig passed to plotting functions
  ```

**Return Values:**
- Predictable types: functions return exactly one type (not `X | None` unless documented)
- DataFrames/Series always indexed consistently (preserved from input where possible)
- Models return as objects, not dicts (flat API in `prediction/__init__.py`); bundle dicts only in `classifier.py` for test support

**Immutability:**
- Functions do not mutate inputs unless explicitly documented
- `.copy()` used before modifying: `def func(df): df = df.copy(); df["col"] = ...; return df`
- Verified in tests: `def test_does_not_mutate_input(self, raw_macro_df): original_cols = list(raw_macro_df.columns); func(raw_macro_df); assert list(raw_macro_df.columns) == original_cols`

## Module Design

**Exports:**
- Public functions: no prefix (e.g., `load()`, `fetch_all()`, `build_profiles()`)
- Private functions/constants: `_prefix` (e.g., `_fetch_one()`, `_MAX_WORKERS`, `_REQUIRED_SECTIONS`)
- Package-level re-exports in `__init__.py` for convenience (e.g., `plotting/__init__.py` re-exports all plot functions)

**Barrel Files:**
- Used in `plotting/__init__.py` and `monitoring/__init__.py` to re-export all submodule functions
- Pattern: `from .submodule import *` with explicit `__all__` list (when needed for clarity)
- Enables `from trading_crab_lib.plotting import plot_regime_timeline` without knowing the submodule

**Circular imports:**
- Avoided via explicit imports where possible
- Lazy imports (`from ... import X` inside a function) used only for optional dependencies or `__getattr__` patterns
- No known circular dependency chains in current codebase (see CLAUDE.md)

---

*Convention analysis: 2026-07-09*
