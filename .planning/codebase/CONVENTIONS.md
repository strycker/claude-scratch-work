# Coding Conventions

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


## Naming Patterns

**Files:**
- Library modules: `lowercase_with_underscores.py` (e.g., `transforms.py`, `checkpoints.py`)
- Pipeline steps: Numbered prefix `NN_descriptive_name.py` (e.g., `01_ingest.py`, `02_features.py`)
- Test files: `test_<module>.py` for unit tests; `test_<feature>.py` for integration tests
- App modules: `lowercase_with_underscores.py` (e.g., `cli.py`, `pipeline.py`)

**Functions:**
- Verb-noun pattern: `fetch_all()`, `apply_log_transforms()`, `build_profiles()`, `fit_clusters()`
- Helper functions: Prefixed with underscore `_fetch_one()`, `_config_hash()`, `_get_nested()`
- Boolean predicates: `is_fresh()`, `should_write()`, `preservation_checkpoint_should_write()`
- All public functions have complete type hints (`def func(x: int, y: str | None = None) -> dict[str, Any]`)
- Return exactly one type (not `X | None` unless documented); use `X | None` for optional returns
- Optional kwargs use keyword-only syntax to prevent accidental positional misuse
- `cls` in classmethods left unannotated per Python convention

**Variables:**
- DataFrames: Noun describing contents (`features`, `pca_df`, `clustered`, `returns`, `profiles`)
- Series: Noun for single values (`labels`, `cluster`, `sp500_prices`, `transitions`)
- Config/dict: `cfg`, `meta`, `frames`, `results`
- Loop indices: Single letters accepted for math/ML contexts only (`i`, `j`, `k`, `n`, `q`, `v`, `t`, `h`); never for meaningful data
- Temporary/intermediate: Short-lived names (`tmp_path`, `session_dir`, `result`, `df`)

**Constants & Private Availability Flags:**
- Module-level constants: `UPPER_CASE` (e.g., `_MAX_WORKERS`, `PRESERVATION_CHECKPOINT_NAMES`, `CHECKPOINT_DIR`)
- Availability flags: `_HMM_AVAILABLE`, `_STATSMODELS_AVAILABLE`, `HAS_LIGHTGBM` (assigned in try/except blocks)
- Private availability flags: Allowed by pylint via `.pylintrc` `variable-rgx = (_[A-Z][A-Z0-9_]*$)|...`
- Type aliases: Avoided; use explicit union types (`X | Y` instead of `Optional[X]`)

## Code Style

**Formatting & Linting:**
- Line length: 127 characters (set in `pyproject.toml` `[tool.ruff]` and `.pylintrc`)
- Python version: 3.10+ required (enables `match`, `X | Y` union syntax, walrus operator)
- Required in all source files: `from __future__ import annotations` (at top, PEP 563 postponed evaluation)
- No trailing whitespace (enforced by pre-commit `trailing-whitespace` hook)
- All files end with single newline (enforced by pre-commit `end-of-file-fixer` hook)
- Union types: `X | Y` not `Union[X, Y]`; optional: `X | None` not `Optional[X]`

**Active Linters (CI/CD matrix: Python 3.10–3.13):**
- **ruff** (primary): Rules `E`, `F`, `W`, `I`, `UP` (pycodestyle, pyflakes, isort, pyupgrade)
  - Ignores: `E741` (ambiguous variable names—common in ML code)
  - Per-file ignores: `tests/**/*.py = ["F811"]` (re-imported/shadowed fixtures are expected)
  - Entry: `.github/workflows/python-package.yml` runs `ruff check src/ tests/`
  - Pre-commit: `ruff` hook (check-only, no --fix, fails if violations found)

- **flake8** (secondary, syntax only): Select `E9,F63,F7,F82` only (syntax errors + undefined names)
  - Line length: 127 (matches ruff)
  - Excluded: `gsd-scratch-work`, `trading-crab-lib`, `legacy`
  - Pre-commit: `flake8` hook with same --select flags

- **pylint** (informational, exit-zero): Disabled false-positives:
  - Documentation: `C0114,C0115,C0116` (enforce at code-review, not lint time)
  - Complexity: `R0914,R0912,R0915,R0913,R0917,C0302` (legitimate in ML pipelines)
  - Duplication: `R0801` (expected in train/eval/plotting patterns)
  - Class shape: `R0902,R0903` (dataclasses and sklearn objects vary)
  - Test patterns: `W0613,W0212` (fixtures use protected access + unused args)
  - Import errors: `E0401` (optional dependencies)
  - Wildcard imports: `W0401,W0614` (used in plotting/__init__.py re-export pattern)
  - Broad exception catching: `W0718` (network ingestion code, marked `# noqa: BLE001`)
  - Other: `W0104,C0415,R0401,E1136,R0911,E1101,W0621` (false positives on pandas, IPython, fixtures)
  - Entry: `.github/workflows/python-package.yml` runs `pylint src/trading_crab src/trading_crab_lib --rcfile=.pylintrc || true`

- **mypy** (informational, exit-zero): Incremental type-checking
  - Config: `python_version = "3.10"`, `ignore_missing_imports = true`
  - Scope: `files: "^src/"` (library + app only, not tests)
  - Exit-zero: Not strict yet; enable stricter settings incrementally as coverage improves
  - Next targets: `warn_return_any`, `warn_unreachable`, `disallow_untyped_defs`

## Import Organization

- **Order** (handled by ruff's isort integration):
  - Standard library imports (`from pathlib import Path`, `import logging`)
  - Third-party imports (`import pandas as pd`, `from sklearn import ...`)
  - Local imports (`from trading_crab_lib.transforms import ...`, `from . import submodule`)

- **Path aliases:** Not used at project level (absolute imports preferred for clarity)

- **Absolute imports:** Preferred
  - `from trading_crab_lib.transforms import engineer_all`
  - `from trading_crab_lib.platform.labeling import classify_regime`

- **Optional dependencies** (graceful handling in try/except):
  - `hmmlearn`, `statsmodels`, `hdbscan`, `lightgbm`, `k-means-constrained`
  - Availability flags assigned at module load: `_HMM_AVAILABLE = False` in except block
  - Error message provides install instructions: `"Install with: pip install 'trading-crab-lib[ingestion]'"`
  - Tests skip via `pytest.mark.skipif` when optional deps unavailable

- **Lazy imports** (inside function bodies only for):
  - Optional dependencies requiring conditional logic
  - `__getattr__` patterns for convenience re-exports (e.g., `trading_crab_lib.RunConfig`)

## Error Handling

- **Specific exception types always:** No bare `except:`
  - Example: `except (FileNotFoundError, ValueError) as e:` (not `except Exception:`)
  - Broad exception catching only in network ingestion code, marked `# noqa: BLE001` for ruff

- **Fail-fast for config errors:**
  - `validate_config()` in `config.py` checks required sections + scalar types at load time
  - Raises single `ValueError` listing every issue (not cascading failures)
  - Called at end of `load()` before any pipeline step runs

- **Missing/invalid files:** Raise `FileNotFoundError` with full path included
  - Example: `raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")`

- **Network failures:** Caught and logged at WARNING level; pipeline continues with empty/partial data
  - Graceful degradation: missing data is acceptable; ingestion completion report flags it
  - No hard fails for network timeouts or transient errors

## Logging

- **Module-level logger:** Each module starts with `log = logging.getLogger(__name__)`
- **Root logger config:** `setup_logging(level: str = "INFO")` in `config.py`
  - Format: `"%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"`
  - Date format: `"%Y-%m-%d %H:%M:%S"`
- **Verbosity control:** `RunConfig.apply_logging()` sets root to DEBUG if `verbose=True`

**Logging levels by context:**
- **DEBUG:** Checkpoint freshness checks, detailed step progression (only when `--verbose` flag set)
- **INFO:** Normal pipeline progress, checkpoint saves/loads, ingestion completion counts, regime naming
- **WARNING:** Missing/invalid config, corrupt checkpoint metadata, network failures, data staleness
- **ERROR:** Avoided; critical failures raise exceptions instead (use `log.error()` before raising)

**Logging style:**
- Use `%` formatting: `log.info("Processed %d rows in %.2f seconds", n, elapsed)`
- Avoid f-strings in log calls: `log.warning(f"Failed: {msg}")` → `log.warning("Failed: %s", msg)`
- Always include actionable context: `log.warning("Checkpoint is %d days old; consider --refresh", age_days)`

## Comments

- **Algorithm intent:** Explain *why* a non-obvious approach is chosen
  - Example: "Bernstein gap fill in log space (not linear) preserves exponential growth patterns"
  - Example: "Publication-lag shifts GDP +1 quarter to prevent look-ahead bias"

- **Complex math:** Document formula or reference
  - Example: "Derivative of a linear series should be roughly constant; large jumps indicate NaN regions"

- **Intentional simplifications:** Mark with `# ponytail: explanation` naming the simplification and upgrade path
  - Example: `# ponytail: use StandardScaler instead of PCA for speed; revisit if perf OK`

- **Non-obvious dependencies:** Document in comments
  - Example: "# transform order matters: gap-fill AFTER log transform, BEFORE derivatives"

- **Section dividers:** Horizontal lines using `# ── Name ──` (en-dashes, exactly 2 dashes before/after)
  - Used to organize long modules into logical sections (e.g., `# ── Gap filling ──`)

- **Avoid:** Over-commenting obvious code; let docstrings and type hints speak
  - `x = 1  # set x to 1` ← unnecessary
  - `return df.copy()  # preserve input` ← necessary (non-obvious why)

## Function Design

- **Target size:** Short enough to fit on one screen without scrolling (~50 lines max)
- **Longer functions acceptable only for:**
  - Pipeline step orchestrators (inherently complex; use helper functions to break up)
  - Exploration notebooks (linear workflows)

- **Helper functions:** Break into helpers when exceeding ~40 lines
  - Example: `_fetch_one_series()` extracted from `fetch_all()` in `ingestion/fred.py`

- **Arguments:**
  - Config objects passed whole: `cfg: dict[str, Any]` (not unpacked into 5 separate kwargs)
  - RunConfig always as positional parameter: `def step(df: pd.DataFrame, run_cfg: RunConfig)`
  - Optional flags as keyword-only: `def func(df, *, verbose=False, save_plots=True)`
  - Max reasonable count: ~10 args; excess suggests refactoring

- **Return values:**
  - Predictable single type (not `X | None` unless documented as optional)
  - DataFrames/Series preserve index consistency from input
  - Models returned as objects, not dicts (flat API in `prediction/__init__.py`; bundle dicts only in `classifier.py` for test support)
  - Use early returns for guard clauses: `if not df.empty: return df` (not nested if blocks)

- **Input mutation prevention:**
  - Functions don't mutate inputs unless explicitly documented in docstring
  - Use `.copy()` before modifying: `def func(df): df = df.copy(); df["col"] = ...; return df`
  - Tests verify: `def test_does_not_mutate_input(self, raw_macro_df): original_cols = list(raw_macro_df.columns); func(raw_macro_df); assert list(raw_macro_df.columns) == original_cols`

## Module Design

- **Public vs private:**
  - Public functions: No prefix (e.g., `load()`, `fetch_all()`, `build_profiles()`)
  - Private functions/constants: `_prefix` (e.g., `_fetch_one()`, `_MAX_WORKERS`, `_REQUIRED_SECTIONS`)
  - Protected (subclass access): Single underscore `_method()` (documented in docstring)

- **Package-level re-exports:** In `__init__.py` for convenience
  - Used in: `plotting/__init__.py`, `monitoring/__init__.py`
  - Pattern: `from .submodule import *` with explicit `__all__` list (when needed for clarity)
  - Enables: `from trading_crab_lib.plotting import plot_regime_timeline` without knowing submodule
  - Accepted exception to "no wildcard imports" rule

- **Circular dependency prevention:**
  - Lazy imports (`from ... import X` inside function) used sparingly
  - Known good pattern: `__getattr__` for convenience re-exports avoids circular import at module load
  - Example: `trading_crab_lib.__init__.py` uses `__getattr__` to delay `RunConfig` and `CheckpointManager` imports

## Docstrings

- **Format:** Triple-quoted on all public functions
  - One-liner summary, then longer description if needed
  - Args/Returns/Raises sections when needed
  - Example:
    ```python
    def load_config(path: Path | None = None) -> dict[str, Any]:
        """Load config from YAML, validate schema, inject secrets from environment.
        
        Args:
            path: Path to settings.yaml; None uses default config/settings.yaml
            
        Returns:
            Validated config dict with keys: data, fred, multpl, features, ...
            
        Raises:
            ValueError: If config is missing required sections or has invalid types
        """
    ```

- **Module docstrings:** Always present
  - First: Explain module purpose and usage example
  - Key concepts: Link to relevant ADRs or architecture docs
  - Example: `src/trading_crab_lib/platform/transforms_monthly.py` line 1-33

## Special Conventions

**Legacy quarterly pipeline is frozen:**
- `src/trading_crab_lib/transforms.py` (`engineer_all()` and helpers): Do not modify
- Step modules: `pipelines/01_ingest.py` through `pipelines/09_tactics.py`
- Reference implementation: `legacy/unified_script.py` (ground truth for all formulas)
- Modifications to frozen modules break all downstream steps; use the platform layer instead

**Platform layer (active development):**
- All new features in `src/trading_crab_lib/platform/`
- Separate monthly spine, labeling, and evaluation modules
- Imports frozen incumbent modules only for comparison/reference (see `CLAUDE.md` D-01)

---

*Convention analysis: 2026-10-05*
