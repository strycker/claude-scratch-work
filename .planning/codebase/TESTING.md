# Testing Patterns

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


## Test Framework

**Runner:**
- pytest 8.0+ (configured in `pyproject.toml` `[tool.pytest.ini_options]`)
- Test paths: `tests/` (root), `tests/unit/`, `tests/integration/`
- Entry point: `pytest tests/ -v --tb=short` (short traceback for readability)

**Assertion Library:**
- pytest built-in assertions: `assert condition`, `assert x == y`
- pandas testing: `pd.testing.assert_series_equal()`, `pd.testing.assert_frame_equal()`
- NumPy testing: `np.testing.assert_array_almost_equal()`, etc.

**Configuration:**
- `pythonpath = ["src", "scripts"]` (allows importing from `src/` without installation)
- Markers: `@pytest.mark.network`, `@pytest.mark.real_browser` (for tests needing live resources)
- Filter warnings: statsmodels overflow warnings suppressed (harmless numerical artifacts in synthetic test data)
- Collect info: `pytest --collect-only -q` shows all test names; **2699 tests collected** as of 2026-10-05

## Test File Organization

**Location:**
- `tests/conftest.py` — Shared fixtures (checkpoint isolation, synthetic data, logging setup)
- `tests/unit/test_*.py` — Unit tests for specific modules (~60+ test_platform_*.py files)
- `tests/integration/test_*.py` — Integration tests (mini pipeline, wheel smoke tests)
- `tests/test_*.py` — Root-level tests (CLI smoke, pipeline smoke, email, constraints, scripts)

**Naming:**
- Module-focused: `test_<module_name>.py` matches the module it tests
  - Example: `test_transforms.py` → tests `src/trading_crab_lib/transforms.py`
  - Example: `test_platform_transforms.py` → tests `src/trading_crab_lib/platform/transforms_monthly.py`

**Structure — Class-based with descriptive names:**
- `class TestFunctionName:` for each function or feature
- `def test_<behavior>():` with explicit behavior description
- Example:
  ```python
  class TestAddCrossRatios:
      def test_all_ten_columns_added(self, raw_macro_df):
          result = add_cross_ratios(raw_macro_df)
          expected = ["div_yield2", "price_div", ...]
          for col in expected:
              assert col in result.columns
  ```

**Inline configuration (not from settings.yaml):**
- Platform tests use inline config dicts to stay isolated from concurrent settings.yaml edits
- Example from `test_platform_transforms.py`:
  ```python
  SPLICE_CFG: dict = {
      "equities": {
          "research_name": "equities_tr",
          "method": "total_return_from_price_div",
          ...
      },
      ...
  }
  ```

## Fixture Architecture

**Session-scoped checkpoint isolation (autouse, critical):**
- Fixture: `_isolated_checkpoint_dir` in `conftest.py` (lines 159–241)
- Scope: `autouse=True, scope="session"`
- Behavior:
  1. Creates a session-scoped temporary directory
  2. Copies production checkpoints from `data/checkpoints/` into it (read-fallback)
  3. Patches `trading_crab_lib.checkpoints.CHECKPOINT_DIR` → session temp dir
  4. Sets env var `TC_CHECKPOINT_DIR` so subprocesses also use session dir
  5. **CRITICAL:** Also patches `trading_crab_lib.platform.checkpoints.PLATFORM_CHECKPOINT_DIR` and `trading_crab_lib.platform.honesty.holdout.HOLDOUT_CHECKPOINT_DIR` to protect the 2021+ holdout dataset from being overwritten by tests
  6. Synthesizes minimal checkpoints if production copies unavailable (ensures constraint tests always run)
  7. Restores original paths on session teardown

**Why this matters:**
- Production data is **never** read from or written to during `pytest`
- Non-deterministic behavior eliminated: pytest runs don't interfere with pipeline runs
- 2021+ holdout dataset (the one the honesty framework protects) is preserved—test writes go only to session temp dir

**Synthetic data generators (fallback when production data unavailable):**
- `_synthesize_macro_raw(session_dir)` → `macro_raw.parquet` (100 quarters, synthetic columns)
- `_synthesize_features(session_dir)` → `features_noncausal.parquet` + `features_causal.parquet`
  - Calls `engineer_all()` if config + dependencies available; falls back to minimal DataFrame
- `_synthesize_asset_prices(session_dir)` → `asset_prices.parquet` (8 tickers, quarterly index)
  - Uses configured ETF list from `config/platform_settings.yaml`

**Seeded random data:**
- `np.random.default_rng(0)` for reproducibility
- Quarterly indices: `pd.date_range("2000-03-31", periods=N, freq="QE")`
- Ensures tests pass identically across runs

## Test Structure Patterns

**Mocking network calls (no live API access):**
- unittest.mock.patch for ingestion modules
- Example from `test_platform_transforms.py`:
  ```python
  @patch('trading_crab_lib.platform.ingestion.macro_monthly.fetch_macro_monthly')
  def test_build_monthly_spine(self, mock_fetch_macro):
      mock_fetch_macro.return_value = _make_synthetic_macro(idx)
      ...
  ```

**Monkeypatching module constants:**
- Example from `test_platform_transforms.py`:
  ```python
  @pytest.fixture(autouse=True)
  def _redirect_platform_checkpoints(tmp_path, monkeypatch):
      from trading_crab_lib.platform import checkpoints as platform_checkpoints
      monkeypatch.setattr(platform_checkpoints, "PLATFORM_CHECKPOINT_DIR", tmp_path / "platform")
  ```

**Determinism regression tests:**
- Synthetic data with fixed seeds: `rng = np.random.default_rng(0)`
- Verify output is identical on repeated calls
- Example: `test_derivatives_independent_of_market_code` (guard against label-pattern leakage)

**Input mutation guards:**
- Verify functions don't modify input DataFrames
- Example:
  ```python
  def test_does_not_mutate_input(self, raw_macro_df):
      original_cols = list(raw_macro_df.columns)
      add_cross_ratios(raw_macro_df)
      assert list(raw_macro_df.columns) == original_cols
  ```

**DataFrame equality assertions:**
- `pd.testing.assert_series_equal(result["col"], expected, check_names=False)` (allows index/name mismatch)
- `pd.testing.assert_frame_equal(result, expected)` (strict: index, columns, dtypes, values)
- Useful for testing computed features match expected formulas

## Platform-Specific Testing

**Platform test suite:** ~60 test files dedicated to `src/trading_crab_lib/platform/`
- `test_platform_transforms.py` — Monthly feature-table assembly (DATA-01, DATA-03, DATA-04)
- `test_platform_labeling.py` — Regime classification (labeling/ submodule)
- `test_platform_evaluation_*.py` — Evaluation metrics, deflated Sharpe, churn, disagreement
- `test_platform_backtest_*.py` — Backtest driver, baseline, joint driver, costs
- `test_platform_plotting_*.py` — Visualization (allocation, backtest, drift, features, history, regime, etc.)
- `test_platform_walkforward.py` — Walk-forward validation harness
- `test_platform_honesty_registry.py` — Honesty framework (trial registry, locked holdout)
- `test_platform_cv.py` — Cross-validation split logic
- `test_build_platform_data_guard.py` — Data integrity guards (point-in-time, publication lags)

**Data dependencies for platform tests:**
- Real data: Copied from `data/checkpoints/platform/` into session temp dir (read-fallback)
- Synthetic data: Used when real data unavailable; structure matches production
- Monthly spine: Tests verify consistency across monthly/quarterly resamplings
- Holdout dataset: Protected by autouse fixture; never overwritten by test writes

## Mocking Strategy

**No live network calls — all ingestion is mocked:**
- `unittest.mock.patch` for:
  - `trading_crab_lib.ingestion.fred.fetch_all()` → synthetic FRED time series
  - `trading_crab_lib.ingestion.multpl.fetch_all()` → synthetic multpl data
  - `trading_crab_lib.ingestion.assets.fetch_universe_prices()` → synthetic prices
  - `trading_crab_lib.platform.ingestion.alfred.fetch_all_vintages()` → synthetic ALFRED vintages
  - `trading_crab_lib.platform.ingestion.prices_daily.fetch_universe_prices()` → synthetic daily prices

**Test data helpers (conftest.py + test modules):**
- `_make_monthly_index(start, periods)` → `pd.DatetimeIndex` at month-end
- `_make_quarterly_index(start, periods)` → `pd.DatetimeIndex` at quarter-end
- `_make_synthetic_macro(idx)` → DataFrame with all required columns (FRED, multpl, macrotrends)
- `_make_synthetic_features(idx)` → Already-engineered feature DataFrame

**Fixture-provided test data:**
- `raw_macro_df` (conftest.py) — Real production macro_raw if available, else synthesized
- `quarterly_index` (conftest.py) — 305-quarter index (1950-Q1 through ~2025-Q4)
- `cluster_labels`, `profiles_df` — Loaded from checkpoints if available

## Coverage

**Measurement:**
- Tool: pytest-cov (installed via `[dev]` extras)
- Command: `pytest tests/ --cov=src/ --cov-report=html`
- Report: Generates `htmlcov/index.html` with per-file coverage

**Requirements:**
- **Not enforced:** No minimum coverage threshold in CI/CD
- **Reported:** CI prints coverage summary but exits zero regardless
- **Goal:** Trend toward ~80% line coverage; focus on behavior/logic coverage over line coverage

**Coverage gaps (intentional):**
- Optional dependency fallbacks (`_HMM_AVAILABLE`, `_STATSMODELS_AVAILABLE`) — skipped when deps missing
- Network retry logic (ingestion modules) — tested with mocks only
- Rare error paths (corrupt pickle files, disk full) — practical to skip in CI
- Old legacy code (`legacy/`, `gsd-scratch-work/`) — reference only; not active development

## Running Tests

**Pytest command reference:**
```bash
# Run all tests
pytest tests/ -v

# Run with short traceback
pytest tests/ -v --tb=short

# Run specific test file
pytest tests/unit/test_platform_transforms.py -v

# Run specific test class
pytest tests/unit/test_transforms.py::TestAddCrossRatios -v

# Run specific test
pytest tests/unit/test_transforms.py::TestAddCrossRatios::test_all_ten_columns_added -v

# Run with coverage
pytest tests/ --cov=src/ --cov-report=html

# Collect tests (don't run)
pytest --collect-only -q tests/

# Run tests matching a keyword
pytest tests/ -k "transforms" -v

# Run only markers (network, real_browser)
pytest tests/ -m network -v
```

**Parallel execution (not recommended for this codebase):**
- Checkpoint isolation via session fixture makes tests independent at the session level
- However, individual tests may share in-memory state (fixtures); pytest-xdist parallelization is not tested
- Use sequential execution (default) for reliability

## CI/CD Pipeline

**GitHub Actions matrix (.github/workflows/python-package.yml):**
- Runs on: Every push to main, every PR to main
- Matrix: Python 3.10, 3.11, 3.12, 3.13 (parallel jobs, fail-fast: false)
- Installs: Full `[all,dev]` extras + optional packages (k-means-constrained, hdbscan, etc.)

**Build job steps:**
1. **Lint with flake8** — Syntax errors only: `E9,F63,F7,F82`
2. **Lint with ruff** — Full checks: `E,F,W,I,UP` (GitHub output format for PR comments)
3. **Lint with pylint** — Informational only (exit-zero)
4. **Test with pytest** — `pytest tests/ -v --tb=short` (all 2699 tests)
5. **Type-check with mypy** — Informational only (exit-zero)

**Build-pkg job:**
- Verifies both packages build cleanly (sdist + wheel):
  - `python -m build src/trading_crab_lib/`
  - `python -m build .`

## Test Data & Isolation

**Real data (production checkpoints):**
- Copied into session temp dir on pytest startup
- Read-fallback for tests that need realistic data
- Never written back to production directory
- Location after copy: `session_tmp_dir/macro_raw.parquet`, `session_tmp_dir/features_*.parquet`, etc.

**Synthetic data (generated by conftest.py):**
- Used when real data unavailable (e.g., yfinance unreachable, data dir cleared)
- Structure matches production exactly (same columns, same index frequency)
- Seeded with fixed random state for reproducibility
- Minimal: 40-100 rows (enough to test algorithms, not slow)

**Platform-specific isolation:**
- `data/checkpoints/platform/` → session temp dir's `platform/` subdirectory
- `data/checkpoints/holdout/` → session temp dir's `holdout/` subdirectory
- Protects the 2021+ holdout (the one piece of data the honesty framework exists to preserve)

**Pre-migration platform checkpoints (git show, not live working tree):**
- Tests that verify Phase 7/8 RECORDS (numbers measured on unlagged data) read pre-8.1 checkpoints
- Loaded via `get_platform_checkpoint_git_show()` (reads from git history, not working tree)
- Pattern: `conftest.py` lines 243–249

## Special Test Patterns

**Fixture parametrization (when needed):**
- Example: Testing multiple k values for clustering
  ```python
  @pytest.mark.parametrize("k", [2, 3, 4, 5])
  def test_silhouette_score_improves(k):
      ...
  ```

**Skipping tests conditionally:**
- Missing optional dependency:
  ```python
  @pytest.mark.skipif(not _HMM_AVAILABLE, reason="hmmlearn not installed")
  def test_hmm_labels():
      ...
  ```
- Slow or network-dependent:
  ```python
  @pytest.mark.network
  def test_fred_api_call():
      ...
  ```

**Fixtures with setup/teardown:**
- Example: Redirect checkpoints, restore on cleanup
  ```python
  @pytest.fixture(autouse=True)
  def _redirect_platform_checkpoints(tmp_path, monkeypatch):
      monkeypatch.setattr(platform_checkpoints, "PLATFORM_CHECKPOINT_DIR", tmp_path / "platform")
      yield
      # Cleanup happens automatically on session end
  ```

**Test markers (in pyproject.toml):**
```toml
[tool.pytest.ini_options]
markers = [
    "network: test deliberately makes a real network request",
    "real_browser: test launches a real browser",
]
```

## Best Practices

1. **Keep tests focused:** One behavior per test, descriptive name
2. **Use fixtures heavily:** Avoid duplication of setup code via conftest.py
3. **Mock external resources:** No live API calls, network, browser launches (except marked tests)
4. **Seed randomness:** Use `np.random.default_rng(seed)` for reproducibility
5. **Test both happy path and edge cases:** Empty inputs, NaN rows, missing columns, etc.
6. **Preserve input:** Verify functions don't mutate DataFrame arguments
7. **Use parametrize for variants:** Test same logic with different inputs
8. **Comment non-obvious tests:** Why this edge case matters (e.g., "tests look-ahead bias guard")
9. **Checkpoint isolation:** Trust the session fixture; don't create your own tmp_path teardown logic
10. **Real data is precious:** Use synthetic data for unit tests; reserve real data for integration tests

---

*Testing analysis: 2026-10-05*
