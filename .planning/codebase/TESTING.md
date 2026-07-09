# Testing Patterns

**Analysis Date:** 2026-07-09

## Test Framework

**Runner:**
- pytest ~8.0+ (specified in `requirements-dev.txt` and `pyproject.toml`)
- Config: `pyproject.toml` `[tool.pytest.ini_options]` with testpaths, pythonpath, and filterwarnings

**Assertion Library:**
- pytest native assertions (`assert`, `assert ... in ...`)
- pandas testing utilities: `pd.testing.assert_series_equal()`, `pd.testing.assert_frame_equal()`, `pd.testing.assert_index_equal()`
- NumPy approximate assertions: `pytest.approx()`, `np.testing.assert_allclose()`

**Run Commands:**
```bash
# All tests
pytest tests/ -v

# Watch mode (if pytest-watch installed)
pytest tests/ -v --watch

# Coverage report
pytest tests/ --cov=src/trading_crab_lib --cov-report=html

# Specific test file or class
pytest tests/unit/test_transforms.py -v
pytest tests/unit/test_clustering.py::TestReducePca -v

# Skip tests that require optional deps
pytest tests/ --co -q  # collect-only to see which tests are marked skip
```

## Test File Organization

**Location:**
- Unit tests: `tests/unit/test_<module>.py` (covers a single module or class in isolation)
- Integration tests: `tests/integration/test_<feature>.py` (covers multi-step workflows)
- Pipeline smoke tests: `tests/test_pipelines_<name>.py` (at tests/ root level)
- Model/behavior tests: `tests/test_models_<area>.py` (at tests/ root level for cross-module tests)
- Email/reporting tests: `tests/test_<service>.py` (at tests/ root level)

**Naming:**
- Test files: `test_*.py`
- Test functions: `test_<behavior_description>()`
- Test classes: `Test<ComponentName>` (groups related tests with shared fixtures)
- Fixture functions: lowercase with underscores, typically defined in `conftest.py` or inline with `@pytest.fixture`

**Structure:**
```
tests/
├── conftest.py                          ← shared fixtures (session/function scope)
├── fixtures/                            ← test data files (currently empty)
├── unit/
│   ├── test_transforms.py               ← transform function tests
│   ├── test_clustering.py               ← clustering module tests
│   ├── test_ingestion.py                ← HTTP-mocked ingestion tests
│   ├── test_prediction_flat.py          ← flat prediction API tests
│   ├── test_email_weekly.py             ← email delivery tests
│   └── ... (35+ unit test files)
├── integration/
│   ├── __init__.py
│   └── test_mini_pipeline.py            ← synthetic end-to-end (steps 2-4)
├── test_pipelines_ingest_features.py    ← pipeline steps 1-2 smoke tests
├── test_models_regime.py                ← regime classifier bundle API
├── test_models_behavior.py              ← behavior model tests
└── ... (5+ top-level test files)
```

## Test Structure

**Suite Organization:**
```python
# Example from tests/unit/test_clustering.py

class TestReducePca:
    """Group of tests for reduce_pca function."""
    def test_output_shape(self, feature_df):
        pca_df, _, _ = reduce_pca(feature_df, n_components=5)
        assert pca_df.shape == (len(feature_df), 5)

    def test_index_preserved(self, feature_df):
        pca_df, _, _ = reduce_pca(feature_df, n_components=5)
        pd.testing.assert_index_equal(pca_df.index, feature_df.index)

class TestFitClusters:
    """Separate class for different function."""
    def test_both_columns_present(self, feature_df):
        result = fit_clusters(feature_df, best_k=3, balanced_k=5)
        assert "cluster" in result.columns
```

**Patterns:**
- **Arrange-Act-Assert:** Each test has explicit setup, action, verification
  ```python
  def test_gap_fill_idempotent(self, quarterly_index):
      # Arrange
      df = self._make_gapped_df(quarterly_index)
      # Act
      result1 = apply_gap_fill(df.copy())
      result2 = apply_gap_fill(df.copy())
      # Assert
      pd.testing.assert_frame_equal(result1, result2)
  ```

- **No setup/teardown:** Fixtures handle all initialization; no `setUp()` / `tearDown()` methods
- **Fixtures over test data files:** Temporary data synthesized in fixtures (`_synthesize_macro_raw()`, `_synthesize_features()`) rather than committed to repo
- **Isolation via fixtures:** Session-scoped `_isolated_checkpoint_dir` redirects all checkpoint I/O to pytest temp directory — production data never written during tests

## Mocking

**Framework:** `unittest.mock` (stdlib) — `patch`, `MagicMock`

**Patterns:**
```python
# Example from tests/unit/test_ingestion.py

@patch("trading_crab_lib.ingestion.multpl.time.sleep")
@patch("trading_crab_lib.ingestion.multpl.requests.get")
def test_multpl_scrape_raw_rows(mock_get, mock_sleep):
    """Mock HTTP calls for scraper tests — no real network access."""
    mock_get.return_value = _FakeResponse(SAMPLE_MULTPL_HTML)
    rows = _scrape_raw_rows("https://example.com/table")
    assert len(rows) == 3

# Mocking FRED API
@patch("trading_crab_lib.ingestion.fred.Fred")
def test_fred_fetch_all_basic(mock_fred_cls):
    """Mock fredapi.Fred() constructor and .get_series() method."""
    mock_fred = MagicMock()
    mock_fred.get_series.return_value = _make_mock_fred_series()
    mock_fred_cls.return_value = mock_fred
    
    df = fetch_all(cfg)
    assert "fred_gdp" in df.columns
```

**What to Mock:**
- Network calls: `requests.get`, `fredapi.Fred`, `yfinance.download`
- External services: any HTTP/API endpoint
- Time-sensitive operations: `time.sleep` (for rate-limiting tests)
- File I/O across checkpoint boundaries: handled by `conftest.py` fixture isolation, not individual test mocks

**What NOT to Mock:**
- Core business logic functions: call the real implementation to verify behavior
- Transformations (gap fill, derivatives, log transforms): test with synthetic data, not mocks
- Clustering algorithms: call sklearn directly to verify geometry

**Fixture patterns:**
```python
# From conftest.py

@pytest.fixture(autouse=True, scope="session")
def _isolated_checkpoint_dir(tmp_path_factory: pytest.TempPathFactory):
    """Session-scoped: redirect all checkpoint I/O to tmp dir.
    
    autouse=True means every test session gets this fixture automatically.
    Production data/checkpoints/ is never touched.
    """
    session_dir = tmp_path_factory.mktemp("checkpoints", numbered=False)
    # Copy production checkpoints into session_dir for read-based tests
    # Patch CHECKPOINT_DIR module variable + env var
    # Synthesize minimal stand-in checkpoints when production data missing
    yield session_dir
    # Restore on teardown

@pytest.fixture
def feature_df(quarterly_index):
    """Per-test fixture: fresh feature matrix."""
    rng = np.random.default_rng(42)
    n = 70
    index = pd.date_range("2000-03-31", periods=n, freq="QE")
    return pd.DataFrame(rng.standard_normal((n, 10)), index=index, columns=[f"f{i}" for i in range(10)])
```

## Fixtures and Factories

**Test Data:**
```python
# Minimal fixtures in conftest.py

@pytest.fixture
def quarterly_index():
    """DatetimeIndex with quarterly frequency for test DataFrames."""
    return pd.date_range("2000-03-31", periods=300, freq="QE")

@pytest.fixture
def raw_macro_df(quarterly_index):
    """Synthetic macro DataFrame matching the schema ingested by step 1."""
    rng = np.random.default_rng(0)
    n = len(quarterly_index)
    return pd.DataFrame({
        "sp500": np.abs(rng.uniform(300, 5000, n)) + 200,
        "dividend": rng.uniform(10, 80, n),
        "fred_gdp": rng.uniform(5000, 25000, n),
        "fred_gnp": rng.uniform(4800, 24000, n),
        # ... more columns matching ingestion output
    }, index=quarterly_index)
```

**Factories (helper functions):**
```python
# From conftest.py (synthesis functions)

def _synthesize_macro_raw(session_dir: Path) -> None:
    """Write a minimal synthetic macro_raw checkpoint."""
    # Mirrors ingestion output structure for constraint tests

def _synthesize_features(session_dir: Path) -> None:
    """Write synthetic features checkpoints by running engineer_all()."""
    # Exact schema matching pipeline output
```

**Location:**
- Shared fixtures: `tests/conftest.py` (session/module/function scope)
- Per-test-class fixtures: defined inline in the test class with `@pytest.fixture`
- Temporary data: synthesized via helpers, never committed as files

## Coverage

**Requirements:** No strict minimum enforced; target is 80%+ for critical paths

**View Coverage:**
```bash
pytest tests/ --cov=src/trading_crab_lib --cov-report=term-missing
pytest tests/ --cov=src/trading_crab_lib --cov-report=html  # opens index.html
```

**Coverage gaps noted in CLAUDE.md:**
- Some optional-dependency modules skip when libraries unavailable (HMM, Markov, HDBSCAN, LightGBM)
- Pre-1993 asset data uses proxies only (gold/oil prices unavailable before macrotrends.net backfill)
- Behavior model tests incomplete in early phases

**CI/CD:** `pytest --cov` runs in GitHub Actions but does not fail on coverage threshold (informational only)

## Test Types

**Unit Tests:**
- Scope: single function or class in isolation
- Data: synthetic (fixture-based), no external dependencies
- File location: `tests/unit/test_<module>.py`
- Example: `test_gap_fill_interior_nans_filled()` — call `apply_gap_fill()` on synthetic DataFrame with known NaN positions
- ~500+ unit tests across 30+ test files

**Integration Tests:**
- Scope: multi-step workflow (e.g., steps 2-4: features → clustering → regimes)
- Data: synthetic DataFrames that mimic step outputs
- No checkpoint I/O; no network calls
- File location: `tests/integration/test_mini_pipeline.py`
- Tests: determinism regression, column preservation, NaN handling across pipeline
- ~14 integration tests verifying end-to-end consistency

**Smoke Tests:**
- Scope: CLI and pipeline entry point dispatch (not full execution)
- Data: mocked or minimal fixtures
- File location: `tests/test_pipeline_smoke.py`, `tests/test_cli_smoke.py`
- Tests: argument parsing, step function dispatch, error handling
- ~20 smoke tests verifying CLI wiring

**No E2E tests:** Full pipeline requires fresh network data (10 min runtime) — not run in CI; developer runs manually with `tradingcrab --refresh --recompute`

## Common Patterns

**Async Testing:**
- Not used; pipeline is synchronous
- ThreadPoolExecutor used in FRED ingestion but not tested as async (verified with mocked responses)

**Error Testing:**
```python
# Example from tests/

def test_config_missing_required_section():
    """validate_config raises ValueError with all errors in one message."""
    cfg = {"data": {}}  # missing "fred", "multpl", etc.
    with pytest.raises(ValueError) as exc_info:
        validate_config(cfg)
    assert "validation error(s)" in str(exc_info.value)
    assert "Missing required section" in str(exc_info.value)

def test_checkpoint_not_found():
    """CheckpointManager.load raises FileNotFoundError with path."""
    cm = CheckpointManager(checkpoint_dir=tmp_path)
    with pytest.raises(FileNotFoundError) as exc_info:
        cm.load("nonexistent")
    assert "Checkpoint not found" in str(exc_info.value)
```

**Determinism Tests:**
```python
# Example from tests/unit/test_transforms.py

class TestGapFillDeterminism:
    """Verify gap fill is idempotent and independent of market_code."""
    
    def test_gap_fill_idempotent(self, quarterly_index):
        """Running gap fill twice gives identical output."""
        df = self._make_gapped_df(quarterly_index)
        result1 = apply_gap_fill(df.copy())
        result2 = apply_gap_fill(df.copy())
        pd.testing.assert_frame_equal(result1, result2)
    
    def test_gap_fill_independent_of_market_code(self, quarterly_index):
        """Gap fill on col X must not change when market_code is added/changed."""
        df_no_mc = self._make_gapped_df(quarterly_index)
        result_no_mc = apply_gap_fill(df_no_mc.copy())
        
        df_with_mc = df_no_mc.copy()
        df_with_mc["market_code"] = [0 if i % 2 == 0 else 1 for i in range(len(df_no_mc))]
        result_with_mc = apply_gap_fill(df_with_mc.copy())
        
        # Feature column should be identical regardless of market_code presence
        pd.testing.assert_series_equal(result_no_mc["x"], result_with_mc["x"])
```

**Optional Dependency Skipping:**
```python
# Example from tests/unit/test_hmm.py

@pytest.mark.skipif(not _HMM_AVAILABLE, reason="hmmlearn not installed")
def test_fit_hmm_returns_scores():
    """Skip entire test if hmmlearn unavailable."""
    result = fit_hmm(pca_df, k_range=range(2, 4))
    assert "k" in result.columns
    assert "bic" in result.columns
```

**Warning Suppression:**
- Configured in `pyproject.toml` `[tool.pytest.ini_options] filterwarnings` for warnings that occur across multiple test runs (statsmodels overflow warnings, numpy divide-by-zero, etc.)
- Per-test suppression via `@pytest.mark.filterwarnings("ignore::...")` when specific to one test
- Rationale: third-party libraries generate harmless numerical artefacts on synthetic data

---

*Testing analysis: 2026-07-09*
