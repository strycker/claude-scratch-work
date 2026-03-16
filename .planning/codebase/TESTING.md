# Testing Patterns

**Analysis Date:** 2026-03-16

## Test Framework

**Runner:**
- pytest (dev dependency)
- Config: `pyproject.toml` (`[tool.pytest.ini_options]`)

**Assertion Library:**
- pytest built-in assertions (`assert ...`) plus targeted pandas/numpy helpers where appropriate.
  - Example: `pd.testing.assert_frame_equal(...)` in `tests/unit/test_checkpoints.py`
  - Example: `np.isfinite(...)` assertions in `tests/unit/test_returns.py`

**Run Commands:**

```bash
make test                          # Run all tests (pytest tests/ -v)
make test-fast                     # Stop at first failure (pytest -x -q)
pytest tests/ -v                   # Run all tests (explicit)
pytest tests/ --cov --cov-report=term-missing  # Coverage (pytest-cov installed)
```

## Test File Organization

**Location:**
- Centralized `tests/` directory with `tests/unit/` for unit tests.
  - Examples: `tests/unit/test_transforms.py`, `tests/unit/test_density.py`

**Naming:**
- `test_*.py` files.
- Test functions are grouped into `class Test...:` blocks per unit under test.
  - Example: `class TestApplyDerivatives:` in `tests/unit/test_transforms.py`

**Structure:**

```
tests/
├── conftest.py
└── unit/
    ├── test_checkpoints.py
    ├── test_transforms.py
    ├── test_clustering.py
    ├── test_density.py
    └── ...
```

## Test Structure

**Suite Organization:**

```typescript
# Not applicable (Python-only repo).
```

**Patterns:**
- **Arrange/Act/Assert** via direct local variables and explicit expected values.
  - Example: median calculation check in `tests/unit/test_returns.py`
- **Exception testing** uses `pytest.raises(..., match="...")` to validate message content.
  - Examples: `tests/unit/test_density.py`, `tests/unit/test_clustering_exploration.py`
- **Fixture-driven synthetic data** using numpy RNG for determinism.
  - Example: shared fixtures in `tests/conftest.py`

## Mocking

**Framework:** pytest `monkeypatch`

**Patterns:**

```typescript
# Not applicable (Python-only repo).
```

Observed mocking approach:
- Optional dependencies are tested by monkeypatching `builtins.__import__` to raise `ImportError`.
  - Example: `tests/unit/test_density.py` (`hdbscan` missing cases)
  - Example: `tests/unit/test_clustering_exploration.py` (`kneed` missing case)

**What to Mock:**
- Optional dependency imports to validate clean fallbacks and explicit error messages.
  - Example: `hdbscan` and `kneed` tests above

**What NOT to Mock:**
- Core numeric logic is tested directly with synthetic data (no heavy mocking).
  - Example: `reduce_pca()` and k-means scoring in `tests/unit/test_clustering.py`

## Fixtures and Factories

**Test Data:**

```typescript
# Not applicable (Python-only repo).
```

Key fixture patterns:
- Shared fixtures live in `tests/conftest.py` and provide reusable indices and small synthetic DataFrames/Series.
  - `quarterly_index`: `pd.date_range(..., freq="QE")`
  - `raw_macro_df`: minimal macro columns required by transforms
  - `cluster_labels`: cycling integer regimes
  - `asset_prices`: synthetic price history for return calculations

**Location:**
- `tests/conftest.py` for cross-suite fixtures.
- Some tests define local fixtures in-file when tightly coupled to a module.
  - Example: `pca_df` in `tests/unit/test_density.py`

## Coverage

**Requirements:** None enforced via config (pytest-cov present, but no coverage config detected).

**View Coverage:**

```bash
pytest tests/ --cov --cov-report=term-missing
```

## Test Types

**Unit Tests:**
- Dominant test type: pure functions and deterministic computations with synthetic data.
- Scope includes:
  - Feature engineering transforms (`tests/unit/test_transforms.py`)
  - Clustering utilities (`tests/unit/test_clustering.py`, `tests/unit/test_clustering_exploration.py`)
  - Optional clustering backends and sweeps (`tests/unit/test_density.py`)
  - Checkpoint persistence and metadata (`tests/unit/test_checkpoints.py`)

**Integration Tests:**
- Not detected (no end-to-end pipeline runs under `tests/`).

**E2E Tests:**
- Not used.

## Common Patterns

**Async Testing:**
- Not used; concurrency in production code (e.g., FRED ingestion) is not exercised via async test helpers.

**Error Testing:**
- Standard pattern uses `pytest.raises` with `match=...` to assert both type and message.
  - Example: empty input guards in `tests/unit/test_density.py`

Additional notes:
- Tests generally import the package directly as `market_regime.*`. The repo also sets `pythonpath = ["src"]` for pytest in `pyproject.toml`, but several tests additionally do a manual `sys.path.insert(...)`.
  - Example: `sys.path.insert(0, ...)` in `tests/unit/test_checkpoints.py` and `tests/unit/test_transforms.py`

---

*Testing analysis: 2026-03-16*
