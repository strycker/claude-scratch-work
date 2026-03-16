---
phase: 01-null
plan: 03
type: execute
wave: 3
depends_on:
  - 01-null-01
  - 01-null-02
files_modified:
  - tests/test_constraints_etf_universe.py
  - tests/test_constraints_frequency.py
  - tests/test_pipelines_ingest_features.py
  - config/settings.yaml
autonomous: true
requirements:
  - DATA-01
  - DATA-03
  - CONSTR-01
  - CONSTR-02
must_haves:
  truths:
    - "Core data and feature artifacts only contain ETFs from the configured ETF universe; non-ETF tickers cause tests to fail."
    - "Core artifacts operate at monthly/quarterly resolutions; sub-daily indices are rejected by tests."
    - "Ingestion and feature pipelines can be exercised via tests without network access by mocking external calls."
    - "Constraints around ETF-only universe and non-intraday, non-auto-trading behavior are enforced by automated tests, not just documentation."
  artifacts:
    - path: "config/settings.yaml"
      provides: "Single source of truth for the ETF universe used by constraint tests."
    - path: "tests/test_constraints_etf_universe.py"
      provides: "Tests that assert only configured ETFs appear in core artifacts and that non-ETF tickers cause failures."
    - path: "tests/test_constraints_frequency.py"
      provides: "Tests that assert core artifacts use monthly/quarterly frequencies and reject sub-daily data."
    - path: "tests/test_pipelines_ingest_features.py"
      provides: "Smoke tests for pipelines/01_ingest.py and 02_features.py using mocks, ensuring pipelines can run under Nyquist without network."
  key_links:
    - from: "tests/test_constraints_etf_universe.py"
      to: "data/checkpoints/asset_prices.parquet"
      via: "loading checkpointed ETF prices and checking tickers against config ETF universe"
    - from: "tests/test_constraints_etf_universe.py"
      to: "data/checkpoints/features_noncausal.parquet"
      via: "loading feature artifacts and asserting ETF columns align with config ETF universe"
    - from: "tests/test_constraints_frequency.py"
      to: "data/checkpoints/macro_raw.parquet"
      via: "loading macro data and asserting quarterly index"
    - from: "tests/test_constraints_frequency.py"
      to: "data/checkpoints/features_causal.parquet"
      via: "loading causal features and asserting quarterly index with no sub-daily frequencies"
    - from: "tests/test_pipelines_ingest_features.py"
      to: "pipelines/01_ingest.py"
      via: "mocked runs that assert correct CheckpointManager interactions"
    - from: "tests/test_pipelines_ingest_features.py"
      to: "pipelines/02_features.py"
      via: "mocked runs that assert feature artifacts are produced without network access"
---

<objective>
Enforce ETF-only and non-intraday constraints, and validate the ingestion and feature pipelines via automated tests that operate on checkpointed data and mocks rather than live network calls.

Purpose: Turn Phase 1’s data and constraint requirements into failing tests that guard against regressions and verify DATA-01/03 and CONSTR-01/02 in practice.
Output: New constraint-focused tests and pipeline smoke tests wired into the existing pytest suite.
</objective>

<execution_context>
@.planning/PROJECT.md
@.planning/REQUIREMENTS.md
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/phases/01-null/01-RESEARCH.md
@.planning/phases/01-null/01-null-01-PLAN.md
@.planning/phases/01-null/01-null-02-PLAN.md
</execution_context>

<context>
@CLAUDE.md
@config/settings.yaml
@tests/
@pipelines/01_ingest.py
@pipelines/02_features.py
</context>

<tasks>

<task type="auto">
  <name>Task 1: Add tests enforcing ETF-only universe in core artifacts</name>
  <files>
config/settings.yaml
tests/test_constraints_etf_universe.py
  </files>
  <action>
- In `config/settings.yaml`, ensure the ETF universe is clearly defined in a single list (e.g. `assets.etfs`) that constraint tests can import or load via `market_regime.config.load()`.
- Create `tests/test_constraints_etf_universe.py` with tests that:
  - Load the ETF universe from config via the public config API.
  - Load `asset_prices` and feature checkpoints (e.g. `features_noncausal`, `features_causal`) via `CheckpointManager` or direct parquet reads.
  - Assert that all ticker columns present in these artifacts are subsets of the configured ETF universe.
  - Optionally include a negative test case that constructs a small dummy DataFrame with an out-of-universe ticker and asserts that a helper validation function raises an error.
- Use existing test fixtures and mocking patterns from `tests/conftest.py` where appropriate to avoid network access.
  </action>
  <verify>
- Run `pytest tests/ -k "constraints_etf_universe" -v` to execute the new tests.
  </verify>
  <done>
- Tests fail if any non-ETF ticker appears in core price or feature artifacts, tying CONSTR-01 directly to automated validation.
  </done>
</task>

<task type="auto">
  <name>Task 2: Add tests enforcing non-intraday frequency and cadence</name>
  <files>
tests/test_constraints_frequency.py
  </files>
  <action>
- Create `tests/test_constraints_frequency.py` with tests that:
  - Load macro, price, and feature checkpoints via `CheckpointManager` or direct parquet reads.
  - Assert that:
    - Macro artifacts (e.g. `macro_raw`) have a quarterly index as expected.
    - ETF price artifacts (`asset_prices`) are at monthly/quarterly resolution and do not contain sub-daily timestamps.
    - Feature artifacts (`features_noncausal`, `features_causal`) use a quarterly index with no sub-daily frequency.
- Include edge-case checks where index frequency is inferred (e.g. via Pandas frequency inference) and compared against expected values.
- Ensure tests do not depend on live scraping or external APIs; rely only on existing checkpoints or fixtures.
  </action>
  <verify>
- Run `pytest tests/ -k "constraints_frequency" -v` to execute frequency constraint tests.
  </verify>
  <done>
- Tests fail if any core artifact uses sub-daily resolution or deviates from the intended monthly/quarterly cadence, enforcing CONSTR-02.
  </done>
</task>

<task type="auto">
  <name>Task 3: Add smoke tests for ingestion and feature pipelines using mocks</name>
  <files>
tests/test_pipelines_ingest_features.py
  </files>
  <action>
- Create `tests/test_pipelines_ingest_features.py` with tests that:
  - Use pytest and existing mocking patterns to patch network-dependent calls in `ingestion.fred`, `ingestion.multpl`, and `ingestion.assets`, returning small synthetic DataFrames.
  - Invoke `pipelines/01_ingest.py` and `pipelines/02_features.py` via their `main(run_cfg: RunConfig)` entrypoints with a test `RunConfig` (no plots, save_plots=False where needed).
  - Assert that:
    - `CheckpointManager` is called to write expected checkpoints (`macro_raw`, `asset_prices`, `features_noncausal`, `features_causal`).
    - The pipelines complete without hitting the network or raising errors.
  - Optionally expose a narrow CLI-style harness (e.g. calling `run_pipeline.py` with limited steps under mocks) if that provides better coverage.
- Keep tests fast and deterministic so they can be run as part of the standard Phase 1 validation loop.
  </action>
  <verify>
- Run `pytest tests/ -k "pipelines_ingest_features" -v` to execute the new smoke tests.
  </verify>
  <done>
- Ingestion and feature pipelines can be exercised under Nyquist validation using mocks, with checkpoints written as expected and no reliance on external services.
  </done>
</task>

</tasks>

<verification>
- `pytest tests/ -k "constraints_etf_universe or constraints_frequency or pipelines_ingest_features" -v` passes, confirming ETF-only, non-intraday constraints and pipeline smoke behavior.
- A full `pytest tests/ -v` run passes after earlier Phase 1 plans are executed, integrating the new constraint tests with the existing suite.
</verification>

<success_criteria>
- Violations of ETF-only or non-intraday constraints in core artifacts cause automated tests to fail, preventing silent regressions.
- Ingestion and feature pipelines can be run under test conditions without network access, validating their contract with checkpoints and config.
- Phase 1’s DATA-01/03 and CONSTR-01/02 requirements are backed by concrete, repeatable tests.
</success_criteria>

<output>
After completion, ensure Phase 1 verification and any future `/gsd:verify-work` steps reference these constraint and pipeline tests as part of the standard validation checklist.
</output>

