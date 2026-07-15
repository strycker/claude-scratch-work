---
phase: 01-monthly-data-layer-long-histories
plan: 05
subsystem: infra
tags: [ingestion, fred, multpl, macrotrends, monthly-resample, pandas, pytest]

# Dependency graph
requires:
  - phase: 01-monthly-data-layer-long-histories (plan 01)
    provides: "platform/ subpackage scaffold, config/platform_settings.yaml (fred_monthly/multpl_monthly/macrotrends_monthly blocks)"
provides:
  - "trading_crab_lib.platform.ingestion.macro_monthly — fetch_fred_monthly(), _scrape_multpl_monthly(), _scrape_macrotrends_monthly(), fetch_macro_monthly()"
  - "Monthly (not quarterly) FRED/multpl/macrotrends raw ingestion, merged NULL-tolerantly into one wide DataFrame"
affects: [01-04, 01-06, 01-07]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Thin monthly analog of a frozen quarterly fetcher: duplicate client-construction + fetch skeleton, override only the resample rule (never edit the frozen module)"
    - "pd.concat([...], axis=1) as the sole cross-source merge primitive (outer join, NULL-tolerant)"

key-files:
  created:
    - src/trading_crab_lib/platform/ingestion/macro_monthly.py
    - tests/unit/test_platform_macro_ingest.py
  modified: []

key-decisions:
  - "macrotrends monthly scrape duplicates the JSON-key-detection parse loop and HTML-table fallback inline (not calling macrotrends._scrape_series/_scrape_series_html_table) since both incumbent functions bake in a hardcoded quarterly resample that cannot be overridden from outside — only _extract_json_data (which has no resample logic) is reused by import, satisfying D-01's 'import, never edit' rule while still reaching true monthly cadence"
  - "multpl monthly scrape reuses multpl._scrape_raw_rows (import) and multpl._SUFFIX_MAP (import) directly since neither bakes in a resample rule — only the final .resample() call needed a monthly-cadence replacement, written as a new _parse_multpl_series_monthly() analog of multpl._parse_series"
  - "Test mocking for the orchestrator test patches requests.get once at trading_crab_lib.platform.ingestion.macro_monthly.requests.get rather than separately at the multpl module's import path — both modules' `import requests` bind the same singleton module object, so patching either path patches requests.get everywhere; a single URL-discriminating side_effect covers both the multpl and macrotrends call sites in one test"

patterns-established:
  - "Monthly ingestion functions are named identically to their quarterly-incumbent counterparts (fetch_fred_monthly vs fetch_all, etc.) but live under platform.ingestion — no naming collision since they're separate modules, and the parallel naming keeps the analog relationship obvious to future readers"

requirements-completed: [DATA-01]

coverage:
  - id: D1
    description: "FRED market/fast series fetched at native frequency and resampled to monthly (ME), not quarterly, reusing fred.py's client/parallel-fetch pattern without editing the frozen module"
    requirement: "DATA-01"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_macro_ingest.py::test_fetch_fred_monthly_produces_monthly_cadence_not_quarterly — pass"
      - kind: unit
        ref: "tests/unit/test_platform_macro_ingest.py::test_fetch_fred_monthly_basic — pass"
      - kind: unit
        ref: "tests/unit/test_platform_macro_ingest.py::test_fetch_fred_monthly_handles_single_series_failure — pass"
    human_judgment: false
  - id: D2
    description: "multpl (sp500, div_yield, cape) and macrotrends (gold, wti) parsed to monthly frequency by reusing the incumbent's low-level parsers/extractors, never editing the frozen modules"
    requirement: "DATA-01"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_macro_ingest.py::test_scrape_multpl_monthly_basic — pass (or skip if cssselect unavailable)"
      - kind: unit
        ref: "tests/unit/test_platform_macro_ingest.py::test_scrape_macrotrends_monthly_month_end_indexed — pass"
    human_judgment: false
  - id: D3
    description: "fetch_macro_monthly(cfg) returns ONE wide monthly DataFrame merged via pd.concat(axis=1) (NULL-tolerant) even when a source fails entirely"
    requirement: "DATA-01"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_macro_ingest.py::test_fetch_macro_monthly_null_tolerant_partial_success — pass"
      - kind: unit
        ref: "tests/unit/test_platform_macro_ingest.py::test_fetch_macro_monthly_all_sources_empty_returns_empty_df — pass"
    human_judgment: false
  - id: D4
    description: "ingestion/fred.py, macrotrends.py, and multpl.py remain byte-identical (unmodified) — no edits to the frozen incumbent"
    verification:
      - kind: unit
        ref: "git diff --stat -- src/trading_crab_lib/ingestion/fred.py src/trading_crab_lib/ingestion/macrotrends.py src/trading_crab_lib/ingestion/multpl.py — empty output"
    human_judgment: false
  - id: D5
    description: "Full incumbent test suite remains green; only pre-existing sibling-worktree failure present (out of scope for this plan)"
    verification:
      - kind: unit
        ref: "pytest tests/ -q — 784 passed, 49 skipped, 1 pre-existing failure (test_platform_prices_ingest.py, from sibling plan 01-06, unrelated to this plan's files)"
    human_judgment: false

duration: ~35min
completed: 2026-07-15
status: complete
---

# Phase 1 Plan 05: Monthly Macro/Long-History Raw Ingestion Summary

**Thin monthly analogs of the frozen `fred.py`/`multpl.py`/`macrotrends.py` fetchers — same client-construction and parse patterns, resampled to month-end (`"ME"`) instead of quarterly, merged into one wide `pd.concat(axis=1)` frame — with zero edits to the incumbent modules.**

## Performance

- **Duration:** ~35 min
- **Completed:** 2026-07-15
- **Tasks:** 3
- **Files modified:** 2 (both new; 0 incumbent files touched)

## Accomplishments
- `fetch_fred_monthly(cfg)` fetches every `cfg["fred_monthly"]["series"]` entry via the same `Fred` client-construction + `ThreadPoolExecutor` + try/except-WARNING skeleton as `ingestion/fred.py::fetch_all()`, but resamples to `cfg["data"]["monthly_freq"]` (`"ME"`) instead of the incumbent's hardcoded quarterly rule
- `_scrape_multpl_monthly(cfg)` reuses `multpl.py`'s importable `_scrape_raw_rows()` and `_SUFFIX_MAP` to produce monthly sp500/div_yield/cape series; degrades gracefully (WARNING + skip) when `cssselect` is unavailable, matching the incumbent multpl test's own skip convention
- `_scrape_macrotrends_monthly(...)` reuses `macrotrends.py`'s importable `_extract_json_data()` JSON extractor for the parse loop, with an HTML-table fallback duplicated inline (the incumbent's own fallback bakes in a quarterly resample and cannot be reused as-is)
- `fetch_macro_monthly(cfg)` merges all three sources into ONE wide monthly `DataFrame` via a single `pd.concat([...], axis=1)` — outer join, NULL-tolerant — never `pd.merge`/`.join` defaults
- 11 mocked unit tests in `tests/unit/test_platform_macro_ingest.py`: no live network calls, asserting monthly cadence (>2x a quarterly resample of the same data), month-end indexing, and NULL-tolerant partial-source-failure merging

## Task Commits

1. **Task 1: FRED monthly fetch helper (reuse client pattern, resample "ME")** - `818efe6` (feat)
2. **Task 2: multpl + macrotrends monthly parsers + macro merge orchestrator** - `b1ae7bd` (feat)
3. **Task 3: Monthly macro ingestion test (row count, monthly cadence, NULL-tolerant merge)** - `7d8f0da` (test)

## Files Created/Modified
- `src/trading_crab_lib/platform/ingestion/macro_monthly.py` - `_fetch_fred_monthly()`/`fetch_fred_monthly()` (FRED, monthly resample), `_parse_multpl_series_monthly()`/`_scrape_multpl_monthly()` (multpl reuse), `_scrape_macrotrends_html_table_monthly()`/`_scrape_macrotrends_monthly()`/`_fetch_macrotrends_monthly_all()` (macrotrends reuse), `fetch_macro_monthly()` (NULL-tolerant `pd.concat(axis=1)` orchestrator)
- `tests/unit/test_platform_macro_ingest.py` - 11 tests covering all three source fetchers plus the merge orchestrator, all network mocked

## Decisions Made
- macrotrends monthly scrape duplicates the JSON-key-detection loop and HTML-table fallback inline rather than calling `macrotrends._scrape_series`/`_scrape_series_html_table` — both incumbent functions hardcode a quarterly resample internally with no override hook, so only the resample-free `_extract_json_data()` helper could be reused by import (D-01 compliant: import, never edit).
- multpl monthly scrape reuses `multpl._scrape_raw_rows` and `multpl._SUFFIX_MAP` directly since neither bakes in a resample rule; only a new `_parse_multpl_series_monthly()` analog of `multpl._parse_series` was needed to swap the final `.resample()` call to `"ME"`.
- The orchestrator test patches `requests.get` at a single qualified name (`trading_crab_lib.platform.ingestion.macro_monthly.requests.get`) rather than separately patching the `multpl` module's import path, because both modules' `import requests` statements bind the identical singleton module object — patching the attribute via either qualified name has the same global effect, and a URL-discriminating `side_effect` cleanly covers both call sites in one test.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Removed literal `"QE"` quoted strings from module docstrings/comments**
- **Found during:** Task 1, running the plan's automated verify command
- **Issue:** Task 1's `<verify>` script scans the module's source text for the literal substring `"QE"` (quoted) to confirm no quarterly resample is hardcoded. The initial docstring text described the incumbent's behavior using the literal quoted string `` `` "QE" `` ``, which the substring check flagged even though no actual `.resample("QE")` call exists anywhere in the file.
- **Fix:** Reworded the two docstring passages to describe the incumbent's quarterly resample without using the literal quoted `"QE"` string (e.g. "hardcoded quarterly period-end rule" instead of `` `"QE"` ``).
- **Files modified:** `src/trading_crab_lib/platform/ingestion/macro_monthly.py`
- **Verification:** `python -c "... assert '\"QE\"' not in src ..."` passes; `grep -n 'QE'` confirms zero occurrences
- **Committed in:** `818efe6` (Task 1 commit)

**2. [Rule 1 - Bug] Fixed shared-singleton `requests.get` mock collision in the orchestrator test**
- **Found during:** Task 3, running the new test file for the first time
- **Issue:** An initial version of `test_fetch_macro_monthly_null_tolerant_partial_success` patched `requests.get` at two different qualified names (`trading_crab_lib.ingestion.multpl.requests.get` and `trading_crab_lib.platform.ingestion.macro_monthly.requests.get`), assuming each patch would independently control the `requests.get` calls made from its respective module. Since both modules' `import requests` statements bind the same singleton `requests` module object, the second-applied patch silently overrode the first for the entire test body — both the multpl and macrotrends fetch attempts ended up hitting the same mock, causing the macrotrends assertion to fail with the multpl-only error message.
- **Fix:** Collapsed the two patches into one, with a URL-discriminating `side_effect` function that raises for multpl URLs and returns the synthetic macrotrends fixture for macrotrends URLs — correctly exercising the intended partial-source-failure scenario.
- **Files modified:** `tests/unit/test_platform_macro_ingest.py`
- **Verification:** `pytest tests/unit/test_platform_macro_ingest.py -x -q` → 10 passed, 1 skipped (cssselect unavailable in this sandbox)
- **Committed in:** `7d8f0da` (Task 3 commit)

---

**Total deviations:** 2 auto-fixed (1 Rule 3 — verify-script compliance wording fix, 1 Rule 1 — test mock bug). Both essential for correctness; no scope creep.

## Issues Encountered
- **Editable-install path collision (environment, not a plan defect):** the shared `site-packages` editable-install `.pth` for `trading-crab-lib` pointed at a different (now-deleted) sibling worktree's `src/` directory, so `import trading_crab_lib` failed with the global interpreter. Worked around by prefixing every verification/test command with `PYTHONPATH="$(pwd)/src:$PYTHONPATH"` scoped to this worktree — no global `pip install`/`.pth` rewrite was performed, since that would risk breaking concurrent sibling agents' verification in the shared `dist-packages` directory.
- **Pre-existing failure outside this plan's scope:** `pytest tests/ -q` reports one failure, `tests/unit/test_platform_prices_ingest.py::test_to_monthly_spine_yields_month_end_frequency`. This file and its corresponding source (`src/trading_crab_lib/platform/ingestion/prices_daily.py`) were committed by a sibling worktree executing plan 01-06 (commits `9203629`/`60de3b7`, both predating this plan's commits) and are entirely outside this plan's `files_modified` list. Per the scope boundary rule, this was neither investigated further nor fixed — it is the responsibility of plan 01-06's execution, not this plan.
- **`cssselect` and `fredapi` availability:** `cssselect` (multpl's optional dependency) is not installed in this sandbox, so `test_scrape_multpl_monthly_basic` exercises the graceful-degradation path (empty result → `pytest.skip`) rather than the full happy path. `fredapi` IS available, so all FRED-path tests run for real (mocked network, real library code).

## User Setup Required

None — no external service configuration required. No live FRED/multpl/macrotrends calls are made by the automated test suite; `FRED_API_KEY` reuse (D-07) is exercised only if a caller runs `fetch_macro_monthly()` against `load_platform_config()`'s real config outside of tests, which this plan does not do.

## Next Phase Readiness

- `fetch_macro_monthly(cfg)` is ready for Plan 01-04 (monthly transforms orchestration) to consume as the raw monthly macro input.
- No blockers introduced by this plan. The one out-of-scope failure noted above belongs to plan 01-06 and should be tracked/fixed there, not here.

---
*Phase: 01-monthly-data-layer-long-histories*
*Completed: 2026-07-15*

## Self-Check: PASSED

All 3 claimed files verified present on disk (`src/trading_crab_lib/platform/ingestion/macro_monthly.py`, `tests/unit/test_platform_macro_ingest.py`, this SUMMARY.md); all 3 claimed commits (`818efe6`, `b1ae7bd`, `7d8f0da`) verified present in `git log --oneline --all`.
