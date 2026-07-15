---
phase: 01-monthly-data-layer-long-histories
plan: 03
subsystem: data
tags: [alfred, fredapi, point-in-time, vintage-correction, look-ahead-bias, ingestion]

# Dependency graph
requires:
  - phase: 01-monthly-data-layer-long-histories
    provides: platform package foundation + config loader (Plan 01) — cfg['fred_vintage']['api_key'] injection
provides:
  - "fetch_vintage_series() / fetch_all_vintages(): bulk ALFRED all-releases fetch (one call per series)"
  - "value_as_of(): point-in-time reconstruction from a revision history, no revision look-ahead"
  - "align_with_fallback(): D-06 pre-vintage-era fallback to publication-lag-shifted values"
  - "_detect_vintage_columns(): defensive case-insensitive column detection guarding A1 schema assumption"
  - "docs/vintage_alignment.md: documented D-06 scope, shift-vs-vintage distinction, pre-vintage fallback policy"
affects: ["07-transforms (Plan 07 calls value_as_of/align_with_fallback to align agency series)"]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Bulk-fetch-once ingestion (mirrors fred.py's ThreadPoolExecutor + try/except-WARNING-degrade shape) with a lower _MAX_WORKERS=3 for the larger all-releases payloads"
    - "Defensive column detection (case-insensitive) before consuming an external API's assumed schema, raising a named ValueError rather than misreconstructing silently"

key-files:
  created:
    - src/trading_crab_lib/platform/ingestion/alfred.py
    - tests/unit/test_platform_alfred.py
    - docs/vintage_alignment.md
  modified: []

key-decisions:
  - "value_as_of/align_with_fallback keep rows with realtime_start <= as_of_date and take the latest realtime_start per reference date — the vintage 'active' at that as_of_date, never a later revision"
  - "align_with_fallback derives the pre-vintage cutover at runtime from all_releases[realtime_start].min() rather than hardcoding per-series ALFRED coverage-start dates (RESEARCH A2 uncertainty), so it self-corrects against whatever the live API actually returns"
  - "Vintage correction is treated as subsuming the publication-lag shift once vintages exist (RESEARCH Pitfall 4) — the shift fallback path is used only strictly before the earliest recorded vintage"

patterns-established:
  - "ALFRED bulk-fetch: never loop a single-date vintage lookup per historical date — one get_series_all_releases() call per series covers the full revision history"

requirements-completed: [DATA-03]

coverage:
  - id: D1
    description: "fetch_vintage_series() makes exactly one bulk get_series_all_releases call per series (no per-date loop anti-pattern)"
    requirement: "DATA-03"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_alfred.py::test_fetch_vintage_series_calls_bulk_endpoint_once"
        status: pass
    human_judgment: false
  - id: D2
    description: "value_as_of() reconstructs the point-in-time value knowable at as_of_date, never a later revision"
    requirement: "DATA-03"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_alfred.py::test_value_as_of_respects_vintage"
        status: pass
      - kind: unit
        ref: "tests/unit/test_platform_alfred.py::test_value_as_of_ignores_future_revisions"
        status: pass
    human_judgment: false
  - id: D3
    description: "align_with_fallback() returns the publication-lag-shifted value (not NaN, not an error) for as_of dates before the earliest recorded vintage"
    requirement: "DATA-03"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_alfred.py::test_pre_vintage_fallback"
        status: pass
    human_judgment: false
  - id: D4
    description: "_detect_vintage_columns() locates realtime_start/realtime_end/date/value case-insensitively and raises a clear ValueError naming the missing role when a column is absent"
    requirement: "DATA-03"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_alfred.py::test_detect_vintage_columns_case_insensitive"
        status: pass
      - kind: unit
        ref: "tests/unit/test_platform_alfred.py::test_detect_vintage_columns_raises_on_missing_role"
        status: pass
    human_judgment: false
  - id: D5
    description: "docs/vintage_alignment.md documents the D-06 vintage scope (5 named series), the pre-vintage-era publication-lag fallback as an explicit accepted compromise, and the vintage-vs-shift (revision vs timing look-ahead) distinction"
    requirement: "DATA-03"
    verification:
      - kind: other
        ref: "python -c \"t=open('docs/vintage_alignment.md').read().lower(); [t.index(k) for k in ['vintage','pre-vintage','fallback','publication','gdp','cpi','unrate']]\""
        status: pass
    human_judgment: false

# Metrics
duration: ~55min (across two executor sessions, interrupted by a provider quota kill)
completed: 2026-07-15
status: complete
---

# Phase 1 Plan 03: ALFRED Point-in-Time Vintage Ingestion Summary

**Bulk ALFRED all-releases fetch with point-in-time reconstruction and a documented pre-vintage-era publication-lag fallback for the five revision-heavy agency series (GDP, CPI, UNRATE, INDPRO, PAYEMS).**

## Performance

- **Duration:** ~55 min across two executor sessions (first session was terminated mid-task by a provider quota limit; this continuation session verified and completed the remaining work)
- **Started:** 2026-07-15T15:24:21Z (first RED commit)
- **Completed:** 2026-07-15T22:13:12Z
- **Tasks:** 2/2
- **Files modified:** 3 (1 created source module, 1 created test file, 1 created doc)

## Accomplishments
- `fetch_vintage_series()` / `fetch_all_vintages()`: single bulk `get_series_all_releases()` call per series — never a per-date loop (verified by `call_count == 1` assertion and a grep-based anti-pattern check)
- `value_as_of()`: point-in-time reconstruction — for each reference period, keeps the vintage with the latest `realtime_start <= as_of_date`, so a later revision can never leak backward in time
- `align_with_fallback()`: implements the D-06 pre-vintage-era compromise — falls back to a caller-supplied publication-lag-shifted value for dates before the earliest recorded vintage, never `NaN`/error
- `_detect_vintage_columns()`: defensive, case-insensitive column detection guarding the RESEARCH A1 schema assumption, raising a `ValueError` naming the missing role
- `docs/vintage_alignment.md`: documents D-06 scope, the shift-vs-vintage-correction distinction (RESEARCH Pitfall 4), and the pre-vintage fallback policy, matching the implemented behavior
- 9 mocked unit tests, all green, no live network calls

## Task Commits

Each task was committed atomically. This plan was executed across two interrupted executor sessions; the timeline below reflects the actual commit history on this branch (all pre-existing commits verified, not redone):

1. **Task 1 (TDD RED): ALFRED vintage fetch + point-in-time reconstruction + fallback (mocked test)** — `b706e98` (test) — first session, wrote the failing test suite for `alfred.py` before the module existed
2. **Task 1 (rescue/GREEN): implementation** — `3ec3f96` (wip, rescued mid-write after provider quota kill) — first session's implementation of `alfred.py` was rescued by the orchestrator's auto-commit safety net before the agent was terminated; verified GREEN (all 9 tests passing) at the start of this continuation session
3. **Task 1 (fix) + Task 2 (docs)** — `0976c31` (docs+fix) — this continuation session: rephrased two docstring mentions of the anti-pattern API name that were literally tripping the plan's acceptance-criteria grep check (`grep get_series_as_of_date` must return nothing), and created `docs/vintage_alignment.md`

**Plan metadata:** (this commit, SUMMARY.md)

_Note: the RED→GREEN split happened across two separate executor processes due to the provider quota interruption, not by design — this was not a planned multi-commit TDD refactor step._

## Files Created/Modified
- `src/trading_crab_lib/platform/ingestion/alfred.py` - Bulk ALFRED vintage fetch (`fetch_vintage_series`, `fetch_all_vintages`), point-in-time reconstruction (`value_as_of`), the D-06 pre-vintage fallback (`align_with_fallback`), and defensive column detection (`_detect_vintage_columns`)
- `tests/unit/test_platform_alfred.py` - 9 mocked unit tests covering all five behaviors from the plan's `<behavior>` block
- `docs/vintage_alignment.md` - D-06 vintage scope, shift-vs-vintage-correction distinction, and pre-vintage-era fallback policy

## Decisions Made
- Kept the rescued implementation's approach of deriving the pre-vintage cutover date at runtime (`all_releases[realtime_start].min()`) rather than hardcoding per-series ALFRED coverage-start dates — matches the plan's guidance that RESEARCH A2 per-series dates are unverified and should not be hardcoded
- `align_with_fallback` treats vintage-correction as subsuming the shift once vintages exist (per RESEARCH Pitfall 4): for as_of dates at/after the earliest vintage it uses `value_as_of`, falling back to `shift_series` only when no reference period is known yet at that as_of date (not merely "before the earliest vintage" globally, but per lookup) — this matches the plan action's stated behavior precisely

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Docstrings named the anti-pattern API literally, breaking the acceptance-criteria grep check**
- **Found during:** Continuation-session verification of Task 1's acceptance criteria
- **Issue:** The rescued `alfred.py` implementation documented the anti-pattern it avoids (per-date vintage lookups) by naming the actual `get_series_as_of_date()` method in two docstrings. This is good documentation intent, but the plan's acceptance criterion `grep -n "get_series_as_of_date" ... returns nothing` is a literal string match, so the docstrings themselves tripped the check even though no code call to that method exists.
- **Fix:** Rephrased both docstring mentions to describe the anti-pattern ("the single-date vintage lookup endpoint" / "the single-date vintage lookup") without using the literal method name. No behavior change.
- **Files modified:** `src/trading_crab_lib/platform/ingestion/alfred.py`
- **Verification:** `grep -n "get_series_as_of_date" src/trading_crab_lib/platform/ingestion/alfred.py` now returns nothing (exit 1); all 9 tests still pass; `ruff check` clean.
- **Committed in:** `0976c31`

---

**Total deviations:** 1 auto-fixed (Rule 1 — bug/acceptance-criteria fix)
**Impact on plan:** Cosmetic docstring fix only; no functional or test-coverage impact. No scope creep.

## Issues Encountered
- **Provider quota interruption:** The original executor was terminated mid-plan by a provider quota limit after completing the RED test commit (`b706e98`) and mid-writing the GREEN implementation. The orchestrator's rescue mechanism auto-committed the in-progress `alfred.py` as `3ec3f96` ("wip: rescue uncommitted executor work"), which turned out to be a complete, working implementation (all 9 tests passed on first run in this continuation session, no fixes needed beyond the docstring grep issue above).
- **`FRED_API_KEY` unavailability in this sandbox:** Per the parallel-execution constraints, no live ALFRED/FRED calls are made anywhere in the test suite — `fetch_all_vintages` is exercised only via `@patch("trading_crab_lib.platform.ingestion.alfred.Fred")`. The missing-key path (`fetch_all_vintages` raising `OSError` when `cfg["fred_vintage"]["api_key"]` is falsy) is itself tested (`test_fetch_all_vintages_missing_api_key_raises`), so the module degrades gracefully with or without a real key present in the environment.
- **Full incumbent suite run:** `pytest tests/ -q` shows 774 passed, 48 skipped, and 1 failure (`tests/unit/test_platform_prices_ingest.py::test_to_monthly_spine_yields_month_end_frequency`). This failure is in a file this plan never touches — it was committed by a sibling agent's Plan 01-06 work (`60de3b7`, prior to this branch's base) — and is out of this plan's `files_modified` scope per the parallel-execution/scope-boundary rules. Not fixed here; left for Plan 01-06's owner or a follow-up.

## User Setup Required
None - no external service configuration required. `FRED_API_KEY` reuse (D-07) requires no new credentials.

## Next Phase Readiness
- Plan 07 (transforms layer) can now call `value_as_of()` / `align_with_fallback()` to align the five D-06 agency series without revision look-ahead.
- Manual follow-up (documented, non-blocking): before the first live pipeline run, spot-check `get_series_all_releases()` column names and `get_series_vintage_dates()` per-series coverage against the live ALFRED API (RESEARCH A1/A2) — the defensive column detection means an unexpected schema fails loudly rather than silently.
- Unrelated pre-existing test failure in `tests/unit/test_platform_prices_ingest.py` (Plan 01-06 scope) is a blocker for a fully-green suite but not for this plan's deliverables.

---
*Phase: 01-monthly-data-layer-long-histories*
*Completed: 2026-07-15*

## Self-Check: PASSED

- FOUND: src/trading_crab_lib/platform/ingestion/alfred.py
- FOUND: tests/unit/test_platform_alfred.py
- FOUND: docs/vintage_alignment.md
- FOUND: commit b706e98 (RED test)
- FOUND: commit 3ec3f96 (rescued GREEN implementation)
- FOUND: commit 0976c31 (docstring fix + docs)
