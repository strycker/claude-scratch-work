---
phase: 03-regime-labeling-prediction
plan: 02
subsystem: ml-labeling
tags: [jump-model, checkpointing, monitoring, parquet, sklearn]

# Dependency graph
requires:
  - phase: 03-regime-labeling-prediction (03-01)
    provides: fit_jump_model, canonicalize_states, soft_confidences, standardize_features (platform/labeling/jump_model.py)
provides:
  - "diagnostics.py: occupancy_and_sojourns, label_churn, auto_profile, report_labeling_diagnostics, label_regimes"
  - "regime_labels / regime_confidences / regime_profiles platform checkpoints"
  - "labeling_diagnostics.parquet §4.4 report-only artifact"
affects: [phase-4-allocation-report, phase-3-nowcaster-plan-03-03]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "load-before-save churn: read previous regime_labels checkpoint BEFORE saving the new one, catch FileNotFoundError explicitly for the first-run NaN case"
    - "report-only diagnostics: §4.4 violations log at WARNING, never raise (D-02)"
    - "diagnostics artifact routed through the same checkpoint_dir override as persistence, keeping tests fully isolated from the real OUTPUT_DIR"

key-files:
  created:
    - src/trading_crab_lib/platform/labeling/diagnostics.py
  modified:
    - tests/unit/test_platform_labeling.py

key-decisions:
  - "auto_profile, report_labeling_diagnostics, and label_regimes were all implemented together in Task 1's GREEN commit, not split across Task 1/Task 2 as the plan's task boundaries suggested — the TestChurnTwoRun invariant (Task 1) requires a fully working label_regimes(), which in turn requires auto_profile() and report_labeling_diagnostics() to already exist. Task 2's tests therefore document coverage of already-complete functionality rather than driving new implementation."
  - "report_labeling_diagnostics' output_dir is threaded through label_regimes' checkpoint_dir parameter so test runs never write labeling_diagnostics.parquet into the real outputs/ directory."

patterns-established:
  - "diagnostics.py mirrors gap_lag.py exactly: pure compute_* functions + one report_* function (schema-stable artifact + print + WARNING-on-violation) + __main__ synthetic self-check."

requirements-completed: [L1-02, L1-03]

coverage:
  - id: D1
    description: "occupancy_and_sojourns() reports per-state occupancy fraction (sums to 1.0) and run-length sojourn stats; never-occupied states report 0% without raising"
    requirement: "L1-02"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_labeling.py::TestOccupancySojourn"
        status: pass
    human_judgment: false
  - id: D2
    description: "label_churn() computes trailing-window fraction of differing canonicalized labels; identical->0.0, disjoint->1.0, empty->nan"
    requirement: "L1-03"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_labeling.py::TestLabelChurn"
        status: pass
    human_judgment: false
  - id: D3
    description: "label_regimes() proves load-before-save churn ordering: NaN on first run (no prior checkpoint), 0.0 on an identical repeat run"
    requirement: "L1-03"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_labeling.py::TestChurnTwoRun"
        status: pass
    human_judgment: false
  - id: D4
    description: "auto_profile() generates a deterministic, one-line economic profile per state naming real feature identifiers (D-04)"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_labeling.py::TestAutoProfile"
        status: pass
    human_judgment: false
  - id: D5
    description: "report_labeling_diagnostics() logs a WARNING on a §4.4 occupancy violation and still returns the artifact path without raising (D-02 report-only); schema-stable even when metrics is empty"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_labeling.py::TestReportDiagnosticsReportOnly"
        status: pass
    human_judgment: false
  - id: D6
    description: "regime_labels, regime_confidences, and regime_profiles persist through the platform checkpoint namespace and reload; confidence rows sum to ~1.0"
    requirement: "L1-02"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_labeling.py::TestLabelRegimesPersistence"
        status: pass
    human_judgment: false

duration: 25min
completed: 2026-07-22
status: complete
---

# Phase 3 Plan 02: Labeling Persistence, Churn, and §4.4 Diagnostics Summary

**Label churn monitoring (load-before-save, NaN-on-first-run), §4.4 occupancy/sojourn report-only diagnostics, and D-04 auto-generated economic profiles complete the L1 labeler around Plan 03-01's jump-model core.**

## Performance

- **Duration:** ~25 min
- **Started:** 2026-07-22T19:00:00Z
- **Completed:** 2026-07-22T19:25:00Z
- **Tasks:** 2
- **Files modified:** 2 (1 created, 1 modified)

## Accomplishments
- `occupancy_and_sojourns()` and `label_churn()`: pure, independently-testable compute functions mirroring `gap_lag.py`'s structure — churn is NaN on an empty/first-run input and computed over the trailing window of already-canonicalized labels otherwise.
- `label_regimes()` orchestration: pulls the lean feature set, standardizes, fits the jump model (Plan 03-01), canonicalizes, computes soft confidences, computes churn against the previous `regime_labels` checkpoint **before** overwriting it, then persists `regime_labels`/`regime_confidences`/`regime_profiles` and reports §4.4 diagnostics.
- `auto_profile()` (D-04): one-line economic profile per numeric state id from the sign/magnitude of its top-3 standardized centroid coordinates.
- `report_labeling_diagnostics()` (D-02/D-05): schema-stable `labeling_diagnostics.parquet` artifact; a §4.4 occupancy violation logs a loud WARNING but the function always returns normally — the labeler never gates on it.
- Two-run invariant (`TestChurnTwoRun`) proves the load-before-save ordering end-to-end: run 1 against an empty `tmp_path` checkpoint dir returns `nan`, an identical run 2 returns `0.0`.
- `__main__` synthetic self-check (no network, no Phase-1 data dependency) exercises the full pipeline and prints a live §4.4 WARNING example.

## Task Commits

Each task was committed atomically (TDD RED → GREEN):

1. **Task 1: Label churn (load-before-save, first-run NaN) + occupancy/sojourn stats**
   - `66e1009` (test) — RED: failing tests for `occupancy_and_sojourns`, `label_churn`, `TestChurnTwoRun`
   - `f42bd62` (feat) — GREEN: implemented `occupancy_and_sojourns`, `label_churn`, `auto_profile`, `report_labeling_diagnostics`, and `label_regimes` (all five functions were needed together — see Deviations)
2. **Task 2: Auto profiles (D-04) + report-only §4.4 diagnostics + label_regimes persistence**
   - `de12616` (test) — coverage tests for `auto_profile`, `report_labeling_diagnostics` report-only behavior, and `label_regimes` persistence; all pass against the Task 1 implementation with no further code changes

**Plan metadata:** this SUMMARY.md commit (below)

## Files Created/Modified
- `src/trading_crab_lib/platform/labeling/diagnostics.py` — new module: `occupancy_and_sojourns`, `label_churn`, `auto_profile`, `report_labeling_diagnostics`, `label_regimes`, `__main__` self-check
- `tests/unit/test_platform_labeling.py` — extended with `TestOccupancySojourn`, `TestLabelChurn`, `TestChurnTwoRun`, `TestAutoProfile`, `TestReportDiagnosticsReportOnly`, `TestLabelRegimesPersistence`

## Decisions Made
- Implemented all 5 plan-listed functions in Task 1's GREEN commit rather than splitting churn/occupancy (Task 1) from auto_profile/report/persistence (Task 2), because the `TestChurnTwoRun` two-run invariant (a Task 1 acceptance criterion) calls `label_regimes()`, which cannot complete without `auto_profile()` and `report_labeling_diagnostics()` already existing. Task 2's tests (`TestAutoProfile`, `TestReportDiagnosticsReportOnly`, `TestLabelRegimesPersistence`) therefore exercise already-complete functionality and pass immediately — documented per the TDD fail-fast rule rather than forcing an artificial RED phase.
- `label_regimes()` routes the diagnostics artifact's `output_dir` through its own `checkpoint_dir` parameter. Without this, the first test run wrote a real `labeling_diagnostics.parquet` into the repo's `outputs/reports/model_metrics/` directory (caught via `git status` before committing, fixed inline as a Rule 1 bug fix, verified no stray files remained).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] `report_labeling_diagnostics` wrote into the real `outputs/` directory during tests**
- **Found during:** Task 1, after first GREEN run of `TestChurnTwoRun`
- **Issue:** `label_regimes()` called `report_labeling_diagnostics(metrics)` without an `output_dir` override, so even when `checkpoint_dir=tmp_path` was passed for checkpoint isolation, the diagnostics parquet artifact was written to the project's real `outputs/reports/model_metrics/` directory.
- **Fix:** `label_regimes()` now passes `output_dir=checkpoint_dir` to `report_labeling_diagnostics()`, so the diagnostics artifact is isolated alongside the checkpoints whenever a non-default `checkpoint_dir` is supplied (tests, `__main__` self-check).
- **Files modified:** `src/trading_crab_lib/platform/labeling/diagnostics.py`
- **Verification:** Re-ran the full `tests/unit/test_platform_labeling.py` suite and confirmed `git status --short` showed no untracked files under `outputs/` after the run.
- **Committed in:** `f42bd62` (Task 1 GREEN commit, fixed before commit — not a separate follow-up)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Necessary correctness fix for test isolation; no scope creep. The task-boundary deviation (functions implemented together rather than split across Task 1/Task 2) is a planning-artifact note, not a code change — documented under Decisions Made above.

## Issues Encountered
None beyond the auto-fixed test-isolation bug above.

## User Setup Required
None — no external service configuration required.

## Next Phase Readiness
- `regime_labels`, `regime_confidences`, and `regime_profiles` checkpoints, plus the `labeling_diagnostics.parquet` artifact, are ready for the nowcaster (Plan 03-03) to consume as its training target and for Phase 4's weekly report to display.
- Manual/deferred: once `FRED_API_KEY` is set and the real 1962+ `monthly_features` checkpoint exists, run `label_regimes()` on it and inspect the real §4.4 report-only diagnostics for sane occupancy/sojourns (per plan's `<verification>` section) — non-blocking, tracked as a STATE.md blocker per phase convention.

---
*Phase: 03-regime-labeling-prediction*
*Completed: 2026-07-22*

## Self-Check: PASSED

- FOUND: `src/trading_crab_lib/platform/labeling/diagnostics.py`
- FOUND: `.planning/phases/03-regime-labeling-prediction/03-02-SUMMARY.md`
- FOUND: commit `66e1009` (test RED — churn/occupancy tests)
- FOUND: commit `f42bd62` (feat GREEN — diagnostics.py implementation)
- FOUND: commit `de12616` (test — Task 2 coverage)
- FOUND: `tests/unit/test_platform_labeling.py -q` → 34 passed
- FOUND: `tests/unit/test_platform_*.py -q` → 184 passed, 1 skipped
