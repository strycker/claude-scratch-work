---
phase: 03-regime-labeling-prediction
plan: 04
subsystem: prediction
tags: [pandas, transition-matrix, regime-diagnostics, L2]

# Dependency graph
requires:
  - phase: 03-03
    provides: "platform/prediction/ package (__init__.py) that this plan extends"
provides:
  - "empirical_transition_matrix(states) — pure row-normalized K x K forward-transition diagnostic"
affects: [04-allocation-tactics, weekly-report-forward-regime-distribution]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "compute_* pure-function module (gap_lag.py analog) — no I/O, no model state, full docstring + type hints, __main__ synthetic self-check footer"

key-files:
  created:
    - src/trading_crab_lib/platform/prediction/transition_matrix.py
    - tests/unit/test_platform_transition_matrix.py
  modified: []

key-decisions:
  - "Copied RESEARCH.md Pattern 8 verbatim (pd.crosstab of from/to pairs, row-normalized via counts.div(counts.sum(axis=1), axis=0)) — no deviation from the vetted implementation"
  - "Confirmed this is a v1 diagnostic over the FULL non-embargoed smoothed label sequence, not a training target; TVTP (feature-conditional transitions) explicitly deferred to v2 (L2-V2-02) per module docstring and threat register T-03-13"

patterns-established:
  - "Pattern: pure compute_*-style diagnostic functions in platform/ live standalone with a __main__ synthetic self-check mirroring gap_lag.py, proven with zero network/checkpoint dependency ahead of upstream real-data wiring"

requirements-completed: [L2-02]

coverage:
  - id: D1
    description: "empirical_transition_matrix(states) returns a row-normalized K×K DataFrame where P(next=j | current=i); occupied rows sum to 1.0"
    requirement: "L2-02"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_transition_matrix.py#TestEmpiricalTransitionMatrix::test_occupied_rows_sum_to_one"
        status: pass
      - kind: unit
        ref: "tests/unit/test_platform_transition_matrix.py#TestEmpiricalTransitionMatrix::test_hand_computed_known_counts"
        status: pass
    human_judgment: false
  - id: D2
    description: "A state that is never a 'from' state produces an absent/zero-sum row, not a ZeroDivision/NaN crash"
    requirement: "L2-02"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_transition_matrix.py#TestEmpiricalTransitionMatrix::test_never_a_from_state_does_not_crash"
        status: pass
    human_judgment: false
  - id: D3
    description: "The matrix is a pure function of the label sequence — no model state, no I/O, no incumbent imports"
    requirement: "L2-02"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_transition_matrix.py#TestEmpiricalTransitionMatrix::test_pure_function_no_io"
        status: pass
    human_judgment: false

# Metrics
duration: 12min
completed: 2026-07-22
status: complete
---

# Phase 3 Plan 04: Empirical Transition Matrix Diagnostic Summary

**L2-02 empirical transition matrix — pure row-normalized K×K forward-transition table via `pd.crosstab`, degrading gracefully on never-observed 'from' states**

## Performance

- **Duration:** 12 min
- **Tasks:** 1 (TDD: RED → GREEN)
- **Files modified:** 2 (both new)

## Accomplishments
- `empirical_transition_matrix(states: pd.Series) -> pd.DataFrame` implemented exactly per RESEARCH.md Pattern 8 — a single pure function with no class wrapper, full docstring, type hints on param and return
- Degrades gracefully: a state that never appears as a 'from' state produces no row for it (no ZeroDivision/NaN crash), asserted by a dedicated test
- 4 tests covering row-sum-to-1 (random sequence), hand-computed known-count case, never-a-from-state no-crash case, and pure-function/no-I/O check — all green
- Module docstring documents this is v1 machinery; the feature-conditional TVTP model is explicitly deferred to v2 (L2-V2-02), matching threat register T-03-13's disposition (accept — documented so no consumer treats it as a leak-safe supervised feature)

## Task Commits

Each task was committed atomically (TDD):

1. **Task 1 RED: failing tests for empirical transition matrix** - `22f314d` (test)
2. **Task 1 GREEN: implement empirical transition matrix (L2-02)** - `fb771bc` (feat)

No REFACTOR commit needed — the implementation is a direct, already-clean copy of the vetted RESEARCH.md pattern (3 lines of logic plus docstring).

## Files Created/Modified
- `src/trading_crab_lib/platform/prediction/transition_matrix.py` - `empirical_transition_matrix()` pure function; `__main__` synthetic self-check footer mirroring `gap_lag.py`
- `tests/unit/test_platform_transition_matrix.py` - `TestEmpiricalTransitionMatrix` with 4 tests

## Decisions Made
- Copied RESEARCH.md Pattern 8 verbatim — no deviation from the vetted `pd.crosstab` + `div(sum(axis=1), axis=0)` implementation.
- Confirmed via module docstring and threat register cross-check that this diagnostic consumes the labeler's FULL smoothed state sequence and is NOT embargoed (it is descriptive, not a training input) — matches plan's `key_links` and `must_haves.truths`.

## Deviations from Plan

None — plan executed exactly as written. Task's `<action>` specified copying Pattern 8 verbatim with a module docstring and `__main__` self-check footer; that is exactly what was built.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- `empirical_transition_matrix()` is ready for Phase 4's allocation/weekly-report layer to consume the forward regime distribution implied by history, once wired against real labeler output.
- `ruff check` passes clean on both new files; full platform suite (`tests/unit/test_platform_*.py`) is green: 174 passed, 1 skipped (pre-existing, unrelated).
- No blockers.

---
*Phase: 03-regime-labeling-prediction*
*Completed: 2026-07-22*
