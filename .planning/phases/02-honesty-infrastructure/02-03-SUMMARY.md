---
phase: 02-honesty-infrastructure
plan: 03
subsystem: infra
tags: [honesty-framework, cross-validation, scikit-learn, purged-embargoed-cv]

# Dependency graph
requires:
  - phase: 02-honesty-infrastructure
    provides: "src/trading_crab_lib/platform/honesty/ package (empty barrel), config/platform_settings.yaml cv: section"
provides:
  - "src/trading_crab_lib/platform/honesty/cv.py: PurgedEmbargoedKFold(BaseCrossValidator)"
  - "tests/unit/test_platform_cv.py: property-sweep leakage proof + BaseCrossValidator contract test"
affects: [phase-3-modeling, walk-forward-runner (02-04/02-05)]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Hand-rolled sklearn BaseCrossValidator subclass — no new dependency (mlfinlab relicensed closed-source, timeseriescv unmaintained/SUS per RESEARCH Package Legitimacy Audit)"
    - "Required keyword-only constructor args with no default (label_horizon, embargo) to force every caller to reason about its own target's horizon"

key-files:
  created:
    - src/trading_crab_lib/platform/honesty/cv.py
    - tests/unit/test_platform_cv.py

key-decisions:
  - "Followed RESEARCH.md Pattern 3 exactly, with the one specified change: label_horizon and embargo are required keyword-only args (no = 1 / = 0 defaults) so omitting either raises TypeError"

patterns-established:
  - "cv.py is the canonical purged+embargoed CV splitter for any Phase 3+ supervised component with overlapping labels — drop-in cv= replacement for TimeSeriesSplit"

requirements-completed: [HON-04]

coverage:
  - id: D1
    description: "PurgedEmbargoedKFold implements the BaseCrossValidator contract (get_n_splits, split yielding numpy integer train/test index arrays)"
    requirement: "HON-04"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_cv.py::TestBaseCrossValidatorContract::test_isinstance_basecrossvalidator"
        status: pass
      - kind: unit
        ref: "tests/unit/test_platform_cv.py::TestBaseCrossValidatorContract::test_split_yields_numpy_integer_arrays"
        status: pass
    human_judgment: false
  - id: D2
    description: "No training index ever falls within the purge window before or the embargo window after any test fold, across a parametrized (n, n_splits, label_horizon, embargo) sweep"
    requirement: "HON-04"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_cv.py::TestNoLeakageAcrossPurgeEmbargoWindow::test_no_train_index_in_purge_or_embargo_window[3 param sets]"
        status: pass
    human_judgment: false
  - id: D3
    description: "label_horizon and embargo are required explicit constructor arguments with no silent default"
    requirement: "HON-04"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_cv.py::TestRequiredArgsNoSilentDefault (3 tests: missing both, missing embargo, missing label_horizon)"
        status: pass
    human_judgment: false
  - id: D4
    description: "At a realistic monthly horizon (n=120, n_splits=5, label_horizon=12, embargo=1) the purge removes a non-trivial (>0) number of training rows on at least one interior fold"
    requirement: "HON-04"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_cv.py::TestNontrivialPurgeAtMonthlyHorizon::test_nontrivial_purge_at_monthly_horizon"
        status: pass
    human_judgment: false

# Metrics
duration: 3min
completed: 2026-07-17
status: complete
---

# Phase 2 Plan 03: Purged + Embargoed CV Splitter Summary

**Hand-rolled `PurgedEmbargoedKFold(BaseCrossValidator)` for overlapping-label time series — purges training rows within a required `label_horizon` before each test fold and embargoes a required `embargo` window after it, with a property-sweep test proving zero leakage across three parametrizations.**

## Performance

- **Duration:** ~3 min (2 commits, 20:57:27Z → 20:57:46Z)
- **Tasks:** 1 completed (TDD: RED → GREEN)
- **Files modified:** 2 (both created)

## Accomplishments
- Implemented `PurgedEmbargoedKFold` in `src/trading_crab_lib/platform/honesty/cv.py` exactly per RESEARCH.md Pattern 3, with `label_horizon` and `embargo` as required keyword-only constructor args (no silent default — omitting either raises `TypeError`)
- 12 tests in `tests/unit/test_platform_cv.py`: a parametrized 3-tuple leakage-window sweep, train/test disjointness, full-index fold partition, `get_n_splits` contract, the two no-default `TypeError` cases, the monthly-horizon non-triviality proof (n=120, label_horizon=12, embargo=1), and the `BaseCrossValidator` isinstance + numpy-integer-array contract checks
- Zero new dependencies added (confirmed via `git diff pyproject.toml requirements.txt` — empty), matching the plan's zero-install constraint (mlfinlab relicensed closed-source, timeseriescv unmaintained/SUS per RESEARCH Package Legitimacy Audit)

## Task Commits

Each task was committed atomically:

1. **Task 1: PurgedEmbargoedKFold BaseCrossValidator subclass** - `26ea66c` (test, RED) → `341006d` (feat, GREEN)

**Plan metadata:** SUMMARY commit (this file) follows.

_Note: Task 1 was TDD — RED commit (`26ea66c`) proves the test suite genuinely fails without the implementation (`ModuleNotFoundError: No module named 'trading_crab_lib.platform.honesty.cv'`, verified by collection, not a skip); GREEN commit (`341006d`) restores `cv.py` and all 12 tests pass._

## Files Created/Modified
- `src/trading_crab_lib/platform/honesty/cv.py` - `PurgedEmbargoedKFold(BaseCrossValidator)`: `__init__(n_splits=5, *, label_horizon, embargo)`, `get_n_splits(...)`, `split(X, y=None, groups=None)` yielding `(train_idx, test_idx)` numpy integer arrays
- `tests/unit/test_platform_cv.py` - 12 tests across 7 classes: leakage-window sweep (parametrized over 3 `(n, n_splits, label_horizon, embargo)` tuples), train/test disjoint, fold partition, `get_n_splits` match, required-args no-default (3 cases), monthly-horizon non-triviality, `BaseCrossValidator` contract (2 cases)

## Decisions Made
- Followed RESEARCH.md Pattern 3's `PurgedEmbargoedKFold` body verbatim (purge_start = `max(0, test_start - label_horizon)`, embargo_end = `min(n, test_end + 1 + embargo)`, `np.array_split` for contiguous test folds) — the only deviation from the research snippet is the one the plan explicitly specified: `label_horizon` and `embargo` dropped their `= 1` / `= 0` defaults to become required keyword-only args (Open Question 2 in RESEARCH.md, resolved in favor of "no silent default"). No conflict arose between the plan's spec and sklearn's `BaseCrossValidator` contract — nothing to document as a deviation on that front.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- `PurgedEmbargoedKFold` is ready as a drop-in `cv=` argument for any Phase 3+ supervised trainer with overlapping labels (forward-return / forward-transition targets).
- No blockers. `git diff pyproject.toml requirements.txt` confirmed empty — no new dependency surface introduced.
- Sibling plans in this wave (02-02: registry.py, 02-04: gating.py + gap_lag.py) are unaffected — this plan touched only `cv.py` and its test, staying strictly within `files_modified`.

---
*Phase: 02-honesty-infrastructure*
*Completed: 2026-07-17*

## Self-Check: PASSED

All created files verified present on disk; both task commit hashes (`26ea66c`, `341006d`)
verified present in `git log --oneline --all`.
