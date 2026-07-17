---
phase: 02-honesty-infrastructure
plan: 01
subsystem: infra
tags: [honesty-framework, checkpoints, holdout, pandas, platform-config]

# Dependency graph
requires:
  - phase: 01-monthly-data-layer-long-histories
    provides: platform/checkpoints.py (get_platform_checkpoint_manager), platform/config.py (load_platform_config), monthly_features checkpoint namespace
provides:
  - "config/platform_settings.yaml holdout:/registry:/cv: sections"
  - "src/trading_crab_lib/platform/honesty/ package (empty barrel)"
  - "src/trading_crab_lib/platform/honesty/holdout.py: HOLDOUT_CHECKPOINT_DIR, DEFAULT_HOLDOUT_CUTOFF, get_holdout_checkpoint_manager(), split_by_holdout_boundary(), write_monthly_features_split(), assert_dev_checkpoint_within_boundary()"
  - "invariant test proving the default platform checkpoint manager cannot load post-2020 rows"
affects: [02-02, 02-03, 02-04, 02-05, phase-3-modeling]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Holdout carve as a second CheckpointManager instance pointed at data/holdout/, never a subclass"
    - "No try/except fallback from default (dev) checkpoint path to holdout tree — FileNotFoundError IS the guard"
    - "Config sections read defensively via .get() and NOT added to _REQUIRED_PLATFORM_SECTIONS"

key-files:
  created:
    - src/trading_crab_lib/platform/honesty/__init__.py
    - src/trading_crab_lib/platform/honesty/holdout.py
    - tests/unit/test_platform_holdout.py
  modified:
    - config/platform_settings.yaml

key-decisions:
  - "holdout/registry/cv config sections kept optional (.get()-read only), not added to _REQUIRED_PLATFORM_SECTIONS, to avoid breaking Phase 1 minimal-cfg test fixtures"
  - "assert_dev_checkpoint_within_boundary returns None (no-op) on an empty dev checkpoint rather than raising, matching the plan's stated behavior"

patterns-established:
  - "honesty/ package follows platform/__init__.py convention: no re-exports, downstream modules import directly from the owning submodule"

requirements-completed: [HON-01]

coverage:
  - id: D1
    description: "config/platform_settings.yaml gains holdout:/registry:/cv: sections, readable via load_platform_config() without breaking existing required-section validation"
    requirement: "HON-01"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_taxonomy.py, tests/unit/test_platform_transforms.py (regression, both green)"
        status: pass
      - kind: other
        ref: "python3 -c \"load_platform_config() exposes holdout/registry/cv keys with expected values\""
        status: pass
    human_judgment: false
  - id: D2
    description: "platform/honesty/ package created with empty __init__.py marker (no re-exports)"
    requirement: "HON-01"
    verification:
      - kind: unit
        ref: "python3 -c \"import trading_crab_lib.platform.honesty\" succeeds"
        status: pass
    human_judgment: false
  - id: D3
    description: "holdout.py implements the physical 2021+ holdout carve: split_by_holdout_boundary, write_monthly_features_split, get_holdout_checkpoint_manager, assert_dev_checkpoint_within_boundary — reusing get_platform_checkpoint_manager verbatim, no fallback path to the holdout tree"
    requirement: "HON-01"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_holdout.py::TestHoldoutBoundary::test_default_manager_cannot_load_post_2020_rows"
        status: pass
      - kind: unit
        ref: "tests/unit/test_platform_holdout.py (10 tests total, all classes)"
        status: pass
    human_judgment: false

# Metrics
duration: 5min
completed: 2026-07-17
status: complete
---

# Phase 2 Plan 01: Holdout Carve + Honesty Package Foundation Summary

**Physical 2021+ holdout carve (HON-01) via a second CheckpointManager instance pointed at `data/holdout/`, with an invariant test proving the default platform checkpoint manager structurally cannot return post-2020 rows.**

## Performance

- **Duration:** ~5 min (3 commits, 20:50:51Z → 20:52:53Z)
- **Tasks:** 2 completed
- **Files modified:** 4 (1 modified, 3 created)

## Accomplishments
- Added `holdout:`, `registry:`, `cv:` sections to `config/platform_settings.yaml` — the one shared config file this phase owns, read defensively via `.get()` by all downstream Plans 02-05 without ever re-touching the file
- Created `src/trading_crab_lib/platform/honesty/` package (empty barrel, no re-exports) that every later honesty module imports through
- Implemented `holdout.py`: `get_holdout_checkpoint_manager()` (opt-in only), `split_by_holdout_boundary()`, `write_monthly_features_split()`, `assert_dev_checkpoint_within_boundary()` — all reusing `get_platform_checkpoint_manager()` verbatim per D-01, with zero fallback code path from the default (dev) checkpoint tree to `data/holdout/`
- Invariant test `test_default_manager_cannot_load_post_2020_rows` proves by real load (not monkeypatching) that a holdout-only checkpoint raises `FileNotFoundError` through the default manager, and that a split-written `monthly_features` checkpoint's dev side never exceeds the 2020-12-31 cutoff

## Task Commits

Each task was committed atomically:

1. **Task 1: Config sections + honesty package marker** - `ae0f0e9` (feat)
2. **Task 2: Holdout carve module + invariant test** - `395e1af` (test, RED) → `50dd742` (feat, GREEN)

**Plan metadata:** SUMMARY commit (this file) follows.

_Note: Task 2 was TDD — RED commit (`395e1af`) proves the test genuinely fails without the implementation (ModuleNotFoundError, not collection skip); GREEN commit (`50dd742`) adds `holdout.py` and fixes a test-authoring bug surfaced during GREEN._

## Files Created/Modified
- `config/platform_settings.yaml` - Appended `holdout:` (cutoff, holdout_dir), `registry:` (path), `cv:` (label_horizon/embargo month conventions) sections with en-dash header comments matching file style
- `src/trading_crab_lib/platform/honesty/__init__.py` - Package marker, docstring only, no re-exports (matches `platform/__init__.py` convention)
- `src/trading_crab_lib/platform/honesty/holdout.py` - Holdout carve: constants + 4 functions (see key-files above)
- `tests/unit/test_platform_holdout.py` - 10 tests across 6 classes (split boundary, manager distinctness, split-write round-trip, boundary assertion, headline invariant)

## Decisions Made
- Config sections were kept out of `_REQUIRED_PLATFORM_SECTIONS` per explicit plan instruction — `platform/config.py` is byte-for-byte unchanged (verified via `git diff --stat`), so Phase 1 fixtures building minimal cfg dicts (`tests/unit/test_platform_taxonomy.py::_well_formed_cfg`) are unaffected.
- `assert_dev_checkpoint_within_boundary` treats an empty dev checkpoint as trivially within boundary (returns `None`, no raise) rather than raising on an empty frame — matches the plan's stated behavior ("returns None otherwise").

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed `DatetimeIndex.intersection(...).any()` TypeError in test**
- **Found during:** Task 2 GREEN phase (`test_dev_and_holdout_are_disjoint_and_sum_to_input_length`)
- **Issue:** `pandas.core.nanops` raises `TypeError: datetime64 type does not support operation 'any'` when calling `.any()` on a `DatetimeIndex` (an authoring bug in the test itself, not in `holdout.py`) — pandas does not support boolean reduction on datetime64 arrays.
- **Fix:** Changed `assert not dev_df.index.intersection(holdout_df.index).any()` to `assert len(dev_df.index.intersection(holdout_df.index)) == 0` — semantically identical, works on any dtype.
- **Files modified:** `tests/unit/test_platform_holdout.py`
- **Verification:** `pytest tests/unit/test_platform_holdout.py -x -q` → 10 passed
- **Committed in:** `50dd742` (Task 2 GREEN commit)

---

**Total deviations:** 1 auto-fixed (1 bug fix, test-authoring only — no production code affected)
**Impact on plan:** Zero scope creep. Fix was internal to the new test file and did not touch `holdout.py`, config, or any file outside this plan's scope.

## Issues Encountered
None beyond the deviation above.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- `holdout.py`'s two checkpoint-manager factories and boundary guard are ready for Plans 02-05 (trial registry, purged/embargoed CV, walk-forward runner, causal gating, gap/lag metrics) to build on.
- The `registry:` and `cv:` config sections this plan added are unused by this plan's own code — they exist solely so Plan 02 (registry) and Plan 03 (CV) never need to re-touch `config/platform_settings.yaml`.
- No blockers. `config/platform_settings.yaml` diff is additive-only; `platform/config.py` is untouched.

---
*Phase: 02-honesty-infrastructure*
*Completed: 2026-07-17*

## Self-Check: PASSED

All created files verified present on disk; all task commit hashes (`ae0f0e9`,
`395e1af`, `50dd742`) verified present in `git log --oneline --all`.
