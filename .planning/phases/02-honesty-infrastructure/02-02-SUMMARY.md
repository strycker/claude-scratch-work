---
phase: 02-honesty-infrastructure
plan: 02
subsystem: infra
tags: [honesty-framework, trial-registry, jsonl, append-only, pandas]

# Dependency graph
requires:
  - phase: 02-honesty-infrastructure
    plan: 01
    provides: platform/honesty/ package, config/platform_settings.yaml registry:.path section
provides:
  - "src/trading_crab_lib/platform/honesty/registry.py: append_trial(), read_trials(), config_hash(), DEFAULT_REGISTRY_PATH"
  - "Property test suite proving append-never-truncates + complete row schema"
affects: [02-05, phase-3-modeling]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Append-only JSONL ledger: open('a') only, never 'w'; git history is the tamper-evidence layer"
    - "config_hash = md5(json.dumps(config, sort_keys=True, default=str)) — deterministic, order-independent"
    - "git SHA capture via subprocess.run(['git','rev-parse','HEAD']), catches CalledProcessError/FileNotFoundError only"

key-files:
  created:
    - src/trading_crab_lib/platform/honesty/registry.py
    - tests/unit/test_platform_registry.py

key-decisions:
  - "DEFAULT_REGISTRY_PATH resolves under trading_crab_lib.ROOT (registry/trials.jsonl), never under DATA_DIR, so the ledger is git-tracked (D-01)"
  - "config_hash uses default=str in json.dumps so non-JSON-native config values (e.g. Path objects) hash without raising"

patterns-established:
  - "Ledger row schema: config_hash, config, features, metrics, git_sha, timestamp — every row is self-contained and reproducible (Pitfall 3)"

requirements-completed: [HON-02]

coverage:
  - id: D1
    description: "append_trial() appends exactly one JSON line per call and never truncates or rewrites existing lines"
    requirement: "HON-02"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_registry.py::TestAppendNeverTruncates (2 tests)"
        status: pass
    human_judgment: false
  - id: D2
    description: "Every appended row carries config_hash, config, features, metrics, git_sha, and a tz-aware ISO-8601 timestamp"
    requirement: "HON-02"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_registry.py::TestRowSchema::test_row_schema"
        status: pass
    human_judgment: false
  - id: D3
    description: "config_hash is deterministic and order-independent (12-char hex); read_trials roundtrips the ledger as a DataFrame, empty-safe on a missing/empty file"
    requirement: "HON-02"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_registry.py::TestConfigHash, TestReadTrials (5 tests)"
        status: pass
    human_judgment: false
  - id: D4
    description: "The default ledger path resolves under ROOT/registry/, not data/, and is proven not git-ignored"
    requirement: "HON-02"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_registry.py::TestDefaultPathNotUnderData, test_registry_path_is_git_trackable"
        status: pass
      - kind: other
        ref: "git check-ignore registry/trials.jsonl → exit 1 (not ignored)"
        status: pass
    human_judgment: false

# Metrics
duration: 8min
completed: 2026-07-17
status: complete
---

# Phase 2 Plan 02: Trial Registry Summary

**Append-only JSONL trial registry (HON-02) at `registry/trials.jsonl` — every evaluated configuration logged as an immutable, git-tracked pre-registration row (config_hash, full config, features, metrics, git_sha, UTC timestamp), proven by property tests that appends never truncate or rewrite existing lines.**

## Performance

- **Duration:** ~8 min (2 commits)
- **Tasks:** 2 completed (Task 2's test was folded into the same test file created for Task 1, per plan's own file list)
- **Files modified:** 2 (both created)

## Accomplishments
- `registry.py`: `append_trial()`, `read_trials()`, `config_hash()`, `_git_sha()`, `_resolve_registry_path()`, `DEFAULT_REGISTRY_PATH` (`ROOT/registry/trials.jsonl`)
- 12 tests in `tests/unit/test_platform_registry.py` covering every behavior bullet and acceptance criterion in the plan, including the append-never-truncates byte-stable-prefix proof and the `git check-ignore` git-trackability proof
- Verified via `grep` that the module source never opens the ledger in `"w"` mode — append-only is structurally enforced, not just convention

## Task Commits

Each task was committed atomically (TDD RED → GREEN):

1. **Task 1: Append-only JSONL trial registry module** - `96e8b6d` (test, RED) → `c51e783` (feat, GREEN)
2. **Task 2: Verify ledger path is git-tracked** — folded into the same test file (`test_registry_path_is_git_trackable`, included in the RED commit `96e8b6d` and passing after GREEN `c51e783`); no separate commit needed since Task 2's only file target (`tests/unit/test_platform_registry.py`) was already created in Task 1

_Note: TDD RED commit (`96e8b6d`) proves genuine failure — `ModuleNotFoundError: No module named 'trading_crab_lib.platform.honesty.registry'`, not a collection skip — confirmed by temporarily moving the not-yet-committed implementation file aside before running pytest. GREEN commit (`c51e783`) adds `registry.py` and fixes a self-inflicted test-authoring bug surfaced during GREEN (see Deviations)._

## Files Created/Modified
- `src/trading_crab_lib/platform/honesty/registry.py` — append-only JSONL ledger: `append_trial()`, `read_trials()`, `config_hash()`, `_git_sha()`, `_resolve_registry_path()`, `DEFAULT_REGISTRY_PATH`
- `tests/unit/test_platform_registry.py` — 12 tests across 8 classes (append-never-truncates, one-line-per-call, row schema, config_hash determinism/length, open-mode guard, read_trials roundtrip, git_sha presence, default-path location, git-trackability)

## Decisions Made
- Used `default=str` in `json.dumps` for both `config_hash()` and the row write, matching RESEARCH.md's exact target implementation — handles non-JSON-native config values (e.g. `Path` objects) without raising.
- `read_trials()` checks `.exists()` and `.stat().st_size == 0` before calling `pd.read_json(lines=True)` — an empty file raises `ValueError: Expected object or value` from pandas otherwise; the plan's behavior bullet requires a graceful empty DataFrame instead.
- Task 2 produced no separate commit: its sole file target (`tests/unit/test_platform_registry.py`) was the same file Task 1 already created, so `test_registry_path_is_git_trackable` was authored alongside Task 1's other tests in the RED commit rather than as a follow-up diff.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Module docstring literal `"w"` false-positived the open-mode guard test**
- **Found during:** Task 1 GREEN phase, `TestOpenModeIsAppend::test_open_mode_is_append`
- **Issue:** The module docstring's prose explanation ("Never `"w"`: rewriting or truncating...") contained the literal substring `"w"`, which the test's naive `assert '"w"' not in source` check flagged as a false positive — the docstring mentions the forbidden mode by name, it doesn't use it.
- **Fix:** Rephrased the docstring to describe the invariant without the literal quoted `"w"` token ("append mode only, never write/truncate mode").
- **Files modified:** `src/trading_crab_lib/platform/honesty/registry.py` (docstring only — no code-path change)
- **Verification:** `pytest tests/unit/test_platform_registry.py -x -q` → 12 passed; `grep -n "'w'\|\"w\"" src/trading_crab_lib/platform/honesty/registry.py` → no match
- **Committed in:** `c51e783` (Task 1 GREEN commit)

---

**Total deviations:** 1 auto-fixed (docstring wording only — no production logic changed)
**Impact on plan:** Zero scope creep. The fix was internal to the module's own docstring and did not touch `append_trial`, `read_trials`, `config_hash`, or any behavior under test.

## Issues Encountered
None beyond the deviation above.

## User Setup Required
None — no external service configuration required.

## Next Phase Readiness
- `append_trial()` and `read_trials()` are ready for Plan 05 (walk-forward runner) to wire in automatically per grid cell (D-02: no manual bookkeeping).
- `registry.path` in `config/platform_settings.yaml` (added in Plan 01) is unused by this plan's code — `registry.py` resolves its own default from `trading_crab_lib.ROOT`, matching the plan's stated key-links. A future caller may pass `path=cfg["registry"]["path"]` explicitly if config-driven override is needed.
- No blockers. This plan touched only its own two files (`registry.py`, `test_platform_registry.py`) — no changes to shared orchestrator artifacts, `config/platform_settings.yaml`, `STATE.md`, or `ROADMAP.md`.

---
*Phase: 02-honesty-infrastructure*
*Completed: 2026-07-17*

## Self-Check: PASSED

All created files verified present on disk; both commit hashes (`96e8b6d`, `c51e783`)
verified present in `git log --oneline --all`.
