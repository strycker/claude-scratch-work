---
phase: 02-honesty-infrastructure
plan: 04
subsystem: infra
tags: [honesty-framework, causal-features, look-ahead-bias, gap-lag, checkpoints, pandas, pyarrow]

# Dependency graph
requires:
  - phase: 02-honesty-infrastructure
    provides: "02-01: platform/checkpoints.py::get_platform_checkpoint_manager, config/platform_settings.yaml holdout:/registry:/cv: sections, platform/honesty/ package barrel"
provides:
  - "src/trading_crab_lib/platform/honesty/gating.py: FORBIDDEN_CENTERED_SUFFIXES, assert_causal_features(), select_platform_feature_path()"
  - "src/trading_crab_lib/platform/honesty/gap_lag.py: compute_gap(), compute_detection_lag(), sojourn_lag_ratio(), report_gap_lag(), __main__ CLI demo"
  - "outputs/reports/model_metrics/gap_lag_metrics.parquet artifact convention (D-05)"
affects: [phase-3-modeling, 02-05, phase-4-report-wiring]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Causal-feature gating as an assertion guard (scan columns, refuse loudly by default) rather than a two-file selector — monthly_features is causal-by-construction, resolving RESEARCH.md Open Question 1"
    - "Empty-safe DataFrame + to_parquet idiom (rows list -> DataFrame; if empty, rebuild with an explicit column list) for schema-stable artifacts, ported from model_metrics_artifacts.py"
    - "Pure compute functions (compute_gap, compute_detection_lag, sojourn_lag_ratio) kept separate from the I/O/reporting function (report_gap_lag) in the same module, matching PATTERNS.md guidance"

key-files:
  created:
    - src/trading_crab_lib/platform/honesty/gating.py
    - src/trading_crab_lib/platform/honesty/gap_lag.py
    - tests/unit/test_platform_gating.py
    - tests/unit/test_platform_gap_lag.py
  modified: []

key-decisions:
  - "assert_causal_features raises ValueError (not FileNotFoundError) for a forbidden-column violation, reserving FileNotFoundError for the genuinely-missing-checkpoint case in select_platform_feature_path — keeps the two failure modes distinguishable"
  - "compute_detection_lag takes integer positions into filtered_probs (not label-index timestamps) — matches RESEARCH.md's 'index-aligned pandas Series / integer positions' wording and keeps the function usable before Phase 3's real regime index exists"
  - "An unresolved transition (probability never crosses threshold) reports NaN and is excluded from the median — never silently defaults to 0 or drops the transition entry entirely"
  - "report_gap_lag's output_dir parameter replaces the full target directory (not just the OUTPUT_DIR base) so tests can assert the exact written path without relying on OUTPUT_DIR internals"

patterns-established:
  - "gap_lag.py's __main__ block is both the CLI surface and a synthetic self-check — runnable end-to-end with `python3 -m trading_crab_lib.platform.honesty.gap_lag`, no network, no Phase 3 dependency"

requirements-completed: [HON-05, HON-06]

coverage:
  - id: D1
    description: "assert_causal_features refuses forbidden centered-window columns by default with a loud ValueError naming the offending column(s); allow_noncausal=True opts out with a NONCAUSAL_USED WARNING and returns True"
    requirement: "HON-06"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_gating.py::TestAssertCausalFeatures (3 tests: clean/blocked/opt-out)"
        status: pass
    human_judgment: false
  - id: D2
    description: "select_platform_feature_path resolves monthly_features.parquet via get_platform_checkpoint_manager (or an override dir), reads its schema cheaply via pyarrow, and runs the gating guard — no monthly_features_supervised path referenced anywhere"
    requirement: "HON-06"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_gating.py::TestSelectPlatformFeaturePath (4 tests: clean/blocked/opt-out/missing-file)"
        status: pass
      - kind: other
        ref: "grep -c monthly_features_supervised src/trading_crab_lib/platform/honesty/gating.py -> 0"
        status: pass
    human_judgment: false
  - id: D3
    description: "compute_gap, compute_detection_lag, and sojourn_lag_ratio produce correct values on synthetic smoothed/filtered series and raise ValueError on malformed input (empty series, out-of-bounds transitions, non-positive lag)"
    requirement: "HON-05"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_gap_lag.py::TestComputeGap, TestComputeDetectionLag, TestSojournLagRatio (13 tests)"
        status: pass
    human_judgment: false
  - id: D4
    description: "report_gap_lag() prints a stdout summary and persists a schema-stable parquet artifact under outputs/reports/model_metrics/ (D-05); a __main__ demo proves the full compute -> report path end-to-end on synthetic data with exit 0"
    requirement: "HON-05"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_gap_lag.py::TestReportGapLag (2 tests: writes-artifact, empty-metrics-schema-stable)"
        status: pass
      - kind: other
        ref: "python3 -m trading_crab_lib.platform.honesty.gap_lag (exit 0, prints sojourn/lag ratio line)"
        status: pass
    human_judgment: false

# Metrics
duration: 3min
completed: 2026-07-17
status: complete
---

# Phase 2 Plan 04: Causal-Feature Gating + Gap/Lag Metrics Summary

**Assertion-style causal-feature guard for `monthly_features` (HON-06) plus the generic, tested §5.4 gap/lag compute functions with a CLI + persisted-artifact reporting surface proven end-to-end on synthetic series (HON-05, D-05).**

## Performance

- **Duration:** ~3 min (5 commits, 21:00:02Z → 21:02:44Z)
- **Tasks:** 3 completed
- **Files modified:** 4 (all created; no files outside the plan's scope touched)

## Accomplishments
- `gating.py`: `assert_causal_features()` scans column names for forbidden centered-window suffixes (`_centered`, `_c5`, `_zerophase`) and raises loudly by default; `allow_noncausal=True` opts out with a `NONCAUSAL_USED` WARNING and a returned flag — the P1 look-ahead sin cannot happen silently in the platform namespace
- `select_platform_feature_path()` resolves `monthly_features.parquet` via `get_platform_checkpoint_manager()` (or an override dir for tests), reads its schema cheaply via `pyarrow.parquet.ParquetFile(...).schema_arrow` (no full-file row read), and runs the guard
- Confirmed and preserved the 02-PATTERNS.md resolution of RESEARCH.md Open Question 1: `monthly_features` is causal-by-construction (only trailing `rolling(3).std`, no `center=True` anywhere) — the guard is an assertion, not a two-file selector; `grep -c monthly_features_supervised gating.py` returns 0
- `gap_lag.py`: `compute_gap`, `compute_detection_lag`, and `sojourn_lag_ratio` implement design §5.4's exact definitions as pure, deterministic functions with explicit `ValueError` guards on malformed input, proven on hand-constructed synthetic series where the expected lag/median/ratio are known by construction
- `report_gap_lag()` prints a stdout summary (D-05 CLI run output) and writes a schema-stable parquet artifact under `outputs/reports/model_metrics/gap_lag_metrics.parquet`, using the empty-safe DataFrame + `to_parquet` idiom ported from `model_metrics_artifacts.py`
- `python3 -m trading_crab_lib.platform.honesty.gap_lag` runs the full compute → report pipeline on synthetic data end-to-end (exit 0, prints the sojourn/lag ratio) — no network, no Phase 3 dependency; Phase 3 plugs real jump-model/nowcaster series into the same three compute functions with no interface change

## Task Commits

Each task was committed atomically (Tasks 1 and 2 were TDD — RED then GREEN):

1. **Task 1: Causal-feature gating guard (HON-06)** - `41970e5` (test, RED) → `34f6b3c` (feat, GREEN)
2. **Task 2: Gap/lag pure compute functions (HON-05)** - `0e2819a` (test, RED) → `9506ab6` (feat, GREEN)
3. **Task 3: Gap/lag reporting surface — CLI summary + persisted artifact (HON-05, D-05)** - `8b623e7` (feat)

**Plan metadata:** SUMMARY commit (this file) follows.

_Note: Task 1 and Task 2 RED commits prove genuine test failure (ModuleNotFoundError, not a collection skip) before the corresponding GREEN implementation existed. Task 3 has no `tdd="true"` flag per the plan and was implemented + tested + committed as a single atomic auto task._

## Files Created/Modified
- `src/trading_crab_lib/platform/honesty/gating.py` - `FORBIDDEN_CENTERED_SUFFIXES` constant, `assert_causal_features()`, `select_platform_feature_path()`
- `src/trading_crab_lib/platform/honesty/gap_lag.py` - `compute_gap()`, `compute_detection_lag()`, `sojourn_lag_ratio()`, `report_gap_lag()`, `__main__` synthetic CLI demo
- `tests/unit/test_platform_gating.py` - 7 tests across 2 classes (assert_causal_features clean/blocked/opt-out; select_platform_feature_path clean/blocked/opt-out/missing-file)
- `tests/unit/test_platform_gap_lag.py` - 15 tests across 4 classes (compute_gap; compute_detection_lag known-construction/median/unresolved/ValueError guards; sojourn_lag_ratio; report_gap_lag artifact/empty-schema)

## Decisions Made
- `assert_causal_features` raises `ValueError` (not `FileNotFoundError`) for a forbidden-column violation, reserving `FileNotFoundError` for the genuinely-missing-checkpoint case in `select_platform_feature_path` — keeps the two failure modes cleanly distinguishable for callers.
- `compute_detection_lag` takes integer positions into `filtered_probs` rather than label-index timestamps, matching RESEARCH.md's "index-aligned pandas Series / integer positions" phrasing and keeping the function directly usable once Phase 3's real regime index exists, with no interface change.
- An unresolved transition (probability never crosses the threshold at or after its position) reports `NaN` in the per-transition lag list and is excluded from the median — never silently defaulted to `0` and never dropped from the returned list entirely, so callers can always see which transitions were unresolved.
- `report_gap_lag`'s `output_dir` kwarg replaces the entire target directory (not just an `OUTPUT_DIR` base to append `reports/model_metrics/` onto) — simpler contract for tests, which pass the exact directory they expect the artifact under.
- Used `pyarrow.parquet.ParquetFile(path).schema_arrow.names` for the cheap column read in `select_platform_feature_path`, avoiding a full-frame `pd.read_parquet` load just to inspect column names.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Removed a literal "monthly_features_supervised" token from gating.py's docstring**
- **Found during:** Task 1 GREEN phase, pre-commit acceptance-criteria check
- **Issue:** The module docstring explained the RESOLVED Open Question 1 decision using the literal phrase "no `monthly_features_supervised.parquet` companion file exists" — this is exactly the token the plan's own acceptance criterion negative-greps for (`grep -c "monthly_features_supervised" gating.py` must return 0), and the plan explicitly warned: "do not write a literal forbidden token into a comment that a negative-grep acceptance check would then trip on."
- **Fix:** Reworded the docstring to "no causal/noncausal companion checkpoint file exists or should be created" — same meaning, no literal token.
- **Files modified:** `src/trading_crab_lib/platform/honesty/gating.py`
- **Verification:** `grep -c "monthly_features_supervised" src/trading_crab_lib/platform/honesty/gating.py` → `0`; `pytest tests/unit/test_platform_gating.py -x -q` → 7 passed
- **Committed in:** `34f6b3c` (Task 1 GREEN commit)

---

**Total deviations:** 1 auto-fixed (1 bug fix, docstring wording only — no behavioral change)
**Impact on plan:** Zero scope creep. Fix was internal to `gating.py`'s docstring and required by the plan's own stated acceptance criterion.

## Issues Encountered
None beyond the deviation above. The `python3 -m trading_crab_lib.platform.honesty.gap_lag` demo writes to `outputs/reports/model_metrics/` at the repo's real `OUTPUT_DIR` when run without `output_dir=` override (by design, per D-05) — this generated an untracked `outputs/` directory during verification, which was deleted (`rm -rf outputs/`) before each commit since it is runtime output, not a plan artifact.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- `select_platform_feature_path()` is ready for any Phase 3+ supervised-training entry point in the platform namespace to call before loading `monthly_features` for training.
- `compute_gap()`, `compute_detection_lag()`, and `sojourn_lag_ratio()` are ready for Phase 3 to plug real jump-model-labeler smoothed labels and logistic-nowcaster filtered probabilities into, with no interface change.
- `report_gap_lag()`'s artifact convention (`outputs/reports/model_metrics/gap_lag_metrics.parquet`) is ready for Phase 4's weekly-report wiring (D-05) — no incumbent report path (`email.py`, `reporting.py`, `write_weekly_report_md`) was touched this plan.
- No blockers. This plan's four files are fully independent of sibling Wave 2 plans (02-02 registry.py, 02-03 cv.py) — only shared Wave 1 foundation (`platform/checkpoints.py`, `platform/honesty/__init__.py`) was imported, never modified.

---
*Phase: 02-honesty-infrastructure*
*Completed: 2026-07-17*

## Self-Check: PASSED

All 4 created files verified present on disk; all 6 task/summary commit hashes
(`41970e5`, `34f6b3c`, `0e2819a`, `9506ab6`, `8b623e7`, `2d08149`) verified
present in `git log --oneline --all`.
