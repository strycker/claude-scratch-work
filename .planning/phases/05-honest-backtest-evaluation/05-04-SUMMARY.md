---
phase: 05-honest-backtest-evaluation
plan: 04
subsystem: backtest
tags: [platform, evaluation, honesty-framework, tdd, cross-ai-review-fixes, model-metrics, brier, calibration]

# Dependency graph
requires:
  - phase: 05-honest-backtest-evaluation
    plan: 02
    provides: "backtest/driver.py run_backtest() -> (equity_curve, per_step_metrics); per_step_metrics contract (dates/proba/classes, no loop-sourced y_true, review F2)"
  - phase: 05-honest-backtest-evaluation
    plan: 03
    provides: "evaluation/sojourn_lag.py build_filtered_probs_matrix — the shared union-of-classes/K=5 padding pattern this plan re-implements internally as _reconcile_and_stack_proba (module-import-boundary kept separate per this plan's numpy/pandas/stdlib-only constraint)"
provides:
  - "src/trading_crab_lib/platform/evaluation/model_metrics.py: BIN_EDGES, compute_brier_multiclass, calibration_bins, confusion_tidy, _reconcile_and_stack_proba, report_model_metrics"
  - "brier / calibration / confusion parquet artifacts under OUTPUT_DIR/reports/platform/"
affects: [05-06-report]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "model_metrics.py's three pure metric functions are adapted math (verbatim formulas) ported BY VALUE from the Phase-3 quarterly pipeline's retired metrics helpers — never imported; grep-gated (FoldReport|model_metrics_artifacts|gsd-salvage|classifier import) to prove no import edge into the incumbent bundle-report shape (Pitfall 5, T-05-08)"
    - "_reconcile_and_stack_proba performs union-of-classes reconciliation, padding any per-step proba row missing a state observed elsewhere with 0.0, BEFORE stacking into a rectangular (n_steps, K) array — review F3, prevents a ragged early-history step from corrupting the stack shape"
    - "report_model_metrics reads y_true from per_step_metrics['y_true'] (a date-joined list the report layer supplies) and asserts len(y_true) == len(dates) == len(proba), raising ValueError on mismatch — review F2, never sourced from the in-progress walk-forward loop"
    - "empty-safe schema-stable parquet idiom (declared columns always present, even for zero-row DataFrames) ported from honesty/gap_lag.py, applied to all three artifacts (brier/calibration/confusion)"

key-files:
  created:
    - src/trading_crab_lib/platform/evaluation/model_metrics.py
    - tests/unit/test_platform_evaluation_model_metrics.py
  modified: []

key-decisions:
  - "model_metrics.py implements its own _reconcile_and_stack_proba rather than importing sojourn_lag.py's build_filtered_probs_matrix — the plan's prohibition restricts this module's imports to numpy/pandas/stdlib only (no cross-import into another platform/evaluation module), so the union-of-classes/K-padding ALGORITHM is reused (same logic pattern, verified against the same review-F3 invariant) but the CODE is a separate small function scoped to this module's plain (proba_rows: list[np.ndarray], classes_rows: list[list]) -> (np.ndarray, list) signature, distinct from build_filtered_probs_matrix's (per_step_metrics: dict) -> pd.DataFrame signature."
  - "report_model_metrics indexes per_step_metrics['y_true'] directly (dict key access, not .get()) so a caller who forgets to date-join y_true gets an immediate KeyError rather than a silent None/empty-list fallback that would then pass a misleading len()==0 mismatch message."
  - "Task 2 (not marked tdd=\"true\" in the plan) was committed as a single feat commit covering both the new RaggedClassesReconciliation/YTrueDateAlignment/ReportArtifacts test classes and the implementation, following the plan's literal task-type distinction from Task 1 (which IS tdd=\"true\" and produced separate test/feat commits) — internally the new tests were still written and confirmed RED (ImportError for the not-yet-defined names) before implementing to GREEN, matching the plan's <action> instruction, but the git history reflects the plan's task boundary rather than an extra TDD split not requested for this task."
  - "Empty per_step_metrics (0 predictions) writes three genuinely zero-row parquet files with only the declared schema columns present; a non-empty run always writes brier as a single-row DataFrame (one scalar Brier value per full backtest run) rather than a per-step brier row, since EVAL-04 scores the whole walk-forward run's pooled predictions, not per-step values."

patterns-established:
  - "Pattern: platform/evaluation modules that adapt salvage math keep the module's import list minimal (numpy/pandas/stdlib) and re-express shared cross-module algorithms (e.g. union-of-classes padding) as local, signature-scoped helpers rather than importing a sibling evaluation module — trading off a small amount of logic duplication for a hard import-boundary guarantee that's independently grep-gated."

requirements-completed: [EVAL-04]

coverage:
  - id: D1
    description: "compute_brier_multiclass(y_true, proba, classes) is ~0 for perfectly one-hot-correct predictions and strictly > 0 for a deliberately miscalibrated (flat) probability vector"
    requirement: "EVAL-04"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_evaluation_model_metrics.py::TestBrierKnownAnswer"
        status: pass
    human_judgment: false
  - id: D2
    description: "calibration_bins(y_true, proba, classes) returns per-class rows over the 5 fixed bins [0,0.2,0.4,0.6,0.8,1.0] with class_label/bin/bin_low/bin_high/predicted_prob_mean/observed_freq/n_in_bin, the top bin inclusive of 1.0, and class_label forced to str"
    requirement: "EVAL-04"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_evaluation_model_metrics.py::TestCalibrationBinEdges"
        status: pass
    human_judgment: false
  - id: D3
    description: "confusion_tidy(y_true, y_pred, classes) counts sum to len(y_true); true_label/pred_label forced to str"
    requirement: "EVAL-04"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_evaluation_model_metrics.py::TestConfusionSumsToN"
        status: pass
    human_judgment: false
  - id: D4
    description: "_reconcile_and_stack_proba pads a ragged early-history step (fewer than K observed classes) with 0.0 for the missing states, producing a rectangular (n_steps, K) array with a single reconciled class list; the reconciled stack scores through compute_brier_multiclass/calibration_bins without a shape error (review F3)"
    requirement: "EVAL-04"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_evaluation_model_metrics.py::TestRaggedClassesReconciliation"
        status: pass
    human_judgment: false
  - id: D5
    description: "report_model_metrics scores against a y_true supplied by the caller and joined by date; a length mismatch between y_true and dates/proba raises ValueError (review F2 — never sourced from the in-progress walk-forward loop)"
    requirement: "EVAL-04"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_evaluation_model_metrics.py::TestYTrueDateAlignment"
        status: pass
    human_judgment: false
  - id: D6
    description: "report_model_metrics(per_step_metrics, output_dir) persists three schema-stable brier/calibration/confusion parquet artifacts, each round-trippable via read_parquet with the declared columns, and remains schema-stable (zero rows, declared columns) when per_step_metrics has no predictions at all"
    requirement: "EVAL-04"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_evaluation_model_metrics.py::TestReportArtifacts"
        status: pass
    human_judgment: false
  - id: D7
    description: "model_metrics.py imports only numpy/pandas/stdlib at module level — no import edge into the incumbent prediction package, FoldReport, or the salvage module itself"
    requirement: "EVAL-04"
    verification:
      - kind: other
        ref: "grep -nE 'FoldReport|model_metrics_artifacts|gsd-salvage|^(from|import).*classifier' src/trading_crab_lib/platform/evaluation/model_metrics.py returns nothing"
        status: pass
    human_judgment: false

duration: 15min
completed: 2026-07-24
status: complete
---

# Phase 5 Plan 4: Model Metrics Artifacts Summary

**`model_metrics.py` — multiclass Brier score, 5-bin calibration tables, and tidy confusion counts adapted (math ported by value, never imported) from the retired Phase-3 metrics helpers, orchestrated by a NEW `report_model_metrics` that reconciles ragged per-step classes to a padded K=5 array before scoring (review F3) and scores against a date-joined `y_true` with a hard length-alignment guard (review F2), persisting schema-stable empty-safe parquet artifacts.**

## Performance

- **Duration:** 15 min
- **Started:** 2026-07-24T22:20:00Z
- **Completed:** 2026-07-24T22:35:00Z
- **Tasks:** 2 completed
- **Files modified:** 2 (both created)

## Accomplishments

- Implemented `src/trading_crab_lib/platform/evaluation/model_metrics.py`: `BIN_EDGES` + `compute_brier_multiclass`, `calibration_bins`, `confusion_tidy` — pure functions over plain `(y_true: list, proba: np.ndarray, classes: list)` / `(y_true, y_pred, classes)` inputs, math adapted verbatim from the Phase-3 quarterly pipeline's retired metrics helpers but never importing that file or its incumbent `FoldReport`/bundle-API dataclass (Pitfall 5, T-05-08).
- Grep gate (`FoldReport|model_metrics_artifacts|gsd-salvage|^(from|import).*classifier`) returns nothing against `model_metrics.py` — no import edge into the incumbent prediction package anywhere in the module.
- Implemented `_reconcile_and_stack_proba(proba_rows, classes_rows) -> (np.ndarray, list)`: unions every state observed across all walk-forward steps and pads any per-step proba row missing a state with `0.0`, producing a rectangular `(n_steps, K)` array plus the reconciled class list — review F3. An early step observing only 3 of 5 canonical states (`classes=[0,1,2]`, proba width 3) is padded to width 5 without corrupting the stack or misaligning columns.
- Implemented `report_model_metrics(per_step_metrics, *, output_dir=None) -> dict[str, Path]`: reads `per_step_metrics["y_true"]` — a list the report layer (Plan 06) has already joined by date from the full-sample smoothed reference labeling, never sourced from the in-progress walk-forward loop (review F2) — and asserts `len(y_true) == len(dates) == len(proba)`, raising `ValueError` on any mismatch. Reconciles+stacks the proba via `_reconcile_and_stack_proba`, derives `y_pred` as the argmax reconciled class per step, scores via the three pure functions, and writes three schema-stable parquet artifacts (`model_metrics_brier.parquet`, `model_metrics_calibration.parquet`, `model_metrics_confusion.parquet`) under `OUTPUT_DIR/reports/platform/` (or `output_dir` for tests) — empty-safe (declared columns, zero rows) when there are no predictions at all.
- `__main__` self-check builds a tiny synthetic `per_step_metrics` dict with a deliberately ragged first step (3 of 5 states observed) and a date-joined `y_true`, and writes the three artifacts to a temp dir — runs standalone with no network/checkpoint dependency, mirroring `kpis.py`/`sojourn_lag.py`/`gap_lag.py`'s self-check convention.

## Task Commits

Each task was committed atomically (TDD RED→GREEN for Task 1; single commit for Task 2, not marked `tdd="true"` in the plan):

1. **Task 1: Write model_metrics.py test file (RED)** — `047e29c` (test) — `TestBrierKnownAnswer`, `TestCalibrationBinEdges`, `TestConfusionSumsToN` (8 tests) fail at collection because `trading_crab_lib.platform.evaluation.model_metrics` did not yet exist.
2. **Task 1: Implement the three pure metric functions (GREEN)** — `1fb767e` (feat) — all 8 Task-1 tests pass; grep gate confirms no salvage/incumbent import tokens; module imports only numpy/stdlib at this stage.
3. **Task 2: Implement report_model_metrics orchestrator (review F2/F3)** — `7a3bec4` (feat) — added `TestRaggedClassesReconciliation`, `TestYTrueDateAlignment`, `TestReportArtifacts` to the test file (confirmed RED via `ImportError` for the not-yet-defined `_reconcile_and_stack_proba`/`report_model_metrics` names) then implemented `_reconcile_and_stack_proba` + `report_model_metrics` + `__main__` self-check to GREEN; full test file (14 tests) passes.

## Files Created/Modified

- `src/trading_crab_lib/platform/evaluation/model_metrics.py` — new module: `BIN_EDGES` + `compute_brier_multiclass`/`calibration_bins`/`confusion_tidy` (Task 1) + `_reconcile_and_stack_proba`/`report_model_metrics` + `__main__` self-check (Task 2).
- `tests/unit/test_platform_evaluation_model_metrics.py` — new test file: 14 tests across 6 classes (`TestBrierKnownAnswer`, `TestCalibrationBinEdges`, `TestConfusionSumsToN`, `TestRaggedClassesReconciliation`, `TestYTrueDateAlignment`, `TestReportArtifacts`).

## Decisions Made

- **`_reconcile_and_stack_proba` is a self-contained implementation, not an import of `sojourn_lag.py`'s `build_filtered_probs_matrix`.** Both functions perform the same union-of-classes/zero-padding algorithm (established as a shared pattern in Plan 03's summary), but the plan's prohibitions explicitly scope this module's imports to numpy/pandas/stdlib only — no import edge into any other `platform/evaluation` module. `_reconcile_and_stack_proba` therefore re-implements the pattern locally with a plain `(proba_rows: list[np.ndarray], classes_rows: list[list]) -> (np.ndarray, list)` signature (matching the plan's literal `<action>` spec), distinct from `build_filtered_probs_matrix`'s `(per_step_metrics: dict) -> pd.DataFrame` signature. This is a small, deliberate logic-duplication tradeoff in exchange for a hard, independently grep-gated import boundary (T-05-08).
- **`per_step_metrics["y_true"]` is accessed via direct dict indexing, not `.get()`.** A caller who omits the date-joined `y_true` key gets an immediate `KeyError` rather than a silently defaulted `None`/`[]` that would produce a confusing generic length-mismatch message — the missing-key failure mode is clearer for a field the report layer is contractually required to supply (review F2).
- **Task 2 committed as a single `feat` commit**, following the plan's literal task-type distinction (Task 1 is `tdd="true"`; Task 2 is plain `type="auto"`). The new test classes were still written and confirmed RED (collection-time `ImportError` for the not-yet-defined `_reconcile_and_stack_proba`/`report_model_metrics` names) before implementing to GREEN, per the plan's `<action>` instruction — but this RED→GREEN cycle is reflected within one task commit rather than split into separate `test`/`feat` commits, matching how Plan 02's non-TDD tasks were committed.
- **A full backtest run's Brier score is a single scalar** (one row in `model_metrics_brier.parquet`), not a per-step time series — EVAL-04 scores the pooled predictions across the whole walk-forward run, consistent with how `calibration_bins`/`confusion_tidy` also pool across all steps rather than reporting per-step tables.

## Deviations from Plan

None — the plan's `<action>` blocks were followed as written, including the exact `must_haves.truths` wording for F2/F3, the module/function/constant names, the artifact directory convention (`OUTPUT_DIR/reports/platform/`), and the empty-safe schema-stable parquet idiom.

## Issues Encountered

None.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- `report_model_metrics` is ready for Plan 06 (report layer) to call directly against `run_backtest`'s `per_step_metrics` (Plan 02 output) once the report layer joins a `y_true` list by date from the full-sample smoothed reference labeling (`labeling/diagnostics.py`-produced state series, already available from Phase 3) — the same date-join Plan 06 already performs for `sojourn_lag.py`'s `compute_sojourn_lag_headline` (Plan 03).
- `_reconcile_and_stack_proba`'s union-of-classes/K=5-padding behavior is proven independently of `sojourn_lag.py`'s `build_filtered_probs_matrix`, so Plan 06 can freely use either helper for its respective purpose (multiclass scoring vs. the sojourn/lag detection-lag column lookup) without any cross-dependency between the two `evaluation/` modules.
- All three artifacts (`model_metrics_brier.parquet`, `model_metrics_calibration.parquet`, `model_metrics_confusion.parquet`) write under the same `OUTPUT_DIR/reports/platform/` directory convention already used by `assets/returns.py`, `assets/vol.py`, and `report/weekly.py` — Plan 06's report assembly can read all evaluation artifacts from one directory.
- No blockers.

---
*Phase: 05-honest-backtest-evaluation*
*Completed: 2026-07-24*

## Self-Check: PASSED

Both created files confirmed present on disk; both commit hashes
(047e29c, 1fb767e, 7a3bec4) confirmed present in `git log --oneline --all`.
