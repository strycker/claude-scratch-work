---
phase: 03-regime-labeling-prediction
plan: 03
subsystem: prediction
tags: [sklearn, calibration, logistic-regression, purged-cv, trial-registry, joblib]

# Dependency graph
requires:
  - phase: 02-honesty-infrastructure
    provides: PurgedEmbargoedKFold (honesty/cv.py), append_trial/read_trials (honesty/registry.py), assert_causal_features (honesty/gating.py), get_platform_checkpoint_manager (checkpoints.py)
provides:
  - build_nowcaster_training_set(features_df, labels, *, embargo_months=12) — structural D-01 embargo
  - fit_nowcaster(X, y, *, label_horizon=12, embargo=1, n_splits=5, random_state=42) — CalibratedClassifierCV(LogisticRegression, method="sigmoid") through PurgedEmbargoedKFold
  - transition_window_accuracy(y_true, y_pred, *, window_months=3) — overall/transition/steady-state accuracy dict
  - evaluate_nowcaster(features_df, labels, *, embargo_months=12, min_train=60, registry_path=None, config=None) — end-to-end eval + one registry trial + joblib model persistence
affects: [03-04, phase-4-asset-prediction]

# Tech tracking
tech-stack:
  added: []
  patterns: [structural-once-embargo distinct from per-fold PurgedEmbargoedKFold embargo, never-argmax predict_proba output contract, one-registry-trial-per-evaluation]

key-files:
  created:
    - src/trading_crab_lib/platform/prediction/__init__.py
    - src/trading_crab_lib/platform/prediction/nowcaster.py
    - tests/unit/test_platform_nowcaster.py
  modified: []

key-decisions:
  - "evaluate_nowcaster evaluates in-sample against the full embargoed set rather than a per-step walk-forward refit (nested CalibratedClassifierCV inside an expanding-window loop is impractical this early — small early train slices can't guarantee all 5 classes per calibration fold); still one registry trial per call, still leakage-guarded by D-01 + PurgedEmbargoedKFold"
  - "Model persisted via CheckpointManager.save_model() (existing joblib-backed helper) rather than a new joblib.dump call — reuses incumbent infra (P27/D14) instead of hand-rolling persistence"

requirements-completed: [L2-01, L1-02]

coverage:
  - id: D1
    description: "build_nowcaster_training_set physically excludes the trailing 12 months of labels from the nowcaster training targets (D-01 structural embargo)"
    requirement: "L1-02"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_nowcaster.py::TestEmbargoInvariant#test_trailing_12_months_excluded_on_real_output"
        status: pass
      - kind: unit
        ref: "tests/unit/test_platform_nowcaster.py::TestEmbargoInvariant#test_negative_embargo_raises_before_slicing"
        status: pass
    human_judgment: false
  - id: D2
    description: "fit_nowcaster returns a calibrated multinomial classifier whose predict_proba rows sum to ~1.0 — a distribution, never argmax"
    requirement: "L2-01"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_nowcaster.py::TestNowcasterCalibratedOutput#test_predict_proba_shape_and_rows_sum_to_one"
        status: pass
      - kind: unit
        ref: "tests/unit/test_platform_nowcaster.py::TestNowcasterCalibratedOutput#test_no_multi_class_kwarg_on_logistic_regression"
        status: pass
    human_judgment: false
  - id: D3
    description: "transition_window_accuracy reports overall, transition-window, and steady-state accuracy together, never overall alone (§5.1 persistence-trap)"
    requirement: "L2-01"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_nowcaster.py::TestTransitionWindowAccuracy#test_hand_computed_transition_vs_steady_state"
        status: pass
      - kind: unit
        ref: "tests/unit/test_platform_nowcaster.py::TestTransitionWindowAccuracy#test_no_transitions_returns_nan_transition_accuracy_without_raising"
        status: pass
    human_judgment: false
  - id: D4
    description: "evaluate_nowcaster logs exactly one trial row to the registry per call and persists the model via joblib (never raw pickle, P27)"
    requirement: "L2-01"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_nowcaster.py::TestRegistryLoggingOne#test_exactly_one_new_trial_per_evaluate_call"
        status: pass
      - kind: unit
        ref: "tests/unit/test_platform_nowcaster.py::TestRegistryLoggingOne#test_model_persisted_via_joblib_never_raw_pickle"
        status: pass
    human_judgment: false

duration: 4min
completed: 2026-07-22
status: complete
---

# Phase 3 Plan 3: Calibrated Nowcaster + Structural Label Embargo Summary

**CalibratedClassifierCV-wrapped multinomial LogisticRegression fit through PurgedEmbargoedKFold, with a structural 12-month label embargo and a transition-window-accuracy headline metric that never reports overall accuracy alone**

## Performance

- **Duration:** ~4 min (first RED commit to final GREEN commit)
- **Started:** 2026-07-22T19:12:12Z
- **Completed:** 2026-07-22T19:15:41Z
- **Tasks:** 3
- **Files modified:** 3 (2 created source files, 1 created test file)

## Accomplishments
- `build_nowcaster_training_set` physically excludes the trailing N months (default 12) of labels from ever becoming a training target — proven by a real assertion on real synthetic output, not a mocked boundary (D-01/L1-02)
- `fit_nowcaster` wraps a plain (no `multi_class` kwarg — sklearn 1.9.0 removed it) `LogisticRegression` in `CalibratedClassifierCV(method="sigmoid")` through `PurgedEmbargoedKFold` (Phase 2, reused verbatim, no new CV code), gated by `assert_causal_features` on its inputs
- `transition_window_accuracy` reports `overall_accuracy`, `transition_accuracy`, and `steady_state_accuracy` together, always — verified against a hand-computed 10-month synthetic series with errors deliberately placed inside and outside a ±1-month transition window, plus a no-transition (constant series) nan-not-raise case
- `evaluate_nowcaster` builds the embargoed training set, fits, computes metrics, logs exactly one row to the trial registry per call, and persists the fitted model via the existing joblib-backed `CheckpointManager.save_model()` helper

## Task Commits

Each task was committed atomically with TDD RED→GREEN pairs:

1. **Task 1: Structural embargo training-set builder**
   - `c091ee9` test(03-03): add failing test for structural 12-month label embargo (RED)
   - `e92b8e8` feat(03-03): implement structural 12-month label embargo (GREEN)
2. **Task 2: Calibrated fit through PurgedEmbargoedKFold + registry logging**
   - `77b4a3f` test(03-03): add failing tests for calibrated fit + registry logging (RED)
   - `19c4d1e` feat(03-03): implement calibrated nowcaster fit + registry logging (GREEN)
3. **Task 3: Transition-window accuracy headline metric**
   - `99a2706` test(03-03): add failing test for transition-window accuracy metric (RED)
   - `4b9148d` feat(03-03): implement transition-window accuracy headline metric (GREEN)

_TDD Gate Compliance: every task has a `test(...)` commit before its `feat(...)` commit; all 6 commits confirmed present in `git log`._

## Files Created/Modified
- `src/trading_crab_lib/platform/prediction/__init__.py` - package init (module docstring only, mirrors `honesty/__init__.py`)
- `src/trading_crab_lib/platform/prediction/nowcaster.py` - `build_nowcaster_training_set`, `fit_nowcaster`, `transition_window_accuracy`, `evaluate_nowcaster`
- `tests/unit/test_platform_nowcaster.py` - `TestEmbargoInvariant`, `TestNowcasterCalibratedOutput`, `TestRegistryLoggingOne`, `TestTransitionWindowAccuracy` (12 tests total)

## Decisions Made
- **In-sample evaluation, not per-step walk-forward refit:** `evaluate_nowcaster` fits once against the full embargoed training set rather than nesting `CalibratedClassifierCV` inside `run_walkforward`'s expanding-window loop. The plan's action block explicitly sanctioned this fallback ("if the nested CalibratedClassifierCV-inside-walkforward is impractical on the available window, fall back to a single batch fit + registry.append_trial") — early expanding-window slices (e.g. `min_train=60` rows split 5 ways by calibration CV) cannot reliably guarantee all 5 regime classes appear in every fold's train split, which would intermittently break `LogisticRegression.fit`. The D-01 structural embargo and the per-fold `PurgedEmbargoedKFold` purge/embargo still provide the leakage guarantees this phase requires; `min_train` is recorded in the logged trial config for traceability even though it isn't used to gate a walk-forward loop in this implementation.
- **Model persistence via `CheckpointManager.save_model()`:** rather than adding a bespoke `joblib.dump(...)` call, reused the existing joblib-backed `save_model`/`load_model` helpers already on `CheckpointManager` (incumbent P27/D14 migration) via `get_platform_checkpoint_manager().save_model(model, "nowcaster")` — smaller diff, same guarantee (joblib, never raw pickle), no duplicate persistence code.
- Two "embargo" concepts (D-01 structural, once-applied vs. `PurgedEmbargoedKFold`'s per-fold purge/embargo) are named distinctly in code and comments per RESEARCH.md Pitfall 4 — `build_nowcaster_training_set`'s `embargo_months` vs. `fit_nowcaster`'s `label_horizon`/`embargo`.

## Deviations from Plan

None — plan executed exactly as written, including the explicitly-sanctioned walk-forward-impractical fallback path documented above.

## Issues Encountered

None. `python3 -m pytest tests/unit/test_platform_nowcaster.py -x -q` passes (12/12), the full platform suite (`tests/unit/test_platform_*.py`) passes (150 passed, 1 skipped — pre-existing, unrelated to this plan), and `ruff check` is clean on all new files.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- `platform/prediction/nowcaster.py` is ready for Plan 03-04 (transition-matrix diagnostic, per PATTERNS.md's separate `transition_matrix.py` file) and for Phase 4's asset-prediction/weekly-report layer to consume `evaluate_nowcaster`'s output.
- No blockers. The labeler (L1-01/L1-03, `platform/labeling/`) that produces the real `regime_labels` this module's `labels` parameter expects is built in a sibling plan within this same phase/wave — this plan only depends on the labels' *shape* (an integer-state `pd.Series` on a `DatetimeIndex`), not on the labeler's implementation, so no ordering dependency exists between the two.

---
*Phase: 03-regime-labeling-prediction*
*Completed: 2026-07-22*
