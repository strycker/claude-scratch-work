---
phase: 07-regime-representation
plan: 01
subsystem: platform-backtest
tags: [jump-model, walk-forward, feature-policy, trial-registry, honesty-framework]

# Dependency graph
requires:
  - phase: 06-platform-notebook-suite
    provides: "platform/evaluation/report.py::run_full_backtest_evaluation, _reference_label_columns, and the persisted full_sample_states/filtered_state_probs artifacts (Amendment 3 item H)"
provides:
  - "A single frozen_features/frozen_l1_features keyword threaded from report.py's ONE _reference_label_columns call through run_backtest into both _refit_l1 call sites, so the walk-forward driver and the evaluation reference cannot resolve to different L1 feature spaces."
  - "TestFrozenPolicyEquivalence (6 tests) proving the freeze mechanism, including a real-checkpoint variant comparing driver vs. reference column resolution at 3 sampled decision dates."
  - "trial_tag keyword on run_backtest / no_regime_ablation / run_full_backtest_evaluation, attributing registry rows to the policy variant that produced them, with the exactly-two-rows-per-evaluation fact pinned by test."
affects: ["07-02 (checkpoint recompute — the frozen set grows from 9 to 10 columns once oil's stale-checkpoint artifact is fixed)", "07-03 (policy trials use trial_tag to distinguish the frozen-policy run from the D-03 imputation-variant trial)", "07-04 (ADR-0001 records this plan's frozen column list and the trial-ceiling formula)"]

# Actuals (#2632)
actuals:
  tokens: 8950
  tasks: 2
  commits: 2

tech-stack:
  added: []
  patterns:
    - "Keyword-only X | None = None parameter with a documented None-means-old-behavior fallback (matches run_backtest's existing min_train/registry_path convention) — no new config key for a per-run derived value."
    - "Compute-once-thread-through: a single _reference_label_columns call in report.py, computed BEFORE run_backtest, reused at every downstream L1 consumer instead of recomputed."
    - "Spy-on-canonicalize_states to capture the columns a private function actually used, since _refit_l1 returns only a states Series with no column metadata."

key-files:
  created: []
  modified:
    - src/trading_crab_lib/platform/backtest/driver.py
    - src/trading_crab_lib/platform/backtest/baselines.py
    - src/trading_crab_lib/platform/evaluation/report.py
    - tests/unit/test_platform_backtest_driver.py

key-decisions:
  - "D-01 (07-CONTEXT.md): froze the DRIVER to the REFERENCE's column policy (not vice versa) — the reference's expanding rule would have turned it into a second walk-forward, changing what §5.4 measures."
  - "Relaxed the new first_decision guard in run_full_backtest_evaluation from strict equality to first_decision <= observed minimum: run_backtest excludes degraded steps (T-05-05) from per_step_metrics, so an early degraded step can legitimately push the recorded minimum later than the index-derived first_decision. Only an earlier-than-declared start (a real leak direction) now raises."
  - "baselines.py (no_regime_ablation) was edited even though it is absent from the plan's files_modified frontmatter and <verification> confinement list — the plan's own <action> text for both tasks explicitly requires threading frozen_l1_features and trial_tag through it, and the artifacts_this_phase_produces table lists it as the exact location for the frozen_l1_features parameter. Treated as a plan-documentation gap, not a scope violation."

requirements-completed: [REG-01]

coverage:
  - id: D1
    description: "_refit_l1 accepts frozen_features: list[str] | None; when given, overrides the per-window min_history admission rule outright, using exactly the frozen columns present (in order); when None, reproduces the pre-fix _window_active_features rule exactly."
    requirement: "REG-01"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_backtest_driver.py::TestFrozenPolicyEquivalence::test_frozen_features_override_the_min_history_rule"
        status: pass
      - kind: unit
        ref: "tests/unit/test_platform_backtest_driver.py::TestFrozenPolicyEquivalence::test_frozen_features_none_reproduces_the_expanding_rule"
        status: pass
      - kind: unit
        ref: "tests/unit/test_platform_backtest_driver.py::TestFrozenPolicyEquivalence::test_frozen_list_is_not_reordered_in_transit"
        status: pass
    human_judgment: false
  - id: D2
    description: "An empty or too-short frozen_features list raises ValueError naming the resolved count (and K when applicable) — fails loudly at the boundary rather than surfacing as an opaque error from inside the clustering fit."
    requirement: "REG-01"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_backtest_driver.py::TestFrozenPolicyEquivalence::test_empty_frozen_list_raises_a_named_error"
        status: pass
      - kind: unit
        ref: "tests/unit/test_platform_backtest_driver.py::TestFrozenPolicyEquivalence::test_frozen_list_shorter_than_k_raises"
        status: pass
    human_judgment: false
  - id: D3
    description: "On the real dev monthly_features checkpoint, the driver's frozen-path column resolution and report.py::_reference_label_columns's output are identical (ordered AND set equality) at 3 sampled decision dates — the criterion-1 equivalence proof, run against real data, not skipped."
    requirement: "REG-01"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_backtest_driver.py::TestFrozenPolicyEquivalence::test_real_checkpoint_driver_and_reference_resolve_identical_columns"
        status: pass
    human_judgment: false
  - id: D4
    description: "report.py::run_full_backtest_evaluation computes _reference_label_columns exactly ONCE (before run_backtest), threads frozen_l1_features=ref_cols into both the strategy run_backtest call and the no_regime_ablation call, reuses the same dev_features/ref_cols at the full-sample fit (step d) instead of recomputing, and returns ref_cols under the key frozen_l1_features."
    requirement: "REG-01"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_evaluation_report.py::TestReferenceLabelColumns (unchanged, still green)"
        status: pass
      - kind: integration
        ref: "tests/integration/test_mini_backtest.py::TestRunFullBacktestEvaluationEndToEnd (all pass with the new frozen threading)"
        status: pass
    human_judgment: false
  - id: D5
    description: "run_backtest / no_regime_ablation / run_full_backtest_evaluation accept trial_tag: str | None = None, merged into the registry row's config under 'trial_tag' when set; omitted, trial_config keeps its exact pre-existing four keys. One strategy run_backtest + one no_regime_ablation appends exactly 2 registry rows."
    requirement: "REG-01"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_backtest_driver.py::TestTrialTag::test_trial_tag_lands_in_the_registry_config"
        status: pass
      - kind: unit
        ref: "tests/unit/test_platform_backtest_driver.py::TestTrialTag::test_no_trial_tag_leaves_config_shape_unchanged"
        status: pass
      - kind: unit
        ref: "tests/unit/test_platform_backtest_driver.py::TestTrialTag::test_one_evaluation_appends_exactly_two_rows"
        status: pass
    human_judgment: false

duration: ~55min
completed: 2026-09-14
status: complete
---

# Phase 7 Plan 1: Freeze the L1 Feature Policy Summary

**One `_reference_label_columns` call in `report.py` now feeds both the walk-forward driver's per-step L1 refits and the evaluation reference's full-sample fit, closing audit item A13's driver/reference divergence, and every registry row can now carry a `trial_tag` naming the policy variant that produced it.**

## Performance

- **Duration:** ~55 min
- **Started:** 2026-09-14T21:50:00Z (approx, session start)
- **Completed:** 2026-09-14T22:45:22Z
- **Tasks:** 2
- **Files modified:** 4 (`driver.py`, `baselines.py`, `report.py`, `tests/unit/test_platform_backtest_driver.py`)

## Accomplishments

- `_refit_l1` accepts a keyword-only `frozen_features: list[str] | None = None`. When given, it overrides `_window_active_features`'s per-window `feature_min_history` admission rule outright — the active columns are exactly the members of `frozen_features` present in `train_features.columns`, in the order given. When `None`, behavior is byte-for-byte unchanged.
- `run_backtest` threads a new `frozen_l1_features` keyword into **both** `_refit_l1` call sites in the per-step loop (the strategy branch and the tilt-off debug branch), so the discarded ablation-leg L1 fit can never silently use a different feature set than the strategy leg's.
- `report.py::run_full_backtest_evaluation` computes `_reference_label_columns` **once**, before the strategy `run_backtest` call, and passes that same `ref_cols` into both the strategy and ablation legs. Step (d)'s full-sample smoothed-reference fit reuses the SAME `dev_features`/`ref_cols` rather than recomputing them — one computation, three consumers.
- A permanent guard asserts the index-derived `first_decision` is never *after* the walk-forward loop's observed first recorded decision date, catching a future `expanding_steps` decoupling without falsely tripping on benign early-step degradation (see Deviations).
- `TestFrozenPolicyEquivalence` (6 tests) proves the mechanism, including a real-checkpoint variant (not skipped — `data/checkpoints/platform/monthly_features.parquet` is present) comparing the driver's frozen-path column resolution against `_reference_label_columns`'s output at 3 sampled decision dates, asserting both ordered-list and set equality.
- `run_backtest` / `no_regime_ablation` / `run_full_backtest_evaluation` all accept an optional `trial_tag: str | None = None`, merged into the registry row's `config` dict under `"trial_tag"` when set. `TestTrialTag` pins the exactly-two-rows-per-evaluation formula the Phase 7 ADR's trial-ceiling section will state.

## Task Commits

1. **Task 1: End-to-end — one frozen column list, computed once, consumed by both L1 fit paths** - `cc375db` (feat)
2. **Task 2: Make every registry row attributable to the policy variant that produced it** - `14e9dcc` (feat)

_Note: both tasks are `tdd="true"`; tests were written and run to RED (a `TypeError`/`AttributeError` from the not-yet-existing keyword) before the source edits, then GREEN after, within each commit's development — the executor verified the RED state interactively before committing only the GREEN result, per this repo's convention of one commit per completed task rather than separate RED/GREEN commits for `type="auto"`/`type="tracer"` tasks (not `type: tdd` PLAN-level gating)._

## Files Created/Modified

- `src/trading_crab_lib/platform/backtest/driver.py` — `_refit_l1` gains `frozen_features`; `run_backtest` gains `frozen_l1_features` and `trial_tag`, threaded into both `_refit_l1` call sites and into `trial_config`.
- `src/trading_crab_lib/platform/backtest/baselines.py` — `no_regime_ablation` gains `frozen_l1_features` and `trial_tag`, passed straight through to `run_backtest`.
- `src/trading_crab_lib/platform/evaluation/report.py` — `run_full_backtest_evaluation` computes the frozen list once before step (a), threads it (plus `trial_tag`) into steps (a)/(b), reuses it at step (d) (deleting the second `_reference_label_columns` call and the `per_step_metrics`-sourced `first_decision` recomputation), adds the `first_decision <=` guard, and returns `frozen_l1_features` in the result dict.
- `tests/unit/test_platform_backtest_driver.py` — new `TestFrozenPolicyEquivalence` (6 tests) and `TestTrialTag` (3 tests); existing `_fake_refit_l1`/`spy_refit_l1`/`counting_refit_l1` helpers updated to accept the new `frozen_features` keyword so they still match `_refit_l1`'s (now unconditionally-passed) call signature.

## Decisions Made

- **D-01 followed as locked:** froze the driver to the reference's column policy, not the reverse — recorded in the plan and re-affirmed here since it is the axis every later measurement in this phase depends on.
- **Relaxed the first_decision guard to `<=` instead of strict `==`.** The plan's literal text specified `first_decision == per_step_metrics["dates"].min()`. Running this against `tests/integration/test_mini_backtest.py`'s small synthetic walk-forward revealed that `run_backtest` legitimately excludes degraded steps (T-05-05, an existing documented behavior) from `per_step_metrics["dates"]`, so when the very first attempted step(s) degrade, the first *recorded* date is later than `first_decision`. Strict equality is not a valid invariant of the existing system; `first_decision <= observed_first_decision` preserves the safety-relevant direction (the loop must never appear to have started earlier than declared, which would signal a leak) while tolerating benign degradation-driven delay. This is a Rule 1 auto-fix on code I had just written in this same task, not a weakening of any plan-mandated test — none of the 6+3 required tests in this plan touch `run_full_backtest_evaluation`'s guard at all.
- **`baselines.py` treated as in-scope despite the files_modified/verification omission.** The plan's frontmatter `files_modified` list and `<verification>` section's "confined to driver.py, report.py, and the test file" statement omit `baselines.py`, but the plan's own `<action>` text for both tasks ("Thread the same parameter through `no_regime_ablation`") and the `artifacts_this_phase_produces` table (`frozen_l1_features` / `trial_tag` both listed against `platform/backtest/baselines.py::no_regime_ablation`) explicitly require it. Treated as an incomplete scope declaration in the plan document, not a license to skip required wiring.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Relaxed the first_decision guard from strict equality to `<=`**
- **Found during:** Task 1, running the full test suite after implementation
- **Issue:** The plan's literal action text specified asserting `first_decision == pd.DatetimeIndex(per_step_metrics["dates"]).min()`. This failed on `tests/integration/test_mini_backtest.py`'s synthetic evaluation, whose small/early data causes the walk-forward's first few steps to degrade (an existing, documented `run_backtest` behavior — degraded steps are excluded from `per_step_metrics` entirely). The strict-equality guard as written would raise on any evaluation run with early degradation, which is not itself a bug — it is normal, graceful behavior the driver has always had.
- **Fix:** Changed the assertion to `first_decision <= observed_first_decision`, preserving the guard's real purpose (catch the loop somehow starting *before* its declared first decision date — the direction that would indicate a leak) while tolerating the benign case where early degradation delays the first *recorded* date.
- **Files modified:** `src/trading_crab_lib/platform/evaluation/report.py`
- **Verification:** `tests/integration/test_mini_backtest.py` (6 previously-failing tests) all pass; full suite 1714 passed, 0 skipped.
- **Committed in:** `cc375db` (Task 1 commit)

**2. [Scope correction] Edited `baselines.py`, which is absent from the plan's `files_modified` frontmatter and `<verification>` confinement list**
- **Found during:** Task 1 and Task 2, implementing the plan's own explicit action text
- **Issue:** The plan's `<action>` sections for both tasks explicitly instruct threading `frozen_l1_features` (Task 1) and `trial_tag` (Task 2) through `no_regime_ablation` in `baselines.py`, and the plan's own `artifacts_this_phase_produces` table lists both parameters against that exact function. But `files_modified` in the frontmatter and the `<verification>` section's "git diff --stat shows edits confined to driver.py, report.py, and the test file" statement both omit `baselines.py`.
- **Fix:** Edited `baselines.py` as the plan's action text requires; documented the discrepancy here rather than silently working around it or silently expanding scope without comment.
- **Files modified:** `src/trading_crab_lib/platform/backtest/baselines.py`
- **Verification:** `ruff check` clean; all driver/report/integration/plotting tests pass.
- **Committed in:** `cc375db` (frozen_l1_features half) and `14e9dcc` (trial_tag half)

---

**Total deviations:** 2 (1 Rule 1 bug fix in newly-authored code, 1 scope-declaration correction matching the plan's own explicit action text)
**Impact on plan:** Both are necessary for correctness/completeness. No functional scope creep — no behavior was added beyond what the plan's action text and artifacts table already specified.

## Issues Encountered

- The plan's `<action>` text for the `first_decision` guard was internally inconsistent with `run_backtest`'s own documented graceful-degradation behavior (T-05-05). Resolved as described above (Rule 1) rather than weakening or removing the guard — the guard still fires correctly in the direction that matters (an earlier-than-declared start).
- Both tasks' interleaved edits (frozen_l1_features and trial_tag both touch `run_backtest`'s signature, docstring, and the `trial_config` block in adjacent regions) were implemented together in one working pass for efficiency, then deliberately split into two atomic commits matching the plan's task boundaries by temporarily removing and re-adding the Task 2 (`trial_tag`) pieces, running each task's required verification independently before each commit. No functional difference from executing them sequentially from the start.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Plan 07-02 (D-02-A checkpoint recompute) can now proceed: `report_module._reference_label_columns` is unchanged and still the single source of truth; once `monthly_features` is recomputed from the cached `monthly_raw` checkpoint, the frozen set will grow from the currently-observed **9 columns** (`cape_shiller`, `credit_spread_baa_aaa`, `curve_10y3m`, `div_yield`, `real_rate_level`, `realized_vol_1m`, `realized_vol_3m`, `trailing_return_1m`, `trailing_return_3m`) to **10** (adding `oil`), with no code change required in this plan's deliverables — `TestFrozenPolicyEquivalence`'s real-checkpoint test deliberately asserts equality between the two paths, never a hard-coded length, so it will continue to pass unchanged after 07-02's recompute.
- Plan 07-03 (policy trials) can use `trial_tag` to distinguish the frozen-policy run from the D-03 imputation-variant rejection trial in the registry itself.
- Plan 07-04 (ADR-0001) can cite this plan's exact new signatures, the `registry_rows_added = 2 × N_full_evaluation_runs` formula (now pinned by `TestTrialTag::test_one_evaluation_appends_exactly_two_rows`), and the 9-column pre-07-02 frozen set observed against the live checkpoint.
- No blockers.

---
*Phase: 07-regime-representation*
*Completed: 2026-09-14*

## Self-Check: PASSED

All 4 modified source/test files and this SUMMARY.md confirmed present on disk; both task commits (`cc375db`, `14e9dcc`) confirmed present in git history.
