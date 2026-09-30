---
phase: 08-regime-persistence-stability
plan: 16
subsystem: platform/report (serving build + weekly Bayes filter), platform/backtest (the shared L2 fit)
tags: [gap-closure, CR-01, bayes-filter, class-prior, train-serve-skew, honesty, no-registry, tdd, tracer]
status: complete
requires:
  - "08-15 (weekly page, neutral-posture per-asset rows)"
  - "08-13 (serving builder; fit_l2_nowcaster as the one L2 fit recipe)"
provides:
  - "driver.training_class_prior(y_train, classes): the ONE rule for the likelihood's class prior"
  - "driver.fit_l2_nowcaster -> (model, active, class_prior); pre-drops non-finite rows itself"
  - "serving.SERVING_CLASS_PRIOR = 'nowcaster_class_prior', saved beside the model; facts carry class_prior"
  - "weekly._served_class_prior(cm, nowcaster): validates the artifact against classes_ before any save"
  - "weekly.advance_regime_belief(..., class_prior=...) REQUIRED, no default"
  - "tests: TestTrainingClassPrior, TestTheServedClassPrior, TestTheServedModelOnTheTrackedData, re-derived weekly cold-start test"
affects:
  - "08-17 (switches both backtest drivers to the returned prior; _refit_l2's return type changes there)"
  - "08-19 (re-measures everything the change moves; regenerates the real-data page)"
tech-stack:
  added: []
  patterns:
    - "the fit returns its own training prior; the prior travels with the model as an artifact; the consumer refuses a mismatched pair"
    - "two roles, two rules: pi_0 and A from regime_labels over all K; L_t divided by the training prior over classes_"
key-files:
  created:
    - .planning/phases/08-regime-persistence-stability/08-16-SUMMARY.md
  modified:
    - src/trading_crab_lib/platform/backtest/driver.py
    - src/trading_crab_lib/platform/report/serving.py
    - src/trading_crab_lib/platform/report/weekly.py
    - tests/unit/test_platform_report_serving.py
    - tests/unit/test_platform_report_weekly.py
    - tests/unit/test_platform_pooling_consumers.py
    - README.md
    - CLAUDE.md
decisions:
  - "CR-01 at serve: the Bayes filter's likelihood divides the posterior by the served model's training class prior (the rows fit_l2_nowcaster fit on, over classes_), persisted as nowcaster_class_prior beside the model; never recomputed from regime_labels."
  - "pi_0 and A stay on regime_labels over all K states (two roles, two rules): pi_0 is a belief over every state and enters once; the likelihood prior is what the L2 posterior was calibrated against and enters every step."
  - "advance_regime_belief's class_prior is a required keyword with no default: a default would silently re-open CR-01."
metrics:
  duration: "~35 min"
  completed: 2026-09-29
estimate:
  tokens: 95000
  tasks: 3
actuals:
  tokens: 7000    # chars/4 over the realized diff (28092 chars of +/- lines, adae2c4..HEAD, src/tests/README/CLAUDE)
  tasks: 3
  commits: 5    # 4 task commits + the SUMMARY commit
---

# Phase 8 Plan 16: CR-01 on the Served Path Summary

The weekly Bayes filter now divides the nowcaster's posterior by the class prior of the rows the served model was actually fit on, not by the whole-label distribution. On the real served model this turns state 3's evidence from FOR (ratio 1.96) to AGAINST (0.63), and the cold-start top belief state goes from 3 to 0. The prior is computed once, inside `fit_l2_nowcaster`. It is saved beside the model as `nowcaster_class_prior`, loaded and checked by weekly, and a prior that does not match the model's classes is refused before anything is saved.

## Tasks

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 (tracer, TDD) | RED: served training prior tests | e04d68d | tests/unit/test_platform_report_serving.py, tests/unit/test_platform_report_weekly.py, tests/unit/test_platform_pooling_consumers.py |
| 1 (tracer, TDD) | GREEN: fit returns its prior; serving persists it; weekly divides by it | ff7e1c8 | driver.py, serving.py, weekly.py, tests/unit/test_platform_report_weekly.py |
| 2 (TDD) | Real-data pin on the tracked dev data | 55ee8bd | tests/unit/test_platform_report_serving.py |
| 3 | README run order, live counts 2465 -> 2472 (D-07) | b1c1fbe | README.md, CLAUDE.md |

## RED on HEAD (e04d68d, before the fix)

8 failed, 69 passed on the serving + weekly + pooling files. Each failed for the stated reason:

- `TestFitL2IsShared::test_fit_l2_nowcaster_returns_the_model_and_its_columns`: `ValueError: not enough values to unpack (expected 3, got 2)` (the fit returned 2 values).
- `TestTrainingClassPrior::test_fit_l2_nowcaster_returns_the_prior_of_the_rows_it_fit`: `ValueError: not enough values to unpack (expected 3, got 2)`.
- `TestTrainingClassPrior::test_training_class_prior_refuses_labels_that_are_not_the_models_classes`: `AttributeError: module 'trading_crab_lib.platform.backtest.driver' has no attribute 'training_class_prior'`.
- `test_a_missing_serving_artifact_names_the_command_that_builds_it[nowcaster_class_prior-...]`: `FileNotFoundError: ... nowcaster_class_prior.parquet` (the builder wrote no such artifact to delete).
- `TestTheServedClassPrior::test_the_builder_persists_the_fits_training_prior_beside_the_model`: `FileNotFoundError: Checkpoint not found: .../nowcaster_class_prior.parquet`.
- `TestTheServedClassPrior::test_weekly_divides_by_the_served_training_prior_not_the_label_prior`: `FileNotFoundError: Checkpoint not found: .../nowcaster_class_prior.parquet`.
- `TestTheServedClassPrior::test_a_class_prior_that_is_not_the_models_classes_is_refused_before_any_save`: `Failed: DID NOT RAISE ValueError` (HEAD read no prior artifact and served).
- `TestRegimeBeliefAtServe::test_cold_start_calls_the_shared_helper_and_the_tilt_gets_the_belief` (re-derived): `AssertionError: Series are different`. HEAD divided by the label prior, not the served [0.2, 0.3, 0.5].

## GREEN (ff7e1c8)

- The serving, weekly and pooling files: 77 passed.
- `test_refit_l2_equals_the_inline_backtest_recipe` passes **unmodified**. The pre-drop in `fit_l2_nowcaster` left the posterior bit-identical (`check_exact=True`).
- The backtest driver suites (`test_platform_backtest_driver.py`, `test_platform_backtest_joint_driver.py`) pass unchanged. `_refit_l2` still returns the posterior Series.
- Discriminating fixture: in the synthetic serving world, the served training prior differs from the label prior (restricted to `classes_` and renormalized) by a **max abs difference of 0.010512820512820487**. That is well above 1e-6, so no extra `_serving_world` knob was needed.

## Mutation arms (each applied, run, reverted; tree clean after each)

| Arm | Mutation | Result |
|-----|----------|--------|
| M1 | weekly passes `unconditional_belief(regime_labels)` as `class_prior` | RED: `test_weekly_divides_by_the_served_training_prior_not_the_label_prior` (`Series are different`) and the weekly cold-start test. The first form also tripped the cold-start test's call count (`assert 2 == 1`), so I ran M1b, which calls `regime_filter.unconditional_belief` past the spy. Both tests were RED on values: `Series values are different (100.0 %)`. |
| M2 | prior from the embargoed `y` BEFORE the finite-row drop | RED: `test_fit_l2_nowcaster_returns_the_prior_of_the_rows_it_fit` (`assert_array_equal`, line 136) |
| M3 | serving persists the label prior (restricted to classes_) | RED: `test_the_builder_persists_the_fits_training_prior_beside_the_model` (line 574, persisted != fit prior). The e2e precondition also fired: "the two priors differ by only 0.0". |
| M4 | `_served_class_prior` skips the class-set check | RED: `test_a_class_prior_that_is_not_the_models_classes_is_refused_before_any_save` (`DID NOT RAISE ValueError`; the superset prior would otherwise be served) |
| M5 | cold start switched to the training prior | RED: the e2e start assertion (line 617) and the weekly cold-start test. M5b failed on index names first, so I ran M5c with matching index metadata. It was RED on values: `Series values are different (100.0 %)` at line 617. |
| M6 | `advance_regime_belief` ignores `class_prior`, divides by `unconditional_belief(regime_labels)` | RED: `test_the_training_prior_flips_state_3s_evidence_and_the_served_belief`: `assert 3 == 0`, where `idxmax` of `0 0.272363 / 1 0.212738 / 2 0.066539 / 3 0.369157 / 4 0.011726 / 5 0.067476`. This is an old-prior RED, not a TypeError. |

## Real-data numbers (tracked dev data, session copy, for 08-19)

Every recorded value reproduced. None of the STOP conditions fired.

- Dev labels: 695 months, 1963-02-28 -> 2020-12-31. Counts {0: 40, 1: 228, 2: 71, 3: 200, 4: 84, 5: 72}. Label prior = counts/695.
- Serving fit: `classes_` [0, 3, 4]; training block of 153 rows, 2007-04-30 -> 2019-12-31, counts {0: 11, 3: 137, 4: 5}.
- **Training prior** {0: 11/153 = 0.0718954, 3: 137/153 = 0.8954248, 4: 5/153 = 0.0326797}.
- Served posterior (2020-12-31): [0.41800356506238856, 0.5639928698752229, 0.018003565062388593], matched at rel 1e-9, abs 0.
- State 3's likelihood ratio: **0.6298606502986066** under the training prior (evidence against); **1.9598752228163996** under the label prior (evidence for).
- **New cold-start one-step belief** (exact floats): {0: 0.29999237288846065, 1: 0.29270739835551485, 2: 0.09155165323013899, 3: 0.16323594694900967, 4: 0.059671515442087014, 5: 0.09284111313478884}. The argmax is 0.
- Old rule (label prior): {0: 0.2723628136202596, 1: 0.21273807777327436, 2: 0.06653922256344703, 3: 0.36915714570090175, 4: 0.011726345629889195, 5: 0.06747639471222798}. The argmax is 3.

## Interim state (stated)

After this plan, **only the served path** divides by the training prior. Both backtest drivers (`driver.run_backtest` via `_refit_l2`, and `joint_driver`) still divide by the whole in-window label prior. `_refit_l2` receives the triple and discards the prior (`_class_prior`). Plan 08-17 switches both drivers to the same returned prior. The phase does not close between the two plans. The `driver.py` module docstring's "Known approximation (plan 08-06)" sentence still describes the backtest drivers accurately, so it was left for 08-17.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] The weekly fixture's served prior broke three band/hysteresis scenario tests**
- **Found during:** Task 1 GREEN.
- **Issue:** The plan set `_serve_env`'s `nowcaster_class_prior` to [0.2, 0.3, 0.5] and said the other `_serve_env` tests would keep passing unmodified. They did not. With that prior the cold-start belief is (.72, .22, .06), not (.30, .45, .25). Three tests hard-code constants derived on the old belief, and their preconditions fired. `test_build_inputs_returns_the_hysteresis_output_and_main_renders_it` failed because regime 0 now clears the 0.70 act threshold. `test_the_band_suppresses_one_trade_and_allows_another_at_serve` failed because the SPY target moved to 0.609. `test_the_negative_residual_branch_fires_at_serve` failed because the TLT target moved to 0.391.
- **Fix:** Scanning for a prior that happened to satisfy the band constants would have fitted the fixture to the tests, so I did not do that. The fixture keeps the plan's [0.2, 0.3, 0.5] as its default (module constant `_SERVED_PRIOR`), and the cold-start CR-01 test uses it. `_serve_env` gained a `served_prior` keyword. The three 08-09 scenario tests pass `served_prior=_LABEL_DISTRIBUTION`: a served model whose training block had the label frequencies. The docstring says why. Their assertions are unchanged. They test the band and the hysteresis, not the prior.
- **Files modified:** tests/unit/test_platform_report_weekly.py
- **Commit:** ff7e1c8

**2. [Rule 3 - Blocking] `test_platform_pooling_consumers.py` fake checkpoint manager (outside the plan's file list)**
- **Found during:** Task 1 RED.
- **Issue:** `_FakeCheckpointManager` serves exactly the artifacts `_build_report_inputs` loads, and weekly now loads `nowcaster_class_prior`.
- **Fix:** Added a valid prior over the fake nowcaster's `classes_`, [0, 1, 2]: [0.5, 0.3, 0.2]. This is harness only; no G6 assertion was touched.
- **Commit:** e04d68d

## Threat Flags

None new. The new artifact is a parquet in the platform namespace, written by the same builder as `returns_by_regime` and `asset_returns`. Like those two (and unlike `*.pkl`), it is neither tracked nor git-ignored, so a real serving run leaves it untracked there. That was already true for the two existing serving parquets; this plan follows the same pattern and does not change it.

## Budget and fences

- Registry: `total_trial_count()` == 44; `registry/trials.jsonl` sha256 prefix c957e8fdb360. Checked before Task 1, after Task 2 and at the end.
- The serving fit stays `NO_REGISTRY`. The real-data test poisons `get_holdout_checkpoint_manager` and `load_full_span`, splits at `DEFAULT_HOLDOUT_CUTOFF` and saves nothing. `git status --porcelain -- data outputs registry` is empty.
- `git diff --quiet b7193fd -- registry/ legacy/ gsd-scratch-work/ trading-crab-lib/` holds. `git diff --stat b7193fd -- src/` lists only driver.py, serving.py and weekly.py. `joint_driver.py` and `regime_filter.py` are untouched.
- Live count **2472** at all four sites (CLAUDE.md layout tree and current status; README badge and feature list). `test_docs_recorded_counts.py` and the legacy import ratchet pass.
- ruff and flake8 (E9,F63,F7,F82) are clean on every touched Python file.
- Full suite: `pytest tests/ -q` -> **2472 passed, 5 warnings in 264.95s** (0 failed, 0 skipped, 0 xfailed). Before this plan: 2465.

## Self-Check: PASSED

- All five touched source/test files and this SUMMARY exist.
- Commits e04d68d, ff7e1c8, 55ee8bd and b1c1fbe are present in `git log`.
- Registry: 44, c957e8fdb360. `git status --porcelain -- data outputs registry` is empty.
