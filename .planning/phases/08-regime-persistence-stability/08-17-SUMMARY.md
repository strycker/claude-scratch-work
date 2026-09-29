---
phase: 08-regime-persistence-stability
plan: 17
subsystem: platform/backtest (run_backtest, joint_driver), platform/prediction (regime_filter)
tags: [gap-closure, CR-01, bayes-filter, class-prior, train-serve-skew, honesty, no-registry, tdd, tracer]
status: complete
requires:
  - "08-16 (driver.fit_l2_nowcaster -> (model, active, class_prior); serve persists nowcaster_class_prior)"
provides:
  - "driver._refit_l2 -> (posterior, class_prior); run_backtest's filter divides by that prior"
  - "joint_driver._filtered_belief(prev, states, posterior, class_prior, *, state_index); each classifier passes its own step's prior"
  - "regime_filter.likelihood_ratio refuses a prior whose support is not the posterior's states, or that does not sum to 1 within 1e-9"
  - "tests: identity-spy discriminating tests in both driver suites; serve/backtest parity on the prior; refusal and no-evidence arms"
affects:
  - "08-19 (re-measures the observational l2 leg of joint_driver: curves, B1, S-1; regenerates the belief parquet the nowcaster_recursion S-1 pins read)"
tech-stack:
  added: []
  patterns:
    - "the fit returns its own training prior; every call site passes that object to the filter (identity-tested)"
    - "a wrong prior is refused, never renormalized"
key-files:
  created:
    - .planning/phases/08-regime-persistence-stability/08-17-SUMMARY.md
  modified:
    - src/trading_crab_lib/platform/backtest/driver.py
    - src/trading_crab_lib/platform/backtest/joint_driver.py
    - src/trading_crab_lib/platform/prediction/regime_filter.py
    - tests/unit/test_platform_backtest_driver.py
    - tests/unit/test_platform_backtest_joint_driver.py
    - tests/unit/test_platform_hysteresis.py
    - tests/unit/test_platform_pooling_consumers.py
    - tests/unit/test_platform_report_serving.py
    - tests/unit/test_platform_regime_filter.py
    - README.md
    - CLAUDE.md
decisions:
  - "CR-01 in the backtest: both drivers divide the posterior by the training prior fit_l2_nowcaster returned for that step's fit (via _refit_l2), the same object serve persists. No driver computes a likelihood prior from window labels."
  - "likelihood_ratio enforces the rule: after the unchanged zero-prior-present-state raise, the prior's support must equal the posterior's states and sum to 1 within 1e-9. It refuses; it does not renormalize."
metrics:
  duration: "~60 min"
  completed: 2026-09-29
estimate:
  tokens: 110000
  tasks: 3
actuals:
  tokens: 8200    # chars/4 over the realized diff (32889 chars of +/- lines, 67c9e8f..4927874, src/tests/README/CLAUDE)
  tasks: 3
  commits: 4    # 3 task commits + this SUMMARY commit
---

# Phase 8 Plan 17: CR-01 in the Backtest Drivers Summary

Both backtest drivers now divide the nowcaster's posterior by the training class prior of the fit that produced it. That is the same object `fit_l2_nowcaster` returns and serve persists (08-16). `_refit_l2` returns `(posterior, class_prior)`. `run_backtest` and each classifier in `joint_driver` pass that prior to `filter_step`. `likelihood_ratio` now refuses the old whole-window prior shape, so the defect cannot come back without a loud error. π_0 and `A` still come from the in-window labels, so there are two roles and two rules at all three call sites. The decision-bearing l1only leg does not move: its bit-for-bit pins pass unmodified, and M4 shows they still catch a filter leak.

## Tasks

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 (tracer, TDD) | `_refit_l2` returns the training prior; both drivers divide by it | 7cf5943 | driver.py, joint_driver.py, test_platform_backtest_driver.py, test_platform_backtest_joint_driver.py, test_platform_hysteresis.py, test_platform_pooling_consumers.py, test_platform_report_serving.py |
| 2 (TDD) | `likelihood_ratio` refusals; docstrings (two roles, two rules); demo valid | 668efcf | regime_filter.py, test_platform_regime_filter.py |
| 3 | Recorded counts 2472 -> 2477 (D-07) | 4927874 | README.md, CLAUDE.md |

Task 1 is a single commit. The plan calls the signature change atomic across callers and fakes, and a separate RED commit would have committed a red suite. RED was run against HEAD's source and is recorded below.

## Task 1: RED on HEAD, then GREEN

**RED (HEAD's driver.py and joint_driver.py, new tests):** 8 failed, 6 passed.
- `test_the_likelihood_divides_by_the_fits_training_prior`: FAILED. On HEAD it failed on shape first, because the fake's tuple reached `float()` (`TypeError: float() argument must be a string or a real number, not 'tuple'`).
- `test_the_likelihood_prior_is_the_step_fits_training_prior`: FAILED (`ValueError: too many values to unpack (expected 2)`; HEAD's `_refit_l2` returns a Series).
- The re-derived `test_cold_start_is_the_shared_unconditional_belief` and `test_cold_start_is_unconditional_belief_on_the_steps_own_labels`, plus both serving `_refit_l2` consumers: FAILED on shape.

The value-level RED, which is HEAD's rule applied with the new signature, is **M1** below. With the new code dividing by `unconditional_belief(states)`, the driver test fails exactly on the prior: `assert 0 0.5 / 1 0.5 (window) is 0 0.65 / 1 0.35 (fit prior)`.

**GREEN:** the six files in the Task 1 verify: 226 passed.

**Joint discriminating precondition:** on the synthetic l2 world, 104 filter calls and 104 recorded `_refit_l2` returns. The training prior differs from the restricted, renormalized window prior at **104 of 104** filtered classifier-steps, max abs gap **0.04750123701138054** (well above 1e-6).

## Task 2: RED on HEAD, then GREEN

**RED (HEAD's regime_filter.py):** 3 failed, 16 passed.
- `test_a_posterior_equal_to_the_training_prior_is_no_evidence`: its first half (ratio 1.0 everywhere, belief = normalize(π A) at atol 1e-15) PASSED on HEAD. Its second half failed with `Failed: DID NOT RAISE ValueError` on the old-shaped call.
- `test_a_prior_with_mass_on_a_state_the_posterior_lacks_is_refused`: `Failed: DID NOT RAISE ValueError`.
- `test_a_prior_that_does_not_sum_to_one_is_refused`: `Failed: DID NOT RAISE ValueError`.
- The rewritten `test_absent_state_keeps_its_prediction_share_not_zero` is GREEN on HEAD, as the plan predicted. HEAD's arithmetic is correct when it is given the right prior. This plan's RED evidence is the refusal arms.

**GREEN:** regime_filter + nowcaster_recursion + both driver suites + serving + weekly + hysteresis + pooling: 257 passed. `python -m trading_crab_lib.platform.prediction.regime_filter` exits 0. `test_platform_nowcaster_recursion.py` is **unmodified** and passes.

## Mutation arms (each applied, run, reverted; `cmp` confirmed the tree restored after each)

| Arm | Mutation | Result |
|-----|----------|--------|
| M1 | `run_backtest` passes `unconditional_belief(states, ...)` as `class_prior` | RED: `test_the_likelihood_divides_by_the_fits_training_prior` (`window 0.5/0.5 is not fit prior 0.65/0.35`). Also `test_cold_start_is_the_shared_unconditional_belief` (`assert 13 == 1` calls). |
| M2 | joint_driver passes `prior_1` to classifier #2 | RED: `test_the_likelihood_prior_is_the_step_fits_training_prior` (`class_prior 0.414634/0.585366 is 0.585366/0.414634`) |
| M3 | cold start = training prior (reindexed to K) in both drivers | RED: both cold-start tests (driver `assert 0 == 1` calls; joint `IndexError: list index out of range`, no cold-start call) |
| M4 | joint filter gate widened to `if use_regime_filter:` (l1only one-hot passed as its own prior) | RED: **`test_l1only_curve_is_bit_identical_with_the_filter_on_and_off`** (`return` values different, 100%). Also `test_band_off_l1only_curve_is_the_pre_0809_curve_bit_for_bit[key-absent]`, `[key-null]` (`'0x1.0000000000000p+0' == '0x1.e24880103f884p-1'`), and `test_filter_is_gated_on_the_l2_routing_in_code_not_by_data`. |
| M5 | delete refusal (a), the support check | RED: `test_a_prior_with_mass_on_a_state_the_posterior_lacks_is_refused`. The no-evidence test's old-shape half still raised through refusal (b), because that prior sums to 0.6 over the posterior's states. |
| M6 | renormalize the prior over the posterior's states instead of refusing | RED: `..._is_refused` (mass), `..._does_not_sum_to_one_is_refused`, `..._is_no_evidence` |
| M7 | delete refusal (b), the sum check | RED: `test_a_prior_that_does_not_sum_to_one_is_refused` |
| M8 | move the support check before the zero-prior check | RED: `TestZeroPriorWithPresentPosteriorRaises::test_raises_naming_the_state` (its message match). For this arm to bite, the support check compares the support to the posterior's states for **equality**, per the must-have. Behind the zero-prior check only the "extra mass" direction can reach it. |

## Re-derived expectations (old -> new; no tolerance widened)

| Test | Old | New | Why |
|------|-----|-----|-----|
| `test_platform_backtest_driver.py::test_cold_start_is_the_shared_unconditional_belief` | `unconditional_belief` called once per filtered step (`len(calls) == len(starts)`); step 2's start `!=` `ub(step-2 states)` | called at the cold start only (`len(calls) == 1`); every later start **is** the previous step's filter output (identity) | The class prior no longer comes from `unconditional_belief`, so it has one role (π_0). The second half is stronger: identity, not inequality. |
| `test_platform_backtest_joint_driver.py::test_cold_start_is_unconditional_belief_on_the_steps_own_labels` | `start == class_prior` | assertion dropped; the start is still `unconditional_belief(states_1)` (unchanged); the 4th argument is indexed exactly by the posterior's classes and **is** the first filtered step's `_refit_l2` prior | Two roles, two rules (08-16): π_0 over all K vs the training prior over `classes_`. |
| `test_platform_regime_filter.py::test_absent_state_keeps_its_prediction_share_not_zero` | prior c = [0.3, 0.2, 0.1, 0.4]; state 3 belief 2/7; belief [0.5, 0.3, 0.2, 0.4]/1.4 | training prior [0.5, 1/3, 1/6] over {0, 1, 2}; state 3 belief **0.4** (its prediction share); belief [0.3, 0.18, 0.12, 0.4]; abs 1e-12 as before | The 2/7 was CR-01's inflation artefact (08-REVIEW.md): present states inflated by 1/0.6. |
| `test_platform_regime_filter.py::test_output_plugs_straight_into_filter_step` | class prior = `unconditional_belief(states)` over {0, 1, 2} with a {0, 1} posterior | class prior {0: 0.6, 1: 0.4} over the posterior's classes; start unchanged; sum-to-one assertion unchanged | The old shape is now refused. |
| `test_platform_regime_filter.py::TestDeterminismAndPurity::test_no_argument_is_mutated_and_repeated_calls_are_equal` (not in the plan's list; see Deviations) | class prior [0.4, 0.4, 0.2] over {0, 1, 2}, posterior {0, 2} | {0: 2/3, 2: 1/3} (the old one restricted and renormalized) | Old shape refused. The assertions are purity and determinism only and are unchanged. |
| `test_platform_report_serving.py::test_refit_l2_equals_the_inline_backtest_recipe` | `got = _refit_l2(...)` | `got, _ = _refit_l2(...)`; equality stays `check_exact=True` | Return type. |
| `test_platform_report_serving.py::test_served_posterior_equals_refit_l2_at_full_dev_history` | posterior parity only | posterior parity (exact) **plus** persisted `nowcaster_class_prior` == `_refit_l2`'s prior (index equal, `assert_array_equal`) | Train/serve parity on the prior. |

**Fakes** (harness, no assertion depends on the chosen value except as noted):
- `_fake_refit_l2` (driver): prior {0: 0.55, 1: 0.45}.
- `_varying_fake_refit_l2`: prior {0: 0.65, 1: 0.35}, a new object per call. It is deliberately not the ~0.5/0.5 window distribution; the discriminating test asserts that precondition at every step.
- hysteresis `fake_l2`: {0: 0.5, 1: 0.5}. Its band assertions (`any held`, `any traded`, lower turnover) pass unchanged.
- pooling `_fake_refit_l2`: {0: 0.45, 1: 0.45, 2: 0.10}. The sub-floor state's ratio is 2, so it keeps its mass. The G6 pin (`TestDriverConsumer`) passes: the driver still hands `vol_targeted_tilt` the unpooled table.

## The l1only guarantees

1. `test_platform_backtest_joint_driver.py::TestRegimeFilterWiring::test_l1only_curve_is_bit_identical_with_the_filter_on_and_off`: **unmodified**, passes.
2. `test_platform_backtest_driver.py::TestRegimeFilterInRunBacktest::test_filter_off_reproduces_the_pre_change_curve_exactly` (the `_PRE_0808_CURVE_HEX` pin): **unmodified**, passes. Its fake now returns a tuple, and the posterior values are unchanged.
3. `test_band_off_l1only_curve_is_the_pre_0809_curve_bit_for_bit` (sha256 pin): unmodified, passes.
4. **M4**: widening the gate makes (1), (3) and the gate regex test red, so the pins still discriminate after this change.

The real-record proof, reproducing the committed l1only artifacts byte-identically, is 08-19's.

## Where numbers move (for 08-19)

- Moved: the observational l2 leg of `joint_driver` (curves, B1, S-1), and `run_backtest` with the tilt on.
- Not moved: l1only, which never calls `_refit_l2` and has the filter gated off.
- No artifact was regenerated. The committed belief parquet the nowcaster_recursion S-1 pins read still carries the pre-08-17 numbers, and 08-19 regenerates it.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] `TestDeterminismAndPurity::test_no_argument_is_mutated_and_repeated_calls_are_equal` used the old prior shape**
- **Found during:** Task 2 GREEN.
- **Issue:** It passed a 3-state prior with a {0, 2} posterior, the shape the new refusal rejects. It was not in the plan's read_first list.
- **Fix:** The prior is now {0: 2/3, 2: 1/3}. The purity and determinism assertions are unchanged.
- **Commit:** 668efcf

**2. [Rule 2 - Correctness] The support check tests equality, not only "extra mass"**
- **Found during:** Task 2, planning M8.
- **Issue:** An "extra states only" check placed before the zero-prior check would not fire on `test_raises_naming_the_state`'s input, so M8 could not fail. The must-have says the support must **equal** the posterior's states.
- **Fix:** The check is `support != set(posterior states)`, and the message names the extra and any missing states. Behind the zero-prior check, only the extra direction is reachable.
- **Commit:** 668efcf

**3. [Process] Task 1 source was drafted before its tests**
- The driver edits were written first and then set aside (copied to the scratchpad, HEAD's files restored via `git checkout -- <file>`). RED was run on HEAD's source, and the edits were restored before GREEN. Nothing was committed out of order.

## Budget and fences

- Registry: `total_trial_count()` == **44**; `registry/trials.jsonl` sha256 prefix **c957e8fdb360**. Checked at start and after Task 3.
- `git status --porcelain -- data outputs registry` is empty. No real-data run, no artifact regenerated, no holdout read.
- `git diff --stat b7193fd -- legacy/ gsd-scratch-work/ trading-crab-lib/ registry/` is empty.
- Live count **2477** (was 2472: +2 Task 1 tests, +3 Task 2 tests) at all four sites. `test_docs_recorded_counts.py`: 7 passed.
- Legacy import ratchet: 11 passed (`MAX_LEGACY_IMPORT_SITES = 31`).
- ruff and flake8 (E9,F63,F7,F82) are clean on all nine touched Python files.
- Full suite: `pytest tests/ -q` -> **2477 passed, 5 warnings in 285.18s** (0 failed, 0 skipped, 0 xfailed).

## Self-Check: PASSED

- All nine touched source/test files and this SUMMARY exist.
- Commits 7cf5943, 668efcf and 4927874 are present in `git log`.
- Registry 44 / c957e8fdb360; data/outputs/registry clean.
