---
phase: 08-regime-persistence-stability
plan: 13
subsystem: platform/report (serving path), platform/backtest (L2 fit), platform/assets
tags: [gap-closure, G-08-2, serving, train-serve-skew, honesty, no-registry, tdd, tracer]
status: complete
requires:
  - "08-11 (portable criterion-7 re-derivation; recorded count 2433)"
  - "08-12 (08-SERVING.md §1 measured facts, §2 rulings q1-c / q2-ii)"
  - "08-UAT.md G-08-2 ruling (2026-09-28): the serving fit is NOT a registry trial"
provides:
  - "src/trading_crab_lib/platform/report/serving.py: SERVING_BUILD_COMMAND, SERVING_TRIAL_TAG, build_serving_artifacts, main"
  - "backtest/driver.py::fit_l2_nowcaster (the ONE L2 fit recipe; _refit_l2 delegates to it)"
  - "assets/returns.py::tradable_asset_returns (the one research-to-tradable mapping; evaluation/report.py delegates to it)"
  - "weekly._build_report_inputs scores feature_names_in_ only; named FileNotFoundError / ValueError refusals before any state save"
affects:
  - "08-14 (replaces the interim NaN refusal with q1-c + q2-ii; owns the real-data runs)"
  - "Phase 9 (recipe fix for the input-independent served posterior)"
tech-stack:
  added: []
  patterns:
    - "one fit function, two callers (backtest step + serving) as the train/serve-skew guarantee"
    - "NO_REGISTRY hard-coded at the call site, with a spy asserting `path is NO_REGISTRY`"
    - "holdout entry points poisoned + AST scan of alias/Name/Attribute nodes"
key-files:
  created:
    - src/trading_crab_lib/platform/report/serving.py
    - tests/unit/test_platform_report_serving.py
  modified:
    - src/trading_crab_lib/platform/backtest/driver.py
    - src/trading_crab_lib/platform/assets/returns.py
    - src/trading_crab_lib/platform/evaluation/report.py
    - src/trading_crab_lib/platform/report/weekly.py
    - tests/unit/test_platform_report_weekly.py
    - tests/unit/test_platform_pooling_consumers.py
    - README.md
    - CLAUDE.md
    - .gitignore
decisions:
  - "The serving nowcaster is fit by driver.fit_l2_nowcaster, extracted verbatim from _refit_l2 (same signature for _refit_l2; joint_driver and every monkeypatch of driver._refit_l2 unaffected)"
  - "Serving calls registry.append_trial exactly once with path=registry.NO_REGISTRY hard-coded (no parameter); metrics carry only counts and the training window, never accuracy"
  - "weekly scores monthly_features.iloc[[-1]][feature_names_in_]; a NaN in a model column raises ValueError naming columns, row date and latest complete month, before any state load or save (interim until 08-14)"
  - "nowcaster.pkl in the tracked platform namespace is git-ignored (data/checkpoints/platform/*.pkl, P27)"
metrics:
  duration: "~55 min"
  completed: 2026-09-29
estimate:
  tokens: 105000
  tasks: 3
actuals:
  tokens: 11100
  tasks: 3
  commits: 4
---

# Phase 8 Plan 13: Serving builder and weekly own-column scoring (G-08-2) Summary

**What changed.** A supported command, `python -m trading_crab_lib.platform.report.serving`, now
builds all three artifacts the weekly report reads (`nowcaster.pkl`, `returns_by_regime`,
`asset_returns`). It fits the model through `driver.fit_l2_nowcaster`, the same function the
backtest's `_refit_l2` now calls. It reads DEV data only and writes no registry row.

**What weekly scores.** Weekly now scores only the model's own columns. A missing artifact names
the command that builds it. A NaN in a model column fails loudly, naming the columns, and writes
no state.

**Where G-08-2 stands.** Closed on synthetic data, end to end, against a real fitted model. On
today's real data weekly still refuses, by design and by name, until 08-14 applies ruling q1-c.

## Tasks and commits

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 (tracer) | Shared L2 fit, tradable mapping, serving builder, e2e | `8547f25` | driver.py, returns.py, evaluation/report.py, report/serving.py, test_platform_report_serving.py |
| 2 | weekly scores its model's columns; named refusals | `9ae5e99` | weekly.py, test_platform_report_serving.py, test_platform_report_weekly.py |
| 2 (follow-up) | G6 weekly-arm harness declares `feature_names_in_` | `41ad502` | test_platform_pooling_consumers.py |
| 3 | Run order documented, `*.pkl` ignored, counts 2433 -> 2451 | `1d89ce5` | weekly.py (docstring), README.md, CLAUDE.md, .gitignore |

## RED / GREEN record

### Task 1

- **Characterization pin** `TestFitL2IsShared::test_refit_l2_equals_the_inline_backtest_recipe`:
  - GREEN against the UNMODIFIED `driver.py`: `1 passed` (the other 9 new tests failed, as expected).
  - GREEN after the extraction: `3 passed` (`-k FitL2`).
- **Tracer e2e**, before `serving.py` existed. RED on all three `TestServingEndToEnd` tests:
  `ImportError: cannot import name 'serving' from 'trading_crab_lib.platform.report'`. The
  `TestTradableAssetReturns` tests were RED on the missing `tradable_asset_returns`, and the two
  new `TestFitL2IsShared` tests on the missing `driver.fit_l2_nowcaster`.
- **After implementation:** `10 passed`. One intermediate RED was a fixture artifact, not a
  product defect (see Deviation 3).
- **Task 1 verify block:**
  - driver, joint-driver, hysteresis, pooling, evaluation-report, ratchet and serving suites:
    `197 passed`;
  - AST import check: "serving.py imports only trading_crab_lib.platform.*";
  - `git diff --quiet c2e632d -- registry/ scripts/run_joint_lift.py .../nowcaster.py`: untouched.
- **Tracer feedback gate:** the tracer verify was re-run after the commit, `10 passed`, before any
  expansion task.

### Task 2 (RED lines against the pre-change weekly.py)

- `test_report_scores_the_models_columns_not_the_whole_row`: RED.
  `ValueError: The feature names should match those that were passed during fit. Feature names
  unseen at fit time: - short_hist - starver`.
- `test_a_nan_in_a_model_column_fails_loudly_naming_it`: RED.
  `AssertionError: assert 'f_slope' in 'Input X contains NaN.\nLogisticRegression does not accept
  missing values encoded as NaN natively. ...'`.
- `test_a_missing_serving_artifact_names_the_command_that_builds_it`: RED ×3.
  - `[nowcaster]`: `assert 'python -m trading_crab_lib.platform.report.serving' in 'Model
    checkpoint not found: .../platform/nowcaster.pkl'`.
  - `[returns_by_regime]`: the same assertion against `'Checkpoint not found:
    .../returns_by_regime.parquet'`.
  - `[asset_returns]`: the same assertion against `'Checkpoint not found:
    .../asset_returns.parquet'`.
- GREEN from the moment they were written, as the plan expected, because the shared helper
  already existed: parity, no-skew (`check_exact=True`) and the same-month rerun.
- **After the weekly change:** serving + weekly suites `52 passed`.

## Mutation arms (every new check shown able to fail)

| Check | Mutation | Result |
|---|---|---|
| Characterization pin | `fit_l2_nowcaster` builds the training set with `embargo_months=0` | RED: `AssertionError: Series are different`. Reverted, GREEN. |
| Parity (all-columns, the plan's arm) | serving fits `fit_nowcaster` on every column of the embargoed set | RED: `ValueError: Requesting 3-fold cross-validation but provided less than 3 examples for at least one class.` (the builder cannot even fit). Reverted. |
| Parity (added arm: reaches the column assertion) | serving fits the right columns in REVERSED order | RED: `AssertionError: assert ['f_slope', 'f_level'] == ['f_level', 'f_slope']`. Reverted, GREEN. |
| Discriminating e2e | pre-fix whole-row scorer (the unmodified weekly) | RED, quoted above. The test's own precondition asserts that `predict_proba(full_span.iloc[[-1]])` raises on this fixture. |
| Parity preconditions | (built in) | These assert the window rule excluded `short_hist`, `starver` was old enough, the CV narrowing then excluded `starver`, and `expected != list(X.columns)`. Both exclusion rules fired. |
| NaN-names-it, missing-artifact | the unmodified weekly | RED, quoted above. |
| No-skew | (built in) | Exact `check_exact=True` equality with `_refit_l2` at full dev history. Any change of labels, embargo or columns breaks it. |
| NO_REGISTRY | (built in) | Three independent paths: the spy (`len(calls) == 1 and calls[0]["path"] is registry.NO_REGISTRY`), the tmp copy's bytes and trial count, and the real ledger's sha256. |
| No-holdout | (built in) | `get_holdout_checkpoint_manager` and `load_full_span` are poisoned to raise, and an AST scan covers alias, Name and Attribute nodes. |
| Posterior varies | (built in) | `np.unique(..., axis=0).shape[0] > 1` over the world's complete rows, so a degenerate fixture model fails the e2e. |

## Behavior-preservation evidence for the extraction

1. **The characterization pin** was GREEN before and after the extraction. It compares
   `_refit_l2` exactly with the inline recipe. The `embargo_months=0` mutation turns it RED.
2. **Real dev data, before and after.** Before touching `driver.py`, I scored `_refit_l2` on the
   real dev checkpoints (read-only) at four decision dates: 1995-06-30, 2008-09-30, 2015-03-31
   and 2020-12-31. Each used the training slice strictly before t and `regime_labels`, and I
   pickled the Series to the scratchpad. After the extraction I re-ran the same script.
   **All four Series were IDENTICAL under `assert_series_equal(check_exact=True)`**: 4, 5, 3 and
   3 classes respectively. The 2020-12-31 step gives {0: 0.41818, 3: 0.56364, 4: 0.01818}.
3. **Every existing test stayed green, unmodified:** driver, joint-driver (which imports
   `_refit_l2` by name), hysteresis, pooling and evaluation-report. The decision-bearing
   criterion-7 record tests are included; none of those test files was edited.
   `run_backtest` still calls `_refit_l2` by its module-level name, so every existing
   monkeypatch of `driver._refit_l2` still bites.
4. **The real-data serving model reproduces 08-12 §1 item 6 to the bit:** 231 complete months,
   2007-04-30 to 2026-06-30, 1 distinct vector (0.41800356506238856, 0.5639928698752229,
   0.018003565062388593).

## Frames the builder reads (no 2021+ data used for fitting)

- **`monthly_features`**, through the DEV manager (`get_platform_checkpoint_manager()`), then
  `split_by_holdout_boundary(..., cutoff="2020-12-31")`, keeping the dev side. This frame is
  what gets fit.
- **`regime_labels["state"]`**, through the DEV manager; it ends 2020-12-31. These are the
  training targets, which end 2019-12-31 under the 12-month embargo.
- **`monthly_raw`**, through the DEV manager; it spans 1962-01 to 2026-08. It feeds
  `asset_returns` for the tilt's live volatility estimate. This is looking, not fitting, and it
  selects nothing. `returns_by_regime` joins the dev side of those returns to the dev labels.
- **Never read:** `load_full_span`, `get_holdout_checkpoint_manager`, `HOLDOUT_CHECKPOINT_DIR`.
  The poison test and the AST scan enforce this.

## Real-data run (scratch copies only; Task 3 step 4)

`data/checkpoints` and `data/holdout` were copied to the scratchpad and run with
`TC_DATA_DIR=<scratch>/data TC_OUTPUT_DIR=<scratch>/out`.

**Builder**, `python -m trading_crab_lib.platform.report.serving`: exit 0. The returned facts
match 08-12 §1 exactly:

| Fact | Measured | 08-12 expected |
|---|---|---|
| model columns | 55 | 55 |
| training rows / window | 153, 2007-04-30 → 2019-12-31 | same |
| classes / counts | [0, 3, 4] / {0: 11, 3: 137, 4: 5} | same |
| `n_distinct_posteriors_dev` | 1 (across 165 complete dev months) | 1 |
| `asset_returns` | SPY, TLT, IAU, USO; 1962-01-31 → 2026-08-31 (776 months) | — |
| `returns_by_regime` rows | 24 | — |
| `registry_row_written` | False | False |

**Weekly**, `python -m trading_crab_lib.platform.report.weekly`: exit 1, the interim refusal, verbatim:

> ValueError: The latest monthly_features row (2026-08-31) has NaN in 3 of the nowcaster's 55 model columns: ['fred_m2sl', 'fred_totalsl', 'div_yield']. The latest month observed in every model column is 2026-06-30. The report does not impute, so it refuses to score this row (interim behaviour until plan 08-14 applies the 08-12 ruling).

The scratch dirs held no `weekly_report.md` and none of `regime_belief`, `hysteresis_state` or
`executed_weights`: the refusal preceded every save.

## Registry (ADR-0004 gap-closure budget 0)

- Before: sha256 `c957e8fdb360f04cebd36a6ac19f7efe5c036d098152a5b0c1694f2e73c088ad`,
  `total_trial_count()` **44**.
- After, at plan end: sha256 `c957e8fdb360f04cebd36a6ac19f7efe5c036d098152a5b0c1694f2e73c088ad`,
  `total_trial_count()` **44**.
- `git diff --stat c2e632d -- registry/ legacy/ gsd-scratch-work/ trading-crab-lib/
  scripts/run_joint_lift.py src/trading_crab_lib/platform/prediction/nowcaster.py` is empty.
- `git status --porcelain -- data outputs registry` is empty.

## Counts and ratchet

- The live collection went from **2433** to **2451** (+18, all in
  `tests/unit/test_platform_report_serving.py`). It is written at all four sites:
  `CLAUDE.md` layout tree and current status; `README.md` badge and feature list.
  `test_docs_recorded_counts.py`: `7 passed`.
- Full suite: `2451 passed, 5 warnings`, with 0 failed, 0 skipped and 0 xfailed.
- Legacy-import ratchet: `11 passed`, still at **31**. `serving.py` imports only
  `trading_crab_lib.platform.*`, pandas, numpy and the stdlib, and `cm` is left loosely typed.
- ruff is clean on every touched source and test file. flake8 (E9,F63,F7,F82) is clean on
  serving.py and weekly.py.

## Deviations from Plan

### Auto-fixed issues

**1. [Rule 1 - Bug] The G6 weekly-arm harness broke under own-column scoring.**
- **Found during:** the Task 3 full-suite run.
- **Issue:** `test_platform_pooling_consumers.py::TestWeeklyConsumer` feeds weekly a
  `_FakeNowcaster` with no `feature_names_in_`. The new guard correctly refused it:
  "declares no feature_names_in_". The plan listed the analogous fix only for
  `test_platform_report_weekly.py`.
- **Fix:** the fake declares `feature_names_in_ = np.array(["SPY", "TLT"])`, the columns its
  `load_full_span` stub returns. This is harness only; no G6 assertion was touched.
- **Commit:** `41ad502`.

**2. [Rule 1 - Test correctness] The trial-count assertion compares before with after, not a literal 44.**
- **Issue:** the e2e test copies the REAL ledger to tmp. Hard-coding `== 44` there would turn red
  on the first legitimate Phase 9 trial, a false failure unrelated to serving.
- **Fix:** the test asserts that the tmp copy's count equals the real ledger's before the run and
  is unchanged after it, that the copy is byte-identical, and that the real sha256 is unchanged.
  The literal 44 was verified live, before and after, in the Task 3 real-data run.

**3. [Rule 1 - Test artifact] The no-accuracy scan matched its own tmp path.**
- **Issue:** `tmp_path` is named after the test (`test_no_accuracy_is_computed...`).
  `report_returns_by_regime` logs the artifact path, so the `accura` scan matched the test's own
  name.
- **Fix:** that test uses `tmp_path_factory.mktemp("world")`, with a comment explaining why.

**4. [Rule 2 - Non-vacuity] The same-month rerun test names one account.**
- **Issue:** with `report.accounts: []` the "Trades Implied" section carries no per-asset rows,
  so comparing the two runs' sections would compare two nearly empty strings.
- **Fix:** this test sets one account with no holdings file. Holdings treat it as neutral and
  all-cash; it never crashes. The test asserts the section lists SPY and TLT.

**5. [Rule 1 - Precision] The record-only training block uses `fit_nowcaster`'s finite-row rule.**
- The plan said `dropna`; `_training_block` uses `np.isfinite(...).all(axis=1)`, which is exactly
  the rule the fit applies. The two are identical in the absence of inf.

**6. An extra parity mutation (reversed column order).**
- The plan's all-columns mutation turns parity RED, but through the builder's own fit
  ValueError, before the column assertion runs. The added arm fits cleanly on a wrong column
  list and proves the assertion itself bites.

**7. The tracer gate ran in autonomous form.**
- `workflow.auto_advance` and `_auto_chain_active` read `false`. The plan is
  `autonomous: true`, and the orchestrator asked for a full run and a completion report, so I
  applied the automated gate instead of stopping for human-verify: re-run the tracer verify
  after the commit (`10 passed`) and halt on failure.

**8. Small additions beyond the plan.**
- `TestFitL2IsShared::test_refit_l2_delegates_to_the_shared_fit` asserts that `_refit_l2`
  calls `fit_l2_nowcaster` by module name. It guards against a future re-inlining.
- The builder returns the `class_counts` and the `asset_returns` columns in its facts dict.
- It logs a WARNING for excluded research classes, as `evaluation/report.py` does.

## Known Stubs

None.

## Threat Flags

None beyond the plan's register.
- T-08-70 through T-08-75 are mitigated as specified.
- `serving.py` adds a CLI that writes a joblib pickle into the platform namespace (T-08-74); it
  is covered by the new `.gitignore` rule.

## Open items for STATE

1. **`evaluate_nowcaster` is untouched.** It still writes `nowcaster.pkl` and appends a registry
   trial if anyone calls it. If called, it would overwrite the served model with a
   differently-fit one (the full embargoed set, not the `_cv_safe_active_features` columns),
   and it would spend a trial.
2. **`scripts/run_joint_lift.py` keeps its own copy of the research-to-tradable mapping.** It is
   untouched because it is the decision-bearing harness. It is now the only duplicate of
   `tradable_asset_returns`.
3. **The served posterior is input-independent on real data**: 1 distinct vector across 231
   complete months. That is the recipe's property, ruled on in 08-12 (q2-ii: disclose it). The
   recipe fix is a Phase 9 item.
4. **Weekly refuses on today's real data by design.** 08-14 implements q1-c (the latest complete
   month plus the 3-month staleness cap) and q2-ii (the distinct-count disclosure), and owns the
   real-data runs.
5. **More untracked files in the tracked namespace (new observation).** A real serving run also
   leaves `asset_returns.parquet`, `returns_by_regime.parquet` and their `.meta.json` files
   untracked and unignored in `data/checkpoints/platform/`. Weekly's `regime_belief`,
   `hysteresis_state` and `executed_weights` checkpoints are in the same state already. Only the
   pickles are ignored, per the plan; the parquets are not a code-execution risk but are one
   `git add data/` away from being committed.
6. **Heavier import (new observation).** `weekly.py` now imports `report/serving.py` for
   `SERVING_BUILD_COMMAND`, which pulls in the backtest driver and sklearn at weekly import time.
   This is functionally harmless and noted for anyone optimising import cost.

## Self-Check: PASSED

- FOUND: src/trading_crab_lib/platform/report/serving.py
- FOUND: tests/unit/test_platform_report_serving.py
- FOUND: commits 8547f25, 9ae5e99, 41ad502, 1d89ce5
