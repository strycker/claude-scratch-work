---
phase: 08-regime-persistence-stability
plan: 12
artifact: SERVING — facts and rulings for the weekly report's serving path
gap: G-08-2
measured: 2026-09-29
measured_at_commit: d2c742f
registry_before: 44
registry_after: 44
registry_rows_spent: 0
---

# SERVING — what the weekly report scores, measured before anyone rules on it

> **Cost first.** **registry rows spent: 0.** Every number in §1 came from one read-only Python
> process. It made direct recipe calls (`build_nowcaster_training_set`,
> `_cv_safe_active_features`, `fit_nowcaster`, `predict_proba`) and never called
> `evaluate_nowcaster`, `save`, `save_model` or `append_trial`. `total_trial_count()` read **44**
> before and after. `registry/trials.jsonl` had sha256 `c957e8fdb360f04cebd36a6ac19f7efe5c036d098152a5b0c1694f2e73c088ad` before and after.
> `git status --porcelain -- data outputs registry` was empty afterwards.

---

## 0. Standing ruling (Glenn, 2026-09-28, APPROVED)

Quoted from `08-UAT.md`, gap G-08-2, `missing` item 3:

> "Ruling (APPROVED by Glenn 2026-09-28): the serving fit is NOT a registry trial — it is written
> with the NO_REGISTRY sentinel; it refits an already-evaluated configuration on full history,
> selects nothing, and leaves total_trial_count() at 44"

- The sentinel is `registry.NO_REGISTRY` (`"__no_registry__"`, `platform/honesty/registry.py:72`).
  An append with `path=NO_REGISTRY` builds the row, logs "Registry append SKIPPED", and writes nothing.
- The configuration refit is the one the backtest evaluated: `backtest/driver.py::_refit_l2`,
  lines 306-343.
- **registry rows spent: 0**

### The data window the serving fit may read

- **The serving fit reads the DEV checkpoints only.** Features run to 2020-12-31, labels to
  2020-12-31, and training targets to 2019-12-31 after the 12-month embargo.
- **"Full labelled history" means full DEV history.** `regime_labels` ends at the holdout boundary
  (2020-12-31), so no label exists past it to train on.
- **The 2021+ rows are READ only when scoring**, never when fitting:
  - `load_full_span("monthly_features")`, the explicit "looking" opt-in (`honesty/holdout.py:77`,
    called by `report/weekly.py:441`). It is never called from a fitting path.
  - The tilt's live volatility estimate, through `asset_returns`.
  - Neither of those fits anything.

---

## 1. Facts, as measured

Measured 2026-09-29 at `d2c742f`, on the tracked checkpoints, with the config in
`config/platform_settings.yaml` (`labeling.embargo_months` = 12, `backtest.feature_min_history` = 120,
`backtest.nowcaster_cv_splits` = 5).

**Every fact matches the planner's 2026-09-28 value at `c2e632d`, and the numbers the ruling was
given on. The tracked data did not change between planning and execution.** The one place where
this record is more precise than the plan is item 7. It changes no ruled fact.

| # | Fact, as measured (denominator and window in the line) | Code path | Plan-time value |
|---|---|---|---|
| 1 | Dev `monthly_features`: **708 rows × 55 columns**, 1962-01-31 → 2020-12-31. **543 of 708** rows carry at least one NaN. | `get_platform_checkpoint_manager().load("monthly_features")` | same |
| 2 | Holdout `monthly_features`: **68 rows × 55 columns**, 2021-01-31 → 2026-08-31. | `get_holdout_checkpoint_manager().load("monthly_features")` (`honesty/holdout.py:34`) | same |
| 3 | `regime_labels`: **695 months**, 1963-02-28 → 2020-12-31, **6 states**, counts **{0: 40, 1: 228, 2: 71, 3: 200, 4: 84, 5: 72}**. Elementwise identical to `outputs/reports/platform/backtest_full_sample_states.parquet`: **0 of 695** mismatched. | `cm.load("regime_labels")["state"]` | same |
| 4 | After the **12-month** embargo, training targets run **1963-02-28 → 2019-12-31, 683 rows**; the training matrix is 683 × 55. | `prediction/nowcaster.py::build_nowcaster_training_set` (line 54), as called at `driver.py:327` | same |
| 5 | `_cv_safe_active_features` (min_history 120, n_splits 5) admits **55 of 55** columns. The block where every column is observed starts **2007-04-30**, where **HYG, UEC and UNG** start. That leaves **153 training rows, 2007-04-30 → 2019-12-31, holding 3 of 6 states: {0: 11, 3: 137, 4: 5}**. The fitted model's `classes_` are **[0, 3, 4]**. | `backtest/driver.py::_cv_safe_active_features` (line 160), called at line 333; `fit_nowcaster` (line 86) drops non-finite rows | same |
| 6 | Across **all 231 months complete in the 55 model columns**, 2007-04-30 → 2026-06-30 (contiguous, 0 missing month-ends, 776-row full span 1962-01-31 → 2026-08-31), `predict_proba` returns **1 distinct vector**: **{0: 0.418, 3: 0.564, 4: 0.018}**. Exact floats: 0.41800356506238856, 0.5639928698752229, 0.018003565062388593. `np.ptp` per class over the 231 rows is **[0.0, 0.0, 0.0]**. Measured with `np.unique(P, axis=0)`, which compares exact floats, so the count cannot come out as 1 by rounding. | the recipe of item 5, fit once, scored on `load_full_span("monthly_features")[active]` rows with no NaN | same |
| 7 | The evaluated backtest's per-step posterior (`backtest_filtered_state_probs.parquet`, **501 rows**, 1974-02-28 → 2020-12-31) emits the same value triple **0.418 / 0.564 / 0.018** at its final two steps. At **2020-11-30** it lands on the same state ids, {0: 0.418, 3: 0.564, 4: 0.018}. At **2020-12-31** it lands on ids {0: 0.418, 4: 0.564, 5: 0.018}. A near-identical triple appears at **2020-07-31** ({0: 0.419, 2: 0.562, 5: 0.019}) and **2020-09-30** ({0: 0.418, 4: 0.563, 5: 0.018}). | `evaluation/report.py:1038` writes the matrix; `driver.py::_refit_l2` produces each row | the plan said "the same triple at its final two steps" |
| 8 | The latest live row, **2026-08-31**, has NaN in **`div_yield`, `fred_m2sl`, `fred_totalsl`**, all three of them model columns. **2026-07-31** is missing `div_yield` alone. **2026-06-30** is the latest month complete in all 55 model columns. 2026-06-30 is **2 month-ends** behind the newest row. | `load_full_span("monthly_features")`, the row `weekly.py:446` scores today (`.iloc[[-1]]`) | same |
| 9 | Not re-run here. The planner's prototype ran in scratch copies of `data/` and `outputs/` via `TC_DATA_DIR` / `TC_OUTPUT_DIR`. With the three serving artifacts built and the scored row complete (2026-06-30), `weekly.main([])` completed. The report carried UAT test 2's strings: "active regime: none (neutral posture)", the A7 sentence, and "EXECUTED book after the 5.0% no-trade band". `registry/trials.jsonl` was byte-identical. | planner prototype, 2026-09-28 | — (08-14 re-measures in §3) |

### What the facts say, in plain words

- **Item 7 is precise about state ids, and it does not change the conclusion.** Each backtest step
  re-labels history with its own jump-model fit, so a state id at one step is not comparable with
  the same id at another step. The value triple is what repeats. At 2020-11-30 it repeats on the
  very ids the serving fit uses. **The serving fit is faithful to the evaluated path. The
  degeneracy belongs to the recipe at the end of history, not to the serving code.**
- **Why the posterior is constant (item 5 → item 6).** The complete block holds 153 rows. 137 of
  them are state 3. States 1, 2 and 5 are absent. The calibrated fit returns the same distribution
  for every input. That is **1 distinct vector across 231 months**.
- **Why today's row cannot be scored as it stands (item 8).** The newest row lacks three of the
  model's own columns. That is a publication lag, and it is structural, not a glitch. The logistic
  nowcaster rejects NaN, and the same row raises and **degrades** in the backtest
  (`driver.py:520-537`, `_L2_DEGRADE_EXCEPTIONS`).
- **What is not on the table here.** Changing the recipe would produce a configuration nobody
  evaluated. That includes a different feature rule, imputation, dropping the lagging columns from
  the fit, or a different calibration. Under ADR-0004 (accepted 2026-09-28) such a change needs its
  own phase's pre-declared trial budget, and Phase 8's gap closure runs at budget 0. It may be
  recorded as an open item. It may not be chosen here.
