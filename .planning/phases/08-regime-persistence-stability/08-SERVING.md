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

---

## 2. Rulings

**Who, when, and on what.** Glenn ruled on **2026-09-29**, before plan 08-12 executed, through the
orchestrator's AskUserQuestion (`08-12-PLAN.md`, Task 2, `<ruling-received>`). He ruled on the
orchestrator's independently reproduced numbers:
- the training block: 153 rows, 2007-04-30 → 2019-12-31, classes {0: 11, 3: 137, 4: 5};
- 1 distinct posterior, {0: 0.418, 3: 0.564, 4: 0.018}, across 231 scorable months, 2007-04-30 →
  2026-06-30;
- the latest row: 2026-08-31, missing `fred_m2sl`, `fred_totalsl` and `div_yield`.

§1 re-measured every one of those numbers and **all of them match** (items 5, 6 and 8). The
rulings therefore stand on the facts as measured, not only on the plan's copies.

Both rulings are options from the plan's own list. **Neither implies a recipe change**, so no
ADR-0002-amendment item is raised.

### 2.1 Q1 (the ragged edge) → **q1-c**, latest complete month, with a 3-month staleness cap

**Glenn's words, verbatim (the option he chose):**

> "(c) Latest complete month (Recommended) — Score the newest month where every model column is
> present (2026-06-30 today). The page states the as-of date and names the missing columns. I'd add
> a staleness cap: fail if the scored month is more than 3 months old. Uses only older data, so no
> look-ahead."

**The staleness cap is part of the ruling.** Planned implementation, as the orchestrator relayed
it: raise `ValueError` **before any save** when the scored month is more than 3 month-ends behind
the newest row of `monthly_features`.

**The orchestrator's interpretation, flagged for Glenn's confirmation.** Glenn wrote "more than 3
months old". The orchestrator reads that as **"more than 3 month-ends behind the newest
`monthly_features` row"**, which measures age against the data, **not against the wall clock**.
- Under this reading today's case is 2026-06-30 against 2026-08-31, which is **2 month-ends**, so
  the report serves.
- A wall-clock reading would count from the run date instead. Today's run date is 2026-09-29, and
  counted in calendar months (September 2026 back to June 2026) that is **3**. That sits exactly at
  the cap: it serves only because the ruling says "more than 3". On any run after 2026-09-30 it
  would fail until July's data completes.
- **Glenn: please confirm the data-relative reading.** If you meant the wall clock, it is a
  one-line change at the cap's single site in 08-14.

**What this ruling licenses 08-14 to implement:**
- Score the newest row of `load_full_span("monthly_features")` in which **every** model column
  (`feature_names_in_`) is observed.
- Print the scored as-of month prominently, and name the model columns missing from each newer row.
- Step the belief (`advance_regime_belief`) and the no-trade band on the **scored** month, not the
  newest row. A re-run within the same scored month then neither double-counts nor compounds.
- Apply the staleness cap: `ValueError` before any save, naming the scored month, the newest row
  and the gap.

**What it forbids:**
- **Imputation** of any kind: forward-fill, interpolation, or a fill from another series. Every
  scored value must have been observed.
- **A recipe change**: dropping the lagging columns from the fit, changing the feature rule, or
  changing the calibration. Those are new configurations under ADR-0004 (budget 0 here).
- Scoring the newest row when it is incomplete, and scoring an older month silently.

**Consequence on today's data (§1 item 8):**
- The report scores **2026-06-30**. That is 2 month-ends behind the newest row, 2026-08-31, so it
  is inside the cap.
- The page names `div_yield` as missing from 2026-07-31, and `div_yield`, `fred_m2sl` and
  `fred_totalsl` as missing from 2026-08-31.
- G-08-2's truth ("the weekly report runs end to end") becomes reachable on real data. 08-14's §3
  measures whether it is actually met.

**A stated difference between train and serve, not a hidden one.** The evaluated backtest scored
month *t* with complete month-*t* data and never modelled publication lag. Serving guidance now
runs on information 2 month-ends older than the backtest assumed (up to 3 under the cap). This is
a serve-time behaviour the backtest did not evaluate. It selects nothing and costs no trial, but it
is a known difference. Look-ahead is not possible under it, because only older, observed data is
scored.

- **registry rows spent: 0**

### 2.2 Q2 (the input-independent posterior) → **q2-ii**, disclose on the page

**Glenn's words, verbatim (the option he chose):**

> "(ii) Disclose on page (Recommended) — The report runs, and under the regime distribution it
> prints the exact count of distinct outputs ('1 distinct vector across 231 complete months') so
> the page never passes a constant off as a live signal. The recipe fix is logged as a Phase 9
> item."

**What this ruling licenses 08-14 to implement:**
- At serve time, `weekly` computes `np.unique(P, axis=0)` over the posteriors of **every**
  full-span month complete in the model columns. The comparison is exact, with no rounding and no
  threshold.
- It prints the count, the number of complete months and their window **directly under** "Current
  Regime Distribution".
- When the count is 1, it says in plain words that the distribution does not depend on the
  features.
- This is "looking", not fitting: scoring rows of `load_full_span` with the already-fitted serving
  model.

**What it forbids:**
- **Withholding the report** on the count (that is q2-iii, not chosen).
- **A tolerance or threshold** on "distinct": the count is exact.
- **Imputation**, and **any recipe change** meant to make the count exceed 1. The recipe fix is a
  **Phase 9 item**, under that phase's own pre-declared ADR-0004 trial budget.

**Consequence on today's data (§1 items 5 and 6):**
- The report runs, and shows {0: 0.418, 3: 0.564, 4: 0.018} as the distribution.
- Directly beneath it, it prints **1 distinct vector across 231 complete months
  (2007-04-30 → 2026-06-30)**.
- The disclosure is correct automatically if a future recipe fixes the degeneracy: the count then
  rises above 1.

**Open item (for STATE, owned by the orchestrator):** Phase 9, the recipe fix for the
input-independent serving posterior. Its root cause is the 153-row complete block holding 3 of 6
states ({0: 11, 3: 137, 4: 5}), which is truncated by HYG, UEC and UNG starting 2007-04-30. Any fix
is a new configuration under ADR-0004.

- **registry rows spent: 0**

### 2.3 Order of implementation

- **08-13** lands the conservative interim behaviour first: fail loudly, naming the missing
  columns.
- **08-14** then implements exactly §2.1 and §2.2, and nothing else.

---

## 3. Post-implementation measurement, real tracked data (plan 08-14)

> **Cost first.** **registry rows spent: 0.** `registry/trials.jsonl` had sha256
> `c957e8fdb360f04cebd36a6ac19f7efe5c036d098152a5b0c1694f2e73c088ad` before and after every run
> below, and `total_trial_count()` read **44** before and after. The runs wrote only to scratch
> copies; `git status --porcelain -- data outputs registry` was empty afterwards.

Measured 2026-09-29, with 08-14's two rulings implemented (commits `efbd262` for q1-c and
`eed8057` for q2-ii). `serving.py` and `driver.py` are unchanged since 08-13, so there is no
recipe change.

### 3.1 Commands run

`data/checkpoints` and `data/holdout` were copied, as tracked, into a scratch directory
(`<scratch>/data`), so every checkpoint started exactly as committed. There was no prior belief,
hysteresis or executed-book state (a cold start).

```bash
export TC_DATA_DIR=<scratch>/data TC_OUTPUT_DIR=<scratch>/out
python -m trading_crab_lib.platform.report.serving      # exit 0
python -m trading_crab_lib.platform.report.weekly       # exit 0 (run 1)
python -m trading_crab_lib.platform.report.weekly       # exit 0 (run 2, same month)
```

`python scripts/build_platform_data.py` was **not** run: it needs the network and a FRED key, and
running it would have measured a different dataset from the one §1 and §2 describe. The
measurement is therefore on the tracked data as committed. A fresh build on the Mac may move the
ragged edge (see §3.4).

### 3.2 What the run produced

| Measurement | Value |
|---|---|
| Report produced? | **Yes.** Both weekly runs exited 0 and wrote `weekly_report.md`. |
| As-of month (scored) | **2026-06-30**, 2 month-ends behind the newest row (2026-08-31). Inside the cap `MAX_SCORING_LAG_MONTHS = 3`. |
| Lagging columns named on the page | 2026-07-31 lacks `div_yield`; 2026-08-31 lacks `fred_m2sl`, `fred_totalsl`, `div_yield`. The same as §1 item 8. |
| Served distribution | regime 3: 56.4%, regime 0: 41.8%, regime 4: 1.8%. The same as §1 item 6. |
| Distinct-posterior count and window | **1 distinct posterior vector across 231 complete months (2007-04-30 → 2026-06-30)**, with the does-not-depend sentence. The same as §1 item 6. |
| Filtered belief (cold start, one filter step, as-of 2026-06-30) | 3: 36.9%, 0: 27.2%, 1: 21.3%, 5: 6.7%, 2: 6.7%, 4: 1.2%. The checkpoint's `as_of` is 2026-06-30. |
| Active regime | none (neutral posture). No belief component clears the 0.70 act threshold. |
| Executed book (`executed_weights`, basis `executed`, as-of 2026-06-30) | TLT 0.390654, SPY 0.330723, USO 0.144729, IAU 0.133895; cash residual 0.0%. |
| Second run, same month | The `executed_weights` frame was equal under `assert_frame_equal`, and `weekly_report.md` was **byte-identical** (`cmp`) to run 1's. The belief was reused unchanged (same scored month), and the band re-banded against the same held book. |
| Registry | sha256 `c957e8fdb360...` before and after; `total_trial_count()` 44 before and after. |

The real-data report's key lines, verbatim:

> Scored as of 2026-06-30, the latest month observed in all 55 of the nowcaster's model columns (2 month-ends behind the newest row). Newer rows lack model columns: 2026-07-31 lacks div_yield; 2026-08-31 lacks fred_m2sl, fred_totalsl, div_yield. Nothing is imputed.

> 1 distinct posterior vector across 231 complete months (2007-04-30 → 2026-06-30): the served model scored on every month observed in all 55 model columns, compared exactly (no rounding, no threshold). The distribution above does not depend on the features: it is the same every week.

> - active regime: none (neutral posture)

> The active regime is a reported label: it selects the trajectory and per-asset rows below and gates no weight (audit item A7, 08-A7.md). The weights come from the filtered belief through the no-trade band.

> Targets below are the EXECUTED book after the 5.0% no-trade band (design §5.3 bounded turnover, 08-A7.md): an asset whose target moved by no more than the band from its last executed weight keeps that weight.

Each of UAT test 2's expected strings is present: "active regime: ... none (neutral posture)", the
A7 sentence, and "Targets below are the EXECUTED book after the 5.0% no-trade band". The second
run in the same month gives the same targets.

### 3.3 Each ruling's stated consequence, answered

- **§2.1 (q1-c).** Predicted: the report scores 2026-06-30, 2 month-ends inside the cap, and names
  `div_yield` for 2026-07-31 and all three columns for 2026-08-31. **Measured: exactly that.**
- **§2.2 (q2-ii).** Predicted: the report runs, shows {0: 0.418, 3: 0.564, 4: 0.018}, and prints
  "1 distinct vector across 231 complete months (2007-04-30 → 2026-06-30)" beneath it.
  **Measured: exactly that**, with the does-not-depend sentence.

### 3.4 Status

**G-08-2: CLOSED on real data: the supported commands produce the weekly report on the tracked data as of 2026-06-30** (the scored month; the newest row is 2026-08-31, and the run date is 2026-09-29).

What this does and does not cover:
- It covers steps 2 and 3 of the documented run order on the data as committed. Step 1 was not
  re-run here (§3.1). The Mac re-test of UAT test 2 runs all three.
- It does **not** make the guidance input-dependent. The page now says so: 1 distinct vector,
  and the distribution does not depend on the features. The recipe fix is the Phase 9 item §2.2
  records, under that phase's own ADR-0004 budget.
- **The staleness cap uses the data-relative reading** (month-ends behind the newest
  `monthly_features` row), as the orchestrator relayed it. §2.1's request for Glenn to confirm
  that reading still stands. Under a wall-clock reading today's run (2026-09-29, scoring June)
  sits exactly at 3 and still serves. From 2026-10-01 it would refuse until July's `div_yield`
  arrives. The cap lives at one site, `weekly.MAX_SCORING_LAG_MONTHS` and its one comparison in
  `_scored_row`.
- **The train/serve difference §2.1 declared is now live.** Serving guidance runs on 2026-06-30
  information, 2 month-ends older than the backtest ever assumed.
- **The page shows no per-account trade rows**, because `config/platform_settings.yaml` configures
  no `report.accounts`. The executed book is in the `executed_weights` checkpoint (tabled above)
  but is not listed on the page. This predates 08-14 and is recorded, not changed.

- **registry rows spent: 0**
