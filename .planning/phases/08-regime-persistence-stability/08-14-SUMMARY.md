---
phase: 08-regime-persistence-stability
plan: 14
subsystem: platform/report (weekly serving path)
tags: [gap-closure, G-08-2, serving, ragged-edge, input-independence, honesty, no-registry, tdd, tracer]
status: complete
requires:
  - "08-12 (08-SERVING.md §1 facts; §2 rulings q1-c with the 3-month staleness cap, and q2-ii)"
  - "08-13 (serving builder, own-column scoring, interim NaN refusal; recorded count 2451)"
provides:
  - "weekly.MAX_SCORING_LAG_MONTHS = 3 (data-relative staleness cap, ruling q1-c)"
  - "weekly._model_columns / _scored_row: scores the latest month complete in the model columns; 'Scored as of' note"
  - "weekly._input_sensitivity_note: exact np.unique distinct-posterior count across complete full-span months (ruling q2-ii)"
  - "assemble_weekly_report keywords scored_as_of_note, input_sensitivity_note"
  - "08-SERVING.md §3: post-implementation real-data record; G-08-2 CLOSED on real data as of 2026-06-30"
affects:
  - "UAT test 2 (re-test on the Mac)"
  - "Phase 9 (recipe fix for the input-independent posterior, now disclosed on the page)"
tech-stack:
  added: []
  patterns:
    - "a proxy wrapping the REAL fitted model records every predict_proba frame (scored-row spy) and can fix the output (degenerate arm)"
    - "ruling-only branches: each ruled behaviour has a test arm that fails under the unchosen option"
key-files:
  created:
    - .planning/phases/08-regime-persistence-stability/08-14-SUMMARY.md
  modified:
    - src/trading_crab_lib/platform/report/weekly.py
    - tests/unit/test_platform_report_serving.py
    - README.md
    - CLAUDE.md
    - .planning/phases/08-regime-persistence-stability/08-SERVING.md
decisions:
  - "q1-c implemented with the staleness cap read data-relative (month-ends behind the newest monthly_features row), as the orchestrator relayed it; one constant, one comparison site"
  - "The belief and the no-trade band step on the SCORED month, so reruns while the newest row stays ragged reproduce exactly"
  - "q2-ii count printed with full ISO dates (2007-04-30 → 2026-06-30), as 08-SERVING §2.2 words it, not the plan table's YYYY-MM"
  - "08-13's interim NaN-refusal test was removed: q1-c supersedes the behaviour it pinned"
metrics:
  duration: "~45 min"
  completed: 2026-09-29
estimate:
  tokens: 70000
  tasks: 3
actuals:
  tokens: 7200
  tasks: 3
  commits: 3
---

# Phase 8 Plan 14: Glenn's rulings q1-c and q2-ii on the weekly report, measured on real data (G-08-2)

**What changed.** When the newest data row lacks model columns, the weekly report now scores the
latest month that has them all. It prints "Scored as of" that month and names what each newer row
lacks. It refuses when that month is more than 3 month-ends behind the newest row. Under the
distribution it prints the exact number of distinct posteriors the served model gives across
history. On the real tracked data the report runs: it scores 2026-06-30, discloses **1 distinct
posterior vector across 231 complete months**, and a second run in the same month is
byte-identical. **G-08-2 is CLOSED on real data as of 2026-06-30.**

## Rulings implemented (08-SERVING.md §2)

| Ruling | Implemented | Not implemented |
|---|---|---|
| **q1-c** latest complete month + cap | `_scored_row`: `monthly_features[cols].dropna(how="any").iloc[[-1]]`. ValueError before any save when nothing is complete, or when the month is more than `MAX_SCORING_LAG_MONTHS = 3` month-ends behind the newest row. Belief, hysteresis and band step on the scored month. | q1-a (the raise is replaced), q1-b (no degrade-hold branch). No imputation. |
| **q2-ii** disclose | `_input_sensitivity_note`: `np.unique(predict_proba(complete_rows), axis=0)`, exact; count, months and window under the distribution; the does-not-depend sentence only when the count is 1. | q2-iii (never withholds). No fit, no threshold. |

`fit_l2_nowcaster`, `serving.py` and `driver.py` were not touched (the diff since the last 08-13 commit is empty).

## Tasks and commits

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 (tracer) | q1-c: latest complete month, the note, the cap | `efbd262` | weekly.py, test_platform_report_serving.py |
| 2 | q2-ii: distinct-posterior disclosure | `eed8057` | weekly.py, test_platform_report_serving.py |
| 3 | 08-SERVING §3, README behaviour, counts 2451 → 2459 | `df4e30b` | 08-SERVING.md, README.md, CLAUDE.md |

## RED / GREEN record

### Task 1 (q1-c), RED against 08-13's weekly

All seven q1-c tests were written first and run: `8 failed, 1 passed`. The 8 failures include the
two q2ii tests, which were then written alongside and were failing through the same refusal.
- `test_q1c_scores_the_latest_complete_month_and_names_the_lag`,
  `test_q1c_belief_and_band_step_on_the_scored_month[2-2021-04-30]` and
  `test_q1c_staleness_cap_boundary_exactly_3_behind_serves` were RED with the interim refusal:
  `ValueError: The latest monthly_features row (2021-06-30) has NaN in 1 of the nowcaster's 2 model columns: ['f_slope']. The latest month observed in every model column is 2021-04-30. The report does not impute, so it refuses to score this row (interim behaviour until plan 08-14 applies the 08-12 ruling).`
  The boundary test gave the same message ending "... is 2021-03-31 ...".
- `test_q1c_a_complete_latest_row_is_scored_as_itself_with_no_lag`: RED.
  `AssertionError: assert 'Scored as of 2021-06-30' in '# Trading-Crab Platform Weekly Report\n\n## Current Regime Distribution\n\n- regime 2: 67.0%...'`
- `test_q1c_no_complete_month_raises`: RED.
  `AssertionError: the message names the model columns / assert ('f_slope' in "The latest monthly_features row (2021-06-30) has NaN in 1 of ... ['f_slope'] ..." and 'f_level' in ...)`
- `test_q1c_staleness_cap_refuses_a_month_more_than_3_behind`: RED.
  `AttributeError: module 'trading_crab_lib.platform.report.weekly' has no attribute 'MAX_SCORING_LAG_MONTHS'`
- **GREEN at once, as expected:** `test_q1c_belief_and_band_step_on_the_scored_month[0-2021-06-30]`,
  the complete-row arm. Its value is as the other side of the pair.
- **GREEN after the change:** serving, weekly and pooling-consumer suites `63 passed`.
- **Tracer gate (autonomous form):** re-ran Task 1's verify after the commit: `58 passed`, registry
  44, and data/ outputs/ registry/ untouched. Then expanded.

### Task 2 (q2-ii), RED after q1-c landed (its own RED, not the q1 refusal)

- `test_q2ii_the_page_states_the_distinct_count_and_window`: RED.
  `AssertionError: assert '316 distinct posterior vectors across 316 complete months (1995-01-31 → 2021-04-30)' in '# Trading-Crab Platform Weekly Report...'`
- `test_q2ii_a_constant_posterior_is_disclosed_as_such`: RED.
  `AssertionError: assert '1 distinct posterior vector across 316 complete months (1995-01-31 → 2021-04-30)' in '# Trading-Crab Platform Weekly Report...'`
- One intermediate RED after implementation was a harness artifact:
  `assert_frame_equal ... AssertionError: (None, <MonthEnd>)`. The parquet-loaded frame carries
  index `freq`; values and index were equal. That comparison now uses `check_freq=False`, with a
  comment.
- **GREEN:** `65 passed`.

## Mutation arms (every new check shown able to fail)

Each mutation was applied to `weekly.py`, the relevant tests were run, and the file was restored
(`diff -q` against a saved copy).

| # | Mutation | Result |
|---|---|---|
| M1 | Forward-fill the scored row (`monthly_features[cols].ffill().iloc[[-1]]`) | RED: scored-row spy, and month-stepping `[2-2021-04-30]` |
| M2 | Step belief and band on the newest row, not the scored month | RED: `test_q1c_belief_and_band_step_on_the_scored_month[2-2021-04-30]` |
| M3 | Cap `>` becomes `>=` | RED: `test_q1c_staleness_cap_boundary_exactly_3_behind_serves` |
| M4 | Cap removed | RED: `test_q1c_staleness_cap_refuses_a_month_more_than_3_behind` |
| M5 | Empty-complete raise removed | RED: `test_q1c_no_complete_month_raises` |
| M6 | Note omits the lagging columns | RED: `test_q1c_scores_the_latest_complete_month_and_names_the_lag` |
| Q1 | Does-not-depend sentence always printed | RED: varying arm |
| Q2 | Sentence never printed | RED: constant arm |
| Q3 | Rounded comparison (`np.round(P, 2)`) | RED: varying arm (the count differs from the independent exact count) |
| Q4 | Count over forward-filled rows | RED: both q2ii arms (the counted frame is not the complete rows) |
| Q5 | Withhold when the count is 1 (the unchosen q2-iii) | RED: constant arm |
| Q6 | The note fits a model | RED: both q2ii arms (the fit spy on `CalibratedClassifierCV.fit`) |

## Real-data sequence (scratch copies; `TC_DATA_DIR` / `TC_OUTPUT_DIR` under the scratchpad)

This ran twice: once after Task 1 (q1-c alone), then again from a fresh copy after Task 2 (both
rulings). Each time the order was `python -m trading_crab_lib.platform.report.serving`, then
`python -m trading_crab_lib.platform.report.weekly` twice. It was a cold start, with no prior
state. Every run exited 0.

Verbatim key lines from the final real-data `weekly_report.md`:

- "Scored as of" line:
  > Scored as of 2026-06-30, the latest month observed in all 55 of the nowcaster's model columns (2 month-ends behind the newest row). Newer rows lack model columns: 2026-07-31 lacks div_yield; 2026-08-31 lacks fred_m2sl, fred_totalsl, div_yield. Nothing is imputed.
- Distinct-posterior disclosure:
  > 1 distinct posterior vector across 231 complete months (2007-04-30 → 2026-06-30): the served model scored on every month observed in all 55 model columns, compared exactly (no rounding, no threshold). The distribution above does not depend on the features: it is the same every week.
- Active regime line:
  > - active regime: none (neutral posture)
- A7 line:
  > The active regime is a reported label: it selects the trajectory and per-asset rows below and gates no weight (audit item A7, 08-A7.md). The weights come from the filtered belief through the no-trade band.
- No-trade band line:
  > Targets below are the EXECUTED book after the 5.0% no-trade band (design §5.3 bounded turnover, 08-A7.md): an asset whose target moved by no more than the band from its last executed weight keeps that weight.

Other measurements:
- **Served distribution:** 3: 56.4%, 0: 41.8%, 4: 1.8%.
- **Belief** (`as_of` 2026-06-30): 3: 36.9%, 0: 27.2%, 1: 21.3%, 5: 6.7%, 2: 6.7%, 4: 1.2%.
- **Executed book:** TLT 0.390654, SPY 0.330723, USO 0.144729, IAU 0.133895; cash 0.0%.
- **Second run:** `executed_weights` equal under `assert_frame_equal`, and the report
  **byte-identical** (`cmp`) in both sequences.
- **UAT test 2:** all of its expected strings are present, and the targets were the same on the
  second run.

**G-08-2 status sentence (08-SERVING.md §3.4):** "CLOSED on real data: the supported commands
produce the weekly report on the tracked data as of 2026-06-30". §3.4 states the limits:
- step 1 (`build_platform_data.py`) was not re-run;
- the posterior is still input-independent, now disclosed;
- the cap uses the data-relative reading, still pending Glenn's confirmation;
- the serve-time lag is now live.

## Registry (ADR-0004 gap-closure budget 0)

- Before and after every run, and at plan end: sha256
  `c957e8fdb360f04cebd36a6ac19f7efe5c036d098152a5b0c1694f2e73c088ad`, `total_trial_count()` **44**.
- `git status --porcelain -- data outputs registry` is empty.
- `git diff --stat c2e632d -- legacy/ gsd-scratch-work/ trading-crab-lib/ registry/` is empty.
- `git diff --stat 0da072d -- src/trading_crab_lib/platform/report/serving.py
  src/trading_crab_lib/platform/backtest/driver.py` is empty: no recipe change after 08-13.

## Counts, suite, ratchet, lint

- **Live collection 2459** (2451 − 1 superseded + 7 q1-c + 2 q2-ii), written at all four sites:
  CLAUDE.md layout tree and current status; README badge and feature list.
  `test_docs_recorded_counts.py`: `7 passed`.
- **Full suite:** `2459 passed, 5 warnings in 377.71s`, with 0 failed, 0 skipped, 0 xfailed.
- **Legacy-import ratchet:** `11 passed`, `MAX_LEGACY_IMPORT_SITES = 31`.
- **Lint:** ruff clean; flake8 (E9,F63,F7,F82) clean on weekly.py and test_platform_report_serving.py.

## Re-test instruction for UAT test 2 (on the Mac)

```bash
python scripts/build_platform_data.py
python -m trading_crab_lib.platform.report.serving
python -m trading_crab_lib.platform.report.weekly
python -m trading_crab_lib.platform.report.weekly   # second run, same month
```

Expected under q1-c + q2-ii:
- The report is written to `outputs/reports/platform/weekly_report.md`.
- It carries "Scored as of <month>". That month is the newest one complete in all model columns;
  on today's tracked data it is 2026-06-30. A fresh build may move it forward if the lagging
  series have published.
- It names what each newer row lacks, and says "Nothing is imputed."
- Under the distribution: "1 distinct posterior vector across N complete months (2007-04-30 → …)"
  and the does-not-depend sentence.
- "active regime: none (neutral posture)" (or "regime N"), the A7 sentence, and "Targets below are
  the EXECUTED book after the 5.0% no-trade band".
- The second run gives identical targets.
- If the newest complete month is more than 3 month-ends behind the newest row, weekly instead
  raises a ValueError naming both months and `MAX_SCORING_LAG_MONTHS = 3`, and writes nothing.

## Deviations from Plan

**1. [Rule 1 - Superseded test] 08-13's `test_a_nan_in_a_model_column_fails_loudly_naming_it` was removed.**
- It pinned the interim refusal that q1-c replaces.
- Its fixture knob `nan_in_model_col` became `nan_tail: int`, which NaNs the last N holdout rows of
  `f_slope`. Holdout rows only, so the dev fit is unchanged.
- The q1-c tests cover the same row: the refusal paths still name the columns.

**2. Date format of the disclosure window.**
- The plan's branch table wrote "(2007-04 → 2026-06)". The disclosure prints full ISO dates,
  "(2007-04-30 → 2026-06-30)", which is 08-SERVING §2.2's own wording.

**3. The module docstring sentences landed with the feature commits (Tasks 1 and 2), not Task 3.**
- Each sentence ships with the behaviour it describes. README's sentence landed in Task 3, as
  planned.

**4. The tracer gate ran in autonomous form.**
- `workflow.auto_advance` and `_auto_chain_active` both read `false`.
- The plan is `autonomous: true`, and the orchestrator asked for a full run and a completion
  report. As in 08-13, the automated gate was applied: re-verify after the commit, halt on failure.

**5. [Rule 1 - Harness] `check_freq=False` on one frame comparison.**
- Index `freq` is metadata that the parquet round trip sets. The values stay compared exactly.

**6. Task 1 built the Task 2 tests too (`8 failed` above).**
- The q2ii tests were written in the same pass. For atomic commits they were set aside for
  Task 1's commit and restored for Task 2.
- Their Task 2 RED lines are their own: the missing disclosure, not the q1 refusal.

## Known Stubs

None.

## Threat Flags

None beyond the plan's register.
- T-08-76: mitigated by the scored-row spy and the M1 arm.
- T-08-77: mitigated. §3's status is tied to a produced report.
- T-08-78: mitigated by the q2ii pair and the Q1, Q2 and Q5 arms.
- T-08-79: mitigated by scratch dirs, the empty `git status` and an unchanged sha256.

## Open items for STATE (owned by the orchestrator)

1. **Glenn to confirm the staleness cap's data-relative reading** (08-SERVING §2.1, repeated in
   §3.4). Under a wall-clock reading, runs from 2026-10-01 would refuse until July's `div_yield`
   arrives. The change is one site, `weekly._scored_row`, which compares against
   `MAX_SCORING_LAG_MONTHS`.
2. **The page lists no per-account trades on the default config.** `report.accounts` is
   unconfigured, so the executed book appears only in the `executed_weights` checkpoint. This
   predates 08-14.
3. **Phase 9 recipe fix** for the input-independent posterior. It is unchanged, and is now
   disclosed on the page.
4. Carried from 08-13: `evaluate_nowcaster` would overwrite the served model and spend a trial if
   called; there are untracked weekly/serving parquets in the tracked platform namespace after
   real runs.

## Self-Check: PASSED

- FOUND: src/trading_crab_lib/platform/report/weekly.py (MAX_SCORING_LAG_MONTHS, _scored_row, _input_sensitivity_note)
- FOUND: tests/unit/test_platform_report_serving.py (TestQ1cLatestCompleteMonth, TestQ2iiDistinctPosteriorDisclosure)
- FOUND: .planning/phases/08-regime-persistence-stability/08-SERVING.md §3 with one status sentence
- FOUND: commits efbd262, eed8057, df4e30b
