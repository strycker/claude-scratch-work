---
phase: 06-platform-notebook-suite
plan: 03
subsystem: platform-regime-notebook
tags: [matplotlib, nbformat, A13, economic-history, USREC, sign-off, jupyter]

# Dependency graph
requires:
  - phase: 06-platform-notebook-suite
    provides: "plan 06-01 — platform/plotting/{core,loaders,drift}.py, notebooks/platform/P1_data_spine.ipynb, tests/unit/test_platform_notebooks.py"
  - phase: 06-platform-notebook-suite
    provides: "plan 06-02 — backtest_full_sample_states.parquet (695x1), backtest_filtered_state_probs.parquet (470x5)"
  - phase: 03-regime-labeling-prediction
    provides: "labeling/diagnostics.py label_regimes/occupancy_and_sojourns/auto_profile, prediction/transition_matrix.py"
  - phase: 05-backtest-evaluation
    provides: "backtest/driver.py _window_active_features, honesty/walkforward.py expanding_steps"
provides:
  - "platform/plotting/history.py: ECONOMIC_EVENTS, RECESSION_ERA_LABEL, BASELINE_ERA_LABEL, load_usrec_or_warn, recession_periods, regime_era_contingency, regime_era_marginals, plot_era_contingency"
  - "platform/plotting/regime.py: plot_regime_timeline, plot_occupancy_and_sojourn, plot_transition_matrix, plot_soft_confidences, plot_regime_profiles, plot_sojourn_distribution, active_feature_count_timeline, feature_set_change_dates, plot_active_feature_count, label_disagreement, plot_label_comparison"
  - "notebooks/platform/P3_regime_labeling.ipynb — the cold-start sign-off notebook and the A13 side-by-side, executed with outputs"
affects: [06-07-backtest-notebook]

# Tech tracking
tech-stack:
  added: []  # zero new dependencies
  patterns:
    - "D-12 events-as-module-constant: six dated ranges live in ECONOMIC_EVENTS, with the docstring stating explicitly that a dedicated events file / provenance schema was rejected"
    - "D-10 loud degradation for the phase's only network call: load_usrec_or_warn catches everything, returns None, and logs one WARNING naming the failure, the CONSEQUENCE (shading omitted, event overlay still renders), and the FIX (FRED_API_KEY + reachability)"
    - "D-13 two-directional contingency: era-conditional (rows sum to 1.0) AND state-marginal (rows do not), because one reading alone invites seeing a pattern that is not there; no significance test"
    - "A13 reconstruction imports backtest.driver._window_active_features read-only rather than reimplementing the rule — a test asserts object identity with the driver's own function"
    - "plot_label_comparison never reindexes its inputs onto a common index: a track is blank where its labeling has no value, because differing spans are the finding"
    - "Empty-era rows are omitted from the contingency rather than emitted as an all-zero row, so 'every row sums to 1.0' holds unconditionally"

key-files:
  created:
    - src/trading_crab_lib/platform/plotting/history.py
    - src/trading_crab_lib/platform/plotting/regime.py
    - notebooks/platform/P3_regime_labeling.ipynb
    - tests/unit/test_platform_plotting_history.py
    - tests/unit/test_platform_plotting_regime.py
  modified:
    - .gitignore

key-decisions:
  - "feature_set_change_dates returns an EMPTY list for a constant-count timeline rather than a one-element list containing the first step. The plan's action block says 'including the first step' and its acceptance criteria say 'an empty list for a constant-count synthetic timeline' — both hold only if the first step counts as a change when the count later varies, and not otherwise. Reporting the first step unconditionally would inflate every change count by one and report a change where nothing changed."
  - "label_disagreement between the reference and filtered labelings reports n_compared = 470, NOT the plan's stated 588. This is the same planning defect plan 06-02 already corrected: the filtered artifact only has 470 rows because 118 of 588 walk-forward steps are degraded and excluded from per_step_metrics. 588 is unreachable without inventing rows."
  - "The scratch checkpoint namespace data/checkpoints/platform_notebook/ was added to .gitignore. data/checkpoints/platform/ IS tracked in this repo, so the untracked scratch twin sits one `git add data/` away from being committed — exactly the production-namespace contamination D-10/T-06-12 exists to prevent."
  - "Three figure-layout defects were found only by rendering against real data and fixed before commit: recession shading was invisible behind the opaque regime bands; the disagreement strip overdrew the bottom labeling track; two annotations collided with chart titles. A chart an operator cannot read cannot carry a sign-off (T-06-13)."

requirements-completed: [NB-01]

coverage:
  - id: H1
    description: "The economic-history overlay degrades loudly and actionably when the FRED USREC fetch fails, naming the failure, the consequence, and the fix (D-10)"
    requirement: "NB-01"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_plotting_history.py::TestLoadUsrecOrWarn (4 tests — missing key, raising client, empty config, stubbed success); no test performs network I/O"
        status: pass
    human_judgment: false
  - id: H2
    description: "Both directions of the D-13 regime x era contingency exist, are descriptive only, and the era-conditional reading's rows sum to 1.0 with every value in [0, 1]"
    requirement: "NB-01"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_plotting_history.py::TestRegimeEraContingency, ::TestRegimeEraMarginals"
        status: pass
    human_judgment: false
  - id: R1
    description: "The A13 active-feature reconstruction reproduces the seven documented change points exactly against the real dev checkpoint, using the driver's own rule object"
    requirement: "NB-01"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_plotting_regime.py::TestActiveFeatureCountTimeline::test_real_dev_features_reproduce_the_seven_a13_change_points, ::test_uses_the_drivers_own_rule_object"
        status: pass
    human_judgment: false
  - id: R2
    description: "label_disagreement returns zeros on a disjoint intersection instead of raising, and pct_disagree is always a proportion"
    requirement: "NB-01"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_plotting_regime.py::TestLabelDisagreement (7 tests)"
        status: pass
    human_judgment: false
  - id: R3
    description: "Every public plot function returns a Figure for both a populated fixture and an empty input"
    requirement: "NB-01"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_plotting_regime.py (6 plot classes) + tests/unit/test_platform_plotting_history.py::TestPlotEraContingency"
        status: pass
    human_judgment: false
  - id: N1
    description: "P3 passes the static notebook gate, contains the sign-off tokens, and calls neither run_backtest nor run_full_backtest_evaluation"
    requirement: "NB-01"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_notebooks.py (12 passed; the A13-discipline guard for P3 now runs instead of skipping) + the plan's own static assertion script"
        status: pass
    human_judgment: false
  - id: N2
    description: "P3 runs top-to-bottom against real checkpoints and both persisted labeling artifacts without raising"
    requirement: "NB-01"
    verification:
      - kind: manual
        ref: "notebooks/platform/P3_regime_labeling.ipynb executed via `jupyter execute --inplace` this session from a cleared scratch namespace; 10 code cells, 0 error outputs, 10 figures, outputs committed"
        status: pass
    human_judgment: true
  - id: N3
    description: "The regime timeline read against dated economic history supports or contradicts the labeling — a human judgement recorded in the sign-off cell"
    requirement: "NB-01"
    verification:
      - kind: manual
        ref: "notebooks/platform/P3_regime_labeling.ipynb final markdown cell — Date/Verdict/Reasoning fields are blank and awaiting the operator (D-15). The panel they judge is rendered and legible; the verdict itself is not mine to record."
        status: pending
    human_judgment: true

actuals:
  tokens: 71000
  tasks: 3
  commits: 3

metrics:
  duration_minutes: 65
  completed_date: 2026-09-10
status: complete
---

# Phase 6 Plan 03: P3 Regime Labeling Notebook Summary

Built the economic-history overlay (`history.py`), the regime panels plus the A13
three-labeling comparison surface (`regime.py`), and `P3_regime_labeling.ipynb` —
the one notebook in the suite that carries a cold-start human decision. P3 draws
the five regimes against NBER recession bars and six dated economic eras, renders
all three labelings the codebase produces side by side with the seven feature-set
change dates marked, quantifies their disagreement at **82.8%**, and ends in a
plain markdown sign-off cell awaiting an operator's verdict.

## Task Commits

1. **Task 1: Economic-history overlay and the regime-by-era contingency table**
   - `6fa34b8` (feat) — `platform/plotting/history.py`,
     `tests/unit/test_platform_plotting_history.py` (25 tests)
2. **Task 2: Regime plotting module including the A13 three-labeling comparison**
   - `7c30514` (feat) — `platform/plotting/regime.py`,
     `tests/unit/test_platform_plotting_regime.py` (39 tests)
3. **Task 3: P3_regime_labeling notebook with the cold-start sign-off cell**
   - `5f364d1` (feat) — `notebooks/platform/P3_regime_labeling.ipynb` (executed,
     outputs committed), plus the render-driven layout fixes to `regime.py` and
     one `.gitignore` line

## Verification Results

- `pytest tests/unit/test_platform_plotting_history.py -x -q` → **25 passed**
- `pytest tests/unit/test_platform_plotting_regime.py -x -q` → **39 passed**
- `pytest tests/unit/test_platform_notebooks.py -x -q` → **12 passed, 2 skipped**
  (was 12 passed / 2 skipped with 3 skips before P3 existed — the P3
  A13-discipline guard now runs and passes; P4 and P6 remain skipped by design)
- **Full suite: `pytest tests/ -q` → 1528 passed, 2 skipped, 0 failures** (~76s).
  Baseline entering this plan was 1458 passed / 3 skipped. Net +70 tests, one
  skip converted to a pass, zero regressions.
- `ruff check` clean on all four new source/test files.
- The plan's Task 2 `<verify>` script ran verbatim and printed
  `A13 timeline OK [('1972-01-31', 4), ('1972-02-29', 6), ('1972-04-30', 8),
  ('1973-02-28', 9), ('1986-06-30', 10), ('1995-02-28', 12), ('2000-01-31', 13)]`
  — 588 rows, seven change points, exact match to the audit's reported sequence.
- The plan's Task 3 `<verify>` static script ran verbatim: all nine required
  tokens present, neither banned token present, 20 cells.
- `git status --porcelain src/trading_crab_lib/platform/backtest/driver.py
  outputs/reports/platform/ data/checkpoints/platform/ registry/` → **empty**.
  The driver was read, never edited; no Phase-5 artifact and no production
  checkpoint was touched; the trial registry is untouched.
- Notebook executed end to end from a **cleared** scratch namespace: 10 code
  cells, execution counts 1–10, **0 error outputs**, 10 figures, committed with
  outputs. The live FRED `USREC` fetch succeeded (776 months, 8 recession
  periods over the plotted span); the degradation branch is therefore covered by
  unit tests rather than by this run.
- Secret hygiene: the live `FRED_API_KEY` value does **not** appear anywhere in
  the committed notebook; config is displayed only through
  `pplot.redacted_config(cfg)` (T-06-01).

## The measured disagreement between the two labelings

The headline number this plan was asked to produce:

| quantity | value |
|---|---|
| `n_compared` | **470** (not 588 — see Deviations) |
| `n_disagree` | **389** |
| `pct_disagree` | **0.8276595744680851 (82.8%)** |
| `first_common_date` | 1974-02-28 |
| `last_common_date` | 2020-12-31 |

Reference state (rows) × filtered state (columns) counts:

| | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| **0** | 0 | 3 | 0 | 0 | 3 |
| **1** | 0 | 11 | 44 | 18 | 4 |
| **2** | 0 | 4 | 23 | 33 | 18 |
| **3** | 27 | 55 | 80 | 43 | 36 |
| **4** | 1 | 22 | 29 | 12 | 4 |

**No band is claimed around 82.8%, and it must not be read as a detection lag.**
Three independent reasons, all surfaced in the notebook:

1. **Different feature sets.** The reference is fit on a fixed 9 columns across
   695 months; the walk-forward's active set changes seven times (4 → 6 → 8 → 9 →
   10 → 12 → 13) at 1972-01-31, 1972-02-29, 1972-04-30, 1973-02-28, 1986-06-30,
   1995-02-28, 2000-01-31 — all reproduced exactly from the driver's own rule.
2. **Coverage.** 118 of 588 walk-forward steps (20.1%) are degraded and excluded
   from `per_step_metrics`, so the filtered path has 470 rows and starts
   1974-02 — ~2 years after the backtest's 1972-01 start. The comparison covers
   ~80% of steps and none of the early window.
3. **Independent canonicalization.** The confusion table shows no diagonal
   structure at all: reference state 3 (the 40.6%-occupancy state) scatters
   across all five filtered states. The two labelings are canonicalized
   separately, so their state ids need not denote the same regimes — a high raw
   disagreement is compatible with similar regime *structure* under different id
   assignments. This is stated in the notebook as a caveat, not a conclusion.

**P3 nowhere claims A13 is resolved.** Rendering makes it inspectable.
`core.A13_CAVEAT` is printed verbatim, followed by the 06-02 coverage addendum.

## What the sign-off panel actually shows (criterion 4)

The reference labeling's transition dates, printed by the notebook, are exactly
the six Amendment 2 item E names plus the 1963-02 start:
1963-02-28 → 2, **1973-09-30 → 1**, **1981-10-31 → 4**, **1988-09-30 → 2**,
**1996-08-31 → 3**, **2008-07-31 → 0**, **2009-06-30 → 3**.

Occupancy 1.5827 / 13.9568 / 31.9424 / 40.5755 / 11.9424 percent (sum
1.000000000000, matching `06-VALIDATION.md`'s reference within 0.05pp per state);
pooled median sojourn 97.0 months; state 0 below the 5% §4.4 soft floor, logged
by `drift.assert_regime_occupancy_plausible` as a warning, not a block.

The D-13 contingency is the strongest single piece of evidence the operator has:
the 1973 oil shock is **100% state 1**; the 1987 crash **100% state 4**; LTCM and
the dot-com bust **100% state 3**; the GFC **58% state 0** against a 2% baseline
occupancy — a ~29× lift into the crisis state. The NBER recession row, by
contrast, spreads roughly evenly (0.13 / 0.28 / 0.22 / 0.20 / 0.16), which is
exactly why D-13 demanded both readings: recessions alone cannot separate a
stagflation regime from a credit-crisis regime, and the marginal view says so.

**The verdict itself is deliberately not recorded here.** The sign-off cell's
Date / Verdict / Reasoning fields are blank and await the operator (D-15). Per
D-16 a negative verdict is recorded and does not block.

## Deviations from Plan

### Corrected planning defect (inherited, already established by plan 06-02)

**1. `label_disagreement` reports `n_compared` = 470, not the plan's stated 588**
- **Found during:** Task 3, running the A13 cell against the real artifacts.
- **Issue:** the Task 3 acceptance criteria state "`label_disagreement` between
  the reference and filtered labelings reports `n_compared` equal to 588".
  `backtest_filtered_state_probs.parquet` has 470 rows, because
  `backtest/driver.py:452-455` only appends non-degraded steps to
  `per_step_metrics`. This is the identical defect plan 06-02 already found and
  corrected for the artifact's own row count; it simply propagated into this
  plan's criteria.
- **Resolution:** reported honestly as 470 and surfaced prominently in the
  notebook as the coverage finding this plan was explicitly asked to carry.
  Reaching 588 would require inventing 118 rows that do not exist.
- **Files affected:** none — no code or test asserts 588.

### Auto-fixed Issues

**2. [Rule 1 - Bug] Three figure-layout defects, visible only against real data**
- **Found during:** Task 3, inspecting the executed notebook's rendered figures.
- **Issue (a):** `plot_regime_timeline` drew NBER recession shading as an
  `axvspan` behind the regime bands, which are opaque `vlines` — the shading was
  completely invisible, and the six event labels overlapped each other and the
  chart title. An operator would have signed off against a chart that silently
  showed no recession information at all, which is precisely threat T-06-13.
  **Issue (b):** `plot_label_comparison`'s disagreement strip was drawn at
  y ∈ [0.05, 0.75] while the bottom labeling track occupied [0.2, 1.0] — the
  strip overdrew the third labeling. **Issue (c):** the final (largest)
  annotation in `plot_active_feature_count` and the tallest bar label in
  `plot_occupancy_and_sojourn` both collided with their titles.
- **Fix:** (a) redrew the timeline as three stacked non-overlapping tracks in one
  axes (regime / NBER recession / dated eras) with event labels staggered across
  two rows beneath the event bars and named y-tick labels for each track; (b)
  moved the tracks up one unit and reserved y ∈ [0.10, 0.90] for the strip; (c)
  annotate below the step when it sits at the top of the range, and added
  `margins(y=0.18)` headroom to the sojourn panel.
- **Files modified:** `src/trading_crab_lib/platform/plotting/regime.py`
- **Commit:** `5f364d1`

**3. [Rule 3 - Blocking] `_era_masks` built a numpy array and then called `.to_numpy()` on it**
- **Found during:** Task 1, first test run.
- **Issue:** `(DatetimeIndex >= ts) & (DatetimeIndex <= ts)` returns a plain
  `numpy.ndarray`, not an Index, so the trailing `.to_numpy()` raised
  `AttributeError`.
- **Fix:** `np.asarray(...)` instead.
- **Files modified:** `src/trading_crab_lib/platform/plotting/history.py`
- **Commit:** `6fa34b8`

**4. [Rule 2 - Missing safeguard] `.gitignore` did not cover the notebook scratch namespace**
- **Found during:** Task 3, after the first notebook execution.
- **Issue:** `data/checkpoints/platform/` is **tracked** in this repo, and
  `compute_regime_labeling` writes its scratch twin to
  `data/checkpoints/platform_notebook/`, which appeared as untracked output in
  `git status`. That leaves generated scratch checkpoints one `git add data/`
  away from being committed into the repository — the production-namespace
  contamination D-10 and threat T-06-12 exist to prevent, arriving by a
  different route than the one those controls anticipated.
- **Fix:** one ignore rule with a comment naming `NOTEBOOK_SCRATCH_DIR`, D-10,
  and T-06-12.
- **Files modified:** `.gitignore` (outside the plan's declared `files_modified`;
  recorded here as a deliberate, documented addition)
- **Commit:** `5f364d1`

## Prohibitions — status

| Prohibition | Status | Evidence |
|---|---|---|
| No (K, λ, n_restarts) sweep; shipped configuration only (D-14) | **held** | No cell or function fits with any K/λ other than the config's. `plot_occupancy_and_sojourn` sources every number from `occupancy_and_sojourns`; the only fit in the notebook is the single shipped `label_regimes()` call. |
| No `sign_off()` helper, no machine-readable ledger (D-15) | **held** | The sign-off is cell 19, a markdown cell. `grep -c sign_off` over `src/` returns 0 for anything this plan added. |
| `run_full_backtest_evaluation` / `run_backtest(` appear nowhere in P3 | **held** | Asserted by the plan's own static script and by `tests/unit/test_platform_notebooks.py`. |
| Nothing writes `data/checkpoints/platform/` or modifies `outputs/reports/platform/` | **held** | `git status --porcelain` on both paths is empty after three notebook executions. |
| No task claims A13 is resolved | **held** | The notebook says "makes A13 inspectable and does not resolve it" in two places; `A13_CAVEAT` is printed verbatim; the module docstring repeats it. |

## Known Stubs

None.

## Issues Encountered

- The live `USREC` fetch succeeded in this environment, so the D-10 degradation
  branch was exercised only by unit tests (a monkeypatched raising client and a
  config with no API key). An operator on a machine without `FRED_API_KEY` will
  see the printed degradation banner in cell 5 and must not sign off against
  recession agreement — the notebook says so explicitly.
- `label_regimes()`'s churn metric is `nan` on a first run in a cleared scratch
  namespace, because it compares against a previous `regime_labels` checkpoint
  that does not yet exist. The notebook prints the value and the markdown names
  this as documented first-run behavior. Re-running P3 a second time yields a
  real churn number.

## User Setup Required

None. Zero new dependencies. The FRED credential is the incumbent
`FRED_API_KEY`; no new secret was introduced.

## Next Plan Readiness — what 06-07 (P6) needs to know

- **Render the A13 caveat identically.** P6 displays the same §5.4
  sojourn/lag headline (`median_sojourn` 97.0, `median_lag` 164.0, `ratio`
  0.5914634146341463, `n_resolved` 4 of `n_transitions` 6). It must print
  `pplot.A13_CAVEAT` verbatim — the same single string P3 prints — so the wording
  cannot drift. `tests/unit/test_platform_notebooks.py`'s A13-discipline guard
  will start enforcing the `A13` mention automatically once
  `P6_backtest_evaluation.ipynb` exists.
- **Carry the resolved-transition count with the ratio.** Wherever P6 shows
  0.591, show "4 of 6 transitions resolved" beside it. P3 states the same.
- **Carry the coverage finding too.** The caveat is now two-part: the feature-set
  mismatch *and* the 470/588 (20.1% degraded, filtered path absent before
  1974-02) coverage gap. P3 prints both together in one cell; P6 should present
  them the same way rather than only the small-sample caveat.
- **Reuse `regime.py` rather than re-deriving.** `label_disagreement`,
  `active_feature_count_timeline`, `feature_set_change_dates`, and
  `plot_label_comparison` are all available by submodule import
  (`from trading_crab_lib.platform.plotting import regime as pregime`). They are
  **not** in `platform/plotting/__init__.py`'s `__all__`, by 06-01's design.
- **The measured disagreement is 82.8% over 470 comparable months.** If P6
  recomputes it, it must get the same number; if it differs, something upstream
  changed and that is a finding.
- **Do not re-run the walk-forward.** P6 loads the same persisted artifacts.

## Self-Check: PASSED

- `src/trading_crab_lib/platform/plotting/history.py` — FOUND
- `src/trading_crab_lib/platform/plotting/regime.py` — FOUND
- `notebooks/platform/P3_regime_labeling.ipynb` — FOUND (20 cells, outputs committed)
- `tests/unit/test_platform_plotting_history.py` — FOUND (25 tests)
- `tests/unit/test_platform_plotting_regime.py` — FOUND (39 tests)
- `.gitignore` — FOUND, modified
- commit `6fa34b8` — FOUND
- commit `7c30514` — FOUND
- commit `5f364d1` — FOUND

---
*Phase: 06-platform-notebook-suite*
*Completed: 2026-09-10*
