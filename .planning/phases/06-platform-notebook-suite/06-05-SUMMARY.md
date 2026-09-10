---
phase: 06-platform-notebook-suite
plan: 05
subsystem: platform-nowcaster-notebook
tags: [matplotlib, nbformat, calibration, brier, no-skill-floor, persistence-trap, A8-corroboration, A13-discipline, side-effect-free-fit]

# Dependency graph
requires:
  - phase: 06-platform-notebook-suite
    provides: "plan 06-01 — platform/plotting/{core,loaders,drift}.py (A13_CAVEAT, load_report_artifact, compute_regime_labeling, assert_brier_plausible, drift_report), tests/unit/test_platform_notebooks.py"
  - phase: 03-regime-labeling-and-nowcaster
    provides: "platform/prediction/nowcaster.py — build_nowcaster_training_set, fit_nowcaster, transition_window_accuracy"
  - phase: 05-backtest-and-evaluation
    provides: "outputs/reports/platform/model_metrics_{brier,calibration,confusion}.parquet"
provides:
  - "platform/plotting/nowcaster.py: fit_nowcaster_diagnostics, plot_transition_window_accuracy, plot_proba_over_time, plot_calibration_curve, plot_confusion_matrix"
  - "notebooks/platform/P4_nowcaster.ipynb — persistence-trap split, Brier-vs-no-skill verdict, calibration/confusion, A13 caveat, lean-feature drift; executed with outputs"
  - "tests/unit/test_platform_plotting_nowcaster.py — 25 tests incl. the T-06-20 no-side-effect proof and the D-01 boundary check"
affects: [06-06-allocation-notebook, 06-07-backtest-notebook]

# Tech tracking
tech-stack:
  added: []  # zero new dependencies
  patterns:
    - "a diagnostic-only re-orchestration: fit_nowcaster_diagnostics composes the production primitives verbatim but deliberately omits evaluate_nowcaster's registry-append and checkpoint-save side effects, proven by a monkeypatch-to-raise test plus an AST assertion that neither call site appears in the source at all"
    - "stacked-area legends go BELOW the axes — a stacked area chart fills its whole frame, so an in-axes legend occludes the bands it labels (caught by rendering, not by a Figure-is-not-None check)"
    - "calibration marker area scales with n_in_bin so a bin holding 1 observation cannot be misread as one holding 380 — the real artifact contains both"
    - "axis limits set a hair beyond [0, 1] so a bin sitting exactly at 0.0 or 1.0 draws a whole marker instead of a clipped half"

key-files:
  created:
    - src/trading_crab_lib/platform/plotting/nowcaster.py
    - notebooks/platform/P4_nowcaster.ipynb
    - tests/unit/test_platform_plotting_nowcaster.py
  modified: []

key-decisions:
  - "fit_nowcaster_diagnostics narrows features to taxonomy.lean_feature_set(cfg) before building the training set. The plan's verify passes the whole 53-column monthly_features frame, but fit_nowcaster drops every row with any non-finite feature — with the 22 ETF columns present that discards nearly all history. Narrowing to the 13 lean columns is exactly what backtest/driver.py::_refit_l1/_refit_l2 do, and the plan's own read_first names lean_feature_set for this purpose."
  - "The returned X/y/y_pred/proba are the FINITE-row subset actually scored, not the full (X, y) the plan's action block names. fit_nowcaster drops non-finite rows internally and returns only the model; model.predict(X) on the unfiltered frame raises on the NaN months. The notebook's plot_proba_over_time(result['X'].index, result['proba'], ...) requires the index and the proba row count to agree, so the scored subset is what must be returned."
  - "plot_confusion_matrix uses matplotlib imshow rather than a seaborn heatmap — features.py established that platform/plotting submodules reach matplotlib only through core.plt and own no plotting import of their own, and a seaborn import would break that (an AST test enforces it)."

requirements-completed: [NB-01]

coverage:
  - id: N1
    description: "The walk-forward's persisted multiclass Brier is checked against the K=5 no-skill floor and rendered as the amber finding it is, not hidden inside a 'model trained' statement"
    requirement: "NB-01"
    verification:
      - kind: manual
        ref: "plan Task 2 <verify> run verbatim — {'value': 0.2087462934432449, 'no_skill': 0.16, 'beats_no_skill': False}; the notebook prints the same verdict and narrates it in markdown"
        status: pass
      - kind: unit
        ref: "tests/unit/test_platform_plotting_nowcaster.py::TestP4NotebookSource::test_states_the_brier_no_skill_comparison"
        status: pass
    human_judgment: false
  - id: N2
    description: "Overall, transition-window and steady-state accuracy are shown TOGETHER so the persistence trap is visible rather than masked by a single number"
    requirement: "NB-01"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_plotting_nowcaster.py::TestPlotTransitionWindowAccuracy (4 tests, incl. all-three-figures-always-drawn and the NaN 'n/a' bar)"
        status: pass
      - kind: manual
        ref: "executed notebook cell 5 against the real dev data — overall 0.7139, transition 0.5250, steady 0.7375 over 360 scored months"
        status: pass
    human_judgment: false
  - id: N3
    description: "The in-process diagnostic fit never writes the production nowcaster checkpoint and never appends to the git-tracked trial registry"
    requirement: "NB-01"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_plotting_nowcaster.py::TestFitNowcasterDiagnosticsNoSideEffects (3 tests: monkeypatch-both-to-raise, AST assertion that neither call site appears in the source, and evaluate_nowcaster is never imported)"
        status: pass
      - kind: manual
        ref: "git status --porcelain data/checkpoints/platform/ registry/trials.jsonl -> empty after executing the notebook end to end"
        status: pass
    human_judgment: false
  - id: N4
    description: "Calibration and confusion panels come from the real walk-forward's persisted per-step artifacts, never from the in-process fit and never from a recomputation"
    requirement: "NB-01"
    verification:
      - kind: unit
        ref: "TestPlotCalibrationCurve + TestPlotConfusionMatrix (6 tests); TestP4NotebookSource::test_never_invokes_the_full_walk_forward_entrypoints"
        status: pass
      - kind: manual
        ref: "plan Task 2 <verify> — real 21x7 calibration and 20x3 confusion artifacts render; the confusion pivot is 5x5 over labels '0'..'4'"
        status: pass
    human_judgment: false
  - id: N5
    description: "P4 states its A13 relationship in its own terms, rendering core.A13_CAVEAT verbatim, without claiming A13 is resolved"
    requirement: "NB-01"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_notebooks.py::test_a13_discipline_notebooks_mention_audit_item[P4_nowcaster.ipynb] — converted from skip to pass; plus TestFreshPackageBoundary::test_module_docstring_states_the_a13_relationship"
        status: pass
    human_judgment: false
  - id: N6
    description: "A lean feature whose current-era distribution has shifted from the pre-2021 fitted window appears in P4's drift table"
    requirement: "NB-01"
    verification:
      - kind: manual
        ref: "executed notebook cell 16 — 776 full-span rows through 2026-08-31; 6 of 13 lean features flagged (gold +3.80, cape_shiller +1.72, div_yield -1.31, real_rate_level -1.31, oil +1.16, curve_10y3m -1.06)"
        status: pass
    human_judgment: false
  - id: N7
    description: "P4 runs top-to-bottom against the real monthly_features checkpoint and the persisted Phase 5 artifacts without raising"
    requirement: "NB-01"
    verification:
      - kind: manual
        ref: "executed via `jupyter execute --inplace` this session — 10 code cells, execution counts 1-10, 0 error outputs, 4 figures, outputs committed. Every figure was decoded out of the notebook and visually inspected."
        status: pass
    human_judgment: true

actuals:
  tokens: 62000
  tasks: 3
  commits: 3

metrics:
  duration_minutes: 35
  completed_date: 2026-09-10
status: complete
---

# Phase 6 Plan 05: P4 Nowcaster Notebook Summary

Built `platform/plotting/nowcaster.py` — a diagnostic-only (registry-free,
checkpoint-free) nowcaster fit, the design §5.1 persistence-trap panel, the
probability-path chart, and rendering for the walk-forward's three persisted
model-metrics artifacts — plus `P4_nowcaster.ipynb`, executed against real data
with outputs committed.

P4's most important output is an honest negative result, stated plainly.

## Task Commits

1. **Task 1: diagnostic fit + persistence-trap panel**
   - `eeaae01` (feat) — `src/trading_crab_lib/platform/plotting/nowcaster.py`,
     `tests/unit/test_platform_plotting_nowcaster.py` (18 tests at that point)
2. **Task 2: persisted calibration / confusion rendering + Brier band**
   - `804a1d3` (feat) — same two files extended (25 tests)
3. **Task 3: the P4 notebook**
   - `a386737` (feat) — `notebooks/platform/P4_nowcaster.ipynb` (18 cells, executed)

## The Brier finding, as measured

| quantity | value |
|---|---|
| observed multiclass Brier (`model_metrics_brier.parquet`) | **0.208746** |
| K=5 no-skill floor, `(K−1)/K²` | **0.16** |
| `beats_no_skill` | **False** |

This codebase's `compute_brier_multiclass` computes `mean((p − onehot)²)` over the
full `(n, K)` array, so its range is **[0, 1] — not the textbook [0, 2]**. A band
written against [0, 2] would look comfortably mid-range and be silently wrong.

**The observed value sits ABOVE the no-skill floor. On this metric the nowcaster
cannot presently claim to beat random guessing.** That is amber, not green, and P4
prints both numbers, prints `verdict` itself, and narrates the comparison in prose
directly beneath.

It independently corroborates **audit item A8**: the presence of
`CalibratedClassifierCV` in the pipeline had been standing in for evidence that the
model *is* calibrated, and no pass band had ever been asserted against its output.
Asserting one is what surfaced this.

**Nothing here was tuned to make it look better.** No recalibration, no threshold
search, no reweighting, no swapping the calibrator, no hyperparameter sweep — D-14
and this phase's scope fence forbid it, and the repair belongs to whoever owns L2.
The calibration panel shows where the miscalibration lives: every state except 3
sits well below the perfect-calibration diagonal across most of the probability
range, i.e. systematically over-confident.

## The persistence trap, as measured

The in-process diagnostic fit against the live dev data (360 scored months,
1990-01 → 2019-12, 13 lean features, 5 states):

| figure | value |
|---|---|
| `overall_accuracy` | 0.7139 |
| `transition_accuracy` | **0.5250** |
| `steady_state_accuracy` | 0.7375 |

The gap is real and material: accuracy drops ~21 points at exactly the months
guidance depends on. A single headline number would have hidden it. Note this is a
distinct, cheap, **non-walk-forward** fit — the notebook says so in prose, and the
real walk-forward's numbers are loaded separately in the next section and labelled
as such (T-06-21).

## Evidence the diagnostic fit is inert

Three independent proofs, all in `tests/unit/test_platform_plotting_nowcaster.py`:

1. `TestFitNowcasterDiagnosticsNoSideEffects::test_calls_neither_append_trial_nor_save_model`
   monkeypatches `honesty.registry.append_trial` **and**
   `CheckpointManager.save_model` to raise `AssertionError`, then runs
   `fit_nowcaster_diagnostics` to completion. Neither fires.
2. `test_source_never_references_the_two_side_effect_call_sites` parses the module
   with `ast` and asserts neither attribute call appears anywhere in the source —
   so a later edit reintroducing one fails even if the runtime path happens not to
   reach it.
3. `test_does_not_import_evaluate_nowcaster` — the one production entrypoint that
   carries both side effects is never imported into this module.

Operationally: `git status --porcelain data/checkpoints/platform/ registry/trials.jsonl`
is **empty** after executing the notebook end to end. The notebook's own labeling
call routes through `compute_regime_labeling`, which writes only to
`data/checkpoints/platform_notebook/` (D-10).

## Full-span lean-feature drift (D-06 look-not-fit panel)

776 rows through 2026-08-31; 6 of 13 lean features flagged, matching 06-04's
measurement exactly: `gold` +3.80, `cape_shiller` +1.72, `div_yield` −1.31,
`real_rate_level` −1.31, `oil` +1.16, `curve_10y3m` −1.06. Per D-07 I have **not**
written a `.planning/POST-2020-OBSERVATIONS.md` entry — deciding that an observation
is decision-changing is the operator's judgement, not mine.

## Verification Results

- `pytest tests/unit/test_platform_plotting_nowcaster.py -q` → **25 passed**.
- `pytest tests/unit/test_platform_notebooks.py -q` → **23 passed, 1 skipped**
  (P4 joined the glob automatically and **converted the P4 A13-discipline guard
  from skip to pass**; the single remaining skip is P6, not yet built).
- **Full suite: `pytest tests/ -q` → 1613 passed, 1 skipped, 0 failures** (~76s).
  Baseline entering this plan was 1578 passed / 2 skipped. **Net +35 tests, one
  skip converted to a pass, zero regressions, zero skips added.**
- `ruff check` clean on both new source/test files.
- All three plan `<verify>` scripts ran verbatim and printed their sentinels:
  `P4 diagnostic fit smoke OK {...}`, `P4 persisted-artifact smoke OK {'value':
  0.2087462934432449, 'no_skill': 0.16, 'beats_no_skill': False}`,
  `P4 static checks OK 18 cells`.
- `git status --porcelain src/trading_crab_lib/platform/prediction/
  outputs/reports/platform/ data/checkpoints/platform/ registry/trials.jsonl`
  → **empty**. Phase 3/5 code and every reference artifact were read, never modified.
- Secret hygiene (T-06-01/T-06-19): the live `FRED_API_KEY` value does not appear
  anywhere in the committed notebook (checked by direct substring search against
  the loaded env value); config is displayed only through `pplot.redacted_config(cfg)`.
- Notebook executed from a clean state via `jupyter execute --inplace`: 10 code
  cells, execution counts 1-10, **0 error outputs**, 4 figures, committed with outputs.

## Rendering — every figure was extracted and looked at

Per the 06-03 lesson, all four committed figures were decoded out of the executed
notebook and visually inspected. Two layout fixes came out of that inspection rather
than out of the plan:

1. **`plot_proba_over_time`'s legend moved below the axes.** A stacked area chart
   fills its entire frame, so the first draft's `loc="upper left"` legend sat
   directly on top of the state-4 band it was labelling. Every `<verify>` still
   passed — a Figure came back either way.
2. **`plot_calibration_curve`'s axis limits extended a hair past [0, 1].** The real
   artifact contains bins at exactly `observed_freq = 0.0` and `= 1.0`; at limits
   of exactly `(0, 1)` those markers were drawn half-clipped by the frame.

The 5×5 confusion heatmap was checked for text/color contrast (cell text flips
white above half the max count) and the calibration legend was checked against the
plotted lines for occlusion — neither has any.

## Deviations from Plan

### Implementation choices the plan left open or got slightly wrong about the data

**1. `fit_nowcaster_diagnostics` narrows to the lean feature set before building
the training set.** The plan's `<verify>` passes the whole 53-column
`monthly_features` frame straight through, and its action block does not name a
narrowing step — but `fit_nowcaster` drops every row with *any* non-finite feature,
and with the 22 ETF/price columns present that would discard nearly the whole
history before the latest-starting column. Narrowing to
`taxonomy.lean_feature_set(cfg)` (13 columns) is what `backtest/driver.py::_refit_l1`
/`_refit_l2` do, and the plan's own `read_first` names `lean_feature_set` for exactly
this. Result: 360 scored months rather than a handful. A dedicated test
(`test_features_are_narrowed_to_the_lean_taxonomy_set`) pins the behavior.

**2. The returned `X`/`y`/`y_pred`/`proba` are the finite-row subset actually
scored.** The plan's action block says `y_pred = pd.Series(model.predict(X),
index=y.index)` against the same `X` passed to `fit_nowcaster`. That raises:
`fit_nowcaster` drops non-finite rows internally and returns only the model, so
`predict` on the unfiltered frame hits the NaN months (VIX starts 1990-01). The
same mask is therefore applied for scoring, and the scored frames are what the dict
returns — which is also what the notebook needs, since
`plot_proba_over_time(result["X"].index, result["proba"], ...)` requires the index
length and the proba row count to agree. Documented in the function docstring.

**3. `plot_confusion_matrix` uses matplotlib `imshow`, not a seaborn heatmap.**
06-04 established that platform plotting submodules own no plotting-library import
and reach matplotlib only through `core.plt`; an AST test enforces it. A seaborn
heatmap would need a direct seaborn import and break that boundary.

### Structural

**4. TDD RED/GREEN commits were not split.** Tasks 1 and 2 carry `tdd="true"`, but
this phase's established convention (06-01, 06-03, 06-04) is one commit per task
rather than a `test:`/`feat:` pair, and the plan's acceptance criteria are framed
per task. Tests and implementation were written and iterated together; every
behavior case named in the `<behavior>` blocks exists as a test. Unlike 06-04, the
three tasks here **did** land as three separate commits.

No auto-fix under deviation Rules 1-3 was needed: no bug, no missing critical
functionality, no blocking issue. Every numeric reference the plan asserted was
factually correct about the data this time — 0.208746, no-skill 0.16, 21×7
calibration, 20×3 confusion pivoting to 5×5, 776 full-span rows — each verified
before being relied on.

## Known Stubs

None.

## Threat Flags

None. This plan added no network endpoint, no auth path, and no schema at a trust
boundary. It added one *fit* path, which is precisely why T-06-20's no-side-effect
proof is a hard test rather than a comment: the only new write is the committed
notebook itself, scanned for secret leakage (see Verification Results).

## Issues Encountered

None blocking. Reconfirming the standing trap for 06-06 / 06-07: `tests/conftest.py`
does not seed real `data/holdout/` content, so any test asserting on the real
776-row full span sees 708 under pytest. Every test here stays on synthetic frames;
the live numbers were verified through the plan's own `python -c` `<verify>` scripts,
which run outside pytest.

## Next Phase Readiness — what 06-06 / 06-07 need to know

- **`platform/plotting/__init__.py` is still untouched.** `nowcaster.py` is imported
  by submodule path (`from trading_crab_lib.platform.plotting import nowcaster as
  pnowcaster`) and a test asserts `pplot` does not re-export
  `fit_nowcaster_diagnostics`. Add `allocation.py` / `backtest.py` the same way.
- **The Brier finding is P4's to state, and it is now stated.** 06-07's P6 should
  reference it rather than re-derive it; `drift.assert_brier_plausible` remains the
  single source of the `(K−1)/K²` floor.
- **`core.A13_CAVEAT` now has two live callers** (P3 and P4). P6 is the third and
  last; `test_a13_discipline_notebooks_mention_audit_item[P6_backtest_evaluation.ipynb]`
  is the **one remaining skip in the whole suite** and will convert when P6 lands.
- **P4 loads only the three `model_metrics_*` artifacts.** P6 owns
  `backtest_kpi_table.parquet`, the equity curves, and the §5.4 sojourn/lag
  headline — no contention.
- **The persistence-trap split (0.714 / 0.525 / 0.738) is an in-process diagnostic
  number, not a walk-forward number.** Do not quote it in P6 as walk-forward
  accuracy; P6's per-step metrics are a different computation.
- **`test_platform_notebooks.py` still needs zero edits.** P4 joined its glob
  automatically. Its constraints are unchanged, and note the additional ones this
  plan's own test file adds for its notebook: no `append_trial`, no `save_model`,
  no `evaluate_nowcaster`, no `run_backtest(`, no `run_full_backtest_evaluation`
  in any code cell.
- **Trailing semicolons on plot cells** (the P2 convention) work as intended — all
  four P4 figures appear exactly once each, as `display_data`.

## Self-Check: PASSED

- `src/trading_crab_lib/platform/plotting/nowcaster.py` — FOUND
- `notebooks/platform/P4_nowcaster.ipynb` — FOUND
- `tests/unit/test_platform_plotting_nowcaster.py` — FOUND
- commit `eeaae01` — FOUND
- commit `804a1d3` — FOUND
- commit `a386737` — FOUND

---
*Phase: 06-platform-notebook-suite*
*Completed: 2026-09-10*
