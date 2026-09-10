---
phase: 06-platform-notebook-suite
plan: 06
subsystem: platform-allocation-notebook
tags: [matplotlib, nbformat, weight-bands, long-only, vol-targeting, regime-conditional-sharpe, ewma-vol, read-only-universe]

# Dependency graph
requires:
  - phase: 06-platform-notebook-suite
    provides: "plan 06-01 — platform/plotting/{core,loaders,drift}.py (assert_portfolio_weights_plausible, drift_report, load_platform_checkpoint, load_full_span_checkpoint, compute_regime_labeling, redacted_config), tests/unit/test_platform_notebooks.py"
  - phase: 04-assets-and-allocation
    provides: "platform/assets/{returns,vol}.py — compute_monthly_returns, returns_by_regime_stats, ewma_vol, MONTHLY_ANNUALIZATION; platform/allocation/tilt.py — vol_targeted_tilt"
  - phase: 01-monthly-data-layer
    provides: "platform/splice.py build_core_research_series; monthly_raw / monthly_features checkpoints"
provides:
  - "platform/plotting/allocation.py: investable_asset_returns, plot_returns_by_regime_heatmap, plot_ewma_vol_timeline, compute_smoothed_tilt_weights_over_time, plot_tilt_weights_over_time"
  - "notebooks/platform/P5_assets_allocation.ipynb — regime-conditional Sharpe/mean heatmaps, EWMA vol timeline, plausibility-checked tilt weights over time, full-span asset-return drift; executed with outputs"
  - "tests/unit/test_platform_plotting_allocation.py — 42 tests incl. the per-row weight-band pass-through, the D-01 boundary closure, the T-06-24 no-persistence proof, and 7 P5 notebook source-discipline assertions"
affects: [06-07-backtest-notebook]

# Tech tracking
tech-stack:
  added: []  # zero new dependencies
  patterns:
    - "a read-only re-derivation: investable_asset_returns mirrors report.py's splice_cfg dispatch using only public functions, so the notebook's universe is provably the backtest's rather than a hardcoded ticker list that would silently drift"
    - "heatmap cell text flips to white past 0.6x the symmetric color limit — black on the saturated end of RdYlGn is legible only just, and a regime-conditional Sharpe table is read for its extremes"
    - "the plausibility guard runs over EVERY rendered row, not only the last one, and runs BEFORE the chart is drawn — a normalization defect stops the notebook instead of producing a plausible-looking chart of invalid weights"
    - "the illustrative caveat lives in the plot's own title, not only in the surrounding markdown: a weights chart is the artifact most likely to be screenshotted away from its context (T-06-25)"
    - "AST-level (not substring) assertions for 'imports no CheckpointManager' / 'never calls run_backtest' — the module docstring names both by design, so a substring check would fail on its own documentation"

key-files:
  created:
    - src/trading_crab_lib/platform/plotting/allocation.py
    - notebooks/platform/P5_assets_allocation.ipynb
    - tests/unit/test_platform_plotting_allocation.py
  modified: []

key-decisions:
  - "The D-01/T-06-24 source assertions are AST-based, not substring-based. allocation.py's module docstring deliberately states 'It calls neither run_backtest nor run_full_backtest_evaluation' and 'It imports no CheckpointManager' — a naive substring scan fails on the very documentation that records the constraint. The tests parse the module with ast and check imported names / called names, which is also the literal reading of the plan's criterion ('contains no CheckpointManager import')."
  - "plot_returns_by_regime_heatmap uses matplotlib imshow with a symmetric diverging scale rather than a seaborn heatmap — 06-04/06-05 established that platform plotting submodules own no plotting-library import and reach matplotlib only through core.plt, and an AST test in this plan's own file enforces it."
  - "compute_smoothed_tilt_weights_over_time skips the first `min_obs` months of `states` as warmup (the plan's 'starting after the first min_obs months') and returns a frame indexed by the visited dates only — 360 of the labeling's 372 months on live data."

requirements-completed: [NB-01]

coverage:
  - id: A1
    description: "P5 shows regime-conditional statistics for the SAME 4-asset universe run_backtest tilts across, computed via returns_by_regime_stats, never a hand-rolled parallel statistic"
    requirement: "NB-01"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_plotting_allocation.py::TestInvestableAssetReturns (4 tests), TestNoAllocationMathIsReDerived::test_calls_vol_targeted_tilt_and_returns_by_regime_stats, TestP5NotebookSource::test_hardcodes_no_ticker_list"
        status: pass
      - kind: manual
        ref: "plan Task 1 <verify> run verbatim against the real monthly_raw checkpoint — 'P5 universe/vol smoke OK (776, 4)', columns exactly {SPY, TLT, IAU, USO}, cash_ret length 776 == len(monthly_raw)"
        status: pass
    human_judgment: false
  - id: A2
    description: "Each asset's trailing annualized EWMA vol is shown over the full span so a vol regime shift is visible by eye"
    requirement: "NB-01"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_plotting_allocation.py::TestPlotEwmaVolTimeline (5 tests, incl. the ragged-inception case asserting all 4 lines are still drawn)"
        status: pass
      - kind: manual
        ref: "executed notebook cell 11 — figure decoded and visually inspected; SPY/TLT run from 1962, IAU/USO from 1985, USO's 2020 spike reaches 1.19 annualized"
        status: pass
    human_judgment: true
  - id: A3
    description: "Every rendered weight row satisfies drift.assert_portfolio_weights_plausible — each weight in [0,1], assets plus cash summing to 1.0"
    requirement: "NB-01"
    verification:
      - kind: unit
        ref: "TestComputeSmoothedTiltWeightsOverTime::test_every_row_passes_the_exact_weight_band_contract (calls assert_portfolio_weights_plausible per row directly), ::test_rows_sum_to_one_and_carry_every_asset_plus_cash, ::test_no_weight_is_negative_long_only_by_design"
        status: pass
      - kind: manual
        ref: "executed notebook cell 13 — 'all 360 rendered rows pass the same check'; sum(assets)+cash = 1.000000000000 on the last row"
        status: pass
    human_judgment: false
  - id: A4
    description: "The weights-over-time panel is explicitly labeled a smoothed/hindsight illustration, distinct from a live weekly recommendation"
    requirement: "NB-01"
    verification:
      - kind: unit
        ref: "TestPlotTiltWeightsOverTime::test_title_states_the_illustrative_caveat, TestP5NotebookSource::test_states_the_weights_panel_is_illustrative"
        status: pass
    human_judgment: false
  - id: A5
    description: "An asset whose current-era return distribution has shifted relative to the pre-2021 fitted window appears in P5's drift table"
    requirement: "NB-01"
    verification:
      - kind: manual
        ref: "executed notebook cell 16 — 776 full-span rows through 2026-08-31; all 4 assets ranked by |standardized mean shift|, 0 flagged (TLT largest at -0.383). See 'The drift finding, as measured' below."
        status: pass
    human_judgment: true
  - id: A6
    description: "P5 runs top-to-bottom against the real monthly_raw/monthly_features checkpoints without raising"
    requirement: "NB-01"
    verification:
      - kind: manual
        ref: "executed via `jupyter execute --inplace` this session — 10 code cells, execution counts 1-10, 0 error outputs, 4 figures, 8.2s wall clock; outputs committed. Every figure decoded and visually inspected."
        status: pass
    human_judgment: true
  - id: A7
    description: "P5 carries no sign-off cell and no per-run gate (D-15 stays P3-exclusive), and writes no production checkpoint"
    requirement: "NB-01"
    verification:
      - kind: unit
        ref: "TestP5NotebookSource::test_carries_no_sign_off_cell, ::test_never_writes_a_checkpoint_or_a_trial, ::test_never_invokes_the_full_walk_forward_entrypoints; TestNeverPersistsAnything (3 tests on the module)"
        status: pass
      - kind: manual
        ref: "git status --porcelain data/checkpoints/platform/ outputs/reports/platform/ registry/ src/trading_crab_lib/platform/{allocation,assets}/ -> empty after executing the notebook end to end"
        status: pass
    human_judgment: false

actuals:
  tokens: 58000
  tasks: 3
  commits: 3

metrics:
  duration_minutes: 30
  completed_date: 2026-09-10
status: complete
---

# Phase 6 Plan 06: P5 Assets & Allocation Notebook Summary

Built `platform/plotting/allocation.py` — a read-only re-derivation of the real
investable universe, the regime-conditional returns heatmap, the EWMA vol
timeline, and the smoothed-labeling tilt weights over history — plus
`P5_assets_allocation.ipynb`, executed against real data with outputs committed.

Every one of the 360 rendered weight rows is routed through
`drift.assert_portfolio_weights_plausible` **before** any chart is drawn.

## Task Commits

1. **Task 1: investable universe + L3 regime/vol panels**
   - `8bd42ac` (feat) — `src/trading_crab_lib/platform/plotting/allocation.py`,
     `tests/unit/test_platform_plotting_allocation.py` (20 tests at that point)
2. **Task 2: smoothed-labeling tilt weights + the exact weight-band contract**
   - `eec7fe2` (feat) — same two files extended (35 tests)
3. **Task 3: the P5 notebook**
   - `bc964f8` (feat) — `notebooks/platform/P5_assets_allocation.ipynb`
     (18 cells, executed) plus 7 P5 source-discipline tests (42 tests)

## The investable universe, as re-derived

`investable_asset_returns(monthly_raw, cfg)` against the real 776x45
`monthly_raw` checkpoint:

| ticker | research class | method | first observed month |
|---|---|---|---|
| **SPY** | `equities_tr` | `total_return_from_price_div` | 1962-02-28 |
| **TLT** | `long_duration_tr` | `cmt_par_bond_repricing` | 1962-02-28 |
| **IAU** | `gold` | `single_source` (`gold_spot`, candidate 1 of 2) | 1985-03-31 |
| **USO** | `oil` | `single_source` (`wti_crude`, candidate 1 of 2) | 1985-03-31 |

`asset_returns` is (776, 4); `cash_ret` is 776 months. **Cash (FZFXX) is
excluded from the tilt by design** — it is the vol-target residual that earns
`cash_ret`, never a tilted risk position. Nothing in the notebook types a
ticker out: a test (`test_hardcodes_no_ticker_list`) asserts no code cell
contains a literal `"SPY"`/`"IAU"`, so the day a splice class is added, renamed
or falls back to a different source, P5 follows the config rather than lying.

The `optional: true` fallback path is covered by a dedicated test: dropping
`gold_spot` from the raw frame yields `{SPY, TLT, USO}` plus a WARNING naming
the excluded class, mirroring `report.py`'s own `_excluded` handling.

## The weight bands, as measured on real data

360 monthly weight rows, 1991-01-31 -> 2020-12-31 (the labeling spans 372
months from 1990-01; the first 12 are `portfolio_vol_min_obs` warmup):

| check | result |
|---|---|
| rows passing `assert_portfolio_weights_plausible` | **360 / 360** |
| `sum(weights) + cash` range across all rows | `[0.9999999999999999, 1.0000000000000002]` |
| minimum weight anywhere | `0.0` |
| maximum weight anywhere | `0.664` |
| any negative weight | **none** (long-only by design) |
| last row (2020-12-31) | SPY 0.2208, TLT 0.2226, IAU 0.2133, USO 0.0828, cash 0.2605 — sums to `1.000000000000` |

The guard runs over **every** rendered row, not only the last one, and it runs
*before* `plot_tilt_weights_over_time` is called. This is the mechanism that
would catch a normalization defect in `allocation/tilt.py` rather than
rendering a plausible-looking chart of invalid weights.

Long-only is a project-level constraint (no shorts, no options — PROJECT.md),
not a preference, and `drift.assert_portfolio_weights_plausible` is the single
source of the rule. Neither `allocation.py` nor the notebook re-derives it.

The chart itself is economically coherent as a sanity check: cash is 0.0 for
most of the 1990s, spikes to ~0.30 through 2008-2010, and spikes again in 2020
— exactly where vol targeting should be pulling risk off. IAU and USO carry
zero weight before their 1985 inception (they carry weight throughout the
rendered 1991+ window).

## The drift finding, as measured

Full span 776 rows through 2026-08-31, baseline = pre-2021 fitted window:

| asset | n_baseline | n_current | baseline_mean | current_mean | standardized shift | flagged |
|---|---|---|---|---|---|---|
| TLT | 707 | 68 | 0.00554 | **−0.00183** | **−0.383** | False |
| IAU | 430 | 68 | 0.00534 | 0.01381 | +0.191 | False |
| SPY | 707 | 68 | 0.00872 | 0.01256 | +0.107 | False |
| USO | 430 | 68 | 0.00718 | 0.01424 | +0.065 | False |

**Zero assets are flagged today.** That is the honest current reading, and the
table is rendered ranked regardless so the largest mover is visible whether or
not it crosses a threshold. TLT is the one worth an operator's attention: its
post-2020 mean monthly return is *negative* where the pre-2021 window it was
Sharpe-ranked on was +0.55%/month — a −0.38 sigma shift with a KS p-value of
0.035. The tilt currently ranks TLT highly in states 1 and 2 on the strength of
that pre-2021 window.

Per D-07 I have **not** written a `.planning/POST-2020-OBSERVATIONS.md` entry.
Deciding an observation is decision-changing is the operator's judgement, not
mine; the table is the surface that makes the call possible.

## Verification Results

- `pytest tests/unit/test_platform_plotting_allocation.py -q` → **42 passed**.
- `pytest tests/unit/test_platform_plotting_allocation.py tests/unit/test_platform_notebooks.py -q`
  → **70 passed, 1 skipped** (P5 joined the notebook-gate glob automatically;
  the single skip is P6's A13 guard, 06-07's, not this plan's).
- **Full suite: `pytest tests/ -q` → 1660 passed, 1 skipped, 0 failures** (~81s).
  Baseline entering this plan was 1613 passed / 1 skipped. **Net +47 tests,
  zero regressions, zero skips added, nothing weakened or xfailed.**
- `ruff check` clean on both new source/test files.
- Both plan `<verify>` scripts ran verbatim and printed their sentinels:
  `P5 universe/vol smoke OK (776, 4)` and `P5 static checks OK 18 cells`.
- `git status --porcelain src/trading_crab_lib/platform/allocation/
  src/trading_crab_lib/platform/assets/ outputs/reports/platform/
  data/checkpoints/platform/ registry/` → **empty**. Phase 1/4/5 code and every
  reference artifact were read, never modified.
- Secret hygiene (T-06-01/T-06-23): the live `FRED_API_KEY` value does not
  appear anywhere in the committed notebook (checked by direct substring search
  against the loaded env value); config is displayed only through
  `pplot.redacted_config(cfg)`.
- Notebook executed from a clean state via `jupyter execute --inplace`:
  10 code cells, execution counts 1-10, **0 error outputs**, 4 figures,
  8.2s wall clock, committed with outputs.

## Rendering — every figure was extracted and looked at

All four committed figures were decoded out of the executed notebook and
visually inspected. One fix came out of that inspection rather than out of the
plan:

**Heatmap cell text on the saturated ends of RdYlGn.** The first draft drew
every annotation in black. The live table's extremes (SPY 1.728 in state 4,
SPY −1.182 in state 0) sit on dark green / dark red, where black text is
legible only just. Text now flips to white past `0.6 * limit`, mirroring the
confusion-matrix convention 06-05 established. The `<verify>` passed either way
— a Figure came back regardless.

The stacked-area weights panel was checked specifically for the two failure
modes this phase has already hit: the legend sits **below** the axes (a stacked
area fills its whole frame, so an in-axes legend occludes the bands it labels —
06-05's finding), and the band ordering puts cash last in neutral gray so it
never reads as one of the four regime-palette assets. The EWMA vol timeline's
legend is below the axes for the same reason: vol spikes reach the top corners
of the frame at exactly the crisis months an operator most wants to read.

All four figures appear exactly once each as `display_data` — the trailing-`;`
convention works as intended.

## Deviations from Plan

### Implementation choices the plan left open

**1. The D-01 / T-06-24 source assertions are AST-based, not substring-based.**
The plan's acceptance criterion says "a test asserts `allocation.py` contains no
`CheckpointManager` import", and its prohibitions call for "a source assertion
that no P5 cell references either function name". Implemented literally as a
substring scan of the module source, both fail immediately — `allocation.py`'s
module docstring deliberately *states* "It imports no `CheckpointManager`" and
"It calls neither `run_backtest` nor `run_full_backtest_evaluation`", which is
exactly the documentation that records the constraint. The tests therefore parse
the module with `ast` and check the set of imported names and the set of called
names. Substring scanning is retained where it is safe and stricter — for the
notebook's *code cells* (which carry no such prose) and for the tilt-formula
names (`vol_target_scale`, `regime_tilt_weights`, `portfolio_vol(`) that must
appear nowhere in `allocation.py` at all.

**2. `plot_returns_by_regime_heatmap` uses matplotlib `imshow`, not seaborn.**
Same reason 06-05 recorded: platform plotting submodules own no plotting-library
import and reach matplotlib only through `core.plt`; an AST test in this plan's
own file enforces it, and a seaborn heatmap would need a direct seaborn import.

**3. TDD RED/GREEN commits were not split.** Tasks 1 and 2 carry `tdd="true"`,
but this phase's established convention (06-01, 06-03, 06-04, 06-05) is one
commit per task rather than a `test:`/`feat:` pair, and the plan's acceptance
criteria are framed per task. Tests and implementation were written and iterated
together; every behavior case named in the `<behavior>` blocks exists as a test.
The three tasks landed as three separate commits.

### Correction to a plan expectation about the data

**4. The plan's must-have truth "An asset's return distribution whose current-era
mean/vol has shifted relative to the pre-2021 fitted window appears in P5's
drift table" is satisfied structurally, but zero assets are currently flagged.**
The drift table renders all four assets ranked by absolute standardized shift
whether or not any crosses the flag threshold, so a shifted asset *would* appear
— and TLT's −0.383 shift with a negative current-era mean is visible at the top
of the table. But no row has `flag == True` today, unlike 06-04/06-05's
lean-feature table where 6 of 13 features flag. Reporting the real zero rather
than implying a flag exists.

No auto-fix under deviation Rules 1-3 was needed: no bug, no missing critical
functionality, no blocking issue in library code.

## Known Stubs

None.

## Threat Flags

None. This plan added no network endpoint, no auth path, and no schema at a
trust boundary. It added no *write* path of any kind — `allocation.py` is pure
(DataFrame in, DataFrame/Figure out) and imports no `CheckpointManager`, which
is why T-06-24's proof is three hard tests rather than a comment. The only new
committed artifact is the notebook itself, scanned for secret leakage (see
Verification Results).

## Issues Encountered

None blocking. Reconfirming the standing trap for 06-07: `tests/conftest.py`
does not seed real `data/holdout/` content, so any test asserting on the real
776-row full span sees 708 under pytest. Every test here stays on synthetic
frames; the live numbers were verified through the plan's own `python -c`
`<verify>` scripts and direct notebook execution, both of which run outside
pytest.

## Next Phase Readiness — what 06-07 (P6, the last plan) needs to know

- **`platform/plotting/__init__.py` is still untouched.** `allocation.py` is
  imported by submodule path (`from trading_crab_lib.platform.plotting import
  allocation as pallocation`) and a test asserts `pplot` does not re-export
  `investable_asset_returns`. Add `backtest.py` the same way.
- **P6 owns the only remaining skip in the suite.**
  `test_a13_discipline_notebooks_mention_audit_item[P6_backtest_evaluation.ipynb]`
  converts from skip to pass when P6 lands. Suite entering 06-07: **1660 passed,
  1 skipped**.
- **`investable_asset_returns` is reusable and already proven against real
  data.** If P6 needs the asset universe (e.g. to align an equity curve or a
  per-leg attribution), import it rather than re-deriving a third copy —
  `from trading_crab_lib.platform.plotting import allocation as pallocation`.
- **The tilt weights P5 shows are SMOOTHED, not walk-forward.** 360 rows,
  1991-2020, driven by a one-hot on the full-sample smoothed state with
  full-sample regime stats. Do **not** quote them in P6 as what the strategy
  held. P6's equity curve and per-step metrics are a different computation, and
  the *comparison* between the two is precisely the smoothed-vs-filtered gap
  P6 is responsible for framing (with `core.A13_CAVEAT` attached).
- **P5 loads only `monthly_raw`, `monthly_features`, and the full-span
  `monthly_raw`.** It touches none of `backtest_kpi_table.parquet`, the equity
  curves, `backtest_full_sample_states.parquet`, or
  `backtest_filtered_state_probs.parquet` — no contention with P6.
- **`assert_kpi_table_plausible` (06-01) is still unused by any notebook.** P6
  is its first and only intended caller; 06-01 verified it returns a 10-row
  verdict frame (5 legs x 2 metrics, all `pass`) against the live KPI table.
- **P5 carries no sign-off cell.** D-15 remains P3-exclusive, and P6 should not
  add one either.
- **Notebook-gate constraints are unchanged and need zero edits** — P5 joined
  the glob automatically. Note the extra ones this plan's own test file adds for
  its notebook: no `Sign-Off`, no `run_backtest(` / `run_full_backtest_evaluation`,
  no `.save(` / `append_trial` / `report_returns_by_regime`, no hardcoded ticker
  literal, and no `target_vol_annual=` / `halflife=` / `K=` hyperparameter
  override in any code cell (D-14).

## Self-Check: PASSED

- `src/trading_crab_lib/platform/plotting/allocation.py` — FOUND
- `notebooks/platform/P5_assets_allocation.ipynb` — FOUND
- `tests/unit/test_platform_plotting_allocation.py` — FOUND
- commit `8bd42ac` — FOUND
- commit `eec7fe2` — FOUND
- commit `bc964f8` — FOUND

---
*Phase: 06-platform-notebook-suite*
*Completed: 2026-09-10*
