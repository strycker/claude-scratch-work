---
phase: 06-platform-notebook-suite
plan: 07
subsystem: platform-backtest-notebook
tags: [matplotlib, nbformat, equity-curves, calibration, sojourn-lag, a13-caveat, deterministic-baselines, honesty-framework]

# Dependency graph
requires:
  - phase: 06-platform-notebook-suite
    provides: "plan 06-01 — platform/plotting/{core,loaders,drift}.py (A13_CAVEAT, assert_kpi_table_plausible, assert_brier_plausible, load_report_artifact, load_full_sample_states, load_filtered_state_probs, load_full_span_checkpoint, redacted_config), tests/unit/test_platform_notebooks.py"
  - phase: 06-platform-notebook-suite
    provides: "plan 06-02 — backtest_full_sample_states.parquet / backtest_filtered_state_probs.parquet, the two artifacts the sojourn/lag headline recomputes from"
  - phase: 05-evaluation-and-reporting (Phase 5, referenced)
    provides: "backtest_equity_curve_strategy.parquet, backtest_equity_curve_ablation.parquet, backtest_kpi_table.parquet, model_metrics_calibration.parquet, model_metrics_brier.parquet — the five persisted evaluation artifacts P6 reads"
provides:
  - "platform/plotting/backtest.py: recompute_baseline_curves, plot_equity_curves, compute_ablation_delta, plot_kpi_table_bars, plot_calibration_summary, plot_sojourn_lag_headline (all landed under commit faa0e35, prior session)"
  - "notebooks/platform/P6_backtest_evaluation.ipynb — the phase's evaluation capstone: five-leg equity curves, KPI gauntlet, ablation delta, calibration summary, sojourn/lag headline with A13 caveat; executed with outputs, committed"
  - "tests/unit/test_platform_plotting_backtest.py — 31 tests (prior session), now exercised end-to-end by the committed notebook"
affects: []  # last plan of Phase 6 — closes the phase

# Actuals (#2632)
actuals:
  tokens: 8000   # this session's work: verification, no source/test edits; notebook content itself authored in the prior (rate-limited) session
  tasks: 1        # only Task 3 (commit + close-out) was outstanding on arrival; Tasks 1-2 were already committed (faa0e35)
  commits: 1

tech-stack:
  added: []
  patterns:
    - "resumed-plan closeout: verify a prior executor's uncommitted, already-executed artifact against every must-have and prohibition before committing, rather than re-deriving or re-executing it"
    - "figure-by-figure visual audit as a standing discipline: all 5 committed PNGs decoded from the notebook's own output cells and inspected, catching the exact defect class (legend occlusion, flattened bars, illegible text) that passed <verify> in 4 prior plans"

key-files:
  created:
    - notebooks/platform/P6_backtest_evaluation.ipynb
  modified: []

key-decisions:
  - "No code changes were needed. Both open items from the handoff (resolved-transition count rendering, ablation-delta value rendering) were already present and correct in the executed notebook — verified by decoding cell text output and the headline figure's own text artists, not just re-reading source."
  - "All five figures were extracted from the notebook's embedded PNG outputs and visually inspected before committing, per this plan's own standing lesson (06-03 through 06-06 each shipped a rendering defect that a passing <verify> did not catch). None found here — see 'Rendering' below."

requirements-completed: [NB-01]

coverage:
  - id: A1
    description: "P6 renders all five legs' equity curves (strategy, no-regime ablation, SPY buy-and-hold, 60/40, Faber SMA) on one chart, twice — over each leg's own span and over the common comparable span"
    requirement: "NB-01"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_plotting_backtest.py::TestPlotEquityCurves (5-leg + empty-dict cases)"
        status: pass
      - kind: manual
        ref: "cell4_fig1.png / cell6_fig2.png decoded and visually inspected — 5 distinguishable line styles (solid bold blue, dashed purple, solid red, dash-dot orange, dotted green), legend below the axes with terminal values annotated, linear y-axis showing real separation between legs, no occlusion"
        status: pass
    human_judgment: true
  - id: A2
    description: "The KPI table renders under assert_kpi_table_plausible gated BEFORE the bar chart, and the no-regime-ablation delta is computed live from the KPI table (never hardcoded)"
    requirement: "NB-01"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_plotting_backtest.py::TestComputeAblationDelta, TestPlotKpiTableBars"
        status: pass
      - kind: manual
        ref: "cell 8/9 output: verdict table all 'pass' before fig_kpi renders; cell 13 output: 'wealth_delta: +0.379267', 'dd_delta: -0.014364' printed live from compute_ablation_delta(kpi_table)"
        status: pass
    human_judgment: false
  - id: A3
    description: "The calibration summary renders from the same persisted model_metrics_calibration.parquet P4 loads, gated on assert_brier_plausible"
    requirement: "NB-01"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_plotting_backtest.py::TestPlotCalibrationSummary"
        status: pass
      - kind: manual
        ref: "cell 16/17 output + cell17_fig4.png decoded: brier_verdict printed (brier=0.208746, beats_no_skill=False) before the 21-point scatter renders; 5 states color-coded, point size ~ n_in_bin, 45-degree reference line legible"
        status: pass
    human_judgment: true
  - id: A4
    description: "The sojourn/lag headline renders the resolved-transition count (4 of 6) and the ratio, with core.A13_CAVEAT rendered verbatim and unconditionally — both in prose before any number and inside the figure itself"
    requirement: "NB-01"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_plotting_backtest.py::TestPlotSojournLagHeadline (incl. NaN-ratio does-not-crash case, text-content assertion for 'A13' + '4 of 6')"
        status: pass
      - kind: manual
        ref: "cell 20 stream output prints 'resolved : n_resolved=4 of n_transitions=6 transitions (the ratio is a median over 4 observations)'; cell21_fig5.png decoded — subtitle reads '4 of 6 transitions resolved', caveat box below the bars reads 'NOT INTERPRETABLE (audit item A13): ...' verbatim"
        status: pass
    human_judgment: true
  - id: A5
    description: "No void historical KPI figure (111.06 log wealth, Faber -99.7%, 60/40 -2.3%/-2.27%) is cited anywhere in the notebook as current; every KPI value is described as read live from backtest_kpi_table.parquet"
    requirement: "NB-01"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_notebooks.py (P6 static source checks, parameterized)"
        status: pass
      - kind: other
        ref: "grep -c '111.06\\|99.7%\\|-2.3%\\|-2.27%' notebooks/platform/P6_backtest_evaluation.ipynb -> 0"
        status: pass
    human_judgment: false
  - id: A6
    description: "P6 runs top-to-bottom against real checkpoints and all nine persisted outputs/reports/platform/ artifacts without raising, carries no sign-off cell, calls neither run_backtest() nor run_full_backtest_evaluation(), and writes to no production checkpoint or report artifact"
    requirement: "NB-01"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_notebooks.py -x -q -> 73 passed (incl. the cross-notebook A13 discipline check, which converts from skip to pass with this notebook's arrival)"
        status: pass
      - kind: manual
        ref: "26 cells, 14 code cells, 0 error outputs, 5 figures (verified this session); git status --porcelain over evaluation/, backtest/, outputs/reports/platform/, data/checkpoints/platform/, registry/ -> empty"
        status: pass
    human_judgment: false

metrics:
  duration_minutes: 20
  completed_date: 2026-09-10
status: complete
---

# Phase 6 Plan 07: P6 Backtest Evaluation Notebook Summary (Phase 6 close-out)

**Verified and committed the phase's evaluation capstone notebook — all five ROADMAP criterion-5 panels confirmed to render correctly, including the two items a prior executor's context-limited handoff could not confirm from outside: the sojourn/lag resolved-transition count (4 of 6) and the live ablation-delta value (+0.3793), both present exactly as required.**

## Performance

- **Duration:** ~20 min
- **Completed:** 2026-09-10
- **Tasks:** 1 of 3 (Tasks 1-2 were already committed under `faa0e35` by a prior, rate-limited executor session; this session's job was Task 3's close-out — verify, fix if needed, commit, summarize)
- **Files modified:** 1 new file committed (`notebooks/platform/P6_backtest_evaluation.ipynb`); zero source or test files touched this session

## What I found on arrival

The handoff was accurate. The uncommitted notebook (540KB, 26 cells, 14 code
cells, all executed, 0 error outputs, 5 figures) already satisfied every
must-have in `06-07-PLAN.md`. I did not need to add code, re-execute the
notebook, or touch `backtest.py` / its test file. My work was verification,
not authorship:

1. **Decoded and read every cell's source and output text** — not just
   grepped for tokens, but read the printed values to confirm they say what
   the plan requires.
2. **Extracted all 5 embedded PNG figures and visually inspected each one**,
   per this plan's own standing lesson (06-03 through 06-06 each shipped a
   rendering defect — invisible recession shading, bars collapsed onto zero,
   an occluding legend, illegible dark-cell text — that passed a green
   `<verify>` because "returns a Figure" is not "is legible"). None of that
   class of defect is present here.
3. **Ran the full test suite** and the plan's static-source `<verify>` script
   verbatim.
4. **Committed** the notebook and wrote this summary.

No code changes were needed anywhere. This is itself worth stating plainly:
the previous session's work was correct and complete; the interruption was
purely a rate-limit, not an unfinished implementation.

## The two items the handoff could not confirm

**1. The resolved-transition count (4 of 6).** Confirmed rendering in two
independent places:
- Cell 20's stream output: `resolved : n_resolved=4 of n_transitions=6
  transitions (the ratio is a median over 4 observations)` — the explicit
  callout that a median over four observations is a materially different
  object than one over sixty is present verbatim, word for word.
- The headline figure itself (`cell21_fig5.png`): the subtitle line reads
  `sojourn/lag ratio = 0.5915  |  4 of 6 transitions resolved  |
  act_threshold = 0.7`, directly under the chart title, above the bars.

**2. The ablation delta (+0.3793).** Confirmed live-computed, not hardcoded:
cell 13's output prints `wealth_delta: +0.379267` and `dd_delta: -0.014364`
directly from `pbacktest.compute_ablation_delta(kpi_table)`'s return dict —
the plan's preferred approach (computed at render time from the live table)
rather than a hardcoded expectation. Both signs are interpreted correctly in
prose immediately after: the regime layer added terminal wealth (True) but
did **not** reduce max drawdown (False) relative to its own regime-free twin.

Both items render exactly as required. No fix was needed.

## Rendering — every figure was decoded and looked at

All 5 figures extracted from the committed notebook's `display_data` outputs
and inspected:

| # | cell | figure | what it shows | verdict |
|---|---|---|---|---|
| 1 | 4 | 5-leg equity curves, own span | strategy (bold blue solid), no_regime_ablation (dashed purple), spy_buy_hold (solid red), sixty_forty (dash-dot orange), faber_sma (dotted green); legend below axes with terminal values annotated | legible — 5 distinct line styles + colors, no occlusion, linear axis preserves real separation |
| 2 | 6 | same 5 legs, common span (1972-2020) | same styling, re-anchored to the comparable window | legible, consistent with #1 |
| 3 | 9 | KPI bars (terminal log wealth / max drawdown) | grouped bars, one subplot per metric, value labels on every bar | legible — no bars collapsed to zero, all 5 legs visible at distinct heights |
| 4 | 17 | calibration summary (21 class×bin points) | scatter, point size ∝ n_in_bin, colored by state (5-color regime palette), 45° reference line | legible — largest points (state 0, low-probability, well-populated bins) don't obscure the diagonal; no text-on-fill contrast issue since this chart carries no in-marker text |
| 5 | 21 | sojourn/lag headline (2-bar + caveat box) | median sojourn / median lag bars, ratio+resolved-count in the subtitle, full A13 caveat text in a bordered box below the axes | legible — caveat box uses black text on light-yellow fill (high contrast), sits below the chart (not overlapping data), and is the single most load-bearing panel in the notebook |

None of the four specific defect patterns named in the task brief (invisible
recession shading, bars collapsed onto zero on a linear axis, a legend
sitting on top of the band it labels, black text on saturated dark cells)
appear in any of these 5 figures. Five overlaid equity curves are the
riskiest case named in the brief for legend occlusion / color confusion / a
flattening log-vs-linear axis; both equity-curve figures use a **linear**
y-axis (cumulative log wealth is already a log-space quantity, so the axis
itself is linear-log-wealth, not a log-scaled axis compounding two log
transforms) and a legend placed below, not inside, the plot area.

## Verification Results

- `.venv/bin/python -m pytest tests/unit/test_platform_plotting_backtest.py tests/unit/test_platform_notebooks.py -q` → **73 passed** (0 skipped — the P6 A13-discipline guard converted from its prior `skip` to a `pass` now that the notebook exists).
- Plan Task 3's static `<verify>` python block ran verbatim: `P6 static checks OK 26 cells`.
- `grep -c '111.06\|99.7%\|-2.3%\|-2.27%' notebooks/platform/P6_backtest_evaluation.ipynb` → **0** — no void historical figure is present anywhere in the committed notebook.
- `git status --porcelain src/trading_crab_lib/platform/evaluation/ src/trading_crab_lib/platform/backtest/ outputs/reports/platform/ data/checkpoints/platform/ registry/` → **empty**. Phase 3/5 code and every reference artifact were read, never modified; nothing was written to a checkpoint, a report artifact, or the registry.
- **Full suite: `pytest tests/ -q` → 1705 passed, 0 skipped, 0 failures** (~92s). Baseline entering this plan was 1660 passed / 1 skipped. Net **+45 tests, the 1 standing skip converted to a pass, zero regressions.**

## Task Commits

1. **Task 1: Deterministic baseline recompute, equity-curve chart, KPI bars, ablation delta** — `faa0e35` (feat, prior session)
2. **Task 2: Calibration summary and the sojourn/lag headline with the mandatory A13 caveat** — `faa0e35` (feat, same commit as Task 1, prior session)
3. **Task 3: P6_backtest_evaluation notebook** — `ab5da0d` (feat, this session)

## Files Created/Modified

- `notebooks/platform/P6_backtest_evaluation.ipynb` — the phase's evaluation capstone; 26 cells (14 code, executed with outputs), all five ROADMAP criterion-5 panels

(`src/trading_crab_lib/platform/plotting/backtest.py` and
`tests/unit/test_platform_plotting_backtest.py` were already committed under
`faa0e35` before this session started; neither was modified this session.)

## Decisions Made

None new. The prior session's implementation choices (documented in the
handoff, not repeated here) stood unchanged after verification.

## Deviations from Plan

None — plan executed exactly as written across both sessions. No auto-fix
under deviation Rules 1-3 was needed this session: nothing was broken,
nothing critical was missing, and no blocking issue surfaced.

## Known Stubs

None.

## Threat Flags

None. This plan closes Phase 6 without adding a new network endpoint, auth
path, or schema at a trust boundary. `T-06-27` (config leak via
`redacted_config`), `T-06-28` (sojourn/lag ratio mistaken for a settled
signal), `T-06-29` (void historical figures mistaken for current), and
`T-06-30` (accidental re-run of the full walk-forward) — all four threats
named in this plan's own threat register — were checked directly against the
committed notebook this session and found mitigated as designed: config is
displayed only through `pplot.redacted_config(cfg)`; the A13 caveat renders
three times (markdown before any number, inside the headline figure
unconditionally, and again in the closing summary) with `n_resolved`/
`n_transitions` printed alongside every appearance of the ratio; zero void
numerals appear in the committed source; no cell calls `run_backtest(` or
`run_full_backtest_evaluation`.

## Issues Encountered

None. The plan's `<verify>` scripts, the full test suite, and the manual
figure-by-figure inspection all passed on the first attempt with the
artifact exactly as the prior session left it.

## Phase 6 close-out — what the phase verifier should check

This is the last plan of Phase 6 (`06-platform-notebook-suite`). For the
phase-level verification pass:

- **All six notebooks now exist, are committed with outputs, and pass
  `tests/unit/test_platform_notebooks.py`'s parameterized suite** — P1
  through P6, 26 cells / 14 code cells in P6 specifically, 73 tests passing
  across the two backtest-plotting + notebook-gate test files, 1705 passing
  across the whole repo.
- **ROADMAP criterion 5's five elements are ALL rendered in P6**, each traced
  to a specific cell/figure in the "Rendering" table above and in the
  `coverage:` block's A1-A4 entries.
- **ROADMAP criterion 3** (every figure in the notebook suite comes from
  `platform/plotting/`) holds for P6: no cell imports `matplotlib` or
  `seaborn` directly (enforced by the plan's own static `<verify>` and by an
  AST-level test in `test_platform_plotting_backtest.py`).
- **D-15 (sign-off cell is P3-exclusive)** holds: P6 carries no `Sign-Off`
  heading, confirmed by both the static `<verify>` script and a dedicated
  unit test.
- **The A13 position is stated, not resolved, everywhere it could plausibly
  be mistaken for resolved.** This is by design (design-locked audit item);
  the phase verifier should not expect or request a numeric resolution here.
- **Left open for the operator (not a defect, a standing decision point):**
  the ablation delta shows the regime layer added terminal wealth (+0.3793
  log wealth) but *increased* max drawdown relative to its own regime-free
  ablation twin (dd_delta -0.0144, i.e., the regime-tilted book's drawdown is
  slightly worse, not better). The notebook states this plainly and scopes
  the claim correctly (this delta says nothing about whether either leg
  beats the baseline gauntlet — Faber and 60/40 both still beat the strategy
  on terminal wealth, and Faber beats it on drawdown too, per the KPI bars).
  Whether this specific dd sign is acceptable given the wealth gain is a
  design/policy call outside this plan's scope.

## Self-Check: PASSED

- `notebooks/platform/P6_backtest_evaluation.ipynb` — FOUND
- `src/trading_crab_lib/platform/plotting/backtest.py` — FOUND (pre-existing, `faa0e35`)
- `tests/unit/test_platform_plotting_backtest.py` — FOUND (pre-existing, `faa0e35`)
- commit `faa0e35` — FOUND
- commit `ab5da0d` — FOUND

---
*Phase: 06-platform-notebook-suite*
*Completed: 2026-09-10*
