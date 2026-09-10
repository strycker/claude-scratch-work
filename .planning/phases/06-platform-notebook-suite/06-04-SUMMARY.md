---
phase: 06-platform-notebook-suite
plan: 04
subsystem: platform-features-notebook
tags: [matplotlib, nbformat, taxonomy, plausibility-bands, drift-detection, A4-regression-guard, causal-only]

# Dependency graph
requires:
  - phase: 06-platform-notebook-suite
    provides: "plan 06-01 — platform/plotting/{core,loaders,data,drift}.py, notebooks/platform/P1_data_spine.ipynb, tests/unit/test_platform_notebooks.py"
  - phase: 01-monthly-data-layer
    provides: "monthly_features / monthly_raw platform checkpoints, platform/taxonomy.py classify_feature/lean_feature_set"
  - phase: 02-honesty-infrastructure
    provides: "honesty/holdout.py DEFAULT_HOLDOUT_CUTOFF + load_full_span, honesty/gating.py FORBIDDEN_CENTERED_SUFFIXES"
provides:
  - "platform/plotting/features.py: tier_frames, feature_range_table, plot_feature_ranges, assert_feature_ranges_plausible, check_agency_level_discontinuities, plot_drift_summary, _RANGE_CHECKS and its six named domain-bound constants"
  - "notebooks/platform/P2_features_taxonomy.ipynb — tiered coverage, ranges, quality gate, full-span drift, causal-only framing; executed with outputs"
  - "tests/unit/test_platform_plotting_features.py — 45 tests incl. the D-01 boundary check and the P2 notebook-source discipline assertions"
affects: [06-05-nowcaster-notebook, 06-06-allocation-notebook, 06-07-backtest-notebook]

# Tech tracking
tech-stack:
  added: []  # zero new dependencies
  patterns:
    - "symlog range axis: a single taxonomy tier spans realized_vol_1m at 8.1e-05 alongside gold at 1971.68 — six orders of magnitude. A linear axis collapses every small-valued feature onto the zero line, producing a chart that looks complete and shows nothing (the 06-03 lesson applied preemptively)"
    - "features.py reaches matplotlib ONLY through core (core.plt), owning no plotting-library import of its own — a test asserts this by AST, so the Agg-backend guard stays single-sourced"
    - "check_agency_level_discontinuities catches drift.assert_no_level_discontinuity's per-call raise once per column so the first violation never short-circuits the rest, then raises once — the collect-then-raise idiom applied around a raise-per-call primitive"
    - "the plausibility/quality gate runs on the DEV frame only; the full-span (post-2020) read is confined to the drift panel. A gate that can stop the notebook must not be evaluated against data the fitting side never saw (D-06)"
    - "notebook code cells that end in a plot call carry a trailing semicolon to suppress the duplicate execute_result figure the inline backend would otherwise render alongside plt.show()'s display_data"

key-files:
  created:
    - src/trading_crab_lib/platform/plotting/features.py
    - notebooks/platform/P2_features_taxonomy.ipynb
    - tests/unit/test_platform_plotting_features.py
  modified: []

key-decisions:
  - "Tasks 1 and 2 landed in ONE commit rather than two. Both extend the same new file, and splitting them would have required either partial-staging or writing then deleting then re-adding Task 2's function bodies. The commit message names both tasks explicitly."
  - "assert_feature_ranges_plausible's verdict frame carries exactly the four columns the plan specifies (feature, bound, observed, verdict) with `bound` and `observed` as human-readable strings ('> 0', 'min=9.51', 'in [0, 0.15]', 'min=0.0111, max=0.0624'). A numeric `observed` cannot represent a two-sided range check in one row, and the plan fixes both the column set and the one-row-per-checked-feature shape."
  - "A feature whose observed range is all-NaN is recorded as `verdict=pass, observed='no observations'` rather than as a violation. A dead column has no observed value to contradict a bound; it is a finding for the coverage panel, and treating absence as impossibility would make the gate fire on a missing series rather than a wrong one."
  - "check_agency_level_discontinuities' documented return type (dict[str, list[pd.Timestamp]]) is only ever realized as {} — a violation raises. drift.assert_no_level_discontinuity embeds its offending dates in the message string rather than returning them, and parsing them back out to populate a mapping would be re-implementing what the plan explicitly forbids re-implementing."

requirements-completed: [NB-01]

coverage:
  - id: F1
    description: "monthly_features' 53 columns group into fast/slow/agency/untagged via taxonomy.classify_feature, never a re-derived tier list"
    requirement: "NB-01"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_plotting_features.py::TestTierFrames (4 tests, incl. test_tier_membership_comes_from_taxonomy_not_a_local_list)"
        status: pass
      - kind: manual
        ref: "plan Task 1 <verify> run verbatim against the live dev checkpoint — 10/3/5/35, 53 range-table rows, 'P2 range/tier smoke OK'"
        status: pass
    human_judgment: false
  - id: F2
    description: "A feature whose observed range falls outside a named domain bound stops the notebook with a ValueError naming the feature and the bound"
    requirement: "NB-01"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_plotting_features.py::TestAssertFeatureRangesPlausible (10 parameterized out-of-band cases incl. fred_vix min=-5.0 and div_yield max=0.40, plus the collect-then-raise case)"
        status: pass
    human_judgment: false
  - id: F3
    description: "The A4 CPI level-discontinuity guard re-runs on every P2 open and passes against the live monthly_raw"
    requirement: "NB-01"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_plotting_features.py::TestCheckAgencyLevelDiscontinuities (5 tests: clean, tripling-raises-naming-column-and-date, absent-column-warns, multi-column collect, no-reimplementation)"
        status: pass
      - kind: manual
        ref: "plan Task 2 <verify> run verbatim — check_agency_level_discontinuities(monthly_raw, columns=('fred_cpi',)) == {}"
        status: pass
    human_judgment: false
  - id: F4
    description: "A lean feature whose current-era distribution has shifted from the pre-2021 fitted window appears in a ranked drift table and a magnitude-sorted bar chart"
    requirement: "NB-01"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_plotting_features.py::TestPlotDriftSummary (4 tests, incl. flagged-bars-use-a-distinct-color)"
        status: pass
      - kind: manual
        ref: "rendered against the real 776-row full span: gold +3.80, cape_shiller +1.72, div_yield -1.31, real_rate_level -1.31, oil +1.16, curve_10y3m -1.06 flagged"
        status: pass
    human_judgment: false
  - id: F5
    description: "P2 builds, seeks, and overlays no centered-feature variant, and says in prose why (Amendment 3 item I)"
    requirement: "NB-01"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_plotting_features.py::TestP2NotebookSource::test_no_code_cell_builds_a_centered_variant (checks center=True and every FORBIDDEN_CENTERED_SUFFIXES literal in code cells)"
        status: pass
    human_judgment: false
  - id: F6
    description: "P2 carries no sign-off cell and no per-run gate (D-15 stays P3-exclusive)"
    requirement: "NB-01"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_plotting_features.py::TestP2NotebookSource::test_carries_no_sign_off_cell"
        status: pass
    human_judgment: false
  - id: F7
    description: "P2 runs top-to-bottom against the real monthly_features/monthly_raw checkpoints without raising"
    requirement: "NB-01"
    verification:
      - kind: manual
        ref: "notebooks/platform/P2_features_taxonomy.ipynb executed via `jupyter execute --inplace` this session — 8 code cells, execution counts 1-8, 0 error outputs, 7 figures, outputs committed. Every figure was extracted and visually inspected."
        status: pass
    human_judgment: true

actuals:
  tokens: 58000
  tasks: 3
  commits: 2

metrics:
  duration_minutes: 40
  completed_date: 2026-09-10
status: complete
---

# Phase 6 Plan 04: P2 Features & Taxonomy Notebook Summary

Built `platform/plotting/features.py` — taxonomy tiering, the feature range
table, the D-11 named-domain-bound plausibility guard, the audit-item-A4
level-discontinuity regression guard, and the drift-summary chart — and
`P2_features_taxonomy.ipynb`, executed against the real checkpoints. P2 is
also this phase's explicit answer to Amendment 3 item I: it states in prose
why no causal-vs-centered panel exists rather than leaving the absence to
look like an oversight.

## Task Commits

1. **Tasks 1 + 2: tiering, range table, plausibility guard, A4 guard, drift chart**
   - `b3c3e7c` (feat) — `src/trading_crab_lib/platform/plotting/features.py`,
     `tests/unit/test_platform_plotting_features.py` (43 tests at that point)
2. **Task 3: the P2 notebook**
   - `74b9c47` (feat) — `notebooks/platform/P2_features_taxonomy.ipynb`
     (15 cells, executed, outputs committed)

## The measured fast / slow / agency / untagged split

Against the live dev `monthly_features` (708 rows x 53 columns,
1962-01-31 -> 2020-12-31) — this matches the plan's stated 10/3/5/35 exactly:

| tier | columns | members |
|---|---|---|
| **fast** | 10 | `credit_spread_baa_aaa`, `curve_10y2y`, `curve_10y3m`, `fred_vix`, `gold`, `oil`, `realized_vol_1m`, `realized_vol_3m`, `trailing_return_1m`, `trailing_return_3m` |
| **slow** | 3 | `cape_shiller`, `div_yield`, `real_rate_level` |
| **agency** | 5 | `fred_cpi`, `fred_gdp`, `fred_indpro`, `fred_payems`, `fred_unrate` |
| **untagged** | 35 | **22 ETF/equity price columns** — `AGG COST DBA EEM EFA GDX HYG IAU IEF IWM LQD MCD O QQQ SHY SLV SPY TSM UEC UNG VNQ VYM` — plus **13 raw/intermediate/splice columns**: `wti_fred`, `fred_gs10`, `fred_tb3ms`, `fred_aaa`, `fred_baa`, `fred_t10y3m`, `fred_t10y2y`, `sp500`, `gold_spot`, `wti_crude`, `equities_tr`, `long_duration_tr`, `cash` |

The lean set (`fast ∪ slow`) is 13 columns, all present in the full-span frame.

**Coverage findings worth an operator's attention** (all expected, none defects):
`curve_10y2y` starts 1976-06, `gold`/`oil` start 1985-02, `fred_vix` starts
1990-01 — so three of the ten fast-tier features do not span the labeler's
1962+ history. No series in any tier stops updating before the right edge.

## Plausibility verdict against the live data (every bound passes)

| feature | bound | observed |
|---|---|---|
| `fred_vix` | `> 0` | min=9.51 |
| `realized_vol_1m` | `>= 0` | min=8.114e-05 |
| `realized_vol_3m` | `>= 0` | min=0.000724 |
| `credit_spread_baa_aaa` | `>= -0.005` | min=0.32 |
| `cape_shiller` | `> 0` | min=6.64 |
| `gold` | `> 0` | min=254.6 |
| `oil` | `> 0` | min=10.25 |
| `div_yield` | `in [0, 0.15]` | min=0.0111, max=0.0624 |

`check_agency_level_discontinuities(monthly_raw, columns=("fred_cpi",))`
returns `{}` — **the A4 ALFRED-rebasing defect has not regressed.**

## Full-span drift (D-06 look-not-fit panel)

`load_full_span_checkpoint("monthly_features")` returns **776 rows through
2026-08-31** (708 dev + 68 holdout), matching the plan's stated reference. Six
of the 13 lean columns are flagged at or beyond the ±1σ threshold:

| column | standardized mean shift | reading |
|---|---|---|
| `gold` | **+3.80** | current mean 2552 vs baseline 723 — the largest shift by far |
| `cape_shiller` | **+1.72** | 34.5 vs 20.6 |
| `div_yield` | **−1.31** | 0.0140 vs 0.0290 |
| `real_rate_level` | **−1.31** | −0.81 vs +2.27 |
| `oil` | **+1.16** | 77.1 vs 43.8 |
| `curve_10y3m` | **−1.06** | 0.21 vs 1.50 |
| `curve_10y2y` | −0.80 | in band |
| `credit_spread_baa_aaa` | −0.54 | in band |
| `trailing_return_3m` / `_1m` | +0.17 / +0.11 | in band |
| `fred_vix` | −0.06 | in band |
| `realized_vol_3m` / `_1m` | −0.04 / +0.02 | in band |

Note the pattern: the two *slow*-tier valuation anchors (`cape_shiller`,
`div_yield`) and `real_rate_level` have all moved more than a baseline sigma,
while every *fast*-tier volatility/return measure is essentially unmoved. That
is a question for whoever tunes the strategic-tilt layer, not a defect — and
per D-07 it belongs in `.planning/POST-2020-OBSERVATIONS.md` if it changes a
decision. **I have not written that entry**: recording an observation as
decision-changing is the operator's judgement, not mine.

## Verification Results

- `pytest tests/unit/test_platform_plotting_features.py -q` → **45 passed**
  (43 at the Tasks-1+2 commit; the two P2-notebook-source tests were skip-guarded
  until Task 3 landed and now run).
- `pytest tests/unit/test_platform_notebooks.py -q` → **17 passed, 2 skipped**
  (P2 joins the glob parameterization automatically; the 2 remaining skips are
  the A13-discipline guards for the not-yet-built P4 and P6).
- **Full suite: `pytest tests/ -q` → 1578 passed, 2 skipped, 0 failures** (~75s).
  Baseline entering this plan was 1528 passed / 2 skipped. **Net +50 tests, zero
  regressions, zero skips added.**
- `ruff check` clean on both new source/test files.
- Both plan `<verify>` scripts ran verbatim and printed their success sentinels
  (`P2 range/tier smoke OK`, `P2 plausibility/discontinuity smoke OK`), as did
  Task 3's static-token script (`P2 static checks OK 15 cells`).
- `git status --porcelain src/trading_crab_lib/platform/taxonomy.py
  src/trading_crab_lib/platform/transforms_monthly.py outputs/reports/platform/
  data/checkpoints/platform/ registry/` → **empty**. Phase 1-5 code and every
  reference artifact were read, never modified.
- Secret hygiene (T-06-15): the live `FRED_API_KEY` value does not appear
  anywhere in the committed notebook; the only `api_key` occurrence is inside
  an explanatory comment. Config is displayed exclusively through
  `pplot.redacted_config(cfg)`.
- Notebook executed from a clean state via `jupyter execute --inplace`:
  8 code cells, execution counts 1-8, **0 error outputs**, 7 figures, committed
  with outputs.

## Rendering — every figure was extracted and looked at

Per the 06-03 lesson (a `<verify>` that only checks "a Figure came back" cannot
tell you the chart is legible), all seven committed figures were decoded out of
the executed notebook and visually inspected. Two layout decisions came out of
that inspection rather than out of the plan:

1. **`plot_feature_ranges` uses a symlog x-axis, not a linear one.** The fast
   tier alone spans `realized_vol_1m` at 8.1e-05 and `gold` at 1971.68 — six
   orders of magnitude, with `curve_10y3m` reaching −2.65 on the negative side.
   On a linear axis, seven of the ten bars render as an invisible smear on the
   zero line while `gold` and `oil` take the whole width. Each row is
   additionally annotated with its literal `[min .. max]` so the numbers do not
   depend on reading a log axis correctly.
2. **The plot cells end in a trailing semicolon.** The inline backend renders
   the figure once from `_save_or_show`'s `plt.show()` (as `display_data`) and
   again from the returned `Figure`'s repr (as `execute_result`), so the last
   chart in each cell appeared twice. P1 carries this duplication; P2 does not.

The 35-row untagged coverage grid and the 13-row drift chart were both checked
for collided tick labels and occluded layers — neither has any.

## Deviations from Plan

### Structural

**1. Tasks 1 and 2 landed in a single commit (`b3c3e7c`) rather than two.**
- **Why:** both tasks extend the same new file, `platform/plotting/features.py`.
  Producing two atomic commits would have meant either interactive partial
  staging (unavailable) or writing Task 2's functions, deleting them, committing,
  and re-adding them — churn with no audit value. The commit message names both
  tasks and both bodies of work explicitly.
- **Effect on verification:** none. Both tasks' `<verify>` scripts and all
  acceptance criteria were run and satisfied before the commit.

**2. TDD RED/GREEN commits were not split.** Both tasks carry `tdd="true"`, but
this phase's established convention (06-01, 06-03) is one commit per task rather
than a `test:`/`feat:` pair, and the plan's own `files_modified` and acceptance
criteria are framed per task. Tests and implementation were written and iterated
together; every behavior case named in the `<behavior>` blocks exists as a test.

### Implementation choices the plan left open

**3. The plausibility verdict frame's `bound` and `observed` are strings.**
The plan fixes the column set to exactly `feature, bound, observed, verdict` and
the shape to one row per checked feature. `div_yield`'s check is two-sided, so a
single numeric `observed` cannot represent it. Both columns are therefore
human-readable strings (`"in [0, 0.15]"`, `"min=0.0111, max=0.0624"`), which is
also what the notebook prints.

**4. An all-NaN feature passes rather than violating.** `_check_one_feature`
short-circuits a NaN observed range to `verdict=pass, observed="no observations"`.
A column with no observations has no value that contradicts a bound; firing the
gate on absence would conflate "this series is missing" (a coverage finding, one
panel up) with "this series holds an impossible value" (what the gate is for).

**5. `check_agency_level_discontinuities` returns `{}` on the clean path and
never a populated mapping.** Its declared return type is
`dict[str, list[pd.Timestamp]]` per the plan, but a violation raises rather than
returning. Populating the mapping would require parsing offending dates back out
of `drift.assert_no_level_discontinuity`'s message string — re-deriving what the
plan explicitly forbids re-implementing. The docstring states this.

No auto-fix under deviation Rules 1-3 was needed: no bug, no missing critical
functionality, and no blocking issue was encountered. Every plan acceptance
criterion was factually correct about the data this time (10/3/5/35, 53 rows,
776 full-span rows, every named range in band, `fred_cpi` clean) — verified
before relying on any of them.

## Known Stubs

None.

## Threat Flags

None. This plan added no network endpoint, no auth path, no file-write path,
and no schema at a trust boundary. The only new file access is read-only via
06-01's existing loaders; the only new write is the committed notebook itself,
which was scanned for secret leakage (see Verification Results).

## Issues Encountered

None blocking. One point future plans should know: `tests/conftest.py`'s
session-scoped checkpoint-isolation fixture does not seed real `data/holdout/`
content, so **any test asserting on the real 776-row full span will see 708
under pytest**. This plan avoided the trap entirely by keeping every test on
synthetic frames and verifying the live numbers through the plan's own
`python -c` `<verify>` scripts, which run outside pytest. 06-01 hit this and
06-03 worked around it; it remains true for 06-05 through 06-07.

## Next Phase Readiness — what plans 06-05 / 06-06 / 06-07 need to know

- **`platform/plotting/__init__.py` is still untouched** and must stay that way.
  `features.py` is imported by submodule path
  (`from trading_crab_lib.platform.plotting import features as pfeatures`), and
  a test in this plan's file asserts `pplot` does *not* re-export `tier_frames`.
  Add `nowcaster.py` / `allocation.py` / `backtest.py` the same way.
- **`core.plt` is the sanctioned matplotlib handle.** `features.py` imports no
  plotting library of its own and reaches pyplot through `core`, which owns the
  Agg-backend guard; `data.py` (06-01) imports `matplotlib.pyplot` directly.
  Either passes the D-01 boundary test, but routing through `core` is the
  stricter and now-tested convention.
- **Reuse `features._RANGE_CHECKS` and `drift.py`'s named constants** rather
  than re-declaring a bound locally. Any later notebook displaying a feature
  range should call `assert_feature_ranges_plausible` on a
  `feature_range_table`, not hand-roll a comparison.
- **The symlog lesson generalizes.** Any chart that puts multiple platform
  quantities on one axis will span several orders of magnitude (returns at 1e-2
  next to price levels at 1e3, `equities_tr` at 5e5). Check the rendered PNG,
  not just that a `Figure` came back.
- **`test_platform_notebooks.py` still needs zero edits.** P2 joined its glob
  automatically. Remember its constraints: first cell markdown naming a
  `scripts/` prerequisite; no `import matplotlib` / `import seaborn` / bare
  `plt.` or `sns.` token in any code cell; no `.save(`; and if you call
  `load_platform_config`, reference `redacted_config` and never `print(cfg)`.
- **P2 deliberately carries no A13 mention** and is not in
  `_A13_GATED_NOTEBOOKS` — the §5.4 lag/ratio headline does not appear in it.
  P4 and P6 are still gated and still skipping.
- **The six flagged drift columns above** (`gold`, `cape_shiller`, `div_yield`,
  `real_rate_level`, `oil`, `curve_10y3m`) are the current-era shifts P5's
  allocation notebook and P6's backtest notebook will be reading regime behavior
  through. Three of them are slow-tier valuation anchors.

## Self-Check: PASSED

- `src/trading_crab_lib/platform/plotting/features.py` — FOUND
- `notebooks/platform/P2_features_taxonomy.ipynb` — FOUND
- `tests/unit/test_platform_plotting_features.py` — FOUND
- commit `b3c3e7c` — FOUND
- commit `74b9c47` — FOUND

---
*Phase: 06-platform-notebook-suite*
*Completed: 2026-09-10*
