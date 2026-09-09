---
phase: 06-platform-notebook-suite
plan: 01
subsystem: platform-plotting-spine
tags: [matplotlib, nbformat, jupyter, drift-detection, plausibility-bands, notebook-testing]

# Dependency graph
requires:
  - phase: 01-monthly-data-layer
    provides: monthly_raw/monthly_features platform checkpoints, get_platform_checkpoint_manager()
  - phase: 02-honesty-infrastructure
    provides: platform/config.py load_platform_config(), platform/honesty/holdout.py (DEFAULT_HOLDOUT_CUTOFF, split_by_holdout_boundary, load_full_span)
  - phase: 03-regime-labeling-prediction
    provides: platform/labeling/diagnostics.py label_regimes()
  - phase: 05 (backtest evaluation)
    provides: outputs/reports/platform/backtest_kpi_table.parquet, model_metrics_brier.parquet
provides:
  - "platform/plotting/core.py: CUSTOM_COLORS, REGIME_CMAP, PLATFORM_PLOT_DIR, A13_CAVEAT, _in_jupyter, _regime_color, _save_or_show(fig, *, save_path, show) -> Figure"
  - "platform/plotting/loaders.py: load_platform_checkpoint, load_full_span_checkpoint, load_report_artifact, load_full_sample_states, load_filtered_state_probs, redacted_config, compute_regime_labeling, NOTEBOOK_SCRATCH_DIR"
  - "platform/plotting/data.py: plot_coverage_timeline(df, *, title, max_columns, save_path, show) -> Figure"
  - "platform/plotting/drift.py: no_skill_brier, assert_terminal_log_wealth_plausible, assert_max_drawdown_plausible, assert_brier_plausible, assert_turnover_plausible, assert_cvar_plausible, assert_regime_occupancy_plausible, assert_portfolio_weights_plausible, assert_no_level_discontinuity, assert_kpi_table_plausible, baseline_and_current, compute_drift, drift_report"
  - "notebooks/platform/P1_data_spine.ipynb — first platform verification notebook, executed against the real monthly_raw checkpoint"
  - "tests/unit/test_platform_notebooks.py — static notebook gate, parameterized over notebooks/platform/*.ipynb, reusable by all five downstream plans"
affects: [06-02-nowcaster-evaluation-artifacts, 06-03-features-taxonomy, 06-04-regime-labeling-notebook, 06-05-nowcaster-notebook, 06-06-allocation-notebook, 06-07-backtest-notebook]

# Tech tracking
tech-stack:
  added: []  # zero new dependencies — matplotlib, seaborn, nbformat, scipy already installed
  patterns:
    - "D-02 no-RunConfig plot signature: plot_x(data, *, save_path: Path | None = None, show: bool = False) -> plt.Figure, returned unclosed so callers/tests can inspect it"
    - "D-10 load-or-raise-actionably: every loader wraps a FileNotFoundError and re-raises one naming the missing artifact, its directory, and the exact rebuild command"
    - "D-06 full-span opt-in: load_full_span_checkpoint wraps honesty.holdout.load_full_span (never a hand-rolled two-manager concat) and logs post-cutoff row count at WARNING, pointing at POST-2020-OBSERVATIONS.md"
    - "D-01 fresh-package boundary verified by static AST import-graph closure (importlib.util.find_spec + ast.walk), not sys.modules inspection, to stay immune to test-order pollution from other test files that DO import the legacy trading_crab_lib.plotting package"
    - "D-11 plausibility raise/warn idiom copied verbatim in shape from platform/splice.py::assert_yield_units_plausible: named-constant bounds, hard ValueError with an explanatory f-string, log.warning only on the ambiguous direction"
    - "D-09 drift baseline is always the pre-2021 fitted window (honesty.holdout.split_by_holdout_boundary), never a rolling trailing window"
    - "assert_kpi_table_plausible collects every violation before raising once (config.validate_config collect-then-raise idiom)"
    - "__init__.py re-exports ONLY core+loaders symbols by design (documented in its own docstring) so the five wave-2 plans each add one new plotting submodule without contending for the same barrel file"

key-files:
  created:
    - src/trading_crab_lib/platform/plotting/__init__.py
    - src/trading_crab_lib/platform/plotting/core.py
    - src/trading_crab_lib/platform/plotting/loaders.py
    - src/trading_crab_lib/platform/plotting/data.py
    - src/trading_crab_lib/platform/plotting/drift.py
    - notebooks/platform/P1_data_spine.ipynb
    - tests/unit/test_platform_plotting.py
    - tests/unit/test_platform_plotting_data.py
    - tests/unit/test_platform_plotting_drift.py
    - tests/unit/test_platform_notebooks.py
    - .planning/POST-2020-OBSERVATIONS.md
  modified: []

key-decisions:
  - "D-01 verification implemented as a static transitive-closure scan (ast.parse + importlib.util.find_spec) over trading_crab_lib.* imports reachable from trading_crab_lib.platform.plotting, rather than inspecting sys.modules at runtime — sys.modules is global and polluted by other test files in the same pytest session that legitimately import the legacy trading_crab_lib.plotting package, which would make a naive presence check falsely fail"
  - "assert_kpi_table_plausible returns a long-format verdict frame (one row per leg x metric, 10 rows for the live 5-leg table) rather than one row per leg, matching the plan's literal column spec (leg, metric, value, universal_band, domain_band, verdict) — the plan's acceptance-criteria wording '5 leg rows' is read as '5 distinct legs represented' (verdict['leg'].nunique() == 5), since a wide format would need two different value columns and the action block explicitly fixes this column set"
  - "compute_regime_labeling and NOTEBOOK_SCRATCH_DIR are implemented and unit-covered (differs-from-PLATFORM_CHECKPOINT_DIR) but not exercised end-to-end in this plan — no notebook in this wave calls label_regimes; P3 (a later plan) is the first real caller"

requirements-completed: [NB-01]

coverage:
  - id: D1
    description: "A plot function returns a matplotlib Figure with no RunConfig-shaped object on the platform side (D-02)"
    requirement: "NB-01"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_plotting.py::TestSaveOrShow, tests/unit/test_platform_plotting_data.py::TestPlotCoverageTimeline"
        status: pass
    human_judgment: false
  - id: D2
    description: "A missing platform checkpoint stops with a message naming the checkpoint and the rebuild command, not a bare FileNotFoundError (D-10)"
    requirement: "NB-01"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_plotting.py::TestLoadPlatformCheckpoint, TestLoadFullSpanCheckpoint, TestLoadReportArtifact, TestLoadFullSampleStatesAndFilteredProbs"
        status: pass
    human_judgment: false
  - id: D3
    description: "Terminal log wealth 111.06 raises; the 60/40-shaped leg at -2.27% max drawdown raises despite being inside the universal [-1,0] bound"
    requirement: "NB-01"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_plotting_drift.py::TestAssertTerminalLogWealthPlausible::test_historical_regression_111_raises, TestAssertMaxDrawdownPlausible::test_historical_regression_sixty_forty_raises"
        status: pass
    human_judgment: false
  - id: D4
    description: "The live KPI table and Brier value pass their bands, with the K=5 no-skill Brier reference surfaced as 0.16 and 0.2087 flagged as not beating it"
    requirement: "NB-01"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_plotting_drift.py::TestAssertKpiTablePlausible::test_live_kpi_table_passes_and_returns_verdict_frame, TestAssertBrierPlausible::test_in_band_but_above_no_skill"
        status: pass
    human_judgment: false
  - id: D5
    description: "A feature whose current-era distribution has shifted from the pre-2021 fitted window appears in a ranked drift table with a standardized mean shift"
    requirement: "NB-01"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_plotting_drift.py::TestComputeDrift, TestDriftReport::test_sorted_by_descending_absolute_shift"
        status: pass
    human_judgment: false
  - id: D6
    description: "Every notebook under notebooks/platform/ parses as valid nbformat and defines no plotting logic in its own cells"
    requirement: "NB-01"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_notebooks.py"
        status: pass
    human_judgment: false
  - id: D7
    description: "The tracer notebook P1_data_spine.ipynb runs top-to-bottom against the real monthly_raw checkpoint without raising"
    requirement: "NB-01"
    verification:
      - kind: manual
        ref: "notebooks/platform/P1_data_spine.ipynb — executed via jupyter execute --inplace this session against the real monthly_raw checkpoint (776 rows x 45 cols, 1962-01 -> 2026-08); outputs committed"
        status: pass
    human_judgment: true

actuals:
  tokens: 62000
  tasks: 3
  commits: 3

metrics:
  duration_minutes: 55
  completed_date: 2026-09-09
status: complete
---

# Phase 6 Plan 01: Platform Plotting Spine + P1 Tracer Notebook Summary

Built the platform's first `plotting/` package — `core` (D-02 no-RunConfig
save/show/palette), `loaders` (D-10 actionable-raise checkpoint/report
loaders + D-06 full-span opt-in), `data` (the P1 coverage plot), and `drift`
(D-09 drift-against-baseline + reinstated D-11 plausibility bands) — and
proved the spine end-to-end with `notebooks/platform/P1_data_spine.ipynb`
executed against the real 776x45 `monthly_raw` checkpoint.

## Task Commits

1. **Task 1: End-to-end "see the data spine" slice**
   - `7b9689e` (feat) — `core.py`, `loaders.py`, `data.py`, `__init__.py`,
     `notebooks/platform/P1_data_spine.ipynb` (executed, outputs committed),
     `tests/unit/test_platform_plotting.py`,
     `tests/unit/test_platform_plotting_data.py`
2. **Task 2: Drift-against-baseline (D-09) and reinstated plausibility bands (D-11)**
   - `23f6423` (feat) — `drift.py`, `tests/unit/test_platform_plotting_drift.py`
3. **Task 3: Static notebook gate and the D-07 post-2020 observation record**
   - `eff5a13` (test) — `tests/unit/test_platform_notebooks.py`,
     `.planning/POST-2020-OBSERVATIONS.md`, plus a small amendment to
     `notebooks/platform/P1_data_spine.ipynb`'s setup cell (see Deviations)

## Accomplishments

- `platform/plotting/core.py`: `CUSTOM_COLORS` (5 hex values, matches
  `labeling.K=5`), `REGIME_CMAP`, `PLATFORM_PLOT_DIR`, `A13_CAVEAT` (the
  single-source not-interpretable caveat string every later module/notebook
  reuses wherever the §5.4 lag/ratio appears), `_in_jupyter`, `_regime_color`,
  `_save_or_show(fig, *, save_path, show) -> Figure` (returns the figure
  unclosed, per D-02).
- `platform/plotting/loaders.py`: `load_platform_checkpoint`,
  `load_full_span_checkpoint` (wraps `honesty.holdout.load_full_span`, logs
  post-cutoff row count at WARNING pointing at
  `POST-2020-OBSERVATIONS.md`), `load_report_artifact`,
  `load_full_sample_states`/`load_filtered_state_probs` (raise actionably
  today — the artifacts they read are written by plan 06-02, absent in this
  wave), `redacted_config` (recursive `*api_key` redaction, non-mutating),
  `compute_regime_labeling` (routes `label_regimes` at
  `NOTEBOOK_SCRATCH_DIR = data/checkpoints/platform_notebook/`, never the
  production platform checkpoint namespace), `NOTEBOOK_SCRATCH_DIR`.
- `platform/plotting/data.py`: `plot_coverage_timeline` — binary
  column-availability grid sorted by first-valid date, annotated with
  first-valid year, `max_columns` bound, empty-input-safe.
- `platform/plotting/drift.py`: pure functions, zero matplotlib import.
  Plausibility half — `no_skill_brier`,
  `assert_terminal_log_wealth_plausible`, `assert_max_drawdown_plausible`
  (universal `[-1,0]` then per-leg domain band), `assert_brier_plausible`
  (`[0,1]`, not the textbook `[0,2]`), `assert_turnover_plausible`,
  `assert_cvar_plausible`, `assert_regime_occupancy_plausible`,
  `assert_portfolio_weights_plausible`, `assert_no_level_discontinuity`
  (public counterpart of `transforms_monthly.py`'s private detector),
  `assert_kpi_table_plausible` (collect-then-raise, returns a tidy verdict
  frame on success). Drift half — `baseline_and_current` (delegates to
  `split_by_holdout_boundary`), `compute_drift` (standardized mean shift, KS
  statistic, 2-sigma exceedance, NaN-safe on empty/zero-std input),
  `drift_report` (ranked by descending absolute shift).
- `notebooks/platform/P1_data_spine.ipynb`: 5 cells, built via `nbformat`'s
  programmatic v4 API, executed once against the real `monthly_raw`
  checkpoint (776 rows x 45 cols, 1962-01-31 -> 2026-08-31), outputs
  committed. Demonstrates `redacted_config` in its setup cell.
- `tests/unit/test_platform_notebooks.py`: static-only gate (no
  nbmake/papermill), parameterized over `notebooks/platform/*.ipynb` — valid
  nbformat, no forbidden plotting tokens, no `.save(` calls, config-redaction
  discipline, markdown-header prerequisite, and an existence-guarded A13
  discipline check for P3/P4/P6.
- `.planning/POST-2020-OBSERVATIONS.md`: the D-07 record — markdown table
  with one seeded format-example row, explicit statement that an empty real
  log is a valid state.

## Verification Results

- `pytest tests/unit/test_platform_plotting.py tests/unit/test_platform_plotting_data.py tests/unit/test_platform_plotting_drift.py tests/unit/test_platform_notebooks.py -x -q` → **61 passed, 3 skipped** (the 3 skips are the A13-discipline guards for P3/P4/P6, which don't exist until later plans — confirmed via `pytest.skip`, not a failure).
- `pytest tests/ -q` (full suite) → **1454 passed, 3 skipped, 0 failures** (baseline was 1393 passed; net +61 new tests, 0 regressions).
- `git status --porcelain outputs/reports/platform/` → empty (no Phase-5 artifact touched).
- Both non-negotiable historical regressions from `06-VALIDATION.md` confirmed live: `assert_max_drawdown_plausible(-0.0227, leg="sixty_forty")` raises while `-0.269625` passes; `assert_terminal_log_wealth_plausible(111.06, leg="strategy")` raises while `4.0265` passes.
- `assert_kpi_table_plausible` run against the real `outputs/reports/platform/backtest_kpi_table.parquet` returns a 10-row verdict frame (5 legs x 2 metrics), all `verdict == "pass"`.
- `assert_brier_plausible(0.208746, n_classes=5)` returns `no_skill=0.16`, `beats_no_skill=False`, matching `06-VALIDATION.md`'s reference exactly.
- `notebooks/platform/P1_data_spine.ipynb` executed via `jupyter execute --inplace` against the real `monthly_raw` checkpoint with no exception; committed with its outputs (shape/index-bounds prints, the coverage figure, and the redacted-config section list).
- Manually demonstrated (then discarded) that injecting a direct-`matplotlib`-import cell into a scratch copy of P1 (built in a `tempfile.TemporaryDirectory()`, never touching the real notebook or repo) trips `test_notebook_has_no_forbidden_plotting_tokens` — confirms Task 3's acceptance criterion "introducing a forbidden cell makes the gate fail."

## Files Created/Modified

- `src/trading_crab_lib/platform/plotting/core.py` — palette, save/show, A13 caveat
- `src/trading_crab_lib/platform/plotting/loaders.py` — checkpoint/report loaders, redaction, scratch-namespace labeling
- `src/trading_crab_lib/platform/plotting/data.py` — `plot_coverage_timeline`
- `src/trading_crab_lib/platform/plotting/drift.py` — D-09 drift + D-11 plausibility
- `src/trading_crab_lib/platform/plotting/__init__.py` — core+loaders-only barrel
- `notebooks/platform/P1_data_spine.ipynb` — tracer notebook, executed with outputs
- `tests/unit/test_platform_plotting.py` — core/loaders tests (24 tests)
- `tests/unit/test_platform_plotting_data.py` — `plot_coverage_timeline` tests (3 tests)
- `tests/unit/test_platform_plotting_drift.py` — drift/plausibility tests (28 tests)
- `tests/unit/test_platform_notebooks.py` — static notebook gate (9 tests, 3 skip today)
- `.planning/POST-2020-OBSERVATIONS.md` — D-07 record

## Decisions Made

- **D-01 verification via static AST closure, not `sys.modules`.** The full
  test suite runs many files in one process; several already import the
  legacy `trading_crab_lib.plotting` package for unrelated reasons, which
  pollutes `sys.modules` globally. A `sys.modules`-presence check would
  therefore be a false positive regardless of whether `platform.plotting`
  itself imports the legacy package. Implemented instead as a true
  reachability computation: parse every `trading_crab_lib.platform.plotting`
  source file's imports via `ast`, resolve each `trading_crab_lib.*` name to
  its file via `importlib.util.find_spec`, and recurse — this only reports
  modules actually reachable from the import graph rooted at
  `platform.plotting`, independent of what else the test process happens to
  have imported.
- **`assert_kpi_table_plausible`'s verdict frame is long-format** (one row
  per leg x metric — 10 rows for the live 5-leg table), matching the
  action block's explicit column spec (`leg`, `metric`, `value`,
  `universal_band`, `domain_band`, `verdict`) verbatim. The acceptance
  criteria's phrase "5 leg rows" is satisfied as "5 distinct legs
  represented" (`verdict['leg'].nunique() == 5`) rather than literally 5
  rows, since the column spec has one `value` per (leg, metric) pair and a
  wide format would need two differently-typed value columns.
- **`compute_regime_labeling`/`NOTEBOOK_SCRATCH_DIR` are unit-tested but not
  end-to-end exercised in this plan.** No notebook in wave 1 calls
  `label_regimes` — P1 only needs `load_platform_checkpoint` and
  `plot_coverage_timeline`. The scratch-namespace routing is verified
  directly (differs from `PLATFORM_CHECKPOINT_DIR`); its first real caller
  is a later plan's P3 notebook.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] P1 notebook's setup cell did not demonstrate `redacted_config`, tripping the T-06-01 gate Task 3 defines**
- **Found during:** Task 3, writing `test_notebook_never_displays_unredacted_config`
- **Issue:** Task 1's action block specified the setup cell should call
  `load_platform_config()` but did not require displaying anything from
  `cfg`. Task 3's literal spec, however, requires that *any* notebook whose
  source contains `load_platform_config` also reference `redacted_config`
  somewhere in the notebook — a blanket rule, not conditioned on whether
  `cfg` is actually printed. P1 as built in Task 1 loaded `cfg` but never
  referenced `redacted_config`, so it failed the new test as soon as Task 3
  wrote it.
- **Fix:** Added one line to the setup cell —
  `print("platform config sections:", sorted(pplot.redacted_config(cfg).keys()))`
  — which both satisfies the gate and gives the notebook a genuinely useful
  sanity-check line (visible config sections without any secret value).
  Re-executed the notebook via `jupyter execute --inplace` and recommitted
  the updated cell + outputs.
- **Files modified:** `notebooks/platform/P1_data_spine.ipynb`
- **Commit:** `eff5a13`

**2. [Rule 1 - Bug] Three drift/plausibility test cases as first drafted were numerically or environmentally invalid**
- **Found during:** Task 2, running `tests/unit/test_platform_plotting_drift.py` for the first time
- **Issue (a):** `test_out_of_domain_band_but_in_universal_raises` used
  `9.9` as an "out-of-domain-but-in-universal" terminal-log-wealth value,
  but the domain band `[-3, 12]` is wider than the universal
  `abs(x) < 10` bound on the upper side, so no positive value can be
  in-universal-but-out-of-domain there; `9.9` passed both bounds and the
  test failed to raise. **Issue (b):** the "uniformly wrong series shows
  zero drift" test constructed `np.full(776, 1e128)` and expected an exact
  zero/NaN standardized shift, but squaring/summing values near float64's
  practical precision ceiling for variance computation introduces
  floating-point noise (`std() ≈ 1.9e112` instead of exactly `0`), producing
  a spurious ~0.999 "shift" that was an artifact of extreme magnitude, not
  of the intended lesson. **Issue (c):**
  `test_full_span_has_more_rows_than_dev` asserted `776 > 708` against the
  real production checkpoints, but `tests/conftest.py`'s session-scoped
  checkpoint-isolation fixture deliberately does NOT copy real
  `data/holdout/` content into its session directory (documented in
  `conftest.py` as intentional — that data is the one dataset the honesty
  framework exists to protect and is not reproducible from a dev rebuild),
  so under pytest `load_full_span_checkpoint` correctly falls back to
  dev-only rows and the counts were equal (708 == 708), not a bug in the
  implementation.
- **Fix:** (a) changed the test value to `-5.0` (abs=5 <10 passes universal,
  below the `-3.0` domain floor, fails domain — a value that actually
  exercises the intended branch). (b) replaced the test with
  `test_drift_is_scale_invariant_so_a_uniform_units_error_is_invisible`,
  which demonstrates the real lesson correctly: multiplying both baseline
  and current windows of a normal series by the same wrong constant (100x)
  leaves `standardized_mean_shift` and `flag` unchanged, because the metric
  is scale-invariant — this is *why* drift alone cannot catch a uniform
  units error, independent of any floating-point edge case. (c) rewrote the
  test to build its own synthetic dev+holdout checkpoint pair under
  `monkeypatch`-redirected `PLATFORM_CHECKPOINT_DIR`/`HOLDOUT_CHECKPOINT_DIR`
  rather than depending on real holdout data being present in the isolated
  test session; verified the real 776-vs-708 relationship separately via a
  direct (non-pytest) `python -c` invocation matching the plan's own
  `<verify>` command, which is not subject to the isolation fixture.
- **Files modified:** `tests/unit/test_platform_plotting_drift.py`,
  `tests/unit/test_platform_plotting.py`
- **Commits:** `23f6423`, and the `test_full_span_has_more_rows_than_dev`
  fix was made before the Task 1 commit (`7b9689e`) landed, so it is
  included there.

None of these required any change to library code (`core.py`, `loaders.py`,
`data.py`, `drift.py`) — all three were test-authoring issues caught and
fixed before commit.

## Known Stubs

None. `load_full_sample_states` and `load_filtered_state_probs` intentionally
raise `FileNotFoundError` today (the artifacts they read are written by plan
06-02, not this plan) — this is documented behavior with dedicated
regression tests, not a stub masking missing functionality.

## Issues Encountered

- The pytest session-scoped checkpoint-isolation fixture in
  `tests/conftest.py` (added in an earlier phase to protect
  `data/holdout/`) means any future platform test asserting on real
  dev-vs-full-span row counts must either run outside pytest or build its
  own synthetic dev/holdout pair with `monkeypatch`. This is documented
  inline in `tests/unit/test_platform_plotting.py`'s
  `test_full_span_has_more_rows_than_dev` docstring/comment for whoever
  writes the next such test in plan 06-02 onward.

## User Setup Required

None. Zero new dependencies (matplotlib, seaborn, scipy, nbformat, jupyter
were already installed in `.venv`). No secrets, no service configuration.

## Next Phase Readiness — what plans 06-02 through 06-07 need to know

- **Exact `__init__.py` `__all__` contents:** `A13_CAVEAT`, `CUSTOM_COLORS`,
  `PLATFORM_PLOT_DIR`, `REGIME_CMAP`, `NOTEBOOK_SCRATCH_DIR`,
  `compute_regime_labeling`, `load_filtered_state_probs`,
  `load_full_sample_states`, `load_full_span_checkpoint`,
  `load_platform_checkpoint`, `load_report_artifact`, `redacted_config`.
  Per-layer plot functions (this plan's `data.plot_coverage_timeline`, and
  every function the five downstream plans add) are **never** added here —
  import by submodule path: `from trading_crab_lib.platform.plotting import
  features as pfeat` (or `regime`, `nowcaster`, `allocation`, `backtest`).
- **`core._save_or_show(fig, *, save_path: Path | None, show: bool) ->
  plt.Figure`** — every new `plot_x` function in every downstream submodule
  must call this exact helper as its return statement, and must itself take
  `(data, *, save_path: Path | None = None, show: bool = False) ->
  plt.Figure`. No `RunConfig`-shaped parameter, ever.
- **`loaders.load_full_sample_states()` and `loaders.load_filtered_state_probs()`
  still raise `FileNotFoundError` until plan 06-02 lands** (it is the plan
  that extends `platform/evaluation/report.py` per Amendment 3 item H to
  persist `backtest_full_sample_states.parquet` and
  `backtest_filtered_state_probs.parquet`). Once 06-02 lands, no change to
  `loaders.py` is required — the functions already read the correct
  filenames and column-renaming contract; they simply start succeeding.
- **`drift.py`'s named constants are the single source of truth for every
  plausibility band** any downstream notebook displays — reuse
  `_DOMAIN_MAX_DRAWDOWN_BANDS`, `_BRIER_RANGE`, etc. rather than
  re-declaring bounds locally.
- **`core.A13_CAVEAT`** is ready for P3/P4/P6 to interpolate wherever the
  §5.4 detection-lag/sojourn-ratio headline is displayed —
  `tests/unit/test_platform_notebooks.py`'s A13-discipline guard will start
  enforcing this automatically once those three notebooks exist (currently
  3 skipped tests, by design).
- **`tests/unit/test_platform_notebooks.py` needs zero edits** as
  downstream plans add notebooks — it globs `notebooks/platform/*.ipynb`.
  New notebooks must: start with a markdown cell containing `scripts/`;
  never contain `import matplotlib`/`import seaborn`/bare `plt.`/`sns.`;
  never contain `.save(`; and if they call `load_platform_config`, must
  also reference `redacted_config` somewhere and never `print(cfg)` or end
  a cell with a bare `cfg` expression.

## Self-Check: PASSED

- `src/trading_crab_lib/platform/plotting/__init__.py` — FOUND
- `src/trading_crab_lib/platform/plotting/core.py` — FOUND
- `src/trading_crab_lib/platform/plotting/loaders.py` — FOUND
- `src/trading_crab_lib/platform/plotting/data.py` — FOUND
- `src/trading_crab_lib/platform/plotting/drift.py` — FOUND
- `notebooks/platform/P1_data_spine.ipynb` — FOUND
- `tests/unit/test_platform_plotting.py` — FOUND
- `tests/unit/test_platform_plotting_data.py` — FOUND
- `tests/unit/test_platform_plotting_drift.py` — FOUND
- `tests/unit/test_platform_notebooks.py` — FOUND
- `.planning/POST-2020-OBSERVATIONS.md` — FOUND
- commit `7b9689e` — FOUND
- commit `23f6423` — FOUND
- commit `eff5a13` — FOUND

---
*Phase: 06-platform-notebook-suite*
*Completed: 2026-09-09*
