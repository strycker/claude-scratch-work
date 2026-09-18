---
phase: 07-regime-representation
plan: 05
subsystem: ml-platform
tags: [jump-model, feature-engineering, fred-ingestion, regime-labeling, honesty-framework]

# Dependency graph
requires:
  - phase: 07-regime-representation (wave 1)
    provides: frozen ten-column feature policy for classifier #1 (D-02-A), the platform's
      report-only/no-gate posture (D-07), the legacy-import ratchet test
provides:
  - "canonicalize_states(sort_column=...) keyword-only parameter that raises ValueError on
    an absent sort column instead of silently falling back to centroid column 0"
  - "platform/features/relative.py — classifier #2's raw candidate columns (relative
    strength, trailing momentum, rolling cross-correlation, CPI acceleration, plus INV-01's
    two named invariant ratios), ported from the legacy library, never imported"
  - "M2SL and TOTALSL ingested into monthly_raw through the existing config-driven FRED
    path (fred_m2sl, fred_totalsl)"
affects: [07-06 (deflated Sharpe / trial registry), classifier-#2-fit, joint-tilt-blending,
  ADR-0002-l1-leadership-axis]

# Actuals (#2632)
actuals:
  tokens: 11018
  tasks: 3
  commits: 3

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Ported, never imported: legacy algorithm bodies copied with attribution comments,
      never `from trading_crab_lib.momentum import ...` — keeps the legacy-import ratchet
      (tests/unit/test_platform_legacy_import_ratchet.py) unchanged at 31."
    - "Window-constant re-derivation in prose: legacy quarterly numerals are never carried
      across a cadence boundary unconverted — the arithmetic is written out in a comment."
    - "Raise, never warn-and-fallback, on a caller-supplied identifier absent from the
      fitted feature set — applied to canonicalize_states' sort_column."

key-files:
  created:
    - src/trading_crab_lib/platform/features/__init__.py
    - src/trading_crab_lib/platform/features/relative.py
    - tests/unit/test_platform_features_relative.py
  modified:
    - src/trading_crab_lib/platform/labeling/jump_model.py
    - config/platform_settings.yaml
    - tests/unit/test_platform_labeling.py
    - tests/unit/test_platform_macro_ingest.py

key-decisions:
  - "canonicalize_states' sort_column defaults to trailing_return_1m (classifier #1's
    ordering key) and is keyword-only — classifier #1's three production call sites
    (driver.py:262, evaluation/report.py:924, labeling/diagnostics.py:286) pass it
    positionally with 3 args and are therefore byte-identical after the change."
  - "compute_rolling_cross_correlation's default pairs are derived from
    DEFAULT_RELATIVE_PAIRS (one source of truth) rather than a separate
    DEFAULT_CORRELATION_PAIRS constant, since D-10/D-11's candidate set has no reason to
    vary the window per pair (unlike the legacy per-pair-window design)."
  - "M2SL/TOTALSL route through fred_monthly.series (the fast-layer path used by GS10/
    WTISPLC/etc.), NOT align_agency_monthly's ALFRED vintage path — both are natively
    monthly and need no forward-fill or vintage correction, matching every other
    fred_monthly.series entry's treatment (none of which carry a discontinuity guard
    either)."

requirements-completed: [REG-01, INV-01]

coverage:
  - id: D1
    description: "canonicalize_states takes a keyword-only sort_column, raises ValueError
      (never warns) when it is absent from feature_names, and is byte-identical to prior
      behavior for classifier #1's three production call sites."
    requirement: REG-01
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_labeling.py::TestCanonicalizeStatesSortColumn (4 tests)"
        status: pass
      - kind: unit
        ref: "tests/unit/test_platform_backtest_driver.py::TestFrozenPolicyEquivalence (positional 3-arg spy, unchanged)"
        status: pass
    human_judgment: false
  - id: D2
    description: "platform/features/relative.py computes classifier #2's candidate
      leadership columns (relative strength, momentum, rolling correlation, CPI
      acceleration) at monthly cadence with re-derived window constants, disjoint from
      classifier #1's 13 raw columns, with no new legacy import."
    requirement: REG-01
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_features_relative.py (21 tests)"
        status: pass
      - kind: unit
        ref: "tests/unit/test_platform_legacy_import_ratchet.py (ratchet unchanged at 31)"
        status: pass
    human_judgment: false
  - id: D3
    description: "M2SL and TOTALSL ingest through the existing config-driven FRED path;
      INV-01's two named invariant ratios (m2_gdp, credit_gdp) are computable and their
      boundary/adjacency behavior is pinned by test; live smoke fetch confirms both
      series' real start dates."
    requirement: INV-01
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_macro_ingest.py::TestInvariantSeriesIngestion (4 tests)"
        status: pass
      - kind: other
        ref: "live FRED smoke fetch: M2SL first_valid=1959-01-01 (811 obs), TOTALSL first_valid=1943-01-01 (1003 obs)"
        status: pass
    human_judgment: false

duration: ~10min (commit-to-commit span; excludes reading/research time)
completed: 2026-09-17
status: complete
---

# Phase 7 Plan 05: Canonicalization Fix + Classifier #2 Feature Substrate + INV-01 Ingestion Summary

**Closed the `canonicalize_states` landmine that would have silently corrupted every
classifier-#2 fit, ported classifier #2's leadership/relative-strength feature set to
monthly cadence, and ingested M2SL/TOTALSL for INV-01's named invariant ratios.**

## Performance

- **Duration:** ~10 min (span between first and last commit; research/reading time not
  separately tracked)
- **Started:** 2026-09-17T14:22:39Z (first commit)
- **Completed:** 2026-09-17T14:28:35Z (last commit)
- **Tasks:** 3/3 completed
- **Files modified:** 7 (3 created, 4 modified)

## Accomplishments

- **`canonicalize_states` now raises, never warns, on a missing sort column.** Added a
  keyword-only `sort_column: str = "trailing_return_1m"` parameter. The prior
  `log.warning` + "fall back to centroid column 0" branch is deleted entirely. Classifier
  #1's three production call sites (`driver.py:262`, `evaluation/report.py:924`,
  `labeling/diagnostics.py:286`) are unchanged in the diff and pass their 3 positional
  args exactly as before — the frozen ten-column set (D-02-A) always contains
  `trailing_return_1m`, so none of them can reach the new `ValueError`. The positional
  spy in `tests/unit/test_platform_backtest_driver.py:492`
  (`_real_canonicalize_states(states, centroids, feature_names)`) still calls the
  function correctly, proving the change is additive.
- **`platform/features/relative.py` built as classifier #2's raw feature substrate.**
  Four functions ported (never imported) from `src/trading_crab_lib/momentum.py`:
  `compute_trailing_momentum`, `compute_relative_strength`,
  `compute_rolling_cross_correlation`, `compute_inflation_acceleration`, plus a new
  `compute_invariant_ratios` for INV-01. `DEFAULT_RELATIVE_PAIRS` produces
  `rs_equities_bonds` (`equities_tr` / `long_duration_tr`) and `rs_oil_equities`
  (`oil` / `equities_tr`); `gold` is deliberately excluded (D-11's 1972+ freeze — gold
  starts 1985-02, oil survives from 1962-01).
- **M2SL and TOTALSL ingested through the existing FRED path.** Two entries added to
  `fred_monthly.series`, copying the `WTISPLC` shape exactly — no new ingestion code. A
  live smoke fetch confirms both series' real start dates and non-null counts (below).

## Task Commits

Each task was committed atomically:

1. **Task 1: Canonical state ordering for a disjoint feature set** - `5c4741e` (fix)
2. **Task 2: platform/features/relative.py — leadership features ported to monthly cadence** - `be0ffc6` (feat)
3. **Task 3: INV-01 ingestion — M2SL and TOTALSL into the monthly FRED spine** - `5ff27c8` (feat)

**Plan metadata:** this SUMMARY's own commit (docs, made separately per orchestrator instruction)

_Note: none of these tasks required a TDD RED→GREEN→REFACTOR gate sequence beyond the
tests-alongside-implementation pattern already used throughout `platform/`; tests were
written and verified passing together with each task's implementation before commit._

## Files Created/Modified

- `src/trading_crab_lib/platform/labeling/jump_model.py` - `canonicalize_states` gains
  keyword-only `sort_column`, raises `ValueError` (never warns) when absent
- `src/trading_crab_lib/platform/features/__init__.py` - new package marker for
  classifier #2's feature substrate
- `src/trading_crab_lib/platform/features/relative.py` - ported leadership/relative-
  strength functions + INV-01 invariant ratios, monthly cadence
- `config/platform_settings.yaml` - `M2SL`/`TOTALSL` added to `fred_monthly.series`,
  with a comment recording `BCNSDODNS`/`TOTBKCR` as considered-and-rejected
- `tests/unit/test_platform_labeling.py` - `TestCanonicalizeStatesSortColumn` (4 tests)
- `tests/unit/test_platform_features_relative.py` - new file (21 tests)
- `tests/unit/test_platform_macro_ingest.py` - `TestInvariantSeriesIngestion` (4 tests)

## Decisions Made

- **`canonicalize_states`'s default stays `trailing_return_1m`, keyword-only.** Matches
  the codebase's established shape for additive signature changes (`frozen_l1_features`,
  `trial_tag`, `min_train: int | None = None`). No production call site needed touching.
- **Correlation pairs reuse `DEFAULT_RELATIVE_PAIRS`'s (num, denom) columns** rather than
  a separate `DEFAULT_CORRELATION_PAIRS` constant — one source of truth for the two asset
  pairs classifier #2 cares about, and the window is a single re-derived constant
  (`MONTHLY_CORRELATION_WINDOW = 24`) rather than per-pair, since nothing in D-10/D-11
  calls for varying it.
- **M2SL/TOTALSL go through `fred_monthly.series` (fast layer), not
  `align_agency_monthly` (ALFRED vintage path).** Both series are natively monthly
  (verified live), so — like every other `fred_monthly.series` entry — they need no
  forward-fill or vintage correction. This matches the plan's own action text
  ("neither needs the quarterly-repeat forward-fill `fred_gdp` receives, and neither
  needs a new ingestion code path"). The threat model's read_first note pointed at
  `_warn_on_level_discontinuity`/`_series_kind` for context; those guards are specific
  to the ALFRED agency path and are not applied to any other `fred_monthly.series`
  member (`GS10`, `WTISPLC`, etc.) today, so not adding them here keeps M2SL/TOTALSL
  consistent with their sibling series rather than introducing a bespoke exception.

## Deviations from Plan

None — plan executed exactly as written. One naming clarification: the plan's
`<read_first>` for Task 1 refers to "the existing `TestCanonicalizeStates` class" —
the actual class in the codebase is named `TestCanonicalize` (not `TestCanonicalizeStates`).
This was a pre-existing naming detail in the codebase, not a plan defect requiring a
fix; `TestCanonicalizeStatesSortColumn` was added alongside it as instructed, without
modifying the existing class.

## Issues Encountered

None.

## Verification Results

```
pytest tests/unit/test_platform_labeling.py tests/unit/test_platform_features_relative.py \
  tests/unit/test_platform_macro_ingest.py tests/unit/test_platform_legacy_import_ratchet.py \
  tests/unit/test_platform_backtest_driver.py -q
=> 112 passed
```

```
pytest tests/unit/test_platform_labeling.py -k sort_column -q
=> 4 passed
pytest tests/unit/test_platform_macro_ingest.py -k Invariant -q
=> 4 passed
pytest tests/unit/test_platform_features_relative.py -q
=> 21 passed
```

**Legacy-import ratchet:** 31 before this plan, **31 after** — unchanged (`
MAX_LEGACY_IMPORT_SITES = 31` in `tests/unit/test_platform_legacy_import_ratchet.py` is
untouched in the diff; the AST scan was independently re-run and returned 31).

**`lean_feature_set(load_platform_config())`:** returns exactly **13** — unaffected by
the M2SL/TOTALSL additions (neither was added to `taxonomy.fast`/`.slow`).

**Live FRED smoke fetch (INV-01, no data committed):**

| Series | First valid | Non-null count |
|---|---|---|
| M2SL | 1959-01-01 | 811 |
| TOTALSL | 1943-01-01 | 1003 |

Both match `07-RESEARCH.md`'s live-verified figures exactly, confirming a real (not
mocked) fetch was performed.

**Full suite:** `pytest tests/ -q` → **1792 passed, 0 skipped, 0 failed** in ~117s
(baseline was 1752 passed, 0 skipped @ `main`/`34ffa30`). This run reflects the current
working tree, which also includes the concurrently-running sibling plan 07-06's
uncommitted work (`evaluation/deflated_sharpe.py`, `honesty/registry.py`,
`tests/unit/test_platform_honesty_registry.py`) — those files were left untouched and
unstaged, per the wave-2 isolation instruction. The 40-test delta over baseline
(1792 − 1752) is the combined new-test count from this plan (29: 4 + 21 + 4) plus
plan 07-06's own concurrent additions, not attributable to this plan alone.

## Next Phase Readiness

- The blocking `canonicalize_states` landmine is closed — classifier #2 can now be fit
  on a disjoint feature set with an explicit `sort_column` and will fail loudly, never
  silently, if that column is ever absent.
- `platform/features/relative.py` provides the raw candidate columns
  (`rs_equities_bonds`, `rs_oil_equities`, trailing-momentum columns, rolling
  correlation, `cpi_acceleration`) classifier #2's fit will consume, plus INV-01's
  `m2_gdp`/`credit_gdp` (guarded, computable once `monthly_raw` is rebuilt with the two
  new FRED series).
- Not yet done by this plan (later plans, per the plan's own scope note): the actual
  classifier #2 fit, `measure_labeling_dependence`, `blend_regime_tilts`,
  `total_trial_count`/`deflated_sharpe_ratio`, and ADR-0002. This plan's artifacts are
  the substrate those later plans build on.
- No blockers. A rebuild of the live `monthly_raw` checkpoint (via `build_monthly_spine`,
  a network-dependent pure function of cached raw data — no new re-ingest required for
  this plan's own tests, which are all synthetic-data unit tests) will be needed before
  `compute_invariant_ratios` produces real `m2_gdp`/`credit_gdp` values on the actual
  dev checkpoint, since `monthly_raw` on disk predates this plan's config addition.

---
*Phase: 07-regime-representation*
*Completed: 2026-09-17*

## Self-Check: PASSED

All 7 created/modified files verified present on disk (jump_model.py,
platform/features/__init__.py, platform/features/relative.py,
test_platform_features_relative.py, test_platform_labeling.py,
test_platform_macro_ingest.py, config/platform_settings.yaml) plus this SUMMARY. All 3
task commits (`5c4741e`, `be0ffc6`, `5ff27c8`) verified present in `git log --oneline --all`.
