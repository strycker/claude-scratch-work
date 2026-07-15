---
phase: 01-monthly-data-layer-long-histories
plan: 07
subsystem: data
tags: [transforms, pandas, alfred, taxonomy, monthly-spine, pytest, tdd]

# Dependency graph
requires:
  - phase: 01-monthly-data-layer-long-histories
    provides: "platform package foundation + config/checkpoints (Plan 01), splice engine (Plan 02), ALFRED vintage ingestion (Plan 03), monthly macro ingestion (Plan 05), daily universe prices (Plan 06)"
provides:
  - "align_agency_monthly() — point-in-time agency-series alignment onto a monthly index via alfred.value_as_of/align_with_fallback, with a pre-vintage-era shift fallback built from each series' own first-published values"
  - "build_monthly_spine() — orchestrates macro_monthly + prices_daily + splice + align_agency_monthly into ONE monthly feature table, merged NULL-tolerantly via pd.concat(axis=1)"
  - "compute_lean_features() / tag_feature_columns() — the taxonomy-tagged fast/slow lean feature set (curve_10y3m, curve_10y2y, credit_spread_baa_aaa, fred_vix, gold, oil, trailing_return_1m/3m, realized_vol_1m/3m, cape_shiller, div_yield)"
  - "daily_raw / monthly_raw / monthly_features checkpoints in the platform namespace — the labeler's input of record"
affects: [phase-2-regime-labeling, phase-3-asset-prediction]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Fixed step-order orchestrator mirroring engineer_all()'s named-helper-per-step structure, applied to the monthly platform pipeline"
    - "Dependent-feature guard via `{'a', 'b'} <= cols` before computing (mirrors transforms.py::add_cross_ratios/add_yield_curve_features) — a missing source column skips the feature, never crashes"
    - "Concat-then-dedupe(keep='last') for merging passthrough lean columns with their monthly_raw source, avoiding duplicate column names without a manual column-existence branch per feature"

key-files:
  created:
    - src/trading_crab_lib/platform/transforms_monthly.py
    - tests/unit/test_platform_transforms.py
  modified: []

key-decisions:
  - "align_agency_monthly's pre-vintage-era shift_series is built from the SAME all_releases frame passed to it (each reference period's earliest realtime_start row, resampled to the monthly spine, ffilled, then shifted by one period) rather than requiring a separately-fetched plain series — keeps the function self-contained with a single ALFRED bulk-fetch call per series, at the cost of the fallback being NaN before that series' very first recorded reference period (acceptable: no data exists to fall back to before recorded history begins)"
  - "curve_10y2y is a direct passthrough of fred_t10y2y (FRED already publishes the 10Y-2Y spread natively) rather than a re-derivation from separately-ingested 10Y/2Y yield legs — no GS2 series is ingested, and FRED's own spread series is the correct source"
  - "realized_vol_1m is a naive |1-month-return| proxy (documented with a `ponytail:` comment), not a true intra-period realized vol — the equities_tr research series is monthly-only (derived from multpl price + dividend yield), so no daily granularity exists to compute a genuine sub-month vol; upgrade path noted for if/when a daily equity total-return series exists"
  - "Duplicate passthrough columns (gold/oil/fred_vix/cape_shiller/div_yield appear identically in both monthly_raw and the lean feature set) are resolved via pd.concat(axis=1) + drop_duplicates(keep='last') rather than pre-filtering compute_lean_features' output — keeps the literal 'concat axis=1' merge convention the plan specifies while avoiding a duplicate-column DataFrame"

patterns-established:
  - "Monthly platform orchestrator (build_monthly_spine) follows the same 'fetch → merge → derive → persist' shape as the incumbent's per-step pipeline, but as a single function rather than 9 CLI steps — appropriate for this phase's single deliverable (one feature table, not a multi-stage pipeline with intermediate human review points)"

requirements-completed: [DATA-01, DATA-03, DATA-04]

coverage:
  - id: D1
    description: "align_agency_monthly() point-in-time-aligns every fred_vintage series onto a monthly index via alfred.value_as_of/align_with_fallback — no revision or timing look-ahead"
    requirement: "DATA-03"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_transforms.py::TestBuildMonthlySpineEndToEnd::test_quarterly_series_alignment"
        status: pass
      - kind: unit
        ref: "tests/unit/test_platform_transforms.py::TestBuildMonthlySpineEndToEnd::test_pre_vintage_fallback_applied"
        status: pass
    human_judgment: false
  - id: D2
    description: "build_monthly_spine() orchestrates macro_monthly + prices_daily + splice + align_agency_monthly into one monthly feature table with ~12 rows/year cadence, merged NULL-tolerantly, persisting daily_raw/monthly_raw/monthly_features checkpoints in the platform namespace"
    requirement: "DATA-01"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_transforms.py::TestBuildMonthlySpineEndToEnd::test_monthly_row_count"
        status: pass
      - kind: unit
        ref: "tests/unit/test_platform_transforms.py::TestBuildMonthlySpineEndToEnd::test_short_history_satellite_null_tolerant"
        status: pass
      - kind: unit
        ref: "tests/unit/test_platform_transforms.py::TestBuildMonthlySpineLeanFeatures::test_persists_monthly_features_checkpoint"
        status: pass
    human_judgment: false
  - id: D3
    description: "compute_lean_features()/tag_feature_columns() produce the taxonomy-tagged lean fast/slow feature set with curve_10y3m == fred_gs10 - fred_tb3ms and credit_spread_baa_aaa == fred_baa - fred_aaa, and check_columns_tagged() on the produced columns returns empty"
    requirement: "DATA-04"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_transforms.py::TestComputeLeanFeatures (4 tests)"
        status: pass
      - kind: unit
        ref: "tests/unit/test_platform_transforms.py::TestTagFeatureColumns (2 tests)"
        status: pass
      - kind: unit
        ref: "tests/unit/test_platform_transforms.py::TestBuildMonthlySpineEndToEnd::test_every_feature_tagged"
        status: pass
    human_judgment: false
  - id: D4
    description: "No frozen incumbent module modified; full incumbent + platform test suite remains green"
    verification:
      - kind: unit
        ref: "git diff --stat -- transforms.py fred.py multpl.py macrotrends.py assets.py — empty"
        status: pass
      - kind: unit
        ref: "pytest tests/ -q — 797 passed, 49 skipped, 0 failed"
        status: pass
    human_judgment: false

# Metrics
duration: ~20min
completed: 2026-07-15
status: complete
---

# Phase 1 Plan 07: Monthly Feature Table Assembly Summary

**`build_monthly_spine()` — the phase's vertical join point: orchestrates macro_monthly + prices_daily + splice + ALFRED point-in-time alignment into one monthly feature table (1962+), with a taxonomy-tagged lean fast/slow feature set and three platform checkpoints.**

## Performance

- **Duration:** ~20 min
- **Completed:** 2026-07-15
- **Tasks:** 3 (Task 2 was TDD: RED → GREEN)
- **Files modified:** 2 (both new)

## Accomplishments
- `align_agency_monthly()` point-in-time-aligns every `cfg['fred_vintage']['series']` column onto a monthly index — `alfred.value_as_of` reconstruction where ALFRED vintages exist, publication-lag-shift fallback (built from each series' own first-published values) before the earliest recorded vintage. No revision or timing look-ahead (DATA-03 runtime application of Plan 03's ALFRED module).
- `build_monthly_spine()` orchestrates `macro_monthly.fetch_macro_monthly` + `prices_daily.fetch_universe_prices` + `splice.build_core_research_series` + `align_agency_monthly` into ONE monthly feature table, merged exclusively via `pd.concat([...], axis=1)` (NULL-tolerant outer join) onto a canonical month-end index spanning `cfg['data']['start_date']` forward — persisting `daily_raw`, `monthly_raw`, and `monthly_features` checkpoints in the platform namespace.
- `compute_lean_features()` derives the taxonomy-listed lean fast/slow feature set: `curve_10y3m` (fred_gs10 − fred_tb3ms), `curve_10y2y` (fred_t10y2y passthrough), `credit_spread_baa_aaa` (fred_baa − fred_aaa), `fred_vix`/`gold`/`oil` passthroughs, `trailing_return_1m`/`3m` and `realized_vol_1m`/`3m` on the equities research series, `cape_shiller`/`div_yield` passthroughs — every column name matches `config/platform_settings.yaml`'s taxonomy block exactly.
- `tag_feature_columns()` maps every produced column to its taxonomy tier via `taxonomy.classify_feature`, with a `taxonomy.check_columns_tagged` WARNING-only defensive gate (DATA-04's "every feature classified into exactly one tier" guarantee).
- 12 mocked unit tests in `tests/unit/test_platform_transforms.py` — no network/live checkpoint access — proving monthly cadence, no-look-ahead point-in-time agency alignment, pre-vintage fallback, taxonomy coverage, and NULL-tolerant satellite merging.
- No frozen incumbent module modified; full test suite (incumbent + all prior platform plans + this plan) remains green — 797 passed, 49 skipped, 0 failed.

## Task Commits

1. **Task 1: Monthly spine assembly + point-in-time agency alignment (DATA-01, DATA-03)** - `ee0a261` (feat)
2. **Task 2: Lean feature computation + taxonomy tagging + monthly_features checkpoint (DATA-01, DATA-04)** — TDD:
   - RED: `78a9e27` (test) — failing tests, `compute_lean_features`/`tag_feature_columns` did not exist yet
   - GREEN: `7d94820` (feat) — implementation, all 7 tests (Task 1's implicit coverage + Task 2's new tests) pass
3. **Task 3: Transforms test — monthly cadence, agency alignment (no look-ahead), taxonomy coverage** - `e05d05d` (test)

_No refactor commit needed — GREEN implementation was minimal and clean on first pass._

## Files Created/Modified
- `src/trading_crab_lib/platform/transforms_monthly.py` - `align_agency_monthly()`, `_shift_fallback_series()`, `compute_lean_features()`, `tag_feature_columns()`, `build_monthly_spine()`
- `tests/unit/test_platform_transforms.py` - 12 tests: 4 for `compute_lean_features`, 2 for `tag_feature_columns`/`check_columns_tagged`, 1 for the persisted `monthly_features` checkpoint, and 5 end-to-end tests (monthly row count, look-ahead guard, pre-vintage fallback, taxonomy coverage on the real persisted `monthly_raw` checkpoint, NULL-tolerant satellite merging)

## Decisions Made
- `align_agency_monthly`'s pre-vintage-era `shift_series` fallback is derived from the *same* `all_releases` bulk-fetch response passed into it (first-published value per reference period, resampled + ffilled + shifted by one period) rather than requiring a second, separately-fetched plain series — keeps the function self-contained behind a single ALFRED bulk call per series (RESEARCH Pitfall 3's cost concern), at the acceptable cost that the fallback itself is NaN before a series' very first recorded reference period.
- `curve_10y2y` is a direct passthrough of `fred_t10y2y` (already the FRED-published 10Y-2Y spread) rather than a re-derivation — no GS2 series is separately ingested by this phase's config.
- `realized_vol_1m` uses a naive `|1-month-return|` proxy (documented with a `ponytail:` comment naming the upgrade path) since the equities research series is monthly-only; a true intra-period vol would need daily equity total-return data not currently produced by this phase.
- Duplicate passthrough columns between `monthly_raw` and the lean feature set (gold/oil/fred_vix/cape_shiller/div_yield) are resolved via `pd.concat(axis=1)` + `drop_duplicates(keep="last")`, honoring the plan's literal "join them (concat axis=1)" instruction while avoiding duplicate-named columns in the final `monthly_features` frame.

## Deviations from Plan

None - plan executed exactly as written. All three tasks' `<verify>`/`<acceptance_criteria>` blocks pass as specified; no Rule 1-4 fixes were needed.

## Issues Encountered

None. The environment already had all required dependencies installed (pandas, pyarrow, fredapi, etc., from Plan 01's session setup) and no network/live-FRED calls were needed since all ingestion is mocked at the `macro_monthly.fetch_macro_monthly` / `prices_daily.fetch_universe_prices` / `alfred.fetch_all_vintages` boundary per the plan's `<read_first>`/`<action>` guidance. `FRED_API_KEY` was not exercised (no live calls made by this plan or its tests) — consistent with prior plans in this phase.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- `build_monthly_spine(load_platform_config())` is ready for a live end-to-end run (documented follow-up per the plan's `<verification>` block: "a single live run produces a `monthly_features` checkpoint with a month-end index reaching ~1962 and materially more rows than the quarterly incumbent" — needs `FRED_API_KEY` + network, not exercised by the automated suite).
- Phase 1's DATA-01/DATA-03/DATA-04 deliverables are now all runtime-complete: the monthly feature table exists, agency series are point-in-time aligned, and every lean feature carries exactly one taxonomy tag. Downstream phases (Phase 3 regime labeling) can consume `monthly_features` directly via `get_platform_checkpoint_manager().load("monthly_features")`.
- No blockers. This plan's `files_modified` list (`transforms_monthly.py`, `test_platform_transforms.py`) exactly matches what was created — no scope creep into sibling plans' files.

---
*Phase: 01-monthly-data-layer-long-histories*
*Completed: 2026-07-15*

## Self-Check: PASSED

All 2 claimed files verified present on disk (`src/trading_crab_lib/platform/transforms_monthly.py`,
`tests/unit/test_platform_transforms.py`); all 4 claimed commits (`ee0a261`, `78a9e27`, `7d94820`,
`e05d05d`) verified present in `git log --oneline --all`.
