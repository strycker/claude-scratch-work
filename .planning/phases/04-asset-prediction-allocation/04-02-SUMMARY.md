---
phase: 04-asset-prediction-allocation
plan: 02
subsystem: analytics
tags: [pandas, ewma, regime-conditioning, l3-asset-prediction]

# Dependency graph
requires:
  - phase: 04-asset-prediction-allocation (plan 01)
    provides: monthly_raw checkpoint (asset price/level columns), regime_labels checkpoint (state column)
provides:
  - "compute_monthly_returns / returns_by_regime_stats / report_returns_by_regime (L3-01)"
  - "ewma_vol / latest_ewma_vol / report_vol (L3-02) — single decay implementation for downstream reuse"
  - "returns_by_regime checkpoint + outputs/reports/platform/{returns_by_regime,ewma_vol}.parquet artifacts"
affects: [04-03 (allocation tilt — imports ewma_vol for portfolio_vol), 04-04 (tripwire — imports ewma_vol for vol-spike signal), 04-05 (report layer — consumes _MIN_OBS_FLAG low-confidence framing)]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "compute_*() pure function + report_*() persist/print + synthetic __main__ self-check (gap_lag.py shape)"
    - "index.intersection alignment, never iloc slicing (Pitfall P3 analog)"
    - "schema-stable empty-safe parquet artifact (_ARTIFACT_COLUMNS constant)"

key-files:
  created:
    - src/trading_crab_lib/platform/assets/__init__.py
    - src/trading_crab_lib/platform/assets/returns.py
    - src/trading_crab_lib/platform/assets/vol.py
    - tests/unit/test_platform_returns.py
    - tests/unit/test_platform_vol.py
  modified: []

key-decisions:
  - "returns_by_regime_stats never imports incumbent asset_returns.py/reporting.py (D-01) — inspiration only, verified by a source-grep test"
  - "_MIN_OBS_FLAG = 6 module constant surfaces D11 low-confidence (regime, asset) cells via n_obs rather than dropping or hiding them"
  - "report_returns_by_regime persists both the returns_by_regime platform checkpoint AND a schema-stable parquet, matching the gap_lag.py dual-persistence shape"
  - "ewma_vol is the single .ewm(halflife=...).std() implementation; Plans 03/04 import it rather than re-deriving the decay math"

patterns-established:
  - "assets/ package follows the honesty/gap_lag.py module shape exactly: pure compute -> report_* persist+print -> __main__ synthetic self-check"

requirements-completed: [L3-01, L3-02]

coverage:
  - id: D1
    description: "returns_by_regime_stats produces one row per (regime, asset) with mean/std/annualized-Sharpe/hit-rate/max-drawdown/n_obs, index-intersection aligned to regime labels"
    requirement: "L3-01"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_returns.py::TestReturnsByRegimeStats"
        status: pass
      - kind: unit
        ref: "tests/unit/test_platform_returns.py::TestReportReturnsByRegime"
        status: pass
    human_judgment: false
  - id: D2
    description: "ewma_vol matches a hand-computed RiskMetrics-style EWMA on a synthetic series and annualizes correctly for daily (sqrt(252)) and monthly (sqrt(12))"
    requirement: "L3-02"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_vol.py::TestEwmaVol"
        status: pass
      - kind: unit
        ref: "tests/unit/test_platform_vol.py::TestLatestEwmaVol"
        status: pass
      - kind: unit
        ref: "tests/unit/test_platform_vol.py::TestReportVol"
        status: pass
    human_judgment: false

# Metrics
duration: 20min
completed: 2026-07-22
status: complete
---

# Phase 4 Plan 02: Returns-by-Regime Tables + EWMA Vol Forecasts Summary

**`assets/returns.py` (regime-conditional return stats, D11 NULL-tolerant) + `assets/vol.py` (single reusable `.ewm(halflife=...)` decay implementation) — zero new dependencies, both proven via synthetic `__main__` self-checks and 18 new unit tests.**

## Performance

- **Duration:** ~20 min
- **Tasks:** 2 (both TDD RED->GREEN)
- **Files modified:** 5 (3 new source files, 2 new test files)

## Accomplishments
- `returns_by_regime_stats(returns, states)` conditions monthly asset returns on Phase-3 regime labels via `.index.intersection` (never `.iloc` slicing), returning one row per (regime, asset) with the seven-column stats schema the allocation tilt (Plan 03) and report layer (Plan 05) both need.
- `_MIN_OBS_FLAG = 6` module constant surfaces D11 short-history assets (e.g. a satellite ETF with only 3 usable return months) as explicitly low-confidence `n_obs` rows rather than dropping the column or crashing.
- `ewma_vol()` is the single RiskMetrics-style EWMA volatility function (`.ewm(halflife=..., min_periods=2).std() * sqrt(af)`) — verified against a hand-rolled `alpha = 1 - exp(-ln(2)/h)` reference — that Plan 03's `portfolio_vol()` and Plan 04's tripwire vol-spike signal will both import rather than re-derive.
- Both modules persist schema-stable, empty-safe parquet artifacts (`outputs/reports/platform/{returns_by_regime,ewma_vol}.parquet`); `returns.py` additionally writes the `returns_by_regime` platform checkpoint.

## Task Commits

Each task was committed atomically (TDD RED -> GREEN):

1. **Task 1: Returns-by-regime table** - `a89c689` (test: RED) -> `6223c30` (feat: GREEN)
2. **Task 2: EWMA volatility** - `c67408b` (test: RED) -> `a090eb5` (feat: GREEN)

**Plan metadata:** this commit (docs: complete plan)

## Files Created/Modified
- `src/trading_crab_lib/platform/assets/__init__.py` - empty package marker with module docstring
- `src/trading_crab_lib/platform/assets/returns.py` - `compute_monthly_returns`, `returns_by_regime_stats`, `report_returns_by_regime`, `_MIN_OBS_FLAG`
- `src/trading_crab_lib/platform/assets/vol.py` - `ewma_vol`, `latest_ewma_vol`, `report_vol`, `DAILY_ANNUALIZATION`, `MONTHLY_ANNUALIZATION`
- `tests/unit/test_platform_returns.py` - 9 tests: schema, D11 NULL-tolerance, Sharpe formula, index-intersection alignment, hit-rate/drawdown, persistence, no-incumbent-import grep
- `tests/unit/test_platform_vol.py` - 9 tests: hand-rolled EWMA reference match, annualization scaling, min_periods=2 NaN semantics, constant-series zero-vol, artifact persistence

## Decisions Made
- Followed RESEARCH.md Pattern 4 / PATTERNS.md verbatim for the stats-row construction and Pattern 1 (gap_lag.py shape) for both modules' overall structure — no deviation from the plan's provided code.
- `report_returns_by_regime` accepts both `output_dir` and `cm` kwargs so tests can fully isolate checkpoint and parquet I/O to `tmp_path` without touching the real platform checkpoint namespace.

## Deviations from Plan

None - plan executed exactly as written. Both tasks' `<action>` code closely mirrored the worked examples already provided in 04-RESEARCH.md Pattern 1/4 and 04-PATTERNS.md, requiring no architectural judgment calls.

## Issues Encountered
None.

## Next Phase Readiness
- `ewma_vol` and `returns_by_regime_stats`/`report_returns_by_regime` are ready for Plan 03 (allocation tilt) to import: `portfolio_vol()` per RESEARCH.md Pattern 5 calls `ewma_vol` directly on a blended weighted-return series.
- Plan 04 (tripwire) can import `ewma_vol` for its daily vol-spike signal without any interface change.
- Plan 05 (report layer) can read the `returns_by_regime` checkpoint and flag `n_obs < _MIN_OBS_FLAG` cells using the already-exported constant.
- No blockers.

---
*Phase: 04-asset-prediction-allocation*
*Completed: 2026-07-22*
