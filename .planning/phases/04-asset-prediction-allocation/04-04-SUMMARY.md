---
phase: 04-asset-prediction-allocation
plan: 04
subsystem: platform-tripwire
tags: [tripwire, risk-monitoring, or-logic, ewma-vol, credit-spread, drawdown, cli]

# Dependency graph
requires:
  - phase: 04-asset-prediction-allocation (Plan 01)
    provides: daily_raw['SPY'] checkpoint (universe price ingestion), fred_daily_raw['fred_daaa'/'fred_dbaa'] checkpoint (daily credit ingestion)
  - phase: 04-asset-prediction-allocation (Plan 02)
    provides: ewma_vol() in platform/assets/vol.py (reused for the vol-spike signal, not duplicated)
provides:
  - "Daily tripwire monitor (L4-04): TripwireEscalation enum, escalate() count-driven OR-logic, 3 independent signal functions, run_tripwire orchestrator, standalone CLI"
affects: [phase-4-weekly-report, phase-5-honest-backtest]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "compute + __main__ synthetic self-check module shape (gap_lag.py analog), CLI prints final enum value as its last stdout line"
    - "count-driven OR-logic escalation: sum(bools) determines tier, never signal identity"
    - "defensive cfg.get('tripwire', {}) config reads with in-code defaults, not added to _REQUIRED_PLATFORM_SECTIONS"

key-files:
  created:
    - src/trading_crab_lib/platform/tripwire/__init__.py
    - src/trading_crab_lib/platform/tripwire/monitor.py
    - tests/unit/test_platform_tripwire.py
  modified: []

key-decisions:
  - "Skipped the optional tripwire_status.parquet artifact (plan marked it optional) — run_tripwire prints + returns only; add a report_tripwire() persistence helper if the weekly report needs tripwire history later"
  - "realized_vol_spike computes EWMA vol independently over the recent short-window slice and the trailing baseline slice (both via ewma_vol), rather than slicing a single full-series EWMA curve — keeps the two windows statistically independent and reuses the existing decay implementation without adding a new one"

requirements-completed: [L4-04]

coverage:
  - id: D1
    description: "escalate() OR-logic: 0 triggers -> NONE, exactly 1 -> RUN_WEEKLY_SCORING_EARLY (any of the 3 signals), 2 or 3 -> TIER1_DERISK_REVIEW; count-driven not identity-driven"
    requirement: "L4-04"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_tripwire.py#TestEscalateOrLogic::test_all_8_combinations"
        status: pass
      - kind: unit
        ref: "tests/unit/test_platform_tripwire.py#TestEscalateOrLogic::test_count_driven_not_identity_driven"
        status: pass
    human_judgment: false
  - id: D2
    description: "realized_vol_spike (vol family), credit_spread_velocity (credit family), spy_drawdown_from_peak (price family) each fire/quiet correctly on synthetic fire-case and quiet-case series"
    requirement: "L4-04"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_tripwire.py#TestRealizedVolSpike"
        status: pass
      - kind: unit
        ref: "tests/unit/test_platform_tripwire.py#TestCreditSpreadVelocity"
        status: pass
      - kind: unit
        ref: "tests/unit/test_platform_tripwire.py#TestSpyDrawdownFromPeak"
        status: pass
    human_judgment: false
  - id: D3
    description: "run_tripwire orchestrator works with injected series (tests) and loads daily_raw['SPY'] / fred_daily_raw['fred_daaa'/'fred_dbaa'] from get_platform_checkpoint_manager() when series are omitted (live path)"
    requirement: "L4-04"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_tripwire.py#TestRunTripwire::test_reads_daily_raw_and_fred_daily_raw_when_series_omitted"
        status: pass
    human_judgment: false
  - id: D4
    description: "Standalone daily CLI (python3 -m trading_crab_lib.platform.tripwire.monitor) exits 0 and prints one escalation enum value as its last stdout line, no network"
    requirement: "L4-04"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_tripwire.py#TestCLI::test_cli_prints_escalation_value_as_last_line_no_network"
        status: pass
    human_judgment: false

# Metrics
duration: 35min
completed: 2026-07-23
status: complete
---

# Phase 4 Plan 04: Daily Tripwire Monitor Summary

**Standalone daily risk tripwire — realized-vol spike, credit-spread velocity, and SPY drawdown-from-peak combined by count-driven OR-logic into a 3-tier escalation enum, runnable via `python3 -m trading_crab_lib.platform.tripwire.monitor`.**

## Performance

- **Duration:** ~35 min
- **Started:** 2026-07-23T08:15:00Z (approx)
- **Completed:** 2026-07-23T08:50:34Z
- **Tasks:** 1 (TDD: RED then GREEN)
- **Files modified:** 3 (2 created source, 1 created test)

## Accomplishments
- `TripwireEscalation(str, Enum)` with `NONE` / `RUN_WEEKLY_SCORING_EARLY` / `TIER1_DERISK_REVIEW`
- `escalate(vol_spike, credit_velocity, spy_drawdown)` — pure count-driven OR-logic, proven against all 8 combinations of the truth table plus an explicit identity-swap invariant test
- `realized_vol_spike()` (vol family) — recent short-window EWMA vol vs trailing-baseline EWMA vol, reusing `ewma_vol()` from `platform/assets/vol.py` (no duplicate decay math)
- `credit_spread_velocity()` (credit family) — DBAA−DAAA spread widening in bps over a lookback window
- `spy_drawdown_from_peak()` (price family) — `cummax()`-based drawdown-from-peak
- `run_tripwire()` orchestrator — accepts injected series for tests, falls back to `get_platform_checkpoint_manager().load("daily_raw")["SPY"]` and `.load("fred_daily_raw")[["fred_daaa","fred_dbaa"]]` on the live path; thresholds read via `cfg.get("tripwire", {})` matching `config/platform_settings.yaml`'s already-present `tripwire:` section
- `__main__` CLI — synthetic self-check (no network, no checkpoint dependency), prints the escalation enum's `.value` as the last stdout line

## Task Commits

1. **Task 1 (RED): tests for tripwire signals + OR-logic + CLI** - `9c9d4fe` (test)
2. **Task 1 (GREEN): tripwire monitor implementation** - `b045b41` (feat)

_TDD plan: RED (failing tests, `ModuleNotFoundError`) confirmed before GREEN implementation._

## Files Created/Modified
- `src/trading_crab_lib/platform/tripwire/__init__.py` - package docstring, no logic
- `src/trading_crab_lib/platform/tripwire/monitor.py` - enum, 3 signal functions, `escalate`, `run_tripwire`, `__main__` CLI
- `tests/unit/test_platform_tripwire.py` - 27 tests: 8-combination truth table, identity-invariance, per-signal fire/quiet cases, orchestrator (injected + checkpoint-mocked), literal/grep checks (T-04-12, cfg.get pattern, provisional marker), CLI subprocess test

## Decisions Made
- Skipped the plan's explicitly-optional `tripwire_status.parquet` artifact — `run_tripwire()` prints (`log.info` + a one-line `print`) and returns the enum; no parquet persistence added. The plan text marks this "Optionally persist" and the artifacts list calls it "(optional)" — deferred until the weekly report or a future plan actually needs tripwire history. `report_tripwire()`-style helper (mirroring `gap_lag.py`/`vol.py`) is the natural upgrade path if that need arises.
- `realized_vol_spike` slices `daily_returns` into a `short_window` tail and a `baseline_window` slice immediately preceding it, computing `ewma_vol()` independently on each slice rather than reading two points off one continuous EWMA curve — keeps the two windows as genuinely separate vol estimates (satisfies "recent vs trailing baseline" framing) while still reusing the single `ewma_vol()` implementation for both (grep-verified in tests).
- Numeric thresholds are NOT hardcoded as bare literals — every default mirrors the exact keys and provisional values already present in `config/platform_settings.yaml`'s `tripwire:` section (written by an earlier Wave 0/1 plan), and the module docstring + inline comments carry the `provisional-until-Phase-5-backtest` (A3) language so `grep -i provisional` finds it.

## Deviations from Plan

None — plan executed as written. The one explicitly-optional deliverable (parquet artifact) was deliberately not built; see Decisions Made above (not a deviation from the plan's own instructions, which mark it optional).

## Issues Encountered
None. `trading_crab_lib` resolves to the main checkout's `site-packages`-linked copy by default in this environment; `PYTHONPATH=src` was used for every test/CLI invocation (per this plan's parallel-execution constraint: no `pip install -e`) to ensure the worktree's own `src/trading_crab_lib/platform/tripwire/` module was exercised, not a stale copy.

## User Setup Required
None — no external service configuration required. Live data wiring (`FRED_API_KEY`, `yfinance` network access) remains a deferred human-verification item per RESEARCH.md, same as Phase 1; the CLI's `__main__` self-check is fully synthetic and proves the logic end-to-end without it.

## Next Phase Readiness
- L4-04 tripwire monitor is complete and independently runnable (`python3 -m trading_crab_lib.platform.tripwire.monitor`).
- `run_tripwire(cfg)` (no injected series) is ready to be wired into a future daily-cron or weekly-report-early-trigger flow once live `daily_raw`/`fred_daily_raw` data exists.
- No blockers for Phase 4 Plan 03 (allocation) or the weekly report plan — this plan touched only `platform/tripwire/` and its test file, no shared module was modified.

---
*Phase: 04-asset-prediction-allocation*
*Completed: 2026-07-23*
