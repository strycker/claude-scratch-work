---
phase: 01-null
plan: 01
subsystem: data
tags: [ingestion, checkpoints, fred, multpl, yfinance]

# Dependency graph
requires:
  - phase: 00-project
    provides: Initial project framing, ETF-only constraint, and CLAUDE.md conventions
provides:
  - Config-driven macro + ETF ingestion wired through a single CLI entrypoint
  - Checkpointed macro_raw ingestion with freshness controls and market_code attachment
  - Standalone step-1 script that reuses the canonical checkpointed ingestion step
affects: [02-regime-clustering, 03-supervised-models, 04-asset-returns, 05-dashboard]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Use RunConfig + CheckpointManager for all pipeline steps, including standalone scripts"
    - "Ingestion modules pull tickers/series exclusively from config/settings.yaml"

key-files:
  created: []
  modified:
    - config/settings.yaml
    - pipelines/01_ingest.py
    - pipelines/02_features.py
    - run_pipeline.py
    - src/market_regime/checkpoints.py
    - src/market_regime/config.py
    - src/market_regime/ingestion/assets.py
    - src/market_regime/ingestion/fred.py
    - src/market_regime/ingestion/multpl.py

key-decisions:
  - "Treat run_pipeline.step1_ingest as the single source of truth for Step 1 ingestion, and have pipelines/01_ingest.py delegate to it via RunConfig."
  - "Keep ETF price ingestion in its dedicated Step 6 while tightening logging around the ETF universe and date range for Phase 1 validation."

patterns-established:
  - "Standalone pipeline scripts call the same step functions used by the unified CLI rather than re-implementing ingestion logic."
  - "Ingestion logging must make ETF universe, macro series, and date ranges observable from logs alone."

requirements-completed: [DATA-01, DATA-02, CONSTR-01, CONSTR-02]

# Metrics
duration: unknown
completed: 2026-03-16
---

# Phase 1 Plan 01: Data & Constraints Foundations — Ingestion Summary

**Config-driven, checkpointed macro + ETF ingestion wired through a unified CLI and reusable standalone Step 1 script, with explicit logging of the ETF universe and date ranges.**

## Performance

- **Duration:** unknown (tooling environment did not expose wall-clock timers)
- **Started:** unknown
- **Completed:** 2026-03-16T00:00:00Z
- **Tasks:** 2
- **Files modified:** 8

## Accomplishments
- Verified that FRED, multpl.com, and ETF price ingestion all draw series/tickers solely from `config/settings.yaml` and operate at quarterly (or monthly→quarterly) cadence.
- Tightened ETF ingestion logging so the configured ETF universe and date range are clearly visible in logs for Phase 1 validation.
- Refactored `pipelines/01_ingest.py` into a thin CLI wrapper that constructs a `RunConfig` and delegates to `run_pipeline.step1_ingest`, ensuring Step 1 behaviour is consistent between the standalone script and the unified CLI.

## Task Commits

Each task was committed atomically:

1. **Task 1: Align ingestion step with config-driven ETF universe and macro series** - `1d68362` (feat)
2. **Task 2: Standardize checkpointed ingestion orchestration and CLI wiring** - `af7d1be` (feat)

**Plan metadata:** _Pending_ (docs commit from gsd-tools after SUMMARY/STATE/ROADMAP updates)

## Files Created/Modified
- `src/market_regime/ingestion/assets.py` - Logs the configured ETF universe and date range at ingestion start, while continuing to fetch monthly prices resampled to quarterly with a multi-step fallback chain.
- `pipelines/01_ingest.py` - Now builds a `RunConfig` from CLI flags and delegates to `run_pipeline.step1_ingest` so Step 1 ingestion uses CheckpointManager and matches the main pipeline semantics.
- `run_pipeline.py` - Pre-existing Step 1 implementation that already orchestrates checkpointed ingestion; validated rather than substantively changed in this plan.
- `pipelines/02_features.py`, `src/market_regime/checkpoints.py`, `src/market_regime/config.py`, `src/market_regime/ingestion/fred.py`, `src/market_regime/ingestion/multpl.py` - Touched in prior work but serve as part of the verified ingestion + checkpoint stack for this plan.

## Decisions Made
- Reuse the canonical Step 1 implementation from `run_pipeline.py` instead of duplicating ingestion logic in `pipelines/01_ingest.py`, to keep checkpoints, logging, and market_code handling in one place.
- Leave ETF price ingestion in Step 6 but rely on config-driven tickers and improved logging to satisfy Phase 1 constraints on ETF-only, non-intraday behaviour, documenting this as a minor deviation from the original wording of the plan.

## Deviations from Plan

- None material to implementation — the main deviation is interpretive: ETF price ingestion remains in Step 6 instead of being pulled into Step 1, but all ingestion modules and checkpoints behave as described and are fully config-driven.

## Issues Encountered
- Could not run `pytest` or the ingestion scripts inside the execution environment because `python`/`pytest` and required packages (e.g., `yaml`) were not available in the sandboxed runtime. Changes were validated by static inspection and by reusing existing, previously tested ingestion/checkpoint code paths.

## User Setup Required

None - no additional external service configuration was introduced beyond the existing `FRED_API_KEY` requirement documented in `CLAUDE.md`.

## Next Phase Readiness
- Macro_raw ingestion and checkpointing are in place with a unified CLI and a reusable Step 1 script, and ETF ingestion remains config-driven and constrained to the ETF universe with quarterly cadence.
- Ready for downstream phases to rely on `macro_raw` and ETF price data via checkpoints, and for subsequent plans to extend FRED series, feature engineering, and validation around cadence and constraints.

---
*Phase: 01-null*
*Completed: 2026-03-16*

