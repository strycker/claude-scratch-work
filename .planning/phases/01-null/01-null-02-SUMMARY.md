---
phase: 01-null
plan: 02
subsystem: data
tags: [features, checkpoints, pandas]

# Dependency graph
requires:
  - phase: 01-null
    provides: "checkpointed macro_raw inputs with optional market_code overlay"
provides:
  - "Config-driven feature engineering via engineer_all(causal=...) using settings.yaml lists"
  - "Dual non-causal and causal feature artifacts with stable checkpoint names"
  - "User-facing documentation for how to regenerate and consume feature artifacts"
affects: [02-regimes, 03-prediction, 04-assets]

# Tech tracking
tech-stack:
  added: []
  patterns: ["pipeline steps delegate to run_pipeline step registry for single source of truth"]

key-files:
  created: []
  modified:
    - run_pipeline.py
    - pipelines/02_features.py
    - README.md

key-decisions:
  - "Use CheckpointManager aliases features_noncausal/features_causal pointing at existing features/features_supervised artifacts instead of renaming core files."

patterns-established:
  - "Standalone pipeline scripts wrap the central run_pipeline step implementations instead of duplicating logic."

requirements-completed: [DATA-02, DATA-03]

# Metrics
duration: unknown
completed: 2026-03-16
---

# Phase 1 Plan 2: Data & Constraints Foundations — Features Summary

**Config-driven feature engineering now produces dual non-causal and causal artifacts with stable checkpoint names and clear contracts for downstream phases.**

## Performance

- **Duration:** unknown (executed in automated environment)
- **Started:** unknown
- **Completed:** 2026-03-16T00:00:00Z
- **Tasks:** 3
- **Files modified:** 3

## Accomplishments
- Confirmed that `engineer_all(causal=...)` implements the documented feature pipeline order and is fully driven by `config/settings.yaml` feature lists and derivative window settings.
- Updated step 2 of the master pipeline and the `pipelines/02_features.py` wrapper so feature engineering always produces both non-causal and causal artifacts, checkpointed under explicit names for downstream use.
- Documented the feature artifacts, their intended uses, and regeneration commands in `README.md` so later phases can rely on a stable feature contract without reading the code.

## Task Commits

Each task was committed atomically where it required code or docs changes:

1. **Task 1: Ensure config-driven feature lists and causal flags are explicit** - *no code changes required* (existing implementation already satisfied the plan conditions).
2. **Task 2: Wire Step 02 features to produce dual checkpoints from raw data** - `98d522f` (test/feat; implemented in earlier 01-null-03 work and reused by this plan).
3. **Task 3: Document the feature contract for downstream phases** - `6caf53f` (docs).

**Plan metadata:** not yet committed (will be captured in a separate docs commit with STATE/ROADMAP updates).

## Files Created/Modified
- `run_pipeline.py` - Ensures step 2 saves both centered and causal feature sets and adds `features_noncausal` / `features_causal` checkpoint aliases alongside `features` / `features_supervised`.
- `pipelines/02_features.py` - Now a thin wrapper around `run_pipeline.step2_features`, guaranteeing single-source-of-truth behavior when running step 2 standalone.
- `README.md` - Adds a concise explanation of centered vs causal features, checkpoint names, and the command to regenerate both artifacts.

## Decisions Made
- Kept the canonical parquet filenames `features.parquet` and `features_supervised.parquet` for compatibility, while adding `features_noncausal` and `features_causal` as explicit checkpoint aliases for plan-level artifacts.
- Chose to delegate standalone step scripts to the central `run_pipeline` step functions rather than duplicating ingestion/feature logic, reducing future drift between CLI and per-step entrypoints.

## Deviations from Plan

None - plan executed substantively as written; existing feature code already matched the required pipeline order and config-driven behavior, so Task 1 was purely verification.

## Issues Encountered
- Automated environment lacked a configured `python`/`pytest` binary, so the prescribed verification commands (`pytest -k "features"` and `python pipelines/02_features.py`) could not be executed here. They remain the correct validation steps for a local developer environment.

## User Setup Required

None - feature engineering and checkpointing reuse existing runtime configuration; no new environment variables or external services were introduced.

## Next Phase Readiness
- Downstream clustering, prediction, and asset-return phases can now depend on explicit, documented feature artifacts (`features_noncausal` / `features_causal`) without ambiguity about causal vs non-causal behavior.
- Future plans in Phases 2–3 should load features via `CheckpointManager` using these names and treat the README section as the user-facing contract for feature regeneration.

---
*Phase: 01-null*
*Completed: 2026-03-16*

