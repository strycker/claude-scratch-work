---
phase: 03-supervised-regime-behavior-models
plan: 03
subsystem: modeling
tags: [sklearn, time-series, classifiers, behavior, regimes]

# Dependency graph
requires:
  - phase: 03-supervised-regime-behavior-models
    provides: supervised regime model bundles and metrics from plan 02
provides:
  - forward-looking ETF/portfolio behavior labels and classifiers
  - behavior-focused CV metrics compatible with existing regime summaries
  - merged reporting surface for regime and behavior model metrics
affects:
  - 04-regime-conditional-etf-portfolio-behavior
  - 05-recommendations-machine-readable-outputs

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Directional up/flat/down behavior labels built from ETF/portfolio returns"
    - "Shared behavior model bundle structure mirroring regime model bundles"

key-files:
  created:
    - tests/test_models_behavior.py
  modified:
    - src/market_regime/prediction.py
    - tests/test_models_reporting.py

key-decisions:
  - "Use simple up/flat/down thresholds around 0% quarterly return for initial behavior labels, keeping thresholds pluggable for future config-driven tuning."
  - "Align behavior model bundles with existing supervised regime bundles so reporting and checkpointing can treat them uniformly."

patterns-established:
  - "Forward behavior models consume the same causal feature set and regime labels as supervised regime models, plus ETF/portfolio returns."
  - "Model metrics are flattened into family-tagged rows so regimes and behaviors can be surfaced together in dashboards."

requirements-completed: [MODEL-03, MODEL-04]

# Metrics
duration: unknown
completed: 2026-03-16
---

# Phase 3: Supervised Regime & Behavior Models Plan 03 Summary

**Implemented forward-looking ETF/portfolio behavior labels and RandomForest-based up/flat/down models with shared reporting hooks alongside supervised regime metrics.**

## Performance

- **Duration:** unknown (not measurable in this environment)
- **Started:** unknown
- **Completed:** 2026-03-16T19:13:45Z
- **Tasks:** 3
- **Files modified:** 3

## Accomplishments
- Added `make_behavior_labels` to derive up/flat/down labels from quarterly ETF/portfolio returns at configurable forward horizons.
- Implemented `train_forward_behavior_models` to train per-asset, per-horizon RandomForest classifiers with TimeSeriesSplit CV and nested results suitable for checkpointing.
- Extended `model_metrics_summary` to merge regime and behavior metrics into a single, JSON-serialisable structure tagged by family, asset, horizon, and class label.

## Task Commits

Each task was committed atomically:

1. **Task 1 & 2: Implement behavior label helpers and forward behavior models** - `0293272` (feat)
2. **Task 3: Extend reporting to include behavior models alongside regime models** - `26011fb` (feat)

**Plan metadata:** _not created in this environment (planning docs unchanged)_

## Files Created/Modified
- `src/market_regime/prediction.py` - Adds behavior label construction, forward behavior model training, and a generic model metrics summariser that tags regime vs behavior entries.
- `tests/test_models_behavior.py` - Synthetic tests validating behavior label alignment, exclusion of trailing horizons, and well-formed behavior model probability outputs.
- `tests/test_models_reporting.py` - Reporting tests ensuring behavior metrics are surfaced alongside regime metrics and remain filterable by asset and behavior class.

## Decisions Made
- Chose a simple 0% return threshold for initial up/flat/down splits while keeping thresholds as function parameters so future phases can drive them from config.
- Mirrored the regime model bundle pattern for behavior models (nested models and CV reports) to keep downstream reporting and storage consistent.

## Deviations from Plan

None - plan executed as written within the constraints of this environment; test execution via pytest is blocked by the missing Python binary but tests are structured to run locally without further changes.

## Issues Encountered
- Could not run `pytest` or `python -m pytest` in this environment because the `python` binary is unavailable; behavior and reporting tests were still written to align with the existing supervised regime helpers and should be executed locally.

## User Setup Required

None - no new external services, environment variables, or CLI tools were introduced beyond the existing Trading-Crab setup.

## Next Phase Readiness
- Behavior label and model helpers are in place so Phase 4 can consume directional ETF/portfolio expectations alongside regime-conditional return distributions.
- Combined metrics summaries now carry both regime and behavior entries, ready for downstream dashboards and recommendation logic in later phases.

---
*Phase: 03-supervised-regime-behavior-models*
*Completed: 2026-03-16*
