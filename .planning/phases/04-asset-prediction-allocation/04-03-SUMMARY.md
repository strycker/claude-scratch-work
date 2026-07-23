---
phase: 04-asset-prediction-allocation
plan: 03
subsystem: allocation
tags: [hysteresis, schmitt-trigger, vol-targeting, pandas, ewma, checkpoints]

# Dependency graph
requires:
  - phase: 04-asset-prediction-allocation (Plan 01/02, wave 1)
    provides: platform/assets/vol.py (ewma_vol), config/platform_settings.yaml allocation:/tripwire:/report: sections
provides:
  - "platform/allocation/hysteresis.py — Schmitt-trigger state machine on P(held regime), load-before-save persistence"
  - "platform/allocation/tilt.py — vol_target_scale, portfolio_vol (naive no-covariance), regime_tilt_weights, vol_targeted_tilt"
affects: [04-04-weekly-report, 04-05-tripwire]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Schmitt trigger on the HELD entity's own probability (never argmax) for anti-flicker state machines"
    - "load-before-save checkpoint ordering (mirrors labeling/diagnostics.py _churn_against_previous)"
    - "conservative-fallback ceiling: linear-sum-of-vols over-estimates on purpose so it can only under-lever"

key-files:
  created:
    - src/trading_crab_lib/platform/allocation/__init__.py
    - src/trading_crab_lib/platform/allocation/hysteresis.py
    - src/trading_crab_lib/platform/allocation/tilt.py
    - tests/unit/test_platform_hysteresis.py
    - tests/unit/test_platform_tilt.py
  modified: []

key-decisions:
  - "regime_tilt_weights(regime, returns_by_regime, probs, min_obs_flag) blends per-regime Sharpe-rank tilts by nowcaster probs; `regime` is only consulted as a single-regime point-bet fallback when probs is empty/all-zero (Claude's discretion, D-03/§7)"
  - "vol_targeted_tilt accepts regime_or_probs as either a probability mapping (blended tilt) or a bare regime id (100% point bet on that regime)"
  - "Cold-start hysteresis rule (A1, RESEARCH-flagged assumption): argmax(probs) if max >= act_threshold, else None — documented in the docstring for report visibility, not silently baked in"

patterns-established:
  - "Pattern: Schmitt-trigger state machines track P(the currently-held entity), never argmax(P) — same shape reusable for any future anti-flicker gate"

requirements-completed: [L4-01]

coverage:
  - id: D1
    description: "Hysteresis Schmitt-trigger state machine (load_active_regime, save_active_regime, update_active_regime) — headline no-flip invariant on an oscillating 0.65<->0.72 probability path"
    requirement: "L4-01"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_hysteresis.py::TestNoFlipInvariant#test_oscillating_probability_never_flips_active_regime"
        status: pass
      - kind: unit
        ref: "tests/unit/test_platform_hysteresis.py::TestNoFlipInvariant#test_competitor_spike_does_not_steal_active_regime"
        status: pass
      - kind: unit
        ref: "tests/unit/test_platform_hysteresis.py::TestUnwindAndSwitch"
        status: pass
      - kind: unit
        ref: "tests/unit/test_platform_hysteresis.py::TestColdStart"
        status: pass
      - kind: unit
        ref: "tests/unit/test_platform_hysteresis.py::TestPersistence"
        status: pass
    human_judgment: false
  - id: D2
    description: "Vol-targeted tilt (vol_target_scale, portfolio_vol, regime_tilt_weights, vol_targeted_tilt) — leverage cap at 1.0, zero-vol degrade, cash residual"
    requirement: "L4-01"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_tilt.py::TestVolTargetScale"
        status: pass
      - kind: unit
        ref: "tests/unit/test_platform_tilt.py::TestPortfolioVol"
        status: pass
      - kind: unit
        ref: "tests/unit/test_platform_tilt.py::TestRegimeTiltWeights"
        status: pass
      - kind: unit
        ref: "tests/unit/test_platform_tilt.py::TestVolTargetedTilt"
        status: pass
    human_judgment: false

# Metrics
duration: 25min
completed: 2026-07-23
status: complete
---

# Phase 4 Plan 3: Vol-Targeted Regime Tilt + Hysteresis Summary

**Schmitt-trigger hysteresis on P(held regime) plus a leverage-capped, cash-residual vol-targeted tilt — 25 tests, TDD RED/GREEN throughout, zero new dependencies.**

## Performance

- **Duration:** 25 min
- **Tasks:** 2
- **Files modified:** 5 (all new)

## Accomplishments
- `hysteresis.py`: `update_active_regime` is a pure Schmitt-trigger state machine tracking the HELD regime's own probability (never argmax) — proven by a test that a 0.65<->0.72 oscillation never flips `active_regime`, and that a competitor spiking to 0.72 while the held regime is still >=0.40 does NOT steal the tilt.
- `load_active_regime`/`save_active_regime` persist via `get_platform_checkpoint_manager()` (parquet), None-safe on both cold-start (`FileNotFoundError`) and a persisted null value.
- `tilt.py`: `vol_target_scale` hard-caps at `min(1.0, target/actual)` and degrades to `0.0` on non-positive vol — no code path can lever.
- `portfolio_vol` imports `ewma_vol` from Plan 02's `platform/assets/vol.py` (no duplicate EWMA math); falls back to the documented `# ponytail:` conservative linear-sum-of-vols estimate when joint history is short, which can only under-lever, never over-lever.
- `regime_tilt_weights` ranks assets by within-regime annualized Sharpe, clips negatives to 0 (long-only), normalizes to 1.0, and blends across regimes by nowcaster probabilities.
- `vol_targeted_tilt` assembles the full pipeline: base tilt -> portfolio_vol -> scale -> `{"weights", "cash", "scale", "portfolio_vol"}`, with `cash == 1 - scale` and total weight == 1.0.

## Task Commits

Each task was committed as a TDD RED/GREEN pair:

1. **Task 1: Hysteresis state machine** — RED `2cf4d90` (test), GREEN `cbe4a5f` (feat)
2. **Task 2: Vol-targeted tilt** — RED `fb1570f` (test), GREEN `7f15b6c` (feat)

**Plan metadata:** this SUMMARY commit (docs)

## Files Created/Modified
- `src/trading_crab_lib/platform/allocation/__init__.py` - new package docstring
- `src/trading_crab_lib/platform/allocation/hysteresis.py` - `load_active_regime`, `save_active_regime`, `update_active_regime`
- `src/trading_crab_lib/platform/allocation/tilt.py` - `vol_target_scale`, `portfolio_vol`, `regime_tilt_weights`, `vol_targeted_tilt`
- `tests/unit/test_platform_hysteresis.py` - 11 tests: no-flip invariant, unwind+switch, cold start, pure-function grep guard, load-before-save round trip
- `tests/unit/test_platform_tilt.py` - 14 tests: scale cap/degrade, blended-EWMA vs linear-sum-fallback portfolio_vol, tilt clipping/normalization/determinism, cash residual

## Decisions Made
- `regime_tilt_weights`'s `regime` positional parameter (required by the plan's artifact signature) is used only as a single-regime point-bet fallback when `probs` is empty or sums to zero — the normal path always blends across every regime present in `probs`, matching the design's "shrink brutally, never a single point-forecast bet" doctrine (§7). Documented in the function's docstring.
- `vol_targeted_tilt`'s `regime_or_probs` accepts either a probability mapping or a bare regime id, resolved via `isinstance` — keeps the function usable both from the nowcaster's full probability output and from a degenerate single-regime caller without a second code path.
- Cold-start hysteresis rule (argmax if max(probs) >= act_threshold, else None) is explicitly flagged in the docstring as `[ASSUMED — A1]` per RESEARCH.md's own flagging — visible for report-time confirmation rather than silently baked in.

## Deviations from Plan

None - plan executed exactly as written. Both tasks followed the plan's exact `<action>` specifications (function names, signatures, and RED-test invariants), and both TDD RED gates were confirmed failing (ModuleNotFoundError) before the corresponding GREEN implementation.

## Issues Encountered

None. Threat-model mitigations (T-04-07 no-flip invariant, T-04-08 leverage cap, T-04-10 load-before-save ordering) are all directly covered by the RED-gate tests listed above; T-04-09 (no tuning against holdout data) required no code change — the thresholds are read from `config/platform_settings.yaml` (already populated by Wave 1) via `.get()` defaults, never computed against data in this plan.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- `hysteresis.py` and `tilt.py` are ready for Plan 04/05 (weekly report, tripwire) to consume: `update_active_regime` for the report's regime-hold decision, `vol_targeted_tilt` for the report's target-weight table.
- `regime_tilt_weights` expects the long-format `returns_by_regime` table (columns `regime`/`asset`/`sharpe_annualized`/`n_obs`) — matches Plan 02's `returns_by_regime_stats()` output shape; no adapter needed.
- Full platform suite green: 242 passed, 1 skipped (pre-existing, unrelated) across `tests/unit/test_platform_*.py`.

---
*Phase: 04-asset-prediction-allocation*
*Completed: 2026-07-23*
