---
phase: 05-honest-backtest-evaluation
plan: 01
subsystem: backtest
tags: [platform, config, transaction-costs, turnover, tdd, honesty-framework]

# Dependency graph
requires:
  - phase: 02-honesty-infrastructure
    provides: platform/config.py load/validate pattern, platform/honesty/holdout.py DEFAULT_HOLDOUT_CUTOFF
  - phase: 04-asset-prediction-allocation
    provides: platform/allocation/tilt.py vol_targeted_tilt() weight Series shape (input to compute_turnover)
provides:
  - "backtest: config section in config/platform_settings.yaml (cost_bps, min_train_months, refit_frequency_months, sixty_forty_rebalance, apply_cost_to_baselines, skip_l1l2_for_ablation, crisis_windows)"
  - "src/trading_crab_lib/platform/backtest/ package (backtest/costs.py: compute_turnover, apply_transaction_cost)"
  - "src/trading_crab_lib/platform/evaluation/ empty package scaffold"
  - "known-answer test proving the D-03 cost = turnover x bps honesty invariant"
affects: [05-02-walkforward-driver, 05-05-baselines]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "backtest: config section read defensively via cfg.get('backtest', {}), never added to _REQUIRED_PLATFORM_SECTIONS"
    - "index-union reindex(fill_value=0.0) for weight-Series diffing — never positional diffing"

key-files:
  created:
    - src/trading_crab_lib/platform/backtest/__init__.py
    - src/trading_crab_lib/platform/backtest/costs.py
    - src/trading_crab_lib/platform/evaluation/__init__.py
    - tests/unit/test_platform_backtest_costs.py
  modified:
    - config/platform_settings.yaml

key-decisions:
  - "crisis_windows default list hard-bounded to 4 in-sample crises (1973-74, 1980-82, 2000-02, 2008-09), every end date <= 2020-12-31; no 2020/2022 window present"
  - "compute_turnover uses index-union reindex(fill_value=0.0) alignment, not positional diffing, so cold starts (empty prev) and asset-set changes (new/dropped assets) are handled correctly"
  - "turnover convention is target-vs-target (not drift-adjusted) — documented in costs.py docstring as Pitfall 3's defensible simplification"

patterns-established:
  - "Pattern: new optional platform config sections always paired with a load-time acceptance-criteria assertion in the plan, never touching _REQUIRED_PLATFORM_SECTIONS"

requirements-completed: [EVAL-01]

coverage:
  - id: D1
    description: "backtest: config section added to platform_settings.yaml, loads via load_platform_config() without touching _REQUIRED_PLATFORM_SECTIONS"
    requirement: "EVAL-01"
    verification:
      - kind: other
        ref: "python -c \"from trading_crab_lib.platform.config import load_platform_config; c=load_platform_config(); assert c['backtest']['cost_bps']==10 ...\" (plan acceptance_criteria, task 1)"
        status: pass
    human_judgment: false
  - id: D2
    description: "backtest/ and evaluation/ package scaffolds created and import cleanly"
    requirement: "EVAL-01"
    verification:
      - kind: other
        ref: "importlib.import_module('trading_crab_lib.platform.backtest'); importlib.import_module('trading_crab_lib.platform.evaluation')"
        status: pass
    human_judgment: false
  - id: D3
    description: "compute_turnover + apply_transaction_cost implement the D-03 cost = turnover x bps identity, TDD RED->GREEN"
    requirement: "EVAL-01"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_backtest_costs.py#TestTurnover, TestCostIdentity"
        status: pass
    human_judgment: false

duration: 3min
completed: 2026-07-24
status: complete
---

# Phase 5 Plan 1: Backtest Config + Cost/Turnover Identity Summary

**backtest: config section (cost_bps, crisis_windows hard-bounded to <=2020-12-31, ablation skip flag) plus a TDD-proven turnover/transaction-cost module (compute_turnover + apply_transaction_cost) that the Plan 02 walk-forward driver and Plan 05 baseline gauntlet both import.**

## Performance

- **Duration:** 3 min
- **Started:** 2026-07-24T21:36:56Z
- **Completed:** 2026-07-24T21:39:20Z
- **Tasks:** 2 completed
- **Files modified:** 5 (1 modified, 4 created)

## Accomplishments
- Added a `backtest:` section to `config/platform_settings.yaml` with `cost_bps`, `min_train_months`, `refit_frequency_months`, `sixty_forty_rebalance`, `apply_cost_to_baselines`, `skip_l1l2_for_ablation`, and a hard-bounded `crisis_windows` list (all end dates <= 2020-12-31) — read defensively via `cfg.get("backtest", {})`, `_REQUIRED_PLATFORM_SECTIONS` in `platform/config.py` untouched.
- Scaffolded the two new platform sub-packages (`platform/backtest/`, `platform/evaluation/`) with one-line docstrings matching the `platform/honesty/__init__.py` style.
- Implemented `compute_turnover` (index-union absolute-diff sum, cold-start-safe) and `apply_transaction_cost` (`gross - turnover * cost_bps / 1e4`) in `platform/backtest/costs.py`, test-first (RED commit then GREEN commit), including a synthetic `__main__` self-check.
- Proved the honesty invariant named in 05-CONTEXT: for the same weight sequence, `frictionless_net - costed_net == turnover * cost_bps / 1e4` to float precision.

## Task Commits

Each task was committed atomically:

1. **Task 1: Add the backtest: config section + create both package scaffolds** - `a91bd35` (feat)
2. **Task 2: Implement costs.py (turnover + transaction-cost identity) test-first** - `7137229` (test, RED) then `e012ac3` (feat, GREEN)

_Note: Task 2 is a TDD task — RED (`7137229`) then GREEN (`e012ac3`); no REFACTOR commit was needed, the implementation was already clean on first pass._

## Files Created/Modified
- `config/platform_settings.yaml` - gains the `backtest:` section (config knobs + crisis windows)
- `src/trading_crab_lib/platform/backtest/__init__.py` - new package, one-line docstring
- `src/trading_crab_lib/platform/backtest/costs.py` - `compute_turnover`, `apply_transaction_cost`, `__main__` self-check
- `src/trading_crab_lib/platform/evaluation/__init__.py` - new package, one-line docstring
- `tests/unit/test_platform_backtest_costs.py` - `TestTurnover` (5 tests), `TestCostIdentity` (3 tests)

## Decisions Made
- `crisis_windows` default list: 1973-01..1974-12, 1980-01..1982-12, 2000-01..2002-12, 2008-01..2009-12 — chosen from the 4 candidates named in 05-CONTEXT.md, all hard-bounded to end on or before `DEFAULT_HOLDOUT_CUTOFF` (2020-12-31, `honesty/holdout.py`). No 2020/2022 window is present.
- `compute_turnover` aligns via `Index.union()` + `reindex(fill_value=0.0)` rather than positional diffing, so it correctly handles a cold-start empty `prev_weights`, and assets present in only one of the two Series — matching the plan's `must_haves.truths` spec exactly.
- Turnover convention documented as target-vs-target (not drift-adjusted) directly in the `costs.py` module docstring — a defensible, explicitly-flagged simplification (Pitfall 3), not an oversight.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- `backtest/costs.py`'s `compute_turnover` / `apply_transaction_cost` are ready for Plan 02 (walk-forward driver) and Plan 05 (baseline gauntlet) to import unchanged.
- The `backtest:` config section (`min_train_months`, `refit_frequency_months`, `crisis_windows`, `skip_l1l2_for_ablation`) is available for the driver plan to read via `cfg.get("backtest", {})`.
- No blockers. `platform/backtest/` and `platform/evaluation/` package namespaces are now owned by this plan, so later same-wave plans in Phase 5 will not collide on the `__init__.py` files.

---
*Phase: 05-honest-backtest-evaluation*
*Completed: 2026-07-24*

## Self-Check: PASSED

All 5 created/modified files confirmed present on disk; all 4 commit hashes
(a91bd35, 7137229, e012ac3, 492606a) confirmed present in `git log --oneline --all`.
