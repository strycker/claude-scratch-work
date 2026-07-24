---
phase: 05-honest-backtest-evaluation
plan: 02
subsystem: backtest
tags: [platform, walk-forward, honesty-framework, tdd, cross-ai-review-fixes]

# Dependency graph
requires:
  - phase: 05-honest-backtest-evaluation
    plan: 01
    provides: "backtest/costs.py compute_turnover + apply_transaction_cost; backtest: config section (min_train_months, cost_bps, skip_l1l2_for_ablation)"
  - phase: 02-honesty-infrastructure
    provides: "honesty/walkforward.py expanding_steps spine; honesty/holdout.py split_by_holdout_boundary/DEFAULT_HOLDOUT_CUTOFF; honesty/registry.py append_trial"
  - phase: 03-regime-labeling-prediction (via 04-asset-prediction-allocation wave)
    provides: "labeling/jump_model.py fit_jump_model+canonicalize_states+standardize_features; prediction/nowcaster.py build_nowcaster_training_set+fit_nowcaster"
  - phase: 04-asset-prediction-allocation
    provides: "assets/returns.py returns_by_regime_stats; allocation/tilt.py vol_targeted_tilt; allocation/hysteresis.py update_active_regime"
provides:
  - "src/trading_crab_lib/platform/backtest/driver.py: run_backtest(monthly_features, asset_returns, cfg, *, min_train=None, cash_returns=None, use_regime_tilt=True, registry_path=None) -> (equity_curve, per_step_metrics)"
  - "private helpers _refit_l1, _refit_l2, _realized_return (module-level, monkeypatch-friendly for downstream test suites)"
  - "per_step_metrics contract (dates/proba/classes, no loop-sourced y_true) consumed by Plan 04 (EVAL-04 model-metrics artifacts)"
affects: [05-03-evaluation-kpis, 05-04-model-metrics-artifacts, 05-05-baselines, 05-06-report]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "walk-forward per-step body refits L1+L2 fresh on train_index-only data every step (never the Phase 3/4 one-shot nowcaster checkpoint) — expanding_steps is the only reused component from honesty/walkforward.py, the loop body is new code (A1)"
    - "ablation (use_regime_tilt=False) feeds a degenerate constant single-state label/probability pair through the SAME vol_targeted_tilt/regime_tilt_weights code path — never a hand-rolled parallel implementation"
    - "cash residual (tilt['cash']) earns the supplied cash_returns series for the test month via a keyword-only cash_return arg on _realized_return, defaulting to 0.0 only when no series is supplied (review F4)"
    - "per-step L2 refit wrapped in try/except ValueError, degrading to hold-previous-weights + WARNING rather than crashing (RESEARCH.md Pitfall 2); degraded steps are excluded from per_step_metrics"
    - "holdout boundary applied via split_by_holdout_boundary to BOTH monthly_features and asset_returns before constructing expanding_steps — never relies on config end_date"

key-files:
  created:
    - src/trading_crab_lib/platform/backtest/driver.py
    - tests/unit/test_platform_backtest_driver.py
  modified: []

key-decisions:
  - "Task boundary matched the plan literally: Task 2 committed the core loop WITHOUT the holdout split or L2 resilience (TestHoldoutBoundary intentionally still RED after Task 2's commit); Task 3 added split_by_holdout_boundary + try/except degrade-and-continue, turning the last test GREEN — this makes the incremental TDD narrative visible in git history rather than landing everything in one commit."
  - "All 6 test classes monkeypatch the module-level _refit_l1/_refit_l2 (and, for the cash-residual test, vol_targeted_tilt) private functions rather than exercising the real jump-model/nowcaster fit on tiny synthetic data. This keeps the suite fast and deterministic and isolates each invariant (holdout boundary, compounding, registry logging, train-window-only refit, cash accrual, ablation-skip) from real-fit quality/degeneracy, which is exactly Pitfall 2's territory and is separately proven end-to-end via the module's __main__ self-check (degrades gracefully on tiny synthetic data with a real fit)."
  - "_L2_DEGRADE_EXCEPTIONS is scoped to ValueError only (not a broad except) — matches what CalibratedClassifierCV/LogisticRegression actually raise for a degenerate single-class fold, and keeps the broad-except-discouraged project convention (CLAUDE.md Error Handling: no bare except, broad catching only for network ingestion code)."
  - "Fixed the __main__ self-check's synthetic cfg (added a taxonomy section) after discovering during Task 3 verification that lean_feature_set(cfg) silently returns an empty set without one, degrading every single step. Rule 1 auto-fix — the self-check is documentation/verification code, not a behavior change to run_backtest itself."

patterns-established:
  - "Pattern: private per-step helper functions (_refit_l1, _refit_l2, _realized_return, and any imported allocation/tilt function used inside a walk-forward loop) are module-level names specifically so downstream test suites can monkeypatch them for fast, deterministic, invariant-isolated testing of orchestration logic without re-fitting real ML models on synthetic data every test."

requirements-completed: [EVAL-01]

coverage:
  - id: D1
    description: "run_backtest steps expanding_steps(dev_index, min_train) across the holdout-bounded index and returns (equity_curve, per_step_metrics) with dates/proba/classes"
    requirement: "EVAL-01"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_backtest_driver.py::TestHoldoutBoundary, TestEquityCurveCompounding"
        status: pass
    human_judgment: false
  - id: D2
    description: "L1 and L2 refit fresh on train_index-only data every step; no reuse of the Phase 3/4 one-shot nowcaster checkpoint"
    requirement: "EVAL-01"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_backtest_driver.py::TestRefitFromTrainWindowOnly"
        status: pass
      - kind: other
        ref: "grep -nE 'get_holdout_checkpoint_manager\\(|\\.load_model\\(' src/trading_crab_lib/platform/backtest/driver.py returns nothing"
        status: pass
    human_judgment: false
  - id: D3
    description: "strategy cash residual earns the cash_returns series' return for the test month, not a hard 0% (review F4)"
    requirement: "EVAL-01"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_backtest_driver.py::TestCashResidualEarnsCashReturn"
        status: pass
    human_judgment: false
  - id: D4
    description: "the no-regime ablation (use_regime_tilt=False) skips the expensive L1/L2 refits when skip_l1l2_for_ablation is set, and produces an identical equity curve with or without the skip (review F5)"
    requirement: "EVAL-01"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_backtest_driver.py::TestAblationSkipInvariant"
        status: pass
    human_judgment: false
  - id: D5
    description: "run_backtest calls registry.append_trial exactly once per full run"
    requirement: "EVAL-01"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_backtest_driver.py::TestRegistryLogging"
        status: pass
    human_judgment: false
  - id: D6
    description: "an L2 refit failure in an early small-sample step is caught, logged at WARNING, and degrades to holding previous weights rather than crashing the run"
    requirement: "EVAL-01"
    verification:
      - kind: other
        ref: "python -m trading_crab_lib.platform.backtest.driver self-check on 48-month synthetic data: several steps log 'L2 refit degraded ... holding previous weights' and the run completes to a full equity curve without raising"
        status: pass
    human_judgment: false

duration: 12min
completed: 2026-07-24
status: complete
---

# Phase 5 Plan 2: Walk-Forward Backtest Driver Summary

**`run_backtest()` — the honest end-to-end EVAL-01 walk-forward driver: expanding-window refit of L1 (jump-model) + L2 (nowcaster) on train-only data every step, L3/L4 vol-targeted regime tilt with hysteresis, cost-adjusted compounding, and all three cross-AI review fixes (F2 y_true off-by-one, F4 cash-residual convention, F5 ablation-skip) proven by dedicated tests.**

## Performance

- **Duration:** 12 min
- **Started:** 2026-07-24T21:41:00Z
- **Completed:** 2026-07-24T21:53:00Z
- **Tasks:** 3 completed
- **Files modified:** 2 (both created)

## Accomplishments

- Implemented `src/trading_crab_lib/platform/backtest/driver.py`: `run_backtest(monthly_features, asset_returns, cfg, *, min_train=None, cash_returns=None, use_regime_tilt=True, registry_path=None)` plus private helpers `_refit_l1`, `_refit_l2`, `_realized_return`.
- Applied `split_by_holdout_boundary` to both `monthly_features` and `asset_returns` BEFORE constructing `expanding_steps` — the visited train/test index never exceeds `DEFAULT_HOLDOUT_CUTOFF` (2020-12-31) even when the input frames physically extend into 2021+; no call to the holdout-namespace checkpoint manager getter anywhere in the module (verified by grep gate).
- L1 (`fit_jump_model` + `canonicalize_states`) and L2 (`build_nowcaster_training_set` + `fit_nowcaster`) refit fresh on `train_index`-only data at every step; no code path loads the Phase 3/4 one-shot nowcaster checkpoint (`.load_model(` also grep-gated absent).
- `use_regime_tilt=False` (the design §8.7 no-regime ablation) feeds a degenerate constant single-state label/probability pair through the SAME `vol_targeted_tilt`/`regime_tilt_weights` code path (never a hand-rolled parallel implementation); when additionally `cfg["backtest"].get("skip_l1l2_for_ablation", True)`, the L1/L2 refits are skipped entirely for that step since their output is discarded anyway (review F5) — proven byte-identical (`pd.testing.assert_frame_equal`) between skip=True and skip=False.
- The strategy's cash residual (`tilt["cash"]`) earns the supplied `cash_returns` series' return for the test month via `_realized_return`'s keyword-only `cash_return` arg — not a hard 0% (review F4) — proven by a symmetric-delta test comparing a run with `cash_returns` supplied against `cash_returns=None`.
- A per-step L2 refit failure (RESEARCH.md Pitfall 2 — an early small post-embargo window starving a `CalibratedClassifierCV` K-fold) is caught (`ValueError` only, no broad except), logged at WARNING naming the step date, and degrades that step to holding the previous weights/active-regime/cash rather than crashing the run; degraded steps are excluded from `per_step_metrics`.
- `per_step_metrics` (`dates`/`proba`/`classes`) never sources `y_true` from the in-progress loop (review F2) — the report layer (Plan 06) joins ground truth retroactively against the full-sample smoothed reference labeling by decision date.
- Exactly one `registry.append_trial` call per full run, logging `{"phase":"05-backtest","use_regime_tilt","min_train","cost_bps"}`, the full feature-column list, and `{"n_steps","terminal_log_wealth"}`.

## Task Commits

Each task was committed atomically:

1. **Task 1: Write the driver test file (RED)** — `e26a5ca` (test) — six test classes (`TestHoldoutBoundary`, `TestEquityCurveCompounding`, `TestRegistryLogging`, `TestRefitFromTrainWindowOnly`, `TestCashResidualEarnsCashReturn`, `TestAblationSkipInvariant`) fail at collection because `trading_crab_lib.platform.backtest.driver` did not yet exist.
2. **Task 2: Implement run_backtest core loop** — `6569adf` (feat) — refit/size/compound/log-one-trial loop over the full (not-yet-holdout-bounded) index; the 5 tests named in this task's acceptance criteria pass; `TestHoldoutBoundary` intentionally remains RED (holdout boundary not yet applied).
3. **Task 3: Holdout-bound the index, add early-step resilience** — `2ed6d69` (feat) — added `split_by_holdout_boundary` before `expanding_steps` and try/except degrade-and-continue around the L2 refit; all 6 tests pass; grep gate confirms no `get_holdout_checkpoint_manager(`/`.load_model(` calls.

## Files Created/Modified

- `src/trading_crab_lib/platform/backtest/driver.py` — new module: `run_backtest` + `_refit_l1`/`_refit_l2`/`_realized_return` + `__main__` self-check.
- `tests/unit/test_platform_backtest_driver.py` — new test file: 6 test classes, all monkeypatch-based for speed/determinism.

## Decisions Made

- **Task boundary matched the plan literally.** Rather than writing the finished driver in one shot, Task 2's commit deliberately omits the holdout split and L2-failure resilience (both land in Task 3), so `TestHoldoutBoundary` is provably RED after Task 2 and GREEN after Task 3 — the incremental TDD narrative is visible in git history, not collapsed into one commit.
- **Monkeypatch-first test design.** All 6 test classes monkeypatch the module-level `_refit_l1`/`_refit_l2` (and, for the cash-residual test, `vol_targeted_tilt`) private functions rather than exercising the real jump-model/nowcaster fit on tiny synthetic data. This keeps the six orchestration invariants (holdout boundary, compounding, registry logging, train-window-only refit, cash accrual, ablation-skip) fast, deterministic, and isolated from real-fit degeneracy — which is exactly RESEARCH.md Pitfall 2's territory. The real end-to-end fit path (jump model + nowcaster, including graceful degradation) is separately exercised by the module's `__main__` self-check on 48 months of synthetic data.
- **`_L2_DEGRADE_EXCEPTIONS` scoped to `ValueError` only**, matching what `CalibratedClassifierCV`/`LogisticRegression` actually raise for a degenerate single-class fold, and keeping with the project's no-broad-except convention (root `CLAUDE.md` Error Handling).
- **`__main__` self-check cfg fix (Rule 1 auto-fix).** Discovered during Task 3 verification that the self-check's synthetic cfg lacked a `taxonomy` section, so `lean_feature_set(cfg)` silently returned an empty set and every step degraded. Added a `taxonomy: {fast, slow, agency}` block matching the self-check's lean columns so the self-check now genuinely exercises the real L1/L2 fit path (and correctly shows the graceful-degradation behavior on the genuinely early/small-sample steps, not on every step).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] `__main__` self-check cfg missing `taxonomy` section**
- **Found during:** Task 3 verification (running the self-check directly showed every step degrading with a cryptic numpy error)
- **Issue:** The synthetic `_cfg` dict in `driver.py`'s `__main__` block had no `taxonomy` section, so `lean_feature_set(cfg)` returned an empty set, `_refit_l1` selected zero columns, and `standardize_features` raised on an empty-column DataFrame.
- **Fix:** Added a `taxonomy: {"fast": ..., "slow": ..., "agency": []}` block covering the same 13 lean columns used elsewhere in the self-check.
- **Files modified:** `src/trading_crab_lib/platform/backtest/driver.py`
- **Commit:** `2ed6d69`

No other deviations — the plan's `<action>` blocks were followed as written, including the exact `must_haves.truths` wording for F2/F4/F5.

## Issues Encountered

- Running the `__main__` self-check directly (outside pytest) wrote a real `registry/trials.jsonl` file to the repo root via `registry.append_trial`'s default path. This was test-run pollution from manual verification, not committed data — deleted (`rm -rf registry/`) before staging Task 3's commit. No registry directory was ever staged or committed.
- Initial phrasing of the module docstring/comments literally contained the substring `get_holdout_checkpoint_manager(` (with trailing paren, inside prose describing the absence of such a call), which the Task 3 grep gate (`grep -nE "get_holdout_checkpoint_manager\(|\.load_model\("`) correctly flagged as a false-positive match on its own documentation. Reworded to "the holdout-namespace checkpoint manager getter" (no literal parenthesized call-form substring) in three places; grep gate then returns nothing as required.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- `run_backtest`'s `(equity_curve, per_step_metrics)` return contract is ready for Plan 03 (evaluation KPIs — terminal log wealth, max drawdown, CVaR, crisis capture ratios) and Plan 04 (model-metrics artifacts: `per_step_metrics["dates"]`/`["proba"]`/`["classes"]` feed multiclass Brier/calibration/confusion computations once the report layer joins `y_true` by date).
- Plan 05 (baseline gauntlet: SPY buy-and-hold, 60/40, Faber SMA) can reuse `backtest/costs.py` (already available from Plan 01) with the same `apply_cost_to_baselines` convention; `run_backtest(use_regime_tilt=False)` is the no-regime ablation baseline required by design §8.7 / EVAL-02.
- No blockers. `driver.py`'s private helpers (`_refit_l1`, `_refit_l2`, `_realized_return`, and the imported `vol_targeted_tilt` name) are intentionally monkeypatch-friendly module-level names for any downstream test suite that needs to isolate orchestration logic from real-fit cost.

---
*Phase: 05-honest-backtest-evaluation*
*Completed: 2026-07-24*

## Self-Check: PASSED

All 3 created/modified files confirmed present on disk; all 3 commit hashes
(e26a5ca, 6569adf, 2ed6d69) confirmed present in `git log --oneline --all`.
