---
phase: 05-honest-backtest-evaluation
plan: 03
subsystem: backtest
tags: [platform, evaluation, honesty-framework, tdd, cross-ai-review-fixes, kpis, sojourn-lag]

# Dependency graph
requires:
  - phase: 05-honest-backtest-evaluation
    plan: 01
    provides: "backtest/costs.py compute_turnover + apply_transaction_cost; backtest: config section incl. crisis_windows"
  - phase: 05-honest-backtest-evaluation
    plan: 02
    provides: "backtest/driver.py run_backtest() -> (equity_curve, per_step_metrics); per_step_metrics contract (dates/proba/classes, no loop-sourced y_true)"
  - phase: 02-honesty-infrastructure
    provides: "honesty/gap_lag.py compute_detection_lag + sojourn_lag_ratio; honesty/holdout.py DEFAULT_HOLDOUT_CUTOFF"
  - phase: 03-regime-labeling-prediction
    provides: "labeling/diagnostics.py occupancy_and_sojourns"
provides:
  - "src/trading_crab_lib/platform/evaluation/kpis.py: terminal_log_wealth, max_drawdown_and_duration, cvar, crisis_capture_ratio"
  - "src/trading_crab_lib/platform/evaluation/sojourn_lag.py: build_filtered_probs_matrix, compute_sojourn_lag_headline (EVAL-03 headline, review F1 fixed)"
affects: [05-04-model-metrics-artifacts, 05-06-report]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "sojourn_lag.py reimplements ZERO underlying math — it imports and orchestrates occupancy_and_sojourns + compute_detection_lag + sojourn_lag_ratio unchanged (grep-gated: no local redefinition of any of the three)"
    - "review F1: ex-post transitions are grouped by their OWN target state and checked against ONLY that state's own filtered-probs column — never a class-agnostic max-across-classes series, which would systematically understate detection lag"
    - "build_filtered_probs_matrix performs union-of-classes reconciliation: any state not observed in a given walk-forward step is padded with 0.0, never dropped or NaN (shared helper for F1's per-target-state conversion and F3's model-metrics artifacts, review F1/F3)"
    - "crisis_capture_ratio hard-rejects (ValueError) any crisis window whose end date exceeds DEFAULT_HOLDOUT_CUTOFF (2020-12-31) before computing anything — Pitfall 4 guard, proven at the boundary (2020-12-31 accepted, 2021+ rejected)"

key-files:
  created:
    - src/trading_crab_lib/platform/evaluation/kpis.py
    - src/trading_crab_lib/platform/evaluation/sojourn_lag.py
    - tests/unit/test_platform_evaluation_kpis.py
    - tests/unit/test_platform_evaluation_sojourn_lag.py
  modified: []

key-decisions:
  - "max_drawdown_and_duration's duration_months is defined literally per the plan's <action> text: the longest run of CONSECUTIVE periods strictly below the prior running peak (drawdown < 0), not a peak-to-trough-only measure — for a series with exactly one drawdown episode the two coincide, but a never-recovered drawdown's duration correctly extends to the end of the series."
  - "cvar's floor(alpha*n) is clamped to a minimum of 1 tail observation — an unclamped floor would silently produce an empty-mean NaN for any sample small enough that alpha*n < 1, which is not documented behavior anywhere in the plan and would be a confusing failure mode for a KPI function."
  - "compute_sojourn_lag_headline signature takes full_sample_states as a pd.Series with a DatetimeIndex (not a bare array) specifically so filtered_probs_matrix's own-target-state column can be reindexed onto it by date — this is what makes 'pre-decision months carry NaN and never cross' a real property of the alignment rather than an assumption."
  - "A transition into a target state with NO observed column anywhere in filtered_probs_matrix is treated as fully unresolved (NaN) for every one of its transitions, consistent with compute_detection_lag's own documented 'unresolved' convention — not an error, since a state genuinely never observed during the entire walk-forward run is a real (if degenerate) scenario."

patterns-established:
  - "Pattern: the `compute_* / report_* (where applicable) / __main__ self-check` module shape from honesty/gap_lag.py and labeling/diagnostics.py is followed exactly for evaluation/kpis.py and evaluation/sojourn_lag.py — no new module shape introduced."

requirements-completed: [EVAL-01, EVAL-03]

coverage:
  - id: D1
    description: "terminal_log_wealth(returns) == np.log1p(returns).sum() for a known monthly return sequence, including the all-zero edge case"
    requirement: "EVAL-04"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_evaluation_kpis.py::TestTerminalLogWealth"
        status: pass
    human_judgment: false
  - id: D2
    description: "max_drawdown_and_duration returns the correct peak-to-trough magnitude and the longest-underwater-run duration for hand-built curves, including a never-recovered and a flat (zero-drawdown) case"
    requirement: "EVAL-04"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_evaluation_kpis.py::TestMaxDrawdown"
        status: pass
    human_judgment: false
  - id: D3
    description: "cvar(returns, alpha) equals the mean of the worst floor(alpha*n) returns for known sequences at both the default and a custom alpha"
    requirement: "EVAL-04"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_evaluation_kpis.py::TestCVaR"
        status: pass
    human_judgment: false
  - id: D4
    description: "crisis_capture_ratio computes strategy-cum/benchmark-cum per configured window, accepts a window ending exactly at the holdout cutoff, and raises ValueError for any window ending after it"
    requirement: "EVAL-04"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_evaluation_kpis.py::TestCrisisCaptureBounded"
        status: pass
    human_judgment: false
  - id: D5
    description: "compute_sojourn_lag_headline wires occupancy_and_sojourns + compute_detection_lag + sojourn_lag_ratio into the EVAL-03 headline ratio, and treats full_sample_states/filtered_probs_matrix as genuinely distinct inputs (a transition before the filtered-probs date range resolves NaN rather than crashing or misaligning)"
    requirement: "EVAL-03"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_evaluation_sojourn_lag.py::TestSojournLagHeadline, TestDistinctSmoothedFiltered"
        status: pass
    human_judgment: false
  - id: D6
    description: "compute_sojourn_lag_headline pools PER-TARGET-STATE lags (each transition checked against P(its own target state)) and the resulting median_lag is strictly greater than a class-agnostic max-across-classes construction would produce, on a known K=5 synthetic construction (review F1)"
    requirement: "EVAL-03"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_evaluation_sojourn_lag.py::TestPerTransitionTargetStateProbs"
        status: pass
    human_judgment: false
  - id: D7
    description: "build_filtered_probs_matrix stacks per-step (proba, classes) rows into a (n_steps, K) frame indexed by decision dates, padding any state not observed in a given step with 0.0"
    requirement: "EVAL-03"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_evaluation_sojourn_lag.py::TestBuildFilteredProbsMatrix"
        status: pass
    human_judgment: false
  - id: D8
    description: "none of the three underlying gap_lag/diagnostics functions is redefined locally in sojourn_lag.py (orchestration-only, T-05-07)"
    requirement: "EVAL-03"
    verification:
      - kind: other
        ref: "grep -nE 'def (compute_detection_lag|sojourn_lag_ratio|occupancy_and_sojourns)' src/trading_crab_lib/platform/evaluation/sojourn_lag.py returns nothing"
        status: pass
    human_judgment: false

duration: 18min
completed: 2026-07-24
status: complete
---

# Phase 5 Plan 3: Evaluation KPIs + Sojourn/Lag Headline Summary

**Strategy KPIs (`kpis.py`: terminal log wealth, max drawdown+duration, CVaR(5%), holdout-bounded crisis capture ratios) and the EVAL-03 headline (`sojourn_lag.py`: median sojourn / detection lag), with the cross-AI review F1 fix — the headline now checks each ex-post transition against its OWN target-state probability column, never a class-agnostic max-across-classes series.**

## Performance

- **Duration:** 18 min
- **Started:** 2026-07-24T22:00:00Z
- **Completed:** 2026-07-24T22:18:00Z
- **Tasks:** 2 completed
- **Files modified:** 4 (all created)

## Accomplishments

- Implemented `src/trading_crab_lib/platform/evaluation/kpis.py`: `terminal_log_wealth`, `max_drawdown_and_duration`, `cvar`, `crisis_capture_ratio` — ~30 lines of auditable vectorized pandas per function, no empyrical/quantstats dependency, matching the `assets/returns.py` drawdown one-liner convention.
- `crisis_capture_ratio` hard-rejects any crisis window whose `end` date exceeds `DEFAULT_HOLDOUT_CUTOFF` (2020-12-31) with a `ValueError` before computing anything — proven at the exact boundary (2020-12-31 accepted, any 2021+ end date rejected), closing Pitfall 4.
- Implemented `src/trading_crab_lib/platform/evaluation/sojourn_lag.py`: `build_filtered_probs_matrix` (new multiclass→per-state matrix helper) and `compute_sojourn_lag_headline` (thin orchestration, zero new math — grep-gated to prove none of the three underlying `gap_lag`/`diagnostics` functions is redefined locally).
- **Review F1 fix implemented exactly as specified:** `compute_sojourn_lag_headline` derives ex-post transitions from the smoothed states' own change points, GROUPS them by their target state, and for each target state reindexes ONLY that state's own `filtered_probs_matrix` column onto the smoothed states' positional index before calling `compute_detection_lag` — never a class-agnostic row-max. `TestPerTransitionTargetStateProbs` proves the per-target-state median (2.0) is strictly greater than the class-agnostic max-across-classes construction (0.5) on a known K=5 synthetic matrix with deliberate spurious spikes in unrelated classes.
- `build_filtered_probs_matrix` performs the union-of-classes reconciliation: unions every state observed across all walk-forward steps, then pads any state absent from a given step's `classes` list with `0.0` (never dropped or left `NaN`) — the shared helper both F1 (this plan) and F3 (Plan 04's model-metrics artifacts) rely on.
- `TestDistinctSmoothedFiltered` proves `full_sample_states` and `filtered_probs_matrix` are genuinely separate inputs (Pitfall 1): a transition occurring before the filtered-probs matrix's date range even starts (simulating the walk-forward warmup period) resolves to `NaN` (unresolved, excluded from the median) rather than crashing or silently misaligning.

## Task Commits

Each task was committed atomically (TDD RED→GREEN):

1. **Task 1: Write kpis.py test file (RED)** — `fb7e485` (test) — 11 tests across `TestTerminalLogWealth`, `TestMaxDrawdown`, `TestCVaR`, `TestCrisisCaptureBounded`; fail at collection because `kpis.py` did not yet exist.
2. **Task 1: Implement kpis.py (GREEN)** — `1d7200b` (feat) — all 11 tests pass; crisis-window holdout-bound guard proven at the exact 2020-12-31 boundary.
3. **Task 2: Write sojourn_lag.py test file (RED)** — `42c8916` (test) — 5 tests including `TestPerTransitionTargetStateProbs` (review F1); fail at collection because `sojourn_lag.py` did not yet exist.
4. **Task 2: Implement sojourn_lag.py (GREEN)** — `23d3e42` (feat) — all 5 tests pass; grep gate confirms none of `compute_detection_lag`/`sojourn_lag_ratio`/`occupancy_and_sojourns` is redefined locally.

## Files Created/Modified

- `src/trading_crab_lib/platform/evaluation/kpis.py` — `terminal_log_wealth`, `max_drawdown_and_duration`, `cvar`, `crisis_capture_ratio` + `__main__` self-check.
- `src/trading_crab_lib/platform/evaluation/sojourn_lag.py` — `build_filtered_probs_matrix`, `compute_sojourn_lag_headline` + `__main__` self-check.
- `tests/unit/test_platform_evaluation_kpis.py` — 11 tests, 4 classes.
- `tests/unit/test_platform_evaluation_sojourn_lag.py` — 5 tests, 4 classes (incl. `TestPerTransitionTargetStateProbs`, `TestBuildFilteredProbsMatrix`).

## Decisions Made

- **`duration_months` is the longest underwater run, not strictly peak-to-trough-to-recovery.** Followed the plan's `<action>` text literally ("duration = longest run below the prior peak"): implemented as the longest consecutive run of periods with `drawdown < 0`. For a series with exactly one drawdown episode (the test's primary case) this coincides with peak-to-trough-to-recovery; a never-recovered drawdown correctly reports duration extending to the end of the series rather than an undefined/infinite recovery time.
- **`cvar`'s tail-count floor is clamped to at least 1.** `floor(alpha * n)` can be 0 for small samples; clamping to `max(1, floor(alpha*n))` avoids an empty-mean `NaN` that neither the plan's `must_haves.truths` nor any test explicitly required guarding against, but which would otherwise be a silent-failure trap for any downstream caller with a short return history (documented in the docstring per the plan's explicit "document each convention" instruction).
- **`compute_sojourn_lag_headline` requires `full_sample_states` to carry a `DatetimeIndex`** (a `pd.Series`, not a bare array) specifically so `filtered_probs_matrix`'s own-target-state column can be `reindex`ed onto it by date. This is the mechanism that makes "pre-decision months carry NaN and never cross" a structural property of the date-alignment rather than an assumption the caller must maintain manually.
- **A target state with no observed column anywhere in `filtered_probs_matrix`** (e.g., a state the walk-forward nowcaster genuinely never assigned nonzero probability to across the entire run) has every one of its transitions treated as unresolved (`NaN`), consistent with `compute_detection_lag`'s own documented convention for a threshold that's never crossed — not an error, since this is a real (if degenerate) scenario a production run could hit.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] `__main__` self-check in sojourn_lag.py raised ValueError on first attempt**
- **Found during:** Task 2 verification (running `python -m trading_crab_lib.platform.evaluation.sojourn_lag` directly after the test suite passed)
- **Issue:** The initial self-check built `filtered_probs_matrix` from `rng.dirichlet(np.ones(3))` random per-step probabilities. For that particular RNG draw, two of the three ex-post transitions resolved to `NaN` (their own target-state column never crossed the 0.70 threshold) and the third resolved to exactly `0.0`, so `median_lag == 0.0` — and `sojourn_lag_ratio` correctly raises `ValueError` for a non-positive lag (by design, per `gap_lag.py`'s own contract). This was a self-check construction flaw, not a bug in `compute_sojourn_lag_headline` itself (the test suite, which uses deterministic hand-built matrices, all passed).
- **Fix:** Replaced the random Dirichlet self-check data with a deterministic construction where each target state's own probability column ramps from 0.2 to 0.8 exactly 2 periods after its ex-post transition — guaranteeing every transition resolves with a known, positive lag (2 months), producing a meaningful non-degenerate self-check output (`median_sojourn=5.0, median_lag=2.0, ratio=2.5`).
- **Files modified:** `src/trading_crab_lib/platform/evaluation/sojourn_lag.py` (only the `__main__` block; no production code changed)
- **Commit:** `23d3e42` (part of Task 2's GREEN commit)

---

**Total deviations:** 1 auto-fixed (Rule 1, self-check documentation code only)
**Impact on plan:** No production code was affected — the fix was isolated to the module's own runnable documentation/self-check block. All test-file assertions (RED-phase, unchanged) still pass identically after the fix.

## Issues Encountered

- `ruff --fix` reordered the import block in `tests/unit/test_platform_evaluation_sojourn_lag.py` (isort: `evaluation.sojourn_lag` import moved after `honesty.gap_lag` alphabetically). Applied before committing the GREEN task; no test assertions changed.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- `kpis.py`'s four functions are ready for Plan 06 (report layer) to call directly against the walk-forward `equity_curve` (`backtest/driver.py`'s Plan 02 output) and the baseline gauntlet's return series (Plan 05).
- `sojourn_lag.py`'s `build_filtered_probs_matrix` is the shared helper Plan 04 (EVAL-04 model-metrics artifacts, review F3) will also consume to build its multiclass Brier/calibration/confusion inputs from the same `per_step_metrics` contract — no duplicate union-of-classes logic needed.
- `compute_sojourn_lag_headline` is ready to be wired against real data once Plan 06 has both a full-sample smoothed L1 fit (`labeling/diagnostics.py::label_regimes` output, already available from Phase 3) and a walk-forward `per_step_metrics` (Plan 02's `run_backtest` output) — the two are structurally guaranteed distinct inputs by the function's own signature, closing Pitfall 1 at the interface level.
- No blockers.

---
*Phase: 05-honest-backtest-evaluation*
*Completed: 2026-07-24*

## Self-Check: PASSED

All 4 created files confirmed present on disk; all 4 commit hashes
(fb7e485, 1d7200b, 42c8916, 23d3e42) confirmed present in `git log --oneline --all`.
