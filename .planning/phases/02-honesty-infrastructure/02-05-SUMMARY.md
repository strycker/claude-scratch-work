---
phase: 02-honesty-infrastructure
plan: 05
subsystem: infra
tags: [honesty-framework, walk-forward, sklearn, trial-registry]

# Dependency graph
requires:
  - phase: 02-honesty-infrastructure
    provides: "02-02 registry.py (append_trial/read_trials), 02-01 honesty package"
provides:
  - "src/trading_crab_lib/platform/honesty/walkforward.py: expanding_steps(), run_walkforward()"
  - "Phase-0 exit proof: trivial model (DummyClassifier prior) runs end-to-end with refit-before-t at every step"
  - "Automatic single-trial registry logging per walk-forward run (D-02, Pitfall 3)"
affects: [phase-3-modeling, phase-5-backtest]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "clone-per-step via sklearn.base.clone (mirrors monitoring/prediction.py clone-per-fold idiom)"
    - "append_trial wired INSIDE the runner, never left to callers (RESEARCH Pitfall 3)"

key-files:
  created:
    - src/trading_crab_lib/platform/honesty/walkforward.py
    - tests/unit/test_platform_walkforward.py
  modified: []

key-decisions:
  - "DummyClassifier(strategy='prior') as the Phase-2 trivial model (ponytail rung 5 — sklearn already implements 'predict the prior'); Phase 3 swaps real models against the same interface"
  - "Hit-rate + n_steps as the aggregate metrics logged per trial — enough for the ledger denominator, real metrics arrive with real models"

patterns-established:
  - "Walk-forward interface frozen: expanding_steps(index, *, min_train, step=1) yields (t, train_index, test_index) with train strictly before t; run_walkforward returns decisions list and logs exactly one trial"

requirements-completed: [HON-03]

coverage:
  - id: T1
    description: "Leakage invariant: train_index[-1] < test_index[0] and test row never in train set, at every step"
    requirement: "HON-03"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_walkforward.py::TestExpandingStepsLeakageInvariant"
        status: pass
    human_judgment: false
  - id: T2
    description: "One decision dict {t, prediction, params} per step; len == len(index) - min_train"
    requirement: "HON-03"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_walkforward.py::TestOneDecisionPerStep"
        status: pass
    human_judgment: false
  - id: T3
    description: "End-to-end on synthetic monthly data, no network, default trivial model"
    requirement: "HON-03"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_walkforward.py::TestEndToEndNoNetwork"
        status: pass
    human_judgment: false
  - id: T4
    description: "Exactly one registry trial per run, logged automatically by the runner; two runs append two rows"
    requirement: "HON-03"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_walkforward.py::TestExactlyOneRegistryTrial, TestRegistryLoggingIsAutomatic, TestMultipleRunsLogMultipleTrials"
        status: pass
    human_judgment: false

## Execution notes

TDD RED commit `ed91597` (failing tests), GREEN commit `a8b135c` (implementation).
6 tests in test_platform_walkforward.py; full honesty suite
`pytest tests/unit/test_platform_*.py` green at 138 passed, 1 skipped.

**Deviation:** the original worktree executor lost its isolated worktree after a
provider quota interruption and its RED commit landed directly on the integration
branch; the orchestrator (Fable, per CLAUDE.md model routing — look-ahead-bias
guards stay on Fable) completed the GREEN implementation in-place on the branch.
Single-plan wave, no parallel-agent contention, so no conflicting merges resulted.
