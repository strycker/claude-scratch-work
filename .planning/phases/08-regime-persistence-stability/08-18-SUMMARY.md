---
phase: 08-regime-persistence-stability
plan: 18
subsystem: scripts/run_joint_lift, platform/backtest (run_backtest, joint_driver)
tags: [gap-closure, CR-05, CR-06, registry-integrity, adr-0004, trial-budget, honesty, tdd, tracer]
status: complete
requires:
  - "08-17 (committed; registry 44, sha256 prefix c957e8fdb360)"
provides:
  - "scripts/run_joint_lift.py: ROWS_PER_DECISION_BEARING_RUN = 2, preflight_trial_budget(), required --declared-ceiling for decision-bearing runs, explicit raises only (no assert), _registry_block() with declared_ceiling / declared_ceiling_source"
  - "run_backtest registry rows: config['use_regime_filter'] = use_regime_filter and use_regime_tilt"
  - "run_joint_backtest registry rows: config['use_regime_filter'] = use_regime_filter and routing == L2_NOWCAST; 'harness_plan': '07-11' replaces 'plan': '07-11'"
  - "tests/unit/test_platform_run_joint_lift_budget.py (10 tests; the real ledger's sha256 is pinned by a module fixture)"
affects:
  - "the next decision-bearing joint-lift run: it must pass --declared-ceiling from its phase CONTEXT's '## Trial budget' and rename the 08-10 tags"
  - "Phase 8.1: the two remaining hard-coded-44 sites of ADR-0004 (test_platform_gate_tiers.py, test_platform_joint_diagnostics_record.py)"
tech-stack:
  added: []
  patterns:
    - "the budget check runs before any work that can append; the ceiling is declared, not hard-coded"
    - "guards raise explicitly, so python -O cannot strip them (AST test plus -O subprocess)"
    - "registry rows record the EFFECTIVE configuration flag, not the requested one"
key-files:
  created:
    - tests/unit/test_platform_run_joint_lift_budget.py
    - .planning/phases/08-regime-persistence-stability/08-18-SUMMARY.md
  modified:
    - scripts/run_joint_lift.py
    - src/trading_crab_lib/platform/backtest/driver.py
    - src/trading_crab_lib/platform/backtest/joint_driver.py
    - tests/unit/test_platform_backtest_driver.py
    - tests/unit/test_platform_backtest_joint_driver.py
    - README.md
    - CLAUDE.md
decisions:
  - "CR-06: run_joint_lift reads total_trial_count() first and refuses a decision-bearing run before load_platform_config/build_inputs/either leg when count_before + 2 > --declared-ceiling. The boundary is inclusive (ADR-0004: phase ceiling = opening + budget). No ceiling on a decision-bearing run is a ValueError, never a default. ADR_0002_CEILING = 44 is removed."
  - "CR-05: run_backtest and run_joint_backtest rows carry the EFFECTIVE use_regime_filter. Prospective only. A row without the key ran unfiltered (verified against both ledgers)."
  - "The joint harness's 'plan': '07-11' becomes 'harness_plan': '07-11' on newly appended rows. Historical rows keep 'plan' (append-only) and are identified by trial_tag. No code or test reads the 'plan' key."
metrics:
  duration: "~45 min"
  completed: 2026-09-29
actuals:
  tokens: 7700
  tasks: 3
  commits: 6
---

# Phase 8 Plan 18: Registry-integrity gap closure (CR-05, CR-06) Summary

`run_joint_lift` now checks a declared ADR-0004 ceiling before it builds inputs or runs a leg. Every guard in the script raises explicitly, so `python -O` cannot strip it. Both backtest drivers' registry rows now record whether the curve was filtered, and the joint row names the plan that built the harness instead of claiming the run's plan. No registry rows were spent.

## Tasks

| # | Task | Commits |
|---|------|---------|
| 1 (tracer) | run_joint_lift checks the budget before any append; explicit raises; `--declared-ceiling` | `7dcb781` (RED test), `70bcd6c` (fix) |
| 2 | Effective `use_regime_filter` in both drivers' `trial_config`; `harness_plan` | `4184fbd` (RED test), `332f9b9` (fix) |
| 3 | Recorded counts 2477 -> 2494 (D-07), full suite, lint | `faccf26` |

Tracer gate (auto mode): after `70bcd6c`, the tracer's verification was re-run end to end. It showed 10/10 budget tests passing and the registry at `44 c957e8fdb360`. Tasks 2 and 3 proceeded from there.

## RED on HEAD (Task 1)

This is the exact CR-06 sequence. The HEAD call shape (no ceiling) was run at fake count 44 with fake legs that append one row each:

```
AssertionError: CR-06 sequence: 1 input build(s) and 2 leg call(s) (fake count now 46) BEFORE
AssertionError: total_trial_count() is 46, above ADR-0002's stated ceiling of 44. Exceeding it
requires an explicit ADR amendment — this is the silent search-creep D-17 exists to make visible.
```

HEAD built the inputs, ran both legs (in the real ledger, 44 -> 46), and only then raised, through an `assert` that `-O` strips.

Other RED lines on HEAD:
- `TypeError: run() got an unexpected keyword argument 'declared_ceiling'`: the refused-before-append, at-the-ceiling and rows-added arms.
- `assert statements (stripped by python -O) at lines [110, 114, 248, 253]`: the AST arm.
- `AttributeError: module 'run_joint_lift' has no attribute 'preflight_trial_budget'`: the -O subprocess arm (non-zero exit, but no RuntimeError).
- `assert not True where True = hasattr(R, 'ADR_0002_CEILING')`: the stale-constant arm.
- The 2 observational/dry-run arms (l2, and l1 with `--dry-run`, both without a ceiling) passed on HEAD, as expected. They are preservation arms: they pin that 0-row runs need no ceiling and pass `NO_REGISTRY`.

Result on HEAD: 8 failed, 2 passed.

GREEN: `tests/unit/test_platform_run_joint_lift_budget.py`, **10 passed**.

## RED on HEAD (Task 2)

All 7 new cases failed with `KeyError: 'use_regime_filter'`:
- the driver's 4 cases, (tilt, filter) ∈ {T,F}²;
- the joint harness's 3 cases: (l1only, T), (l2, T) and (l2, F).

GREEN: both driver suites, **78 passed**.

## Mutation arms

Each arm was applied, the named tests were run, and the file was restored. `cmp` against a backup confirmed each restore.

| Arm | Mutation | Result |
|-----|----------|--------|
| M1 | pre-flight moved after both legs | 3 failed: refused-before-append, HEAD-shape arm, without-a-ceiling |
| M2 | `>` becomes `>=` | 1 failed: `test_exactly_at_the_ceiling_proceeds` |
| M3 | pre-flight written as an `assert` | 3 failed: `test_the_preflight_guard_survives_python_O`, `test_no_assert_statement_guards_the_script`, and refused-before-append (AssertionError, not RuntimeError) |
| M4 | ceiling defaults to 44 when absent | 1 failed: `test_a_decision_bearing_run_without_a_declared_ceiling_is_refused_before_any_work` |
| M5 | the requested flag recorded, not the effective one (both drivers) | 2 failed: driver `[filter=True, tilt=False]`, joint `[L1_ONLY, True -> False]` |
| M6 | `plan` key kept beside `harness_plan` | 3 failed: all three joint cases (`"plan" not in config`) |

## Ledger convention evidence (read only)

Both `registry/trials.jsonl` and `registry/archive/trials-pre-P7W1-reset.jsonl` were loaded read-only.

- **`run_backtest`-shaped rows** (`phase == "05-backtest"`, `trial_tag == "run_backtest"`, or a `use_regime_tilt` key): **42**, all in the archive. Timestamps run from `2026-07-27T18:17:48Z` to `2026-09-15T00:10:54Z`. All of them predate `ec354b1` (`2026-09-23T20:10:50+00:00`, when `run_backtest` gained the filter). None carries `use_regime_filter`.
- **Joint-harness rows** (`criterion == 7` or `routing` present): **4**, all in the live ledger. Timestamps run from `2026-09-21T14:21:29Z` to `2026-09-28T14:44:48Z`. The tags are `07-11-c1-alone-L1only`, `07-11-joint-c1xc2-L1only`, `08-10-c1-alone-L1only-notrade5pp` and `08-10-joint-c1xc2-L1only-notrade5pp`. **All 4 are `L1_ONLY_LAST_FILTERED_STATE`**, and all carry `plan: "07-11"`.
- The other 3 live rows are the `REGISTRY-RESET-P7W1` provenance header and 2 `07-07-inv01-screen` rows. They are neither shape.

No row violates the convention: **a row without `use_regime_filter` ran unfiltered.** The convention is written into both drivers' `use_regime_filter` Args docstrings. From this plan on, the key is authoritative on every row.

**The `plan` key convention.** Historical joint rows keep `"plan": "07-11"`, because the ledger is append-only. Newly appended rows carry `"harness_plan": "07-11"` and no `plan` key. A grep of `tests/`, `src/` and `scripts/` found no reader of a row's `plan` key. The decision-bearing criterion-7 record tests (`test_platform_joint_diagnostics_record.py`, which reads the committed record's `adr_0002_ceiling == 44`) were left unmodified and still pass.

## ADR-0004 migration status

**1 of 3 hard-coded-44 sites is migrated.** `scripts/run_joint_lift.py`'s `ADR_0002_CEILING` has been replaced by the required `--declared-ceiling`. Two sites remain for Phase 8.1, before its first registry row: `tests/unit/test_platform_gate_tiers.py` and `tests/unit/test_platform_joint_diagnostics_record.py`. The second of these pins the committed 08-10 record's historical `adr_0002_ceiling` key.

**For the orchestrator's STATE update:** the next decision-bearing joint-lift run must:
- pass `--declared-ceiling N`, taken from its phase CONTEXT.md `## Trial budget` declaration (none exists yet; `grep -l "^## Trial budget" .planning/phases/*/*-CONTEXT.md` printed nothing at execution);
- rename `BASELINE_TAG` / `JOINT_TAG`, which are still pinned to 08-10's names.

## Residual (stated in the script's docstring)

Some post-run guards necessarily run AFTER the appends:
- `rows_added != planned_rows`;
- `count_after > declared_ceiling`;
- `_leg_kpis`' terminal-log-wealth and max-drawdown plausibility checks.

A failure there leaves real, counted rows in the ledger. It is recorded as a finding under ADR-0004 §2, and the rows are never deleted. These guards now raise `RuntimeError`, not `AssertionError`.

## Verification

- `pytest tests/ -q`: **2494 passed**, 0 failed / skipped / xfailed / errors (5 warnings). Count before: 2477. Count after: 2494 (+10 budget, +4 driver, +3 joint).
- Recorded counts: all four sites (CLAUDE.md × 2, README.md × 2) = 2494. `test_docs_recorded_counts.py`: 7 passed.
- Legacy import ratchet: 11 passed (`MAX_LEGACY_IMPORT_SITES = 31`).
- Registry: `total_trial_count() == 44`, sha256 prefix `c957e8fdb360`, unchanged. `git status --porcelain -- data outputs registry` is empty. `git diff --quiet b7193fd -- registry/ legacy/ gsd-scratch-work/ trading-crab-lib/` passes.
- **`scripts/run_joint_lift.py` was never executed** against the real registry, in any mode. It was only imported, by the tests and by the `python -c "import run_joint_lift"` check that `diagnose_s1_truncation.py` depends on (exit 0). The unchanged sha256 is the evidence.
- ruff and flake8 (E9,F63,F7,F82) are clean on all 6 touched Python files.

## Deviations from Plan

None. The plan was executed as written.

- `_registry_block` also takes `planned_rows`, which it uses to null the ceiling on 0-row runs, as the plan's registry-block spec requires.
- The new post-run over-ceiling check is guarded on `planned_rows and declared_ceiling is not None`, which is equivalent to "decision_bearing" once the pre-flight has passed.

## Known Stubs

None.

## Self-Check: PASSED

- FOUND: tests/unit/test_platform_run_joint_lift_budget.py, scripts/run_joint_lift.py, driver.py, joint_driver.py
- FOUND commits: 7dcb781, 70bcd6c, 4184fbd, 332f9b9, faccf26
