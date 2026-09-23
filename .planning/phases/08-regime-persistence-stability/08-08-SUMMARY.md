---
phase: 08-regime-persistence-stability
plan: 08
subsystem: platform/backtest + platform/report + platform/evaluation
tags: [criterion-1, criterion-2, PER-02, PER-03, bayes-filter, S-1, leakage-guard, HALTED, honesty-framework]
status: halted
requires:
  - "08-06 (prediction/regime_filter.py; compute_signed_detection_offsets; the synthetic arms and _world() fixture)"
  - "08-01 (churn.py; joint_lift_probs_{1,2}_l2.parquet; B0 = 221/487 and 66/487)"
provides:
  - "evaluation/sojourn_lag.py::classify_negative_offsets — the pre-registered held-through-return rule, pinned on the synthetic arms"
  - "joint_driver.run_joint_backtest(use_regime_filter=True): belief carried as a loop variable under ROUTING_L2_NOWCAST only"
  - "driver.run_backtest(use_regime_filter=True) and report/weekly.py load/save_regime_belief + advance_regime_belief: one cold-start rule (the same function object)"
  - "outputs/reports/platform/joint_lift/joint_lift_belief_{1,2}_l2.parquet — committed as EVIDENCE of the halt, not as a result"
  - ".planning/phases/08-regime-persistence-stability/08-CHURN.md — the halt record"
affects:
  - "08-09: must not proceed on the belief until a human rules on the S-1 halt (T-08-40 / T-08-40b)"
  - "08-10: l1only decision-bearing leg re-confirmed byte-identical on real data, filter on AND off"
  - "tests/unit/test_platform_pooling_consumers.py::TestWeeklyConsumer is RED (outside scope; see Blockers)"
decisions:
  - "Filter literally gated: `if routing == ROUTING_L2_NOWCAST and use_regime_filter:`; under l1only prev_belief stays None and no belief bucket is written"
  - "joint_driver: on ANY degraded step both classifiers advance by predict_only_step when their labels exist (even if one classifier's L2 succeeded), because the step is excluded from metrics and weights are held"
  - "run_backtest: L1 and L2 share one try, so a degraded step has no labels and the belief is held with a WARNING (no A to advance by)"
  - "run_backtest: the filter is never applied on the use_regime_tilt=False ablation (a constant one-state vector is not a posterior)"
  - "weekly: one filter step per MONTH — the belief is persisted with its as-of month; a same-month re-run reuses it, a k-month gap applies k-1 predict-only steps then one filter step"
  - "On the S-1 LEAD: nothing registered, no B1 churn computed or recorded, diagnostics_l2_observational.json not regenerated, the rule's clauses untouched"
tech-stack:
  added: []
  patterns: ["literal routing gate + bit-for-bit pin", "golden pre-change curve as exact float hex recovered from git", "spy on the consumer to prove what it receives", "AST detector for target assertions, shown to fire"]
key-files:
  created:
    - outputs/reports/platform/joint_lift/joint_lift_belief_1_l2.parquet
    - outputs/reports/platform/joint_lift/joint_lift_belief_2_l2.parquet
    - .planning/phases/08-regime-persistence-stability/08-CHURN.md
  modified:
    - src/trading_crab_lib/platform/evaluation/sojourn_lag.py
    - src/trading_crab_lib/platform/backtest/joint_driver.py
    - src/trading_crab_lib/platform/backtest/driver.py
    - src/trading_crab_lib/platform/report/weekly.py
    - scripts/run_joint_lift.py
    - tests/unit/test_platform_evaluation_sojourn_lag.py
    - tests/unit/test_platform_backtest_joint_driver.py
    - tests/unit/test_platform_backtest_driver.py
    - tests/unit/test_platform_report_weekly.py
    - tests/unit/test_platform_nowcaster_recursion.py
metrics:
  duration: "about 1h (first commit 19:54Z, halt commit 20:47Z), plus the pre-interruption reading"
  completed: 2026-09-23
actuals:
  tokens: 17984   # chars/4 over the added lines of the realized diff 6a88638..HEAD, parquets excluded
  tasks: 3        # Tasks 1 and 2 complete; Task 3 HALTED at the S-1 gate
  commits: 4
---

# Phase 8 Plan 08: Bayes filter wired at all three call sites; the real-data leakage guard HALTED the measurement

The filter now runs in `joint_driver` (l2 only), `run_backtest` and the weekly serve path.
All three call sites share one cold start, `regime_filter.unconditional_belief`, the same
function object. On the decision-bearing leg it is byte-for-byte inert: the real l1only
curves are identical with the filter on, with it off, and to git. **The S-1 guard on the
real belief path found 3 LEADs for classifier #2** under the pre-registered
held-through-return rule. Per instruction the plan stopped there:
- no B1 churn was computed;
- nothing was registered;
- the rule was not touched.

## Tasks and commits

| task | commit | status |
|---|---|---|
| sub-step: `classify_negative_offsets` + synthetic pins (before any real data) | `4752bba` | done |
| 1 (tracer): joint_driver l2 wiring, l1only pinned | `a78edc3` | done |
| 2: run_backtest + weekly, one cold-start rule | `ec354b1` | done |
| 3: measurement | `2649865` | **HALTED at the S-1 gate**. The halt evidence and 08-CHURN.md are committed |

**Tracer gate** (autonomous, per the orchestrator). The tracer verify was re-run: 34
passed, and the literal gate regex passed. The real l1only leg was then run end to end
before any expansion work. It came back byte-identical to git (below).

## The rule, pinned before real data (`4752bba`)

`classify_negative_offsets` implements the plan's three clauses verbatim. `q−1 >= 0` is
guarded before indexing. NaN belief months never satisfy clause (iii), so missing data
cannot exempt an offset. All four prototype results reproduced exactly:

| arm | leads | misses |
|---|---|---|
| caveat (A diag 0.999) | [] | [45] |
| Arm 2 (smoothed substitution) | [14, 45, 69] | [] |
| Arm 1 (honest) | no negative offsets | |
| clause (iii) dropped entirely (monkeypatched) | [14, 69] | [45] |

Clause pins were added for: start-of-series (the wrap guard), a non-return `t→r→s`, early
re-entry (the belief registers `r`), and a missing month at q−1.

## The three churn numbers

| | #1 | #2 | window, degraded |
|---|---|---|---|
| **A** (`state_N`, both routings) | **246 / 587**, unchanged | **24 / 587**, unchanged | 588 steps 1972-01-31 → 2020-12-31; 0 degraded l1only, 100 l2 |
| **B0** (raw-posterior argmax, l2) — the control | **221 / 487**, unchanged | **66 / 487**, unchanged | 488 rows 1974-02-28 → 2020-12-31, **100 degraded** |
| **B1** (filtered-belief argmax, l2) | **not computed**: halted by S-1 | **not computed** | belief matrix: 488 rows, same index as B0, 100 degraded |

B0 is stronger than "count unchanged": both raw posterior matrices are **byte-identical**
to 08-01's committed files (max abs diff 0.0). Track A holds elementwise: the l2 run's
`state_1`, `state_2` and `degraded` columns equal the tracked l2 curves, in both legs.

## l1only byte identity (the decision-bearing pin)

The real `--routing l1 --dry-run` run was dumped to scratch. Every comparison below came
back **`cmp`-identical** and `assert_frame_equal` exact:
- `joint_lift_{baseline,joint}_l1only.parquet` and `joint_lift_probs_{1,2}_l1only.parquet`
  against `git show 6a88638:<path>`;
- filter on;
- filter off (`use_regime_filter=False` injected);
- on against off.

Criterion 7 reproduces to the last digit in both runs: `wealth_delta`
**−0.12343826162064975**, `dd_delta` **+0.02408401236666291**. No belief artifact is
written under l1only.

## Real-data S-1 guard

| clf | transitions / resolved | min offset | n_negative | **n_lead** | n_held_through_miss |
|---|---|---|---|---|---|
| #1 | 25 / 20 | −6 | 1 | **0** | **1**: position 300, 1988-02-29, −6, into 4, over the state-0 run 1987-10..1988-01 |
| #2 | 12 / 11 | −31 | 3 | **3** | 0 |

The #2 LEADs:
- **236**: 1982-09, −1. Fails (ii): `1→0→2` is not a return.
- **387**: 1995-04, −12. Fails (iii): belief[2] was 0.468 at q−1 = 1994-02.
- **596**: 2012-09, −31. Fails (ii): `2→3→4` is not a return.

Context recorded in 08-CHURN.md §3. It is not a reclassification:
- the raw posterior has **no** negative offsets (min +3 / +110);
- the pre-existing L1-only one-hot for #2 already "leads" at 596 by −264 months;
- three candidate readings (leak / reference timing / label disagreement) are listed for a
  human.

## S-3 readings

Not computed on the belief, because of the halt. The pre-filter context stands:
- l1only 9.5 / 4.0 = 2.375;
- raw-posterior max probability below 0.70 in 355 / 488 rows (#1) and 476 / 488 (#2), per
  08-01.

## Evidence that each discriminating check can fail

- **Classifier:** M1 unguarded q−1 → 1 red. M2 clause (ii) always true → 4 red. M3 a
  NaN-tolerant (iii) → 1 red. M4 (iii) dropped → 3 red. M5 dropping only the `belief[r]`
  half → **0 red**. That confirms the plan's measured statement that a half-drop arm
  cannot fail at 0.70, so none was written.
- **joint_driver:** filter applied under l1only → l1only pin red. A no-op filter on #1 →
  belief-to-tilt test red. Tilt fed the raw posterior → red. Degraded step implemented as
  a hold → red. Uniform cold start → red.
- **driver / weekly:** each of the following turns a test red:
  - serve tilt fed the raw posterior;
  - save before load;
  - uniform cold start;
  - a same-month re-filter;
  - a copied `unconditional_belief` (fails the identity-object test);
  - driver tilt fed the raw posterior;
  - `use_regime_filter` ignored (the filter-off vs pre-change golden goes red).
- **The real S-1 gate itself fired** on #2. The check is live on real data.

## Deviations from Plan

1. **[HALT, T-08-40] Task 3 stopped at the S-1 gate.** B1, the S-3 readings on the belief,
   and the regenerated `diagnostics_l2_observational.json` do not exist. The diagnostics
   extension (`walk_forward_belief` block, S-1/S-3 readings) and its record tests were
   written and are **uncommitted by design**, because their tests need the regenerated
   JSON, which would record B1. The patch is saved at
   `/tmp/claude-0/-home-user-claude-scratch-work/64f0b80b-f0fb-5914-ae2e-935a933024c0/scratchpad/0808-task3-unapplied-diagnostics-and-record-tests.patch`
   (session scratch; regenerable from the plan). Those two files were restored with
   `git checkout -- <file>`.
2. **The real-data gate test is committed RED on classifier #2** (`2649865`). Marking it
   xfail or weakening it would be adjusting the guard.
3. **[Rule 1] The driver test fake `_fake_refit_l1` was all-zero** while `_fake_refit_l2`
   named state 1. With the filter on the default path, `likelihood_ratio` correctly raised
   on that impossible pairing (8 red). The fake now emits both states. It is in
   `files_modified`.
4. **[Rule 2] Weekly cadence vs a monthly filter.** Re-filtering on every weekly run would
   count the same month's evidence up to 4 times. The belief is now as-of dated: a
   same-month re-run is reused, and a gap applies predict-only for each unobserved month.
   Tested.
5. **The belief artifacts were produced in a scratch run and copied into the tracked dir.**
   The filter-on l2 equity curves are **not** committed (they are not in
   `files_modified`), so the tracked l2 curves remain the pre-filter observational leg.
   Their state and degraded columns are identical to the filter-on run's.
6. **`classify_negative_offsets` was committed ahead of the wiring**, as its own
   sub-step, per the orchestrator's instruction.
7. **The synthetic l2 fixture was new**, because the default `_frames()` degrades every l2
   step. It uses a 96-month square wave: 60 steps, 8 degraded.

## Blockers / Found, not fixed

- **RED, outside scope:** `tests/unit/test_platform_pooling_consumers.py::TestWeeklyConsumer::test_weekly_tilt_call_receives_the_unpooled_returns_by_regime`.
  Its `_FakeCheckpointManager.load` raises `KeyError` for any unserved name. The real
  manager raises `FileNotFoundError`, which `load_regime_belief` handles. The test already
  monkeypatches `load/save_active_regime` for exactly this reason. The fix is two
  analogous lines for `load/save_regime_belief`. **Not applied**: the file is outside
  `files_modified`.
- **The S-1 halt needs a human ruling** (08-CHURN.md §3). Changing the rule after seeing
  real data is T-08-40b.
- The hysteresis sees classifier #1 alone while the tilt blends both classifiers. This is
  recorded in `joint_driver`'s docstring for 08-09.

## Verification

| check | result |
|---|---|
| Task 1 verify (pytest + literal gate regex) | 34 passed; gate found |
| Task 2 verify (pytest, one-cold-start object check, scope diff) | 47 passed; one rule across all three modules; nowcaster/tilt/hysteresis untouched |
| Task 3 grep verify on 08-CHURN.md | ok |
| Task 3 JSON verify | not run: JSON not regenerated (halt) |
| `total_trial_count()` | **42 before, 42 after** every run |
| legacy-import ratchet | 11 passed, 31 |
| 2021+ holdout | not read; every artifact ends 2020-12-31; the real-data arm asserts it |
| ruff + flake8 (E9,F63,F7,F82) | clean on every committed .py, before each commit |
| `pytest tests/ -q` | **2240 collected: 2238 passed, 2 failed, 0 skipped** (2212 at wave-2 close + 28 new). The 2 failures are the expected ones: the S-1 gate on classifier #2 (the halt) and the out-of-scope `test_platform_pooling_consumers.py::TestWeeklyConsumer` fixture (Blockers) |

## Known Stubs

None.

## Threat Flags

| Flag | File | Description |
|------|------|-------------|
| threat_flag: new persisted state | src/trading_crab_lib/platform/report/weekly.py | New `regime_belief` checkpoint (parquet: state, belief, as_of) in the platform checkpoint namespace; local file only, same pattern as `hysteresis_state` |

## Self-Check: PASSED

- FOUND: both belief parquets (tracked), 08-CHURN.md, and every modified source/test file
- FOUND commits on `claude/keen-galileo-zqcml6-w5`: `4752bba`, `a78edc3`, `ec354b1`, `2649865`. Not pushed.
- Registry 42. STATE.md, ROADMAP.md and REQUIREMENTS.md not touched (orchestrator instruction).
