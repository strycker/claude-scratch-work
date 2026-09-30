---
phase: 08-regime-persistence-stability
plan: 08
subsystem: platform/backtest + platform/report + platform/evaluation
tags: [criterion-1, criterion-2, PER-02, PER-03, bayes-filter, causal-invariance, S-1, S-3, honesty-framework]
status: complete
requires:
  - "08-06 (prediction/regime_filter.py; compute_signed_detection_offsets; the synthetic arms and _world() fixture)"
  - "08-01 (churn.py; joint_lift_probs_{1,2}_l2.parquet; B0 = 221/487 and 66/487)"
provides:
  - "evaluation/sojourn_lag.py::classify_negative_offsets — the pre-registered held-through-return rule (S-1, observational since the ruling)"
  - "joint_driver.run_joint_backtest(use_regime_filter=True): belief carried as a loop variable under ROUTING_L2_NOWCAST only"
  - "driver.run_backtest(use_regime_filter=True) and report/weekly.py load/save_regime_belief + advance_regime_belief: one cold-start rule (the same function object)"
  - "the GOVERNING leakage guard: an every-month causal-invariance sweep on 08-06's synthetic world, plus real-data p-1 adjudication (s1_truncation_invariance.json)"
  - "B1 + S-3 readings in diagnostics_l2_observational.json; joint_lift_belief_{1,2}_l2.parquet"
affects:
  - "08-09: the belief is what the hysteresis/tilt consume under l2; hysteresis sees classifier #1 only while the tilt blends two (recorded in joint_driver's docstring)"
  - "08-10: l1only decision-bearing leg re-confirmed byte-identical on real data, filter on AND off; the tracked l2 equity curves are still PRE-FILTER until 08-10 regenerates the l2 record"
decisions:
  - "Filter literally gated: `if routing == ROUTING_L2_NOWCAST and use_regime_filter:`; under l1only prev_belief stays None and no belief artifact is written"
  - "joint_driver: on ANY degraded step both classifiers advance by predict_only_step when their labels exist; hold + WARNING only when no labels"
  - "run_backtest: L1 and L2 share one try, so a degraded step has no labels and the belief is held with a WARNING; the use_regime_tilt=False ablation is never filtered"
  - "weekly: one filter step per MONTH — as-of dated belief; same-month re-run reuses it; a k-month gap applies k-1 predict-only steps"
  - "RULING 2026-09-23 (Glenn), after the S-1 halt: causal invariance governs; S-1 is observational; S-1's clauses unchanged (T-08-40b)"
  - "Real data is two-stage: S-1 detects, a truncation cut at p-1 adjudicates; 4 of 4 negative offsets bit-identical"
tech-stack:
  added: []
  patterns: ["literal routing gate + bit-for-bit pin", "golden pre-change curve as exact float hex recovered from git", "exhaustive truncation sweep with fixed A/prior", "two-stage detect/adjudicate on real data", "AST detector for target assertions, shown to fire"]
key-files:
  created:
    - outputs/reports/platform/joint_lift/joint_lift_belief_1_l2.parquet
    - outputs/reports/platform/joint_lift/joint_lift_belief_2_l2.parquet
    - outputs/reports/platform/joint_lift/s1_truncation_invariance.json
    - .planning/phases/08-regime-persistence-stability/08-CHURN.md
  modified:
    - src/trading_crab_lib/platform/evaluation/sojourn_lag.py
    - src/trading_crab_lib/platform/backtest/joint_driver.py
    - src/trading_crab_lib/platform/backtest/driver.py
    - src/trading_crab_lib/platform/report/weekly.py
    - scripts/run_joint_lift.py
    - scripts/joint_lift_diagnostics.py
    - outputs/reports/platform/joint_lift/diagnostics_l2_observational.json
    - tests/unit/test_platform_evaluation_sojourn_lag.py
    - tests/unit/test_platform_backtest_joint_driver.py
    - tests/unit/test_platform_backtest_driver.py
    - tests/unit/test_platform_report_weekly.py
    - tests/unit/test_platform_nowcaster_recursion.py
    - tests/unit/test_platform_joint_diagnostics_record.py
metrics:
  duration: "about 2h across two sessions: 19:54Z to 20:51Z (to the halt), then 21:27Z to about 21:45Z after the ruling"
  completed: 2026-09-23
actuals:
  tokens: 33561   # chars/4 over the added lines of this plan's ten commits (parquets excluded; orchestrator commits excluded)
  tasks: 3
  commits: 11
---

# Phase 8 Plan 08: Bayes filter wired at three call sites; S-1 halted, the ruling adjudicated, three churn numbers measured

**What was built.** The filter runs in three places, all sharing one cold start
(`regime_filter.unconditional_belief`, the same function object):
- `joint_driver`, on the l2 routing only;
- `run_backtest`;
- the weekly serve path.

**The decision-bearing leg is untouched.** The real l1only curves are byte-identical with
the filter on, with it off, and against git.

**The halt, the ruling, and the adjudication.** The first real-data S-1 run found 3 LEADs
on classifier #2. As instructed, the plan halted at that point without adjusting anything.
Glenn then ruled that **causal invariance governs** and that S-1 becomes an observational
reading.
- **Governing guard, unit suite:** an every-month truncation sweep on the synthetic world.
  The honest filter is invariant at all 111 cuts. The smoothed-substitution arm breaks at
  20 cuts.
- **Real data:** all 4 negative offsets were adjudicated by a cut at p−1, and all 4 are
  bit-identical.

**The churn numbers (all l2).** Track A and B0 are unchanged; B1 is new. Each is
count / 487 adjacent pairs.

| | #1 | #2 |
|---|---|---|
| B1 (filtered belief) | **81 / 487** | **30 / 487** |
| B0 (raw posterior, unchanged) | 221 / 487 | 66 / 487 |

## Tasks and commits

| step | commit | what |
|---|---|---|
| rule pinned before real data | `4752bba` | `classify_negative_offsets` + synthetic pins; the four prototype results reproduce |
| Task 1 (tracer) | `a78edc3` | joint_driver l2 wiring; l1only pinned on/off |
| Task 2 | `ec354b1` | `run_backtest` + weekly; one cold-start function object |
| Task 3 — halt | `2649865` | real-data S-1 arm red on #2 (3 LEADs); belief artifacts as evidence; halt recorded |
| (summary at halt) | `189fa6c` | superseded by this file |
| *(orchestrator)* | `94d5283`, `639d56b`, `28bb9ba` | pooling-consumer fake stubbed; halt investigated; **RULING** recorded in the plan |
| Task 3 — step 1 | `6c33d7f` | S-1 real-data test becomes a pin of the measured counts |
| Task 3 — step 2 | `57651bd` | governing invariance test, every-month sweep |
| Task 3 — step 3 | `57467d3` | `s1_truncation_invariance.json` (4 cuts) + the record test citing it |
| Task 3 — step 4 | `94a9ecf` | B1 + S-3 in the diagnostics record; JSON regenerated (additions only) |
| Task 3 — record | `4b26cc3` | 08-CHURN.md: ruling, adjudication, three numbers, S-3 |

**Tracer gate.** The tracer's verify passed (34 tests plus the literal gate regex). The real
l1only leg was then run end to end before any expansion work, and was byte-identical to git.

## The halt and the ruling

**At the halt.** Under the pre-registered held-through-return rule, #1 had 0 LEADs and 1
held-through miss, at position 300 (1988-02, −6). #2 had 3 LEADs:

| position | reference transition | offset |
|---|---|---|
| 236 | 1982-09 | −1 |
| 387 | 1995-04 | −12 |
| 596 | 2012-09 | −31 |

At that point:
- nothing was registered;
- no B1 was computed;
- the rule was not touched.

**RULING — 2026-09-23 (Glenn).** Causal invariance is the governing leakage guard. S-1 is
observational. S-1's clauses are unchanged (T-08-40b).

**Step 1 — S-1 becomes a pin.** The S-1 test now pins the counts exactly as measured, so it
fails if any count moves. It no longer gates.

**Step 2 — governing invariance, unit suite.** The sweep cuts at every one of 111 months.
A and the class prior are supplied once, fixed.
- The honest filter is bit-identical at every cut.
- **Arm 2 breaks at 20 cuts:** {11:1, 12:2, 13:3, 14:4, 24:1, 36:1, 42:1, 43:2, 44:3, 45:4,
  56:1, 57:2, 66:1, 67:2, 68:3, 69:4, 79:1, 80:2, 88:1, 100:1}.
- Arm 2 is **invariant at 30, 60, 90 and 110**, pinned so the reason for the exhaustive
  sweep lives in the test.
- **Deviation, measured and not adjusted:** the ruling's recorded prototype result, "Arm 2
  breaks exactly at {13, 44, 68}, 3 rows each", reproduces exactly only when restricted to
  the p−1 cut family (the month before each reference transition). Under the every-month
  sweep the ruling itself requires, it breaks at 20 cuts. Both are pinned. The same
  measurement shows that a cut at the turn month p catches every transition (the non-led
  ones by 1 row), which no p−1 cut can see.
- **Further pins:**
  - a one-month structural read-ahead breaks at exactly the cuts where month T+1's
    evidence differs from month T's, and at no steady cut;
  - re-deriving the prior per truncation (`_run_filter`) raises at cut 13 and breaks the
    honest filter at cut 50, which is why A and the prior are fixed.

**Step 3 — real-data adjudication** (`s1_truncation_invariance.json`, 0 breaks):

| cut T | adjudicates | rows ≤ T | #1 | #2 |
|---|---|---|---|---|
| 1982-08-31 | #2 LEAD 236 | 66 | bit-identical | bit-identical |
| **1988-01-31** | **#1 held-through miss 300** — run by this plan, 98.8 s, NO_REGISTRY | **123** | **bit-identical, max diff 0.0** | **bit-identical, max diff 0.0** |
| 1995-03-31 | #2 LEAD 387 | 209 | bit-identical | bit-identical |
| 2012-08-31 | #2 LEAD 596 | 407 | bit-identical | bit-identical |

## The three churn numbers (each count / adjacent pairs, window, degraded count)

| | #1 | #2 |
|---|---|---|
| **A** `state_N`, l1only | 246 / 587 (41.91%), unchanged. 588 steps 1972-01-31 → 2020-12-31, 0 degraded | 24 / 587 (4.09%), unchanged |
| **A** `state_N`, l2 | 246 / 587, unchanged, 100 degraded | 24 / 587, unchanged, 100 degraded |
| **B0** raw-posterior argmax, l2 (control) | 221 / 487 (45.38%), unchanged. 488 rows 1974-02-28 → 2020-12-31, 100 degraded. Matrix byte-identical to 08-01's | 66 / 487 (13.55%), unchanged; byte-identical |
| **B1** filtered-belief argmax, l2 | **81 / 487 (16.63%)**, 488 rows 1974-02-28 → 2020-12-31, **100 degraded** | **30 / 487 (6.16%)**, same window, **100 degraded** |

- argmax(belief) and argmax(posterior) differ in **273 / 488** months (#1) and
  **155 / 488** (#2).
- No target was pre-declared, and none is asserted. An AST check that is shown to fire
  forbids one.

## S-3 readings (raw posterior → belief; directions only)

| | #1 | #2 |
|---|---|---|
| median lag (resolved) | 111.0 (15/25) → **72.5** (20/25) | 122.0 (2/12) → **44.0** (11/12) |
| sojourn / lag ratio | 0.086 → **0.131** | 0.238 → **0.659** |
| transition-window accuracy | 0.111 → 0.202 | 0.189 → 0.208 |
| steady-state accuracy | 0.165 → 0.062 | 0.448 → 0.418 |
| overall accuracy | 0.154 → 0.090 | 0.420 → 0.396 |
| max prob < 0.70 | 355/488 → **181/488** | 476/488 → **73/488** |

- **No collapse signature.** Lag falls and the ratio rises, which is the opposite of S-3's
  collapse direction.
- **Steady-state accuracy falls** for both classifiers.
- **Caveat.** Every row compares walk-forward output with the full-sample reference. For #2
  the two labelings share a vocabulary only 41.3% after the best 1:1 relabelling (08-CHURN
  §5.2). #1's overall accuracy is below 1/6, which suggests a mismatch there too; that has
  not been measured.
- **Unchanged:** the l1only headline, 9.5 / 4.0 = 2.375.

## l1only byte identity

The real `--routing l1 --dry-run` run was dumped to scratch. Every comparison below was
`cmp`-identical and `assert_frame_equal` exact:
- `joint_lift_{baseline,joint}_l1only.parquet` and `joint_lift_probs_{1,2}_l1only.parquet`
  against `git show 6a88638:`;
- filter on;
- filter off;
- on against off.

Criterion 7 reproduces exactly: wealth_delta **−0.12343826162064975**, dd_delta
**+0.02408401236666291**.

## Evidence each discriminating check can fail

- **Classifier rule:** the unguarded q−1, a clause (ii) that is always true, a
  NaN-tolerant (iii), and (iii) dropped each go red. Dropping one half of (iii) gives
  0 red, as the plan measured, so no arm for it was written.
- **joint_driver:** each of these goes red:
  - the filter applied under l1only;
  - a no-op filter;
  - the tilt fed the raw posterior;
  - a degraded step implemented as a hold;
  - a uniform cold start.
- **driver / weekly:** 7 mutations, each red:
  - the serve and driver tilts fed the raw posterior;
  - save before load;
  - a uniform cold start;
  - a same-month re-filter;
  - a copied cold-start helper;
  - the filter flag ignored (caught against the golden recovered from git).
- **Governing invariance:** Arm 2 breaks at 20 cuts, and the read-ahead breaks at every
  changing month, so the honest-invariance assertion would fail on either. Pinning Arm 2's
  invariance at cut 30 shows that a sparse cut set would pass it.
- **S-1 pin:** dropping 387, or dropping miss 300, goes red.
- **Adjudication record:** removing the 1988 cut, or flipping one `bit_identical`, goes red.
- **Diagnostics record:** each of these goes red:
  - B0 moved;
  - Track A moved;
  - B1's source pointed at the raw posterior (2 red);
  - B1's degraded count moved.
- **No-target detector:** fires on a target snippet and on a direction snippet, and passes
  a value re-derived from the artifact.

## Deviations from Plan

1. **[HALT, then RULING]** Task 3 halted on 3 real-data LEADs. The rule was not adjusted.
   It was completed after Glenn's ruling, under the ruling's text.
2. **The ruling's recorded prototype result {13, 44, 68} does not hold under its own
   every-month sweep.** It holds for the p−1 cut family. Both are pinned as measured
   (above).
3. **[Rule 1]** The driver test fake `_fake_refit_l1` returned only state 0 while
   `_fake_refit_l2` named state 1. The filter correctly raised on that pairing, and the fake
   now emits both states.
4. **[Rule 2]** The weekly belief is as-of dated, so that weekly re-runs do not count the
   same month's evidence twice.
5. **`diagnostics_l1only.json` was not regenerated** (it is not in `files_modified`). The
   l1only record test accepts a missing `walk_forward_belief` key as "not applicable".
6. **The tracked `joint_lift_{baseline,joint}_l2.parquet` equity curves remain PRE-FILTER
   until 08-10 regenerates the l2 record.** They are not in `files_modified`. Their `state_*`
   and `degraded` columns are identical to the filter-on run's.
   `diagnostics_l2_observational.json` reads only those columns, so it is unaffected.
7. **New synthetic l2 fixture** (a 96-month square wave), because the default fixture
   degrades every l2 step.
8. **`classify_negative_offsets` was committed ahead of the wiring** as its own sub-step.
   The S-1 real-data test was split, so that its adjudication half landed with the JSON it
   reads.

## Verification

| check | result |
|---|---|
| Task 1 verify | 34 passed; literal gate found |
| Task 2 verify | 47 passed; one cold-start object across driver, joint_driver and weekly; nowcaster/tilt/hysteresis untouched |
| Task 3 pytest verify | 81 passed (recursion + diagnostics record) |
| Task 3 JSON verify | `A 246 B0 221 B1 81`; degraded 100 for B0 and B1 |
| Task 3 CHURN grep verify | ok |
| `total_trial_count()` | **42 before, 42 after** every run, including the 1988-01-31 cut |
| legacy-import ratchet | 11 passed, 31 |
| 2021+ holdout | not read; every artifact and cut ends ≤ 2020-12-31 |
| ruff + flake8 (E9,F63,F7,F82) | clean on every touched .py, before each commit |
| `pytest tests/ -q` | **2263 passed, 0 failed, 0 skipped** (2240 at the halt + 23: the S-1 pin, 5 invariance tests, the adjudication test, 16 record tests; the orchestrator's `94d5283` turned the pooling-consumer test green) |

## Known Stubs

None.

## Threat Flags

| Flag | File | Description |
|------|------|-------------|
| threat_flag: new persisted state | src/trading_crab_lib/platform/report/weekly.py | New `regime_belief` checkpoint (parquet: state, belief, as_of) in the platform checkpoint namespace; local file only, same pattern as `hysteresis_state` |

## Self-Check: PASSED

- FOUND (tracked): both belief parquets, `s1_truncation_invariance.json`, the regenerated `diagnostics_l2_observational.json`, `08-CHURN.md`
- FOUND commits on `claude/keen-galileo-zqcml6-w5`: `4752bba`, `a78edc3`, `ec354b1`, `2649865`, `189fa6c`, `6c33d7f`, `57651bd`, `57467d3`, `94a9ecf`, `4b26cc3`, and this summary. Not pushed.
- Registry 42. STATE.md, ROADMAP.md and REQUIREMENTS.md were not touched (orchestrator instruction).
