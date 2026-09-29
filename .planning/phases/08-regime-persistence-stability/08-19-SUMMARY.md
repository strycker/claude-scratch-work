---
phase: 08-regime-persistence-stability
plan: 19
subsystem: platform/evaluation records (joint lift l2, diagnostics, S-1), phase records (08-MEASUREMENTS, 08-SERVING)
tags: [gap-closure, CR-01, re-measurement, no-registry, amendment, honesty, tracer]
status: complete
requires:
  - "08-15 (neutral-posture per-asset display)"
  - "08-16 (served training prior)"
  - "08-17 (both backtest drivers divide by the training prior)"
  - "08-18 (registry integrity; run_joint_lift ceiling pre-flight)"
provides:
  - "l1only record proven byte-unchanged in three forms on the final source"
  - "l2 curves, belief matrices, measurement and diagnostics records regenerated behind controls (NO_REGISTRY)"
  - "s1_truncation_invariance.json (plan 08-19): 3 cuts, both comparators, n_break 0"
  - "_MEASURED_S1 at the adjudicated new reading"
  - "08-MEASUREMENTS §11 and 08-SERVING §4 amendments, old -> new"
affects:
  - "Phase 8.1 (CR-02, CR-03 still qualify every L2 number)"
  - "Glenn: open question on the keep-absolute threshold ruling's cited counts"
tech-stack:
  added: []
  patterns:
    - "controls before install: probs, states and degraded identical, else halt"
    - "a second, NaN-aware comparator beside the truncation script's nanmax (WR-08)"
    - "amend, never overwrite: old numbers stand beside new ones"
key-files:
  created:
    - .planning/phases/08-regime-persistence-stability/08-19-SUMMARY.md
  modified:
    - outputs/reports/platform/joint_lift/joint_lift_baseline_l2.parquet
    - outputs/reports/platform/joint_lift/joint_lift_joint_l2.parquet
    - outputs/reports/platform/joint_lift/joint_lift_belief_1_l2.parquet
    - outputs/reports/platform/joint_lift/joint_lift_belief_2_l2.parquet
    - outputs/reports/platform/joint_lift/measurement_l2_observational.json
    - outputs/reports/platform/joint_lift/diagnostics_l2_observational.json
    - outputs/reports/platform/joint_lift/s1_truncation_invariance.json
    - tests/unit/test_platform_nowcaster_recursion.py
    - .planning/phases/08-regime-persistence-stability/08-MEASUREMENTS.md
    - .planning/phases/08-regime-persistence-stability/08-SERVING.md
decisions:
  - "None taken. The keep-absolute ruling's cited counts moved (307 -> 273, 415 -> 408 of 488). This is recorded as an open question for Glenn, not decided."
metrics:
  duration: "~65 min"
  completed: 2026-09-29
estimate:
  tokens: 90000
  tasks: 3
actuals:
  tokens: 7400    # chars/4 over the realized text diff (29535 chars of +/- lines in .md/.json/.py; parquets binary, excluded)
  tasks: 3
  commits: 3      # Task 2, Task 3, this SUMMARY (Task 1 writes only to scratch, by plan)
---

# Phase 8 Plan 19: CR-01 Re-measured — l1only Proven Unchanged, l2/B1/S-1 Re-measured Behind Controls, Served Belief Now Regime 0

Three things are shown here. The decision-bearing l1only record did not move: it is byte-identical in three forms. The observational l2 leg moved only where the fix acts: the posterior, the states and the degraded set are identical. B1 fell to 64/487 and 27/487. S-1 shrank to 2 LEADs (#2), each adjudicated bit-identical by two comparators. On the real tracked data, the served belief now puts regime 0 on top (0.300; before, regime 3 at 0.369). That vector equals 08-16's unit-test floats exactly. Every old number stands in 08-MEASUREMENTS §11 and 08-SERVING §4, next to its new value.

## Tasks

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 (tracer) | Both legs re-run into scratch; l1only byte proof; l2 controls | none (scratch only, by plan) | — |
| 2 | Install l2 artifacts, regenerate diagnostics, adjudicate S-1, move the pin | 324d2c2 | 7 joint_lift artifacts, test_platform_nowcaster_recursion.py |
| 3 | Served re-run on real data; both amendments | 9fb9c9f | 08-MEASUREMENTS.md, 08-SERVING.md |

## Commands and wall times

All runs used `NO_REGISTRY`. `$S` is the session scratchpad.

| Command | Wall | Exit |
|---|---|---|
| `python scripts/run_joint_lift.py --routing l1 --dry-run --dump-curves $S/rerun/l1 --out $S/rerun/l1/measurement_l1only_dryrun.json` | 1296 s | 0 |
| `python scripts/run_joint_lift.py --routing l2 --dump-curves $S/rerun/l2 --out $S/rerun/l2/measurement_l2_observational.json` | 1626 s | 0 |
| `python scripts/joint_lift_diagnostics.py --curves outputs/reports/platform/joint_lift --suffix l2 --out outputs/reports/platform/joint_lift/diagnostics_l2_observational.json` | 3 s | 0 |
| `python scripts/diagnose_s1_truncation.py 1982-08-31 $S/cuts/1982-08-31` | 488 s | 0 |
| `python scripts/diagnose_s1_truncation.py 1995-03-31 $S/cuts/1995-03-31` | 1051 s | 0 |
| `python scripts/diagnose_s1_truncation.py 2012-08-31 $S/cuts/2012-08-31` | 1208 s | 0 |
| `TC_DATA_DIR=$S/serve2/data TC_OUTPUT_DIR=$S/serve2/out python -m trading_crab_lib.platform.report.serving` | 4 s | 0 |
| same env, `python -m trading_crab_lib.platform.report.weekly` (run 1, run 2) | 2 s, 2 s | 0, 0 |
| `pytest tests/ -q -p no:cacheprovider` | 298 s | 0 |

The two joint-lift runs overlapped, and the three cuts ran in parallel (4 CPUs). l2 is bit-reproducible within one environment (§7). Load changes wall time only.

## Task 1: the l1only proof (HOLDS, all three forms)

- **cmp:** `joint_lift_{baseline,joint}_l1only.parquet` and `joint_lift_probs_{1,2}_l1only.parquet` from the dry run are **IDENTICAL** to the committed files.
- **Record fields**, compared exactly (value and type):

  | Block | Keys compared | Mismatches | Excluded |
  |---|---|---|---|
  | `lift` | 28 | 0 | none |
  | `baseline_leg`, `joint_leg` | 27 each | 0 | `registry_row_written` |
  | `deflated_sharpe.baseline`, `.joint` | 9 each | 0 | `n_trials_read_at` |
  | top-level configuration (`routing`, `blend_weight_1`, `no_trade_band`, `use_regime_filter` False, `K_1`, `lambda_1`, `K_2`, `lambda_2`, `frozen_features_1/2`, `first_decision`) | 11 | 0 | none |

- **Excluded fields and why a dry run differs in each (old → new):**
  - `decision_bearing`: True → False (a dry run is never decision-bearing).
  - `dry_run`: False → True.
  - `registry` block: {42 → 44, rows_added 2, the 08-10 tags, `adr_0002_ceiling` 44} → {44 → 44, rows_added 0, "(NO_REGISTRY)" ×2, 08-18's `declared_ceiling` null}. A dry run writes no row. The block's schema is 08-18's.
  - `quality_tier.governs`: True → False, and `quality_tier.verdict`: "FAILED on 2 of 2 legs (baseline, joint)" → "NOT GOVERNING - observational or dry run; ...". `hurdle`, `n_trials` (44), `baseline_ok` and `joint_ok` are the SAME.
  - `n_trials_read_at` is the read time. `n_trials` itself is equal (44).
  - `registry_row_written`: True → False.
- **git diff:** `git diff --quiet b7193fd --` over the six l1only files is CLEAN.

## Task 1: the l2 controls (HOLD)

- `joint_lift_probs_{1,2}_l2.parquet`: value-identical (`assert_frame_equal`, `check_exact=True`, 488 rows each), and **`cmp` IDENTICAL**. They were not re-installed.
- Both new l2 curves: `state_1`, `state_2` and `degraded` are identical to the committed ones. Degraded is 100 of 588 on both legs (87 on #1, 13 on #2).
- Differing columns, both legs: `return`, `turnover`, `cost`, `active_regime`, `scale`. That is exactly the expected set. First differing date: **1974-02-28** for `return`, `turnover`, `cost` and `scale` (the first non-degraded step); **1976-11-30** for `active_regime`.
- Record: `registry.rows_added` 0; both tags "(NO_REGISTRY)".
- Window: 588 steps, 1972-01-31 → 2020-12-31, 100 degraded. Matrix rows: 488, 1974-02-28 → 2020-12-31.
- The tracer gate re-ran the plan's two `<verify>` commands end to end, green, before any install.

## Task 2: S-1 cuts

The cut set is the p−1 dates (on classifier #2's dev reference) of the two new LEADs, plus the standing 1982-08-31 spot-check. #1 has no negative offset on the new path.

| T | adjudicates | rows ≤ T | script `bit_identical` (#1 / #2) | NaN-aware `assert_frame_equal(check_exact)` (#1 / #2) | seconds |
|---|---|---|---|---|---|
| 1982-08-31 | standing spot-check. 08-08's LEAD at 236 is gone | 66 | True / True (max abs diff 0.0) | True / True (0 NaN cells) | 484.7 |
| 1995-03-31 | #2 LEAD at 387 (1995-04-30, −12) | 209 | True / True | True / True | 1048.6 |
| 2012-08-31 | #2 LEAD at 596 (2012-09-30, −32) | 407 | True / True | True / True | 1205.3 |

`n_negative_offsets` 2, `n_adjudicated` 3 (it counts cuts, including the spot-check), `n_break` 0.

The NaN-aware comparator was itself shown to fail first. On a copy of an old cut with one NaN injected, `nan_aware_exact` was False, naming row 1976-08-31, while the script's `bit_identical` was True.

The old pin was then shown to fail on the new path. HEAD's `test_platform_nowcaster_recursion.py`, run against the installed belief, was 2 failed: "classifier #1: S-1 moved from ... [300] to {'n_negative': 0, ...}" and "classifier #2: ... [236, 387, 596] to {'n_negative': 2, 'lead_positions': [387, 596], ...}". Only after the adjudication was `_MEASURED_S1` moved. The 08-08 reading is kept verbatim in its comment.

## Old → new (as in 08-MEASUREMENTS §11)

Curve window: 588 steps, 1972-01-31 → 2020-12-31, 100 degraded. Matrix window: 488 rows (487 pairs), 1974-02-28 → 2020-12-31. **All observational.**

| quantity | old | new |
|---|---|---|
| B1 #1 / #2 | 81/487 = 16.63% / 30/487 = 6.16% | **64/487 = 13.14% / 27/487 = 5.54%** |
| argmax(belief) ≠ argmax(posterior) #1 / #2 | 273/488 / 155/488 | **380/488 / 178/488** |
| belief clearing 0.70 #1 / #2 | 307/488 / 415/488 | **273/488 / 408/488** |
| l2 `wealth_delta` / `dd_delta` | −0.30060767886336137 / +0.06101144035366535 | **−0.2967834422163289 / +0.0632580887963281** |
| l2 terminal log wealth, #1-alone / joint | 4.228027 / 3.927420 | **4.262866 / 3.966083** |
| l2 max drawdown, #1-alone / joint | −0.254700 (32 mo) / −0.193688 (68 mo) | **−0.267290 (33 mo) / −0.204032 (33 mo)** |
| l2 mean turnover, #1-alone / joint | 0.141119 / 0.068775 | **0.139896 / 0.070970** |
| l2 Sharpe, #1-alone / joint | 1.263438 / 1.157769 | **1.178896 / 1.159271** |
| l2 DSR, #1-alone / joint | 6.520758e-35 / 1.018363e-45 at n=42, hurdle 2.208694 | **3.978469e-46 / 3.171019e-48** at n=44, hurdle 2.226891 (at n=42 they would be 1.308430e-44 / 1.151274e-46; the 42 → 44 is the registry read date, not the fix) |
| S-1 #1 | 1 (held-through miss 300, −6) | **0** |
| S-1 #2 | 3 LEADs: 236 (−1), 387 (−12), 596 (−31) | **2 LEADs: 387 (−12), 596 (−32)** |
| S-3 #1 overall / transition | 44/488 = 0.0902 / 20/99 = 0.2020 | **32/488 = 0.0656 / 18/99 = 0.1818** |
| S-3 #2 overall / transition | 193/488 = 0.3955 / 11/53 = 0.2075 | **158/488 = 0.3238 / 10/53 = 0.1887** |
| belief median lag #1 / #2 | 72.5 (20/25 resolved) / 44.0 (11/12) | **52.0 (18/25) / 59.0 (11/12)** |
| served belief top state | 3 (0.369157) | **0 (0.299992)** |

Unchanged, each with a control: Track A 246/587 and 24/587; B0 221/487 and 66/487; series identity 184/488 and 276/488; degraded 100/588; the l1only record.

## Task 3: the served re-run (real tracked data, scratch copies, cold start)

- `nowcaster_class_prior` = {0: 11/153, 3: 137/153, 4: 5/153}, exactly as floats.
- Served distribution (unchanged): 3: 56.4%, 0: 41.8%, 4: 1.8%. The distinct count is still 1 across 231 complete months (2007-04-30 → 2026-06-30). Scored as-of 2026-06-30.
- Belief: {0: 0.29999237288846065, 1: 0.29270739835551485, 2: 0.09155165323013899, 3: 0.16323594694900967, 4: 0.059671515442087014, 5: 0.09284111313478884}. The argmax is 0. It is **exactly equal** to the value recomputed through 08-16's unit-test path (`fit_l2_nowcaster` → `advance_regime_belief`), and to 08-16's recorded floats.
- State 3's likelihood ratio: 1.9599 → 0.6299.
- Active regime: none (neutral posture).
- Executed book: TLT 0.390654 → **0.433693**, SPY 0.330723 → **0.320510**, USO 0.144729 → **0.135247**, IAU 0.133895 → **0.110549**. Cash residual 0.0%.
- Per-Asset Signals: 24 unlabelled rows → the 08-15 neutral sentence, with no rows.
- Run 2's `weekly_report.md` is `cmp` byte-identical to run 1's.

## Registry at every checkpoint

`total_trial_count()` read 44 and `registry/trials.jsonl` had sha256 `c957e8fdb360f04cebd36a6ac19f7efe5c036d098152a5b0c1694f2e73c088ad` at each of these points:
- before the runs;
- inside both joint-lift records (`count_before` = `count_after` = 44);
- after the l2 run;
- after the served re-run;
- after the three cuts;
- after Task 2's tests;
- after the full suite.

The cut record also stores 44/44. `git status --porcelain -- data registry` stayed empty throughout. `git status --porcelain -- outputs` was empty outside Task 2's install.

## Verification

- `pytest tests/ -q`: **2494 passed**, 0 failed / skipped / xfailed (5 warnings), 295.65 s. The count is unchanged from 08-18, because this plan adds no test.
- `test_docs_recorded_counts.py` + the legacy-import ratchet: 18 passed; `MAX_LEGACY_IMPORT_SITES = 31`.
- `git diff --quiet b7193fd -- registry/ legacy/ gsd-scratch-work/ trading-crab-lib/`: clean. The six l1only files and the two l2 probs files are unchanged since b7193fd.
- No source change: `git diff --quiet bdf401e -- src/ scripts/` (the last 08-18 commit).
- The amendment diff check: 0 removed lines in 08-MEASUREMENTS.md and 08-SERVING.md since b7193fd.
- No holdout path was read by any fit. The joint-lift and cut runs carve at 2020-12-31, and weekly reads the full span through its existing opt-in only.

## Pins moved, and the control that licensed each

- `_MEASURED_S1` (test_platform_nowcaster_recursion.py). Licensed by the three cuts, each bit-identical under both comparators, after the old pin was shown RED on the new path.
- The regenerated l2 artifacts that the record suites read (B0 pins 221/66 and degraded 100 untouched). Licensed by the probs, states and degraded controls.
- `test_platform_joint_diagnostics_record.py` passes unmodified against the new l2 files: 164 passed with the churn suite. That includes the re-derivation of B1 and of the l2 lift from their own artifacts, and the AST ban on B1 targets.

## Deviations from Plan

**1. [Process] The served re-run ran during Task 1, not after Task 2.** It was independent of the l2 leg: it reads only the tracked checkpoints, copied to scratch. It ran while the joint-lift runs were in flight. Its numbers were recorded only after Task 2's controls and cuts passed.

**2. [Note] The regenerated l2 record carries 08-18's registry-block schema.** `adr_0002_ceiling` is gone; `declared_ceiling` and `declared_ceiling_source` are null; `ceiling_respected` is null. This is the run script's output for a `NO_REGISTRY` run, which has no ceiling to check. No test reads the l2 `adr_0002_ceiling`.

Otherwise the plan was executed as written. No source change and no registry row. No control failed, so no halt.

## Open question for Glenn

Glenn's keep-absolute threshold ruling (08-09) cited the belief clearing 0.70 in 307/488 (#1) and 415/488 (#2) months. On the CR-01-fixed path those counts are **273/488** and **408/488**. The ruling kept 0.70/0.40 and changed nothing, so no quantity moves. Does he want to revisit it on the new counts? This is not decided here.

## Standing qualifications

CR-02 (belief carried across refits with unaligned ids; `regime_belief` not fingerprinted at serve) and CR-03 (unmodelled publication lag in two L2 columns) qualify every L2 number above. Both are Phase 8.1. The vocabulary finding (§6) still qualifies S-1 and S-3.

## Known Stubs

None.

## Threat Flags

None. No new surface: no source change; the records are regenerated in their existing schema, with one added boolean per cut block.

## Self-Check: PASSED

- FOUND: all seven joint_lift artifacts, test_platform_nowcaster_recursion.py, the 08-MEASUREMENTS.md §11 and 08-SERVING.md §4 headings.
- FOUND commits: 324d2c2, 9fb9c9f.
- Registry 44 / c957e8fdb360.
