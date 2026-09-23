---
phase: 08-regime-persistence-stability
plan: 01
subsystem: platform/evaluation
tags: [criterion-0, criterion-2, criterion-9, churn, track-a, track-b, f-4, honesty-framework]
status: complete
requires:
  - "07-11 (run_joint_backtest's per_step_metrics_1/_2 meta keys; run_joint_lift.py; joint_lift_diagnostics.py)"
  - "d704f5d (T0.12: compute_sojourn_lag_headline raises on non-integer columns)"
provides:
  - "src/trading_crab_lib/platform/evaluation/churn.py: state_change_count, churn_rate, write_/read_probability_matrix, argmax_churn"
  - "outputs/reports/platform/joint_lift/joint_lift_probs_{1,2}_{l1only,l2}.parquet: the per-step probability matrices (ROADMAP criterion 0)"
  - "diagnostics_{l1only,l2_observational}.json: walk_forward_filtered (Track A) + walk_forward_nowcast (Track B) + series_identity"
  - "F-4 fixed: walk_forward_filtered.transition_rate = n_transitions / n_pairs"
affects:
  - "08-06..08-10: Track B is now the series a §5.1 change has to move. Track A cannot move on an L2 change."
  - "08-04: measured --collect-only count below"
  - "08-10: l1only decision-bearing leg is confirmed byte-reproducible; the l2 leg is not reproducible against the Sep-21 artifacts at the 1e-8 level"
decisions:
  - "Track B is computed ONLY from the persisted matrix. diagnose() raises if the matrix is missing and has no fallback to state_N, because under l1only that column IS the argmax (joint_driver.py:431)."
  - "Only the joint leg's matrices are written. The baseline leg differs only in blend_weight_1 and the suite already pins that both legs share the state path."
  - "test_recorded_rate_equals_count_over_steps keeps its name so it traces to the plan that moved it; the docstring says its body now divides by pairs."
tech-stack:
  added: []
  patterns: ["string state_{k} on disk, int64 in memory, raise on anything else", "identity pin on the degenerate routing + non-identity pin on the other"]
key-files:
  created:
    - src/trading_crab_lib/platform/evaluation/churn.py
    - tests/unit/test_platform_evaluation_churn.py
    - outputs/reports/platform/joint_lift/joint_lift_probs_1_l1only.parquet
    - outputs/reports/platform/joint_lift/joint_lift_probs_2_l1only.parquet
    - outputs/reports/platform/joint_lift/joint_lift_probs_1_l2.parquet
    - outputs/reports/platform/joint_lift/joint_lift_probs_2_l2.parquet
  modified:
    - scripts/run_joint_lift.py
    - scripts/joint_lift_diagnostics.py
    - tests/unit/test_platform_joint_diagnostics_record.py
    - outputs/reports/platform/joint_lift/diagnostics_l1only.json
    - outputs/reports/platform/joint_lift/diagnostics_l2_observational.json
metrics:
  duration: "about 2h of active work, plus about 1h45m of harness wall-clock across two sessions (the l2 leg took 44 min under sibling load)"
  completed: 2026-09-23
actuals:
  tokens: 9782   # chars/4 over the added lines of the realized text diff (parquets excluded)
  tasks: 3
  commits: 3
---

# 08-01: The probability matrix is on disk, and Track A and Track B are separate rows

## Tasks

| task | status | commit |
|---|---|---|
| 1 (tracer): `evaluation/churn.py`, 20 tests, `run_joint_lift.py` persists the matrices, four artifacts | complete | `3e271ff` |
| 2: `walk_forward_nowcast` + `series_identity` in `diagnose()`, 18 new test cases, both JSONs regenerated | complete | `992df99` |
| 3: F-4 fix. Source, moved pin, rejection test and regenerated JSONs in one commit | complete | **`f1c6b90`** |

Tracer gate: auto mode. After Task 1 the tracer's `<verify>` was re-run end to end on all
four matrices, not only the one the plan names, before any expansion task started. All passed.

## The four probability artifacts

Every artifact was written by the harness. None was patched by hand. Each file has one row per NON-degraded step.

| file | rows | degraded | rows + degraded | window (first → last row) |
|---|---|---|---|---|
| `joint_lift_probs_1_l1only.parquet` | 588 | 0 | 588 | 1972-01-31 → 2020-12-31 |
| `joint_lift_probs_2_l1only.parquet` | 588 | 0 | 588 | 1972-01-31 → 2020-12-31 |
| `joint_lift_probs_1_l2.parquet` | 488 | 100 | 588 | 1974-02-28 → 2020-12-31 |
| `joint_lift_probs_2_l2.parquet` | 488 | 100 | 588 | 1974-02-28 → 2020-12-31 |

The l2 matrices' index equals the curve's non-degraded dates exactly. All rows sum to 1 within 3e-16.
Columns: classifier #1 {0..5} (K=6), classifier #2 {0..4} (K=5).

## Both churn series, both classifiers, both routings

Both tracks are denominated in **adjacent pairs**.

| routing | classifier | Track A (`state_N`, L1) | Track B (argmax of posterior, L2) | series identical? |
|---|---|---|---|---|
| l1only | #1 | 246 / 587 = **41.91%** (588 steps, 1972-01-31 → 2020-12-31) | 246 / 587 = 41.91% (588 rows, 0 degraded) | **yes**: 0 of 588 mismatched |
| l1only | #2 | 24 / 587 = **4.09%** | 24 / 587 = 4.09% (588 rows, 0 degraded) | **yes**: 0 of 588 mismatched |
| l2 | #1 | 246 / 587 = 41.91% | **221 / 487 = 45.38%** (488 rows, **100 degraded**, 1974-02-28 → 2020-12-31) | **no**: 184 of 488 mismatched |
| l2 | #2 | 24 / 587 = 4.09% | **66 / 487 = 13.55%** (488 rows, 100 degraded) | **no**: 276 of 488 mismatched |

How to read the table:

- **Under l1only the two series are the same object.** This is a degeneracy, and it is pinned as one.
  Any measurement of "nowcast churn" on the decision-bearing routing is the L1 label churn under another name.
- **Track A is identical across routings (246 and 24),** which matches the S-4 routing-consistency signature.
- **Track B is measured for the first time. Under l2 it is higher than Track A for #1 (45.38% vs 41.91%) and
  about 3.3× higher for #2 (13.55% vs 4.09%).** Its window is 100 steps shorter.
- Context for F-2: the max probability is below 0.70 in 355 of 488 rows for #1 and in 476 of 488 rows for #2.
  The plan's branch-structure bound for #1 was ≥309, and the measured 355 is consistent with it.

## F-4

`walk_forward_filtered.transition_rate` now divides by `n_pairs = n_steps − 1` through `churn.churn_rate`.

| classifier | before (÷588 months) | after (÷587 pairs) |
|---|---|---|
| #1 | 0.418367 (41.84%) | **0.418980 (41.91%)** |
| #2 | 0.040816 (4.08%) | **0.040886 (4.09%)** |

The counts are unchanged (246 and 24). The JSON diff is `n_pairs` and `transition_rate` only.
**Existing pin moved:** `test_recorded_rate_equals_count_over_steps` now asserts `n_pairs == n_steps − 1` and
`rate == count / n_pairs`. The new `test_recorded_rate_is_not_the_month_denominated_value` rejects ÷588 at 1e-9.
It was observed failing 4 of 4 against the pre-fix record before the fix went in.
`test_filtered_churn_is_an_order_of_magnitude_above_full_sample` (rel=0.02 around 11.63) still passes because the ratio moved by 0.17%.

## Reproducibility

### l1only (decision-bearing): bit-reproducible

`--routing l1 --dry-run` re-wrote `joint_lift_{baseline,joint}_l1only.parquet`. Both are **byte-identical to HEAD**
(`cmp` clean; `git status` shows no modification), and every column is exactly equal. The dry-run KPI record matches
the committed `measurement_l1only.json` exactly in every field except the registry, dry-run and timestamp fields.
Criterion 7 reproduces exactly: `wealth_delta` −0.12343826162064975, `dd_delta` +0.02408401236666291.
Plan 08-10's 1e-12 decision-bearing equality is not threatened.

### l2 (observational): drift against HEAD, but reproducible run to run in this environment

This refines the orchestrator's finding.

- **Two l2 runs in this environment agree bit for bit.** Run 1 was Sep-22 16:31. Run 2 was Sep-23, after a
  container restart, and dumped to a scratch directory so no tracked curve was touched. Their probability
  matrices are `DataFrame.equals` True (max abs diff 0.0, 0 of 488 rows differ, both classifiers), and their KPI records
  are identical in every non-timestamp field.
- **Both runs differ from HEAD's l2 curves,** which were produced Sep-21 about 14:45 for `d5c3ac9`. Run 2 vs HEAD, max abs diff per column:

  | leg | return | turnover | cost | scale | active_regime / degraded / state_1 / state_2 |
  |---|---|---|---|---|---|
  | baseline | 8.73e-10 (527 rows) | **8.54e-08** (470) | 8.54e-11 (416) | 1.32e-09 (207) | exactly equal |
  | joint | 4.36e-10 (550 rows) | 4.27e-08 (486) | 4.27e-11 (427) | 4.20e-10 (203) | exactly equal |

  KPI-level drift: l2 `wealth_delta` −0.13450527951 (HEAD) vs −0.13450527919 (now), and `dd_delta` differs at ~1e-11.
- **Cause: not established.** No src/ or config/ commit between the two productions changes numerics.
  `42161dd` is control flow only (separate try blocks per classifier). The 08-03 commits touch `labeling/stability.py`,
  and `d704f5d` / `230c91c` touch modules outside the driver path. l1only, which runs through the same allocation
  arithmetic, is byte-identical, so the drift enters through the L2 nowcaster's probabilities. The hypothesis is an
  environment dependence (BLAS kernel or CPU) in the L2 solver path. It is **unverified**: HEAD never persisted
  l2 probabilities, so the drift cannot be localized to them directly.
- **Effect on Track B: none at the argmax level.** The minimum top-1 − top-2 probability gap is 0.00205 (#1)
  and 0.00223 (#2), which is 4–5 orders of magnitude above the observed drift. Argmax, churn counts and
  `series_identity` are robust to it. The HEAD l2 curves were left untouched: my scratch-dir rerun never wrote to
  the tracked directory, and the orchestrator had already restored the ones my first run overwrote.
- **Tolerance note for later plans:** any cross-run comparison of l2 float quantities needs a tolerance of about 1e-7 or wider
  against the Sep-21 artifacts. None of the checks in this plan compares l2 floats across runs.
  Every l2 check reads a single artifact or compares integer counts.

## Evidence that the discriminating checks can fail

- **Masquerade mutation.** Monkeypatching `diagnose()` to compute Track B from `state_N` turns l2
  `argmax_equals_state_elementwise` to True with 0 mismatches, and `n_rows`/`n_degraded` to 588/0. The l2 non-identity pin
  and the 488/100 denominator pin both fail, for both classifiers.
- **Missing matrix.** `diagnose()` on a directory without `joint_lift_probs_*` raises `FileNotFoundError`. The message
  names `python scripts/run_joint_lift.py --routing l2 --dump-curves …` and explains why there is no fallback.
- **Round trip.** `assert_frame_equal(..., check_column_type=True)` fails on an object-dtype column index. `read_probability_matrix`
  raises on `["a","b"]`, `["0","1"]` and `state_x`. A restored matrix is accepted by the T0.12 guard.
- **churn_rate.** Returns 246/587 and is asserted not to equal 246/588 at 1e-9. It raises for fewer than 2 rows.
- **Self-check footer.** `python -m trading_crab_lib.platform.evaluation.churn` caught my own hand-count: I wrote 5 changes and the
  sequence has 4. The comment was corrected. The code was already right.
- **The plan's line-scoped stale-string check** failed on my first reflow ("superseded" wrapped to the next line). I fixed the
  text. I did not change the check.

## Verification

| check | result |
|---|---|
| `pytest tests/unit/test_platform_evaluation_churn.py -q` | 20 passed |
| `pytest tests/unit/test_platform_joint_diagnostics_record.py -q` | 52 passed (30 before this plan, plus 18 from Task 2 and 4 from Task 3) |
| Task 1 shape verify (plan names l2 #1; run on all four) | rows + degraded == 588 for all four |
| Task 2 verify (l1only identity True, l2 False, Track A 246) | pass |
| Task 3 verify (`abs(rate−246/587)<1e-12 and abs(rate−246/588)>1e-9`) | pass, 41.91 |
| Task 3 stale-string verify | pass |
| `band_3a_ok` / `band_3b_ok`, both records, both classifiers | all True (unchanged) |
| `pytest tests/ -q` | **2150 passed**, 0 failed, 0 skipped |
| `pytest --collect-only -q \| tail -1` | **2150 tests collected** (tree at 2026-09-23 about 15:45, including sibling plans 08-02..08-05 tests; 42 of them are this plan's) |
| `total_trial_count()` | **42 before, 42 after** (both harness runs asserted `rows_added == 0`) |
| legacy-import ratchet (`MAX_LEGACY_IMPORT_SITES = 31`) | 11 passed; `churn.py` imports only `platform.evaluation.sojourn_lag` |
| 2021+ holdout | not read. All runs go through `split_by_holdout_boundary(..., DEFAULT_HOLDOUT_CUTOFF)` in `build_inputs`, and every artifact ends 2020-12-31 |
| ruff + flake8 (E9,F63,F7,F82) on every touched .py | clean |

## Deviations from Plan

1. **[Rule 3 – scope/coordination] The l2 curves were not committed, and they were not regenerated in the tracked directory.** My first l2
   run (session 1) overwrote `joint_lift_{baseline,joint}_l2.parquet`, which are not in `files_modified`. The orchestrator
   restored them to HEAD. The second l2 run dumped to a scratch directory, and was used only to measure drift. The committed
   `joint_lift_probs_*_l2.parquet` come from run 1, and run 2 reproduces them bit for bit.
2. **The Task 1 shape verify was run on all four matrices, not only `probs_1_l2`.** This is a wider check, not a changed one.
3. **Extra tests beyond the plan's list:** `test_track_b_is_re_derivable_from_its_own_named_artifact` recomputes Track B from
   the parquet the record names, and `test_each_block_names_its_own_track_in_the_record`. The l2 non-identity and denominator pins are
   parametrized over both classifiers, not only #1. That is justified because both classifiers measure non-identical under l2.
4. **`series_identity` carries three extra keys** beyond the two the plan specified: `n_compared`, `state_column` and `probs_source`.
5. **No `diagnose()` unit test for the missing-matrix raise.** `diagnose()` loads gitignored checkpoints (`data/`), so such a
   test would fail in CI. The raise was verified by running the script against a directory without the matrix (shown above).

## Found, not fixed (out of this plan's scope)

- **F-4's off-by-one also exists in `scripts/run_joint_lift.py::_leg_kpis`:** `state_{1,2}_transition_rate = count / max(1, n_steps)`.
  It feeds the committed `measurement_{l1only,l2_observational}.json` (`state_1_transition_rate` = 0.418367).
  Task 3 scoped the fix to the diagnostics record. Fixing this one means regenerating the measurement JSONs, and the l2 one
  would pick up the ~1e-8 drift above. It needs its own decision, so it is recorded here and not applied. After this plan, the
  diagnostics record and the measurement record state different rates for the same 246 changes.
- **l2 cross-environment drift** (above). Anyone planning to re-assert l2 float quantities at tight tolerance should know about it.

## Known Stubs

None.

## Threat Flags

None. No new network, auth or schema surface. The only new I/O is local parquet under `outputs/reports/`.

## Self-Check: PASSED

- FOUND: all six created files and five modified files listed in key-files
- FOUND: commits `3e271ff`, `992df99`, `f1c6b90` on `claude/keen-galileo-zqcml6-w5`
- Registry 42, unchanged. Not pushed.
