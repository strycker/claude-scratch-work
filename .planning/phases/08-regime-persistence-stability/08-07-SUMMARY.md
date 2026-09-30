---
phase: 08-regime-persistence-stability
plan: 07
subsystem: ml-platform
tags: [regime-stability, subsample-stability, hungarian-matching, criterion-3, honesty-framework, jump-model]

# Dependency graph
requires:
  - phase: 08-regime-persistence-stability
    provides: "08-03 labeling/stability.py — fit_for_stability, match_states, split_half_null,
      state_episodes, stability_row and the four scheme functions"
  - phase: 07-regime-representation
    provides: "the ten-column frozen policy (D-02-A), both re-pinned labelings (#1 K=6/λ=10.0,
      #2 K=5/λ=16.0) and their tracked checkpoints"
provides:
  - "scripts/run_subsample_stability.py — criterion-3 runner: in-process reference reproduction
    with an elementwise identity raise, four schemes, partner-keyed rows, both cost-matrix unit
    spaces, spawn-context worker pool, registry-unchanged assertion"
  - "outputs/reports/platform/stability/ — stability_record.json (77 rows),
    stability_rows.parquet (8,883 per-refit rows), stability_cost_matrices.parquet (98,526)"
  - "08-STABILITY.md — the prediction quoted from 2363752, the full results, the condition (i)
    verdict on classifier #1 state 0, and the caveats"
affects: [08 phase close, ADR-0001 exemption condition (i), ADR-0002 (classifier #2 single-episode states), labeling/stability.py (run_stability keying, distance units)]

# Actuals (#2632)
actuals:
  tokens: 25000   # chars/4 over the three authored text files (script 11.1k + tests 5.6k + record 8.1k); the generated JSON (36.1k) excluded
  tasks: 3
  commits: 4

tech-stack:
  added: []
  patterns:
    - "Reference identity as a raise, not a log line: the in-process refit is asserted elementwise
      against the tracked checkpoint before any subsample is fit, and the raise path is tested on
      a perturbed series and a shifted index."
    - "Partner keying: every per-state quantity on a row (occupancy, null, episodes) is read off
      the Hungarian-matched partner, with a synthetic non-identity test that fails under same-id
      keying and a pin that reproduces run_stability exactly under identity."
    - "Tables generated from the artifact: 08-STABILITY.md's result tables are filled from the
      committed JSON, not transcribed."
    - "Determinism checked by re-run: two full runs compared frame-equal in all three artifacts."

key-files:
  created:
    - scripts/run_subsample_stability.py
    - tests/unit/test_platform_subsample_stability_record.py
    - outputs/reports/platform/stability/stability_record.json
    - outputs/reports/platform/stability/stability_rows.parquet
    - outputs/reports/platform/stability/stability_cost_matrices.parquet
    - .planning/phases/08-regime-persistence-stability/08-STABILITY.md
  modified: []

key-decisions:
  - "Occupancy, the split-half null and episode counts are keyed on the matched PARTNER, not on
    the same-id subsample state as stability.run_stability does. Classifier #1's assignment is
    non-identity under every contiguous scheme and in 97-100% of bootstrap replicates, so
    same-id keying would have put another state's occupancy beside nearly every distance, and
    could mask an evaporated partner."
  - "The holdout is carved BEFORE deriving anything. run_joint_lift.build_inputs computes
    classifier #2's relative features and asset returns on un-carved monthly_raw; the runner calls
    the same functions in the same order after the carve. Elementwise identity (696/0) proves it
    changed nothing."
  - "A reference-SD companion distance is reported beside 08-03's winsorized-unit distance,
    never replacing it: the winsorized distance is 98.7% oil+CAPE for #1 and 100%
    rs_equities_bonds for #2. No verdict depends on either distance."
  - "Condition (i) is evaluated only on calendar-order schemes (both decade drops, LOO). On
    bootstrap rows it carries holds=None with the reason: a resampled series has no temporal
    separation and its episode count counts seams."
  - "Bootstrap rows are summarised over replicates in the JSON (distance/null over
    non-evaporated replicates; evaporated = any replicate evaporated, count carried); every
    replicate is kept in the parquet."

requirements-completed: [PER-06]

coverage:
  - id: D1
    description: "Both reference labelings reproduce their tracked checkpoints elementwise
      (695/6/0 mismatches; 696/5/0) via a raise whose failure path is tested."
    requirement: PER-06
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_subsample_stability_record.py::TestReferenceIdentity, ::TestMismatchPathRaises, ::TestFrozenListIdentity"
        status: pass
      - kind: other
        ref: "mutation: the mismatch raise replaced with log.warning -> test_a_single_perturbed_month_raises_with_the_count fails"
        status: pass
    human_judgment: false
  - id: D2
    description: "Rows are keyed on the matched partner; an evaporated partner is flagged under a
      non-identity assignment; identical to run_stability under identity."
    requirement: PER-06
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_subsample_stability_record.py::TestPartnerKeying (3 tests)"
        status: pass
      - kind: other
        ref: "mutation: occupancy/null keyed on the same id -> 2 of 3 fail (the identity pin passes, as it must)"
        status: pass
    human_judgment: false
  - id: D3
    description: "All four schemes ran for both classifiers; exact row count (77) and per-refit
      coverage asserted; every row carries occupancy, own-n null, margin, episodes, evaporated;
      the persisted cost matrix is cross-checked against every row's matched distance."
    requirement: PER-06
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_subsample_stability_record.py::TestArtifact* (16 tests)"
        status: pass
      - kind: other
        ref: "plan Task 2 verify one-liners (coverage/occupancy/null/state-2 degeneracy; no threshold key)"
        status: pass
    human_judgment: false
  - id: D4
    description: "08-STABILITY.md quotes the 2363752 prediction ahead of the results, reads every
      distance against its own-n null, renders only condition (i), and states the caveats."
    requirement: PER-06
    verification:
      - kind: other
        ref: "plan Task 3 verify blocks (string presence; state-0 episode counts in the record)"
        status: pass
    human_judgment: true

duration: ~24min (first commit 18:29, last task commit 18:41; excludes reading and the ~4-min runs)
completed: 2026-09-23
status: complete
---

# Phase 8 Plan 07: §4.4 Criterion 3 Run for Both Classifiers Summary

**Criterion 3 has now been run on both classifiers, against references reproduced elementwise
from the tracked checkpoints. The one verdict this phase may render is that classifier #1's
crisis state satisfies AMENDMENT condition (i), with 5, 9 and 9 episodes. The run also turned up
four things nobody predicted. Classifier #1's canonical ordering flips under every subsample.
Classifier #2 has two single-episode states. The evaporation flag never fired, even where every
month of a state had been removed. And the 08-03 distance is nearly one-dimensional.**

## Performance

- **Tasks:** 3/3. **Commits:** 4 (3 task + this summary).
- **Refits:** 1,615 (808 for #1, 807 for #2) in 166 s and then 231 s across two runs, on 4 spawn
  workers. The two runs are bit-identical.
- **Fit cost:** measured at ~0.3–0.5 s per fit, not the 0.111 s §5.7 assumed. Criterion 3 is still
  cheap.
- **Tests:** 31 in the new file, all passing, 0 skipped. Full suite: **2212 passed, 0 skipped, 0
  failed** (197.6 s).

## Task Commits

1. **Task 1 (tracer): reproduce both references; harness end to end.** `0f30661` (feat). The tracer
   gate was re-run green before any expansion work. Auto-advance is off, but the plan is
   `autonomous: true` and the orchestrator asked for the full run, so no interactive checkpoint was
   raised.
2. **Task 2: four schemes, both classifiers, artifacts plus full cost matrices.** `d47d2de` (feat).
3. **Task 3: 08-STABILITY.md.** `65f4c2e` (docs).

## Reference Identity (asserted before any subsample)

| classifier | checkpoint | months | states | window | mismatches |
|---|---|---|---|---|---|
| #1 (K=6, λ=10.0, 10 frozen cols, `trailing_return_1m`) | `regime_labels` | 695 | 6 | 1963-02-28 → 2020-12-31 | **0** |
| #2 (K=5, λ=16.0, 8 config cols, `rs_equities_bonds`) | `regime_labels_2` | 696 | 5 | 1963-01-31 → 2020-12-31 | **0** |

## Results

`dist` is 08-03's distance: winsorized units, beside the within-state split-half null at that
row's **own** subsample n. `rSD` is the reference-SD companion (see Deviation 3). Occupancy is the
matched partner's. **Evaporated = False on all 8,883 per-refit rows.** Full per-row tables, with
null p10/p90, margins and overlap counts, are in `08-STABILITY.md` §4.

### Contiguous schemes

| # | scheme | st | ptr | dist / null (n) | rSD dist / null | occ (of sub) | eps | ref-in-sub / overlap | evap |
|---|---|---|---|---|---|---|---|---|---|
| 1 | drop_first | 0 | 0 | 4.840 / 5.595 (21) | 1.818 / 1.605 | 21/575 | 5 | 37 / 18 | F |
| 1 | drop_first | 1 | 3 | 10.222 / 1.210 (148) | 1.076 / 0.350 | 148/575 | 6 | 118 / 96 | F |
| 1 | drop_first | 2 | 4 | 19.981 / 2.010 (180) | 1.303 / 0.318 | 180/575 | 4 | 71 / 62 | F |
| 1 | drop_first | 3 | 2 | 27.158 / 1.690 (75) | 1.261 / 0.465 | 75/575 | 2 | 200 / 70 | F |
| 1 | drop_first | 4 | 5 | 1.769 / 1.957 (69) | 0.345 / 0.555 | 69/575 | 2 | 84 / 69 | F |
| 1 | drop_first | 5 | 1 | 1.567 / 1.794 (82) | 0.903 / 0.635 | 82/575 | 2 | 65 / 58 | F |
| 1 | drop_last | 0 | 0 | 3.142 / 5.890 (25) | 1.185 / 1.336 | 25/575 | 9 | 34 / 24 | F |
| 1 | drop_last | 1 | 1 | 0.502 / 0.906 (245) | 0.173 / 0.254 | 245/575 | 5 | 228 / 223 | F |
| 1 | drop_last | 2 | 3 | 12.590 / 2.331 (133) | 0.903 / 0.411 | 133/575 | 3 | 71 / 70 | F |
| 1 | drop_last | 3 | 2 | 17.808 / 3.819 (35) | 1.685 / 0.914 | 35/575 | 2 | 86 / 28 | F |
| 1 | drop_last | 4 | 5 | 1.712 / 1.335 (72) | 0.253 / 0.555 | 72/575 | 1 | 84 / 72 | F |
| 1 | drop_last | 5 | 4 | 1.289 / 2.024 (65) | 0.529 / 0.680 | 65/575 | 3 | 72 / 59 | F |
| 1 | LOO s0 (1973-11→1974-09, 11) | 0 | 0 | 12.076 / 5.981 (34) | 1.232 / 1.114 | 34/684 | 9 | 29 / 28 | F |
| 1 | LOO s1 (1963-02→1970-04, 87) | 1 | 2 | 5.051 / 1.080 (156) | 0.810 / 0.323 | 156/608 | 3 | 141 / 126 | F |
| 1 | **LOO s2 (1996-07→2002-05, 71) DEGENERATE** | 2 | 2 | 19.444 / 1.218 (107) | 3.118 / 0.339 | 107/624 | 5 | **0 / 0** | F |
| 1 | LOO s3 (2011-11→2020-02, 100) | 3 | 4 | 4.692 / 3.078 (87) | 0.511 / 0.533 | 87/595 | 2 | 100 / 78 | F |
| 1 | LOO s4 (1981-10→1987-09, 72) | 4 | 5 | 8.210 / 4.267 (41) | 2.884 / 0.823 | 41/623 | 5 | 12 / 4 | F |
| 1 | LOO s5 (1978-04→1981-08, 41) | 5 | 3 | 9.411 / 1.057 (133) | 2.669 / 0.331 | 133/654 | 6 | 31 / 23 | F |
| 2 | drop_first | 0 | 0 | 186.651 / 110.231 (24) | 3.124 / 0.751 | 24/576 | 2 | 116 / 24 | F |
| 2 | drop_first | 1 | 1 | 40.749 / 69.097 (107) | 1.805 / 0.364 | 107/576 | 4 | 50 / 12 | F |
| 2 | drop_first | 2 | 2 | 72.778 / 118.445 (179) | 0.474 / 0.268 | 179/576 | 2 | 144 / 141 | F |
| 2 | drop_first | 3 | 3 | 0.000 / 171.168 (166) | 0.005 / 0.352 | 166/576 | 1 | 166 / 166 | F |
| 2 | drop_first | 4 | 4 | 4.941 / 209.580 (100) | 0.009 / 0.349 | 100/576 | 1 | 100 / 100 | F |
| 2 | drop_last | 0 | 0 | 219.012 / 129.697 (24) | 3.124 / 0.751 | 24/576 | 2 | 116 / 24 | F |
| 2 | drop_last | 1 | 1 | 218.369 / 40.939 (227) | 0.902 / 0.225 | 227/576 | 4 | 158 / 120 | F |
| 2 | drop_last | 2 | 2 | 1015.081 / 46.217 (38) | 1.903 / 0.556 | 38/576 | 2 | 156 / 37 | F |
| 2 | drop_last | 3 | 3 | 1504.034 / 134.244 (139) | 2.111 / 0.284 | 139/576 | 1 | 146 / 0 | F |
| 2 | drop_last | 4 | 4 | 2284.632 / 159.631 (148) | 2.081 / 0.349 | 148/576 | 1 | **0 / 0** | F |
| 2 | LOO s0 (1974-01→1982-08, 104) | 0 | 0 | 96.047 / 45.779 (151) | 2.259 / 0.209 | 151/592 | 2 | 12 / 0 | F |
| 2 | LOO s1 (1963-01→1970-11, 95) | 1 | 1 | 1200.689 / 109.830 (225) | 1.738 / 0.249 | 225/601 | 2 | 63 / 58 | F |
| 2 | LOO s2 (1989-05→1994-02, 58) | 2 | 2 | 166.770 / 277.011 (70) | 0.918 / 0.405 | 70/638 | 4 | 98 / 70 | F |
| 2 | **LOO s3 (1998-11→2012-08, 166) DEGENERATE** | 3 | 3 | 1812.304 / 127.037 (158) | 2.735 / 0.273 | 158/530 | 5 | **0 / 0** | F |
| 2 | **LOO s4 (2012-09→2020-12, 100) DEGENERATE** | 4 | 4 | 2435.057 / 146.242 (168) | 2.000 / 0.341 | 168/596 | 1 | **0 / 0** | F |

Assignment identity: classifier #1 is **N on every contiguous row**. Classifier #2 is Y except
LOO s4.

### Circular block bootstrap (200 replicates each; medians; evaporated replicates = 0 everywhere)

| # | L | st | id frac | dist med / null med | rSD dist / null | occ med (min) | eps med |
|---|---|---|---|---|---|---|---|
| 1 | 6 | 0 / 1 / 2 / 3 / 4 / 5 | 0.01 | 8.206/5.233 · 2.249/0.984 · 8.844/1.830 · 6.638/2.329 · 3.541/1.567 · 4.750/1.989 | 2.240/1.147 · 0.419/0.250 · 1.220/0.417 · 0.495/0.315 · 0.838/0.529 · 1.328/0.631 | 33 (13) · 224.5 (30) · 103 (19) · 161.5 (19) · 78 (18) · 69.5 (16) | 10 · 23 · 13.5 · 20 · 12 · 11 |
| 1 | 12 | 0..5 | 0.01 | 9.013/4.903 · 2.537/0.928 · 8.412/1.675 · 5.187/2.249 · 4.410/1.386 · 6.469/1.762 | 2.538/1.171 · 0.573/0.254 · 1.128/0.392 · 0.563/0.301 · 0.910/0.498 · 1.719/0.624 | 33.5 (11) · 202.5 (46) · 108 (13) · 164 (30) · 84 (28) · 70 (22) | 8 · 13 · 9 · 12 · 7 · 7 |
| 1 | 24 | 0..5 | 0.00 | 9.834/4.611 · 3.642/0.919 · 6.792/1.490 · 7.373/2.174 · 5.040/1.206 · 6.669/1.673 | 2.635/1.109 · 0.772/0.259 · 0.990/0.409 · 0.586/0.304 · 1.004/0.468 · 1.780/0.585 | 35 (12) · 181 (43) · 104.5 (23) · 161 (36) · 88 (22) · 76.5 (23) | 8 · 8 · 6 · 7 · 5 · 6 |
| 1 | 48 | 0..5 | 0.03 | 10.222/4.387 · 4.814/0.863 · 7.428/1.459 · 11.245/2.118 · 5.600/1.237 · 6.248/1.748 | 2.675/1.022 · 0.799/0.257 · 1.181/0.380 · 0.747/0.309 · 1.101/0.478 · 1.898/0.587 | 37.5 (10) · 175.5 (32) · 120.5 (14) · 150.5 (19) · 93 (17) · 77 (11) | 7 · 6 · 4 · 6 · 5 · 5 |
| 2 | 6 | 0..4 | 0.86 | 202.0/91.0 · 250.4/80.9 · 458.3/135.7 · 699.1/187.4 · 931.2/210.9 | 2.629/0.612 · 0.850/0.240 · 0.862/0.320 · 1.407/0.361 · 1.064/0.323 | 41 (8) · 220.5 (16) · 121.5 (24) · 125 (19) · 144.5 (30) | 7 · 18 · 13 · 13 · 16 |
| 2 | 12 | 0..4 | 0.85 | 218.7/73.7 · 167.7/63.8 · 586.6/128.8 · 778.5/161.9 · 1099.3/190.6 | 2.529/0.508 · 0.884/0.238 · 0.813/0.307 · 1.520/0.333 · 1.123/0.307 | 49.5 (9) · 199 (22) · 122 (30) · 131 (22) · 166.5 (29) | 5 · 12 · 9 · 8 · 10 |
| 2 | 24 | 0..4 | 0.83 | 209.75/66.2 · 230.5/60.2 · 651.7/110.7 · 574.3/153.9 · 1030.3/189.9 | 2.001/0.415 · 1.155/0.246 · 1.024/0.306 · 1.328/0.320 · 1.018/0.311 | 80.5 (12) · 166 (17) · 120.5 (12) · 135 (16) · 144.5 (32) | 4 · 7 · 6 · 5 · 6 |
| 2 | 48 | 0..4 | 0.70 | 200.0/61.6 · 177.4/65.8 · 719.6/102.3 · 773.6/151.5 · 1140.2/177.8 | 2.185/0.415 · 1.579/0.267 · 1.144/0.301 · 1.471/0.320 · 1.045/0.311 | 82 (9) · 148 (19) · 124 (14) · 144.5 (12) · 142 (15) | 3 · 5 · 5 · 4 · 4 |

### Seam counts per bootstrap replicate (median [min, max])

| L | #1 | #2 |
|---|---|---|
| 6 | 116 [114, 119] | 116 [113, 119] |
| 12 | 58 [56, 61] | 58 [56, 61] |
| 24 | 29 [27, 33] | 29 [27, 33] |
| 48 | 15 [13, 18] | 15 [13, 18] |

Politis–White anchor n^(1/3) = 8.86 months. It is quoted and not obeyed.

## Condition (i) Verdict — Classifier #1 State 0

**SATISFIED under criterion 3 applied directly.** The matched partner's episodes in the subsample
were 5 under drop-first-decade (21/575 months), **9** under drop-last-decade (25/575), and **9**
under LOO of its longest episode (1973-11 → 1974-09, 11 months; 34/684). The state did not
evaporate under any scheme or replicate; its minimum bootstrap occupancy was 10 months (L=48).
Condition (i) is **not** in question, so **no sentence needs to be carried into
`.planning/STATE.md`'s open items under D-05.**

## Did the Pre-Registered Prediction (2363752) Hold?

- **State 0: the verdict held, the arithmetic did not.** It predicted "six to seven" episodes;
  the measured counts were 5 and 9. The refit is not the reference minus the dropped months.
  Under drop-first, 37 reference-crisis months remain, but the partner holds 21, of which 18
  overlap.
- **State 2: the structural prediction held exactly.** Neither decade drop touches it
  (ref-in-sub 71/71 both times). LOO is degenerate: 71 months dropped over 1996-07 → 2002-05, and
  the refit's slot 2 is built entirely from other months (overlap 0), at 19.44 against a null of
  1.22. Per the plan, the degeneracy is the answer. Whether state 2 "fails" was **not** declared
  (08-STABILITY.md §7).

## Deviations from Plan

**1. [Rule 2 - Correctness] Rows keyed on the matched partner, not the same-id state.**
- **Found during:** Task 1, on the first real scheme. Classifier #1's drop-first-decade
  assignment was non-identity.
- **Issue:** `stability.run_stability` reads `occupancy[state]`, the null on `sub_states == state`
  and `episodes[state]` using the **reference** state id, while `matched_distance` belongs to the
  partner. Under a non-identity assignment each row mixes two states. That can hide an evaporated
  partner, which is Trap B through a side door.
- **Fix:** the runner composes 08-03's public building blocks (`fit_for_stability`,
  `match_states`, `split_half_null`, `state_episodes`, `stability_row`) keyed on the partner, and
  imports the private `_check_frozen_columns` for Trap C at the same three points.
- **Tests:**
  - A synthetic non-identity test fails under same-id keying (mutation-checked).
  - A pin test shows the runner reproduces `run_stability`'s rows field by field under identity.
- **`stability.py` was NOT modified.** The defect is **reported, not fixed**. `run_stability`
  remains wrong under non-identity for any other caller. Recommend a follow-up that keys it on
  the partner.
- **Commit:** `0f30661`.

**2. [Rule 3 / CRITICAL constraint] Holdout carved first rather than reusing `build_inputs`
verbatim.** The plan says to reuse `run_joint_lift.build_inputs`. That function computes
`add_relative_features` and the asset returns on **un-carved** `monthly_raw`, which reads
post-2020 rows. `build_frames` calls the same functions, with the same arguments and in the same
order, after `split_by_holdout_boundary` on both frames. Both identities still reproduce with 0
mismatches, which proves the carve changed nothing. Commit `0f30661`.

**3. [Rule 2 - Correctness] Reference-SD companion distance added; primary unchanged.** Found
during Task 2 from the magnitude of classifier #2's distances (hundreds to thousands) and then
quantified structurally on the references:
- the winsorized-unit distance is **79.5% oil + 19.2% CAPE** for #1, with 7 of 10 columns
  (including every crisis-defining one) at ≤ 0.1%;
- it is **100% `rs_equities_bonds`** for #2;
- §5.2 benchmarked centroid distance on unit-scale columns.

The primary is reported as built. The companion divides both sides by the reference fit's SD,
with no refit, and is reported beside it along with its own Hungarian partner, its null and its
cost matrix. It was not substituted, because switching metrics after seeing numbers is the
forbidden shape. No verdict depends on either distance. Whether 08-03's distance should change is
flagged for a later plan. Commit `d47d2de`.

**4. [Rule 3 - Blocking] Worker pool uses spawn, not fork.** The first run deadlocked: 4 workers at
0% CPU, because they were forked after the BLAS/OpenMP threads existed. The fix is
`mp_context=spawn` with BLAS pinned to one thread in the children. Commit `0f30661`.

**5. [Scope interpretation] Condition (i) is not evaluated on bootstrap rows.** Those rows carry
`holds: null` and the reason. The plan asks for the field on classifier #1 state 0 rows. A
block-resampled series is not in calendar order, so "temporally separated episodes" is undefined
on it, and its episode count counts seams. Computing a pass/fail there would apply the condition
to a quantity it does not describe.

**6. [Plan-number correction] Runtime.** The plan and §5.7 assumed 0.111 s per fit, about 3
minutes serial. Measured fits take ~0.3–0.5 s; the parallel run took 166–231 s. The conclusion
is unchanged: it is cheap.

**7. [Own-test catch] `no_verdict_note` renamed to `scope_note`.** My own key scan, which
includes "verdict", caught a prose-note key I had added. The fix was a re-run, not a hand edit.
That re-run doubled as the determinism check: all three artifacts were frame-equal across both
runs.

No architectural change, no package install, no `legacy/` or submodule edit, no post-2020 data
read, no secret written, and no STATE/ROADMAP/REQUIREMENTS edit (the orchestrator owns those).

## Findings for the Orchestrator (not deviations; recorded, no verdict)

1. **The evaporation flag never fired in 8,883 rows, including where it arguably should matter.**
   Classifier #2 state 4 loses **all** of its months under drop-last-decade. Three LOOs (#1 s2,
   #2 s3, #2 s4) remove all of a state's months. In all of these the K-fixed refit fills the slot
   with other months, so `evaporated = False`. `reference_months_in_subsample` and
   `partner_overlap_months` (0 / 0) carry that reading instead. **The flag is necessary but not
   sufficient.** This is the "can only confirm" shape, and it is documented in 08-STABILITY.md §4.4.
2. **Classifier #1's canonical ordering is unstable.** The assignment is non-identity under every
   contiguous scheme and in 97–100% of bootstrap replicates. Any id-keyed downstream number
   (profiles, lift, occupancy by id) inherits this.
3. **Classifier #2 has two single-episode states.** State 3 is 1998-11 → 2012-08 (166 months) and
   state 4 is 2012-09 → 2020-12 (100 months). ADR-0002's λ re-pin comment says states recur at
   2n. Three of five do.
4. **`run_stability`'s same-id keying** (Deviation 1) and **the scale-dominated distance**
   (Deviation 3) are defects or limitations in 08-03's module. Both are reported here, not fixed.

## Verification Results

```
# Task 1
pytest tests/unit/test_platform_subsample_stability_record.py -q -k "reference or frozen or mismatch"
=> 9 passed, 3 deselected
python scripts/run_subsample_stability.py --classifier 1 --schemes drop_first_decade --out-dir /tmp/stability_smoke
=> exit 0; 6 rows; both identities 0 mismatches; registry 42 -> 42

# Task 2
pytest tests/unit/test_platform_subsample_stability_record.py -q       => 31 passed
<plan one-liner: coverage/occupancy/null/state-2 degeneracy>           => rows 77; all present
<plan one-liner: threshold-shaped keys>                                 => none

# Task 3
<plan string-presence check>                                            => ok
<plan state-0 episode counts in record>                                 => present

# Phase verification
pytest tests/unit/test_platform_subsample_stability_record.py -q       => 31 passed, 0 skipped
pytest tests/ -q                                                        => 2212 passed, 0 skipped, 0 failed (197.6 s)
total_trial_count()                                                     => 42 (before and after; also asserted in-run)
pytest tests/unit/test_platform_legacy_import_ratchet.py -q            => 11 passed (ratchet 31)
ruff check / flake8 --select=E9,F63,F7,F82 on both .py files            => clean
artifacts under outputs/reports/platform/stability/                     => committed, not gitignored
```

Mutation checks: same-id keying makes 2 of 3 partner tests fail, and a log-instead-of-raise
identity check makes the mismatch test fail.

## Known Stubs

None.

## Threat Flags

None. No new network endpoint, auth path, or trust-boundary schema. The runner reads tracked
checkpoints and writes three report artifacts.

---
*Phase: 08-regime-persistence-stability*
*Completed: 2026-09-23*

## Self-Check: PASSED
