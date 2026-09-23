---
phase: 08-regime-persistence-stability
plan: 08
created: 2026-09-23
status: complete — halted by S-1 (3 LEADs, #2), ruled 2026-09-23 (causal invariance governs), all 4 negative offsets adjudicated bit-identical, Task 3 finished
---

# 08-CHURN — Bayes filter wired; S-1 halted, the ruling adjudicated, the three churn numbers measured

> **Decision-bearing caveat, first.** The Bayes filter changes nothing on the
> decision-bearing `ROUTING_L1_ONLY`: there `probs` is a one-hot on the L1 label, which is
> not a likelihood, and the filter is literally gated off (`if routing == ROUTING_L2_NOWCAST
> and use_regime_filter:`). Everything below about the belief is on the **observational,
> firewalled l2 leg**, and per ADR-0002 decision (e) **nothing downstream may change on the
> basis of it**. That includes the halt and every number in §7–§8. None of it moves criterion 7,
> which was re-measured byte-identical (§1).

## 0. The halt, in one paragraph (historical — resolved by §6)

The S-1 signed-offset guard was run on the REAL filtered belief paths
(`joint_lift_belief_{1,2}_l2.parquet`) against the full-sample reference labels. Every
strictly negative offset was then classified by `classify_negative_offsets`, the
held-through-return rule that was pre-registered in `08-08-PLAN.md` (AMENDED 2026-09-23)
before any real 08-08 number existed. The rule was pinned against 08-06's synthetic arms
first, and all four prototype results reproduced. **Classifier #2 has 3 LEADs. Classifier
#1 has 0 LEADs and 1 held-through miss.** One LEAD halts the plan (T-08-40). As
instructed:
- nothing was registered;
- **no B1 churn number was computed or recorded;**
- `diagnostics_l2_observational.json` was not regenerated;
- the rule's three clauses were not touched (T-08-40b).

The finding is recorded below for a human decision. **Resolved:** Glenn ruled the same day that
causal invariance governs (§6). All four negative offsets were adjudicated bit-identical, and
Task 3 then ran to completion (§7, §8).

## 1. Invariants — all held, all checked before the guard was read

| invariant | result | how measured |
|---|---|---|
| l1only curve, filter ON vs git | **byte-identical** (`cmp` clean) for `joint_lift_{baseline,joint}_l1only.parquet` and `joint_lift_probs_{1,2}_l1only.parquet`; `assert_frame_equal(check_exact=True)` | the real `--routing l1 --dry-run` run, dumped to scratch and compared against `git show 6a88638:<path>` (the pre-change tracked file) |
| l1only curve, filter OFF vs git and vs ON | **byte-identical**, all four files, both comparisons | same run with `run_joint_backtest` wrapped by `functools.partial(use_regime_filter=False)` |
| criterion 7 (l1only, 588 steps 1972-01-31 → 2020-12-31) | `wealth_delta` **−0.12343826162064975**, `dd_delta` **+0.02408401236666291**, identical in both runs | the dry-run KPI record |
| Track A (`state_N` changes) | **246 / 587** (#1) and **24 / 587** (#2) under **both** routings. The l2 run's `state_1`, `state_2` and `degraded` columns are elementwise equal to the tracked l2 curves, both legs | new l2 curves vs tracked |
| B0 (raw-posterior argmax churn) — the control | **221 / 487** (#1) and **66 / 487** (#2), 488 rows, 100 degraded, 1974-02-28 → 2020-12-31. Same as 08-01-SUMMARY.md. The raw posterior matrices are **byte-identical** to 08-01's committed `joint_lift_probs_{1,2}_l2.parquet` (max abs diff 0.0) | new vs tracked matrices |
| degraded steps (l2) | **100 of 588**, unchanged; the belief matrix has 488 rows, the same index as the posterior's | curve `degraded` column; belief row count |
| registry | **42 before, 42 after** every run (`rows_added == 0`, `NO_REGISTRY`) | `total_trial_count()` and each run's record |

**S-4.** Track A is 246 before and after. That is the **expected** result under the
corrected causal model (08-CONTEXT.md AMENDMENT), not a failure of the fix: the filter
acts on L2's posterior, and L1's terminal-month label cannot move because of it.

**The control paragraph.** B0's invariance is what would make B1 attributable. The raw
posterior is byte-identical before and after, so any difference between B1 and B0 is
caused by the filter and by nothing else. Had B0 moved, the nowcaster (or its inputs)
would have changed, and a B1 movement could not have been told apart from a nowcaster
change. That is the unattributed-number defect this phase's AMENDMENT records. Here B0
did not move, so the attribution is clean. The halt is therefore unambiguous: the three
negative offsets below are properties of the **filter's** output. The raw posterior shows
none (§3).

## 2. The finding — S-1 on the real belief path

Reference: the full-sample `regime_labels` / `regime_labels_2` checkpoints, split at
2020-12-31 (#1: 695 months 1963-02-28 → 2020-12-31; #2: 696 months 1963-01-31 →
2020-12-31). The belief covers 488 non-degraded steps, 1974-02-28 → 2020-12-31.
`act_threshold` = 0.70 (config).

| clf | transitions | resolved | min offset | median offset | n_negative | **n_lead** | n_held_through_miss |
|---|---|---|---|---|---|---|---|
| #1 | 25 | 20 | −6 | 72.5 | 1 | **0** | 1 (position 300) |
| #2 | 12 | 11 | −31 | 44.0 | 3 | **3** (positions 236, 387, 596) | 0 |

### The three LEADs (classifier #2) — positions, offsets, and the reference runs they span

| position | reference transition | offset | into `s` | preceding reference run `r`, span `[q, p−1]` | month `q−1` | clause that fails |
|---|---|---|---|---|---|---|
| **236** | 1982-09-30 | **−1** | 2 | state 0, positions 132–235 (1974-01 → 1982-08, 104 months) | state 1 | **(ii)**: `1 → 0 → 2` is not a return |
| **387** | 1995-04-30 | **−12** | 2 | state 1, positions 374–386 (1994-03 → 1995-03, 13 months) | state 2 (a return) | **(iii)**: `belief[2]` = 0.468 at q−1 (1994-02) and 0.659 at q (1994-03), below 0.70 |
| **596** | 2012-09-30 | **−31** | 4 | state 3, positions 430–595 (1998-11 → 2012-08, 166 months) | state 2 | **(ii)**: `2 → 3 → 4` is not a return |

Belief (B) and raw posterior (P) around each lead, as measured:

- **236.** 1982-05 and 1982-06 are degraded (NaN), so the backward walk stops there. B[2] was
  0.299 at 1982-07, **0.752 at 1982-08** and 0.919 at 1982-09; P[2] was 0.269, 0.663, 0.662.
  The belief crossed one month before the reference switch. The raw posterior never crossed.
- **387.** B[2] climbed from 0.205 (1993-12) to 0.659 (1994-03), then reached **0.826 at
  1994-04** and stayed 0.89–0.99 through 1995-04. B[1] stayed ≤ 0.025 throughout.
  P[2] was 0.36–0.48 and P[1] 0.01–0.09 over the same months. The belief never registered
  the reference's 13-month state-1 run. It came within one clause of the held-through
  pattern, and failed that clause because it was still *rising* into state 2 at q−1.
- **596.** B[4] crossed **0.713 at 2010-02** and stayed 0.83–0.99 until the reference
  switched in 2012-09. P[4] was 0.33–0.46 (a plurality) and P[3] 0.18–0.40 over the same
  span.

### The one held-through miss (classifier #1) — reported, not dropped

Position **300**, 1988-02-29, offset **−6**, into state 4. The preceding reference run is
state 0 over positions 296–299 (1987-10 → 1988-01, the October 1987 crash), and q−1
(1987-09) is state 4. B[4] = 0.88–0.96 and B[0] ≤ 0.01 across [1987-09, 1988-01]: the
belief held state 4 straight through the 4-month state-0 run and never registered it. This
is the exact pattern 08-06's caveat test measured on synthetic data. Every other #1 offset
is ≥ +1. The minimum over non-miss transitions is 1.0 (2008-09).

## 3. Context for the human decision — measured, and NOT a reclassification

None of the following changes any verdict above. Changing the rule after seeing these
numbers is exactly T-08-40b, and it was not done.

1. **The raw posterior shows no negative offset for either classifier.** On the same
   reference, B0's matrix gives min offset **+3.0** (#1) and **+110.0** (#2), n_negative 0.
   The LEADs appear only after filtering.
2. **The pre-existing decision-bearing L1-only one-hot (Track A) for #2 already "leads"
   at position 596, by −264 months** (the same rule classifies it a LEAD; #1's L1 one-hot
   is clean, min +1.0, as 08-06 recorded). The walk-forward L1 labeler for #2 read state 4
   at its terminal month in every one of the 264 months 1990-09 → 2012-08 (measured). The
   full-sample reference has state 3 for 1998-11 → 2012-08 and places the 3 → 4 switch at
   2012-09. That predates this plan and was never checked for #2.
3. **The belief amplifies a persistent raw plurality.** At 387 and 596 the raw posterior
   put a steady 0.35–0.48 on the incoming state for months. The filter's `A` and prior come
   from the step's in-window L1 labels, and at 596 those labels already favour state 4
   (item 2). Compounding one month of evidence per step lifts the belief over 0.70.
4. **Candidate readings, in no order, for a human to adjudicate:**
   - **(a) A genuine leak.** No code path that carries post-*t* data into the belief is
     known. `A`, the prior and the posterior are all built from `train_index` (strictly
     before *t*) plus the feature row at *t*. The rule's premise is that a lead is proof
     of one, and the rule was pre-registered, so it stands.
   - **(b) Reference timing.** S-1's premise is that "a causal filter cannot lead"
     the reference switch. That premise assumes the full-sample smoothed reference places
     each switch where the data turned. A two-sided jump model with a sticky penalty (λ/d
     = 2.0 for #2) can place a switch *later* than a causal reading of the same data.
     Item 2 is consistent with this for 596.
   - **(c) Label disagreement.** The in-window labels that `A` and the prior are built
     from disagree with the full-sample reference. This is 08-06's "second catch" (A is
     fitted on smoothed in-window labels), which was defensible because no information at
     or after *t* enters.

   Deciding among these is a judgement about the guard's premise. It belongs to a human
   (T-08-40b), and it was not made here.

## 4. Known approximation — class prior (found in 08-06), stated

The filter's class prior (and cold start) is `unconditional_belief` over the **whole**
in-window label series. The nowcaster trains on the D-01 **embargoed** subset (trailing
`embargo_months` dropped, non-finite rows dropped). So `posterior / prior` divides by a
prior slightly different from the one the classifier learned. The plan's rule was kept (a
superset cannot fire the zero-prior raise on a state the nowcaster saw). The mismatch is
also stated in `joint_driver.py`'s, `driver.py`'s and `report/weekly.py`'s docstrings.

## 5. Orchestrator investigation of the halt (2026-09-23) — evidence for the ruling, NOT a reclassification

The halt stands. S-1's three clauses are unchanged and its verdict — **3 LEADs on classifier
#2** — is recorded as measured. What follows tests candidate readings (a) and (c) directly, so
the human ruling rests on measurement rather than on argument.

### 5.1 Candidate (a), a genuine leak — REFUTED by a causal-invariance test

Method (`scripts/diagnose_s1_truncation.py`): physically truncate `monthly_features` and
`monthly_raw` at a cutoff T **before** `build_inputs`, so every derived frame and both frozen
column lists are rebuilt from truncated data; run the l2 joint leg (`NO_REGISTRY`); compare
the filtered belief at every month ≤ T with the committed full-run belief matrices. Each T is
the month immediately before one LEAD's reference transition.

| T | covers LEAD | months ≤ T compared | classifier #1 | classifier #2 |
|---|---|---|---|---|
| 1982-08-31 | 236 (offset −1) | 66 | bit-identical, max diff 0.0 | bit-identical, max diff 0.0 |
| 1995-03-31 | 387 (offset −12) | 209 | bit-identical, max diff 0.0 | bit-identical, max diff 0.0 |
| 2012-08-31 | 596 (offset −31) | 407 | bit-identical, max diff 0.0 | bit-identical, max diff 0.0 |

**What three cutoffs establish — and, corrected the same day, what they do not.**
- **Structural lookahead — ruled out everywhere.** The driver runs the same code at every
  step, so a code path that *reads* rows *j* months ahead would do so at every step, and any
  truncation would perturb the last *j* months before it. All months ≤ T — including month T
  — are identical at three independent cutoffs, so no structural read-ahead of any size exists.
- **Smoothing-type influence — ruled out at the three LEADs, not everywhere.** A two-sided
  decode leaks future into past only where the evidence is ambiguous. Measured on 08-06's
  synthetic world: Arm 2's smoothed substitution breaks invariance at cuts 13, 44 and 68 (the
  month before each led turn) and is **invariant at arbitrary cuts 30, 60, 90 and 110**. Each
  real-data cutoff here sits at the month before a LEAD — exactly where such a leak would have
  manufactured it — so **every belief value that produced each of the three LEADs was computed
  from data dated at or before that LEAD.** A claim that no smoothing-type influence exists at
  *any* month would need a cut at every month; the unit-suite guard does that on the synthetic
  world, and the real-data runs do not.

An earlier version of this section said the three cutoffs rule out "a leak everywhere". That
over-claimed for smoothing-type influence and is corrected above. Registry 42 before and after;
nothing tracked was written by the runs.

### 5.2 Candidate (c), label disagreement — CONFIRMED for classifier #2

Classifier #2's walk-forward terminal-month labels (08-02's `c2_lag1_state`, 588 months,
1971-12 → 2020-11) against its full-sample reference `regime_labels_2`:

- Raw state-id agreement **16.7%**; after the best one-to-one (Hungarian) relabelling **41.3%**.
- The walk-forward labeler assigns **state 4 to 359 of the 361 months 1990-09 → 2020-12**. The
  reference splits the same span into state 2 (1990-09 → 1998-10, mostly), state 3
  (1998-11 → 2012-08) and state 4 (2012-09 → 2020-12).

The belief and the reference therefore **do not share a state vocabulary** for classifier #2.
S-1's premise — that an offset between them measures *when* the belief learned of a
transition — does not hold: at 596, "a 31-month lead into reference state 4" is the belief
sitting in the walk-forward labeler's catch-all state, which happens to carry the same id.

### 5.3 What this does and does not establish

- **Does:** no post-*t* information reaches the belief (5.1). Classifier #2's LEADs are
  artefacts of comparing two labelings that do not agree on what the states are (5.2), and
  are consistent with (b) as well.
- **Does not:** establish that the filter is *useful*, or that classifier #2's walk-forward
  vocabulary is sound. 41.3% aligned agreement is itself a finding about classifier #2 — every
  #2 metric that compares walk-forward output to the full-sample reference (detection lag,
  §5.4 ratio, S-1) is measured across two vocabularies.
- **Does not:** relax S-1. Whether S-1 remains the governing leakage guard, is superseded by
  the invariance test, or is scoped to classifiers whose vocabularies align, is the human
  ruling T-08-40b reserves.

## 6. The ruling, and the real-data adjudication under it

**RULING — 2026-09-23 (Glenn), recorded in `08-08-PLAN.md` Task 3 and T-08-40. It was made
*after* the S-1 guard fired.** Causal invariance is the **governing** leakage guard. S-1 is
demoted to an **observational** reading. S-1's three clauses are **unchanged**, and T-08-40b
still forbids changing them.

**Governing guard, unit suite** (`tests/unit/test_platform_nowcaster_recursion.py`). On
08-06's synthetic world, the belief rows ≤ T are compared between a run truncated at T and
the full run, for **every** T (all 111 months). A and the class prior are supplied once,
fixed.

| arm | cuts that break | as measured |
|---|---|---|
| honest filter | **none** | bit-identical at all 111 cuts |
| Arm 2 (smoothed two-sided decode) | **20 cuts**: {11:1, 12:2, 13:3, 14:4, 24:1, 36:1, 42:1, 43:2, 44:3, 45:4, 56:1, 57:2, 66:1, 67:2, 68:3, 69:4, 79:1, 80:2, 88:1, 100:1} (cut: rows differing) | invariant at 30, 60, 90 and 110 |
| Arm 2 at the p−1 cut family only | exactly **{13, 44, 68}**, 3 rows each | the ruling's recorded prototype result, reproduced |
| one-month structural read-ahead | exactly the cuts where month T+1's evidence differs from month T's | invisible at every steady cut |

The ruling text said Arm 2 "breaks exactly at {13, 44, 68}". That reproduces exactly for
cuts at the month before each reference transition. Under the every-month sweep the ruling
requires, Arm 2 also breaks in the run-up to each led turn, and by one row at every turn
month itself. Both are pinned as measured. The ruling's point stands, and more strongly:
a sparse or convenient cut set can miss a smoothing leak and a read-ahead alike.

**Real data: two-stage — S-1 detects, invariance adjudicates.** Every strictly negative
offset gets a truncation cut at p−1 (`scripts/diagnose_s1_truncation.py`, NO_REGISTRY). Any
break halts. Persisted to `outputs/reports/platform/joint_lift/s1_truncation_invariance.json`
and cited by `test_every_real_negative_offset_was_adjudicated_bit_identical`.

| cut T | adjudicates | S-1 verdict | rows ≤ T | classifier #1 | classifier #2 |
|---|---|---|---|---|---|
| 1982-08-31 | #2, position 236 (1982-09, −1) | LEAD | 66 | bit-identical | bit-identical |
| **1988-01-31** | **#1, position 300 (1988-02, −6)** | held-through miss | **123** | **bit-identical** | **bit-identical** |
| 1995-03-31 | #2, position 387 (1995-04, −12) | LEAD | 209 | bit-identical | bit-identical |
| 2012-08-31 | #2, position 596 (2012-09, −31) | LEAD | 407 | bit-identical | bit-identical |

The 1988-01-31 cut was run by this plan (98.8 s). The other three were run by the orchestrator
(08-CHURN.md §5.1). **4 of 4 negative offsets were adjudicated, with 0 breaks**: no
post-*t* information reached any belief value that produced a negative offset. The S-1
counts are now pinned as measured (#1: 0 LEADs, 1 miss at 300; #2: 3 LEADs at 236, 387,
596) and no longer gate. Registry 42 before and after.

## 7. The three churn numbers — before and after

Each cell gives count / adjacent pairs, rate, window and degraded count. "Before" is quoted
from 08-01-SUMMARY.md, not re-derived.

| | #1 before | #1 after | #2 before | #2 after |
|---|---|---|---|---|
| **A** — `state_N` (L1), l1only | 246 / 587 = 41.91%, 588 steps 1972-01-31 → 2020-12-31, 0 degraded | **246 / 587 = 41.91%**, same window, 0 degraded | 24 / 587 = 4.09%, same | **24 / 587 = 4.09%**, same |
| **A** — `state_N` (L1), l2 | 246 / 587 = 41.91%, 588 steps, 100 degraded | **246 / 587**, 100 degraded | 24 / 587, 100 degraded | **24 / 587**, 100 degraded |
| **B0** — argmax of the RAW posterior, l2 (the control) | 221 / 487 = 45.38%, 488 rows 1974-02-28 → 2020-12-31, 100 degraded | **221 / 487 = 45.38%**, same window, 100 degraded; matrix byte-identical | 66 / 487 = 13.55%, same | **66 / 487 = 13.55%**, same; byte-identical |
| **B1** — argmax of the FILTERED belief, l2 | — (did not exist) | **81 / 487 = 16.63%**, 488 rows 1974-02-28 → 2020-12-31, **100 degraded** | — | **30 / 487 = 6.16%**, same window, **100 degraded** |

- **B1 is not B0.** argmax(belief) and argmax(posterior) differ in **273 of 488** months
  (#1) and **155 of 488** (#2), measured from the two artifacts.
- **Attribution.** B0 did not move and the degraded count did not move, so the difference
  between B1 and B0 is the filter's.
- **Reading.** The numbers are reported, not judged against anything. B1 is lower than B0
  for both classifiers. Per §8 there is no target, and neither direction was pre-declared.
  The S-3 readings below are what say whether the lower churn was bought with timing.

## 8. S-3 readings — directions, no thresholds

Reference: full-sample labels split at 2020-12-31. `act_threshold` 0.70. "raw" is the
posterior (B0's object), "belief" is B1's. Sojourn/lag is design §5.4's ratio.

| reading | #1 raw | #1 belief | #2 raw | #2 belief |
|---|---|---|---|---|
| median sojourn | 9.5 | 9.5 | 29.0 | 29.0 |
| median detection lag (months) | 111.0 (15 of 25 resolved) | **72.5** (20 of 25) | 122.0 (2 of 12) | **44.0** (11 of 12) |
| sojourn / lag ratio | 0.086 | **0.131** | 0.238 | **0.659** |
| overall accuracy vs reference | 0.154 | 0.090 | 0.420 | 0.396 |
| transition-window accuracy (±3 months) | 0.111 | **0.202** | 0.189 | **0.208** |
| steady-state accuracy | 0.165 | 0.062 | 0.448 | 0.418 |
| rows with max prob < 0.70 | 355 / 488 | **181 / 488** | 476 / 488 | **73 / 488** |
| max-prob quantiles q10 / q50 / q90 | 0.348 / 0.572 / 0.763 | 0.498 / 0.773 / 0.964 | 0.352 / 0.458 / 0.613 | 0.633 / 0.948 / 0.999 |

Directions, as measured:
- **The filter makes the path sharper and more resolvable, not more collapsed.** More
  transitions resolve (#1 15 → 20, #2 2 → 11). Median lag falls (#1 111 → 72.5, #2 122 → 44).
  The sojourn/lag ratio **rises** (#1 0.086 → 0.131, #2 0.238 → 0.659). S-3's collapse
  signature is lag rising toward the sojourn and the ratio falling toward 1 or below; that
  is **not** what moved. Transition-window accuracy **rises** for both classifiers.
- **Steady-state and overall accuracy fall for both.** This matters most for #1. Against the
  full-sample reference, #1's overall accuracy is 0.154 raw and 0.090 filtered, below
  uniform 1/6.
- **Caveat.** Every row of this table compares walk-forward output against the full-sample
  reference. For #2 those two labelings share a vocabulary only 41.3% after the best 1:1
  relabelling (§5.2), and #1's sub-uniform overall accuracy suggests a vocabulary mismatch
  there too; that one is unmeasured here. These are readings, not evidence of skill.
- **F-2's lower bound is replaced by a direct measurement.** 08-RESEARCH derived "at least
  309 of 488 non-degraded steps below 0.70" for #1. The raw posterior measures 355 / 488
  (08-01). Under the belief the count is **181 / 488** (#1) and **73 / 488** (#2).
- **The l1only headline is unchanged:** 9.5 / 4.0 = 2.375, 25 of 25. That object is Track
  A's one-hot, which the filter does not touch.

## 9. At the halt: what was withheld (historical — superseded by §6–§8)

- **B1 (argmax churn of the filtered belief): not computed, not recorded.** The
  instruction on a LEAD is not to record a churn number as a result. The belief artifacts
  are committed as **evidence of the finding**, not as a result.
- **S-3 readings on the belief** (sojourn/lag ratio, `transition_window_accuracy`, the
  belief's max-probability distribution): not computed, for the same reason. The
  pre-filter S-3 context stands as recorded elsewhere:
  - l1only headline 9.5 / 4.0 = **2.375** (25 of 25 resolved);
  - raw-posterior max probability below 0.70 in 355 of 488 rows (#1) and 476 of 488 (#2),
    per 08-01.
- **`diagnostics_l2_observational.json`: not regenerated.** Regenerating it would write
  B1 into a tracked record.
- **Still true after the ruling — the l2 equity curves with the filter on: not committed.** The tracked
  `joint_lift_{baseline,joint}_l2.parquet` remain the **pre-filter** observational leg.
  Their `state_*` and `degraded` columns are identical to the filter-on run's; their
  return, turnover and scale columns are not.

Everything else in this section was produced after the ruling: B1 and S-3 in §7–§8, and the
regenerated `diagnostics_l2_observational.json`. The l2 equity curves stay pre-filter until 08-10
regenerates the l2 record.

## 10. No target

No target was pre-declared for either churn metric (08-CONTEXT.md), and none is asserted
anywhere. The roadmap forbids addressing the number by smoothing it, because design §5.4
says to report that quantity prominently. B1 (§7) is reported beside B0 and is not framed
against any target. An AST check in `test_platform_joint_diagnostics_record.py`
(`TestNoChurnTargetIsAsserted`) fails if any test compares B1 to a fixed number or to B0
with a direction. The check is shown to fire on two violating snippets.

## 11. Reproduce

```bash
python scripts/run_joint_lift.py --routing l2 --dump-curves <dir>        # NO_REGISTRY, 0 rows
pytest tests/unit/test_platform_nowcaster_recursion.py                   # S-1 pin, governing invariance, adjudication
python scripts/diagnose_s1_truncation.py <T> <dir>                        # one real-data cut (NO_REGISTRY)
python scripts/joint_lift_diagnostics.py --curves outputs/reports/platform/joint_lift --suffix l2
pytest tests/unit/test_platform_evaluation_sojourn_lag.py -k HeldThrough  # the rule's synthetic pins
```

