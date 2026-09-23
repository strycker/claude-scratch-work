---
phase: 08-regime-persistence-stability
plan: 08
created: 2026-09-23
status: HALTED — S-1 real-data leakage guard found 3 LEADs (classifier #2); T-08-40
---

# 08-CHURN — Bayes filter wired; the real-data leakage guard HALTED the measurement

> **Decision-bearing caveat, first.** The Bayes filter changes nothing on the
> decision-bearing `ROUTING_L1_ONLY`: there `probs` is a one-hot on the L1 label, which is
> not a likelihood, and the filter is literally gated off (`if routing == ROUTING_L2_NOWCAST
> and use_regime_filter:`). Everything below about the belief is on the **observational,
> firewalled l2 leg**, and per ADR-0002 decision (e) **nothing downstream may change on the
> basis of it**. That includes the halt: it stops this plan's measurement. It does not move
> criterion 7, which was re-measured byte-identical (below).

## 0. The halt, in one paragraph

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

The finding is recorded below for a human decision.

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

## 5. What is NOT in this record, and why

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
- **The l2 equity curves with the filter on: not committed.** The tracked
  `joint_lift_{baseline,joint}_l2.parquet` remain the **pre-filter** observational leg.
  Their `state_*` and `degraded` columns are identical to the filter-on run's; their
  return, turnover and scale columns are not.

## 6. No target

No target was pre-declared for either churn metric (08-CONTEXT.md), and none is asserted
anywhere. The roadmap forbids addressing the number by smoothing it, because design §5.4
says to report that quantity prominently. No B1 number exists in this record to be framed
against a target, and none should be framed that way when one exists.

## 7. Reproduce

```bash
python scripts/run_joint_lift.py --routing l2 --dump-curves <dir>        # NO_REGISTRY, 0 rows
pytest tests/unit/test_platform_nowcaster_recursion.py -k real_data       # #1 green, #2 RED: 3 LEADs
pytest tests/unit/test_platform_evaluation_sojourn_lag.py -k HeldThrough  # the rule's synthetic pins
```

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

**Why three cutoffs rule out a leak everywhere, not just at three months:** the driver runs
the same code at every step, so a code path reading *j* months ahead would read ahead at every
step, and any truncation would perturb the last *j* months before it. All months ≤ T are
identical — including month T itself — at three independent cutoffs, so no lookahead of any
size exists. **Every belief value that produced each LEAD was computed from data dated at or
before that LEAD.** Registry 42 before and after; nothing tracked was written by the runs.

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
