---
phase: 08-regime-persistence-stability
plan: 10
artifact: closing record — every criterion, its number, its window, its verdict
created: 2026-09-28
registry: "42 -> 44 (ADR-0002 ceiling 44; headroom 0)"
suite: "2018 -> 2392 passed, 0 skipped, 0 xfailed"
---

# Phase 8 — Measurements: what each criterion actually returned

**House rule, from `07-MEASUREMENTS.md`:** every number carries its denominator and its window
in the same cell. **No criterion below is MET on a shape check.** For each MET, the evidence
column names the arm that can fail and what makes it fail.

**Vocabulary used in this record.**
- **Decision-bearing** means the `L1_ONLY_LAST_FILTERED_STATE` routing (`l1only`). It is the
  only routing whose numbers may change anything.
- **Observational** means the `L2_NOWCAST` routing (`l2`). Per ADR-0002 decision (e) it is
  firewalled: it is appended with `NO_REGISTRY` and **nothing downstream may change on its
  basis**.
- "Criterion 7" with no other qualifier is **Phase 7's** criterion 7, the joint-allocation lift.
  Phase 8's criterion 6 re-measures it.

---

## 1. The ten criteria

| # | criterion (ROADMAP Phase 8) | number, with denominator and window | evidence that can fail | verdict |
|---|---|---|---|---|
| 0 | per-step probability matrix persisted | `joint_lift_probs_{1,2}_l1only.parquet`: **588 rows / 588 steps**, 0 degraded, 1972-01-31 → 2020-12-31. `joint_lift_probs_{1,2}_l2.parquet`: **488 rows / 588 steps**, 100 degraded, 1974-02-28 → 2020-12-31. Re-produced **byte-identical** by both 08-10 runs | Track B is recomputed from these files, not trusted from JSON (`test_track_b_is_re_derivable_from_its_own_named_artifact`). `diagnose()` raises when a matrix is missing and has no fallback (08-01) | **MET** |
| 1 | a prior-state belief propagates, and cannot leak | The belief moves the output: argmax(belief) ≠ argmax(posterior) in **273 / 488** months (#1) and **155 / 488** (#2), l2, 100 degraded. No leak: the honest filter is bit-identical at **all 111** truncation cuts (synthetic). The substituted smoothed arm breaks at **20** cuts. On real data, **4 of 4** negative offsets were cut at p−1 and all were bit-identical | The governing guard is causal invariance (Glenn, 2026-09-23). The discriminating arm shows it can fail. **HALTED first:** S-1 fired on 3 LEADs (#2) and the plan stopped. After the ruling, S-1 is observational (§3) | **MET** — on the observational leg only. The filter is **inert on the decision-bearing leg** by construction (`routing == ROUTING_L2_NOWCAST` gate) |
| 2 | both churn series reported, neither masquerades | **A** (`state_N`, L1): **246 / 587 = 41.91%** (#1) and **24 / 587 = 4.09%** (#2), 588 steps 1972-01-31 → 2020-12-31, identical under both routings. **B0** (raw posterior argmax, l2): **221 / 487 = 45.38%** and **66 / 487 = 13.55%**, 488 rows, 100 degraded, 1974-02-28 → 2020-12-31. **B1** (belief argmax, l2): **81 / 487 = 16.63%** and **30 / 487 = 6.16%**, same window | Under l1only the identity A ≡ B is pinned as True. Under l2 it is pinned as False (184 and 276 of 488 mismatched). An AST check forbids any B1 target or direction and is shown to fire | **MET-AS-A-MEASUREMENT** — no target was declared for either series |
| 3 | Track A diagnosed before it is changed | Churn vs `iloc[-k]`, k = 1…6, #1: **246 / 242 / 245 / 248 / 247 / 249** of 587 pairs. #2: **24–25 / 587**. 588 steps 1972-01-31 → 2020-12-31, 0 degraded. k = 1 anchored to `state_1` with **0 / 588** mismatches | Recomputed from the 588 × 25 parquet. A mutation of the JSON turns 2 tests red (08-02) | **MET-AS-A-MEASUREMENT** — **flat in k**, so the terminal-month edge artefact is refuted. λ/d is **not isolated** from K, features and d |
| 4 | §5.3 anti-flicker gates allocation, closing A7 | A7 **closed by rewording** (Glenn, 2026-09-24). A **5pp no-trade band** on the executed book, not swept, one helper at all three call sites. Decision-bearing effect (08-10, 588 steps 1972-01-31 → 2020-12-31): mean monthly turnover **0.163251 → 0.127412** (#1-alone), **0.119788 → 0.080972** (joint). Months with no trade at all: **0 / 588 → 304 / 588** and **0 / 588 → 341 / 588** | Band-on vs band-off with suppress **and** allow arms at every site; 6 helper mutations and 5 wiring mutations go red. The band-off l1only curve is byte-identical to git (re-confirmed in 08-10) | **MET** under the reworded text in `08-A7.md` §2. **NOT MET as originally worded:** the hysteresis gates **no** weight; `active_regime` is a reported label. The rewording is proposed text; ROADMAP is not yet edited |
| 5 | §4.4 criterion 3 RUN, both classifiers, four schemes | **8,883** per-refit rows, 4 schemes, both classifiers, frames carved at 2020-12-31. AMENDMENT condition (i), #1 state 0: **5 / 9 / 9** episodes (drop first decade / drop last decade / LOO), against the quoted minimum of 3 | Hungarian matching plus the same-n split-half null; Trap A/B/C mutations go red (08-03). But see the three qualifications in §6 | **MET-AS-A-MEASUREMENT** — condition (i) holds. No persistence threshold exists, so every other row is a reading |
| 6 | criterion 7 re-measured, both legs, one harness, window inline | l1only: `wealth_delta` **−0.125307**, `dd_delta` **+0.026164**. l2: **−0.300608** / **+0.061011**. Both over **588 steps, 1972-01-31 → 2020-12-31**. Full table in §4 | `TestCriterion7ReMeasuredIn0810` re-derives the lift and the gate from the committed curves. It also reads the tracked ledger for the 2 rows the record claims | **MET-AS-A-MEASUREMENT** |
| 7 | A11 revisited and answered | Answered `b-promote-dsr` (Glenn, 2026-09-21), written as the reversal of 2026-09-18. Wired into `joint_lift_table` by 08-10 (`bbd88e2`) **before** the re-measurement (`4bfb81b`). The gate's verdict on criterion 7: **FAILED on 2 of 2 legs** (§5) | Pass and fail arms in one table; the hurdle is read live (a patched registry moves it); the boundary is exclusive (mutating `>` to `>=` goes red) | **MET** — the question is answered. The answer fails the lift |
| 8 | G6 pinned as the known non-compliance | **2 of 3** `vol_targeted_tilt` consumers receive the **unpooled** estimate (`driver.py`, `weekly.py`). 1 pools (`joint_tilt.py`). The fixture's pooled weights move by **1.2303 pp** | Both halves plus a precondition in every arm. The first fixture was caught being a no-op on weights. Last changed `94d5283`, and it passes **unmodified** after 08-09 and 08-10 | **MET** — as a pin of the non-compliance, which is **not** fixed |
| 9 | recorded counts match reality; F-4 fixed | Live `pytest --collect-only`: **2392**. CLAUDE.md ×2 and README.md ×2 (badge URL and feature list) now all say 2392, up from 1705. F-4: diagnostics records since 08-01 and measurement records since 08-10 say **246 / 587 = 0.419080** and **24 / 587 = 0.040886** | `test_docs_recorded_counts.py` asserts **equality** against a subprocess collection; its failure path is exercised on a synthetic 1705 document; a 2391 badge turns exactly one arm red. The F-4 arms reject ÷588 at 1e-9 | **MET** |

## 2. The two tracks, side by side

| | Track A | Track B |
|---|---|---|
| object | `state_N` — the **L1** jump model's terminal-month label | `argmax` of the **L2** per-step probability vector |
| #1 | **246 / 587 = 41.91%**, 588 steps 1972-01-31 → 2020-12-31, 0 degraded (l1only). The same 246 under l2 | B0 **221 / 487 = 45.38%**. B1 **81 / 487 = 16.63%**. 488 rows, **100 degraded**, 1974-02-28 → 2020-12-31 |
| #2 | **24 / 587 = 4.09%**, same window | B0 **66 / 487 = 13.55%**. B1 **30 / 487 = 6.16%**. Same window, **100 degraded** |
| what can move it | L1's (K, λ); nothing in this phase changed either | §5.1's Bayes filter (08-08) |

**The 41.91% is an L1 quantity, and §5.1 changes L2. The fix cannot move that number. That it
did not move — 246 before, 246 after, under both routings, re-read from the 08-10 curves — is
the predicted result, not a failure of the fix.**

One question this phase raised and did not answer: Track A counts id changes across **588
separate fits**. 08-07 found that classifier #1's canonical ordering is unstable across refits.
The assignment is non-identity under every contiguous subsample scheme and in 97–100% of
bootstrap replicates. So an **unmeasured** share of the 246 could be id relabelling between
refits rather than regime change. Nothing in this phase decomposes it.

## 3. What is decision-bearing and what is not

| number(s) | source | status |
|---|---|---|
| Track A 246/587 and 24/587; the k = 1…6 diagnostic | L1 fits, routing-agnostic; l1only curves | **decision-bearing input** |
| criterion 7 l1only lift, turnover, the A11 gate verdict | l1only, 2 registry rows | **decision-bearing** |
| F-2's one-hot identity (`active_regime == state_1` in **588 / 588**) | l1only curves | decision-bearing, and a **degeneracy** |
| sojourn/lag headline 9.5 / 4.0 = 2.375 (25 of 25); signed offsets min 1.0, median 4.0 | l1only one-hot vs full-sample reference | decision-bearing input, **qualified by the vocabulary finding (§6)** |
| B0, B1, S-1, S-3, belief max-probability counts, l2 lift and turnover | l2 | **observational, firewalled** |
| §4.4 criterion 3 (08-STABILITY) | full-sample refits on subsamples | neither — a property of the L1 labelings |
| G6, A11 | synthetic fixtures and a ruling | neither |

**Mechanisms that are inert on the decision-bearing leg — stated in those words.**
- **The Bayes filter is inert on the decision-bearing leg.** It is gated off by a literal
  routing test. The l1only curve was pinned byte-identical with the filter on and off.
- **The hysteresis state machine is inert on the decision-bearing leg.** On a one-hot input it
  is the identity on argmax, across 36 admissible threshold pairs. Under the ruled mechanism it
  gates no weight on any leg.
- **The no-trade band is NOT inert.** It is the one mechanism in this phase that moves a
  decision-bearing number, which is why it cost 2 registry rows.

**One l2 number did inform a ruling.** Glenn's `keep-absolute` threshold ruling (08-09 Task 2)
cited the post-filter belief clearing 0.70 in **307 / 488** (#1) and **415 / 488** (#2)
non-degraded l2 months. The ruling **kept** 0.70/0.40 and changed nothing, so no downstream
quantity moved on an l2 basis. It is recorded here because the firewall's wording is "nothing
downstream may change on its basis", and a *keep* decision citing l2 is as close to that line
as this phase came.

## 4. Criterion 7, re-measured

The prior value is a **comparison point, not a target**. It comes from 07-11, before the band,
at 42 trials.

| leg / quantity | prior (07-11, band off) | **08-10 (band on, 5pp)** | window |
|---|---|---|---|
| **l1only `wealth_delta`** (joint − #1-alone, nats) | −0.123438 | **−0.125307** (−0.1253065774082902) | 588 steps, 1972-01-31 → 2020-12-31, 0 degraded |
| l1only terminal-wealth shortfall, joint vs #1-alone | 11.61% | **11.78%** | same |
| **l1only `dd_delta`** (joint − #1-alone max drawdown) | +0.024084 | **+0.026164** (+0.026164042149829925) | same |
| l1only max drawdown, #1-alone / joint | −0.281442 (47 mo) / −0.257358 (40 mo) | **−0.293991 (46 mo) / −0.267826 (39 mo)** | same |
| l1only terminal log wealth, #1-alone / joint | 4.718723 / 4.595285 | **4.606331 / 4.481024** | same |
| l1only mean monthly turnover, #1-alone / joint | 0.163251 / 0.119788 | **0.127412 / 0.080972** | 588 of 588 steps |
| l1only total cost, #1-alone / joint | 0.095992 / 0.070435 | **0.074918 / 0.047612** | same |
| l1only annualized Sharpe, #1-alone / joint | 0.917073 / 0.914903 | **0.896446 / 0.899378** | 588 months |
| l2 `wealth_delta` / `dd_delta` — observational | −0.134505 / −0.001137 (pre-filter, pre-band) | **−0.300608 / +0.061011** (filter + band) | 588 steps, 1972-01-31 → 2020-12-31, **100 degraded** |
| l2 mean turnover, #1-alone / joint — observational | 0.098907 / 0.060451 (pre-filter) | **0.141119 / 0.068775** | 588 of 588 steps |
| plausibility band flags (both routings) | universal ok ×2, domain note ×2 False | **unchanged:** `*_universal_ok` True, `*_domain_note` False | — |
| registry | 40 → 42 (07-11) | **42 → 44** | ADR-0002 ceiling **44**; headroom **0** |

**How the decision-bearing run was established.**
- **Control first.** With the band forced **off** (`NO_REGISTRY`), on this plan's final source,
  the l1only curves and both probability matrices were **`cmp`-identical** to git.
  `wealth_delta` and `dd_delta` reproduced **exactly**. So any movement is the band's.
- **Then the evaluation, run once.** Tags `08-10-c1-alone-L1only-notrade5pp` and
  `08-10-joint-c1xc2-L1only-notrade5pp`, `no_trade_band: 0.05`. Exactly 2 rows were appended.
- **Only weight columns moved.** `state_1`, `state_2`, `active_regime` and `degraded` are
  identical to git. Only `return`, `turnover`, `cost` and `scale` changed. The first differing
  cell is **1972-03-31, `return`**.

**Reading, reported and not acted on.**
- The band did what A7 asked: turnover fell and trading costs fell on both legs.
- It did not improve the lift. `wealth_delta` moved −0.001869 nats, and **both legs** ended
  with lower terminal wealth and deeper maximum drawdowns than without the band.
- τ = 5pp was taken from outside the backtest and **not swept**. No second value was tried, and
  none may be chosen on this basis without a registry row that does not exist.

## 5. The deflated Sharpe — the A11 gate's verdict

Rule (ADR-0003, `08-A11.md`): a decision-bearing leg PASSES iff
`observed_sharpe > expected_max_sharpe(total_trial_count(), sharpe_variance)`, i.e. DSR > 0.5.

| routing / leg | observed Sharpe (588 months) | DSR | hurdle | verdict |
|---|---|---|---|---|
| **l1only #1-alone** | 0.896446 | **1.8159919159760753e-13** | `expected_max_sharpe(44, 1.0)` = **2.226891** | **FAILED** |
| **l1only joint** | 0.899378 | **4.1314825526463704e-12** | 2.226891 | **FAILED** |
| l2 #1-alone — observational, NOT GOVERNING | 1.263438 | 6.520758194176635e-35 | `expected_max_sharpe(42, 1.0)` = 2.208694 | computed, not acted on |
| l2 joint — observational, NOT GOVERNING | 1.157769 | 1.0183630234410774e-45 | 2.208694 | computed, not acted on |

- **Criterion 7 is met as a measurement and FAILED as a result, on both decision-bearing
  legs.** Glenn accepted this in advance when he ruled A11 (2026-09-21, `08-A11.md` §3.4). It
  is reported here as measured, not negotiated.
- Neither leg's Sharpe is within a factor of two of the bar.
- The l2 run read the registry at 42, before the decision-bearing rows landed. The l1only run
  read it at 44, after its own two rows. A trial spent raises the bar its own result must clear.
- **Standing caveat:** `sharpe_variance` is `DEGENERATE_SHARPE_VARIANCE` = **1.0, a declared
  placeholder**, until 20 independent Sharpe-bearing trials exist (ADR-0002 open item 8). It is
  now load-bearing on a gate rather than on a report. When that trigger lands, the hurdle moves
  and this verdict is re-dated.

## 6. Findings that qualify recorded numbers

**The vocabulary finding (08-08, orchestrator measurement).** Neither classifier's walk-forward
labeling shares a state vocabulary with its own full-sample reference. Terminal-month labels,
588 months:

| classifier | raw id agreement | best 1:1 relabel | chance |
|---|---|---|---|
| #1 (K = 6) | 23.0% | **35.7%** | 16.7% |
| #2 (K = 5) | 16.7% | **41.3%** | 20.0% |

#2's walk-forward labeler assigns state 4 to **359 of 361** months from 1990-09.

**Every number that compares walk-forward output to the full-sample reference is measured across
two vocabularies and is qualified by this.** That covers:
- the sojourn/lag headline (median lag 4.0, ratio 2.375);
- 08-06's signed offsets on real inputs;
- the §5.4 ratios;
- all of S-1 (the 3 LEADs);
- all of S-3 (#1's overall accuracy 0.154 → 0.090 is below chance, which this explains).

**It does not qualify criterion 7.** Each step's tilt is keyed on that step's own in-window
labels, so the allocation is internally consistent. It does not qualify 08-STABILITY either,
which matches full-sample fits to each other explicitly. **Unchecked:** whether a human-facing
surface (the weekly report's regime *name*, via `regime_labels.yaml`) names a walk-forward state
with a full-sample label.

**The S-1 halt, and the ruling that causal invariance governs** (`08-CHURN.md` §5–§6).
- **The halt.** S-1 was pre-registered before any real 08-08 number existed. It fired on
  **3 LEADs** for classifier #2, at positions 236 / 387 / 596 with offsets −1 / −12 / −31, and
  on 1 held-through miss for classifier #1 (position 300, offset −6). The plan halted as
  written. Nothing was registered and no B1 number was recorded.
- **The investigation.** It refuted a leak: the beliefs are bit-identical under truncation at
  each LEAD's p−1. It confirmed label disagreement for #2 (41.3% aligned).
- **The ruling.** Glenn, 2026-09-23, **after** the guard fired: causal invariance governs, and
  S-1 is observational. S-1's three clauses are unchanged.
- **Under the ruling,** 4 of 4 negative offsets were adjudicated bit-identical. Only then was B1
  measured.

**The orchestrator's own over-claims, corrected in the record rather than left standing.**
1. `08-CHURN.md` §5.1 first said three cutoffs rule out "a leak everywhere". That over-claimed
   for smoothing-type influence. Three cuts rule it out *at the three LEADs*, not at every
   month. The orchestrator corrected it the same day.
2. The ruling text said the smoothed arm "breaks exactly at {13, 44, 68}". That holds only for
   the p−1 cut family. The every-month sweep the ruling itself required breaks at **20** cuts
   (08-08 Deviation 2).
3. "`active_regime` changes **462 / 587 (78.7%)**" (F-2, relayed during scoping) counted every
   None → None pair as a change, because `NaN != NaN`. Treating None as a state gives
   **122 / 587 (20.8%)**, on the same pre-filter l2 curve. "Flickers nearly twice the raw label"
   does not hold. The **387 / 588** all-cash count stands. Found in 08-09 and corrected in
   ROADMAP on 2026-09-24. It framed Track B's motivation but bore on no ruling.
4. The orchestrator authored 08-09's registry check as "42 + both spends". That check would
   have **failed on the correct answer**, 42, at 08-09. It was corrected at `36bbf28`.
5. The A11 ruling text labelled the two l1only legs' DSRs as "(l1only)" and "(l2)". 08-05
   corrected this in `08-A11.md` §2.

**08-07's findings (08-STABILITY.md), none fixed.**
- **The `evaporated` flag cannot fire under K-fixed refits.** It flagged **0 of 8,883** rows.
  In **36** rows every one of the reference state's months is absent from the subsample. The
  refit repopulates every slot. The true signal is `reference_months_in_subsample`.
  Redefining the flag is a decision.
- **`stability.run_stability` keys rows on the reference state id.** Occupancy, null and
  episodes are read from the same-id state, not the matched partner. 08-03's tests used only
  identity assignments. 08-07's runner is keyed correctly. The library function is not.
- **The primary distance is nearly one-dimensional.** For #1 it is **79.5% `oil` + 19.2%
  `cape_shiller`**, with 7 of 10 columns ≤ 0.1%. For #2 it is **100% `rs_equities_bonds`**.
  It therefore cannot see the columns that define #1's crisis state. A reference-SD companion
  is reported beside it. No verdict rests on either.

**A transcription error found by this plan.**
- 246 / 587 is **0.419080**. Six sites recorded it as "0.418980":
  - `08-01-SUMMARY.md`, `08-01-PLAN.md`, `08-10-PLAN.md` and `.planning/STATE.md`;
  - `evaluation/churn.py`'s docstring;
  - a docstring in `test_platform_evaluation_churn.py`.
- Every JSON record has always carried 0.4190800681431005. No check reads prose, so nothing
  could fail on it. The percentage 41.91% was always right.
- Not edited here: none of the six sites is in this plan's `files_modified`. Recorded for
  correction (§7).

## 7. Open items carried forward — STATE-ready

- **Registry at the ADR-0002 ceiling: 44 / 44, headroom 0** (08-10, 2026-09-28). Any further
  evaluated configuration — a λ value, a band width, a threshold re-pin — needs an ADR-0002
  amendment **first**.
- **Criterion 7 FAILED the A11 quality tier on both decision-bearing legs.** The DSRs are
  1.816e-13 (#1-alone) and 4.131e-12 (joint), against 2.226891 at 44 trials, over 588 steps
  1972-01-31 → 2020-12-31. The hurdle rests on the `sharpe_variance = 1.0` placeholder
  (ADR-0002 open item 8), now load-bearing on a gate.
- **Phase 7 criterion 6 (dependence verdict) remains UNRESOLVED.** The pre-registration at
  `298b1bc` forbids a tie-break. This phase computed no dependence statistic.
- **ADR-0001 condition (iv): the covariance clause is still unimplemented**, because no
  per-regime covariance exists at L4-01. The Sharpe half is **pinned as a known
  non-compliance** by G6: `driver.py` and `weekly.py` consume the unpooled estimate. Not fixed.
- **AMENDMENT condition (i): satisfied** for classifier #1's crisis state under all three
  calendar-order schemes (5 / 9 / 9 episodes against a minimum of 3). Nothing carried under
  D-05. Readings still carried as structure, not verdicts:
  - #1 state 2 is one episode, 1996-07 → 2002-05, and its LOO is degenerate;
  - #2 states 3 and 4 are each a single episode;
  - #1's canonical ordering is unstable under every subsample scheme.
- **Track A: a λ ruling is owed if Track A is ever to move.** The edge artefact is refuted (flat
  in k, 242–249 / 587). λ/d is not isolated from K, features and d, and only a λ sweep would
  isolate it. That is unauthorised and now also blocked by the ceiling. Also open, and
  unmeasured: whether part of the 246 is id relabelling between refits (§2).
- **PRIORITY — the vocabulary finding** (35.7% / 41.3% best-aligned). It needs its own ruling.
  Check the weekly report's regime *name* surface.
- **`evaporated` redefinition** (decision): `reference_months_in_subsample == 0`, with 08-07's
  artifacts regenerated.
- **`run_stability` keying defect:** key on `assignment[state]`, with a non-identity fixture
  that fails first.
- **Criterion 3's distance** is scale-dominated. Whether it should change is a later plan's
  question.
- **`build_inputs` derives on uncut `monthly_raw`.** Verified clean. Recommended: a test that
  pins the invariance.
- **The sojourn/lag headline counts 3 pre-first-decision transitions** (1970-05, 1970-08,
  1971-03). Changing it is a deliberate decision.
- **l2 drifts ~1e-7 across environments** and is bit-reproducible within one. l1only is
  unaffected.
- **A7 follow-ups:**
  - the reworded Phase 4 criterion 3 (`08-A7.md` §2) is proposed text, not yet applied to
    ROADMAP;
  - a K-relative act/unwind rule must be adopted before L2 is ever decision-bearing;
  - a joint R1 × R2 state space is future work;
  - two executor readings are open to Glenn's overrule: leading degraded steps execute nothing,
    and the weekly report bands once per month.
- **ADR index:** `platform_design/adr/README.md` still lacks the row for 0003 (08-05 follow-up).
- **Transcription:** "0.418980" → **0.419080** at the six sites listed in §6.
- **CLAUDE.md line 607** still says "(10 skipped: HDBSCAN + cssselect optional)". This
  environment runs **0 skipped**. The count beside it is now pinned, but this parenthetical is
  not. Left for a docs pass.

## 8. What this phase did NOT do

- **No λ sweep** and **no re-pin of K or λ.** Both classifiers stayed at their pinned
  (6, 10.0) and (5, 16.0).
- **No dependence statistic** of any kind: no ARI, NMI or Cramér's V (`298b1bc`).
- **No migration work.** That is Phase 9.
- **No 2021+ holdout use** for any decision. Every run carves at 2020-12-31, and every window
  above ends 2020-12-31.
- **No target** was pre-declared for either churn metric, or for the lift.
- **No sweep of τ.** The band is 5pp, and no other value was evaluated on any leg.
- **No fix of the G6 non-compliance**, and no pooling added to `driver.py` or `weekly.py`.
- **No band was tightened or re-tuned.** The four plausibility bands are unchanged, and
  plausibility-only.
- `legacy/` and the reference submodules are untouched. The legacy-import **ratchet stays at
  31**.

## 9. Baselines

| | phase start | phase end |
|---|---|---|
| suite | 2018 passed, 0 skipped, 0 xfailed | **2392 passed, 0 skipped, 0 xfailed** (live collection 2392) |
| legacy-import ratchet | 31 | **31** (measured by the ratchet's own scan; may only decrease) |
| registry (`total_trial_count()`) | 42 | **44**: A11 spent 0, A7 authorised 2, 08-10 consumed 2 |
| DSR hurdle | `expected_max_sharpe(42, 1.0)` = 2.208694 | `expected_max_sharpe(44, 1.0)` = **2.226891** |
| criterion 7, l1only | −0.123438 / +0.024084 | **−0.125307 / +0.026164**, band on |

## 10. The phase's own defects

This phase was scoped on a wrong causal model, and it corrected itself before executing. As
first written, criterion 2 measured `state_1`, an **L1** label, to judge a fix to **L2**. It would
have reported **246 → 246** whether the fix worked or not. That was the **sixth** recorded
instance of this project's signature defect, a check that can only confirm. Claude authored it
while scoping a phase whose purpose was catching exactly that. It was caught by the phase's own
research (08-RESEARCH F-1), not at UAT.

What changed as a result:
- the two tracks were split and never shared a plan;
- every plan was required to show that its checks could fail;
- a "reproduce, else halt" ladder guarded the one decision-bearing number.

**It did not stop there.** Counted from the SUMMARYs, `08-STABILITY.md`, `08-CHURN.md` and
STATE, execution produced **seven more checks that could only confirm**:

1. **08-03** — the first `n_seams` tests were satisfied by the broken `n_seams = n_blocks`. The
   author's own mutation check caught it before commit.
2. **08-04** — G6's first contrast fixture made pooling a no-op **on the weights**. Caught when
   the arm failed.
3. **08-06** — the plan's no-change verify only re-read the committed JSON, so it could not see
   a refactor. The planner wrote it; the executor strengthened it.
4. **08-03 → 08-07** — the `evaporated` flag cannot fire under K-fixed refits: 0 of 8,883 rows
   flagged, 36 that should have been. Found by 08-07. **Not fixed.**
5. **08-03 → 08-07** — `run_stability`'s tests exercised only identity assignments, where its
   keying defect is invisible. **Not fixed.**
6. **Orchestrator** — "three cutoffs rule out a leak everywhere". Corrected the same day.
7. **Orchestrator / ruling text** — "Arm 2 breaks exactly at {13, 44, 68}", a convenient cut
   set. The every-month sweep found 20 breaking cuts.

So the count stands at **thirteen**: six before this phase, seven during it. Two of the seven
were authored by the orchestrator, and two remain unfixed in the library.

**Two checks of the opposite polarity would have failed on the correct answer.**
- S-1's original "one strictly negative offset halts" would halt an honest filter. 08-06
  measured this, and the rule was amended before any real 08-08 number existed.
- The orchestrator's own 08-09 registry check ("42 + both spends") would also have failed on
  the correct answer. It was corrected at `36bbf28`.

**Two recorded numbers were wrong in prose while right in data:**
- the orchestrator's **462 / 587** (NaN ≠ NaN), corrected to 122 / 587;
- the six-site **"0.418980"**, found by this plan.

That recorded-number shape is the one D-07 named. It recurred twice more in this phase.

**The honest summary.**
- Most of these were caught **inside the plan that made them**, by a mutation check or by an
  arm that failed. That is the process change working.
- Five were caught **one or more plans after they were made**:
  - the evaporation flag and the `run_stability` keying (08-03 → 08-07). Both are still live
    in `labeling/stability.py`;
  - S-1's original rule (planning → 08-06);
  - the 462 (scoping → 08-09);
  - the "0.418980" (08-01 → 08-10).
- The orchestrator is not exempt from the defect it polices. All five corrections listed in §6
  under "the orchestrator's own over-claims" are to statements the orchestrator made.
- This record claims no better than that.

## 11. AMENDMENT 2026-09-29 — CR-01: the filter's likelihood prior (plans 08-15..08-19)

§1–§10 above stand as written. Every old number in them was measured on the pre-CR-01 filter.
This section records, old → new, what the fix moved and what it did not. No old number is
replaced. Every re-measurement below ran with `NO_REGISTRY`, and **registry rows spent: 0**.

### 11.1 What changed

Before the fix, the Bayes filter turned the L2 nowcaster's posterior into a likelihood by
dividing it by the **whole in-window label distribution**, restricted to the posterior's classes.
The posterior is calibrated against the class frequencies of the rows the model was **fit on**,
which is a different distribution (08-REVIEW CR-01). Now L_t divides by each step fit's
**training class prior**. `fit_l2_nowcaster` returns that prior (08-16). Serve persists it as
`nowcaster_class_prior` (08-16), and both backtest drivers pass it to `filter_step` (08-17). π_0
and A are unchanged: both still come from the in-window labels over all K states. A likelihood
prior of the old shape (mass on a state the posterior lacks, or not summing to one over the
posterior's classes) is now **refused**, not renormalized (08-17).

### 11.2 What did NOT change, each with its proof

| quantity | value | proof (plan 08-19, final source) |
|---|---|---|
| **the decision-bearing l1only record** | byte-unchanged | Three forms. **(a)** `git diff --quiet b7193fd --` over the six committed l1only files (`measurement_l1only.json`, `diagnostics_l1only.json`, `joint_lift_{baseline,joint}_l1only.parquet`, `joint_lift_probs_{1,2}_l1only.parquet`): clean. **(b)** `run_joint_lift.py --routing l1 --dry-run` (`NO_REGISTRY`) reproduced all four l1only parquets **`cmp`-identical**. **(c)** In the dry-run record, every field equals the committed record exactly: `lift` (28 keys), `baseline_leg` and `joint_leg` (27 keys each), `deflated_sharpe.{baseline,joint}` (9 keys each) and the 11 configuration keys (`use_regime_filter` False). The exceptions are the fields a dry run must differ in by construction, and they are listed rather than hidden: `decision_bearing` True → False; `dry_run` False → True; the `registry` block (42 → 44 with 2 rows and the 08-10 tags → 44 → 44 with 0 rows and "(NO_REGISTRY)"); `quality_tier.governs` True → False and its `verdict`; `n_trials_read_at` (the read time); `registry_row_written` True → False |
| Track A | **246 / 587** (#1) and **24 / 587** (#2), 588 steps 1972-01-31 → 2020-12-31 | `state_1` and `state_2` in the new l2 curves are identical to the committed ones (`assert_series_equal`, exact); the new record reads 246 and 24 |
| B0 (raw posterior argmax, l2) | **221 / 487** and **66 / 487**, 488 rows 1974-02-28 → 2020-12-31 | `joint_lift_probs_{1,2}_l2.parquet` from the re-run are `cmp`-identical to the committed files (and value-identical, `check_exact`); the regenerated diagnostics read 221 and 66 |
| argmax(posterior) ≠ `state_N` (l2 series identity) | **184 / 488** (#1) and **276 / 488** (#2) | unchanged in the regenerated diagnostics (the posterior and the states did not move) |
| degraded steps | **100 / 588** (87 on #1, 13 on #2) | the `degraded` column in both new l2 curves is identical to the committed one |
| registry | **44**, sha256 `c957e8fdb360f04cebd36a6ac19f7efe5c036d098152a5b0c1694f2e73c088ad` | read before and after every run in 08-19 |

In the new l2 curves, only `return`, `turnover`, `cost`, `scale` and `active_regime` differ. The
first differing date is **1974-02-28** for `return`, `turnover`, `cost` and `scale` (the first
non-degraded l2 step), and **1976-11-30** for `active_regime`.

### 11.3 Old → new

All L2 numbers here are **observational** (`NO_REGISTRY`, firewalled; §3). The curve window is
**588 steps, 1972-01-31 → 2020-12-31, 100 degraded**. The matrix window is **488 non-degraded
rows (487 adjacent pairs), 1974-02-28 → 2020-12-31**.

| quantity | old (08-08 / 08-10) | **new (08-19)** | denominator and window |
|---|---|---|---|
| B1 (belief argmax churn), #1 | 81 / 487 = 16.63% | **64 / 487 = 13.14%** | 487 pairs, 488 rows, 1974-02-28 → 2020-12-31, 100 degraded |
| B1, #2 | 30 / 487 = 6.16% | **27 / 487 = 5.54%** | same |
| argmax(belief) ≠ argmax(posterior), #1 | 273 / 488 | **380 / 488** | 488 rows, same window |
| argmax(belief) ≠ argmax(posterior), #2 | 155 / 488 | **178 / 488** | same |
| belief max-prob clearing 0.70, #1 | 307 / 488 | **273 / 488** | 488 rows, same window; act threshold 0.70 |
| belief max-prob clearing 0.70, #2 | 415 / 488 | **408 / 488** | same |
| l2 `wealth_delta` (joint − #1-alone, nats) | −0.300608 (−0.30060767886336137) | **−0.296783** (−0.2967834422163289) | 588 steps, 1972-01-31 → 2020-12-31, 100 degraded |
| l2 `dd_delta` | +0.061011 (+0.06101144035366535) | **+0.063258** (+0.0632580887963281) | same |
| l2 terminal log wealth, #1-alone / joint | 4.228027 / 3.927420 | **4.262866 / 3.966083** | same |
| l2 max drawdown, #1-alone / joint | −0.254700 (32 mo) / −0.193688 (68 mo) | **−0.267290 (33 mo) / −0.204032 (33 mo)** | same |
| l2 mean monthly turnover, #1-alone / joint | 0.141119 / 0.068775 | **0.139896 / 0.070970** | 588 of 588 steps |
| l2 total cost, #1-alone / joint | 0.082978 / 0.040440 | **0.082259 / 0.041730** | same |
| l2 annualized Sharpe, #1-alone / joint | 1.263438 / 1.157769 | **1.178896 / 1.159271** | 588 months |
| l2 DSR, #1-alone | 6.520758e-35, `n_trials` 42, hurdle `expected_max_sharpe(42, 1.0)` = 2.208694 | **3.978469e-46**, `n_trials` 44, hurdle `expected_max_sharpe(44, 1.0)` = 2.226891 | 588 months; `sharpe_variance` 1.0 (placeholder) |
| l2 DSR, joint | 1.018363e-45, 42, 2.208694 | **3.171019e-48**, 44, 2.226891 | same |
| S-1, #1 (belief vs full-sample reference) | **1** negative offset: 0 LEADs, 1 held-through miss at 300 (1988-02-29, −6) | **0** negative offsets | 25 reference transitions; belief resolved **20 → 18** of 25 |
| S-1, #2 | **3** negative offsets, all LEADs: 236 (1982-09-30, −1), 387 (1995-04-30, −12), 596 (2012-09-30, −31) | **2**, both LEADs: 387 (1995-04-30, −12), 596 (2012-09-30, **−32**) | 12 reference transitions; 11 of 12 resolved (unchanged) |
| S-1 adjudication | 4 of 4 cuts at p−1 bit-identical (08-08: 1982-08-31, 1988-01-31, 1995-03-31, 2012-08-31) | **3 of 3** cuts bit-identical by the script's test **and** by a NaN-aware exact frame comparison: 1995-03-31 and 2012-08-31 (p−1 of the two LEADs) and 1982-08-31 (the standing spot-check). `n_break` 0 | `s1_truncation_invariance.json` (plan 08-19) |
| S-3 belief `transition_window_accuracy`, #1 overall / transition | 44/488 = 0.0902 / 20/99 = 0.2020 | **32/488 = 0.0656 / 18/99 = 0.1818** | 488 rows; 99 transition-window, 389 steady-state |
| S-3, #2 overall / transition | 193/488 = 0.3955 / 11/53 = 0.2075 | **158/488 = 0.3238 / 10/53 = 0.1887** | 488 rows; 53 transition-window, 435 steady-state |
| belief sojourn/lag, #1 (median lag, months) | 72.5 (20 of 25 resolved) | **52.0 (18 of 25)** | median sojourn 9.5, unchanged |
| belief sojourn/lag, #2 | 44.0 (11 of 12) | **59.0 (11 of 12)** | median sojourn 29.0, unchanged |
| served belief, top state (real tracked data, cold start, as-of 2026-06-30) | **3** (0.369157) | **0** (0.299992); then 1: 0.292707, 3: 0.163236 | 08-SERVING.md §4 |

**The DSR's registry read moved 42 → 44 for a reason unrelated to the fix.** The 08-10 l2 run
read the registry at 42, before the two decision-bearing rows landed. The 08-19 re-run read it at
44. Holding the old read, the new Sharpes give 1.308430e-44 (#1-alone) and 1.151274e-46 (joint)
at 42. So both the Sharpe change and the hurdle change lowered the DSR. Neither DSR is anywhere
near 0.5.

**The S-1 counts moved because the belief path moved.** The raw posterior did not move. Each new
negative offset was adjudicated before its pin moved: `_MEASURED_S1` in
`test_platform_nowcaster_recursion.py` now holds the new reading, with the 08-08 reading kept
verbatim beside it.

### 11.4 The firewall

Every l2 number above stays **observational**, and nothing downstream changes on its basis.

One l2 number did inform a ruling. Glenn's `keep-absolute` threshold ruling (08-09 Task 2) cited
the belief clearing 0.70 in **307 / 488** (#1) and **415 / 488** (#2) months. Those counts are now
**273 / 488** and **408 / 488**, over the same 488 rows. The ruling kept 0.70/0.40 and changed
nothing, so no quantity moves.

**Open question for Glenn (not decided here):** does he want to revisit the keep-absolute ruling
on the new counts?

### 11.5 Standing qualifications

- **CR-02 (Phase 8.1):** the filter carries a belief across walk-forward refits whose state ids
  are not aligned. Every belief-derived number above (B1, the mismatches, max-prob, S-1, S-3, the
  l2 lift and Sharpes) is qualified by it.
- **CR-03 (Phase 8.1):** two L2 model columns have a real publication lag that the backtest does
  not model. Every L2-routed number above is qualified by it.
- **The vocabulary finding (§6)** still qualifies S-1 and S-3, which compare walk-forward output
  to a full-sample reference across two vocabularies.
- This amendment fixes none of them.
