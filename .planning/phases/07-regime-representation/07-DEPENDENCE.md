# Phase 7 Plan 9 — Criterion-6 Dependence Record

**Measured:** 2026-09-18, on branch `claude/keen-galileo-zqcml6-w3` at commit `3c33262`.
**Module:** `src/trading_crab_lib/platform/evaluation/dependence.py`
(`measure_labeling_dependence`, which delegates alignment and the cross-tabulation to
`platform/plotting/regime.py::label_disagreement` — one alignment, never two).

**Read this document with D-15 in view.** No pass/fail threshold is declared or applied
anywhere in the code. The three flag levels quoted below (ARI 0.7, NMI 0.5, Cramér's V 0.5) are
`[ASSUMED]` per `07-VALIDATION.md` and word the report; nothing branches on them. The judgement
is a human's and it is recorded at plan 07-09 Task 3.

---

## 0. Provenance — how each input was produced (read before the numbers)

### 0.1 `regime_labels_2` (classifier #2) — loaded from disk as plan 07-08 left it

`data/checkpoints/platform/regime_labels_2.parquet`, written 2026-09-18 16:40 by the
re-pinned K = 5, λ = 16.0 (2n) production fit recorded in
`platform_design/adr/0002-l1-second-classifier.md` § RE-PIN 2026-09-18. **696 months,
1963-01-31 → 2020-12-31.** Not recomputed by this plan.

### 0.2 `regime_labels` (classifier #1) — did NOT exist and was produced by this plan

`data/checkpoints/platform/regime_labels.parquet` was absent from the tracked checkpoint
namespace before this plan ran; the directory held only the `*_2` trio. It was produced here by
running classifier #1's own labeling entry point —
`platform/labeling/diagnostics.py::label_regimes` — on the holdout-carved `monthly_features`
checkpoint restricted to the **frozen** L1 feature columns:

```
dev_features, _   = split_by_holdout_boundary(monthly_features, DEFAULT_HOLDOUT_CUTOFF)   # 0 rows carved; the checkpoint is already dev
lean_cols         = sorted(lean_feature_set(cfg) & set(dev_features.columns))              # 13
first_decision    = dev_features.index[cfg.backtest.min_train_months]                      # 1972-01-31
ref_cols          = _reference_label_columns(dev_features, lean_cols, first_decision)      # 10 (D-01/D-11 freeze rule, reused unmodified)
label_regimes(dev_features[ref_cols], cfg)                                                 # writes regime_labels
```

Frozen columns (10 of 13): `cape_shiller`, `credit_spread_baa_aaa`, `curve_10y3m`, `div_yield`,
`oil`, `real_rate_level`, `realized_vol_1m`, `realized_vol_3m`, `trailing_return_1m`,
`trailing_return_3m`. Excluded by the freeze rule: `curve_10y2y`, `fred_vix`, `gold`.

**Equivalence check, run before the measurement.** The labeling this produces is
`np.array_equal`-identical, on an identical index, to `evaluation/report.py`'s own step-(d)
full-sample smoothed reference construction recomputed independently from the same checkpoint.
So `regime_labels` here is classifier #1's frozen-policy labeling, not a variant of it.

**`outputs/reports/platform/backtest_full_sample_states.parquet` was deliberately NOT used**,
although it exists and is superficially the right artifact. Its occupancy is
1.5827 / 13.9568 / **31.9424** / **40.5755** / 11.9424 %. That is reproducible today **only by
dropping `oil` from the frozen set** — verified directly: refitting on the 9 non-`oil` frozen
columns reproduces those five figures to four decimal places. The on-disk artifact is therefore
the pre-`oil`-fix 9-column labeling, one month short (695 rows from 1963-02-28 — same span, but
a different labeling). Using it would have made every statistic below describe something other
than what this record claims.

### 0.3 Classifier #1's occupancy has MOVED since `07-MEASUREMENTS.md`, and is reported as measured

| state | `07-MEASUREMENTS.md` frozen column (2026-09-14) | measured here (2026-09-18) |
|---|---|---|
| 0 | 11.51 % | **1.5827 %** |
| 1 | 9.06 % | **13.9568 %** |
| 2 | 35.54 % | **33.2374 %** |
| 3 | 31.51 % | **39.2806 %** |
| 4 | 12.37 % | **11.9424 %** |

**Attribution:** commit `dbd3fcc` re-resolved the `oil` splice from macrotrends `wti_crude`
(1985-02+) to FRED `wti_fred` (1962-01+). `oil` is one of the ten frozen columns, so its values
changed from 1985 onward and its history extended 23 years further back; classifier #1's
labeling legitimately changed with it. `07-MEASUREMENTS.md`'s frozen column was produced against
the hand-recomputed, stale-`oil` checkpoint and **cannot be reproduced today**. That record is
left as it stands — it is not retro-edited to match, and the figures above are not adjusted
toward it.

**Consequence that must not be buried:** on today's data classifier #1 **breaches design §4.4
criterion 1 at both ends** — state 0 at 1.58 % is far below the ~8 % floor (11 months of 695),
and state 3 at 39.28 % is above the ~35 % cap. `label_regimes` emitted both WARNINGs
(report-only, D-02). Classifier #1's §4.4 standing is out of this plan's scope to fix, but a
reader of the dependence numbers below needs to know that one of the two labelings being
compared has a state occupying 11 months.

---

## 1. The three statistics

Both labelings' own spans, so the reader can judge whether the common window is the intersection
expected:

| labeling | K | own span | own months |
|---|---|---|---|
| classifier #1 (`regime_labels`) | 5 | 1963-02-28 → 2020-12-31 | 695 |
| classifier #2 (`regime_labels_2`) | 5 | 1963-01-31 → 2020-12-31 | 696 |
| **common window** | — | **1963-02-28 → 2020-12-31** | **`n_compared` = 695** |

| statistic | value | measured over | flag level | flag status |
|---|---|---|---|---|
| **Adjusted Rand** (ARI) | **0.4686082735798639** | `n_compared` = 695 months, 1963-02-28 → 2020-12-31, K₁ = 5 vs K₂ = 5 | 0.7 `[ASSUMED]` | at or below the flag level |
| **Normalized mutual information** (NMI) | **0.6023322651194867** | `n_compared` = 695 months, 1963-02-28 → 2020-12-31, K₁ = 5 vs K₂ = 5 | 0.5 `[ASSUMED]` | **ABOVE the flag level** |
| **Cramér's V** | **0.6558187277453474** | `n_compared` = 695 months, 1963-02-28 → 2020-12-31, K₁ = 5 vs K₂ = 5 | 0.5 `[ASSUMED]` | **ABOVE the flag level** |

`suspicious = False`; `suspicious_reason` empty (neither perfect-statistic signature fired — see
§4).

**Is the common window the intersection expected?** Yes, and the one-month difference is
explained rather than assumed: `n_compared` = 695 is the *complete* intersection of a 695-month
and a 696-month labeling. Classifier #1 starts one month later because `real_rate_level`, one of
its ten frozen columns, has its first valid month at 1963-02-28, and `label_regimes` drops rows
with any NaN. Nothing was silently lost.

**A note on "the 1972+ decision window" (D-11), so 695 is not misread.** D-11's 1972+ first
decision date governs *which columns qualify for the frozen set* — it is the freeze rule's
reference date, not a truncation of the labeled span. Both classifiers label their full
available history once frozen, which is why both run back to 1963 and `n_compared` = 695 rather
than the 588 months from 1972-01 to 2020-12. Both were frozen at the same 1972-01-31 date by the
same `_reference_label_columns` call, which is what makes them month-for-month comparable.

---

## 2. The cross-tabulation

Classifier #1's states (rows) by classifier #2's states (columns), with both marginals:

```
 #2 →        0     1     2     3     4   |  ALL
 #1 ↓
  0          0     0     0    11     0   |    11
  1         93     4     0     0     0   |    97
  2          0   136    95     0     0   |   231
  3          0     0    18   155   100   |   273
  4         23    17    43     0     0   |    83
 ----------------------------------------|-------
 ALL       116   157   156   166   100   |   695
```

Row marginals (classifier #1 occupancy over the common window):
1.5827 / 13.9568 / 33.2374 / 39.2806 / 11.9424 %.
Column marginals (classifier #2 occupancy over the common window):
16.6906 / 22.5899 / 22.4460 / 23.8849 / 14.3885 %.

**Does it show diagonal structure?** Not a literal diagonal — and a literal diagonal is not the
thing to look for. The two state numbering schemes are **independent**: classifier #1 orders its
states on ascending `trailing_return_1m` centroid, classifier #2 on ascending
`rs_equities_bonds`. Neither ordering has any reason to line up with the other, so a diagonal is
meaningful only after accounting for that, and **a near-permutation of the identity is as strong
a dependence signal as a literal diagonal**.

What the table shows is a strong near-permutation. Read row by row, each classifier-#1 state
concentrates into one or two classifier-#2 states:

- #1 state 0 (11 mo) → **100 %** into #2 state 3
- #1 state 1 (97 mo) → **95.9 %** into #2 state 0
- #1 state 2 (231 mo) → 58.9 % into #2 state 1, 41.1 % into #2 state 2 (two cells hold 100 %)
- #1 state 3 (273 mo) → 56.8 % into #2 state 3, 36.6 % into #2 state 4 (two cells hold 93.4 %)
- #1 state 4 (83 mo) → 51.8 % / 27.7 % / 20.5 % across #2 states 2, 0, 1 — the only genuinely
  diffuse row

Fourteen of the twenty-five cells are exactly zero. Read the other way, every classifier-#2
column is dominated by one classifier-#1 row: 80.2 %, 86.6 %, 60.9 %, 93.4 % and **100 %** for
columns 0–4 respectively. That concentration is what drives NMI and Cramér's V above their flag
levels.

---

## 3. Verdict, in criterion 6's own terms

Two of the three statistics exceed their flag levels — NMI at 0.6023 against 0.5, and Cramér's V
at 0.6558 against 0.5 — and the cross-tabulation is a strong near-permutation with fourteen empty
cells.

**Classifier #2 represents a failure to add an axis.**

ROADMAP criterion 6 states that high dependence is recorded as a failure to add an axis, and
this is that recording. It is the deliverable.

---

## 4. Suspicion section — what would be a wiring bug rather than a finding

A dependence statistic is not self-validating. Each named signature was checked explicitly.

### 4.1 ARI or NMI at or extremely near 1.0 — *both classifiers reading the same labels*

**Not fired.** ARI = 0.4686, NMI = 0.6023; `measure_labeling_dependence` sets
`suspicious = True` on an exact 1.0 and did not. Independently: the two checkpoint files are
distinct on disk (`regime_labels.parquet`, 695 rows, written 17:19 by `label_regimes`;
`regime_labels_2.parquet`, 696 rows, written 16:40 by `label_leadership_regimes`), they have
different row counts and different start months, and their raw label sequences disagree on
441 of 695 months. The two fits also used disjoint feature sets (classifier #1's ten frozen
columns above; classifier #2's eight `labeling_2` columns), different λ (52.0 vs 16.0) and
different ordering columns. They are not the same labeling.

### 4.2 All three statistics at or near 0.0 simultaneously — *an alignment bug*

**Not fired, and not close.** All three are well above zero. The opposite failure — the module
silently returning zeros — is also excluded: `measure_labeling_dependence` returns NaN, never
0.0, on a zero overlap, and `n_compared` here is 695.

### 4.3 `n_compared` materially below the months in the shared decision range — *a silent index mismatch*

**Not fired.** `n_compared` = **695**, an integer, against a **695**-month classifier-#1 span and
a **696**-month classifier-#2 span. 695 is the full intersection; the single non-overlapping
month (1963-01-31) is accounted for in §1 by `real_rate_level`'s first valid month. There is no
shortfall.

The silent-zero coercion trap was also excluded by construction: both `state` columns load from
parquet as `int64`, not as `state_N` strings, so `label_disagreement`'s internal
`pd.to_numeric(..., errors="coerce")` had nothing to NaN out. Had they been strings,
`n_compared` would have been 0 and the module would have returned NaN with a warning naming the
coercion — not `pct_disagree = 0.0`, which reads as "totally resolved" while comparing nothing.

### 4.4 One classifier-#2 state absorbing nearly every observation — *the wrong checkpoint loaded*

**Not fired.** The crosstab's column marginals were compared against plan 07-08's recorded
occupancy explicitly:

| #2 state | 07-08 / ADR-0002 § RE-PIN recorded (696 mo) | crosstab column marginal (695 mo) | agree? |
|---|---|---|---|
| 0 | 16.6667 % | 16.6906 % | yes (+0.024 pp) |
| 1 | 22.7011 % | 22.5899 % | yes (−0.111 pp) |
| 2 | 22.4138 % | 22.4460 % | yes (+0.032 pp) |
| 3 | 23.8506 % | 23.8849 % | yes (+0.034 pp) |
| 4 | 14.3678 % | 14.3885 % | yes (+0.021 pp) |

**They agree.** Every difference is under 0.12 pp and is fully explained by the denominator
differing by the one dropped month (695 vs 696). The largest column is 23.88 % and the smallest
14.39 %, so no state absorbs the table. The correct classifier-#2 checkpoint was loaded.

Classifier #1's row marginals likewise match the occupancy `label_regimes` reported when it
wrote the checkpoint (1.58 / 13.96 / 33.24 / 39.28 / 11.94 %), including the two §4.4 WARNINGs
described in §0.3.

### 4.5 Recorded alongside, **not** as mitigation of §3

One further fact about both inputs is recorded here because a reader will notice it and should
not have to rediscover it. It does **not** qualify the verdict in §3, and it is not offered as a
reason the numbers might be acceptable.

Both labelings are extremely blocky in time. Over the same 695 months, classifier #1 changes
state **6** times and classifier #2 **12** times — seven and thirteen contiguous blocks
respectively. Two step functions over one shared time axis share the time axis as a common
factor, so some association between them is expected before any shared economics is invoked.
Whether an ARI/NMI/Cramér's V computed between two heavily-smoothed block partitions of the same
timeline is the right instrument for "did classifier #2 add an axis" is a question this plan
does not answer and has no pre-declared answer for (D-15). It is routed to the human at Task 3
alongside the verdict, **not** applied to soften it.

Related, and already on the record in ADR-0002 § RE-PIN's own "Two limitations": classifier #2's
states 3 and 4 each occur exactly once (the final two blocks, 1998-11 → 2012-09 and 2012-09 →
2020-12), and §4.4 criterion 3 (subsample stability) has never been run for either classifier.

---

## 5. Comparison to wave 1's disagreement baseline — calibration only

Wave 1 measured `pct_disagree` = **0.80899 over 288 of 356 months**
(`07-MEASUREMENTS.md` §2).

**That is a different quantity and the two numbers are not comparable.** Wave 1 compared **one**
labeling against its **own** walk-forward filtered version — the same state space, the same
numbering, so a raw label mismatch is meaningful. This plan compares **two different labelings**
produced by two different classifiers whose state numbering schemes are independent by
construction, so a raw label mismatch between them is close to meaningless as a dependence
statistic.

It is stated here only so a reader who remembers the 80.9 % figure does not mistake this plan's
numbers for it. For completeness: over the same alignment used above,
`pct_disagree` = 0.6345323741007194 (441 of 695). **Do not read that as "the two labelings agree
34 % of the time."** It is an artifact of two independent numbering schemes and is not one of
criterion 6's statistics. The three statistics in §1 are the measurement; that figure is not.

---

## 6. Sign-off

**Pending.** Plan 07-09 Task 3 is a blocking `checkpoint:human-verify`. D-15 requires a human to
read these numbers and requires the judgement to be recorded; criterion 6 requires that a high
result be recorded as a failure to add an axis, which §3 does. The human's judgement — (a) an
axis was added, (b) classifier #2 failed to add an axis, or (c) the numbers are not trustworthy
— goes here verbatim and is carried into ADR-0002 by plan 07-11.

No agent has marked this satisfied.

---

## PRE-REGISTRATION 2026-09-18 — the decision rule, committed before the control exists

Glenn deferred the criterion-6 verdict at plan 07-09's Task 3 gate and directed that a
block-permutation control be run first, with its reading **pre-registered**. This section is
committed **before any control code is written or run**. The commit timestamp is the evidence
that the rule predates the number; if this section and the result ever appear in the same
commit, the pre-registration is void and must be re-done.

### Why a control is needed at all

The three statistics assume exchangeable observations. These labelings are not exchangeable —
they are step functions on a shared time axis. Classifier #1 has 7 contiguous blocks over 695
months, classifier #2 has 13. Two heavily-smoothed block partitions of one timeline share the
axis itself as a common factor, so a positive association is expected *before* any shared
economics is invoked. ARI, NMI and Cramér's V cannot separate "these two classifiers see the
same market structure" from "these two classifiers are both slow".

This is not a reason to discount the measured numbers. It is a reason to measure the null.

### What the observed numbers are (measured, fixed, not to be recomputed to taste)

Measured on the labelings in force at the time of writing, n_compared = 695:

| statistic | observed |
|---|---|
| Adjusted Rand | 0.4686082735798639 |
| Normalized mutual information | 0.6023322651194867 |
| Cramér's V | 0.6558187277453474 |

Entropy decomposition of the same table: H(c1) = 1.3274, H(c2) = 1.5912, I(c1;c2) = 0.8790 nats.
H(c1|c2) = 0.4484, H(c2|c1) = 0.7122. So 66.2% of classifier #1's entropy is explained by #2 and
55.2% of #2's by #1 — neither a relabeling (both fractions would approach 1.0) nor a strict
refinement (H(c1|c2) would approach 0).

### The null

Preserve each labeling's own block-length structure exactly; randomise only the arrangement of
those blocks (permutation of block order and/or circular shift), independently for each labeling.
Recompute all three statistics per resample. This holds sojourn structure and occupancy fixed and
destroys only the *alignment* between the two labelings, which is precisely the quantity in
question. At least 2000 resamples, seeded and recorded.

### The rule — binding, and it decides the verdict mechanically

Read on **NMI**, the statistic whose flag the observed value exceeds and which is invariant to
label permutation. The other two are reported alongside and do not change the verdict.

- observed NMI **≤ 95th percentile** of the null → verdict **(b) failure to add an axis**.
  The association is what two block partitions of one timeline produce by construction.
- observed NMI **> 99th percentile** of the null → verdict **(a) added an axis**.
  The shared structure exceeds what blockiness alone explains.
- observed NMI **between the 95th and 99th percentiles** → **inconclusive**, recorded as
  inconclusive in exactly those terms, with no tie-break and no further tests run to break it.

No other reading is permitted. If the result is inconclusive it is reported as inconclusive; the
project does not then go looking for a fourth statistic that resolves it, which would be the
garden of forking paths this pre-registration exists to close.

### Scope note — this will be re-run

Glenn also directed that classifier #1 be re-pinned against §4.4 criterion 1, which it currently
breaches at both ends (state 0 at 1.5827%, 11 months; state 3 at 39.2806%). Re-pinning changes
every one of classifier #1's labels, so the observed numbers above become void and the dependence
measurement is re-run against the new labeling. **The rule above survives that re-run unchanged.**
It is a rule about how to read a control, not about these particular numbers, and re-stating it
after seeing new numbers would defeat its purpose.

---

## RESULT 2026-09-18 — re-measured after both re-pins; the rule returns INCONCLUSIVE

Both classifiers were re-pinned before this measurement, so the numbers in §3 and in the
PRE-REGISTRATION's "observed" table are **void** and are superseded here.

- classifier #1: K 5 → **6**, λ 52.0 → **10.0** (ADR-0001 § RE-PIN), under design §4.4's amended
  criterion 1 with the recurrence exemption.
- classifier #2: K 3 → **5**, λ 32.0 → **16.0** (ADR-0002 § RE-PIN).

### Re-measured dependence

n_compared = 695 (1963-02-28 → 2020-12-31), `suspicious = False`.

| statistic | previous (void) | **re-measured** |
|---|---|---|
| Adjusted Rand | 0.4686 | **0.354841** |
| Normalized mutual information | 0.6023 | **0.464088** |
| Cramér's V | 0.6558 | **0.589748** |

Every statistic fell. NMI no longer exceeds its 0.5 prose flag; Cramér's V still does.

### The control

2000 resamples, seed 20260918. Block counts actually used: classifier #1 **26**, classifier #2
**13**. Note #1's block count rose from 7 to 26 under the re-pin — the earlier measurement was
taken against a far blockier, near-degenerate labeling.

| statistic | observed | null p50 | p95 | p99 | max | observed percentile |
|---|---|---|---|---|---|---|
| Adjusted Rand | 0.354841 | 0.2164 | 0.3332 | 0.4086 | 0.4602 | 96.75 |
| NMI | 0.464088 | 0.3406 | 0.4492 | 0.5012 | 0.5532 | **96.60** |
| Cramér's V | 0.589748 | 0.5214 | 0.6178 | 0.6584 | 0.7261 | 87.70 |

**The null median NMI is 0.3406.** A substantial share of the raw association is temporal
blockiness, which is what the control was built to expose. Cramér's V — the statistic whose flag
the raw number exceeded — sits at the 87.70th percentile, comfortably inside the null, and on its
own evidences nothing.

### Verdict, by the rule committed at `298b1bc` before this code existed

> p95 0.449202 < observed NMI 0.464088 ≤ p99 0.501195 → **INCONCLUSIVE**

Recorded as **inconclusive**, in exactly those terms. Per the pre-registration: no tie-break, and
no further statistic is computed to resolve it. The association is elevated above what blockiness
alone explains, but not by enough to call, and the honest summary is that **this measurement
cannot decide whether classifier #2 added an axis.**

### What this does and does not license

- It does **not** license reading criterion 6 as satisfied. It is not satisfied; it is unresolved.
- It does **not** license a fourth dependence statistic, a different null, or a re-run at another
  seed. All three would be the garden of forking paths the pre-registration closed.
- It does **not** invalidate criterion 7. Joint allocation lift is a *separate* criterion with its
  own pre-planned measurement, asking a different and more decision-relevant question: does the
  blend beat classifier #1 alone? That is not a tie-break on dependence — it is the test the phase
  already intended to run, and its answer stands on its own footing whatever criterion 6 says.

### Known bias in the null, stated

`_shuffle_blocks` merges same-state blocks that land adjacent, so each resample is slightly
blockier than its input. That raises association by chance and shifts the null **up**, making it
harder for the observed value to clear. The bias is therefore one-directional and conservative:
it can only push a verdict toward (b), never manufacture an (a). An inconclusive result under a
null biased toward (b) is, if anything, mildly favourable to classifier #2 — which is stated here
rather than left for a reader to work out.
