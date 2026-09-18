---
phase: 07-regime-representation
plan: 10
artifact: plausibility-band evidence package
status: awaiting human decision
assembled: 2026-09-18
assembled_before: "plan 07-11's joint-lift measurement — no joint-lift number existed when this was written"
decided: null
---

# The Four Load-Bearing Plausibility Bands — Evidence for Decision

> **What this document is.** An evidence package, not a decision. Task 3 of plan 07-10 is a
> blocking human-verify checkpoint: Glenn decides **confirm / revise / retire** for each of the
> four bands. Nothing here asserts anything until the decision column in the closing table is
> filled.
>
> **Why now.** D-07 kept all four bands *advisory* through wave 1, so nothing turned on their
> values. `07-VALIDATION.md` makes criterion 7 judge joint lift against them, and makes
> confirming or revising all four an explicit gate. This document is deliberately written
> **before** plan 07-11 produces any joint-lift number, so the bands cannot be adjusted to fit a
> result already seen. That ordering is the point — `.planning/UAT-AUDIT-2026-09-09.md` indicts
> exactly the reverse.
>
> **What a plausibility band is (D-07, restated).** It asserts a number is *physically possible*.
> It never asserts a number is *good*. Audit item A11 — that no gate in this project can fail on
> a bad-but-working model, only on a broken one — stays open and conscious. Changing that is a
> design-freeze decision, not a wave-2 one.

---

## 0. Two findings that apply to all four bands

### 0.1 Every one of the four is `[ASSUMED]`. No measurement in this project derived any of them.

All four carry the `[ASSUMED]` tag in `07-VALIDATION-WAVE1.md`'s Plausibility Bands table and are
inherited unchanged by `07-VALIDATION.md`. Each was written by analogy or by arithmetic on
*other* bands, never fitted to or derived from a measurement. Their prior use is:

| Band | Prior use | Was anything gated on it? |
|---|---|---|
| `abs(wealth_delta) < 5` | wave 1 (plans 07-03 … 07-06), advisory | No — D-07 |
| `dd_delta ∈ [-0.5, 0.5]` | wave 1, advisory | No — D-07; a quality veto on `dd_delta` was offered in discussion and **declined** (ADR-0001 § Selection criterion) |
| `n_transitions > 30` implausible | wave 1, advisory | No — D-07 |
| `pct_disagree < 0.02` suspicious | wave 1, advisory — and it **caught a real bug** (§4.3) | No, but it would have |

### 0.2 Only two of the four are joint-lift quantities. The gate's framing over-reaches.

`07-VALIDATION.md`'s gate says "criterion 7 judges joint lift against them" for all four. Read
against what criterion 7 actually compares — the joint (#1 ⊕ #2) leg versus the #1-alone leg,
both from one harness differing only in `blend_weight_1` (plan 07-11) — that is true of
`wealth_delta` and `dd_delta` only.

- `n_transitions` describes **a labeling**, not a comparison of two backtest legs.
- `pct_disagree` describes **the filtered-vs-smoothed agreement of one labeler with itself**, not
  a lift.

Both remain worth deciding — they guard classifier #2's own labeling diagnostics in plan
07-11 — but a decision that treats them as joint-lift acceptance bounds would be applying them to
a comparison they cannot describe. Named here rather than left for a reader to notice.

---

## 1. Band: `abs(wealth_delta) < 5`

### 1.1 What the band asserts, in the quantity as actually computed

`evaluation/report.py:259` (and identically at `:988`):

```python
wealth_delta = strategy_kpis["terminal_log_wealth"] - ablation_kpis["terminal_log_wealth"]
```

and `evaluation/kpis.py:43`:

```python
def terminal_log_wealth(returns): return float(np.log1p(returns).sum())
```

**Units: a difference of terminal LOG wealths — natural-log units (nats), dimensionless.**
It is **not** a difference of wealth levels and **not** a percentage. A `wealth_delta` of +1.0
means the strategy leg ended at **e ≈ 2.718 times** the ablation leg's terminal wealth, not one
percent or one dollar more. A `wealth_delta` of 5 means a factor of **e⁵ ≈ 148×**.

This distinction is the whole reason the band exists — see §1.4.

### 1.2 Where the number came from

`[ASSUMED]`. `07-VALIDATION-WAVE1.md` records its derivation as: universal bound
`abs(x) < 15` obtained arithmetically from the two per-leg terminal-log-wealth bounds
(`[-3, 12]`, itself `[CITED]` from `06-VALIDATION.md`), and the domain band `abs(x) < 5` set
against "recorded history spans +0.073 to +0.379". **No measurement in this project derived 5.**
It is a round number roughly an order of magnitude above the largest value then on record.

### 1.3 Every value this project has measured against it

All from the strategy-minus-ablation leg pair over the **588-step walk-forward, first decision
1972-01-31, terminal 2020-12-31**:

| Value | Variant / window |
|---|---|
| **+0.3778473581475139** | frozen 10-column policy (accepted), 588 steps, 1972-01-31 → 2020-12-31, run 2026-09-14 |
| **+0.379267** | pre-fix 9-column stale-checkpoint compound baseline, same 588-step window |
| **+0.40094727234882077** | impute-13-column variant (rejected, isolated under `outputs/reports/platform/trials/impute-13col/`), same 588-step window |
| **+0.073 … +0.379** | the range `07-VALIDATION-WAVE1.md` cites as "recorded history" at band-writing time |

Largest magnitude ever recorded: **0.401**. The band is **12.5×** that.

### 1.4 What criterion 7 will ask of it

Criterion 7 applies it to `terminal_log_wealth(joint leg) − terminal_log_wealth(#1-alone leg)`,
both legs produced by `run_joint_backtest` differing only in `blend_weight_1` (plan 07-11).

A value outside the band has **two possible meanings, and they are not the same finding**:

1. **The strategy did something extreme.** A blend that concentrated into one asset class and
   compounded away from the baseline. Implausible for two long-only, 10%-vol-targeted legs over
   the same window, but not impossible.
2. **The measurement is broken.** This is what the band is for.
   `.planning/UAT-AUDIT-2026-09-09.md` §"The closure record compounds it" records
   `05-VERIFICATION.md` tabulating terminal log wealth improving **32.18 → 111.06** and
   presenting it as progress. **e¹¹¹ ≈ 10⁴⁸.** The phase closed by celebrating a move from 10¹³
   to 10⁴⁸. Two conclusions were drawn from that run and had to be re-opened.

**A band read against the wrong units would not have caught it.** If `wealth_delta` were read as
a *percentage* difference, 111 would read as "111% better" — implausible but arguable. Read
correctly as nats, it is a factor of 10⁴⁸ and unarguable. This is why §1.1 leads with units.

### 1.5 Proposed disposition

**CONFIRM at `abs(wealth_delta) < 5` for the domain tier.** Rationale: 5 nats is `e⁵ ≈ 148×`,
about 12.5× the largest magnitude ever recorded here and still small enough to catch an
order-of-magnitude arithmetic break like 111.06 several times over. **Tightening it toward the
observed 0.38 would convert a plausibility band into a quality band** — the precise move D-07 and
ADR-0001's Selection criterion refuse, and the one A11 keeps consciously open.

See §5 for the wide-versus-narrow resolution (`abs < 15` universal vs `abs < 5` domain).

---

## 2. Band: `dd_delta ∈ [-0.5, 0.5]`

### 2.1 What the band asserts, in the quantity as actually computed

`evaluation/report.py:260` (and identically at `:989`):

```python
dd_delta = strategy_kpis["max_drawdown"] - ablation_kpis["max_drawdown"]
```

`max_drawdown` comes from `kpis.py::max_drawdown_and_duration`, the peak-to-trough
`(cum / cum.cummax() - 1).min()` convention.

**Units: a difference of two drawdown FRACTIONS, each a negative number in `[-1, 0]`.**
`-0.2642` is a 26.42% drawdown. So `dd_delta` is in fractional units, and the band `[-0.5, 0.5]`
means **±50 percentage points of drawdown**. `report.py:273` formats it `{dd_delta:+.2%}`, so
the *printed* value carries a `%` sign while the *stored* value is a fraction — a reader
comparing the printed `-6.61%` against the stored band `0.5` is comparing two different scales.

**Arithmetic consequence, which matters in §2.5:** since each leg's `max_drawdown ∈ [-1, 0]`,
their difference is confined by construction to **`[-1, 1]`**.

### 2.2 Where the number came from

`[ASSUMED]`. `07-VALIDATION-WAVE1.md`: universal bound `x ∈ [-2, 2]`; domain band `abs(x) < 0.5`
justified as "both legs are vol-targeted, long-only, similarly levered; current value −0.014364".
No measurement derived either number.

### 2.3 Every value this project has measured against it

Same 588-step window, first decision 1972-01-31 → 2020-12-31:

| Value | Variant / window |
|---|---|
| **−0.0661242048614149** | frozen 10-column policy (accepted), 588 steps, 1972-01-31 → 2020-12-31 |
| **−0.014364** | pre-fix 9-column compound baseline, same window |
| **+0.01898849550676429** | impute-13-column variant (rejected, isolated), same window |

Supporting per-leg values, same window: strategy `max_drawdown` −26.42% over 58 months (frozen),
−21.24% over 33 months (pre-fix), −17.91% over 44 months (impute).

Largest magnitude ever recorded: **0.0661** (6.61 percentage points). The band is **7.6×** that.

### 2.4 What criterion 7 will ask of it

Applied to `max_drawdown(joint leg) − max_drawdown(#1-alone leg)`. Outside the band means
either:

1. **The strategy did something extreme** — one leg spent a crisis fully invested while the other
   went to cash. For two legs sharing a 10% annualized vol target, a 50pp drawdown gap would
   require the vol targeting to have failed on one of them.
2. **The measurement is broken** — a leg whose equity curve is not a return series, a
   `max_drawdown` computed on levels rather than returns, or a window mismatch between the legs.
   Threat T-07-26 names the window-mismatch case specifically.

### 2.5 Proposed disposition

**Two parts.**

**(a) RETIRE the universal bound `[-2, 2]` and replace it with the arithmetic bound `[-1, 1]`.**
Arithmetic: `max_drawdown ∈ [-1, 0]` for each leg, so
`dd_delta = a − b ∈ [-1 − 0, 0 − (−1)] = [-1, 1]`. **A bound of ±2 is wider than the quantity's
own arithmetic range and therefore cannot fail — it can only confirm.** That is the same
evidence-shape failure `UAT-AUDIT-2026-09-09` indicts and the same class as criterion 8's
`grep -v platform` exit check and the invented 5% occupancy floor's missing cap. Replacing it
with `[-1, 1]` makes the bound *definitional*: a value outside it is not an extreme strategy, it
is proof that one leg's `max_drawdown` is not a fraction in `[-1, 0]`, i.e. the KPI is broken.
**What now catches what `[-2, 2]` was watching for:** nothing is lost, because `[-2, 2]` caught
nothing — every value it admits that `[-1, 1]` rejects is arithmetically unreachable.

**(b) CONFIRM the domain band at `abs(dd_delta) < 0.5`.** Rationale: 50 percentage points is 7.6×
the largest magnitude recorded and, for two long-only legs sharing a 10% vol target over one
window, is only reachable if the vol targeting is not being applied to one of them. As with §1.5,
tightening toward the observed 6.6pp would make it a quality gate.

---

## 3. Band: `n_transitions > 30` treated as implausible

### 3.1 What the band asserts, in the quantity as actually computed

There are **two different quantities** wearing this name in this project, and the band has
already been read against both.

**(a) `n_transitions` as computed in `evaluation/sojourn_lag.py:185`:**

```python
n_transitions = sum(len(p) for p in transitions_by_state.values())
n_resolved    = len(resolved)
```

This counts **reference-labeling state transitions that fall inside the evaluated comparison
window and are candidates for detection-lag resolution**, with `n_resolved ≤ n_transitions` by
construction. **Units: a count of transition events, dimensionless.** It is a property of the
*comparison window*, not of the labeling as a whole — ADR-0001 states explicitly that the frozen
policy's "7 of 7" means 7-of-7-available-within-the-narrower-356-month-window, not 7-of-7 over
the full 588-step decision range.

**(b) The full-sample labeling's run-length change count** — the number of months at which
`state[t] != state[t-1]` across the whole labeling. `07-08-SUMMARY.md` quotes classifier #2's
"3 transitions (1974-01-31, 1982-12-31, 1998-09-30)" in this second sense.
**Units: also a count, but over a different population** — every month of the labeling, not the
months inside an evaluation comparison window.

`07-VALIDATION-WAVE1.md`'s own justification mixes the two: it places the band on the
`n_resolved` / `n_transitions` row (sense **a**) while justifying it "by analogy to the 4–6
transitions observed historically... at K=5, λ=52" (sense **b**). **A band read against the wrong
population is the §1.4 failure in a different costume.**

### 3.2 Where the number came from

`[ASSUMED]`. Verbatim from `07-VALIDATION-WAVE1.md`: *"implausible if `> 30` over 588 months at
K=5, λ=52 — by analogy to the 4–6 transitions observed historically; a high count signals
over-segmentation / λ not applied."*

**Its stated calibration point no longer exists.** Both classifiers were re-pinned on
2026-09-18: classifier #1 from K=5, λ=52.0 to **K=6, λ=10.0** (ADR-0001 § RE-PIN); classifier #2
from K=3, λ=32.0 to **K=5, λ=16.0**. λ = 10 is **one fifth** of the λ the band was calibrated
against, and λ is exactly the jump penalty whose absence the band claims to detect.

### 3.3 Every value this project has measured against it

**Sense (a) — `sojourn_lag`'s within-window count**, all from the 588-step evaluations of
2026-09-14:

| `n_resolved` of `n_transitions` | Variant / window |
|---|---|
| **7 of 7** | frozen 10-column policy, resolved within the 356-month comparison window 1974-02-28 → 2017-05-31 |
| **4 of 6** | pre-fix 9-column baseline, 470-month comparison window 1974-02-28 → 2020-12-31 |
| **3 of 6** | impute-13-column variant (rejected, isolated), 470-month window |

**Sense (b) — full-sample labeling run-length changes.** Measured live on **2026-09-18** from the
git-tracked checkpoints `data/checkpoints/platform/regime_labels.parquet` and
`regime_labels_2.parquet`, both written after the 2026-09-18 re-pins:

| Labeling | Transitions | Window |
|---|---|---|
| **Classifier #1**, K=6, λ=10.0 | **25** | 695 months, 1963-02-28 → 2020-12-31 |
| **Classifier #2**, K=5, λ=16.0 | **12** | 696 months, 1963-01-31 → 2020-12-31 |
| Classifier #2, K=3, λ=32.0 (**superseded**) | 3 | 696 months, same window — `07-08-SUMMARY.md` |

The occupancy vectors recomputed in the same read reproduce ADR-0001 § RE-PIN's
`5.7554 / 32.8058 / 10.2158 / 28.7770 / 12.0863 / 10.3597 %` and ADR-0002's
`16.67 / 22.70 / 22.41 / 23.85 / 14.37 %` exactly, so the two counts above are from the same
artifacts those documents describe.

**Classifier #1's 25 is inside a band of 30 — with 17% of headroom, at a λ five times smaller
than the one the band was calibrated against.** A single further λ reduction would breach it,
and nothing in the band's text says whether that breach would mean anything.

### 3.4 What criterion 7 will ask of it

Per §0.2, criterion 7's joint-lift comparison does not produce an `n_transitions`. What plan
07-11 will produce is a §5.4 sojourn/detection-lag readout for classifier #2 alongside
classifier #1's, and each carries a `n_resolved` / `n_transitions` pair.

Outside the band means either:

1. **Over-segmentation is real** — λ is too small and the labeling is chattering. This is a
   labeling finding, not a lift finding.
2. **The measurement is broken** — λ is not reaching the fit at all (the exact defect ADR-0001
   § RE-PIN found: λ = 52 had been computed on the 13-member lean set while the fit reads 10
   frozen columns, so the penalty was scaled to a feature set the model never sees), or the
   transition counter is running on the wrong population.

A fixed count of 30 cannot separate these because **it is length-dependent**: the same
segmentation *rate* breaches or clears it purely according to how many months the labeling
covers.

### 3.5 Proposed disposition

**REVISE, and split the band in two by the quantity it governs.** Arithmetic shown.

**(a) `n_transitions` (`sojourn_lag`, within-window):** replace `> 30` with the **definitional**
bound `n_resolved ≤ n_transitions ≤ n_label_transitions`. You cannot resolve more transitions
than the comparison window contains, and the window cannot contain more than the labeling has.
Recorded values 7, 6 and 6 all satisfy it. A breach is arithmetically a counting bug, never a
strategy outcome.

**(b) `n_label_transitions` (full-sample labeling):** replace the fixed `> 30` with the **rate**
band `n_label_transitions > 0.10 × n_months` is implausible — more than one state change per ten
months on average.

Arithmetic:

| Labeling | Transitions | Months | Rate | Proposed threshold `0.10 × n_months` |
|---|---|---|---|---|
| Classifier #1, K=6, λ=10 | 25 | 695 | **3.60%** | 69.5 |
| Classifier #2, K=5, λ=16 | 12 | 696 | **1.72%** | 69.6 |
| Classifier #2, K=3, λ=32 (superseded) | 3 | 696 | 0.43% | 69.6 |

Why 10%: design §4.4 criterion 2 requires no state to have a median sojourn below **3 months**. A
labeling honouring that cannot average more than one transition every three months, i.e. 33%; a
10% rate is three times more permissive than criterion 2's own floor implies, so it is a
plausibility bound rather than a quality one and cannot fail on a merely-chatty-but-valid
labeling. The old fixed 30 corresponds to **4.3%** at 695 months — *tighter* than the proposed
rate at this length, and looser at any shorter one. Replacing a length-dependent count with a
rate removes that dependence.

**What now catches what `> 30` was watching for:** the rate band catches over-segmentation and
λ-not-applied at any labeling length, which the fixed count did only at 588–695 months. Nothing
is lost.

---

## 4. Band: `pct_disagree < 0.02` treated as suspicious

### 4.1 What the band asserts, in the quantity as actually computed

From `evaluation/disagreement.py` via `label_disagreement`, reported as the pair
`{"n_compared": int, "pct_disagree": float}`.

**Units: a fraction in `[0, 1]`** — the share of *compared* months on which the walk-forward
*filtered* labeling differs from the full-sample *smoothed* reference labeling.
**`n_compared` is its denominator and is not optional context**: `pct_disagree` read without it
is not a reading. ADR-0001's own consequences table foregrounds this — 0.8277 (n=470) and 0.8090
(n=356) are "different populations, spanning different date ranges", not a 1.9-point improvement.

This band is the project's **third check class**: not "an impossible number" but "a number that
*looks* resolved and is actually a different bug".

### 4.2 Where the number came from

`[ASSUMED]`. `07-VALIDATION-WAVE1.md`: *"a hindsight full-sample fit vs. a walk-forward per-step
fit should still disagree somewhat under one feature space. Near-zero is as suspicious as 82.8%
was bad; it suggests the driver is seeing full-sample data."* No measurement derived 0.02.

### 4.3 Every value this project has measured against it

| Value | `n_compared` | Window | Variant |
|---|---|---|---|
| **0.8089887640449438** | 288 of 356 | 1974-02-28 → 2017-05-31 | frozen 10-column (accepted) |
| **0.8276595744680851** | 389 of 470 | 1974-02-28 → 2020-12-31 | pre-fix 9-column baseline |
| **0.8319148936170213** | n = 470 | 1974-02-28 → 2020-12-31 | impute-13-column (rejected, isolated) |
| **0.0** | **n_compared = 0** | — | the **bug**, §4.4 below |

**This band has already earned its keep.** `07-VALIDATION-WAVE1.md` § "One band earned its keep
already" records that passing the persisted `state_N` **strings** to `label_disagreement` returns
`{'n_compared': 0, 'pct_disagree': 0.0}` **silently, with no exception** — a reading of "0%
disagreement, A13 totally fixed" produced by comparing nothing at all. Reproduced at plan time;
now additionally pinned by `test_platform_evaluation_disagreement` asserting `n_compared` is
non-zero.

### 4.4 What criterion 7 will ask of it

Per §0.2, not a joint-lift quantity. It governs classifier #2's own filtered-vs-smoothed
readout, if plan 07-11 computes one — and ADR-0002's L2-window decision makes that readout's
sample size a live concern already.

Outside the band (i.e. *below* 0.02) means either:

1. **The two labelings genuinely agree** — a real and surprising finding, which under D-15's
   posture is reported, not gated.
2. **The measurement is broken** — and this is the recorded case, twice over in shape: an empty
   comparison returning 0.0, or the driver seeing full-sample data (look-ahead), which would make
   the filtered fit converge on the smoothed one for the wrong reason.

The failure actually observed produced `pct_disagree = 0.0` through `n_compared = 0`. **The
threshold alone would have flagged it, but only because 0.0 < 0.02 by coincidence of an empty
average.** A comparison of two months, one disagreeing, gives 0.5 and sails past.

### 4.5 Proposed disposition

**CONFIRM the 0.02 threshold and REVISE the band's statement to carry its denominator.**
The band becomes:

> `pct_disagree` is suspicious if `pct_disagree < 0.02` **OR** `n_compared == 0` **OR**
> `n_compared` is materially below the decision-range month count without a recorded reason.

Rationale: the threshold is sound — an 80%-disagreement history makes a sub-2% reading
extraordinary — but the bug it caught was a *denominator* failure that the threshold caught only
incidentally. Writing `n_compared` into the band makes the guard fire on the mechanism rather
than on a coincidence. The third clause is what ADR-0001's own window-narrowing limitation
(232 of 588 steps L2-degraded, leaving 356) would have triggered had it been in force; that
narrowing was found by hand, not by a band.

**No arithmetic revision to 0.02 is proposed** — no measurement in this project bears on where
between 0 and 0.80899 the suspicion threshold belongs, and inventing one would be exactly the
`[ASSUMED]`-masquerading-as-measured move this document exists to avoid.

---

## 5. The wide-versus-narrow tension, named and a resolution proposed

`07-VALIDATION.md`'s wave-2 Plausibility Bands table lists, for the **joint-vs-#1 deltas**:

> `abs(wealth_delta) < 15`, `dd_delta ∈ [-2, 2]` — **[CITED]** universal / **[ASSUMED]** domain

Twelve lines further down, the ⚠ GATE names:

> `abs(wealth_delta) < 5` · `dd_delta ∈ [-0.5, 0.5]` · …

**Two different numbers on the same two quantities, in one document.** Left as-is, criterion 7
could be reported as passing against `< 15` and failing against `< 5` on the same measurement,
and both statements would cite `07-VALIDATION.md`.

**They are not in fact rival bands — they are two tiers with different jobs**, and the tension is
that the document never says which governs a *verdict*. `07-VALIDATION-WAVE1.md`'s table has both
in adjacent columns headed "Universal bound" and "Domain band"; `07-VALIDATION.md` reproduces only
one tier in the table and only the other in the gate.

### Proposed resolution — one governing value per quantity

| Tier | Job | Failure means | Governs criterion 7's verdict? |
|---|---|---|---|
| **Universal / arithmetic** | Is the number physically possible for this quantity? | The **measurement is broken**. Halt and fix; do not report a lift. | **YES — this is the governing tier** |
| **Domain / advisory** | Is the number within what this stack has ever produced? | **Record a note** naming the value, the window, and which of §1.4's two meanings is more likely. Criterion 7 still reports. | No |

Applied:

| Quantity | Governing (universal) value for criterion 7 | Advisory (domain) trigger |
|---|---|---|
| `wealth_delta` | `abs(x) < 15` | `abs(x) ≥ 5` → recorded note |
| `dd_delta` | `x ∈ [-1, 1]` (**revised** from `[-2, 2]`, §2.5a) | `abs(x) ≥ 0.5` → recorded note |

This keeps D-07's posture exactly: the governing tier can only fail on a *broken* measurement,
never on a bad-but-working model, so **A11 stays open and conscious** rather than being quietly
closed by promoting a domain band to a gate. If Glenn wants the domain tier to gate instead, that
is a deliberate reversal of D-07 and A11 and should be recorded as one.

---

## 6. Limitations this document must not paper over

Two facts about classifier #1's labeling bear on any band touching drawdown or transition counts,
and are recorded here because `07-BANDS.md` is where criterion 7's reader will look:

1. **The crisis state's median sojourn is 3.0 months — exactly on design §4.4 criterion 2's
   boundary** (`no state with median < 3 months`), against design §5.4's typical **1–3 month**
   detection lag. The lag consumes most or all of the sojourn. **This labeling identifies crises
   *ex post*; whether L2 can nowcast them in time to act is a separate and unanswered question,
   and no band in this document answers it.** Any `dd_delta` or transition-count reading on the
   joint leg inherits that caveat: a drawdown avoided in the labeling is not a drawdown avoided
   in real time.

2. **Design §4.4 criterion 3's formal subsample-stability test (Hungarian matching on emission
   distributions) has never been run for either classifier.** The nine-episode recurrence
   recorded in ADR-0001 § RE-PIN is strong evidence for the crisis state *specifically*, and is
   not a substitute for the test across all states. No band here rests on criterion 3 having
   passed, and none should be read as implying it has.

3. **Criterion 6 is UNRESOLVED** — neither met nor failed. The pre-registered block-permutation
   control returned INCONCLUSIVE (observed NMI 0.464088 at the 96.60th percentile; p95 0.4492,
   p99 0.5012), and the pre-registration at `298b1bc` forbids tie-breaks. **Nothing in this
   document should be read as establishing that classifier #2 adds an independent axis.** The
   bands decided here govern how a joint-lift *number* is judged; they say nothing about whether
   the second axis exists.

---

## 7. Closing table — for the developer's decision

**Nothing in the "Developer's decision" column is filled by an agent.** Task 3 of plan 07-10 is a
blocking human-verify checkpoint. `07-VALIDATION.md` makes confirming or revising all four a gate
before any joint-lift number is assessed, and three of four decided is not decided.

| # | Band | Current value | Proposed disposition | Proposed new value | Developer's decision |
|---|---|---|---|---|---|
| 1 | `wealth_delta` (nats, terminal log wealth, strategy − ablation) — domain | `abs(x) < 5` | **Confirm** | `abs(x) < 5` (unchanged) | |
| 1u | `wealth_delta` — universal / governing | `abs(x) < 15` | **Confirm**, and designate as the tier governing criterion 7's verdict (§5) | `abs(x) < 15` (unchanged) | |
| 2 | `dd_delta` (fraction of peak, strategy − ablation) — domain | `abs(x) < 0.5` | **Confirm** | `abs(x) < 0.5` (unchanged) | |
| 2u | `dd_delta` — universal / governing | `x ∈ [-2, 2]` | **Revise** — unreachable, cannot fail (§2.5a) | `x ∈ [-1, 1]` (arithmetic: each leg's `max_drawdown ∈ [-1, 0]`) | |
| 3a | `n_transitions` (`sojourn_lag`, within comparison window) | `> 30` implausible | **Revise** — wrong population, and the band's stated K=5/λ=52 calibration no longer exists (§3.2) | `n_resolved ≤ n_transitions ≤ n_label_transitions` (definitional) | |
| 3b | `n_label_transitions` (full-sample labeling run-length changes) | `> 30` implausible | **Revise** — a fixed count is length-dependent (§3.5b) | `> 0.10 × n_months` implausible (69.5 at 695 months; measured 25 → 3.60% and 12 → 1.72%) | |
| 4 | `pct_disagree` (fraction of compared months) | `< 0.02` suspicious | **Confirm the threshold, revise the statement** to carry its denominator (§4.5) | `< 0.02` **OR** `n_compared == 0` **OR** `n_compared` materially below the decision-range month count without a recorded reason | |
| 5 | Wide-vs-narrow governance (§5) | both tiers in play, neither designated | **Resolve** — universal tier governs the verdict, domain tier records a note | one governing value per quantity: `wealth_delta` `abs < 15`; `dd_delta` `∈ [-1, 1]` | |

**Decision date (to be filled at Task 3, and it must predate plan 07-11's first joint-lift run):**

---

*Assembled 2026-09-18 by plan 07-10 Task 2, before any joint-lift number existed. Sources:*
*`07-VALIDATION.md`, `07-VALIDATION-WAVE1.md`, `07-MEASUREMENTS.md`,*
*`.planning/UAT-AUDIT-2026-09-09.md`, `07-CONTEXT.md` D-06/D-07,*
*`platform_design/adr/0001-l1-feature-policy.md` (incl. § AMENDMENT 2026-09-17, § ADDENDUM and*
*§ RE-PIN 2026-09-18), `platform_design/adr/0002-l1-second-classifier.md`, `07-08-SUMMARY.md`,*
*`07-09-SUMMARY.md`, `src/trading_crab_lib/platform/evaluation/kpis.py`,*
*`evaluation/report.py`, `evaluation/sojourn_lag.py`, and a live read of*
*`data/checkpoints/platform/regime_labels{,_2}.parquet` on 2026-09-18.*

---

## 8. DECISIONS — Glenn, 2026-09-18 (plan 07-10 Task 3, human-verify gate)

Recorded before any joint-lift number exists, as the gate requires.

| # | Band | Disposition |
|---|---|---|
| 1 | `abs(wealth_delta) < 5` | **CONFIRM** as the *domain* tier. Not tightened toward the observed 0.38 — that would convert a plausibility band into a quality band, which D-07 and ADR-0001's Selection criterion refuse. |
| 2 | `dd_delta` universal `[-2, 2]` | **REVISE to `[-1, 1]`.** The old bound was wider than the quantity's own arithmetic range (`max_drawdown ∈ [-1, 0]` per leg ⇒ difference ∈ `[-1, 1]`) and therefore **could only confirm**. The revised bound is definitional: a breach proves a leg's `max_drawdown` is not a fraction in `[-1, 0]`, i.e. the KPI is broken. Domain band `abs(dd_delta) < 0.5` **CONFIRMED** unchanged. |
| 3 | `n_transitions > 30` | **REVISE and SPLIT by quantity.** (a) `sojourn_lag` within-window count → definitional `n_resolved ≤ n_transitions ≤ n_label_transitions`; a breach is a counting bug, never a strategy outcome. (b) full-sample labeling → **rate** band, implausible above `0.10 × n_months`. A fixed count is length-dependent; a rate is not, and the original's calibration point (K=5, λ=52) no longer exists. |
| 4 | `pct_disagree < 0.02` | **CONFIRM 0.02 unrevised; REVISE the statement to carry its denominator.** Now suspicious if `pct_disagree < 0.02` **OR** `n_compared == 0` **OR** `n_compared` materially below expectation without a *recorded* reason. No number was invented for the threshold — nothing measured in this project bears on where between 0 and 0.80899 it belongs. |

### Governance — which tier decides criterion 7

| Tier | Job | A breach means | Governs the verdict? |
|---|---|---|---|
| Universal / arithmetic | Is this number physically possible for this quantity? | The **measurement is broken** — halt, do not report a lift. | **YES** |
| Domain / advisory | Is it within what this stack has produced? | **Record a note** with the value and window; criterion 7 still reports. | No |

| Quantity | Governing value | Advisory trigger |
|---|---|---|
| `wealth_delta` | `abs(x) < 15` | `abs(x) ≥ 5` → note |
| `dd_delta` | `x ∈ [-1, 1]` (revised) | `abs(x) ≥ 0.5` → note |

**A11 stays open and conscious.** The governing tier can only fail on a broken measurement, never
on a bad-but-working model. Glenn declined to promote the domain tier to a gate, which would have
closed A11 by gating on `[ASSUMED]` numbers.

### Implemented in code by this decision

Band 4's denominator clauses are live in `platform/evaluation/disagreement.py`, not left as prose.
The defect they fix was real: `n_compared == 0` previously **logged** a warning but set
`suspicious = False`, so a caller reading the flag rather than the log saw "not suspicious" on the
one case the band exists to catch. Five tests pin it, including that a *recorded* reason suppresses
the coverage clause while an unexplained shortfall does not.

Bands 1–3 are contracts for plan 07-11 to implement at the point criterion 7 is measured; no code
computes `wealth_delta` or `dd_delta` bounds yet.
