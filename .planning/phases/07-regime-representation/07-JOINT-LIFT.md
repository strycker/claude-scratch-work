---
phase: 07-regime-representation
plan: 11
artifact: criterion-7 joint allocation lift — the measured record
measured: 2026-09-21
routing: L1_ONLY_LAST_FILTERED_STATE (ADR-0002 decision (e), decision-bearing)
status: measured
---

# Criterion 7 — Joint (#1 × #2) Allocation Lift, Measured

> **The organizing rule of this document.** Every number appears with the window it
> was measured on, in the same sentence or table cell. This is the binding condition
> wave 1's UAT attached to criterion 3 and which the orchestrator extends to
> criterion 7. A reader must not be able to see a delta without seeing its step count
> and its date range.
>
> **D-06 applies.** Criterion 7 asks whether lift was *measured*, not whether it was
> *positive*. A negative, honestly measured and clearly windowed, satisfies it.

---

## 1. The routing, stated before any number

**Chosen routing: `L1_ONLY_LAST_FILTERED_STATE` — decision-bearing** (ADR-0002
decision (e), pinned 2026-09-17, before any run).

Each classifier's per-step probability vector is a **degenerate one-hot on its own
last filtered state**: the state it assigns to the most recent month of
`train_index`, which `expanding_steps` guarantees is strictly before the decision
date. No L2 nowcaster is consulted on this path. Concretely, at decision date *t*:

- classifier #1 is refit by `driver.py::_refit_l1` on `train_index` only, on
  ADR-0001's ten frozen columns;
- classifier #2 is refit on `train_index` only, on its own eight frozen columns,
  with `sort_column="rs_equities_bonds"`;
- each contributes `{last_state: 1.0}` to `allocation/joint_tilt.py::blend_regime_tilts`,
  which applies ADR-0001 condition (iv)'s partial pooling before either tilt is formed.

**What this routing means for the measurement window.** The L1-only path runs no L2
refit, so wave 1's dominant degradation mechanism — an early small post-embargo
window starving a K-fold — is absent. Both legs ran **588 of 588 steps with 0
degraded steps, 1972-01-31 → 2020-12-31**.

**⚠ These deltas are NOT comparable to wave 1's figures**, on two independent
grounds, and any table placing them side by side without saying so is misreporting:

1. **Different comparison.** Wave 1's `wealth_delta` **+0.377847** and `dd_delta`
   **−0.066124** are *strategy minus no-regime ablation*. Criterion 7's are *joint
   (#1 × #2) minus classifier-#1-alone*. These are different subtractions.
2. **Different measurement path and window.** Wave 1's were measured **through L2, on
   356 steps ending 2017-05**, against a 470-step ablation ending 2020-12. Criterion
   7's #1-alone baseline is **re-run through this harness** on 588 steps ending
   2020-12; it is never reused from wave 1.

**The L2 leg is computed and reported in §7 for human reading only.** Per ADR-0002's
firewall clause it was appended with the `NO_REGISTRY` sentinel, contributes zero
registry rows, counts toward neither D-16 nor D-17, and **nothing downstream in this
phase may change on the basis of it**.

---

## 2. The two legs, side by side — decision-bearing (L1-only routing)

Both legs come from one harness, `backtest/joint_driver.py::run_joint_backtest`,
differing only in `blend_weight_1`. `blend_weight_1 = 1.0` **is** the #1-alone leg:
same loop, same refits, same cost model, same condition-(iv) shrinkage path.

| | classifier #1 alone | joint (#1 × #2) |
|---|---|---|
| `blend_weight_1` | 1.00 | 0.50 (ADR-0002 (f), never swept) |
| **Window** | **588 steps, 1972-01-31 → 2020-12-31** | **588 steps, 1972-01-31 → 2020-12-31** |
| Degraded steps | 0 of 588 | 0 of 588 |
| Terminal log wealth (nats, 588 steps 1972-01-31 → 2020-12-31) | **4.718723** | **4.595285** |
| Max drawdown (588 steps 1972-01-31 → 2020-12-31) | **−28.1442%**, 47 months underwater | **−25.7358%**, 40 months underwater |
| Annualized Sharpe (588 monthly obs, 1972-01-31 → 2020-12-31) | **0.917073** | **0.914903** |
| Mean monthly turnover (588 steps, same window) | 0.163251 | 0.119788 |
| Total transaction cost (588 steps, same window) | 0.095992 | 0.070435 |
| Mean invested scale (588 steps, same window) | 0.934882 | 0.942452 |
| Worst month (in window) | −16.5950% (2008-10-31) | −15.4201% (2008-10-31) |
| Best month (in window) | +53.1696% (1974-01-31) | +53.0926% (1974-01-31) |
| Return skew / raw kurtosis (588 obs, same window) | 7.395819 / 125.806389 | 7.584288 / 135.569083 |
| Classifier #1 filtered-state changes (588 steps, same window) | 246 | 246 |
| Classifier #2 filtered-state changes (588 steps, same window) | 24 | 24 |
| Registry tag | `07-11-c1-alone-L1only` | `07-11-joint-c1xc2-L1only` |

The two legs' equity-curve indexes are **equal element-wise** (verified in-harness and
re-verified independently from the persisted parquet at
`outputs/reports/platform/joint_lift/`), and their degraded-step *sets* are identical
by construction: neither refit depends on `blend_weight_1`, and a step degrades if
*either* classifier fails, so the two legs cannot diverge in which months they hold.

---

## 3. The lift, both axes, each with its window inline

| Axis | Value | Window (inline) | Governing (universal) band | Inside? | Advisory (domain) trigger | Note fired? |
|---|---|---|---|---|---|---|
| `wealth_delta` = joint − baseline terminal log wealth | **−0.123438 nats** | **588 steps, 1972-01-31 → 2020-12-31** | `abs(x) < 15` | **yes** | `abs(x) ≥ 5` | no |
| `dd_delta` = joint − baseline max drawdown | **+0.024084** (+2.41 percentage points, i.e. the joint leg's drawdown is *shallower*) | **588 steps, 1972-01-31 → 2020-12-31** | `x ∈ [−1, 1]` (**revised**, 07-BANDS.md §8 band 2) | **yes** | `abs(x) ≥ 0.5` | no |

**Both governing bands hold, so the measurement is not broken and criterion 7
reports.** Neither advisory trigger fired, so no domain note is recorded. Per
07-BANDS.md §8's governance table, the universal tier decides; the domain tier would
only have produced a recorded note, never a halt.

**Reading the two numbers in the units they are actually in:**

- `wealth_delta = −0.123438` nats means the joint leg ended at **e^(−0.123438) ≈
  0.8839×** the #1-alone leg's terminal wealth over those 588 months — an **11.61%
  shortfall in terminal wealth**, not 0.12% and not 12.3 percentage points of return.
  (07-BANDS.md §1.1: this is a difference of *terminal log wealths*, in nats.)
- `dd_delta = +0.024084` is a difference of two drawdown *fractions*. The joint leg's
  worst peak-to-trough loss was **2.41 percentage points shallower** (−25.74% vs
  −28.14%) and its longest underwater stretch **7 months shorter** (40 vs 47), over
  the same 588 steps.

**Costs are not the explanation for the wealth shortfall.** The joint leg paid
**0.025557 less** in total transaction cost over the same 588 steps (0.070435 vs
0.095992) and still ended **0.123438 nats lower**. The shortfall is in the gross
allocation the blend produced, not in what it cost to hold it.

**The ablation is real, not a no-op.** The two legs' monthly returns differ in **586
of 588 months**. The harness is additionally pinned by
`test_weight_one_equals_classifier_one_alone`, which re-runs at `blend_weight_1 = 1.0`
with a *different* classifier-#2 signal and asserts the equity curve is unchanged.

---

## 4. The deflated Sharpe, for both legs

`sharpe_variance` and the trial count were read live; the verdict strings are
`format_dsr_verdict`'s own output, quoted verbatim.

| Leg | Observed Sharpe (588 obs, 1972-01-31 → 2020-12-31) | `n_trials` | Trial count read at (UTC) | `sharpe_variance` | DSR | Verdict (verbatim) |
|---|---|---|---|---|---|---|
| classifier #1 alone | 0.917073 | **42** | 2026-09-21T14:24:23.984933Z | 1.0 | **2.28151 × 10⁻¹²** | "Deflated Sharpe ratio 0.0000 does not clear the multiple-testing hurdle — statistically indistinguishable from a skill-less discovery given the number of trials searched." |
| joint (#1 × #2) | 0.914903 | **42** | 2026-09-21T14:24:23.984933Z | 1.0 | **1.46904 × 10⁻¹¹** | "Deflated Sharpe ratio 0.0000 does not clear the multiple-testing hurdle — statistically indistinguishable from a skill-less discovery given the number of trials searched." |

**Neither leg clears the multiple-testing hurdle.** Both DSRs are far below 0.5. The
joint leg does not clear it and the #1-alone leg does not clear it. There is no target
to fall short of — `07-VALIDATION.md` sets none for this number, precisely so there is
nothing to miss narrowly.

**Deflation does not treat the two legs differently in any way that matters.** Their
DSRs differ by an order of magnitude in a range where both round to 0.0000; the joint
leg's raw Sharpe (0.914903) is *lower* than the baseline's (0.917073) over the same
588 steps, so the blend did not help on this axis either, before deflation or after.

### 4.1 What the DSR number here is, and is not

Stated separately so it qualifies nothing above. The verdict stands as written.

- `expected_max_sharpe(42, 1.0) = 2.208694`. The hurdle scales with
  `sqrt(sharpe_variance)`, and `sharpe_variance` here is the **1.0 placeholder**, not a
  measured quantity: `registry_sharpe_variance()` logged "only 0 usable Sharpe
  observation(s) found (need >= 2)". This is the declared degenerate-case policy in
  `07-DSR-ESTIMATOR-NOTE.md` §4, recorded long before these runs — it is a property of
  the estimator, not a caveat invented for this result.
- **No `sharpe` key was written into this plan's two registry rows, deliberately.**
  Writing these two near-identical legs' Sharpes would flip `registry_sharpe_variance`
  from the placeholder to a sample variance of order 10⁻⁶, collapsing
  `expected_max_sharpe` to roughly zero and **silently disabling the multiple-testing
  correction for every future DSR in this project** — the "systematically
  under-penalize search" direction `07-RESEARCH.md` names as this project's closest
  analog to a security defect. The Sharpes are reported here and in
  `outputs/reports/platform/joint_lift/measurement_l1only.json` instead. Whether the
  registry should carry Sharpe metrics at all is an ADR-0002 amendment, not a script
  edit, and is **left open**.

---

## 5. The trial arithmetic

| Item | Value | Source |
|---|---|---|
| `total_trial_count()` **before** both tagged runs | **40** | live read, 2026-09-21T14:18:34.285851Z |
| `total_trial_count()` **after** both tagged runs | **42** | live read, 2026-09-21T14:24:23.984933Z |
| Rows added by this plan | **2** (one per tagged run) | difference of the two live reads |
| Rows added by plan 07-07's INV-01 screen | 2 (38 → 40) | ADR-0002 § Trial ceiling, component 1 |
| ADR-0002 ceiling for the remainder of this phase | **44** (= 40 + 4) | ADR-0002 § Trial ceiling |
| Ceiling respected? | **yes — 42 ≤ 44**, with 2 rows of the budget unspent | in-script assertion; the run refuses to report otherwise |

This harness appends **exactly one** row per call, not two:
`run_full_backtest_evaluation`'s factor of two comes from its own two `append_trial`
sites (strategy + ablation) and does not apply here. ADR-0002's component-2 ceiling of
4 therefore over-budgets this harness by 2 rows; **spending fewer rows than the
ceiling is the only direction a ceiling permits.**

**Sentinel (`NO_REGISTRY`) runs performed, stated even though they added nothing:**
**6 `run_joint_backtest` invocations across 3 script invocations** — one two-leg
`--dry-run` wiring verification under the L1-only routing before any tagged run, and
two two-leg L2 observational runs (the second re-run after a degrade-attribution fix,
§7). All added **0** rows; the count moved 40 → 42 and no further. Unit tests use
`tmp_path` ledgers or the sentinel exclusively and can never touch the live ledger.

**No configuration was tuned, varied or re-run on seeing a result.** The tagged runs
were executed once, at the pinned `blend_weight_1 = 0.50`, and reproduce the earlier
sentinel dry run bit-for-bit (`fit_jump_model` is seeded, `random_state=42`) — which
is itself the evidence that nothing moved between the wiring check and the record.

---

## 6. Read against criterion 6

**Criterion 6's recorded judgement (07-DEPENDENCE.md, 2026-09-18):** the
pre-registered block-permutation control returned **INCONCLUSIVE** — observed NMI
0.464088 at the 96.60th percentile, against p95 0.449202 and p99 0.501195 — and the
pre-registration at `298b1bc` forbids tie-breaks. **Criterion 6 is UNRESOLVED: neither
met nor failed.**

How that conditions this result: **neither of the two cases plan 07-11 anticipated
occurred.** It anticipated (a) positive lift alongside high dependence — the blend
merely re-weighting classifier #1's own signal — or (b) positive lift alongside low
dependence, the result this phase was built to look for. What happened is a third
case: **a negative wealth lift and a modest drawdown improvement, alongside an
unresolved dependence verdict.**

The consequence is a narrowing of what can be claimed, in both directions:

- **Nothing here establishes that classifier #2 adds an independent axis.** Criterion
  6 did not establish it and criterion 7 cannot substitute for it.
- **Nothing here refutes it either.** A negative `wealth_delta` at one pinned blend
  weight, on one routing, is not evidence that the second axis does not exist — it is
  evidence that *this blend, at this weight, on this path, over these 588 months* did
  not increase terminal wealth.
- **No further dependence statistic was computed for this document.** The
  pre-registration forbids tie-breaks and that constraint is honoured here.

---

## 7. The L2 observational leg — reported, firewalled, not acted on

Computed under `ROUTING_L2_NOWCAST`, appended with `NO_REGISTRY` (0 rows), per
ADR-0002 decision (e). **Nothing downstream in this phase changes on the basis of
anything in this section.**

| | classifier #1 alone | joint (#1 × #2) |
|---|---|---|
| **Window** | **588 steps, 1972-01-31 → 2020-12-31** | **588 steps, 1972-01-31 → 2020-12-31** |
| Degraded steps | **100 of 588** (87 classifier #1's nowcaster, 13 classifier #2's) | **100 of 588** (87 / 13 — identical set) |
| Terminal log wealth (588 steps, 1972-01-31 → 2020-12-31) | 4.103637 | 3.969131 |
| Max drawdown (same window) | −26.6546%, 32 months underwater | −26.7683%, 32 months underwater |
| Annualized Sharpe (588 obs, same window) | 1.204395 | 1.171926 |
| Mean monthly turnover (same window) | 0.098907 | 0.060451 |
| DSR (`n_trials` 42, read 2026-09-21T14:45:17.284854Z, variance 1.0) | 3.25419 × 10⁻³⁰ | 6.14432 × 10⁻³³ |

**L2 lift, both axes, with the window inline:** `wealth_delta` **−0.134505** over
**588 steps, 1972-01-31 → 2020-12-31**; `dd_delta` **−0.001137** over the same **588
steps, 1972-01-31 → 2020-12-31**. Both inside their governing bands. Both legs' degrade
counts match exactly (100 each, same steps), so the two L2 legs also share one window.

**ADR-0002's named failure signature for this routing — "the two legs disagree in
sign on joint lift" — half fired, and is recorded rather than resolved:**

- On `wealth_delta` the two routings **agree**: −0.123438 (L1-only) and −0.134505
  (L2), both negative, similar magnitude.
- On `dd_delta` they **disagree in sign**: **+0.024084** (L1-only, a 2.41pp
  improvement) versus **−0.001137** (L2, a 0.11pp deterioration). The L2 magnitude is
  near zero — about one twentieth of the L1-only figure and about 1/20th of a
  percentage point of drawdown — so this is better described as "the L2 path shows no
  drawdown effect" than as a contradiction. **It is nonetheless a disagreement in
  sign on one of the two axes and is not explained here.** Per the firewall clause,
  it is reported and not acted on.

---

## 8. Labeling diagnostics — the two bands criterion 7's comparison does not itself produce

07-BANDS.md §0.2 records that `n_transitions` and `pct_disagree` describe *labelings*,
not a comparison of two backtest legs. They are measured here, from the persisted
curves, by `scripts/joint_lift_diagnostics.py` (which touches the registry not at all).

| Quantity | Classifier #1 | Classifier #2 | Band (07-BANDS.md §8) | Verdict |
|---|---|---|---|---|
| Full-sample label transitions | **25 over 695 months, 1963-02-28 → 2020-12-31 (3.597%)** | **12 over 696 months, 1963-01-31 → 2020-12-31 (1.724%)** | 3b: implausible above `0.10 × n_months` (69.5 / 69.6) | **inside** |
| `sojourn_lag` `n_resolved` / `n_transitions` | **25 / 25** (588-step window, 1972-01-31 → 2020-12-31) | **5 / 12** (same window) | 3a: `n_resolved ≤ n_transitions ≤ n_label_transitions` → 25 ≤ 25 ≤ 25; 5 ≤ 12 ≤ 12 | **inside** |
| §5.4 median sojourn / median detection lag / ratio | 9.5 mo / 4.0 mo / **2.375** (588 steps, 1972-01-31 → 2020-12-31) | 29.0 mo / 27.0 mo / **1.074** (same window) | — | see §9 limitation 3 |
| `pct_disagree` (filtered vs full-sample smoothed) | **0.787415, n_compared = 588 of 588 expected** (1972-01-31 → 2020-12-31) | **0.831633, n_compared = 588 of 588 expected** (same window) | 4: `< 0.02` **OR** `n_compared == 0` **OR** coverage below 90% of expectation without a recorded reason | **not suspicious** on any clause |

Band 4's denominator clauses were exercised live: `expected_n_compared=588` was passed
in both cases and coverage came back at exactly 588, so the third clause could have
fired and did not. The 78.7% / 83.2% figures are in line with wave 1's 80.9% (n=288 of
356) and 82.8% (n=389 of 470) — different populations, different windows, quoted with
their denominators for that reason.

**The walk-forward filtered labeling is a separate matter and is reported as a
finding, not a band verdict.** Classifier #1's per-step filtered state changed in
**246 of 588 decision months (41.84%)**, against a full-sample rate of 3.60%.
Classifier #2's changed in **24 of 588 (4.08%)**, against a full-sample 1.72%. Band 3b
governs the *full-sample* labeling (07-BANDS.md §3.5b) and says nothing about the
filtered one, so no band fired — but a labeler that re-labels its own most recent
month in two months out of five is churning, and that churn is what the degenerate
one-hot routing feeds straight into the tilt. It is the most likely mechanical source
of classifier #1's 0.163 mean monthly turnover.

---

## 9. Suspicion — what here would mean a broken measurement, and what was checked

Each of the five signatures plan 07-11 named, with the check performed and the
conclusion. **This section is written to be capable of failing.**

**(1) A terminal log wealth outside a plausible range for a five-decade monthly
series.** The reference failure is `05-VERIFICATION.md` tabulating 32.18 → 111.06 as
progress; e¹¹¹ ≈ 10⁴⁸. *Checked:* an in-script assertion `abs(terminal_log_wealth) <
10` runs on **both** legs before any number is reported, and would have raised on
111.06 eleven times over. Measured: **4.718723** and **4.595285** over 588 months —
e^4.72 ≈ 112× and e^4.60 ≈ 99× cumulative, i.e. **9.9% and 9.6% annualized over 49
years** for a 10%-vol-targeted long-only book. *Concluded:* plausible, and the guard
that would have caught the failure ran.

**(2) Two legs whose indexes differ.** *Checked* three ways: `joint_lift_table` returns
`indexes_identical` in the same mapping as the deltas and logs a WARNING naming both
sizes if they differ; the harness's degraded-step *set* is identical by construction
(no refit consults `blend_weight_1`, and a step degrades if either classifier fails);
and the persisted parquet curves were re-read independently and compared
element-wise. Measured: **`index.equals` is True, 588 = 588, degraded 0 = 0**, and an
independent recompute of both deltas from the parquet reproduced −0.123438 and
+0.024084 to the last digit. *Concluded:* one window, shared.

**(3) A transition count above the governing threshold — labeling churning rather
than regime-switching.** *Checked:* full-sample run-length changes against band 3b's
rate, and the `sojourn_lag` within-window counts against band 3a's definitional chain.
Measured: **3.597% (25/695) and 1.724% (12/696)**, both far inside the 10% rate; 3a's
chain holds for both. *Concluded:* the bands hold. **But the check surfaced something
the bands do not govern:** the *filtered* labeling changes state in 41.84% of decision
months for classifier #1 (§8). That is reported as a finding, not laundered through a
band that was never about it.

**(4) A lift of exactly zero — the blend weight never took effect.** *Checked:*
`wealth_delta` inspected for exact zero, and the per-month returns compared.
Measured: **`wealth_delta = −0.123438`, and 586 of 588 monthly returns differ between
the legs**. The two identical months are early steps where classifier #2's tilt was
degenerate. Independently, `test_legs_differ_in_returns_so_the_ablation_is_not_a_no_op`
and `test_weight_one_equals_classifier_one_alone` pin both halves of the ablation.
*Concluded:* the blend took effect.

**(5) A DSR of exactly 0.0 or 1.0 — a degenerate variance input.** *Checked:* the raw
float DSRs, not the four-decimal verdict string. Measured: **2.28151 × 10⁻¹²** and
**1.46904 × 10⁻¹¹** — strictly positive, strictly below 1, and *different from each
other*, so the computation is not returning a constant. `deflated_sharpe_ratio`'s own
non-normality-denominator guard (which raises rather than returning NaN) did not fire.
*Concluded:* not degenerate in the arithmetic sense. **It is, however, running on the
`1.0` placeholder variance** because the registry holds zero Sharpe-bearing rows — a
declared assumption (§4.1), logged at WARNING on every call, not a silent default.

### 9.1 Additional suspicions raised by the data itself, not on the list

**A +53.17% single strategy month in January 1974.** Traced to the asset universe:
**`oil` returned +134.57% in 1974-01-31** (the OPEC embargo repricing). That single
month is what produces both legs' skew of ~7.4 and raw kurtosis of ~126, which in turn
feed the DSR's non-normality denominator. Two related data facts, found while
checking it and both shared identically by the two legs so they cannot bias the
*delta*: **`oil` has 92 of 588 months (15.65%) with a return of exactly 0.0000** — it
is a step-function posted-price series in the early era, not a traded monthly return
series — and **`gold` has no data before 1985-03-31** (430 of 588 months), so it is
absent from the tilt for the first 26 years of the window. Both are level facts about
the asset universe, not about classifier #2; both legs inherit them equally.

**A silent zero caught while building §8's diagnostics.**
`compute_sojourn_lag_headline` expects **integer** state columns while
`measure_label_disagreement` expects **`state_{k}` strings**. Handing the former a
string-columned matrix returns `n_resolved = 0`, `median_lag = NaN`, `ratio = NaN`
with no error — a reading that looks like "detection never happened" and is actually
"the matrix was the wrong shape". This is this project's signature defect (the check
that can only confirm) in a new costume, and it is the fifth instance after criterion
8's `grep -v platform`, the invented 5% floor with no cap, D-02-A's unreproducible
recompute and `dd_delta`'s `[-2, 2]`. The first draft of §8 reported `0 of 25` before
the shape was corrected. `scripts/joint_lift_diagnostics.py` now asserts the matrix
shares at least one state column with the labeling, so the ambiguity cannot recur
silently.

---

## 10. Limitations this record carries rather than buries

1. **The crisis state's median sojourn is 3.0 months — exactly on design §4.4
   criterion 2's boundary — against §5.4's typical 1–3 month detection lag.**
   Criterion 7's `dd_delta` of **+0.024084 over 588 steps, 1972-01-31 → 2020-12-31**
   must **not** be read as evidence that crises are nowcastable in time to act. This
   labeling identifies crises *ex post*; whether L2 can nowcast them early enough to
   trade is L2's question and it is **unanswered**. A drawdown avoided in the labeling
   is not a drawdown avoided in real time.
2. **Design §4.4 criterion 3's formal subsample-stability test (Hungarian matching on
   emission distributions) has never been run for either classifier.** Nothing in this
   document rests on it having passed, and nothing here should be read as implying it
   has.
3. **Classifier #2's §5.4 ratio is 1.074** (median sojourn 29.0 months, median
   detection lag 27.0 months, **5 of 12 transitions resolved**, over 588 steps
   1972-01-31 → 2020-12-31). The lag very nearly consumes the sojourn, and more than
   half its transitions never resolve at the 0.70 action threshold at all. Whatever
   leadership structure classifier #2 encodes, this measurement gives **no evidence it
   is detectable in real time early enough to allocate on** — which is a caveat on
   criterion 7's headline number, since the joint leg's tilt is fed by exactly that
   filtered labeling.
4. **One routing, one blend weight, one window.** `blend_weight_1` is pinned at 0.50
   and was **not** swept — sweeping it is an unregistered selection dimension the
   trial ceiling does not budget for (D-13, ADR-0002 (f)). This record says nothing
   about what any other weight would produce.
5. **The DSR's `sharpe_variance` is the `1.0` placeholder**, not a measured quantity
   (§4.1).
6. **The `n_degraded_classifier_2` counter under `ROUTING_L2_NOWCAST` originally
   attributed every L2-block failure to classifier #2.** It was corrected to catch each
   nowcaster separately and the observational leg re-run under the sentinel; §7's
   87/13 split is from the corrected run. The L1-only decision-bearing legs never
   execute that block and are unaffected.

---

## 11. The honest bottom line

**Classifier #2 was measured, and it did not help on wealth.** Blended at
`blend_weight_1 = 0.50` against the classifier-#1-alone leg on the identical 588-step
window 1972-01-31 → 2020-12-31, the joint leg ended **0.123438 nats lower** — about
**11.6% less terminal wealth** — while making the worst drawdown **2.41 percentage
points shallower** (−25.74% vs −28.14%) and 7 months shorter. Both deltas sit inside
their governing plausibility bands, both legs ran 588 of 588 steps with zero
degradation on one shared window, and **neither leg's deflated Sharpe clears the
multiple-testing hurdle at 42 trials**.

**Criterion 7 is satisfied: the lift was measured, honestly and with its window.**
D-06 makes measurement the gate, not the sign. What remains unproven is everything the
sign would have needed to mean something — **criterion 6 is unresolved**, so no
independent second axis is established; **§4.4 criterion 3 has never been run** for
either classifier; **classifier #2's detection lag (27.0 months) nearly equals its own
median sojourn (29.0 months)**, so its labels may not be actionable in real time at
all; and the drawdown improvement, the one axis where the blend helped, sits behind
the §10.1 caveat that this labeling identifies crises after the fact.
