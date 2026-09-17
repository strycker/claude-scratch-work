# 07-08 Task 1 — Decision checkpoint record

**Resolved:** 2026-09-17 by Glenn (VP Decision Sciences), at the blocking
`checkpoint:decision` gate opening plan 07-08.

All six values below were fixed BEFORE classifier #2 was fit and before either
walk-forward ran, per D-13 (no selection trials on these) and D-17 (trial ceiling
precedes the runs). Each carries its failure signature, per the gate's `<action>`.

---

## (a) Classifier #2's frozen candidate columns — "Lean 8"

```
rs_equities_bonds                      first valid 1962-01-31
rs_oil_equities                        first valid 1962-01-31
equities_tr_mom_12m                    first valid 1963-01-31
long_duration_tr_mom_12m               first valid 1963-01-31
oil_mom_12m                            first valid 1963-01-31
corr_equities_tr_long_duration_tr_24m  first valid 1963-01-31
cpi_acceleration                       first valid 1962-04-30
m2_gdp                                 first valid 1962-02-28
```

n = 8. One momentum horizon per sleeve; the 6m/24m twins are dropped as
near-collinear, and `credit_gdp` is dropped because 07-07 measured its correlation
with `m2_gdp` at 0.957–0.969 across eras.

Disjointness from classifier #1's lean 13 verified live:
`set(candidates) & lean_feature_set(cfg) == set()`.
All eight clear D-11's 1972+ freeze with margin (latest first-valid 1963-01-31).

**Failure signature.** If 12m is the wrong memory length for a leadership axis, the
fitted states will flip far more often than classifier #1's — visible as a
change-point count well above #1's over the same span, and as median sojourn
falling below the §5.4 detection-lag ratio.

> **Glenn's stated position, recorded verbatim in substance:** feature selection,
> dimensionality reduction and hyperparameter optimization are things an ML model
> should do, and should not be hardcoded. He accepted the pinned Lean 8 *for this
> phase only*, conditional on a roadmap item to revisit once selection can be done
> honestly. Filed as **ROADMAP T0.9** (registered nested selection inside the
> walk-forward loop). The constraint is not the selection itself but doing it
> without look-ahead and without silently understating D-16's denominator.

## (b) K = 3

By construction: three asset sleeves appear in the candidate set (equities,
long duration, oil), so three leadership states. The K falls out of the feature
set's structure; no fit was run to choose it.

**Failure signature.** A state outside §4.4 criterion 1's band, or two states whose
centroids differ on no sleeve.

> **CORRECTION 2026-09-17 — this failure signature was stated against the wrong
> threshold, and it fired.** I wrote "§4.4's 5% floor", repeating an error that
> originates in ADR-0001. §4.4 criterion 1 is two-sided: every state ≥ ~8% AND
> ≤ ~35%. The live fit is 15.3736 / 46.1207 / 38.5057 % — states 1 and 2 breach
> the cap, state 1 by 11pp, so **K = 3 as pinned here fails criterion 1**.
>
> Worse, the K = 3 rationale was unsound in a way I should have caught at the
> checkpoint: three states summing to 100% under a ~35% cap must each sit in
> [30%, 35%], which is forced balance — the thing §4.3 set out to replace. I
> reasoned K = 3 from the three asset sleeves and never scored it against the
> acceptance criteria. That error is mine, not the executor's.
>
> K and λ are pending a re-pin against §4.4 criteria 1, 2 and 6. Per §4.3 that
> tuning is an acceptance gate, not a D-13 selection trial. Items (a), (d), (e)
> and (f) are unaffected.

> Same standing objection from Glenn as (a) — K should be selected, not fixed.
> Accepted for this phase; also covered by **ROADMAP T0.9**.

## (c) λ = 32.0 — derived, not chosen

λ = 4n = 4 × 8. Matches the feature-count formula D-13 reuses and that classifier #1
instantiates (13 columns → λ 52.0). No discretion: the gate REJECTS any other value.

## (d) Canonical ordering column = `rs_equities_bonds`

RESEARCH Assumption A1's proposal, risk-rated low: `sort_column` decides only which
state index is called 0, never occupancy, dependence or lift.

**Failure signature.** If this column is ever dropped from the frozen list, plan
07-05's fix raises `ValueError` at fit time rather than silently reordering — the
failure is loud by construction. (The pre-07-05 fallback to centroid column 0 is
deleted; 0 occurrences remain in the tree.)

## (e) Criterion-7 routing — L1 decision-bearing, L2 observational and firewalled

Glenn's instruction, which is a fourth option not among the three the plan offered:

> "Only use L1 for any choices, decisions, D-16 denominator, Sharpe calculation,
> D-17 requirements, etc., however, also calculate and report L2 for human
> evaluation. Do not include any L2 calculations towards D-16, D-17, Sharpe, etc."

Implementation contract:
- The **L1-only leg is decision-bearing**: it is what criterion 7's joint-lift number
  reports, it appends to the trial registry with a `trial_tag`, and it counts toward
  D-16's deflated-Sharpe denominator and D-17's ceiling.
- The **L2 leg is observational**: computed and reported for human reading, appended
  with the `NO_REGISTRY` sentinel so it does NOT count toward D-16, D-17 or the DSR.
- Both legs state **the window they were measured on, inline with the number**
  (wave 1's UAT binding condition).
- The report states that the L1-only leg is **not comparable** to wave 1's measured
  `wealth_delta` +0.377847 / `dd_delta` -0.066124, because the #1-alone baseline is
  re-run through the L1-only harness rather than reused.

**Residual risk, recorded rather than resolved.** The honesty framework fences
*fitting*, not looking — but a number a human reads and reacts to is selection in
effect, registered or not. `NO_REGISTRY` keeps the arithmetic honest; it cannot keep
the reader honest. ADR-0002 must therefore bind that **nothing downstream in this
phase changes on the basis of the L2 leg** — it is reported and not acted on. If a
future phase does act on it, that is a recorded amendment, not a silent reuse.

**Failure signature.** The two legs disagree in sign on joint lift. That would mean
the L1-only reading and the production stack disagree about whether classifier #2
helps at all, and neither number should be trusted until the disagreement is
explained.

## (f) Blend weight `allocation.blend_weight_1` = 0.50

Equal weight — the no-information prior. Nothing known before the runs favours either
classifier, and any other value would need an empirical justification, which is a
selection trial D-13 forbids.

**Failure signature.** If the blended tilt tracks one classifier's solo tilt almost
exactly, the blend is adding nothing and the weight was never the binding constraint.

> **Glenn challenged the premise**, not the number: he does not think the two regimes
> should be mixed at all, preferring the (regime₁, regime₂) combinations one-hot
> encoded as features with selection deciding which cells carry signal — dropping
> regime₂ entirely if unpredictive.
>
> Investigated at the checkpoint. **There is no learner at L3 or L4 to consume such a
> design matrix.** `assets/returns.py::returns_by_regime_stats` is descriptive
> statistics, explicitly "NOT ... a supervised-learning target" and exempt from the
> registry and purged CV; `allocation/tilt.py::regime_tilt_weights` takes a
> probability-weighted average of that table. No coefficients anywhere. The fixed
> weight exists because two probability vectors must become one weight vector and
> there is no model to learn the combination — not because blending was preferred
> over interaction.
>
> D-14 therefore stands for this phase. The proposal is filed as **ROADMAP T0.10**
> (learned regime-conditional allocation), with the caveat that one-hot encoding does
> not escape cell thinness: a cell holding 8 months yields a column with 8 nonzero
> rows, so selection either drops it (collapsing toward the marginals, ≈ the blend)
> or retains it and overfits. At ~590 decision months and K₁×K₂ = 15 the expectation
> is mostly collapse — a prior to be tested, not a measured result.
