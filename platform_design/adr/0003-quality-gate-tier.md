# ADR-0003: A Quality Gate Tier — the Deflated-Sharpe Hurdle Governs, and A11 Closes

## Status

**Accepted, 2026-09-21**, by plan 08-05 of Phase 8 (`08-regime-persistence-stability`), wave 1.
Requirement: **PER-08**. Audit item: **A11**.

**This ADR reverses a decision taken on 2026-09-18.** It is not a cleanup of an oversight. On
2026-09-18, at plan 07-10's human-verify gate, Glenn **declined** to promote any band to a quality
gate and left A11 open *by deliberate choice*; ADR-0002 § *Deferrals and open items at acceptance*
item 10 records that state in those words. On 2026-09-21 Glenn reopened A11 and answered it
**YES**. The reversal record — the prior reasoning quoted verbatim, what changed, and what survives
of the objection being reversed — is
`.planning/phases/08-regime-persistence-stability/08-A11.md`. That document, not this one, is the
narrative of the reversal; this ADR is the architecture decision it produced.

**What is accepted, stated before anything else so it cannot be over-read.** What is accepted is
that **exactly one** threshold in this project is promoted to a governing quality tier: the
deflated-Sharpe hurdle. **No plausibility band is promoted.** D-07 is untouched and the four
`[ASSUMED]` bands remain plausibility-only. Nothing in this document may be cited as having
promoted a band.

**This decision was recorded before any number produced by Phase 8 existed** — before plans 08-01
and 08-02 executed, and four waves before plan 08-10 re-measures criterion 7. That ordering is the
point: a gate chosen after seeing the value it judges is a tie-break, which the pre-registration at
`298b1bc` forbids. The ordering is a fact about what happened, not a guarantee the dependency graph
supplied; 08-01 and 08-02 sit in the same wave with no ordering constraint against 08-05.

## Context

ADR-0002 § *The four plausibility bands, as confirmed or revised* closes with the disposition this
ADR reverses: *"Per D-07 these remain plausibility bands, not quality gates. The governing tier can
only fail on a **broken measurement**, never on a bad-but-working model. **Audit item A11 therefore
remains open and conscious** — Glenn declined to promote the domain tier to a gate, which would
have closed A11 by gating on `[ASSUMED]` numbers."*

`.planning/UAT-AUDIT-2026-09-09.md` raised A11 as the gap spanning all five prior phases: *no
criterion anywhere asks whether the output is any good.* Through 2026-09-18 that was an abstract
worry about a class of failure that had not yet occurred.

**On 2026-09-21 it has an instance, and the instance is this project's own headline measurement.**
Criterion 7, routing `L1_ONLY_LAST_FILTERED_STATE`, decision-bearing, **588 steps 1972-01-31 →
2020-12-31**, 0 degraded steps on either leg
(`outputs/reports/platform/joint_lift/measurement_l1only.json`):

| Quantity | Value | Verdict as recorded |
|---|---|---|
| `wealth_delta` | **−0.123438 nats** — joint leg at ≈ **0.8839×** the #1-alone leg's terminal wealth, an **11.61% terminal-wealth shortfall** | `wealth_delta_universal_ok` **True**, `wealth_delta_domain_note` **False** |
| `dd_delta` | **+0.024084** | `dd_delta_universal_ok` **True**, `dd_delta_domain_note` **False** |
| DSR, #1-alone leg (observed Sharpe 0.9170725133308871) | **2.28151091802503 × 10⁻¹²** | *does not clear the multiple-testing hurdle* |
| DSR, joint leg (observed Sharpe 0.914903185594245) | **1.4690427074211624 × 10⁻¹¹** | *does not clear the multiple-testing hurdle* |

A working model lost 11.61% of terminal wealth, both legs' deflated Sharpe ratios were
indistinguishable from a skill-less discovery, and **not one of the four band flags fired — not
even the two advisory ones.** That is A11 stated as a measurement rather than a worry, and it is
why the item was reopened.

Two constraints bound every available answer: **no 2021+ holdout data may inform it**, and the
trial registry stands at a live-read **42 of ADR-0002's 44 ceiling**, so any answer requiring a
calibrated threshold would spend from a budget with two rows left — rows that plan 08-09's own
blocking decision needs.

## Decision

**A decision-bearing leg of a criterion-7 measurement FAILS the quality tier when its own estimated
Sharpe ratio does not exceed the expected maximum Sharpe of an equal number of skill-less trials:**

```
PASS  iff  observed_sharpe > expected_max_sharpe(total_trial_count(), sharpe_variance)
```

which is **identically** `deflated_sharpe_ratio(...) > 0.5`, because the DSR is `norm.cdf` of a z
whose numerator is exactly that difference. At `observed_sharpe` equal to the hurdle the DSR is
exactly 0.5; the boundary is exclusive. `format_dsr_verdict`'s existing `_VERDICT_HURDLE = 0.5` is
therefore the same threshold, and **this decision invents no new constant.**

At the recorded trial count of **42** and the placeholder `sharpe_variance` of **1.0**:

```
expected_max_sharpe(42, 1.0) = 2.208694      (2.2086935028832686)
expected_max_sharpe(44, 1.0) = 2.226891      (2.2268911497604993)
```

**Both arguments are read live at measurement time. Neither may be written as a literal.** A
hard-coded hurdle would freeze the bar while the search kept growing — the systematic
under-penalization T-07-05 names as this project's closest analog to a security defect.

The option identifier Glenn selected is **`b-promote-dsr`**, recorded with its full rationale in
`08-A11.md` §3.

**Three tiers now exist, and only the first two can halt anything:**

| Tier | Job | A breach means | Governs? |
|---|---|---|---|
| Universal / arithmetic | Is this number physically possible for this quantity? | The **measurement is broken** — halt, do not report a lift. | YES (unchanged) |
| **Quality (this ADR)** | Is this result distinguishable from a skill-less discovery given the search performed? | The result **FAILED**. Reported plainly; not waivable by a reviewer. | **YES (new)** |
| Domain / advisory | Is it within what this stack has produced? | Record a note; the measurement still reports. | No (unchanged) |

## Considered Options

The four answers put to Glenn are enumerated with their option identifiers in `08-05-PLAN.md`
Task 1 and in `08-A11.md` §3.3. **Only the selected identifier is repeated here** — the three
rejected identifiers are deliberately not written into this directory, so the automated check that
*exactly one* selected option id reaches the ADRs stays discriminating rather than matching on
every id it finds.

1. **Answer NO — plausibility bands stay plausibility-only, and A11 closes as answered.**
   Rejected. It costs zero trials and preserves D-07 exactly, and its objection — that promoting a
   band means gating on `[ASSUMED]` numbers — **has never been refuted**. But it ships the platform
   with **no** gate that can fail on a bad model, leaving § Context's 11.61% shortfall with four
   green flags standing as the documented consequence. The reversal would have been real but
   modest: the item stops being open without anything behaving differently.

2. **Answer YES by promoting the existing domain/advisory tier to governing.** Rejected
   **decisively, and on measurement rather than on argument.** On the actual result the two domain
   triggers are `abs(wealth_delta) ≥ 5` and `abs(dd_delta) ≥ 0.5`; the measured values are
   **−0.123438** and **+0.024084**, so **both flags are already recorded `False`** in
   `measurement_l1only.json`. Promoting them buys the *appearance* of a quality gate with none of
   the substance — it would not have fired on the very instance that reopened A11 — while incurring
   the full weight of `07-BANDS.md` §0.1's objection that no measurement in this project derived
   any of the four. Tightening them toward the observed values to make them bite would convert a
   plausibility band into a quality band fitted to a result already seen, which D-07 and ADR-0001's
   selection criterion refuse.

3. **Answer YES with a brand-new pre-registered quality threshold, declared now and applied at
   08-10.** Rejected. It is the only rejected option that would both bite and be genuinely
   pre-registered, but there is no measurement in this project from which to derive such a
   threshold — which is exactly why the existing four bands are all `[ASSUMED]`. Calibrating one
   would spend from the **two rows** remaining against ADR-0002's 44 ceiling, and those two rows
   are claimed by plan 08-09's `b-bounded-turnover` option. Two independent decisions cannot each
   claim the last two rows. And a pre-registered number chosen from intuition is still an
   intuition; it is merely an honest one.

4. **The selected answer, `b-promote-dsr`.** Accepted. `expected_max_sharpe(n_trials,
   sharpe_variance)` is **arithmetic on the trial count** — Bailey & López de Prado (2014) Eq. 5/6,
   an Extreme Value Theory result — making it **the only quality threshold in this project that is
   derived rather than asserted by analogy.** It already exists, is already computed on every run,
   and already carries an unsoftened verdict string; promoting it costs **zero** registry trials.
   It gates on exactly the failure this project's honesty framework was built to catch.

## Consequences

1. **Criterion 7's already-recorded `✅ MET 2026-09-21` becomes FAILED retroactively, on BOTH
   legs** — DSR **2.28151091802503 × 10⁻¹²** (#1-alone) and **1.4690427074211624 × 10⁻¹¹** (joint)
   against the **2.208694** hurdle. Neither leg's Sharpe is within a factor of two of the bar.
   **This was accepted knowingly and in advance**, at the moment of the decision, not discovered at
   plan 08-10. The measurement itself does not change and is not re-described: criterion 7 remains
   **MET as a measurement** and is now **FAILED as a result**, and the record says both.

2. **`07-BANDS.md` §8's `[ASSUMED]` objection is narrowed, not answered and not overridden.** The
   hurdle's dependence on the trial count is derived arithmetic over a live read of this project's
   own registry, so on the axis §8 objects to — provenance of the threshold — this gate is
   categorically unlike the four bands. **But `sharpe_variance = 1.0` is
   `DEGENERATE_SHARPE_VARIANCE`, a declared placeholder** standing until **20 independent
   Sharpe-bearing trials** exist, so **2.208694** rests on an assumption. That is the same species
   of objection in weaker form — the placeholder is conservative by construction, documented at its
   definition as an assumption, and carries a countable replacement trigger, none of which is true
   of the four bands. **ADR-0002 open item 8 therefore escalates from governing a report to
   governing a gate**, and remains open. When the twentieth independent Sharpe-bearing trial lands,
   the hurdle moves and every verdict this gate has issued is re-dated.

3. **Registry cost: zero.** `registry rows spent: 0`. Live `total_trial_count()` = **42** against
   the ceiling of **44**; **both** remaining rows are left intact for plan 08-09 Task 1.

4. **No code changes in this decision.** Plan 08-05 writes no `src/`. The implementation — surfacing
   the quality-tier verdict alongside the existing `*_universal_ok` / `*_domain_note` flags and
   honouring it in the criterion-7 record — is executed by **plan 08-10, under this decision,
   before criterion 7 is re-measured.** `joint_driver.py` is owned by 08-01 in wave 1 and 08-10 in
   wave 5; the dependency edge 08-10 → 08-05 enforces the ordering.

5. **A quality-tier failure is not waivable by a reviewer.** Reviewers approve on the honesty and
   completeness of a record, never on whether the numbers look good (D-06). A failure is reported,
   not negotiated; it may be superseded only by a later ADR stating what changed, or re-dated when
   `sharpe_variance` stops being a placeholder — never by a re-reading of the same numbers.

6. **No 2021+ holdout data may inform this gate, its threshold, or any waiver of it.**

7. **What did NOT change.** The four plausibility bands remain plausibility-only; a green
   `*_universal_ok` still asserts only that the measurement is possible; a `*_domain_note` remains
   an advisory note that changes no verdict; no band may be tightened toward an observed value.

## Pinned by

`tests/unit/test_platform_gate_tiers.py` — names the **rejected** values (observed Sharpe
0.9170725133308871 and 0.914903185594245, DSR 2.28151091802503e-12 and 1.4690427074211624e-11) and
the **accepted** value (observed Sharpe 2.60, DSR 0.752687478118391) on the same moments and track
length, asserts the hurdle turns over at exactly 0.5, asserts the hurdle rises with the trial count
rather than being a literal, and asserts the variance inside it is still a declared placeholder — so
the day it stops being one, the pin goes red and this ADR is re-dated on purpose. **No skipped or
xfailed branch.**

## Related

- `.planning/phases/08-regime-persistence-stability/08-A11.md` — the reversal record.
- `platform_design/adr/0002-l1-second-classifier.md` — § *Deferrals and open items at acceptance*
  item 10 (the state this ADR reverses) and item 8 (the placeholder variance).
- `.planning/phases/07-regime-representation/07-BANDS.md` §§ 0.1, 8 — the prior reasoning.
- `.planning/UAT-AUDIT-2026-09-09.md` — where A11 was raised.
