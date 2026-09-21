# Phase 7 wave 2 — UAT sign-off

**Signed off by Glenn, 2026-09-21.** Conversational UAT, conducted against what wave 2 set out
to do rather than against whether its code runs (that is validation's question, audited
separately in `07-VALIDATION.md`).

---

## The question wave 2 existed to answer

> Does a second, independent regime axis help?

**The measured answer: no on wealth, yes on drawdown, and independence unproven.**

| | |
|---|---|
| `wealth_delta` | **−0.123438 nats** — the joint leg ended **11.61% below** classifier #1 alone |
| `dd_delta` | **+0.024084** — worst drawdown **2.41pp shallower**, 40 vs 47 months underwater |
| window | 588 steps, 1972-01-31 → 2020-12-31, both legs, 0 degraded |
| criterion 6 | **UNRESOLVED** — pre-registered rule returned INCONCLUSIVE |
| deflated Sharpe | baseline 2.28151e-12, joint 1.46904e-11 — **neither clears** at 42 trials |

## UAT 1 — was the intent met? **YES**

Glenn's verdict: *the question was answered honestly*. The phase existed to find out, not to
succeed. It measured the thing, recorded a negative result without dressing it up, and declined
to claim an axis it could not demonstrate.

What is now known that was not before, all of it measured, windowed and reproducible:

- classifier #2 costs 11.61% terminal wealth and buys 2.41pp of drawdown;
- the two labelings' dependence is unresolved, not favourable and not damning;
- neither leg clears the multiple-testing hurdle;
- classifier #1 churns 41.84% of decision months in real time;
- classifier #2's detection lag (27.0 mo) nearly equals its own median sojourn (29.0 mo).

A phase that returns "no, and here is exactly how much no, with its window" has done its job.

## UAT 2 — the filtered-churn finding: **BLOCKING, fix before Phase 8**

| measure | value | what it is |
|---|---|---|
| full-sample transition rate | 3.60% | smoothed, hindsight |
| **filtered** state changes | **41.84%** (246 of 588) | **real-time, actionable** |

Glenn's verdict: **blocking defect, fix before Phase 8.**

The reasoning is design §5.4's own: the gap between smoothed and filtered *is* "the measured
hindsight content of the strategy". A ~12× gap is large and was never budgeted for. The 41.84%
figure is what a weekly report would actually consume — the 3.60% rate is the hindsight view and
is **not** what anyone would trade on. A labeler that re-labels its own most recent month two
months in five feeds that instability straight into the tilt, and therefore into criterion 7's
number.

**No band governs this.** Band 3b is a full-sample band and does not see the filtered series.
That is a genuine coverage gap, not merely an unflattering statistic.

**Consequence:** Phase 8 does not start until this is addressed. Recorded in `ROADMAP.md`.

## UAT 3 — phase verdict: **ACCEPT with the recorded caveats**

Same disposition wave 1 received.

| item | as signed off |
|---|---|
| REG-01 | **PARTIAL** — criterion 6 the named open item |
| INV-01 | **COMPLETE** |
| criteria 5, 7, 8 | MET |
| criterion 6 | **UNRESOLVED** — not met, not failed |
| open items | all twelve carried forward as they stand |
| A11 | **open**, by Glenn's deliberate choice |
| suite | 1983 passed, 0 skipped, 0 xfailed; ratchet 31 |

### What this sign-off does NOT assert

- That classifier #2 adds an independent axis. It does not; criterion 6 is unresolved and no
  tie-break is permitted.
- That the drawdown improvement offsets the wealth loss. Those are two numbers, recorded
  separately and deliberately not netted.
- That crises are nowcastable in time to act. The crisis state's median sojourn is 3.0 months
  against a 1–3 month detection lag; `dd_delta` is not evidence on that question.
- That §4.4 criterion 3 holds. Its Hungarian subsample test has never been run for either
  classifier.
