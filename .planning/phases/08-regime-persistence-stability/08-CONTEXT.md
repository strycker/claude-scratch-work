---
phase: 8
slug: regime-persistence-stability
created: 2026-09-21
source: Phase 7 wave-2 UAT + validation audit; decisions taken with Glenn 2026-09-21
---

# Phase 8 — Regime Persistence & Stability: context

Every number below was **measured**, not assumed, and is reproducible from the tracked
checkpoints on `main` at `7939124`. Where a figure is an assumption it says so.

---

## D-01 — The problem, measured

Phase 7's UAT measured classifier #1's **filtered** (real-time) labeling:

| quantity | value |
|---|---|
| filtered state changes | **246 / 588 (41.84%)** |
| full-sample transitions | 22 (3.74%) |
| **median filtered run length** | **1.0 month** |
| runs of exactly one month | **136 of 247** |
| changes >3 months from any real transition | **184 (74.8%)** |
| churn rate *inside* ±3-month transition windows | 49.2% |
| churn rate *outside* them | **39.8%** |

**The labeling is effectively memoryless.** A model appropriately uncertain at turns would churn
at boundaries and settle between them. Quiet-period churn at 39.8% against boundary churn at
49.2% is not that. Window: 588 steps, 1972-01-31 → 2020-12-31, from
`outputs/reports/platform/joint_lift/joint_lift_baseline_l1only.parquet`.

**Why it matters:** design §5.4 calls the smoothed-vs-filtered gap *"the measured hindsight
content of the strategy"*. Here it is ~11× and was never budgeted. The filtered series is what a
weekly report consumes; the full-sample rate is hindsight. This churn feeds the allocation tilt
directly and therefore criterion 7's measured lift.

## D-02 — Root cause, confirmed and named by the design

`prediction/nowcaster.py::build_nowcaster_training_set` is `X = features_df.loc[common]`. There is
**no prior state distribution in the feature set**. Design §5.1, line 176, verbatim:

> **Include prior predicted state distribution as a feature** (recursive structure). Without it,
> persistence is discarded and predictions flicker.

## D-03 — The trap this phase must not fall into

The honest feature is the prior **predicted (filtered)** distribution, produced recursively within
each walk-forward step. The prior **smoothed label** is built from future data.

Substituting the smoothed label is **P1**, this project's documented first sin, in its easiest
form — and it fails in the worst possible way: CV accuracy would look excellent while production
flickered exactly as it does now. **A guard test must FAIL if the smoothed label is substituted.**
A test that merely checks the feature exists is the evidence-shape failure this project has now
recorded five times.

## D-04 — §5.3's machinery is built and unwired (audit A7)

`allocation/hysteresis.py::update_active_regime` is imported by `backtest/driver.py:67`,
`backtest/joint_driver.py:97` and `report/weekly.py:41`. It computes `active_regime` — and
per audit item **A7** (rated **High**, open since 2026-09-09) that value gates nothing:
weights come from `vol_targeted_tilt(regime_probs, …)`.

> A7: *"Decide Phase 4 C3: wire hysteresis into allocation, or reword the criterion. Today the
> thresholds stabilize a label, not a portfolio."*

Either allocation acts on it or the criterion is reworded. Leaving it inert is not an option.
**§5.3 work and closing A7 are the same job.**

**Ordering decision (Glenn, 2026-09-21): §5.1 first, then §5.3.** They do different jobs — §5.1
fixes the predictions, §5.3 fixes the allocation response. Wiring §5.3 alone would leave the
41.84% unchanged (it is a labeling metric) and merely stop the churn propagating, which is
treating the symptom. The roadmap note is explicit: **do not** address this by smoothing the
reported number, since §5.4 says to report that quantity prominently.

## D-05 — §4.4 criterion 3 has never been run, for either classifier

`grep` for `hungarian|linear_sum_assignment|subsample_stab` across `src/` and `tests/` returns
**nothing**. Criterion 3 reads:

> **Stability:** re-estimate on subsamples (drop first decade / drop last decade / block
> bootstrap); states persist with matched emission parameters (match via Hungarian algorithm on
> distribution distances to defeat label switching). A "regime" that evaporates when 2008–09 is
> dropped is an *episode*, not a regime.

**This is the test that decides whether classifier #1's crisis state is a regime or an episode.**
The §4.4 AMENDMENT of 2026-09-18 admitted that state below the ~8% floor on a *nine-episode
recurrence* argument — 1970, 1973-74, 1981, 1987, 1990, 2002, 2008-09, 2011, 2020. That is
evidence *for* the exemption, and it is **not** this test. Whichever way the result falls, it is
recorded; if the crisis state fails, the exemption's condition (i) is in question and that must be
said.

Both classifiers were re-pinned on 2026-09-18 (#1 → K=6, λ=10.0; #2 → K=5, λ=16.0), so the
labelings are fresh and this is the right moment.

## D-06 — A11 is reopened, and that is a reversal

Glenn left **A11** (*"no gate fails on a bad model"*) deliberately open on 2026-09-18 when the band
tiers were decided — universal/arithmetic gates criterion 7's verdict, domain/advisory only
records a note — and **chose on 2026-09-21 to reopen it**. The question: should any gate fail on a
bad-but-working model, rather than only on a broken measurement?

It is answered either way and **written as the reversal of a prior decision that it is**, not as a
cleanup. The prior reasoning is in `07-BANDS.md` §8: promoting the domain tier would mean gating
on `[ASSUMED]` numbers that no measurement in this project supports.

## D-07 — Two small carried items

- **Validation gap G6.** No test pins that non-joint consumers of `vol_targeted_tilt` — notably
  `backtest/driver.py:497` — receive the **unpooled** per-regime estimate. The wave-2 Nyquist
  audit could not write it because that file was outside its write scope. §5.3's wiring puts this
  phase in `driver.py` anyway.
- **Stale recorded counts.** `CLAUDE.md` claims **1705** tests at lines 128 and 606; `README.md`'s
  badge says **1705**. The suite is at **2018**. Trivial — and exactly the shape of recorded
  number that has misled this project three times already.

---

## Scope fences

- **No migration work.** That is Phase 9 (renumbered from 8 on 2026-09-21 when this phase was
  inserted ahead of it).
- **No dependence statistic of any kind.** Criterion 6 is UNRESOLVED and the pre-registration at
  `298b1bc` forbids tie-breaks. Nothing in this phase may recompute or relitigate it.
- **No 2021+ holdout use** for any selection decision.
- **`legacy/` and the reference submodules are untouched.**
- **The legacy-import ratchet is 31** and may only decrease.
- **ADR-0001 condition (iv)'s covariance clause is NOT in scope** — no per-regime covariance
  exists at L4-01; it falls to L3 (design §6.2).

## Baselines this phase must not regress

| | |
|---|---|
| suite | **2018 passed, 0 skipped, 0 xfailed** |
| legacy-import ratchet | **31** |
| criterion 7 (L1-only, 588 steps 1972-01-31 → 2020-12-31) | `wealth_delta` −0.123438, `dd_delta` +0.024084 |
| registry | 42 trials, ceiling 44 (ADR-0002) |
| DSR hurdle | `expected_max_sharpe(42, 1.0)` = 2.208694; `sharpe_variance` is the **1.0 placeholder** until 20 independent Sharpe-bearing trials exist |

Criterion 7 is **re-measured** under the changed labeling. Its previous value is a comparison
point, not a target — a change in it is the expected consequence of fixing the labeling, and is
reported with its window either way. **No target is pre-declared for the churn reduction.**

---

## AMENDMENT 2026-09-21 — D-01 through D-03 had the causal model wrong

`08-RESEARCH.md`'s F-1 overturns this document's own framing. Verified independently before
acceptance; every figure below was re-derived from the tracked artifacts at `b1c3519`.

### What was wrong

D-01 measured the churn on `state_1`. D-02 named §5.1's missing prior-state feature as the cause.
**Both are individually true and the link between them is false.**

- `joint_driver.py:502` — `state_1 = states_1.iloc[-1]`, the **jump model's** label (L1).
- `joint_driver.py:431` — under the decision-bearing `ROUTING_L1_ONLY`,
  `probs_1 = _last_state_one_hot(states_1)`; **`_refit_l2` is never called.**
- Classifier #2 runs the **same code path** and churns **4.09%** (24/587).

§5.1 changes L2. The 41.91% is L1. **The fix cannot move the number**, and the phase's original
criterion 2 would have reported 246 → 246 whether it worked perfectly or not at all.

The match between the measurement and §5.1's *"without it, predictions flicker"* was
**linguistic, not causal**. It was never checked that the measured series comes from the component
§5.1 fixes. Sixth recorded instance of this project's signature defect — a check that can only
confirm — authored by Claude while scoping a phase whose purpose is catching exactly that.

### The corrected model: two problems

**A — L1 terminal-month churn (41.91%).** The DP's terminal month is the only month with no right
neighbour, so deviating there costs **λ** where an interior deviation costs **2λ**; the filtered
labeling reads exactly that month, every step. λ/d is **1.0** for #1 and **2.0** for #2 — a
consequence of the λ re-pin of 2026-09-18 (coefficient 4 → 1). *Existence* follows from the
objective at `jump_model.py:43-45`; *magnitude* is unmeasured and is what the zero-trial
diagnostic settles.

**B — L2 flat posteriors.** `active_regime` changes **462/587 (78.7%)** under `l2` routing and is
all-cash in **387/588**, implying **≥309/488** non-degraded steps had max probability below 0.70.
That bound follows from the function's branch structure, not an estimate. 0.70 is **4.2× uniform
at K=6**; it was written against §4.2's HMM-filter intuition where posteriors are sharp.

### Decisions taken on this amendment (Glenn, 2026-09-21)

- **Target both, correctly separated.** Criterion 2 splits into A's metric (`state_1` churn) and
  B's metric (`argmax(regime_probs)` churn, never measured). Neither may masquerade as the other.
- **Criterion 1 is REWORDED to accept an explicit Bayes filter** — `π_t ∝ [Σ π_{t−1} A] · L_t` —
  over a trained prior-state column. Zero train/serve skew, zero leakage surface, zero free
  parameters, free cold start. It satisfies §5.1's intent and fails the original wording, and the
  wording was changed deliberately rather than reinterpreted. Decisive reason: the L1 labeler is
  **non-causal within its window by design**, so a trained prior-state column carries post-t
  information that **no holdout, purge or embargo can detect**. Eliminating the channel beats
  guarding it.

### Resolved by Claude with reasons stated, not silently

- **Q-5 — leave-one-episode-out added** as a fourth subsample scheme. Classifier #1's state 2 is
  **one contiguous episode (1996-07 → 2002-05)** and **neither decade-drop touches it**. The three
  named schemes miss the actual failure mode. Costs no trials; invents no threshold.
- **Q-7 — the zero-trial λ/d diagnostic is IN scope**; a **λ sweep is NOT**, since it costs one
  registry trial per value against 42/44 used and needs its own ruling.
- **Q-8 — F-4's denominator is fixed**, not annotated: 246/587 = 41.91%, recorded as 41.84%. The
  fix breaks the pin at `test_platform_joint_diagnostics_record.py:133`; both in one commit.
- **Cold start** follows from the Bayes filter: the unconditional class distribution, free, with
  the *same rule at train and serve* — a rule that differs between them *is* train/serve skew.

### Still open — for Glenn, as plan checkpoints

- **Q-3** — §5.3 mechanism: hard gate / bounded turnover / magnitude-scaling. Research recommends
  magnitude-scaling as the only one satisfying both A7's letter and §5.1's "consume the
  probabilities, never the argmax". Hard gate would put the book **100% cash in 387/588 months**.
- **Q-4** — the 0.70/0.40 pair: re-pin (costs trials) or define relative to 1/K.
- **Q-6/A11** — reopened; must be written as a reversal.

### One further inconsistency found, recorded not scheduled

`report/weekly.py` computes the hysteresis state, persists it, hands `vol_targeted_tilt` the raw
probabilities anyway, and `assemble_weekly_report` then recomputes its own `probs.idxmax()` at
line 117 — while printing an explanation of the hysteresis cold-start rule at lines 134-139. **The
report narrates a state machine whose output it does not show.** Whoever touches §5.3 should look.
