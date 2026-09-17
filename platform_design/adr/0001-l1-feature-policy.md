# ADR-0001: L1 Feature Policy — Freeze the Walk-Forward Driver to the Evaluation's Reference Set

## Status

**Accepted, 2026-09-15.** Phase 7 (`07-regime-representation`), Wave 1. Requirement coverage:
**REG-01 in part** (see "Deferrals and open items" below for exactly which clauses); **INV-01
is deferred to wave 2 in full** (D-09) and is not addressed by this ADR beyond naming what it
will require.

## Context

Audit item A13 (`.planning/UAT-AUDIT-2026-09-09.md`) found that the platform's two L1
(regime-labeling) fits were not reading the same feature space:

- `backtest/driver.py::_window_active_features` — the walk-forward's per-step L1 refit —
  **expands**: a feature is admitted once it has accumulated `feature_min_history` (120) months
  of in-window history. Its active set therefore grows over the backtest.
- `evaluation/report.py::_reference_label_columns` — the ONE full-sample "smoothed reference"
  fit used as the ground truth against which the walk-forward's real-time "filtered" labeling is
  scored — **freezes**: a column qualifies only if it is non-NaN for every month from the first
  decision date onward, computed once.

Measured directly against the pre-fix checkpoint, the walk-forward's active set changed **seven
times** across the 588-step backtest, while the reference stayed fixed at nine columns:

| Decision date | Active feature count (pre-fix, walk-forward) |
|---|---|
| 1972-01-31 | 4 |
| 1972-02-29 | 6 |
| 1972-04-30 | 8 |
| 1973-02-28 | 9 |
| 1986-06-30 | 10 |
| 1995-02-28 | 12 |
| 2000-01-31 | 13 |

Design §5.4's headline metric — the sojourn/detection-lag ratio — is defined as how long the
*filtered* labeling takes to agree with the *smoothed* reference. When the two are fit on
different and time-varying feature sets, their disagreement is not purely detection delay; it
also contains a feature-set difference that grows and shrinks over the backtest. This made the
§5.4 ratio **not interpretable** as reported in `.planning/BASELINE-v1-tracer-bullet.md`
(0.591, 164-month detection lag).

**This is a policy disagreement, not a defect.** `_window_active_features` and
`_reference_label_columns` were each internally correct for the purpose they were separately
written for — an expanding admission rule is a reasonable way to let a walk-forward use more
data as it becomes available; a frozen full-decision-range set is a reasonable way to guarantee
one hindsight fit covers every decision date. Neither function has a bug. The problem is that
§5.4 silently assumed they agreed. That is why this is a policy decision recorded as an ADR,
not a patch to either function.

## Decision

**The walk-forward driver's L1 labeler is frozen to a constant feature space, computed once per
evaluation run by `report.py::_reference_label_columns` and threaded through as an explicit
parameter** (`frozen_l1_features` / `_refit_l1`'s `frozen_features` keyword) into every L1 fit
the run performs — the strategy leg, the no-regime-ablation leg, and the full-sample smoothed
reference fit itself. All three now resolve to the exact same feature list by construction,
never by convergent-but-independent computation.

**The frozen set is these TEN columns** (order as returned by `_reference_label_columns`,
verified against the D-02-A-recomputed dev `monthly_features` checkpoint):

1. `cape_shiller`
2. `credit_spread_baa_aaa`
3. `curve_10y3m`
4. `div_yield`
5. `oil`
6. `real_rate_level`
7. `realized_vol_1m`
8. `realized_vol_3m`
9. `trailing_return_1m`
10. `trailing_return_3m`

(Not ten from the start of this phase — see "Considered Options" #3 below for why the set grew
from nine to ten mid-phase, and why that growth was itself a decision, not an accident.)

**`_refit_l2` and `driver.py::_cv_safe_active_features` (L2's separate, still-expanding
admission path) are deliberately left untouched.** L2's admission rule narrows for a distinct
and legitimate reason unrelated to A13: it exists so `CalibratedClassifierCV` always has
`n_splits` examples of every class present before fitting the nowcaster. A13 is diagnosed and
fixed as an **L1** problem — the labeler's own internal consistency between two fits of itself.
Freezing L2's path as well was raised during this phase's discussion and consciously **not**
pursued; it is recorded here as a conscious choice with a structural parallel worth someone
re-examining later, not an oversight this ADR forgot to close.

## Considered Options

### 1. Expand the reference to match the driver's expanding rule — rejected on the merits

Instead of freezing the driver, widen `_reference_label_columns` so it also expands its
admitted set over time, matching the driver's `_window_active_features` rule. **Rejected**:
this would turn the reference from ONE full-sample hindsight fit into a SECOND walk-forward
fit. Design §5.4's detection lag is specifically "how long a real-time causal estimator takes
to agree with a non-causal, full-information estimator" — if the reference itself became
causal (expanding, refit-per-window), the ratio would compare two causal estimators, which is a
different quantity from detection lag entirely. Only freezing the driver preserves what §5.4 is
actually supposed to measure.

### 2. All 13 lean features, with pre-start back-fill imputation — rejected, evidence-backed

Rather than dropping the four late-starting lean columns (`curve_10y2y`, `gold`, `oil`,
`fred_vix` in the pre-D-02-A frame), back-fill each one's pre-start gap with its own first
observed value, so all 13 lean features qualify for the frozen set. This was not rejected by
argument alone — it was **executed once as a logged trial** (07-CONTEXT.md D-03; plan
`07-03-PLAN.md` Task 2; registry `trial_tag=P7-W1-impute-13col-REJECTED`, two rows, isolated
under `outputs/reports/platform/trials/impute-13col/`, never touching the published artifacts).
Its measured numbers, cited here so the rejection is evidence-backed rather than argued:

| Quantity | Frozen 10-col (accepted, published) | Impute 13-col (rejected, isolated) |
|---|---|---|
| Strategy `terminal_log_wealth` | 4.025085120485386 | 4.048185034686693 |
| Strategy `max_drawdown` | -26.42% (58 mo) | -17.91% (44 mo) |
| `wealth_delta` | +0.3778473581475139 | +0.40094727234882077 |
| `dd_delta` | -0.0661242048614149 | +0.01898849550676429 |
| §5.4 ratio (`n_resolved` of `n_transitions`) | 0.6014492753623188 (7 of 7) | 1.2686567164179106 (**3 of 6** — indicative only) |
| `pct_disagree` (`n_compared`) | 0.8089887640449438 (n=356) | 0.8319148936170213 (n=470) |
| Regime occupancy (min state) | 9.06% (no state below §4.4's ~8% floor) | **1.29%** — below design §4.4's ~8% occupancy floor |

**On several of these raw numbers the rejected variant looks nominally better** — a higher
terminal log wealth, a positive `dd_delta` versus the accepted variant's negative one. This is
recorded honestly, not hidden: **the rejection does not rest on these numbers.** It rests on the
substantive, structural reason declared before either variant ran (D-03/D-04): the imputation
is **non-causal by construction**. For each of `curve_10y2y`, `gold`, and `fred_vix` (see the
note on `oil` below), the pre-start gap is back-filled with that column's own first observed
value — there is no history before a series begins, so any pre-start fill fabricates data the
model could not have had at the time. Fabricating pre-1990 VIX levels would place an invented
stress feature inside a crisis classifier, which is precisely the class of look-ahead
contamination this project's honesty framework exists to prevent. The variant's isolated
occupancy floor breach (1.29%, against §4.4's ~8% floor) is additional corroborating evidence that its geometry is
degenerate, not the reason for rejection.

*Note on `oil`:* by the time this trial ran, `oil` had already been added to the frozen set
independently (D-02-A, see Considered Option #3), so the imputation variant's back-fill on
`oil` was a documented no-op — the imputation's *effective* policy imputes only `curve_10y2y`,
`gold`, and `fred_vix`. This does not change the rejection reasoning above; it is noted for
completeness (`07-MEASUREMENTS.md` §3).

### 3. Pin the frozen set to the pre-fix nine-column checkpoint and document the staleness — rejected

The frozen set was originally measured (D-02) as **nine** common-support columns against the
`monthly_features` checkpoint as it stood at plan time. Research flagged, and plan-time
verification against the live code and data confirmed, that this checkpoint was **stale**:
`compute_lean_features` assigns `features["oil"] = monthly_raw["oil"]`
(`transforms_monthly.py:253`) — an unwindowed passthrough with no truncation — yet
`monthly_raw.oil` runs from **1962-01-31** (776 non-NaN months) while the on-disk
`monthly_features.oil` began only at **1985-02-28** (431 non-NaN months). The on-disk feature
column could not have been produced from the current `monthly_raw`; it predates a splice-work
rebuild that gave `oil` its full history. Rebuilding `monthly_features` from the cached
`monthly_raw` (plan `07-02-PLAN.md`, no network, offline recompute) changed nothing except
`oil` — every other column was byte-for-byte identical before and after — and
`_reference_label_columns` itself, called against the corrected checkpoint, now resolves to
**ten** columns, not nine.

The option under consideration here was to accept the nine-column set as measured and document
its staleness as a known limitation, rather than spend a plan recomputing the checkpoint.
**Rejected**: Phase 7's entire purpose is to decide what the L1 labeler is allowed to see.
Locking in a set that is wrong by *accident* — an artifact of a stale checkpoint, not an
analytical choice — would undermine the phase's own reason for existing, and certifying numbers
against evidence already known to be stale is exactly the failure mode
`.planning/UAT-AUDIT-2026-09-09.md` was written to catch and prevent from recurring.

## Consequences

**Every affected figure moved, and moved for a COMPOUND reason this record cannot separate.**
Two changes landed together: (1) this ADR's driver freeze (D-01), and (2) the checkpoint
correction that grew the frozen set from nine to ten columns (D-02-A, Considered Option #3
above). The pre-fix numbers below reflect BOTH the pre-fix expanding driver policy AND the
stale nine-column checkpoint simultaneously. **Attributing the whole delta between the pre-fix
and post-fix columns to either cause alone would repeat exactly the "fooled by its own
backtest" failure mode `.planning/UAT-AUDIT-2026-09-09.md` documents** — this record does not
attempt that attribution, and no reader of this ADR should either.

Measured outcomes under **three labelled states** (`07-MEASUREMENTS.md`, produced by two real
588-step L1+L2 walk-forward evaluations on 2026-09-14):

| Quantity | Pre-fix (9-col, stale checkpoint, compound baseline) | Post-fix, frozen (10-col, **accepted**) | Post-fix, impute (13-col, **rejected**, isolated) |
|---|---|---|---|
| `pct_disagree` (`n_compared`) | 0.8276595744680851 (389/470) | 0.8089887640449438 (288/356) | 0.8319148936170213 (n=470, isolated run) |
| §5.4 ratio (`n_resolved`/`n_transitions`) | 0.591 (4 of 6) | 0.6014492753623188 (7 of 7) | 1.2686567164179106 (3 of 6) |
| Multiclass Brier | 0.2087 | 0.20717126722358387 (n_steps=356) | recorded, isolated |
| `wealth_delta` | +0.379267 | +0.3778473581475139 | +0.40094727234882077 |
| `dd_delta` | -0.014364 | -0.0661242048614149 | +0.01898849550676429 |
| Strategy `terminal_log_wealth` | 4.0265 | 4.025085120485386 | 4.048185034686693 |
| Strategy `max_drawdown` | -21.24% (33 mo) | -26.42% (58 mo) | -17.91% (44 mo) |

### ⚠ Named limitation, foregrounded here (not a footnote): the window-narrowing on `pct_disagree` and the §5.4 ratio

**Read the `pct_disagree` and §5.4-ratio rows above together with their `n_compared` /
`n_resolved` denominators, not as bare percentages or ratios.** The pre-fix column's `470` and
the frozen column's `356` are **different populations, spanning different date ranges**:
pre-fix spans **1974-02-28 → 2020-12-31**; the frozen post-fix column spans
**1974-02-28 → 2017-05-31** — 232 further months are simply absent from the frozen comparison.

**82.77% → 80.90% is NOT a clean 1.9-point improvement measured over the same population.** The
frozen policy's walk-forward produced **232 of 588** L2-degraded steps ("at least 2 classes"
CV-fold solver failures, an existing documented graceful-degrade behavior), leaving only **356**
non-degraded steps to compare — versus **118 of 588** degraded steps under the pre-fix baseline
(and, separately, under the rejected imputed variant, which reproduces the pre-fix baseline's
470-month, 2020-08-dated window almost exactly). The frozen policy's `pct_disagree` and §5.4
ratio therefore rest on a **smaller and differently-dated** sample than either the pre-fix
comparison or the (rejected) imputed alternative would give. The same caveat applies to the
ratio's "7 of 7" resolved-transition count: it means 7-of-7-**available-within-the-narrower-
356-month-window**, not 7-of-7 over the full 588-step decision range.

**Mechanism, verified vs. inferred (do not overstate this).** Verified directly against
`driver.py`: `frozen_l1_features` is threaded into `run_backtest`'s per-step loop and reaches
**only** `_refit_l1` (`driver.py` lines 461 and 473, both call sites). `_refit_l2`'s signature
(`driver.py` lines 266-271) takes **no** `frozen_features`/`frozen_l1_features` parameter at
all — L2 can only be affected through `train_states`, i.e. through L1's changed OUTPUT LABELS,
never directly by which columns L1 was frozen to. This is **consistent with** (not proof of) the
observed occupancy shift in the smoothed reference (state 0: 1.6% pre-fix → 11.51% frozen
post-fix) and the degrade-count shift (232/588 frozen vs. 118/588 pre-fix) — a different L1
label sequence changes which per-step CV folds present only one class. **The specific causal
chain from feature-set to per-window class-imbalance degrade is NOT traced or measured here** —
this record states only that the code path makes label-mediation the only *possible* channel,
and explicitly declines to assert more than that.

**This finding does not reopen the policy decision (D-04, restated below), and it did not favor
the rejected imputed alternative** even though the imputed variant's window happens to match the
pre-fix baseline's more closely — see "Selection criterion" immediately below.

**Every Phase 5 evaluation artifact is re-dated by this change** (`outputs/reports/platform/*`
now reflects the frozen ten-column policy, generated 2026-09-14/15). **The A13 caveat is
retired** on the strength of two artifacts, together: (a) `TestFrozenPolicyEquivalence`
(`tests/unit/test_platform_backtest_driver.py`), proving the driver and reference resolve to
identical column sets at every sampled decision date, and (b) the §5.4 ratio now published with
its resolved-transition denominator (see `platform/plotting/core.py::A13_CAVEAT`'s rewritten
text and `platform/evaluation/report.py::assemble_backtest_report`'s pre/post comparison
section, both landed alongside this ADR).

## Selection criterion (restates D-04)

**The policy was chosen on a structural requirement, declared before either wave-1 run in this
document executed: the driver and the reference must fit on the same feature space.** This is
the SAME criterion Considered Option #1 uses to reject expanding the reference instead. The
disagreement percentage, the §5.4 ratio, and both ablation deltas (`wealth_delta`, `dd_delta`)
are **reported, never used to choose or re-run**. This held even after the window-narrowing
finding above surfaced — which superficially looks unfavorable for the frozen policy's sample
size — and even though the rejected imputed variant's `wealth_delta`/`dd_delta` are nominally
more favorable (Considered Option #2's table). **Using `dd_delta` as a veto that sends the
policy back for revision was explicitly offered during this phase's discussion and declined**:
that would be fitting the feature policy to the drawdown metric, which is exactly the kind of
post-hoc rationalization this project's honesty framework exists to prevent (design §22, the
trial registry, deflated Sharpe). Gate semantics for what happens next are separate from
selection criteria — see "Wave-2 gate" below.

## Trial ceiling

**Stated as a formula, not a fixed number**: `registry_rows_added = 2 x N_full_evaluation_runs`.
Derived from the two independent `append_trial` call sites inside one
`run_full_backtest_evaluation` call: the strategy leg's `run_backtest`, and the ablation leg's
`no_regime_ablation`, which itself delegates to `run_backtest` a second time — each call
appends exactly one registry row (pinned by
`tests/unit/test_platform_backtest_driver.py::TestTrialTag::test_one_evaluation_appends_exactly_two_rows`).

**Live count, read via `read_trials()` at ADR write time: 42 rows, read 2026-09-15T00:14:31Z.**
This is a live read, not a figure quoted from a planning document — the count has already moved
more than once during this phase alone (34 before wave 1 began, per `07-CONTEXT.md`; 38 at the
end of plan 07-03; 42 now, after two additional full evaluation runs performed during this
plan's Task 1 development to verify and correct the pre/post comparison section's date
formatting against the real checkpoint). **`30`, `34`, and `~35` are all stale figures and are
explicitly rejected as ceiling values** — they were quoted from earlier planning-stage estimates
before wave 1's actual trial count was known. **Exceeding the ceiling implied by this formula
for a given set of runs requires an explicit amendment to this ADR** — the formula itself is
not a cap the code enforces; it is the arithmetic a reader can use to check whether search-creep
occurred silently between two counts of `read_trials()`.

## Deferrals and open items

Each item below is recorded as a **decision**, with its rationale — not as an omission this ADR
forgot to close.

- **INV-01 is entirely deferred to wave 2, by D-09.** This phase's wave 1 (this ADR) scopes
  itself to the L1 feature-policy decision only; wave 2 gets its own `/gsd-plan-phase 7` planning
  pass with wave 1's actual numbers in hand. INV-01 requires: ingesting M2 (FRED `M2SL`, 1959+)
  and a credit aggregate (candidates: `TOTALSL` or `BCNSDODNS`), a dimensional-reduction
  screening pass over the resulting invariant candidates, and named survivors reported with their
  agency-tier alignment treatment. **None of this was attempted in wave 1.**

- **REG-01 is satisfied in part.** Covered by this ADR/wave 1: one documented, shared L1 feature
  policy between the driver and the reference (this decision), a test that fails if they diverge
  (`TestFrozenPolicyEquivalence`), the §5.4 ratio made interpretable with its resolved-transition
  denominator, and the ablation delta re-measured on both axes (`wealth_delta` and `dd_delta`,
  both reported above under all three labelled states). **Deferred to wave 2**: a second,
  independent labeler (classifier #2) on a disjoint relative/leadership feature set, the
  disjointness assertion, the statistical dependence measurement between the two labelings, and
  the joint (#1 × #2) allocation lift measured walk-forward against #1 alone.

- **Market-cap/GDP (the "Buffett indicator") stays blocked**, for its already-documented reason,
  restated here rather than quietly worked around: `config/platform_settings.yaml`'s
  `taxonomy` block comment (line ~248) records that no free 1962+ market-cap source exists in
  current ingestion — FRED's Wilshire series starts only around 1970 — so `buffett_indicator`
  remains a documented slow-layer *candidate*, not a lean-set member, until a Wilshire/market-cap
  series is ingested and `compute_lean_features()` gains a matching derivation.

- **Audit item A11 remains open and conscious.** No gate in this project can fail on a
  bad-but-working model, only on a broken one (D-07's plausibility-bands-only posture, which this
  ADR's own wave-1 measurements follow — every band above is `[ASSUMED]`/advisory, never a
  gate). A quality band on `dd_delta` (e.g. "fail if `dd_delta < 0`") was offered during this
  phase's discussion and declined, consistent with Phase 5 D-01 and Phase 6 D-16. To be
  re-examined at design freeze, not before.

- **Freezing L2's separate admission path (`_cv_safe_active_features`) was raised and
  consciously not pursued** in wave 1 — see "Decision" above. It remains a parallel worth a
  future contributor confirming is genuinely independent of A13's fix, not a follow-up task this
  ADR is implicitly assigning.

- **`canonicalize_states`' sort-key fallback (audit item A14) is flagged for wave 2's planning,
  not resolved here.** A14 was measured (2026-09-09) as affecting 0.2% of backtest steps under
  the pre-fix policy and closed as negligible at that time. It resurfaces as a wave-2 concern
  because the fallback's default key, `trailing_return_1m`, is one of classifier #1's admitted
  columns — and wave 2's D-10 disjointness rule excludes every one of classifier #1's thirteen
  lean columns from classifier #2's feature set. If classifier #2's own labeling ever needs the
  same fallback mechanism, `trailing_return_1m` will not be present in its feature space, so
  every classifier #2 fit that hits the fallback path would instead fall through to a
  centroid-column-0 default. This is flagged here so wave 2's planning pass inherits the
  question rather than rediscovering it.

- **The eight spec-less probe edges** (this plan's `<edge_assumptions>`):

  | Requirement | Edge | Status | Resolution |
  |---|---|---|---|
  | REG-01 | adjacency | **resolved** | Driver and reference sets are compared as ORDERED lists (plus a set comparison for the failure message), because `_reference_label_columns` derives a deterministic sorted order and `canonicalize_states` depends on column position. |
  | REG-01 | empty | **resolved** | An empty frozen list, or one shorter than K, raises a named `ValueError` at the boundary; at the measurement layer `n_compared == 0` is reported as a disjoint-span finding with a WARNING, never as perfect agreement; at the data layer a rebuilt frame narrower than the existing checkpoint refuses to narrow silently. |
  | REG-01 | ordering | **resolved** | Output order is stable and load-bearing; nothing in the call chain re-sorts the frozen list, pinned by test. |
  | INV-01 | boundary | **deferred**, wave 2 (D-09) | No invariant candidate constructed this pass. |
  | INV-01 | adjacency | **deferred**, wave 2 (D-09) | — |
  | INV-01 | empty | **deferred**, wave 2 (D-09) | — |
  | INV-01 | ordering | **deferred**, wave 2 (D-09) | — |
  | INV-01 | precision | **deferred**, wave 2 (D-09) | — |

- **Wave-2 direction decisions, recorded as carried-forward, not as silence.** Each was recorded
  during this phase's discussion to give the wave-2 planning pass its direction, and each was
  **deliberately not implemented in wave 1**, per D-09:
  - **D-10**: Disjointness for classifier #2 is asserted on RAW columns against classifier #1's
    full thirteen lean columns (not the ten actually frozen); derived RATIOS built from those
    columns remain admissible, since a ratio is scale-invariant and genuinely a different
    quantity from a level.
  - **D-11**: Classifier #2 uses the same freeze rule and the same 1972+ decision window as
    classifier #1, so the two labelings stay month-for-month comparable (required by criterion
    6's dependence measurement). Net effect combined with D-10: oil/equity relative strength
    **survives** (raw `oil` now runs 1962+, per this ADR's Considered Option #3); gold/equity
    relative strength does **not** (`gold` starts 1985-02 and fails the common-support freeze at
    the 1972+ window).
  - **D-12**: INV-01 ingests M2 and a credit aggregate; market-cap/GDP stays out, per the
    deferral above.
  - **D-13**: Classifier #2's (K, λ) is set by construction (λ from the existing
    feature-count formula, K pre-declared) — zero selection trials, consistent with the
    explicit non-goal against (K, λ) sweeps in wave 1 and wave 2 alike.
  - **D-14**: Both labelings' probability vectors feed the allocation tilt as two SEPARATE
    inputs — no product state space. Rationale: allocation already consumes probabilities, not
    labels (audit item A7 found `active_regime` from hysteresis gates nothing); a product space
    would also thin badly across ~590 decision months.
  - **D-15**: Statistical dependence between the two labelings is reported with several
    statistics (adjusted Rand, Cramér's V, normalized mutual information) side by side, with NO
    pre-declared pass/fail threshold — a human reads it and the judgement is recorded.
  - **D-16**: The deflated Sharpe at design freeze uses the WHOLE trial registry since project
    start (currently 42 rows, this ADR's own live read above), not just this phase's trials.
  - **D-17**: A trial ceiling is written into an ADR BEFORE running (this section, above) —
    exactly what this ADR does for wave 1's own trial budget.

## Wave-2 gate (restates D-06)

**Measurement is the gate, not the sign.** Wave 2 (classifier #2, the leadership/relative-
feature labeler) proceeds regardless of what `dd_delta` turned out to be in wave 1 — here,
**-6.61%**, more negative than the pre-fix (compound) baseline's -1.44% — provided it was
honestly measured and recorded, which it was (see "Consequences" above). Classifier #2 is a
different axis fit on disjoint features; its value does not depend on classifier #1 "winning",
and a weak classifier #1 result is, if anything, an argument FOR adding a second axis. This
phase's explicit non-goals (no raising K on classifier #1, no (K, λ) sweeps) also leave "halt
and fix classifier #1" with no sanctioned move available inside this phase — so there is no
alternative next step this gate could plausibly withhold wave 2 for.

---

*Amends: `platform_design/platform_design.md` §5.4 (Honesty metrics for L2). See the
cross-reference at that section.*

*Phase: 07-regime-representation. See also: `.planning/phases/07-regime-representation/07-CONTEXT.md`
(D-01 through D-17), `07-MEASUREMENTS.md` (the full measured record), `07-PREFIX-EVIDENCE.md`
(the reproduced pre-fix baseline), `.planning/UAT-AUDIT-2026-09-09.md` (A7, A11, A13, A14, A15).*

---

## AMENDMENT 2026-09-17 — the "§4.4 five-percent floor" cited above does not exist

**What was wrong.** This ADR cited design §4.4's occupancy criterion as a *five-percent floor*,
in the comparison table and again in the rejection rationale. §4.4 criterion 1 reads, verbatim:

> **Occupancy:** every state ≥ ~8% and ≤ ~35% of months.

There is no 5% threshold anywhere in §4.4, and the criterion is **two-sided** — it carries a
**~35% cap** that this ADR never mentioned. The error propagated from here into
`adr/0002-l1-second-classifier.md`, plans `07-08`/`07-10`/`07-12`, `07-MEASUREMENTS.md`,
`platform/labeling/classifier2.py`, and `platform/labeling/diagnostics.py`, whose
`_MIN_OCCUPANCY_THRESHOLD` was set to `0.05`. The implementation had **no cap check at all**, so
no regime solution in this project had ever been tested against half of criterion 1.

**What this does NOT change.** The rejection of the imputed variant stands, unaltered. It rested
on a structural argument declared before either variant ran — the imputation is non-causal by
construction — and explicitly *not* on the occupancy number. The corroborating observation also
survives the correction: **1.29% fails the real ~8% floor exactly as it failed the fabricated
5% one**, and by a wider margin. The accepted variant's 9.06% minimum clears the real floor too.

**What it does change.** Scored against the correct two-sided criterion, using the occupancy
recorded in `07-MEASUREMENTS.md`:

| variant | occupancy by state | floor ≥~8% | cap ≤~35% |
|---|---|---|---|
| pre-fix | 1.6 / 14.0 / 31.9 / 40.6 / 11.9 % | **state 0 fails** | **state 3 fails (40.6%)** |
| frozen (accepted) | 11.51 / 9.06 / 35.54 / 31.51 / 12.37 % | all pass | state 2 at 35.54% — **marginal** |

Two consequences worth stating plainly:

1. **The frozen policy looks better under the correct criterion, not worse.** It repairs a breach
   at *both* ends — floor 1.6% → 11.51%, cap 40.6% → 35.54%. That strengthens this ADR's
   decision rather than undermining it.
2. **State 2's 35.54% is a 0.54pp overshoot against an explicitly approximate bound** ("≤ ~35%").
   It is recorded as marginal and within tolerance, not as a failure. Read the number, not the
   boolean. It is close enough to the line to be worth watching on the next refit.

**Where this was caught.** Executing plan 07-08 (classifier #2) produced occupancy of
15.37 / 46.12 / 38.51 %, which the occupancy test passed because that test checked only the
invented 5% floor. Classifier #2 breaches the real cap on two states, one of them by 11pp. The
test could confirm and never fail — the same evidence-shape failure class as criterion 8's
`grep -v platform` exit check. `diagnostics.py` now implements both bounds (report-only per
D-02), and `tests/unit/test_platform_labeling_classifier2.py` carries a `strict=True` xfail
recording classifier #2's breach, which fails the suite the moment the breach is fixed.
