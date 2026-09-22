# ADR-0002: Classifier #2 — A Second L1 Labeler on a Disjoint Leadership/Relative Feature Set

## Status

**Accepted, 2026-09-21**, by plan 07-12. Phase 7 (`07-regime-representation`), wave 2.

**What this acceptance covers, stated before anything else so it cannot be over-read.** What is
accepted is the **decision**: a second labeler was specified on a disjoint leadership/relative
feature set, its constants were pinned by rule rather than chosen by search, it was fit, and its
relationship to classifier #1 was measured. **This is not a finding that classifier #2 adds an
independent axis, and nothing in this document may be cited as one.** Criterion 6's
pre-registered test returned **INCONCLUSIVE** and the criterion is **UNRESOLVED**; criterion 7's
wealth lift came out **negative** (−0.123438 nats over 588 steps, 1972-01-31 → 2020-12-31).
Both are recorded in full below, with their windows.

Requirement coverage is split accordingly and is claimed at exactly that width:
**REG-01 is claimed PARTIALLY** — criterion 6 is the named open item — and **INV-01 in full**
(D-09's wave-2 deferral, closed by plan 07-07's screen). See § Requirement coverage.

### How this status was reached — the original pre-declaration, kept verbatim

The two paragraphs below are the Status section as written on 2026-09-17, before classifier #2
was fit. They are preserved rather than replaced, because the ordering they describe is the
evidence that the constants were not chosen to flatter a result:

> **Proposed, 2026-09-17.** Phase 7 (`07-regime-representation`), wave 2 (planned and executed in
> wave 3). Requirement coverage: **REG-01's deferred clauses** (a second, independent labeler on a
> disjoint feature set; the disjointness assertion; the dependence measurement; the joint
> allocation lift) and **INV-01 in full** (D-09's wave-2 deferral, closed by plan 07-07's screen).
>
> **This ADR is written at Proposed BEFORE classifier #2 is fit and before any evaluation run
> happens.** That ordering is the point, not an accident of scheduling: D-17 requires the trial
> ceiling to be on the record before the runs it budgets for, and D-13 spends **zero** selection
> trials choosing classifier #2's constants. A record written after the fit could not distinguish
> "pinned by construction" from "chosen because it looked better." Plan 07-11 amends this ADR to
> Accepted once the measurements from plans 07-09 and 07-10 exist — those measurements will be
> appended to a decision already made, never used to revise it.

One correction to that text, labelled rather than silently applied: acceptance was carried out by
plan **07-12**, not 07-11. Plan 07-11 measured criterion 7 and wrote `07-JOINT-LIFT.md`; 07-12 is
the closure plan that accepts this ADR. The substance — measurements appended to a decision
already made, never used to revise it — held.

## Context

Phase 6's P3 sign-off accepted classifier #1's regimes **with an explicit caveat**: they are
*crisis* regimes, not *allocation* regimes. Classifier #1 is fit on the 13-member lean taxonomy
(fast ∪ slow), of which ADR-0001 froze **ten** columns for the walk-forward — `cape_shiller`,
`credit_spread_baa_aaa`, `curve_10y3m`, `div_yield`, `oil`, `real_rate_level`,
`realized_vol_1m`, `realized_vol_3m`, `trailing_return_1m`, `trailing_return_3m`. Read together
these describe one axis: how stressed is the market. They say almost nothing about *which sleeve
is leading* — equities, long duration, or commodities — which is the question L4's allocation
tilt actually needs answered.

REG-01's remaining clauses therefore ask for a **second** labeler, fit unsupervised on a feature
set disjoint from classifier #1's, with the two labelings' statistical dependence measured
(criterion 6) and their joint allocation lift measured walk-forward against #1 alone
(criterion 7).

### The four premises the WAVE-2 OPENING AMENDMENT corrected

Every decision below was written against the amendment's corrected facts, not against the
pre-wave-1 premises D-10…D-17 were originally drafted under. Stating them here is what makes
this ADR readable a year from now without re-deriving why the constants look the way they do:

1. **Criterion 8 was FALSE — `platform/` DOES import from the legacy library.** ROADMAP's
   "verified fully decoupled" rested on a grep whose final `| grep -v platform` discarded every
   violation, so the check **could not fail**. An AST scan finds **31 real legacy import sites**,
   all predating Phase 7. Effect here: D-10/D-11's "port the relative-strength algorithms, do
   not import them" still stands, but because the coupling must not be *widened* — not because
   `platform/` was ever clean. `tests/unit/test_platform_legacy_import_ratchet.py` pins the count
   at 31 and may only decrease. This ADR's work adds **zero** new sites.
2. **The trial ledger was reset behind a provenance header carrying 38 prior trials.** Wave 1
   appended 4 untagged wiring-verification rows which inflated D-16's denominator; the ledger was
   archived intact (`registry/archive/trials-pre-P7W1-reset.jsonl`, 42 rows) and restarted with a
   header recording `prior_genuine_trials=38`, `discarded_smoke_rows=4`. Effect here: every trial
   count in the "Trial ceiling" section below is a **live** `total_trial_count()` read that
   consumes that header — never a raw post-reset row count and never a figure quoted from a
   planning document.
3. **`canonicalize_states` would have silently canonicalized classifier #2 on centroid column
   0.** Its default sort key was `trailing_return_1m`, one of classifier #1's own 13 columns, so
   a labeler fit on a set disjoint from those 13 would hit the fallback path on **every** fit and
   assign state IDs by a degenerate default — while occupancy, dependence and joint lift all kept
   appearing to pass. Plan 07-05 deleted the fallback and made `sort_column` a keyword parameter
   that **raises** when absent. Effect here: classifier #2 passes its own ordering column
   explicitly (see Decision (d)).
4. **L2 degradation narrowed wave 1's comparison window to 2017-05.** The frozen policy degraded
   **232 of 588** L2 steps (vs 118 pre-fix), so wave-1 measurements rest on **356** steps ending
   2017-05 against a **470**-step baseline ending 2020-12. Classifier #2's different label
   sequence would move that count a third time. Effect here: the routing decision below, which
   the amendment's item 4 explicitly forbids absorbing quietly.

## Decision

Six values, pinned at a blocking decision checkpoint by the developer on **2026-09-17** before
classifier #2 was fit and before either walk-forward ran. The verbatim record, including each
value's failure signature, is
`.planning/phases/07-regime-representation/07-DECISIONS-07-08.md`. **Zero selection trials were
spent choosing any of them** (D-13) — no fit was run and no result was looked at in order to
pick a value here.

### (a) The frozen candidate column list — "Lean 8" (D-10, D-11)

| # | Column | First valid month | Source |
|---|---|---|---|
| 1 | `rs_equities_bonds` | 1962-01-31 | `features/relative.py` relative strength |
| 2 | `rs_oil_equities` | 1962-01-31 | `features/relative.py` relative strength |
| 3 | `equities_tr_mom_12m` | 1963-01-31 | `features/relative.py` trailing momentum |
| 4 | `long_duration_tr_mom_12m` | 1963-01-31 | `features/relative.py` trailing momentum |
| 5 | `oil_mom_12m` | 1963-01-31 | `features/relative.py` trailing momentum |
| 6 | `corr_equities_tr_long_duration_tr_24m` | 1963-01-31 | `features/relative.py` rolling cross-correlation |
| 7 | `cpi_acceleration` | 1962-04-30 | `features/relative.py` inflation acceleration |
| 8 | `m2_gdp` | 1962-02-28 | INV-01 survivor, plan 07-07 |

**n = 8.** One momentum horizon per sleeve: the 6m and 24m twins are dropped as near-collinear
with the 12m, and `credit_gdp` — the other INV-01 survivor — is dropped because plan 07-07
measured its correlation with `m2_gdp` at **0.957–0.969 across eras**, so admitting both would
add a near-duplicate column and inflate λ (which is `4n`) without adding information.

**Disjointness (D-10) verified live before the checkpoint resolved:**
`set(candidates) & lean_feature_set(cfg) == set()`, with `lean_feature_set` returning exactly 13.
D-10 asserts disjointness on RAW columns against all 13 of classifier #1's lean columns, not
against the 10 ADR-0001 actually froze. Ratios *derived from* #1's columns remain admissible — a
ratio is scale-invariant and a genuinely different quantity from a level — which is precisely why
`rs_oil_equities` survives despite `oil` being one of classifier #1's ten.

**Freeze rule (D-11) verified:** every one of the eight is non-NaN for every month from the
1972-01-31 first decision date onward, with margin — the latest first-valid month is 1963-01-31,
nine years early. Classifier #2 uses the **same** freeze rule (`_reference_label_columns`,
reused unmodified, never reimplemented) at the **same** 1972+ decision window as classifier #1,
so the two labelings stay month-for-month comparable, which criterion 6's dependence measurement
requires.

### (b) K = 3 (D-13)

By construction, not by search: three asset sleeves appear in the candidate set — equities, long
duration, oil — so three leadership states. The K falls out of the feature set's structure. No
fit was run to choose it, and re-fitting at a different K to see which looks better would be a
selection trial D-13 forbids.

Classifier #1's K = 5 is deliberately **not** inherited. A leadership axis is a different kind of
question from a crisis axis and carries no obligation to the same state count.

### (c) λ = 32.0 — derived, not chosen (D-13)

`λ = 4n = 4 × 8 = 32.0`.

This is the same feature-count-scaled formula classifier #1 instantiates (13 lean columns → λ
52.0, `config/platform_settings.yaml`'s `labeling` block) and that 03-RESEARCH Pitfall 5
documents. There is no discretion in this value: the checkpoint **rejects** any λ other than
`4 × len(features)`, and `classifier2_config()` raises a named `ValueError` at config-read time
if the two ever drift apart. D-13's arithmetic is an invariant the code fails on, not a comment.

### (d) Canonical ordering column = `rs_equities_bonds` (the parameter plan 07-05 added)

`canonicalize_states(..., sort_column="rs_equities_bonds")`, passed explicitly at classifier #2's
single call site. States are numbered by ascending centroid coordinate of the equity/bond
relative-strength ratio: state 0 is the most bond-leading, state K−1 the most equity-leading.

`07-RESEARCH.md` Assumption A1 proposed this column and rated the risk **low**, because
`sort_column` decides only which state index is called 0 — never occupancy, never dependence,
never lift. Its failure mode is loud by construction: if this column were ever dropped from the
frozen list, plan 07-05's amended `canonicalize_states` raises `ValueError` at fit time rather
than silently reordering. The pre-07-05 fallback to centroid column 0 is deleted; zero
occurrences remain in the tree.

### (e) Criterion-7 measurement routing — L1 decision-bearing, L2 observational and firewalled

**This is a fourth option, not one of the three the plan offered.** It is recorded as its own
choice rather than relabelled as the nearest listed option, because relabelling it would hide the
firewall clause that is its defining feature. The developer's instruction, verbatim:

> "Only use L1 for any choices, decisions, D-16 denominator, Sharpe calculation, D-17
> requirements, etc., however, also calculate and report L2 for human evaluation. Do not include
> any L2 calculations towards D-16, D-17, Sharpe, etc."

See "The L2-window decision" section below, where this is stated as an implementation contract
with its named consequence.

### (f) `allocation.blend_weight_1` = 0.50 (D-14)

Equal weight on classifier #1's and classifier #2's probability vectors in
`blend_regime_tilts` (plan 07-10). Equal weight is the **no-information prior**: nothing known
before the runs favours either classifier, and any other value would need an empirical
justification, which is exactly the selection trial D-13 forbids. It is declared here rather than
in 07-10 because sweeping it later would be an unregistered selection dimension the trial ceiling
does not budget for.

### Config

Both `labeling_2` (K, lambda, n_restarts, sort_column, features) and
`allocation.blend_weight_1` live in `config/platform_settings.yaml`, read defensively via
`cfg.get()` at every consumer and **not** added to `_REQUIRED_PLATFORM_SECTIONS` — the
established convention for additive platform config. `lean_feature_set()` still returns exactly
13 and `validate_platform_config()` still passes. The `labeling` and `taxonomy` blocks are
untouched: classifier #1 is byte-identical after this change.

## Considered Options

### 1. Route criterion 7's joint lift through L1's own labels only — partially adopted

Skip L2 entirely, feeding the tilt from L1's labels via `driver.py`'s existing `use_regime_tilt`
branch. **Pros:** removes the L2-degradation confound so both legs are measured on the same full
window. **Cost if adopted alone:** classifier #2 never exercises L2, so nothing whatsoever is
learned about whether its labels are nowcastable — a real gap, since a leadership labeling that
cannot be nowcast is useless in production regardless of its in-sample lift. That cost is what
the adopted hybrid pays to avoid: the L1-only leg is kept as the decision-bearing measurement,
and the L2 leg is computed anyway, for reading only.

### 2. Route joint lift through L2 as classifier #1 does — rejected on the merits

Run the full L1→L2→L4 stack the production weekly report uses, reusing `run_backtest` unchanged.
**Rejected**: classifier #2's different label sequence will move the degrade count a third time,
so the joint and #1-alone legs would be measured on **two different windows** — a comparison this
phase cannot honestly make. Resolving it properly means making L2's CV robust to rare classes,
which re-dates every prior phase that touched L2 and is out of this phase's scope. Adopting this
as the *decision-bearing* routing would have bought production fidelity at the price of an
apples-to-oranges headline number.

### 3. Measure both routings and count both toward the registry — rejected on cost

Run both and log both as trials. **Rejected**: doubling the registered evaluation runs raises
D-16's denominator for **every headline number in the project**, weakening the deflated Sharpe of
results that have nothing to do with classifier #2. D-17's spirit is to minimize registry growth.
The adopted routing takes this option's information benefit while paying none of its denominator
cost, by appending the L2 leg with the `NO_REGISTRY` sentinel — see the firewall clause below for
the residual risk that trade does not eliminate.

### 4. `BCNSDODNS` as the INV-01 credit aggregate — rejected, pre-ingestion

Nonfinancial corporate business debt securities. **Rejected** for being **quarterly-native**: 305
observations from 1945-10, so putting it on a monthly spine requires exactly the forward-fill
treatment `M2SL` and `TOTALSL` avoid by being monthly-native. A forward-filled quarterly series
inside a monthly labeler manufactures two months of stale, invariant readings per quarter, which
a jump model with a per-jump penalty will happily read as regime persistence that is an artifact
of the interpolation.

### 5. `TOTBKCR` as the INV-01 credit aggregate — rejected, pre-ingestion

Total bank credit. **Rejected**: starts in **1973**, after the 1962 spine start and after the
1972-01 first decision date, so it cannot survive D-11's common-support freeze. `TOTALSL`
(1943-01, monthly) was ingested instead.

### 6. Market-cap/GDP (the "Buffett indicator") — stays blocked, restated not worked around

D-12 blocks `buffett_indicator` for an already-recorded reason, restated here rather than quietly
routed around: **no free 1962+ market-cap source exists in current ingestion** — FRED's Wilshire
series starts around 1970, after the spine start. It remains a documented slow-layer *candidate*,
not a lean-set member and not a classifier-#2 candidate, until a Wilshire/market-cap series is
ingested and `compute_lean_features()` gains a matching derivation. Nothing in this ADR changes
that status.

### 7. Gold/equity relative strength — admissible in principle, excluded in fact

D-10's ratio rule admits it (a gold/equity ratio is a different quantity from `gold` the level,
which is one of classifier #1's 13). D-11's freeze rule excludes it: `gold` begins **1985-02** in
`monthly_raw` and cannot be non-NaN back to the 1972-01 decision date. `oil` runs from
**1962-01** and does survive, which is why `rs_oil_equities` is in the frozen eight and no gold
ratio is. `features/relative.py`'s `DEFAULT_RELATIVE_PAIRS` encodes this exclusion at the source
and a test asserts no produced column name contains "gold".

### 8. A product `(state_1, state_2)` state space — rejected per D-14

Cross the two labelings into K₁ × K₂ = 5 × 3 = 15 joint states and tilt on the product.
**Rejected**: across roughly **590** decision months with occupancy never uniform, rare cells
fall below design §4.4's ~8% floor — at perfectly uniform occupancy each cell would hold
~39 months, and occupancy is not uniform, so the thin tail is thinner than that. The two
probability vectors feed the allocation tilt as two **separate** inputs instead, blended at the
weight pinned in (f). The developer's counter-proposal at the checkpoint — one-hot encoding the
(regime₁, regime₂) cells as features and letting selection decide which carry signal — is filed
as **ROADMAP T0.10** rather than adopted; see "Standing objections" below for why it cannot be
expressed against today's L3/L4.

### 9. Selecting the feature list, K and λ empirically instead of pinning them — rejected for this phase, filed as ROADMAP T0.9

This is the option the developer argued for at the checkpoint. It is recorded here as a
considered option rather than buried, because pinning is a **current-capability constraint, not a
claim that selection is wrong**. See "Standing objections" below.

## The L2-window decision

**Chosen routing: the L1-only leg is decision-bearing; the L2 leg is observational and
firewalled.** Stated as an implementation contract that plans 07-09, 07-10 and 07-11 must honor:

- **The L1-only leg is decision-bearing.** It is what criterion 7's joint-lift number reports. It
  appends to the trial registry with a `trial_tag`, it counts toward **D-16's** deflated-Sharpe
  denominator, and it counts toward **D-17's** ceiling in the section below.
- **The L2 leg is observational.** It is computed and reported for human reading, appended with
  the `NO_REGISTRY` sentinel so it contributes **zero** rows and does **not** count toward D-16,
  D-17 or the deflated Sharpe.
- **Both legs must state the window they were measured on, inline with the number** — the same
  binding condition wave 1's UAT imposed on criterion 3. A bare percentage or a bare delta with
  its denominator in a footnote is not an acceptable report.
- **⚠ The L1-only leg is NOT comparable to wave 1's measured numbers.** Wave 1's
  `wealth_delta` of **+0.377847** and `dd_delta` of **−0.066124** were measured **through L2, on
  356 steps ending 2017-05**, against a 470-step baseline ending 2020-12. Criterion 7's #1-alone
  baseline is **re-run through the L1-only harness**, not reused from wave 1, so its deltas
  describe a different measurement path over a different window. Any table placing the two
  side by side without saying so is misreporting. This is foregrounded here, not footnoted,
  exactly as ADR-0001 foregrounded its own window-narrowing limitation.
- **⚠ The firewall clause, binding: nothing downstream in this phase may change on the basis of
  the L2 leg.** It is reported and not acted on. The honesty framework fences *fitting*, not
  *looking* — but a number a human reads and reacts to is selection in effect, registered or not.
  `NO_REGISTRY` keeps the arithmetic honest; it cannot keep the reader honest. That gap is closed
  by this constraint being stated, not by assuming it. If a future phase does act on the L2 leg,
  that is a **recorded amendment to this ADR**, never a silent reuse.

**Failure signature for this routing:** the two legs disagree in **sign** on joint lift. That
would mean the L1-only reading and the production stack disagree about whether classifier #2
helps at all, and neither number should be trusted until the disagreement is explained.

## Consequences

- **Classifier #1 is untouched.** Classifier #2 is an `add-alongside`, not a `promote`: it
  persists to new checkpoint names `regime_labels_2` / `regime_confidences_2`, reads a new
  `labeling_2` config section, and adds no survivor to the `taxonomy` tiers. §5.4's ratio, the
  Brier and confusion tables (whose `y_true` is classifier #1's labeling), and every wave-1
  measured number are therefore **not** re-dated by this ADR. Promoting #2 would have re-opened
  all of them, and this phase's explicit non-goals leave no sanctioned remedy if that went badly.
- **Criterion 7's headline number changes meaning**, per the routing section above: it measures
  the L1→L4 path, not the production L1→L2→L4 path.
- **Two new registry-bearing evaluation runs are authorized** and budgeted below. No others are.
- **What would later force a promote**, recorded so the bar is explicit rather than negotiable
  after the fact: low measured dependence (criterion 6) *together with* positive joint lift on
  both `wealth_delta` and `dd_delta` inside the confirmed plausibility bands and a DSR clearing
  0.5; or classifier #2's own leg beating #1's on both §23.1 axes while #1's `dd_delta` stays
  negative; or design freeze licensing a blend-weight sweep, at which point (f) stops being a
  constant and the blend earns its own ADR.

## Deflated-Sharpe estimator

Lifted verbatim from `.planning/phases/07-regime-representation/07-DSR-ESTIMATOR-NOTE.md` (plan
07-06), which read the primary source — Bailey & López de Prado (2014), *The Deflated Sharpe
Ratio* — in full:

> **Chosen `sharpe_variance` estimator: sample variance of this project's own trial registry's
> observed Sharpe-ratio metrics (non-header rows), read live via `registry_sharpe_variance()`,
> falling back to a fixed placeholder constant of `1.0` — never `0.0` — whenever fewer than two
> such observations exist.** Justification: the registry is design §22's own declared "true DSR
> denominator," so its own population is the only source that can ever reflect what this
> project's search actually explored, and a fixed proxy divorced from the registry would report
> the same number regardless of how much or how little was searched. Named limitation: verified
> live this session, the registry currently holds **zero** usable Sharpe-bearing rows (0 of 42
> archived, 0 of 1 post-reset), so the estimator runs entirely on its `1.0` placeholder fallback
> today — a deliberate, logged, non-silent assumption, not a measured quantity — and the registry
> additionally mixes heterogeneous strategy legs rather than repeated draws from one strategy
> class, so even once real Sharpe-bearing rows accumulate, the resulting sample variance will be
> a cruder, pooled-population estimate than the paper's own "one strategy class" framing strictly
> assumes. This is why the result is reported as "a deflated Sharpe ratio," not "the deflated
> Sharpe design §22 specifies," until both gaps close.

The honest claim boundary carried forward with it: `0.0` is rejected as a fallback specifically
because `expected_max_sharpe(n_trials, 0.0)` returns exactly `0.0` for every trial count, which
would silently **disable** the multiple-testing correction — the "systematically under-penalize
search" pattern `07-RESEARCH.md` names as this project's closest analog to a security defect.

## Trial ceiling

**Stated as arithmetic, in three separately-named components — never as one conflated number.**

**Component 1 — the INV-01 invariant screen (already spent).**
`rows_added = len(INVARIANT_CANDIDATES) = 2`, one `append_trial` row per candidate screened.
Measured by plan 07-07: `total_trial_count()` read **38** before the screen
(2026-09-17T15:09:17Z) and **40** after (2026-09-17T15:10:37Z), both rows tagged
`07-07-inv01-screen`. These two rows are **not** folded into component 2's formula — they come
from a different call site (`features/invariants.py`) with different arithmetic.

**Component 2 — criterion 7's evaluation runs (not yet spent).**
`rows_added = rows_per_run × N_runs`.

- `rows_per_run = 2`, derived from the two independent `append_trial` call sites inside one
  `run_full_backtest_evaluation` call: the strategy leg's `run_backtest`, and the ablation leg's
  `no_regime_ablation`, which delegates to `run_backtest` a second time. Each call appends
  exactly one row, pinned by
  `tests/unit/test_platform_backtest_driver.py::TestTrialTag::test_one_evaluation_appends_exactly_two_rows`.
  **This factor is re-derived for the chosen routing, not inherited from ADR-0001**: the L1-only
  routing changes *which* code path feeds the tilt (`use_regime_tilt` / a degenerate
  state-probability pair) but not *how many* `append_trial` sites one evaluation traverses, so 2
  is confirmed rather than assumed.
- `N_runs = 2` — the joint (#1 × #2) leg and the #1-alone baseline leg, both re-run through the
  L1-only harness (the baseline cannot be reused from wave 1; see the routing section).
- **Component 2 ceiling: 2 × 2 = 4 rows.** If plan 07-10 shares one ablation leg across both
  configurations rather than re-running it, the actual count is 3 — fewer than the ceiling, which
  is the only direction a ceiling permits.

**Component 3 — the L2 observational leg: exactly 0 rows.** Appended with the `NO_REGISTRY`
sentinel per decision (e). This is the arithmetic consequence of the firewall clause, stated so a
later reader can check it rather than trust it.

**Live cumulative reading: `total_trial_count()` = 40, read 2026-09-17T16:42:52Z**, at the time
this ADR was written, from `registry/trials.jsonl` via the provenance header (`38` prior genuine
trials + the 2 rows from component 1). This is a live read performed for this document, not a
figure copied from a planning file.

**Ceiling implied for the remainder of this phase: 40 + 4 = 44.**

**Standing warning, restated from ADR-0001 and extended: `30`, `34`, `35`, `38` and `42` are all
stale figures and none may be used as a ceiling value.** `38` is the header's
`prior_genuine_trials`, not a current count; `42` is the *archived* pre-reset ledger's row count.
**Exceeding the ceiling above requires an explicit amendment to this ADR** — the formula is not a
cap the code enforces; it is the arithmetic a reader uses to check whether search-creep occurred
silently between two `total_trial_count()` readings.

## INV-01 coverage

**Named survivors: `["m2_gdp", "credit_gdp"]`** — `m2_gdp = fred_m2sl / fred_gdp` and
`credit_gdp = fred_totalsl / fred_gdp`, both documented economic ratios a human can read and
reason about. Of the two, **only `m2_gdp` enters classifier #2's frozen eight**; `credit_gdp` is
dropped for near-collinearity (0.957–0.969 correlation with `m2_gdp` across eras), a rejection
recorded rather than silent.

**No anonymous principal component was admitted anywhere (design decision R4).** Dimensional
reduction appears in plan 07-07's screen **only as a discovery tool** — to look at how the named
candidates relate to one another. `compute_candidate_loadings()` reads only `pca.components_`
(the loadings matrix); `pca.transform()` is never called anywhere in
`platform/features/invariants.py`, so no component *score* was ever computed, let alone handed to
a caller who could mistake it for a feature. R4 is held structurally, not by convention.

**Era-stability tolerance: 0.15** (`LOADING_STABILITY_TOLERANCE`), assessed over **10 eras**
ending 1972-01-31 through 2017-01-31 via `loading_stability_across_eras()` on expanding windows
(`min_train=120`, `step=60`). Both candidates classified **stable**, with per-era PC1 loadings of
0.70711 and ranges of 4.4e-16 / 2.2e-16.

**⚠ Read that stability result with plan 07-07 §4's own caveat, not as a strong finding.** A PC1
loading of exactly 1/√2 on **two** standardized candidates is a mathematical property of
screening exactly two positively-correlated series — the first component must split its weight
evenly — not, by itself, evidence of genuine five-decade economic stability. The screening record
says so explicitly and this ADR does not upgrade its confidence in passing.

## Standing objections, recorded rather than absorbed

Two objections were raised by the developer at the checkpoint and are **not** resolved by this
ADR. They are recorded here, beside the constants they bear on, so a reader sees that pinning is
a **current-capability constraint and not a claim that selection is wrong**.

- **ROADMAP T0.9 — registered nested selection inside the walk-forward loop.** The developer's
  position: feature selection, dimensionality reduction and hyperparameter optimization are
  things an ML model should do, and should not be hardcoded. The Lean 8, K = 3 and λ = 32.0 in
  Decision (a)–(c) are accepted **for this phase only**, conditional on this roadmap item. The
  binding constraint is not selection itself but doing it *without look-ahead* and *without
  silently understating D-16's denominator* — nested selection inside the walk-forward loop, with
  every inner-loop configuration registered. Until that exists, pinning by construction is the
  only way to spend zero selection trials honestly; it is the lesser of two evils, not a
  preference.
- **ROADMAP T0.10 — learned regime-conditional allocation (interaction-aware L3/L4).** The
  developer challenged (f)'s premise, not its number: he does not think the two regimes should be
  mixed at all, preferring the (regime₁, regime₂) combinations one-hot encoded as features with
  selection deciding which cells carry signal — dropping regime₂ entirely if unpredictive.
  Investigated at the checkpoint: **there is no learner at L3 or L4 to consume such a design
  matrix.** `assets/returns.py::returns_by_regime_stats` is descriptive statistics, explicitly
  "NOT a supervised-learning target" and exempt from the registry and purged CV;
  `allocation/tilt.py::regime_tilt_weights` takes a probability-weighted average of that table.
  There are no coefficients anywhere. **The fixed weight exists because two probability vectors
  must become one weight vector and there is no model to learn the combination — not because
  blending was preferred over interaction.** D-14 therefore stands for this phase. Caveat carried
  forward with the proposal: one-hot encoding does not escape cell thinness — a cell holding 8
  months yields a column with 8 nonzero rows, so selection either drops it (collapsing toward the
  marginals, ≈ the blend) or retains it and overfits. At ~590 decision months and K₁ × K₂ = 15
  the expectation is mostly collapse, which is a prior to be tested, not a measured result.

## Deferrals and open items

Each is recorded as a **decision**, not as an omission.

- **Criterion 6's dependence measurement is plan 07-09**, not this ADR. Per **D-15** it will be
  reported with several statistics side by side (adjusted Rand, Cramér's V, normalized mutual
  information) with **no pre-declared pass/fail threshold** — a human reads it and the judgement
  is recorded.
- **Criterion 7's joint tilt and lift are plan 07-10**, under the routing pinned above.
- **This ADR's acceptance is plan 07-11.** It stays Proposed until the measurements exist.
- **Classifier #2 is never nowcast by L2 in this phase.** A leadership labeling that cannot be
  nowcast is useless in production; the L2 observational leg is the only evidence this phase
  gathers on that question, and it is firewalled from decisions. Whether to build a second
  nowcaster is left to a later phase with this phase's numbers in hand.
- **The four `[ASSUMED]` plausibility bands** (`abs(wealth_delta) < 5`,
  `dd_delta ∈ [−0.5, 0.5]`, `n_transitions > 30` implausible, `pct_disagree < 0.02` suspicious)
  become load-bearing at criterion 7 and must be confirmed or revised before plan 07-10 judges
  against them. D-07's report-only posture is not withdrawn here.
- **Audit item A11 remains open and conscious** — no gate in this project can fail on a
  bad-but-working model, only on a broken one. Unchanged by this ADR.
- **Freezing L2's separate admission path (`_cv_safe_active_features`)** remains raised and not
  pursued, as in ADR-0001. The adopted routing sidesteps it for criterion 7 rather than resolving
  it.
- **Audit item A14 (`canonicalize_states`' sort-key fallback) is CLOSED** by plan 07-05, which
  deleted the fallback and made `sort_column` a raising keyword parameter. ADR-0001 flagged it
  for wave-2 planning; wave 2 closed it. No occurrence of the fallback remains in the tree.

### The eight spec-less probe edges — all resolved, none deferred

ADR-0001 resolved REG-01's three and deferred INV-01's five to wave 2. Wave 2 is where they
close.

| Requirement | Edge | Status | Resolution |
|---|---|---|---|
| REG-01 | adjacency | **resolved** (wave 1); **re-opened and re-resolved** by classifier #2 | Wave 1: driver and reference feature sets compared as ORDERED lists plus a set comparison for the failure message. Classifier #2 re-opens it because it has its own ordered frozen list: `freeze_classifier2_columns` reuses `_reference_label_columns` **unmodified**, so wave 1's ordered-list treatment carries across unchanged, and order is load-bearing because `canonicalize_states` locates `sort_column` by position. Named tests: `test_platform_labeling_classifier2.py::TestFreezeClassifier2Columns::test_returns_every_candidate_in_declaration_order`, `::test_matches_reference_label_columns_called_directly` (the reuse is asserted, not assumed), `::test_late_starting_candidate_excluded_by_name_with_its_first_valid_month` and `::test_candidate_absent_from_the_frame_is_excluded_not_a_keyerror`. The **criterion-5 disjointness assertion** compares ordered lists at the same boundary: `test_platform_features_relative.py::test_disjoint_from_lean_feature_set`, `::test_configured_feature_list_is_disjoint_from_the_lean_set` and `::test_resolved_frozen_list_is_disjoint_from_the_lean_set` — the last of which asserts the *resolved* frozen eight, not merely the declared candidates, is disjoint from classifier #1's lean 13. |
| REG-01 | empty | **resolved** (wave 1); **re-opened and re-resolved** by classifier #2 | Wave 1: an empty or shorter-than-K frozen list raises a named `ValueError` at the `_refit_l1` boundary. Classifier #2 mirrors both messages in shape at its own boundary (`freeze_classifier2_columns` raises on an empty resolved list and on a list shorter than K — K is **5** after the 2026-09-18 RE-PIN, not the 3 this row was drafted against), so the two boundaries read alike and neither can surface as an opaque error from inside the clustering fit. Named tests: `test_platform_labeling_classifier2.py::TestFreezeClassifier2Columns::test_empty_frozen_list_raises_naming_the_count` and `::test_shorter_than_K_raises_naming_the_count_and_K`, plus `::test_all_candidates_nan_after_the_decision_date_raises`. The dependence side of the same edge is pinned by `test_platform_evaluation_dependence.py::test_disjoint_index_gives_nan_statistics_and_warns_without_raising`, `::test_disjoint_index_never_reports_zero` and `::test_disjoint_span_renders_as_a_finding_not_as_a_number` — a zero `n_compared` is reported as a disjoint-span finding with NaN statistics and a WARNING, **never** as agreement or as independence. |
| REG-01 | ordering | **resolved** (wave 1); **re-opened and re-resolved** by classifier #2 — the substantive one | Wave 1: output order is stable and load-bearing, nothing in the call chain re-sorts the frozen list. Classifier #2 genuinely re-opens this: its state numbering cannot use classifier #1's `trailing_return_1m` default (D-10 excludes it), so it passes `sort_column="rs_equities_bonds"` **explicitly**, and a test asserts that a feature frame lacking that column **raises** rather than warning. Plan 07-05 deleted the fallback precisely so this edge cannot be resolved silently. Named tests that re-resolve it: `test_platform_labeling_classifier2.py::TestLabelLeadershipRegimes::test_missing_ordering_column_raises` (the raise, not a warning), `::test_states_are_numbered_by_ascending_sort_column_centroid` (the ordering is the pinned column's, not centroid column 0's), `::test_two_calls_on_the_same_frame_return_identical_states` (idempotence), `::test_returns_frozen_columns_states_and_confidences_aligned`, and `test_platform_labeling.py::test_disjoint_feature_set_end_to_end_explicit_sort_column` (permutation-stability end-to-end for a feature list disjoint from classifier #1's). The frozen list's order is load-bearing because `canonicalize_states` locates `sort_column` **by position** among the centroid columns; `TestFreezeClassifier2Columns::test_returns_every_candidate_in_declaration_order` pins that order. |
| INV-01 | boundary | **resolved** (wave 2, plan 07-07) | **The ratio-admissibility boundary, named first because it is the one D-11's freeze turns on:** a ratio's first admissible month is the **later** of its two sources' first valid months, and must fall at or before the **1972-01-31** first decision date to survive the freeze. Pinned by `tests/unit/test_platform_macro_ingest.py::test_boundary_m2_gdp_first_valid_is_the_later_source_start`. Two further boundaries, both pinned: (`test_stable_at_exact_tolerance_boundary` — a range of exactly `LOADING_STABILITY_TOLERANCE` classifies *stable*, i.e. `<=` not `<`), and the holdout cutoff (`test_holdout_boundary_applied_before_any_ratio_is_computed`, `test_no_result_admissible_month_exceeds_the_holdout_cutoff`). |
| INV-01 | adjacency | **resolved** (wave 2, plan 07-07) | **The denominator-adjacency case, stated first:** `fred_gdp` is quarterly and is forward-filled across the months of a quarter, so **adjacent months of a GDP-denominated ratio share a denominator by construction**. No interpolation was added, and none is implied by the forward fill. Pinned by `tests/unit/test_platform_macro_ingest.py::test_adjacency_denominator_not_interpolated_across_a_quarter`, which recovers the denominator back out of the ratio and asserts it is **constant within a quarter** rather than sloping. Separately, era windows are adjacent expanding slices; `test_era_windows_never_leak_future_values_into_an_early_era` asserts an early era's loading is unchanged by values that only exist in a later era, and `test_era_count_is_a_handful_not_one_per_month` pins the stride so eras are a handful, not one per month. |
| INV-01 | empty | **resolved** (wave 2, plan 07-07) | `test_no_computable_candidate_raises` / `test_no_candidate_computable_raises` — zero computable candidates raises rather than returning an empty survivor list; `test_all_nan_candidate_excluded_with_warning` and `test_missing_column_excluded_with_warning` cover the per-candidate empty case with a named WARNING. |
| INV-01 | ordering | **resolved** (wave 2, plan 07-07) | `test_returns_frame_indexed_by_ordered_candidate_names` and `test_results_are_in_invariant_candidates_declared_order` — results follow `INVARIANT_CANDIDATES`' declared order, and the loadings frame is indexed by candidate NAME, never by component index (`test_rejects_integer_or_component_label_index`, which is also R4's structural guard). |
| INV-01 | precision | **resolved** (wave 2, plan 07-07) | **The named tolerance a stability claim would fail at is `LOADING_STABILITY_TOLERANCE = 0.15`** (`platform/features/invariants.py`), applied as `range <= 0.15` — inclusive at the boundary. Loadings are compared **unrounded**; rounding to five decimals happens only for display in the screening record. The measured per-era loading ranges are 4.4e-16 and 2.2e-16 — machine precision, not drift — and are classified *stable* rather than being mistaken for movement. The exact-boundary comparison test above pins the float comparison as inclusive, and plan 07-07 §4's caveat records that a 1/√2 loading on two standardized candidates is an arithmetic identity rather than evidence, so "stable at machine precision" is not over-read. |

**AMENDMENT 2026-09-21 (plan 07-12), labelled rather than silent.** The eight rows above were
drafted at Proposed. At acceptance every row was re-checked against the tree and **five rows were
extended, none weakened and none re-classified**: REG-01 `adjacency` and `empty` and all four
INV-01 rows now name the specific test that resolves them rather than describing it, REG-01
`ordering` names the five tests that re-resolve it against classifier #2, INV-01 `boundary` and
`adjacency` lead with the two `test_platform_macro_ingest.py` tests the wave-2 plan specified,
and INV-01 `precision` states the numeric tolerance (0.15) instead of calling it small. One
factual correction is carried in the `empty` row: it was drafted against **K = 3** and K is now
**5** after the 2026-09-18 RE-PIN. Every named test was verified to exist in the tree by name on
2026-09-21. No row's status changed; none was deferred at Proposed and none is deferred now.

---

*Amends: `platform_design/platform_design.md` §4.4 (occupancy), §5.4 (honesty metrics), §6.2/§7
(allocation). See also `platform_design/adr/0001-l1-feature-policy.md`, which this ADR extends
rather than supersedes.*

*Phase: 07-regime-representation. See also:
`.planning/phases/07-regime-representation/07-DECISIONS-07-08.md` (the verbatim checkpoint
record with every failure signature), `07-CONTEXT.md` (D-10 through D-17 and the WAVE-2 OPENING
AMENDMENT), `07-INV01-SCREENING.md` (the INV-01 screen), `07-DSR-ESTIMATOR-NOTE.md` (the
estimator reading), `ROADMAP.md` T0.9 and T0.10 (the two standing objections).*

---

## CORRECTION 2026-09-17 — classifier #2 as pinned FAILS design §4.4 criterion 1

This ADR is still **Proposed**, so nothing below is a reversal of an accepted decision.

**Measured on the live fit:** occupancy 15.3736 / 46.1207 / 38.5057 %. Design §4.4 criterion 1
requires every state ≥ ~8% and ≤ ~35%. **States 1 and 2 breach the cap**, state 1 by 11pp. This
is not a marginal overshoot.

Three further facts, all measured rather than inferred:

1. **K = 3 is near-infeasible against the band by arithmetic.** Three states summing to 100%
   under a ≤35% cap must each land in **[30%, 35%]** — effectively forced balance, which §4.3
   set out to replace. The K = 3 choice was reasoned from the three asset sleeves in the
   candidate set and was never checked against §4.4. At K = 5 the band is comfortable.
2. **λ = 32 is badly over-penalized for this feature set.** The fit yields **3 transitions in
   696 months** (1974-01, 1982-12, 1998-09), then state 2 unbroken from 1998-09 to 2020-12 —
   268 months, **45.6% of the decision window, and the most recent 45.6%**, spanning dotcom, the
   GFC and COVID with no change. §4.4 criterion 6 anticipates ~15–30 independent transitions.
3. **Criterion 7 could not mean what it appears to under this fit.** Post-1998 classifier #2
   contributes a constant, so blending it with #1's time-varying tilt measures the blend weight,
   not leadership information.

**The repair is design-sanctioned, not a D-13 breach.** §4.3 states: "λ is the single
interpretable persistence knob. Tune λ (and K) until **acceptance criteria** (§4.4) pass —
occupancy and sojourn targets become the tuning objective, not a distortion of the geometry."
D-13 forbids spending trials to make *lift* look good; §4.4 is a structural gate the design
explicitly says to tune against, and §14 Phase 2's exit is "all six acceptance criteria pass."
Any such tuning must still record how much searching it took, so search-creep stays visible.

**Status of the pinned constants.** (a) the frozen Lean 8 columns, (d) `sort_column`, (e) the
routing and (f) the blend weight are unaffected. **(b) K and (c) λ must be re-pinned** against
§4.4 criteria 1, 2 and 6 before plans 07-09 through 07-12 run. The `strict=True` xfail in
`tests/unit/test_platform_labeling_classifier2.py` fails the suite the moment they are.

---

## RE-PIN 2026-09-18 — K = 5, lambda = 16.0 (2n); §4.4 criterion 1 now passes

Resolves the CORRECTION above. This ADR remains **Proposed**.

### What changed, and what did not

| item | was | now |
|---|---|---|
| (a) frozen columns | Lean 8 | **unchanged** |
| (b) K | 3 | **5** |
| (c) lambda | 32.0 (= 4n) | **16.0 (= 2n)** |
| (d) `sort_column` | `rs_equities_bonds` | **unchanged** |
| (e) routing | L1 decision-bearing / L2 observational | **unchanged** |
| (f) `blend_weight_1` | 0.50 | **unchanged** |

**D-13's rule survives.** lambda is still a *formula* of the frozen feature count, never a free
knob; only the coefficient moved. Design §4.3 licenses precisely this: *"λ is the single
interpretable persistence knob. Tune λ (and K) until **acceptance criteria** (§4.4) pass —
occupancy and sojourn targets become the tuning objective, not a distortion of the geometry."*
§14 Phase 2's exit is "all six acceptance criteria pass." This is an acceptance gate, not a
selection trial for lift, so no registry row was written and D-16's denominator is unchanged.
Classifier #1 is untouched: it keeps its own section and its own 4 × 13 = 52.0.

### Measured results (live, dev-carved span 1963-01-31 → 2020-12-31, 696 months)

| state | occupancy | floor ≥~8% | cap ≤~35% | median sojourn |
|---|---|---|---|---|
| 0 | 16.6667% | PASS | PASS | 58 mo |
| 1 | 22.7011% | PASS | PASS | 25 mo |
| 2 | 22.4138% | PASS | PASS | 29 mo |
| 3 | 23.8506% | PASS | PASS | 166 mo |
| 4 | 14.3678% | PASS | PASS | 100 mo |

Occupancy sum error exactly **0.0**. **§4.4 criterion 1 passes on all five states**; criterion 2
passes (no median sojourn below 3 months).

12 transitions, state sequence `[1, 2, 1, 0, 2, 0, 2, 1, 2, 1, 2, 3, 4]`, transitions at 1970-12,
1971-12, 1974-01, 1982-09, 1983-11, 1984-11, 1987-04, 1989-05, 1994-03, 1995-04, 1998-11,
2012-09. Final run 100 months (14.4% of the span), against the pre-re-pin fit's 268 months
(45.6%).

### Why these two values, by rule rather than by preference

**K = 5.** K = 3 was arithmetically near-infeasible against criterion 1: three states summing to
100% under a ~35% cap must each sit in **[30%, 35%]** — forced balance, the thing §4.3 set out to
replace. The original K = 3 rationale (three asset sleeves in the candidate set) was never scored
against §4.4. K = 5 also matches classifier #1's state-space size, which keeps the ARI / NMI /
Cramér's V dependence statistics (D-15) comparing like with like.

**lambda = 2n.** At 4n = 32 with K = 5 the fit produced five contiguous blocks in strict sequence
(`0→1→2→3→4`) with **no state ever recurring** — a time-segmentation, not a regime model. §4.4
criterion 3 names that failure directly: a state that appears once is an *episode*, not a regime.
The measured acceptance window at K = 5 is **lambda ∈ [8, 24]**: inside it criteria 1 and 2 pass
*and* states recur; at 6 and below criterion 1 breaks (occupancy 3.6% and 36.9%). 2n = 16 sits
mid-window and on a plateau — 12 and 16 produce identical labelings — so it is robust rather than
a cliff-edge pick.

### Search extent (D-17 — recorded so search-creep stays visible)

Eleven fits total, every one an acceptance-gate evaluation against §4.4, **zero** performance or
lift evaluations and **zero** registry rows:

- 1 at K=3, λ=32 — confirming the oil data fix did not move occupancy (it did not: 15.37 /
  46.12 / 38.51%, identical to the pre-fix fit, so this re-pin rests on its own merits);
- 1 at K=5, λ=32;
- 8 sweeping λ ∈ {32, 24, 16, 12, 8, 6, 4, 2} at K=5;
- 1 production re-fit at the pinned K=5, λ=16.

### Two limitations, stated rather than left to be found

1. **States 3 and 4 each occur exactly once**, as the final two blocks (1998-11 → 2012-09 and
   2012-09 → 2020-12). States 0, 1 and 2 recur properly (2, 4 and 5 occurrences). So the
   episode-vs-regime concern is **reduced, not eliminated** — it now applies to the last 22 years
   rather than to the whole span. Whether that is a genuine secular shift or an artifact of K is
   open; it is not resolved here.
2. **§4.4 criterion 3 (subsample stability under Hungarian matching) has not been run** for either
   classifier. Criteria 1 and 2 pass and are tested; criterion 4 has per-state profiles; criterion
   5 (decision-relevance) is plan 07-11's; criterion 6 constrains complexity. This ADR therefore
   claims criteria 1 and 2, not "all six".

`tests/unit/test_platform_labeling_classifier2.py::TestClassifier2LiveOccupancyAgainstDesign44`
now asserts criterion 1 against the live fit as a plain passing test; before the re-pin it carried
a `strict=True` xfail recording the breach.

---

## ACCEPTANCE 2026-09-21 — Consequences, as measured

This section is appended at acceptance. Nothing above it is revised by it; where a figure above
was superseded by a re-pin or a re-measurement, that is said here rather than by editing the
earlier text. **Every number below carries its window in the same table cell or sentence** — the
binding condition wave 1's UAT attached to criterion 3, extended to criteria 5, 6 and 7.

**D-06 governs: measurement is the gate, not the sign.** That is not a licence to present an
unfavourable number favourably. Where a result came out badly it is stated in the units it is
actually in, first, with no clause after it doing repair work.

### Criterion 5 — occupancy and disjointness: **MET**

Classifier #2 as re-pinned (K = 5, λ = 16.0; see § RE-PIN 2026-09-18), fit on the dev-carved
span:

| state | occupancy (696 months, 1963-01-31 → 2020-12-31) | §4.4 floor ≥~8% | §4.4 cap ≤~35% |
|---|---|---|---|
| 0 | **16.6667%** | PASS | PASS |
| 1 | **22.7011%** | PASS | PASS |
| 2 | **22.4138%** | PASS | PASS |
| 3 | **23.8506%** | PASS | PASS |
| 4 | **14.3678%** | PASS | PASS |

Occupancy **sum error exactly 0.0** over those 696 months. All five states sit inside §4.4
criterion 1's **~8%–35%** band; **classifier #2 has no sub-floor state and does not invoke the
recurrence exemption** design §4.4 gained on 2026-09-18 (classifier #1 does — see § Requirement
coverage). Disjointness from classifier #1's lean 13 is asserted on the **resolved** frozen eight,
not merely on the declared candidate list, by
`tests/unit/test_platform_features_relative.py::test_resolved_frozen_list_is_disjoint_from_the_lean_set`.

**Correction carried, not buried: there is no "§4.4 five-percent floor."** ROADMAP criterion 5 as
originally worded cited one. §4.4 criterion 1 reads "every state ≥ ~8% and ≤ ~35% of months" — a
**two-sided** band. The 5% figure was a project-wide misquote corrected on 2026-09-17
(ADR-0001 § AMENDMENT 2026-09-17); the table above is scored against the real, two-sided
criterion, which is the stricter reading in both directions.

### Criterion 6 — dependence: **UNRESOLVED**. Neither met nor failed.

Re-measured after **both** classifiers were re-pinned, so the statistics recorded at Proposed and
in `07-DEPENDENCE.md` §3 are **void** and are superseded here.

| statistic | value | measured over |
|---|---|---|
| Adjusted Rand | **0.354841** | `n_compared` = **695** months, **1963-02-28 → 2020-12-31**, K₁ = 6 vs K₂ = 5 |
| Normalized mutual information | **0.464088** | `n_compared` = **695** months, **1963-02-28 → 2020-12-31**, K₁ = 6 vs K₂ = 5 |
| Cramér's V | **0.589748** | `n_compared` = **695** months, **1963-02-28 → 2020-12-31**, K₁ = 6 vs K₂ = 5 |

Read against the **block-permutation control** (2000 resamples, seed 20260918; block counts used:
classifier #1 **26**, classifier #2 **13**), which holds each labeling's own sojourn structure and
occupancy fixed and destroys only the alignment between them:

| statistic | observed | null p50 | p95 | p99 | observed percentile |
|---|---|---|---|---|---|
| Adjusted Rand | 0.354841 | 0.2164 | 0.3332 | 0.4086 | 96.75 |
| **NMI** (the pre-registered read) | **0.464088** | 0.3406 | **0.449202** | **0.501195** | **96.60** |
| Cramér's V | 0.589748 | 0.5214 | 0.6178 | 0.6584 | 87.70 |

**Verdict, by the rule committed at `298b1bc` before the control code existed:**
p95 0.449202 < observed NMI 0.464088 ≤ p99 0.501195 → **INCONCLUSIVE**.

**What that means, said plainly and without repair.** The association is elevated above what
temporal blockiness alone explains, but not by enough to call. **This measurement cannot decide
whether classifier #2 added an axis, and nothing else in this ADR decides it either.** Criterion
6 is therefore **open**, and that is why REG-01 is claimed PARTIALLY below rather than in full.

**No tie-break was run and none may be.** The pre-registration forbids a fourth statistic, a
different null and a re-run at another seed; all three would be the garden of forking paths it
exists to close. **Criterion 7 is not a tie-break on criterion 6** — it is a separate,
independently pre-planned measurement asking a different question, and it is not offered here as
evidence about dependence.

One property of the null is recorded because it bears on how the number should be read, and it
cuts in classifier #2's favour rather than against it: `_shuffle_blocks` merges same-state blocks
that land adjacent, so each resample is slightly blockier than its input, shifting the null
**up**. The bias is one-directional and conservative — it can push a verdict toward "failed to add
an axis" but can never manufacture "added an axis." An inconclusive result under such a null is
mildly favourable to classifier #2. **It is not a resolution of the criterion and is not offered
as one.**

### Criterion 7 — joint allocation lift: **MET as a measurement. The wealth sign is negative.**

Routing `L1_ONLY_LAST_FILTERED_STATE`, decision-bearing, pinned at decision (e) before any run.
Both legs from one harness differing only in `blend_weight_1`; `blend_weight_1 = 1.0` **is** the
#1-alone leg.

| | classifier #1 alone | joint (#1 × #2) |
|---|---|---|
| **Window** | **588 steps, 1972-01-31 → 2020-12-31** | **588 steps, 1972-01-31 → 2020-12-31** |
| Degraded steps | **0 of 588** | **0 of 588** |
| Terminal log wealth (588 steps, 1972-01-31 → 2020-12-31) | **4.718723** nats | **4.595285** nats |
| Max drawdown (same 588 steps, same window) | **−28.1442%**, 47 months underwater | **−25.7358%**, 40 months underwater |
| Annualized Sharpe (588 monthly obs, same window) | **0.917073** | **0.914903** |
| Registry tag | `07-11-c1-alone-L1only` | `07-11-joint-c1xc2-L1only` |

| Axis | Value | Window (inline) | Governing (universal) band | Inside? |
|---|---|---|---|---|
| `wealth_delta` | **−0.123438 nats** | **588 steps, 1972-01-31 → 2020-12-31** | `abs(x) < 15` | yes |
| `dd_delta` | **+0.024084** (+2.41 pp; the joint leg's drawdown is *shallower*) | **588 steps, 1972-01-31 → 2020-12-31** | `x ∈ [−1, 1]` (**revised**) | yes |

**The wealth result, in the units it is actually in.** `wealth_delta = −0.123438` nats means the
joint leg ended at e^(−0.123438) ≈ **0.8839×** the #1-alone leg's terminal wealth over those 588
months — an **11.61% shortfall in terminal wealth**. **Blending classifier #2 in at the pinned
weight made the book poorer over the measured window.** Costs are not the explanation: the joint
leg paid **0.025557 less** in total transaction cost over the same 588 steps (0.070435 vs
0.095992) and still ended 0.123438 nats lower. The shortfall is in the gross allocation the blend
produced.

**The drawdown result is a separate number and does not offset the wealth result.** `dd_delta =
+0.024084` is a difference of two drawdown fractions: the joint leg's worst peak-to-trough loss
was **2.41 percentage points shallower** (−25.74% vs −28.14%) and its longest underwater stretch
**7 months shorter** (40 vs 47), over the same 588 steps. That is a real, measured improvement on
one axis. **It is not a reason to describe the phase's headline lift as favourable**, and it is
additionally qualified by the § Named limitation below: this labeling identifies crises *ex post*,
so a drawdown avoided in the labeling is not a drawdown avoided in real time.

**Criterion 7 is satisfied because the lift was measured honestly and with its window, not
because it came out well.** D-06 makes measurement the gate. The sign is stated, not softened.

**The ADR's own named failure signature for this routing — "the two legs disagree in sign on
joint lift" — half fired, and is recorded rather than explained.** On `wealth_delta` the two
routings agree (−0.123438 L1-only, −0.134505 L2, both over 588 steps 1972-01-31 → 2020-12-31). On
`dd_delta` they disagree in sign: **+0.024084** (L1-only) versus **−0.001137** (L2, over the same
588 steps). The L2 magnitude is about one twentieth of the L1-only figure, so "the L2 path shows
no drawdown effect" describes it better than "contradiction" — but it **is** a sign disagreement
on one of two axes and it is **not explained here**. Per the firewall clause it is reported and
not acted on.

### The deflated Sharpe — **neither leg clears the multiple-testing hurdle**

| Leg | Observed Sharpe (588 obs, 1972-01-31 → 2020-12-31) | `n_trials` | Trial count read at (UTC) | `sharpe_variance` | DSR |
|---|---|---|---|---|---|
| classifier #1 alone | 0.917073 | **42** | 2026-09-21T14:24:23.984933Z | 1.0 (placeholder) | **2.28151 × 10⁻¹²** |
| joint (#1 × #2) | 0.914903 | **42** | 2026-09-21T14:24:23.984933Z | 1.0 (placeholder) | **1.46904 × 10⁻¹¹** |

`format_dsr_verdict`'s own output, quoted verbatim for **both** legs:

> "Deflated Sharpe ratio 0.0000 does not clear the multiple-testing hurdle — statistically
> indistinguishable from a skill-less discovery given the number of trials searched."

**The result does not clear the multiple-testing hurdle.** Both DSRs are far below 0.5. There is
no near-miss to report and no target to have fallen short of: `07-VALIDATION.md` sets none for
this number, precisely so there is nothing to miss narrowly. The joint leg's raw Sharpe
(0.914903) is also *lower* than the baseline's (0.917073) over the same 588 steps, so the blend
did not help on this axis before deflation either.

`expected_max_sharpe(42, 1.0) = 2.208694`. **`sharpe_variance` is the declared `1.0` placeholder,
not a measured quantity** — `registry_sharpe_variance()` logs that it found too few usable Sharpe
observations on every call. That is a property of the estimator declared in
`07-DSR-ESTIMATOR-NOTE.md` long before these runs, not a caveat invented for this result, and it
is stated here **after** the verdict because it qualifies nothing above.

### Estimator repair, 2026-09-21 — recorded because it changed the rules, not the verdicts

`07-DSR-ESTIMATOR-NOTE.md` § AMENDMENT 2026-09-21 (commit `230c91c`) repaired a live trap in the
estimator, decided by Glenn after plan 07-11 declined to write a `sharpe` key and flagged why.
`_MIN_USABLE_SHARPE_OBSERVATIONS` was **2**: any two Sharpe-bearing rows switched
`registry_sharpe_variance` off the conservative placeholder onto a computed value. Measured on
plan 07-11's own two rows (Sharpe 0.917073 and 0.914903), sample variance 2.354450e-06 collapses
`expected_max_sharpe(42, ·)` from **2.208694** to **0.003389** — a **99.85% collapse** of the
hurdle, after which essentially any strategy clears DSR. The deeper error was a category one:
those two rows are two arms of **one** ablation, not independently tried configurations.

Two independent defences were implemented: `_MIN_USABLE_SHARPE_OBSERVATIONS` **2 → 20**, and
`config["independent_trial"] is False` **excludes a row entirely** from the across-trials
variance. Trial rows now do carry `metrics["sharpe"]`; plan 07-11's two rows were backfilled with
`independent_trial: False`. **Today's verdicts are unchanged** — `sharpe_variance` is still 1.0,
`expected_max_sharpe(42, 1.0)` is still 2.208694, and both DSRs above still do not clear.
`DEGENERATE_SHARPE_VARIANCE = 1.0` **remains a declared assumption** governing every DSR this
project reports, until 20 independent Sharpe-bearing trials exist — a longer road than before the
repair, deliberately, because the previous road was short because it was wrong.

---

## ⚠ Named limitation, foregrounded here (not a footnote): what criterion 7's window and routing make this number, and what they make it not

This is a section, in ADR-0001's own style, because a limitation that decides how a headline
number may be cited does not belong in a parenthetical.

**Criterion 7's lift was measured on 588 steps, 1972-01-31 → 2020-12-31, through the L1-only
routing.** That routing runs **no L2 refit**, which is why both legs ran 588 of 588 steps with
**0 degraded steps** — wave 1's dominant degradation mechanism (an early small post-embargo
window starving a K-fold) is structurally absent from this path. The clean window is a
*consequence of the routing*, not evidence that the production stack behaves this way.

**What this number is therefore not comparable to.** Wave 1's `wealth_delta` **+0.377847** and
`dd_delta` **−0.066124** were measured **through L2, on 356 steps ending 2017-05**, against a
470-step ablation ending 2020-12, and they are a **different subtraction**: *strategy minus
no-regime ablation*, where criterion 7's is *joint (#1 × #2) minus classifier-#1-alone*.
**Any table placing +0.377847 beside −0.123438 without stating both differences is
misreporting.** Criterion 7's #1-alone baseline was re-run through this harness and was never
reused from wave 1.

**What the routing costs in meaning.** Each classifier contributes a **degenerate one-hot on its
own last filtered state**, so criterion 7 measures the L1→L4 path, not the production L1→L2→L4
path. The L2 observational leg exists precisely because that gap is real; it is firewalled, and
per the firewall clause **nothing downstream in this phase changed on the basis of it**.

**What feeds that path, stated because it is uncomfortable rather than despite it.** Classifier
#1's per-step **filtered** labeling changes state in **246 of 588 decision months (41.84%)**,
against a full-sample label-transition rate of **3.60% (25 of 695 months, 1963-02-28 →
2020-12-31)**. Classifier #2's filtered labeling changes in **24 of 588 (4.08%)** against a
full-sample **1.72% (12 of 696 months, 1963-01-31 → 2020-12-31)**. **No plausibility band governs
the filtered rate** — band 3b governs the full-sample labeling and says nothing about it — so
nothing fired, and this is recorded as a finding rather than laundered through a band that was
never about it. A labeler that re-labels its own most recent month in two months out of five is
churning, that churn feeds the tilt directly under this routing, and it is the most likely
mechanical source of the #1-alone leg's 0.163251 mean monthly turnover.

**And the drawdown improvement must not be read as crisis-timing skill.** Classifier #1's crisis
state has a **median sojourn of 3.0 months — exactly on design §4.4 criterion 2's boundary** —
against §5.4's typical 1–3 month detection lag. This labeling identifies crises *ex post*.
Whether L2 can nowcast them early enough to trade is **L2's question and it is unanswered**.
`dd_delta = +0.024084 over 588 steps, 1972-01-31 → 2020-12-31` is **not** evidence that crises are
nowcastable in time to act.

---

## Requirement coverage

### REG-01 — claimed **PARTIALLY**. Criterion 6 is the named open item.

Phase-wave 1 (plans 07-01…07-04, ADR-0001) satisfied the **feature-policy clauses**: one
documented policy shared by the walk-forward driver and the evaluation reference, the §5.4
sojourn/lag ratio made interpretable with its resolved-transition count, and classifier #1's
ablation delta re-measured on both axes.

Phase-wave 2 (plans 07-05…07-12, this ADR) satisfies **three** of REG-01's four remaining
clauses and leaves **one open**:

| REG-01 clause | Status | Evidence |
|---|---|---|
| A second, independent classifier fit **unsupervised** on features **disjoint** from classifier #1's | **satisfied** | Classifier #2, K = 5, λ = 16.0, frozen Lean 8; occupancy table above (696 months, 1963-01-31 → 2020-12-31) |
| The **disjointness assertion** (a test, not a claim) | **satisfied** | `test_platform_features_relative.py::test_resolved_frozen_list_is_disjoint_from_the_lean_set` and two sibling tests |
| Orthogonality **measured, not assumed** | **measured; the verdict is UNRESOLVED** | `07-DEPENDENCE.md` — ARI 0.354841 / NMI 0.464088 / Cramér's V 0.589748 over `n_compared` = 695 months, 1963-02-28 → 2020-12-31; block-permutation control returns **INCONCLUSIVE** at the 96.60th percentile against p95 0.449202 / p99 0.501195 |
| Joint allocation lift **assessed walk-forward** against classifier #1 alone, every configuration registry-logged | **satisfied** | `07-JOINT-LIFT.md` — `wealth_delta` **−0.123438**, `dd_delta` **+0.024084**, both over 588 steps 1972-01-31 → 2020-12-31; tags `07-11-c1-alone-L1only` and `07-11-joint-c1xc2-L1only` |

**The measurement clause is discharged; the question it was asked to answer is not.** REG-01 is
therefore **PARTIAL, not complete**, and the open item is named rather than folded in:
**criterion 6's dependence verdict is unresolved, no tie-break is permitted, and no independent
second axis is established by this phase.** Closing it requires either more data or a different
instrument, in a phase that pre-registers its rule the way this one did.

### INV-01 — claimed **IN FULL**. All four clauses, each with its artifact.

ADR-0001 recorded INV-01 as "entirely deferred to wave 2" under D-09. This closes that
enumeration, clause by clause:

| # | INV-01 clause | Artifact that satisfies it |
|---|---|---|
| 1 | Named invariant candidates **constructed and screened**, with dimensional reduction used as a **discovery tool** only | `07-INV01-SCREENING.md` §1–§2; `platform/features/invariants.py::screen_invariant_candidates`. `compute_candidate_loadings()` reads only `pca.components_`; **`pca.transform()` is never called anywhere in the module**, so no component *score* was ever computed |
| 2 | **Loading stability tested across eras**, with a named tolerance | `loading_stability_across_eras()` over **10 eras** ending **1972-01-31 → 2017-01-31** (expanding windows, `min_train=120`, `step=60`), tolerance **`LOADING_STABILITY_TOLERANCE = 0.15`**, inclusive at the boundary; both candidates classified **stable**, per-era PC1 loading 0.70711 with ranges 4.4e-16 and 2.2e-16 |
| 3 | **Every candidate logged to the trial registry** and assessed walk-forward | 2 rows tagged `07-07-inv01-screen`, written to the real `registry/trials.jsonl` (not `NO_REGISTRY`); `total_trial_count()` read **38** at 2026-09-17T15:09:17Z and **40** at 2026-09-17T15:10:37Z |
| 4 | Survivors admitted as **named** features, never anonymous principal components (design decision R4) | Named survivors **`m2_gdp`** (`fred_m2sl / fred_gdp`) and **`credit_gdp`** (`fred_totalsl / fred_gdp`). Only **`m2_gdp`** enters classifier #2's frozen eight; `credit_gdp` is dropped for near-collinearity (0.957–0.969 with `m2_gdp` across eras) — a rejection recorded, not silent. R4 is held **structurally**, pinned by `test_platform_features_invariants.py::test_rejects_integer_or_component_label_index` |

**⚠ Read clause 2's result at plan 07-07 §4's own strength, which this ADR does not upgrade.** A
PC1 loading of exactly 1/√2 on **two** standardized candidates is an **arithmetic identity** —
the first component of exactly two positively-correlated standardized series must split its
weight evenly — not, by itself, evidence of genuine five-decade economic stability. The
screening record says so; era screening was **survived**, and that is all that is claimed.

---

## Rejections, restated rather than quietly dropped

Five candidate directions were rejected during this phase. Each is restated here with its reason
so the record of what was *not* built is as durable as the record of what was:

1. **`BCNSDODNS`** — quarterly-native, 305 observations from 1945-10. Using it would require
   exactly the forward-fill treatment the chosen series avoid. Rejected **pre-ingestion**.
2. **`TOTBKCR`** — starts **1973**, after the **1962** spine start, so it cannot serve a
   labeling frozen at the 1972-01 first decision date. Rejected **pre-ingestion**.
3. **Market-cap/GDP (the "Buffett indicator")** — **stays blocked** (D-12): no free source back
   to 1962; FRED's Wilshire series starts around 1970. Restated, not worked around.
4. **Gold/equity relative strength** — **admissible in principle** under D-10's ratio rule,
   **excluded in fact** by D-11's freeze: `gold` starts **1985-02** while `oil` runs from
   **1962-01**, so the ratio cannot clear the 1972-01 first decision date.
5. **A product `(state_1, state_2)` state space** — rejected per **D-14**: K₁ × K₂ cells over
   roughly 590 decision months thins rare cells far below §4.4's **~8%** floor. The two
   probability vectors are consumed independently and only the resulting weight Series are
   combined; no product index is ever formed.

---

## The four plausibility bands, as confirmed or revised

Decided by Glenn on 2026-09-18 at plan 07-10's human-verify gate, **before any joint-lift number
existed** (`07-BANDS.md` §8). Governing values:

| Quantity | Governing (universal) value | Advisory (domain) trigger | Disposition |
|---|---|---|---|
| `wealth_delta` | `abs(x) < 15` | `abs(x) ≥ 5` → recorded note | Domain band `abs(x) < 5` **CONFIRMED** as the domain tier; deliberately **not** tightened toward the observed value |
| `dd_delta` | **`x ∈ [−1, 1]`** | `abs(x) ≥ 0.5` → recorded note | **REVISED** from `[−2, 2]`. The old bound was wider than the quantity's own arithmetic range (`max_drawdown ∈ [−1, 0]` per leg ⇒ difference ∈ `[−1, 1]`) and **could therefore only confirm** — this project's signature defect shape. The revised bound is definitional: a breach proves a leg's `max_drawdown` is not a fraction in `[−1, 0]`, i.e. the KPI is broken |
| `n_transitions` | **REVISED and SPLIT** | — | (a) `sojourn_lag` within-window count → definitional `n_resolved ≤ n_transitions ≤ n_label_transitions`; a breach is a counting bug, never a strategy outcome. (b) full-sample labeling → a **rate** band, implausible above `0.10 × n_months`. A fixed count is length-dependent; a rate is not |
| `pct_disagree` | `< 0.02` **OR** `n_compared == 0` **OR** coverage materially below expectation without a *recorded* reason | — | Threshold **CONFIRMED unrevised at 0.02**; the **statement** revised to carry its denominator. Live in `platform/evaluation/disagreement.py`, not left as prose — the defect it fixes was real: `n_compared == 0` previously logged a warning but set `suspicious = False` |

**There is no "five-percent floor" anywhere in this project's design.** §4.4 criterion 1 is
`≥ ~8%` and `≤ ~35%`; the 5% figure was a misquote corrected 2026-09-17 and is not a band.

**Per D-07 these remain plausibility bands, not quality gates.** The governing tier can only fail
on a **broken measurement**, never on a bad-but-working model. **Audit item A11 therefore remains
open and conscious** — Glenn declined to promote the domain tier to a gate, which would have
closed A11 by gating on `[ASSUMED]` numbers. A11 is open by deliberate choice, and this ADR does
not close it.

At criterion 7 both governing bands held (`abs(−0.123438) < 15`; `+0.024084 ∈ [−1, 1]`), so the
measurement is not broken and criterion 7 reports. **Neither advisory trigger fired**, so no
domain note was recorded. Band 3's clauses hold for both labelings, and band 4's denominator
clauses were **exercised live** — `expected_n_compared = 588` was passed and coverage came back
at exactly 588, so the third clause **could have fired and did not**.

---

## Trial arithmetic at acceptance

| Item | Value | Source |
|---|---|---|
| Component 1 — INV-01 screen rows (spent) | **2** (38 → 40) | live reads 2026-09-17T15:09:17Z and 15:10:37Z |
| Component 2 — criterion 7 evaluation rows (ceiling **4**, actual **2**) | **2** (40 → 42) | live reads 2026-09-21T14:18:34.285851Z and 14:24:23.984933Z |
| Component 3 — the L2 observational leg | **0** rows | `NO_REGISTRY` sentinel, per decision (e) |
| **Live `total_trial_count()` at acceptance** | **42**, read **2026-09-21T15:37:29.833014Z** | live read performed for this section, not copied from a planning document |
| Ceiling for the remainder of this phase | **44** (= 40 + 4) | § Trial ceiling above |
| Ceiling respected? | **yes — 42 ≤ 44**, with **2** rows of the budget unspent | arithmetic on the reading above |

The component-2 harness appends **exactly one** row per call, not two —
`run_full_backtest_evaluation`'s factor of two comes from its own two `append_trial` sites and
does not apply to `run_joint_backtest`. So component 2 spent 2 of its budgeted 4. **Spending
fewer rows than a ceiling is the only direction a ceiling permits.**

**Standing warning, restated from ADR-0001 and extended: any literal count is stale the moment it
is written.** `30`, `34`, `35`, `38`, `40` and `42` are all figures of a particular moment. `38`
is the provenance header's `prior_genuine_trials`; **`42` is a live reading at
2026-09-21T15:37:29.833014Z and nothing more.** Exceeding **44** requires an explicit amendment to
this ADR; the formula is not a cap the code enforces, it is the arithmetic a reader uses to check
whether search-creep occurred silently between two readings.

---

## Deferrals and open items at acceptance

Recorded as decisions and carried forward, not closed by acceptance:

1. **Criterion 6 is UNRESOLVED.** No tie-break was run and none may be — the pre-registration at
   `298b1bc` forbids it. **No further dependence statistic may be computed on these labelings.**
2. **§4.4 criterion 3 (subsample stability under Hungarian matching) has never been run** for
   **either** classifier. Nothing in this ADR rests on it having passed and nothing here implies
   it has. Both ADR-0001 and ADR-0002 claim criteria 1 and 2, never "all six."
3. **ADR-0001 condition (iv) is implemented for per-regime Sharpe only.** `allocation/joint_tilt.py`
   flags every sub-floor regime at WARNING and partially pools its per-regime Sharpe toward the
   all-history estimate with credibility `min(1, occupancy / floor)`. **Its covariance clause is
   NOT implemented** — no per-regime covariance exists at L4-01 to shrink. That clause falls to
   **L3** (design §6.2) and **condition (iv) must not be described as complete.**
4. **`vol_targeted_tilt` and `driver.py:497` remain condition-(iv)-non-compliant** for consumers
   other than the joint harness: they apply no pooling. Only the `blend_regime_tilts` path
   applies it today.
5. **Classifier #1's filtered labeling changes state in 246 of 588 decision months (41.84%)**
   against a **3.60%** full-sample rate. **No band governs the filtered rate.** That churn feeds
   the tilt directly under the chosen routing.
6. **Classifier #2's §5.4 ratio is 1.074** — median sojourn **29.0 months**, median detection lag
   **27.0 months**, **only 5 of 12 transitions resolved**, over 588 steps 1972-01-31 → 2020-12-31.
   The lag very nearly consumes the sojourn. **This phase gathered no evidence that classifier
   #2's leadership structure is detectable in real time early enough to allocate on.**
7. **The crisis state's median sojourn is 3.0 months, exactly on criterion 2's boundary**, against
   a 1–3 month detection lag. Criterion 7's `dd_delta` is **not** evidence crises are nowcastable
   in time to act.
8. **`DEGENERATE_SHARPE_VARIANCE = 1.0` remains a declared assumption** governing every DSR this
   project reports, until **20** independent Sharpe-bearing trials exist.
9. **Whether the registry should carry Sharpe metrics at all** is resolved for the recording
   convention (it does, with `independent_trial: False` on non-independent arms) but the
   estimator still runs on the placeholder. Left open.
10. **Audit item A11 remains open and conscious** — no gate in this project can fail on a
    bad-but-working model, only on a broken one. **Open by Glenn's deliberate choice**, not by
    oversight.
11. **The L2 CV-robustness question was routed around, not resolved.** The L1-only routing
    sidesteps `_cv_safe_active_features` for criterion 7 rather than freezing L2's separate
    admission path. It **remains open with its own future ADR**, exactly as ADR-0001 left it.
12. **Classifier #2 is never nowcast by L2 in this phase.** The L2 observational leg is the only
    evidence gathered on that question and it is firewalled from decisions.
13. **The `add-alongside` decision stands, and the measured results confirm it was the right
    call.** § Consequences recorded three conditions that would together force a promote: low
    measured dependence, positive joint lift on **both** axes inside the confirmed bands, and a
    DSR clearing 0.5. **None of the three landed** — dependence is unresolved, `wealth_delta` is
    negative, and neither DSR clears. A promote-or-product design is therefore **recorded as a
    named follow-up and not acted on inside this phase**, since D-13 forbids the blend-weight
    search a promote would require. `blend_weight_1` was pinned at **0.50** and **never swept**;
    this ADR says nothing about what any other weight would produce.

### The three caveats wave 2 inherited from `07-UAT.md`

| Wave-1 caveat | Status at acceptance |
|---|---|
| 1 — criterion 3's comparison window (356 steps ending 2017-05 vs a 470-step baseline ending 2020-12), cause understood as L2's CV degradation; resolution needs an L2 CV design decision | **STILL STANDING.** Routed around, not resolved — see open item 11. The L1-only routing produced a clean 588-of-588 window *because it runs no L2 refit*, which does not answer wave 1's question |
| 2 — drawdown got worse under the frozen policy (`dd_delta` −0.014364 → −0.066124), and the strategy remains last of five legs | **STILL STANDING**, and not discharged by criterion 7's `+0.024084`: that is a **different subtraction** over a **different window and routing** (see § Named limitation). Nothing measured in wave 2 re-rates the wave-1 comparison |
| 3 — four `[ASSUMED]` bands unconfirmed; confirm or revise before wave 2 leans on them | **DISCHARGED.** All four were confirmed or revised at plan 07-10's gate on 2026-09-18, **before any joint-lift number existed** — `07-BANDS.md` §8, restated above. Band 2 was revised because it could only confirm; band 4's clauses were implemented in code and exercised live |

---

## A11 RULING 2026-09-21 — open item 10 is CLOSED, and closed by a reversal

Open item 10 above ("Audit item A11 remains open and conscious … Open by Glenn's deliberate choice") was **reversed and CLOSED** on 2026-09-21: A11 is ANSWERED YES, promoting the deflated-Sharpe hurdle `expected_max_sharpe(total_trial_count(), sharpe_variance)` to a governing quality tier — see [`0003-quality-gate-tier.md`](0003-quality-gate-tier.md) and `.planning/phases/08-regime-persistence-stability/08-A11.md`, which together record the consequence that criterion 7's `✅ MET` above is **FAILED** on both legs, and note that open item 8's placeholder `sharpe_variance` now governs a gate rather than a report.
