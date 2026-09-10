# Phase 7: Regime Representation - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-09-10
**Phase:** 7-Regime Representation
**Areas discussed:** Feature policy + selection criterion; Gate semantics on a negative result; Classifier #2's feature set + INV-01; Joint labeling, orthogonality rule, trial budget

---

## Feature policy + selection criterion

### Q1 — Which direction reconciles the expanding driver and the frozen reference?

| Option | Description | Selected |
|--------|-------------|----------|
| Freeze the driver | Driver stops expanding; both paths fit on one constant space. The only direction preserving §5.4's smoothed-vs-filtered meaning. Cost: late starters dropped or imputed. | ✓ |
| Expand the reference to match | Keep the expanding driver; refit the reference per decision date. Cost: the reference becomes a second walk-forward, so §5.4 measures something else. | |
| Evaluate both, pre-declared criterion | Run both, ADR records the loser. Roughly doubles walk-forward runs and registry rows. | |

**User's choice:** Freeze the driver
**Notes:** The two directions are not symmetric — this was the key structural point. Expanding the reference would destroy the very quantity §5.4 exists to measure.

### Q2 — What does the frozen set contain?

| Option | Description | Selected |
|--------|-------------|----------|
| The 9 common-support features | Adopt what the reference already computes. Zero new math; the equivalence test becomes trivial. | |
| All 13 with explicit imputation | Keeps VIX/gold/oil/10y2y by backfilling. Invents pre-1990 VIX levels. | |
| 9 now, imputation as a logged trial | Ship the 9-feature policy; ALSO run the 13+imputation variant once as a registered trial so the ADR's rejection is evidence-backed. | ✓ |

**User's choice:** 9 now, imputation as a logged trial
**Notes:** Verified empirically before asking — exactly 9 of 13 span the decision range from 1972-01; the 4 excluded are `curve_10y2y` (1976-06), `gold` (1985-02), `oil` (1985-02), `fred_vix` (1990-01). Also separated two figures the proposal conflates: 9 is common support across the decision range; 4 is what the driver's first window held at ≥120 months.

### Q3 — What rule picks and validates the policy?

| Option | Description | Selected |
|--------|-------------|----------|
| Interpretability only, pre-declared | Chosen on a structural criterion declared before any run; all downstream numbers reported, never used to choose. | ✓ |
| Structural + stability tiebreak | Break ties on label stability. Adds a selection dimension, so adds trials to the DSR denominator. | |
| Structural + ablation delta veto | Reject a policy whose `dd_delta` is worse than −0.014364. Honestly named: fitting the feature policy to the drawdown metric. | |

**User's choice:** Interpretability only, pre-declared
**Notes:** The third option was listed explicitly as what most teams actually do, and explicitly advised against.

### Q4 — Does wave-2 disjointness measure against the 9 in use or the 13 named?

| Option | Description | Selected |
|--------|-------------|----------|
| Against all 13 | Roadmap criterion 5 already says this. Keeps released level/stress features out of the leadership classifier. | ✓ |
| Against the 9 in use | Literal disjointness; would let #2 use VIX/gold/oil/10y2y. Makes high dependence far more likely. | |
| Against 13, with derived ratios allowed | Raw columns off-limits, ratios admissible. | |

**User's choice:** Against all 13
**Notes:** Raised because freezing #1 at 9 mechanically *releases* 4 features. Later refined in the classifier-#2 area, where ratios of #1's columns were admitted (see that area's Q1) — so the settled position is raw-column disjointness against all 13, with derived ratios allowed.

### Q5 — How are the superseded Phase 5/6 numbers handled?

| Option | Description | Selected |
|--------|-------------|----------|
| Side-by-side pre/post table | Every affected number under both policies, pre-fix column labelled as baseline. | ✓ |
| Supersede and archive | New numbers become the record; old ones archived. Loses the at-a-glance comparison. | |
| Pre/post table + explicit y_true warning | The table plus a stated caveat that Brier/confusion moved because labels changed. | |

**User's choice:** Side-by-side pre/post table
**Notes:** The `y_true` mechanism (`report.py:617` reindexes the reference labeling onto decision dates) is recorded in CONTEXT.md as a mechanical fact the planner must carry into the table, since it explains why Brier moves.

---

## Gate semantics on a negative result

### Q1 — If `dd_delta` is still negative after the fix, what does the gate do?

| Option | Description | Selected |
|--------|-------------|----------|
| Measurement is the gate | Wave 2 proceeds regardless of sign, provided the number is honestly measured and recorded. | ✓ |
| Negative dd_delta halts wave 2 | Stop and fix #1 first — but the explicit non-goals forbid the obvious remedies. | |
| Negative dd_delta redirects wave 2 | Same code, different declared hypothesis, recorded before the run. | |

**User's choice:** Measurement is the gate
**Notes:** Rationale carried into CONTEXT.md: #2 is a different axis on disjoint features, so its value does not depend on #1 winning; and "halt and fix #1" has no sanctioned move inside this phase.

### Q2 — What band makes each re-measured number pass or fail?

| Option | Description | Selected |
|--------|-------------|----------|
| Plausibility bands only | Assert physically possible, not good. Holds the Phase 5 D-01 / Phase 6 D-16 posture. | ✓ |
| Plausibility + quality band on dd_delta | Would be the project's first criterion that fails on a bad-but-working model (audit A11). | |
| Plausibility + regression bands | Assert the new numbers have not moved absurdly far from baseline. | |

**User's choice:** Plausibility bands only
**Notes:** A11 was surfaced explicitly as a real change in posture and consciously declined for now — recorded as a deferred idea, to be re-examined at design freeze.

### Q3 — What artifact licenses removing the A13 caveat?

| Option | Description | Selected |
|--------|-------------|----------|
| Equivalence test + resolved count | A test asserting identical column sets at every decision date, plus the ratio published with its transition denominator. | ✓ |
| Equivalence test alone | Structural proof only; reader cannot judge whether the ratio rests on 6 transitions or 60. | |
| Equivalence test + count + degenerate-labeling flag | Both, plus an explicit flag when the labeling is near-constant. | |

**User's choice:** Equivalence test + resolved count

### Q4 — How do the two waves get planned?

| Option | Description | Selected |
|--------|-------------|----------|
| Plan wave 1 only, replan wave 2 after | Land wave 1's results, then run `/gsd-plan-phase` again with real numbers in hand. | ✓ |
| Plan both waves upfront | Cheaper; wave 2's hypothesis gets written before the evidence that should shape it exists. | |
| Plan both, with a declared wave-2 revision point | Plan both, revise at a wave-1 exit checkpoint only if needed. | |

**User's choice:** Plan wave 1 only, replan wave 2 after
**Notes:** This materially changes the handoff — the next planning pass is scoped to wave 1 deliverables only. Wave 2's decisions were still captured to give that second pass its direction.

---

## Classifier #2's feature set + INV-01

*Two facts established before asking: (1) none of INV-01's three named invariants has an ingested source — no M2, no Wilshire/market-cap, no credit aggregate; (2) disjointness-against-13 collides with the proposal's own candidate list, which names gold/equity and commodity/equity relative strength while `gold` and `oil` are among the 13.*

### Q1 — How does disjointness handle ratios built from classifier #1's columns?

| Option | Description | Selected |
|--------|-------------|----------|
| Ratio of a #1 column is admissible | Test asserts no RAW column shared; a ratio is a genuinely different quantity (scale-invariant vs level). | ✓ |
| Strict — no ratio touching the 13 | Cleanest orthogonality story; drops gold/equity and oil/equity entirely. | |
| Admissible, but flagged in the dependence report | Allowed, with per-feature dependence reporting. | |

**User's choice:** Ratio of a #1 column is admissible

### Q2 — What does wave 2 do about INV-01's un-ingested invariants?

| Option | Description | Selected |
|--------|-------------|----------|
| Ingest M2 and credit; leave market-cap out | M2SL (1959+) and a credit aggregate are free and reach back far enough; market-cap stays blocked for its documented reason. | ✓ |
| Build only what exists today | Smallest scope; INV-01 only partly satisfied by this phase. | |
| Ingest all three, accept a shorter invariant history | Reintroduces the staggered-start problem wave 1 just eliminated. | |

**User's choice:** Ingest M2 and credit; leave market-cap out
**Notes:** Flagged for research — M2 is monthly 1959+, but credit aggregates and GDP are quarterly *agency* series, and `lean_feature_set()` deliberately excludes the agency tier (design §9). Classifier #2 is not the lean set, so this may be fine, but it needs a stated taxonomy position.

### Q3 — Does wave 1's freeze rule apply to classifier #2?

| Option | Description | Selected |
|--------|-------------|----------|
| Same freeze rule, same 1972+ window | One policy platform-wide; the two labelings stay month-for-month comparable, which the dependence measurement needs. | ✓ |
| #2 gets a later start window | Keeps gold-in-equities; creates a time-varying joint model. | |
| Same rule, #2's set fixed by construction first | Name the list in the ADR before looking at data, then freeze to its common support. | |

**User's choice:** Same freeze rule, same 1972+ window
**Notes:** Net effect of Q1 + Q3 combined, recorded in CONTEXT.md so the planner does not misread it: oil/equity relative strength survives (raw `oil` is 1962+); gold/equity does not (1985-02 start fails the freeze). The ratio rule opens the door; the freeze rule closes it for gold.

### Q4 — How is classifier #2's (K, λ) set?

| Option | Description | Selected |
|--------|-------------|----------|
| By construction, zero trials | λ from the 4 × n_features formula, K pre-declared. Nothing added to the DSR denominator. | ✓ |
| Small pre-declared grid | A named tiny grid; each cell is a real trial the DSR must pay for. | |
| By construction, with K chosen for joint tractability | Same, but K set low because K₁ × K₂ governs joint state count. | |

**User's choice:** By construction, zero trials

---

## Joint labeling, orthogonality rule, trial budget

### Q1 — How does the joint (#1 × #2) labeling enter allocation?

| Option | Description | Selected |
|--------|-------------|----------|
| Two separate probability inputs | No product space; matches how allocation already consumes `regime_probs`; clean single-change ablation. | ✓ |
| Product state space | Most expressive; K₁ × K₂ cells over ~590 months means rare cells fall below the §4.4 5% floor. | |
| Two inputs now, product as a reported diagnostic | Two inputs for allocation; product cross-tab computed and reported with occupancy per cell. | |

**User's choice:** Two separate probability inputs
**Notes:** Grounded in audit item A7 — `active_regime` from hysteresis gates nothing; weights come from `vol_targeted_tilt(regime_probs, …)` in both the driver and the weekly report.

### Q2 — Which dependence statistic, and what makes it a failure?

| Option | Description | Selected |
|--------|-------------|----------|
| Report several, pre-declare no threshold | Adjusted Rand, Cramér's V and NMI side by side with the cross-tab; no pass/fail line. | ✓ |
| Cramér's V with a pre-declared threshold | Genuinely falsifiable, but the threshold would have no empirical basis in this project yet. | |
| Several statistics + a declared directional expectation | No hard line, but a pre-declared expected direction and rough magnitude. | |

**User's choice:** Report several, pre-declare no threshold
**Notes:** Consistent with the plausibility-bands-only posture and with "a number, not a goal."

### Q3 — Which deflated-Sharpe denominator?

| Option | Description | Selected |
|--------|-------------|----------|
| Whole registry since project start | Every configuration ever evaluated contributed to selection. Harshest penalty; the reason the registry is git-tracked as tamper-evidence. | ✓ |
| This phase's trials only | Friendlier number; quietly pretends Phases 3–5's searches never happened. | |
| Both, reported side by side | Full-registry figure as headline, phase-only shown as the marginal cost. | |

**User's choice:** Whole registry since project start
**Notes:** `registry/trials.jsonl` holds 30 entries as of 2026-09-10.

### Q4 — Does the phase declare a hard trial ceiling?

| Option | Description | Selected |
|--------|-------------|----------|
| Declare a ceiling in the ADR | Expected count written down before running; exceeding it requires an explicit amendment. Makes silent search-creep visible. | ✓ |
| No ceiling, registry is enough | The DSR penalty scales automatically; a ceiling adds bureaucracy. | |
| Ceiling on wave 2 only | Wave 1 is bounded; wave 2 is where a feature-set search could expand. | |

**User's choice:** Declare a ceiling in the ADR
**Notes:** ~5 expected trials (2 wave-1 policy runs, 1 classifier-#2 fit, 1 joint, 1 #1-alone), putting the registry near 35 at phase end.

---

## Claude's Discretion

- Whether L2's `_cv_safe_active_features` admission path is also frozen (recommendation: leave it in wave 1, note the parallel in the ADR — it is L2, and it exists for a different reason).
- Where the ADR physically lives and its numbering scheme.
- Exact form of the criterion-1 equivalence test.
- Module layout for the new relative-strength code inside `platform/`.
- Which credit aggregate series to use for INV-01, and its agency-tier alignment treatment.
- Report and plot layout for the pre/post table and the dependence cross-tabulation.

## Deferred Ideas

- A quality gate that can fail on a bad-but-working model (audit A11) — offered and declined; open until design freeze.
- Freezing L2's `_cv_safe_active_features` path.
- Market-cap/GDP as an invariant — blocked on a free 1962+ market-cap source.
- gold/equity relative strength — inadmissible under the 1972+ freeze; would need a longer gold splice.
- A7 — making hysteresis actually gate the portfolio rather than a reported label.
- Raising K on classifier #1 / (K, λ) sweeps — explicit non-goals; v2 requirement L1-V2-01.
