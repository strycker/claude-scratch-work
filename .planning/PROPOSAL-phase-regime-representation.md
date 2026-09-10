# Proposed Phase — Regime Representation (A13/A15 + the leadership axis)

**Status:** DRAFT FOR APPROVAL. Nothing committed to `ROADMAP.md` or `REQUIREMENTS.md`.
**Raised:** 2026-09-10, from the Phase 6 P3 sign-off (accept-with-caveats).
**Supersedes nothing.** Absorbs INV-01 (current Phase 8) — see "Numbering" at the end.

## Why this phase exists

Two independent findings converged on the same root cause.

**From the audit:** A13 (HIGHEST) and A15 (Medium) are open and **assigned to no phase**.
Phase 6 was scoped only to *display* A13, which it did — the reference and filtered
labelings disagree on **389 / 470 = 82.8%** of compared months, with no diagonal structure
in the confusion table. The §5.4 sojourn/lag headline is therefore not interpretable, and
nothing is scheduled to fix that.

**From the sign-off:** the regimes are crisis regimes, not allocation regimes. The
operator's decision is relative — *"would portfolio A outperform portfolio B now?"* — and
the current labeling does not inform it during ordinary months, which is where most months
live.

Both trace to the same thing: **what the labeler is allowed to see.**

## A13's mechanism — diagnosed, not speculated

`backtest/driver.py::_window_active_features` admits a feature once it has
`feature_min_history` months of data *in that window*. Config sets it to **120 months**.
Combined with staggered start dates that fully explains the 7 changes:

| feature | first observed | +120 months | observed change date |
|---|---|---|---|
| `curve_10y2y` | 1976-06 | 1986-06 | **1986-06** |
| `gold`, `oil` | 1985-02 | 1995-02 | **1995-02** |
| `fred_vix` | 1990-01 | 2000-01 | **2000-01** |

Meanwhile `evaluation/report.py::_reference_label_columns` picks a **fixed** set at the
first decision date and holds it for all 695 months. So the driver's labeling is fit on a
growing feature space while the reference is fit on a frozen one — and §5.4 compares them
as if they were the same estimator. **This is a policy disagreement, not a defect.**

## Goal

Decide what the regime labeler should see, prove that decision walk-forward, and add a
**second, independent labeler on relative/leadership features** so the platform can inform
allocation during ordinary markets — not only crisis avoidance.

## Scope

### Wave 1 — Resolve A13/A15 and establish whether classifier #1 earns its keep

**This wave is a gate. Wave 2 does not start unless it passes.**

1. **Enumerate the candidate feature policies** and state each one's cost honestly:
   - *(a) Fixed common-support set* — only features spanning the full window (~4: `cape_shiller`,
     `credit_spread_baa_aaa`, `curve_10y3m`, `div_yield`). Stable estimator, discards VIX/gold/oil.
   - *(b) Expanding set* (current driver) — uses all data but the estimator changes 7 times.
   - *(c) Fixed full set with explicit backfill/imputation* — constant feature space, but
     imputation is a modelling assumption that must itself be validated.
   - *(d) Fixed set + a documented `feature_min_history` change* — e.g. 60 months.
2. **Make driver and report use one policy.** Whatever is chosen, both paths use it. The
   §5.4 comparison becomes apples-to-apples by construction.
3. **Re-measure the disagreement** under the chosen policy. The 82.8% figure is the
   pre-fix baseline; the post-fix number is wave 1's headline result.
4. **Re-run the walk-forward and re-measure the ablation delta.** Today
   `wealth_delta +0.379267` but `dd_delta −0.014364` — the regime layer makes max drawdown
   slightly *worse* than its regime-free twin. Wave 1 must state plainly whether classifier
   #1 earns its keep on the dimension it was built for.
5. **Every policy evaluated goes in the trial registry.** No exceptions — this is a
   feature-selection search and it is exactly what the registry exists to count.

**Gate to wave 2:** the §5.4 ratio is interpretable, and classifier #1's contribution is
measured (positive or negative — a negative result is a valid gate outcome and is recorded,
not worked around).

### Wave 2 — The leadership classifier (classifier #2)

1. **Build relative/invariant features natively in `platform/`.**
   **Do NOT import the legacy `momentum.py` / `divergence.py`.** Verified this session:
   `platform/` currently imports **nothing** from the legacy library — it is already fully
   decoupled. Importing them would create the first coupling seam and partly undo what the
   migration phase exists to achieve. Port the algorithms instead.
   Candidate features (all *ratios and relative strengths*, deliberately disjoint from the
   13 level/stress features classifier #1 uses):
   - equity/bond relative strength, and **the stock–bond rolling correlation** — the single
     most important missing variable; it changed sign around 2000 and again around 2021 and
     governs whether bonds diversify equities at all
   - commodity/equity and gold/equity relative strength (S&P-in-Gold, S&P-in-Oil, Gold-in-Oil)
   - CPI acceleration (second derivative) — a direct inflation/stagflation signal
   - curve *changes* rather than levels (steepening/flattening as a leadership signal)
   - named macro invariants per INV-01: M2/GDP, market-cap/GDP, credit/GDP
2. **Fit classifier #2 unsupervised on those features** — same jump-model machinery,
   separate instance, separate (K, λ).
   **It must NOT be fit to maximise portfolio outperformance.** A classifier fit directly on
   forward relative returns is a return predictor wearing a regime costume, and the deflated
   Sharpe will correctly destroy it. Discovery unsupervised; validation walk-forward.
3. **Verify orthogonality is real, not assumed.** Measure mutual information / Cramér's V
   between the two labelings. If they are highly dependent, classifier #2 has rediscovered
   the stress axis and adds no lift — that is a falsifiable failure condition, and it should
   be stated as one.
4. **Carry both labelings jointly** into the downstream allocation layer as interacting
   features, and measure the lift of the pair against classifier #1 alone, walk-forward.

## Success criteria (falsifiable, with bands — per the Phase 6 lesson)

1. Driver and report label under **one documented feature policy**; a test fails if they
   diverge. The policy choice and its rejected alternatives are recorded as an ADR.
2. The §5.4 sojourn/lag ratio is **interpretable**: computed between two labelings fit on
   the same feature space, with its resolved-transition count. The A13 caveat is *removed*
   only because the cause is fixed — never because the wording was softened.
3. Post-fix labeling disagreement is **measured and reported** against the 82.8% pre-fix
   baseline. (No target is set — a number, not a goal, to avoid fitting to it.)
4. Classifier #1's ablation delta is re-measured and reported on **both** axes
   (`wealth_delta` and `dd_delta`), each within its `06-VALIDATION.md` band.
5. Classifier #2 exists, is fit unsupervised on a feature set **disjoint** from classifier
   #1's 13 (a test asserts the disjointness), and its labeling has documented occupancy
   summing to 1.0 with no state below the §4.4 5% floor unmarked.
6. Statistical dependence between the two labelings is measured and reported. High
   dependence is a **failure to add an axis** and is recorded as such.
7. The joint (#1 × #2) allocation lift is measured walk-forward against #1 alone, with
   every configuration in the trial registry and the deflated-Sharpe correction applied for
   the full trial count.
8. `platform/` still imports nothing from the legacy library (existing import-guard test
   extended to the new modules).

## Explicit non-goals

- **No fitting to forward returns.** (Criterion 2 of wave 2.)
- **No raising K on classifier #1** — the crisis classifier is kept as-is. Subdividing a
  stress-dominated space yields finer stress gradations, not new axes, and state 0 already
  holds ~11 of 695 months.
- **No 2021+ holdout use for any selection decision.** Unchanged.
- **No migration work** — that is its own phase, unaffected.

## What is reused unchanged

The jump-model labeler, all honesty rails (walk-forward, purged CV, trial registry,
holdout), the `platform/plotting/` package, and all six notebooks. P3 and P6 gain panels;
they are not rewritten. This phase is **additive** to Phase 6's output.

## Cost and risk

| | |
|---|---|
| **Largest risk** | Trial-space expansion. Two classifiers × (K, λ) × two feature sets. The deflated-Sharpe penalty scales with trial count — this is a real statistical cost, not bookkeeping. |
| **Mitigation** | Wave 1 is a gate; wave 2 only runs if #1 is sound. Feature sets are chosen by construction (disjoint, named, economically motivated), not by search. |
| **Second risk** | Classifier #2 rediscovers the stress axis. Mitigated by making orthogonality a measured, falsifiable criterion (#6) rather than an assumption. |
| **Rework avoided** | Solving A13 and the feature architecture together avoids touching the labeling machinery twice. |

## Numbering — needs a decision

This phase absorbs **INV-01** (current Phase 8), whose named-invariant discovery is exactly
wave 2's feature work. Two options:

- **Option A (recommended):** new **Phase 7 — Regime Representation** (carries a new REG-01
  plus INV-01); migration becomes **Phase 8**; old Phase 8 is retired into this one.
- **Option B:** insert as **Phase 6.5** (GSD decimal convention), leaving 7 = migration and
  8 = invariants numbered as-is, with INV-01 reduced to whatever wave 2 does not cover.

Option A is cleaner — INV-01 and wave 2 are the same work, and leaving a hollowed-out
Phase 8 invites drift. Option B avoids renumbering references in existing docs.
