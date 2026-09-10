# Proposal — Dual Regime Classifiers (crisis axis × leadership axis)

**Raised:** 2026-09-10, at the Phase 6 P3 cold-start sign-off (accept-with-caveats).
**Status:** proposal — not scheduled. Sequencing recommendation below is the load-bearing part.

## The operator's diagnosis, confirmed by inspection

> "We may be TOO focused on the negative (drawdowns) vs actionable states … it's more
> important to know at any given time whether a particular proposed portfolio A would
> likely outperform a different portfolio B."

This is correct, and the mechanism is visible in the feature set rather than in the model.

## Why the current regimes are crisis regimes — a representation limit, not a tuning failure

`taxonomy.lean_feature_set(cfg)` gives the labeler exactly **13 features**:

| group | features | n |
|---|---|---|
| stress / volatility | `credit_spread_baa_aaa`, `fred_vix`, `realized_vol_1m`, `realized_vol_3m` | 4 |
| term structure | `curve_10y2y`, `curve_10y3m`, `real_rate_level` | 3 |
| valuation | `cape_shiller`, `div_yield` | 2 |
| commodity **levels** | `gold`, `oil` | 2 |
| own-price momentum | `trailing_return_1m`, `trailing_return_3m` | 2 |

Two structural facts follow:

1. **Nearly a third of the set is direct stress measurement.** A jump model with an L2
   switching penalty finds the dominant variance axis; here that axis is calm-vs-stressed.
   The regimes recovered are the regimes *findable* in this space.
2. **Not one feature is cross-sectional or relative.** Every column is a level or a
   single-asset property. `gold` and `oil` enter as levels, never as ratios to equities.
   There is no equity/bond, growth/value, large/small, or commodity/equity relative
   strength anywhere in the set.

"Tech growth leadership", "strong bonds", "stagflation" are **relative** states. A
classifier with no relative information cannot represent them, at any K. This is why the
labeling is crisis-shaped, and why it is not a defect of the labeler.

## Why raising K is the wrong lever

- It subdivides the existing stress axis — "mild / moderate / severe stress" — rather than
  adding a new axis. No new information enters.
- State 0 already holds ~11 of 695 months (1.58%) and trips the §4.4 5% soft floor. At
  K = 8–10 several states fall below ~20 observations, making per-regime asset statistics
  unusable. The legacy pipeline hit exactly this and adopted balanced clustering for it
  (ADR #3: "a cluster of 10 quarters has unreliable mean/std estimates").
- More states in the same space is strictly more overfit for strictly less interpretability.

## Why two independent classifiers is the right lever

**It is a factorisation, not a finer partition.** Two K=5 classifiers span 25 joint cells
using ten states' worth of parameters. A flat K=25 classifier would need 25 states over
695 months — ~28 months each, unusable. This is the standard factorial-HMM argument
(Ghahramani & Jordan, 1997): factored state spaces are far more sample-efficient than the
flat product space they represent.

**The interaction is the payoff, as the operator identified.** "Inflation regime AND
crisis" implies a different mix than "inflation regime AND calm". A single classifier must
spend states to encode that cross product; two classifiers hand it to the downstream
optimiser for free as an interaction term.

## The design requirement: orthogonality

If classifier #2 is fit on features correlated with the stress set, the two classifiers say
the same thing — no lift, plus collinear features into the optimiser. Classifier #2's
feature set must be deliberately **disjoint** and **relative**.

Machinery that already exists and is directly reusable (legacy library, D17/D18):

- `momentum.compute_relative_strength()` — S&P-in-Gold, S&P-in-Oil, Gold-in-Oil
- `momentum.compute_inflation_acceleration()` — CPI second derivative; a direct
  inflation/stagflation-regime signal
- `divergence.compute_rolling_correlation()` / `compute_divergence()` — **the stock–bond
  correlation is the single most important missing leadership variable.** It changed sign
  around 2000 and again around 2021; it governs whether bonds diversify equities at all,
  and it appears nowhere in the current 13.

Candidate additions: equity/bond relative strength, commodity/equity relative strength,
curve *changes* rather than levels, and a growth/value or large/small proxy where free data
permits.

## The honest-framework cost — the real risk

1. **Trial-space expansion.** Two classifiers means 2 × (K, λ) × two feature sets. Every
   evaluated configuration goes in the trial registry; the deflated-Sharpe correction grows
   with the number of trials. This is a material cost, not bookkeeping.
2. **The objective is one step from fitting returns.** Classifier #1 was fit *unsupervised*
   on macro data. Classifier #2 is motivated by "which portfolio outperforms" — and if it is
   fit to maximise outperformance directly, it is a return predictor wearing a regime
   costume. That is precisely where backtest overfitting lives.
   **Recommendation: fit #2 unsupervised on relative-strength features, then *measure*
   allocation lift walk-forward.** Discovery unsupervised; validation walk-forward. Same
   discipline that makes #1 trustworthy.

## Sequencing — the strongest caution

Two facts say **do this second, not next**:

- **A13 is unresolved.** Filtered vs smoothed labelings disagree on 389/470 = **82.8%** of
  compared months with no diagonal structure. Classifier #1's real-time behaviour is not yet
  trustworthy.
- **Classifier #1 has not yet demonstrated it earns its keep.** The ablation delta splits:
  `wealth_delta +0.379267` but `dd_delta −0.014364` — the regime layer currently makes max
  drawdown slightly *worse* than its own regime-free twin, on the very dimension it exists
  to improve.

Adding a second classifier on top of an unvalidated first one compounds uncertainty and
makes attribution ambiguous: if the pair helps, which one helped? Settle A13, establish #1's
contribution, then add #2.

## Roadmap fit

**Phase 8 — "Invariants & Dimensional Reduction"** already exists to *"discover named,
era-stable conserved quantities as candidate regime features"*, with a success criterion
that survivors be admitted as **named** features, never anonymous principal-component
blocks. That is precisely the feature-discovery work classifier #2 requires, and precisely
the interpretability constraint a leadership classifier needs.

Natural shape: Phase 8 discovers and validates the leadership feature set; classifier #2 is
then fitted on it as a follow-on (extended Phase 8, or its own phase). No roadmap change is
made by this proposal — recorded for the operator to schedule.
