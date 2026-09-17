# Phase 7: Regime Representation - Research

**Researched:** 2026-09-14
**Domain:** Feature-policy unification for a walk-forward regime labeler + honest-evaluation
metric interpretability (quant-research Python codebase, no new external services)
**Confidence:** HIGH for wave 1 (every claim below is grounded in a file read this session, or
an empirical measurement run this session). MEDIUM for wave 2 (direction only, per D-09 — this
research pass supports wave-1 planning; wave 2 gets its own research/plan pass after wave 1
lands).

**Scope of this document:** Per `07-CONTEXT.md` D-09, **this phase-planning pass covers WAVE 1
ONLY.** Wave 2 material below (feature engineering, classifier #2, dependence measurement,
joint allocation lift) is included as forward-looking direction — required reading for the
*next* `/gsd-plan-phase 7` invocation — but the planner reading this document now should scope
plans to wave 1's deliverables (ROADMAP criteria 1-4) exclusively.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Feature policy — the A13/A15 resolution (wave 1)**

- **D-01: Freeze the driver to a constant feature space.** A13 is an asymmetry between an
  expanding driver (`driver.py::_window_active_features`, admitting a feature at
  `feature_min_history` = 120 months in-window) and a frozen reference
  (`report.py::_reference_label_columns`, fixed at the first decision date). Of the two
  reconciliation directions, **only freezing the driver preserves what §5.4 measures**.
  Expanding the reference to match would turn it from ONE hindsight fit into a second
  walk-forward, so §5.4 would then compare two causal estimators — a different quantity
  from detection lag. — **Reversibility:** costly — reverting means re-deriving every
  labeling-dependent number again (§5.4 ratio, disagreement %, both ablation deltas, and
  the nowcaster's Brier/confusion, which depend on it via the y_true join at
  `report.py:617`).

- **D-02: The frozen set is the 9 common-support features.** Verified empirically this
  session against the live `monthly_features` checkpoint: of the 13 lean features, exactly
  **9** are non-NaN across the whole decision range from 1972-01, and 4 are not —
  `curve_10y2y` (1976-06), `gold` (1985-02), `oil` (1985-02), `fred_vix` (1990-01). The
  driver adopts what the reference already computes, so the two paths converge by
  construction and the criterion-1 equivalence test is trivial to write.
  **Note two figures the proposal conflates:** 9 is the common-support set across the
  decision range; 4 is what the driver's FIRST window had at ≥120 months. Different
  quantities.

- **D-03: The 13-feature + imputation variant runs once as a logged trial.** The 9-feature
  policy is the decision, but the imputation alternative is executed once and registered so
  the ADR's rejection is evidence-backed rather than argued. Rationale for rejecting it on
  the merits: imputation would fabricate pre-1990 VIX levels, and an invented stress feature
  is the worst possible place to put a modelling assumption inside a crisis classifier.

- **D-04: Selection criterion is structural and pre-declared — interpretability only.** The
  policy is chosen on the structural requirement that driver and reference fit on the same
  space, declared before any run. The disagreement %, the §5.4 ratio, and both ablation
  deltas are **reported, never used to choose**. Explicitly rejected: using `dd_delta` as a
  veto that sends you back to try another policy — that is fitting the feature policy to the
  drawdown metric, and it is what this project's honesty framework exists to prevent.

- **D-05: Superseded numbers get a side-by-side pre/post table.** Every affected figure
  (82.8% disagreement, §5.4 ratio 0.591, Brier 0.2087, `wealth_delta` +0.379267,
  `dd_delta` −0.014364) appears under both policies, with the pre-fix column labelled as the
  baseline it is. **Mechanical fact the planner must carry into that table:** Brier and the
  confusion tables move because their LABELS changed (`full_sample_states` is reindexed onto
  the decision dates as `y_true`), not because the nowcaster improved.

**Gate semantics (wave 1 → wave 2)**

- **D-06: Measurement is the gate, not the sign.** Wave 2 proceeds whatever `dd_delta`
  turns out to be, provided it is honestly measured and recorded. Rationale: classifier #2
  is a *different axis on disjoint features* — its value does not depend on #1 winning, and
  a weak #1 is an argument FOR adding an axis. Also practical: the explicit non-goals forbid
  the obvious remedies (no raising K, no (K, λ) tuning), so "halt and fix #1" has no
  sanctioned move available inside this phase.

- **D-07: Plausibility bands only — no quality gate.** Bands assert each number is
  physically possible (`dd_delta` ∈ [−1, 1], disagreement ∈ [0, 1], ratio > 0), not that it
  is good. This holds the Phase 5 D-01 / Phase 6 D-16 posture: the project has no gate that
  can fail on a bad-but-working model, only on a broken one, and that stays a deliberate
  tracer-bullet choice. **Audit item A11 remains open and conscious**, to be re-examined at
  design freeze — it was offered as a quality band on `dd_delta` this session and declined.

- **D-08: The A13 caveat is licensed off by two artifacts, together.** (a) A test asserting
  driver and reference resolve to identical column sets at every decision date — this is
  criterion 1's failing test. (b) The §5.4 ratio published with its resolved-transition
  denominator, so a reader can see whether it rests on 6 transitions or 60. The caveat is
  removed because the cause is fixed, never because the wording softened.

- **D-09: Plan and execute WAVE 1 ONLY; wave 2 gets a second planning pass.** After wave 1's
  results land, run `/gsd-plan-phase 7` again to plan wave 2 with the actual numbers in
  hand. Costs one extra planning cycle and buys a wave-2 hypothesis that is honest to what
  wave 1 found. **The planner must scope this pass to wave 1's deliverables only.** Wave 2's
  decisions below (D-10…D-16) are recorded to give that second pass its direction, not to be
  planned now. — **Reversibility:** reversible.

**Classifier #2 — feature set and INV-01 (wave 2 direction)**

- **D-10: Disjointness is asserted on RAW columns; ratios derived from #1's columns are
  admissible.** The test asserts no raw column is shared with classifier #1's **13** (not
  the 9 actually used — the released `fred_vix`/`gold`/`oil`/`curve_10y2y` stay out of #2,
  because admitting level/stress features into a leadership classifier invites exactly the
  criterion-6 failure). A *ratio* is a genuinely different quantity — relative strength is
  scale-invariant where a level is not — so gold-in-equities and oil-in-equities remain
  admissible in principle.

- **D-11: Same freeze rule, same 1972+ decision window for classifier #2.** One documented
  policy across the platform, and the two labelings stay month-for-month comparable — which
  criterion 6's dependence measurement requires.
  **Net effect of D-10 and D-11 combined, which the planner must not misread:**
  oil/equity relative strength **survives** (raw `oil` in `monthly_raw` runs 1962+);
  gold/equity **does not** (`gold` starts 1985-02 and fails the common-support freeze). The
  ratio rule opens the door; the freeze rule is what closes it for gold specifically.

- **D-12: INV-01 — ingest M2 and a credit aggregate; leave market-cap/GDP out.** Verified
  this session: **none** of INV-01's three named invariants has an ingested source. M2SL
  (1959+) and a credit aggregate (e.g. TOTALSL / BCNSDODNS) are free and reach back far
  enough. Market-cap/GDP stays blocked for its already-documented reason — no free 1962+
  market-cap source; FRED's Wilshire starts ~1970 — and that block is **restated in the
  ADR**, not quietly worked around. See the `buffett_indicator` comment at
  `config/platform_settings.yaml:248`.

- **D-13: Classifier #2's (K, λ) is set by construction — zero selection trials.** λ from
  the existing feature-count formula (λ = 4 × n_features, per 03-RESEARCH Pitfall 5) and K
  pre-declared. No sweep, nothing added to the DSR denominator. Consistent with the v2
  deferral of (K, λ) tuning and with the proposal's own stated mitigation ("feature sets are
  chosen by construction, not by search").

**Joint labeling, dependence, and trial budget (wave 2 direction)**

- **D-14: Two separate probability inputs — no product state space.** Both labelings'
  probability vectors feed the allocation tilt as separate inputs. Rationale: allocation
  already consumes probabilities, not labels — audit item **A7** found `active_regime` from
  hysteresis gates nothing; weights come from `vol_targeted_tilt(regime_probs, …)` in both
  the backtest driver and the weekly report. A product space would also thin badly (K₁ × K₂
  cells over ~590 decision months, with occupancy never uniform, so rare cells fall below
  §4.4's ~8% floor (§4.4 crit. 1 is ~8%–~35%; see ADR-0001 § AMENDMENT 2026-09-17)). Keeping two inputs makes the lift-vs-#1-alone comparison a clean
  single-change ablation. — **Reversibility:** costly — switching to a product space later
  changes the allocation input contract and invalidates the lift comparison.

- **D-15: Dependence is reported with several statistics and NO pre-declared threshold.**
  Report adjusted Rand, Cramér's V and normalized mutual information side by side with the
  cross-tabulation. No pass/fail line — consistent with D-07's plausibility-only posture and
  with the proposal's "a number, not a goal". A human reads it and the judgement is recorded.
  A threshold was offered and declined on the grounds that it would have no empirical basis
  in this project yet.

- **D-16: Deflated Sharpe uses the WHOLE registry since project start.** `registry/trials.jsonl`
  holds **30 trials** as of 2026-09-10 [now **34** as of this research session, see Pitfall
  6 below — the count keeps growing from re-runs and must be re-read at execution time, not
  assumed]. Every configuration ever evaluated on this data contributed to the selection
  process that produced today's model, whether or not this phase ran it — that is why the
  registry is git-tracked as tamper-evidence. Counting only this phase's trials was offered
  and declined.

- **D-17: A trial ceiling is written into the ADR before running.** Expected count is roughly
  5 (2 wave-1 policy runs, 1 classifier-#2 fit, 1 joint, 1 #1-alone), putting the registry
  near 35 at phase end. Exceeding the ceiling requires an explicit amendment. This makes
  silent search-creep visible — the failure mode the registry can record but cannot prevent.
  **Correction found this session (Pitfall 6): each `run_full_backtest_evaluation()` call
  appends TWO registry rows, not one — see below. Re-derive the ceiling accordingly.**

### Claude's Discretion

- **Whether L2's separate admission path is also frozen.** `driver.py::_cv_safe_active_features`
  (line 120) ALSO expands, but it governs the **nowcaster** (L2), not the labeler (L1), and
  it exists for a distinct reason — it narrows until `CalibratedClassifierCV` has
  `n_splits` examples of every class present. A13 is an L1 problem. Recommendation: leave L2
  alone in wave 1, note the parallel in the ADR, and let the researcher confirm the two paths
  are genuinely independent. Raised in discussion and consciously not pursued.
  **This research confirms the recommendation is executable as stated**: `_cv_safe_active_features`
  is called only from `_refit_l2` (`driver.py:243`); freezing `_refit_l1`'s admission does not
  touch it. See Architecture Patterns, Pattern 1.
- Where the ADR physically lives (`platform_design/adr/`, `.planning/`, or the phase dir) and
  its numbering scheme.
- The exact form of the criterion-1 equivalence test (parametrized over decision dates vs. a
  single set-comparison assertion at each of a sampled few).
- Module layout for the new relative-strength code inside `platform/` (a new
  `platform/features/relative.py`, or extending `transforms_monthly.py`).
- Which credit aggregate series to use for INV-01, and its agency-tier alignment treatment.
- Report and plot layout for the pre/post table and the dependence cross-tabulation, within
  the existing `platform/plotting/` conventions.

### Deferred Ideas (OUT OF SCOPE)

- **A quality gate that can fail on a bad-but-working model** (audit item A11) — offered
  this session as a `dd_delta > 0` band and declined in favour of plausibility-only bands
  (D-07). Stays open and conscious, to be re-examined at design freeze.
- **Freezing L2's `_cv_safe_active_features` admission path** — the nowcaster expands too,
  but for a different and legitimate reason (CV class counts). Noted in the ADR; not acted
  on in wave 1. See Claude's Discretion.
- **Market-cap/GDP as an invariant** (part of INV-01) — blocked on a free 1962+ market-cap
  source. FRED's Wilshire starts ~1970. Revisit if a source appears, or if the paid-provider
  seams (Norgate/Tiingo/EODHD) are ever activated.
- **gold/equity relative strength** — inadmissible under D-11's freeze at the 1972+ window
  because `gold` starts 1985-02. Would become available if the decision window were ever
  shortened, or if a longer gold splice (macrotrends back to 1915) were successfully
  ingested — note the macrotrends source is still only wiring-verified, not live-verified.
- **A7 — making hysteresis actually gate the portfolio** — `active_regime` stabilizes a
  reported label, not a portfolio. D-14 works with the system as it is rather than fixing
  this. Needs its own decision, per the audit.
- **Raising K on classifier #1 / (K, λ) sweeps** — explicit non-goals here; v2 requirement
  L1-V2-01.
- **No fitting to forward returns; no 2021+ holdout use for any selection decision; no
  migration work** — hard boundaries restated from ROADMAP for this phase.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| REG-01 (wave 1 portion) | One documented feature policy shared by the walk-forward driver and the evaluation reference (A13/A15), §5.4 ratio made interpretable, ablation delta re-measured on both wealth and drawdown | Architecture Patterns Pattern 1 gives the exact restructuring (`report.py`'s `_reference_label_columns` computed BEFORE `run_backtest`, threaded in as a frozen column list); Common Pitfalls 1-3 cover the concrete traps; Validation Architecture gives bands for the re-measured numbers; Code Examples gives the before/after sketch |
| REG-01 (wave 2 portion) | Second independent classifier on disjoint relative/leadership features, orthogonality measured, joint lift assessed | Recorded as direction only (D-09) — see "Wave 2 Direction" subsections throughout; not planned this pass |
| INV-01 | Named invariant candidates (M2/GDP, market-cap/GDP, credit/GDP), screened and logged, survivors admitted as named features | Recorded as direction only (D-09) — confirmed this session that M2SL and credit aggregates are NOT yet ingested (`config/platform_settings.yaml` `fred_monthly.series` has no M2/credit entry, verified by direct read); not planned this pass |

</phase_requirements>

## Summary

Wave 1 is a **restructuring of two existing, already-correct functions so they compute from
one shared call site**, not new algorithm work. `backtest/driver.py::_window_active_features`
(an expanding admission rule keyed on `feature_min_history`) and
`evaluation/report.py::_reference_label_columns` (a frozen-at-first-decision-date common-support
rule) already do different, individually-correct things — A13 is a policy disagreement, not a
bug, exactly as `07-CONTEXT.md` frames it. The fix D-01/D-02 lock in is to have the driver's L1
labeler (`_refit_l1` only — **not** `_refit_l2`, whose separate admission path is deliberately
left alone per Claude's Discretion) consume the SAME 9-column list the reference already
computes, rather than compute its own expanding list. This session traced the exact call graph
and found the cleanest implementation is to **compute `_reference_label_columns`'s output once,
before `run_backtest()` runs**, and thread it through as a parameter — because the `first_decision`
value the reference needs (`dev_features.index[min_train]`, verified — `expanding_steps` yields
`index[i]` for `i` starting at `min_train`, `honesty/walkforward.py:48-49`) does not actually
depend on anything `run_backtest()` produces; the current code only computes it *after*
`run_backtest()` runs because that is where `per_step_metrics["dates"]` happens to already be
available, not because of a true data dependency. This removes the apparent circularity between
"the driver needs the frozen list" and "the frozen list is computed from the driver's own dates."

Three concrete, code-grounded findings surfaced this session materially affect wave-1 planning
and are not yet reflected in `07-CONTEXT.md`: **(1)** the `monthly_features.parquet` checkpoint
on disk is measurably STALE relative to `monthly_raw.parquet` — `oil` has zero NaN in
`monthly_raw` from 1962-2020 (fully populated) but `monthly_features["oil"]` still starts
1985-02, identical to `gold`, even though `compute_lean_features()` is a direct passthrough
with no windowing (`transforms_monthly.py:252-253`). This means a rebuild would likely promote
`oil` into the common-support set, changing D-02's frozen set from 9 to a probable **10**
columns. **(2)** the trial registry currently holds **34** entries (not the 30 `07-CONTEXT.md`
records), and each `run_full_backtest_evaluation()` call appends **two** rows, not one (strategy
+ ablation, both via `run_backtest`'s own `registry.append_trial()`) — D-17's "~5 trials, ~35
total" ceiling needs re-deriving from these two corrections before it goes in the ADR. **(3)**
an empirical timing run this session shows the full 588-step L1+L2 walk-forward refit loop costs
roughly **2 minutes of wall-clock** (measured: 15 sampled steps averaged 0.111s L1 + 0.109s L2,
extrapolated to 588 steps ≈ 129s), so re-running the full evaluation for BOTH the 9-feature
policy and the D-03 13-feature+imputation trial is cheap — there is no reason to try to shortcut
via reused artifacts, and doing so would risk exactly the kind of staleness bug finding (1)
just caught.

**Primary recommendation:** resolve finding (1) FIRST — confirm whether `monthly_features` needs
a rebuild before computing the frozen set, since D-02's "9" was verified against a stale
checkpoint — then implement the frozen-column threading in `driver.py`/`report.py` as one shared
function call, write the criterion-1 equivalence test against the real re-run data, and produce
the D-05 pre/post table from two full `run_full_backtest_evaluation()` runs (cheap, ~2 min each)
rather than partial recomputation.

## Architectural Responsibility Map

This project's own five-layer design (`platform_design.md` §14: L0 data → L1 regime labeling →
L2 regime prediction → L3 asset prediction → L4 allocation, plus an Evaluation/Honesty
cross-cutting layer) is the correct tier taxonomy here — not a browser/API/DB web-app split.

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Shared feature-policy computation (frozen 9/10-column set) | L1 (labeling) | Evaluation/Honesty | The list is a *labeling* input, but must be identical wherever a labeling fit occurs — driver (walk-forward) and report (smoothed reference) are both L1 consumers |
| Driver's `_refit_l1` admission | L1 (labeling) | — | Consumes the shared list; no longer computes its own |
| Driver's `_refit_l2` admission (`_cv_safe_active_features`) | L2 (prediction) | — | Deliberately untouched (Claude's Discretion) — a distinct CV-safety concern, not a feature-policy concern |
| §5.4 sojourn/lag interpretability | Evaluation/Honesty | L1 | Consumes two L1 outputs (smoothed + filtered labelings) but the metric itself lives in `evaluation/sojourn_lag.py`, outside any modeling layer |
| Ablation delta re-measurement | Evaluation/Honesty | L4 (allocation) | `no_regime_ablation` reuses the L4 tilt code path with regime input disabled — the delta is an Evaluation-layer diagnostic over an L4 output |
| Pre/post comparison table + ADR | Evaluation/Honesty (reporting) | — | Pure presentation of already-computed numbers; no new modeling |
| A13_CAVEAT string update | Evaluation/Honesty (plotting) | — | `platform/plotting/core.py:74-81`, consumed by 3 plotting modules; single edit point |
| Relative-strength feature engineering (wave 2) | L1 (labeling input) | L0 (data) | New derived features feed a second L1 labeler fit; the port work itself is data-transform (L0-adjacent) |
| Classifier #2 fit (wave 2) | L1 (labeling) | — | Second instance of the same jump-model machinery |
| Dependence measurement (wave 2) | Evaluation/Honesty | — | Cross-tabulation of two L1 outputs |
| Joint allocation lift (wave 2) | L4 (allocation) | Evaluation/Honesty | Two probability inputs into `vol_targeted_tilt`; lift measured by the Evaluation layer |

## Standard Stack

### Core

No new external packages are required for wave 1. Every function used (`pandas`, `numpy`,
`scikit-learn` for the jump model / nowcaster, `pyarrow` for parquet) is already an installed,
already-used dependency of `trading_crab_lib` and `trading_crab_lib_lib` per `pyproject.toml`
(confirmed by successfully importing and running `_refit_l1`/`_refit_l2` from this environment
this session).

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pandas | already installed (2.0+ per `.claude/CLAUDE.md`) | DataFrame slicing for the frozen column set | Already the platform's sole tabular data structure |
| numpy | already installed (1.25+) | Array ops inside `jump_model.py`/`sojourn_lag.py` | Unchanged — no new math |
| scikit-learn | already installed (1.4+) | `KMeans`, `StandardScaler`, `CalibratedClassifierCV` (L1/L2, unchanged this wave) | Unchanged |

### Supporting (wave 2 direction — not installed by this pass)

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| scikit-learn (`sklearn.metrics.adjusted_rand_score`, `normalized_mutual_info_score`) | already installed | Criterion-6 dependence statistics | Verified importable this session (`sklearn.metrics.adjusted_rand_score`, `sklearn.metrics.normalized_mutual_info_score`) — **[VERIFIED: local import test this session]** |
| scipy (`scipy.stats.contingency.association`) | 1.17.1, already installed | Cramér's V | Verified importable this session (`from scipy.stats.contingency import association`) — **[VERIFIED: local import test this session]** |
| fredapi | already installed | M2SL / credit-aggregate ingestion (INV-01) | Same client already used by `ingestion/macro_monthly.py::fetch_fred_monthly` — no new adapter needed, just a new `fred_monthly.series` entry |

**No new packages are needed anywhere in this phase.** Wave 1 touches zero dependencies; wave 2's
dependence statistics and M2/credit ingestion reuse libraries already installed and already used
elsewhere in `platform/`.

**Installation:** none required.

## Package Legitimacy Audit

**Not applicable.** This phase installs no new external packages in wave 1, and wave-2 direction
(recorded for the next planning pass) also uses only already-installed dependencies
(`scikit-learn`, `scipy`, `fredapi`) verified present in this environment this session. The
Package Legitimacy Gate protocol is skipped per its own trigger condition ("whenever this phase
installs external packages").

## Architecture Patterns

### System Architecture Diagram

```
                     config/platform_settings.yaml
                     (taxonomy.fast + taxonomy.slow = 13 lean cols)
                                    |
                                    v
                    monthly_features checkpoint (1962-2020, dev-bounded)
                                    |
                    +---------------+----------------+
                    |                                 |
         [NEW: compute once, BEFORE run_backtest]      |
     _reference_label_columns(dev_features,             |
       lean_cols, first_decision=index[min_train])       |
                    |                                 |
          frozen_cols (9 or 10, TBD — see Pitfall 1)     |
                    |                                 |
      +-------------+----------------+                 |
      |                              |                 |
      v                              v                 v
 run_backtest()                report.py's OWN
 per step t:                   full-sample fit
   _refit_l1(train,             (unchanged --
     frozen_cols)  <--SAME-->    already uses
     [was: _window_             _reference_label_
     active_features             columns; now
     expanding rule]             reuses the SAME
   _refit_l2(train,              precomputed list,
     ...)  <-- UNCHANGED,        never recomputes)
     still uses
     _cv_safe_active_features
     (L2, untouched)
      |                              |
      v                              v
 per_step_metrics            full_sample_states
 (filtered, walk-forward)    (smoothed, hindsight)
      |                              |
      +--------------+---------------+
                     |
                     v
        build_filtered_probs_matrix +
        compute_sojourn_lag_headline
        (evaluation/sojourn_lag.py, UNCHANGED --
         only its INPUTS now come from one
         feature space instead of two)
                     |
                     v
        assemble_backtest_report()
        (D-05 pre/post table: NEW section,
         two full runs' worth of numbers
         side by side)
                     |
                     v
        platform/plotting/core.py::A13_CAVEAT
        (D-08: update or remove once the
         equivalence test + resolved-transition
         count both exist)
```

A reader can trace the primary use case (a decision date's labeling) from `monthly_features`
through the frozen-column computation, into both the walk-forward driver and the smoothed
reference, and out through the sojourn/lag headline into the report — the SAME box
(`_reference_label_columns`) now feeds both downstream paths, which is the literal mechanism
that makes criterion 1's equivalence test trivial: both consumers read the same in-memory list,
so a divergence can only happen if a future edit reintroduces a second computation.

### Recommended Project Structure

No new files or packages for wave 1. Modified files only:

```
src/trading_crab_lib/platform/
├── backtest/driver.py          # _refit_l1 gains a frozen-cols parameter; _window_active_features
│                                #   call removed from _refit_l1 (kept for _refit_l2/_cv_safe_active_features)
├── evaluation/report.py        # first_decision computed BEFORE run_backtest, not after;
│                                #   frozen_cols threaded into run_backtest(); D-05 pre/post table
│                                #   section added to assemble_backtest_report()
├── plotting/core.py            # A13_CAVEAT updated/removed per D-08 (single edit point)
platform_design/adr/            # NEW (or .planning/ — Claude's Discretion): the ADR file itself
tests/unit/
├── test_platform_backtest_driver.py    # equivalence test (criterion 1) added
├── test_platform_evaluation_report.py  # TestReferenceLabelColumns extended for the new frozen-cols
│                                        #   threading; D-05 pre/post table test added
```

### Pattern 1: Compute the frozen column list once, before the walk-forward loop

**What:** `report.py::run_full_backtest_evaluation` currently computes `first_decision =
pd.DatetimeIndex(per_step_metrics["dates"]).min()` (`report.py:598`) AFTER calling
`run_backtest()` (`report.py:569-571`), purely because that is where the dates happen to already
be available. But `first_decision` is actually `dev_features.index[min_train]` — a value fully
determined by the index and `min_train`, independent of anything `run_backtest` computes
(`honesty/walkforward.py:48-49`: `expanding_steps` yields `index[i]` for `i` starting at
`min_train`). This means the frozen list can — and for criterion 1, must — be computed BEFORE
`run_backtest()` runs, then passed IN, rather than computed independently by each side after the
fact (which is what currently makes A13 possible: `_window_active_features` and
`_reference_label_columns` are two independent computations that happen to run at different
times against different windows).

**When to use:** Any time two consumers must agree on a derived value that is expensive or
error-prone to recompute independently — this is the general "compute once, thread through"
principle Phase 3's DP-decode oracle test exemplifies for correctness; here it is applied to
avoid a *policy* divergence rather than a numerical one.

**Example (sketch — exact signatures are the executor's to finalize):**
```python
# report.py::run_full_backtest_evaluation, BEFORE step (a):
dev_features, _ = split_by_holdout_boundary(monthly_features, cutoff=DEFAULT_HOLDOUT_CUTOFF)
lean_cols = sorted(lean_feature_set(cfg) & set(dev_features.columns))
min_train = cfg.get("backtest", {}).get("min_train_months", 120)
first_decision = dev_features.index[min_train]  # == expanding_steps(...)'s first yielded t
frozen_cols = _reference_label_columns(dev_features, lean_cols, first_decision)

equity_curve, per_step_metrics = run_backtest(
    monthly_features, asset_returns, cfg,
    cash_returns=cash_ret, use_regime_tilt=True, registry_path=registry_path,
    frozen_l1_features=frozen_cols,   # NEW param, threaded into _refit_l1 only
)
# ... later, step (d) reuses frozen_cols directly instead of recomputing:
X_df = dev_features[frozen_cols].dropna()
```
```python
# driver.py::_refit_l1, new signature:
def _refit_l1(train_features, cfg, *, frozen_features: list[str] | None = None) -> pd.Series:
    ...
    if frozen_features is not None:
        active = [c for c in frozen_features if c in train_features.columns]
    else:
        active = _window_active_features(train_features, lean_cols, min_history=min_history)
    ...
```
`_refit_l2` (and its `_cv_safe_active_features` call) is UNCHANGED — this is the Claude's
Discretion boundary confirmed this session: `_cv_safe_active_features` is only ever called from
`_refit_l2` (`driver.py:243`), so freezing `_refit_l1`'s admission cannot accidentally touch L2.

### Pattern 2: Side-by-side pre/post table as a pure function

**What:** `assemble_backtest_report()` is already a pure markdown builder over precomputed dicts
(`report.py:94-136`, no I/O) — the existing pattern for D-05's table is to add a new section
function that takes TWO dicts (pre-fix baseline values, hard-coded from
`BASELINE-v1-tracer-bullet.md`'s "Current reference run" table since that run predates the
fix, vs. post-fix values from the new run) and renders them side by side. This mirrors exactly
how `assemble_backtest_report` already handles the baseline-gauntlet table (`report.py:227-255`).

**When to use:** Any comparison between a superseded baseline value and a newly-measured one
where BOTH numbers must remain visible (never silently overwritten) — D-05's explicit
requirement.

### Anti-Patterns to Avoid

- **Recomputing the frozen list independently in two places.** This is literally what A13 is —
  do not "fix" it by writing a second, hopefully-identical computation in `driver.py`. The
  equivalence test can only be trivially true if there is ONE call site.
- **Trusting the on-disk `monthly_features.parquet` checkpoint without checking staleness
  first.** See Common Pitfall 1 — this checkpoint is currently stale relative to `monthly_raw`,
  and the "9" in D-02 was computed against it.
- **Treating `_cv_safe_active_features` (L2) as part of this fix.** It solves a different
  problem (CV class-count safety) for a different layer (nowcaster, not labeler). Freezing it
  too would be scope creep beyond what D-01/D-02 authorize and beyond what Claude's Discretion
  recommends.
- **Skipping a full re-run in favor of patching persisted artifacts.** The persisted
  `backtest_full_sample_states.parquet` / `backtest_filtered_state_probs.parquet` artifacts
  (Phase 6 D-Amendment-3-H) were written under the PRE-fix policy. Patching them in place risks
  exactly the kind of silent staleness this session's Pitfall 1 finding already caught once.
  Full re-runs are empirically cheap (~2 minutes each, measured this session) — there is no
  performance reason to shortcut.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Common-support feature selection | A new "frozen feature set" function in `driver.py` | `report.py::_reference_label_columns`, called once and threaded through | It already exists, is already tested (`TestReferenceLabelColumns`, `test_platform_evaluation_report.py:334-366`), and reusing the identical function is what makes the criterion-1 test trivially provable rather than a second parallel implementation a test must police |
| Statistical dependence between two labelings (wave 2) | A hand-rolled Cramér's V / mutual-information function | `sklearn.metrics.adjusted_rand_score`, `sklearn.metrics.normalized_mutual_info_score`, `scipy.stats.contingency.association` | All three verified importable and installed this session; hand-rolling a contingency-table statistic is exactly the kind of "deceptively complex" problem this project's own `Don't Hand-Roll` culture (see incumbent CLAUDE.md) warns against |
| Deflated-Sharpe trial count | Manually counting `07-CONTEXT.md`'s stated "30" | `registry.read_trials(path=...)` (`honesty/registry.py:88-96`) — re-read the live ledger at execution time | The count has already changed once since context-gathering (30 → 34, confirmed this session) purely from other work re-running the same evaluation; a hard-coded count will be stale by the time the ADR is written |
| A13 caveat text | A second copy of the caveat string in each plotting module | `platform/plotting/core.py::A13_CAVEAT` (single string, `core.py:74-81`), consumed by `backtest.py`, `nowcaster.py`, `regime.py` | Confirmed by grep this session: exactly one definition, three consumers via `core.A13_CAVEAT` reference — one edit updates all three renderings |

**Key insight:** every "don't hand-roll" item above is not a third-party-library substitution
(the incumbent CLAUDE.md's more familiar sense) but an **in-repo single-source-of-truth**
substitution — the entire wave-1 fix is architecturally about eliminating a *second*
implementation of something that already exists once, correctly, in `report.py`.

## Common Pitfalls

### Pitfall 1: The on-disk `monthly_features.parquet` checkpoint is stale — verified this session

**What goes wrong:** D-02's "exactly 9 of 13" common-support count was verified against the
current `monthly_features.parquet` checkpoint, but that checkpoint's `oil` column starts
1985-02-28, IDENTICAL to `gold`, while `compute_lean_features()` (`transforms_monthly.py:252-253`)
is a direct, unwindowed passthrough: `features["oil"] = monthly_raw["oil"]`. Reading
`monthly_raw.parquet` directly this session shows `oil` has **zero NaN values from 1962-01
through 2020-12** (708/708 non-null) — **[VERIFIED: data/checkpoints/platform/monthly_raw.parquet,
read directly this session]**. A direct passthrough of a fully-populated 1962+ column cannot
legitimately produce a 1985-02 start in `monthly_features` unless the checkpoint predates a
`monthly_raw` rebuild that extended oil's coverage.

**Why it happens:** the platform has no `--recompute`-only mode (unlike the legacy quarterly
pipeline's checkpoint freshness/`--recompute` distinction) — `scripts/build_platform_data.py`
always re-ingests AND recomputes features together (confirmed by reading its `main()` — no
argparse flags for selective steps). If `monthly_raw` was rebuilt (e.g. during the 2026-09-09
CPI/ALFRED fix work, `976f7c8`/`8095498`) without a subsequent full re-run of
`build_monthly_spine()`, the derived `monthly_features` checkpoint silently falls behind its own
source data.

**How to avoid:** before computing (or re-verifying) the frozen column set for the ADR, the
planner must decide: (a) rebuild `monthly_features` from the current `monthly_raw` (network-
dependent — `build_monthly_spine()` re-fetches from FRED/multpl-equivalent/macrotrends/yfinance,
confirmed by reading its call chain; FRED access was verified live and working in this
environment this session via `fredapi.Fred.get_series()`, but macrotrends/yfinance reachability
was not separately re-verified this session — see Environment Availability), then recompute the
common-support set (likely 10, not 9, since `oil` would newly qualify); or (b) explicitly
document that the frozen set is pinned to the CURRENT (stale) checkpoint's 9 columns as a
conscious, dated decision, and record the staleness as a known limitation in the ADR rather than
silently inheriting it. **Either choice is defensible; silently treating "9" as settled fact
without addressing this is not** — it directly affects what column list the equivalence test
asserts.

**Warning signs:** if the executor reruns `_reference_label_columns` against a freshly-rebuilt
`monthly_features` and gets 10 (not 9) qualifying columns, that is this exact issue resurfacing,
not a new bug.

### Pitfall 2: `first_decision` looks like it depends on `run_backtest`'s output but does not

**What goes wrong:** a naive implementation of D-01 might conclude the frozen column list can
only be computed AFTER `run_backtest()` runs (since that is where the current code sources
`first_decision` from `per_step_metrics["dates"].min()`), creating an apparent chicken-and-egg
problem (the driver needs the frozen list to run; the frozen list needs the driver's output to
compute).

**Why it happens:** the current code takes the path of least resistance — `per_step_metrics`
happens to already carry the dates, so `report.py:598` reads them from there instead of deriving
them independently.

**How to avoid:** `first_decision` is exactly `dev_features.index[min_train]` — the same value
`expanding_steps(dev_features.index, min_train=min_train)`'s first yielded step would produce
(`honesty/walkforward.py:48-49`, verified by reading the function this session: `for i in
range(min_train, len(index), step): yield index[i], ...`). Compute it directly from the index
and `min_train` BEFORE calling `run_backtest()`, with no dependency on the walk-forward loop
having executed. See Architecture Patterns Pattern 1 for the code sketch.

**Warning signs:** if a plan task proposes running the walk-forward loop TWICE (once to get
`first_decision`, once for real), that is a sign this pitfall was not resolved.

### Pitfall 3: The registry trial count and per-run row count are both undercounted in `07-CONTEXT.md`

**What goes wrong:** `07-CONTEXT.md` D-16 states "30 trials as of 2026-09-10" and D-17 budgets
"~5 expected trials... putting the registry near 35 at phase end." Reading `registry/trials.jsonl`
directly this session shows **34 entries already** — **[VERIFIED: registry/trials.jsonl, read
directly this session, `wc -l` = 34]** — four more than the 30 recorded five days ago, from two
extra `run_full_backtest_evaluation()`-equivalent runs on 2026-09-11 (visible as two adjacent
`use_regime_tilt: true`/`false` pairs at those timestamps in the ledger, presumably from the
R1/R4 tech-debt verification activity noted in `STATE.md`). Separately, `no_regime_ablation()`
delegates to `run_backtest(use_regime_tilt=False)`, which calls `registry.append_trial()` itself
(`backtest/baselines.py:264` docstring: "its own registry trial via `run_backtest`'s single
`append_trial`") — so **every single `run_full_backtest_evaluation()` call appends TWO rows**,
not one (confirmed: `report.py:569-576` calls `run_backtest` once directly for the strategy leg
and once via `no_regime_ablation` for the ablation leg, each independently hitting
`driver.py:476-481`'s `registry.append_trial()`).

**Why it happens:** D-17's "2 wave-1 policy runs" language undercounts if it meant 2 registry
rows rather than 2 `run_full_backtest_evaluation()` calls; each call is actually 2 rows.

**How to avoid:** at ADR-writing time, (a) call `registry.read_trials()` fresh rather than
trusting the 30 or 34 figures recorded in planning documents — the count is a live, growing
number; (b) budget the ceiling as `2 rows × N full-evaluation runs`, e.g. wave 1's two policy
variants (9-feature; 13-feature+imputation, per D-03) cost **4 rows**, not 2, pushing the
registry to roughly 38 at wave-1's end (34 + 4), not the "~35" D-17 anticipated from the older
30-count baseline.

**Warning signs:** an ADR trial-ceiling section that states a number without a
`registry.read_trials()` call backing it up at write time.

### Pitfall 4: `canonicalize_states`' default sort key is a member of classifier #1's 13 (wave 2 — flag now, resolve later)

**What goes wrong (wave 2, recorded here so it is not rediscovered from scratch next pass):**
`labeling/jump_model.py::canonicalize_states` sorts states by ascending centroid coordinate of
`trailing_return_1m`, falling back to raw centroid column 0 with a WARNING if that column is
absent from the fit's feature set (`jump_model.py:214-221`). D-10 requires classifier #2's raw
columns to be disjoint from classifier #1's 13 — and `trailing_return_1m`/`trailing_return_3m`
ARE two of those 13 (`config/platform_settings.yaml` `taxonomy.fast`, verified by direct read
this session). So classifier #2's feature set structurally cannot include `trailing_return_1m`,
meaning EVERY fit of classifier #2 will silently fall back to the "centroid column 0" sort order
— the same fallback A14 diagnosed as corrupting cross-window label comparability for classifier
#1's early windows (`UAT-AUDIT-2026-09-09.md` N2/A14; closed as negligible there ONLY because it
fired on 1 of 588 steps — for classifier #2 it would fire on effectively every fit).

**Why it happens:** `canonicalize_states` was written assuming its caller's feature set always
includes `trailing_return_1m`; D-10's disjointness rule breaks that assumption for a second
caller.

**How to avoid (wave 2 planning, not this pass):** `canonicalize_states` needs either a caller-
supplied sort column (e.g., the ratio-based relative-strength feature most analogous to
"bear→bull" ordering for a leadership axis) or an explicit, non-default sort convention
documented before classifier #2 is fit — an unstable sort order would corrupt criterion 6's
dependence measurement exactly the way it corrupted early-window comparability for classifier #1.

### Pitfall 5: `A13_CAVEAT` is prose that must be updated, not deleted blindly

**What goes wrong:** D-08 says the caveat is "removed... never because the wording softened" —
a naive fix might delete the `A13_CAVEAT` string entirely once the equivalence test exists.

**Why it happens:** the string currently hard-codes the OLD mechanism description ("changes 7
times... 4 -> 6 -> 8 -> 9 -> 10 -> 12 -> 13 features", `core.py:75-80` — **[VERIFIED:
src/trading_crab_lib/platform/plotting/core.py:74-81]**, quoted verbatim in Code Examples
below), which becomes FALSE (not just outdated) the moment the driver is frozen — there is no
longer a 7-step feature-count progression to describe.

**How to avoid:** replace the string's content with the D-08 resolution narrative (what the
frozen policy is, and a pointer to the equivalence test + resolved-transition count), rather
than deleting the constant or its three call sites (`backtest.py:435/443`, `nowcaster.py:49`,
`regime.py:19` reference it) — those call sites still need SOME caption text for the sojourn/lag
panel; only the CONTENT changes.

### Pitfall 6: A rebuild of `monthly_raw`/`monthly_features` is network-dependent; the walk-forward re-run is not

**What goes wrong:** conflating "re-run the evaluation" (cheap, ~2 minutes, pure computation on
existing checkpoints — verified by empirical timing this session) with "rebuild the checkpoints"
(potentially expensive and network-dependent — `build_monthly_spine()` re-fetches from FRED,
multpl-equivalent scraping, macrotrends, and yfinance).

**Why it happens:** both are colloquially "re-running the pipeline," but they have very
different cost/risk profiles and only one is required by Pitfall 1's resolution IF the planner
chooses the rebuild path.

**How to avoid:** if Pitfall 1 is resolved by rebuilding, budget it as a separate, potentially
flaky, network-dependent task (FRED access was verified live and functional this session via
`fredapi`; macrotrends has documented wiring-only status per `STATE.md`'s ⚠ notes — see
Environment Availability). If resolved by pinning to the current checkpoint instead, no rebuild
is needed and the ~2-minute re-run estimate applies directly to both wave-1 policy variants.

## Code Examples

### The exact functions being reconciled (both already exist, both already correct in isolation)

```python
# src/trading_crab_lib/platform/backtest/driver.py:98-117 — EXPANDING rule (to be
# removed from _refit_l1's call path only; _refit_l2 keeps its own admission logic)
def _window_active_features(
    features: pd.DataFrame, cols: list[str], *, min_history: int
) -> list[str]:
    return [c for c in cols if int(features[c].notna().sum()) >= min_history]
```

```python
# src/trading_crab_lib/platform/evaluation/report.py:442-457 — FROZEN rule
# (already exactly what D-02 wants; the fix is WHERE/WHEN this gets called, not
# its own logic)
def _reference_label_columns(
    dev_features: pd.DataFrame, lean_cols: list[str], first_decision: pd.Timestamp
) -> list[str]:
    decision_slice = dev_features.loc[dev_features.index >= first_decision]
    return [c for c in lean_cols if bool(decision_slice[c].notna().all())]
```

```python
# src/trading_crab_lib/platform/honesty/walkforward.py:38-49 — where first_decision
# actually comes from (verified this session — no dependency on run_backtest's output)
def expanding_steps(index: pd.Index, *, min_train: int, step: int = 1):
    for i in range(min_train, len(index), step):
        yield index[i], index[:i], index[i : i + 1]
    # -> the FIRST t is always index[min_train]
```

### The A13_CAVEAT string that must be revised under D-08 (verbatim, current text)

```python
# src/trading_crab_lib/platform/plotting/core.py:74-81
A13_CAVEAT: str = (
    "NOT INTERPRETABLE (audit item A13): this ratio compares a fixed-feature "
    "'smoothed reference' labeling against the walk-forward's own per-window "
    "'filtered' labeling, whose active feature set changes 7 times across the "
    "backtest (4 -> 6 -> 8 -> 9 -> 10 -> 12 -> 13 features). Their disagreement "
    "is therefore not purely detection delay, and no plausibility band around "
    "this number resolves that until A13 is settled."
)
```

### Existing equivalence-adjacent test pattern to extend (criterion 1's foundation)

```python
# tests/unit/test_platform_evaluation_report.py:334-366 — ALREADY TESTS
# _reference_label_columns in isolation. Criterion 1's new test should assert
# that driver._refit_l1's active column list, at every decision date, equals
# this same function's output — not write a second independent assertion.
class TestReferenceLabelColumns:
    def test_drops_late_start_keeps_warmup_and_complete(self):
        idx = pd.date_range("1962-01-31", periods=240, freq="ME")
        first_decision = idx[120]  # ~1972, like min_train=120
        ...
```

### Registry row-count reality (verified this session, informs Pitfall 3 / D-17)

```
$ python3 -c "import json; print(sum(1 for _ in open('registry/trials.jsonl')))"
34
```
Last 3 pairs' `config` field is identical `{"phase": "05-backtest", "use_regime_tilt": True/False,
"min_train": 120, "cost_bps": 10}` for each pair — confirming the registry does not dedupe
identical configs; every call appends fresh rows.

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|---------------|--------|
| Driver expands feature admission per-window (`feature_min_history`) | Driver's L1 labeler consumes a frozen, pre-computed column list | This phase (wave 1) | Removes the A13 policy disagreement; §5.4 ratio becomes interpretable |
| `A13_CAVEAT` describes a 7-step feature-count progression as an active, unresolved defect | Caveat text describes the RESOLVED policy and points to the equivalence test + resolved-transition count | This phase (wave 1, D-08) | Caveat text becomes historically accurate rather than perpetually "not interpretable" |
| Two independent `_reference_label_columns`-shaped computations (implicit — the driver never called this function, it had its own logic) | One shared call site | This phase (wave 1) | Structural elimination of the divergence class A13 exemplifies |

**Deprecated/outdated:** the current `A13_CAVEAT` string's literal claim ("changes 7 times...")
becomes false, not merely outdated, once the driver is frozen — it must be edited, not left as
stale-but-harmless prose (see Pitfall 5).

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | The ADR's physical location (`platform_design/adr/` vs `.planning/` vs the phase dir) — left as Claude's Discretion by `07-CONTEXT.md`; this research does not resolve it further | Recommended Project Structure | Low — purely organizational, easily moved |
| A2 | Rebuilding `monthly_features` (Pitfall 1, option a) would promote `oil` from the excluded set into the frozen set, making it 10 columns not 9 — inferred from `oil`'s zero-NaN 1962-2020 span in `monthly_raw` and `compute_lean_features`'s unwindowed passthrough, but NOT verified by actually running a rebuild this session (macrotrends/yfinance reachability from this environment was not independently confirmed) | Common Pitfall 1 | Medium — if the rebuild fails or macrotrends is unreachable (per `STATE.md`'s documented wiring-only status), the planner may need to fall back to option (b) (pin to the current stale-checkpoint 9) regardless of preference |
| A3 | Freezing only `_refit_l1`'s admission (leaving `_refit_l2`/`_cv_safe_active_features` untouched) fully resolves A13 without any L2-side side effect — based on tracing that `_cv_safe_active_features` is called only from `_refit_l2` and operates on the nowcaster's training matrix, a distinct concern from the labeler's feature set | Architecture Patterns Pattern 1, Claude's Discretion confirmation | Low-Medium — if L2's calibration quality is implicitly sensitive to which L1 labels it trains against (rather than which columns it itself sees), a knock-on effect on Brier/calibration numbers beyond the documented y_true-reindex mechanism (D-05) is possible and should be watched for in the pre/post table |
| A4 | The empirical ~2-minute full-run timing estimate (129s extrapolated from a 15-sample average of every 40th walk-forward step) generalizes to the ACTUAL full 588-step run including the ablation leg, baseline computation, and report assembly overhead not captured in the sampled L1+L2-only timing | Summary, Common Pitfall 6 | Low — even a 5-10x underestimate still puts a full run at well under 30 minutes, which does not change the "cheap, run it twice" recommendation |

**If this table is empty:** N/A — see rows above.

## Open Questions (ALL RESOLVED — see resolution notes below, added 2026-09-14 at plan time)

> All three were resolved before planning completed. The original text of each is kept intact
> below so the reasoning that led to the resolution stays auditable; each carries a
> **RESOLVED** note naming where the answer now lives.

1. **Does the frozen set become 9 or 10 columns?**
   - ✅ **RESOLVED — it is TEN.** Answered by the user at plan time and recorded as
     `07-CONTEXT.md` **D-02-A**, which supersedes D-02's "9". Verified three independent ways
     before the decision: `monthly_raw.oil` holds 776 non-NaN from 1962-01 while
     `monthly_features.oil` holds 431 from 1985-02; `compute_lean_features` assigns
     `features["oil"] = monthly_raw["oil"]` (`transforms_monthly.py:253`, an unwindowed
     passthrough) so the checkpoint cannot derive from the current raw; and running
     `report.py::_reference_label_columns` both ways returns 9 as-is and 10 rebuilt, with
     `oil` the sole addition (+277 months, no other column changed).
   - The feasibility sub-question is also resolved: **no network is needed.** The recompute is a
     pure function of the cached `monthly_raw`, so the macrotrends/yfinance reachability concern
     raised below does not apply — it would only apply to a full `build_monthly_spine()`
     re-ingest, which `07-02-PLAN.md` explicitly forbids.
   - Implemented by `07-02-PLAN.md` Task 2; documented in the ADR by `07-04-PLAN.md` Task 3.

   - What we know: the current on-disk `monthly_features` checkpoint yields 9 (verified this
     session); a rebuild would very likely yield 10 by promoting `oil` (verified `oil`'s
     underlying `monthly_raw` coverage is complete 1962-2020, and `compute_lean_features` is an
     unwindowed passthrough).
   - What's unclear: whether a rebuild is feasible/desirable within wave 1's scope, and whether
     macrotrends/yfinance (also touched by any full `build_monthly_spine()` rebuild) are
     reachable from the execution environment (FRED alone was confirmed reachable this session;
     the others were not independently tested).
   - Recommendation: the planner should make this an explicit Wave-0/first-task decision point —
     either commit to a rebuild-then-freeze sequence, or explicitly document (in the ADR) that
     the 9-column freeze is pinned to the currently-committed checkpoint as a conscious, dated
     choice, with the staleness noted as a known limitation rather than silently inherited.

2. **What is the true current trial-registry count and ceiling at ADR-writing time?**
   - What we know: 34 as of this research session (2026-09-14), growing from re-runs unrelated
     to this phase; each full evaluation run contributes 2 rows, not 1.
   - What's unclear: exactly how many rows wave 1 itself will add depends on how many full
     re-runs the plan schedules (at minimum 2: the 9-feature policy and the D-03
     13-feature+imputation trial = 4 rows; possibly a third re-run if Pitfall 1's rebuild
     changes the frozen set and a 10-feature variant must ALSO be evaluated = 6 rows).
   - Recommendation: state the ceiling as a formula (`2 × N_policy_variants_evaluated`) in the
     ADR rather than a fixed number, and have the executing task call `registry.read_trials()`
     fresh immediately before and after to record the actual before/after count.
   - ✅ **RESOLVED — the recommendation was adopted.** The ADR records the ceiling as the
     formula `2 × N_full_evaluation_runs`, not a fixed number, and the executing tasks call
     `read_trials()` live rather than trusting the 34 measured here. See `07-01-PLAN.md`
     (registry `trial_tag` attribution) and `07-04-PLAN.md` Task 3 (ADR trial-ceiling section).
     This matters because deflated Sharpe is computed over the whole registry (D-16), so a
     stale hard-coded count would understate the trial burden.

3. **Is there any knock-on effect on L2's `_cv_safe_active_features` degrade pattern from
   freezing L1's admission?**
   - What we know: the two functions are structurally independent (different callers, different
     purposes) per this session's trace.
   - What's unclear: whether the FREQUENCY of L2 degrades (currently ~112/588 steps hold
     previous weights per `BASELINE-v1-tracer-bullet.md`'s "19% of steps degraded" caveat) shifts
     at all once L1's labels come from a stable feature space instead of a shifting one — L2
     trains on L1's OUTPUT labels (`build_nowcaster_training_set(train_features, train_states,
     ...)`, `driver.py:235`), so a more stable L1 label sequence could plausibly change L2's
     class-balance-per-window profile even though L2's own feature admission is untouched.
   - Recommendation: report this as an observed side effect (if any) in the pre/post table
     rather than assuming a priori that "L2 untouched" means "L2 numbers unaffected" — the
     y_true-reindex mechanism (D-05) already establishes that L1-side changes ripple into
     L2-adjacent metrics (Brier, confusion) without any L2 code changing.
   - ✅ **RESOLVED — the recommendation was adopted.** `07-03-PLAN.md` measures and records the
     L2 degrade frequency alongside the L1 numbers, and `07-04-PLAN.md`'s three-state pre/post
     table reports any shift as an observed side effect rather than asserting in advance that
     L2 is unaffected. Nothing is assumed a priori in either direction.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| FRED API (via `fredapi`) | Any rebuild of `monthly_raw`/`monthly_features` (Pitfall 1, option a); wave-2 M2/credit ingestion | ✓ — **[VERIFIED: live `fredapi.Fred.get_series("GS10", ...)` call this session returned 3 real rows]** | `FRED_API_KEY` present, 32-char key, functional | — |
| Direct HTTPS to `api.stlouisfed.org` via raw `curl` | N/A (not used by any code path) | ✗ observed — a raw `curl` to the FRED REST endpoint timed out in this session | — | Not a real blocker: the `fredapi` client (verified above) is the code path actually used, and it succeeded — the raw-curl failure likely reflects a proxy/routing difference for that specific tool, not the FRED API itself |
| macrotrends.net reachability | Rebuild path for `gold`/`oil` splices in `monthly_raw` (if a rebuild is chosen) | Not independently re-tested this session | — | `STATE.md` records this source as wiring-verified only, historically blocked by bot-detection from this class of environment (see STATE.md macrotrends notes) — treat a rebuild's gold/oil coverage as at-risk and confirm before depending on it |
| yfinance reachability | ETF/equity price ingestion within a rebuild | Not independently re-tested this session | — | Historically functional per STATE.md; not re-verified this session — low risk but unconfirmed |
| No new external packages | Wave 1 entirely; wave 2's `sklearn`/`scipy` dependence stats | ✓ — **[VERIFIED: `sklearn.metrics.adjusted_rand_score`, `normalized_mutual_info_score`, `scipy.stats.contingency.association` all imported successfully this session]** | sklearn 1.4+, scipy 1.17.1 | — |

**Missing dependencies with no fallback:** none for wave 1 — it is pure in-repo computation on
already-fetched checkpoints, no network required at all unless Pitfall 1's rebuild path is
chosen.

**Missing dependencies with fallback:** macrotrends/yfinance reachability, IF a rebuild is
chosen — fallback is to pin the frozen set to the current (stale) checkpoint instead (Pitfall 1,
option b), which requires zero network access.

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest 8.0+ |
| Config file | root `pyproject.toml` → `[tool.pytest.ini_options]` |
| Quick run command | `pytest tests/unit/test_platform_backtest_driver.py tests/unit/test_platform_evaluation_report.py tests/unit/test_platform_evaluation_sojourn_lag.py -x` |
| Full suite command | `pytest tests/ -q` |
| New dependencies | none |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| REG-01 (criterion 1) | Driver and report resolve to identical column sets at every sampled decision date | unit (equivalence) | `pytest tests/unit/test_platform_backtest_driver.py -k equivalence -x` | ❌ Wave 0 — new test |
| REG-01 (criterion 2) | §5.4 ratio computed on ONE feature space, shown with resolved-transition count | unit | `pytest tests/unit/test_platform_evaluation_sojourn_lag.py -x` | ✅ existing file — extend, `n_resolved`/`n_transitions` fields already present per `sojourn_lag.py:185-195` |
| REG-01 (criterion 3) | Post-fix disagreement % computed identically to how the 82.8% baseline was computed | unit + integration (real re-run) | `pytest tests/unit/test_platform_evaluation_report.py -x` + a real `run_full_backtest_evaluation()` invocation | Partial — the comparison LOGIC (`build_filtered_probs_matrix` vs `full_sample_states`) exists; a new test asserting the SAME comparison methodology (same date alignment, same counting rule) as whatever originally produced 389/470 must be added — see Pitfall/Open-Question note below |
| REG-01 (criterion 4) | Both `wealth_delta` and `dd_delta` re-measured, each within its 06-VALIDATION.md band | integration (real re-run) | full `run_full_backtest_evaluation()` re-run against real checkpoints | Manual-only real-data verification, mirroring Phase 5's own final plan (05-07) |
| REG-01 (ADR) | Policy choice + rejected alternatives + trial ceiling formula recorded | manual (document review) | N/A | N/A |

**Note on criterion 3's test gap:** this research did not locate the exact code that originally
produced "389/470 = 82.8%" (it is not `compute_sojourn_lag_headline`'s own output — that
function returns sojourn/lag/ratio/transition counts, not a raw month-by-month agreement
percentage). The planner should treat "reproduce the 389/470 methodology exactly" as a Wave-0
research/discovery task within the plan itself if the originating computation cannot be found in
`evaluation/`, `labeling/`, or a Phase 6 notebook cell — it may live in a Phase 6 notebook cell
(`P3_regime_labeling` or `P6_backtest_evaluation`) rather than in library code, since Phase 6's
scope was only to DISPLAY A13, not resolve it. **[ASSUMED — not verified this session; flagged
as an open question above (see risk table) is close but this is the more concrete, test-blocking
version of it.]**

### Sampling Rate

- **Per task commit:** quick command above (~10-20s, no real-data dependency for the unit
  portions)
- **Per wave merge:** full suite (`pytest tests/ -q`)
- **Phase gate:** at least one real `run_full_backtest_evaluation()` run against live checkpoints
  per policy variant (2 minimum: 9-feature, 13-feature+imputation; 3 if Pitfall 1's rebuild
  changes the count to 10) — mirrors Phase 5's own human-verify final plan pattern

### Wave 0 Gaps

- [ ] `tests/unit/test_platform_backtest_driver.py` — new equivalence test: `_refit_l1`'s active
      column list (with `frozen_features` threading in place) equals
      `_reference_label_columns`'s output, asserted at a sampled few decision dates (Claude's
      Discretion: parametrize vs. sample — either satisfies criterion 1 as literally worded)
- [ ] Locate (or write, if genuinely absent from library code) the exact function/notebook cell
      that produces the "389/470 = 82.8%" disagreement percentage, so the post-fix number is
      computed by the SAME methodology and is genuinely comparable (criterion 3's literal
      requirement)
- [ ] Framework install: none required

## Plausibility Bands — extending `06-VALIDATION.md`'s contract to wave 1's numbers

Every band `06-VALIDATION.md` already established (terminal log wealth, max drawdown, Brier,
turnover, CVaR, occupancy, portfolio weights) is UNCHANGED and reused as-is — wave 1 does not
introduce a new metric TYPE for those. The genuinely new quantities wave 1 re-measures or
introduces are below. Existing bands quoted are **[CITED: .planning/phases/06-platform-notebook-suite/06-VALIDATION.md]**;
new bands proposed for wave 1's new/changed quantities are **[ASSUMED — proposed this session,
no prior project precedent; the planner/discuss-phase should confirm before treating as locked]**.

| Quantity | Universal bound | Domain band | Source / reasoning |
|---|---|---|---|
| Terminal log wealth (588-mo span), strategy & ablation legs | `abs(x) < 10.0` | `[-3, 12]` per leg | [CITED: 06-VALIDATION.md] — unchanged, reused for both pre-fix and post-fix columns of the D-05 table |
| Max drawdown, any leg | `x ∈ [-1, 0]` | buy-and-hold `< -0.30`; blended 60/40 `< -0.10`; trend-following `> -0.50` | [CITED: 06-VALIDATION.md] — unchanged |
| Multiclass Brier, K=5 | `x ∈ [0, 1]` | no-skill floor `(K-1)/K² = 0.16` | [CITED: 06-VALIDATION.md] — unchanged; D-05 mechanically expects this to move because `y_true` changes, NOT the nowcaster — the pre/post table must caption this explicitly, per the CONTEXT.md mechanical fact |
| Regime occupancy | each `∈[0,1]`, `Σ=1.0` | soft warning `<0.05` (`_MIN_OCCUPANCY_THRESHOLD`, `labeling/diagnostics.py:67`) | [CITED: 06-VALIDATION.md] — unchanged |
| Post-fix disagreement % (criterion 3) | `x ∈ [0, 1]` (it is a fraction of compared months) | **suspicious-if-near-zero band: `x < 0.02` should trigger a "confirm the two labelings are not accidentally sharing state" check** | [ASSUMED] — the two labelings remain a hindsight full-sample fit vs. a walk-forward per-step fit even under one feature space; genuine detection lag means SOME disagreement is expected even after the fix. A near-0% post-fix number is not automatically good — it would be as suspicious in its own way as the pre-fix 82.8% was bad, and could indicate an implementation bug (e.g. the driver accidentally seeing full-sample data) |
| §5.4 ratio (post-fix) | `ratio > 0` (already established, `06-VALIDATION.md`: `∈[0, 588]` for lag/sojourn individually) | **now interpretable per D-08 once both artifacts land — no numeric target is set (D-04 forbids using it to select the policy)** | [CITED: 06-VALIDATION.md] for the individual lag/sojourn bounds; [ASSUMED] that the "not interpretable" caveat itself becomes removable, contingent on the two D-08 artifacts, not on the ratio's value |
| `n_transitions` (resolved-transition count, criterion 2) | `n_transitions >= 0`, integer, `n_resolved <= n_transitions` | domain band: `n_transitions` implausible if `> 30` over a 588-month/49-year span at K=5, λ=52 (a jump-model penalty this large should produce single-digit-to-low-double-digit transitions, consistent with the historically observed 4-6) | [ASSUMED] — proposed this session by analogy to the historically observed transition counts (4, 5, 6 across the project's recorded runs); flagging an implausibly HIGH transition count as a regression signal (over-segmentation / λ effectively not being applied) is the wave-1 analog of `06-VALIDATION.md`'s "catch a suspiciously perfect value" philosophy, not just "catch an impossible one" |
| `wealth_delta` (ablation, criterion 4) | derivable from the two terminal-log-wealth bounds: `abs(wealth_delta) < 15` | domain band: `abs(wealth_delta) < 5` — the project's own recorded history shows deltas of +0.073 to +0.379 (a 5x range); a value outside `[-5, 5]` on a re-measurement of the SAME strategy under a feature-policy change (not a new model) would be surprising enough to warrant a second look before publishing | [ASSUMED] — proposed this session; the universal bound is arithmetic (both terms individually bounded to `[-3,12]`), the domain band is proposed by analogy to observed history |
| `dd_delta` (ablation, criterion 4) | `x ∈ [-2, 2]` (both terms individually bounded to `[-1,0]`) | domain band: `abs(dd_delta) < 0.5` — both strategy and ablation are vol-targeted, long-only, similarly-levered books; a drawdown delta beyond half a full drawdown range between two variants of the SAME allocation code path (`no_regime_ablation` reuses the L4 tilt code, D-02 Phase 5) would be surprising | [ASSUMED] — proposed this session; current recorded value is −0.014364, well within this proposed band — this band would NOT have caught the historical 111.06-wealth-shaped failure class on its own, but that class is already caught by the terminal-log-wealth universal bound above (defense in depth, per `06-VALIDATION.md`'s own "two check classes catch disjoint failure modes" framing) |
| Trial registry row count, per full evaluation run | exactly `2` (strategy + ablation) — **[VERIFIED: registry/trials.jsonl inspected this session; `run_backtest`/`no_regime_ablation` call chain traced in `report.py`/`baselines.py`/`driver.py`]** | a run that appends `!= 2` rows to the registry indicates either a partial failure (crashed before the ablation leg) or a code change to the append pattern that the ADR's trial-ceiling formula must account for | [VERIFIED: this session, both by reading the ledger and by tracing the call chain] |

**Existing plausibility patterns to extend, not reinvent:** the two check CLASSES from
`06-VALIDATION.md` — universal physical-possibility bounds, and domain-informed per-leg bounds —
both apply directly; this phase adds a THIRD class the previous phase did not need: a
**"suspiciously-close-to-zero-or-perfect" band** for the disagreement percentage and the
transition count, because wave 1's headline risk is not "an impossible number" but "a number
that LOOKS resolved but is actually a different kind of bug" (e.g., the driver and reference
accidentally converging by both reading the same in-memory object rather than by the intended
mechanism) — directly the same failure class `06-VALIDATION.md` itself names but does not yet
have a worked example for: *"a valid-looking number computed against a mismatched target... no
numeric band around that ratio resolves A13, and no plan task may claim otherwise"* (quoted
verbatim from `06-VALIDATION.md`'s own framing, **[CITED: 06-VALIDATION.md]**). The bands above
are a first attempt at closing that gap for wave 1's specific numbers, not a claim that they
fully close it — a plausibility band cannot substitute for the equivalence TEST (criterion 1)
itself.

## Security Domain

`security_enforcement: true` in `.planning/config.json`. This phase makes no changes to
authentication, session management, access control, or any network-facing surface — it is an
internal restructuring of two Python functions in a batch-evaluation pipeline with no new
inputs from untrusted sources (all data is either already-ingested FRED/checkpoint data, or
config-file-driven feature-column lists). Most ASVS categories genuinely do not apply.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-------------------|
| V2 Authentication | No | No auth surface touched |
| V3 Session Management | No | No session surface touched |
| V4 Access Control | No | No access-control surface touched |
| V5 Input Validation | Marginal | The new `frozen_l1_features` / `frozen_cols` parameter threaded into `run_backtest` should be validated as a subset of `train_features.columns` (already the existing pattern: `active = [c for c in cols if int(...) >= min_history]` filters against the DataFrame's actual columns; the frozen variant should do the same defensive filtering rather than assume the frozen list and the DataFrame always agree, since the frozen list is now computed once and threaded through multiple call sites over time) |
| V6 Cryptography | No | Not applicable — no cryptographic operations anywhere in this phase |

### Known Threat Patterns for this stack

Not applicable in the traditional STRIDE sense (no network-facing endpoint, no user input, no
authentication). The closest analog this project's own honesty framework already treats as a
security-adjacent concern is **information leakage across the holdout boundary** (HON-01) —
wave 1 does not touch the holdout mechanism (`split_by_holdout_boundary` calls in `driver.py`/
`report.py` are unchanged), and the existing `assert_dev_checkpoint_within_boundary` /
grep-gated DSR-read guards remain the controls. No new threat surface is introduced.

## Sources

### Primary (HIGH confidence — read directly this session)

- `src/trading_crab_lib/platform/backtest/driver.py` (full file) — `_window_active_features`,
  `_cv_safe_active_features`, `_refit_l1`, `_refit_l2`, `run_backtest`
- `src/trading_crab_lib/platform/evaluation/report.py` (full file) — `_reference_label_columns`,
  `run_full_backtest_evaluation`, `assemble_backtest_report`
- `src/trading_crab_lib/platform/evaluation/sojourn_lag.py` (full file) —
  `compute_sojourn_lag_headline`, `build_filtered_probs_matrix`
- `src/trading_crab_lib/platform/taxonomy.py` (full file) — `lean_feature_set`
- `config/platform_settings.yaml` lines 232-420 — `taxonomy`, `holdout`, `registry`,
  `labeling`, `allocation`, `backtest`, `fred_monthly` sections
- `src/trading_crab_lib/platform/honesty/gating.py`, `registry.py`, `holdout.py`,
  `walkforward.py` (`expanding_steps`, lines 38-49) — full files
- `src/trading_crab_lib/platform/labeling/jump_model.py`, `labeling/diagnostics.py` (full files)
- `src/trading_crab_lib/platform/allocation/tilt.py`, `allocation/hysteresis.py` (full files)
- `src/trading_crab_lib/platform/transforms_monthly.py` lines 220-320 — `compute_lean_features`
- `src/trading_crab_lib/platform/plotting/core.py` lines 70-84 — `A13_CAVEAT`
- `tests/unit/test_platform_evaluation_report.py` lines 334-460 — `TestReferenceLabelColumns`,
  `TestSmoothedHindsightUniverse`
- `tests/unit/test_platform_plotting.py` lines 50-179 — `TestFreshPackageBoundary` (the
  import-guard test pattern for criterion 8)
- `data/checkpoints/platform/monthly_raw.parquet`, `monthly_features.parquet` — read directly
  this session via pandas to confirm the `oil`/`gold`/`fred_vix`/`curve_10y2y` common-support
  facts and diagnose the staleness finding
- `registry/trials.jsonl` — read directly this session (34 rows, last 6 inspected for the
  duplicate-config finding)
- `config/platform_settings.yaml` `fred_monthly.series` — confirmed no M2/credit entry exists
- Empirical timing measurement: `_refit_l1`/`_refit_l2` called directly this session on real
  checkpoint data, sampled every 40th walk-forward step, extrapolated to the full 588-step run
- Live `fredapi.Fred.get_series("GS10", ...)` call, this session, confirmed FRED access works
- `python3 -c "import sklearn.metrics; import scipy.stats.contingency"` this session, confirmed
  wave-2 dependence-statistic dependencies are already installed

### Secondary (MEDIUM confidence — planning documents, not independently re-verified against code)

- `.planning/phases/07-regime-representation/07-CONTEXT.md`, `07-DISCUSSION-LOG.md` (locked
  decisions, copied verbatim above)
- `.planning/PROPOSAL-phase-regime-representation.md`, `PROPOSAL-dual-regime-classifiers.md`
  (wave 2 direction)
- `.planning/ROADMAP.md` Phase 7 section (success criteria, non-goals)
- `.planning/REQUIREMENTS.md` (REG-01, INV-01 definitions)
- `.planning/BASELINE-v1-tracer-bullet.md` (pre-fix reference numbers for the D-05 table)
- `.planning/UAT-AUDIT-2026-09-09.md` (A13/A14/A15 diagnosis history)
- `.planning/phases/06-platform-notebook-suite/06-VALIDATION.md` (the plausibility-band
  contract this document extends)
- `.planning/STATE.md` (registry count history, R1/R4 tech-debt notes, macrotrends/yfinance
  reachability history)
- `.planning/POST-2020-OBSERVATIONS.md` (empty log — confirmed no post-2020 observations have
  changed a decision to date)

### Tertiary (LOW confidence)

- None — this research used only direct code reads, direct checkpoint/registry reads, and
  empirical measurements run this session; no web search or third-party documentation was
  needed (no new libraries, no new external services).

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — zero new dependencies, all verified importable this session
- Architecture: HIGH — the exact restructuring (Pattern 1) is grounded in reading both target
  functions and the `expanding_steps` dependency in full, plus a traced call graph confirming
  `_cv_safe_active_features` is L2-only
- Pitfalls: HIGH — every pitfall in this document was either directly observed this session
  (checkpoint staleness, registry count/row-per-run, canonicalization default) or is a direct
  logical consequence of code read this session (the `first_decision` circularity)
- Validation architecture bands: MEDIUM — the reused `06-VALIDATION.md` bands are HIGH
  confidence (cited directly); the NEW bands proposed for wave-1-specific quantities are
  explicitly [ASSUMED] and flagged for discuss-phase/planner confirmation before being treated
  as locked criteria

**Research date:** 2026-09-14
**Valid until:** short — this document's most load-bearing finding (the `monthly_features`
staleness, Pitfall 1) is itself time-sensitive: if the checkpoint is rebuilt or otherwise
changes before planning proceeds, several concrete numbers above (particularly "9 vs 10
columns," "34 registry rows") must be re-verified, not assumed to still hold. Re-run the direct
checkpoint/registry reads in this document's Sources section before treating any specific number
here as current if more than a few days elapse before execution.

## Project Constraints (from CLAUDE.md)

Two CLAUDE.md files govern this repo; the platform-specific one (`.claude/CLAUDE.md`) is
authoritative for every file this phase touches (all are under `src/trading_crab_lib/platform/`).
The root `CLAUDE.md` documents the separate, frozen legacy quarterly pipeline and its own ADRs —
not touched by this phase, and its conventions (e.g. `ADR #12`'s dual prediction API) do not
apply here.

**From `.claude/CLAUDE.md` (platform-specific, directly applicable):**
- `from __future__ import annotations` required at the top of every module touched — already
  present in every file this phase edits (`driver.py`, `report.py`, `sojourn_lag.py`, `core.py`
  plotting module), confirmed by reading their headers this session.
- `X | None`, not `Optional[X]` — the new `frozen_l1_features: list[str] | None = None`
  parameter sketch in this document already follows this convention.
- Type hints on all public functions; `log = logging.getLogger(__name__)` per module; no bare
  `except:`; specific exception types only — all unchanged by this phase's edits, and every
  function this phase modifies already follows these conventions (confirmed by direct read).
- Config sections read defensively via `cfg.get(...)`, never added to
  `_REQUIRED_PLATFORM_SECTIONS` — the `backtest`/`labeling`/`allocation` sections this phase
  reads are all already read this way; no change needed.
- **`platform/` imports nothing from the legacy library** (criterion 8's constraint) — wave 1
  adds no new imports of any kind; wave 2's relative-strength code must be PORTED from
  `src/trading_crab_lib/momentum.py`/`divergence.py`, never imported (confirmed those two files'
  function signatures this session: `compute_relative_strength`, `compute_rolling_correlation`,
  `compute_rolling_cross_correlation`, `compute_inflation_acceleration`,
  `compute_divergence_triggers`, `compute_derivative_divergence` — these are the six candidates
  to port, per `07-CONTEXT.md`'s canonical_refs).
- **GSD Workflow Enforcement** (root `CLAUDE.md`) — direct file edits must go through a GSD
  command (`/gsd-execute-phase` for this planned work); this research document does not itself
  edit any source file, consistent with that rule.

**From the root `CLAUDE.md` (legacy pipeline, NOT directly applicable but worth confirming
non-conflict):** the legacy `prediction/` dual-API convention (ADR #12), the quarterly pipeline's
checkpoint/RunConfig conventions, and the `market_code` naming rule are all scoped to
`src/trading_crab_lib/`'s NON-platform modules and are untouched by this phase. No conflict was
found between the two CLAUDE.md files for the specific modules this phase edits.
