# Phase 7 WAVE 2: Regime Representation (Leadership-Axis Classifier) — Research

**Researched:** 2026-09-15
**Domain:** unsupervised regime labeling (second jump-model instance), relative-strength feature
engineering ported from a legacy library, statistical dependence measurement between two
categorical labelings, walk-forward joint-allocation lift, deflated-Sharpe trial accounting.
**Confidence:** HIGH — every claim below is grounded in a file read this session, a live code
execution this session, or a live FRED API call this session. Where the codebase does not yet
contain something wave 2 needs (there is **no** deflated-Sharpe function anywhere in the repo —
verified by grep), that absence is stated as a finding, not glossed over.

**Scope of this document:** wave 2 only (ROADMAP criteria 5, 6, 7 + INV-01). Wave 1 (criteria
1-4) is CLOSED — verified, validated, UAT accept-with-caveats 2026-09-15 — and is not replanned
here. `07-RESEARCH-WAVE1.md` is preserved unmodified; its wave-2 material was explicitly
"direction only, MEDIUM confidence" and is superseded by this document wherever the two disagree
(the WAVE-2 OPENING AMENDMENT in `07-CONTEXT.md` is the authoritative diff).

<user_constraints>
## User Constraints (from CONTEXT.md)

### WAVE-2 OPENING AMENDMENT (2026-09-15) — read this before D-10…D-17 below

Wave 1's execution changed four premises D-10…D-17 were written against. None of the decisions
is withdrawn; each needs re-reading against the corrected facts.

1. **Criterion 8 is FALSE — `platform/` DOES import from the legacy library.** An AST scan finds
   **31 real legacy import sites**. All predate Phase 7; wave 1 added none. D-10/D-11's "port,
   don't import" still stands — but because the coupling must not be *widened*, not because
   `platform/` is already clean. Enforced by `tests/unit/test_platform_legacy_import_ratchet.py`,
   a ratchet pinned at **31** that may only decrease. **Wave 2 must not raise it.**

2. **D-16's registry premise moved — the ledger was reset.** Wave 1 appended 4 untagged rows
   from wiring-verification runs; the ledger was archived to
   `registry/archive/trials-pre-P7W1-reset.jsonl` (42 rows) and restarted with a **provenance
   header row** carrying `prior_genuine_trials=38`, `discarded_smoke_rows=4`. D-16's "whole
   registry since project start" now means **38 + rows appended after the header** — NOT the raw
   post-reset row count. `append_trial` now **refuses** untagged rows; `NO_REGISTRY` exempts
   smoke runs; the report CLI requires `--trial-tag` or `--smoke`.

3. **`canonicalize_states` conflicts with D-10 — a blocking landmine.** Its default sort key is
   `trailing_return_1m`, one of classifier #1's 13 columns. D-10 asserts disjointness on those
   13, so classifier #2 **always** hits the "fallback to centroid column 0" warning path unless
   wave 2 decides #2's own sort convention BEFORE #2 is fit.

4. **L2 degradation narrowed the comparison window, and wave 2 inherits it.** The frozen policy
   degraded 232/588 L2 steps (vs 118 pre-fix), so wave-1 measurements rest on 356 steps ending
   2017-05 against a 470-step baseline ending 2020-12. Not a wiring defect. Resolving it needs
   its own ADR — either make L2's CV robust to rare classes, or move §5.4 off L2's `proba` onto
   L1's own filtered labels (changes what §5.4 measures). **Decide this explicitly for
   criterion 7; do not quietly measure joint lift on a window wave 2 never acknowledges.**

5. **Four `[ASSUMED]` plausibility bands are now load-bearing.** `abs(wealth_delta) < 5`,
   `dd_delta ∈ [-0.5, 0.5]`, `n_transitions > 30` implausible, `pct_disagree < 0.02` suspicious.
   D-07 kept them advisory through wave 1. **Criterion 7 judges against them — confirm or revise
   all four before then.**

**Wave-1 reference numbers wave 2 compares against** (`07-MEASUREMENTS.md`): frozen set **10**
columns; §5.4 ratio **0.60145 (7/7 resolved)**; `pct_disagree` **0.80899 (288/356)**;
`wealth_delta` **+0.377847**; `dd_delta` **−0.066124**; strategy terminal log wealth
**4.025085**, max DD **−26.42%**; occupancy 11.51/9.06/35.54/31.51/12.37%. Strategy is still
**last of five legs**; Faber 6.3726 / −18.94% beats it on both §23.1 axes.

### Locked Decisions (D-10 through D-17 — wave 2's direction)

- **D-10: Disjointness is asserted on RAW columns; ratios derived from #1's columns are
  admissible.** The test asserts no raw column is shared with classifier #1's **13** (not the 9/10
  actually frozen — the released `fred_vix`/`gold`/`oil`/`curve_10y2y` stay out of #2, because
  admitting level/stress features into a leadership classifier invites exactly the criterion-6
  failure). A *ratio* is a genuinely different quantity — relative strength is scale-invariant
  where a level is not — so gold-in-equities and oil-in-equities remain admissible in principle.

- **D-11: Same freeze rule, same 1972+ decision window for classifier #2.** One documented policy
  across the platform, and the two labelings stay month-for-month comparable — required by
  criterion 6. Net effect of D-10 and D-11 combined: oil/equity relative strength **survives**
  (raw `oil` in `monthly_raw` runs 1962+); gold/equity **does not** (`gold` starts 1985-02 and
  fails the common-support freeze). The ratio rule opens the door; the freeze rule closes it for
  gold specifically.

- **D-12: INV-01 — ingest M2 and a credit aggregate; leave market-cap/GDP out.** None of INV-01's
  three named invariants has an ingested source. M2SL (1959+) and a credit aggregate (e.g.
  TOTALSL / BCNSDODNS) are free and reach back far enough. Market-cap/GDP stays blocked — no free
  1962+ market-cap source; FRED's Wilshire starts ~1970 — restated in the ADR, not quietly
  worked around.

- **D-13: Classifier #2's (K, λ) is set by construction — zero selection trials.** λ from the
  existing feature-count formula (λ = 4 × n_features) and K pre-declared. No sweep, nothing added
  to the DSR denominator.

- **D-14: Two separate probability inputs — no product state space.** Both labelings'
  probability vectors feed the allocation tilt as separate inputs. Allocation already consumes
  probabilities, not labels (audit item A7: `active_regime` gates nothing). A product space would
  also thin badly (K₁ × K₂ cells over ~590 decision months, rare cells below §4.4's ~8% floor (§4.4 crit. 1 is ~8%–~35%; see ADR-0001 § AMENDMENT 2026-09-17)).
  Keeping two inputs makes the lift-vs-#1-alone comparison a clean single-change ablation.
  — **Reversibility: costly.**

- **D-15: Dependence is reported with several statistics and NO pre-declared threshold.** Report
  adjusted Rand, Cramér's V and normalized mutual information side by side with the
  cross-tabulation. No pass/fail line. A threshold was offered and declined (no empirical basis
  in this project yet).

- **D-16: Deflated Sharpe uses the WHOLE registry since project start.** Every configuration ever
  evaluated on this data contributed to the selection process that produced today's model,
  whether or not this phase ran it. Counting only this phase's trials was offered and declined.
  **See the amendment above — the arithmetic changed.**

- **D-17: A trial ceiling is written into the ADR before running.** Expected count is roughly 5
  (2 wave-1 policy runs, 1 classifier-#2 fit, 1 joint, 1 #1-alone). Exceeding the ceiling requires
  an explicit amendment. **See the amendment above and Common Pitfall 8 below — the ceiling
  formula from wave 1's own ADR (`registry_rows_added = 2 × N_full_evaluation_runs`) is the
  correct unit to re-derive from, not a fixed number.**

### Claude's Discretion

- Whether L2's separate admission path (`_cv_safe_active_features`) is also frozen for
  classifier #2's own L2 (if wave 2 ever fits an L2 nowcaster on classifier #2's labels — not
  required by any of criteria 5-7, which only need L1 labeling + allocation lift).
- Where the ADR physically lives and its numbering (wave 1 used `platform_design/adr/0001-*`;
  wave 2's decision most naturally amends or extends it as a new ADR, e.g. `0002-l1-leadership-axis.md`).
- Module layout for the new relative-strength code inside `platform/` (a new
  `platform/features/relative.py`, or extending `transforms_monthly.py`).
- Which credit aggregate series to use for INV-01, and its agency-tier alignment treatment —
  **this research recommends TOTALSL over BCNSDODNS; see Priority 3 below and Common Pitfall 7.**
- Report and plot layout for the dependence cross-tabulation and joint-lift comparison, within
  the existing `platform/plotting/` conventions.
- The exact blend mechanism and blend weight for the two-probability-input joint tilt (D-14) —
  **this research recommends a fixed, pre-declared 50/50 blend of two independently-computed
  `regime_tilt_weights()` outputs; see Architecture Patterns Pattern 4.** No grid search over the
  blend weight — that would be an unregistered selection trial.

### Deferred Ideas (OUT OF SCOPE)

- No fitting to forward returns. No raising K on classifier #1 / no (K, λ) sweeps (v2, L1-V2-01).
- No 2021+ holdout use for any selection decision. No migration work (Phase 8).
- Market-cap/GDP as an invariant — blocked on a free 1962+ market-cap source (restated, not
  re-litigated, by D-12).
- gold/equity relative strength — inadmissible under D-11's freeze (gold starts 1985-02).
- A7 — making hysteresis actually gate the portfolio — not fixed by D-14; D-14 works with the
  system as it is.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| REG-01 (wave 2 portion) | Second, independent regime classifier fit unsupervised on relative/leadership features disjoint from classifier #1's, with orthogonality measured (not assumed) and joint allocation lift assessed walk-forward against classifier #1 alone, every configuration logged to the trial registry | Priority 1 (canonicalize_states fix) resolves the blocking landmine before any fit; Priority 2 fixes the method (reuse `fit_jump_model`/`canonicalize_states`, port 3 legacy functions for the feature set); Priority 4 gives the dependence-statistic combination; Priority 5/6 give the trial budget and the joint-tilt architecture gap; Architecture Patterns Pattern 4 gives the concrete allocation-layer design D-14 does not fully specify |
| INV-01 | Named, era-stable invariant candidates (M2/GDP, credit/GDP; market-cap/GDP excluded) constructed and screened, survivors admitted as named features never anonymous PCs | Priority 3 — live-verified FRED series (M2SL, TOTALSL, BCNSDODNS), frequency-mismatch finding for BCNSDODNS, and the GDP-alignment note (fred_gdp is already monthly-forward-filled, so M2/GDP needs no new interpolation work) |

</phase_requirements>

## Summary

Wave 2 is genuinely new algorithm-adjacent work, but less new than it looks: the labeling
*method* is not in question — D-13's own wording ("classifier #2 is a separate instance of this
same machinery") and the ADR's Deferrals section both already settle that classifier #2 reuses
`labeling/jump_model.py::fit_jump_model` / `canonicalize_states`, the identical pure-numpy DP
decode classifier #1 uses, refit with its own (K, λ) on a disjoint column list. The two things
that are genuinely new are (1) the **feature engineering** feeding that fit — six candidate
functions ported (not imported) from the legacy `trading_crab_lib.momentum`/`.diagnostics`
modules, renamed onto the platform's monthly column names and re-derived for monthly (not
quarterly) cadence — and (2) the **allocation-layer wiring** for D-14's "two separate probability
inputs," which this research finds is **not** already supported by `allocation/tilt.py`'s
existing functions and needs a small, genuinely new blending function.

Before any of that, one blocking landmine must be resolved: `canonicalize_states`
(`labeling/jump_model.py:190-225`) hardcodes its canonical-ordering sort key to
`trailing_return_1m`, one of classifier #1's frozen 13 columns. Because D-10 requires
classifier #2's raw feature set to be disjoint from those 13, **every single fit of
classifier #2 will hit the "fallback to centroid column 0" warning path** unless the function's
signature is extended with an explicit, caller-supplied `sort_column` parameter before wave 2
fits anything. This research verified (by grep) that **no existing test currently exercises or
pins that fallback path** — so the fix is a safe, additive signature change (`sort_column:
str = "trailing_return_1m"`, defaulting to classifier #1's exact current behavior at every one
of its four call sites), not a breaking change. The fallback's silent WARNING-and-continue
behavior should also be hardened to a raised `ValueError` — since #1 will never hit it (its
frozen list always contains `trailing_return_1m`) and #2 must never hit it either, once #2 is
required to pass its own `sort_column` explicitly.

Two further findings materially change how the planner should scope criterion 7. First, **no
deflated-Sharpe function exists anywhere in this codebase** (verified by grep across
`src/` and `tests/`) — design §22 and the D13/D16 decisions describe *what* it must correct for
(selection bias under multiple testing, non-normality) but the platform has never implemented
Bailey & López de Prado (2014)'s formula. Wave 2 must write it from scratch — a small,
well-specified numpy/scipy computation, not a new dependency (skfolio has a PR adding this, but
pulling in a full portfolio-optimization library for a ~45-row registry is disproportionate).
Second, `allocation/tilt.py::vol_targeted_tilt`/`regime_tilt_weights` accept exactly **one**
probability vector today; D-14's "two separate probability inputs" is a real, unimplemented
allocation-layer interface, not existing machinery classifier #2 merely "instances." This
research recommends the smallest change that honors D-14 literally: compute two independent
`regime_tilt_weights()` outputs (one per classifier, each against its own
`returns_by_regime_stats` table) and blend them with a fixed, pre-declared weight — never a
product state space, never a search over the blend weight.

**Primary recommendation:** fix `canonicalize_states` first (Priority 1), in the same commit
that adds the disjointness test — nothing about classifier #2's fit is trustworthy until that
landmine is closed. Then port the three feature functions this phase actually needs (equity/bond
relative strength + rolling correlation, oil/equity relative strength, CPI acceleration) into a
new `platform/features/relative.py`, reusing `_reference_label_columns` unmodified to freeze
classifier #2's own common-support set at the 1972+ window (D-11). Implement the dependence
report as an extension of the existing `disagreement.py`/`label_disagreement` cross-tabulation
pattern, not a new implementation. Write the deflated-Sharpe function against the registry's
`prior_genuine_trials` provenance header, not the raw post-reset row count. Decide the L2-window
question (Common Pitfall 3) explicitly, in its own ADR section, before criterion 7 is judged —
and confirm the four now-load-bearing `[ASSUMED]` bands before leaning on them.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| `canonicalize_states` sort-column parameterization | L1 (labeling) | — | Shared machinery both classifiers call; the fix must not change #1's behavior at any of its 4 existing call sites |
| Relative-strength / leadership feature engineering (ported) | L0 (data) | L1 (labeling input) | New derived columns feed classifier #2's fit; the port work itself is a pure data transform |
| Classifier #2 fit (`fit_jump_model` + `canonicalize_states`, own K/λ) | L1 (labeling) | — | Second instance of existing machinery — no new clustering algorithm |
| INV-01 ingestion (M2SL, credit aggregate) | L0 (data) | — | New FRED series into `monthly_raw`; agency-tier alignment question for the credit aggregate (Common Pitfall 7) |
| Disjointness assertion (criterion 5) | Evaluation/Honesty | L1 | A test over `taxonomy.lean_feature_set(cfg)` vs. classifier #2's raw column list |
| Occupancy / 5%-floor reporting for classifier #2 | L1 (labeling) | Evaluation/Honesty | Reuses `labeling/diagnostics.py::occupancy_and_sojourns` / `report_labeling_diagnostics` unmodified — same report-only, never-gate posture (D-02) |
| Dependence measurement (criterion 6) | Evaluation/Honesty | — | Extends `plotting/regime.py::label_disagreement`'s cross-tabulation with `sklearn`/`scipy` statistics |
| Joint-tilt blending (criterion 7, D-14) | L4 (allocation) | — | **Genuinely new code** — `vol_targeted_tilt`/`regime_tilt_weights` today accept one probability input only (verified this session); needs a new blending function, not a parameter tweak |
| Joint walk-forward re-run (criterion 7) | L4/Backtest | Evaluation/Honesty | Either two `run_backtest()` calls (one per classifier) feeding the new blend function, or a `driver.py` extension running both L1 fits in the same per-step loop — see Architecture Patterns Pattern 4 for the tradeoff |
| Deflated-Sharpe computation (criterion 7) | Evaluation/Honesty | — | **Does not exist yet anywhere in the codebase** (verified by grep) — new, small, well-specified function reading `registry.read_trials()`'s provenance header |
| L2-degradation-window decision (amendment item 4) | Evaluation/Honesty | L2 (prediction) | A design decision (ADR), not a code capability by itself — gates how criterion 7's comparison window is read |

## Standard Stack

### Core

No new external packages for wave 2's labeling/dependence work. All libraries used are already
installed, already used elsewhere in `platform/`, and were re-verified importable this session:

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| numpy | already installed | `canonicalize_states` sort-column change; ported feature math | Unchanged |
| pandas | already installed (2.0+) | Ported feature functions operate on DataFrames, same as `momentum.py`'s originals | Unchanged |
| scikit-learn | 1.9.1, live-verified this session | `sklearn.metrics.adjusted_rand_score`, `normalized_mutual_info_score` (criterion 6) | Already used for the jump model's `KMeans`/`StandardScaler` warm start |
| scipy | 1.17.1, live-verified this session | `scipy.stats.contingency.association` (Cramér's V, criterion 6); `scipy.stats.norm` (deflated-Sharpe quantiles, criterion 7) | Already an installed dependency |
| fredapi | already installed, live-verified this session (`M2SL`/`TOTALSL`/`BCNSDODNS`/`TOTBKCR` all fetched successfully) | INV-01's M2/credit ingestion | Same client `ingestion/macro_monthly.py::fetch_fred_monthly` already uses |

**No new packages are needed anywhere in wave 2.** A candidate was considered and rejected: the
`skfolio` library has an open PR (`skfolio/skfolio#255`) adding `deflated_sharpe_ratio` and
`expected_max_sharpe_ratio` [CITED: github.com/skfolio/skfolio/pull/255, found via WebSearch this
session] — **not adopted**, because (a) it is a PR, not a released feature, on a full
portfolio-optimization library with a much heavier dependency footprint (`cvxpy` and friends)
than a ~45-row trial registry justifies, and (b) the formula itself (Bailey & López de Prado
2014) is short enough (see Code Examples) to hand-implement in ~40 lines of numpy/scipy with a
docstring citing the paper — consistent with wave 1's own "zero new dependencies" posture.

**Installation:** none required.

## Package Legitimacy Audit

**Not applicable.** Wave 2 installs no new external packages. `scikit-learn`, `scipy`, and
`fredapi` are all pre-existing dependencies, live-verified importable/functional this session.

## Architecture Patterns

### System Architecture Diagram

```
config/platform_settings.yaml (taxonomy.fast ∪ .slow = 13 lean cols, classifier #1's space)
                    |
        _reference_label_columns(dev_features, lean_cols, first_decision)
                    |
             frozen_cols_1 (10, per D-02-A)  ──────────────────────────────┐
                    |                                                       |
   [NEW] platform/features/relative.py — ported momentum.py functions,     |
   renamed onto monthly_raw columns (equities_tr, long_duration_tr,        |
   oil, gold[excluded], fred_cpi) ── produces NEW column names,            |
   disjoint from the 13 by construction (D-10)                             |
                    |                                                       |
        _reference_label_columns(dev_features, candidate_cols_2,           |
          first_decision)   [SAME function, D-11's freeze rule]            |
                    |                                                       |
             frozen_cols_2 (K_2 raw ratio/relative columns)                 |
                    |                                                       |
   canonicalize_states(states, centroids, frozen_cols_2,                   |
      sort_column=<#2's own leading feature>)  <-- FIX REQUIRED FIRST       |
      [see Pattern 1 — the blocking landmine]                              |
                    |                                                       |
   fit_jump_model(X_2, K=K_2, lam=4*len(frozen_cols_2))  (D-13, by          |
      construction, zero selection trials)                                 |
                    |                                                       |
        classifier #2 labels (states_2)             classifier #1 labels (states_1,
                    |                                  produced identically via
                    |                                  _refit_l1/full-sample fit)
                    |                                                       |
        +-----------+---------------------------------+                    |
        |                                                                  |
        v                                                                  v
  measure_labeling_dependence(states_1, states_2)              returns_by_regime_stats(
  [NEW, extends label_disagreement's crosstab pattern —          asset_returns, states_1)
   adjusted Rand + Cramér's V + NMI, D-15, no threshold]         returns_by_regime_stats(
        |                                                          asset_returns, states_2)
        v                                                                  |
  dependence report (criterion 6) ── human reads it,             regime_tilt_weights(probs_1, ...)
  records "failure to add an axis" or not                        regime_tilt_weights(probs_2, ...)
                                                                            |
                                                          [NEW] blend_regime_tilts(tilt_1, tilt_2,
                                                            weight=0.5)  -- D-14, two SEPARATE
                                                            probability inputs, NO product space
                                                                            |
                                                                            v
                                                          vol_target_scale + portfolio_vol (REUSED
                                                            unmodified from allocation/tilt.py)
                                                                            |
                                                                            v
                                                          joint equity curve, walk-forward
                                                                            |
                                                                            v
                                            registry.append_trial(...) x N  -->  read_trials()
                                            reads provenance header's prior_genuine_trials=38
                                                                            |
                                                                            v
                                            [NEW] compute_deflated_sharpe(sharpe, n_trials,
                                              trial_sharpe_variance, skew, kurtosis, n_obs)
                                              -- does not exist anywhere in the repo today
```

A reader can trace the primary use case (does adding classifier #2 help) from the taxonomy
config through both classifiers' independent fits, into the dependence report (a stop/no-stop
finding, criterion 6), and out through the joint tilt into a walk-forward equity curve and its
deflated-Sharpe-corrected trial-registry entry (criterion 7) — every box already exists except
the four marked `[NEW]`, and three of those four extend an existing pattern rather than
introducing a new one.

### Recommended Project Structure

```
src/trading_crab_lib/platform/
├── labeling/
│   └── jump_model.py            # canonicalize_states signature change (sort_column param)
├── features/
│   └── relative.py              # NEW — ported momentum.py functions, monthly-cadence, platform-
│                                 #   native column names, no legacy imports (ratchet-safe)
├── evaluation/
│   ├── disagreement.py          # extended (or a new sibling module) with
│   │                             #   measure_labeling_dependence()
│   └── deflated_sharpe.py       # NEW — Bailey–López de Prado DSR, reads registry's
│                                 #   provenance header, zero new dependencies
├── allocation/
│   └── joint_tilt.py            # NEW — blend_regime_tilts(), the D-14 two-input wiring
│                                 #   vol_targeted_tilt/regime_tilt_weights do not provide today
├── backtest/
│   └── driver.py                # extended: either a second _refit_l1-analog call site per
│                                 #   step, or a separate joint-evaluation orchestration function
│                                 #   (see Pattern 4 tradeoff)
tests/unit/
├── test_platform_labeling.py            # canonicalize_states sort_column tests (new)
├── test_platform_features_relative.py   # NEW — disjointness test (criterion 5), ported-function
│                                         #   parity tests against legacy behavior on synthetic data
├── test_platform_evaluation_dependence.py  # NEW — criterion 6
├── test_platform_evaluation_deflated_sharpe.py  # NEW — criterion 7, oracle test against a
│                                                 #   hand-computed small-N example
├── test_platform_allocation_joint_tilt.py  # NEW — criterion 7's blending function
├── test_platform_legacy_import_ratchet.py  # UNCHANGED count assertion (31) — a regression here
│                                            #   means the port leaked an import instead of copying
```

### Pattern 1: Parameterize `canonicalize_states`'s sort column — the blocking landmine, fixed

**What:** `labeling/jump_model.py:190-225` hardcodes the canonical-ordering sort key:

```python
# CURRENT — src/trading_crab_lib/platform/labeling/jump_model.py:214-221
if "trailing_return_1m" in feature_names:
    sort_col = feature_names.index("trailing_return_1m")
else:
    sort_col = 0
    log.warning(
        "trailing_return_1m not in feature_names — falling back to centroid "
        "column 0 for canonicalization sort order"
    )
```

This is meaningful for classifier #1 because `trailing_return_1m` IS classifier #1's own
defining axis — the labeler's states are fundamentally ordered along a return-level (bear→bull)
gradient, and sorting ascending on that column gives a canonical, economically interpretable
state numbering that is stable across restarts (the function's own docstring: "so state
numbering is stable across restarts and refreshes"). For a **leadership/relative-strength**
axis, there is no equivalent bear/bull polarity — state semantics are about **which asset class
is leading**, not about good/bad. The correct generalization is the same principle applied to
classifier #2's own defining feature: sort ascending on **classifier #2's primary/first-listed
relative-strength column** (this research recommends the equity/bond relative-strength ratio —
see Priority 3/Code Examples — because the proposal itself names the stock–bond relationship as
"the single most important missing variable"), so state 0 = bonds-leading regime, state K-1 =
equities-leading regime, exactly mirroring how state 0/state K-1 anchor #1's bear/bull axis.

**The fix — an additive, backward-compatible signature change:**

```python
def canonicalize_states(
    states: np.ndarray,
    centroids: np.ndarray,
    feature_names: list[str],
    *,
    sort_column: str = "trailing_return_1m",
) -> tuple[np.ndarray, np.ndarray]:
    if sort_column not in feature_names:
        raise ValueError(
            f"sort_column={sort_column!r} not in feature_names — a canonicalization "
            "sort column must be present in the fitted feature set. Pass the caller's "
            "own defining column explicitly (never rely on a silent fallback)."
        )
    sort_col = feature_names.index(sort_column)
    order = np.argsort(centroids[:, sort_col])
    remap = {old: new for new, old in enumerate(order)}
    new_states = np.array([remap[s] for s in states])
    return new_states, centroids[order]
```

**Why this is safe, verified this session:**
- All 4 existing call sites (`driver.py::_refit_l1` line 262, `evaluation/report.py` line 924,
  `labeling/diagnostics.py::label_regimes` line 286) never pass `sort_column` today, so the
  default (`"trailing_return_1m"`) reproduces current behavior byte-for-byte — and classifier #1
  ALWAYS has `trailing_return_1m` in its frozen 10-column set (D-02-A), so it can never hit the
  new `ValueError`.
- **No existing test pins the old warning-and-fallback behavior** — verified by grep across
  `tests/unit/test_platform_labeling.py`: the file's only `caplog`-based WARNING test
  (`TestReportDiagnosticsReportOnly::test_violation_warns_but_does_not_raise`) is about a
  *different* warning (the §4.4 occupancy floor in `report_labeling_diagnostics`), not
  `canonicalize_states`'s fallback. Hardening the fallback into a raised error changes zero
  passing tests.
- Classifier #2's own call site passes `sort_column=<its own leading feature>` explicitly,
  so it never touches the fallback path at all — the landmine is closed by construction, not by
  catching the warning after the fact.

**The test for criterion 5's disjointness AND this fix together:**

```python
def test_classifier2_disjoint_from_classifier1_raw_columns(cfg):
    lean_1 = lean_feature_set(cfg)                       # 13 columns
    candidates_2 = ["rs_equities_bonds", "corr_equities_bonds_12m", ...]
    assert set(candidates_2).isdisjoint(lean_1)

def test_canonicalize_states_leadership_axis_no_fallback(caplog):
    # classifier #2's own sort column is present -> the fallback is never hit
    states, centroids = canonicalize_states(
        raw_states, raw_centroids, candidates_2, sort_column="rs_equities_bonds",
    )
    assert not any("falling back to centroid column 0" in r.message for r in caplog.records)
    # oracle check, mirroring the DP-decode brute-force pattern (03-RESEARCH Pattern 1 precedent)
    expected_order = np.argsort(raw_centroids[:, candidates_2.index("rs_equities_bonds")])
    ...

def test_canonicalize_states_raises_on_missing_sort_column():
    with pytest.raises(ValueError, match="sort_column"):
        canonicalize_states(states, centroids, ["some_other_col"], sort_column="rs_equities_bonds")
```

### Pattern 2: Reuse `_reference_label_columns` unmodified for classifier #2's own freeze

**What:** D-11 requires the SAME freeze rule and 1972+ decision window classifier #1 uses.
`evaluation/report.py::_reference_label_columns` is already a generic function of
`(dev_features, candidate_cols, first_decision)` — nothing about its implementation is specific
to classifier #1's 13 columns. Calling it a second time with classifier #2's candidate ratio
column list produces classifier #2's own frozen set by the exact same mechanism wave 1 used to
close criterion 1 — a single shared call site, never a second parallel implementation.

**When to use:** any time a second labeling instance must honor the same common-support
discipline the first one does. This is the "Don't Hand-Roll" reuse from wave 1's research,
applying unchanged.

**Concrete effect on the candidate list (verified this session, live checkpoint reads):**
`equities_tr`, `long_duration_tr`, `cash`, `oil`, `fred_gs10`, `fred_tb3ms`, `fred_baa`,
`fred_aaa` are all **776/776 non-NaN from 1962-01-31** in `monthly_raw.parquet`
[VERIFIED: `data/checkpoints/platform/monthly_raw.parquet`, read directly this session]; `gold`
is 499/776 non-NaN, first valid **1985-02-28** [VERIFIED: same file, same session]. Any relative
strength or correlation column built purely from the full-history raw columns (equity/bond,
equity/oil, CPI acceleration) will trivially pass `_reference_label_columns`'s 1972+
common-support freeze with 100% of the decision window; anything touching `gold` will not — this
is D-11's stated net effect, now independently re-confirmed against the live checkpoint rather
than quoted from `07-CONTEXT.md`.

### Pattern 3: Extend `label_disagreement`'s cross-tabulation for criterion 6 — do not reimplement it

**What:** `platform/plotting/regime.py::label_disagreement(reference_states, comparison_states)`
already computes exactly the raw material criterion 6 needs: it aligns two label Series on their
common index and returns `per_state_confusion` — a `pd.crosstab(reference, comparison)` — plus
`n_compared`. `evaluation/disagreement.py::measure_label_disagreement` already shows the correct
pattern for wrapping it defensively (coercing `state_N`-string columns, guarding the
silent-zero-`n_compared` trap). Criterion 6's new function should follow the identical shape:

```python
# NEW — platform/evaluation/dependence.py (or extend disagreement.py)
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.stats.contingency import association

def measure_labeling_dependence(states_1: pd.Series, states_2: pd.Series) -> dict:
    """D-15: several statistics, no pre-declared threshold. Reuses
    label_disagreement's alignment/crosstab exactly — never a second
    alignment implementation."""
    from trading_crab_lib.platform.plotting.regime import label_disagreement
    base = label_disagreement(states_1, states_2)   # reuses the SAME crosstab machinery
    if base["n_compared"] == 0:
        return {**base, "adjusted_rand": float("nan"), "nmi": float("nan"), "cramers_v": float("nan")}
    common = states_1.index.intersection(states_2.index)
    a = states_1.loc[common].to_numpy()
    b = states_2.loc[common].to_numpy()
    return {
        **base,
        "adjusted_rand": adjusted_rand_score(a, b),
        "nmi": normalized_mutual_info_score(a, b),
        "cramers_v": float(association(pd.crosstab(a, b).to_numpy(), method="cramer")),
    }
```

Live-verified this session (synthetic 6-element example): `adjusted_rand_score` = 0.1667,
`normalized_mutual_info_score` = 0.5794, `association(..., method="cramer")` = 0.5 for a
constructed 2×2 contingency table — all three functions run correctly against real numpy inputs
in this environment [VERIFIED: local execution this session].

**Why these three, and what each one actually answers (Priority 4):**

| Statistic | Question it answers | What it misses |
|---|---|---|
| **Adjusted Rand Index (ARI)** | "Do the two labelings agree on which months are grouped together?" — pairwise agreement corrected for chance. Symmetric, `-1` to `1`, `0` = chance-level agreement. | Says nothing about the *information content* one labeling gives about the other — two labelings can have low ARI yet still be highly informative of each other if the state counts differ (K₁ ≠ K₂). |
| **Normalized Mutual Information (NMI)** | "How much does knowing #2's state reduce uncertainty about #1's state?" — a genuinely different question from ARI: shared information, not pairwise grouping agreement. Robust to different K. | Not chance-corrected the way ARI is (though `normalized_mutual_info_score` does apply a normalization by entropy, not by an expected-value-under-random-labeling correction the way `adjusted_mutual_info_score` does — `adjusted_mutual_info_score` is a defensible alternative/addition if NMI's own un-adjusted baseline seems too generous for the small K here). |
| **Cramér's V** | "How strong is the association, on a single unit-interval scale comparable across different-sized contingency tables?" — a chi-squared-based effect size, useful as the single "at a glance" number in a report. | Purely a strength-of-association number; it does not distinguish which direction the dependence runs, nor is it interpretable as "percent of variance explained" the way R² is. |

**This combination is deliberately not just "three flavors of the same thing":** ARI answers a
clustering-agreement question, NMI answers an information-theoretic question, Cramér's V answers
a classical-statistics association-strength question. Reporting all three (D-15) is what makes
"high dependence is a failure to add an axis" a legible, falsifiable finding rather than a single
number a reader has to take on faith — if all three point the same direction (all near their
respective "no relationship" values, or all near their "identical" values), the finding is robust
to which statistic a skeptical reader trusts most. **A directional statistic (Theil's U /
uncertainty coefficient) is NOT recommended as a fourth addition** — it answers "how well does #1
predict #2" vs. "how well does #2 predict #1" asymmetrically, which is a meaningful refinement
only if the report needs to argue one classifier is more informative than the other; D-15's
framing ("failure to add an axis") is a symmetric question, so the three symmetric statistics
above are sufficient and simpler to report side by side.

### Pattern 4: D-14's "two separate probability inputs" is NOT already-existing machinery — a small new blending function is required

**What goes wrong if the planner assumes otherwise:** `allocation/tilt.py::vol_targeted_tilt`
(and the `regime_tilt_weights` it calls) accepts exactly **one** `regime_or_probs` argument
[VERIFIED: `src/trading_crab_lib/platform/allocation/tilt.py`, read in full this session,
signature `vol_targeted_tilt(regime_or_probs, returns_by_regime, asset_returns, *, ...)`].
`regime_tilt_weights` blends per-regime tilts using ONE `returns_by_regime` table keyed by ONE
classifier's state ids. Nothing in `allocation/` today accepts two probability vectors and two
`returns_by_regime` tables simultaneously. D-14's decision record ("Both labelings' probability
vectors feed the allocation tilt as separate inputs") states the REQUIREMENT, not an
already-implemented mechanism — this is new allocation-layer code, however small.

**The recommended mechanism (smallest change that honors D-14 literally, avoids a product
state space, and needs zero new selection trials per D-13's spirit):**

```python
# NEW — platform/allocation/joint_tilt.py
def blend_regime_tilts(
    probs_1: pd.Series, returns_by_regime_1: pd.DataFrame,
    probs_2: pd.Series, returns_by_regime_2: pd.DataFrame,
    asset_returns: pd.DataFrame,
    *,
    weight_1: float = 0.5,     # fixed, pre-declared — NOT swept (D-13's spirit extended)
    target_vol_annual: float, halflife: float, min_obs: int,
) -> dict:
    """D-14: two SEPARATE probability-weighted tilts, blended at the WEIGHT
    level, never a product (state_1, state_2) cell. Each classifier's own
    returns_by_regime table only needs K cells, not K1*K2 — sidesteps the
    thinning D-14 explicitly rejected a product space to avoid."""
    from trading_crab_lib.platform.allocation.tilt import (
        regime_tilt_weights, portfolio_vol, vol_target_scale,
    )
    tilt_1 = regime_tilt_weights(probs_1.idxmax(), returns_by_regime_1, probs_1)
    tilt_2 = regime_tilt_weights(probs_2.idxmax(), returns_by_regime_2, probs_2)
    blended = (weight_1 * tilt_1).add((1 - weight_1) * tilt_2, fill_value=0.0)
    total = blended.sum()
    base_weights = blended if total <= 0 else blended / total
    if base_weights.empty or base_weights.sum() <= 0:
        return {"weights": pd.Series(dtype=float), "cash": 1.0, "scale": 0.0}
    port_vol = portfolio_vol(base_weights, asset_returns, halflife=halflife, min_obs=min_obs)
    scale = vol_target_scale(target_vol_annual, port_vol)
    return {"weights": base_weights * scale, "cash": 1.0 - scale, "scale": scale}
```

`portfolio_vol` and `vol_target_scale` ARE reused unmodified — only the pre-scaling weight-blend
step is new. **The blend weight (0.5 here) must be fixed and declared in the ADR before either
classifier's walk-forward is run**, exactly like D-13's (K, λ)-by-construction philosophy —
sweeping it would add an unregistered selection dimension the trial ceiling did not budget for.

**Wiring this into a walk-forward run — the two-run vs. one-loop tradeoff the planner must pick:**
1. **Two independent `run_backtest()` calls** (one per classifier, each with its own
   `frozen_l1_features`), each producing its own `per_step_metrics["proba"]` stream, combined
   post-hoc at each shared decision date via `blend_regime_tilts`. Simpler to implement (zero
   changes to `driver.py`'s per-step loop), but re-fits L1 and L2 for BOTH classifiers on every
   step of BOTH runs even though only the tilt step needs combining — doubles the walk-forward's
   wall-clock cost (each run was ~2 minutes per wave-1's own measurement, so ~4 minutes total,
   still cheap).
2. **Extend `driver.py`'s per-step loop** to fit both L1s (and optionally both L2s) inside one
   pass and call `blend_regime_tilts` in place of the single `vol_targeted_tilt` call — avoids
   redundant re-computation of shared inputs (train/test index slicing, cost/turnover
   accounting) but is a more invasive change to a load-bearing, heavily-tested function
   (`run_backtest` is exercised by dozens of existing tests per wave 1's own SUMMARY).

**Recommendation: option 1 (two independent runs, combined post-hoc).** It is a pure-function
composition over TWO calls to already-tested code (`run_backtest`, unmodified) rather than a
structural change to it, it costs an extra ~2 minutes of wall-clock (cheap, per wave 1's own
timing measurement), and it keeps the "which classifier changed the number" attribution trivial
— exactly the auditability property this project's honesty framework optimizes for everywhere
else (D-04's own reasoning for wave 1's frozen-cols threading applies here by analogy).

### Anti-Patterns to Avoid

- **Reusing `trading_crab_lib.momentum`/`.diagnostics` by `import` instead of porting the
  function bodies.** This is the single fastest way to raise the legacy-import ratchet past 31
  and fail `test_platform_legacy_import_ratchet.py`. The ratchet is an AST scan of every `.py`
  file under `platform/` — adding `from trading_crab_lib.momentum import compute_relative_strength`
  anywhere under `platform/` is caught exactly the same way a bare `import trading_crab_lib`
  would be.
- **Treating classifier #2's feature-set design as a search problem.** D-10/D-11/D-13 together
  fix the candidate list, the freeze rule, and (K, λ) all by construction — there is no sweep
  anywhere in wave 2's scope. A plan task that proposes "try a few candidate feature subsets and
  see which gives the best occupancy" is scope creep against D-13's explicit "zero selection
  trials."
- **Computing dependence or joint lift on non-overlapping windows without saying so.** Wave 1's
  own hard-won lesson (the 356-vs-470-month window narrowing) applies with equal force here:
  classifier #1's frozen labels and classifier #2's frozen labels may resolve over different
  degraded-step counts once threaded through `driver.py`'s L2 admission path (Common Pitfall 3)
  — report both windows inline, not as a footnote, exactly as `07-04`'s D-05 table did.
- **Assuming `regime_tilt_weights`/`vol_targeted_tilt` already support two inputs.** Verified
  false this session (Pattern 4) — do not discover this mid-implementation.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Common-support feature freeze for classifier #2 | A second `_reference_label_columns`-shaped function | `evaluation/report.py::_reference_label_columns`, called a second time with classifier #2's candidate list | Already exists, already tested, and D-11 requires the SAME rule — calling it twice IS the "one documented policy" this project's honesty culture insists on |
| Statistical dependence between two labelings | A hand-rolled contingency-table statistic | `sklearn.metrics.adjusted_rand_score`, `normalized_mutual_info_score`, `scipy.stats.contingency.association` | All three verified importable AND numerically exercised this session; contingency-table statistics have well-known off-by-one/normalization pitfalls that a maintained library already handles |
| Cross-tabulation / date alignment between two label series | A new alignment implementation | `platform/plotting/regime.py::label_disagreement`'s existing `pd.crosstab` + index-intersection logic | Already handles the disjoint-span edge case (returns zeros, never raises) and the silent-zero trap `disagreement.py` already documents |
| Deflated Sharpe ratio | An ad-hoc "penalize by number of trials" heuristic | Bailey & López de Prado (2014)'s actual formula (expected-max-SR under the null + probabilistic Sharpe ratio correction for skew/kurtosis) — hand-implemented, ~40 lines, zero new dependencies | The formula is precise and short enough to implement directly; an ad-hoc penalty would not be the "true DSR" design §22 and D-16 both name explicitly, and would not survive scrutiny at design freeze |
| Vol-scaling / leverage capping for the joint tilt | A new vol-targeting implementation for the blended weights | `allocation/tilt.py::portfolio_vol` + `vol_target_scale`, called AFTER the new blend step | These two functions are generic over any weight vector — no classifier-specific logic inside them; only the blend-then-scale ORDER is new |

**Key insight:** as in wave 1, almost everything wave 2 needs is a second call to, or a thin
extension of, machinery that already exists and is already tested. The two places that are
genuinely novel — the joint-tilt blend function and the deflated-Sharpe computation — are both
small, well-specified, and free of any new external dependency.

## Common Pitfalls

### Pitfall 1: `canonicalize_states`'s hardcoded sort key — see Architecture Patterns Pattern 1

Fully covered above; repeated here only as an index entry since this is the single blocking item
for this phase. **Resolve before fitting classifier #2 even once**, per the WAVE-2 OPENING
AMENDMENT item 3.

### Pitfall 2: `vol_targeted_tilt`/`regime_tilt_weights` do not accept two probability inputs today

**What goes wrong:** a plan task that reads D-14 ("two separate probability inputs feed the
allocation tilt") and assumes this is already how `allocation/tilt.py` works will discover,
mid-implementation, that it is not — `regime_or_probs` is a single positional argument
[VERIFIED: `allocation/tilt.py`, full file read this session].

**Why it happens:** D-14's decision record states the DESIRED property, not an audit of the
current code; it was written to rule out a product state space, not to certify the allocation
layer already supports two inputs.

**How to avoid:** budget a small new function (`blend_regime_tilts`, Pattern 4) as its own task,
with its own test, rather than folding it silently into "wire up classifier #2's probabilities."

**Warning signs:** a plan task titled "pass classifier #2's probs into vol_targeted_tilt" with no
corresponding new function or test — that phrasing assumes the interface already exists.

### Pitfall 3: The L2-degradation window is inherited, not re-derived, and criterion 7 must say which reading it uses

**What goes wrong:** joint-lift is measured by comparing classifier-#1-alone's walk-forward
equity curve against the joint (#1×#2) curve. Both curves are produced by `run_backtest()`,
whose L2 admission path (`_cv_safe_active_features`, `driver.py:120-167`) degrades a step
whenever any class in a training window has fewer than `n_splits` examples
[VERIFIED: `driver.py:161-163`, read this session — the exact condition:
`if len(counts) >= 2 and int(counts.min()) >= n_splits:`]. Wave 1 already measured that freezing
L1's admission (unrelated to L1 vs. L2 choice, but changing L1's OUTPUT LABEL sequence) shifted
the degrade count from 118/588 to 232/588 steps, narrowing the comparison window from
2020-12 to 2017-05 [CITED: ADR-0001 Consequences, "Named limitation" section]. **Classifier #2's
own L1 fit will produce a DIFFERENT label sequence again**, and if wave 2's joint evaluation uses
`use_regime_tilt=True` with L2 nowcasting in the loop (which criterion 7 does not strictly
require — see below), the degrade count could shift a THIRD time, on top of the already-narrowed
window wave 1 left behind.

**Why it happens:** L2's admission gate is about CV-fold class balance, which is sensitive to
whatever label sequence L1 (whichever classifier) produced that step — a structurally
independent mechanism from L1's own feature-set choice, but one that reacts to L1's OUTPUT every
time L1 changes.

**How to avoid — the two options the amendment names, restated with an implementation lens:**
(a) **Make L2's CV robust to rare classes** (e.g., adaptive `n_splits` per window, or a
class-aware fold strategy) — this changes numbers for every prior phase that touched L2
(Phase 5's original ablation deltas, wave 1's own re-measurement), a large blast radius.
(b) **Move criterion 7's joint-lift measurement off L2's `proba` onto L1's own filtered labels**
— i.e., compute the tilt from each classifier's OWN L1 states directly (a degenerate
single-state-probability input per the `use_regime_tilt=False`/ablation code path's own pattern,
or a simpler L1-only probability proxy), sidestepping L2 (and its degrade mechanism) entirely for
the specific purpose of measuring joint-lift. **This second option is available NOW, cheaply**:
`driver.py`'s own ablation branch (`use_regime_tilt=False`) already demonstrates the pattern of
running the tilt off a degenerate state/probability pair without invoking L2 at all — building
classifier #2's evaluation the same way (skip L2 nowcasting entirely; feed L1's own smoothed or
filtered labels directly into `blend_regime_tilts`) removes the L2-degradation confound from
criterion 7's comparison ENTIRELY, at the cost of not exercising L2 for classifier #2 at all
(which criteria 5-7 never require — L2 nowcasting for classifier #2 is not named in any of the
three criteria).

**Recommendation:** since none of criteria 5, 6, or 7 explicitly requires an L2 nowcaster for
classifier #2 (criterion 7 says "joint allocation lift measured walk-forward" — the L1 labeling
+ L4 allocation loop, not necessarily L2's calibrated probabilities), **wave 2 should route the
joint-lift comparison through L1's own labels/confidences (option (b))** rather than resolving
the L2 CV-robustness question (option (a)) inside this phase. This sidesteps the inherited
window-narrowing confound cleanly, keeps criterion 7's comparison on the SAME, already-measured
window classifier #1 alone uses (whatever L1-only equivalent of `per_step_metrics` that produces
— see Open Questions), and defers the harder L2-design question to its own ADR exactly as the
amendment instructs. **This must be stated explicitly in wave 2's own ADR section — do not let
the plan silently inherit L2's window narrowing without saying so, per the amendment's own
closing instruction.**

**Warning signs:** a plan task that runs `run_backtest(use_regime_tilt=True, ...)` for both
classifiers without any discussion of which L2-degradation reading it is using, or a joint-lift
number reported without its own `n_compared`/window-end-date caveat inline (mirroring D-05's
binding condition from wave 1's own UAT sign-off).

### Pitfall 4: Reading the trial registry's row count naively double-counts or under-counts against D-16's true denominator

**What goes wrong:** `registry/trials.jsonl` currently has exactly **1 row** — the provenance
header itself [VERIFIED: `wc -l registry/trials.jsonl` this session = 1;
`registry/archive/trials-pre-P7W1-reset.jsonl` has 42 rows, also verified this session]. A naive
`len(read_trials())` or `wc -l` at DSR-write time will therefore undercount the true trial total
by 38 (the header's own `prior_genuine_trials`) UNLESS the header row is added back in, and it
must ALSO not itself be counted as a 39th "trial" (its `config_hash` is the literal string
`"RESET"`, not a real config hash, and its `metrics` field holds accounting metadata, not a
performance measurement).

**The correct read, verified against the live file this session:**

```python
# registry/trials.jsonl, row 0 (verified this session):
{"config_hash": "RESET", "config": {"trial_tag": "REGISTRY-RESET-P7W1",
 "record_type": "provenance_header", ..., "prior_genuine_trials": 38,
 "discarded_smoke_rows": 4, ...}, "features": [], "metrics": {"prior_genuine_trials": 38, ...}, ...}
```

`config["record_type"] == "provenance_header"` is the load-bearing discriminator. The correct
trial count for D-16's deflated-Sharpe denominator is:

```python
def total_trial_count(path=None) -> int:
    df = registry.read_trials(path)
    if df.empty:
        return 0
    is_header = df["config"].apply(lambda c: isinstance(c, dict) and c.get("record_type") == "provenance_header")
    prior = int(df.loc[is_header, "config"].apply(lambda c: c["prior_genuine_trials"]).sum()) if is_header.any() else 0
    return prior + int((~is_header).sum())
```

**Why it happens:** the reset mechanism (WAVE-2 OPENING AMENDMENT item 2) is new this phase and
has no existing reader anywhere in the codebase — `registry.py`'s `read_trials()` returns the raw
DataFrame including the header row with no special-casing (verified by reading `registry.py` in
full this session: `read_trials` is a bare `pd.read_json(..., lines=True)`, no header-awareness
at all).

**How to avoid:** write `total_trial_count()` (or equivalent) as its OWN small, directly-tested
function before wiring it into the deflated-Sharpe computation — do not inline the header-parsing
logic into the DSR function itself, so a future reset (a second header row, should the registry
ever need resetting again) has one clear place to extend.

**Warning signs:** a deflated-Sharpe implementation that calls `len(registry.read_trials())`
directly, or that hardcodes `38` or `42` as a starting offset rather than reading the header row
live (repeating exactly the "34, then 38, then 42 — all stale the moment you write them down"
lesson wave 1's own ADR documents about trial counts).

### Pitfall 5: Ported legacy functions assume quarterly cadence; the platform is monthly

**What goes wrong:** `momentum.py::DEFAULT_RELATIVE_STRENGTH_PAIRS`,
`DEFAULT_CORRELATION_PAIRS` (window=8 "quarters"), and `compute_trailing_momentum`'s default
`windows=[2, 4, 8]` are all sized for the legacy quarterly pipeline (2/4/8 quarters ≈ 6/12/24
months). A literal line-for-line port that keeps these numeric defaults would silently compute
2-month, 4-month, and 8-month windows instead of the intended 6/12/24-month equivalents — a
subtle, silent semantic error, not a crash.

**Why it happens:** the ported functions (`compute_relative_strength`,
`compute_rolling_cross_correlation`, `compute_trailing_momentum`, `compute_inflation_acceleration`)
are pure, parameterized, and generic over any period unit [VERIFIED: `src/trading_crab_lib/momentum.py`,
full file read this session — no hardcoded quarterly assumption INSIDE the math itself, only in
the DEFAULT arguments and docstrings, e.g. `pct_change(periods=w)` where `w` is unit-agnostic].
The functions themselves are period-agnostic; only their DEFAULT constants and docstrings
("quarterly time-series columns") assume the old cadence.

**How to avoid:** when porting, explicitly re-derive every window constant for monthly cadence
(e.g., re-declare `rs_window`/`rm_window`/correlation windows in month-equivalents — 12/24-month
windows rather than 4/8-"quarter" windows, consistent with design §9's own guidance: "one-sided
EMAs at multiple half-lives" for the monthly spine) rather than copying the numeric literals
unchanged. Rename every docstring reference from "quarterly" to "monthly." Do not port
`compute_rrg`/`rolling_zscore`/`percentile_rank`/`normalize_100` (legacy `diagnostics.py`) for
wave 2 at all — verified this session (`diagnostics.py`, full file read) that RRG quadrant
classification is a TACTICAL/asset-rotation concept for individual ETF diagnostics (design's
§23.2 tripwire monitor territory), not a regime-LABELING input; none of D-10/D-11/D-12's
candidate feature list references RRG quadrants, and importing it would add porting surface
(218 lines) with no criterion-5/6/7 payoff.

**Warning signs:** a ported function whose default window arguments still read `[2, 4, 8]` or
`window: int = 8` with a "quarters" docstring after the port — a signal the semantic
re-derivation step was skipped, not just a cosmetic rename.

### Pitfall 6: `TOTALSL` is the better credit-aggregate candidate than `BCNSDODNS` — a frequency mismatch, verified live

**What goes wrong:** the ADR's candidate list (D-12, restated in `07-CONTEXT.md`) names
"`TOTALSL` or `BCNSDODNS`" as equally plausible credit-aggregate candidates. Live-verified this
session via `fredapi.Fred.get_series()`:

| Series | First valid | Last valid | n (non-null) | Implied native frequency |
|---|---|---|---|---|
| `M2SL` | 1959-01-01 | 2026-07-01 | 811 | Monthly (matches the platform spine natively) |
| `TOTALSL` (Total Consumer Credit Owned and Securitized) | 1943-01-01 | 2026-07-01 | 1003 | Monthly |
| `BCNSDODNS` (Nonfinancial Corporate Business; Debt Securities and Loans, Liability, Level) | 1945-10-01 | 2026-04-01 | 305 | **Quarterly** (305 obs / ~80 years ≈ 3.8/year) |
| `TOTBKCR` (Total Bank Credit, All Commercial Banks) | 1973-01-03 | 2026-09-02 | 2801 | Weekly/monthly, but starts too late (1973, not 1962) |

[VERIFIED: live `fredapi.Fred.get_series(...)` calls this session for all four series IDs]

`BCNSDODNS` is a Federal Reserve Z.1 Flow-of-Funds series, natively quarterly — using it in the
platform's monthly spine would require the SAME forward-fill/interpolation treatment `fred_gdp`
already gets (verified this session: `monthly_raw["fred_gdp"]` repeats its quarterly value across
each month within the quarter, e.g. 3758.147 for 1962-02/03/04
[VERIFIED: `data/checkpoints/platform/monthly_raw.parquet`, read directly this session]), plus
the ALFRED point-in-time-vintage question design §9's agency-tier taxonomy raises for
revision-prone Fed aggregates. `TOTALSL` is natively monthly, requires no interpolation, and
reaches back to 1943 — comfortably covering the 1962+ decision window with native-frequency data.

**Recommendation:** use `TOTALSL` as the primary credit-aggregate candidate; document `BCNSDODNS`
as considered-and-rejected (frequency mismatch) rather than silently dropping it from the ADR's
candidate list, consistent with this project's "state what a rejected option would have cost"
practice (mirroring wave 1 ADR's own D-03 imputation-trial treatment).

**Warning signs:** a plan task that ingests `BCNSDODNS` without any interpolation/alignment step
— the resulting monthly series would have long flat stretches (quarter-repeated values) unless
explicitly handled, and if left un-aligned to the monthly index at all, would produce mostly-NaN
months that fail `_reference_label_columns`'s common-support freeze outright.

### Pitfall 7: The legacy import ratchet is an AST scan of the WHOLE `platform/` tree, not just new files

**What goes wrong:** a developer might reasonably assume the ratchet test
(`test_platform_legacy_import_ratchet.py`) only checks NEW files wave 2 adds. It does not —
`_legacy_import_sites()` walks every `.py` file under `PLATFORM_ROOT` on every test run
[VERIFIED: `tests/unit/test_platform_legacy_import_ratchet.py:44-58`, read in full this session].
Adding even one new `from trading_crab_lib.momentum import ...` or
`from trading_crab_lib.diagnostics import ...` anywhere under `platform/` — including inside the
new `platform/features/relative.py` module this research recommends — raises the count from 31
to 32 and fails `test_legacy_import_count_does_not_grow` immediately.

**Why it happens:** it is tempting, when porting, to reach for the fastest path (`import` the
already-tested legacy function directly) rather than copying its body — especially since the
legacy functions have ZERO legacy-library-internal dependencies themselves (verified this
session: `momentum.py`'s only imports are `numpy`, `pandas`, `logging`, `typing` — no
`CheckpointManager`, no `config.load()`, nothing that would normally justify treating the import
as "coupling" in spirit). The ratchet test does not distinguish "harmless pure-function import"
from "coupling" — it is a pure syntactic AST scan by design (so it cannot be fooled the way the
old `grep -v platform` check was).

**How to avoid:** copy the function BODIES (not `from X import Y`) into the new platform-native
module, with attribution comments naming the source file and line range this research verified
(`momentum.py:77-110` for `compute_relative_strength`, `:123-157` for
`compute_rolling_cross_correlation`, `:162-180` for `compute_inflation_acceleration`) — mirroring
exactly how `platform_settings.yaml`'s own comments already document "ported, not imported" for
other seams (e.g. macrotrends: "Reused verbatim from `trading_crab_lib.ingestion.macrotrends`
(D-01: import, never edit)" is the OPPOSITE pattern used elsewhere in this SAME repo for a
different, intentionally-imported seam — do not confuse the two conventions).

**Warning signs:** `pytest tests/unit/test_platform_legacy_import_ratchet.py -v` failing with
"platform/ legacy imports rose to 32" after the port lands — the exact, named failure mode the
test's own docstring predicts for "a later phase" (that phase is this one).

### Pitfall 8: The trial ceiling formula from wave 1's ADR must be re-applied, not re-derived from scratch

**What goes wrong:** D-17's original "~5 trials, ~35 total" estimate is stale twice over — once
by wave 1's own finding that each full evaluation run appends 2 rows not 1 (ADR-0001's "Trial
ceiling" section), and again by the registry reset (amendment item 2, this document's opening
section). A plan that budgets wave 2's trials against "~35" or "~42" (either number appearing
literally in prior planning documents) will be judging against a number two decisions have
already superseded.

**How to avoid:** re-state the ceiling using wave 1's own formula
(`registry_rows_added = 2 × N_full_evaluation_runs`) applied to wave 2's ACTUAL planned run
count, then separately state the true cumulative total using Pitfall 4's `total_trial_count()`
(`prior_genuine_trials` + post-header rows), not the two numbers conflated. Given D-13's zero
selection trials and the recommended two-independent-runs approach (Pattern 4): classifier #2's
own fit is not itself a registry trial in the `run_backtest` sense (fitting a jump model is not
a call to `run_backtest`) — the registry-relevant runs are (a) classifier #2 alone (if measured
as its own leg, optional), (b) the joint #1×#2 evaluation, and (c) the #1-alone baseline
(already measured in wave 1, MAY be re-usable without a new run if wave 1's exact frozen-cols
configuration is unchanged — verify before re-running unnecessarily, since D-17's spirit is to
minimize registry growth, not run everything twice reflexively). At minimum 2 new full-evaluation
runs (joint + one comparison leg) × 2 rows = **4 new rows**, pushing the live-read total (1
header row representing 38 prior + whatever wave 1 itself appended after its own header, verified
live at ADR-write time, not assumed) up by 4. **State this as a formula in wave 2's ADR section,
exactly as wave 1's ADR did, and call `total_trial_count()` live immediately before and after.**

## Code Examples

### `canonicalize_states` — current hardcoded behavior (verbatim, the landmine)

```python
# src/trading_crab_lib/platform/labeling/jump_model.py:214-221
if "trailing_return_1m" in feature_names:
    sort_col = feature_names.index("trailing_return_1m")
else:
    sort_col = 0
    log.warning(
        "trailing_return_1m not in feature_names — falling back to centroid "
        "column 0 for canonicalization sort order"
    )
```

### The DP-decode / jump-model machinery classifier #2 reuses unmodified

```python
# src/trading_crab_lib/platform/labeling/jump_model.py:140-187 (fit_jump_model, unmodified)
# Classifier #2's own call, illustrative:
from trading_crab_lib.platform.labeling.jump_model import fit_jump_model, standardize_features, canonicalize_states

X_df = dev_features[frozen_cols_2].dropna(axis=0, how="any")
X = standardize_features(X_df)
K_2 = cfg["labeling_2"]["K"]              # pre-declared, D-13
lam_2 = 4 * len(frozen_cols_2)            # same formula as classifier #1's λ = 4 × n_features
fit = fit_jump_model(X, K=K_2, lam=lam_2, n_restarts=10)
states_2, _ = canonicalize_states(
    fit["states"], fit["centroids"], list(X_df.columns),
    sort_column="rs_equities_bonds",      # classifier #2's OWN leading feature, never #1's
)
```

### Registry provenance header — the exact current row (verbatim, live-read this session)

```json
{"config_hash": "RESET", "config": {"trial_tag": "REGISTRY-RESET-P7W1", "record_type": "provenance_header", "reset_reason": "Phase 7 wave 1 appended 4 untagged rows from wiring-verification (smoke) runs of trading_crab_lib.platform.evaluation.report, which are not evaluated configurations. D-16 deflates Sharpe over the whole registry, so those rows would have inflated the trial count. Ledger archived intact and restarted; no row was deleted.", "archived_to": "registry/archive/trials-pre-P7W1-reset.jsonl", "archived_row_count": 42, "prior_genuine_trials": 38, "discarded_smoke_rows": 4, "deflation_note": "D-16's trial count since project start = prior_genuine_trials + rows appended after this header. Do NOT read the post-reset row count as the project total."}, "features": [], "metrics": {"prior_genuine_trials": 38, "discarded_smoke_rows": 4}, "git_sha": null, "timestamp": "2026-09-15T14:43:46.377815+00:00"}
```

`wc -l registry/trials.jsonl` = **1** at time of writing [VERIFIED: this session] — every trial
wave 1 itself logged after this header (the two `P7-W1-frozen-10col` rows, two
`P7-W1-impute-13col-REJECTED` rows referenced in `07-MEASUREMENTS.md` §8) was appended to the
PRE-reset ledger and is now inside `registry/archive/trials-pre-P7W1-reset.jsonl`'s 42 rows — the
reset happened AFTER wave 1's measurement work, per the header's own timestamp
(2026-09-15T14:43:46Z) versus `07-MEASUREMENTS.md`'s registry-accounting section (dated
2026-09-14). **This means wave 1's own 4 measurement rows (34→36→38, per `07-MEASUREMENTS.md` §8)
are counted inside the archived 42, and the header's `prior_genuine_trials=38` already includes
them** — confirm this reconciliation explicitly at ADR-write time rather than assuming it; it is
inferred from timestamps here, not independently re-verified against the archive file's own
contents this session (see Assumptions Log A3).

### Deflated Sharpe ratio — the formula to hand-implement (no existing code; grep-confirmed absent)

```python
# NEW — platform/evaluation/deflated_sharpe.py (sketch; formula per Bailey & López de Prado 2014,
# "The Deflated Sharpe Ratio," https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551
# [CITED via WebSearch this session])
import numpy as np
from scipy.stats import norm

def expected_max_sharpe(n_trials: int, sharpe_variance: float) -> float:
    """Expected maximum Sharpe ratio of n_trials skill-less (SR=0) trials,
    Bailey-Lopez de Prado (2014) eq. 8 approximation. sharpe_variance is the
    variance of the SR estimates ACROSS the registry's trials (proxies for
    how dispersed the search was)."""
    euler_mascheroni = 0.5772156649
    if n_trials <= 1:
        return 0.0
    return np.sqrt(sharpe_variance) * (
        (1 - euler_mascheroni) * norm.ppf(1 - 1.0 / n_trials)
        + euler_mascheroni * norm.ppf(1 - 1.0 / (n_trials * np.e))
    )

def deflated_sharpe_ratio(
    observed_sharpe: float, n_trials: int, sharpe_variance: float,
    skew: float, kurtosis: float, n_obs: int,
) -> float:
    """Probabilistic Sharpe Ratio of observed_sharpe against the expected-max-
    under-the-null hurdle, correcting for the trial's own skew/kurtosis and
    track length (n_obs = number of return observations, e.g. 356-588
    monthly steps depending on which L2-window reading Pitfall 3 selects)."""
    sr0 = expected_max_sharpe(n_trials, sharpe_variance)
    denom = np.sqrt(1 - skew * observed_sharpe + ((kurtosis - 1) / 4) * observed_sharpe**2)
    z = (observed_sharpe - sr0) * np.sqrt(n_obs - 1) / denom
    return float(norm.cdf(z))
```

**This is a sketch to ground the planner's task sizing, not a final implementation** — the exact
`sharpe_variance` estimator (population variance of the registry's historical Sharpe-bearing
trials vs. some other dispersion proxy) is a genuine design choice the ADR must record, since
design §22/§8 name the CONCEPT (Bailey–López de Prado DSR) but do not pin an exact estimator
choice for this codebase. **Flagged in Open Questions below.**

## State of the Art

| Old Approach | Current/Recommended Approach | When Changed | Impact |
|---|---|---|---|
| One regime labeler (crisis/stress axis only) | Two independent labelers (crisis axis + leadership/relative-strength axis), fed as separate probability inputs | This phase (wave 2) | Adds the axis the Phase 6 P3 sign-off flagged as missing ("crisis regimes, not allocation regimes") |
| `canonicalize_states` implicitly assumes every caller shares classifier #1's return-level axis | Explicit, caller-supplied `sort_column` per classifier | This phase (wave 2), landmine fix | Prevents a systemic, ~100%-of-fits fallback corruption class (vs. #1's own historically negligible 0.2%, A14) |
| No deflated-Sharpe implementation anywhere in the codebase | A small, hand-implemented DSR function reading the registry's provenance header | This phase (wave 2) | Makes design §22/D-16's "deflated Sharpe for headline results" an actual, runnable check rather than a stated intention |
| Registry trial count read as a raw row count | Registry trial count read as `prior_genuine_trials` (header) + post-header rows | This phase (wave 2), inherited from wave 1's reset | Prevents undercounting D-16's true denominator by 38 |

**Deprecated/outdated:** any planning-document reference to "30", "34", "35", "38", or "42" as
THE trial count is stale the moment it is written down (wave 1's own ADR makes this point about
itself) — always re-read live via `total_trial_count()` (Pitfall 4) at ADR-write time.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `rs_equities_bonds` (equity/bond relative strength, built from `equities_tr`/`long_duration_tr`) is the correct choice for classifier #2's `canonicalize_states` sort column — chosen by analogy to the proposal's own "single most important missing variable" framing, not independently derived from a fitted classifier #2's actual centroid structure (which does not exist yet) | Architecture Patterns Pattern 1, Code Examples | Low — the sort column only affects state NUMBERING (which state is called 0 vs K-1), never occupancy, dependence, or lift; a different, equally-reasonable choice (e.g. the stock-bond rolling correlation itself) would not change any of criteria 5-7's substantive findings, only which state index is "leading" |
| A2 | `TOTALSL` is the better credit-aggregate candidate than `BCNSDODNS` for INV-01, based on native monthly vs. quarterly frequency verified via live FRED calls this session — not independently checked against ALFRED vintage/revision history for either series | Common Pitfall 6 | Medium — if `TOTALSL` turns out to have material revision history requiring ALFRED point-in-time treatment (unverified this session), the "no interpolation needed" advantage could be partly offset by an agency-tier alignment cost neither series currently has budgeted |
| A3 | The registry reset (provenance header, timestamped 2026-09-15T14:43:46Z) happened AFTER wave 1's own 4 measurement-run registry appends (dated 2026-09-14 per `07-MEASUREMENTS.md` §8), such that `prior_genuine_trials=38` already includes those 4 rows and wave 2 does not need to separately account for them — inferred from timestamp ordering, NOT independently verified by reading `registry/archive/trials-pre-P7W1-reset.jsonl`'s actual row contents this session | Code Examples ("Registry provenance header") | Low-Medium — if wrong, D-16's denominator would be off by up to 4 trials at wave 2's ADR-write time; cheaply checked by reading the archive file directly before wave 2's ADR is written |
| A4 | No L2 nowcaster is required for classifier #2 to satisfy criteria 5, 6, or 7 as literally worded — criterion 7 says "joint allocation lift measured walk-forward," which this research reads as satisfiable via L1 labels/confidences directly (Pitfall 3's recommended option (b)), not requiring a full L2 CalibratedClassifierCV nowcaster for the SECOND classifier | Common Pitfall 3 | Medium — if the discuss-phase or planner reads criterion 7 as requiring full L2-equivalent probability calibration for classifier #2 too, this recommendation under-scopes the work; worth confirming explicitly before locking the plan, since it changes whether Pitfall 3's L2-degradation confound applies to classifier #2 at all |
| A5 | Deflated-Sharpe's `sharpe_variance` input should be estimated from the registry's own historical population of trial Sharpe ratios (a var-of-observed-SRs estimator) rather than a different dispersion proxy (e.g., assumed iid normal trials at some canonical variance) — this is a genuine implementation choice Bailey–López de Prado's paper itself discusses multiple approaches for, and this research did not read the primary paper in full this session (only its abstract/summary via WebSearch) | Code Examples (deflated Sharpe sketch) | Medium — the DSR estimator choice materially affects the deflated number; the ADR must record which estimator was used and why, since "we implemented *a* DSR" is not the same claim as "we implemented *the* DSR design §22 specifies" |

**If this table is empty:** N/A — see rows above.

## Open Questions

1. **Exactly which raw columns constitute classifier #2's final candidate list?**
   - What we know: the proposal names equity/bond relative strength + rolling correlation,
     commodity/equity and (excluded) gold/equity relative strength, CPI acceleration, and curve
     *changes* (not levels) as candidates; D-11's freeze rule admits oil/equity but excludes
     gold/equity; INV-01 adds M2/GDP and credit/GDP once ingested.
   - What's unclear: the exact final K_2 (how many raw columns after applying D-11's freeze to
     each candidate) — this determines λ_2 = 4 × K_2 by D-13's formula, so it must be pinned
     before the fit, not discovered after.
   - Recommendation: treat "finalize and freeze the candidate list, compute K_2 and λ_2" as an
     explicit Wave-0 task with its own automated check (assert the frozen list's length matches
     the λ formula's input), mirroring wave 1's own D-02-A discovery-then-lock pattern.

2. **Does the discuss-phase intend classifier #2 to have its own L2 nowcaster at all?**
   - What we know: none of criteria 5, 6, 7 names L2 for classifier #2 explicitly; D-14 only
     discusses PROBABILITY inputs to the allocation tilt, which could come from L1's own
     confidences (`soft_confidences()`, already generic and reusable — `jump_model.py:94-109`)
     rather than a full L2 nowcaster.
   - What's unclear: whether "walk-forward" in criterion 7 implicitly means the SAME L1+L2 loop
     structure `run_backtest` already implements, or whether an L1-only walk-forward loop
     (refit classifier #2's jump model on each expanding window, without any L2 step) suffices.
   - Recommendation: confirm this explicitly in discuss-phase/plan-time — this research
     recommends the L1-only reading (Pitfall 3, Assumption A4) because it is cheaper, avoids the
     L2-degradation confound entirely, and is not contradicted by any criterion's literal wording,
     but it is a genuine interpretive choice, not a settled fact.

3. **What is the exact deflated-Sharpe `sharpe_variance` estimator this project should use?**
   - What we know: the concept (expected-max-SR under N skill-less trials, corrected for the
     trial's own skew/kurtosis) is well-established (Bailey & López de Prado 2014); no prior
     implementation exists in this codebase to follow as precedent.
   - What's unclear: whether to estimate `sharpe_variance` from the registry's own historical
     trial population (heterogeneous strategies, likely NOT iid) or some other proxy.
   - Recommendation: read the primary paper (or a close secondary source) in full before
     implementing, and record the chosen estimator's justification in the wave-2 ADR — flagged
     here as unresolved rather than silently picked (Assumption A5).

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| FRED API (via `fredapi`) | INV-01's M2SL/TOTALSL ingestion | ✓ — **[VERIFIED: live `fredapi.Fred.get_series()` calls this session for M2SL, TOTALSL, BCNSDODNS, TOTBKCR, all returned real data]** | `FRED_API_KEY` present, functional | — |
| `sklearn.metrics.adjusted_rand_score` / `normalized_mutual_info_score` | Criterion 6 | ✓ — **[VERIFIED: imported and numerically exercised this session, sklearn 1.9.1]** | 1.9.1 | — |
| `scipy.stats.contingency.association` | Criterion 6 (Cramér's V) | ✓ — **[VERIFIED: imported and numerically exercised this session, scipy 1.17.1]** | 1.17.1 | — |
| `scipy.stats.norm` | Criterion 7 (deflated Sharpe quantiles) | ✓ — **[VERIFIED: same scipy install, `norm.ppf`/`norm.cdf` are core scipy.stats API, present in every scipy 1.x release]** | 1.17.1 | — |
| No new external packages | Wave 2 entirely | ✓ | — | — |

**Missing dependencies with no fallback:** none — wave 2's own new code (canonicalize_states fix,
ported features, dependence stats, joint tilt, DSR) is pure in-repo computation over
already-available libraries. The only genuinely NEW network dependency is INV-01's M2/credit
FRED ingestion, and FRED access is live-verified functional in this environment this session.

**Missing dependencies with fallback:** none identified for wave 2.

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest 8.0+ |
| Config file | root `pyproject.toml` → `[tool.pytest.ini_options]` |
| Quick run command | `pytest tests/unit/test_platform_labeling.py tests/unit/test_platform_features_relative.py tests/unit/test_platform_evaluation_dependence.py tests/unit/test_platform_allocation_joint_tilt.py tests/unit/test_platform_legacy_import_ratchet.py -x` |
| Full suite command | `pytest tests/ -q` (1752 tests + wave 2's new files, ~72-90s per wave 1's own measurement) |
| New dependencies | none |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| REG-01 (criterion 5, disjointness) | Classifier #2's raw candidate columns are disjoint from `lean_feature_set(cfg)`'s 13 | unit | `pytest tests/unit/test_platform_features_relative.py -k disjoint -x` | ❌ Wave 0 — new test |
| REG-01 (criterion 5, canonicalize fix) | `canonicalize_states` never hits the silent-fallback path for classifier #2's own call; raises `ValueError` if a caller's `sort_column` is absent | unit | `pytest tests/unit/test_platform_labeling.py -k canonicalize -x` | ❌ Wave 0 — new test (extends existing `TestCanonicalizeStates` class per file structure verified this session) |
| REG-01 (criterion 5, occupancy) | Classifier #2's occupancy sums to 1.0; every state below §4.4's ~8% floor (§4.4 crit. 1 is ~8%–~35%; see ADR-0001 § AMENDMENT 2026-09-17) is flagged (reuses `_MIN_OCCUPANCY_THRESHOLD=0.08` + `_MAX_OCCUPANCY_THRESHOLD=0.35`, `labeling/diagnostics.py:67`) | unit | `pytest tests/unit/test_platform_labeling.py -k occupancy -x` | ✅ `occupancy_and_sojourns`/`report_labeling_diagnostics` already exist and are generic — extend the existing test class with a classifier-#2-shaped fixture |
| REG-01 (criterion 6) | `measure_labeling_dependence` returns ARI, NMI, Cramér's V, plus the existing `label_disagreement` crosstab fields, with NO pass/fail gate (D-15) | unit | `pytest tests/unit/test_platform_evaluation_dependence.py -x` | ❌ Wave 0 — new test/module |
| REG-01 (criterion 7, joint tilt) | `blend_regime_tilts` produces weights summing to `scale` (matching `vol_targeted_tilt`'s existing contract), degrades gracefully when either input is empty | unit | `pytest tests/unit/test_platform_allocation_joint_tilt.py -x` | ❌ Wave 0 — new test/module |
| REG-01 (criterion 7, deflated Sharpe) | `deflated_sharpe_ratio`/`total_trial_count` computed against a hand-worked small-N oracle example; `total_trial_count` correctly reads the provenance header | unit | `pytest tests/unit/test_platform_evaluation_deflated_sharpe.py -x` | ❌ Wave 0 — new test/module; **no existing implementation anywhere to extend (verified by grep)** |
| REG-01 (criterion 7, walk-forward, real data) | Joint lift measured against classifier #1 alone on the live checkpoint, inside its own reported window (Pitfall 3) | integration (real re-run) | manual, mirroring wave 1's `run_full_backtest_evaluation()`-style human-verify pattern | Manual-only — needs a real multi-minute walk-forward against live checkpoints |
| — (ratchet regression guard) | Porting relative-strength features does not raise the legacy-import count past 31 | unit | `pytest tests/unit/test_platform_legacy_import_ratchet.py -x` | ✅ exists unchanged — this IS the guard; a red result here means the port leaked an import |
| INV-01 | M2SL/TOTALSL ingested, aligned to the monthly spine, no interpolation needed (both native monthly) | unit + live (network) | `pytest tests/unit/test_platform_ingestion_macro_monthly.py -k m2_or_credit -x` (new cases) + one live `fetch_fred_monthly()` call | ❌ Wave 0 — new ingestion config entries + test cases |
| ADR (wave 2) | Policy choices recorded: sort-column convention, credit-aggregate choice, blend weight, L2-window decision, DSR estimator, trial ceiling formula | manual (document review) | N/A | N/A |

### Sampling Rate

- **Per task commit:** quick run command above (~15-30s, no real-data dependency for unit portions)
- **Per wave merge:** full suite (`pytest tests/ -q`)
- **Phase gate:** at least one real walk-forward evaluation of the joint (#1×#2) tilt against
  classifier #1 alone, on live checkpoints — mirrors wave 1's own phase-gate pattern

### Wave 0 Gaps

- [ ] `tests/unit/test_platform_labeling.py` — extend with `canonicalize_states`'s `sort_column`
      parameter tests (no-fallback oracle test, missing-column `ValueError` test)
- [ ] `tests/unit/test_platform_features_relative.py` — NEW file: disjointness test, ported-
      function parity tests (each ported function's output checked against a hand-computed
      small synthetic example, mirroring `momentum.py`'s own module-level self-check pattern)
- [ ] `tests/unit/test_platform_evaluation_dependence.py` — NEW file: ARI/NMI/Cramér's V against
      known small examples (e.g., identical labelings → ARI=NMI=1, Cramér's V=1; independent
      random labelings → all near their "no relationship" values)
- [ ] `tests/unit/test_platform_allocation_joint_tilt.py` — NEW file: `blend_regime_tilts`
      against hand-computed small examples, degenerate-input edge cases (one classifier's probs
      empty, both empty)
- [ ] `tests/unit/test_platform_evaluation_deflated_sharpe.py` — NEW file: DSR formula against
      a hand-worked oracle (e.g., N=1 trial should return the raw Sharpe's own significance,
      not a deflated one; N→∞ with fixed observed SR should drive DSR toward the null); AND
      `total_trial_count()` tested against a synthetic registry file containing a
      `record_type: provenance_header` row
- [ ] Framework install: none required
- [ ] INV-01 ingestion config entries (`fred_monthly.series` additions for M2SL/TOTALSL) + one
      live ingestion smoke test

## Plausibility Bands — extending `06-VALIDATION.md` and `07-VALIDATION.md`'s contract to wave 2

Every band from `06-VALIDATION.md` (terminal log wealth, max drawdown, Brier, turnover, CVaR,
occupancy, portfolio weights) and `07-VALIDATION.md` (pct_disagree, §5.4 ratio, n_transitions,
wealth_delta, dd_delta, trial-row-count-per-run) is UNCHANGED and reused for classifier #2's own
version of each quantity. New quantities wave 2 introduces:

| Quantity | Universal bound | Domain band | Source / reasoning |
|---|---|---|---|
| Classifier #2 occupancy | each `∈[0,1]`, `Σ=1.0` | soft warning `<0.05` (`_MIN_OCCUPANCY_THRESHOLD`, `labeling/diagnostics.py:67`) | [CITED: same constant classifier #1 already uses] — unchanged, reused as-is (criterion 5's literal wording) |
| Adjusted Rand Index (criterion 6) | `x ∈ [-1, 1]` (can be slightly negative for worse-than-chance agreement) | **no numeric pass/fail (D-15)** — but `x > 0.7` should be flagged in prose as "high dependence, candidate failure to add an axis"; `x` near 0 as "the target case, an independent axis" | [ASSUMED] — proposed this session by convention (ARI ≈ agreement fraction after chance-correction); no prior project precedent for this specific statistic |
| Normalized Mutual Information (criterion 6) | `x ∈ [0, 1]` | **no numeric pass/fail (D-15)** — `x > 0.5` flagged in prose as suspicious given K₁=5 possible states already carry meaningful baseline entropy | [ASSUMED] — proposed this session |
| Cramér's V (criterion 6) | `x ∈ [0, 1]` | **no numeric pass/fail (D-15)** — `x > 0.5` (conventionally "large effect" in social-science usage) flagged in prose | [ASSUMED] — proposed this session, using the conventional Cohen-style effect-size language as the flagging language, not a gate |
| Joint-tilt weights (criterion 7) | each `>= 0` (long-only, D-14 inherits classifier #1's `regime_tilt_weights` long-only clipping), `Σ <= scale <= 1` | matches `vol_targeted_tilt`'s existing output contract exactly | [CITED: `allocation/tilt.py`, unchanged downstream vol-scaling] |
| Joint-vs-#1-alone `wealth_delta`/`dd_delta` (criterion 7) | same universal bounds as wave 1's ablation deltas: `abs(wealth_delta) < 15`, `dd_delta ∈ [-2, 2]` | **the four now-load-bearing `[ASSUMED]` bands from the amendment must be CONFIRMED (not just reused) before this row is judged**: `abs(wealth_delta) < 5`, `dd_delta ∈ [-0.5, 0.5]` | [CITED: `07-VALIDATION.md`] for the universal bounds; **the domain bands are the exact four items the amendment names as now-load-bearing — this research recommends a human confirm or revise all four in wave 2's discuss-phase, before any joint-lift number is judged against them** |
| Deflated Sharpe (criterion 7) | `x ∈ [0, 1]` (it is a probability, per the PSR/DSR formula's `norm.cdf` output) | **no numeric target set** — design §22/D-16 name the correction, not a pass/fail threshold; a DSR near 0.5 or below on the joint strategy should be reported plainly as "does not clear the multiple-testing hurdle," not softened | [ASSUMED] — the `[0,1]` universal bound follows directly from the formula's own structure (a CDF value); the "no target" domain posture mirrors D-04/D-07's established plausibility-only philosophy applied to a new metric |
| `total_trial_count()` (registry accounting) | `x >= prior_genuine_trials` (38, or whatever the live header states) — **a count LESS than the header's own stated prior is a parsing bug, not a legitimate reading** | — | [VERIFIED: derivable directly from the header's own semantics, read this session] |

### The third check class, extended

Wave 1 introduced a "suspiciously resolved" band class (a number that LOOKS good but signals a
different bug). Wave 2 needs the SAME class applied to dependence: **a dependence statistic that
looks "too clean" (ARI/NMI exactly 1.0, or exactly 0.0) is itself a signal to check** — exactly
1.0 likely means the two classifiers are reading the same underlying labels (a wiring bug, the
same failure class `disagreement.py`'s `suspicious`/`suspicious_reason` fields already guard
against for criterion 3); exactly 0.0 across ALL three statistics simultaneously (not just one)
is unusually clean for real financial data and warrants the same "confirm this isn't an alignment
bug" scrutiny `06-VALIDATION.md`'s own philosophy already establishes.

## Security Domain

`security_enforcement: true`, `security_asvs_level: 1` in `.planning/config.json`. Wave 2 adds
one new network-facing surface (M2/credit FRED ingestion) beyond wave 1's pure-computation scope;
otherwise the same posture as wave 1 applies — no auth, no session, no access-control surface.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-------------------|
| V2 Authentication | No | No auth surface touched |
| V3 Session Management | No | No session surface touched |
| V4 Access Control | No | No access-control surface touched |
| V5 Input Validation | Marginal | New FRED series (M2SL/TOTALSL) should reuse the existing `_warn_on_level_discontinuity`/`_series_kind` guard pattern (`transforms_monthly.py`, already used for other agency/vintage series) rather than a bespoke validation path — an aggregate series with a units/rebasing error would otherwise silently corrupt the M2/GDP invariant candidate |
| V6 Cryptography | No | Not applicable |

### Known Threat Patterns for this stack

Not applicable in the STRIDE sense (no network-facing endpoint accepting untrusted input, no
authentication). The one project-specific honesty-adjacent concern: a hand-implemented
deflated-Sharpe function that silently under-reads the trial count (Pitfall 4) is this project's
closest analog to a "security" defect — it would systematically UNDER-penalize search, the exact
failure mode the honesty framework (HON-01/HON-02) exists to prevent. Treat
`total_trial_count()`'s test coverage with the same rigor as a holdout-boundary guard.

## Sources

### Primary (HIGH confidence — read/executed directly this session)

- `src/trading_crab_lib/platform/labeling/jump_model.py` (full file) — `canonicalize_states`,
  `fit_jump_model`, `decode_states_dp`, `soft_confidences`
- `src/trading_crab_lib/platform/backtest/driver.py` (full file) — `_window_active_features`,
  `_cv_safe_active_features`, `_refit_l1`, `_refit_l2`, `run_backtest`
- `src/trading_crab_lib/platform/honesty/registry.py` (full file) — `append_trial`,
  `read_trials`, `NO_REGISTRY`
- `src/trading_crab_lib/platform/allocation/tilt.py` (full file) — `vol_targeted_tilt`,
  `regime_tilt_weights`, `portfolio_vol`, `vol_target_scale`
- `src/trading_crab_lib/platform/evaluation/disagreement.py` (full file) —
  `measure_label_disagreement`, the silent-zero-trap guard pattern
- `src/trading_crab_lib/platform/plotting/regime.py::label_disagreement` — cross-tabulation
  pattern criterion 6 extends
- `src/trading_crab_lib/platform/evaluation/report.py` lines 680-940 —
  `run_full_backtest_evaluation`'s step-by-step docstring, `_reference_label_columns` call site
- `src/trading_crab_lib/platform/taxonomy.py` (full file) — `lean_feature_set`,
  `validate_taxonomy`
- `src/trading_crab_lib/platform/labeling/diagnostics.py` (full file) —
  `_MIN_OCCUPANCY_THRESHOLD=0.08` + `_MAX_OCCUPANCY_THRESHOLD=0.35` (line 67), `occupancy_and_sojourns`,
  `report_labeling_diagnostics`
- `src/trading_crab_lib/momentum.py` (full file, 236 lines) — `compute_relative_strength`
  (:77-110), `compute_trailing_momentum` (:30-64), `compute_rolling_cross_correlation`
  (:123-157), `compute_inflation_acceleration` (:162-180) — the port candidates
- `src/trading_crab_lib/diagnostics.py` (partial — header + `compute_rrg`) — read to confirm
  it is NOT a wave-2 candidate (tactical RRG, not a labeling input)
- `src/trading_crab_lib/platform/transforms_monthly.py` lines 150-260 —
  `align_agency_monthly`, `compute_lean_features`, the `fred_gdp` monthly-forward-fill behavior
- `tests/unit/test_platform_legacy_import_ratchet.py` (full file) — the AST-scan ratchet
  mechanism, `MAX_LEGACY_IMPORT_SITES = 31`
- `tests/unit/test_platform_labeling.py` — confirmed NO existing test pins
  `canonicalize_states`'s fallback-warning behavior
- `config/platform_settings.yaml` lines 1-300 — `fred_monthly.series`, `taxonomy`,
  `labeling` (K=5, λ=52.0), `holdout`, `registry` sections; confirmed no M2/credit entries exist
- `platform_design/adr/0001-l1-feature-policy.md` (full file, 382 lines) — the wave-1 ADR,
  its "Deferrals and open items" section (D-10 through D-17 restated with wave-1's own live
  numbers), the trial-ceiling formula
- `platform_design/platform_design.md` — §4.3/§4.4 (occupancy criteria), §5.4 (honesty metrics),
  §9 (data span/taxonomy), §22 ((K,λ) selection protocol, "true DSR denominator"), §12 (reading
  list, Bailey & López de Prado citation) — grepped for §8.4/§8.7 (referenced in `07-CONTEXT.md`
  canonical_refs) and found **those exact section numbers do not exist in the current document**
  (design doc renumbered since those references were written; the DSR/trial-registry content
  they pointed to now lives at §22 and in the R12/D13 rows of §11's table) — flagged as a stale
  cross-reference in the phase's own canonical_refs, not a code finding
- `registry/trials.jsonl` (1 row, the provenance header) and
  `registry/archive/trials-pre-P7W1-reset.jsonl` (42 rows) — both read directly this session
- `data/checkpoints/platform/monthly_raw.parquet` — read directly this session to verify
  `equities_tr`/`long_duration_tr`/`cash`/`oil`/`fred_gs10`/`fred_tb3ms`/`fred_baa`/`fred_aaa`
  (776/776 non-NaN from 1962-01-31) and `gold` (499/776, first valid 1985-02-28), and `fred_gdp`'s
  quarter-repeated monthly values
- Live `fredapi.Fred.get_series()` calls this session for `M2SL`, `TOTALSL`, `BCNSDODNS`,
  `TOTBKCR`, `GS10` — all succeeded, exact first/last-valid dates and observation counts recorded
  above
- `python3 -c "..."` this session — confirmed `sklearn.metrics.adjusted_rand_score`,
  `normalized_mutual_info_score`, `scipy.stats.contingency.association`, `scipy.stats.chi2_contingency`
  all import and execute correctly against real numpy inputs
- `grep -rn "deflated\|prior_genuine_trials\|DSR"` across `src/` and `tests/` — confirmed
  **no deflated-Sharpe implementation exists anywhere in the codebase**

### Secondary (MEDIUM confidence — planning documents, not independently re-verified against code)

- `.planning/phases/07-regime-representation/07-CONTEXT.md` (WAVE-2 OPENING AMENDMENT, D-10
  through D-17, canonical_refs)
- `.planning/phases/07-regime-representation/07-MEASUREMENTS.md`, `07-UAT.md`,
  `07-VERIFICATION.md`, `07-VALIDATION.md` (wave 1's closed record; the numbers wave 2 compares
  against)
- `.planning/PROPOSAL-phase-regime-representation.md`, `.planning/PROPOSAL-dual-regime-classifiers.md`
  (wave 2's original candidate feature list and cost/risk framing)
- `.planning/REQUIREMENTS.md` (REG-01, INV-01 verbatim), `.planning/ROADMAP.md` Phase 7 section
  (criteria 5-7 verbatim, criterion 8's 2026-09-15 correction)
- `.planning/BASELINE-v1-tracer-bullet.md`, `.planning/UAT-AUDIT-2026-09-09.md` (A7, A14
  diagnosis history — A14's "0.2% of steps" negligibility finding for classifier #1 is the
  precedent this research uses to argue #2 would hit the SAME fallback at ~100% instead)

### Tertiary (LOW confidence)

- `github.com/skfolio/skfolio/pull/255`, Wikipedia's "Deflated Sharpe ratio" page, and the
  Bailey–López de Prado SSRN abstract — found via one WebSearch this session, used only to
  confirm the DSR formula's existence and general shape (expected-max-SR + skew/kurtosis-adjusted
  PSR), NOT to derive the exact estimator this codebase should use (Open Question 3, Assumption
  A5) — the primary paper was not read in full this session.

## Metadata

**Confidence breakdown:**
- `canonicalize_states` landmine diagnosis and fix: HIGH — read the function in full, confirmed
  by grep that no test pins the fallback path, confirmed all 4 call sites' current arguments
- Classifier #2's method (reuse jump-model machinery): HIGH — directly stated in the ADR's own
  Deferrals section and D-13; verified the machinery's functions are already generic/parameterized
- Feature-porting scope and pitfalls: HIGH — read the exact candidate legacy functions in full,
  verified their dependencies are legacy-lib-import-free, verified the ratchet mechanism directly
- Dependence-statistic combination: HIGH — all three functions verified importable and executed
  against real data this session; the "which question does each answer" analysis is standard
  statistical reasoning, not novel
- Joint-tilt architecture gap: HIGH — read `allocation/tilt.py` in full, confirmed the
  single-probability-input signature directly; the recommended blend mechanism is a design
  proposal, not yet implemented or tested (MEDIUM confidence on the EXACT blend mechanism,
  HIGH confidence that SOME new code is required)
- Deflated Sharpe: MEDIUM — HIGH confidence that no implementation exists (grep-verified); MEDIUM
  confidence on the exact formula/estimator, since the primary paper was not read in full this
  session (Open Question 3)
- INV-01 ingestion: HIGH — all four candidate FRED series live-verified this session with exact
  dates/frequencies
- L2-degradation-window recommendation (Pitfall 3): MEDIUM — the mechanism is HIGH-confidence
  (verified code paths), but the RECOMMENDATION to route criterion 7 through L1-only labels
  rather than resolving L2's CV robustness is a judgment call this research makes, not a decision
  already locked in `07-CONTEXT.md` — flagged as Open Question 2 / Assumption A4 for
  discuss-phase confirmation

**Research date:** 2026-09-15
**Valid until:** short, matching wave 1's own posture — the registry's live trial count and the
current `monthly_raw`/`monthly_features` checkpoint contents are both time-sensitive; re-run the
direct checkpoint/registry reads in this document's Sources section before treating any specific
number here as current if more than a few days elapse before wave 2's plan executes.

## Project Constraints (from CLAUDE.md)

Two CLAUDE.md files govern this repo; `.claude/CLAUDE.md` (platform-specific) is authoritative
for every file wave 2 touches (all under `src/trading_crab_lib/platform/` or new files in that
tree). The root `CLAUDE.md` documents the separate, frozen legacy quarterly pipeline and does not
apply to platform code, EXCEPT as the source of the functions being ported (read-only reference).

**From `.claude/CLAUDE.md` (directly applicable):**
- `from __future__ import annotations` required at the top of every new/touched module — the
  legacy source files being ported (`momentum.py`) already have it; the new platform-native
  module must too.
- `X | None`, not `Optional[X]`; type hints on all public functions; `log =
  logging.getLogger(__name__)` per module; no bare `except:`; specific exception types only.
- Config sections read defensively via `cfg.get(...)`, never added to
  `_REQUIRED_PLATFORM_SECTIONS` — any new `labeling_2`/`taxonomy_2`-style config section for
  classifier #2 must follow this pattern, matching how `labeling` itself is read today.
- **`platform/` must not import from the legacy library beyond the existing 31 sites (the
  ratchet)** — the single most consequential constraint for wave 2's porting work (Common
  Pitfall 7).
- **GSD Workflow Enforcement** (root `CLAUDE.md`) — direct file edits must go through a GSD
  command (`/gsd-execute-phase` for this planned work); this research document does not itself
  edit any source file, consistent with that rule.

**From the root `CLAUDE.md` (legacy pipeline — read-only source for the port, not itself
touched):** `momentum.py` and `diagnostics.py` are quarterly-pipeline library modules following
that codebase's own conventions (functions-only, `from __future__ import annotations`, no
`print()` in library code) — these conventions are compatible with `.claude/CLAUDE.md`'s and
require no reconciliation; only the numeric window defaults need re-deriving for monthly cadence
(Common Pitfall 5), not the coding style itself.
