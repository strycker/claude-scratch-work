# Phase 7: Regime Representation - Context

**Gathered:** 2026-09-10
**Status:** Ready for planning — **WAVE 1 ONLY** (see D-09)

<domain>
## Phase Boundary

Decide what the regime labeler is allowed to see, prove that decision walk-forward, and
add a second, independent labeler on relative/leadership features so the platform can
inform allocation during ordinary markets — not only crisis avoidance. Covers **REG-01**
and absorbs **INV-01**.

**Two waves. Wave 1 is a gate; wave 2 does not start until it passes.**

**Wave 1 — resolve A13/A15 (in scope, planned now):**
- One documented feature policy shared by `backtest/driver.py` and `evaluation/report.py`,
  with a test that fails if they diverge (ROADMAP criterion 1).
- The §5.4 sojourn/lag ratio made interpretable, with its resolved-transition count
  (criterion 2).
- Post-fix labeling disagreement measured against the 82.8% pre-fix baseline (criterion 3).
- Classifier #1's ablation delta re-measured on **both** axes — `wealth_delta` and
  `dd_delta` (criterion 4).
- An ADR recording the policy choice and its rejected alternatives.

**Wave 2 — the leadership classifier (in scope for the PHASE, planned in a second pass):**
- Relative/invariant features built **natively in `platform/`** — ported, never imported
  (criterion 8).
- Classifier #2 fit unsupervised on a feature set disjoint from classifier #1's, with a
  disjointness test (criterion 5).
- Statistical dependence between the two labelings measured and reported (criterion 6).
- Joint (#1 × #2) allocation lift measured walk-forward against #1 alone, every
  configuration in the trial registry, deflated Sharpe applied (criterion 7).

**Out of scope (hard boundaries, from ROADMAP "Explicit non-goals"):**
- **No fitting to forward returns.** Classifier #2 is discovered unsupervised and
  validated walk-forward. A classifier fit on forward relative returns is a return
  predictor wearing a regime costume.
- **No raising K on classifier #1** and no (K, λ) sweeps — deferred to v2 (L1-V2-01).
- **No 2021+ holdout use for any selection decision.**
- **No migration work** — that is Phase 8, unaffected.

</domain>

<decisions>
## Implementation Decisions

### Feature policy — the A13/A15 resolution (wave 1)

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

### Gate semantics (wave 1 → wave 2)

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

### Classifier #2 — feature set and INV-01 (wave 2 direction)

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

### Joint labeling, dependence, and trial budget (wave 2 direction)

- **D-14: Two separate probability inputs — no product state space.** Both labelings'
  probability vectors feed the allocation tilt as separate inputs. Rationale: allocation
  already consumes probabilities, not labels — audit item **A7** found `active_regime` from
  hysteresis gates nothing; weights come from `vol_targeted_tilt(regime_probs, …)` in both
  the backtest driver and the weekly report. A product space would also thin badly (K₁ × K₂
  cells over ~590 decision months, with occupancy never uniform, so rare cells fall below
  the §4.4 5% floor). Keeping two inputs makes the lift-vs-#1-alone comparison a clean
  single-change ablation. — **Reversibility:** costly — switching to a product space later
  changes the allocation input contract and invalidates the lift comparison.

- **D-15: Dependence is reported with several statistics and NO pre-declared threshold.**
  Report adjusted Rand, Cramér's V and normalized mutual information side by side with the
  cross-tabulation. No pass/fail line — consistent with D-07's plausibility-only posture and
  with the proposal's "a number, not a goal". A human reads it and the judgement is recorded.
  A threshold was offered and declined on the grounds that it would have no empirical basis
  in this project yet.

- **D-16: Deflated Sharpe uses the WHOLE registry since project start.** `registry/trials.jsonl`
  holds **30 trials** as of 2026-09-10. Every configuration ever evaluated on this data
  contributed to the selection process that produced today's model, whether or not this phase
  ran it — that is why the registry is git-tracked as tamper-evidence. Counting only this
  phase's trials was offered and declined.

- **D-17: A trial ceiling is written into the ADR before running.** Expected count is roughly
  5 (2 wave-1 policy runs, 1 classifier-#2 fit, 1 joint, 1 #1-alone), putting the registry
  near 35 at phase end. Exceeding the ceiling requires an explicit amendment. This makes
  silent search-creep visible — the failure mode the registry can record but cannot prevent.

### Claude's Discretion

- **Whether L2's separate admission path is also frozen.** `driver.py::_cv_safe_active_features`
  (line 120) ALSO expands, but it governs the **nowcaster** (L2), not the labeler (L1), and
  it exists for a distinct reason — it narrows until `CalibratedClassifierCV` has
  `n_splits` examples of every class present. A13 is an L1 problem. Recommendation: leave L2
  alone in wave 1, note the parallel in the ADR, and let the researcher confirm the two paths
  are genuinely independent. Raised in discussion and consciously not pursued.
- Where the ADR physically lives (`platform_design/adr/`, `.planning/`, or the phase dir) and
  its numbering scheme.
- The exact form of the criterion-1 equivalence test (parametrized over decision dates vs. a
  single set-comparison assertion at each of a sampled few).
- Module layout for the new relative-strength code inside `platform/` (a new
  `platform/features/relative.py`, or extending `transforms_monthly.py`).
- Which credit aggregate series to use for INV-01, and its agency-tier alignment treatment.
- Report and plot layout for the pre/post table and the dependence cross-tabulation, within
  the existing `platform/plotting/` conventions.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase scope (authoritative for this phase)
- `.planning/PROPOSAL-phase-regime-representation.md` — **the full scope document**, named
  by ROADMAP Phase 7 as "Full scope". A13's diagnosed mechanism, the four candidate
  policies, wave 2's candidate feature list, the cost/risk table.
- `.planning/ROADMAP.md` § "Phase 7: Regime Representation" — goal, the 8 success criteria,
  explicit non-goals.
- `.planning/REQUIREMENTS.md` — **REG-01** and **INV-01** (both mapped to Phase 7);
  L1-V2-01 bounds what stays deferred to v2.

### Design (authoritative)
- `platform_design/platform_design.md` §5.4 — smoothed-vs-filtered gap, detection lag, and
  the median-sojourn / detection-lag ratio. The metric wave 1 makes interpretable.
- `platform_design/platform_design.md` §4.4 — jump-model acceptance criteria, incl. the 5%
  occupancy floor referenced by criterion 5 and by D-14.
- `platform_design/platform_design.md` §8.4, §8.7, §22 — deflated Sharpe against the trial
  registry (D-16), the no-regime ablation that produces `dd_delta`, and the (K, λ) selection
  protocol that D-13 deliberately does not invoke.
- `platform_design/platform_design.md` §9 — data span and the fast/slow/agency feature
  taxonomy that D-12's new series must be classified into.
- `platform_design/platform_design.md` §14 — tracer-bullet phase plan; the "beats nothing
  yet — that's fine" stance underwriting D-07.

### Analysis and baselines
- `.planning/BASELINE-v1-tracer-bullet.md` — the current trustworthy reference numbers that
  D-05's pre/post table is built against. **All prior recorded numbers are void.**
- `.planning/UAT-AUDIT-2026-09-09.md` — A13 (top open item, this phase's subject), **A7**
  (hysteresis gates nothing — the basis for D-14), **A11** (no gate fails on a bad model —
  left open by D-07), A8 (Brier's bounded range).
- `.planning/POST-2020-OBSERVATIONS.md` — the recorded-decision log required whenever a
  post-2020 observation changes a decision.
- `.planning/PROPOSAL-dual-regime-classifiers.md` — earlier framing of the two-classifier
  idea.

### Prior-phase context (constraints this phase inherits)
- `.planning/phases/06-platform-notebook-suite/06-CONTEXT.md` — **D-01** (fresh
  self-contained `platform/`, importing nothing from the legacy lib — the constraint behind
  criterion 8); the AMENDMENT sections carry the A13 surfacing decision.
- `.planning/phases/06-platform-notebook-suite/06-VALIDATION.md` — **the plausibility-band
  contract**. D-07 is an application of it. Read before writing any band.
- `.planning/phases/05-honest-backtest-evaluation/05-CONTEXT.md` — D-01 (Phase 5 is a
  diagnostic, not a gate) and D-02 (the no-regime ablation that produces both deltas).
- `.planning/phases/02-honesty-infrastructure/02-CONTEXT.md` — the frozen walk-forward
  interface and the one-`append_trial`-per-run convention D-16/D-17 rely on.
- `.planning/PROJECT.md` § Constraints — the **amended** honesty discipline: the fence is on
  fitting, not looking; post-2020 observations that change a decision are recorded with their
  date.

### Codebase (the files this phase changes)
- `src/trading_crab_lib/platform/backtest/driver.py` — `_window_active_features` (line 98,
  the expanding admission rule D-01 freezes), `_cv_safe_active_features` (line 120, L2's
  separate path — see Claude's Discretion), `_refit_l1` (line 176, reads
  `backtest.feature_min_history`).
- `src/trading_crab_lib/platform/evaluation/report.py` — `_reference_label_columns`
  (line 442, the frozen reference set D-02 adopts); `run_full_backtest_evaluation` step (d)
  at ~line 594 where `full_sample_states` is fit; **line 617, where that reference is
  reindexed onto the decision dates as `y_true`** — the third consumer D-05 must account for.
- `src/trading_crab_lib/platform/taxonomy.py` + `config/platform_settings.yaml` §`taxonomy`
  (line 232) — the 13-feature lean set; the `buffett_indicator` block comment at line 248 is
  the documented reason D-12 leaves market-cap out.
- `config/platform_settings.yaml` §`backtest` (line 375) — `feature_min_history: 120`,
  `min_train_months: 120`; §`labeling` (line 290) — K=5, λ=52.0, and the λ = 4 × n_features
  derivation note D-13 reuses.
- `src/trading_crab_lib/platform/honesty/registry.py` + `registry/trials.jsonl` — the
  append-only ledger; **30 entries** as of 2026-09-10 (D-16's denominator).
- `src/trading_crab_lib/platform/honesty/gating.py` — `FORBIDDEN_CENTERED_SUFFIXES`; every
  new wave-2 feature must pass this causal gate.
- `src/trading_crab_lib/platform/labeling/jump_model.py` — `fit_jump_model`,
  `canonicalize_states`; classifier #2 is a separate instance of this same machinery.
- `src/trading_crab_lib/platform/allocation/{tilt,hysteresis}.py` — `vol_targeted_tilt`
  consumes `regime_probs`; the A7 finding behind D-14.
- `src/trading_crab_lib/momentum.py`, `src/trading_crab_lib/divergence.py` — **PATTERN
  SOURCE ONLY, DO NOT IMPORT.** `compute_relative_strength`, `compute_rolling_correlation`,
  `compute_rolling_cross_correlation`, `compute_inflation_acceleration` are the algorithms to
  **port** into `platform/` for wave 2.
- `src/trading_crab_lib/platform/transforms_monthly.py` — `compute_lean_features` (line 227),
  where the 13 lean features are derived; the shape wave 2's new derivations follow.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `report.py::_reference_label_columns` — already computes exactly the 9-feature
  common-support set D-02 adopts. The driver can call this same function, so criterion 1's
  "one documented feature policy" becomes a shared call site rather than two parallel
  implementations that a test must police.
- `honesty/registry.py` `append_trial` and the frozen `run_walkforward` interface (Phase 2)
  — every policy trial and every classifier-#2 fit logs through the existing rail. No new
  registry mechanism is needed for D-03, D-16 or D-17.
- `labeling/jump_model.py` `fit_jump_model` / `canonicalize_states` — classifier #2 is a
  second instance of the same machinery with its own (K, λ), not new algorithm work.
- `evaluation/sojourn_lag.py` `compute_sojourn_lag_headline` / `build_filtered_probs_matrix`
  — already scores each transition against P(its own target state) rather than a
  class-agnostic max (Phase 5 review F1). Wave 1 changes what feeds it, not how it works.
- `platform/plotting/` (Phase 6, 11 submodules) — the pre/post table and the dependence
  cross-tabulation render through the existing conventions; P3 and P6 gain panels rather
  than being rewritten.

### Established Patterns
- **`platform/` imports nothing from the legacy library.** Re-verified this session:
  `grep` over all 59 `platform/*.py` modules returns zero legacy imports. Criterion 8 is
  currently TRUE and must stay true — wave 2's relative-strength code is ported, and the
  import-guard test (static AST import-graph closure, Phase 6 D-01, not `sys.modules`) is
  extended to the new modules.
- Functions-only library, `from __future__ import annotations`, type hints on public
  functions, `log = logging.getLogger(__name__)`, no `print()` in library code.
- New config sections read defensively via `cfg.get()`, never added to
  `_REQUIRED_PLATFORM_SECTIONS` (Phase 2/4 pattern).
- Every displayed number carries a stated plausibility band (`06-VALIDATION.md`), with the
  two historical failures pinned as regression cases that must RAISE.
- The strongest verification pattern in the project is Phase 3's DP-decode oracle test
  (proven identical to brute-force enumeration across 7 cases) — worth reaching for wherever
  a brute-force reference is affordable.

### Integration Points
- **Consumes:** `monthly_features` and `monthly_raw` platform checkpoints (dev tree,
  1962-01 → 2020-12, 708 rows); `config/platform_settings.yaml`; `registry/trials.jsonl`.
- **Produces (wave 1):** a changed feature-admission path in `driver.py`, an equivalence
  test, re-run backtest artifacts under `outputs/reports/platform/`, the pre/post table, and
  the policy ADR.
- **Produces (wave 2):** new `platform/` relative-feature module(s), new M2/credit
  ingestion, classifier #2's labeling, the dependence report, and the joint-lift comparison.
- **Touches, and therefore re-dates:** every Phase 5 evaluation artifact. This is expected
  and is what D-05's table exists to make legible.

### Data facts verified this session (2026-09-10)
- Lean set is **13**; common-support from 1972-01 is **9**; the 4 excluded are
  `curve_10y2y` (1976-06), `gold` (1985-02), `oil` (1985-02), `fred_vix` (1990-01).
- Dev `monthly_features`: 1962-01 → 2020-12, 708 rows. Holdout carve is applied at rest.
- `monthly_raw` has `equities_tr`, `long_duration_tr`, `cash`, `oil`, `fred_cpi`,
  `fred_gs10`, `fred_tb3ms`, `fred_baa`, `fred_aaa` all from **1962-01/02** — enough for
  stock–bond rolling correlation, equity/bond relative strength, oil/equity relative
  strength, CPI acceleration and curve changes, all full-history.
- **No M2, no Wilshire/market-cap, no credit aggregate** in `monthly_raw` or in
  `config/platform_settings.yaml`. D-12's M2 + credit ingestion is genuinely new work.
- ⚠ **Discrepancy for research to confirm:** `oil` in `monthly_raw` starts **1962-01**, but
  the `oil` FEATURE in `monthly_features` starts **1985-02**, despite
  `compute_lean_features` being a direct passthrough. Most likely `monthly_features` on disk
  predates a `monthly_raw` rebuild. This matters — if `oil` genuinely has 1962+ coverage it
  may belong in the frozen 9 after a rebuild, which would change D-02's set. **Resolve before
  finalizing the frozen set.**
- `registry/trials.jsonl`: **30 entries**, keys `config_hash`, `config`, `features`,
  `metrics`, `git_sha`, `timestamp`.

</code_context>

<specifics>
## Specific Ideas

- **"A policy disagreement, not a defect."** A13's framing matters and should survive into
  the ADR. The driver and the report were each internally correct; §5.4 simply compared two
  estimators fit on different feature spaces. The fix is a decision, not a bug fix, which is
  why it earns an ADR rather than a patch.
- **The 82.8% is a baseline, not a target.** Criterion 3 sets no goal deliberately — a
  number, not a goal, to avoid fitting to it. The post-fix figure is wave 1's headline
  result whatever it is.
- **Classifier #2 must not be fit to outperformance.** The proposal's sharpest line, worth
  quoting in wave 2's plan: a classifier fit directly on forward relative returns "is a
  return predictor wearing a regime costume, and the deflated Sharpe will correctly destroy
  it." Discovery unsupervised; validation walk-forward.
- **High dependence is a falsifiable failure, and is reported as one.** If classifier #2
  turns out to have rediscovered the stress axis, that is the honest finding — recorded as a
  failure to add an axis, not softened.
- **The stock–bond rolling correlation is the single most important missing variable**
  (proposal, wave 2 item 1). It changed sign around 2000 and again around 2021, and it
  governs whether bonds diversify equities at all. Full-history buildable from
  `equities_tr` and `long_duration_tr`.

</specifics>

<deferred>
## Deferred Ideas

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

</deferred>

---

*Phase: 7-Regime Representation*
*Context gathered: 2026-09-10*
