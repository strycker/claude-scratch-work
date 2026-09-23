---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
current_phase_name: Regime Persistence & Stability
status: executing
stopped_at: Phase 8 wave 1 complete (5/10 plans); wave 2 (08-06, 08-07) next
last_updated: "2026-09-23T00:00:00.000Z"
progress:
  total_phases: 9
  completed_phases: 7
  total_plans: 57
  completed_plans: 52
current_phase: 8
last_activity: 2026-09-23
last_activity_desc: "Phase 8 WAVE 1 COMPLETE — 08-01..08-05 executed. A11 ANSWERED b-promote-dsr (registry rows spent: 0; criterion 7 FAILED retroactively on both l1only legs). Track A terminal-month edge artefact REFUTED — #1 churn flat in k (242-249/587 for k=1..6). l1only bit-reproducible. Suite 2150 passed, 0 skipped. Registry 42/44. Ratchet 31."
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-07-09)

**Core value:** Honest, regime-aware weekly guidance that beats buy-and-hold SPY net of
avoided drawdowns — never fooled by its own backtest.
**Current focus:** Phase 7 — Regime Representation (approved, not yet discussed/planned)

## Current Position

Phase: **8 — Regime Persistence & Stability — EXECUTING**
Status: **WAVE 1 COMPLETE (5 of 10 plans).** Next: wave 2 — `08-06` (Bayes filter + signed
detection offset, dep 08-01) and `08-07` (§4.4 criterion 3 run, dep 08-03). Both autonomous.
Full suite **2150 passed, 0 skipped, 0 failed**; ruff/flake8 clean; both wheels build and ship
`evaluation/churn.py` + `labeling/stability.py`. Registry **42 of 44**. Ratchet **31**.

**Requirements closed by wave 1:** PER-01 (08-01), PER-04 (08-02), PER-08 (08-05), PER-09 (08-04).
**Partially delivered, completed later:** PER-03 (08-08), PER-06 (08-07 runs 08-03's machinery),
PER-10 (08-01 did the F-4 half; 08-10 pins the suite count).

**What wave 1 established, as measured:**

- **A11 ANSWERED — `b-promote-dsr`** (Glenn, 2026-09-22), taken *before* 08-01/08-02 produced
  any number, so the pre-registration claim stands unqualified. `registry rows spent: 0`.
  **Criterion 7's MET becomes FAILED retroactively** on both legs of the l1only routing —
  DSR 2.28e-12 (baseline) and 1.47e-11 (joint) against hurdle 2.208694. ADR-0003 Accepted.
  The gate is decided and tested but **not yet wired**: 08-10 must add it to `joint_lift_table`
  before re-measuring criterion 7.
- **Track A's terminal-month edge artefact is REFUTED** by criterion 3's pre-registered rule.
  Classifier #1 churn is flat in k: **246 / 242 / 245 / 248 / 247 / 249** of 587 pairs for
  k = 1…6 (1972-01-31 → 2020-12-31); #2 is 24–25. k=1 anchored to `state_1` at 0/588
  mismatches. The 41.91% is the labeler's own path, not its edge. **The lever is λ, and a λ sweep
  is not authorized** — so classifier #1's churn is not fixable in this phase, and **08-09's
  bounded turnover is the only mechanism left that can move the decision-bearing leg.**
  Not ruled out: edge effects longer than 6 months; λ/d not isolated from K, features and d.
- **Both churn series exist and are separately denominated** (08-01). l1only: Track A ≡ Track B
  (identity pinned True). l2: #1 Track B **221/487 = 45.38%**, #2 **66/487 = 13.55%** (100
  degraded), identity pinned False. Max posterior < 0.70 in **355/488** rows for #1.
- **l1only is bit-reproducible** — curves byte-identical to HEAD; criterion 7 reproduces to the
  last digit. 08-10's 1e-12 reproduction gate is safe.

---

### Previously: Phase 7 — Regime Representation — CLOSED 2026-09-21
Status: **PHASE CLOSED.** 12 of 12 plans executed. ADR-0002 **Accepted 2026-09-21**; all eight
probe edges resolved with a named test each; legacy-import ratchet re-measured at **31**
(unchanged, constant untouched); full suite **1983 passed, 0 skipped, 0 xfailed**.

**Requirements, claimed at exactly the width of the evidence:**

- **INV-01 — DELIVERED IN FULL.** All four clauses closed by artifact (see ADR-0002
  § Requirement coverage). Claimed at the strength of the evidence and no further: the era
  screen was *survived*; a PC1 loading of exactly 1/√2 on two standardized candidates is an
  arithmetic identity, not by itself evidence of five-decade economic stability.
- **REG-01 — DELIVERED PARTIALLY.** The open item is **named, not folded in: criterion 6's
  dependence verdict is UNRESOLVED.** The pre-registered block-permutation control returned
  **INCONCLUSIVE** — NMI **0.464088** at the **96.60th percentile** against p95 **0.449202** and
  p99 **0.501195**, `n_compared` = **695 months, 1963-02-28 → 2020-12-31**. The pre-registration
  at `298b1bc` forbids a tie-break; **no further dependence statistic may be computed.**
  **No independent second axis is established by this phase.**

**The three criteria, as measured:**

| # | Verdict | The number, with its window |
|---|---|---|
| 5 — occupancy & disjointness | **MET** | Classifier #2 occupancy **16.6667 / 22.7011 / 22.4138 / 23.8506 / 14.3678 %** over **696 months, 1963-01-31 → 2020-12-31**, sum error **0.0**, all inside §4.4's real two-sided **~8–35%** band (there is **no** five-percent floor — misquote corrected 2026-09-17). Disjointness from the lean 13 asserted on the *resolved* frozen eight |
| 6 — dependence | **UNRESOLVED** | ARI 0.354841 / **NMI 0.464088** / Cramér's V 0.589748, `n_compared` = 695, 1963-02-28 → 2020-12-31; control 2000 resamples, seed 20260918 → **INCONCLUSIVE** |
| 7 — joint lift | **MET as a measurement; negative on wealth** | `wealth_delta` **−0.123438 nats** (≈ **0.8839×**, an **11.61% terminal-wealth shortfall**) and `dd_delta` **+0.024084** (**2.41pp shallower**, 40 vs 47 months underwater), both over **588 steps, 1972-01-31 → 2020-12-31**, 0 degraded steps. **Neither leg's DSR clears the multiple-testing hurdle** at 42 trials |

**The drawdown improvement does not offset the wealth loss and is not recorded as doing so.**
D-06 makes honest measurement the gate, not the sign — which is why criterion 7 is met and why
the number is quoted rather than framed.

**Open items carried into the record** (full list in ADR-0002 § Deferrals and open items at
acceptance): criterion 6 unresolved, no tie-break permitted; §4.4 criterion 3's Hungarian
subsample-stability test **never run for either classifier**; ADR-0001 condition (iv)'s
**covariance clause unimplemented** — no per-regime covariance exists at L4-01, it falls to L3
(design §6.2) — with `vol_targeted_tilt` and `driver.py:497` still (iv)-non-compliant for
consumers other than the joint harness; classifier #1's **filtered** labeling changing state in
**246 of 588 decision months (41.84%)** against a 3.60% full-sample rate, governed by **no
band**, feeding the tilt directly; classifier #2's §5.4 ratio **1.074** (median sojourn 29.0 mo,
median detection lag 27.0 mo, only **5 of 12** transitions resolved); the crisis state's **3.0
month** median sojourn exactly on criterion 2's boundary against a 1–3 month detection lag, so
criterion 7's `dd_delta` is **not** evidence crises are nowcastable in time to act;
`DEGENERATE_SHARPE_VARIANCE = 1.0` still a declared assumption governing every DSR until 20
independent Sharpe-bearing trials exist; the L2 CV-robustness question routed around, not
resolved; and **audit item A11 open by deliberate choice.**

---

### Previously: wave 2's plan table (8 plans, `07-05`…`07-12`, checker PASS 2026-09-15) — all executed

| Plan | Wave | Depends | Auto | Delivers |
|---|---|---|---|---|
| 07-05 | 1 | — | yes | **tracer** — `sort_column` fix (raises), relative-features port, M2SL/TOTALSL config |
| 07-06 | 1 | — | yes | deflated Sharpe from scratch + `total_trial_count` reading the header |
| 07-07 | 2 | 05 | yes | INV-01 screen — PCA discovery-only, era stability, named survivors |
| 07-08 | 3 | 05,07 | **no** | `checkpoint:decision` pinning 6 constants · ADR-0002 Proposed · classifier #2 fit |
| 07-09 | 4 | 08 | **no** | criterion 6 — ARI/NMI/Cramér's V, no gate · human-verify |
| 07-10 | 4 | 08 | **no** | `blend_regime_tilts` · human-verify confirming the four `[ASSUMED]` bands |
| 07-11 | 5 | 06,09,10 | yes | criterion 7 — joint lift + DSR, window inline |
| 07-12 | 6 | 07,09,11 | yes | ADR-0002 Accepted · 8 probe edges · REG-01 + INV-01 claimed |

**Both requirements are now CLAIMED — but at different widths, which the plan table above could
not anticipate.** Wave 1's plans carried `deferred_requirements: [INV-01]`; that deferral ended
here. **INV-01 is claimed in full; REG-01 is claimed PARTIALLY** because criterion 6's verdict
came back inconclusive under its own pre-registered rule. **Three blocking checkpoints** were
taken as planned: 07-08 decision, 07-09 and 07-10 human-verify.

**The sequencing that matters** (verified, not asserted): the `canonicalize_states` `sort_column`
fix is wave **1**; the first classifier-#2 fit is wave **3**. The band-confirmation checkpoint is
wave **4**; the first joint-lift number is wave **5**. Both gaps are structural, not conventional.

---

### Previously: **PHASE-WAVE 1 CLOSED.** Executed 4/4, verified 4/4 criteria by live re-derivation

(`07-VERIFICATION.md`), validated `nyquist_compliant: true` (`07-VALIDATION.md`), UAT signed
**accept-with-caveats** 2026-09-15 (`07-UAT.md`, 3/3 items closed). Suite **1752 passed, 0 skipped**. Wave 2 remains deliberately unplanned per D-09
and needs its own `/gsd-plan-phase 7` pass. Suite 1705 → **1735 passed, 0 skipped**.

### What phase-wave 1 delivered

| Criterion | Outcome |
|---|---|
| 1 — one documented feature policy, test fails on divergence, ADR | ✅ `frozen_l1_features` threaded from ONE `_reference_label_columns` call; `TestFrozenPolicyEquivalence`; `platform_design/adr/0001-l1-feature-policy.md` |
| 2 — §5.4 ratio interpretable, with resolved-transition count | ✅ **0.60145 (7/7 resolved)**, both labelings on one feature space; A13 caveat retired by cause (D-08), not softened |
| 3 — post-fix disagreement vs the 82.8% baseline | ⚠️ **80.90% (288/356)** vs 82.77% (389/470) — satisfied as worded, but see the named limitation below |
| 4 — ablation delta on both axes, in band | ✅ `wealth_delta` +0.377847, `dd_delta` −0.066124; both in band (both bands `[ASSUMED]`) |

**D-02-A landed:** `monthly_features` recomputed offline from cached `monthly_raw` — `oil`
431 → **708** non-NaN dev months, frozen set 9 → **10**, shape `(708, 53)`, holdout fence intact
(68 rows from 2021-01). A13 golden constant re-pinned `4,6,8,9,10,12,13` → `5,7,9,10,11,12,13`,
still exact list equality.

### ⚠ Named limitation on criterion 3 — human-approved with this condition attached

The frozen policy produced **232/588** L2-degraded steps versus **118/588** pre-fix, so the
post-fix disagreement rests on **356** steps ending **2017-05** against a baseline of **470**
ending **2020-12**. `82.77% → 80.90%` is **not** a clean 1.9-point improvement — different
sample, different window. Foregrounded inline (same table cell as the numbers) in both
`backtest_report.md` and ADR-0001, per the human sign-off at 07-03's checkpoint.

Verified, not inferred: the degrade is L1-label-mediated, **not** a wiring defect —
`frozen_l1_features` reaches only `_refit_l1` (`driver.py:461`, `:473`); `_refit_l2`
(`driver.py:266-271`) takes no such parameter. **Why** freezing L1 nearly doubles L2's degrade
rate is inferred (occupancy shifted, state 0: 1.6% → 11.51%), **not measured** — do not cite a
cause as established.

### ⚠ Criterion 8's "platform is fully decoupled" was FALSE — corrected 2026-09-15

Found by the wave-1 verifier. `MIGRATION-PLAN.md`'s decoupling exit check ended in
`| grep -v platform`; every match line begins with a path containing `platform`, so the filter
discarded **every** violation. The check returned nothing whether or not the code was decoupled
— **it could not fail** — and that false negative is what backed ROADMAP criterion 8's
"Verified 2026-09-10: platform is currently fully decoupled."

An AST scan finds **31 real legacy import sites** in `platform/`: bare `trading_crab_lib` ×16
(mostly `ROOT`/`OUTPUT_DIR`), `.checkpoints` ×7, `.ingestion.http` ×3, `.ingestion.browser` ×2,
`.ingestion` ×1, `.ingestion.assets` ×1, `.email` ×1. **All predate Phase 7; wave 1 added none**
(confirmed by `git blame`). Vendoring them is `MIGRATION-PLAN.md` P0 / Phase 8 criterion 1.

**Wave 2's "port, don't import" instruction still stands** — but because the coupling must not be
*widened*, not because `platform/` is already clean. Now guarded by
`tests/unit/test_platform_legacy_import_ratchet.py`: an AST ratchet pinned at 31 that may only
decrease, plus an allowlist check that fails on a new seam, plus a test that fails if anyone
restores the unfalsifiable grep. ROADMAP criteria 7.8 and 8.1 and `MIGRATION-PLAN.md` all
corrected.

### ✅ Trial registry contamination — FIXED 2026-09-15

The registry stands at **42** rows. Rows 35–38 are this phase's real policy trials
(`P7-W1-frozen-10col` ×2, `P7-W1-impute-13col-REJECTED` ×2). **Rows 39–42 are UNTAGGED and are
not policy evaluations** — they were appended by two end-to-end wiring-verification runs of
`python -m trading_crab_lib.platform.evaluation.report` during 07-04 (the plan's own
`<verification>` section anticipated this).

**Resolved (user-approved: archive + reset, keep count).** The ledger was archived **intact** to
`registry/archive/trials-pre-P7W1-reset.jsonl` (42 rows, nothing deleted) and restarted with a
single provenance header carrying `prior_genuine_trials=38`, `discarded_smoke_rows=4`, and an
explicit note that **D-16's project total = 38 + rows appended after the header**. A bare
post-reset row count must not be read as the project total.

Three structural fixes so it cannot recur:

- `append_trial` **refuses** to persist a row without a non-empty `trial_tag` (blank and
  whitespace rejected; a refused write leaves no file).

- New `NO_REGISTRY` sentinel: smoke runs build the row, log the skip, return `written=False`,
  write nothing — and are exempt from the tag requirement, because a smoke run is not a trial.

- The report CLI now **requires** `--trial-tag` or `--smoke` (mutually exclusive). The exact
  invocation that caused this can no longer be typed; bare `python -m ...report` exits non-zero.

- `run_backtest` / `fit_nowcaster` / `run_walkforward` default to naming their own call site, so
  every persisted row is attributable without breaking callers.

| Plan | Wave | depends_on | Tasks | Autonomous | Covers |
|---|---|---|---|---|---|
| 07-01 | 1 | — | 2 (leads `type="tracer"`) | yes | criterion 1 — frozen policy threading, equivalence test |
| 07-02 | 2 | 07-01 | 3 | yes | D-02-A recompute, pre-fix evidence, A13 re-pin |
| 07-03 | 3 | 07-01, 07-02 | 3 (one `checkpoint:human-verify`) | **no** | criteria 2–4 — disagreement, policy trials, measurements |
| 07-04 | 4 | 07-01..03 | 3 | yes | three-state report, A13 caveat rewrite, ADR-0001 |

Phase 7 gates internally: **wave 1 (resolve A13/A15) must pass before wave 2 (the
leadership-axis classifier) starts.** After wave 1's numbers land, run `/gsd-plan-phase 7`
again to plan wave 2 with real numbers in hand (D-09).

**Requirement scoping:** `requirements: [REG-01]` (PARTIAL — feature-policy clauses only);
`deferred_requirements: [INV-01]` (entirely wave 2, D-09 cited). Recorded as a decision, not
an omission.

**Resume file:** None

### ⚠ Carried into execution — three things that are not settled

1. **Four `[ASSUMED]` plausibility bands are load-bearing and unlocked**: `abs(wealth_delta) < 5`,
   `dd_delta ∈ [-0.5, 0.5]`, `n_transitions > 30` implausible, `pct_disagree < 0.02` suspicious.
   Every plan depending on one labels it provisional and uses it as an advisory flag that
   triggers a recorded note — **never a hard gate** (D-07). Confirm or revise before trusting
   any verdict that rests on one.

2. **The decision-coverage gate cannot parse `07-CONTEXT.md`.** `check decision-coverage-plan`
   returns `total: 0, "no trackable decisions"` and therefore `passed: true` — a **skipped gate
   reporting a pass**, not a verification. D-10's title wraps across two lines before its closing
   `**` (unlike every other bullet), but `total: 0` means the parser finds no decisions at all,
   so the format mismatch is broader than that one bullet. Coverage was confirmed by direct
   citation count instead: all nine wave-1 decisions appear across the plans (D-09 34×, D-02 15×,
   D-02-A 13×, D-04 12×, D-08 12×, D-03 8×, D-01 7×, D-05 4×, D-07 4×, D-06 3×). **Do not read a
   future green from this gate as evidence until the parse is fixed.**

3. **The first plan-checker run (haiku) returned a PASS that was not trustworthy** — it declared
   `07-VALIDATION.md` and `07-PATTERNS.md` absent (both exist and are committed), skipped the
   Nyquist and Pattern dimensions on that false premise, and scored coverage against an invented
   criteria table that marked disagreement measurement as "deferred to wave 2" when it is
   criterion 3 and squarely wave 1. Re-run on sonnet with a prompt requiring file-existence
   proof and verbatim criteria quoting: **PASS with one WARNING** (since closed). The recorded
   verdict for this phase is the sonnet run, not the haiku one.

---

### Phase 6 — Platform Notebook Suite (CLOSED 2026-09-10)

Executed 7/7, verified 5/5 criteria (`06-VERIFICATION.md`, status `human_needed` for two
deliberate human items), UAT closed 2/2 (`06-UAT.md`). The fresh-run item was settled by
executing all six notebooks from cleared state via `nbclient`; the P3 cold-start sign-off
was recorded as **accept-with-caveats** — the regimes are *crisis* regimes, not *allocation*
regimes. That caveat is the direct motivation for Phase 7's wave 2.

The as-executed plan map, retained because Phase 7 builds on these modules:

| Plan | Notebook | Wave | depends_on | New module |
|---|---|---|---|---|
| 06-01 | P1_data_spine | 1 | — | `plotting/{core,loaders,data,drift}.py` |
| 06-02 | *(artifact persistence)* | 1 | — | additive write in `evaluation/report.py` |
| 06-03 | P3_regime_labeling | 2 | 06-01, 06-02 | `plotting/{history,regime}.py` |
| 06-04 | P2_features_taxonomy | 2 | 06-01 | `plotting/features.py` |
| 06-05 | P4_nowcaster | 2 | 06-01 | `plotting/nowcaster.py` |
| 06-06 | P5_assets_allocation | 2 | 06-01 | `plotting/allocation.py` |
| 06-07 | P6_backtest_evaluation | 2 | 06-01, 06-02 | `plotting/backtest.py` |

19 tasks. Zero new dependencies. The five wave-2 plans are genuinely parallel — verified no
`files_modified` overlap. That is by design: `plotting/__init__.py` re-exports **only** shared
constants and loaders (created once in 06-01); per-layer plot functions are deliberately NOT
barrelled and are imported by submodule path, so the per-layer plans never contend for that
one file. The reason is recorded in the module docstring.

**Planning artifacts:** `06-RESEARCH.md` (900 lines), `06-VALIDATION.md` (the plausibility-band
contract), `06-PATTERNS.md` (analog map, 15/19 files matched), `06-CONTEXT.md` AMENDMENT 3.

**Two research open questions settled as AMENDMENT 3 before planning:**

- **(H) A13's filtered path — persist a new artifact.** Two of three A13 ingredients are cheap
  (reference labeling 0.62s; the 588-step active-feature-count timeline 1.2s, which reproduced
  the audit's exact change points 4→6→8→9→10→12→13 and independently corroborates A13). The
  third needs a full `run_full_backtest_evaluation()`. Decision: extend
  `platform/evaluation/report.py` to additionally persist `full_sample_states` and
  `filtered_state_probs`. **Additive write only — every existing Phase-5 artifact must be
  byte-identical**, and 06-02 Task 2 byte-compares all seven to prove it. `full_sample_states`
  was already computed in-function (report.py:593); it only needed adding to the existing
  `artifacts` dict. Written once by 06-02, read by both P3 and P6.

- **(I) P2 has no causal-vs-centered panel.** `transforms_monthly.py` has zero occurrences of
  center/centered/causal, and `honesty/gating.py` defines
  `FORBIDDEN_CENTERED_SUFFIXES = ("_centered","_c5","_zerophase")` and raises on sight. The
  platform did not inherit legacy ADR #1's split. P2 says so in prose instead.

**Corrections to CONTEXT.md found during research (CONTEXT.md is stale on these):**

- The **Faber / 60-40 KPI anomalies are already fixed.** The live artifacts show Faber
  6.3726/−18.94% and 60/40 5.0471/−26.96%, matching BASELINE. The −99.7% / −2.3% figures
  CONTEXT.md told P6 to surface are void; P6 narrates the fix history from live values instead.

- **`regime_labels` / `regime_confidences` / `regime_profiles` do NOT exist on disk**
  (contradicting Amendment 1 item C). P3/P4/P5 call `label_regimes()` themselves via
  `loaders.compute_regime_labeling`, routed at a scratch checkpoint namespace so no notebook
  can write `data/checkpoints/platform/`.

- **Brier is bounded [0,1] here, not the textbook [0,2]** — `compute_brier_multiclass` computes
  `mean(diff²)` over the full (n,K) array. The K=5 no-skill floor is (K−1)/K² = 0.16 and the
  observed 0.2087 sits **above** it, so the metric cannot currently claim the nowcaster beats
  random guessing. P4 surfaces this; fixing it is explicitly out of scope. Independently
  corroborates audit item **A8**.

**The plausibility contract (D-11 reversed).** Every task that displays a number carries a
stated numeric band. 06-01 Task 2 builds the bands as pure functions and pins the two
historical failures as regression cases that must RAISE: a 60/40-shaped leg at −2.27% max DD
fails its **domain per-leg** band (the universal [−1,0] bound does not catch it — that is the
whole lesson), and terminal log wealth 111.06 fails the universal `abs(x) < 10` band.

Phase 6 history: planned 2026-09-09 (research → validation strategy → pattern map → 7 plans
→ checker PASS), executed and closed 2026-09-10.

### ⚠ UAT audit outcome (2026-09-09) — `.planning/UAT-AUDIT-2026-09-09.md`

Phases 1 and 5 were signed off against arithmetically impossible output (Phase 5's
closure record tabulates terminal log wealth of 111.06 — e¹¹¹ ≈ 10⁴⁸ — as an
*improvement*). Neither phase is re-opened as failed: **every criterion is satisfied as
phrased**. What is void is the recorded evidence and the conclusions drawn from it.

Restated scores: **Phase 1 — 2 of 5 genuinely verified, 3 certified over invalid
evidence. Phase 5 — 1 of 4 genuinely verified, 3 wiring-verified with void values.**

**The systemic finding:** every criterion in both phases is satisfied by evidence of one
of four shapes — existence, shape, pure-function correctness, or placement. None can
detect a physically impossible value. `gsd-tools query audit-uat` returns 0 items for
the same reason: it only surfaces pending/skipped/blocked, never
"passed-on-evidence-that-no-longer-holds". Audit item **A3** proposes a plausibility
gate as a standing criterion for every phase emitting numeric output.

**A4 — found by applying that very lens, now CLOSED EMPIRICALLY (`976f7c8` + `8095498`).**
ALFRED point-in-time vintages of *rebased index* series are not level-comparable across
rebasings, and the pre-vintage fallback compounded it by picking each period's
first-published value from whichever vintage happened to be earliest. Verified on the
2026-09-09 rebuild: zero discontinuity warnings, `real_rate_level` −4.91…9.27 (+3.38
mean through 1981, −3.26 through 1974 — economically correct), occupancy matching the
repair experiment exactly.

**The regime structure is now legible**: 1973-09 oil shock, 1981-10 Volcker, 1988-09
disinflation, 1996-08 late-90s expansion, **2008-07 → the 1.6% crisis state (GFC)**,
2009-06 recovery. Before the fix one state held 49.6% and another 1.7%.

**A13 is now the top open item.** The driver's active feature set changes **7 times**
across the backtest (4→6→8→9→10→12→13) while the report's reference is a fixed 9
columns. §5.4's detection lag measures how long the filtered path takes to agree with
that reference, so their disagreement is not purely detection delay — which is why the
current 164-month lag / 0.591 ratio is **not interpretable**. A14 was measured at 1 of
588 steps (0.2%) and closed as negligible.

### First trustworthy baseline — `.planning/BASELINE-v1-tracer-bullet.md`

All prior recorded numbers are void. **Current reference (2026-09-09, both CPI defects
fixed):** strategy 4.0265 log wealth / −21.24% max DD (33 mo underwater), ablation delta
**+0.3793** (5× the original baseline), CVaR −0.0463, turnover 0.0734, Brier 0.2087,
crisis down-capture .12/.83/.05/.10. Still **last of five legs**; Faber 6.3726 / −18.94%
still beats it on both §23.1 dimensions. The §5.4 ratio (0.591) is **not interpretable**
pending A13.

### Phase 6 discussion outcome (2026-08-04)

The notebooks were reframed during discussion, and the reframe reached up into the
project constraints:

- **Purpose:** periodic verification & validation (are the regimes still well-defined?
  is the data still behaving as historic?), **not** a per-run human gate. One cold-start
  selection gate at `P3_regime_labeling`; the other five carry no gate.

- **Honesty framing amended** in `PROJECT.md`: the fence is on *fitting*, not *looking*.
  Notebooks read the full span (incl. post-2020) via the explicit
  `get_holdout_checkpoint_manager()` opt-in — which `holdout.py` already documented as
  the live-scoring path. Fitting stays fenced at 2020-12. Post-2020 observations that
  change a decision are recorded with their date. The prior "firewalled from all
  selection decisions" wording was unworkable: catching feature decay *is* a selection
  decision informed by recent data.

- **`ROADMAP.md` criterion 2 rewritten** (+ new 2b on the holdout opt-in) and **NB-01
  reworded** in `REQUIREMENTS.md` to match.

- **Scope held:** no tuning ((K,λ) sweeps stay deferred to v2 per Phase 3 D-02), no
  report wiring, no package restructuring (Phase 7), no holdout-carve repair (separate).

### ✅ RESOLVED (2026-09-08) — fitting is now fenced at rest

*Was:* `data/holdout/` did not exist, the dev-tree `monthly_features` carried
post-cutoff rows to 2026-08, and nothing outside `tests/unit/test_platform_holdout.py`
called `write_monthly_features_split()` or `assert_dev_checkpoint_within_boundary()`.
A tested mechanism that no production path invokes is not a fence.

*Now:* `build_monthly_spine()` writes through the split, and
`scripts/build_platform_data.py` calls `assert_dev_checkpoint_within_boundary()`
and **fails the build** on violation. Applied to the checkpoint on disk: dev
1962-01 → **2020-12** (708 rows), holdout holds the 68 post-cutoff rows.

New `honesty.holdout.load_full_span()` is the explicit *looking* opt-in (dev +
holdout concatenated), matching the amended framing. `report/weekly.py` live
scoring now uses it — **required, not cosmetic**: it scores
`monthly_features.iloc[[-1]]`, so carving without it would have scored December
2020 as "today" every week. Three tests pin the wiring specifically and fail
against the unwired code.

Progress: [███████▌░░] 75% (6 of 8 phases; phase 7 in flight, 11 of its 12 plans done)

### Roadmap restructure (2026-08-04)

Phase 6 was "Migration to Public Repo". It is now split and extended:

| Phase | Was | Now |
|---|---|---|
| 6 | Migration to Public Repo | **Platform Notebook Suite** (NB-01) |
| 7 | — | **Migration to Public Repo** (MIG-01) |
| 8 | — | **Invariants & Dimensional Reduction** (INV-01) |

Rationale: the migration's per-step validation gate is "run the notebook and verify",
but the platform has **zero** notebooks — all 12 in `notebooks/` cover the legacy
quarterly pipeline. Notebooks are a prerequisite, not a nice-to-have. Full analysis in
`.planning/STATUS-REVIEW-2026-08.md`.

## Performance Metrics

**Velocity:**

- Total plans completed: **47 of 47** written (7 + 5 + 4 + 5 + 7 + 7 across Phases 1–6, plus
  **12 of 12** in Phase 7).

- Full test suite: **1983 passed, 0 skipped, 0 xfailed** (measured 2026-09-21 at Phase 7
  closure, branch `claude/keen-galileo-zqcml6-w4`). Previously: 1705 collected, 0 skipped
  (2026-09-14 on `main` @ `c6605a4`).

**By Phase:**

| Phase | Plans | Status |
|-------|-------|--------|
| 1 — Monthly Data Layer | 7/7 | Verified passed (FRED_API_KEY verified 2026-07-23 — human item cleared) |
| 2 — Honesty Infrastructure | 5/5 | Verified passed 5/5 |
| 3 — Regime Labeling & Prediction | 4/4 | Complete 2026-07-22 |
| 4 — Asset Prediction & Allocation | 5/5 | Complete 2026-07-23 |
| 5 — Honest Backtest & Evaluation | 7/7 | Complete 2026-07-27, closed 2026-08-04 |
| 6 — Platform Notebook Suite | 7/7 | Verified 5/5 + UAT closed 2026-09-10 |
| 7 — Regime Representation | 12/12 | **CLOSED 2026-09-21** — INV-01 in full; **REG-01 PARTIAL** (criterion 6 UNRESOLVED) |
| 8 — Migration to Public Repo | 0/TBD | Not started |

⚠ Phases 1 and 5 carry the UAT-audit caveat below: criteria satisfied as phrased, recorded
numeric evidence void. See `.planning/UAT-AUDIT-2026-09-09.md`.

**Per-Plan Metrics:**

| Plan | Duration | Tasks | Files |
|------|----------|-------|-------|
| Phase 05 P01 | 3min | 2 tasks | 5 files |
| Phase 05 P02 | 12min | 3 tasks | 2 files |
| Phase 05 P03 | 18min | 2 tasks | 4 files |
| Phase 05 P04 | 15min | 2 tasks | 2 files |
| Phase 05 P05 | 22min | 3 tasks | 2 files |
| Phase 05 P06 | 7min | 3 tasks | 2 files |
| Phase 06 P01 | 55min | 3 tasks | 11 files |
| Phase 7 P08 | 1 session | 3 tasks | 6 files |
| Phase 07 P11 | 1h05m | 3 tasks | 6 files |
| Phase 07 P12 | 35min | 2 tasks | 4 files |

*Durations were not recorded for Phase 05 P07 or Phase 06 P02–P07.*
*Updated after each plan completion.*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- Roadmap: Data layer (Phase 1) precedes Honesty infrastructure (Phase 2) — HON-01
  (holdout carve) and HON-06 (causal-feature gating) need Phase 1's files and feature
  taxonomy to operate on; both still land before any modeling phase per design §14.

- Roadmap: L1 (labeling) + L2 (prediction) merged into one phase (3); L3 (asset
  prediction) + L4 (allocation & report) merged into one phase (4) — these are tightly
  coupled steps of the same tracer-bullet vertical slice (design §14 Phase 1).

- Roadmap: MIG-01 kept as its own final phase (6) per explicit orchestrator instruction,
  despite being a single requirement.

- Phase 2: walk-forward interface frozen (expanding_steps + run_walkforward with
  automatic single append_trial per run); Phase 3 models plug into this interface.

- [Phase 05]: Phase 5 Plan 1: crisis_windows default list hard-bounded to 4 in-sample crises (1973-74, 1980-82, 2000-02, 2008-09), no 2020/2022 window; compute_turnover uses index-union reindex(fill_value=0.0), not positional diffing — Keeps holdout discipline (Pitfall 4) and correctly handles cold starts / asset-set changes
- [Phase 05-02]: Task boundary matched the plan literally: Task 2 lands the core loop without holdout split/L2 resilience (TestHoldoutBoundary intentionally RED); Task 3 adds split_by_holdout_boundary + try/except degrade-and-continue, turning it GREEN — Makes the incremental TDD narrative visible in git history rather than one large commit
- [Phase 05-02]: All 6 driver tests monkeypatch module-level _refit_l1/_refit_l2 (and vol_targeted_tilt for the cash-residual test) instead of exercising the real jump-model/nowcaster fit — Keeps orchestration-invariant tests fast/deterministic and isolated from real-fit degeneracy (Pitfall 2 territory); the real fit path is separately proven via the module's __main__ self-check
- [Phase 05-03]: compute_sojourn_lag_headline groups ex-post transitions by their OWN target state and checks each against only that state's own filtered-probs column, never a class-agnostic max-across-classes series — Review F1 fix — class-agnostic max would systematically understate detection lag ('fooled by its own backtest')
- [Phase 05-03]: max_drawdown_and_duration's duration_months is the longest run of consecutive underwater periods (drawdown < 0), not strictly peak-to-trough — Matches the plan's literal <action> text; a never-recovered drawdown extends duration to end of series
- [Phase 05-04]: model_metrics.py implements its own _reconcile_and_stack_proba rather than importing sojourn_lag.py's build_filtered_probs_matrix — the plan's numpy/pandas/stdlib-only import constraint keeps the two evaluation modules independently grep-gated, even though both implement the same union-of-classes/K-padding pattern (review F3).
- [Phase 05-04]: report_model_metrics indexes per_step_metrics['y_true'] via direct dict access (not .get()), and asserts len(y_true)==len(dates)==len(proba), raising ValueError on mismatch — y_true is always joined by date by the report layer, never sourced from the walk-forward loop (review F2).
- [Phase 05-05]: no_regime_ablation adds a cash_returns passthrough kwarg not in the plan's literal one-line-delegation snippet (Rule 2 auto-fix) - omitting it would silently default the ablation's cash residual to 0%, breaking F4 cash-return symmetry; the function stays a single delegating return statement.
- [Phase 05-06]: run_full_backtest_evaluation computes the smoothed-vs-filtered gap via a hindsight-oracle vol_targeted_tilt driven by the full-sample smoothed states at each real walk-forward decision date (never inventing new allocation math) rather than a simpler point-estimate proxy — matches the design's non-causal batch-labeling doctrine and keeps the gap input genuinely distinct from the filtered strategy performance (Pitfall 1).
- [Phase 05-06]: The investable asset_returns universe fed to run_backtest excludes the 'cash' splice class (FZFXX) — cash is never tilted into as a risk position; it is the vol-target residual that earns cash_ret directly via run_backtest's cash_returns parameter (review F4).
- [Phase ?]: 06-01: D-01 fresh-package boundary verified via static AST import-graph closure (not sys.modules) to survive test-order pollution
- [Phase ?]: 07-08: classifier #2 pinned before the fit (Lean 8, K=3, lambda=32.0=4n, sort_column=rs_equities_bonds, blend_weight_1=0.50) and recorded in ADR-0002 at Proposed
- [Phase ?]: 07-08: criterion-7 routing is L1 decision-bearing / L2 observational via NO_REGISTRY, firewalled from D-16/D-17/DSR — a fourth option, not one of the plan's three
- [Phase 07-11]: Criterion 7 measured: wealth_delta -0.123438, dd_delta +0.024084, both over 588 steps 1972-01-31 to 2020-12-31 (joint bw1=0.50 minus classifier-#1-alone bw1=1.00, L1-only routing). Both governing bands held; neither leg's DSR clears the multiple-testing hurdle at 42 trials.
- [Phase 07-11]: Joint harness degrades a step if EITHER classifier fails, so the two legs share a degraded-step SET rather than merely an index — the strongest available guarantee that the comparison is a one-parameter ablation.
- [Phase 07-11]: No sharpe key written into the trial registry: two near-identical legs would collapse registry_sharpe_variance from its 1.0 placeholder to ~1e-6, silently disabling the multiple-testing correction project-wide. Left open as an ADR-0002 amendment question.
- [Phase 07-12]: ADR-0002 accepted at a SCOPED width: what is accepted is the decision (classifier #2 was built on a disjoint set, pinned by rule, fit and measured), NOT a finding that an independent axis was added. Criterion 6 is unresolved and the ADR says so without a mitigating clause.
- [Phase 07-12]: REG-01 claimed PARTIALLY rather than in full, by Glenn's explicit decision of 2026-09-18, with criterion 6 named as the open item rather than folded in. The plan as written assumed both requirements could be claimed outright; that assumption did not survive the INCONCLUSIVE dependence verdict.
- [Phase 07-12]: The ADR's pre-declared sections were not edited in place. The Status section's original text is preserved verbatim under a subsection, and the probe-edge table's five extended rows carry a labelled AMENDMENT 2026-09-21 note — D-17's before-the-run guarantee is worthless if a pre-declaration can be silently rewritten afterward.
- [Phase 07-12]: Ratchet left at 31 rather than touched. The constant may only decrease and the re-measurement found no decrease; ROADMAP criterion 8's correction block was left byte-identical and the re-measurement recorded outside it.

### Pending Todos

- **(Phase 8, found in 08-06, NOT fixed — predates the phase) the sojourn/lag headline counts
  transitions the platform could never have acted on.** `compute_detection_lag`'s forward search
  has no upper bound, and 3 of the 25 reference transitions (**1970-05, 1970-08, 1971-03**) fall
  **before the first decision date, 1972-01-31**. They contribute lags of **44, 54 and 28 months**
  to the committed median of **4.0** (ratio 2.375). The headline is pinned byte-for-byte by
  08-06 and read by 08-08's S-3, so changing it is a deliberate decision, not a cleanup.

- **(Phase 8, found in 08-01, NOT fixed) `_leg_kpis` carries F-4's off-by-one.**
  `scripts/run_joint_lift.py::_leg_kpis` computes `state_{1,2}_transition_rate` as changes /
  `n_steps` (588), not / pairs (587). It feeds the committed `measurement_*.json`, which still
  say **0.418367** while `diagnostics_*.json` now say **0.418980** — two records, same 246 changes,
  different rates. **Folded into 08-10 Task 1 — approved by Glenn 2026-09-23.** Source fix and
  record regeneration land in one commit; a verify accepts /587 and rejects /588 (fails today).

- **(Phase 8, found in 08-01) the l2 leg drifts ~1e-7 across environments, not across runs.**
  Within one container l2 is bit-reproducible run to run; against HEAD's l2 curves (produced in
  a different container, 2026-09-21) numeric columns differ by up to **8.54e-08**, KPI
  `wealth_delta` by ~3e-10. `degraded` / `state_*` / `active_regime` identical; argmax unaffected
  (min top-2 gap 0.002). Cause unconfirmed — most plausibly BLAS/CPU. **l1only is unaffected.**
  Any cross-environment l2 comparison needs a ~1e-7 tolerance.

- **Release-engineering tech debt, recorded as `ROADMAP.md` Tier 0.5 (R1–R4), deferred
  deliberately 2026-09-11/14 — none block Phase 7, but R1 touches it directly.**

  - **R1 (HIGH)** — partial ingestion silently degrades `monthly_features` while the
    checkpoint merge repairs `monthly_raw` from disk, concealing it. Observed 2026-09-11:
    a lost `fred_aaa` meant `credit_spread_baa_aaa` was never derived; features went
    53 → 51 columns while raw looked fine. **`credit_spread_baa_aaa` is one of the 9
    features in the frozen common-support set Phase 7 D-02 locks** — a silent loss changes
    the frozen set from 9 to 8. Caught only because a real-data test is pinned to the
    seven A13 change points.

  - **R2 (MED)** — `build-pkg` CI builds both packages but never installs them, so an
    empty wheel can reach `main` and stay invisible until release.

  - **R3 (LOW)** — legacy `trading_crab_lib.plotting` raises a bare `ModuleNotFoundError`
    instead of the guarded `ImportError` naming the extra.

  - **R4 (MED, Phase 7 relevant)** — the enumerated `[tool.setuptools] packages` list is
    unguarded by any test, and the publish smoke test imports only the top-level package.
    A new Phase 7 subpackage omitted from that list would ship missing with every gate
    green. Verified in sync 2026-09-14 (17/17).

- Verifier informational note: `holdout.py`/`registry.py` use hardcoded constants that
  match `config/platform_settings.yaml` sections rather than reading them — wire to
  config if/when the values ever need to change (future-divergence risk only).

### Blockers/Concerns

- ~~Live 1962+ data run still pending FRED_API_KEY in the claude.ai/code environment
  (Phase 1 human-verification item).~~ **RESOLVED 2026-07-23** — `FRED_API_KEY` is present
  in the environment (32-char key) and functionally verified against the live FRED API
  (authenticated GDP series fetch succeeded). No longer a blocker; do not re-flag.

- **Macrotrends fix is wiring-verified, NOT live-verified** (quick task 260805-570). It
  fetches through a browser-impersonating client, but the container's agent proxy
  MITM-terminates TLS and resets curl_cffi's impersonated ClientHello, so it could not be
  confirmed against the live site. Unit tests prove the client is used; they do NOT prove
  impersonation defeats the bot check. Run the `<human-check>` in `260805-570-PLAN.md`
  Task 1 on a residential connection before trusting this source.

- **Macrotrends: headless-browser fallback now wired at both call sites (quick task
  260805-r7w). A residential-connection diagnostic (2026-08-05) confirmed a real headless
  browser DOES reach the live macrotrends page (HTTP 200, real rendered content, no
  interstitial) — the browser-reachability half of this fallback is no longer speculative.
  BUT the same diagnostic found the embedded-JSON regex (`_DATA_PATTERN`) does NOT match on
  the rendered page (the series lives inside a Highcharts closure, not a named `window`
  global), so the rendered HTML falls through to the `pandas.read_html` table path, not the
  JSON path. The page's only `<table>` has no distinguishing class (`historical_data_table`,
  assumed from a third-party snippet, does NOT exist — the real class is plain `"table"`), and
  its header cell text can be a squashed multi-line label (e.g. "Gold PricesMonthly Closing
  Price"). `BROWSER_WAIT_SELECTOR = "table"` is that unconfirmed candidate selector and is
  used ONLY with `require_selector=False` for exactly this reason. Column detection in
  `_scrape_series_html_table` / `_scrape_macrotrends_html_table_monthly` was extended to match
  a "month"-titled date column (with a value/date collision guard, since a squashed header
  containing "Monthly" also matches the substring "month"). What this diagnostic does NOT
  confirm: whether the end-to-end parse of a REAL macrotrends series (not the synthetic test
  fixtures in this task) produces correct data — that remains untested against the live site
  from this container (all egress here is proxy-reset). Run the `<human-check>` in
  `260805-r7w-PLAN.md` Task 3 to confirm the full chain end-to-end.**

- **Stooq: TLS impersonation confirmed insufficient; headless-Chromium fallback now wired,
  also NOT live-verified** (quick task 260805-jt2, superseding the 260805-570 Stooq
  finding). On a residential connection, `impersonate="chrome"` and `impersonate="safari"`
  — each with and without a hand-written header override, each with and without a
  same-session quote-page warm-up — ALL returned the identical 796-byte JavaScript
  browser-verification challenge page and set zero cookies. The challenge requires
  executing JavaScript; no TLS fingerprint can satisfy it. A headless-Chromium challenge
  solver (`ingestion/browser.py`) is now wired as the Stooq fallback behind the optional
  `[browser]` packaging extra, tried only when the plain HTTP path recovers zero tickers.
  This browser path is wiring-verified by mocked unit tests ONLY — it has NOT been
  confirmed against the live site, because all container egress is reset by the agent
  proxy (Chromium launches here, but even `https://example.com` returns
  `net::ERR_CONNECTION_RESET`). Run the `<human-check>` in `260805-jt2-PLAN.md` Task 3 on
  a residential machine to find out whether it defeats the challenge.

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 260805-570 | fix stooq and macrotrends bot-blocking, ALFRED vintage schema, build guard, yfinance rate-limit | 2026-08-05 | e98f1f0 | [260805-570-fix-stooq-and-macrotrends-bot-blocking-a](./quick/260805-570-fix-stooq-and-macrotrends-bot-blocking-a/) |
| 260805-jt2 | restore Stooq via a headless-Chromium challenge solver behind the optional `[browser]` extra (TLS impersonation proven insufficient) | 2026-08-05 | c57d44d | [260805-jt2-restore-stooq-via-headless-chromium-chal](./quick/260805-jt2-restore-stooq-via-headless-chromium-chal/) |
| 260805-od0 | implement the live Tiingo daily-price adapter as first fallback ahead of yfinance | 2026-08-05 | ed6e425 + de4b038 | [260805-od0-implement-tiingo-daily-price-adapter-as-](./quick/260805-od0-implement-tiingo-daily-price-adapter-as-/) ⚠ no SUMMARY.md written |
| 260805-r7w | generalize browser.py to fetch_page_html/fetch_urls_as_text, add Selenium as a second engine, route macrotrends through the browser fallback at both call sites | 2026-08-05 | (see directory) | [260805-r7w-generalize-browser-module-and-add-seleni](./quick/260805-r7w-generalize-browser-module-and-add-seleni/) |
| 260806-u89 | never lose data: multi-source checkpoint merge + fallback chains so a partial fetch cannot shrink a saved checkpoint | 2026-08-06 | 555cd9b | [260806-u89-never-lose-data-checkpoint-merge-multi-s](./quick/260806-u89-never-lose-data-checkpoint-merge-multi-s/) |
| 260908-qwe | fix hindsight-oracle IndexError: restrict the smoothed oracle's per-step universe to assets that have started (phantom IAU/USO weight in 1974), guard portfolio_vol's per-asset EWMA fallback conservatively | 2026-09-08 | 18f68af | [260908-qwe-fix-hindsight-oracle-indexerror-phantom-](./quick/260908-qwe-fix-hindsight-oracle-indexerror-phantom-/) |
| 260908-rh4 | fix percent-vs-decimal yield units at the splice boundary (long_duration_tr compounded to 2.3e128; cash booked yield CHANGES as returns), add an asymmetric units guard, correct three test defects incl. an integration test whose 24/24 steps all degraded | 2026-09-08 | 75dedc7 | [260908-rh4-fix-percent-vs-decimal-yield-units-at-th](./quick/260908-rh4-fix-percent-vs-decimal-yield-units-at-th/) |
| 260908-fnc | close the holdout fence at rest: carve at build, assert at build, add load_full_span() looking opt-in, repoint live weekly scoring | 2026-09-08 | 7f99548 | (in this STATE entry) |
| 260909-0hs | stop pytest destroying the production holdout checkpoint | 2026-09-09 | eca8f06 | [260909-0hs-stop-pytest-writing-to-production-platfo](./quick/260909-0hs-stop-pytest-writing-to-production-platfo/) |
| 260909-0og | fix ALFRED vintage index-base discontinuity — chain vintages on within-vintage growth (audit item A4, closed empirically) | 2026-09-09 | 976f7c8 + 8095498 | [260909-0og-fix-alfred-vintage-index-base-discontinu](./quick/260909-0og-fix-alfred-vintage-index-base-discontinu/) |
| 260910-vyi | fix PyPI publish: corrected doubled dist path (dist/dist), added loud PUBLISH/SKIP gate logging so a skipped matrix leg is no longer indistinguishable from a publish, added a Verify built artifacts guard that fails on empty dist or tag/artifact version mismatch; bumped both packages 0.1.2 -> 0.1.4 (0.1.3 burned on PyPI) | 2026-09-10 | 9d0153c + 0c6fb8d | [260910-vyi-fix-pypi-publish-workflow-dist-path-and-](./quick/260910-vyi-fix-pypi-publish-workflow-dist-path-and-/) |
| 260911-kkj | add a PyPI token-presence guard: fails the job before any build/upload when the leg's API token secret is absent or empty, naming the exact secret and noting that a dynamic secrets[...] lookup yields an empty string on a name mismatch rather than erroring | 2026-09-11 | 118bd06 | [260911-kkj-add-a-token-presence-guard-to-the-pypi-p](./quick/260911-kkj-add-a-token-presence-guard-to-the-pypi-p/) |
| 260911-la3 | fix trading-crab-lib's blank PyPI page (wired real README into pyproject readme key, twine check --strict WARNING -> PASSED), add a twine check --strict gate before every upload, add a workflow_dispatch target input (testpypi default) for a TestPyPI dry-run path with target-aware secret selection, write docs/RELEASING.md | 2026-09-11 | c25cbc6 + beabc63 + e97c8a5 + ca837cd | [260911-la3-harden-the-release-procedure-twine-check](./quick/260911-la3-harden-the-release-procedure-twine-check/) |
| 260911-nt7 | fix the empty trading-crab-lib wheel (0.1.0–0.1.4 shipped zero Python modules): explicit package-dir + enumerated packages list, install-and-import smoke gate, --no-deps so parallel matrix legs cannot couple; bumped both packages to 0.1.5 | 2026-09-11 | f204de3 + 926ef21 | [260911-nt7-fix-empty-trading-crab-lib-wheel-add-ins](./quick/260911-nt7-fix-empty-trading-crab-lib-wheel-add-ins/) |

## Deferred Items

Items acknowledged and carried forward from previous milestone close:

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| *(none — first milestone)* | | | |

## Session Continuity

Last session: 2026-09-21T16:05:00.000Z
Stopped at: Completed 07-12-PLAN.md — **Phase 7 CLOSED**
Resume file: `platform_design/adr/0002-l1-second-classifier.md` § Deferrals and open items at
acceptance — the thirteen items Phase 7 carries forward, and the starting point for Phase 8.
**Read § ACCEPTANCE 2026-09-21 first: criterion 6 is unresolved and REG-01 is only partially
delivered. Nothing in Phase 7 establishes that classifier #2 adds an independent axis.**

Between the 2026-09-10 context session and this one, the release-engineering work
(quick tasks 260910-vyi, 260911-kkj, 260911-la3, 260911-nt7) was carried out **partly
outside GSD** because of model rate limits. 260911-nt7 shipped code and a PLAN but no
SUMMARY; that summary and the STATE/ROADMAP status entries were back-filled on
2026-09-14. No code was changed during the reconciliation.

### Audit Part II — Phases 2, 3, 4 (2026-09-09)

**Evidence quality is not uniform, and these three hold up far better than 1 and 5.**
Phase 3's DP-decode oracle test — proven identical to brute-force enumeration across
7 (T,K,λ) cases — is the strongest verification artifact in the project and should be
the pattern wherever a brute-force reference is affordable.

Five criteria need attention, not a blanket re-verification:

- **P2 C1** (holdout) — mechanism verified, application never invoked. Closed 2026-09-08.
- **P2 C5** (causal gating) — the rail IS live (`nowcaster.py:114`), but the gate is a
  name-suffix scan; a centered feature not following the convention passes silently.

- **P3 C4** (calibration) — "uses `CalibratedClassifierCV`" stands in for "is
  calibrated". Brier is 0.1816; no pass band exists.

- **P4 C2** (EWMA vol) — verified by import-grep; the consumer `portfolio_vol` then
  crashed on ragged universes, the condition holding for 43 of 49 backtest years.

- **P4 C3** (hysteresis) — **the criterion's purpose clause was never tested.**
  `update_active_regime` is a correct Schmitt trigger, but `active_regime` gates
  nothing: weights come from `vol_targeted_tilt(regime_probs, …)` in both the backtest
  driver and the weekly report. The thresholds stabilize a *reported label*, not a
  *portfolio*. Needs an explicit decision (item A7).

**The gap spanning all five phases:** no criterion anywhere asks whether the output is
any good. Phase 3 proved the DP decode is exactly optimal — and it is exactly decoding
a degenerate solution of 6 transitions in 59 years. Phase 5 D-01 and Phase 6 D-16 make
this deliberate, which is a defensible tracer-bullet stance, but it means the project
has **no gate that can fail on a bad model, only on a broken one**. Item A11: make that
a conscious choice re-examined at design freeze.
