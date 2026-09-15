---
phase: 07-regime-representation
plan: 03
subsystem: platform-backtest
tags: [jump-model, walk-forward, feature-policy, trial-registry, honesty-framework, disagreement-measurement]

# Dependency graph
requires:
  - phase: 07-regime-representation
    provides: "07-01's frozen driver/reference L1 feature-column mechanism (trial_tag, frozen_l1_features) and 07-02's D-02-A-recomputed dev monthly_features checkpoint (708x53, 10-column frozen reference set)"
provides:
  - "measure_label_disagreement() — a first-class, tested criterion-3 measurement delegating to plotting/regime.py::label_disagreement, closing the state_N-string silent-zero coercion trap."
  - "scripts/run_policy_trials.py — runs both wave-1 policy variants (frozen 10-col decision, imputed 13-col D-03 rejection) for real, with hard physical-impossibility guards and advisory-only [ASSUMED]-band flags."
  - "07-MEASUREMENTS.md — every wave-1 number against its pre-fix baseline, its band, and its verdict, plus a NAMED window-narrowing limitation on criterion 3 that MUST be carried forward into 07-04 (binding condition, see below)."
  - "Human sign-off (relayed via orchestrator) closing Task 3's checkpoint: criterion 3 satisfied as worded, D-04 holds, one binding condition attached."
affects: ["07-04 (ADR-0001 — MUST foreground the window-narrowing limitation inline at the point pct_disagree/§5.4-ratio numbers are shown, not as a discussion-only footnote; see 'Binding Condition for 07-04' below)"]

# Actuals (#2632)
actuals:
  tokens: 15000
  tasks: 3
  commits: 4

tech-stack:
  added: []
  patterns:
    - "Delegation-not-reimplementation for methodology-equality: measure_label_disagreement() wraps the located label_disagreement() rather than recomputing the comparison, so a post-fix number is comparable to a pre-fix baseline BY CONSTRUCTION, not by inspection."
    - "Pinned-commit git-show fixture for a regression test that survives its own plan overwriting the artifacts it tests against (test_reproduces_the_prefix_baseline reads outputs/reports/platform/*.parquet via `git show <sha>:<path>`, not the live working tree)."
    - "Hard-vs-advisory assertion split in a measurement script: physically-impossible values (terminal log wealth, drawdown bounds, registry-row-count) raise; [ASSUMED] domain bands (wealth_delta, dd_delta, n_transitions, pct_disagree thresholds) are recorded as flags only, per D-07 — never gate, never trigger a re-run."

key-files:
  created:
    - src/trading_crab_lib/platform/evaluation/disagreement.py
    - tests/unit/test_platform_evaluation_disagreement.py
    - scripts/run_policy_trials.py
    - .planning/phases/07-regime-representation/07-MEASUREMENTS.md
  modified:
    - outputs/reports/platform/backtest_report.md
    - outputs/reports/platform/backtest_kpi_table.parquet
    - outputs/reports/platform/backtest_full_sample_states.parquet
    - outputs/reports/platform/backtest_filtered_state_probs.parquet
    - outputs/reports/platform/backtest_equity_curve_strategy.parquet
    - outputs/reports/platform/backtest_equity_curve_ablation.parquet
    - outputs/reports/platform/model_metrics_brier.parquet
    - outputs/reports/platform/model_metrics_calibration.parquet
    - outputs/reports/platform/model_metrics_confusion.parquet
    - registry/trials.jsonl

key-decisions:
  - "D-04 followed: no measured number (pct_disagree, §5.4 ratio, wealth_delta, dd_delta) was used to select, revise, or re-run the frozen ten-column policy — including after discovering the window-narrowing finding, which looks unfavorable for the frozen policy's sample size but was NOT treated as grounds to prefer the rejected 13-column+imputation alternative."
  - "Human sign-off settled: (1) criterion 3 is satisfied AS WORDED — measured by the located methodology, reported against the 82.8% baseline, no target set; the window-narrowing limitation qualifies interpretation, it does not fail the criterion. (2) D-04 holds, unchanged by the degrade finding."
  - "BINDING CONDITION attached by the human sign-off, carried forward to 07-04: the window-narrowing (232/588 L2-degraded steps under the frozen policy vs. 118/588 under pre-fix/imputed, producing different n_compared AND different end dates — 356 months to 2017-05 vs. 470 months to 2020-12/2020-08) must be foregrounded as a named limitation at the point the numbers are shown in 07-04's ADR and its own copy of the D-05 pre/post table — inline, not only in a later discussion section. See 'Binding Condition for 07-04' section below."
  - "L1/L2 mechanism verified vs. inferred, recorded in 07-MEASUREMENTS.md §5: frozen_l1_features reaches only _refit_l1 (driver.py lines 461, 473); _refit_l2's signature (driver.py lines 266-271) takes no such parameter — L2 can only be affected through _refit_l1's changed OUTPUT LABELS. This is consistent with the observed occupancy shift and degrade-count shift but does NOT constitute a traced causal chain — recorded explicitly as inferred, not measured, per the human's request not to assert a cause beyond what was verified."

requirements-completed: [REG-01]

coverage:
  - id: D1
    description: "measure_label_disagreement() delegates to the located label_disagreement() and reproduces the 389/470=82.8% baseline exactly (to 1e-12), proving methodological equality; also closes the state_N-string silent-zero coercion trap and adds an advisory-only suspicious flag."
    requirement: "REG-01"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_evaluation_disagreement.py::TestMeasureLabelDisagreement (6 tests, including test_reproduces_the_prefix_baseline, NOT skipped)"
        status: pass
    human_judgment: false
  - id: D2
    description: "Both wave-1 policy variants (frozen 10-col decision; imputed 13-col D-03 rejection) ran to completion on real checkpoints via scripts/run_policy_trials.py, passing every hard physical-impossibility guard; registry grew by exactly 4 rows (34->38, 2 per run); the imputed variant's artifacts never touched the published outputs/reports/platform/* paths (verified: published kpi_table strategy row = 4.025085, the frozen variant's value)."
    requirement: "REG-01"
    verification:
      - kind: manual_procedural
        ref: "python scripts/run_policy_trials.py --variant frozen (exit 0); python scripts/run_policy_trials.py --variant impute (exit 0); read_trials() count 34->36->38 confirmed live"
        status: pass
    human_judgment: false
  - id: D3
    description: "07-MEASUREMENTS.md records every wave-1 number against its pre-fix baseline, band, and verdict; names the window-narrowing limitation on criterion 3 inline in the pre/post table (not only as a footnote); records the compound-baseline caveat; restates D-04. Criteria 2, 3, and 4 all satisfied with human sign-off."
    requirement: "REG-01"
    human_judgment: true
    rationale: "Reading whether a measurement record is honest and complete — and whether a window-narrowing finding is foregrounded prominently enough — is exactly the judgment call this phase's honesty framework requires a human to make, per the Task 3 checkpoint's own design (mirrors Phase 5's 05-07 pattern)."

duration: ~35min (this session, post-checkpoint continuation)
completed: 2026-09-14
status: complete
---

# Phase 7 Plan 3: Wave-1 Policy Trials — Measurements, Human Sign-Off, and a Binding Condition for 07-04

**Both wave-1 policy variants ran for real on the D-02-A-recomputed checkpoint (frozen 10-column decision: strategy terminal log wealth 4.025085, `wealth_delta` +0.377847, `dd_delta` −0.066124; imputed 13-column D-03 rejection: recorded, isolated, never published); the human reviewed and approved, with one binding condition: the frozen policy's disagreement/ratio numbers rest on a narrower, differently-dated sample (356 of 588 months, ending 2017-05, vs. the pre-fix baseline's 470 months ending 2020-12) — 07-04 MUST show this inline wherever the numbers appear, not as a footnote.**

## Performance

- **Duration:** ~35 min (this continuation, after the Task 3 checkpoint hand-off)
- **Tasks:** 3 (2 `auto`, 1 `checkpoint:human-verify`)
- **Files modified:** 4 created, 10 modified (this continuation: 1 doc edit to `07-MEASUREMENTS.md` + this SUMMARY)

## Accomplishments

- **Task 1** (prior turn, commit `14987cd`): `measure_label_disagreement()` created as a first-class, tested wrapper around the located `label_disagreement()` methodology. 6 tests, including a baseline-reproduction test that reads the pre-fix artifacts via `git show <pinned-sha>` so it survives this plan's own artifact regeneration.
- **Task 2** (prior turn, commit `5edef3e`): `scripts/run_policy_trials.py` ran both wave-1 variants for real (~2 min each) — the frozen ten-column policy (published) and the 13-feature+imputation alternative (isolated, D-03's evidence-backed rejection). Registry grew by exactly 4 rows. `07-MEASUREMENTS.md` was written with the full pre/post table.
- **Task 3** (this turn): the human reviewed the inline checkpoint report, **approved**, and attached one binding condition (below). `07-MEASUREMENTS.md` was amended — NOT re-run, NOT re-measured, no numeric value changed — to:
  1. Add a new §5, "Named limitation on criterion 3: the window-narrowing," placed BEFORE the compound-baseline section, stating the differing sample size (470 vs. 356) and end date (2020-12 vs. 2017-05) inline wherever `pct_disagree` and the §5.4 ratio appear — in the §2 pre/post table itself (new "Sample window" column + an unmissable callout above the table), not only in discussion.
  2. Record the L1/L2 mechanism verification (frozen_l1_features reaches only `_refit_l1`; `_refit_l2` takes no such parameter) as **verified**, and explicitly flag everything beyond that as **inferred, not measured** — per the human's own corroborating check and their explicit instruction not to assert an unproven cause.
  3. State explicitly, per the human's decision: criterion 3 is satisfied as worded (measured + reported against the baseline, no target); D-04 holds (the finding is not grounds to reopen the policy or prefer the rejected variant).
  4. Renumber all subsequent sections and fix cross-references; update the closing "Sign-off" section from "pending" to "APPROVED" with the two settled points and a pointer to this SUMMARY's binding-condition section.

## Task Commits

1. **Task 1: A first-class criterion-3 disagreement measurement, matching the located baseline methodology** - `14987cd` (feat)
2. **Task 2: Run both policy variants for real and capture every wave-1 number** - `5edef3e` (feat)
3. **Task 3: Incorporate the human sign-off decision and the binding window-narrowing condition into `07-MEASUREMENTS.md`** - `27e08e2` (docs)

**Plan metadata:** `[this commit]` (docs: complete plan 03, this SUMMARY) — see the final response for the exact hash

## Files Created/Modified

- `src/trading_crab_lib/platform/evaluation/disagreement.py` — `measure_label_disagreement()`, delegating to `plotting/regime.py::label_disagreement`.
- `tests/unit/test_platform_evaluation_disagreement.py` — 6 tests.
- `scripts/run_policy_trials.py` — `run_variant("frozen"|"impute")`, hard vs. advisory assertion split.
- `outputs/reports/platform/*` — regenerated under the frozen policy (published).
- `outputs/reports/platform/trials/impute-13col/*` — the D-03 rejected variant's isolated artifacts.
- `registry/trials.jsonl` — +4 rows (`P7-W1-frozen-10col` ×2, `P7-W1-impute-13col-REJECTED` ×2).
- `.planning/phases/07-regime-representation/07-MEASUREMENTS.md` — the full record, amended this turn per the human's binding condition.

## BINDING CONDITION FOR 07-04 (do not miss this)

**The human sign-off is conditional on this being carried forward.** When plan 07-04 writes the
ADR and its own copy of the D-05 pre/post table, it **MUST**:

1. **State the differing sample size and end date INLINE, at the point `pct_disagree` and the
   §5.4 ratio numbers are shown** — in the same table row/cell, not deferred to a later
   discussion paragraph. `07-MEASUREMENTS.md` §2 (the pre/post table) and §5 (the named
   limitation section) show the required treatment; copy that pattern, don't just link to it.
2. **Never present 82.77% → 80.90% as a clean 1.9-point improvement.** The two percentages are
   measured over different populations: pre-fix `n_compared=470` spanning 1974-02→2020-12;
   post-fix (frozen) `n_compared=356` spanning 1974-02→**2017-05**. The frozen policy's
   walk-forward produced 232/588 L2-degraded steps (vs. 118/588 for the pre-fix baseline and
   for the imputed variant) — a real, verified difference in comparable sample, not a
   presentation nuance.
3. **The same treatment applies to the §5.4 ratio's `n_resolved`/`n_transitions`.** "7 of 7"
   for the frozen policy means 7-of-7-available-within-the-356-month-window, not 7-of-7-over-
   the-full-588-step range.
4. **Keep the mechanism claim scoped to what was verified.** It is VERIFIED that
   `frozen_l1_features` reaches only `_refit_l1` (`driver.py` lines 461, 473) and that
   `_refit_l2` (`driver.py` lines 266-271) takes no such parameter — so L2 can only be affected
   through L1's changed output labels. It is **NOT** verified (only plausible) that this fully
   explains the specific 232-vs-118 degrade-count difference. 07-04 must preserve this
   verified/inferred distinction rather than presenting the mechanism as proven.
5. **This finding is evidence, never a selection input.** D-04 held throughout this plan: the
   window-narrowing was NOT used to prefer the imputed (rejected) alternative, even though the
   imputed variant's sample happens to match the pre-fix baseline's window more closely. 07-04
   inherits this same discipline — it may not use the finding to reopen the frozen-policy
   decision either.

## Decisions Made

- **D-04 held throughout, including after the window-narrowing discovery.** No number — not
  `dd_delta` coming back more negative than pre-fix (−0.0661 vs. −0.014364), not the narrower
  disagreement sample, not the imputed variant's nominally better `wealth_delta`/`dd_delta` —
  was used to revise or re-run the policy. This is the single most load-bearing decision in
  this plan and is restated in `07-MEASUREMENTS.md` §7 and §5.
- **Human sign-off settled two points explicitly** (relayed via the orchestrator, not a
  self-approval): criterion 3 is satisfied as worded; D-04 holds. Both are now recorded in
  `07-MEASUREMENTS.md`'s closing section.
- **The window-narrowing binding condition is a process requirement on 07-04, not a data
  correction on this plan.** No measured number in `07-MEASUREMENTS.md` was changed by this
  turn's edits — only presentation (table structure, an inline callout, a new named-limitation
  section, and cross-reference renumbering).
- **L1/L2 mechanism: verified claim kept narrow.** Confirmed directly against `driver.py`
  (lines 461, 473 for `_refit_l1`'s frozen-features call sites; lines 266-271 for `_refit_l2`'s
  signature) that L2 has no direct parameter dependency on `frozen_l1_features` — only an
  indirect one through L1's output labels. The specific causal chain from feature-set to
  per-window class-imbalance degrade is explicitly NOT claimed as measured.

## Deviations from Plan

None beyond what was already documented in Tasks 1–2 (no deviations in either). Task 3's
completion consisted of amending `07-MEASUREMENTS.md` per the human's binding condition —
this is the checkpoint's own designed outcome (a conditional approval with a carried-forward
requirement), not a deviation from the plan.

## Issues Encountered

None. The checkpoint worked as designed: the inline report surfaced every number with its
band, the human read it, approved, and attached a scoping condition that was incorporated
without touching any measured value.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- **Plan 07-04 (ADR-0001) is ready to proceed, WITH the binding condition above.** It must cite
  this plan's exact numbers from `07-MEASUREMENTS.md`, preserve the three-labelled-states
  structure (pre-fix / post-fix frozen / post-fix imputed-rejected), and foreground the
  window-narrowing limitation inline — not as a footnote — wherever `pct_disagree` or the §5.4
  ratio is shown.
- The frozen ten-column policy is confirmed as the published decision:
  `outputs/reports/platform/backtest_kpi_table.parquet`'s `strategy` row is `4.025085120485386`
  (frozen), not `4.048185034686693` (imputed/rejected).
- Registry stands at 38 rows; the last 4 carry `trial_tag` values
  `P7-W1-frozen-10col`, `P7-W1-frozen-10col`, `P7-W1-impute-13col-REJECTED`,
  `P7-W1-impute-13col-REJECTED`.
- Full suite: **1726 passed, 0 skipped, 0 failed** (measured after both variant runs
  regenerated the published artifacts; unchanged by this turn's doc-only edit).
- No blockers. Wave 1 (D-09) is complete; wave 2 gets its own `/gsd-plan-phase 7` pass per D-09.

---
*Phase: 07-regime-representation*
*Completed: 2026-09-14*

## Self-Check: PASSED

All 4 created files (`disagreement.py`, its test file, `run_policy_trials.py`,
`07-MEASUREMENTS.md`) confirmed present on disk; both prior task commits (`14987cd`,
`5edef3e`) confirmed present in git history; `07-MEASUREMENTS.md`'s section numbering
(1 through 10 plus the closing sign-off) confirmed sequential and cross-references
consistent after the renumbering edit; the published `backtest_kpi_table.parquet` reconfirmed
to hold the frozen variant's `4.025085120485386`, not the imputed variant's value.
