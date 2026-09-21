---
phase: 07-regime-representation
plan: 12
subsystem: planning-record
tags: [adr, requirements, roadmap, phase-closure, honesty-framework]
status: complete
requires: ["07-07", "07-09", "07-11"]
provides:
  - "ADR-0002 at status Accepted, 2026-09-21"
  - "All eight spec-less probe edges resolved, each naming its test"
  - "REG-01 marked PARTIAL (criterion 6 unresolved); INV-01 marked delivered in full"
  - "ROADMAP Phase 7 criteria 5, 6, 7 marked against their evidence"
  - "Legacy-import ratchet re-measured at phase close"
affects:
  - platform_design/adr/0002-l1-second-classifier.md
  - .planning/ROADMAP.md
  - .planning/REQUIREMENTS.md
  - .planning/STATE.md
tech-stack:
  added: []
  patterns:
    - "Pre-declared sections amended by a labelled amendment, never edited in place"
    - "Every measured figure carries its window inline, in the same cell or sentence"
key-files:
  created:
    - .planning/phases/07-regime-representation/07-12-SUMMARY.md
  modified:
    - platform_design/adr/0002-l1-second-classifier.md
    - .planning/ROADMAP.md
    - .planning/REQUIREMENTS.md
    - .planning/STATE.md
decisions:
  - "REG-01 claimed PARTIALLY, not in full — criterion 6 named as the open item rather than folded in"
  - "Acceptance of ADR-0002 scoped to the decision, explicitly not to a finding that an axis was added"
  - "Ratchet left at 31: re-measured, no decrease found, constant untouched"
  - "ROADMAP criterion 6 marked UNRESOLVED rather than satisfied, departing from the plan's instruction to mark criteria 5-7 satisfied"
metrics:
  duration: 35min
  completed: 2026-09-21
actuals:
  tokens: 28000
  tasks: 2
  commits: 3
---

# Phase 7 Plan 12: Phase Closure Summary

ADR-0002 accepted at a deliberately scoped width — the decision, not a finding — with criterion
6 recorded as UNRESOLVED and criterion 7's negative wealth result stated in the units it is
actually in; REG-01 claimed PARTIALLY and INV-01 in full.

## What was done

### Task 1 — ADR-0002 → Accepted (commit `60a4915`)

**Status: Accepted, 2026-09-21.** The acceptance is scoped in the Status section itself, before
anything else: what is accepted is the **decision** — classifier #2 was specified on a disjoint
leadership feature set, pinned by rule rather than by search, fit, and measured. **It is not a
finding that classifier #2 adds an independent axis**, and the ADR says so in its own first
paragraph.

The pre-declared Status text was **not overwritten**. It is preserved verbatim as a block quote
under a `### How this status was reached` subsection, with one labelled correction (acceptance was
carried out by plan 07-12, not 07-11). D-17's before-the-run guarantee is worthless if a
pre-declaration can be silently rewritten afterward.

Appended `## ACCEPTANCE 2026-09-21 — Consequences, as measured`, carrying:

| Criterion | Verdict as the ADR states it | The number, with its window |
|---|---|---|
| 5 — occupancy & disjointness | **MET** | Classifier #2 occupancy **16.6667 / 22.7011 / 22.4138 / 23.8506 / 14.3678 %** over **696 months, 1963-01-31 → 2020-12-31**; sum error exactly **0.0**; all five states inside §4.4's real two-sided **~8–35%** band; disjointness asserted on the *resolved* frozen eight |
| 6 — dependence | **UNRESOLVED — neither met nor failed** | ARI **0.354841**, NMI **0.464088**, Cramér's V **0.589748**, `n_compared` = **695 months, 1963-02-28 → 2020-12-31**, K₁ = 6 vs K₂ = 5. Control: 2000 resamples, seed 20260918 → observed NMI at the **96.60th percentile** against p95 **0.449202** and p99 **0.501195** → **INCONCLUSIVE** by the rule committed at `298b1bc` |
| 7 — joint lift | **MET as a measurement; the wealth sign is negative** | `wealth_delta` **−0.123438 nats** (≈ 0.8839×, an **11.61% terminal-wealth shortfall**) and `dd_delta` **+0.024084** (**2.41pp shallower**, 40 vs 47 months underwater), both over **588 steps, 1972-01-31 → 2020-12-31**, 0 degraded steps on both legs |

Also recorded: both legs' DSR with `format_dsr_verdict`'s output quoted **verbatim** — *"Deflated
Sharpe ratio 0.0000 does not clear the multiple-testing hurdle — statistically indistinguishable
from a skill-less discovery given the number of trials searched"* — at `n_trials` **42**,
`sharpe_variance` **1.0** (placeholder); the 2026-09-21 estimator repair (`_MIN_USABLE_SHARPE_OBSERVATIONS`
2 → 20 plus the `independent_trial` exclusion, after the 99.85% hurdle collapse it would have
caused) with today's verdicts explicitly unchanged; all five rejections restated with reasons;
the four plausibility bands with `dd_delta`'s universal bound at the **revised `[-1, 1]`**, the
universal/domain governance split, and **A11 left open by deliberate choice**; and trial
arithmetic reconciled against a live `total_trial_count()` = **42**, read
**2026-09-21T15:37:29.833014Z**, against the ceiling of **44**.

A `## ⚠ Named limitation` **section** (not a footnote) records the measurement-window consequence
of the chosen routing: the clean 588-of-588 window is a *consequence of running no L2 refit*, not
evidence the production stack behaves that way; wave 1's +0.377847 / −0.066124 are a **different
subtraction over a different window and path** and may not be tabled beside these numbers; the
filtered-labeling churn (**246 of 588 decision months, 41.84%**, against a 3.60% full-sample rate)
that feeds the tilt is governed by no band; and the drawdown gain must not be read as crisis-timing
skill, because the crisis state's median sojourn is **3.0 months — exactly on criterion 2's
boundary** — against a 1–3 month detection lag.

### Task 2 — probe edges, requirements, roadmap, ratchet (commit `5447684`)

**The eight probe edges** are all resolved with no deferred row. Five rows were extended to name
their tests; none was weakened or re-classified, and the change carries a labelled
`AMENDMENT 2026-09-21` note beneath the table. Every named test was verified to exist in the tree
by name:

| Requirement | Edge | Test(s) that resolve it |
|---|---|---|
| REG-01 | adjacency | `test_platform_labeling_classifier2.py::TestFreezeClassifier2Columns::{test_returns_every_candidate_in_declaration_order, test_matches_reference_label_columns_called_directly, test_late_starting_candidate_excluded_by_name_with_its_first_valid_month, test_candidate_absent_from_the_frame_is_excluded_not_a_keyerror}`; criterion-5 disjointness via `test_platform_features_relative.py::{test_disjoint_from_lean_feature_set, test_configured_feature_list_is_disjoint_from_the_lean_set, test_resolved_frozen_list_is_disjoint_from_the_lean_set}` |
| REG-01 | empty | `TestFreezeClassifier2Columns::{test_empty_frozen_list_raises_naming_the_count, test_shorter_than_K_raises_naming_the_count_and_K, test_all_candidates_nan_after_the_decision_date_raises}`; `test_platform_evaluation_dependence.py::{test_disjoint_index_gives_nan_statistics_and_warns_without_raising, test_disjoint_index_never_reports_zero, test_disjoint_span_renders_as_a_finding_not_as_a_number}` |
| REG-01 | ordering — **re-opened by classifier #2 and re-resolved** | `TestLabelLeadershipRegimes::{test_missing_ordering_column_raises, test_states_are_numbered_by_ascending_sort_column_centroid, test_two_calls_on_the_same_frame_return_identical_states, test_returns_frozen_columns_states_and_confidences_aligned}`; `test_platform_labeling.py::test_disjoint_feature_set_end_to_end_explicit_sort_column` |
| INV-01 | boundary | `test_platform_macro_ingest.py::test_boundary_m2_gdp_first_valid_is_the_later_source_start`; plus `test_stable_at_exact_tolerance_boundary`, `test_holdout_boundary_applied_before_any_ratio_is_computed`, `test_no_result_admissible_month_exceeds_the_holdout_cutoff` |
| INV-01 | adjacency | `test_platform_macro_ingest.py::test_adjacency_denominator_not_interpolated_across_a_quarter` (recovers the denominator from the ratio, asserts it is constant within a quarter); plus `test_era_windows_never_leak_future_values_into_an_early_era`, `test_era_count_is_a_handful_not_one_per_month` |
| INV-01 | empty | `test_no_computable_candidate_raises`, `test_no_candidate_computable_raises`, `test_all_nan_candidate_excluded_with_warning`, `test_missing_column_excluded_with_warning` |
| INV-01 | ordering | `test_returns_frame_indexed_by_ordered_candidate_names`, `test_results_are_in_invariant_candidates_declared_order`, `test_rejects_integer_or_component_label_index` |
| INV-01 | precision | Tolerance stated numerically: **`LOADING_STABILITY_TOLERANCE = 0.15`**, applied as `range <= 0.15` (inclusive), loadings compared unrounded; boundary pinned by `test_stable_at_exact_tolerance_boundary` |

**Requirements.** Scoped `Edit` calls on the two entries and their coverage rows only:

- **REG-01 → `[~]` DELIVERED PARTIALLY.** Three of four wave-2 clauses satisfied; the
  orthogonality clause was **measured but its verdict is unresolved**. Coverage row reads
  *"Partial — criterion 6 (dependence) UNRESOLVED; see ADR-0002"*.
- **INV-01 → `[x]` DELIVERED IN FULL**, with all four clauses enumerated and each pointed at its
  artifact.

**ROADMAP.** Scoped `Edit` calls on the Phase 7 section only (100 insertions, 10 deletions; every
deletion an intentional replacement). Criterion 8's correction block appears **only as context
lines in the diff — byte-identical**; the ratchet re-measurement is recorded *outside* it.
Criterion 5's evidence block corrects the *"§4.4 5% floor"* misquote inline and records the
2026-09-18 recurrence exemption (classifier #2 does not invoke it; classifier #1 does, at
**5.7554%** across nine named episodes, 1970 through 2020). The phase's "claims both in full"
coverage paragraph was corrected, and the carried-forward open items listed.

**Ratchet.** AST scan at phase close, 2026-09-21: **31** legacy import sites — `trading_crab_lib`
×16, `.checkpoints` ×7, `.ingestion.http` ×3, `.ingestion.browser` ×2, `.ingestion` ×1,
`.ingestion.assets` ×1, `.email` ×1. **Unchanged from 2026-09-15.** Wave 2 added none and removed
none, so `MAX_LEGACY_IMPORT_SITES` was **not edited** — the constant may only decrease and there
was no decrease to record.

## Deviations from Plan

### 1. [Rule 1 — correctness] ROADMAP criterion 6 marked UNRESOLVED, not satisfied

**Found during:** Task 2.
**Issue:** The plan instructed that criteria 5, 6 and 7 be marked satisfied, on the reasoning that
criterion 6 asks only that dependence be measured and reported. That instruction was written
before the dependence result existed. `07-DEPENDENCE.md` § RESULT states in its own words that the
outcome *"does not license reading criterion 6 as satisfied. It is not satisfied; it is
unresolved."*
**Fix:** Criterion 6's ROADMAP block records that the measurement and reporting clause is
discharged and that **the criterion itself is UNRESOLVED**, with the numbers and the pre-registered
rule. REG-01 is claimed PARTIALLY as a consequence.
**Files modified:** `.planning/ROADMAP.md`, `.planning/REQUIREMENTS.md`.
**Commit:** `5447684`.

### 2. [Rule 2 — missing critical correctness] The "§4.4 5% floor" misquote corrected inline

**Found during:** Task 2.
**Issue:** ROADMAP criterion 5's own wording cites a *"§4.4 5% floor"* that does not exist. §4.4
criterion 1 is two-sided (`≥ ~8%` and `≤ ~35%`); the misquote was corrected project-wide on
2026-09-17, and the implementation it produced had no cap check at all — an occupancy test that
**could confirm and never fail**.
**Fix:** The criterion text is left as written for the historical record, with an evidence block
stating that the floor does not exist, that the criterion was scored against the real two-sided
band (stricter in both directions), and that §4.4 was subsequently amended to carry a recurrence
exemption which classifier #2 does not invoke.
**Files modified:** `.planning/ROADMAP.md`, `platform_design/adr/0002-l1-second-classifier.md`.
**Commits:** `60a4915`, `5447684`.

### 3. [Rule 3 — blocked] STATE.md updated by hand; the GSD state handlers were not used

**Found during:** state update.
**Issue:** `state.advance-plan` fails on this repo (`Cannot parse Current Plan or Total Plans in
Phase from STATE.md` — that file has never carried those fields), and `state.update-progress` /
`roadmap.update-plan-progress` have twice written wrong values here (phase counts, a truncated
sentence, a blanked table row).
**Fix:** `.planning/STATE.md` was edited by hand with scoped replacements and the result read back
against the document body. Frontmatter now reads 8 phases / **7** complete / 47 plans / **47**
complete.
**Commit:** `5447684`.

### 4. [Rule 2] The ADR's probe-edge table was extended, and the edit labelled

**Found during:** Task 1/2.
**Issue:** The table existed at Proposed but several rows described their resolution in prose
without naming a test, the precision row called the tolerance small without stating it, and the
`empty` row was drafted against **K = 3** (now **5** after the 2026-09-18 RE-PIN).
**Fix:** Five rows extended, the numeric tolerance stated, the K correction carried, and an
`AMENDMENT 2026-09-21` note added beneath the table recording exactly what changed. No row's
status changed.
**Commit:** `60a4915`.

## Known Stubs

None. This plan creates no source modules.

## Assumptions in the plan that did not survive contact

The plan's `must_haves.truths` asserted *"REG-01 and INV-01 are both claimed as delivered."*
**REG-01 is claimed PARTIALLY**, by Glenn's explicit decision of 2026-09-18, with criterion 6
named as the open item. The plan was written before the dependence verdict existed; claiming
REG-01 in full would have required reading an INCONCLUSIVE result as a pass, which is the
document-level version of the evidence failure `UAT-AUDIT-2026-09-09` indicts.

The plan's `assumption_delta_decision` also anticipated only two outcomes for criterion 7
(positive lift with high dependence, or positive lift with low dependence). **A third occurred:**
a negative wealth lift and a modest drawdown improvement alongside an unresolved dependence
verdict. The `add-alongside` decision stands, and now stands on measured grounds: none of the
three conditions ADR-0002 named as forcing a promote landed.

## Open items carried into the closing record

1. **Criterion 6 UNRESOLVED.** No tie-break was run and none may be — the pre-registration at
   `298b1bc` forbids it. **No further dependence statistic may be computed on these labelings.**
2. **§4.4 criterion 3's Hungarian subsample-stability test has never been run** for either
   classifier.
3. **ADR-0001 condition (iv)'s covariance clause is unimplemented** — no per-regime covariance
   exists at L4-01; it falls to L3 (design §6.2). Condition (iv) is **not** complete. Only the
   per-regime Sharpe half is implemented, in `allocation/joint_tilt.py`.
4. **`vol_targeted_tilt` and `driver.py:497` remain (iv)-non-compliant** for consumers other than
   the joint harness.
5. **Classifier #1's filtered labeling changes state in 246 of 588 decision months (41.84%)**
   against a 3.60% full-sample rate. No band governs it; that churn feeds the tilt directly.
6. **Classifier #2's §5.4 ratio is 1.074** — median sojourn 29.0 months, median detection lag
   27.0 months, only **5 of 12** transitions resolved. The lag nearly consumes the sojourn.
7. **The crisis state's median sojourn is 3.0 months**, exactly on criterion 2's boundary, against
   a 1–3 month detection lag. Criterion 7's `dd_delta` is **not** evidence crises are nowcastable
   in time to act.
8. **`DEGENERATE_SHARPE_VARIANCE = 1.0` remains a declared assumption** governing every DSR, now
   until 20 independent Sharpe-bearing trials exist.
9. **Audit item A11 remains open by Glenn's deliberate choice** — not closed here.
10. **The L2 CV-robustness question was routed around, not resolved**, and keeps its own future ADR.
11. **`blend_weight_1` was pinned at 0.50 and never swept.** Nothing is known about other weights.
12. **The ADR's own failure signature half fired:** the two routings disagree in sign on `dd_delta`
    (+0.024084 L1-only vs −0.001137 L2, same 588 steps). Recorded, not explained.

## Verification

- `pytest tests/unit/test_platform_legacy_import_ratchet.py -q` → **11 passed**; measured count
  **31**, constant **31**, not raised.
- `pytest tests/ -q` → **1983 passed, 0 skipped, 0 xfailed** (123.92s). Baseline **1983 / 0 / 0**
  held exactly; no new skips.
- ADR-0002 Task-1 verify command → **PASS** (Status section contains no `Proposed`; all twelve
  required keys present).
- Probe-edge table → **8 rows, none with a deferred status** (checked programmatically).
- `git diff --stat .planning/ROADMAP.md` → 100 insertions, 10 deletions, scoped to the Phase 7
  section; criterion 8's correction block byte-identical.

## A note on checks that can only confirm

This project's signature defect is the check that can only confirm — five recorded instances. Two
statements in this plan's output are worth flagging explicitly rather than leaving to be found:

- **The probe-edge table is documentation, not a test.** It *names* tests that can fail, and each
  named test was verified to exist by name on 2026-09-21, but the table itself asserts nothing the
  suite would catch if it drifted. That is a real gap and it is stated rather than papered over.
- **The ratchet's "unchanged at 31" is a genuine measurement that could have come out otherwise.**
  The AST scan is falsifiable in both directions — it would have reported 32 had wave 2 added an
  import, and the suite would have failed. It reported 31.

## Self-Check: PASSED

- `platform_design/adr/0002-l1-second-classifier.md` — FOUND (1044 lines, Status: Accepted)
- `.planning/ROADMAP.md` — FOUND (Phase 7 criteria 5/6/7 marked)
- `.planning/REQUIREMENTS.md` — FOUND (REG-01 `[~]`, INV-01 `[x]`)
- `.planning/STATE.md` — FOUND (7 of 8 phases complete, 47 of 47 plans)
- `tests/unit/test_platform_legacy_import_ratchet.py` — FOUND, unmodified, 11 passed
- Commit `60a4915` — FOUND
- Commit `5447684` — FOUND
