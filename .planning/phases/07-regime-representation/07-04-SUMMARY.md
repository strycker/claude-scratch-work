---
phase: 07-regime-representation
plan: 04
subsystem: platform-backtest
tags: [jump-model, walk-forward, feature-policy, adr, honesty-framework, trial-registry]

# Dependency graph
requires:
  - phase: 07-regime-representation
    provides: "07-01's frozen driver/reference L1 feature-column mechanism (trial_tag, frozen_l1_features, TestFrozenPolicyEquivalence), 07-02's D-02-A-recomputed ten-column checkpoint, and 07-03's measured wave-1 numbers (07-MEASUREMENTS.md) plus its binding window-narrowing condition"
provides:
  - "A three-state pre/post comparison section in assemble_backtest_report (policy_comparison keyword, default None, byte-identical output when omitted), wired end-to-end in run_full_backtest_evaluation from the run's own live numbers plus a dated, never-recomputed _PREFIX_BASELINE_20260914 historical constant."
  - "A13_CAVEAT rewritten from 'NOT INTERPRETABLE ... changes 7 times' to a resolution narrative naming both D-08 licensing artifacts (TestFrozenPolicyEquivalence, the resolved-transition denominator), retired on the strength of the fix, never by softened wording."
  - "platform_design/adr/ — new to this repo, no prior precedent. README.md establishes the convention (sequence-prefixed filenames, five canonical headers, superseded-in-place). 0001-l1-feature-policy.md records the policy, three evidence-backed rejected alternatives, the trial ceiling as a formula against a live read_trials() count, and every wave-1/wave-2 deferral as an explicit decision."
affects: ["wave 2's second /gsd-plan-phase 7 pass (D-09) — INV-01 and REG-01's remaining clauses (criteria 5-7), which this ADR names but does not implement"]

# Actuals (#2632)
actuals:
  tokens: 16350
  tasks: 3
  commits: 3

tech-stack:
  added: []
  patterns:
    - "Optional keyword-only dict parameter with a None default that guarantees byte-identical output when omitted (assemble_backtest_report's policy_comparison) — the same 'additive, never breaking' shape as prior optional report keywords (excluded_assets, trial_tag)."
    - "Dated, provenance-commented module-level historical constant (_PREFIX_BASELINE_20260914) that is read from but never recomputed — the pre-fix numbers are transcribed once from two upstream documents and diffed against live per-run numbers, never re-derived."
    - "ADR directory convention established fresh (no prior precedent): sequence-prefixed filename, five canonical section headers matching GSD's adr-parser.cjs vocabulary, README.md as the convention record, cross-referenced from the design doc by a single pointer line rather than duplicated content."
    - "Content-only constant re-pin, matching plan 07-02's established convention: A13_CAVEAT's two invalidated assertions in test_platform_plotting.py are removed/replaced with a comment naming the date and the licensing decision (D-08), never silently loosened."

key-files:
  created:
    - platform_design/adr/README.md
    - platform_design/adr/0001-l1-feature-policy.md
  modified:
    - src/trading_crab_lib/platform/evaluation/report.py
    - src/trading_crab_lib/platform/plotting/core.py
    - tests/unit/test_platform_evaluation_report.py
    - tests/unit/test_platform_plotting.py
    - platform_design/platform_design.md
    - outputs/reports/platform/backtest_report.md
    - registry/trials.jsonl

key-decisions:
  - "Window-narrowing binding condition (07-03-SUMMARY.md) carried forward and foregrounded INLINE in both artifacts this plan produces: every disagreement/ratio table cell embeds its own n_compared/n_resolved denominator and date window directly in the cell string (never a separate footnote), and the ADR's Consequences section states the 232-vs-118 degraded-step difference and the 356-vs-470-month windows immediately below the pre/post table, before any other discussion."
  - "D-08's two licensing artifacts verified present in the SAME task that retires the caveat: TestFrozenPolicyEquivalence re-run green (Task 2's own verify block), and the resolved-transition denominator regression-pinned in Task 1 (test_headline_still_emits_the_resolved_denominator). The caveat was rewritten only after both were confirmed, never on the strength of wording alone."
  - "D-03's rejected imputation alternative recorded with its own real numbers, including where they look nominally BETTER than the accepted policy (wealth_delta +0.4009 vs +0.3778, dd_delta +0.0190 vs -0.0661) — the ADR states explicitly that the rejection rests on the pre-declared non-causal-imputation reasoning (D-04), not on these numbers, so a reader cannot mistake the favorable-looking figures for the reason it was rejected."
  - "Trial ceiling stated as the formula `2 x N_full_evaluation_runs` (ASCII 'x', matching the plan's exact grep target — an earlier draft used the Unicode multiplication sign and was corrected) plus a LIVE read_trials() count (42, read 2026-09-15T00:14:31Z) rather than any planning-document figure; 30/34/~35 are named explicitly as rejected stale values."
  - "wealth_delta/dd_delta recomputed locally in run_full_backtest_evaluation (the same one-line subtraction assemble_backtest_report's own section 3 already performs) rather than importing plotting/backtest.py::compute_ablation_delta into evaluation/report.py — avoids adding a new evaluation-to-plotting import dependency for a formula that is one line either way."
  - "Criterion-3 disagreement measurement (measure_label_disagreement) is now computed IN-PROCESS inside run_full_backtest_evaluation from the same in-memory full_sample_states/filtered_probs_matrix objects step (d) already produced, rather than re-read from the persisted parquet artifacts afterward — one fewer round-trip, and the disagreement/policy_comparison keys are now part of the function's own return dict for any future caller."

requirements-completed: [REG-01]

coverage:
  - id: D1
    description: "assemble_backtest_report gains an optional policy_comparison keyword (default None, byte-identical rendering when omitted) rendering a three-labelled-column table (quantity / superseded 9-col pre-fix / frozen 10-col post-fix) with every disagreement and §5.4-ratio cell carrying its own n_compared/n_resolved denominator and date window inline, plus the compound-cause caveat and the Brier mechanical-explanation prose, placed between the ablation-delta and smoothed-vs-filtered-gap sections."
    requirement: "REG-01"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_evaluation_report.py::TestPolicyComparisonSection (5 tests: all-three-states, omitted-by-default byte-identical, compound-cause prose, Brier mechanical note, resolved-denominator regression pin)"
        status: pass
      - kind: unit
        ref: "tests/unit/test_platform_evaluation_report.py tests/unit/test_platform_plotting_backtest.py -q (64 tests)"
        status: pass
      - kind: integration
        ref: "tests/integration/test_mini_backtest.py (6 tests, unaffected by the new wiring on synthetic data)"
        status: pass
      - kind: manual_procedural
        ref: "python -m trading_crab_lib.platform.evaluation.report against the real dev checkpoint: n_steps=356, brier=0.20717126722358387, matching 07-MEASUREMENTS.md exactly; rendered table inspected directly in outputs/reports/platform/backtest_report.md"
        status: pass
    human_judgment: false
  - id: D2
    description: "A13_CAVEAT retired to a resolution narrative (no longer claims uninterpretability, no longer carries the 7-change/4-13-feature progression) on the strength of both D-08 licensing artifacts, named explicitly in the new text; constant and all three consumers (plotting/backtest.py, nowcaster.py, regime.py) preserved unchanged; the two invalidated assertions in test_platform_plotting.py re-pinned with a dated D-08 comment, matching plan 07-02's established re-pin convention."
    requirement: "REG-01"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_plotting.py::TestA13CaveatResolution (4 tests) and test_platform_plotting_backtest.py (unmodified, still green including the line-410 generic caveat-rendering pin)"
        status: pass
      - kind: unit
        ref: "tests/unit/test_platform_backtest_driver.py -k FrozenPolicyEquivalence (6 tests, D-08 artifact (a) re-verified in the same task)"
        status: pass
    human_judgment: false
  - id: D3
    description: "platform_design/adr/0001-l1-feature-policy.md and platform_design/adr/README.md created; the ADR names all ten frozen columns, records three considered options each with a stated rejection reason (the 13-col+imputation option cited against its real logged-trial numbers), restates D-04's selection criterion, states the trial ceiling as a formula plus a live read_trials() count, and records INV-01's D-09 deferral, REG-01's partial coverage, the market-cap/GDP block, A11's open status, the canonicalize_states/A14 flag, all eight spec-less probe edges, and the full D-10..D-17 wave-2 direction as explicit decisions. One cross-reference line added to platform_design.md §5.4."
    requirement: "REG-01"
    verification:
      - kind: other
        ref: "grep-based header/coverage verify block from 07-04-PLAN.md's own <verify> (ADR_HEADERS_OK, ADR_COVERAGE_OK, both echoed); python -c read_trials() length check (42)"
        status: pass
    human_judgment: false

duration: ~2h10min
completed: 2026-09-15
status: complete
---

# Phase 7 Plan 4: Feature-Policy Pre/Post Comparison, A13 Retirement, and ADR-0001

**The backtest report now carries a three-labelled-state pre/post comparison with the window-narrowing binding condition foregrounded inline in every affected cell, the A13 caveat is retired to a resolution narrative on the strength of its two D-08 licensing artifacts (never by softened wording), and `platform_design/adr/0001-l1-feature-policy.md` records the ten-column policy, its three evidence-backed rejected alternatives, and every wave-1/wave-2 deferral as an explicit decision — establishing this repo's first ADR convention.**

## Performance

- **Duration:** ~2h10min
- **Tasks:** 3 (all `auto`, Task 1 and Task 2 `tdd="true"`)
- **Files modified:** 5 source/test files, 2 new ADR files, 1 design-doc cross-reference, 2 regenerated output artifacts (`backtest_report.md`, `registry/trials.jsonl`)

## Accomplishments

- **Task 1** (`85e5925`): `assemble_backtest_report` gained an optional keyword-only `policy_comparison: dict | None = None` parameter. When supplied, a new section renders between the no-regime-ablation delta and the smoothed-vs-filtered gap: an 11-row table (quantity name, pre-fix column explicitly headed as the superseded 9-column stale baseline, post-fix column explicitly headed as the frozen 10-column policy), with `pct_disagree`/§5.4-ratio cells carrying their `n_compared`/`n_resolved` denominators and date windows embedded directly in the cell string — the 07-03 human sign-off's binding condition applied literally, not as a separate paragraph. Below the table: a compound-cause caveat (D-01 driver freeze + D-02-A feature-space correction changed together, cannot be separated), a sample-comparability note, the ten frozen column names listed explicitly, and the Brier mechanical-explanation paragraph. `run_full_backtest_evaluation` was rewired to build this mapping from its own live-computed `strategy_kpis`/`ablation_kpis`/`sojourn_lag`, a new in-process `measure_label_disagreement` call (reusing the already-in-memory `full_sample_states`/`filtered_probs_matrix` rather than re-reading parquet), a Brier value read back from what `report_model_metrics` just wrote, and a new dated `_PREFIX_BASELINE_20260914` module constant transcribed from `07-PREFIX-EVIDENCE.md`/`BASELINE-v1-tracer-bullet.md`. Verified twice against the real dev checkpoint (`n_steps=356`, `brier=0.20717126722358387`, matching `07-MEASUREMENTS.md` exactly); the second run fixed a date-formatting defect (`Timestamp.__str__` emitting a spurious `00:00:00`) found during the first verification run.
- **Task 2** (`8175ff8`): `A13_CAVEAT`'s content rewritten from "NOT INTERPRETABLE ... changes 7 times ... (4 -> 6 -> 8 -> 9 -> 10 -> 12 -> 13 features)" to a resolution narrative stating the ratio now compares two labelings fit on one shared, frozen feature space, proved identical at every sampled decision date by `TestFrozenPolicyEquivalence`, published with its resolved-transition denominator, still a small-sample indicative number. The retired seven-change progression moved to the ADR (its correct home as historical record). Constant, re-export, and all three consumers (`plotting/backtest.py`, `nowcaster.py`, `regime.py`) untouched — content only. The two invalidated assertions in `test_platform_plotting.py` were re-pinned (removed the "not interpretable" check; kept the audit-identifier check) and a new `TestA13CaveatResolution` class (4 tests) added, checking the uninterpretability claim is gone from BOTH the constant and its surrounding source comment (so it cannot survive by hiding one line up), the audit identifier survives, both D-08 artifacts are named by string, and the constant/re-export still resolve identically. `test_platform_plotting_backtest.py:410` (the generic caveat-rendering regression pin) was left completely unmodified and stayed green.
- **Task 3** (`f1246f6`): Created `platform_design/adr/` (no prior precedent in this repo — the legacy root `CLAUDE.md` ADR log is scoped to the frozen legacy pipeline). `README.md` establishes the convention (zero-padded sequence prefix, kebab-case slug, one decision per file, five canonical headers matching GSD's `adr-parser.cjs` vocabulary, superseded-in-place never deleted). `0001-l1-feature-policy.md` records: the A13 mechanism as a policy disagreement not a defect; the decision (freeze the driver to the ten named columns, `_refit_l2`/`_cv_safe_active_features` deliberately untouched); three considered options with rejection reasons (expand-the-reference rejected on the merits; all-13-plus-imputation rejected WITH the logged trial's real numbers, including where they look nominally better and why that didn't matter; pin-to-stale-9-column rejected on D-02-A's `oil`-truncation evidence); the compound-baseline consequences under all three labelled states with the window-narrowing binding condition foregrounded immediately below the table (not a footnote); D-04's selection criterion restated (the `dd_delta`-veto offer and its decline); the trial ceiling as the literal formula `2 x N_full_evaluation_runs` plus a live `read_trials()` count (42, read 2026-09-15T00:14:31Z) with `30`/`34`/`~35` named as explicitly rejected stale figures; and every deferral (INV-01 to wave 2 by D-09, REG-01's partial coverage named clause-by-clause, the market-cap/GDP block restated verbatim, A11 left open and conscious, the `canonicalize_states` sort-key/A14 flag for wave 2, all eight spec-less probe edges with one-line resolutions or deferrals, and the full D-10 through D-17 wave-2 direction) recorded as an explicit decision rather than an omission. One cross-reference line added to `platform_design.md` §5.4.

## Task Commits

1. **Task 1: A three-state pre/post comparison section in the backtest report** — `85e5925` (feat)
2. **Task 2: Retire the A13 caveat on the strength of its two licensing artifacts** — `8175ff8` (fix)
3. **Task 3: The feature-policy ADR, with its rejected alternatives and its open deferrals** — `f1246f6` (docs)

_Note: Tasks 1-2 are `tdd="true"`; tests were written and verified RED (missing keyword/removed assertions failing against the pre-fix constant) before the corresponding source edits, then GREEN after, within each task's development, consistent with this repo's one-commit-per-completed-`type="auto"`-task convention._

## Files Created/Modified

- `src/trading_crab_lib/platform/evaluation/report.py` — `assemble_backtest_report` gains `policy_comparison`; new `_PREFIX_BASELINE_20260914` module constant; new `_build_policy_comparison`/`_fmt_dd` helpers; `run_full_backtest_evaluation` computes `disagreement` in-process after step (d), moves `kpi_table`/`wealth_delta`/`dd_delta` earlier, reads the Brier value back from its own just-written parquet, builds and threads `policy_comparison`, and returns `disagreement`/`policy_comparison` as two new result keys.
- `src/trading_crab_lib/platform/plotting/core.py` — `A13_CAVEAT` content and its explanatory comment block rewritten to the resolution narrative; dated, names D-01/D-02-A/D-08.
- `tests/unit/test_platform_evaluation_report.py` — new `TestPolicyComparisonSection` (5 tests).
- `tests/unit/test_platform_plotting.py` — `test_a13_caveat_names_audit_item` trimmed to its still-valid assertion; new `TestA13CaveatResolution` (4 tests).
- `platform_design/adr/README.md` (created) — the ADR convention.
- `platform_design/adr/0001-l1-feature-policy.md` (created) — the policy ADR.
- `platform_design/platform_design.md` — one cross-reference line at §5.4.
- `outputs/reports/platform/backtest_report.md`, `registry/trials.jsonl` — regenerated by two real verification runs of `python -m trading_crab_lib.platform.evaluation.report` against the dev checkpoint (registry grew by 4 untagged rows, 2 per run).

## Decisions Made

See `key-decisions` in the frontmatter for the six load-bearing decisions this plan made (window-narrowing inline placement, D-08 same-task verification, D-03's evidence-backed-not-flattering-numbers framing, the ASCII-`x` trial-ceiling formula, the local-arithmetic vs. cross-module-import choice for `wealth_delta`/`dd_delta`, and the in-process disagreement computation).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Date formatting emitted a spurious `00:00:00` in the pre/post table's window strings**
- **Found during:** Task 1, first real-checkpoint verification run of `python -m trading_crab_lib.platform.evaluation.report`
- **Issue:** `disagreement["first_common_date"]`/`["last_common_date"]` are `pd.Timestamp` objects; the initial f-string interpolation (`f"{ts}"`) rendered the full `Timestamp.__str__` representation (`1974-02-28 00:00:00`) instead of the clean `YYYY-MM` format used on the pre-fix constant's side, making the two windows visually inconsistent in the same table row — exactly the kind of presentation defect the binding condition exists to prevent.
- **Fix:** Added a local `_fmt_month` helper inside `_build_policy_comparison` using `pd.Timestamp(value).strftime("%Y-%m")`, with an `"n/a"` fallback for `None` (the `n_compared == 0` disjoint-span case).
- **Files modified:** `src/trading_crab_lib/platform/evaluation/report.py`
- **Verification:** Re-ran `python -m trading_crab_lib.platform.evaluation.report` against the real checkpoint; confirmed the rendered table now reads `1974-02 -> 2017-05` (not `1974-02-28 00:00:00 -> ...`); `pytest tests/unit/test_platform_evaluation_report.py` re-run green.
- **Committed in:** `85e5925` (Task 1 commit — the fix landed before the commit, so no separate fix commit was needed).

**2. [Rule 1 - Bug] Trial-ceiling formula used the Unicode multiplication sign instead of the literal ASCII text the plan's verify grep targets**
- **Found during:** Task 3, running the plan's own `<verify>` grep for `'2 x N_full_evaluation_runs'`
- **Issue:** The ADR's first draft wrote the formula as `2 × N_full_evaluation_runs` (Unicode `×`), which reads identically to a human but does not literally match `grep -qF '2 x N_full_evaluation_runs'` (ASCII `x`) — the plan's own automated `ADR_COVERAGE_OK` check would have failed silently if not caught.
- **Fix:** Changed the formula's rendering to the literal ASCII string `2 x N_full_evaluation_runs`, matching the plan's acceptance criteria exactly. Other, unrelated Unicode `×` uses elsewhere in the ADR's prose (e.g. "joint (#1 × #2) allocation lift") were left as-is since they are not the tested string.
- **Files modified:** `platform_design/adr/0001-l1-feature-policy.md`
- **Verification:** `grep -qF '2 x N_full_evaluation_runs' platform_design/adr/0001-l1-feature-policy.md && echo MATCH` — confirmed.
- **Committed in:** `f1246f6` (Task 3 commit — caught and fixed before committing).

---

**Total deviations:** 2 auto-fixed (both Rule 1 bugs found and fixed during this plan's own development, before either task's commit).
**Impact on plan:** Both are presentation-correctness fixes with no effect on any measured number. No scope creep.

## Issues Encountered

None beyond the two auto-fixed issues above. The plan's own literal verify commands (grep for the exact formula string, `read_trials()` count, `pytest tests/ -q`) caught both defects before they could land in a commit.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- **Wave 1 (D-09) is now complete in full.** All four wave-1 success criteria are satisfied: criterion 1 (shared feature policy + failing-test-on-divergence + ADR), criterion 2 (A13 caveat retired with both licensing artifacts present, ratio published with its denominator), criterion 3 (post-fix disagreement measured against the pre-fix baseline, satisfied as worded per the human sign-off), and criterion 4 (ablation delta re-measured on both axes, reported under all three labelled states).
- **Per D-09, the phase is ready for its second planning pass.** Run `/gsd-plan-phase 7` again to plan wave 2 (criteria 5, 6, 7, and INV-01) — `platform_design/adr/0001-l1-feature-policy.md`'s "Deferrals and open items" section is the authoritative starting point, carrying forward D-10 through D-17 and the `canonicalize_states`/A14 flag.
- Live trial registry stands at **42 rows** as of this plan's ADR write (2026-09-15T00:14:31Z); wave 2's planning pass should re-read it live rather than quoting this figure, per the ADR's own discipline.
- The published `outputs/reports/platform/backtest_report.md` now carries the new comparison section end-to-end against real data; no further regeneration is required before wave 2 unless wave 2's own work changes the underlying numbers.
- No blockers.

---
*Phase: 07-regime-representation*
*Completed: 2026-09-15*

## Self-Check: PASSED

`platform_design/adr/README.md` and `platform_design/adr/0001-l1-feature-policy.md` confirmed present on disk; all three task commits (`85e5925`, `8175ff8`, `f1246f6`) confirmed present in git history via `git log --oneline -3`; `python -c "from trading_crab_lib.platform.honesty.registry import read_trials; print(len(read_trials()))"` re-confirmed 42; full suite re-confirmed at 1735 passed, 0 skipped after Task 3 (docs-only, no test-count change from Task 2's 1735).
