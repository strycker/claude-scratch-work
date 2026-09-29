---
phase: 08-regime-persistence-stability
plan: 11
subsystem: testing
tags: [gap-closure, G-08-1, portability, pytest-approx, honesty-framework]
status: complete
gap_closure: true
closes: [G-08-1]
requirements: [PER-07, PER-10]
dependency_graph:
  requires: [08-10 (criterion-7 records and curves, commit 4bfb81b), 07-11 band-off record (d5c3ac9)]
  provides: [one portable record-vs-curve comparison routine with discrimination arms; swept defect class]
  affects: [UAT test 1 on macOS]
tech-stack:
  added: []
  patterns: ["pytest.approx(rel=1e-9, abs=0.0) as the single float rule for recomputed-vs-committed floats"]
key-files:
  created: []
  modified:
    - tests/unit/test_platform_joint_diagnostics_record.py
    - CLAUDE.md
    - README.md
decisions:
  - "Recomputed-vs-committed floats compare at rel=1e-9 with abs=0.0; booleans by identity; ints/strings exact"
  - "Record-vs-record equalities (one committed run) stay exact"
metrics:
  duration: "~25 min"
  completed: 2026-09-29
  tests_before: 2393
  tests_after: 2433
actuals:
  tokens: 3800
  tasks: 3
  commits: 4
---

# Phase 8 Plan 11: Portable criterion-7 re-derivation (G-08-1) Summary

The criterion-7 re-derivation and the live A11 hurdle check now go through one routine,
`_record_mismatches`. Floats are compared at `rel=1e-9, abs=0.0` and booleans by identity. The
arms show it accepts the exact macOS/Linux pair from UAT test 1 and 1e-13 noise. They also show it
rejects the band-off record, a one-part-per-million error in any field, a zeroed DSR and a flipped
gate. No record, curve or registry row was touched.

## Commits

| Task | Commit | Message |
|------|--------|---------|
| 1 (RED) | 5491bad | test(08-11): add discrimination arms for the criterion-7 re-derivation (RED) |
| 1 (GREEN) | ea6744a | fix(08-11): compare the criterion-7 re-derivation at rel=1e-9, abs=0.0 (G-08-1) |
| 2 | a1ec15a | fix(08-11): route the live A11 hurdle comparison through the portable float rule |
| 3 | a180c86 | docs(08-11): recorded test count 2393 -> 2433 from a live collection (D-07) |

## RED / GREEN

- **RED** (5491bad, `_record_mismatches` still using `==` on floats):
  `15 failed, 32 passed, 95 deselected in 4.78s`. The 15 are the 14
  `test_cross_platform_last_bit_noise_is_absorbed[{7 keys}-{l1only,l2}]` cases plus
  `test_the_uat_macos_pair_is_not_a_mismatch`. The Linux suite now reproduces G-08-1.
- **GREEN** (ea6744a, float branch changed to `_floats_agree`):
  `47 passed, 95 deselected in 4.29s` on `-k "TestCriterion7ReMeasuredIn0810 or TestTheReDerivationToleranceDiscriminates"`.
- Band-off pin check: the verify re-read `git show d5c3ac9:...measurement_l1only.json` and
  printed `band-off pin == d5c3ac9 record on all four fields`.

## How each check can fail (mutations run, then reverted with `git checkout -- <file>`)

| Arm | Observed |
|---|---|
| Re-derivation (`test_the_record_is_re_derivable_from_its_own_curves`) | GREEN on both suffixes. It fails if any of the nine fields differs by more than 1e-9 relative. |
| Absorbed noise (1e-13 x 7 keys x 2 suffixes) + UAT macOS pair | RED under exact `==` (15 failed, above), GREEN under the rule. |
| Band-off (`== [baseline_mean_turnover, dd_delta, joint_mean_turnover, wealth_delta]`) | GREEN. **Mutation rel=2e-2, rel=1e-1, or abs floor 1e-2: RED.** Mutation rel=1e-2: still GREEN, because wealth_delta's genuine divergence is 1.5e-2. That loosening is caught by the 1-ppm arm instead (14 of 14 RED at rel=1e-2). |
| 1 ppm (7 keys x 2 suffixes, exact `[key]`) | GREEN. **Mutation rel=1e-2: 14/14 RED.** **Mutation dropping `abs=0.0`: the 4 DSR cases RED** (joint/baseline x l1only/l2). |
| DSR = 0.0 (2 keys x 2 suffixes) | GREEN. **Mutation dropping `abs=0.0`: 3 of 4 RED** (baseline_dsr-l1only, baseline_dsr-l2, joint_dsr-l2). joint_dsr-l1only (4.13e-12) stays GREEN because it sits above the default 1e-12 floor. That case is covered by the 1-ppm DSR arm. |
| Flipped gate (2 keys x 2 suffixes) | GREEN. **Mutation replacing `is` with a truthiness/always-true bool compare: 4/4 RED.** |
| Hurdle 1 ppm (`test_a_one_part_per_million_hurdle_error_is_rejected`, 2 suffixes) | GREEN. **Mutation rel=1e-5: both RED** (with the 2 `quality_tier_hurdle` 1-ppm arms). |
| Recorded-count pin (`test_docs_recorded_counts.py`) | 7 passed at 2433. It fails if any of the four sites is off the live collection. |

## Sweep (Task 2)

Rule as stated in the plan: IN class = a float recomputed at test time by reduction or
transcendental arithmetic, compared with exact equality to a float read from a committed
artifact. OUT: (a) ints/strings/dates; (b) exact binary fractions from integers (or a correctly
rounded decimal literal); (c) both sides from committed files of the same run; (d) both sides
computed in-process (synthetic or tmp_path); (e) already `approx`/`allclose`/pandas default rtol;
(f) a deliberate bit-identity flag.

File list: `grep -rlE --include=*.py "outputs/reports|data/checkpoints|data/holdout|registry/|read_parquet|json.loads" tests/`
matched 36 files. No `platform`/quote filter was used.

| File:line | What is compared | IN/OUT | Rule | Action |
|---|---|---|---|---|
| test_platform_joint_diagnostics_record.py:768 (was 676-692) | lift (5 floats + 2 bools) and 2 mean turnovers, recomputed from committed curves vs the committed record | **IN** | - | **Fixed** (Task 1): `_record_mismatches(*_rederived(suffix))` |
| test_platform_joint_diagnostics_record.py:872-873 | `lift.quality_tier_hurdle == q.hurdle == expected_max_sharpe(...)` (live) | **IN** (live part) | c for record-vs-record | **Fixed** (Task 2): record-vs-record stays exact; live-vs-record goes through `_record_mismatches` + 1-ppm arm |
| test_platform_joint_diagnostics_record.py:880 | `observed_sharpe > hurdle` (live hurdle) | OUT | not an equality | Left. The margin is 0.90 vs 2.23 (l1only) and 1.16/1.26 vs 2.21 (l2), so it is not ULP-sensitive |
| test_platform_joint_diagnostics_record.py:877-878, 858 | DSR / observed Sharpe / transition-rate record-vs-record | OUT | c | Left |
| test_platform_joint_diagnostics_record.py:939 | registry terminal_log_wealth vs record | OUT | c, e | Left (approx abs=1e-12) |
| test_platform_joint_diagnostics_record.py:130-340, 442-492 | transition counts, n_rows, n_degraded, mismatch counts from curves | OUT | a | Left |
| test_platform_joint_diagnostics_record.py:153, 303, 478, 190, 851 | churn rates / ratio | OUT | e | Left (already approx) |
| test_platform_joint_diagnostics_record.py:583, 601 | `quality_tier_hurdle == expected_max_sharpe(...)` | OUT | d | Left (both in-process) |
| test_platform_evaluation_sojourn_lag.py:371 | median_sojourn/median_lag/ratio (9.5, 4.0, 2.375) + counts vs diagnostics_l1only.json | OUT | b, a | Left |
| test_platform_evaluation_sojourn_lag.py:(TestSignedOffset real-dev arm) | integer lags/offsets | OUT | a | Left |
| test_platform_subsample_stability_record.py:77 | refit integer states vs committed regime_labels | OUT | a | Left (see note below) |
| test_platform_subsample_stability_record.py:238 | runner vs summarize, synthetic | OUT | d | Left |
| test_platform_subsample_stability_record.py:426 | cost matrix vs matched_distance, both committed | OUT | c, e | Left (allclose atol 1e-9) |
| test_platform_subsample_stability_record.py:306-460 | row counts, occupancy, episode tables | OUT | a | Left |
| test_platform_terminal_month_diagnostic.py:334, 357-358, 382 | rates / lambda_over_d | OUT | e | Left |
| test_platform_terminal_month_diagnostic.py:355 | `block["lambda"] == float(cfg lambda)` | OUT | b | Left (a parsed decimal literal, correctly rounded on every IEEE platform) |
| test_platform_terminal_month_diagnostic.py:312-381 (rest) | n_steps, dates, n_changes, k | OUT | a | Left |
| test_platform_nowcaster_recursion.py:334 | S-1 counts/positions on committed belief | OUT | a | Left |
| test_platform_nowcaster_recursion.py:495 | `bit_identical is True`, `max_abs_diff == 0.0` from s1_truncation_invariance.json | OUT | f, c | Left |
| test_platform_nowcaster_recursion.py:(invariance sweeps) | synthetic belief paths | OUT | d | Left |
| test_platform_monthly_spine_consistency.py:88 | passthrough column raw vs features, both committed | OUT | e, c | Left |
| test_platform_monthly_spine_consistency.py:100, 112 | resolved source name, first_valid_index | OUT | a | Left |
| test_platform_recompute_monthly_features.py:97 | oil passthrough, synthetic | OUT | e, d | Left |
| test_platform_evaluation_disagreement.py:164 | prefix baseline n_compared/n_disagree + pct_disagree | OUT | a, e | Left |
| test_platform_labeling_classifier2.py:365 | live occupancy inside a band | OUT | no equality on a committed float | Left |
| test_platform_backtest_driver.py:587 | resolved column names from real monthly_features | OUT | a | Left |
| test_platform_plotting_regime.py:224 | change-point dates/counts from real monthly_features | OUT | a | Left |
| test_platform_honesty_registry.py:160 | live ledger count >= 38 | OUT | a | Left |
| test_platform_holdout.py:273-296 | checkpoint dir paths are not production | OUT | a | Left |
| tests/test_constraints_frequency.py:50-63 | index cadence of seeded checkpoints | OUT | a | Left |
| tests/conftest.py | seeds checkpoints, no assertions | OUT | no equality on a committed float | Left |
| test_platform_transforms.py:316, 477 | checkpoint written by the test from synthetic data | OUT | d | Left |
| test_platform_labeling.py:451-477 | tmp_path checkpoints, allclose | OUT | d, e | Left |
| test_platform_snapshots.py:191-193 | tmp_path checkpoint meta | OUT | d | Left |
| test_checkpoints.py | tmp_path CheckpointManager | OUT | d | Left |
| test_platform_returns.py:181 | tmp_path | OUT | d | Left |
| test_platform_vol.py:113-123 | tmp_path | OUT | d | Left |
| test_platform_gap_lag.py:129-142 | tmp_path | OUT | d | Left |
| test_platform_evaluation_model_metrics.py:271-273 | tmp_path output_dir | OUT | d | Left |
| test_platform_evaluation_report.py | tmp_path reports | OUT | d | Left |
| test_platform_evaluation_churn.py | synthetic matrices | OUT | d | Left |
| test_platform_features_invariants.py | tmp_path registry | OUT | d | Left |
| test_platform_registry.py | tmp_path ledger | OUT | d | Left |
| test_platform_nowcaster.py | tmp_path registry/checkpoints | OUT | d | Left |
| test_platform_plotting.py | tmp holdout dir | OUT | d | Left |
| test_platform_plotting_allocation.py | synthetic | OUT | d | Left |
| test_platform_plotting_backtest.py | synthetic | OUT | d | Left |
| test_platform_plotting_nowcaster.py | synthetic | OUT | d | Left |
| test_platform_splice.py:714 | tmp_path JSON | OUT | d | Left |
| test_reporting.py:206 | tmp_path parquet | OUT | d | Left |
| tests/test_pipelines_ingest_features.py:95-132 | tmp_path | OUT | d | Left |
| tests/integration/test_mini_backtest.py:174 | tmp_path reports | OUT | d | Left |

**Result:** exactly two sites are IN class, both in `test_platform_joint_diagnostics_record.py`,
and both are fixed. The sweep confirms the planner's pre-screen. No other file needed changes.

**Note (a different defect class, left alone):** `test_platform_subsample_stability_record.py:77`
refits a float model and compares INTEGER states elementwise. A near-tie argmax could flip on another
libm. That is a discrete-outcome portability risk, not a float-equality one, and it passed on the
Mac in UAT test 1. It is recorded here and not changed.

## Deviations from Plan

None. The plan was executed as written, and the only file touched in Tasks 1 and 2 is the planned test file.

One honest calibration of the plan's evidence note: the band-off arm does NOT fail at `rel=1e-2`.
The 07-11 vs 08-10 wealth_delta divergence is 1.5e-2 relative, so the arm fails only at `rel=2e-2`
and looser, or with an absolute floor of 1e-2. A `rel=1e-2` loosening is caught by the 1-ppm arm
(14/14 RED). The DSR-zero arm catches a dropped `abs=0.0` on 3 of its 4 cases, because
joint_dsr-l1only (4.13e-12) sits above the default 1e-12 floor. The 1-ppm DSR arm catches all 4.

## Verification

- `pytest tests/unit/test_platform_joint_diagnostics_record.py -q`: 144 passed.
- Live collection: **2393 before -> 2433 after** (`pytest --collect-only -q -p no:cacheprovider tests/`),
  written at CLAUDE.md:128, CLAUDE.md:606, README.md:5 (badge URL) and README.md:477.
  `tests/unit/test_docs_recorded_counts.py`: 7 passed.
- Full suite: `2433 passed, 5 warnings in 391.53s`. That is 0 failed, 0 skipped and 0 xfailed.
- Ratchet: `test_platform_legacy_import_ratchet.py` 11 passed, `MAX_LEGACY_IMPORT_SITES = 31`.
- `git diff --stat c2e632d -- legacy/ gsd-scratch-work/ trading-crab-lib/ registry/ outputs/ data/`: empty.
- `registry/trials.jsonl` sha256 is `c957e8fd...88ad` both before and after, and `total_trial_count() == 44`. No holdout read.

## UAT test 1 re-test (Mac)

`pytest tests/ -q`: expect `2433 passed`, 0 failed.

## Self-Check: PASSED

- FOUND: tests/unit/test_platform_joint_diagnostics_record.py, CLAUDE.md, README.md (modified)
- FOUND commits: 5491bad, ea6744a, a1ec15a, a180c86
