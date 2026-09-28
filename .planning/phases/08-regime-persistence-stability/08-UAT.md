---
status: testing
phase: 08-regime-persistence-stability
source: [08-01-SUMMARY.md, 08-02-SUMMARY.md, 08-03-SUMMARY.md, 08-04-SUMMARY.md, 08-05-SUMMARY.md, 08-06-SUMMARY.md, 08-07-SUMMARY.md, 08-08-SUMMARY.md, 08-09-SUMMARY.md, 08-10-SUMMARY.md]
started: 2026-09-28T15:48:37Z
updated: 2026-09-28T15:48:37Z
---

## Current Test
<!-- OVERWRITE each test - shows where we are -->

number: 1
name: Fresh pull — full suite green at the recorded count
expected: |
  After `git pull` of claude/keen-galileo-zqcml6-w5 and reinstalling both packages
  (pip install -e "src/trading_crab_lib/[all,dev]" && pip install -e ".[dev]"),
  `pytest tests/ -q` ends with "2393 passed" — 0 failed, 0 skipped, 0 xfailed
  (a handful of third-party warnings is normal). The README badge and CLAUDE.md
  both say 2393.
awaiting: user response

## Tests

### 1. Fresh pull — full suite green at the recorded count
expected: After `git pull` and reinstalling both packages, `pytest tests/ -q` ends with "2393 passed" and 0 failed / 0 skipped / 0 xfailed; README badge and CLAUDE.md both say 2393.
result: [pending]

### 2. Weekly report — the trading surface shows the band and the rewording
expected: `python -m trading_crab_lib.platform.report.weekly` (no --send-email) completes. The markdown shows "active regime: regime N" (or "none (neutral posture)"), the sentence "The active regime is a reported label ... gates no weight (audit item A7, 08-A7.md)", and above the trades "Targets below are the EXECUTED book after the 5.0% no-trade band". Running it a second time in the same month gives the same targets — the band does not compound.
result: [pending]

### 3. Registry at the ADR-0002 ceiling, rows correctly marked
expected: `python -c "from trading_crab_lib.platform.honesty.registry import total_trial_count as t; print(t())"` prints 44. `registry/trials.jsonl` has 7 lines; the last two are tagged 08-10-c1-alone-L1only-notrade5pp and 08-10-joint-c1xc2-L1only-notrade5pp, each with "independent_trial": false and sharpe 0.896446 / 0.899378.
result: [pending]

### 4. Criterion 7 record — band on, A11 gate FAILED as accepted in advance
expected: outputs/reports/platform/joint_lift/measurement_l1only.json shows lift.wealth_delta -0.125307, lift.dd_delta +0.026164 over 588 steps; quality_tier.verdict "FAILED on 2 of 2 legs (baseline, joint)" at hurdle 2.226891; both legs' state_1_transition_rate 0.419080.
result: [pending]

### 5. A11 ruling record reads as your decision, as a reversal
expected: 08-A11.md records your ruling b-promote-dsr, written as a reversal of the 2026-09-18 decision to leave A11 open, with "registry rows spent: 0", the criterion-7 MET -> FAILED consequence accepted in advance, and the sharpe_variance = 1.0 placeholder caveat stated plainly. ADR-0003 is Accepted and names b-promote-dsr.
result: [pending]

### 6. A7 / no-trade band ruling record matches what you decided
expected: 08-A7.md carries your four 2026-09-24 rulings (b-bounded-turnover as a 5pp no-trade band; hysteresis per-classifier with the blend mismatch declared; tau not swept; keep 0.70 / 0.40), the phrase "not swept", "registry rows spent: 2", and states that active_regime gates no weight.
result: [pending]

### 7. S-1 halt and your invariance ruling are recorded honestly
expected: 08-CHURN.md records the halt as measured (3 LEADs on classifier #2), the four truncation cuts all bit-identical, your 2026-09-23 ruling "causal invariance governs; S-1 is observational", and the orchestrator's two corrected over-claims ("a leak everywhere"; "exactly {13, 44, 68}") marked as corrections, not silently rewritten.
result: [pending]

### 8. Track A diagnosis — terminal-month edge artefact refuted
expected: 08-TRACK-A.md shows classifier #1 churn flat in k (246 / 242 / 245 / 248 / 247 / 249 of 587 for k = 1..6), classifier #2 24-25, the verdict "refuted" per criterion 3's pre-registered rule, and both named limits (edge effects > 6 months not ruled out; lambda/d not isolated).
result: [pending]

### 9. Stability results — your judgment on the recorded picture
expected: 08-STABILITY.md quotes the 2363752 pre-registered prediction ahead of the results; reports every classifier x scheme x state row against its split-half null with occupancy; records classifier #1 state 2 as DEGENERATE under leave-one-episode-out without calling it a failure; invents no threshold; and states that the evaporated flag fired 0 of 8,883 times while reference_months_in_subsample carries the truth. You judge whether this is an honest record you can act on.
result: [pending]

### 10. Closing record — every criterion, number, window and verdict
expected: 08-MEASUREMENTS.md gives each of criteria 0-9 a verdict with its number, denominator and window, none MET on a shape check; separates decision-bearing from observational; states the priority vocabulary finding (35.7% / 41.3% best-aligned) and which numbers it qualifies; and tallies 13 confirm-only checks including the orchestrator's own.
result: [pending]

### 11. 08-03 D1 — standardization_params inverts standardize_features elementwise
expected: covered by tests/unit/test_platform_labeling_stability.py
result: pass
source: automated
coverage_id: 08-03-D1

### 12. 08-03 D2 — Trap A test fails on the trap (de-standardized matched distance)
expected: covered by tests/unit/test_platform_labeling_stability.py
result: pass
source: automated
coverage_id: 08-03-D2

### 13. 08-03 D3 — Hungarian matching recovers a known non-identity permutation
expected: covered by tests/unit/test_platform_labeling_stability.py
result: pass
source: automated
coverage_id: 08-03-D3

### 14. 08-03 D4 — split-half null non-zero at n=40 and falling with n
expected: covered by tests/unit/test_platform_labeling_stability.py
result: pass
source: automated
coverage_id: 08-03-D4

### 15. 08-03 D5 — Trap B zero-occupancy state flagged evaporated (SYNTHETIC ONLY)
expected: covered by tests/unit/test_platform_labeling_stability.py on synthetic fixtures. CAVEAT recorded by the orchestrator: on real data the flag cannot fire under a K-fixed refit (0 of 8,883 rows; 36 rows lost every reference month). The real-data question is carried by test 9 and STATE.md, not by this auto-pass.
result: pass
source: automated
coverage_id: 08-03-D5

### 16. 08-03 D6 — four subsample schemes; circular bootstrap preserves length n
expected: covered by tests/unit/test_platform_labeling_stability.py
result: pass
source: automated
coverage_id: 08-03-D6

### 17. 08-03 D7 — Trap C: run_stability raises on any column-set difference
expected: covered by tests/unit/test_platform_labeling_stability.py
result: pass
source: automated
coverage_id: 08-03-D7

### 18. 08-03 D8 — no threshold constant, verdict field or Wasserstein estimator
expected: covered by tests/unit/test_platform_labeling_stability.py
result: pass
source: automated
coverage_id: 08-03-D8

### 19. 08-07 D1 — both reference labelings reproduce their checkpoints elementwise
expected: covered by tests/unit/test_platform_subsample_stability_record.py
result: pass
source: automated
coverage_id: 08-07-D1

### 20. 08-07 D2 — rows keyed on the matched partner
expected: covered by tests/unit/test_platform_subsample_stability_record.py
result: pass
source: automated
coverage_id: 08-07-D2

### 21. 08-07 D3 — all four schemes ran for both classifiers, exact row counts
expected: covered by tests/unit/test_platform_subsample_stability_record.py
result: pass
source: automated
coverage_id: 08-07-D3

## Summary

total: 21
passed: 11
issues: 0
pending: 10
skipped: 0
blocked: 0

## Gaps

[none yet]
