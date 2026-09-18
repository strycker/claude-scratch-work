---
phase: 07-regime-representation
plan: 09
subsystem: platform/evaluation
tags: [dependence, criterion-6, pre-registration, block-permutation, honesty-framework]
status: complete
requires:
  - "07-08 (regime_labels_2 — classifier #2's labeling)"
  - "ADR-0001 § RE-PIN 2026-09-18 (classifier #1 at K=6, lambda=10)"
  - "design §4.4 AMENDMENT 2026-09-18 (recurrence exemption)"
provides:
  - "platform/evaluation/dependence.py — measure_labeling_dependence, format_dependence_report, block_permutation_null"
  - "07-DEPENDENCE.md — pre-registration, measured record, control result"
  - "criterion 6 — measured, verdict INCONCLUSIVE (NOT met, NOT failed)"
affects:
  - "07-12 (must claim REG-01 partially, naming criterion 6 as open)"
---

# 07-09 — Labeling dependence (criterion 6)

## Outcome

**Criterion 6 is UNRESOLVED.** Not satisfied, not failed. The pre-registered decision rule
returned inconclusive, and that is the recorded result.

## Tasks

| task | status |
|---|---|
| 1 — `dependence.py` + oracle tests | complete (`a603f0e`, `3c33262`) |
| 2 — measure on real labelings, write the record | complete (`70c62fe`, `4e67813`) |
| 3 — **human-verify gate** | **SIGNED OFF by Glenn, 2026-09-18** |

## Task 3 sign-off, recorded verbatim in substance

Glenn declined to rule on the raw statistics, directing instead that a block-permutation control
be run **with its reading pre-registered before the control existed**, and that classifier #1 be
re-pinned first so the control would run on final labelings rather than on a classifier about to
be discarded. On the outcome he directed: proceed to 07-10 through 07-12 with criterion 6
recorded as unresolved, and have 07-12 claim REG-01 **partially**, naming criterion 6 as the open
item rather than folding it in quietly.

## What was measured

Both classifiers were re-pinned before the final measurement, voiding the first numbers:
classifier #1 K 5→6 / λ 52→10, classifier #2 K 3→5 / λ 32→16.

| statistic | first (void) | final |
|---|---|---|
| Adjusted Rand | 0.4686 | **0.354841** |
| NMI | 0.6023 | **0.464088** |
| Cramér's V | 0.6558 | **0.589748** |

Control: 2000 resamples, seed 20260918, block counts #1=26 / #2=13. Observed NMI sits at the
**96.60th percentile** of the null; p95 = 0.4492, p99 = 0.5012. Null median NMI 0.3406 — a
substantial share of the raw association was temporal blockiness. Cramér's V sits at the 87.70th
percentile, inside the null.

Rule committed at `298b1bc`, before `block_permutation_null` was written:
`p95 < observed ≤ p99` → **INCONCLUSIVE**, no tie-break.

## Deviations from plan

1. **The plan had no rule for "inconclusive."** Its stated branch was (a)/(b)/(c) with (c)
   stopping 07-10. The pre-registration created a fourth outcome. Resolved by the human, not by
   an agent picking the nearest branch.
2. **The control is not in the plan.** `block_permutation_null` and six oracle tests were added
   on Glenn's direction. The oracles fail in both directions by construction — identical
   labelings must clear p99 (else the control could never return (a)); independent blocky
   labelings must not (else it could never return (b)).
3. **Two re-pins preceded the measurement**, neither in this plan's scope: ADR-0002 § RE-PIN and
   ADR-0001 § RE-PIN plus the §4.4 amendment that made the latter possible.
4. **`07-PATTERNS.md`'s primary oracle was found unrealizable** — ARI 0.1667 / NMI 0.5794 /
   Cramér's V 0.5 on a 2×2 is not jointly achievable. Both corrected oracles are pinned plus a
   third guard, rather than one being silently chosen.

## Open obligations this plan hands downstream

1. **ADR-0001 exemption condition (iv) is unimplemented.** Classifier #1's crisis state must
   carry low-n flags and must not feed unshrunk per-regime Sharpe or covariance without partial
   pooling. Currently a promise, not code. **07-10 and 07-11 are where it binds.**
2. **§4.4 criterion 3's Hungarian subsample test has never been run** for either classifier.
3. **Crisis-state median sojourn is 3.0 months** against §5.4's typical 1–3 month detection lag.
   Criterion 7's drawdown number must not be read as evidence crises are nowcastable in time to
   act — that is L2's question and it is unanswered.

## Verification

`tests/unit/test_platform_evaluation_dependence.py` — 27 passed. Full suite **1907 passed, 0
skipped, 0 xfailed**. Ratchet unchanged at 31. `ruff` clean. D-15 enforced structurally: an AST
test asserts the flag constants are referenced in no function but `format_dependence_report`.
