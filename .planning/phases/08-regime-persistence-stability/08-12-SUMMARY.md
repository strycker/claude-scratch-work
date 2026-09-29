---
phase: 08-regime-persistence-stability
plan: 12
subsystem: platform/report (serving path) — planning record only
tags: [gap-closure, G-08-2, serving, ruling, honesty, no-registry]
status: complete
requires:
  - "08-UAT.md G-08-2 (the NO_REGISTRY serving-fit ruling, 2026-09-28)"
  - "ADR-0004 trial-budgeting policy (gap closure budget 0)"
provides:
  - ".planning/phases/08-regime-persistence-stability/08-SERVING.md §0-§2 (§3 reserved for 08-14)"
  - "Recorded rulings q1-c (latest complete month + 3-month staleness cap) and q2-ii (disclose distinct-posterior count)"
affects:
  - "08-13 (interim fail-loud), 08-14 (implements exactly q1-c + q2-ii)"
  - "Phase 9 (recipe fix for the input-independent serving posterior, logged as an open item)"
tech-stack:
  added: []
  patterns: ["read-only measurement via direct recipe calls; registry sha256 + total_trial_count before/after"]
key-files:
  created:
    - .planning/phases/08-regime-persistence-stability/08-SERVING.md
  modified: []
decisions:
  - "Q1 -> q1-c: weekly scores the newest month complete in every model column (2026-06-30 today), names the as-of month and the missing columns, and raises ValueError before any save when the scored month is more than 3 month-ends behind the newest monthly_features row (the orchestrator's data-relative reading of 'more than 3 months old', flagged for Glenn's confirmation)"
  - "Q2 -> q2-ii: weekly prints the exact np.unique count of distinct posterior vectors across all full-span complete months under the regime distribution; the recipe fix goes to Phase 9"
metrics:
  duration: "~20 min"
  completed: 2026-09-29
estimate:
  tokens: 30000
  tasks: 3
actuals:
  tokens: 5500
  tasks: 3
  commits: 3
---

# Phase 8 Plan 12: Serving facts and rulings record (G-08-2) Summary

The serving nowcaster was re-measured read-only with the evaluated `_refit_l2` recipe. It trains on
153 rows (2007-04-30 → 2019-12-31) holding 3 of 6 states, {0: 11, 3: 137, 4: 5}. Across all 231
complete months (2007-04-30 → 2026-06-30) it returns exactly one distinct posterior,
{0: 0.418, 3: 0.564, 4: 0.018}. The latest live row, 2026-08-31, lacks `div_yield`, `fred_m2sl`
and `fred_totalsl`. Glenn's rulings are recorded verbatim against those facts: q1-c (score the
latest complete month, with a 3-month staleness cap) and q2-ii (disclose the distinct-vector count
on the page). **registry rows spent: 0.**

## Tasks

| # | Task | Commit | Files |
|---|------|--------|-------|
| 1 | Measure the serving facts read-only; write 08-SERVING.md §0 and §1 | `aed8fe8` | 08-SERVING.md |
| 2 | Checkpoint: decision (ruled before execution; see below) | — | — |
| 3 | Record both rulings verbatim in §2; reserve §3 for 08-14 | `b2683a2` | 08-SERVING.md |

**Task 2 was not a stop.** Glenn answered before execution (`<ruling-received>`, 2026-09-29). The
checkpoint rule said to continue only if Task 1's re-measured facts matched the ruling block, and
every one of them did:

| Fact | Ruling block | Re-measured (2026-09-29, `d2c742f`) |
|---|---|---|
| Training block | 153 rows, 2007-04-30 → 2019-12-31, {0: 11, 3: 137, 4: 5} | **same** |
| Distinct posteriors | 1, {0: 0.418, 3: 0.564, 4: 0.018} | **same**: 1 via `np.unique`; `np.ptp` = [0, 0, 0] |
| Scorable window | 231 months, 2007-04-30 → 2026-06-30 | **same** (contiguous, 0 gaps) |
| Latest row | 2026-08-31, missing `fred_m2sl`, `fred_totalsl`, `div_yield` | **same** (2026-07-31 missing `div_yield` only) |

The other plan-time facts matched too:
- dev features: 708 × 55, with 543 of 708 rows carrying a NaN;
- holdout features: 68 rows, 2021-01 → 2026-08;
- labels: 695 months, 6 states, 0 of 695 mismatched against the evaluated run's full-sample
  labeling;
- training targets after the embargo: 683;
- active columns: 55 of 55.

## Rulings recorded (08-SERVING.md §2)

- **q1-c.** Score the latest month in which every model column is observed, and state the as-of
  month and the missing columns on the page.
  - The **staleness cap** is part of the ruling: raise `ValueError` before any save when the scored
    month is more than 3 month-ends behind the newest `monthly_features` row.
  - "Behind the newest row" is the orchestrator's interpretation of Glenn's words "more than 3
    months old". It is recorded as that and flagged for his confirmation.
  - Today the gap is 2, so the report serves.
- **q2-ii.** Print the exact distinct-posterior count under the regime distribution: today,
  "1 distinct vector across 231 complete months". The recipe fix is a Phase 9 item.
- **The standing NO_REGISTRY ruling** (Glenn, 2026-09-28) is quoted verbatim in §0.
- Every ruling carries **registry rows spent: 0**, and each forbids imputation and any recipe change.

## Deviations from Plan

None in substance. There were two precision notes, both recorded in 08-SERVING.md rather than
absorbed:
1. **§1 item 7.** The plan said the evaluated backtest emits "the same triple at its final two
   steps". The value triple does repeat. It lands on the same state ids {0, 3, 4} at 2020-11-30,
   but on ids {0, 4, 5} at 2020-12-31, because each step re-labels history. The conclusion stands:
   the serving fit is faithful and the degeneracy belongs to the recipe.
2. **§2.1 staleness cap.** Measured by the wall clock from today's run date (September 2026 back to
   June 2026), the gap is exactly 3. That is why the data-relative interpretation matters, and it
   is flagged for Glenn.

## Honesty checks

- `total_trial_count()`: **44** before and after.
- `registry/trials.jsonl` sha256: `c957e8fd…2e73c088ad`, unchanged.
- `git status --porcelain -- data outputs registry`: empty.
- No `evaluate_nowcaster`, `save`, `save_model` or `append_trial` call was made. The measurement
  script lives in the session scratchpad, outside the repo.
- No file outside `.planning/` changed.

## Self-Check: PASSED

- FOUND: `.planning/phases/08-regime-persistence-stability/08-SERVING.md` (§0, §1, §2, and §3 as a heading)
- FOUND: commit `aed8fe8` (Task 1)
- FOUND: commit `b2683a2` (Task 3)
