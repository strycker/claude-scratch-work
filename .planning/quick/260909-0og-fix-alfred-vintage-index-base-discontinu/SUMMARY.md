---
id: 260909-0og
slug: fix-alfred-vintage-index-base-discontinu
date: 2026-09-09
status: complete
---

# Summary

Audit item **A4**. ALFRED point-in-time vintages of a *rebased index* are not
level-comparable across rebasings, and `align_with_fallback` spliced published
levels as if they were.

## Root cause

BLS rebased CPI from 1967=100 to 1982-84=100 in January 1988. Consecutive output
values come from consecutive *vintages*, so the value published Jan 1988 (345.9,
old base) and the one published Feb 1988 (115.9, new base) describe the same price
level on different bases. Two cliffs resulted, both the same ~2.99x factor:

| | | ratio |
|---|---|---|
| 1970-12 `39.60` → 1971-01 `119.03` | shift_series → vintage handoff | 3.0058 |
| 1988-01 `345.9` → 1988-02 `115.9` | the rebasing itself | 2.9845 |

`real_rate_level` inherited both, ranging **−209.49 … +74.51**, while being a
defining feature of labeler states 2 and 3 — **64.3% of occupancy**.

## Fix

Two joins, handled separately:

- **Within the vintage era** — apply the growth measured *inside* the current
  vintage, where the base cancels. Strictly point-in-time: only rows already
  filtered to `realtime_start <= as_of` are read, and a revision to an earlier
  period is correctly seen as growth inside the newer vintage.
- **At the shift_series handoff** — ratio-splice the vintage segment onto
  shift_series' base at a same-date join (`splice.ratio_splice`'s idiom). An
  earlier draft chained off the previous *output* instead; that let a stale
  fallback value distort every later value and broke the telescoping property
  below. Caught by an existing test.

**Bounding property:** with a single consistent base the ratios telescope, so
chained output equals the raw published level *exactly*. Rates (`fred_unrate`)
and headcounts (`fred_payems`) are provably untouched.

The series stays on shift_series' modern base, so the level remains interpretable.

## A3 guard, applied here

`_warn_on_level_discontinuity()` flags month-over-month steps >50% at the agency
alignment boundary. Verified against the real corrupted checkpoint: it flags
exactly `1971-01-31` and `1988-02-29`.

**It also produced a false positive, which changed the design.** A blanket ratio
guard flags `fred_gs10` in March 2020 (yield fell 42%) and would flag UNRATE in
April 2020 (4.4% → 14.7%, a genuine 3.3x move). Rates really do move like that.
So the guard applies only to `kind: index` series, with `kind: rate` declared in
config for UNRATE.

That distinction is a fact about the series, not a threshold fitted to the data —
which is the direct answer to Phase 6 D-11's objection that plausibility bounds
"get tuned to whatever the current data happens to look like".

## Verification

- 5 new ALFRED tests, incl. a synthetic reproduction of the real 1988 rebasing.
  `test_vintage_rebasing_does_not_create_a_level_discontinuity` and
  `test_growth_is_preserved_across_the_rebasing` both **fail** pre-fix.
  Two are deliberately *property* tests that pass in both states — they guard
  the telescoping no-op and the point-in-time guarantee against regression.
- 6 new guard tests, including one pinning the shipped config.
- Suite **1391 passed** (was 1382). ruff clean. No production data touched.

## Not verifiable here — action required

This container has no `FRED_API_KEY`, so the fix could not be run against live
ALFRED. **The committed `monthly_raw.parquet` still contains the corrupted
`fred_cpi`** — the fix is at ingestion, so it takes effect only on rebuild.

On the next `python scripts/build_platform_data.py`, expect the guard to stay
silent for `fred_cpi`. If it warns, the fix is incomplete and the warning names
the dates.
