---
id: 260908-qwe
slug: fix-hindsight-oracle-indexerror-phantom-
date: 2026-09-08
status: planned
---

# Fix hindsight-oracle IndexError: phantom-asset allocation + unguarded EWMA fallback

`python -m trading_crab_lib.platform.evaluation.report` crashes with
`IndexError: single positional indexer is out-of-bounds` **after** the
walk-forward backtest completes 588/588 steps successfully.

## Evidence (reproduced against the committed checkpoints, traced — not inferred)

Crash chain:

```
report.py:626  run_full_backtest_evaluation
report.py:410  _smoothed_hindsight_perf
tilt.py:166    vol_targeted_tilt
tilt.py:67     portfolio_vol
tilt.py:68     lambda s: ewma_vol(s.dropna(), ...).iloc[-1]   <-- IndexError
```

First failure, instrumented:

| decision date | rows of history | non-NaN obs per asset | weights |
|---|---|---|---|
| 1974-10-31 | 154 | `IAU: 0, SPY: 153, TLT: 153, USO: 0` | `IAU 1.7%, SPY 7.3%, TLT 90.2%, USO 0.8%` |

`aligned = asset_returns[common].dropna()` yielded **0** rows.

## Root cause

Two defects, one of which is the actual cause.

### Defect 1 — correctness (`report.py:406`)

`_smoothed_hindsight_perf` computes
`smoothed_stats = returns_by_regime_stats(asset_returns, full_sample_states)`
over the **full sample**. IAU (ETF inception 2005) and USO (2006) therefore
receive regime-conditional Sharpe entries and are allocated real weight at
**1974** — assets that did not exist at the decision date.

The walk-forward driver does **not** have this bug: `driver.py:415` computes
stats from `dev_asset_returns.loc[train_index]`, so absent assets get no
stats and no weight. That asymmetry is exactly why the backtest completes
and the oracle crashes.

This matters beyond the crash. `_smoothed_hindsight_perf` feeds the
smoothed-vs-filtered gap (design §5.4, Pitfall 1) — the headline honesty
metric. Its own docstring states the only difference from the walk-forward
leg is the label source ("the SAME allocation math, no new formula").
Full-sample **regime labels** are deliberate hindsight and must be kept.
Full-sample **asset existence** is not: it is a second, undocumented leak,
and the phantom sleeve mechanically earns 0 (`asset_returns.loc[t, "IAU"]`
is NaN and `.sum()` skips NaN), so it silently distorts the reported gap.

### Defect 2 — robustness (`tilt.py:67-70`)

`portfolio_vol`'s per-asset fallback calls `.iloc[-1]` on the EWMA of a
possibly-empty series. `ewma_vol` uses `min_periods=2`, so a 1-observation
series yields NaN (fine) but a 0-observation series yields an **empty**
Series and `.iloc[-1]` raises.

## Fixes

**F1 (`report.py`)** — restrict the per-step tradable universe to assets
that have actually started by `t`, via a precomputed `first_valid_index()`
per asset. Keep the full-sample regime stats (intended hindsight); filter
that stats frame per step so `regime_tilt_weights` normalizes within the
genuinely available set. The function already passes `asset_returns.loc[:t]`
as the vol input — per-step causality on the returns frame was the author's
intent; the stats frame simply was not given the same treatment. Extend the
docstring to state precisely which hindsight is intended (labels) and which
is not (asset existence).

**F2 (`tilt.py`)** — guard the per-asset fallback. Semantics constraint: the
documented safety property is that the linear-sum fallback **over**-estimates
portfolio vol so it under-levers rather than over-levers
(`scale = min(1, target/sigma)`, so under-estimating sigma over-levers). A
NaN vol must therefore **not** be silently treated as zero by `.sum()`'s
skipna. Impute unestimable assets with the **max** of the estimable ones
(conservative, preserves the over-estimate property). If no asset is
estimable, return `0.0`, which `vol_target_scale` already maps to all-cash.
Do **not** return NaN: `min(1.0, target/nan)` evaluates to `1.0` in Python —
full position on zero information, the exact opposite of the intended
failure mode.

**F3 (comment only)** — `aligned = asset_returns[common].dropna()` at
`tilt.py:63` is row-wise, so a single zero-history column collapses the
frame to 0 rows and kills the diversification-aware EWMA branch for the
entire pre-2005 period even though SPY and TLT each have 153 observations.
F1 removes the phantom columns on the oracle path, but the sharp edge
remains for any caller passing a ragged universe. Document it.

## Tests

Project rule: every new/edited test must FAIL against pre-change code and
PASS after — verified by `git stash` round-trip. No tautological tests.

1. `portfolio_vol` with one all-NaN column returns a finite float
   (currently raises `IndexError`).
2. `portfolio_vol` conservatism: imputed result is **strictly greater** than
   the estimable-assets-only sum. Catches the skipna-treats-NaN-as-zero
   regression, which would silently over-lever.
3. `portfolio_vol` with all columns all-NaN returns `0.0` (all-cash), not NaN.
4. `_smoothed_hindsight_perf` gives **zero weight** to an asset with no
   history at the decision date — captured from the real allocation call.
5. `_smoothed_hindsight_perf` result is **identical** whether or not a
   not-yet-started asset is present as a column, when every decision date
   precedes that asset's inception.
6. A fixture guard asserting both synthetic assets have positive full-sample
   Sharpe. This is not ceremony: the first draft used seeded RNG and handed
   SPY a Sharpe of **−0.61**, which `_per_regime_tilt` clips to zero weight,
   zeroing BOTH legs — test 5 passed vacuously, for a reason unrelated to the
   universe restriction. Caught by explicitly checking that tests 4 and 5 fail
   under an F2-only build; the fixture is now deterministic.
7. End-to-end: `python -m trading_crab_lib.platform.evaluation.report` runs
   to completion against `data/checkpoints/platform/`.

Discrimination verified three ways — no fixes (IndexError), F2-only (tests 4
and 5 fail on behavior), both fixes (all pass).

## Gates

- Full pytest suite (1354 currently passing) — no regressions.
- `ruff check` on touched files.
- Commit to `claude/gsd-discuss-phase-6-8wne0z`. Do not push.
