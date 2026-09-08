---
id: 260908-qwe
slug: fix-hindsight-oracle-indexerror-phantom-
date: 2026-09-08
status: complete
---

# Summary

`python -m trading_crab_lib.platform.evaluation.report` now runs to completion
against the committed platform checkpoints. Two defects fixed.

## What was wrong

The crash (`IndexError: single positional indexer is out-of-bounds`) landed in
`tilt.py`, but the cause was in `report.py`.

`_smoothed_hindsight_perf` derived its regime-conditional stats from the
**full sample**, so IAU (ETF inception 2005) and USO (2006) drew Sharpe
entries and were allocated real weight at **1974** — 1.7% and 0.8% of the book
at the traced 1974-10-31 decision date, where the universe was
`{IAU: 0, SPY: 153, TLT: 153, USO: 0}` non-NaN observations. Those two
zero-history columns then (a) collapsed the row-wise `dropna()` to zero rows,
forcing `portfolio_vol` onto its per-asset fallback, and (b) made that
fallback index an empty series.

The walk-forward driver never hit this: it derives stats from
`dev_asset_returns.loc[train_index]`, so an asset with no history in the train
window simply gets no stats and no weight. That asymmetry is why the backtest
completed 588/588 steps while the oracle died immediately after.

This was not only a crash. `_smoothed_hindsight_perf` feeds the
smoothed-vs-filtered gap (design §5.4, Pitfall 1) — the headline honesty
metric — whose stated premise is that the ONLY difference from the
walk-forward leg is the label source. Full-sample *labels* are intended
hindsight. Full-sample *asset existence* was a second, undocumented leak, and
the phantom sleeve mechanically earned 0 (NaN return, skipped by `.sum()`),
silently distorting the gap the function exists to measure.

## Changes

- **`report.py`** — per-step universe restriction via a precomputed
  `first_valid_index()` per asset; the full-sample stats frame is filtered to
  assets that have started by `t`. Docstring now states explicitly which
  hindsight is intended (labels) and which is not (asset existence).
- **`tilt.py`** — new `_latest_asset_vol()` returns NaN instead of raising on
  an unestimable series. `portfolio_vol` imputes unestimable assets with the
  **max** estimable vol, preserving the documented over-estimate property
  (`scale = min(1, target/sigma)`, so under-estimating sigma OVER-levers, and
  pandas' skipna would have done exactly that). All-unestimable returns `0.0`,
  which `vol_target_scale` maps to all-cash. Returning NaN would have been
  actively unsafe — `min(1.0, x/nan)` is `1.0` in Python.
- **`tilt.py`** — comment on the row-wise `dropna()` sharp edge: one all-NaN
  column kills the diversification-aware branch for every caller passing a
  ragged universe, regardless of how much joint history the other assets share.

## Verification

- 6 new tests. Discrimination checked three ways: **no fixes** → IndexError;
  **guard only** → the two behavior tests fail; **both fixes** → all pass.
- Full suite **1360 passed** (was 1354). `ruff check` clean on touched files.
- End-to-end report run completes and writes all artifacts.

## Note on process

The first draft of the universe test used seeded RNG and passed vacuously
under a guard-only build: seed 7 gave SPY a full-sample Sharpe of **−0.61**,
which `_per_regime_tilt` clips to zero weight, zeroing both legs. It was
caught only by explicitly running the test against a partially-fixed build
rather than trusting that red-then-green implied it tested the right thing.
The fixture is now deterministic and carries its own guard assertion.

## Not addressed (carried forward)

- The **holdout fence** is still open: `write_monthly_features_split()` and
  `assert_dev_checkpoint_within_boundary()` have no caller outside their unit
  test, and `data/holdout/` does not exist. The backtest driver does split
  internally (`driver.py:343-344`), so the backtest itself is fenced, but the
  on-disk checkpoint still carries post-2020 rows.
- The report's headline numbers are not yet credible (Faber max drawdown
  −99.69%, crisis capture 2199) — expected of the tracer-bullet naive layers,
  but they need their own investigation before any of this is decision-grade.
