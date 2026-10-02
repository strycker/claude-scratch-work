# Platform module map (M0–M7, plus M8 and M9+)

Written in Phase 8.2 (08.2-02, decision D-07). It maps each module of the build order to its
interface, today's files, its notebook and its gate. The interfaces and gates come from
`REBUILD-FROM-SCRATCH-GUIDE.md` §3; the decisions behind them are indexed in
`platform_design/DECISIONS.md`. File paths are relative to `src/trading_crab_lib/platform/` unless
they start with `scripts/` or `notebooks/`.

## Modules

| # | Module | Interface (in → out) | Current files | Notebook | Gate |
|---|---|---|---|---|---|
| M0 | Skeleton | config, checkpoint manager, CLI, CI | `config.py`, `checkpoints.py`, `snapshots.py` | none | `tradingcrab --help` runs; CI green |
| M1 | Data spine | raw sources → `monthly_raw`, publication lags applied | `ingestion/{alfred,eodhd,macro_daily,macro_monthly,norgate,prices_daily,publication_lags,tiingo}.py`, `splice.py`, `transforms_monthly.py`, `scripts/build_platform_data.py` | `P1_data_spine.ipynb` | coverage plot; point-in-time test passes; splices continuous |
| M2 | Features | `monthly_raw` → `monthly_features` (causal only) | `taxonomy.py`, `features/invariants.py` | `P2_features_taxonomy.ipynb` | taxonomy table; no centred smoothing in causal features |
| M3 | Baseline allocator + backtest | returns → vol-targeted weights → walk-forward curve vs SPY, 60/40, Faber | `assets/{returns,vol}.py`, `allocation/tilt.py`, `allocation/hysteresis.py::execute_rebalance`, `backtest/{driver,baselines,costs}.py`, `evaluation/{kpis,report}.py` | `P5_assets_allocation.ipynb`, `P6_backtest_evaluation.ipynb` | scoreboard table renders; costs applied |
| M4 | Weekly report (serving) | latest data + weights + holdings → markdown | `report/{weekly,serving,holdings}.py`, plus `report/scoreboard.py` (added by 08.2-03) | N4 = `P7_serving_report.ipynb` (08.2-03) | two runs on real data give an identical report; as-of date shown |
| MVP-1 | Usable product, no regime model | | | | Glenn trades from it (A-13: the page runs the no-regime core mix) |
| M5 | Regime labeler | features → states and profiles, aligned across refits | `labeling/{jump_model,diagnostics}.py` | `P3_regime_labeling.ipynb` | timeline vs recessions; occupancy; profiles make sense |
| M6 | Nowcaster + filter | causal features → P(regime now) → filtered belief | `prediction/{nowcaster,regime_filter,transition_matrix}.py` | `P4_nowcaster.ipynb`; N6 = `P8_filtered_belief.ipynb` (08.2-03) | distinct-output count above 1; truncation (no look-ahead) test. Today the count is 1, hence A-14 (view suspended) |
| M7 | Regime tilt + ablation | belief → tilted weights; scoreboard compares tilt with ablation | `allocation/tilt.py`, `backtest/driver.py` (tilt-off ablation leg) | N7 = `P9_does_regime_pay.ipynb` (08.2-03) | tilt beats ablation net of costs, or it stays advisory. E-06 says NO, so it is advisory (G-06) |
| MVP-2 | Regime-aware product | | | | |
| M8 | Honesty | registry budgets, DSR, holdout evaluation | `honesty/{cv,gap_lag,gating,holdout,registry,walkforward}.py`, `evaluation/deflated_sharpe.py` (now also holds `quality_tier`, `annualized_sharpe`, `QUALITY_TIER_RULE`) | none yet | DSR computed against the registry |
| M9+ | Tripwire, stability, second classifier, mixture of experts, tactics | as designed | `tripwire/monitor.py` (advisory on the page from 08.2-03); stability and classifier #2 are parked, see below | per module | per module |

## Parked (08.2-02)

Moved with `git mv` (history kept, `git log --follow` works), no shims. Active `src` must not import
`platform/parked/`; see "Boundary".

| Parked file | What it was | Why parked | How to un-park |
|---|---|---|---|
| `parked/classifier2.py` (was `labeling/classifier2.py`) | Leadership / relative classifier #2, K = 5, λ = 16, pinned by rule | L1-04 DEFER: added no lift, independence unproven (criterion 5 breaks under a 1-month M2 lag); G-07 | `git mv` back to `labeling/`, repoint its importers |
| `parked/joint_driver.py` (was `backtest/joint_driver.py`) | Two-classifier joint backtest behind criterion 7 | E-02 / E-03: the joint lift is not met as a result; G-07 | `git mv` back to `backtest/`, repoint its importers |
| `parked/stability.py` (was `labeling/stability.py`) | Subsample-stability suite for criterion 3 | G-07 scope cut: the weekly product does not use it | `git mv` back to `labeling/`, repoint its importers |

To un-park any of them: `git mv` it back, edit the importers (listed in the table below), and drop
the module from `PARKED_MODULES` in `tests/unit/test_platform_parked_boundary.py`.

Importers that were repointed in 08.2-02: `scripts/run_joint_lift.py`, `scripts/run_subsample_stability.py`,
`scripts/terminal_month_diagnostic.py`, and the tests `labeling_classifier2`, `labeling_stability`,
`subsample_stability_record`, `backtest_joint_driver`, `joint_diagnostics_record`, `report_weekly`,
`terminal_month_diagnostic` and `features_relative` (import lines only, with one approved exception in
`joint_diagnostics_record`: the monkeypatch target of `test_the_boundary_is_exclusive` now names
`evaluation.deflated_sharpe`, because `quality_tier` moved there).

Reuse note: `parked/stability.py` holds the Hungarian matching (`scipy.optimize.linear_sum_assignment`)
that aligns states across refits. The regime rebuild may reuse it for CR-02.

## Parked-only helpers left in place

Each has a reason it did not move.

| File | Reason it stays |
|---|---|
| `allocation/joint_tilt.py` | Only the parked `joint_driver` uses it, but the record test `test_platform_pooling_consumers` imports it |
| `evaluation/churn.py` | Used by scripts only, but `test_platform_nowcaster_recursion` and `test_platform_joint_diagnostics_record` import it |
| `evaluation/dependence.py` | An orphan: only its own test imports it |
| `evaluation/disagreement.py` | Not parked-only: the active `evaluation/report.py` imports it |
| `features/relative.py` | `compute_invariant_ratios` is used by `features/invariants.py` |
| `scripts/{run_joint_lift,joint_lift_diagnostics,terminal_month_diagnostic,run_subsample_stability,diagnose_s1_truncation}.py` | Parked-only, but their `_SCRIPTS_DIR` and `sys.path` logic breaks if they move. Each still resolves: `--help` runs for the first four, and `diagnose_s1_truncation` imports `evaluation.churn` only |

## Boundary

`tests/unit/test_platform_parked_boundary.py` denies the weekly path any route to parked code:

- A fresh interpreter imports every module under `report/` and `tripwire/` (found by glob, so new
  modules are covered without an edit). It fails if `sys.modules` holds anything under
  `platform.parked`, or `allocation.joint_tilt`, `evaluation.churn` or `evaluation.dependence`.
  A positive control checks that `report.weekly` was loaded.
- An AST scan of every module outside `parked/` fails on any import of `platform.parked`, including
  one inside a function. The scanner checks itself against a function-level import.
- Each parked module must be found under `parked/` and absent from its old path.
- `quality_tier`, `annualized_sharpe` and `QUALITY_TIER_RULE` in `parked/joint_driver.py` must be the
  same objects as in `evaluation/deflated_sharpe.py`.
