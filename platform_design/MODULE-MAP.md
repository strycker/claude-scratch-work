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
| M2 | Features | `monthly_raw` → `monthly_features` (causal only) | `taxonomy.py` | `P2_features_taxonomy.ipynb` | taxonomy table; no centred smoothing in causal features |
| M3 | Baseline allocator + backtest | returns → vol-targeted weights → walk-forward curve vs SPY, 60/40, Faber | `assets/{returns,vol}.py`, `allocation/tilt.py`, `allocation/hysteresis.py::execute_rebalance`, `backtest/{driver,baselines,costs}.py`, `evaluation/{kpis,report}.py` | `P5_assets_allocation.ipynb`, `P6_backtest_evaluation.ipynb` | scoreboard table renders; costs applied |
| M4 | Weekly report (serving) | latest data + weights + holdings → markdown | `report/{weekly,serving,holdings}.py`, plus `report/scoreboard.py` (added by 08.2-03) | N4 = `P7_serving_report.ipynb` (08.2-03) | two runs on real data give an identical report; as-of date shown |
| MVP-1 | Usable product, no regime model | | | | Glenn trades from it (A-13: the page runs the no-regime core mix) |
| M5 | Regime labeler | features → states and profiles, aligned across refits | `labeling/{jump_model,diagnostics}.py` | `P3_regime_labeling.ipynb` | timeline vs recessions; occupancy; profiles make sense |
| M6 | Nowcaster + filter | causal features → P(regime now) → filtered belief | `prediction/{nowcaster,regime_filter,transition_matrix}.py` | `P4_nowcaster.ipynb`; N6 = `P8_filtered_belief.ipynb` (08.2-03) | distinct-output count above 1; truncation (no look-ahead) test. Today the count is 1, hence A-14 (view suspended) |
| M7 | Regime tilt + ablation | belief → tilted weights; scoreboard compares tilt with ablation | `allocation/tilt.py`, `backtest/driver.py` (tilt-off ablation leg) | N7 = `P9_does_regime_pay.ipynb` (08.2-03) | tilt beats ablation net of costs, or it stays advisory. E-06 says NO, so it is advisory (G-06) |
| MVP-2 | Regime-aware product | | | | |
| M8 | Honesty | registry budgets, DSR, holdout evaluation | `honesty/{cv,gap_lag,gating,holdout,registry,walkforward}.py`, `evaluation/deflated_sharpe.py` (now also holds `quality_tier`, `annualized_sharpe`, `QUALITY_TIER_RULE`) | none yet | DSR computed against the registry |
| M9+ | Tripwire, stability, second classifier, mixture of experts, tactics | as designed | `tripwire/monitor.py` (advisory on the page from 08.2-03); stability and classifier #2 were retired 2026-10-05, see below | per module | per module |

## Retired research code (2026-10-05, KISS: DECISIONS P-07, G-13)

The code parked in 08.2-02 and everything that existed only to serve it has been **deleted**, not parked.
It remains retrievable from git history, and from the 08.5 archive tag. The weekly product never used any of
it.

- **Parked modules:**
  - `platform/parked/classifier2.py`: classifier #2;
  - `platform/parked/joint_driver.py`: the two-classifier joint backtest;
  - `platform/parked/stability.py`: the subsample-stability suite. Its Hungarian matching
    (`scipy.optimize.linear_sum_assignment`) is the reference for a CR-02 rebuild.
- **Parked-only helpers:**
  - `allocation/joint_tilt.py`;
  - `evaluation/dependence.py`;
  - the whole `features/` package (`relative.py` and `invariants.py`, classifier #2's feature substrate);
  - the `labeling_2` config block.
- **Research scripts:** `run_joint_lift.py`, `run_subsample_stability.py`, `terminal_month_diagnostic.py`,
  `joint_lift_diagnostics.py` and `diagnose_s1_truncation.py`.
- **Their tests:** the classifier-#2, stability, joint-driver, joint-tilt, pooling-consumers, features,
  dependence and parked-boundary tests, plus the joint-lift and subsample record tests that imported them.

`evaluation/churn.py` and `evaluation/disagreement.py` stay, because active code uses them. Record tests that
read committed joint-lift outputs without importing retired code (e.g. the S-1 truncation guard in
`test_platform_nowcaster_recursion.py`) retire with the tracked outputs in 08.5.
