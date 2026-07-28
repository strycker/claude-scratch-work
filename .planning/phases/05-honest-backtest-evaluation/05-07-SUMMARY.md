---
phase: 05-honest-backtest-evaluation
plan: 07
completed: 2026-07-27
type: checkpoint:human-verify
requirements: [EVAL-01, EVAL-02, EVAL-03, EVAL-04]
---

# Plan 05-07 Summary — Real 1972–2020 Backtest (human-verified)

**Task type:** `checkpoint:human-verify` (blocking-human) — the design §14 Phase 1
exit criterion. No source files changed (`files_modified: []`); this plan executes and
human-verifies the machinery built in Plans 01–06.

## What was run

`python -m trading_crab_lib.platform.evaluation.report` against the real Phase 1
monthly checkpoints (spliced 1962+ monthly spine built via
`scripts/build_platform_data.py`), on the operator's local machine with a live
`FRED_API_KEY`. The walk-forward executed **588 monthly steps, 1972-01 → 2020-12**,
end-to-end through L1→L4, and wrote the report + artifacts.

## Human-verified evidence (operator paste, 2026-07-27)

- **Headline (EVAL-03):** sojourn/detection-lag ratio is the FIRST metrics section —
  median sojourn 46.5 mo / median detection lag 107.0 mo → **ratio 0.435**.
- **Baseline gauntlet (EVAL-02):** SPY (log wealth 5.68, DD −48.95%), 60/40 (131.03,
  −2.27%), Faber 10-mo SMA (1.14, −99.69%), plus the **no-regime ablation** (113.89,
  −68.08%) — all present. Ablation delta vs. strategy: **−81.71 log wealth / −14.37% DD**.
- **Strategy KPIs (EVAL-01):** terminal log wealth 32.18; max drawdown −82.45%
  (215 months underwater); CVaR(5%) −0.208; mean monthly turnover 0.0153; in-sample
  crisis capture ratios for 1973-74 / 1980-82 / 2000-02 / 2008-09.
- **Smoothed-vs-filtered gap:** −3.372 (distinct smoothed vs. walk-forward series).
- **Model-metrics artifacts (EVAL-04):** `model_metrics_{brier,calibration,confusion}.parquet`
  written and round-trip; equity-curve (strategy + ablation) and KPI-table parquets written.
- **Holdout boundary:** strategy equity curve ends **2020-12-31** — no 2021+ row touched.
- **Trial registry:** gained exactly **2** rows this run — strategy (`use_regime_tilt: true`)
  and no-regime ablation (`use_regime_tilt: false`), both `n_steps: 588`, same git SHA.
- **D-04:** no `deflated`/`dsr` and no 2021+ dates anywhere in the report.
- **Gold exclusion:** macrotrends is IP-blocked (403), so `gold` was skipped via the
  optional-splice toggle and flagged **⚠ Excluded assets: `gold`** in the report's
  Conventions section; oil fell back to FRED `wti_fred`.

## Honest diagnostic finding (D-01)

Per D-01 this phase passes on "it ran and reported correctly," NOT on beating a
benchmark — and it correctly reports that **the naive tracer-bullet regime layer does
not pay rent yet**: the regime strategy (log wealth 32.18) underperformed its own
no-regime ablation (113.89) and 60/40 (131.03), and the sojourn/detection-lag ratio
(0.43) is far below the design's eventual ≥~5 bar. These become the standing targets
for the later L4-upgrade milestone (D-01a). This is the intended honest outcome, not a
failure of the phase.

## Known caveats / follow-ups (tracked, non-blocking for D-01)

- **Regime layer effectively active 1990-onward.** `fred_vix` (VIXCLS) begins 1990-01,
  and L1's any-NaN row-drop (`driver.py:_refit_l1`) collapses pre-1990 training, so the
  walk-forward holds the naive allocation 1972–~1990. This inflates the 107-month median
  detection lag. Follow-up: drop all-NaN-in-window feature columns so the regime model
  trains pre-1990 on the available features.
- **Gold excluded** for lack of a free long-history spot source (macrotrends 403; FRED
  delisted LBMA gold). Re-includes automatically when macrotrends is reachable
  (`optional: true` → set `false` to require it).

## Self-Check: PASSED

Real 1972–2020 walk-forward ran end-to-end, reported every required number, stopped at
the holdout boundary, logged 2 registry rows, and was human-verified — design §14
Phase 1 exit satisfied as a diagnostic (D-01).
