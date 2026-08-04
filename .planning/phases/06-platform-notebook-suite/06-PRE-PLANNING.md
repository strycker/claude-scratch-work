# Phase 6 Pre-Planning — Platform Notebook Suite

**Status:** groundwork for `/gsd-discuss-phase 6`. Facts gathered from the codebase, plus
the open decisions a human needs to settle before planning. **Not a plan** — no task
breakdown, no waves. That comes from `/gsd-plan-phase 6` after discussion.

**Requirement:** NB-01 · **Depends on:** Phase 5 (closed 2026-08-04)

---

## 1. The scope surprise: there is no `platform/plotting/`

The phase is not "write six notebooks." It is **"build a plotting library, then write six
notebooks."**

Verified by grep:

```
src/trading_crab_lib/platform/plotting/     → does not exist
matplotlib|seaborn imports under platform/  → ZERO matches
```

48 platform modules, 6,977 lines, and not one line of visualization code. Every plot the
notebooks need must be written from scratch.

This matters for planning because **ADR #11** (`CLAUDE.md`) is explicit:

> Notebooks call functions from `plotting.py`; they do not define plotting logic inline.
> Reasons: reusability, testability, consistency, DRY. **If you need a new plot, add it to
> `plotting.py` first, then call it from the notebook.**

So the phase has two halves, and the library half is the larger one:

| Half | Work |
|---|---|
| **A. `platform/plotting/`** | New package, ~6 submodules mirroring L0–L4 + evaluation, plus a `core.py` for save/show handling and the regime palette. Needs tests (the legacy `test_plotting.py` pattern: `test_does_not_crash` + empty-input edge cases). |
| **B. Six notebooks** | Thin — load checkpoint, call plot function, assert-and-explain, human sign-off cell. |

**Reusable prior art:** `src/trading_crab_lib/plotting/` (9 submodules) has directly
analogous functions — `plot_regime_timeline`, `plot_transition_matrix`,
`plot_calibration_curve`, `plot_soft_probabilities`, `plot_feature_regime_overlay`,
`plot_cv_fold_accuracy`, `_save_or_show`, `_regime_color`, `CUSTOM_COLORS`, `REGIME_CMAP`.
These target the quarterly pipeline's data shapes, so they are a **pattern source, not a
drop-in** — an open decision below.

## 2. What the notebooks have to draw from

Everything already persisted by the platform. No new computation required.

### Checkpoints (`data/checkpoints/platform/`, via `get_platform_checkpoint_manager()`)

| Checkpoint | Written by | Feeds notebook |
|---|---|---|
| `monthly_raw`, `monthly_raw_daily`, `daily_raw`, `fred_daily_raw` | L0 ingestion | P1 |
| `monthly_features` | `transforms_monthly.build_monthly_spine` | P1, P2 |
| `regime_labels`, `regime_confidences`, `regime_profiles` | `labeling/diagnostics.label_regimes` | P3 |
| `nowcaster` (joblib model) | `prediction/nowcaster` | P4 |
| `returns_by_regime` | `assets/returns` | P5 |

### Report artifacts (`outputs/reports/platform/`)

`labeling_diagnostics.parquet`, `ewma_vol.parquet`, `returns_by_regime.parquet`,
`gap_lag_metrics.parquet`, `model_metrics_brier.parquet`,
`model_metrics_calibration.parquet`, `model_metrics_confusion.parquet`,
`backtest_equity_curve_strategy.parquet`, `backtest_kpi_table.parquet`,
`backtest_report.md`, `weekly_report.md`

### Config surface

`config/platform_settings.yaml`, 18 sections: `allocation`, `backtest`, `cv`, `data`,
`fred_daily`, `fred_monthly`, `fred_vintage`, `holdout`, `labeling`,
`macrotrends_monthly`, `multpl_monthly`, `paid_providers`, `registry`, `report`,
`splice`, `taxonomy`, `tripwire`, `universe`.

Lean feature set (L1/L2), **13 features**: `cape_shiller`, `credit_spread_baa_aaa`,
`curve_10y2y`, `curve_10y3m`, `div_yield`, `fred_vix`, `gold`, `oil`, `real_rate_level`,
`realized_vol_1m`, `realized_vol_3m`, `trailing_return_1m`, `trailing_return_3m`.
`K=5`, `λ=52.0`, `n_restarts=10`.

## 3. The six notebooks

Per the roadmap success criteria. P3 and P6 are load-bearing — they are where the
"regime layer doesn't pay rent yet" finding is either explained or contradicted.

| # | Notebook | Renders | Human confirms |
|---|---|---|---|
| P1 | `P1_data_spine` | Coverage timeline per series, splice join points, ALFRED vintage vs revised, NaN map | 1962+ histories real? Splice joins continuous? |
| P2 | `P2_features_taxonomy` | Lean-13 time series, fast/slow/agency grouping, causal-vs-centered overlay, correlation matrix | Features economically sane? Any look-ahead? |
| **P3** | **`P3_regime_labeling`** | **Regime timeline vs dated economic history**, sojourn distribution, transition matrix, soft confidences, label churn, per-regime feature profiles | **Do the 5 regimes correspond to real history?** |
| P4 | `P4_nowcaster` | Calibration curve, transition-window vs overall vs steady-state accuracy, detection lag, filtered-probability paths | Calibrated? Beats persistence? |
| P5 | `P5_assets_allocation` | Returns-by-regime, EWMA vol, tilt weights over time, hysteresis state path, turnover | Tilts defensible? Hysteresis stable? |
| **P6** | **`P6_backtest_evaluation`** | **Equity curves (strategy / ablation / SPY / 60-40 / Faber)**, drawdown, KPI table, Brier, confusion, sojourn-lag headline **with resolved-transition count** | **Does the strategy earn its complexity?** |

## 4. Facts that constrain the plan

| # | Constraint | Source |
|---|---|---|
| C1 | Plotting logic goes in the library, notebooks only call it | ADR #11 |
| C2 | No look-ahead: P2 must show causal vs centered as distinct, never mixed | ADR #1, P1 pitfall |
| C3 | Notebooks must not read 2021+ data | HON-01, holdout carve |
| C4 | Regime palette is 5-color `CUSTOM_COLORS`; keep it consistent | legacy convention |
| C5 | Tests must not require network; notebooks may, and are not part of `pytest` | `tests/conftest.py` convention |
| C6 | Running a notebook must not overwrite production checkpoints | P20/D5 precedent |
| C7 | Notebooks need real checkpoints, which need `FRED_API_KEY` + network | Phase 1 human-verification precedent |

**C7 is the significant one.** This sandbox cannot build real data (no egress to
Yahoo/macrotrends). Notebook *code* can be written and its imports//plot functions
unit-tested here, but "runs top-to-bottom against real checkpoints" is a **human
verification item**, exactly like Phase 1's and Phase 5's. Plan for that split.

## 5. Open decisions for `/gsd-discuss-phase 6`

These change the plan materially; I have not assumed answers.

| # | Decision | Options |
|---|---|---|
| **D1** | How does `platform/plotting/` relate to the legacy `plotting/`? | (a) fresh package, legacy as pattern reference only — clean, some duplication; (b) import and adapt legacy functions — less code, re-couples platform to the legacy lib and fights MIGRATION-PLAN P0's decoupling; (c) extract a shared `plotting/core.py` both use. **(a) looks right given P0 decoupling**, but it is your call. |
| **D2** | How much per-notebook? | (a) six focused notebooks; (b) fewer, larger; (c) six + a `P0_index` overview. |
| **D3** | How is the human sign-off recorded? | (a) markdown cell the operator edits with date/verdict; (b) a `sign_off()` helper writing to a YAML ledger; (c) checkbox list only. Affects whether library code is needed. |
| **D4** | Are notebooks executed in CI? | (a) no — human-run only (matches C7); (b) yes with synthetic fixtures via `nbmake`/`papermill`; (c) smoke-import only. Drives test strategy and CI cost. |
| **D5** | Does P3 need a curated economic-events dataset (recessions, inflation eras, credit events) to overlay? | If yes, that's a small data-sourcing task (NBER recession dates are free via FRED `USREC`) and belongs in the plan. |
| **D6** | Notebook naming/location | `notebooks/P1_*.ipynb` alongside the legacy 12, or `notebooks/platform/`? Latter is cleaner for the Phase 7 move. |

## 6. Suggested plan shape (input to the planner, not a decision)

Roughly three waves:

- **Wave 1 — plotting foundation.** `platform/plotting/core.py` (save/show, palette,
  regime coloring) + package scaffold + tests. Everything else depends on it.
- **Wave 2 — per-layer plot functions + notebooks**, parallelizable by layer:
  P1/P2 (data+features), P3/P4 (regime+nowcaster), P5/P6 (allocation+evaluation).
- **Wave 3 — human validation run.** Operator runs all six against real checkpoints and
  records sign-off. Blocking, like 05-07.

**Sequencing suggestion:** build **P3 first** within Wave 2. It carries the phase's real
question — if the 5 regimes don't map onto recognizable economic history, that finding
reshapes Phases 7 and 8, and it is much cheaper to learn now than after migration.

## 7. Ready-to-plan checklist

- [x] Phase 5 closed (`05-VERIFICATION.md` Phase Closure Record)
- [x] `gsd-tools audit-open` → all artifact types clear
- [x] `gsd-tools validate consistency` → passed; `validate health` → healthy
- [x] Full suite green on `main` (1157 passing)
- [x] Roadmap + requirements updated (NB-01 → Phase 6)
- [x] Data/artifact surface inventoried (§2)
- [x] Constraints identified (§4)
- [ ] **D1–D6 answered** → `/gsd-discuss-phase 6`
- [ ] `06-CONTEXT.md`, `06-PATTERNS.md`, `06-RESEARCH.md` produced
- [ ] `/gsd-plan-phase 6`
