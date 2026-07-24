# Phase 5: Honest Backtest & Evaluation - Research

**Researched:** 2026-07-23
**Domain:** Walk-forward strategy backtesting, honesty-framework evaluation metrics, calibration diagnostics — Python/pandas/sklearn on the existing `platform/` codebase
**Confidence:** HIGH (all code paths read from source; design claims cited by section number)

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01: Phase 5 is a DIAGNOSTIC, not a gate.** The backtest must RUN end-to-end and
  REPORT every required number honestly; verification passes on "it ran and reported
  correctly," **not** on "it beat a benchmark." Rationale: design §14 Phase 1 explicitly
  says the tracer bullet "beats nothing yet — that's fine." Migration (Phase 6) proceeds
  regardless of whether the naive skeleton wins.
- **D-01a:** The Faber 10-month SMA comparison (log wealth AND max drawdown, per §23.1)
  and the sojourn/detection-lag ratio are surfaced **prominently** in the report as the
  **standing targets** for the later L4-upgrade milestone — recorded, highlighted, but
  not pass/fail for this phase.
- **D-02: Mandated 3 + no-regime ablation.** Baselines are SPY (buy-and-hold), 60/40,
  Faber 10-month SMA, **and** the same L3/L4 allocation pipeline with the regime tilt
  disabled (regime-agnostic: vol-target-only / flat-weight variant). Design §8.7: "no-
  regime versions of every regime-conditional model — the regime layer must pay rent."
  This is the one baseline that directly attributes any edge to the regime layer vs. plain
  vol-targeting. Cheap to build (same pipeline, tilt flag off).
- **D-03: Token config-driven transaction cost + turnover reported separately.** Apply a
  small per-rebalance haircut on traded notional — a config knob under a new
  `backtest:` / `evaluation:` section (default ~10 bps; document derivation) — AND report
  turnover as its own diagnostic. Not full tax/friction modeling (out of scope), but a
  monthly tilt must not look free. Hysteresis (Phase 4) already keeps turnover low, so the
  cost should be a light touch. Frictionless-only was rejected: it flatters higher-turnover
  configs and undercuts the honesty framing.
- **D-04: Defer deflated Sharpe and the 2021+ holdout evaluation entirely.** Phase 5
  computes neither. Both are design Phase 6 (freeze) / a future milestone. Keeps holdout
  discipline intact — the point of the honesty framework.

### Claude's Discretion

- Backtest module layout inside `src/trading_crab_lib/platform/` (e.g. `platform/backtest/`
  and/or `platform/evaluation/` — planner's call), following the established
  `compute_* / report_* / __main__ self-check / parquet artifact` module shape.
- Exact min_train warmup that yields the 1972 first-decision start from 1962+ monthly data
  (~120 months) — set so the first backtested rebalance lands in 1972, via the frozen
  `expanding_steps(min_train=…)` interface.
- Which in-sample crisis windows to report capture ratios for (candidates: 1973–74,
  1980–82, 2000–02, 2008–09; NOT 2020/2022 — those are in the holdout). Config-driven.
- 60/40 rebalance convention (monthly vs annual reconstitution) and the exact "no-regime
  ablation" construction (equal-weight vs vol-target-only) — documented in the report.
- EWMA/turnover/CVaR computation conventions, and report layout (markdown + plots).
- Whether the backtest strategy leg reuses the exact Phase 4 allocation entry point or a
  thin backtest-driver wrapper around it (prefer reuse; no forking of allocation logic).

### Deferred Ideas (OUT OF SCOPE)

- Deflated Sharpe ratio vs. the full trial registry — design Phase 6 (freeze); a future
  milestone (D-04).
- Single 2021+ holdout evaluation — design Phase 6; only after design freeze (D-04).
- Faber/SPY-beating as a hard gate — deferred to the L4-upgrade milestone (design Phase 5);
  recorded here as a standing target only (D-01a).
- Diebold–Mariano / Mincer–Zarnowitz forecast-KPI depth (§8.8 full) — beyond EVAL-04's
  Brier/calibration/confusion; revisit when the MoE/nowcaster upgrades land (v2).
- BL/HRP/Kelly/stops/crash-dashboard allocation upgrades — v2 (L4-V2-*); the backtest here
  evaluates only the naive allocation.

</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| EVAL-01 | Honest walk-forward backtest 1972–2020 runs end-to-end through all layers (design §14 Phase 1 exit) | §"New Backtest Driver Architecture" — `expanding_steps(min_train=120)` reused as the loop spine; a NEW per-step orchestration function drives L1→L2→L3→L4 (does not reuse `run_walkforward()`'s single-sklearn-model body, which is a different shape). See Pattern 1, Code Example 1. |
| EVAL-02 | Baseline gauntlet in the backtest report: buy-and-hold SPY, 60/40, Faber 10-month SMA (design §8.7, §23.1) + no-regime ablation (D-02) | §"Baseline Gauntlet Construction" — all 4 baselines built from the same spliced core research series (`splice.build_core_research_series`), same window, same cost convention. See Pattern 2. |
| EVAL-03 | Sojourn/detection-lag ratio reported prominently — the go/no-go number for regime timing (design §5.4) | §"Sojourn/Detection-Lag Ratio (EVAL-03)" — `occupancy_and_sojourns()` (L1) supplies median sojourn; `gap_lag.compute_detection_lag()` (already built, Phase 2) supplies lag; `gap_lag.sojourn_lag_ratio()` combines them. No new compute function needed — orchestration only. |
| EVAL-04 | Model metrics artifacts (multiclass Brier, calibration bins, confusion tables) persisted per run (salvaged `model_metrics_artifacts.py`; design §8.8) | §"Model Metrics Artifacts (EVAL-04)" — extract the three pure helper functions from the salvaged file (`_compute_brier_multiclass`, `_calibration_bins`, `_confusion_tidy`); do NOT import the salvage file or its `FoldReport`/`classifier.py` dependency (D-01 no incumbent imports); rewrite an orchestrator shaped like `gap_lag.py`. |

</phase_requirements>

## Summary

Phase 5 is an **orchestration and reporting** phase, not a new-algorithm phase — every
mathematical primitive it needs (expanding-window walk-forward, purged/embargoed CV,
smoothed-vs-filtered gap, detection lag, sojourn/lag ratio, EWMA vol, vol-targeted tilt,
hysteresis, returns-by-regime, trial registry) already exists and is frozen in
`src/trading_crab_lib/platform/`. The work is: (1) a **new backtest driver** that steps
through `expanding_steps(monthly_index, min_train=120)` from 1972 to 2020 and at each step
refits L1 (jump-model labeler) and L2 (nowcaster) on data ≤ t, computes L3 (returns-by-regime
+ EWMA vol) and L4 (vol-targeted regime tilt) target weights, and accrues a cost-adjusted
equity curve; (2) a **baseline gauntlet** built from the same spliced core research series
(SPY buy-hold ≈ `equities_tr`, 60/40 blend, Faber 10-month SMA switch, and a no-regime
ablation that is the *same* L3/L4 code path fed a degenerate single-state label series); (3)
an **evaluation report** surfacing the sojourn/detection-lag ratio first (the headline
number per D-01a/§5.4), followed by strategy KPIs (terminal log wealth, max drawdown +
duration, CVaR(5%), turnover, in-sample crisis capture ratios) and model-metrics artifacts
(Brier, calibration, confusion — adapted from the salvaged file's pure helper functions).

Two important architecture findings from reading the frozen interfaces directly:

1. **`run_walkforward()` cannot be reused verbatim for EVAL-01.** It is shaped around a
   single sklearn-style `.fit()/.predict()` estimator predicting one target per step. The
   Phase 5 backtest drives four cooperating layers (L1 non-causal batch fit, L2 supervised
   fit, L3 descriptive stats, L4 deterministic allocation math) per step, not one estimator.
   The correct reuse is the **pure generator `expanding_steps()`** directly, with a new
   backtest-specific loop body that still ends by calling `registry.append_trial()` exactly
   once per full run (mirroring, not calling, `run_walkforward`'s convention).
2. **The `evaluate_nowcaster()` function built in Phase 3 explicitly does NOT walk-forward
   refit** — its own docstring states "v1 evaluates against the full embargoed set rather
   than re-fitting at every step of an expanding walk-forward window (impractical here...);
   a full walk-forward refit loop is future work, not required by this phase's must_haves."
   **Phase 5 IS that future work.** The backtest driver must call `fit_jump_model` +
   `build_nowcaster_training_set` + `fit_nowcaster` fresh inside its own loop, using only
   `features_df.loc[train_index]` at each step — it must not reuse the single
   already-fitted `nowcaster` checkpoint produced by Phase 3/4's one-shot fit.

**Primary recommendation:** build a new `platform/backtest/driver.py` housing
`run_backtest(monthly_features, core_prices, cfg, *, use_regime_tilt=True) -> pd.DataFrame`
that loops `expanding_steps(monthly_index, min_train=120)`, refits L1+L2 every step (design
§8.1: "O(TK²) HMM/jump fits are minutes over 60 years — no computational excuse"), and reuses
Phase 4's `vol_targeted_tilt`/`regime_tilt_weights`/`returns_by_regime_stats` unchanged. The
no-regime ablation is the *same* function called with a constant-label series (not a forked
code path) — this also gives the invariant test named in CONTEXT's code_context section for
free: "a no-regime ablation with tilt-off must reproduce the vol-target baseline exactly."

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Walk-forward loop / expanding window | Backend (batch/library) | — | Pure function over a DatetimeIndex; `platform/honesty/walkforward.py::expanding_steps` (frozen). |
| L1 regime refit per step | Backend (batch/library) | — | `platform/labeling/jump_model.py` — batch (non-causal within-window) fit on `data ≤ t` only. |
| L2 nowcaster refit per step | Backend (batch/library) | — | `platform/prediction/nowcaster.py` — supervised, causal-feature-gated. |
| L3 returns-by-regime + vol | Backend (batch/library) | — | `platform/assets/returns.py`, `platform/assets/vol.py` — descriptive stats over the train window. |
| L4 allocation (tilt + hysteresis) | Backend (batch/library) | — | `platform/allocation/tilt.py`, `hysteresis.py` — deterministic math, no I/O. |
| Baseline construction (SPY/60-40/Faber/ablation) | Backend (batch/library) | — | New `platform/backtest/baselines.py`; consumes `platform/splice.py` output. |
| Transaction cost / turnover accounting | Backend (batch/library) | — | New `platform/backtest/costs.py`; pure arithmetic on weight vectors. |
| Strategy KPIs (log wealth, DD, CVaR, capture) | Backend (batch/library) | — | New `platform/evaluation/kpis.py`; pure functions over an equity-curve Series. |
| Sojourn/detection-lag ratio | Backend (batch/library) | — | Orchestration only — reuses `labeling/diagnostics.py::occupancy_and_sojourns` + `honesty/gap_lag.py` unchanged. |
| Model metrics (Brier/calibration/confusion) | Backend (batch/library) | — | New `platform/evaluation/model_metrics.py`; adapted pure helpers, no I/O in the compute layer. |
| Report assembly (markdown + parquet artifacts) | Backend (batch/library) | Report/CLI | New `platform/evaluation/report.py`, mirrors `report/weekly.py` / `honesty/gap_lag.py::report_*` shape. |
| Persistence (registry trial, checkpoints, artifacts) | Backend (batch/library) | — | `honesty/registry.py::append_trial` (one row per full backtest run); `platform/checkpoints.py`. |

There is no browser/frontend/API tier in this project — it is a batch CLI/library pipeline
(mirrors the existing `platform/report/weekly.py` `__main__` pattern). No capability in this
phase belongs anywhere but the library/backend tier.

## Standard Stack

No new external packages are required. Phase 5 is pure orchestration over already-installed
dependencies:

| Library | Version (installed) | Purpose | Why Standard |
|---------|------|---------|--------------|
| pandas | ≥2.0 (project pin) | Equity-curve Series/DataFrame, monthly index arithmetic | Already the project's sole DataFrame layer |
| numpy | ≥1.25 (project pin) | Log-wealth, CVaR quantile, drawdown vectorized math | Already used throughout `platform/` |
| scikit-learn | ≥1.4 (project pin) | `KMeans` (jump model warm start), `LogisticRegression`/`CalibratedClassifierCV` (nowcaster) — reused unchanged inside the new loop | Already the L1/L2 dependency (Phase 3) |
| pyarrow | ≥14.0 (project pin) | Parquet artifact I/O for the new KPI/metrics/baseline artifacts | Matches every existing `report_*`/`compute_*` module |
| matplotlib (optional, report plots) | ≥3.8 (project pin) | Equity-curve / drawdown / calibration plots if the planner chooses "markdown + plots" (CONTEXT discretion) | Already a project dependency; not required for the markdown+parquet artifacts alone |

**Installation:** none — no `pip install` step needed for this phase.

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Hand-rolled equity-curve compounding | `empyrical`/`quantstats`/`pyfolio` for KPIs (Sharpe, max DD, CVaR) | Rejected: adds a new external dependency for ~30 lines of vectorized pandas the project already writes by hand elsewhere (see `assets/returns.py::returns_by_regime_stats`'s own max-drawdown one-liner); CLAUDE.md functions-only-lib convention favors small, auditable, in-repo math over a backtest-framework dependency for a diagnostic-only phase. |
| Custom walk-forward loop from scratch | `run_walkforward()` unmodified | Rejected as a direct reuse (see Summary finding 1) — but its inner generator `expanding_steps()` IS reused; only the per-step body differs. |

## Package Legitimacy Audit

Not applicable — this phase installs no new external packages. All functionality is built
from the project's existing pinned dependencies (pandas, numpy, scikit-learn, pyarrow,
matplotlib), already verified and in use by Phases 1–4.

## Architecture Patterns

### System Architecture Diagram

```
                    monthly_features (Phase 1 checkpoint, 1962+)
                    core research prices (splice.build_core_research_series)
                              │
                              ▼
              ┌───────────────────────────────────┐
              │  expanding_steps(index, min_train=120)  │  <- REUSED, frozen (walkforward.py)
              │  yields (t, train_index, test_index)     │
              │  for t = 1972-01 .. 2020-12 (588 steps)  │
              └───────────────────────────────────┘
                              │  per step t
                              ▼
        ┌──────────────────────────────────────────────────────┐
        │  L1 refit (data ≤ t only)                             │
        │  standardize_features -> fit_jump_model -> canonicalize│
        └──────────────────────────────────────────────────────┘
                              │  labels (train-window only)
                              ▼
        ┌──────────────────────────────────────────────────────┐
        │  L2 refit (data ≤ t only)                             │
        │  build_nowcaster_training_set (embargo=12) -> fit_nowcaster│
        │  predict_proba(features at t)  -> filtered regime_probs│
        └──────────────────────────────────────────────────────┘
                              │  regime_probs (t)
                              ▼
        ┌──────────────────────────────────────────────────────┐
        │  L3 (train-window descriptive stats)                  │
        │  compute_monthly_returns -> returns_by_regime_stats    │
        │  ewma_vol (feeds portfolio_vol inside tilt)             │
        └──────────────────────────────────────────────────────┘
                              │  returns_by_regime, asset_returns
                              ▼
        ┌──────────────────────────────────────────────────────┐
        │  L4 (deterministic, no refit)                          │
        │  update_active_regime (hysteresis) -> vol_targeted_tilt│
        └──────────────────────────────────────────────────────┘
                              │  target_weights(t)
                              ▼
        ┌──────────────────────────────────────────────────────┐
        │  Backtest driver step-close (NEW code)                 │
        │  turnover = |Δweights|.sum(); cost = turnover * bps    │
        │  realized_return(t+1) = Σ w_i * asset_return_i - cost   │
        │  append to equity-curve record; prev_weights = weights │
        └──────────────────────────────────────────────────────┘
                              │  (loop back to next step)
                              ▼
              one equity-curve DataFrame per run
                              │
         ┌────────────────────┼─────────────────────┐
         ▼                    ▼                      ▼
   Baseline gauntlet    Sojourn/lag ratio      Model-metrics artifacts
   (SPY, 60/40, Faber,   (occupancy_and_sojourns  (Brier/calibration/
    no-regime ablation)   + gap_lag.*)             confusion, from the
                                                    collected per-step
                                                    (y_true, proba) pairs)
         │                    │                      │
         └────────────────────┼──────────────────────┘
                              ▼
              Evaluation report (markdown + parquet artifacts)
              headline: sojourn/lag ratio, then Faber comparison,
              then no-regime-ablation delta, then full KPI table
                              │
                              ▼
              registry.append_trial()  (ONE row per full backtest run)
```

### Recommended Project Structure

```
src/trading_crab_lib/platform/
├── backtest/
│   ├── __init__.py
│   ├── driver.py        # run_backtest() — the EVAL-01 walk-forward loop (new)
│   ├── baselines.py      # spy_buy_hold(), sixty_forty(), faber_sma(), no_regime_ablation() (EVAL-02, new)
│   └── costs.py          # compute_turnover(), apply_transaction_cost() (D-03, new)
└── evaluation/
    ├── __init__.py
    ├── kpis.py            # terminal_log_wealth(), max_drawdown(), cvar(), crisis_capture_ratio() (new)
    ├── sojourn_lag.py     # orchestrates occupancy_and_sojourns() + gap_lag.* for EVAL-03 (new, thin)
    ├── model_metrics.py   # brier/calibration/confusion adapted from salvage (EVAL-04, new)
    └── report.py          # assemble_backtest_report() + write artifacts (new, mirrors report/weekly.py)
```

Both new top-level packages live under `platform/`, consistent with the module shape convention
(`compute_* / report_* / __main__ self-check / parquet artifact`) established by
`honesty/gap_lag.py` and reused by every Phase 1–4 module.

### Pattern 1: Backtest driver reuses `expanding_steps()`, not `run_walkforward()`

**What:** The loop spine is the pure generator, not the sklearn-model-shaped wrapper.
**When to use:** Any multi-layer (non-single-estimator) walk-forward evaluation.
**Example:**

```python
# Source: adapted directly from the read source of
# src/trading_crab_lib/platform/honesty/walkforward.py (frozen interface)
from trading_crab_lib.platform.honesty.walkforward import expanding_steps
from trading_crab_lib.platform.honesty import registry

def run_backtest(monthly_features, core_returns, cfg, *, min_train=120,
                  use_regime_tilt=True, registry_path=None):
    records = []
    prev_weights = pd.Series(dtype=float)
    prev_active_regime = None
    y_true_log, proba_log, classes_log = [], [], []  # feeds EVAL-04

    for t, train_index, test_index in expanding_steps(monthly_features.index, min_train=min_train):
        train_features = monthly_features.loc[train_index]
        # --- L1 refit on data <= t only ---
        labels = _refit_l1(train_features, cfg)          # jump_model.fit_jump_model + canonicalize
        # --- L2 refit on data <= t only ---
        regime_probs, nowcaster = _refit_l2(train_features, labels, cfg)  # nowcaster.build_*/fit_nowcaster
        # --- L3 descriptive stats over train window ---
        returns = core_returns.loc[train_index]
        states_for_stats = labels if use_regime_tilt else pd.Series(0, index=labels.index)
        stats = returns_by_regime_stats(returns, states_for_stats)
        # --- L4 allocation (deterministic) ---
        active = update_active_regime(regime_probs, prev_active_regime, **cfg["allocation"]["hysteresis"])
        tilt = vol_targeted_tilt(regime_probs if use_regime_tilt else pd.Series({0: 1.0}),
                                  stats, returns, target_vol_annual=cfg["allocation"]["target_vol_annual"],
                                  halflife=cfg["allocation"]["ewma_halflife_months"])
        # --- cost + realized return (step-close) ---
        turnover = _turnover(prev_weights, tilt["weights"])
        cost = turnover * cfg["backtest"]["cost_bps"] / 1e4
        realized = _realized_return(tilt["weights"], tilt["cash"], core_returns.loc[test_index[0]]) - cost
        records.append({"date": t, "return": realized, "turnover": turnover, "cost": cost,
                         "active_regime": active, "scale": tilt["scale"]})
        prev_weights, prev_active_regime = tilt["weights"], active
        y_true_log.append(labels.iloc[-1]); proba_log.append(regime_probs.values); classes_log.append(regime_probs.index)

    equity_curve = pd.DataFrame(records).set_index("date")
    registry.append_trial(
        config={"phase": "05-backtest", "use_regime_tilt": use_regime_tilt, "min_train": min_train,
                "cost_bps": cfg["backtest"]["cost_bps"]},
        features=list(monthly_features.columns),
        metrics={"n_steps": len(equity_curve), "terminal_log_wealth": float(np.log1p(equity_curve["return"]).sum())},
        path=registry_path,
    )
    return equity_curve, {"y_true": y_true_log, "proba": proba_log, "classes": classes_log}
```

### Pattern 2: Baseline gauntlet from the same spliced series

**What:** All 4 baselines (SPY, 60/40, Faber, ablation) share one price source so
comparisons are apples-to-apples over 1972–2020.
**When to use:** EVAL-02.
**Example:**

```python
# Source: derived directly from the read source of
# src/trading_crab_lib/platform/splice.py::build_core_research_series (design §23.1 for Faber)
research = build_core_research_series(monthly_raw, cfg)   # columns: equities_tr, long_duration_tr, gold, oil, cash
equity_ret = research["equities_tr"].pct_change()
bond_ret   = research["long_duration_tr"].pct_change()
cash_ret   = research["cash"].pct_change()   # yield_as_return method — already a monthly return series

def spy_buy_hold(equity_ret):
    return equity_ret  # single purchase, cost-free by construction

def sixty_forty(equity_ret, bond_ret, *, rebalance="monthly"):
    # monthly reconstitution to fixed 60/40 -> turnover = |0.6 - drifted_60|+|0.4 - drifted_40| each month
    ...

def faber_sma(equity_level, cash_ret, *, window=10):
    sma = equity_level.rolling(window).mean()
    signal = (equity_level > sma).shift(1)  # decide with data through t, act at t+1 (no look-ahead)
    return signal.map({True: "equities_tr", False: "cash"})

def no_regime_ablation(monthly_features, core_returns, cfg):
    # SAME driver, SAME L4 code, constant-label L1 output — not a forked implementation.
    return run_backtest(monthly_features, core_returns, cfg, use_regime_tilt=False)
```

### Pattern 3: Sojourn/lag ratio is orchestration, not new math

**What:** EVAL-03's headline number combines two ALREADY-BUILT compute functions from two
different layers.
**When to use:** EVAL-03 report assembly.
**Example:**

```python
# Source: read directly from src/trading_crab_lib/platform/labeling/diagnostics.py
#         and src/trading_crab_lib/platform/honesty/gap_lag.py (both frozen)
from trading_crab_lib.platform.labeling.diagnostics import occupancy_and_sojourns
from trading_crab_lib.platform.honesty.gap_lag import compute_detection_lag, sojourn_lag_ratio

def compute_sojourn_lag_headline(full_sample_states, filtered_probs_series, act_threshold=0.70):
    sojourn_stats = occupancy_and_sojourns(full_sample_states)
    median_sojourn = sojourn_stats["overall_median_sojourn_months"]

    transitions = [i for i in range(1, len(full_sample_states))
                   if full_sample_states[i] != full_sample_states[i - 1]]
    lag_result = compute_detection_lag(transitions, filtered_probs_series, threshold=act_threshold)

    ratio = sojourn_lag_ratio(median_sojourn, lag_result["median"])
    return {"median_sojourn": median_sojourn, "median_lag": lag_result["median"], "ratio": ratio}
```

Note: `full_sample_states` for the *headline* diagnostic should be the FULL 1962–2020
smoothed labeling (a single non-walk-forward L1 fit over the whole span, exactly analogous to
how `gap_lag.py`'s docstring frames "smoothed" vs "filtered" — smoothed = full-sample
hindsight fit; filtered = the walk-forward driver's per-step `regime_probs`). This is
distinct from, and complementary to, the per-step walk-forward refits inside
`run_backtest()` — do not conflate "the smoothed reference labeling used for the gap/lag
diagnostic" with "the walk-forward-refit labels used to size the actual strategy" (see
Pitfall 1 below).

### Anti-Patterns to Avoid

- **Reusing the single Phase 3/4 `nowcaster` checkpoint for the whole backtest:** that model
  was fit once on the full embargoed history — using it to generate "regime_probs at t" for
  early 1970s months is look-ahead leakage (the model has seen 2020 data). The backtest MUST
  refit inside its own loop on `train_index`-only data (see Summary finding 2).
- **Computing the no-regime ablation as a hand-written separate equal-weight function:**
  CONTEXT's code_context explicitly frames it as "cheap to build (same pipeline, tilt flag
  off)" and names the invariant test "a no-regime ablation with tilt-off must reproduce the
  vol-target baseline exactly" — this only holds if the ablation calls the SAME
  `vol_targeted_tilt`/`regime_tilt_weights` functions with a degenerate single-state label
  series, not a hand-rolled parallel implementation.
- **Applying transaction costs asymmetrically without documenting it:** D-03 scopes cost
  realism to "the strategy leg." Applying costs to SPY buy-and-hold (near-zero turnover) but
  NOT to the 60/40 and Faber baselines (which DO rebalance/switch) would flatter the strategy
  unfairly in the Faber/§23.1 comparison — document the convention explicitly (apply the same
  bps convention to every baseline that rebalances) so the "beats it on log wealth AND max
  drawdown" comparison (recorded, not gating, per D-01a) is apples-to-apples.
- **Silently truncating the backtest window at the config's `end_date` (`null` → today) instead
  of the holdout cutoff:** `monthly_features` from Phase 1 spans past 2020 by construction
  (the checkpoint has no default upper bound). The backtest driver must explicitly slice to
  `≤ 2020-12` — either via `holdout.split_by_holdout_boundary()` or an equivalent explicit
  filter — before ever constructing `expanding_steps()`. Do not rely on the walk-forward loop
  "just happening" to stop at 2020 if `monthly_features` continues into 2021+.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Expanding-window walk-forward loop | A new date-stepping loop | `honesty/walkforward.py::expanding_steps()` | Frozen interface (Phase 2); already leakage-tested (`train_index[-1] < test_index[0]` invariant). |
| Purged/embargoed CV inside the L2 refit | A new CV splitter | `honesty/cv.py::PurgedEmbargoedKFold` (already used inside `fit_nowcaster`) | No change needed — the backtest driver calls `fit_nowcaster()` unmodified per step; it already applies purge/embargo internally. |
| Smoothed-vs-filtered gap / detection lag | New gap/lag math | `honesty/gap_lag.py::compute_gap`, `compute_detection_lag`, `sojourn_lag_ratio` | HON-05, already built and tested in Phase 2 against exactly this need. |
| Median sojourn per state | A new run-length scanner | `labeling/diagnostics.py::occupancy_and_sojourns()` | Already computes `overall_median_sojourn_months` — the exact EVAL-03 input. |
| Vol-targeted position sizing | New sizing math | `allocation/tilt.py::vol_targeted_tilt`, `vol_target_scale`, `portfolio_vol` | L4-01, frozen; reused unchanged inside the backtest loop and for the no-regime ablation. |
| EWMA volatility | A new decay implementation | `assets/vol.py::ewma_vol` | "Single RiskMetrics-style EWMA... do not duplicate the decay math elsewhere" (module docstring, explicit). |
| Regime-conditional asset stats | A new groupby | `assets/returns.py::returns_by_regime_stats` | Already NULL-tolerant, already produces `sharpe_annualized`/`max_drawdown`/`n_obs` per (regime, asset). |
| Multiclass Brier / calibration bins / confusion counts | New metric functions from scratch | Adapt the 3 PURE helper functions from `ideas/gsd-salvage/prediction/model_metrics_artifacts.py` (`_compute_brier_multiclass`, `_calibration_bins`, `_confusion_tidy`) | Already correct, already tested logic (multiclass Brier = mean squared error vs one-hot; 5-bin calibration table; tidy confusion counts) — see EVAL-04 section for the exact adaptation boundary. |
| Trial registry logging | A new ledger | `honesty/registry.py::append_trial` | HON-02, append-only, git-tracked; the backtest driver calls it exactly once per full run, mirroring (not calling) `run_walkforward`'s pattern. |
| Treasury total-return synthesis / equity total-return / core research series | New splicing code | `platform/splice.py::build_core_research_series` | Already builds the exact 5 series (`equities_tr`, `long_duration_tr`, `gold`, `oil`, `cash`) the backtest and every baseline need, spliced back to 1962. |

**Key insight:** Phase 5 has essentially zero new numerical algorithms to invent. Every
piece of math it needs was deliberately built in Phases 1–4 with this exact phase's
consumption in mind (see each module's docstring — several explicitly say "Phase 5 plugs
in unchanged" or equivalent). The engineering risk is entirely in **orchestration
correctness** (walk-forward discipline, cost accounting, ablation construction, holdout
boundary), not in new formulas.

## Common Pitfalls

### Pitfall 1: Conflating the "smoothed reference" labeling with the "walk-forward" labeling

**What goes wrong:** Using the walk-forward driver's LAST step's labels (fit on data ≤
2020-12, i.e. effectively the full sample) as if they were a genuinely independent
"smoothed" reference series for the gap/lag diagnostic, when they were actually produced
inside the same loop that also drives strategy sizing.
**Why it happens:** `gap_lag.py`'s docstring already draws this distinction (the module's
`compute_gap(smoothed_perf, filtered_perf)` takes two performance numbers computed under two
different labelings — full-sample hindsight vs. real-time), but nothing in the existing code
forces you to build genuinely separate smoothed/filtered series; it's easy to reuse one
series for both roles by accident.
**How to avoid:** Build the "smoothed" reference explicitly as ONE full-sample
(non-walk-forward) call to `fit_jump_model`/`canonicalize_states` over the ENTIRE
1962–2020 span (this IS the intentional non-causal batch behavior the labeler's own
docstring documents — "the labeler is intentionally non-causal at the batch level"). Build
the "filtered" series as the walk-forward driver's per-step `regime_probs` collected across
all 588 steps. `compute_gap` then compares a strategy-performance metric computed under each.
**Warning signs:** If `smoothed_perf == filtered_perf` (or the gap is suspiciously near
zero) after implementation, check whether the two series were actually built independently.

### Pitfall 2: L2 refit starvation in the earliest walk-forward steps

**What goes wrong:** `build_nowcaster_training_set(embargo_months=12)` drops the trailing 12
months of labels from the (already only ~120-month) 1972 train window, leaving ~108 labeled
months. `fit_nowcaster`'s default `PurgedEmbargoedKFold(n_splits=5, ...)` then needs enough
per-class rows per fold across K=5 regime classes — with 108/5 ≈ 21 rows per fold split
across up to 5 classes, some early folds may have zero or one example of a rare regime,
which can make `CalibratedClassifierCV(method="sigmoid")`'s underlying `LogisticRegression`
fit degenerate or raise.
**Why it happens:** min_train=120 is sized to land the FIRST DECISION at 1972 (design's
explicit target), not to guarantee CV fold balance — those are two independent constraints
that happen to collide in exactly the early-history region the design cares about most
(1972–74 is one of the named crisis windows).
**How to avoid:** Wrap the per-step L2 refit in a try/except that logs a WARNING and either
(a) skips the tilt update for that step (hold previous weights / go neutral) or (b) reduces
`n_splits` for early steps where the post-embargo sample is small, and document the coverage
gap explicitly in the report per the D-01 diagnostic framing ("ran and reported correctly,"
including reporting where it degraded). Do not let an early-step exception crash the whole
1972–2020 run.
**Warning signs:** A stack trace from `sklearn` during the first several years of the loop;
an equity curve that mysteriously starts several years later than 1972.

### Pitfall 3: Turnover computed against the WRONG previous-weights baseline

**What goes wrong:** Computing `turnover_t = |weights_t - weights_{t-1}|.sum()` using
`weights_{t-1}` as the TARGET weights from the previous rebalance rather than the ACTUAL
drifted weights the portfolio would hold going into month t (asset prices moved between
rebalances, so held weights drift away from the prior target even with no new trade).
**Why it happens:** The simplest implementation just diffs consecutive target-weight
vectors, which slightly overstates true turnover (ignoring price drift) but is a defensible,
DOCUMENTED simplification — NOT computing drift-adjusted weights is fine as long as it is
stated, since design explicitly scopes cost realism to "a light touch," not full portfolio
accounting (D-03: "not full tax/friction modeling").
**How to avoid:** Pick one convention (target-vs-target, the simpler one, is acceptable per
D-03's explicit scope) and document it in the report next to the turnover number — do not
silently mix conventions between the strategy leg and the baseline legs (60/40, Faber).
**Warning signs:** Turnover numbers that look implausibly small or large relative to the
known effect of hysteresis (Phase 4 CONTEXT: hysteresis "already keeps turnover low" — a
sanity check the report should be able to confirm, not contradict).

### Pitfall 4: Crisis-window capture ratios accidentally including holdout dates

**What goes wrong:** A capture-ratio helper written generically (e.g., "trailing N months
from a bear-market peak") could, if not bounded, extend a computed window past 2020-12 for
a poorly-chosen anchor date, or a config typo could list `2020-02` (COVID crash) or
`2022-01` among the "in-sample" windows.
**Why it happens:** CONTEXT is explicit ("NOT 2020/2022 — those are in the holdout") but the
crisis windows are config-driven (Claude's discretion) — a config default that isn't
carefully bounded is an easy typo away from a holdout violation, which is exactly the kind
of silent contamination the whole honesty framework exists to prevent.
**How to avoid:** Hard-code the config default crisis window list to `1973-01..1974-12`,
`2000-01..2002-12`, `2008-01..2009-12` (optionally `1980-01..1982-12`) and add an explicit
assertion (or reuse `holdout.split_by_holdout_boundary`) that every configured window's END
date is `<= 2020-12-31` before computing any capture ratio.
**Warning signs:** A capture-ratio table row referencing dates in 2021+.

### Pitfall 5: EVAL-04's salvaged helpers assume incumbent `FoldReport`/bundle-API objects

**What goes wrong:** Importing `ideas/gsd-salvage/prediction/model_metrics_artifacts.py`
directly (or its sibling `classifier.py`) pulls in the incumbent quarterly pipeline's
bundle-API dataclass (`FoldReport`) and its `.report`/`.proba_test`/`.class_order`
attributes — a shape the platform's walk-forward driver does not produce and, per this
project's D-01 conventions (see CONTEXT canonical refs: "adapting salvaged
`model_metrics_artifacts.py`", never "importing"), must not depend on.
**Why it happens:** The salvage file's outer orchestrator (`write_model_metrics_artifacts`)
is written specifically against `regime_current_bundle["cv_scores"]`/`forward_models`/
`behavior_bundle` — three incumbent-pipeline-shaped inputs that don't exist in `platform/`.
**How to avoid:** Extract and adapt ONLY the three pure functions that operate on plain
`(y_true: list, proba: np.ndarray, classes: list)` / `(y_true, y_pred, classes)` —
`_compute_brier_multiclass`, `_calibration_bins`, `_confusion_tidy` — into the new
`platform/evaluation/model_metrics.py`. Write a NEW orchestrator shaped like
`gap_lag.py`'s `compute_* + report_*` pattern that consumes the backtest driver's own
per-step `(y_true, proba, classes)` collection (see Pattern 1's `y_true_log`/`proba_log`/
`classes_log` accumulators) — no `FoldReport`, no per-fold/per-horizon/per-asset grouping
dimensions unless the planner deliberately wants to extend EVAL-04 to also cover a
walk-forward CV inside each step (out of scope per the salvage-adaptation framing in
CONTEXT).
**Warning signs:** A `grep -rn "FoldReport\|classifier\.py\|gsd-salvage" src/trading_crab_lib/platform/` hit anywhere under `platform/evaluation/`.

## Code Examples

### EVAL-04: adapted Brier/calibration/confusion helpers (verbatim math, new call shape)

```python
# Source: adapted (math verbatim, call shape rewritten) from
# ideas/gsd-salvage/prediction/model_metrics_artifacts.py — read directly this session.
# Do NOT import that file or trading_crab_lib.prediction.classifier.FoldReport (D-01, Pitfall 5).
from __future__ import annotations
import numpy as np
import pandas as pd

BIN_EDGES = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], dtype=float)

def compute_brier_multiclass(y_true: list, proba: np.ndarray, classes: list) -> float:
    """Mean over samples/classes of (p_k - onehot_k)^2 — verbatim salvage math."""
    class_to_idx = {c: i for i, c in enumerate(classes)}
    onehot = np.zeros_like(proba, dtype=float)
    for i, yt in enumerate(y_true):
        idx = class_to_idx.get(yt)
        if idx is not None:
            onehot[i, idx] = 1.0
    diff = proba - onehot
    return float(np.mean(diff * diff))

def calibration_bins(y_true: list, proba: np.ndarray, classes: list) -> list[dict]:
    """Tidy per-class calibration bins — verbatim salvage math, 5 fixed bins."""
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_arr = np.array(y_true, dtype=object)
    rows: list[dict] = []
    for c in classes:
        p = proba[:, class_to_idx[c]]
        true_mask = y_arr == c
        for b in range(len(BIN_EDGES) - 1):
            low, high = float(BIN_EDGES[b]), float(BIN_EDGES[b + 1])
            mask = (p >= low) & (p <= high) if b == len(BIN_EDGES) - 2 else (p >= low) & (p < high)
            n = int(mask.sum())
            if n == 0:
                continue
            rows.append({"class_label": str(c), "bin": b + 1, "bin_low": low, "bin_high": high,
                         "predicted_prob_mean": float(p[mask].mean()),
                         "observed_freq": float(true_mask[mask].mean()), "n_in_bin": n})
    return rows
```

### EVAL-01: holdout-bounded index before constructing the walk-forward loop

```python
# Source: pattern combining honesty/holdout.py::split_by_holdout_boundary (read this session)
# with honesty/walkforward.py::expanding_steps.
from trading_crab_lib.platform.honesty.holdout import split_by_holdout_boundary, DEFAULT_HOLDOUT_CUTOFF

dev_features, _holdout_features = split_by_holdout_boundary(monthly_features, cutoff=DEFAULT_HOLDOUT_CUTOFF)
# dev_features now guaranteed <= 2020-12-31 — expanding_steps(dev_features.index, min_train=120)
# can never yield a test_index past the holdout boundary (Pitfall 4 / D-04 discipline).
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|---------------|--------|
| Incumbent quarterly pipeline: single full-sample CV accuracy, no walk-forward, no baseline gauntlet | Platform: expanding-window walk-forward with refit at every step, brutal-baseline gauntlet including a no-regime ablation | Phase 2 (walk-forward infra) → Phase 5 (this phase, first real use) | The incumbent's reported CV metrics are look-ahead-contaminated by construction (full-sample cluster fit before any CV split, per design §11 R10); Phase 5 is the first honest end-to-end number for the platform. |
| Phase 3's `evaluate_nowcaster()` one-shot full-embargoed-set fit | Phase 5's per-step walk-forward refit inside `run_backtest()` | This phase | `evaluate_nowcaster()` remains valid as a quick diagnostic/registry-trial tool but is NOT the backtest — see Summary finding 2. |

**Deprecated/outdated:** none — all consumed interfaces are current/frozen for this
milestone; nothing in this phase's dependency chain is being replaced.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `run_walkforward()` cannot be reused verbatim for the L1→L4 backtest and a new per-step orchestration is needed; only `expanding_steps()` is reused directly | Summary, Pattern 1 | Low — this is a direct reading of `run_walkforward`'s source (single sklearn-model `.fit/.predict` body) against the multi-layer requirement; if the planner disagrees, the alternative (subclassing/parameterizing `run_walkforward` to accept an arbitrary step-callback) is a reasonable variant that still satisfies EVAL-01, just via a different code shape. |
| A2 | Refitting L1 (jump model) + L2 (nowcaster) at EVERY monthly step (588 refits across 1972–2020) is computationally feasible within a normal dev-loop budget, based on the algorithmic complexity read from `jump_model.py`/`nowcaster.py` (K=5, T≤~700, n_restarts=10, O(TK) DP decode; small-sample `LogisticRegression`) | Summary, Pattern 1 | Medium — not benchmarked in this research session (no execution attempted). If wall-clock time proves prohibitive during implementation, the documented fallback is a coarser refit cadence (e.g., annual refit, monthly `predict_proba` on the frozen model in between) — this is explicitly a Claude's-Discretion-shaped engineering knob, not a design-locked requirement (design mandates "walk-forward" discipline — data ≤ t only — but does not literally mandate monthly refit granularity). Recommend a `backtest.refit_frequency_months` config key so this can be tuned without code changes. |
| A3 | The "no-regime ablation" is best constructed as `returns_by_regime_stats(returns, pd.Series(0, index=...))` (a constant single-state label series) fed through the UNCHANGED `regime_tilt_weights`/`vol_targeted_tilt` — producing a pooled-Sharpe, vol-targeted-but-not-regime-conditioned baseline | Pattern 1, Pattern 2, code_context invariant | Low-medium — this is the most direct reading of D-02's "same L3/L4 pipeline with the regime tilt disabled (regime-agnostic: vol-target-only / flat-weight variant)" and of the CONTEXT-named invariant test. An equal-weight-vol-target alternative (skip the Sharpe-tilt ranking entirely) is the documented alternative construction if the planner judges "flat-weight" more literally — CONTEXT leaves both readings open ("equal-weight vs vol-target-only... documented in the report"). |
| A4 | Faber's 10-month SMA signal should be computed with a one-month decision lag (signal known at close of month t decides the position HELD during month t+1), matching the project's existing causal-feature convention | Pattern 2 | Low — this is standard Faber (2007) implementation practice and consistent with every other causal-gating convention in this codebase (`assert_causal_features`, publication-lag shifts); an alternative same-month-execution convention exists in some backtest literature but would introduce a subtle look-ahead inconsistent with the rest of the platform's honesty framework. |
| A5 | 60/40 and Faber baselines should have the SAME `cost_bps` transaction-cost convention applied on their own turnover, for an apples-to-apples §23.1 comparison, even though D-03's text scopes cost realism to "the strategy leg" | Anti-Patterns, Pitfall 3 | Low-medium — if the planner instead reads D-03 literally (costs apply ONLY to the regime-tilt strategy, baselines stay frictionless), the Faber/§23.1 "beats it on log wealth AND max drawdown" comparison becomes cost-asymmetric; this should be an explicit planning decision, documented either way in the report, not a silent default. |
| A6 | Crisis-window capture ratio = strategy cumulative return over the window ÷ SPY (`equities_tr`) cumulative return over the same window (down-capture convention; both typically negative) | Summary (implied), not directly quoted from design | Low — design §8.9 names "capture ratios in ex-post crisis windows" without giving the exact formula; down-capture (strategy/benchmark, both negative-return periods) is the standard finance-industry definition and is consistent with the platform's crash-avoidance framing (§23.1: "absorb first ~10-15%... sidestep the subsequent ~30-40%" — capture ratio is exactly what quantifies this). |

## Open Questions

1. **Exact refit cadence for L1/L2 inside the walk-forward loop (monthly vs. periodic).**
   - What we know: design §8.1 mandates refitting "at each rebalance t" and asserts this is
     computationally cheap ("minutes over 60 years"); the monthly modeling spine is the
     project's stated cadence (root CLAUDE.md: "monthly modeling spine, weekly scoring").
   - What's unclear: whether "each rebalance" literally means every one of the 588 monthly
     steps, or whether a coarser (e.g., annual) L1/L2 refit — still fully causal/walk-forward
     — satisfies the design's "walk-forward everything" intent while being materially cheaper
     to implement and debug.
   - Recommendation: default to true monthly refit (matches the design text most literally,
     and Assumption A2's complexity estimate suggests it is tractable); expose
     `backtest.refit_frequency_months` in `config/platform_settings.yaml` as an escape hatch
     the planner can lower if implementation proves too slow, documenting the tradeoff either
     way in the phase's verification notes.

2. **Where exactly does the trial-registry row for the FULL backtest run get logged — once
   per (baseline × ablation) combination, or once for the whole gauntlet?**
   - What we know: `registry.append_trial()`'s existing convention (both `run_walkforward`
     and `evaluate_nowcaster`) is "exactly one call per evaluated configuration" — and D-02's
     no-regime ablation is explicitly framed as its own "evaluated configuration."
   - What's unclear: whether the 3 non-regime baselines (SPY, 60/40, Faber) — which involve
     no model-fitting/tuning at all, just deterministic price-series arithmetic — also count
     as "evaluated configurations" for the trial registry (relevant to the eventual DSR
     denominator, D-04, though DSR itself is out of scope this phase).
   - Recommendation: log ONE trial for the regime-tilt strategy run and ONE trial for the
     no-regime ablation (both are genuinely "configurations" with fitted models inside them);
     treat the 3 price-only baselines as report-only comparisons, NOT registry trials (they
     involve no tunable parameters or model selection, so they are not part of the
     multiple-testing surface the registry exists to bound) — but flag this as a planning
     decision to confirm, since CONTEXT doesn't resolve it explicitly.

3. **Does the "smoothed" reference series for the gap/lag diagnostic (Pattern 3 / Pitfall 1)
   need its own registry trial, separate from the walk-forward "filtered" run?**
   - What we know: it's a full-sample, non-walk-forward L1 fit purely for the honesty
     diagnostic — not a strategy configuration that is itself evaluated for performance
     selection.
   - What's unclear: whether logging it to the registry at all is appropriate (it's
     diagnostic, not a candidate strategy).
   - Recommendation: do NOT log the smoothed reference fit as a registry trial — it is a
     measurement tool for the gap metric, analogous to how `returns_by_regime_stats` itself
     is explicitly exempted from the registry per Phase 4's `assets/returns.py` docstring
     ("explicitly exempt from the trial registry... it is descriptive statistics... not an
     evaluated model configuration").

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest ≥8.0 (project pin) |
| Config file | `pyproject.toml` `[tool.pytest.ini_options]` (existing, no changes needed) |
| Quick run command | `pytest tests/unit/test_platform_backtest_*.py tests/unit/test_platform_evaluation_*.py -x -q` |
| Full suite command | `pytest tests/ -v` |

Follows the established `tests/unit/test_platform_*.py` convention exactly (see
`test_platform_walkforward.py`/`test_platform_gap_lag.py` read this session): synthetic
monthly DataFrames constructed in-file (no network, no real checkpoint data), `tmp_path`
fixtures for any file I/O, one `class Test<Behavior>` per behavior with a docstring per test
method naming the invariant being proved.

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| EVAL-01 | Backtest never sees data past the holdout cutoff (train_index/test_index always ≤ 2020-12-31) | unit (invariant, synthetic index extending past 2020) | `pytest tests/unit/test_platform_backtest_driver.py::TestHoldoutBoundary -x` | ❌ Wave 0 |
| EVAL-01 | Equity-curve compounding is correct (log-wealth = cumsum(log1p(returns)) for a known synthetic return sequence) | unit (property/known-answer) | `pytest tests/unit/test_platform_backtest_driver.py::TestEquityCurveCompounding -x` | ❌ Wave 0 |
| EVAL-01 | `run_backtest` logs exactly one registry trial per run (mirrors `run_walkforward`'s existing convention) | unit | `pytest tests/unit/test_platform_backtest_driver.py::TestRegistryLogging -x` | ❌ Wave 0 |
| EVAL-01/D-03 | cost = turnover × bps identity holds exactly for a hand-constructed weight sequence | unit (known-answer) | `pytest tests/unit/test_platform_backtest_costs.py::TestCostIdentity -x` | ❌ Wave 0 |
| EVAL-02/D-02 | No-regime ablation (tilt-off, constant-label input) reproduces byte-identical weights/equity-curve to a hand-computed pooled-Sharpe vol-target baseline on synthetic data | unit (invariant) | `pytest tests/unit/test_platform_backtest_baselines.py::TestNoRegimeAblationInvariant -x` | ❌ Wave 0 |
| EVAL-02 | Faber SMA signal never uses same-month price to decide same-month position (1-step lag) | unit (leakage invariant, synthetic price series) | `pytest tests/unit/test_platform_backtest_baselines.py::TestFaberNoLookahead -x` | ❌ Wave 0 |
| EVAL-03 | `sojourn_lag_ratio` orchestration wires `occupancy_and_sojourns` output correctly into the already-tested `gap_lag.sojourn_lag_ratio` (known-construction synthetic state series) | unit (known-answer, reuses existing gap_lag tests as ground truth) | `pytest tests/unit/test_platform_evaluation_sojourn_lag.py -x` | ❌ Wave 0 |
| EVAL-04 | Multiclass Brier score is ~0 for perfectly-calibrated one-hot predictions and >0 for miscalibrated ones (known-answer) | unit | `pytest tests/unit/test_platform_evaluation_model_metrics.py::TestBrierKnownAnswer -x` | ❌ Wave 0 |
| EVAL-04 | Confusion-table counts sum to n (invariant) | unit | `pytest tests/unit/test_platform_evaluation_model_metrics.py::TestConfusionSumsToN -x` | ❌ Wave 0 |
| EVAL-01/EVAL-02/EVAL-03/EVAL-04 | End-to-end: full backtest + all 4 baselines + report assembly runs on a small synthetic monthly frame (≥ min_train + a few steps) with no network access, producing all documented artifacts | integration (synthetic-frame, no-network, mirrors `tests/integration/test_mini_pipeline.py`) | `pytest tests/integration/test_mini_backtest.py -x` | ❌ Wave 0 |

### Sampling Rate

- **Per task commit:** the relevant `test_platform_backtest_*.py` / `test_platform_evaluation_*.py` file (quick run command above).
- **Per wave merge:** `pytest tests/ -v` (full suite — the project runs 863+ tests today; the walk-forward loop over even a small synthetic frame should stay well under the existing suite's runtime budget).
- **Phase gate:** Full suite green before `/gsd-verify-work`; additionally, since this phase's headline artifact IS a real 1972–2020 run (not just unit tests), plan a `checkpoint:human-verify` or scripted `__main__` self-check step that actually executes `run_backtest()` against real Phase 1 checkpoint data end-to-end (analogous to every existing module's `if __name__ == "__main__":` synthetic self-check, but here the phase's OWN exit criterion — design §14 "honest backtest 1972–2020 runs end to end" — requires a REAL run, not only a synthetic-data unit test pass).

### Wave 0 Gaps

- [ ] `tests/unit/test_platform_backtest_driver.py` — covers EVAL-01 (holdout boundary, equity-curve compounding, registry logging, per-step L1/L2 refit-from-train-window-only invariant)
- [ ] `tests/unit/test_platform_backtest_baselines.py` — covers EVAL-02 (SPY, 60/40, Faber no-lookahead, no-regime-ablation invariant)
- [ ] `tests/unit/test_platform_backtest_costs.py` — covers D-03 (turnover, cost identity)
- [ ] `tests/unit/test_platform_evaluation_sojourn_lag.py` — covers EVAL-03 (orchestration correctness against known synthetic construction)
- [ ] `tests/unit/test_platform_evaluation_model_metrics.py` — covers EVAL-04 (Brier known-answer, calibration bin edges, confusion sums)
- [ ] `tests/unit/test_platform_evaluation_kpis.py` — covers strategy KPIs (terminal log wealth, max drawdown+duration, CVaR(5%), crisis capture ratio window-bounding per Pitfall 4)
- [ ] `tests/unit/test_platform_evaluation_report.py` — covers report assembly (headline ordering: sojourn/lag ratio first, per D-01a/specifics)
- [ ] `tests/integration/test_mini_backtest.py` — synthetic end-to-end (mirrors `test_mini_pipeline.py`'s pattern), no network, no real checkpoints
- [ ] Framework install: none — pytest already installed and configured project-wide

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | No | No auth surface — batch CLI/library, single-operator local execution (matches every other `platform/` module). |
| V3 Session Management | No | No sessions. |
| V4 Access Control | No | No multi-tenant/access-control surface. |
| V5 Input Validation | Yes | Config values (`cost_bps`, `crisis_windows`, `min_train`, `refit_frequency_months`) read via `cfg.get(...)` with documented defaults, never `eval`/`exec`; YAML loaded via `yaml.safe_load` only (matches `platform/config.py` and `report/holdings.py`'s existing V5 pattern). Crisis-window bounds validated against the holdout cutoff before use (Pitfall 4) — this IS the security-relevant control in this phase (preventing accidental holdout-data disclosure into a dev-visible report, the closest analog to an information-disclosure control this phase has). |
| V6 Cryptography | No | No new cryptographic material; reuses existing `FRED_API_KEY` handling unchanged (no new secrets introduced by this phase). |

### Known Threat Patterns for this stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Holdout boundary violation (2021+ data leaking into a dev-visible report/artifact/registry trial) | Information Disclosure | `holdout.split_by_holdout_boundary()` applied to `monthly_features` BEFORE constructing `expanding_steps()`; explicit assertion that every configured crisis window ends `<= 2020-12-31` (Pitfall 4); no code path in this phase reads `get_holdout_checkpoint_manager()`. |
| Pickle/joblib deserialization of an untrusted model artifact | Tampering | N/A for new code in this phase — the walk-forward driver fits fresh models in-memory per step and does not load a persisted nowcaster/jump-model artifact from disk; if the planner chooses to also persist the FINAL step's fitted models for inspection, use `joblib.dump`/`joblib.load` only (project convention, P27 in root CLAUDE.md), never raw `pickle`. |
| Config-driven crisis-window / cost-bps values used unsanitized in a report string | Tampering (low severity — local trusted config file, not user input) | Values are numeric (float/date), read via `.get()` with type expectations; no string interpolation into shell commands or file paths derived from these config values. |

## Sources

### Primary (HIGH confidence — read directly from repo this session)

- `src/trading_crab_lib/platform/honesty/walkforward.py` — `expanding_steps()`, `run_walkforward()` full source read.
- `src/trading_crab_lib/platform/honesty/gap_lag.py` — `compute_gap`, `compute_detection_lag`, `sojourn_lag_ratio`, `report_gap_lag` full source read.
- `src/trading_crab_lib/platform/honesty/registry.py`, `holdout.py`, `cv.py` — full source read.
- `src/trading_crab_lib/platform/allocation/tilt.py`, `hysteresis.py` — full source read.
- `src/trading_crab_lib/platform/assets/returns.py`, `vol.py` — full source read.
- `src/trading_crab_lib/platform/labeling/jump_model.py`, `diagnostics.py` (partial — `occupancy_and_sojourns`, `label_regimes`) — full/partial source read.
- `src/trading_crab_lib/platform/prediction/nowcaster.py`, `transition_matrix.py` — full source read.
- `src/trading_crab_lib/platform/splice.py`, `transforms_monthly.py`, `taxonomy.py`, `config.py`, `checkpoints.py` — full/partial source read.
- `src/trading_crab_lib/platform/report/weekly.py` — full source read (report-module shape reference).
- `config/platform_settings.yaml` — full file read (no `backtest:`/`evaluation:` section present yet, confirming CONTEXT's "gains a new section" framing).
- `ideas/gsd-salvage/prediction/model_metrics_artifacts.py` — full source read (EVAL-04 adaptation source).
- `tests/unit/test_platform_walkforward.py`, `test_platform_gap_lag.py` — full source read (test-convention reference).
- `platform_design/platform_design.md` §5.4, §8, §9, §11, §14, §23.1 — full section text read (design citations throughout this document).
- `.planning/phases/05-honest-backtest-evaluation/05-CONTEXT.md`, `.planning/REQUIREMENTS.md`, `.planning/STATE.md` — full read.
- `CLAUDE.md` (root), `.claude/CLAUDE.md` — read at session start (project conventions, functions-only lib, TDD, config `.get()` reads).

### Secondary (MEDIUM confidence)

- None — every claim in this document traces to a directly-read source file or design section; no web search was needed (Phase 5's entire technical surface is internal to this repo).

### Tertiary (LOW confidence)

- None.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — no new dependencies; every reused function's exact signature was read from source this session.
- Architecture: HIGH for the "what already exists and what's frozen" claims (all read directly); MEDIUM for the "how the new backtest driver should be shaped" recommendations (original design synthesis, clearly marked as such in the Assumptions Log — A1/A2/A3 particularly).
- Pitfalls: HIGH — all 5 pitfalls derive from specific, cited tensions between existing docstrings/code and the phase's stated requirements (e.g. `evaluate_nowcaster`'s own docstring stating it does NOT walk-forward refit; `build_nowcaster_training_set`'s embargo interacting with `min_train`'s 1972 target).

**Research date:** 2026-07-23
**Valid until:** No expiry driver — this research is entirely internal-codebase-grounded (no external library API surface that could drift); revisit only if Phase 1-4 frozen interfaces change before Phase 5 planning begins.
