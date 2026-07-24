# Phase 5: Honest Backtest & Evaluation - Context

**Gathered:** 2026-07-23
**Status:** Ready for planning

<domain>
## Phase Boundary

Wire the already-built L1→L2→L3→L4 skeleton (jump-model labeler → calibrated
nowcaster → returns-by-regime + EWMA vol → naive vol-targeted regime-tilt allocation)
into a single **expanding-window walk-forward backtest over 1972–2020**, compare it to
real baselines, and report the honesty metrics that reveal whether regime timing adds
value. Covers **EVAL-01…04**.

**In scope:**
- End-to-end walk-forward backtest 1972–2020 through all layers, producing a strategy
  equity curve (EVAL-01).
- Baseline gauntlet: buy-and-hold SPY, 60/40, Faber 10-month SMA, **plus a no-regime
  ablation** of the same L3/L4 pipeline (EVAL-02 + design §8.7).
- Median-sojourn / detection-lag ratio reported prominently as the headline go/no-go
  number (EVAL-03), reusing Phase 2 `honesty/gap_lag.py`.
- Per-run model-metrics artifacts: multiclass Brier, calibration bins, confusion tables
  (EVAL-04), adapting salvaged `model_metrics_artifacts.py`.
- Strategy KPIs (design §8.9): terminal log wealth, max drawdown + duration, CVaR(5%),
  turnover, crisis-window capture ratios (in-sample crises only).

**Out of scope (hard boundaries):**
- **No 2021+ holdout access** and **no deflated-Sharpe-vs-registry** — both belong to
  the design's Phase 6 "freeze & single holdout eval", a later milestone. Phase 5 is
  strictly ≤2020-12.
- No allocation upgrades (BL/HRP, Kelly, model-driven stops, crash dashboard) — those are
  the design's L4 milestone; v1 allocation stays deliberately naive.
- No new modeling. Phase 5 evaluates the frozen Phase 1–4 skeleton; it does not tune it.

</domain>

<decisions>
## Implementation Decisions

### Verdict framing (EVAL-01/03 — how the phase is judged)
- **D-01: Phase 5 is a DIAGNOSTIC, not a gate.** The backtest must RUN end-to-end and
  REPORT every required number honestly; verification passes on "it ran and reported
  correctly," **not** on "it beat a benchmark." Rationale: design §14 Phase 1 explicitly
  says the tracer bullet "beats nothing yet — that's fine." Migration (Phase 6) proceeds
  regardless of whether the naive skeleton wins.
- **D-01a:** The Faber 10-month SMA comparison (log wealth AND max drawdown, per §23.1)
  and the sojourn/detection-lag ratio are surfaced **prominently** in the report as the
  **standing targets** for the later L4-upgrade milestone — recorded, highlighted, but
  not pass/fail for this phase.

### Baseline gauntlet depth (EVAL-02)
- **D-02: Mandated 3 + no-regime ablation.** Baselines are SPY (buy-and-hold), 60/40,
  Faber 10-month SMA, **and** the same L3/L4 allocation pipeline with the regime tilt
  disabled (regime-agnostic: vol-target-only / flat-weight variant). Design §8.7: "no-
  regime versions of every regime-conditional model — the regime layer must pay rent."
  This is the one baseline that directly attributes any edge to the regime layer vs. plain
  vol-targeting. Cheap to build (same pipeline, tilt flag off).

### Cost realism (EVAL-01 strategy leg)
- **D-03: Token config-driven transaction cost + turnover reported separately.** Apply a
  small per-rebalance haircut on traded notional — a config knob under a new
  `backtest:` / `evaluation:` section (default ~10 bps; document derivation) — AND report
  turnover as its own diagnostic. Not full tax/friction modeling (out of scope), but a
  monthly tilt must not look free. Hysteresis (Phase 4) already keeps turnover low, so the
  cost should be a light touch. Frictionless-only was rejected: it flatters higher-turnover
  configs and undercuts the honesty framing.

### DSR / holdout scope (EVAL boundary)
- **D-04: Defer deflated Sharpe and the 2021+ holdout evaluation entirely.** Phase 5
  computes neither. Both are design Phase 6 (freeze) / a future milestone. Keeps holdout
  discipline intact — the point of the honesty framework. (A preliminary, non-binding DSR
  was considered and rejected to avoid any risk of it creeping into selection.)

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

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Design (authoritative)
- `platform_design/platform_design.md` §8 — Evaluation & Honesty Framework: walk-forward
  everything (§8.1), brutal baselines incl. no-regime ablation (§8.7), forecast KPIs
  (§8.8), strategy KPIs vs SPY (§8.9). **The spec for this phase.**
- `platform_design/platform_design.md` §5.4 — Honesty metrics for L2: smoothed-vs-filtered
  gap, detection lag, and the **median-sojourn / detection-lag ratio** (EVAL-03 headline).
- `platform_design/platform_design.md` §14 — Phase plan. **Phase 1 exit** ("honest backtest
  1972–2020 runs end to end; gap + detection lag reported; beats nothing yet — that's
  fine") is the definition of done for GSD Phase 5. **Phase 6** (DSR-vs-registry + single
  2021+ eval) is explicitly OUT (D-04).
- `platform_design/platform_design.md` §23.1 — Honest expectations: the Faber bar
  ("beat it on log wealth AND max drawdown or the machinery isn't earning its complexity")
  — recorded as the standing target (D-01a).
- `platform_design/platform_design.md` §9 — Data span (1962+) and feature taxonomy.

### Requirements & roadmap
- `.planning/REQUIREMENTS.md` — EVAL-01, EVAL-02, EVAL-03, EVAL-04 (this phase);
  MIG-01 (next phase); v2 deferrals bounding scope.
- `.planning/ROADMAP.md` — Phase 5 goal + 4 success criteria.

### Prior-phase context (interfaces this phase consumes)
- `.planning/phases/02-honesty-infrastructure/02-CONTEXT.md` — honesty rails; the frozen
  walk-forward interface contract.
- `.planning/phases/04-asset-prediction-allocation/04-CONTEXT.md` — the L3/L4 allocation
  this backtest drives (target vol 10%, hysteresis 0.7/0.4, universe, naive-by-design).
- `.planning/phases/03-regime-labeling-prediction/03-CONTEXT.md` — L1/L2 APIs.

### Salvage
- `ideas/gsd-salvage/prediction/model_metrics_artifacts.py` — adapt for EVAL-04
  (multiclass Brier, calibration bins, confusion tables persisted per run).

### Codebase (frozen interfaces + reusable module shape)
- `src/trading_crab_lib/platform/honesty/walkforward.py` — `run_walkforward()` /
  `expanding_steps()`; the expanding-window loop + automatic single `append_trial` per run.
- `src/trading_crab_lib/platform/honesty/registry.py` — `append_trial()` / `read_trials()`
  (the multiple-testing ledger; every backtest config logs here).
- `src/trading_crab_lib/platform/honesty/holdout.py` — `split_by_holdout_boundary()` /
  holdout checkpoint manager; enforces the ≤2020-12 boundary (D-04).
- `src/trading_crab_lib/platform/honesty/gap_lag.py` — smoothed-vs-filtered gap + detection
  lag (EVAL-03 inputs; the sojourn/lag ratio builds on this).
- `src/trading_crab_lib/platform/honesty/cv.py` — PurgedEmbargoedKFold (if any supervised
  refit inside the loop needs CV).
- `src/trading_crab_lib/platform/allocation/{tilt,hysteresis}.py`,
  `src/trading_crab_lib/platform/assets/{returns,vol}.py`,
  `src/trading_crab_lib/platform/labeling/jump_model.py`,
  `src/trading_crab_lib/platform/prediction/{nowcaster,transition_matrix}.py` — the layers
  the backtest drives.
- `src/trading_crab_lib/platform/checkpoints.py`, `platform/config.py` — persistence +
  config (`.get()` reads; new `backtest:`/`evaluation:` section not added to
  `_REQUIRED_PLATFORM_SECTIONS`).
- `config/platform_settings.yaml` — gains a `backtest:` / `evaluation:` section (cost bps,
  crisis windows, baseline toggles).
- `CLAUDE.md` (root) — conventions (functions-only lib, `from __future__ import
  annotations`, no-network synthetic-frame tests, TDD for correctness-critical math).

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `run_walkforward()` / `expanding_steps()` — the expanding-window spine already exists and
  auto-logs one trial per run. The backtest is a walk-forward driver on top of this exact
  interface (Phase 2 froze it precisely so Phase 5 plugs in unchanged).
- `gap_lag.py` — smoothed-vs-filtered gap + detection lag already computed as first-class
  outputs (HON-05); EVAL-03's sojourn/lag ratio extends these, doesn't rebuild them.
- Phase 4 allocation (`allocation/tilt.py` + `hysteresis.py`) and asset layer
  (`assets/returns.py`, `assets/vol.py`) — driven per rebalance to produce the strategy
  equity curve; the no-regime ablation is the same code with the tilt disabled.
- `registry.append_trial()` — every evaluated backtest configuration lands in the ledger
  (the DSR denominator, even though DSR itself is deferred).
- `holdout.split_by_holdout_boundary()` — the mechanism that keeps the backtest ≤2020-12.
- The `honesty/gap_lag.py` module shape (`compute_* + report_* + __main__ self-check +
  parquet artifact`) is the template for the new backtest/evaluation modules.

### Established Patterns
- New platform config sections read via `.get()`, never added to
  `_REQUIRED_PLATFORM_SECTIONS` (Phase 2/4 pattern).
- Synthetic-frame, no-network tests; invariant-style headline tests (e.g. a frictionless
  vs. costed run differ only by turnover×bps; a no-regime ablation with tilt-off must
  reproduce the vol-target baseline exactly).
- TDD RED→GREEN for correctness-critical math (equity-curve compounding, cost/turnover
  accounting, sojourn/lag ratio, Brier/calibration).

### Integration Points
- Consumes: monthly features / labels / nowcaster probabilities / daily spliced prices /
  returns-by-regime + EWMA vol / allocation weights — all Phase 1–4 outputs.
- Produces: backtest report (markdown + plots) + parquet artifacts (equity curves per
  strategy/baseline, KPI table, metrics artifacts) under `outputs/reports/platform/` (or a
  `backtest/` subdir — planner's call); trial-registry rows.
- Nothing modifies the incumbent quarterly pipeline; allocation logic is reused, not forked.

</code_context>

<specifics>
## Specific Ideas

- The **sojourn/detection-lag ratio is the headline number** of the whole phase — it must
  be the first thing the report shows, with the §5.4 interpretation ("sojourn ≈ 18m, lag
  ≈ 2m → most of the regime captured; sojourn ≈ 5m → the lag eats the trade"). Design
  hints a ratio ≥ ~5 is the eventual bar (§14 Phase 3), noted for context only.
- The **no-regime ablation** is the scientifically load-bearing baseline: report the
  regime strategy, the ablation, and the delta between them side-by-side — that delta IS
  "does the regime layer pay rent."
- **Smoothed-vs-filtered gap** must be reported as the measured hindsight content of the
  strategy (§5.4) — the honest gap between a lazy full-sample backtest and the walk-forward.
- Report crisis-window capture ratios for **in-sample** crises only (1973–74, 2000–02,
  2008–09…); 2020/2022 are holdout and must not appear.

</specifics>

<deferred>
## Deferred Ideas

- Deflated Sharpe ratio vs. the full trial registry — design Phase 6 (freeze); a future
  milestone (D-04).
- Single 2021+ holdout evaluation — design Phase 6; only after design freeze (D-04).
- Faber/SPY-beating as a hard gate — deferred to the L4-upgrade milestone (design Phase 5);
  recorded here as a standing target only (D-01a).
- Diebold–Mariano / Mincer–Zarnowitz forecast-KPI depth (§8.8 full) — beyond EVAL-04's
  Brier/calibration/confusion; revisit when the MoE/nowcaster upgrades land (v2).
- BL/HRP/Kelly/stops/crash-dashboard allocation upgrades — v2 (L4-V2-*); the backtest here
  evaluates only the naive allocation.

None of the above are in Phase 5 scope; captured so they are not lost.

</deferred>

---

*Phase: 5-Honest Backtest & Evaluation*
*Context gathered: 2026-07-23*
