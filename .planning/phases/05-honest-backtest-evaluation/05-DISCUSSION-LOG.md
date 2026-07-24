# Phase 5: Honest Backtest & Evaluation - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-07-23
**Phase:** 5-Honest Backtest & Evaluation
**Areas discussed:** Verdict framing, Baseline depth, Cost realism, DSR / holdout scope

**Process note:** User selected all four gray areas to discuss, then (after the
per-area questions were presented) directed "continue from where you left off." Per the
stated ground rule that unpicked/unanswered areas take Claude's discretion, the four
decisions were locked to the design-grounded recommended options and captured in
CONTEXT.md for user review before planning.

---

## Verdict framing

| Option | Description | Selected |
|--------|-------------|----------|
| Diagnostic | Run the honest backtest, report all numbers, proceed to Phase 6 regardless of whether it beats a benchmark (design §14 Phase 1: "beats nothing yet — that's fine"). Faber/sojourn comparisons recorded as standing targets. | ✓ |
| Gate on Faber | Phase fails unless the walk-forward strategy beats Faber 10-mo SMA on log wealth AND max drawdown (§23.1). | |
| Gate on sojourn/lag | Pass only if median-sojourn / detection-lag ratio clears ~5. | |

**User's choice:** Diagnostic (recommended default).
**Notes:** §14 Phase 1 explicitly frames the tracer bullet as not-yet-winning; the Faber
bar (§23.1) is the later L4-milestone target, recorded prominently but non-blocking here.

---

## Baseline depth

| Option | Description | Selected |
|--------|-------------|----------|
| Mandated 3 + no-regime ablation | SPY, 60/40, Faber PLUS same L3/L4 pipeline with regime tilt disabled — "the regime layer must pay rent" (§8.7). | ✓ |
| Mandated 3 only | SPY, 60/40, Faber (exactly EVAL-02). | |

**User's choice:** Mandated 3 + no-regime ablation (recommended default).
**Notes:** The ablation is the load-bearing baseline — the only one that attributes edge to
the regime layer vs. plain vol-targeting. Cheap (same code, tilt off).

---

## Cost realism

| Option | Description | Selected |
|--------|-------------|----------|
| Token config-driven cost + turnover | Small per-rebalance bps haircut (config knob, ~10 bps default) AND turnover reported separately. | ✓ |
| Frictionless + turnover reported | No haircut; turnover as a separate diagnostic only. | |

**User's choice:** Token config-driven cost + turnover (recommended default).
**Notes:** Tax/friction modeling is out of scope, but a monthly tilt shouldn't look free;
hysteresis keeps turnover low so the cost is a light touch. Frictionless flatters
higher-turnover configs.

---

## DSR / holdout scope

| Option | Description | Selected |
|--------|-------------|----------|
| Defer both | No DSR-vs-registry, no 2021+ access; Phase 5 strictly ≤2020-12. Both belong to design Phase 6 (freeze), a later milestone. | ✓ |
| Preliminary DSR now | Compute a diagnostic-only, non-binding deflated Sharpe against the current registry (still no 2021+). | |

**User's choice:** Defer both (recommended default).
**Notes:** Preserves holdout discipline — the point of the honesty framework. Preliminary
DSR rejected to avoid it creeping into selection decisions.

---

## Claude's Discretion

- Backtest module layout inside `platform/` (backtest/ and/or evaluation/).
- min_train warmup that lands the first backtested rebalance in 1972.
- In-sample crisis windows for capture ratios (1973–74, 1980–82, 2000–02, 2008–09; not
  2020/2022).
- 60/40 rebalance convention; exact no-regime ablation construction (equal-weight vs
  vol-target-only).
- EWMA/turnover/CVaR conventions; report layout (markdown + plots).
- Reuse Phase 4 allocation entry point vs. thin backtest-driver wrapper (prefer reuse).

## Deferred Ideas

- Deflated Sharpe vs. full trial registry — design Phase 6 / future milestone.
- Single 2021+ holdout evaluation — design Phase 6, post-freeze only.
- Faber/SPY-beating as a hard gate — L4-upgrade milestone (design Phase 5); standing target here.
- Full Diebold–Mariano / Mincer–Zarnowitz forecast KPIs (§8.8) — v2.
- BL/HRP/Kelly/stops/crash-dashboard allocation upgrades — v2 (L4-V2-*).
