# Phase 4: Asset Prediction & Allocation - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-07-22
**Phase:** 4-Asset Prediction & Allocation
**Areas discussed:** Holdings YAML schema, Report delivery, Target volatility, Tripwire signals

---

## Holdings YAML schema (L4-03)

| Option | Description | Selected |
|--------|-------------|----------|
| Weights (fractions) per account | No price lookups; weight-vs-weight comparison | ✓ |
| Shares + cash balance | Matches Fidelity display; needs current prices | |

**User's choice:** (a) Weights

---

## Weekly report delivery (L4-02)

| Option | Description | Selected |
|--------|-------------|----------|
| Markdown always + opt-in email via existing machinery | Same pattern as incumbent --send-email | ✓ |
| Markdown only, email later | | |

**User's choice:** (a)

---

## Portfolio target volatility (L4-01)

| Option | Description | Selected |
|--------|-------------|----------|
| 10% annualized | Defensive; matches beat-SPY-net-of-drawdowns objective | ✓ |
| 12% | Middle ground | |
| ~15% | SPY-like risk | |

**User's choice:** (a) 10%, config knob

---

## Tripwire v1 signal set (L4-04)

| Option | Description | Selected |
|--------|-------------|----------|
| Roadmap trio: vol spike, credit-spread velocity, SPY drawdown-from-peak | One per independent family; thresholds config-driven | ✓ |
| Swap a family | | |

**User's choice:** (a) Confirm trio

---

## Claude's Discretion

- EWMA half-life convention (config-exposed)
- Regime-tilt construction details (naive, transparent, documented in report)
- Hysteresis state persistence location
- Report layout; module split in platform/
- Tripwire threshold defaults

## Deferred Ideas

- BL/HRP/Kelly, stops, crash dashboard, covariance layer, MoE, fair-value gap,
  full tripwire orchestrator, Fidelity CSV parser, automation, tactical sleeve —
  all tracked as v2 requirements (see CONTEXT deferred section)

(Options presented as plain-text numbered lists per established session pattern;
user answered "1a 2a 3a 4a".)
