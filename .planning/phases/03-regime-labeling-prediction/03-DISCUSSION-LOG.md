# Phase 3: Regime Labeling & Prediction - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-07-22
**Phase:** 3-Regime Labeling & Prediction
**Areas discussed:** Embargo window width, §4.4 enforcement, Soft confidences, Regime naming

---

## Embargo window width (L1-02)

| Option | Description | Selected |
|--------|-------------|----------|
| 12 months | Conservative end of 6–12 range; matches churn window edge | ✓ |
| 6 months | Keeps more recent training data for the nowcaster | |
| Config knob default 12 | | (folded in — value exposed in config, default 12) |

**User's choice:** (a) 12 months

---

## §4.4 acceptance criteria enforcement

| Option | Description | Selected |
|--------|-------------|----------|
| Report-only diagnostics | Labeler always completes; violations logged loudly + persisted | ✓ |
| Hard gate | Run fails on occupancy/sojourn violation at default λ, K=5 | |

**User's choice:** (a) Report-only in v1; hard-gating deferred to v2 tuning

---

## Soft label confidences (L1-02)

| Option | Description | Selected |
|--------|-------------|----------|
| Softmax over negative centroid distances | Cheap, self-contained, no new deps | ✓ |
| Companion HMM forward-backward γ | Better founded, but pulls v2 benchmark machinery into v1 | |

**User's choice:** (a) Distance softmax

---

## Regime naming (v1)

| Option | Description | Selected |
|--------|-------------|----------|
| Numeric labels + auto one-line profiles | Human-pinned names deferred to Phase 4 report | ✓ |
| Pin names now in platform_regime_labels.yaml | | |

**User's choice:** (a) Numeric + auto profiles

---

## Claude's Discretion

- Default λ derivation convention (researcher confirms); λ and K exposed in config
- DP implementation details (vectorization, restart count, tolerance)
- Nowcaster calibration method (isotonic vs sigmoid), CV via PurgedEmbargoedKFold
- Standardization/winsorization for labeler inputs
- Module layout inside platform/ and checkpoint names

## Deferred Ideas

- (K, λ) grid + stability + t-HMM benchmark — v2 (L1-V2-01)
- Recursive prior-state feature + γ weights — v2 (L2-V2-01)
- TVTP transition model with regime age — v2 (L2-V2-02)
- Human-pinned regime names — Phase 4
- Hard-gating §4.4 — v2

(Note: options presented as plain-text numbered lists per established session pattern —
AskUserQuestion widget has crashed repeatedly; user answered "1a 2a 3a 4a".)
