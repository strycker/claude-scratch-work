---
phase: 07-regime-representation
plan: 10
subsystem: platform/allocation
tags: [joint-tilt, D-14, plausibility-bands, adr-0001-condition-iv, honesty-framework]
status: complete
requires:
  - "07-08 (allocation.blend_weight_1 = 0.50; ADR-0002 (f))"
  - "ADR-0001 § RE-PIN (classifier #1 crisis state, recurrence exemption)"
provides:
  - "platform/allocation/joint_tilt.py — blend_regime_tilts, blend_weight_from_config, low_n_regime_flags, pool_low_n_regime_sharpe"
  - "07-BANDS.md with all four dispositions decided"
  - "ADR-0001 exemption condition (iv) implemented for per-regime Sharpe"
  - "disagreement.py band-4 denominator clauses"
affects:
  - "07-11 (consumes blend_regime_tilts for BOTH legs; implements bands 1-3)"
---

# 07-10 — Joint tilt, plausibility bands, and condition (iv)

## Tasks

| task | status |
|---|---|
| 1 — `blend_regime_tilts` + 37 tests | complete (`cba2674` RED, `7ddbd6c` GREEN) |
| 2 — `07-BANDS.md` assembly | complete (`a13cfeb`) |
| 3 — **human-verify gate** | **SIGNED OFF by Glenn, 2026-09-18** |
| + — condition (iv), folded in by Glenn's direction | complete |

## Task 3 dispositions

Recorded in full in `07-BANDS.md` § 8. In brief: band 1 confirmed as domain tier; band 2's
universal bound **revised `[-2, 2]` → `[-1, 1]`** because it could only confirm; band 3 revised and
split into a definitional bound plus a length-independent rate band; band 4's threshold confirmed
with denominator clauses added. Universal tier governs criterion 7's verdict, domain tier records a
note — **A11 stays open**, not closed by gating on `[ASSUMED]` numbers.

## Condition (iv), implemented rather than promised

ADR-0001's recurrence exemption required the crisis state's estimates be flagged and shrunk. Now
true in code: `low_n_regime_flags` surfaces occupancy/floor/credibility/`low_n`;
`pool_low_n_regime_sharpe` replaces each sub-floor Sharpe with `c·raw + (1−c)·pooled`, credibility
`c = min(1, occupancy/floor)`, the pooled target reconstructed exactly from per-regime sufficient
statistics. At classifier #1's 5.7554%, `c = 0.7194`.

**Credibility is derived, not tuned.** §4.4's floor *is* the criterion's own proxy for "enough
observations"; giving a state the credibility its occupancy earns against that floor takes the
weight from the criterion being exempted. Consequence: a state meeting the floor gets `c = 1.0`, so
pooling is a strict no-op for any compliant labeling — which is why the exact reduction to
`vol_targeted_tilt` at `weight_1 = 1.0` survives.

**Ablation run for real:** disabling the shrinkage fails 3 tests, including one asserting the
blended **portfolio weights** differ and in which direction — not merely that a flag field exists.

## Known gaps, stated

1. **(iv)'s covariance clause is not implemented.** No per-regime covariance exists at L4-01;
   `portfolio_vol` is a linear-sum/EWMA estimate on the blended weight vector. It falls to L3
   (design §6.2, Ledoit–Wolf within regime) and is **not** claimed here.
2. **`vol_targeted_tilt` remains (iv)-non-compliant on its own**, by design — `tilt.py` was not
   modified, and `backtest/driver.py:497` calls it directly. This does **not** affect criterion 7:
   07-11 builds a separate `joint_driver.py` where `blend_regime_tilts` at `weight_1 = 1.0` *is*
   the #1-alone baseline, so both legs share the (iv)-compliant path and the comparison stays a
   one-parameter ablation. Other consumers of `driver.py` remain unshrunk.
3. Criterion 6 **UNRESOLVED**; §4.4 criterion 3's Hungarian test never run; crisis-state median
   sojourn 3.0 months against a 1–3 month detection lag — this labeling identifies crises *ex
   post* and no band answers whether L2 can nowcast them in time.

## Verification

Suite **1949 passed, 0 skipped, 0 xfailed**. `test_platform_tilt.py` passes unchanged and
`git diff` on `tilt.py` is empty — proof it was not modified. Ratchet 31. `ruff` clean.
