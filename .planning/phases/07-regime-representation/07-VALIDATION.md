---
phase: 7
slug: regime-representation
scope: phase-wave 2 ONLY (criteria 5, 6, 7 + INV-01)
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-09-15
supersedes: 07-VALIDATION-WAVE1.md (wave 1's, status validated — reused, not replaced)
---

# Phase 7 — Wave 2 Validation Strategy

> Wave 1's contract is preserved at `07-VALIDATION-WAVE1.md` and every band in it is **reused
> unchanged** for classifier #2's own version of each quantity. This file adds only what wave 2
> introduces.
>
> Source: `07-RESEARCH.md` §Validation Architecture + §Plausibility Bands.

---

> **CORRECTION 2026-09-15:** two test filenames in the original draft of this file did not
> exist — I invented `test_platform_allocation_joint_tilt.py` and
> `test_platform_ingestion_macro_monthly.py`. The real files are **`test_platform_tilt.py`**
> and **`test_platform_macro_ingest.py`**. Caught by the wave-2 planner checking the tree
> rather than trusting this document. Corrected throughout.

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.0+ |
| **Config** | root `pyproject.toml` → `[tool.pytest.ini_options]` |
| **Quick run** | `pytest tests/unit/test_platform_labeling.py tests/unit/test_platform_features_relative.py tests/unit/test_platform_evaluation_dependence.py tests/unit/test_platform_tilt.py tests/unit/test_platform_legacy_import_ratchet.py -x` |
| **Full suite** | `pytest tests/ -q` |
| **Current baseline** | **1752 passed, 0 skipped, ~72s** (measured on `main` @ `34ffa30`) |
| **New dependencies** | **none** |

---

## Sampling Rate

- **Per task commit:** quick run (~15–30s)
- **Per wave merge:** full suite
- **Phase gate:** ≥1 real walk-forward of the joint (#1×#2) tilt vs #1 alone on live checkpoints
- **Any drop below 1752, or any new skip, is a regression to FIX** — not to report and move past

---

## Requirements → Test Map

| Criterion | Behavior | Command | Exists? |
|---|---|---|---|
| **5** disjointness | #2's raw candidates disjoint from `lean_feature_set(cfg)`'s 13 | `-k disjoint` in `test_platform_features_relative.py` | ❌ Wave 0 |
| **5** canonicalize fix | `canonicalize_states` never silently falls back for #2; **raises `ValueError`** on an absent `sort_column` | `-k canonicalize` in `test_platform_labeling.py` | ❌ Wave 0 (extends `TestCanonicalizeStates`) |
| **5** occupancy | sums to 1.0; every state under §4.4's ~8% floor (§4.4 crit. 1 is ~8%–~35%; see ADR-0001 § AMENDMENT 2026-09-17) flagged (`_MIN_OCCUPANCY_THRESHOLD`, `labeling/diagnostics.py:67`) | `-k occupancy` | ✅ generic helpers exist — extend with a #2-shaped fixture |
| **6** dependence | ARI + NMI + Cramér's V + crosstab, **no pass/fail gate** (D-15) | `test_platform_evaluation_dependence.py` | ❌ Wave 0 |
| **7** joint tilt | `blend_regime_tilts` weights sum to `scale`; degrades on empty input | `test_platform_tilt.py` | ❌ Wave 0 |
| **7** deflated Sharpe | DSR vs a hand-worked oracle; `total_trial_count` **reads the provenance header** | `test_platform_evaluation_deflated_sharpe.py` | ❌ Wave 0 — **nothing exists to extend** |
| **7** joint lift, real data | measured vs #1 alone, inside its own reported window | manual | Manual-only (multi-minute real walk-forward) |
| **8** ratchet guard | porting must not raise legacy imports past 31 | `test_platform_legacy_import_ratchet.py` | ✅ **exists — this IS the guard** |
| **INV-01** | M2SL/TOTALSL ingested, aligned monthly, no interpolation | new cases in `test_platform_macro_ingest.py` + one live fetch | ❌ Wave 0 |
| **ADR** | sort convention, credit aggregate, blend weight, L2-window decision, DSR estimator, trial ceiling | document review | N/A |

---

## Wave 0 Gaps

- [ ] `test_platform_labeling.py` — `sort_column` no-fallback oracle + missing-column `ValueError`
- [ ] `test_platform_features_relative.py` — **NEW**: disjointness + ported-function parity vs hand-computed examples
- [ ] `test_platform_evaluation_dependence.py` — **NEW**: identical labelings → ARI=NMI=1; independent → near null
- [ ] `test_platform_tilt.py` — **NEW**: blend vs hand-computed; degenerate inputs
- [ ] `test_platform_evaluation_deflated_sharpe.py` — **NEW**: DSR oracle (N=1 → raw significance; N→∞ → toward null) + `total_trial_count` against a synthetic header row
- [ ] INV-01 ingestion config entries + live smoke
- [ ] Framework install: none

---

## Plausibility Bands — wave 2's new quantities

All `06-VALIDATION.md` and `07-VALIDATION-WAVE1.md` bands are **reused unchanged**.

| Quantity | Universal bound | Domain band | Tag |
|---|---|---|---|
| #2 occupancy | each `∈[0,1]`, `Σ=1.0` | `<0.05` soft warning | **[CITED]** — same constant #1 uses |
| **ARI** (crit 6) | `x ∈ [-1, 1]` | **no gate (D-15)**; `>0.7` flagged in prose as candidate failure-to-add-an-axis; near 0 is the *target* | **[ASSUMED]** |
| **NMI** (crit 6) | `x ∈ [0, 1]` | **no gate**; `>0.5` flagged | **[ASSUMED]** |
| **Cramér's V** (crit 6) | `x ∈ [0, 1]` | **no gate**; `>0.5` flagged | **[ASSUMED]** |
| Joint-tilt weights | each `≥0`, `Σ ≤ scale ≤ 1` | matches `vol_targeted_tilt`'s existing contract | **[CITED]** |
| Joint-vs-#1 deltas | `abs(wealth_delta) < 15`, `dd_delta ∈ [-2, 2]` | ⚠️ **the four load-bearing bands below must be CONFIRMED first** | **[CITED]** universal / **[ASSUMED]** domain |
| **Deflated Sharpe** | `x ∈ [0, 1]` (a `norm.cdf` output) | **no target**; a DSR at/below 0.5 must be reported plainly as "does not clear the multiple-testing hurdle" — **not softened** | **[ASSUMED]** |
| `total_trial_count()` | `x ≥ prior_genuine_trials` (38) — **a count below the header's own stated prior is a parsing bug, not a reading** | — | **[VERIFIED]** this session |

### ⚠ GATE: four bands must be confirmed before criterion 7 is judged

`abs(wealth_delta) < 5` · `dd_delta ∈ [-0.5, 0.5]` · `n_transitions > 30` implausible ·
`pct_disagree < 0.02` suspicious.

D-07 kept these advisory through wave 1, so nothing turned on their values. **Criterion 7 judges
joint lift against them.** Confirm or revise all four before any joint-lift number is assessed —
otherwise criterion 7 is measured against numbers nobody has agreed to.

### The "suspiciously clean" class, extended to dependence

Wave 1 added a third check class: a number that *looks* good but signals a different bug. Applied
to criterion 6:

- **ARI or NMI exactly 1.0** almost certainly means both classifiers are reading the same
  underlying labels — a wiring bug, the same failure class `disagreement.py`'s
  `suspicious`/`suspicious_reason` fields already guard for criterion 3.
- **All three statistics exactly 0.0 simultaneously** is unusually clean for real financial data
  and warrants the same "confirm this isn't an alignment bug" scrutiny.

A dependence statistic is not self-validating. High dependence is a **finding** (criterion 6 says
so explicitly); a *perfect* statistic in either direction is a **suspicion**.

---

## Evidence-Shape Requirement

This project has been burned twice, both times by evidence that could only confirm:

1. **`UAT-AUDIT-2026-09-09`** — terminal log wealth **111.06** (e¹¹¹ ≈ 10⁴⁸) tabulated as an
   *improvement*. Every criterion passed as phrased; all evidence was existence-, shape-,
   pure-function- or placement-shaped.
2. **2026-09-15** — criterion 8's "verified fully decoupled" rested on a grep ending
   `| grep -v platform`, which discarded every match line and **could not fail**. 31 real
   imports were hiding behind it.

**Contract:** every verification row must be able to name **the value or state it would reject**.
A check that cannot fail is worse than no check — it manufactures confidence.

---

## Validation Sign-Off

- [ ] Every requirement has an automated command or a documented manual-only reason
- [ ] No 3 consecutive tasks without an automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] Suite ≥ 1752, 0 new skips
- [ ] **Four load-bearing `[ASSUMED]` bands confirmed or revised**
- [ ] Legacy-import ratchet still ≤ 31 after the port
- [ ] `nyquist_compliant: true`

**Approval:** pending
