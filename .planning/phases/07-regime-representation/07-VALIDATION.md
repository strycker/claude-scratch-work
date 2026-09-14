---
phase: 7
slug: regime-representation
# status lifecycle: draft (seeded by plan-phase) → validated (set by validate-phase §6)
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-09-14
---

# Phase 7 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> **Scope: WAVE 1 ONLY** (per `07-CONTEXT.md` D-09). Wave 2 gets its own pass.
>
> Source: `07-RESEARCH.md` §Validation Architecture + §Plausibility Bands.
> Bands marked **[ASSUMED]** below were proposed during research with no prior project
> precedent and are **not yet locked** — confirm before treating as binding.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.0+ |
| **Config file** | root `pyproject.toml` → `[tool.pytest.ini_options]` |
| **Quick run command** | `pytest tests/unit/test_platform_backtest_driver.py tests/unit/test_platform_evaluation_report.py tests/unit/test_platform_evaluation_sojourn_lag.py -x` |
| **Full suite command** | `pytest tests/ -q` |
| **Estimated runtime** | quick ~10–20s · full ~92s (1705 tests, measured 2026-09-14) |
| **New dependencies** | none |

---

## Sampling Rate

- **After every task commit:** quick run command (~10–20s)
- **After every plan wave:** `pytest tests/ -q` (~92s)
- **Phase gate:** at least one real `run_full_backtest_evaluation()` per policy variant —
  **2 variants** (10-feature frozen policy per D-02-A; 13-feature + imputation per D-03).
  Measured walk-forward cost ≈ **2 min** per full 588-step L1+L2 refit run, so full re-runs
  are affordable; do **not** patch persisted artifacts in place.
- **Before `/gsd-verify-work`:** full suite green
- **Max feedback latency:** 92 seconds

---

## Per-Task Verification Map

> Task IDs are filled in by the planner. Rows below are the requirement-level contract the
> planner must satisfy; every criterion needs at least one row with an automated command,
> except the two explicitly listed as manual under **Manual-Only Verifications**.

| Criterion | Req | Behavior | Test Type | Automated Command | File Exists | Status |
|---|---|---|---|---|---|---|
| 1 | REG-01 | Driver and report resolve to **identical column sets** at every sampled decision date; a test fails if they diverge | unit (equivalence) | `pytest tests/unit/test_platform_backtest_driver.py -k equivalence -x` | ❌ Wave 0 — new test | ⬜ pending |
| 2 | REG-01 | §5.4 ratio computed on ONE feature space, reported with its resolved-transition count | unit | `pytest tests/unit/test_platform_evaluation_sojourn_lag.py -x` | ✅ extend (`n_resolved` / `n_transitions` already present, `sojourn_lag.py:185-195`) | ⬜ pending |
| 3 | REG-01 | Post-fix disagreement % computed by the **same methodology** as the 389/470 = 82.8% baseline | unit + real re-run | `pytest tests/unit/test_platform_evaluation_report.py -x` + one real `run_full_backtest_evaluation()` | ⚠️ Partial — **methodology provenance unlocated; Wave 0 discovery task** | ⬜ pending |
| 4 | REG-01 | `wealth_delta` **and** `dd_delta` re-measured, each inside its band | integration (real re-run) | full `run_full_backtest_evaluation()` against real checkpoints | Manual-only real-data verification | ⬜ pending |
| — | REG-01 | Checkpoint recompute (D-02-A) lands and the A13 pinned test is **re-pinned**, not loosened | unit | `pytest tests/unit/test_platform_plotting_regime.py -k a13 -x` | ✅ exists — **will go red until re-pinned** | ⬜ pending |
| — | REG-01 | `platform/` imports nothing from the legacy library (criterion 8) | unit (import guard) | `pytest tests/ -k import_guard -x` | ✅ exists — extend to new modules | ⬜ pending |
| ADR | REG-01 | Policy choice, rejected alternatives, trial-ceiling formula recorded | manual (document review) | N/A | N/A | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] **`tests/unit/test_platform_backtest_driver.py`** — new equivalence test: `_refit_l1`'s
      active column list (with `frozen_features` threaded through) equals
      `_reference_label_columns`'s output at sampled decision dates. This is criterion 1's
      whole foundation — it is the test that must *fail* if driver and report diverge.
- [ ] **Locate the 389/470 = 82.8% methodology.** It is **not** `compute_sojourn_lag_headline`'s
      output (that returns sojourn/lag/ratio/transition counts, not a month-by-month agreement
      percentage). Research could not find it in `evaluation/`, `labeling/`, or library code;
      it likely lives in a Phase 6 notebook cell (`P3_regime_labeling` or
      `P6_backtest_evaluation`), since Phase 6's scope was to *display* A13, not resolve it.
      **Criterion 3 is not satisfiable until this is found or deliberately re-derived** — if
      re-derived, the plan must say so and label the baseline as recomputed, not quoted.
- [ ] **Recompute `monthly_features` from cached `monthly_raw`** (D-02-A) — pure function, no
      network. Re-carve at 2020-12. Sole expected delta: `oil` +277 non-NaN months.
- [ ] Framework install: none required.

---

## Plausibility Bands

Inherited from `06-VALIDATION.md` **unchanged** (wave 1 introduces no new metric *type* for
these): terminal log wealth, max drawdown, Brier, turnover, CVaR, occupancy, portfolio weights.

Wave 1's new or re-measured quantities:

| Quantity | Universal bound | Domain band | Source |
|---|---|---|---|
| Terminal log wealth, any leg | `abs(x) < 10.0` | `[-3, 12]` per leg | [CITED] 06-VALIDATION.md |
| Max drawdown, any leg | `x ∈ [-1, 0]` | buy-hold `< -0.30`; 60/40 `< -0.10`; trend `> -0.50` | [CITED] 06-VALIDATION.md |
| Multiclass Brier, K=5 | `x ∈ [0, 1]` | no-skill floor `(K-1)/K² = 0.16` | [CITED] — expected to move because `y_true` changes, **not** because the nowcaster improved (D-05) |
| Regime occupancy | each `∈[0,1]`, `Σ = 1.0` | soft warning `< 0.05` (`labeling/diagnostics.py:67`) | [CITED] 06-VALIDATION.md |
| **Post-fix disagreement %** | `x ∈ [0, 1]` | **`x < 0.02` must trigger a "confirm the two labelings aren't accidentally sharing state" check** | **[ASSUMED]** — a hindsight full-sample fit vs. a walk-forward per-step fit should still disagree *somewhat* under one feature space. Near-zero is as suspicious as 82.8% was bad; it suggests the driver is seeing full-sample data. |
| **§5.4 ratio (post-fix)** | `ratio > 0`; lag/sojourn each `∈ [0, 588]` | **no numeric target** — D-04 forbids using it to select the policy | [CITED] for bounds; **[ASSUMED]** that the caveat becomes removable, contingent on the two D-08 artifacts landing, never on the ratio's value |
| **`n_transitions`** | integer `≥ 0`; `n_resolved ≤ n_transitions` | implausible if `> 30` over 588 months at K=5, λ=52 | **[ASSUMED]** — by analogy to the 4–6 transitions observed historically; a high count signals over-segmentation / λ not applied |
| **`wealth_delta`** | `abs(x) < 15` (arithmetic from the two log-wealth bounds) | `abs(x) < 5` | **[ASSUMED]** — recorded history spans +0.073 to +0.379 |
| **`dd_delta`** | `x ∈ [-2, 2]` | `abs(x) < 0.5` | **[ASSUMED]** — both legs are vol-targeted, long-only, similarly levered; current value −0.014364 |
| Trial registry rows per evaluation run | exactly `2` (strategy + ablation) | `≠ 2` ⇒ partial failure or an append-pattern change the ADR's trial-ceiling formula must absorb | **[VERIFIED]** this session — ledger inspected and `run_backtest` / `no_regime_ablation` call chain traced |

### The third check class this phase adds

`06-VALIDATION.md` established two classes: **universal physical-possibility** bounds and
**domain-informed per-leg** bounds. Wave 1 needs a third — a **"suspiciously resolved"** band —
because its headline risk is not *an impossible number* but *a number that looks resolved and is
actually a different bug* (driver and reference converging because they read the same in-memory
object rather than by the intended mechanism).

**A plausibility band cannot substitute for the criterion-1 equivalence test.** The bands narrow
what can pass silently; only the test proves the mechanism.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|---|---|---|---|
| Both ablation deltas land inside their bands on real data (criterion 4) | REG-01 | Needs a real ~2-min 588-step walk-forward against live checkpoints; mirrors Phase 5's own 05-07 human-verify pattern | Run `run_full_backtest_evaluation()` per policy variant; read `wealth_delta` / `dd_delta` from the persisted artifacts; check each against the bands above |
| ADR records the choice **and its rejected alternatives** (criterion 1) | REG-01 | Document review — no automated check can judge whether a rejection is evidence-backed | Read the ADR; confirm D-03's 13-feature+imputation variant appears as a *logged trial*, not merely an argued rejection |

---

## Evidence-Shape Requirement (carried from UAT-AUDIT-2026-09-09)

The 2026-09-09 audit found Phases 1 and 5 signed off against arithmetically impossible output —
terminal log wealth 111.06 (e¹¹¹ ≈ 10⁴⁸) tabulated as an *improvement*. Every criterion passed
**as phrased**, because all evidence was of one of four shapes: existence, shape, pure-function
correctness, or placement. None can detect a physically impossible value.

**Contract for this phase:** every verification row above must be able to name **the value it
would reject**. A check that can only confirm a thing ran, exists, or is shaped correctly does
not satisfy a numeric criterion here.

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or a Wave 0 dependency
- [ ] Sampling continuity: no 3 consecutive tasks without an automated verify
- [ ] Wave 0 covers all MISSING references (incl. the 389/470 provenance hunt)
- [ ] No watch-mode flags
- [ ] Feedback latency < 92s
- [ ] `[ASSUMED]` bands confirmed or revised before execution
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
