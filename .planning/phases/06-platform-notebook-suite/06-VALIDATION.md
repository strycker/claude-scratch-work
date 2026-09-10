---
phase: 6
slug: platform-notebook-suite
# status lifecycle: draft (seeded by plan-phase) → validated (set by validate-phase §6)
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-09-09
---

# Phase 6 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> Derived from `06-RESEARCH.md` § Validation Architecture.

---

## The standing lesson this strategy exists to answer

Every criterion across Phases 1–5 was satisfiable by evidence of one of four shapes —
**existence, shape, pure-function correctness, or placement** — and none of them could fail
on a value that is physically impossible. Two phases were signed off against arithmetically
impossible output:

| Historical failure | Caught by drift (D-09)? | Caught by plausibility (D-11)? |
|---|---|---|
| 60/40 baseline at **−2.27% max drawdown** across 1973-74/1980-82/2000-02/2008-09 | **No** — the percent-vs-decimal defect was uniform across the whole span, so nothing drifted relative to itself | **Yes**, but only via a *domain-informed per-leg* band. The universal `∈[-1,0]` bound does **not** catch −2.27% |
| Terminal log wealth **111.06** (e¹¹¹ ≈ 10⁴⁸) | **No**, same reason | **Yes** — the universal `abs(x) < 10` band already coded at `tests/integration/test_mini_backtest.py:258` |

**Neither historical failure was a drift failure.** This is why D-11 was reversed and why
D-09 alone was insufficient. Both check classes are required in every notebook that displays
a number; they catch disjoint failure modes.

A third failure mode exists that **neither** catches: a valid-looking number computed against
a mismatched target. Audit item **A13** is exactly this — the 164-month detection lag and
0.591 ratio are inside every plausible band and still not trustworthy. No numeric band around
that ratio resolves A13, and no plan task may claim otherwise.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.0+ |
| **Config file** | root `pyproject.toml` → `[tool.pytest.ini_options]` |
| **Quick run command** | `pytest tests/unit/test_platform_plotting*.py tests/unit/test_platform_notebooks.py -x` |
| **Full suite command** | `pytest tests/ -q` |
| **Estimated runtime** | quick ~10s · full ~180s (1393 tests currently collected) |
| **New dependencies** | **None.** matplotlib, seaborn, nbformat all present in `.venv`; jupytext/nbmake/papermill deliberately absent (D-17 rejects execution-based CI) |

---

## Sampling Rate

- **After every task commit:** `pytest tests/unit/test_platform_plotting*.py tests/unit/test_platform_notebooks.py -x`
- **After every plan wave:** `pytest tests/ -q` (full suite)
- **Before `/gsd-verify-work`:** full suite green
- **Max feedback latency:** ~10s quick / ~180s full

---

## Plausibility bands — the numeric contract

Every band below is a **stated numeric criterion**, not an existence check. A task that
displays one of these quantities must assert its band.

| Quantity | Universal bound | Domain band | Current reference (2026-09-09) |
|---|---|---|---|
| Terminal log wealth (588-mo span) | `abs(x) < 10.0` | `[-3, 12]` per leg | strategy 4.0265 · SPY 5.6805 · Faber 6.3726 · 60/40 5.0471 · ablation 3.6472 |
| Max drawdown, any leg | `x ∈ [-1, 0]` | buy-and-hold leg `x < -0.30`; blended 60/40 `x < -0.10`; trend-following (Faber) `x > -0.50` | strategy −21.24% (33 mo) · SPY −48.95% · Faber −18.94% · 60/40 −26.96% |
| Multiclass Brier, K=5 | `x ∈ [0, 1]` — **not [0,2]**; this codebase computes `mean(diff²)` over the full (n,K) array | no-skill floor `(K−1)/K² = 0.16` | **0.2087 — ABOVE the no-skill floor.** Amber, not green: this metric cannot presently claim the nowcaster beats random guessing (echoes audit item A8) |
| Monthly turnover | `x ∈ [0, 2]` | mean `< 0.30` for a hysteresis-gated book | 0.0734 |
| CVaR(5%), monthly | `x ∈ [-0.5, 0]` | `[-0.15, -0.01]` for a vol-targeted book | −0.0463 |
| Regime occupancy | each `∈[0,1]`, `Σ = 1.0` exactly | soft warning below 0.05 (already coded: `_MIN_OCCUPANCY_THRESHOLD`, `labeling/diagnostics.py:67`) | 1.6 / 14.0 / 31.9 / 40.6 / 11.9 |
| Detection lag / sojourn (months) | `∈ [0, 588]` | **none — NOT INTERPRETABLE pending A13** | lag 164 · sojourn 97 · ratio 0.591 · 4 of 6 transitions resolved |
| Portfolio / tilt weights | each `∈[0,1]`, `Σw_assets + w_cash = 1.0` exactly | long-only by design (no shorts/options per PROJECT.md) | — |

**Existing plausibility patterns to extend, not reinvent:** `assert_yield_units_plausible`
(`platform/splice.py:124-152`) and `TestKpisAreOnAPlausibleScale`
(`tests/integration/test_mini_backtest.py:218-258`).

---

## Per-Task Verification Map

> Task IDs are filled in by the planner. The rows below fix the *shape* every task must satisfy.

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| TBD | 01 | 0 | NB-01 | unit | `pytest tests/unit/test_platform_plotting.py -x` | ❌ W0 | ⬜ pending |
| TBD | 01 | 0 | NB-01 | unit | `pytest tests/unit/test_platform_plotting_drift.py -x` | ❌ W0 | ⬜ pending |
| TBD | 01 | 0 | NB-01 | unit (static) | `pytest tests/unit/test_platform_notebooks.py -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/unit/test_platform_plotting.py` — every `platform/plotting/` function: does-not-crash
      + empty-input edge cases, following the existing `tests/unit/test_plotting.py` pattern
- [ ] `tests/unit/test_platform_plotting_drift.py` — D-09 drift statistic and D-11 plausibility
      bands as **pure functions**, each band tested with an in-band value AND an out-of-band
      value that must raise/flag. At minimum this file must contain the two regression cases:
      a 60/40-shaped leg at −2.27% max DD must FAIL its domain band, and a terminal log wealth
      of 111.06 must FAIL the universal band.
- [ ] `tests/unit/test_platform_notebooks.py` — static notebook checks (D-17, no execution):
      valid JSON via `nbformat`, resolvable imports, and **criterion 3 enforcement** — no
      `matplotlib`/`seaborn` import appears in any notebook cell (all plotting must route
      through `platform/plotting/`)
- [ ] Framework install: **none required**

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| All six notebooks run top-to-bottom against real checkpoints without error | NB-01 / criterion 1 | Execution-based CI rejected by D-17; needs real artifacts | Open each of P1–P6 in `notebooks/platform/`, Run All, confirm no exception. P1/P2/P6 are executable in this environment today; P3/P4/P5 require `label_regimes()` to be called first (cheap — a single fit, not a walk-forward) |
| P3 cold-start sign-off recorded | NB-01 / criterion 2 | The verdict is inherently a human judgement | Operator fills the P3 sign-off markdown cell with date, verdict, and reasoning. Per D-16 a **negative** verdict is recorded and does not block the phase |
| P3 regimes judged against dated economic history | NB-01 / criterion 4 | Requires human economic judgement | Confirm the overlay shows 1973-09 oil shock, 1981-10 Volcker, 1988-09 disinflation, 1996-08 late-90s expansion, 2008-07 → crisis state (GFC), 2009-06 recovery. Per Amendment 2 item E, expect a **genuine question, not a foregone negative** |

**Degradation that must be loud, not silent:** if the USREC network fetch (D-12) fails, the
recession-shading overlay degrades to the event-list only. Per D-10 this must warn loudly and
actionably — never silently render an unshaded chart that looks complete.

---

## Resolved during plan-phase (research open questions)

**OQ2 — does the platform have a centered-feature variant for P2's causal-vs-centered panel?**
**Resolved: NO, and one is architecturally forbidden.** `transforms_monthly.py` contains zero
occurrences of `center`/`centered`/`causal`, and `honesty/gating.py` defines
`FORBIDDEN_CENTERED_SUFFIXES = ("_centered", "_c5", "_zerophase")` and refuses loudly on sight.
P2 must therefore **not** attempt a causal-vs-centered overlay. Its prose framing instead:
*platform features are causal-only by construction and a centered variant is actively rejected
by the gating rail; the only hindsight in the system is the L1 labeler's deliberate non-causal
batch DP-decode, not the features.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] Every task that displays a number asserts a stated band from the table above
- [ ] The two regression cases (60/40 −2.27%, log wealth 111.06) are covered and fail correctly
- [ ] No watch-mode flags
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
