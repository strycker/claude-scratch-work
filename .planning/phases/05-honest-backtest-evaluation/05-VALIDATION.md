---
phase: 5
slug: honest-backtest-evaluation
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-07-23
---

# Phase 5 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> Source: 05-RESEARCH.md "## Validation Architecture".

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest ≥8.0 (project pin) |
| **Config file** | `pyproject.toml` `[tool.pytest.ini_options]` (existing, no changes) |
| **Quick run command** | `pytest tests/unit/test_platform_backtest_*.py tests/unit/test_platform_evaluation_*.py -x -q` |
| **Full suite command** | `pytest tests/ -v` |
| **Estimated runtime** | ~60–120 s full suite (863+ tests today); walk-forward over a small synthetic frame stays well within budget |

---

## Sampling Rate

- **After every task commit:** Run the relevant `test_platform_backtest_*.py` / `test_platform_evaluation_*.py` file (quick run command).
- **After every plan wave:** Run `pytest tests/ -v` (full suite).
- **Before `/gsd-verify-work`:** Full suite green. Additionally — because this phase's headline artifact IS a real 1972–2020 run (design §14 Phase 1 exit) — a scripted `__main__`/CLI self-check must execute `run_backtest()` against real Phase 1 checkpoint data end-to-end, not only synthetic unit tests.
- **Max feedback latency:** ~120 s.

---

## Per-Task Verification Map

Task IDs are assigned by the planner; rows below are requirement-level and map to Wave 0 test files. Each becomes one or more concrete tasks in PLAN.md.

| Req | Behavior (invariant proved) | Test Type | Automated Command | File Exists |
|-----|-----------------------------|-----------|-------------------|-------------|
| EVAL-01 | Backtest never sees data past holdout cutoff (train/test index ≤ 2020-12-31) | unit invariant | `pytest tests/unit/test_platform_backtest_driver.py::TestHoldoutBoundary -x` | ❌ W0 |
| EVAL-01 | Equity-curve compounding correct (log-wealth = cumsum(log1p(r)) on known sequence) | unit known-answer | `pytest tests/unit/test_platform_backtest_driver.py::TestEquityCurveCompounding -x` | ❌ W0 |
| EVAL-01 | `run_backtest` logs exactly one registry trial per run | unit | `pytest tests/unit/test_platform_backtest_driver.py::TestRegistryLogging -x` | ❌ W0 |
| EVAL-01 | Per-step L1/L2 refit uses train_index-only data (no reuse of Phase 3/4 checkpoint) | unit invariant | `pytest tests/unit/test_platform_backtest_driver.py::TestRefitFromTrainWindowOnly -x` | ❌ W0 |
| EVAL-01/D-03 | cost = turnover × bps identity holds exactly for a hand-built weight sequence | unit known-answer | `pytest tests/unit/test_platform_backtest_costs.py::TestCostIdentity -x` | ❌ W0 |
| EVAL-02/D-02 | No-regime ablation (tilt-off, constant-label) reproduces the vol-target baseline exactly | unit invariant | `pytest tests/unit/test_platform_backtest_baselines.py::TestNoRegimeAblationInvariant -x` | ❌ W0 |
| EVAL-02 | Faber SMA never uses same-month price to set same-month position (1-step lag) | unit leakage invariant | `pytest tests/unit/test_platform_backtest_baselines.py::TestFaberNoLookahead -x` | ❌ W0 |
| EVAL-03 | sojourn/lag orchestration wires `occupancy_and_sojourns` → `gap_lag.sojourn_lag_ratio` correctly | unit known-answer | `pytest tests/unit/test_platform_evaluation_sojourn_lag.py -x` | ❌ W0 |
| EVAL-04 | Multiclass Brier ~0 for perfect one-hot, >0 for miscalibrated (known-answer) | unit | `pytest tests/unit/test_platform_evaluation_model_metrics.py::TestBrierKnownAnswer -x` | ❌ W0 |
| EVAL-04 | Confusion-table counts sum to n | unit invariant | `pytest tests/unit/test_platform_evaluation_model_metrics.py::TestConfusionSumsToN -x` | ❌ W0 |
| KPIs | Crisis-window capture ratios bounded to in-sample crises (≤ 2020-12; never 2020/2022) | unit | `pytest tests/unit/test_platform_evaluation_kpis.py -x` | ❌ W0 |
| all | End-to-end: full backtest + 4 baselines + report on a small synthetic monthly frame, no network | integration | `pytest tests/integration/test_mini_backtest.py -x` | ❌ W0 |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/unit/test_platform_backtest_driver.py` — EVAL-01 (holdout boundary, equity-curve compounding, registry logging, refit-from-train-window-only)
- [ ] `tests/unit/test_platform_backtest_baselines.py` — EVAL-02 (SPY, 60/40, Faber no-lookahead, no-regime-ablation invariant)
- [ ] `tests/unit/test_platform_backtest_costs.py` — D-03 (turnover, cost identity)
- [ ] `tests/unit/test_platform_evaluation_sojourn_lag.py` — EVAL-03 (orchestration vs known synthetic construction)
- [ ] `tests/unit/test_platform_evaluation_model_metrics.py` — EVAL-04 (Brier known-answer, calibration bin edges, confusion sums)
- [ ] `tests/unit/test_platform_evaluation_kpis.py` — strategy KPIs (terminal log wealth, max drawdown+duration, CVaR(5%), crisis capture window-bounding)
- [ ] `tests/unit/test_platform_evaluation_report.py` — report assembly (headline ordering: sojourn/lag ratio first, per D-01a)
- [ ] `tests/integration/test_mini_backtest.py` — synthetic end-to-end (mirrors `test_mini_pipeline.py`), no network, no real checkpoints
- [ ] Framework install: none — pytest already installed and configured project-wide

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Real 1972–2020 backtest runs end-to-end on live Phase 1 checkpoint data and produces all documented artifacts | EVAL-01 (design §14 Phase 1 exit) | Requires real spliced monthly data + FRED_API_KEY (verified present); the automated suite is synthetic-only by the no-network invariant | Run the backtest CLI/`__main__` self-check against real checkpoints; confirm equity curves + baselines + sojourn/lag ratio + metrics artifacts are written and the index stops at ≤ 2020-12 |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 120s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
