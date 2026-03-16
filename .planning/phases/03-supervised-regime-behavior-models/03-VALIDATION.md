---
phase: 3
slug: supervised-regime-behavior-models
status: complete
nyquist_compliant: true
wave_0_complete: true
created: 2026-03-16
---

# Phase 3 — Validation Strategy

> Per-phase validation contract for supervised regime and behavior models.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x |
| **Config file** | pyproject.toml |
| **Quick run command** | `pytest tests/test_models_regime.py tests/test_models_behavior.py tests/test_models_reporting.py -q` |
| **Full suite command** | `pytest -q` |
| **Estimated runtime** | ~90 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/test_models_regime.py tests/test_models_behavior.py tests/test_models_reporting.py -q`
- **After every plan wave:** Run `pytest tests/test_models_regime.py tests/test_models_behavior.py tests/test_models_reporting.py -q`
- **Before `$gsd-verify-work`:** Full suite (`pytest -q`) must be green
- **Max feedback latency:** 90 seconds

---

## Per-Task Verification Map

| Task ID  | Plan | Wave | Requirement             | Test Type | Automated Command                                   | File Exists | Status  |
|----------|------|------|-------------------------|-----------|-----------------------------------------------------|------------|---------|
| 03-01-01 | 01   | 1    | MODEL-01, MODEL-02, MODEL-03, MODEL-04 | unit      | `pytest tests/unit/test_classifier.py::test_scaffolding -q` | ⬜ (to add) | ⬜ pending |
| 03-02-01 | 02   | 2    | MODEL-01, MODEL-02      | unit      | `pytest tests/unit/test_classifier.py::test_current_and_forward_regime_models -q` | ⬜ (to add) | ⬜ pending |
| 03-02-02 | 02   | 2    | MODEL-04                | unit      | `pytest tests/unit/test_classifier.py::test_model_metrics_and_reporting -q` | ⬜ (to add) | ⬜ pending |
| 03-03-01 | 03   | 3    | MODEL-03, MODEL-04      | unit      | `pytest tests/unit/test_classifier.py::test_etf_and_portfolio_behavior_models -q` | ⬜ (to add) | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [x] `tests/test_models_regime.py` — unit tests for current and forward regime models, CV folds, and probabilities.
- [x] `tests/test_models_behavior.py` — unit tests for behavior labels and per-ETF forward behavior models.
- [x] `tests/test_models_reporting.py` — unit tests for metrics aggregation and reporting helpers.

---

## Manual-Only Verifications

| Behavior                                                | Requirement | Why Manual                                   | Test Instructions |
|---------------------------------------------------------|------------|---------------------------------------------|-------------------|
| Inspecting classification reports and confusion matrices over key horizons | MODEL-04   | Requires judgment about “good enough” performance | Run the model training pipelines/notebooks, then inspect generated reports/plots to ensure performance is acceptable for your use case. |

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references
- [x] No watch-mode flags
- [x] Feedback latency < 90s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** 2026-03-16

