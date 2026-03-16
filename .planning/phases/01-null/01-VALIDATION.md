---
phase: 1
slug: data-constraints-foundations
status: complete
nyquist_compliant: true
wave_0_complete: true
created: 2026-03-16
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x |
| **Config file** | pyproject.toml |
| **Quick run command** | `pytest tests/test_pipelines_ingest_features.py tests/test_constraints_etf_universe.py tests/test_constraints_frequency.py -q` |
| **Full suite command** | `pytest -q` |
| **Estimated runtime** | ~60 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/test_pipelines_ingest_features.py tests/test_constraints_etf_universe.py tests/test_constraints_frequency.py -q`
- **After every plan wave:** Run `pytest tests/test_pipelines_ingest_features.py tests/test_constraints_etf_universe.py tests/test_constraints_frequency.py -q`
- **Before `$gsd-verify-work`:** Full suite (`pytest -q`) must be green
- **Max feedback latency:** 60 seconds

---

## Per-Task Verification Map

| Task ID  | Plan | Wave | Requirement       | Test Type | Automated Command                                                                                  | File Exists | Status  |
|----------|------|------|-------------------|-----------|----------------------------------------------------------------------------------------------------|------------|---------|
| 01-01-01 | 01   | 1    | DATA-01, DATA-02, CONSTR-01, CONSTR-02 | smoke    | `pytest tests/test_pipelines_ingest_features.py::test_step01_ingest_writes_macro_raw_without_network -q` | ✅          | ⬜ pending |
| 01-02-01 | 02   | 2    | DATA-02, DATA-03  | smoke     | `pytest tests/test_pipelines_ingest_features.py::test_step02_features_writes_feature_artifacts_without_network -q` | ✅          | ⬜ pending |
| 01-03-01 | 03   | 3    | CONSTR-01         | unit      | `pytest tests/test_constraints_etf_universe.py -q`                                                 | ✅          | ⬜ pending |
| 01-03-02 | 03   | 3    | CONSTR-02, DATA-03 | unit     | `pytest tests/test_constraints_frequency.py -q`                                                    | ✅          | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [x] `tests/test_constraints_etf_universe.py` — stubs and fixtures for ETF-only constraints (added in Phase 1 Plan 03).
- [x] `tests/test_constraints_frequency.py` — stubs and fixtures for cadence constraints (added in Phase 1 Plan 03).
- [x] `tests/test_pipelines_ingest_features.py` — smoke tests for ingestion and features (added in Phase 1 Plan 03).

---

## Manual-Only Verifications

| Behavior                                      | Requirement | Why Manual                                     | Test Instructions |
|-----------------------------------------------|------------|-----------------------------------------------|-------------------|
| Observing ingestion/feature logs for ETF list and date ranges | CONSTR-01, CONSTR-02 | Log content and readability are subjective     | Run `python pipelines/01_ingest.py` and `python pipelines/02_features.py`, then inspect logs for ETF universe and date range lines. |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 60s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending

