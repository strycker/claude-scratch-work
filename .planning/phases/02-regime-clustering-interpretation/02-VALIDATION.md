---
phase: 2
slug: regime-clustering-interpretation
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-16
---

# Phase 2 — Validation Strategy

> Per-phase validation contract for clustering, profiling, and naming behavior.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x |
| **Config file** | pyproject.toml |
| **Quick run command** | `pytest tests/unit/test_clustering.py tests/unit/test_regime.py -q` |
| **Full suite command** | `pytest -q` |
| **Estimated runtime** | ~60 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/unit/test_clustering.py tests/unit/test_regime.py -q`
- **After every plan wave:** Run `pytest tests/unit/test_clustering.py tests/unit/test_regime.py -q`
- **Before `$gsd-verify-work`:** Full suite (`pytest -q`) must be green
- **Max feedback latency:** 60 seconds

---

## Per-Task Verification Map

| Task ID  | Plan | Wave | Requirement     | Test Type | Automated Command                                              | File Exists | Status  |
|----------|------|------|-----------------|-----------|----------------------------------------------------------------|------------|---------|
| 02-01-01 | 01   | 1    | REGIME-01       | unit      | `pytest tests/unit/test_clustering.py -q`                      | ⬜ (to add) | ⬜ pending |
| 02-02-01 | 02   | 2    | REGIME-02, REGIME-03 | unit  | `pytest tests/unit/test_regime.py -q`                          | ⬜ (to add) | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/unit/test_clustering.py` — unit tests for PCA + KMeans clustering, artifact creation under `data/regimes/`.
- [ ] `tests/unit/test_regime.py` — unit tests for regime profiles, naming, and transition matrices driven by `regime_labels.yaml`.

---

## Manual-Only Verifications

| Behavior                                      | Requirement | Why Manual                                     | Test Instructions |
|-----------------------------------------------|------------|-----------------------------------------------|-------------------|
| Visual inspection of regime profiles and clustering stability across reruns | REGIME-01, REGIME-02 | Requires notebook/plot-based inspection        | Run `python pipelines/03_cluster.py` and `python pipelines/04_regime_label.py`, then inspect relevant notebooks/plots. |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 60s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending

