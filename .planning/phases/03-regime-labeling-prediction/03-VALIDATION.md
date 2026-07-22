---
phase: 3
slug: regime-labeling-prediction
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-07-22
---

# Phase 3 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.x (installed, 863 tests green) |
| **Config file** | `pyproject.toml` `[tool.pytest.ini_options]` |
| **Quick run command** | `python3 -m pytest tests/unit/test_platform_*.py -q` |
| **Full suite command** | `timeout 560 python3 -m pytest tests/ -q` |
| **Estimated runtime** | quick ~10 s; full ~50 s |

---

## Sampling Rate

- **After every task commit:** Run `python3 -m pytest tests/unit/test_platform_*.py -q`
- **After every plan wave:** Run `timeout 560 python3 -m pytest tests/ -q`
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 60 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| (filled by planner — one row per task) | | | L1-01..L2-02 | | | unit | `python3 -m pytest tests/unit/test_platform_<module>.py -x -q` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

Existing infrastructure covers all phase requirements — pytest + synthetic-frame
conventions from `tests/unit/test_platform_walkforward.py` and
`tests/integration/test_mini_pipeline.py` are the fixtures to mirror. No framework
install needed.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Labeler run on real 1962+ `monthly_features` checkpoint | L1-01 | Real checkpoint requires FRED_API_KEY live ingestion (Phase 1 pending human item) | After keys configured: run the labeler CLI on the real checkpoint; inspect §4.4 report-only diagnostics for sane occupancy/sojourns |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 60s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
