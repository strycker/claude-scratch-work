---
phase: 4
slug: asset-prediction-allocation
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-07-22
---

# Phase 4 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.x (installed, 913 tests green) |
| **Config file** | `pyproject.toml` `[tool.pytest.ini_options]` |
| **Quick run command** | `python3 -m pytest tests/unit/test_platform_*.py -q` |
| **Full suite command** | `timeout 560 python3 -m pytest tests/ -q` |
| **Estimated runtime** | quick ~15 s; full ~50 s |

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
| (filled by planner — one row per task) | | | L3-01..L4-04 | | | unit | `python3 -m pytest tests/unit/test_platform_<module>.py -x -q` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

Existing pytest infrastructure covers the framework; RESEARCH identifies two data-gap
tasks that behave like Wave 0 prerequisites (SPY ingestion into the universe; daily
BAA/AAA or documented monthly fallback for credit-spread velocity). Synthetic-frame
test conventions from `tests/unit/test_platform_walkforward.py` are the fixtures to
mirror — no framework install needed.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Full weekly report generated from real data | L4-02 | Requires real ingested checkpoints (FRED_API_KEY pending, Phase 1 human item) | After keys configured: run the report CLI end-to-end; read the report; sanity-check trades-implied against a real account YAML |
| Email delivery | L4-02 | Requires SMTP config (config/email.local.yaml) | Run with --send-email against personal SMTP; confirm receipt + rendering |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 60s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
