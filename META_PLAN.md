# META_PLAN.md — Planning Phases for Project Audit & Improvement

This file breaks a large planning/documentation effort into small, independent
phases (P1–P8) that can each be completed in a single Claude Code session
without hitting token limits. Each phase produces specific markdown files
that are committed before the next phase begins.

**Created:** 2026-03-30
**Branch:** `claude/implement-phase-a2-SmHEA`

---

## Context: Audit Findings (already completed)

The following issues were discovered during a codebase audit on 2026-03-30.
They inform the planning phases below.

### CRITICAL: Non-Deterministic Pipeline Results

**Symptom:** Running the pipeline twice with different `--market-code` flags
(or after running pytest) produces different email output.

**Root causes identified:**

1. **`_fill_column()` in `transforms.py:145`** — When `market_code` is present,
   gap-fill uses `df[[col, "market_code"]].dropna()` to find valid rows. Different
   `market_code` sources (grok vs predicted vs clustered) have different NaN patterns,
   so **gap-fill boundaries change**, altering derivative values.

2. **`step5_predict()` uses `.dropna(axis=1, how="any")`** — This drops entire columns
   that have any NaN. Different gap-fill results → different NaN patterns → different
   columns survive → different feature sets fed to RF → different model.

3. **No global numpy/random seed** — While individual sklearn models use `random_state=42`,
   there is no `np.random.seed()` or `random.seed()` at pipeline startup. Any stochastic
   operation not explicitly seeded is non-deterministic.

**Fix strategy:**
- Remove `market_code` from gap-fill valid-row logic (it's a label, not a feature)
- Set global seeds at pipeline startup: `np.random.seed(42)`, `random.seed(42)`
- Add determinism tests that verify identical outputs given identical inputs
- Consider pinning column lists rather than relying on `dropna(axis=1)`

### Pytest Warnings (17 from test_markov.py)

- 17x `RuntimeWarning: invalid value encountered in reduce` — numpy overflow during
  statsmodels MarkovRegression optimization. Harmless but noisy.
- 1x `ValueWarning: A date index has no associated frequency` — statsmodels wants
  `freq` on the DatetimeIndex.

**Fix:** Add `@pytest.mark.filterwarnings` or `pytestmark` to suppress known warnings
in `test_markov.py`. Optionally set `.freq` on test fixture indices.

### Missing `from __future__ import annotations`

`src/trading_crab_lib/transforms.py` is the only module missing this import.

### Print Statements in Library Code

`reporting.py:print_dashboard()` uses `print()` directly (12 instances). Per CLAUDE.md
convention, library code should use `logging`. The `print_dashboard()` function is
user-facing output, so this is semi-intentional, but should be refactored to return
a string and let the caller decide how to display it.

### No Tests for App Package

`src/trading_crab/pipeline.py` (1375 lines) and `src/trading_crab/cli.py` have zero
test coverage. Pipeline step functions are tested indirectly via library module tests,
but the orchestration logic, argument parsing, and step dispatch are untested.

### No Integration Tests

`tests/integration/` and `tests/fixtures/` directories don't exist. There are no
end-to-end tests that run multiple pipeline steps in sequence.

### No Type Checking Setup

No `mypy.ini`, `pyrightconfig.json`, or `[tool.mypy]` in pyproject.toml. Type hints
exist on most public functions but are never validated by CI.

### No Pre-Commit Hooks

No `.pre-commit-config.yaml`. Linting is only enforced in CI (flake8).

### Stale CI Workflows

6 workflow files exist: `publish.yml`, `publish-app.yml`, `publish-lib.yml`,
`python-app.yml`, `python-package.yml`, `python-publish.yml`. Some are likely
duplicates from before the 2-package split.

### Email Output Missing GSD-Style Sections

Current `write_weekly_report_md()` produces:
- `## Current Regime` (regime + confidence)
- `## Recommendations` (BUY/SELL bullets)
- `## Risk & regime transition` (top 3 transition probabilities)
- `## Tactics` (buy_hold / swing / stand_aside — only if tactics_signals.parquet exists)

**Missing from GSD version:**
- `## Diagnostics` section (ratio snapshots by |z|, RRG quadrant counts)
- HTML rendering of markdown in email body
- Inline plot attachments (partially implemented in Phase E email work but not
  wired into the report generation)
- Strongest BUY/SELL ideas showing target % vs current (currently shows delta_pct
  which is allocation change, not price target)

### No Docker Support

No Dockerfile. Would be useful for reproducible pipeline runs and CI.

### No Schema Validation for settings.yaml

Config is loaded as a raw dict with `.get()` defaults scattered across modules.
No centralized validation that required keys exist or have correct types.

---

## Planning Phases

Each phase is a single Claude Code session. Commit output before moving to next.

### P1 — Write REBUILD-FROM-SCRATCH-GUIDE.md

**Scope:** Create a comprehensive guide that documents how to rebuild this project
from scratch, assuming no prior code exists. Covers architecture, module order,
testing strategy, and lessons learned.

**Output:** `REBUILD-FROM-SCRATCH-GUIDE.md` committed to repo.

**Estimated size:** ~500 lines of markdown. Split into 4 sub-sessions to avoid timeouts:

- **P1a** — Sections 1–4: Introduction, two-package architecture, repository layout, build order
- **P1b** — Sections 5–7: Feature pipeline deep-dive, clustering, prediction
- **P1c** — Sections 8–10: Testing strategy, packaging/distribution, lessons learned
- **P1d** — Section 11 (critical invariants) + update META_PLAN.md + commit/push

---

### P2 — Fix Non-Determinism (Critical Bug)

**Scope:** Fix the three root causes of non-deterministic pipeline results.

**Tasks:**
1. Remove `market_code` from `_fill_column()` valid-row logic in `transforms.py`
2. Remove `market_code` from `apply_derivatives()` valid-row logic in `transforms.py`
3. Set global seeds (`np.random.seed(42)`, `random.seed(42)`) in `pipeline.py:main()`
4. Add `random_state` config key to `settings.yaml` under a top-level `pipeline` section
5. Pin the feature column list in step 5 instead of using `dropna(axis=1)` — or at
   minimum log which columns are dropped so the user can see it
6. Add a determinism test: run `engineer_all()` twice on the same input, assert identical output
7. Add `from __future__ import annotations` to `transforms.py`

**Output:** Code changes + tests. Single session.

---

### P3 — Fix Pytest Warnings + Local Dev Setup

**Scope:** Eliminate all pytest warnings and document local setup.

**Tasks:**
1. Suppress known statsmodels warnings in `test_markov.py` via `filterwarnings`
2. Set `.freq` on test fixture DatetimeIndex to eliminate frequency warning
3. Add `hmmlearn` and `statsmodels` to dev requirements so no tests are skipped
4. Add `hdbscan` to dev requirements (currently 11 tests skipped)
5. Update `requirements-dev.txt` with all optional deps needed for full test suite
6. Add a `[tool.pytest.ini_options]` `filterwarnings` section to pyproject.toml
7. Document in README.md how to run tests with zero skips/warnings

**Output:** Config changes + test fixes. Single session.

---

### P4 — Enhanced Email Report (GSD-Style Sections)

**Scope:** Add the missing report sections and HTML rendering.

**Tasks:**
1. Add `## Diagnostics` section to `write_weekly_report_md()`:
   - Ratio snapshots (top by |z-score|) from diagnostics.py
   - RRG quadrant counts from diagnostics.py
2. Wire diagnostics data into step 7 dashboard so it's available for the report
3. Render email body as HTML (markdown → HTML conversion)
4. Attach key plots inline (already partially implemented)
5. Update `config/email.example.yaml` with new `attach_plots` defaults
6. Add tests for the new report sections

**Output:** Code changes + tests. Single session.

---

### P5 — Update NEXT_STEPS.md with New Phases (G–K)

**Scope:** Replace Phase F with more granular phases based on audit findings.

**New phases to define:**
- **Phase G** — Determinism & reproducibility (P2 implementation details)
- **Phase H** — Test hardening (integration tests, pipeline.py tests, mypy)
- **Phase I** — Email & reporting enhancements (P4 implementation details)
- **Phase J** — CI/CD cleanup (deduplicate workflows, add mypy, pre-commit)
- **Phase K** — Migration prep (config independence, Docker, artifact publishing)

Also update: ROADMAP.md status, STATE.md with current test count,
CLAUDE.md with any new conventions discovered.

**Output:** Updated markdown files. Single session.

---

### P6 — CI/CD & Developer Experience

**Scope:** Clean up CI workflows and add developer tooling.

**Tasks:**
1. Audit and deduplicate the 6 GitHub Actions workflow files
2. Add `[tool.mypy]` to pyproject.toml with basic strict settings
3. Create `.pre-commit-config.yaml` (flake8 + mypy + trailing whitespace)
4. Add mypy to CI
5. Consider: is Docker useful? (Answer: yes for reproducible weekly runs,
   but not urgent. Document as future work.)

**Output:** CI config changes. Single session.

---

### P7 — Test Hardening

**Scope:** Fill test coverage gaps.

**Tasks:**
1. Add smoke tests for `trading_crab.pipeline` (test `build_parser()`,
   test `main()` with `--steps 3 --help` or minimal mocked steps)
2. Add smoke test for `trading_crab.cli` entry points
3. Create `tests/integration/` with a mini end-to-end test using synthetic data
4. Add determinism regression tests (run pipeline twice, compare outputs)
5. Reach 100% module coverage (every .py file has a corresponding test file)

**Output:** New test files. Single session.

---

### P8 — REBUILD-FROM-SCRATCH-GUIDE.md Supplements

**Scope:** Additional documentation that supports the rebuild guide.

**Tasks:**
1. Create `LESSONS_LEARNED.md` — pitfalls discovered, what we'd do differently
2. Update README.md — refresh layout tree, add badges, fix any stale references
3. Update CLAUDE.md — add new ADRs for decisions made in P2-P7
4. Answer user's questions about alternative artifact formats (Maven, npm, Docker)
   in a `DISTRIBUTION.md` or section in REBUILD guide

**Output:** Documentation files. Single session.

---

## Execution Order

```
P1 (REBUILD guide)     — can start immediately, no code changes
P2 (determinism fix)   — CRITICAL, do ASAP after P1
P3 (pytest warnings)   — quick win, do alongside or after P2
P5 (update NEXT_STEPS) — do after P2+P3 so new phases reflect fixes
P4 (email enhancements)— after P5 (needs plan finalized)
P6 (CI/CD cleanup)     — independent, any time after P3
P7 (test hardening)    — after P2 (determinism tests) and P4 (email tests)
P8 (documentation)     — last, after all code changes settle
```

**Recommended session order:** P1 → P2 → P3 → P5 → P4 → P6 → P7 → P8

Each session should:
1. Read this META_PLAN.md for context
2. Complete the specified phase
3. Commit and push before ending
4. Update this file's status section (below)

---

## Status Tracker

| Phase | Status | Branch/Commit | Notes |
|-------|--------|---------------|-------|
| P1a | DONE | claude/review-meta-plan-Sqot2 | Sections 1–4 |
| P1b | DONE | claude/review-meta-plan-Sqot2 | Sections 5–7 |
| P1c | DONE | claude/review-meta-plan-Sqot2 | Sections 8–10 |
| P1d | DONE | claude/review-meta-plan-Sqot2 | Section 11 + commit/push |
| P2 | DONE | claude/review-meta-plan-Sqot2 | Critical determinism fix |
| P3 | DONE | claude/review-meta-plan-Sqot2 | Pytest warnings + local setup |
| P4 | DONE | claude/review-meta-plan-Sqot2 | Email enhancements |
| P5 | DONE | claude/p5-doc-updates-1774982009 | Update NEXT_STEPS.md |
| P6 | DONE | claude/p5-doc-updates-1774982009 | CI/CD cleanup |
| P7 | NOT STARTED | | Test hardening |
| P8 | NOT STARTED | | Documentation supplements |
