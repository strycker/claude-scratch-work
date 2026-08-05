---
phase: quick-260805-jt2
plan: 01
subsystem: ingestion
tags: [playwright, stooq, headless-chromium, bot-mitigation, data-ingestion]

requires:
  - phase: quick-260805-570
    provides: "ingestion/http.py browser-impersonating client (curl_cffi), later proven insufficient against Stooq's JS challenge"
provides:
  - "ingestion/browser.py: headless-Chromium Stooq CSV fetcher, optional dependency, never raises"
  - "prices_daily._parse_stooq_csv: single shared CSV-header guard used by both Stooq fetch paths"
  - "prices_daily._batch_stooq_daily: two-path fetch (HTTP then browser fallback), gated on zero-recovery + playwright availability"
  - "[browser] packaging extra (playwright>=1.40) declared in library pyproject, mirrored as Poetry group, included in [all]"
affects: [phase-6-platform-notebook-suite, data-ingestion, live-price-fetch]

actuals:
  tokens: 9000
  tasks: 3
  commits: 3

tech-stack:
  added: ["playwright>=1.40 (optional [browser] extra)"]
  patterns:
    - "Optional-dependency module pattern mirrored from hmm.py: _X_AVAILABLE flag + bound-None fallback name so tests can patch the symbol even when the real package is absent"
    - "Single shared parse/guard helper reached by two independent fetch paths (HTTP first, browser fallback second) so a security-relevant guard cannot drift out of sync between call sites"

key-files:
  created:
    - src/trading_crab_lib/ingestion/browser.py
    - tests/unit/test_ingestion_browser.py
  modified:
    - src/trading_crab_lib/platform/ingestion/prices_daily.py
    - tests/unit/test_platform_prices_ingest.py
    - src/trading_crab_lib/pyproject.toml
    - .planning/STATE.md

key-decisions:
  - "Browser fallback launches exactly one headless Chromium per _batch_stooq_daily call (not per ticker), navigates the JS warm-up page once, then issues context.request.get per ticker through the browser's own context — never a harvested-cookie replay through curl_cffi, which would reintroduce the TLS-fingerprint/identity mismatch this module exists to remove."
  - "CSV-header guard (startswith(\"date,\")) factored into one _parse_stooq_csv helper reached by both the HTTP path and the browser path, verified by a grep gate requiring exactly one literal occurrence in prices_daily.py."
  - "Browser fallback is strictly gated on (HTTP path recovered zero tickers) AND (playwright_available()) — never launched while the cheap HTTP path still works, and never attempted at all when playwright isn't installed."
  - "TC_CHROMIUM_PATH env var is the operator escape hatch for the ambient/expected-build Chromium mismatch discovered in this container; when unset, the executable_path kwarg is omitted entirely (not passed as None) so Playwright's own resolution applies."

requirements-completed: [QUICK-260805-jt2]

coverage:
  - id: D1
    description: "Headless-Chromium Stooq CSV fetcher (ingestion/browser.py): one browser launch, one warm-up navigation, N per-ticker requests, never raises"
    requirement: "QUICK-260805-jt2"
    verification:
      - kind: unit
        ref: "tests/unit/test_ingestion_browser.py (8 tests, playwright mocked entirely)"
        status: pass
    human_judgment: false
  - id: D2
    description: "Single shared CSV-header guard (_parse_stooq_csv) reached by both the HTTP path and the browser fallback path; fallback gated on zero-recovery + playwright availability"
    requirement: "QUICK-260805-jt2"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_prices_ingest.py (9 new tests + 5 pre-existing Stooq tests, all passing)"
        status: pass
    human_judgment: false
  - id: D3
    description: "[browser] packaging extra declared (playwright>=1.40), included in [all], mirrored as Poetry group; scripts/diagnose_stooq.py retired"
    requirement: "QUICK-260805-jt2"
    verification:
      - kind: unit
        ref: "python -c \"import tomllib...\" pyproject parse assertion (Task 3 verify block)"
        status: pass
    human_judgment: false
  - id: D4
    description: "Whether the headless-Chromium path actually defeats the live Stooq JS challenge"
    verification: []
    human_judgment: true
    rationale: "Cannot be verified from this container — all egress is reset by the agent proxy (Chromium launches, but even https://example.com returns net::ERR_CONNECTION_RESET). Requires the <human-check> in the plan's Task 3, run on a residential connection."

duration: 35min
completed: 2026-08-05
status: complete
---

# Quick Task 260805-jt2: Restore Stooq via Headless-Chromium Challenge Solver Summary

**Headless-Chromium fallback (`ingestion/browser.py`) added behind Stooq's TLS-impersonating HTTP path, wired through one shared CSV-header guard, gated on zero-recovery + playwright availability — wiring-verified only, live challenge-defeat unconfirmed.**

## Performance

- **Duration:** 35 min
- **Tasks:** 3
- **Files modified:** 6 (2 created, 4 modified)

## Accomplishments

- New optional module `ingestion/browser.py`: `fetch_stooq_csvs()` launches one headless Chromium, navigates a warm-up page once so the JS challenge executes and commits its cookie, then issues one `context.request.get` per ticker CSV URL. Degrades to `{}` with a WARNING on any failure (missing playwright, launch failure, navigation failure, or per-ticker fetch failure) — never raises.
- `prices_daily._batch_stooq_daily` restructured into two paths through a single new `_parse_stooq_csv` helper: the existing HTTP path (tried first, unchanged in spirit) and the new browser fallback, attempted only when the HTTP path recovers zero tickers AND `playwright_available()` is True.
- `[browser]` packaging extra (`playwright>=1.40`) declared in `src/trading_crab_lib/pyproject.toml`, added to `[all]`, mirrored as a Poetry optional group.
- `scripts/diagnose_stooq.py` deleted — its question (TLS fingerprint vs. JS challenge) is answered, and the answer is now recorded in `ingestion/browser.py`'s module docstring. `scripts/diagnose_yfinance.py` untouched.

## Task Commits

Each task was committed atomically:

1. **Task 1: Headless-Chromium Stooq CSV fetcher** - `ab5ecce` (feat)
2. **Task 2: One shared parse-and-guard helper, used by both Stooq fetch paths** - `4824a7e` (feat)
3. **Task 3: Declare the [browser] extra, retire the answered diagnostic** - `1dc0343` (chore)

_Note: `.planning/STATE.md` was edited but intentionally NOT committed as part of Task 3 — the orchestrator handles the docs commit._

## Files Created/Modified

- `src/trading_crab_lib/ingestion/browser.py` - Headless-Chromium Stooq CSV fetcher (`fetch_stooq_csvs`, `playwright_available`)
- `tests/unit/test_ingestion_browser.py` - 8 tests, playwright mocked entirely
- `src/trading_crab_lib/platform/ingestion/prices_daily.py` - `_parse_stooq_csv` factored out; `_batch_stooq_daily` two-path restructure
- `tests/unit/test_platform_prices_ingest.py` - 9 new tests + 1 existing test repointed (added `playwright_available` patch so it no longer risks launching a real browser now that the fallback wiring actually exists)
- `src/trading_crab_lib/pyproject.toml` - `[browser]` extra + Poetry group
- `scripts/diagnose_stooq.py` - deleted (question answered)
- `.planning/STATE.md` - Blockers/Concerns bullet updated (not committed by this task)

## Decisions Made

- Used the browser CONTEXT's own `request.get` API for per-ticker fetches rather than harvesting the warm-up cookie and replaying it through curl_cffi — a replayed cookie can be bound to the browser's own TLS fingerprint/identity, which would reintroduce exactly the mismatch this module exists to remove (plan constraint, preserved verbatim).
- `TC_CHROMIUM_PATH` kwarg is omitted entirely (not passed as `None`) when unset, so Playwright's own browser resolution applies rather than being short-circuited by an explicit `None`.
- Kept a `recovered_via` tracking variable in `_batch_stooq_daily`'s closing INFO log (rather than inferring path-taken from `playwright_available()` alone) so the log line names which path actually recovered the tickers, not just which path was attempted.

## Deviations from Plan

**1. [Test honesty — Rule 1 adjacent] Repointed `test_batch_stooq_daily_skips_js_challenge_page` to patch `playwright_available` False**
- **Found during:** Task 2, while verifying the full suite would run safely (per orchestrator note #4).
- **Issue:** Playwright is genuinely installed in this session's venv (1.62.0). Before this plan, `_batch_stooq_daily` had no browser fallback at all, so the pre-existing JS-challenge test never touched playwright. After Task 2's restructure, `playwright_available()` would return real `True` in that test, and a challenge-page HTTP response would trigger the (now real) browser fallback — attempting an actual Chromium launch inside a unit test, violating "no browser, no network in any test."
- **Fix:** Added `@patch("...prices_daily.playwright_available")` returning `False` to that one existing test, with a comment explaining it's specifically about the HTTP-path guard, not the fallback (which is covered by new, dedicated tests). Assertion (`out == {}`) is unchanged.
- **Files modified:** `tests/unit/test_platform_prices_ingest.py`
- **Verification:** Confirmed via git-stash round-trip that the full 32-test pair (`test_platform_prices_ingest.py` + `test_ingestion_browser.py`) passes after, and that 9 of those tests fail against pre-change code (referencing symbols that don't yet exist) — genuine RED→GREEN, not a tautology.
- **Committed in:** `4824a7e` (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (test isolation fix, adjacent to Rule 1 — a genuine environment interaction discovered mid-execution, not a plan defect)
**Impact on plan:** Necessary to keep the test suite honest and network-free. No scope creep — the plan's own orchestrator notes anticipated this class of issue.

## Issues Encountered

None beyond the deviation above. All three tasks' automated `<verify>` blocks passed as specified; the plan's `<human-check>` in Task 3 is explicitly NOT runnable from this container (agent-proxy egress reset) and is left for the user.

## Test Honesty Verification

- Task 1 (`browser.py`): confirmed the 8 new tests fail with `ImportError: cannot import name 'browser'` against pre-change code (git stash of `browser.py` only), then pass after restoring it.
- Task 2 (`prices_daily.py` refactor): confirmed 9 of the new/edited tests fail against pre-change code (`AttributeError: ... does not have the attribute '_parse_stooq_csv'`, etc.), then all 32 tests in the pair pass after restoring the change.
- Full suite: `python -m pytest tests/ -q` → **1197 passed**, 0 failed, 9 warnings (pre-existing, unrelated deprecation/pending-deprecation warnings from pytest/seaborn).

## Honesty Caveat (do not weaken)

**The headless-Chromium Stooq path is wiring-verified only — it is NOT confirmed to defeat the live Stooq JS challenge.** All egress from this container is reset by the agent proxy; Chromium launches here but even `https://example.com` returns `net::ERR_CONNECTION_RESET`. This is recorded in `.planning/STATE.md`'s Blockers/Concerns section and in the plan's Task 3 `<human-check>` — run on a residential machine to find out whether it actually works. Nothing in this summary, the commits, or the code claims otherwise.

## Next Phase Readiness

- Stooq now has a second-tier fallback path available; whether it defeats the live challenge is still an open question pending the human-check.
- No blockers introduced for Phase 6 (Platform Notebook Suite) — this quick task did not touch notebook or platform-modeling code.

---
*Phase: quick-260805-jt2*
*Completed: 2026-08-05*
