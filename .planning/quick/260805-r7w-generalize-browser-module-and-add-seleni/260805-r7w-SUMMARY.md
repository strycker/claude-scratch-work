---
phase: quick-260805-r7w
plan: 01
subsystem: ingestion
tags: [playwright, selenium, macrotrends, browser-automation, pandas-read_html]

# Dependency graph
requires:
  - phase: quick-260805-jt2
    provides: "ingestion/browser.py (Stooq-shaped fetch_stooq_csvs, [browser] extra with playwright)"
provides:
  - "fetch_page_html/fetch_urls_as_text as the general browser API, with fetch_stooq_csvs kept as a backwards-compatible wrapper"
  - "Selenium as a second, engine-redundancy browser backend selectable via engine=/TC_BROWSER_ENGINE"
  - "macrotrends browser fallback at both call sites (ingestion/macrotrends.py and platform/ingestion/macro_monthly.py)"
affects: [ingestion, platform-data-layer, packaging]

# Actuals (#2632)
actuals:
  tokens: 20942
  tasks: 3
  commits: 3

# Tech tracking
tech-stack:
  added: ["selenium>=4.15 (optional, [browser] extra)"]
  patterns:
    - "Engine-selection precedence: explicit argument > env var > auto (primary-then-fallback), with explicit-request failures never silently substituting the other engine"
    - "value_col resolved before date_col, with date_col search excluding value_col, to avoid a squashed header (\"...Monthly...\") colliding with a \"month\" substring match intended for the date column"

key-files:
  created: []
  modified:
    - src/trading_crab_lib/ingestion/browser.py
    - tests/unit/test_ingestion_browser.py
    - src/trading_crab_lib/ingestion/macrotrends.py
    - src/trading_crab_lib/platform/ingestion/macro_monthly.py
    - tests/unit/test_macrotrends.py
    - tests/unit/test_platform_macro_ingest.py
    - src/trading_crab_lib/pyproject.toml
    - .planning/STATE.md

key-decisions:
  - "fetch_stooq_csvs kept as a thin wrapper over the new fetch_urls_as_text (expected_prefix=\"date,\") rather than migrated/removed, so prices_daily.py's import needed zero edits"
  - "fetch_urls_as_text stays Playwright-only (no Selenium equivalent of page.expect_download exists) — it does not accept an engine= argument at all, and its unavailable-engine WARNING now says so explicitly"
  - "An explicit engine request (argument or TC_BROWSER_ENGINE) that cannot be honored fails with None + WARNING rather than silently falling through to the other engine — treated as an operator-visible failure, not an auto-recovery opportunity"
  - "BROWSER_WAIT_SELECTOR changed from the plan's assumed 'table.historical_data_table' to 'table' mid-task, based on a residential diagnostic that found the class does not exist on the live page — still passed only with require_selector=False"
  - "Column detection order flipped: value_col resolved first, date_col search then excludes it — discovered via this task's own merged-header regression test, which failed until the fix (a squashed value header containing 'Monthly' also matches the new 'month' date keyword)"

requirements-completed: [QUICK-260805-r7w]

coverage:
  - id: D1
    description: "fetch_page_html renders any URL and returns HTML, with require_selector controlling whether a missing selector returns None or the rendered page anyway"
    requirement: QUICK-260805-r7w
    verification:
      - kind: unit
        ref: "tests/unit/test_ingestion_browser.py::test_fetch_page_html_returns_content_and_closes_browser"
        status: pass
      - kind: unit
        ref: "tests/unit/test_ingestion_browser.py::test_fetch_page_html_required_selector_missing_returns_none"
        status: pass
      - kind: unit
        ref: "tests/unit/test_ingestion_browser.py::test_fetch_page_html_optional_selector_missing_returns_html_anyway"
        status: pass
    human_judgment: false
  - id: D2
    description: "fetch_stooq_csvs unchanged for its existing caller (prices_daily.py); all 16 pre-existing browser tests pass unedited"
    requirement: QUICK-260805-r7w
    verification:
      - kind: unit
        ref: "tests/unit/test_ingestion_browser.py (16 pre-existing tests, appended-only diff)"
        status: pass
      - kind: unit
        ref: "tests/unit/test_platform_prices_ingest.py (27 tests, unedited)"
        status: pass
    human_judgment: false
  - id: D3
    description: "Selenium engine selectable via engine=/TC_BROWSER_ENGINE, Playwright-preferred under auto, explicit-request failures do not silently fall through"
    requirement: QUICK-260805-r7w
    verification:
      - kind: unit
        ref: "tests/unit/test_ingestion_browser.py::test_fetch_page_html_auto_prefers_playwright_when_both_available"
        status: pass
      - kind: unit
        ref: "tests/unit/test_ingestion_browser.py::test_fetch_page_html_explicit_playwright_request_not_honored_does_not_fall_through"
        status: pass
      - kind: unit
        ref: "tests/unit/test_ingestion_browser.py::test_fetch_page_html_explicit_engine_arg_beats_env_var"
        status: pass
    human_judgment: false
  - id: D4
    description: "macrotrends tries plain HTTP first at both call sites and reaches fetch_page_html only when the body yields neither embedded JSON nor a parseable table; both resample rules (QE quarterly, ME monthly) unchanged"
    requirement: QUICK-260805-r7w
    verification:
      - kind: unit
        ref: "tests/unit/test_macrotrends.py::test_scrape_series_interstitial_falls_back_to_browser_render"
        status: pass
      - kind: unit
        ref: "tests/unit/test_platform_macro_ingest.py::test_scrape_macrotrends_monthly_interstitial_falls_back_to_browser_render"
        status: pass
      - kind: unit
        ref: "tests/unit/test_macrotrends.py::test_scrape_series_quarter_end_indexed"
        status: pass
    human_judgment: false
  - id: D5
    description: "Whether a real browser gets past macrotrends' Cloudflare at all, and whether the end-to-end parse of a real series is correct — not testable from this container"
    verification: []
    human_judgment: true
    rationale: "All egress in this container is proxy-reset (Chromium fails even on example.com). A residential-connection diagnostic run mid-task DID confirm browser reachability (HTTP 200, no interstitial) but did NOT confirm the end-to-end parse of a real series — only this task's synthetic fixtures were exercised. The <human-check> in the PLAN.md Task 3 verify block is the only way to close this."

duration: ~90min
completed: 2026-08-05
status: complete
---

# Quick Task 260805-r7w: Generalize browser.py and add Selenium engine, route macrotrends through it

**`ingestion/browser.py` generalized from a Stooq-shaped CSV fetcher into `fetch_page_html`/`fetch_urls_as_text`, Selenium added as an explicit engine-redundancy fallback, and macrotrends now falls back to a rendered page at both call sites when its HTTP response is a Cloudflare interstitial.**

## Performance

- **Duration:** ~90 min
- **Tasks:** 3
- **Files modified:** 7 code files + STATE.md (STATE.md edited but left uncommitted per orchestrator instruction — see below)

## Accomplishments

- `fetch_page_html(url, wait_for_selector=, require_selector=, engine=)` and `fetch_urls_as_text(urls, expected_prefix=)` are now the general API; `fetch_stooq_csvs` is a thin wrapper over `fetch_urls_as_text` and `prices_daily.py`'s import needed zero edits.
- Selenium is a second engine (`selenium_available()`, hardened Chrome options, `WebDriverWait` for selectors, `driver.quit()` in `finally`), selectable via `engine=`/`TC_BROWSER_ENGINE`, Playwright-preferred under `auto`. `fetch_urls_as_text` stays Playwright-only (no Selenium equivalent of `page.expect_download`).
- Both macrotrends modules (`ingestion/macrotrends.py`, `platform/ingestion/macro_monthly.py`) now try plain HTTP first, then fall back to `fetch_page_html(..., wait_for_selector="table", require_selector=False)` only when the HTTP body carries neither embedded JSON nor a parseable table.
- `selenium>=4.15` added to the `[browser]` packaging extra.
- Mid-task diagnostic results (from a residential connection, run by the coordinator) landed and changed the Task 3 design: the assumed `historical_data_table` selector class does not exist on the live page (`BROWSER_WAIT_SELECTOR` changed to `"table"`), and the embedded-JSON regex does not match the rendered page, so the fallback path is expected to land on the `pandas.read_html` branch — which needed a "month" date-keyword extension (plus a value/date collision guard discovered by this task's own regression test) to correctly parse the real table's header phrasing.

## Task Commits

Each task was committed atomically:

1. **Task 1: Generalize browser.py — fetch_page_html + fetch_urls_as_text + viewport, fetch_stooq_csvs as wrapper** - `5a0ca11` (feat)
2. **Task 2: Selenium as a second ENGINE — availability flag, engine selection, hardened driver** - `67cb20e` (feat)
3. **Task 3: Route macrotrends through the browser fallback at both call sites + packaging** - `e0741a4` (feat)

**Plan metadata:** already committed as `9cbc5cc` (docs: plan browser module generalization + selenium engine)

_Note: STATE.md was edited with this task's findings but deliberately NOT committed in any of the three commits above — the orchestrator's top-level constraints for this session say "Do NOT commit .planning/ artifacts ... the orchestrator handles the docs commit," which overrides the plan's default `files_modified` list for Task 3._

## Files Created/Modified

- `src/trading_crab_lib/ingestion/browser.py` - Rewritten: `fetch_page_html`, `fetch_urls_as_text`, `fetch_stooq_csvs` (wrapper), Selenium engine, `_resolve_engine`
- `tests/unit/test_ingestion_browser.py` - 16 pre-existing tests unedited (appended-only diff); 27 new tests (10 Task 1 + 17 Task 2)
- `src/trading_crab_lib/ingestion/macrotrends.py` - `_html_yields_data`, `BROWSER_WAIT_SELECTOR`, browser fallback in `_scrape_series`, value/date column-detection fix
- `src/trading_crab_lib/platform/ingestion/macro_monthly.py` - Same fallback wired into `_scrape_macrotrends_monthly`, same column-detection fix in `_scrape_macrotrends_html_table_monthly`
- `tests/unit/test_macrotrends.py` - 8 pre-existing tests unedited; 7 new tests
- `tests/unit/test_platform_macro_ingest.py` - 11 pre-existing tests unedited; 5 new tests
- `src/trading_crab_lib/pyproject.toml` - `selenium>=4.15` added to `[browser]` extra (both `[project.optional-dependencies]` and `[tool.poetry.group.browser.dependencies]`)
- `.planning/STATE.md` - Blockers/Concerns entry updated with the mid-task diagnostic findings; new Quick Tasks Completed row (edited, not committed — see note above)

## Decisions Made

- `fetch_stooq_csvs` kept as a named, backwards-compatible wrapper rather than migrated away, so `prices_daily.py` needed zero edits (plan requirement).
- `fetch_urls_as_text` deliberately has no `engine=` parameter — Playwright's `page.expect_download` has no Selenium equivalent, so a Selenium-only environment gets `{}` + a WARNING naming both facts (unavailable + Playwright-specific), never a faked download path.
- An explicit engine request (via `engine=` or `TC_BROWSER_ENGINE`) that cannot be honored returns `None` + WARNING rather than silently substituting the other engine — an operator who asked for something specific needs to see that it failed, not get a silent swap.
- `BROWSER_WAIT_SELECTOR` changed from the plan's assumed `"table.historical_data_table"` to `"table"` mid-task, based on the coordinator's residential-connection diagnostic against the live page. Still only ever passed with `require_selector=False`.
- Column detection in both `_scrape_series_html_table` and `_scrape_macrotrends_html_table_monthly` now resolves `value_col` before `date_col`, excluding `value_col` from the date search — a squashed header like "Gold PricesMonthly Closing Price" contains "month" as a substring of "Monthly", which would otherwise misidentify the value column as the date column once "month" was added as a date keyword. This bug was caught by this task's own merged-header regression test failing before the fix, not by inspection.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Value/date column-detection collision from the new "month" keyword**
- **Found during:** Task 3, while writing `test_scrape_series_html_table_handles_merged_header_and_month_column`
- **Issue:** Adding `"month"` as a date-column substring match (per the mid-task diagnostic's finding of a "Month"-titled date column) collided with squashed value headers containing "Monthly" (e.g. "Gold PricesMonthly Closing Price" — "Monthly" contains "month" as a substring), causing the value column to be misidentified as the date column, all dates to fail `pd.to_datetime` parsing, and the resulting Series to be silently empty.
- **Fix:** Reordered detection to resolve `value_col` first (using the pre-existing "value"/"price"/"close" keywords, unaffected by the collision), then search for `date_col` excluding whatever was picked as `value_col`.
- **Files modified:** `src/trading_crab_lib/ingestion/macrotrends.py`, `src/trading_crab_lib/platform/ingestion/macro_monthly.py`
- **Verification:** `test_scrape_series_html_table_handles_merged_header_and_month_column` and `test_scrape_macrotrends_monthly_handles_merged_header_and_month_column` — both constructed with the value column FIRST and date column SECOND specifically to exercise this collision, not the `df.columns[0]` fallback.
- **Committed in:** `e0741a4` (Task 3 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 — bug in code written during this same task, caught before commit by its own test).
**Impact on plan:** No scope creep — the fix is entirely within the column-detection logic this task's diagnostic-driven design change required. Plan-external behavior (resample rules, existing tests, existing call sites) untouched.

## Issues Encountered

- **Mid-task diagnostic arrived and changed Task 3's design** (documented in the orchestrator's message, not a bug in this session's own work): the plan's assumed `historical_data_table` selector class and reliance on the embedded-JSON regex matching the rendered page were both contradicted by a residential-connection run against the live macrotrends page. Resolved by switching the selector to `"table"` and confirming (via the deliberate `mock_fetch_page_html.assert_not_called()` assertions in the JSON-body and table-body tests) that the existing HTTP-first, `_extract_json_data → read_html` chain is unchanged and that the browser fallback lands on the `read_html` branch as the diagnostic predicted.
- Selenium's default `settle_ms` (`_CHALLENGE_SETTLE_MS` = 2000ms) uses real `time.sleep` on the Selenium path (unlike Playwright's mockable `page.wait_for_timeout`), which initially made the new Selenium tests take ~15s wall-clock. Fixed by patching `browser_mod.time.sleep` inside the shared `_patch_selenium` test helper — not a production code issue, purely a test-speed fix.

## User Setup Required

None - no external service configuration required for this task. `selenium>=4.15` is declared behind the optional `[browser]` extra; installing it (plus a Chrome/Chromium binary) is the user's choice, not a requirement of this change.

## Next Phase Readiness

- **Ready:** the wiring is complete and unit-tested end-to-end (mocked). `selenium_available()`, `fetch_page_html(engine=...)`, and the macrotrends browser fallback are all in place behind existing, backwards-compatible entry points.
- **Blocker/concern carried forward (see `.planning/STATE.md` edit, uncommitted):** whether the end-to-end parse of a REAL macrotrends series (not this task's synthetic fixtures) produces correct data is NOT confirmed from this container. The mid-task diagnostic confirmed browser *reachability* (HTTP 200, real content, no interstitial) but not correct end-to-end parsing of the actual live table. The `<human-check>` in `260805-r7w-PLAN.md` Task 3's verify block is the only way to close this — run it on a residential connection.
- The `test_platform_prices_ingest.py` suite (27 tests) and `prices_daily.py`'s import of `fetch_stooq_csvs`/`playwright_available` were confirmed unaffected — Stooq's fallback path is untouched by this task.

## Self-Check: PASSED

- `src/trading_crab_lib/ingestion/browser.py` — FOUND
- `tests/unit/test_ingestion_browser.py` — FOUND
- `src/trading_crab_lib/ingestion/macrotrends.py` — FOUND
- `src/trading_crab_lib/platform/ingestion/macro_monthly.py` — FOUND
- `tests/unit/test_macrotrends.py` — FOUND
- `tests/unit/test_platform_macro_ingest.py` — FOUND
- `src/trading_crab_lib/pyproject.toml` — FOUND
- Commit `5a0ca11` — FOUND in `git log`
- Commit `67cb20e` — FOUND in `git log`
- Commit `e0741a4` — FOUND in `git log`
- `python -m pytest tests/ -q` — 1267 passed, 0 failed

---
*Phase: quick-260805-r7w*
*Completed: 2026-08-05*
