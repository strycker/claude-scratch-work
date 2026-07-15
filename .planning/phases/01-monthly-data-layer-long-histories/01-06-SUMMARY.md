---
phase: 01-monthly-data-layer-long-histories
plan: 06
subsystem: ingestion
tags: [yfinance, pandas, daily-prices, monthly-spine, platform]

# Dependency graph
requires:
  - phase: 01-01
    provides: "platform package skeleton + config/platform_settings.yaml universe lists (satellites/holdings/watchlist/no_price_ingest)"
provides:
  - "universe_fetch_tickers(cfg) — union(satellites, holdings, watchlist) minus no_price_ingest"
  - "fetch_universe_prices(cfg) — daily adjusted-close price fetch with NULL-tolerant outer-join merge"
  - "to_monthly_spine(daily_df) — month-end derived spine from daily prices"
affects: [platform-l0-data, phase-4-tripwire]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Reuse assets._ssl_bypass_curl_session as a pure importable session factory; never reuse assets._batch_yfinance (hardcodes QE resample)"
    - "pd.concat([...], axis=1) outer-join merge for variable-length ticker histories (never pd.merge/join defaults)"

key-files:
  created:
    - src/trading_crab_lib/platform/ingestion/prices_daily.py
    - tests/unit/test_platform_prices_ingest.py
  modified: []

key-decisions:
  - "Daily persisted unresampled; monthly spine derived separately via resample(monthly_freq).last() — daily granularity is required for the Phase 4 tripwire"
  - "assets.py left byte-identical; only _ssl_bypass_curl_session is imported from it"

patterns-established:
  - "NULL-tolerant universe merge: pd.concat(axis=1) outer join, never pd.merge/join defaults — short-history tickers become NaN-padded columns, not dropped rows"

requirements-completed: [DATA-05]

coverage:
  - id: D1
    description: "universe_fetch_tickers(cfg) returns union(satellites, holdings, watchlist) minus no_price_ingest, always including Glenn's holdings and excluding FZFXX/SPAXX"
    requirement: "DATA-05"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_prices_ingest.py#test_universe_fetch_tickers_excludes_money_market_funds"
        status: pass
      - kind: unit
        ref: "tests/unit/test_platform_prices_ingest.py#test_universe_fetch_tickers_includes_holdings_satellites_watchlist"
        status: pass
    human_judgment: false
  - id: D2
    description: "fetch_universe_prices merges variable-length ticker histories via pd.concat(axis=1) outer join — short histories are NaN-padded, never dropped rows or crashed"
    requirement: "DATA-05"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_prices_ingest.py#test_short_history_ticker_becomes_nan_padded_not_dropped"
        status: pass
      - kind: unit
        ref: "tests/unit/test_platform_prices_ingest.py#test_fetch_universe_prices_no_crash_on_short_history"
        status: pass
      - kind: unit
        ref: "tests/unit/test_platform_prices_ingest.py#test_fetch_universe_prices_all_fail_returns_empty"
        status: pass
      - kind: unit
        ref: "tests/unit/test_platform_prices_ingest.py#test_fetch_universe_prices_no_tickers_returns_empty"
        status: pass
    human_judgment: false
  - id: D3
    description: "to_monthly_spine derives a month-end-indexed monthly spine from daily prices without discarding the daily frame"
    requirement: "DATA-05"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_prices_ingest.py#test_to_monthly_spine_yields_month_end_frequency"
        status: pass
      - kind: unit
        ref: "tests/unit/test_platform_prices_ingest.py#test_to_monthly_spine_empty_input_returns_empty"
        status: pass
    human_judgment: false

# Metrics
duration: ~6h47m (includes a provider-quota interruption between the two commits; see Deviations)
completed: 2026-07-15
status: complete
---

# Phase 1 Plan 06: Daily Universe Price Ingestion Summary

**Daily adjusted-close price fetch for the tradable universe (satellites/holdings/watchlist) with NULL-tolerant `pd.concat(axis=1)` merging and a derived monthly spine, reusing `assets._ssl_bypass_curl_session` without touching `assets.py`**

## Performance

- **Duration:** ~6h47m wall clock (provider-quota kill occurred between Task 1's commit and Task 2's test-writing; a rescue commit preserved the in-progress test file, and this continuation executor completed and verified it)
- **Started:** 2026-07-15T15:25:38Z (Task 1 commit)
- **Completed:** 2026-07-15T22:13:02Z
- **Tasks:** 2/2
- **Files modified:** 2

## Accomplishments
- `universe_fetch_tickers(cfg)` computes `union(satellites, holdings, watchlist) minus no_price_ingest`, always including Glenn's current holdings (D-10) and excluding money-market tickers FZFXX/SPAXX (D-12)
- `_batch_yfinance_daily()` fetches all tickers in one `yf.download(interval="1d")` call with NO internal resample — daily granularity preserved for the Phase 4 tripwire
- `fetch_universe_prices(cfg)` merges every ticker's daily Series with `pd.concat([...], axis=1)` (outer join) — short-history tickers become NaN-padded columns spanning the full date range, never dropped rows, never a crash
- `to_monthly_spine(daily_df)` derives a month-end spine via `.resample(monthly_freq).last()` without discarding the daily frame
- `assets.py` is untouched — only `_ssl_bypass_curl_session` is imported from it, verified via `git diff --stat` against the plan's baseline

## Task Commits

Both tasks were committed atomically across two executor sessions (interrupted by a provider quota limit between them):

1. **Task 1: Daily universe price fetch + monthly spine** - `9203629` (feat) — completed by the prior executor session, verified intact by this continuation
2. **Task 2: Universe price ingestion test** - `60de3b7` (wip rescue, prior session) + `bb5e490` (test fix, this session)

**Plan metadata:** this SUMMARY.md commit (docs: complete plan)

## Files Created/Modified
- `src/trading_crab_lib/platform/ingestion/prices_daily.py` - `universe_fetch_tickers()`, `_batch_yfinance_daily()`, `to_monthly_spine()`, `fetch_universe_prices()` (created by prior session, verified correct)
- `tests/unit/test_platform_prices_ingest.py` - 8 HTTP-mocked tests covering ticker-set exclusion/inclusion, NaN-padded short-history merging, all-fetch-failure and no-tickers-configured degradation, and daily→monthly derivation (rescued mid-write by prior session, one assertion fixed by this session)

## Decisions Made
- None new — followed the plan as specified. The only change made in this session was a test-correctness fix (see Deviations), not a design decision.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed incorrect hardcoded month count in `to_monthly_spine` test**
- **Found during:** Task 2 verification (running the rescued test file for the first time)
- **Issue:** The rescued test `test_to_monthly_spine_yields_month_end_frequency` asserted `len(monthly) == 3` for 90 business days starting 2020-01-02. 90 business days (`freq="B"`) spans January through May 2020 (5 calendar months), not 3 — the hardcoded literal was simply wrong arithmetic, unrelated to the module under test.
- **Fix:** Replaced the hardcoded `3` with a dynamically computed expected count (`idx.to_series().dt.to_period("M").nunique()`), so the assertion is correct regardless of the exact date range chosen.
- **Files modified:** `tests/unit/test_platform_prices_ingest.py`
- **Verification:** `pytest tests/unit/test_platform_prices_ingest.py -q` → 8 passed
- **Committed in:** `bb5e490`

---

**Total deviations:** 1 auto-fixed (1 bug in a test written by the interrupted prior session)
**Impact on plan:** No production code changes were needed — `prices_daily.py` from the prior session was correct as written. The only fix was a test-arithmetic error introduced by the quota-interrupted write. No scope creep.

## Issues Encountered
- The prior executor session was terminated mid-plan by a provider quota limit, immediately after writing (but not finishing verification of) `tests/unit/test_platform_prices_ingest.py`. A separate rescue commit (`60de3b7`) preserved that in-progress work before the worktree could be lost. This session verified the rescued file was actually complete (all 8 tests present, covering every acceptance criterion), found and fixed the one incorrect assertion, and confirmed GREEN.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- `fetch_universe_prices(cfg)` and `to_monthly_spine()` are ready for wiring into a pipeline step / checkpoint (not part of this plan's scope — this plan produced the ingestion module and its unit test only)
- Full incumbent test suite verified GREEN (775 passed, 48 skipped for optional deps — hmmlearn/statsmodels/hdbscan/lightgbm/cssselect not installed), confirming no regression from this plan's changes
- `assets.py` confirmed unmodified — no conflict with the incumbent quarterly pipeline

---
*Phase: 01-monthly-data-layer-long-histories*
*Completed: 2026-07-15*

## Self-Check: PASSED
- FOUND: src/trading_crab_lib/platform/ingestion/prices_daily.py
- FOUND: tests/unit/test_platform_prices_ingest.py
- FOUND commit: 9203629
- FOUND commit: bb5e490
