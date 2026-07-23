---
phase: 04-asset-prediction-allocation
plan: 05
subsystem: reporting
tags: [yaml, pandas, email, platform, holdings, weekly-report, tdd]

# Dependency graph
requires:
  - phase: 04-asset-prediction-allocation (04-02)
    provides: returns_by_regime_stats() / report_returns_by_regime() — per-asset signals input
  - phase: 04-asset-prediction-allocation (04-03)
    provides: vol_targeted_tilt(), hysteresis load/update/save — allocation-cycle orchestration
  - phase: 03-regime-labeling-prediction
    provides: fit_nowcaster/predict_proba (regime distribution), empirical_transition_matrix (trajectory)
provides:
  - "load_account_weights(): per-account holdings YAML loader, warn-don't-fail, cash-aware, no-silent-normalize"
  - "trades_implied(): per-asset BUY/SELL/HOLD signal vs a flat no-trade band"
  - "assemble_weekly_report()/write_weekly_report(): the L4-02 weekly report markdown, always written"
  - "main(--send-email): opt-in email delivery via the incumbent email.py, read-only reuse"
affects: [phase-5-honest-backtest, phase-6-migration]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Warn-don't-fail YAML config loader (holdings.py mirrors platform/config.py's collect-then-warn style)"
    - "load->update->tilt->save allocation-cycle orchestration isolated in _build_report_inputs() for testability"
    - "Read-only reuse of incumbent email.py — no fork, no modification"

key-files:
  created:
    - src/trading_crab_lib/platform/report/__init__.py
    - src/trading_crab_lib/platform/report/holdings.py
    - src/trading_crab_lib/platform/report/weekly.py
    - config/accounts/example.yaml
    - tests/unit/test_platform_holdings.py
    - tests/unit/test_platform_report_weekly.py
  modified:
    - .gitignore

key-decisions:
  - "load_account_weights() is a brand-new loader — never reuses incumbent load_portfolio() (silent-normalize, no cash concept conflicts with D-01)"
  - "Out-of-tolerance weights+cash sum logs a WARNING but always returns RAW (un-normalized) values (D-01, T-04-14)"
  - "_build_report_inputs() isolates the load->update->tilt->save allocation cycle from main() so tests can monkeypatch it without real Phase 1/3 checkpoint data on disk"
  - "email delivery is strictly opt-in behind --send-email; main([]) never imports/calls send_weekly_email (D-02)"

patterns-established:
  - "Platform report modules stay pure-function-first (assemble_weekly_report has no I/O beyond per-account holdings reads); side effects (write, email) live in separate callers"

requirements-completed: [L4-02, L4-03]

coverage:
  - id: D1
    description: "Per-account holdings YAML loader (load_account_weights) — warn-don't-fail, cash-aware, never silently normalizes"
    requirement: "L4-03"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_holdings.py#TestWellFormedAccount::test_loads_silently_with_no_warning"
        status: pass
      - kind: unit
        ref: "tests/unit/test_platform_holdings.py#TestOutOfToleranceSum::test_warns_but_returns_raw_unnormalized_weights"
        status: pass
      - kind: unit
        ref: "tests/unit/test_platform_holdings.py#TestMissingFile::test_returns_neutral_dict_with_warning_not_crash"
        status: pass
      - kind: unit
        ref: "tests/unit/test_platform_holdings.py#TestNonNumericWeight::test_non_numeric_weight_skipped_with_warning_not_crash"
        status: pass
    human_judgment: false
  - id: D2
    description: "Real account files gitignored; committed example.yaml documents the schema with obviously-fake data"
    requirement: "L4-03"
    verification:
      - kind: other
        ref: "git check-ignore config/accounts/real_account.yaml (exit 0) && ! git check-ignore config/accounts/example.yaml"
        status: pass
    human_judgment: false
  - id: D3
    description: "trades_implied(): explicit per-asset BUY/SELL/HOLD signal against a flat no-trade band, unioned assets"
    requirement: "L4-02"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_report_weekly.py#TestTradesImplied"
        status: pass
    human_judgment: false
  - id: D4
    description: "assemble_weekly_report(): regime distribution + trajectory + per-asset signals (D11 low-confidence flag) + per-account trades implied"
    requirement: "L4-02"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_report_weekly.py#TestAssembleWeeklyReport"
        status: pass
    human_judgment: false
  - id: D5
    description: "write_weekly_report() ALWAYS writes outputs/reports/platform/weekly_report.md"
    requirement: "L4-02"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_report_weekly.py#TestWriteWeeklyReport"
        status: pass
    human_judgment: false
  - id: D6
    description: "main() delivers email ONLY under --send-email, reusing incumbent email.py read-only"
    requirement: "L4-02"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_report_weekly.py#TestMainOptInEmail"
        status: pass
      - kind: unit
        ref: "tests/unit/test_platform_report_weekly.py#TestReuseConventions"
        status: pass
    human_judgment: false

# Metrics
duration: ~20min
completed: 2026-07-23
status: complete
---

# Phase 4 Plan 05: Weekly Report + Holdings YAML Loader + Trades-Implied + Opt-In Email Summary

**Per-account holdings YAML loader with warn-don't-fail/no-silent-normalize semantics, plus a
weekly-report assembler that always writes markdown and reuses the incumbent `email.py` read-only
for opt-in delivery.**

## Performance

- **Duration:** ~20 min
- **Started:** 2026-07-23T08:5x (RED test 1 committed 08:56:18Z)
- **Completed:** 2026-07-23T09:02:23Z
- **Tasks:** 2 (both TDD: RED -> GREEN)
- **Files modified:** 7 (5 created, 1 modified config file, 1 gitignore edit)

## Accomplishments
- `load_account_weights()` (`report/holdings.py`) — `yaml.safe_load`-only per-account holdings
  loader. Missing file returns a neutral `{"weights": {}, "cash": 1.0}` dict with a WARNING;
  an out-of-tolerance weights+cash sum (>0.02 from 1.0) WARNs but returns the RAW values —
  never normalizes (D-01, T-04-14). Non-numeric weight/cash values are coerced via
  `try/except (TypeError, ValueError)`, WARNING, skip (V5, T-04-15).
- `config/accounts/example.yaml` — committed schema doc with obviously-fake tickers (`ZZZZ`,
  `YYYY`, `XXXX`); `.gitignore` now ignores `config/accounts/*.yaml` except `example.yaml`
  (T-04-16 Information Disclosure control, verified via `git check-ignore`).
- `trades_implied()` (`report/weekly.py`) — one row per unioned target/current asset with
  `current_pct`/`target_pct`/`delta_pct`/`signal`; flat no-trade band: `|delta| < threshold`
  → HOLD, `delta >= +threshold` → BUY, `delta <= -threshold` → SELL.
- `assemble_weekly_report()` — pure-function markdown assembly: current regime distribution,
  trajectory from the empirical transition matrix (with the A1 cold-start rule documented
  inline per the plan's checker note), per-asset signals from `returns_by_regime` flagging
  `n_obs < min_obs_flag` cells as `[LOW-CONFIDENCE — short history, D11]`, and per-account
  target-vs-current trades implied with a one-line regime rationale.
- `write_weekly_report()` — ALWAYS writes `(output_dir or OUTPUT_DIR/reports/platform)/weekly_report.md`.
- `_build_report_inputs()` — isolates the full allocation cycle (load hysteresis state BEFORE
  save, update with current nowcaster probabilities, `vol_targeted_tilt`, persist new state)
  so `main()`'s email test can monkeypatch it without real Phase 1/3 checkpoint data.
- `main(--send-email)` — argparse CLI; the default path (`main([])`) never imports or calls
  `send_weekly_email`; `--send-email` calls `build_weekly_email_body` → `load_email_config` →
  `send_weekly_email` exactly once each (D-02), reusing `trading_crab_lib.email` read-only.

## Task Commits

Each task followed the TDD RED -> GREEN cycle:

1. **Task 1: Per-account holdings YAML loader**
   - `test(04-05): add failing test for holdings YAML loader` — `5b29b9f`
   - `feat(04-05): implement per-account holdings YAML loader (L4-03)` — `dca03b9`
2. **Task 2: Weekly report assembly + trades-implied + opt-in email**
   - `test(04-05): add failing test for weekly report assembly + opt-in email` — `8220322`
   - `feat(04-05): implement weekly report assembly + trades-implied + opt-in email (L4-02)` — `2f560bb`

No REFACTOR commits were needed — both GREEN implementations passed on the first pass with no
cleanup required.

## Files Created/Modified
- `src/trading_crab_lib/platform/report/__init__.py` - package docstring
- `src/trading_crab_lib/platform/report/holdings.py` - `load_account_weights`, `_WEIGHT_SUM_TOLERANCE`
- `src/trading_crab_lib/platform/report/weekly.py` - `trades_implied`, `assemble_weekly_report`,
  `write_weekly_report`, `_build_report_inputs`, `main`
- `config/accounts/example.yaml` - committed schema doc, obviously-fake data
- `.gitignore` - ignore `config/accounts/*.yaml`, keep `example.yaml` tracked
- `tests/unit/test_platform_holdings.py` - 7 tests
- `tests/unit/test_platform_report_weekly.py` - 14 tests

## Decisions Made
- Isolated `_build_report_inputs()` as a separate, monkeypatchable function rather than inlining
  the allocation cycle directly in `main()` — the plan's "orchestration clarification" required
  `main()` to run the full load→update→tilt→save cycle, but no real Phase 1/3 checkpoint data
  exists in this environment yet; separating it keeps `main()`'s email-opt-in test fast and
  network/checkpoint-free while still expressing the real production orchestration in one place.
- `report.accounts` config key (list of account names for `main()`'s per-account report) reads
  defensively via `cfg.get("report", {}).get("accounts", [])` — not added to
  `_REQUIRED_PLATFORM_SECTIONS`, matching the established optional-section pattern. No accounts
  are configured by default (empty list) since no real account YAML exists yet in this repo.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Test assertion over-matched the module's own docstring text**
- **Found during:** Task 1 (holdings.py `TestSecurityConvention` test)
- **Issue:** The initial test asserted `"load_portfolio" not in source` against
  `inspect.getsource(holdings)`, but the module's own docstring legitimately explains *why* it
  does NOT reuse `load_portfolio()`, mentioning the name in prose. This caused a false-positive
  test failure.
- **Fix:** Narrowed the assertion to check for the actual import statement patterns
  (`"import load_portfolio"` / `"from trading_crab_lib.config import"`), which is what the
  acceptance criterion actually cares about (no import of the incumbent function), not a bare
  substring match against documentation.
- **Files modified:** `tests/unit/test_platform_holdings.py`
- **Verification:** `pytest tests/unit/test_platform_holdings.py -x -q` — all 7 tests pass.
- **Committed in:** `dca03b9` (part of the GREEN commit, test fix applied before commit)

---

**Total deviations:** 1 auto-fixed (1 bug — test assertion specificity)
**Impact on plan:** No scope creep; the fix only tightened a test assertion to match its stated
intent. No production code behavior changed.

## Issues Encountered
None.

## User Setup Required
None for the default markdown-only path. Email delivery (`--send-email`) requires SMTP
configuration (`config/email.local.yaml` or `TC_SMTP_*` env vars) — already documented via the
incumbent `email.py`'s existing setup flow; no new setup steps introduced by this plan.

## Next Phase Readiness
- `platform/report/` is complete: holdings loader (L4-03) and weekly report assembly + opt-in
  email (L4-02) are both implemented and tested.
- Phase 4's remaining requirement is L4-04 (daily tripwire), already implemented in a prior wave
  (`platform/tripwire/monitor.py`) per the plan's `depends_on: [04-02, 04-03]`.
- `main()`'s live orchestration (`_build_report_inputs`) has not been run against real ingested
  data yet — this environment has no live FRED/yfinance data (STATE.md: pending `FRED_API_KEY`).
  Live wiring and a real weekly report run are deferred to a human-verification step, consistent
  with Phase 1/3's precedent (RESEARCH.md Environment Availability).
- No blockers for Phase 5 (honest backtest) — this plan's artifacts are report/delivery glue,
  not evaluated model configurations, and are correctly exempt from the trial registry.

---
*Phase: 04-asset-prediction-allocation*
*Completed: 2026-07-23*

## Self-Check: PASSED

All 7 key files verified present on disk; all 4 task commits (`5b29b9f`, `dca03b9`, `8220322`,
`2f560bb`) verified present in `git log`; `pytest tests/unit/test_platform_holdings.py
tests/unit/test_platform_report_weekly.py -x -q` = 21 passed; full `pytest tests/unit/ -k
platform -q` = 290 passed, 2 skipped; full `pytest tests/unit/ -q` = 913 passed, 49 skipped.
No failures.
