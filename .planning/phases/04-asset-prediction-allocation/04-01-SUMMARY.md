---
phase: 04-asset-prediction-allocation
plan: 01
subsystem: data
tags: [fred, yfinance, yaml-config, ingestion, checkpoints]

# Dependency graph
requires:
  - phase: 03-regime-labeling-prediction
    provides: label_regimes()/nowcaster outputs consumed by later Phase 4 plans (not this plan)
  - phase: 01-honesty-infrastructure
    provides: platform/ config + checkpoint conventions this plan extends
provides:
  - SPY in universe.satellites — daily SPY prices now reachable via the existing fetch_universe_prices() path
  - fred_daily config block + macro_daily.py — daily DAAA/DBAA credit-spread ingestion, "fred_daily_raw" checkpoint
  - allocation:/tripwire:/report: config sections with documented defaults, read via cfg.get() only
affects: [04-02, 04-03, 04-04, 04-05]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "gap_lag.py module shape (compute + report/persist + __main__ self-check) applied to a new ingestion module"
    - "macro_monthly.py's ThreadPoolExecutor + try/except-WARNING skeleton reused minus the resample step"

key-files:
  created:
    - src/trading_crab_lib/platform/ingestion/macro_daily.py
    - tests/unit/test_platform_macro_daily.py
  modified:
    - config/platform_settings.yaml
    - docs/splicing_rules.md

key-decisions:
  - "SPY added to universe.satellites (not a new dedicated list) — reuses the existing fetch_universe_prices() ingestion path with zero code changes"
  - "fred_daily config section mirrors fred_monthly.series shape exactly, kept as a separate top-level section (never resampled) rather than a flag on fred_monthly"
  - "fetch_fred_daily() falls back from fred_daily.api_key to fred_monthly.api_key so a single FRED_API_KEY env var covers both sections"
  - "allocation/tripwire/report sections deliberately NOT added to _REQUIRED_PLATFORM_SECTIONS — optional, defensively read via cfg.get() per RESEARCH's explicit constraint"

patterns-established:
  - "New daily-frequency ingestion modules follow macro_monthly.py's client/executor skeleton with the resample step removed, per prices_daily.py's no-internal-resample doctrine"

requirements-completed: [L4-04]

coverage:
  - id: D1
    description: "SPY added to universe.satellites (dual-purpose tradable proxy + tripwire daily price input)"
    requirement: "L4-04"
    verification:
      - kind: unit
        ref: "python3 -c \"from trading_crab_lib.platform.config import load_platform_config as L; assert 'SPY' in L()['universe']['satellites']\""
        status: pass
    human_judgment: true
    rationale: "Confirming prices_daily.fetch_universe_prices() actually ingests SPY into daily_raw requires a live yfinance network run (deferred human-verification item per plan's <verification> block, not blocking this plan)."
  - id: D2
    description: "fred_daily config block (DAAA/DBAA, shift:false) + macro_daily.py module (fetch_fred_daily, assemble_fred_daily, report_fred_daily, __main__ self-check)"
    requirement: "L4-04"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_macro_daily.py (11 tests: assembly, no-resample invariant, per-series failure isolation, checkpoint persistence, artifact writing)"
        status: pass
      - kind: other
        ref: "python3 -m trading_crab_lib.platform.ingestion.macro_daily (synthetic self-check, no network)"
        status: pass
    human_judgment: false
  - id: D3
    description: "allocation:/tripwire:/report: config sections with documented defaults (D-03 target_vol=0.10, hysteresis 0.7/0.4, tripwire trio thresholds)"
    requirement: "L4-04"
    verification:
      - kind: unit
        ref: "python3 -c config assertion (see plan Task 1 <verify>) — target_vol_annual, hysteresis thresholds, tripwire keys, report keys all present"
        status: pass
      - kind: unit
        ref: "grep _REQUIRED_PLATFORM_SECTIONS src/trading_crab_lib/platform/config.py — confirms still exactly 6 entries, unchanged"
        status: pass
    human_judgment: false
  - id: D4
    description: "docs/splicing_rules.md documents SPY dual-purpose, DAAA/DBAA credit source, Fidelity-CSV v2 placeholder seam"
    requirement: "L4-04"
    verification:
      - kind: other
        ref: "grep -q SPY docs/splicing_rules.md && grep -qi tripwire docs/splicing_rules.md && grep -qi Fidelity docs/splicing_rules.md"
        status: pass
    human_judgment: false

duration: 6min
completed: 2026-07-22
status: complete
---

# Phase 4 Plan 1: Data-Gap Ingestion + Config Foundation Summary

**SPY added to the daily universe ticker list, a new `macro_daily.py` module fetches DAAA/DBAA credit spreads at native daily frequency into a `fred_daily_raw` checkpoint, and `config/platform_settings.yaml` gains `allocation:`/`tripwire:`/`report:` sections with literature-derived defaults — closing both data gaps RESEARCH identified before the rest of Phase 4 builds on them.**

## Performance

- **Duration:** 6 min
- **Started:** 2026-07-22T23:46:04Z (Task 1 commit)
- **Completed:** 2026-07-22T23:47:59Z (Task 3 commit)
- **Tasks:** 3
- **Files modified:** 4 (1 config edit, 1 new module, 1 new test file, 1 docs edit)

## Accomplishments
- `SPY` added as the first entry in `universe.satellites` with an inline comment documenting its dual purpose — closes RESEARCH Pitfall 1 (no daily SPY price series existed for the tripwire's drawdown-from-peak signal)
- New `fred_daily:` config block (DAAA/DBAA, `shift: false`) plus `platform/ingestion/macro_daily.py` (`_fetch_fred_daily`, `assemble_fred_daily`, `fetch_fred_daily`, `report_fred_daily`, `__main__` self-check) — closes RESEARCH Pitfall 2 (FRED's `BAA`/`AAA` are monthly-only; the daily credit-spread-velocity tripwire signal needed the daily `DAAA`/`DBAA` counterparts)
- `allocation:`/`tripwire:`/`report:` config sections added with documented, literature-derived defaults (D-03 `target_vol_annual: 0.10`; hysteresis `act_threshold: 0.70`/`unwind_threshold: 0.40`; RiskMetrics-style `vol_halflife_days: 11.2`; tripwire thresholds flagged provisional-until-Phase-5-backtest per RESEARCH A3) — `_REQUIRED_PLATFORM_SECTIONS` in `config.py` left unchanged (still exactly 6 entries)
- `docs/splicing_rules.md` updated with a new "Phase 4 additions" section documenting SPY's dual purpose, DAAA/DBAA as the tripwire's credit source, and the Fidelity positions-CSV parser as a documented v2 (L4-V2-05) placeholder seam

## Task Commits

Each task was committed atomically:

1. **Task 1: Add SPY + fred_daily + allocation/tripwire/report config sections** - `62607e5` (feat)
2. **Task 2: Daily FRED credit ingestion module (macro_daily.py) + test** - `ca65f54` (feat)
3. **Task 3: Document SPY dual-purpose + Fidelity-CSV placeholder seam** - `1d25247` (docs)

_No TDD tasks in this plan (config + straightforward ingestion module, no `tdd="true"` flags)._

## Files Created/Modified
- `config/platform_settings.yaml` - SPY added to `universe.satellites`; new `fred_daily:` block (DAAA/DBAA); new `allocation:`/`tripwire:`/`report:` sections
- `src/trading_crab_lib/platform/ingestion/macro_daily.py` - New module: native-daily FRED fetch (no resample), pure assembly function, checkpoint persistence, parquet artifact reporting, synthetic `__main__` self-check
- `tests/unit/test_platform_macro_daily.py` - 11 mocked tests: assembly shape, no-resample invariant (sub-monthly index preserved), per-series failure isolation, `"fred_daily_raw"` checkpoint save, `fred_daily`→`fred_monthly` API-key fallback, artifact writing (empty and non-empty)
- `docs/splicing_rules.md` - New "§6. Phase 4 additions" section: SPY dual-purpose, DAAA/DBAA daily credit source, Fidelity-CSV v2 placeholder seam

## Decisions Made
- **SPY reuses `universe.satellites`, not a new ticker list** — `prices_daily.fetch_universe_prices()`'s existing `universe_fetch_tickers()` already unions `satellites`/`holdings`/`watchlist`, so adding SPY there required zero code changes to the frozen `prices_daily.py`.
- **`fred_daily` mirrors `fred_monthly.series` shape exactly** but stays a separate top-level section (per plan spec) rather than reusing the inert `daily: true` flag already present (decoratively) on `fred_monthly.T10Y3M`/`T10Y2Y`/`VIXCLS` — RESEARCH confirmed that flag is never read by any code path, so extending its meaning would have been silently ineffective.
- **`fetch_fred_daily()` falls back to `fred_monthly.api_key`** when `fred_daily.api_key` is absent, so a single `FRED_API_KEY` env var (injected by `load_platform_config()` into both sections) covers both — avoids requiring two separate env-var reads for the same credential.
- **`allocation`/`tripwire`/`report` sections were NOT added to `_REQUIRED_PLATFORM_SECTIONS`** — verified via grep that the list is still exactly 6 entries post-edit, satisfying the plan's explicit constraint that these are optional, defensively-read (`cfg.get()`) sections.

## Deviations from Plan

None - plan executed exactly as written. All three tasks matched their `<action>` specs; all `<verify>` automated checks and `<acceptance_criteria>` items pass as specified.

## Issues Encountered
None.

## User Setup Required

None for this plan. `FRED_API_KEY` is already a required env var for the existing `fred_monthly`/`fred_vintage` sections (Phase 1) — `macro_daily.py` reuses the same credential, no new secret needed. Live wiring verification (confirming `fetch_universe_prices()` actually pulls SPY, and `fetch_fred_daily()` actually pulls DAAA/DBAA, against real FRED/yfinance data) is a deferred human-verification item per the plan's `<verification>` block — not a blocker for this plan, consistent with the existing Phase 1 pending-live-data-run precedent noted in STATE.md.

## Next Phase Readiness
- `daily_raw` (once a live `fetch_universe_prices()` run occurs) will contain SPY — unblocking Plan 04-04's (tripwire) drawdown-from-peak signal.
- `fred_daily_raw` checkpoint is ready to be read by the tripwire's credit-spread-velocity signal once live-wired.
- `allocation:`/`tripwire:`/`report:` config sections are in place for Plans 02-05 to read via `cfg.get()` without any further YAML edits — this plan owned the ONLY edit to `config/platform_settings.yaml` for the phase, per the plan's explicit single-owner constraint.
- No blockers identified.

---
*Phase: 04-asset-prediction-allocation*
*Completed: 2026-07-22*
