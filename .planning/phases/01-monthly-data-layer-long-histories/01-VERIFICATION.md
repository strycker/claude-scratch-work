---
phase: 01-monthly-data-layer-long-histories
verified: 2026-07-15T00:00:00Z
status: gaps_found
score: 4/5 success criteria fully verified (1 partial)
behavior_unverified: 0
overrides_applied: 0
gaps:
  - truth: "Every feature is classified fast/slow/agency in config, and a lean full-history (1962+) feature set is defined and usable for labeling (ROADMAP Success Criterion 4 / DATA-04)"
    status: partial
    reason: "taxonomy['slow'] declares 4 features (cape_shiller, div_yield, buffett_indicator, real_rate_level) but compute_lean_features() in transforms_monthly.py only computes 2 of them. buffett_indicator and real_rate_level are listed in config/platform_settings.yaml's taxonomy.slow block and in taxonomy.lean_feature_set()'s output, but no code path ever produces those two columns — they do not exist in monthly_features. Calling lean_feature_set() and then indexing monthly_features by that set would KeyError on those two names."
    artifacts:
      - path: "src/trading_crab_lib/platform/transforms_monthly.py"
        issue: "compute_lean_features() (lines 130-176) has no branch computing buffett_indicator (market-cap/GDP) or real_rate_level (nominal minus inflation expectations) — both require data (Wilshire market cap, GDP, inflation expectations) not currently ingested by macro_monthly.py or splice.py"
      - path: "config/platform_settings.yaml"
        issue: "taxonomy.slow (lines 187-191) declares buffett_indicator and real_rate_level as if they are members of the usable lean set, but no ingestion/computation exists for either"
    missing:
      - "Either compute buffett_indicator and real_rate_level (needs GDP + a market-cap proxy, and a real-rate construction) in a follow-up plan, or remove them from taxonomy.slow / lean_feature_set() until they are actually computed, so lean_feature_set() never advertises a column that does not exist"
deferred: []
human_verification:
  - test: "Run build_monthly_spine(load_platform_config()) live against real FRED_API_KEY + network and confirm monthly_features reaches back to ~1962 with materially more rows than the quarterly incumbent, and that ALFRED calls succeed against fredapi's real get_series_all_releases() schema"
    expected: "A real monthly_features checkpoint is produced with a month-end index starting near 1962-01-31 (bounded by each source's actual first-available date) and materially more rows/year than the quarterly incumbent; ALFRED's live column schema matches the RESEARCH A1 assumption (realtime_start/realtime_end/date/value) that _detect_vintage_columns() defensively checks for"
    why_human: "This sandbox has no FRED_API_KEY and no network access; all 72 platform tests are 100% mocked at the ingestion boundary (macro_monthly.fetch_macro_monthly / prices_daily.fetch_universe_prices / alfred.fetch_all_vintages), which is correct per the CONTEXT.md 'no live network calls in tests' invariant but means the phase's core empirical claim — real spliced histories reaching ~1962 — has never been demonstrated against live data. 01-07-SUMMARY.md itself flags this as an outstanding, non-blocking follow-up ('a single live run... needs FRED_API_KEY + network, not exercised by the automated suite')."
---

# Phase 1: Monthly Data Layer & Long Histories Verification Report

**Phase Goal:** The pipeline ingests and transforms monthly data with long spliced
histories back to ~1962, point-in-time vintages where available, and a documented
feature taxonomy — replacing the quarterly-only spine as the foundation for regime
modeling.

**Verified:** 2026-07-15
**Status:** gaps_found (one success criterion partially unmet; everything else verified)
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths (ROADMAP Success Criteria)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Feature engineering produces a monthly-frequency dataset with quarterly agency series correctly lagged/aligned | ✓ VERIFIED | `build_monthly_spine()` (`transforms_monthly.py:197-242`) resamples to `"ME"`; `test_monthly_row_count` asserts 72 monthly rows over 6 years (`>2×` a quarterly equivalent). Agency alignment is proven with a real look-ahead-bias behavioral test, not just presence: `test_quarterly_series_alignment` (`test_platform_transforms.py:311-327`) shows a GDP value published 2018-04-27 is invisible in the March 2018 row (`fred_gdp == 100.0`, the older vintage) and only appears from April 2018 onward (`fred_gdp == 200.0`) |
| 2 | Core asset histories (S&P TR, Treasury TR synthetic, gold, oil, cash) available back to ~1962, splicing rules documented per asset | ✓ VERIFIED | `splice.py::build_core_research_series()` dispatches all 5 D-03 classes; `docs/splicing_rules.md` (242 lines) documents source, date range, join date, and method per class with an explicit summary table. Par-bond repricing math (`bond_price`/`monthly_total_return`) is behaviorally tested for par pricing (~1.0) and correct rising/falling-yield sign behavior in `tests/unit/test_platform_splice.py` (17 tests, all pass) |
| 3 | Agency series (GDP, CPI, etc.) pull ALFRED point-in-time vintages with a documented fallback for the pre-vintage era | ✓ VERIFIED | `alfred.py::value_as_of()`/`align_with_fallback()` reconstruct point-in-time values (never a later revision); `align_agency_monthly()` wires this into the monthly spine. `docs/vintage_alignment.md` documents the D-06 scope (5 series), the shift-vs-vintage distinction, and the pre-vintage fallback as an explicit accepted compromise. `test_pre_vintage_fallback_applied` proves a pre-2018-06-15 CPI value falls back to the shifted value (42.0) rather than NaN/error |
| 4 | Every feature classified fast/slow/agency in config; lean 1962+ feature set defined and usable for labeling | ⚠ PARTIAL | `taxonomy.py::validate_taxonomy()`/`classify_feature()`/`lean_feature_set()` are correctly implemented and tested (10/10 tests). **But** `lean_feature_set()` returns 14 names (10 fast + 4 slow) while `compute_lean_features()` only ever produces 12 of them — `buffett_indicator` and `real_rate_level` (declared in `platform_settings.yaml`'s `taxonomy.slow`) have no computation path anywhere in the codebase. See gap below |
| 5 | Satellites + holdings ingest NULL-tolerantly; paid-provider seams documented as placeholders; stockcharts/finviz noted | ✓ VERIFIED | `universe_fetch_tickers()`/`fetch_universe_prices()` merge via `pd.concat(axis=1)` outer join; `test_short_history_ticker_becomes_nan_padded_not_dropped` and `test_short_history_satellite_null_tolerant` prove short histories become NaN-padded columns, never dropped rows. `docs/paid_provider_seams.md` documents Norgate/Tiingo/EODHD as inert `NotImplementedError` seams and explicitly notes stockcharts.com/finviz.com as future feature (not price) sources |

**Score:** 4/5 success criteria fully verified (1 partial — infrastructure correct, 2 of 14 declared lean features never computed)

### Context Invariants (01-CONTEXT.md D-01/D-02, "no live network calls in tests")

| Invariant | Status | Evidence |
|-----------|--------|----------|
| Frozen incumbent untouched | ✓ VERIFIED | `git diff b24bc96 HEAD --stat` (pre-phase-1 commit vs. current HEAD) on `fred.py`, `multpl.py`, `macrotrends.py`, `assets.py`, `transforms.py`, `config.py`, `checkpoints.py`, `run_pipeline.py`, `src/trading_crab/pipeline.py` → empty (byte-identical). Full-repo diff outside `platform/`/`docs/`/`config/platform_settings.yaml`/`.planning/` touches only `.gitignore` (added a worktree-ignore line, unrelated to the incumbent pipeline) |
| New code self-contained in `platform/` subpackage | ✓ VERIFIED | All new modules live under `src/trading_crab_lib/platform/`; only imports from the incumbent are the explicitly-sanctioned reuse points (D-01): `multpl._scrape_raw_rows`/`_SUFFIX_MAP`, `macrotrends._extract_json_data`/`HEADERS`/`RATE_LIMIT_SECONDS`, `assets._ssl_bypass_curl_session`, `checkpoints.CheckpointManager`, and `trading_crab_lib.DATA_DIR`/`CONFIG_DIR` — never the frozen fetchers' quarterly-resampling entry points themselves |
| No live network calls in tests | ✓ VERIFIED | Every `requests.get`/`yf.download`/`Fred(...)` call site in the platform test files is `@patch`-mocked (confirmed by grep across all 7 `test_platform_*.py` files); `test_platform_transforms.py` additionally redirects checkpoint I/O to `tmp_path` so no production `data/checkpoints/platform/` file is ever written by tests |

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/trading_crab_lib/platform/{__init__,config,taxonomy,checkpoints}.py` | Foundation scaffold | ✓ VERIFIED | Exist, substantive, wired; `load_platform_config()`/`validate_platform_config()` and `validate_taxonomy()`/`classify_feature()`/`lean_feature_set()`/`check_columns_tagged()` all implemented and exercised by tests |
| `src/trading_crab_lib/platform/splice.py` | Splicing/synthesis engine | ✓ VERIFIED | `ratio_splice`, `bond_price`, `monthly_total_return`, `build_treasury_tr_synthetic`, `build_equity_total_return`, `build_core_research_series` all implemented; 17 tests pass |
| `src/trading_crab_lib/platform/ingestion/alfred.py` | ALFRED vintage fetch + PIT reconstruction | ✓ VERIFIED | `fetch_vintage_series`, `fetch_all_vintages`, `value_as_of`, `align_with_fallback`, `_detect_vintage_columns` all implemented; 9 tests pass, including explicit look-ahead-guard tests |
| `src/trading_crab_lib/platform/ingestion/{norgate,tiingo,eodhd}.py` | Placeholder seams | ✓ VERIFIED | Each imports cleanly, no SDK/network import, `fetch_prices()` raises `NotImplementedError` pointing at `docs/paid_provider_seams.md`; 6 tests pass |
| `src/trading_crab_lib/platform/ingestion/macro_monthly.py` | Monthly FRED/multpl/macrotrends ingestion | ✓ VERIFIED | `fetch_fred_monthly`, `_scrape_multpl_monthly`, `_fetch_macrotrends_monthly_all`, `fetch_macro_monthly` all implemented, monthly (not quarterly) cadence proven by test; 11 tests pass |
| `src/trading_crab_lib/platform/ingestion/prices_daily.py` | Daily universe price ingestion + monthly spine | ✓ VERIFIED | `universe_fetch_tickers`, `fetch_universe_prices`, `to_monthly_spine` all implemented, NULL-tolerant merge proven by test; 8 tests pass |
| `src/trading_crab_lib/platform/transforms_monthly.py` | Monthly spine orchestrator + lean features + tagging | ⚠ PARTIAL | `build_monthly_spine`, `align_agency_monthly` fully correct and tested. `compute_lean_features`/`tag_feature_columns` correct for the 12 columns they do compute, but do not cover 2 of the 14 taxonomy-declared lean features (see gap) |
| `config/platform_settings.yaml` | All declarative config blocks | ✓ VERIFIED | 9 blocks present (data, fred_monthly, fred_vintage, multpl_monthly, macrotrends_monthly, splice, universe, taxonomy, paid_providers) with concrete keys matching what every ingestion/splice/transform module consumes |
| `docs/{splicing_rules,vintage_alignment,paid_provider_seams}.md` | DATA-02/03/06 documentation deliverables | ✓ VERIFIED | All three read in full; each documents source/method/scope/tradeoffs per the phase's D-04 documentation requirement |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|----|--------|---------|
| `transforms_monthly.build_monthly_spine` | `ingestion.macro_monthly.fetch_macro_monthly` | direct call | ✓ WIRED | Line 213 |
| `transforms_monthly.build_monthly_spine` | `ingestion.prices_daily.fetch_universe_prices` | direct call | ✓ WIRED | Line 214 |
| `transforms_monthly.build_monthly_spine` | `splice.build_core_research_series` | direct call | ✓ WIRED | Line 215 |
| `transforms_monthly.build_monthly_spine` | `transforms_monthly.align_agency_monthly` → `ingestion.alfred.fetch_all_vintages`/`align_with_fallback` | direct call chain | ✓ WIRED | Lines 76-124, 216; behaviorally proven via look-ahead test |
| `transforms_monthly.build_monthly_spine` | `checkpoints.get_platform_checkpoint_manager` | `cm.save(daily_raw/monthly_raw/monthly_features)` | ✓ WIRED | Lines 223-236; `test_persists_monthly_features_checkpoint` round-trips a load |
| `transforms_monthly.compute_lean_features` | `taxonomy.lean_feature_set` | declared-vs-produced comparison | ⚠ PARTIAL | Produced columns are always a *subset* of the declared lean set (tested: `set(lean.columns) <= lean_set`), but the reverse is never true and never tested — 2 declared names are permanently absent |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| DATA-01 | 01-05, 01-07 | Monthly data spine; quarterly agency series aligned with lags | ✓ SATISFIED | `fetch_macro_monthly` (monthly cadence) + `align_agency_monthly` (PIT alignment) wired into `build_monthly_spine` |
| DATA-02 | 01-02 | Spliced ~1962+ histories for 5 core assets, documented | ✓ SATISFIED | `build_core_research_series` + `docs/splicing_rules.md` |
| DATA-03 | 01-03, 01-07 | ALFRED PIT vintages + pre-vintage fallback documented | ✓ SATISFIED | `alfred.py` + `docs/vintage_alignment.md` |
| DATA-04 | 01-01, 01-07 | Fast/slow/agency taxonomy in config; lean set defined and usable | ⚠ PARTIAL | Taxonomy validation infra correct; lean set is *defined* but not fully *usable* — 2/14 declared features never computed (see gap) |
| DATA-05 | 01-06 | Satellite/holdings NULL-tolerant ingestion | ✓ SATISFIED | `prices_daily.py` outer-concat merge, tested |
| DATA-06 | 01-04 | Paid-provider seams documented, no implementation | ✓ SATISFIED | `norgate.py`/`tiingo.py`/`eodhd.py` stubs + `docs/paid_provider_seams.md` |

**Note (informational, not a code gap):** `.planning/REQUIREMENTS.md` still shows DATA-01, DATA-02, DATA-03, DATA-05, DATA-06 as `[ ] Pending` in both the checklist and the traceability table, even though ROADMAP.md marks Phase 1 complete and this verification confirms 5 of 6 are fully satisfied (DATA-04 was updated to `[x] Complete` during Plan 01-01, but the other five were never updated). This is a documentation-bookkeeping gap, not a functional one — flagging so the traceability table can be brought current.

### Anti-Patterns Found

Grep across `src/trading_crab_lib/platform/`, all three docs files, and `config/platform_settings.yaml` for `TBD|FIXME|XXX` found **zero** matches. `TODO|HACK|PLACEHOLDER|not yet implemented|not available|coming soon` found one incidental match (`transforms_monthly.py:164`, inside a comment describing why a naive vol proxy is used — a documented `ponytail:` simplification with a stated upgrade path, not a stub). No blocking anti-patterns found in the code.

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| All 7 platform test files pass | `pytest tests/unit/test_platform_*.py -q` | 72 passed, 1 skipped (cssselect unavailable, matches incumbent's own skip convention) | ✓ PASS |
| Full repo suite remains green (no regression from Phase 1) | `pytest tests/ -q` | 797 passed, 49 skipped, 0 failed | ✓ PASS |
| Frozen incumbent modules byte-identical pre/post phase | `git diff b24bc96 HEAD --stat -- <9 incumbent files>` | empty diff | ✓ PASS |
| No live network calls anywhere in platform tests | grep for unmocked `requests.get`/`yf.download`/`Fred(` across `test_platform_*.py` | all call sites `@patch`-wrapped | ✓ PASS |

### Human Verification Required

1 item (see frontmatter `human_verification`): a live, network-connected run of `build_monthly_spine()` against real FRED_API_KEY has never been performed in this environment (no key, no network access here). All 72 tests are mocked at the ingestion boundary, which is correct per the CONTEXT.md "no live network calls in tests" invariant — but it also means the phase's headline empirical claim (spliced histories genuinely reaching back to ~1962 from live sources) has only been proven at the unit level (correct formulas, correct merge semantics, correct look-ahead guards), not end-to-end against real data. 01-07-SUMMARY.md itself flags this as an outstanding, documented, non-blocking follow-up.

## Gaps Summary

One concrete, demonstrable gap: `config/platform_settings.yaml`'s `taxonomy.slow` block declares `buffett_indicator` and `real_rate_level` as members of the lean full-history feature set (matching RESEARCH.md's original design recommendation), but `transforms_monthly.py::compute_lean_features()` never computes either — there is no ingestion or derivation path for market-cap (Buffett Indicator numerator) or a real-rate construction anywhere in the platform subpackage. `taxonomy.lean_feature_set(cfg)` therefore returns 2 names that will never appear as columns in `monthly_features`, which breaks the "usable for labeling" half of Success Criterion 4 for those two features specifically (the other 12 of 14 lean features are fully computed, tagged, and tested). This does not block Phase 2/3 from using the 12 working lean features, but should be closed — either implement the two missing derivations or remove them from the taxonomy declaration — before any downstream code indexes `monthly_features` by the full output of `lean_feature_set()`.

Everything else — monthly cadence, the 5 core-asset splices with documented rules, ALFRED point-in-time vintage alignment with a documented pre-vintage fallback, NULL-tolerant satellite/holdings ingestion, and paid-provider placeholder seams — is genuinely implemented, behaviorally tested (including a real look-ahead-bias regression test, not just presence checks), and does not modify a single byte of the frozen incumbent quarterly pipeline.

---

*Verified: 2026-07-15*
*Verifier: Claude (gsd-verifier)*
