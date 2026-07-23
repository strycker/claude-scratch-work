---
phase: 04-asset-prediction-allocation
verified: 2026-07-23T09:09:32Z
status: passed
score: 5/5 must-haves verified
behavior_unverified: 0
overrides_applied: 0
---

# Phase 4: Asset Prediction & Allocation Verification Report

**Phase Goal:** Glenn can open a weekly report that tells him, per regime, which assets look
favorable, what volatility-targeted portfolio mix to hold, what trades are implied versus
his current holdings, and whether any tripwire condition demands he act sooner.
**Verified:** 2026-07-23
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths (ROADMAP Success Criteria)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Returns-by-regime tables report historical return/risk stats per asset conditional on regime for the v1 asset universe | ✓ VERIFIED | `src/trading_crab_lib/platform/assets/returns.py::returns_by_regime_stats` — one row per (regime, asset) with mean/std/annualized-Sharpe (`(mean/std)*sqrt(12)`)/hit-rate/max-drawdown/n_obs, aligned via `.index.intersection` (never positional slicing — P3 guard). `_MIN_OBS_FLAG = 6` surfaces D11 short-history assets rather than dropping them; `compute_monthly_returns` uses `.pct_change()` and leaves leading NaNs (NULL-tolerant). Docstring explicitly states this table is "historical conditioning, not model selection" and is exempt from the trial registry / purged CV in v1 — matches CONTEXT's locked stance. `report_returns_by_regime` persists both the `returns_by_regime` checkpoint and a schema-stable `outputs/reports/platform/returns_by_regime.parquet`. Live-verified: `tests/unit/test_platform_returns.py` (9/9 pass), `python3 -m trading_crab_lib.platform.assets.returns` synthetic self-check ran and printed a correct 5-row (regime,asset) table. |
| 2 | EWMA volatility forecasts are computed per asset and available to size positions and feed the tripwire | ✓ VERIFIED | `src/trading_crab_lib/platform/assets/vol.py::ewma_vol` — single `.ewm(halflife=...).std() * sqrt(annualization_factor)` implementation (RiskMetrics-style), reused (not re-derived) by `allocation/tilt.py::portfolio_vol` (imports `ewma_vol`) and `tripwire/monitor.py::realized_vol_spike` (imports `ewma_vol`, `DAILY_ANNUALIZATION`) — confirmed by direct import-grep and by reading both call sites. `config/platform_settings.yaml` `allocation.ewma_halflife_months: 6` and `tripwire.vol_halflife_days: 11.2` both drive the same function with different units. Live-verified: `tests/unit/test_platform_vol.py` (9/9 pass, hand-rolled EWMA reference match + daily/monthly annualization scaling), `python3 -m trading_crab_lib.platform.assets.vol` self-check printed `annualized EWMA vol = 16.64%`. |
| 3 | A naive vol-targeted regime-tilt allocation is produced with hysteresis bands (act ~0.7 / unwind ~0.4) so the target mix doesn't flip on every small change | ✓ VERIFIED | `allocation/hysteresis.py::update_active_regime` is a Schmitt trigger on the HELD regime's OWN probability (never argmax) — `probs.get(prev_active, 0.0) >= unwind_threshold` holds regardless of a competitor's probability. Config defaults `act_threshold: 0.70` / `unwind_threshold: 0.40` (confirmed present via a live `load_platform_config()` assertion). `load_active_regime`/`save_active_regime` persist via `CheckpointManager`, load-before-save order followed correctly in `report/weekly.py::_build_report_inputs` (comment-annotated `# load BEFORE save` / `# save AFTER load`). `allocation/tilt.py::vol_target_scale` returns `min(1.0, target/actual)` and `0.0` on non-positive vol — no code path can lever past 1.0. `portfolio_vol` uses the blended-EWMA path when ≥12 overlapping months exist, else a documented `# ponytail:` conservative linear-sum fallback that can only under-lever. `target_vol_annual: 0.10` present in config (D-03). Live-verified: `tests/unit/test_platform_hysteresis.py::TestNoFlipInvariant::test_oscillating_probability_never_flips_active_regime` (0.65↔0.72 oscillation path — read the test source directly, confirmed the exact values) and `test_competitor_spike_does_not_steal_active_regime` both pass; `tests/unit/test_platform_hysteresis.py` (11/11) and `tests/unit/test_platform_tilt.py` (14/14) all pass. |
| 4 | The weekly report (via existing email machinery) shows current regime distribution + trajectory, per-asset signals, and target-vs-current mix with trades implied — with current holdings sourced from a manual per-account YAML file (a Fidelity CSV parser seam is documented as a placeholder only) | ✓ VERIFIED | `report/weekly.py::assemble_weekly_report` emits all 4 required sections (regime distribution, transition-matrix trajectory, per-asset signals flagging `n_obs < min_obs_flag` as `[LOW-CONFIDENCE — short history, D11]`, per-account trades-implied with a one-line regime rationale) — confirmed by a live functional call producing correct markdown end-to-end (BUY/SELL/HOLD rows matched hand-computed deltas). `write_weekly_report` ALWAYS writes `outputs/reports/platform/weekly_report.md` regardless of `--send-email` (confirmed: `main([])` path never imports/calls `send_weekly_email`; `--send-email` calls `build_weekly_email_body`→`load_email_config`→`send_weekly_email` exactly once). `email.py` is unmodified since before this phase (`git diff main...claude/phase-4-execution -- src/trading_crab_lib/email.py` is empty) and is read-only reused with a matching call signature (`build_weekly_email_body(report_dir, subject_prefix)`). `grep` confirms no `import` of `trading_crab_lib.reporting` or `load_portfolio` anywhere under `platform/` (only docstring prose explaining why it is NOT reused). Holdings: `report/holdings.py::load_account_weights` is a new, small YAML loader (`yaml.safe_load` only) — warn-don't-fail, never silently normalizes an out-of-tolerance weights+cash sum (returns RAW values with a WARNING, per D-01/T-04-14). `config/accounts/example.yaml` is git-tracked; `.gitignore` (`config/accounts/*.yaml` + `!example.yaml`) confirmed live via `git check-ignore` (real account files ignored, example.yaml not). Fidelity CSV parser seam documented in `docs/splicing_rules.md` §6 Phase 4 additions (grep-confirmed) and in `holdings.py`'s module docstring — placeholder only, no implementation (matches L4-V2-05 deferral). Live-verified: `tests/unit/test_platform_holdings.py` (7/7), `tests/unit/test_platform_report_weekly.py` (14/14), a direct functional call to `assemble_weekly_report`+`write_weekly_report` produced a real markdown file on disk. |
| 5 | A minimal daily tripwire monitor combines 3 independent signals (e.g. vol spike, credit-spread velocity, drawdown-from-peak) with OR-logic into one escalation output: none / "run weekly scoring early" / "Tier-1 de-risk review" | ✓ VERIFIED | `tripwire/monitor.py::escalate(vol_spike, credit_velocity, spy_drawdown)` is pure count-driven OR-logic (`sum([...])`: 0→NONE, 1→RUN_WEEKLY_SCORING_EARLY, 2/3→TIER1_DERISK_REVIEW) — identity-independent (any single signal alone escalates the same way; test suite proves all 8 truth-table combinations). Three independent-family signals present: `realized_vol_spike` (vol family, reuses `ewma_vol`), `credit_spread_velocity` (credit family, DBAA−DAAA widening in bps over a lookback), `spy_drawdown_from_peak` (price family, `cummax()`-based). `run_tripwire()` falls back to `daily_raw["SPY"]` / `fred_daily_raw[["fred_daaa","fred_dbaa"]]` checkpoints on the live path (no live network call — reads persisted checkpoints only) and accepts injected series for tests/CLI self-check. CLI (`python3 -m trading_crab_lib.platform.tripwire.monitor`) runs live in this sandbox with zero network access, exit code 0, prints exactly one escalation value as its final stdout line (`Tier-1 de-risk review` on the synthetic fixture, matching the expected 2-of-3-triggered case for that fixture). All thresholds are config-driven under `tripwire:` in `config/platform_settings.yaml`, explicitly marked provisional-until-Phase-5-backtest (A3) in both the config comments and the module docstring. Live-verified: `tests/unit/test_platform_tripwire.py` (27/27 pass, including the full 8-combination truth table + identity-invariance + CLI subprocess test). |

**Score:** 5/5 truths verified

### Deferred / Explicitly Accepted Design Stances (not gaps)

- **No trial registry / purged CV for returns-by-regime or tripwire thresholds** — CONTEXT explicitly states this is historical conditioning, not supervised model selection, for v1; `returns.py`'s own docstring documents the exemption citing RESEARCH.md Pitfall 4/9. Confirmed honored: no CV splitter or registry-logging call appears in `assets/returns.py`, `allocation/tilt.py`, or `tripwire/monitor.py`.
- **Live network wiring (FRED_API_KEY / yfinance) not exercised in this sandbox** — consistent with the established Phase 1/3 precedent (see `01-VERIFICATION.md`/`03-VERIFICATION.md`, both `status: passed` without live-network runs). Every module's `__main__` self-check is fully synthetic and was run live in this verification with zero network access, proving the logic end-to-end. Not treated as a gap.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|---|---|---|---|---|
| L3-01 | 04-02 | Returns-by-regime tables for the v1 universe | ✓ SATISFIED | `assets/returns.py`; `tests/unit/test_platform_returns.py` (9/9 live pass) |
| L3-02 | 04-02 | EWMA volatility forecasts per asset feeding sizing + tripwire | ✓ SATISFIED | `assets/vol.py`, reused by `allocation/tilt.py` + `tripwire/monitor.py`; `tests/unit/test_platform_vol.py` (9/9 live pass) |
| L4-01 | 04-03 | Naive vol-targeted regime-tilt allocation with hysteresis bands | ✓ SATISFIED | `allocation/hysteresis.py`, `allocation/tilt.py`; `tests/unit/test_platform_hysteresis.py` (11/11), `tests/unit/test_platform_tilt.py` (14/14) live pass |
| L4-02 | 04-05 | Weekly report reusing existing email machinery | ✓ SATISFIED | `report/weekly.py`; `email.py` unmodified (empty diff); `tests/unit/test_platform_report_weekly.py` (14/14 live pass) + direct functional run producing a real markdown file |
| L4-03 | 04-05 | Manual per-account holdings YAML, Fidelity CSV seam documented | ✓ SATISFIED | `report/holdings.py`, `config/accounts/example.yaml`, `.gitignore`; `tests/unit/test_platform_holdings.py` (7/7 live pass); `git check-ignore` confirms real-vs-example split |
| L4-04 | 04-01, 04-04 | Minimal daily tripwire, 3-signal OR-logic escalation | ✓ SATISFIED | `platform/ingestion/macro_daily.py` (daily credit ingestion), `tripwire/monitor.py`; `tests/unit/test_platform_macro_daily.py` (11/11), `tests/unit/test_platform_tripwire.py` (27/27) live pass; CLI ran live, exit 0 |

**Note (documentation staleness, non-blocking):** `.planning/REQUIREMENTS.md` still shows L3-01/L3-02/L4-01..04 as unchecked `[ ]` with traceability status `Pending` (lines 42-50, 137-142) — the code delivers all six requirements per the evidence above, but the requirements-tracking checkboxes were not flipped to `[x]`/`Complete` as part of this phase's commits (unlike Phases 1-3, whose entries were all updated). This is a tracking-document sync gap, not a functional gap — recommend a follow-up doc-only commit to flip these six lines before Phase 5 planning references requirement status.

### Cross-Cutting Checks

| Check | Result | Evidence |
|---|---|---|
| Frozen incumbent untouched — `email.py` unmodified | ✓ PASS | `git diff main...claude/phase-4-execution -- src/trading_crab_lib/email.py` is empty |
| No platform import of incumbent `reporting.py` / `load_portfolio` | ✓ PASS | `grep -rn "trading_crab_lib.reporting\|load_portfolio" src/trading_crab_lib/platform/` returns only docstring prose, zero import statements |
| Zero new pip dependencies | ✓ PASS | `git diff main...claude/phase-4-execution -- requirements.txt requirements-dev.txt pyproject.toml src/trading_crab_lib/pyproject.toml` produces no added lines |
| `from __future__ import annotations` in all new Phase 4 modules | ✓ PASS | Confirmed present in all 12 new/modified platform source files (macro_daily, assets/{__init__,returns,vol}, allocation/{__init__,hysteresis,tilt}, tripwire/{__init__,monitor}, report/{__init__,holdings,weekly}) |
| Ponytail ceiling comment on no-covariance σ̂_port fallback | ✓ PASS | `allocation/tilt.py` line 43: `# ponytail: linear-sum-of-vols fallback deliberately ignores diversification — it can only make the tilt MORE conservative...` with named upgrade path (L3-V2-01) |
| `_REQUIRED_PLATFORM_SECTIONS` unchanged (allocation/tripwire/report stay optional) | ✓ PASS | Still exactly 6 entries (`data`, `fred_monthly`, `fred_vintage`, `splice`, `universe`, `taxonomy`) |
| Debt markers (TBD/FIXME/XXX/TODO/HACK/PLACEHOLDER) in Phase 4 files | ✓ CLEAN | Zero matches across all 8 new source files + config (one incidental "placeholder" hit in config is a Phase-1-era DATA-06 section comment, unrelated to Phase 4) |
| Full test suite | ✓ PASS | `pytest tests/ -q` → 1015 passed, 49 skipped (matches SUMMARY claims, independently re-run) |
| Targeted Phase 4 test suite | ✓ PASS | `pytest tests/unit/test_platform_{macro_daily,returns,vol,hysteresis,tilt,tripwire,holdings,report_weekly}.py -q` → 102 passed, 0 skipped |

### Anti-Patterns Found

None. No stub returns, no empty handlers, no hardcoded-empty data flowing to output in any Phase 4 file.

### Human Verification Required

None. All 5 success criteria are verified against live-executed code (targeted + full pytest runs, direct `python3 -c` functional calls, and all four modules' `__main__` synthetic self-checks — all run in this verification session, not merely cited from SUMMARY.md). Live network wiring against real FRED/yfinance data remains a known, previously-accepted deferred item (see Phase 1/3 precedent) — not a new gap and not blocking.

### Gaps Summary

No gaps. One non-blocking documentation-staleness note: `.planning/REQUIREMENTS.md` checkbox/traceability rows for L3-01/L3-02/L4-01..04 were not flipped to complete (see Requirements Coverage note above) — recommend a follow-up doc commit, does not block Phase 5.

---

_Verified: 2026-07-23T09:09:32Z_
_Verifier: Claude (gsd-verifier)_
