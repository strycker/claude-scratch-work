---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
current_phase: 6
current_phase_name: Platform Notebook Suite
status: planning
stopped_at: "UAT audit of Phases 1-5 complete 2026-09-09 (.planning/UAT-AUDIT-2026-09-09.md, Parts I+II); A4 (ALFRED index-base splice) FIXED 976f7c8 — REQUIRES A REBUILD to take effect; baseline recorded (.planning/BASELINE-v1-tracer-bullet.md); 06-CONTEXT.md AMENDED (D-11 reversed, D-20 satisfied, D-18/D-19 partly stale). Next: rebuild + confirm the guard stays silent, then /gsd-plan-phase 6."
last_updated: "2026-09-09T00:00:00.000Z"
last_activity: 2026-09-09
last_activity_desc: UAT audit Phases 1-5; A4 ALFRED index-base splice fixed; pytest-clobbers-holdout bug fixed; Phase 6 context amended
progress:
  total_phases: 8
  completed_phases: 5
  total_plans: 28
  completed_plans: 28
  percent: 63
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-07-09)

**Core value:** Honest, regime-aware weekly guidance that beats buy-and-hold SPY net of
avoided drawdowns — never fooled by its own backtest.
**Current focus:** Phase 6 — platform notebook suite (pre-planning)

## Current Position

Phase: 6 — Platform Notebook Suite
Plan: Discussion complete and **amended 2026-09-09** — read `06-CONTEXT.md` *including*
its AMENDMENT section before planning (D-11 reversed, D-20 satisfied, D-18/D-19 partly
stale).
Status: **Ready to plan after one rebuild.** A4 is fixed in code (`976f7c8`) but it is
an *ingestion* fix, so the committed `monthly_raw.parquet` still carries the corrupted
`fred_cpi`. Run `python scripts/build_platform_data.py` with a live `FRED_API_KEY` and
confirm `_warn_on_level_discontinuity` stays silent for `fred_cpi`; then
`/gsd-plan-phase 6` is unblocked.
Last activity: 2026-09-09 — UAT audit of Phases 1–5; A4 fixed; a bug where pytest
destroyed the production holdout checkpoint found and fixed; Phase 6 context amended

### ⚠ UAT audit outcome (2026-09-09) — `.planning/UAT-AUDIT-2026-09-09.md`

Phases 1 and 5 were signed off against arithmetically impossible output (Phase 5's
closure record tabulates terminal log wealth of 111.06 — e¹¹¹ ≈ 10⁴⁸ — as an
*improvement*). Neither phase is re-opened as failed: **every criterion is satisfied as
phrased**. What is void is the recorded evidence and the conclusions drawn from it.

Restated scores: **Phase 1 — 2 of 5 genuinely verified, 3 certified over invalid
evidence. Phase 5 — 1 of 4 genuinely verified, 3 wiring-verified with void values.**

**The systemic finding:** every criterion in both phases is satisfied by evidence of one
of four shapes — existence, shape, pure-function correctness, or placement. None can
detect a physically impossible value. `gsd-tools query audit-uat` returns 0 items for
the same reason: it only surfaces pending/skipped/blocked, never
"passed-on-evidence-that-no-longer-holds". Audit item **A3** proposes a plausibility
gate as a standing criterion for every phase emitting numeric output.

**A4, found by applying that very lens — now FIXED (`976f7c8`), pending a rebuild:** ALFRED
point-in-time vintages of *rebased index* series are not level-comparable across
rebasings. `fred_cpi` carries two artificial ~3× discontinuities (1970-12 `39.60` →
1971-01 `119.03`; 1988-01 `345.9` → 1988-02 `115.9`). `real_rate_level` inherits it,
ranging **−209.49 … +74.51**, and is a defining feature of labeler states 2 and 3 —
**64.3% of occupancy**. This sits directly upstream of Phase 6's P2 and P3 notebooks.

### First trustworthy baseline — `.planning/BASELINE-v1-tracer-bullet.md`

All prior recorded numbers are void. Reference point going forward: strategy 3.7204
log wealth / −21.75% max DD, **last of five legs**; Faber 6.3726 / −18.94% beats it on
both dimensions; regime layer adds +0.0732 while making drawdown 1.95pp worse; headline
sojourn/lag ratio **1.25** against a design bar of ~5.

### Phase 6 discussion outcome (2026-08-04)

The notebooks were reframed during discussion, and the reframe reached up into the
project constraints:

- **Purpose:** periodic verification & validation (are the regimes still well-defined?
  is the data still behaving as historic?), **not** a per-run human gate. One cold-start
  selection gate at `P3_regime_labeling`; the other five carry no gate.
- **Honesty framing amended** in `PROJECT.md`: the fence is on *fitting*, not *looking*.
  Notebooks read the full span (incl. post-2020) via the explicit
  `get_holdout_checkpoint_manager()` opt-in — which `holdout.py` already documented as
  the live-scoring path. Fitting stays fenced at 2020-12. Post-2020 observations that
  change a decision are recorded with their date. The prior "firewalled from all
  selection decisions" wording was unworkable: catching feature decay *is* a selection
  decision informed by recent data.
- **`ROADMAP.md` criterion 2 rewritten** (+ new 2b on the holdout opt-in) and **NB-01
  reworded** in `REQUIREMENTS.md` to match.
- **Scope held:** no tuning ((K,λ) sweeps stay deferred to v2 per Phase 3 D-02), no
  report wiring, no package restructuring (Phase 7), no holdout-carve repair (separate).

### ✅ RESOLVED (2026-09-08) — fitting is now fenced at rest

*Was:* `data/holdout/` did not exist, the dev-tree `monthly_features` carried
post-cutoff rows to 2026-08, and nothing outside `tests/unit/test_platform_holdout.py`
called `write_monthly_features_split()` or `assert_dev_checkpoint_within_boundary()`.
A tested mechanism that no production path invokes is not a fence.

*Now:* `build_monthly_spine()` writes through the split, and
`scripts/build_platform_data.py` calls `assert_dev_checkpoint_within_boundary()`
and **fails the build** on violation. Applied to the checkpoint on disk: dev
1962-01 → **2020-12** (708 rows), holdout holds the 68 post-cutoff rows.

New `honesty.holdout.load_full_span()` is the explicit *looking* opt-in (dev +
holdout concatenated), matching the amended framing. `report/weekly.py` live
scoring now uses it — **required, not cosmetic**: it scores
`monthly_features.iloc[[-1]]`, so carving without it would have scored December
2020 as "today" every week. Three tests pin the wiring specifically and fail
against the unwired code.

Progress: [██████░░░░] 63% (5 of 8 phases)

### Roadmap restructure (2026-08-04)

Phase 6 was "Migration to Public Repo". It is now split and extended:

| Phase | Was | Now |
|---|---|---|
| 6 | Migration to Public Repo | **Platform Notebook Suite** (NB-01) |
| 7 | — | **Migration to Public Repo** (MIG-01) |
| 8 | — | **Invariants & Dimensional Reduction** (INV-01) |

Rationale: the migration's per-step validation gate is "run the notebook and verify",
but the platform has **zero** notebooks — all 12 in `notebooks/` cover the legacy
quarterly pipeline. Notebooks are a prerequisite, not a nice-to-have. Full analysis in
`.planning/STATUS-REVIEW-2026-08.md`.

## Performance Metrics

**Velocity:**

- Total plans completed: 12 (7 Phase 1 + 5 Phase 2) + 4 in Phase 05
- Full test suite: 1120 passed (post 05-05)

**By Phase:**

| Phase | Plans | Status |
|-------|-------|--------|
| 1 — Monthly Data Layer | 7/7 | Verified passed (FRED_API_KEY verified 2026-07-23 — human item cleared) |
| 2 — Honesty Infrastructure | 5/5 | Verified passed 5/5 |

*Updated after each plan completion*
| Phase 05 P01 | 3min | 2 tasks | 5 files |
| Phase 05 P02 | 12min | 3 tasks | 2 files |
| Phase 05 P03 | 18min | 2 tasks | 4 files |
| Phase 05 P04 | 15min | 2 tasks | 2 files |
| Phase 05 P05 | 22min | 3 tasks | 2 files |
| Phase 05 P06 | 7min | 3 tasks | 2 files |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- Roadmap: Data layer (Phase 1) precedes Honesty infrastructure (Phase 2) — HON-01
  (holdout carve) and HON-06 (causal-feature gating) need Phase 1's files and feature
  taxonomy to operate on; both still land before any modeling phase per design §14.

- Roadmap: L1 (labeling) + L2 (prediction) merged into one phase (3); L3 (asset
  prediction) + L4 (allocation & report) merged into one phase (4) — these are tightly
  coupled steps of the same tracer-bullet vertical slice (design §14 Phase 1).

- Roadmap: MIG-01 kept as its own final phase (6) per explicit orchestrator instruction,
  despite being a single requirement.

- Phase 2: walk-forward interface frozen (expanding_steps + run_walkforward with
  automatic single append_trial per run); Phase 3 models plug into this interface.

- [Phase 05]: Phase 5 Plan 1: crisis_windows default list hard-bounded to 4 in-sample crises (1973-74, 1980-82, 2000-02, 2008-09), no 2020/2022 window; compute_turnover uses index-union reindex(fill_value=0.0), not positional diffing — Keeps holdout discipline (Pitfall 4) and correctly handles cold starts / asset-set changes
- [Phase 05-02]: Task boundary matched the plan literally: Task 2 lands the core loop without holdout split/L2 resilience (TestHoldoutBoundary intentionally RED); Task 3 adds split_by_holdout_boundary + try/except degrade-and-continue, turning it GREEN — Makes the incremental TDD narrative visible in git history rather than one large commit
- [Phase 05-02]: All 6 driver tests monkeypatch module-level _refit_l1/_refit_l2 (and vol_targeted_tilt for the cash-residual test) instead of exercising the real jump-model/nowcaster fit — Keeps orchestration-invariant tests fast/deterministic and isolated from real-fit degeneracy (Pitfall 2 territory); the real fit path is separately proven via the module's __main__ self-check
- [Phase 05-03]: compute_sojourn_lag_headline groups ex-post transitions by their OWN target state and checks each against only that state's own filtered-probs column, never a class-agnostic max-across-classes series — Review F1 fix — class-agnostic max would systematically understate detection lag ('fooled by its own backtest')
- [Phase 05-03]: max_drawdown_and_duration's duration_months is the longest run of consecutive underwater periods (drawdown < 0), not strictly peak-to-trough — Matches the plan's literal <action> text; a never-recovered drawdown extends duration to end of series
- [Phase 05-04]: model_metrics.py implements its own _reconcile_and_stack_proba rather than importing sojourn_lag.py's build_filtered_probs_matrix — the plan's numpy/pandas/stdlib-only import constraint keeps the two evaluation modules independently grep-gated, even though both implement the same union-of-classes/K-padding pattern (review F3).
- [Phase 05-04]: report_model_metrics indexes per_step_metrics['y_true'] via direct dict access (not .get()), and asserts len(y_true)==len(dates)==len(proba), raising ValueError on mismatch — y_true is always joined by date by the report layer, never sourced from the walk-forward loop (review F2).
- [Phase 05-05]: no_regime_ablation adds a cash_returns passthrough kwarg not in the plan's literal one-line-delegation snippet (Rule 2 auto-fix) - omitting it would silently default the ablation's cash residual to 0%, breaking F4 cash-return symmetry; the function stays a single delegating return statement.
- [Phase 05-06]: run_full_backtest_evaluation computes the smoothed-vs-filtered gap via a hindsight-oracle vol_targeted_tilt driven by the full-sample smoothed states at each real walk-forward decision date (never inventing new allocation math) rather than a simpler point-estimate proxy — matches the design's non-causal batch-labeling doctrine and keeps the gap input genuinely distinct from the filtered strategy performance (Pitfall 1).
- [Phase 05-06]: The investable asset_returns universe fed to run_backtest excludes the 'cash' splice class (FZFXX) — cash is never tilted into as a risk position; it is the vol-target residual that earns cash_ret directly via run_backtest's cash_returns parameter (review F4).

### Pending Todos

- Verifier informational note: `holdout.py`/`registry.py` use hardcoded constants that
  match `config/platform_settings.yaml` sections rather than reading them — wire to
  config if/when the values ever need to change (future-divergence risk only).

### Blockers/Concerns

- ~~Live 1962+ data run still pending FRED_API_KEY in the claude.ai/code environment
  (Phase 1 human-verification item).~~ **RESOLVED 2026-07-23** — `FRED_API_KEY` is present
  in the environment (32-char key) and functionally verified against the live FRED API
  (authenticated GDP series fetch succeeded). No longer a blocker; do not re-flag.

- **Macrotrends fix is wiring-verified, NOT live-verified** (quick task 260805-570). It
  fetches through a browser-impersonating client, but the container's agent proxy
  MITM-terminates TLS and resets curl_cffi's impersonated ClientHello, so it could not be
  confirmed against the live site. Unit tests prove the client is used; they do NOT prove
  impersonation defeats the bot check. Run the `<human-check>` in `260805-570-PLAN.md`
  Task 1 on a residential connection before trusting this source.

- **Macrotrends: headless-browser fallback now wired at both call sites (quick task
  260805-r7w). A residential-connection diagnostic (2026-08-05) confirmed a real headless
  browser DOES reach the live macrotrends page (HTTP 200, real rendered content, no
  interstitial) — the browser-reachability half of this fallback is no longer speculative.
  BUT the same diagnostic found the embedded-JSON regex (`_DATA_PATTERN`) does NOT match on
  the rendered page (the series lives inside a Highcharts closure, not a named `window`
  global), so the rendered HTML falls through to the `pandas.read_html` table path, not the
  JSON path. The page's only `<table>` has no distinguishing class (`historical_data_table`,
  assumed from a third-party snippet, does NOT exist — the real class is plain `"table"`), and
  its header cell text can be a squashed multi-line label (e.g. "Gold PricesMonthly Closing
  Price"). `BROWSER_WAIT_SELECTOR = "table"` is that unconfirmed candidate selector and is
  used ONLY with `require_selector=False` for exactly this reason. Column detection in
  `_scrape_series_html_table` / `_scrape_macrotrends_html_table_monthly` was extended to match
  a "month"-titled date column (with a value/date collision guard, since a squashed header
  containing "Monthly" also matches the substring "month"). What this diagnostic does NOT
  confirm: whether the end-to-end parse of a REAL macrotrends series (not the synthetic test
  fixtures in this task) produces correct data — that remains untested against the live site
  from this container (all egress here is proxy-reset). Run the `<human-check>` in
  `260805-r7w-PLAN.md` Task 3 to confirm the full chain end-to-end.**

- **Stooq: TLS impersonation confirmed insufficient; headless-Chromium fallback now wired,
  also NOT live-verified** (quick task 260805-jt2, superseding the 260805-570 Stooq
  finding). On a residential connection, `impersonate="chrome"` and `impersonate="safari"`
  — each with and without a hand-written header override, each with and without a
  same-session quote-page warm-up — ALL returned the identical 796-byte JavaScript
  browser-verification challenge page and set zero cookies. The challenge requires
  executing JavaScript; no TLS fingerprint can satisfy it. A headless-Chromium challenge
  solver (`ingestion/browser.py`) is now wired as the Stooq fallback behind the optional
  `[browser]` packaging extra, tried only when the plain HTTP path recovers zero tickers.
  This browser path is wiring-verified by mocked unit tests ONLY — it has NOT been
  confirmed against the live site, because all container egress is reset by the agent
  proxy (Chromium launches here, but even `https://example.com` returns
  `net::ERR_CONNECTION_RESET`). Run the `<human-check>` in `260805-jt2-PLAN.md` Task 3 on
  a residential machine to find out whether it defeats the challenge.

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 260805-570 | fix stooq and macrotrends bot-blocking, ALFRED vintage schema, build guard, yfinance rate-limit | 2026-08-05 | e98f1f0 | [260805-570-fix-stooq-and-macrotrends-bot-blocking-a](./quick/260805-570-fix-stooq-and-macrotrends-bot-blocking-a/) |
| 260805-r7w | generalize browser.py to fetch_page_html/fetch_urls_as_text, add Selenium as a second engine, route macrotrends through the browser fallback at both call sites | 2026-08-05 | (see directory) | [260805-r7w-generalize-browser-module-and-add-seleni](./quick/260805-r7w-generalize-browser-module-and-add-seleni/) |
| 260908-qwe | fix hindsight-oracle IndexError: restrict the smoothed oracle's per-step universe to assets that have started (phantom IAU/USO weight in 1974), guard portfolio_vol's per-asset EWMA fallback conservatively | 2026-09-08 | 18f68af | [260908-qwe-fix-hindsight-oracle-indexerror-phantom-](./quick/260908-qwe-fix-hindsight-oracle-indexerror-phantom-/) |
| 260908-rh4 | fix percent-vs-decimal yield units at the splice boundary (long_duration_tr compounded to 2.3e128; cash booked yield CHANGES as returns), add an asymmetric units guard, correct three test defects incl. an integration test whose 24/24 steps all degraded | 2026-09-08 | 75dedc7 | [260908-rh4-fix-percent-vs-decimal-yield-units-at-th](./quick/260908-rh4-fix-percent-vs-decimal-yield-units-at-th/) |
| 260908-fnc | close the holdout fence at rest: carve at build, assert at build, add load_full_span() looking opt-in, repoint live weekly scoring | 2026-09-08 | 7f99548 | (in this STATE entry) |

## Deferred Items

Items acknowledged and carried forward from previous milestone close:

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| *(none — first milestone)* | | | |

## Session Continuity

Last session: 2026-07-25T14:56:06.038Z
Stopped at: Completed 05-06-PLAN.md (honest backtest report capstone: assemble_backtest_report + write_backtest_report + run_full_backtest_evaluation + main CLI)
Resume file: None

### Audit Part II — Phases 2, 3, 4 (2026-09-09)

**Evidence quality is not uniform, and these three hold up far better than 1 and 5.**
Phase 3's DP-decode oracle test — proven identical to brute-force enumeration across
7 (T,K,λ) cases — is the strongest verification artifact in the project and should be
the pattern wherever a brute-force reference is affordable.

Five criteria need attention, not a blanket re-verification:

- **P2 C1** (holdout) — mechanism verified, application never invoked. Closed 2026-09-08.
- **P2 C5** (causal gating) — the rail IS live (`nowcaster.py:114`), but the gate is a
  name-suffix scan; a centered feature not following the convention passes silently.
- **P3 C4** (calibration) — "uses `CalibratedClassifierCV`" stands in for "is
  calibrated". Brier is 0.1816; no pass band exists.
- **P4 C2** (EWMA vol) — verified by import-grep; the consumer `portfolio_vol` then
  crashed on ragged universes, the condition holding for 43 of 49 backtest years.
- **P4 C3** (hysteresis) — **the criterion's purpose clause was never tested.**
  `update_active_regime` is a correct Schmitt trigger, but `active_regime` gates
  nothing: weights come from `vol_targeted_tilt(regime_probs, …)` in both the backtest
  driver and the weekly report. The thresholds stabilize a *reported label*, not a
  *portfolio*. Needs an explicit decision (item A7).

**The gap spanning all five phases:** no criterion anywhere asks whether the output is
any good. Phase 3 proved the DP decode is exactly optimal — and it is exactly decoding
a degenerate solution of 6 transitions in 59 years. Phase 5 D-01 and Phase 6 D-16 make
this deliberate, which is a defensible tracer-bullet stance, but it means the project
has **no gate that can fail on a bad model, only on a broken one**. Item A11: make that
a conscious choice re-examined at design freeze.
