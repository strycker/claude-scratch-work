---
phase: 01-monthly-data-layer-long-histories
plan: 02
subsystem: data
tags: [splicing, total-return, par-bond-repricing, treasury, pandas, pytest, tdd]

# Dependency graph
requires:
  - phase: 01-monthly-data-layer-long-histories
    provides: platform package scaffold, checkpoint factory, config loader, platform_settings.yaml (Plan 01-01)
provides:
  - "ratio_splice() — literal-join-date 2-segment splice with continuity guarantee, raises on missing join_date"
  - "bond_price() / monthly_total_return() — par-bond repricing math for CMT-yield-to-total-return synthesis"
  - "build_treasury_tr_synthetic() — chained Treasury total-return index from GS10 yields"
  - "build_equity_total_return() — chained equity total-return index from multpl price + dividend yield"
  - "build_core_research_series() — per-class dispatcher producing the 5 D-03 core research series"
  - "docs/splicing_rules.md — DATA-02/D-04 documentation deliverable, one section per core class"
affects: [phase-2-regime-labeling, phase-3-asset-prediction]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Config-driven splice dispatch: build_core_research_series() reads cfg['splice'][class]['method'] and branches to the matching synthesis function — new classes/methods extend via one dict entry + one dispatch branch, no core logic change"
    - "Literal join_date discipline: ratio_splice() raises ValueError on a missing join_date rather than silently interpolating a scale factor (RESEARCH Pitfall 2)"

key-files:
  created:
    - src/trading_crab_lib/platform/splice.py
    - tests/unit/test_platform_splice.py
    - docs/splicing_rules.md
  modified: []

key-decisions:
  - "Equity total return built from already-scraped multpl price + dividend yield, not a new Shiller fetcher — Shiller's file schema was an unverified RESEARCH assumption (A3); multpl requires zero new ingestion code and is already tested"
  - "Treasury total-return synthetic uses GS10 (10Y CMT yield) via par-bond repricing at semiannual compounding, per the locked D-03 mapping — the resulting GS10-vs-TLT (~20Y+) duration mismatch is accepted and explicitly documented rather than corrected (RESEARCH Open Question 2, resolved)"
  - "Gold and oil use single-source macrotrends passthroughs (no splice) since macrotrends already exceeds the 1962 target span; FRED WTISPLC is recorded in config as an oil cross-check, not blended into the research series"
  - "None of the 5 core classes required a 2-segment ratio_splice in this implementation — all sources exceed 1962 individually (RESEARCH Open Question 3, resolved: default to simplest source, add a segment only if a coverage gap is found)"

requirements-completed: [DATA-02]

coverage:
  - id: D1
    description: "ratio_splice() scales old to match new at a literal join_date with a continuous seam, and raises ValueError if join_date is missing from either input"
    requirement: "DATA-02"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_splice.py::test_splice_continuity_at_join"
        status: pass
      - kind: unit
        ref: "tests/unit/test_platform_splice.py (ratio_splice ValueError cases)"
        status: pass
  - id: D2
    description: "bond_price() and monthly_total_return() implement par-bond repricing; a par bond at its own yield prices to ~1.0, rising yields hurt total return, falling yields help"
    requirement: "DATA-02"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_splice.py (bond_price / monthly_total_return sign tests)"
        status: pass
  - id: D3
    description: "build_core_research_series() returns a DataFrame with exactly the 5 research_name columns from cfg['splice'] (equities_tr, long_duration_tr, gold, oil, cash)"
    requirement: "DATA-02"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_splice.py (build_core_research_series construction tests)"
        status: pass
  - id: D4
    description: "docs/splicing_rules.md documents source, date range, join date, and method per core class (DATA-02/D-04 documentation deliverable)"
    requirement: "DATA-02"
    verification:
      - kind: other
        ref: "python3 -c \"t=open('docs/splicing_rules.md').read().lower(); [t.index(k) for k in ['equities','long_duration','gold','oil','cash','join date','par-bond','1962']]\""
        status: pass
    human_judgment: false

duration: ~35min (across two executor sessions; first interrupted by provider quota limit)
completed: 2026-07-15
status: complete
---

# Phase 1 Plan 02: Splicing Engine + DATA-02 Rulebook Summary

**Par-bond repricing Treasury total-return synthetic + multpl-derived equity total return, dispatched per D-03 core class, with a full DATA-02 splicing-rules doc**

## Performance

- **Duration:** ~35 min total (interrupted mid-plan by a provider quota limit; resumed and completed by a continuation executor)
- **Tasks:** 2/2 complete
- **Files created:** 3

## Accomplishments
- `ratio_splice()` implements the literal-join-date 2-segment splice with a continuity guarantee and a hard `ValueError` on a missing join_date — no silent interpolation of the scale factor (RESEARCH Pitfall 2)
- `bond_price()` + `monthly_total_return()` implement the "par-bond repricing" method for turning CMT Treasury yields (GS10) into a total-return series; `build_treasury_tr_synthetic()` chains it into a cumulative index
- `build_equity_total_return()` constructs an S&P 500 total-return index from the incumbent's already-scraped multpl price + dividend-yield columns
- `build_core_research_series()` dispatches per-class on `cfg['splice'][class]['method']` and assembles all 5 D-03 core research series (`equities_tr`, `long_duration_tr`, `gold`, `oil`, `cash`) in one DataFrame
- `docs/splicing_rules.md` documents, per core class, the source(s), date range, join date, and method — including the multpl-vs-Shiller equity source decision, the Treasury par-bond repricing convention and the accepted GS10-vs-TLT duration mismatch, and the gold/oil macrotrends passthroughs with the WTISPLC cross-check for oil

## Task Commits

Each task was committed atomically (across two executor sessions due to a provider quota interruption):

1. **Task 1 — RED: failing test suite for splice engine** — `fb7225d` (test)
2. **Task 1 — WIP rescue: splice.py implementation (GREEN)** — `6df569b` (wip, rescued mid-write by the quota-kill safety net; contains the completed, passing implementation)
3. **Task 2 — docs/splicing_rules.md rulebook** — `977a9ee` (docs)
4. **Deferred-item logging (out-of-scope test failure, see Deviations)** — `c90cbdf` (chore)

_Note: Task 1's GREEN implementation landed inside the quota-kill rescue commit rather than as a distinct `feat(01-02):` commit — see Deviations._

## Files Created/Modified
- `src/trading_crab_lib/platform/splice.py` — `ratio_splice()`, `bond_price()`, `monthly_total_return()`, `build_treasury_tr_synthetic()`, `build_equity_total_return()`, `build_core_research_series()`
- `tests/unit/test_platform_splice.py` — 17 tests covering splice continuity, join-date validation, par-bond pricing at par, rising/falling-yield sign behavior, and per-class assembly
- `docs/splicing_rules.md` — per-class DATA-02/D-04 rulebook (source, date range, join date, method) + summary table

## Decisions Made
- Equity total return built from multpl price + dividend yield (already scraped, zero new ingestion code) rather than adding a new Shiller fetcher, since Shiller's exact file schema was an unverified RESEARCH assumption
- GS10-vs-TLT duration mismatch accepted per the locked D-03 mapping and documented explicitly rather than corrected — "model the index, trade the ETF" already accepts proxy imperfection
- Gold/oil use single-source macrotrends passthroughs; FRED WTISPLC recorded as an oil cross-check in config, not blended in
- No class required a 2-segment `ratio_splice` — all 5 sources individually exceed the 1962 target span

## Deviations from Plan

### Continuation from a quota-interrupted prior session

**1. [Continuation, not a deviation] Task 1's GREEN implementation was recovered from a WIP safety-net commit**
- **Found during:** Resuming this plan after a prior executor was terminated mid-task by a provider quota limit.
- **What happened:** The RED test suite (`fb7225d`) was committed by the prior executor. The GREEN implementation (`src/trading_crab_lib/platform/splice.py`) was in progress when the quota kill occurred; the orchestrator's safety net rescued the uncommitted file into a `wip: rescue uncommitted executor work` commit (`6df569b`) before the worktree was torn down.
- **Verification this session:** Ran `pytest tests/unit/test_platform_splice.py -x -q` first — all 17 tests passed against the rescued implementation with no changes needed. Read the full `splice.py` file and cross-checked every function against the plan's `<behavior>` and `<acceptance_criteria>` blocks (continuity guard, `ValueError` on missing join_date, par-bond-at-own-yield ≈ 1.0, rising/falling-yield sign behavior, exact 5-column output) — all satisfied as written.
- **Action taken:** None required to the implementation itself; completed the remaining Task 2 (docs) and this SUMMARY.
- **Impact:** No code changes beyond what the interrupted executor had already written and verified working; this session's contribution is the documentation deliverable and verification.

### Out-of-scope discovery (logged, not fixed)

**2. [Scope boundary — logged to deferred-items.md] Pre-existing test failure in a sibling plan's file**
- **Found during:** Running the full incumbent suite (`pytest tests/ -q`) as required by this plan's `<verification>` block.
- **Issue:** `tests/unit/test_platform_prices_ingest.py::test_to_monthly_spine_yields_month_end_frequency` fails (`assert len(monthly) == 3` but got `5`).
- **Why not fixed:** This test file belongs to plan 01-06 (`feat(01-06): daily universe price fetch + monthly spine`), introduced by a separate WIP rescue commit (`60de3b7`) merged into this branch via wave dependencies — it is entirely outside plan 01-02's `files_modified` list (`splice.py`, `test_platform_splice.py`, `docs/splicing_rules.md`).
- **Action taken:** Logged to `.planning/phases/01-monthly-data-layer-long-histories/deferred-items.md` per the scope-boundary rule, committed separately (`c90cbdf`). Not fixed here.

---

**Total deviations:** 0 auto-fixes to this plan's own code; 1 continuation-recovery verification, 1 out-of-scope item logged (not fixed).
**Impact on plan:** No scope creep. Plan 01-02's own deliverables are complete and independently verified.

## Issues Encountered
- Prior executor session was terminated mid-task by a provider quota limit; the orchestrator's rescue mechanism preserved the in-progress `splice.py` as a WIP commit, which this continuation session verified was complete and correct before proceeding to Task 2.
- One pre-existing test failure exists elsewhere on the branch (`test_platform_prices_ingest.py`, plan 01-06's file) — see Deviations above; does not affect this plan's own test suite or acceptance criteria.

## Next Phase Readiness
- `build_core_research_series()` is ready for phases 2-3 (regime labeling, asset prediction) to consume the 5 spliced/synthetic research series once wired into a step function that loads `raw` and `cfg['splice']`.
- `docs/splicing_rules.md` gives future contributors (and future plans extending the universe beyond the 5 core classes) a documented pattern to follow for any new splice/synthesis rule.
- No blockers. The one out-of-scope test failure in `test_platform_prices_ingest.py` should be picked up by whichever plan owns `to_monthly_spine()` (plan 01-06).

---
*Phase: 01-monthly-data-layer-long-histories*
*Completed: 2026-07-15*

## Self-Check: PASSED

All claimed files exist (`src/trading_crab_lib/platform/splice.py`, `tests/unit/test_platform_splice.py`,
`docs/splicing_rules.md`, this SUMMARY.md) and all claimed commit hashes
(`fb7225d`, `6df569b`, `977a9ee`, `c90cbdf`) resolve in `git log`.
