---
phase: 01-monthly-data-layer-long-histories
plan: 04
subsystem: infra
tags: [placeholder, ingestion, documentation, DATA-06]

# Dependency graph
requires:
  - phase: 01-01
    provides: "src/trading_crab_lib/platform/ingestion/ package marker; config/platform_settings.yaml paid_providers list (name/module/note per provider)"
provides:
  - "Three inert paid-provider adapter seams (norgate.py, tiingo.py, eodhd.py) under platform/ingestion/ — each raises NotImplementedError with a docs pointer, no SDK imports, no network calls, no new dependency"
  - "docs/paid_provider_seams.md — provider-by-provider comparison (offering + integration shape) plus stockcharts.com/finviz.com noted as future FEATURE sources, not price sources"
affects: []

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Documentation-only adapter seam: module docstring + fetch_prices(cfg) that immediately raises NotImplementedError pointing to a docs file — reserves an import path without building runtime code (YAGNI-driven placeholder pattern)"

key-files:
  created:
    - src/trading_crab_lib/platform/ingestion/norgate.py
    - src/trading_crab_lib/platform/ingestion/tiingo.py
    - src/trading_crab_lib/platform/ingestion/eodhd.py
    - tests/unit/test_platform_paid_provider_stubs.py
    - docs/paid_provider_seams.md
  modified: []

key-decisions:
  - "Stub module filenames (norgate.py/tiingo.py/eodhd.py) match the module field in config/platform_settings.yaml's paid_providers list exactly, so future config-driven dispatch needs no remapping"
  - "No pyproject.toml/requirements.txt changes — the plan explicitly forbids any new runtime dependency for this documentation-only placeholder"

patterns-established:
  - "Placeholder adapter modules for reserved-but-unbuilt integrations: raise NotImplementedError immediately, document the 'why not now' and integration shape in a dedicated docs file, never import a provider SDK"

requirements-completed: [DATA-06]

coverage:
  - id: D1
    description: "Three paid-provider stub modules import cleanly and raise NotImplementedError with a message pointing to docs/paid_provider_seams.md when fetch_prices() is called"
    requirement: "DATA-06"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_paid_provider_stubs.py — TestPaidProviderStubsImportCleanly (3 tests) + TestPaidProviderStubsRaiseNotImplemented (parametrized, 3 cases)"
        status: pass
    human_judgment: false
  - id: D2
    description: "No provider SDK / network imports in any stub module, and no new runtime dependency added to pyproject.toml or requirements files"
    requirement: "DATA-06"
    verification:
      - kind: unit
        ref: "grep -rn \"import requests|import norgatedata|tiingo|eodhd\" on the three stub .py files — zero matches; git diff on pyproject.toml/requirements*.txt across both commits — empty"
        status: pass
    human_judgment: false
  - id: D3
    description: "docs/paid_provider_seams.md documents Norgate, Tiingo, EODHD as v2 placeholder seams and notes stockcharts.com/finviz.com as future FEATURE sources, not price sources"
    requirement: "DATA-06"
    verification:
      - kind: manual_procedural
        ref: "docs/paid_provider_seams.md read in full during close-out verification — contains 'Provider seams' section for all three, plus a dedicated 'Feature sources (not price sources)' section naming stockcharts.com/finviz.com"
        status: pass
    human_judgment: false

# Metrics
duration: N/A (close-out of interrupted execution; implementation work was completed by a prior executor session before a provider quota limit terminated it mid-verification)
completed: 2026-07-15
status: complete
---

# Phase 1 Plan 04: Paid-Provider Adapter Seams Summary

**Three documentation-only stub modules (Norgate, Tiingo, EODHD) that raise `NotImplementedError` from `fetch_prices()`, plus `docs/paid_provider_seams.md` recording provider comparison notes and flagging stockcharts.com/finviz.com as future feature (not price) sources — zero new runtime dependencies.**

## Performance

- **Tasks:** 2/2 completed
- **Files created:** 5 (3 stub modules, 1 test file, 1 doc file)
- **Files modified:** 0

## Accomplishments
- `norgate.py`, `tiingo.py`, `eodhd.py` under `src/trading_crab_lib/platform/ingestion/` — each imports cleanly, performs no I/O, and its single public `fetch_prices(cfg) -> pd.DataFrame` entry point immediately raises `NotImplementedError` with a message directing readers to `docs/paid_provider_seams.md`
- `tests/unit/test_platform_paid_provider_stubs.py` verifies all three modules import cleanly (`hasattr(module, "fetch_prices")`) and that calling `fetch_prices({})` raises `NotImplementedError` with a non-empty message referencing the docs file
- `docs/paid_provider_seams.md` documents what each provider offers, its integration shape (Norgate: subscription desktop DB; Tiingo/EODHD: REST API), and explicitly marks stockcharts.com/finviz.com as candidate feature sources for a later milestone, not price sources for this phase

## Task Commits

Both tasks were committed atomically by the prior executor session (verified present on this branch before close-out began):

1. **Task 1: Three provider stub modules + stub test** - `fdf20f1` (feat)
2. **Task 2: docs/paid_provider_seams.md — provider seam + feature-source notes** - `18c3e5c` (docs)

## Files Created/Modified
- `src/trading_crab_lib/platform/ingestion/norgate.py` - Norgate Data adapter seam stub; `fetch_prices(cfg)` raises `NotImplementedError`
- `src/trading_crab_lib/platform/ingestion/tiingo.py` - Tiingo adapter seam stub; `fetch_prices(cfg)` raises `NotImplementedError`
- `src/trading_crab_lib/platform/ingestion/eodhd.py` - EODHD adapter seam stub; `fetch_prices(cfg)` raises `NotImplementedError`
- `tests/unit/test_platform_paid_provider_stubs.py` - Import-cleanliness + `NotImplementedError` message assertions for all three stubs
- `docs/paid_provider_seams.md` - Provider comparison table, integration-shape notes, v2 adoption checklist, and stockcharts.com/finviz.com feature-source note

## Decisions Made
None new — this close-out verified the prior executor's work against the plan's acceptance criteria rather than making implementation decisions. Two decisions from the original execution are recorded above under `key-decisions` (module-name/config alignment; no new dependency).

## Deviations from Plan

**Provider quota interruption (process deviation, not a code deviation).** The executor that performed the implementation work (both commits `fdf20f1` and `18c3e5c`) was terminated by a provider quota limit while running final verification, before it could write `01-04-SUMMARY.md` or confirm the full test suite. No code was left uncommitted — both commits were already on this branch when the close-out executor resumed. This close-out session:
1. Re-verified the worktree branch/base assertion.
2. Ran `pytest tests/unit/test_platform_paid_provider_stubs.py -q` — 6 passed.
3. Confirmed no SDK/network imports in the three stub files (`grep` for `import requests|import norgatedata|tiingo|eodhd` — zero matches) and no diff to `pyproject.toml`/`requirements*.txt` across both commits.
4. Confirmed the stub module names match `config/platform_settings.yaml`'s `paid_providers` list (`norgate.py`, `tiingo.py`, `eodhd.py`) — this list was authored in Plan 01-01.
5. Confirmed `docs/paid_provider_seams.md` documents all three providers and the stockcharts.com/finviz.com feature-source note per `01-CONTEXT.md`.
6. Ran the full incumbent suite (`pytest tests/ -q`): 774 passed, 48 skipped, 1 failed. The one failure — `tests/unit/test_platform_prices_ingest.py::test_to_monthly_spine_yields_month_end_frequency` — is in a file this plan does not touch (owned by Plan 01-06's daily-universe/monthly-spine work, committed separately as `9203629` by a sibling parallel executor). Per the executor's scope boundary rule, out-of-scope failures in files this plan doesn't own are not auto-fixed; logging it here for the orchestrator/Plan 01-06 owner rather than touching that file.

No genuine defects were found in Plan 01-04's own files. No re-implementation was performed.

## Issues Encountered
None within this plan's scope. (See the out-of-scope test failure noted above under Deviations.)

## User Setup Required
None - no external service configuration required. All three providers remain unimplemented placeholders; no API keys or credentials are needed until a future v2 milestone builds a real adapter.

## Next Phase Readiness
- DATA-06 is satisfied: the three paid-provider seams are documented, inert, and reserved for a future v2 milestone.
- `config/platform_settings.yaml`'s `paid_providers` list (from Plan 01-01) now has a corresponding, verified stub module for each entry.
- One pre-existing, out-of-scope test failure (`test_platform_prices_ingest.py::test_to_monthly_spine_yields_month_end_frequency`) exists on this branch from Plan 01-06's work and should be tracked/fixed by that plan's owner — it does not block this plan's completion.

---
*Phase: 01-monthly-data-layer-long-histories*
*Completed: 2026-07-15*
