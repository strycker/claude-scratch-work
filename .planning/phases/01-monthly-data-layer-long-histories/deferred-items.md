# Deferred Items — Phase 1 (out of scope for individual plans)

**ALL RESOLVED as of 2026-08-04.** No open items remain for Phase 1.

## Deferred Items

- `tests/unit/test_platform_prices_ingest.py::test_to_monthly_spine_yields_month_end_frequency` failed
  during plan 01-02 execution (`assert len(monthly) == 3` but got `5`).
  status: resolved
  resolved_at: 2026-08-04
  resolution: |
    The test now passes. Verified by running `pytest tests/unit/test_platform_prices_ingest.py`
    on `main` — 11 passed, including this test. It was fixed by plan 01-06, which owns
    `to_monthly_spine()`, during that plan's own execution; this entry was simply never
    marked closed at the time.
  original_report: |
    This test file belongs to plan 01-06 (`feat(01-06): daily universe price fetch +
    monthly spine`), not plan 01-02 — it is outside 01-02's `files_modified` list
    (`src/trading_crab_lib/platform/splice.py`, `tests/unit/test_platform_splice.py`,
    `docs/splicing_rules.md`) and was introduced by a WIP rescue commit (`60de3b7`)
    merged in via wave dependencies. Per the executor scope boundary, this was logged
    rather than fixed. Suspected cause at the time (untriaged): business-day index
    resample to month-end producing 5 buckets instead of 3 for a ~90-business-day window
    spanning Jan–May 2020.
