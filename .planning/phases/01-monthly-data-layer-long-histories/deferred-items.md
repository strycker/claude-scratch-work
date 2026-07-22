# Deferred Items — Phase 1 (out of scope for individual plans)

## From Plan 01-02 execution

- **`tests/unit/test_platform_prices_ingest.py::test_to_monthly_spine_yields_month_end_frequency` fails.**
  `assert len(monthly) == 3` but got `5`. This test file belongs to plan 01-06
  (`feat(01-06): daily universe price fetch + monthly spine`), not plan 01-02 — it is
  outside 01-02's `files_modified` list (`src/trading_crab_lib/platform/splice.py`,
  `tests/unit/test_platform_splice.py`, `docs/splicing_rules.md`) and was introduced by
  a WIP rescue commit (`60de3b7`) merged in via wave dependencies. Per the executor scope
  boundary, this was logged rather than fixed. Likely cause (untriaged): business-day
  index resample to month-end producing 5 buckets instead of 3 for a ~90-business-day
  window spanning Jan–May 2020 — needs investigation by whichever plan owns
  `to_monthly_spine()`.
