---
phase: 08-regime-persistence-stability
plan: 15
subsystem: platform/report (weekly page, §3 Per-Asset Signals)
tags: [gap-closure, VERIFY-H4, neutral-posture, display, honesty, no-registry, tdd, tracer]
status: complete
requires:
  - "08-09 (assemble_weekly_report takes the hysteresis OUTPUT active_regime; None = neutral posture reachable)"
provides:
  - "weekly._NEUTRAL_PER_ASSET_SENTENCE: what §3 prints in neutral posture in place of rows"
  - "assemble_weekly_report §3: rows narrowed to active_regime only, each row names its regime ('- SPY (regime 3): mean=...')"
  - "tests/unit/test_platform_report_weekly.py::TestPerAssetSignalsFollowTheActiveRegime (6 items)"
affects:
  - "08-19 (regenerates the real-data page after every gap-closure fix lands)"
  - "verification human item 4 (re-check on the regenerated page)"
tech-stack:
  added: []
  patterns:
    - "one selector (active_regime) drives both §2 Trajectory and §3 Per-Asset Signals; both say so in neutral posture"
key-files:
  created:
    - .planning/phases/08-regime-persistence-stability/08-15-SUMMARY.md
  modified:
    - src/trading_crab_lib/platform/report/weekly.py
    - tests/unit/test_platform_report_weekly.py
    - CLAUDE.md
    - README.md
decisions:
  - "Neutral posture prints one sentence and no per-asset rows (planner's choice; the page's own A7 sentence says the active regime selects the rows). Labelled all-regime rows in neutral posture were NOT implemented and are an open question for Glenn."
  - "Every printed row carries its regime id, so no row is readable without its regime in any posture."
metrics:
  duration: "~15 min"
  completed: 2026-09-29
estimate:
  tokens: 40000
  tasks: 2
actuals:
  tokens: 2500    # chars/4 over the realized diff (9867 chars), be1d7ba..4d852f2
  tasks: 2
  commits: 3
---

# Phase 8 Plan 15: Neutral-posture Per-Asset Signals Summary

In neutral posture the weekly page no longer prints 24 unlabelled, contradictory per-asset rows. §3 now prints one sentence saying no regime is active. When a regime is active it prints only that regime's rows, and each row names its regime id.

## Root cause (confirmed)

`assemble_weekly_report` §3 started from the whole `returns_by_regime` table. It narrowed to `regime == active_regime` only when `active_regime is not None`, and printed each row as `- {asset}: mean=... sharpe=... n_obs=...`, with no regime id. That was unambiguous only while the table was always narrowed to one regime. Before 08-09 (`036ae74`), `active_regime` was `probs.idxmax()` and was never None. 08-09 correctly switched to the hysteresis output, which made None reachable. On today's data None is the live case: no belief component clears the 0.70 act threshold. None then fell through to "no narrowing", so every (regime, asset) row was printed unlabelled.

## Before-state (real-data page, read only)

Page: `scratchpad/serve/out/reports/platform/weekly_report.md`, mtime 2026-09-29 17:52:29 UTC, as-of 2026-06-30, `- active regime: none (neutral posture)`.

- The Per-Asset Signals section held **24 rows**, which is 4 assets × 6 regimes.
- Occurrences per asset: **SPY 6, TLT 6, IAU 6, USO 6**. The verifier wrote "SPY four times"; the page shows six.
- Contradictory example: `- SPY: mean=-4.22% sharpe=-2.21 n_obs=40` and `- SPY: mean=2.17% sharpe=1.97 n_obs=72`, with nothing saying which regime either belongs to.

The page was not regenerated here; 08-19 owns that.

## Tasks

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 (tracer, TDD) | RED: per-asset rows follow the active regime | 999578c | tests/unit/test_platform_report_weekly.py |
| 1 (tracer, TDD) | GREEN: neutral sentence + regime id on every row | 85a207e | src/trading_crab_lib/platform/report/weekly.py |
| 2 | Recorded counts 2459 → 2465 (D-07) | 4d852f2 | CLAUDE.md, README.md |

## RED on HEAD (commit 999578c, before the fix)

4 failed, 2 passed. `[0]` and `[1]` of the parametrized at-most-once arm pass on HEAD by design, because HEAD already narrowed a non-None regime.

- `test_neutral_posture_prints_no_per_asset_row_and_says_why`: `AssertionError: assert ['- SPY: mean....66 n_obs=30'] == []`, "Left contains 4 more items, first extra item: '- SPY: mean=1.23% sharpe=1.11 n_obs=30'"
- `test_no_asset_appears_twice_in_any_posture[None]`: `AssertionError: ['- SPY: mean=1.23% sharpe=1.11 n_obs=30', '- TLT: mean=0.45% sharpe=0.44 n_obs=30', '- SPY: mean=-3.21% sharpe=-2.22 n_obs=30', '- TLT: mean=1.67% sharpe=1.66 n_obs=30']` / `assert 2 <= 1`
- `test_active_posture_rows_are_that_regimes_and_name_it`: "At index 0 diff: '- SPY: mean=-3.21% sharpe=-2.22 n_obs=30' != '- SPY (regime 1): mean=-3.21% sharpe=-2.22 n_obs=30'"
- `test_main_writes_the_neutral_sentence_to_the_page` (tracer): `AssertionError: assert ['- SPY: mean....66 n_obs=30'] == []`, "Left contains 4 more items"

## GREEN (commit 85a207e)

- `TestPerAssetSignalsFollowTheActiveRegime`: 6 passed.
- `pytest tests/unit/test_platform_report_weekly.py tests/unit/test_platform_report_serving.py -q`: **66 passed**. The existing tests are unmodified, including `test_flags_low_confidence_cells_below_min_obs_flag` and 08-09's `TLT in signals and SPY not in signals`.
- Tracer gate: the tracer `<verify>` re-ran end to end, green (66 passed), before Task 2.

## Mutation arms (each applied to weekly.py, run against the new class, then restored; `cmp` against the saved fixed copy = identical)

| Arm | Mutation | Result |
|-----|----------|--------|
| M1 | restore the unnarrowed fall-through for None | 3 failed: neutral, at-most-once[None], main() tracer |
| M2 | drop `(regime {row.regime})` from the row format | 1 failed: active arm |
| M3 | narrow None to the table's first regime (silent default) | 2 failed: neutral, main() tracer |

Each arm turned at least one of its planned tests red. After the arms, `git diff --stat` showed only the intended edit.

## Verification

- Live collection `pytest --collect-only -q tests/` = **2465**, which is 2459 + 6 new items (4 tests, one parametrized ×3). It is written at all four sites: CLAUDE.md:128, CLAUDE.md:606, README.md:5 (badge) and README.md:477. `test_docs_recorded_counts.py`: 7 passed.
- Full suite `pytest tests/ -q -p no:cacheprovider`: **2465 passed**, 0 failed, 0 skipped, 0 xfailed (5 warnings), 273 s.
- Legacy-import ratchet: 11 passed; `MAX_LEGACY_IMPORT_SITES = 31` unchanged.
- ruff: clean. flake8 `--select=E9,F63,F7,F82`: clean on weekly.py and the test file.
- Registry: `total_trial_count() == 44`, `registry/trials.jsonl` sha256 prefix **c957e8fdb360**, byte-identical (ADR-0004 budget 0).
- `git diff --quiet b7193fd -- registry/ legacy/ gsd-scratch-work/ trading-crab-lib/` holds. `git diff --stat b7193fd -- src/` lists only `weekly.py`. No tracked `data/` or `outputs/` file was touched.

## Deviations from Plan

None to the code or tests; the plan was executed as written.

- Process note: auto mode is off (`workflow.auto_advance` false), so the generic interactive tracer gate would have stopped at a human-verify checkpoint after Task 1. I continued without stopping, for three reasons: the orchestrator asked for the whole plan with a completion report, the plan is `autonomous: true`, and the tracer's `<verify>` re-ran green end to end.

## Open question for Glenn (not implemented)

In neutral posture, should §3 print all regimes' rows instead, grouped and labelled per regime (for example, a `### Regime k` block each)? That reading treats the section as reference data rather than the active regime's signal. The page's A7 sentence currently says the active regime selects these rows, so this plan prints the neutral sentence instead. Either choice keeps a regime id on every row.

## Known Stubs

None.

## Self-Check: PASSED

- FOUND: src/trading_crab_lib/platform/report/weekly.py (`_NEUTRAL_PER_ASSET_SENTENCE`)
- FOUND: tests/unit/test_platform_report_weekly.py (`TestPerAssetSignalsFollowTheActiveRegime`)
- FOUND commits: 999578c, 85a207e, 4d852f2
