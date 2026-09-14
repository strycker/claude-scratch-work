---
phase: 07-regime-representation
plan: 02
subsystem: platform-data
tags: [checkpoint-recompute, holdout, feature-policy, regression-pin, D-02-A]

# Dependency graph
requires:
  - phase: 07-regime-representation
    provides: "Plan 07-01's frozen driver/reference L1 feature-column mechanism (report.py::_reference_label_columns, threaded frozen_l1_features), whose real-checkpoint equivalence test this plan re-runs against a changed column count."
provides:
  - "An offline, network-free recompute entry point (scripts/recompute_monthly_features.py) that rebuilds the dev monthly_features checkpoint from the cached monthly_raw checkpoint without ever calling build_monthly_spine()."
  - "A corrected dev monthly_features checkpoint (708 x 53, oil at 708 non-NaN dev months instead of 431) and its re-carved holdout counterpart (68 rows, all post-2020-12-31)."
  - "The frozen L1 reference set grown from 9 to 10 columns (oil the sole addition), verified via report.py::_reference_label_columns itself."
  - "The A13 golden change-point regression constant re-pinned to the exact post-recompute sequence, with a dated D-02-A comment establishing this repo's first re-pin convention."
  - "A reproduced (not quoted) pre-fix labeling-disagreement baseline (07-PREFIX-EVIDENCE.md), captured before the recompute, with an explicit compound-baseline warning for plan 07-04's pre/post table."
affects: ["07-03 (policy trials — the frozen set these trials evaluate is now the corrected 10-column set)", "07-04 (ADR-0001 — cites this plan's 9->10 frozen-set correction and the compound pre-fix baseline in the pre/post table)"]

# Actuals (#2632)
actuals:
  tokens: 6300
  tasks: 3
  commits: 2

tech-stack:
  added: []
  patterns:
    - "Golden-constant re-pin convention (new to this repo, per 07-PATTERNS.md finding #3): a dated block comment above the constant naming the authorizing decision (D-02-A), the root cause, and an explicit statement that the exact list-equality assertion is unchanged — establishing how future data corrections should re-pin a regression pin without softening it."
    - "Offline recompute-from-cache script pattern: a pure frame-builder helper (rebuild_monthly_features) separated from a main() that does I/O, so the single most safety-critical assertion (no raw column dropped) is unit-testable on synthetic data without touching the real checkpoint tree."

key-files:
  created:
    - scripts/recompute_monthly_features.py
    - tests/unit/test_platform_recompute_monthly_features.py
    - .planning/phases/07-regime-representation/07-PREFIX-EVIDENCE.md
  modified:
    - tests/unit/test_platform_plotting_regime.py
    - data/checkpoints/platform/monthly_features.parquet
    - data/checkpoints/platform/monthly_features.meta.json
    - data/holdout/monthly_features.parquet
    - data/holdout/monthly_features.meta.json

key-decisions:
  - "D-02-A (07-CONTEXT.md, applied): recomputed monthly_features offline from the cached monthly_raw rather than a full network rebuild — build_platform_data.py / build_monthly_spine() were deliberately not used or extended, since macrotrends/stooq egress is blocked in this container and a re-ingest would corrupt or fail rather than merely be slow."
  - "07-PATTERNS.md's recompute sketch (write_monthly_features_split(compute_lean_features(raw, cfg), ...)) was NOT followed — it is the exact 13-column bug this plan's Test 1 exists to catch. rebuild_monthly_features() instead reproduces build_monthly_spine's pd.concat([monthly_raw, lean], axis=1) + duplicate-column dedupe (keep='last') tail exactly, preserving all 53 columns."
  - "The pre-fix evidence (Task 1) was captured and committed BEFORE the recompute (Task 2/3), per the plan's sequencing requirement — outputs/ and data/ were verified untouched by that task via git status."

requirements-completed: [REG-01]

coverage:
  - id: D1
    description: "rebuild_monthly_features(monthly_raw, cfg) reproduces build_monthly_spine's tail exactly (compute_lean_features -> concat -> dedupe keep='last' -> index.name='date'), preserving every raw column — proven on synthetic data carrying both lean-source and non-lean columns."
    requirement: "REG-01"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_recompute_monthly_features.py::TestRebuildMonthlyFeatures::test_rebuild_frame_keeps_every_raw_column"
        status: pass
      - kind: unit
        ref: "tests/unit/test_platform_recompute_monthly_features.py::TestRebuildMonthlyFeatures::test_passthrough_columns_are_deduped_keeping_the_lean_copy"
        status: pass
      - kind: unit
        ref: "tests/unit/test_platform_recompute_monthly_features.py::TestRebuildMonthlyFeatures::test_dev_side_stops_at_the_cutoff"
        status: pass
      - kind: unit
        ref: "tests/unit/test_platform_recompute_monthly_features.py::TestRebuildMonthlyFeatures::test_index_name_is_date"
        status: pass
    human_judgment: false
  - id: D2
    description: "The script imports no network-capable module (fredapi/yfinance/requests/curl_cffi/trading_crab_lib.platform.ingestion.*) and never references build_monthly_spine, and every trading_crab_lib import is confined to trading_crab_lib.platform.* — proven by static AST inspection of the script's own source, not by mocking."
    requirement: "REG-01"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_recompute_monthly_features.py::TestScriptStaticImports::test_no_network_module_is_imported"
        status: pass
      - kind: unit
        ref: "tests/unit/test_platform_recompute_monthly_features.py::TestScriptStaticImports::test_imports_only_platform_and_stdlib"
        status: pass
    human_judgment: false
  - id: D3
    description: "On the real cached monthly_raw checkpoint, the recompute is shape-preserving (708x53 dev rows before and after) and changes exactly one column's non-NaN count: oil, 431 -> 708. Every other column is byte-identical, confirming D-02-A's premise held and nothing else in monthly_raw drifted since plan time."
    requirement: "REG-01"
    verification:
      - kind: manual_procedural
        ref: "python scripts/recompute_monthly_features.py --dry-run (logged delta showed exactly one changed column, oil 431->708); python scripts/recompute_monthly_features.py (real write); post-write assertion script confirmed shape==(708,53), oil non-NaN==708, index.max()=='2020-12-31'"
        status: pass
    human_judgment: false
  - id: D4
    description: "The frozen L1 reference set (report.py::_reference_label_columns) resolves to exactly 10 columns on the recomputed checkpoint, with oil the sole addition to the pre-fix 9."
    requirement: "REG-01"
    verification:
      - kind: manual_procedural
        ref: "Direct call to _reference_label_columns(feat, lean_cols, first_decision) against the post-recompute checkpoint returned the 10-column list with 'oil' present and every pre-fix column unchanged."
        status: pass
    human_judgment: false
  - id: D5
    description: "The A13 golden change-point constant is re-pinned to the exact post-recompute sequence (counts +1 across 5 of 7 dates, dates unchanged), with a dated D-02-A comment, and the assertion remains exact list equality (never loosened to an inequality/subset/tolerance). n_active.min() lower bound raised from >=4 to >=5. Lands in the SAME commit as the checkpoint write."
    requirement: "REG-01"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_plotting_regime.py::TestActiveFeatureCountTimeline::test_real_dev_features_reproduce_the_seven_a13_change_points"
        status: pass
    human_judgment: false
  - id: D6
    description: "Plan 07-01's TestFrozenPolicyEquivalence (6 tests, including the real-checkpoint variant) is re-run after the recompute, unmodified, and still passes on the now-10-column checkpoint — evidence the driver/reference equivalence mechanism is structurally invariant to which columns are frozen."
    requirement: "REG-01"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_backtest_driver.py::TestFrozenPolicyEquivalence (all 6 tests)"
        status: pass
    human_judgment: false
  - id: D7
    description: "The pre-fix labeling-disagreement baseline (82.8%, n_compared=470, n_disagree=389) plus the nine pre-fix frozen columns and seven pre-fix A13 change points are reproduced from persisted artifacts (not quoted) and recorded with an explicit compound-baseline statement, BEFORE the recompute overwrites the checkpoint."
    requirement: "REG-01"
    verification:
      - kind: manual_procedural
        ref: ".planning/phases/07-regime-representation/07-PREFIX-EVIDENCE.md; reproduction script re-run independently confirmed n_compared=470, n_disagree=389, pct_disagree=0.8276595744680851, first_common_date=1974-02-28, last_common_date=2020-12-31"
        status: pass
    human_judgment: false

duration: ~50min
completed: 2026-09-14
status: complete
---

# Phase 7 Plan 2: Recompute the Dev Feature Checkpoint and Re-Pin A13 Summary

**The dev `monthly_features` checkpoint is now a current, offline-reproducible function of the cached `monthly_raw` checkpoint — `oil`'s stale 1985-02 truncation is gone, the frozen L1 reference set grows from 9 to 10 columns, and the A13 golden regression constant is re-pinned to the exact new sequence in the same commit as the data that moved it.**

## Performance

- **Duration:** ~50 min
- **Started:** 2026-09-14T22:52:30Z (approx, session start after 07-01)
- **Completed:** 2026-09-14T23:02:43Z
- **Tasks:** 3
- **Files created:** 3 (`scripts/recompute_monthly_features.py`, `tests/unit/test_platform_recompute_monthly_features.py`, `07-PREFIX-EVIDENCE.md`)
- **Files modified:** 4 (`tests/unit/test_platform_plotting_regime.py`, plus the checkpoint parquet/meta pairs for dev and holdout)

## Accomplishments

- **Task 1 — preserved the pre-fix evidence before anything was overwritten.** Reproduced (not quoted) the 82.8% labeling-disagreement baseline directly from `outputs/reports/platform/backtest_full_sample_states.parquet` and `backtest_filtered_state_probs.parquet` via `label_disagreement()`, verifying exactly `n_compared=470`, `n_disagree=389`, `pct_disagree=0.8276595744680851`, `first_common_date=1974-02-28`, `last_common_date=2020-12-31`. Named the `state_N`-string-to-int coercion trap explicitly (skipping it silently returns `n_compared=0`, not an error). Recorded the nine pre-fix frozen columns, the seven pre-fix A13 change points, the artifact vintage (`27fe529` / 2026-09-11), and an explicit compound-baseline statement (pre-freeze driver policy AND stale 9-column checkpoint, so no single cause explains the eventual delta). `git status --porcelain outputs/ data/` was empty after this task, confirmed.
- **Task 2 — built and ran the offline recompute.** `scripts/recompute_monthly_features.py`'s `rebuild_monthly_features(monthly_raw, cfg)` reproduces `build_monthly_spine`'s tail exactly (`compute_lean_features` → `pd.concat([monthly_raw, lean], axis=1)` → dedupe `keep="last"` → `index.name = "date"`), never calling `build_monthly_spine()` itself. Wrote 6 tests first (RED confirmed via `ModuleNotFoundError` before the script existed), all GREEN after: raw-column preservation, passthrough dedupe correctness, holdout-boundary split behavior, index naming, and two static AST checks (no network-capable import, no `build_monthly_spine` reference, and no non-platform `trading_crab_lib` import). `--dry-run` logged the exact expected delta (`oil: 431 -> 708`, shape unchanged at `(708, 53)`) before the real write; `git status --porcelain data/` was empty after the dry run. The real write produced dev `(708, 53)` with `oil` at 708 non-NaN months, index max `2020-12-31`, and a holdout side of 68 rows all dated after `2020-12-31` — all verified against the actual checkpoint after the write.
- **Task 3 — re-pinned the A13 golden constant, exactly, in the same commit.** `EXPECTED_CHANGE_POINTS` moved from `4,6,8,9,10,12,13` to `5,7,9,10,11,12,13` across the same seven dates, with a dated block comment naming D-02-A, the root cause (stale `oil` truncation), and an explicit statement that the exact list-equality assertion is unchanged — establishing this repo's first documented golden-constant re-pin convention (per `07-PATTERNS.md` finding #3, no prior precedent existed). The `n_active.min()` bound moved from `>= 4` to `>= 5`. The a13 test now reports **1 passed** (not skipped — the real checkpoint is present). Plan 07-01's `TestFrozenPolicyEquivalence` (6 tests, including the real-checkpoint variant) was re-run unmodified against the new 10-column checkpoint and passed unchanged, demonstrating the equivalence mechanism generalizes rather than having only worked by coincidence at 9 columns.
- Checkpoint and test-file changes landed in **one commit** (`9496476`), satisfying D-02-A's requirement that the golden test move in the same commit as the data that falsified it — verified via `git log -1 --name-only`.

## Task Commits

1. **Task 1: Preserve the pre-fix evidence before anything is overwritten** — `cf6dadb` (docs)
2. **Tasks 2 + 3: Offline recompute of the dev checkpoint + A13 re-pin (same commit, per D-02-A)** — `9496476` (feat)

## Files Created/Modified

- `scripts/recompute_monthly_features.py` (created) — `rebuild_monthly_features(monthly_raw, cfg)` pure helper + `main()` with `--dry-run`, per-column delta logging, and a narrowing-refusal guard.
- `tests/unit/test_platform_recompute_monthly_features.py` (created) — 6 tests: 4 on synthetic-frame behavior, 2 static AST checks on the script's own source.
- `.planning/phases/07-regime-representation/07-PREFIX-EVIDENCE.md` (created) — reproduced pre-fix baseline, recipe, nine pre-fix columns, seven pre-fix change points, artifact vintage, compound-baseline statement.
- `tests/unit/test_platform_plotting_regime.py` (modified) — `EXPECTED_CHANGE_POINTS` re-pinned with a dated D-02-A comment; `n_active.min()` bound raised to `>= 5`.
- `data/checkpoints/platform/monthly_features.parquet` + `.meta.json` (modified) — recomputed dev checkpoint, 708 rows × 53 columns, `oil` at 708 non-NaN months.
- `data/holdout/monthly_features.parquet` + `.meta.json` (modified) — re-carved holdout checkpoint, 68 rows, all dated after 2020-12-31.

## Verified Numbers (measured directly, not assumed)

| Fact | Before recompute | After recompute |
|---|---|---|
| dev `monthly_features` shape | 708 × 53 | 708 × 53 (unchanged) |
| `oil` dev non-NaN months | 431 (first valid 1985-02-28) | **708** (first valid 1962-01-31) |
| Only column that changed | — | `oil` (the delta log listed exactly one column) |
| dev index max | 2020-12-31 | 2020-12-31 (unchanged) |
| holdout rows | 68 | 68 (unchanged, all > 2020-12-31) |
| frozen L1 reference set (`_reference_label_columns`) | 9 columns | **10 columns** (`oil` added) |
| A13 change points | `4,6,8,9,10,12,13` at the seven dates | `5,7,9,10,11,12,13` at the **same** seven dates |
| `TestFrozenPolicyEquivalence` (07-01, 6 tests) | n/a (07-01 landed on 9-col checkpoint) | all 6 pass, unmodified, on the 10-col checkpoint |

**New frozen 10-column set (order as returned by `_reference_label_columns`):**
`cape_shiller`, `credit_spread_baa_aaa`, `curve_10y3m`, `div_yield`, `oil`, `real_rate_level`, `realized_vol_1m`, `realized_vol_3m`, `trailing_return_1m`, `trailing_return_3m`.

## Full Suite

| | Before this plan | After this plan |
|---|---|---|
| Passed | 1714 | **1720** |
| Skipped | 0 | **0** |

The +6 delta is exactly this plan's new `test_platform_recompute_monthly_features.py` tests. No regressions, no new skips.

## Deviations from Plan

None — plan executed exactly as written. `07-PATTERNS.md`'s recompute sketch (`write_monthly_features_split(compute_lean_features(raw, cfg), ...)`) was consciously NOT followed, per the plan's own warning that PATTERNS is advisory and known-wrong on this exact task; the real assembly (`rebuild_monthly_features`) reproduces `build_monthly_spine`'s tail instead. This is not a deviation from the plan — the plan explicitly named this discrepancy in advance and instructed following the code, not PATTERNS.

## Known Stubs

None.

## Threat Flags

None. All four `mitigate`-disposition threats in this plan's STRIDE register (T-07-06 through T-07-09) were closed by automated assertions already described above (Test 1's raw-column-preservation check, `assert_dev_checkpoint_within_boundary`, the single-column-delta log, and the exact-list-equality A13 pin). No new security-relevant surface was introduced.

## Issues Encountered

None. Every plan-time-verified number (checkpoint shapes, `oil` counts, the A13 sequence, the disagreement baseline) reproduced exactly as `07-02-PLAN.md` predicted — the underlying data had not drifted since plan time.

## User Setup Required

None.

## Next Phase Readiness

- Plan 07-03 (policy trials) can now run against the corrected 10-column frozen set.
- Plan 07-04 (ADR-0001) can cite this plan's exact before/after numbers and `07-PREFIX-EVIDENCE.md`'s compound-baseline statement for its pre/post table — remembering that the pre-fix baseline reflects BOTH the pre-D-01 expanding driver policy AND the stale 9-column checkpoint, so the eventual delta against post-fix numbers must not be attributed to either cause alone.
- No blockers. Wave 1 (`D-09`) continues with plan 07-03.

---
*Phase: 07-regime-representation*
*Completed: 2026-09-14*

## Self-Check: PASSED

All 3 created files and 4 modified files confirmed present on disk; both task commits (`cf6dadb`, `9496476`) confirmed present in git history via `git log --oneline`; the checkpoint's actual on-disk shape and `oil` non-NaN count were re-verified against the values recorded above.
