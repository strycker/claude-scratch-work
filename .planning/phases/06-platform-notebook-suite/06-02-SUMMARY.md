---
phase: 06-platform-notebook-suite
plan: 02
subsystem: platform-evaluation-artifacts
tags: [parquet, reproducibility, blas, floating-point, A13, honesty-framework]

# Dependency graph
requires:
  - phase: 05-backtest-evaluation
    provides: platform/evaluation/report.py::run_full_backtest_evaluation, write_backtest_report
provides:
  - "platform/evaluation/report.py: run_full_backtest_evaluation() persists two additional artifacts — full_sample_states (backtest_full_sample_states.parquet, one `state` int64 column, DatetimeIndex 1963-02-28..2020-12-31) and filtered_state_probs (backtest_filtered_state_probs.parquet, state_0..state_4 float64 columns, DatetimeIndex 1974-02-28..2020-12-31) — additively, at the existing write_backtest_report() call site."
  - "outputs/reports/platform/backtest_full_sample_states.parquet and backtest_filtered_state_probs.parquet — live artifacts, written and verified."
affects: [06-03-features-taxonomy-regime-labeling-notebook, 06-07-backtest-notebook]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Same-file additive dict-literal extension (06-PATTERNS.md 'platform/evaluation/report.py MODIFIED — additive only')"
    - "Same-session code-before/code-after A/B regeneration as the correct way to prove 'additive-only' under non-reproducible BLAS builds — cross-session SHA-256 comparison against on-disk artifacts is the wrong baseline for that claim."

key-files:
  created: []
  modified:
    - src/trading_crab_lib/platform/evaluation/report.py
    - tests/unit/test_platform_evaluation_report.py
    - outputs/reports/platform/backtest_full_sample_states.parquet
    - outputs/reports/platform/backtest_filtered_state_probs.parquet

key-decisions:
  - "Task 1 landed exactly as specified: two new artifacts dict keys (full_sample_states, filtered_state_probs) added at the single write_backtest_report() call site; write_backtest_report() itself untouched; git diff confined to 30 insertions / 2 deletions (both deletions are the two doc/comment lines being extended, per acceptance criteria)."
  - "The byte-identity criterion's correct baseline is SAME-SESSION code-before vs code-after, not cross-session comparison against artifacts already on disk. My first attempt compared a fresh regeneration against artifacts built in an earlier session on a different BLAS DYNAMIC_ARCH kernel selection, which measures the environment, not the code change, and produced 5/7 apparent mismatches at the 10th-16th significant digit. Re-run as a same-session A(post-change)/B(pre-change, report.py reverted to 8eafd98) pair with registry_path overridden: all seven shared artifacts are BYTE-IDENTICAL at SHA-256 between A and B. The additive-only claim is proven at the strictest possible bar. Cross-session SHA-256 equality against stale on-disk artifacts is recorded as NOT a meaningful bar on this OpenBLAS DYNAMIC_ARCH build — a real finding about the criterion, not a failure of the change."
  - "The plan's Task 2 acceptance criterion asserting exactly 588 rows for backtest_filtered_state_probs.parquet was a planning defect. per_step_metrics only appends non-degraded steps (backtest/driver.py:452-455); 118 of 588 walk-forward steps (20.1%) are degraded and excluded, so the natural (correct, un-reindexed) artifact has 470 rows, starting 1974-02-28 rather than 1972-01-31. Corrected acceptance criterion: 470 rows, not 588. Forcing 588 would require inventing 118 rows of data that do not exist in per_step_metrics, directly contradicting the plan's own 'do not reindex' instruction."
  - "The 470/588 gap (20.1% of walk-forward steps degraded, filtered path absent for the first two years, 1972-01 to 1974-02) is recorded as an A13-relevant finding, not merely a row-count correction — see 'Coverage of the filtered path' below."

requirements-completed: [NB-01]

coverage:
  - id: T1
    description: "run_full_backtest_evaluation() persists full_sample_states and filtered_state_probs as two new artifacts dict keys at the existing write_backtest_report() call site, with write_backtest_report() itself unmodified"
    requirement: "NB-01"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_evaluation_report.py::TestNewLabelingArtifacts (4 new tests)"
        status: pass
      - kind: unit
        ref: "tests/unit/test_platform_evaluation_report.py::TestArtifactsWritten (pre-existing, unmodified, still pass)"
        status: pass
    human_judgment: false
  - id: T2
    description: "Regenerate into a scratch directory, prove byte-identity of the seven pre-existing artifacts (same-session A/B, the correct baseline), copy only the two new artifacts into outputs/reports/platform/"
    requirement: "NB-01"
    verification:
      - kind: manual
        ref: "Same-session A(post-change)/B(pre-change, report.py reverted to 8eafd98) regeneration, registry_path overridden to a scratch ledger: all seven shared artifacts (backtest_report.md + 2 equity curves + kpi_table + 3 model_metrics files) are SHA-256 byte-identical between A and B. Reference numbers (median_sojourn 97.0, median_lag 164.0, ratio 0.5914634146341463, 4/6 resolved, Brier 0.208746, strategy terminal log wealth 4.026505, max drawdown -21.2432%) reproduce exactly. Two new artifacts (695x1 states, 470x5 probs) copied from run A into outputs/reports/platform/; the seven pre-existing files confirmed byte-identical (sha256sum) before and after the copy."
        status: pass
    human_judgment: false

actuals:
  tokens: 42000
  tasks: 2
  commits: 3

metrics:
  duration_minutes: 140
  completed_date: 2026-09-09
status: complete
---

# Phase 6 Plan 02: Persist full_sample_states and filtered_state_probs Summary

`run_full_backtest_evaluation()` now additively persists `full_sample_states`
and `filtered_state_probs` at the existing `write_backtest_report()` call
site — proven additive-only by a same-session code-before/code-after A/B
regeneration in which all seven pre-existing artifacts are SHA-256
byte-identical. The two new artifacts are live in
`outputs/reports/platform/`. A first attempt at the byte-identity proof used
the wrong baseline (cross-session comparison against artifacts already on
disk, which differ due to OpenBLAS `DYNAMIC_ARCH` kernel selection varying
between session/host instances) and was correctly escalated rather than
worked around; the orchestrator identified the correct same-session
comparison, which resolves cleanly at the strictest possible bar. The plan's
literal "588 rows" acceptance criterion for `backtest_filtered_state_probs`
was a planning defect — corrected to 470, with the 118-step (20.1%) gap
recorded as an A13-relevant coverage finding for P3/P6 to surface.

## Task Commits

1. **Task 1: Add full_sample_states and filtered_state_probs to the existing artifacts dict**
   - `4bf37b8` (feat) — `src/trading_crab_lib/platform/evaluation/report.py`,
     `tests/unit/test_platform_evaluation_report.py`
2. **Interim: record initial (later-superseded) blocker findings**
   - `7276c8f` (docs) — first draft of this SUMMARY, documenting the
     cross-session byte-identity investigation before the orchestrator
     identified the correct same-session baseline. Superseded by this
     version; kept in history as an accurate record of the investigation.
3. **Task 2: same-session byte-identity proof + artifact copy**
   - (this commit) — `outputs/reports/platform/backtest_full_sample_states.parquet`,
     `outputs/reports/platform/backtest_filtered_state_probs.parquet`,
     this SUMMARY (final version)

## Accomplishments (Task 1)

- `run_full_backtest_evaluation()`'s `artifacts` dict (the single call site at
  `write_backtest_report(...)`) now carries five keys instead of three:
  `equity_curve_strategy`, `equity_curve_ablation`, `kpi_table` (unchanged, in
  their original order) plus `full_sample_states` (the already-computed
  Series `.to_frame()`) and `filtered_state_probs` (the already-computed
  `filtered_probs_matrix`, columns renamed `{0,1,2,3,4} -> {state_0..state_4}`
  via a dict comprehension — no reindex, sort, fill, or round).
- `write_backtest_report()` itself: zero lines changed.
- `assemble_backtest_report`, `_build_kpi_table`, `_leg_kpis`,
  `_smoothed_hindsight_perf`, `_reference_label_columns`,
  `report_model_metrics`: zero lines changed.
- The function's RETURN dict (`report_path`, `sojourn_lag`, `strategy_kpis`,
  `ablation_kpis`, `baseline_kpis`, `gap`, `model_metrics_paths`,
  `equity_curve`, `ablation_curve`, `per_step_metrics`, `full_sample_states`):
  zero keys added, removed, or reordered.
- Docstring: step (h) extended, plus a new paragraph naming both filenames,
  documenting the `state_{k}` parquet-safe column convention and the
  int-conversion requirement for `compute_sojourn_lag_headline`, and stating
  explicitly that persisting these artifacts does NOT resolve audit item A13.
- `git diff --numstat`: **30 insertions, 2 deletions** — both deletions are
  the two lines being extended (the docstring's `(h)` line and the `# (h)`
  code comment), matching the acceptance criterion "fewer than 40 added
  lines and 0 lines deleted other than the lines being extended."

## Verification Results (Task 1)

- `.venv/bin/python -m pytest tests/unit/test_platform_evaluation_report.py -x -q`
  → **20 passed** (16 pre-existing + 4 new `TestNewLabelingArtifacts` tests;
  `TestArtifactsWritten` unmodified and still passing).
- `.venv/bin/python -m pytest tests/unit/test_platform_evaluation_sojourn_lag.py
  tests/unit/test_platform_backtest_driver.py tests/unit/test_platform_evaluation_kpis.py
  tests/unit/test_platform_evaluation_model_metrics.py -q` → **44 passed**.
- `.venv/bin/python -m pytest tests/ -q` (full suite, run after both Task 1
  and Task 2) → **1458 passed, 3 skipped, 0 failures** (baseline before this
  plan was 1454 passed, 3 skipped — net +4, zero regressions).

## TestNewLabelingArtifacts — the new tests, in detail

1. `test_all_five_parquet_files_written_with_backtest_prefix` — a five-key
   artifacts dict (three pre-existing shapes + a one-column `state` frame +
   a `state_{k}`-columned probability frame) all write with the `backtest_`
   prefix.
2. `test_states_frame_round_trips_with_datetime_index_and_int_states` — the
   states frame survives `pd.read_parquet` with its `DatetimeIndex` and
   integer `state` values intact.
3. `test_probability_frame_round_trips_with_state_k_columns_summing_to_one`
   — the probability frame survives round-trip with `state_{k}` string
   columns, every row summing to 1.0 within `1e-9`, every value in `[0, 1]`
   (`assert_frame_equal(..., check_freq=False)` — parquet round-trip does not
   preserve the `pd.DatetimeIndex.freq` attribute, a pandas/pyarrow property,
   not a value difference).
4. `test_round_tripped_pair_feeds_compute_sojourn_lag_headline` — renaming
   `state_{k}` columns back to integers and feeding the pair to
   `compute_sojourn_lag_headline` returns all six documented keys
   (`median_sojourn`, `median_lag`, `ratio`, `n_transitions`, `n_resolved`,
   `act_threshold`), with `n_resolved` an int in `[0, n_transitions]` and
   `ratio` finite or NaN — the exact round trip P6 depends on.

## Task 2: how the byte-identity proof was corrected

### First attempt (superseded) — the wrong baseline

The first regeneration compared a fresh scratch run against the artifacts
already sitting in `outputs/reports/platform/`, which were generated in an
**earlier session**. 5 of 7 files mismatched at SHA-256; characterizing the
difference with `pandas.testing.assert_frame_equal(check_exact=True)` showed
the underlying float64 values differed only at the 10th-16th significant
digit (e.g. `terminal_log_wealth` `4.026504994840959` vs
`4.026504994840789`). Two independent scratch reruns in *that* session
matched each other exactly but not the on-disk artifacts; forcing
single-threaded BLAS did not change the outcome. This pointed at
`numpy.show_config()`'s reported `OpenBLAS ... DYNAMIC_ARCH` build selecting
a different CPU-microarchitecture SIMD kernel depending on the underlying
host — i.e. a property of *which session/host produced the numbers*, not of
this plan's code. Per the plan's explicit "if any hash differs, stop and
report... do not copy anything" instruction, this was correctly escalated
rather than worked around or silently accepted.

### Corrected approach (adopted) — same-session code-before/code-after A/B

The comparison that actually isolates "is this plan's code change additive"
is a same-session, same-host regeneration run twice: once with the current
(post-Task-1) `report.py`, once with `report.py` reverted to its pre-Task-1
state (commit `8eafd98`), both runs using the identical checkpoints, config,
and host/BLAS kernel selection. Executed as:

1. Run A: current code, `run_full_backtest_evaluation(..., output_dir=<scratch_A>, registry_path=<scratch_registry_A>)`. 588/588 steps, 131.0s. Wrote 5 parquets + `backtest_report.md` (9 files with the 3 model-metrics files).
2. `report.py` reverted in the working tree to `git show 8eafd98:...` (Task 1's diff undone locally, not committed).
3. Run B: pre-Task-1 code, same script, `output_dir=<scratch_B>`, `registry_path=<scratch_registry_B>`. 588/588 steps, 131.8s. Wrote 3 parquets + `backtest_report.md` (7 files, as expected — no `full_sample_states`/`filtered_state_probs` keys exist in the pre-change code).
4. `report.py` restored to the post-Task-1 (commit `4bf37b8`) content (`git diff` confirmed empty afterward).
5. SHA-256 compared, A vs B, for the seven files both runs produced:

| Artifact | A (post-change) | B (pre-change) | |
|---|---|---|---|
| `backtest_report.md` | `0c742bf3e7de2b53` | `0c742bf3e7de2b53` | identical |
| `backtest_equity_curve_strategy` | `6c0f6a109433380b` | `6c0f6a109433380b` | identical |
| `backtest_equity_curve_ablation` | `2216a50ae0bd49cc` | `2216a50ae0bd49cc` | identical |
| `backtest_kpi_table` | `c0087ac10ff11e17` | `c0087ac10ff11e17` | identical |
| `model_metrics_brier` | `7002751b9b342359` | `7002751b9b342359` | identical |
| `model_metrics_calibration` | `2c5f0d515dc9b66f` | `2c5f0d515dc9b66f` | identical |
| `model_metrics_confusion` | `8ce3d994136ecf48` | `8ce3d994136ecf48` | identical |

**Every pre-existing artifact is SHA-256 byte-identical, same-session,
code-before vs code-after** — the strictest possible bar, and it holds
cleanly. This is the correct evidence for the additive-only claim; the
plan's byte-identity criterion is retained (not weakened) but **re-baselined
to same-session pre/post regeneration**, since cross-session SHA-256
equality against stale on-disk artifacts is not achievable on this OpenBLAS
`DYNAMIC_ARCH` build (a real finding about the criterion — floating-point
non-associativity across host/kernel selections at the ULP level — not a
failure of the code change).

Reference numbers from run A, reproduced exactly:

```
sojourn_lag={'median_sojourn': 97.0, 'median_lag': 164.0,
             'ratio': 0.5914634146341463, 'n_transitions': 6,
             'n_resolved': 4, 'act_threshold': 0.7}
```

| leg | terminal_log_wealth | max_drawdown |
|---|---|---|
| strategy | 4.026505 | -21.2432% |
| no_regime_ablation | 3.647238 | -19.8068% |
| spy_buy_hold | 5.680455 | -48.9475% |
| sixty_forty | 5.047073 | -26.9625% |
| faber_sma | 6.372592 | -18.9421% |

`model_metrics_brier.parquet` → `brier = 0.208746`. Matches
`BASELINE-v1-tracer-bullet.md`'s "Current reference run" table exactly.

## The corrected "588 rows" acceptance criterion — now 470

The plan's Task 2 `<verify>` script and acceptance criteria assert
`backtest_filtered_state_probs.parquet` has exactly 588 rows. Validating run
A's artifact directly:

```
states shape: (695, 1)   cols: ['state']            dtype: int64
probs  shape: (470, 5)   cols: state_0..state_4      dtype: float64
```

This traces to `backtest/driver.py:452-455`: `per_step_metrics["dates"]`
(and `"proba"`, `"classes"`) only receive an `.append()` call when a step is
**not** flagged `degraded` (insufficient class examples for the L2 CV
refit). `build_filtered_probs_matrix` consumes `per_step_metrics` directly
with zero transformation, so its natural output has exactly as many rows as
non-degraded steps — 470, matching `report_model_metrics`'s own logged
`n_steps: 470`. **Corrected acceptance criterion: 470 rows, not 588.**
Forcing 588 would mean inventing 118 rows of data that do not exist in
`per_step_metrics`, directly contradicting the plan's own "do not reindex,
sort, fill, or round anything" instruction — 470 is the correct,
un-transformed row count.

## Coverage of the filtered path (A13-relevant finding)

The 470/588 gap is a finding in its own right, not merely a number to
correct:

- **118 of 588 walk-forward steps (20.1%) are degraded and excluded** from
  `per_step_metrics`, and therefore from `filtered_state_probs`.
- The filtered path (`backtest_filtered_state_probs.parquet`) does not begin
  until **1974-02-28** — roughly **two years after** the backtest's
  1972-01-31 start — because the earliest steps are the ones most often
  degraded (small post-embargo training windows, insufficient class
  examples for the L2 CV refit; see `backtest/driver.py`'s
  `"L2 refit degraded (early small post-embargo window, RESEARCH Pitfall 2)"`
  warning, which fires repeatedly in the early walk-forward years).
- `full_sample_states` (the smoothed reference) spans **1963-02-28 to
  2020-12-31** (695 months) — i.e. it covers 11 years the filtered path
  never reaches at all, and even within the walk-forward's own 1972-2020
  window, the filtered path is missing for its first two years.

**This strengthens, rather than contradicts, the existing "not
interpretable — see A13" caveat.** A13 already establishes that the
smoothed reference (fixed 9-column feature set, 1963-2020) and the
walk-forward's filtered path (a feature set that changes 7 times, 4→13
active columns) are not tracking the same regimes on the same inputs. The
470/588 coverage gap adds a second, independent reason their disagreement is
not purely detection delay: for 20.1% of walk-forward steps, and for the
entire pre-1974 window, there is no filtered probability to compare against
at all. **P3 (06-03) and P6 (06-07) must surface this coverage — the 470/588
count, the 20.1% figure, and the 1974-02 start date — alongside the
sojourn/lag headline**, not just the existing small-sample
(`n_resolved`/`n_transitions`) caveat.

## Full validation of the two new artifacts (live copy, `outputs/reports/platform/`)

| property | `backtest_full_sample_states.parquet` | `backtest_filtered_state_probs.parquet` |
|---|---|---|
| rows | 695 | **470** (corrected from the plan's stated 588) |
| columns | `state` (int64) | `state_0` .. `state_4` (float64) |
| index | DatetimeIndex, 1963-02-28 → 2020-12-31 | DatetimeIndex, 1974-02-28 → 2020-12-31 |
| value bounds | states ∈ {0,1,2,3,4} | every value ∈ [0.0, 0.912...] ⊂ [0,1] |
| row-sum invariant | n/a | every row sums to 1.0 within 1e-9 (min 0.9999999999999998, max 1.0000000000000002) |
| occupancy (0/1/2/3/4) | 1.5827 / 13.9568 / 31.9424 / 40.5755 / 11.9424 % (sum 1.0000000000000002) — matches the documented reference 1.6/14.0/31.9/40.6/11.9% within 0.05pp per state | n/a |
| coverage vs 588 total walk-forward steps | n/a | 470/588 = **79.9%**; gap = 118/588 = **20.1%** |
| `compute_sojourn_lag_headline` recomputed from this pair | median_sojourn=97.0, median_lag=164.0, ratio=0.5914634146341463, n_resolved=4, n_transitions=6, act_threshold=0.7 — **exact match** to the reference | (same) |

`platform/plotting/loaders.py::load_full_sample_states()` and
`load_filtered_state_probs()` (built in plan 06-01) were exercised directly
against the live artifacts and succeed:

```
states: (695,) int64  1963-02-28 -> 2020-12-31
probs:  (470, 5) columns [0, 1, 2, 3, 4]  1974-02-28 -> 2020-12-31
```

(the `state_{k}` string columns are correctly renamed back to integers by
the loader, per plan 06-01's documented contract.)

## State of the repository after this plan

- `outputs/reports/platform/` contains **nine** files: the original seven
  (confirmed byte-identical via `sha256sum` before and after the copy — zero
  diff) plus `backtest_full_sample_states.parquet` and
  `backtest_filtered_state_probs.parquet`.
- `outputs/reports/platform/` is **not** gitignored in this repo (verified:
  `git check-ignore` returns exit 1; the seven pre-existing files are
  already tracked via `git ls-files outputs/`). The two new parquet files
  are committed alongside the code and tests, consistent with existing
  repository precedent and the plan's own `files_modified` list.
- `registry/trials.jsonl`: fully protected this time — every regeneration
  (initial investigation runs and the final A/B pair) used an explicit
  scratch `registry_path` override. `git status --short registry/` is
  clean throughout Task 2's corrected execution. (The FIRST investigation
  round, before this correction, did append six rows to the live file by
  omission; that was caught and reverted via `git checkout --
  registry/trials.jsonl` before any commit — see the first SUMMARY draft,
  commit `7276c8f`, for that record.)

## Known Stubs

None. Both new artifacts are live and load successfully via the 06-01
loaders.

## Deviations from Plan

### Auto-fixed Issues

None — Rules 1-3 do not apply. No bug was found in the code this plan
wrote.

### Escalated finding, then resolved by the orchestrator (documented, not an auto-fix)

**1. [Escalated] Byte-identity criterion's baseline was cross-session, not same-session**
- **Found during:** first Task 2 regeneration attempt.
- **Issue:** comparing a fresh scratch regeneration against artifacts
  already on disk (built in an earlier session) measures environment-level
  BLAS `DYNAMIC_ARCH` kernel-selection variance, not the code change.
- **Resolution:** orchestrator identified the correct baseline — same-session
  code-before (report.py reverted to `8eafd98`) vs code-after (current) A/B
  regeneration. Executed and confirmed: all seven shared artifacts SHA-256
  byte-identical. The plan's byte-identity criterion is retained at full
  strength, re-baselined to same-session comparison.
- **Files affected:** none (diagnostic only); `report.py` was reverted and
  restored in the working tree, confirmed via `git diff` returning empty
  before proceeding.

**2. [Corrected] "588 rows" acceptance criterion was a planning defect**
- **Found during:** validating the new artifacts against the plan's literal
  `<verify>` script.
- **Issue:** the plan asserts `backtest_filtered_state_probs.parquet` has
  588 rows; the natural (untransformed) artifact has 470, because 118 of
  588 walk-forward steps are `degraded` and excluded from `per_step_metrics`
  by `backtest/driver.py`.
- **Resolution:** acceptance criterion corrected to 470 (documented above
  and orchestrator-confirmed). The 470/588 gap is also recorded as an
  A13-relevant coverage finding (see "Coverage of the filtered path").
- **Files affected:** none (this SUMMARY documents the correction; no code
  or test needed to enforce 588, since Task 1's tests do not assert a row
  count).

## Issues Encountered

- First regeneration script did not override `registry_path`, causing three
  diagnostic runs to append six rows to the live, tracked
  `registry/trials.jsonl`. Caught and reverted before any commit. All
  subsequent regenerations (the corrected A/B pair) used explicit scratch
  `registry_path` overrides — confirmed via `git status --short registry/`
  returning clean throughout.

## User Setup Required

None. Zero new dependencies. No secrets.

## Self-Check: PASSED

- `src/trading_crab_lib/platform/evaluation/report.py` — FOUND, diff
  confirmed additive-only (30 insertions / 2 deletions), confirmed restored
  to post-Task-1 state after the A/B diagnostic (git diff empty)
- `tests/unit/test_platform_evaluation_report.py` — FOUND,
  `TestNewLabelingArtifacts` present, 4 new tests, all pass
- commit `4bf37b8` — FOUND (`git log --oneline` on this branch)
- `outputs/reports/platform/backtest_full_sample_states.parquet` — FOUND,
  695 rows, `state` int64 column, index 1963-02-28..2020-12-31
- `outputs/reports/platform/backtest_filtered_state_probs.parquet` — FOUND,
  470 rows, `state_0`..`state_4` float64 columns, index
  1974-02-28..2020-12-31
- `outputs/reports/platform/`'s seven pre-existing files — FOUND, confirmed
  byte-identical to their pre-plan SHA-256 values (verified before and
  after the copy)
- `platform/plotting/loaders.py::load_full_sample_states()` and
  `load_filtered_state_probs()` — exercised directly against the live
  artifacts, succeed, correct dtypes/shapes/column renaming

---
*Phase: 06-platform-notebook-suite*
*Completed: 2026-09-09*
