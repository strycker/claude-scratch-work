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
  - "Task 1 (LANDED): run_full_backtest_evaluation() persists two additional artifacts — full_sample_states, filtered_state_probs — additively, at the existing write_backtest_report() call site."
  - "Task 2 (BLOCKED): the two artifacts are NOT yet copied into outputs/reports/platform/ — see Execution Blocked below."
affects: [06-03-features-taxonomy-regime-labeling-notebook, 06-07-backtest-notebook]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Same-file additive dict-literal extension (06-PATTERNS.md 'platform/evaluation/report.py MODIFIED — additive only')"

key-files:
  created: []
  modified:
    - src/trading_crab_lib/platform/evaluation/report.py
    - tests/unit/test_platform_evaluation_report.py

key-decisions:
  - "Task 1 landed exactly as specified: two new artifacts dict keys (full_sample_states, filtered_state_probs) added at the single write_backtest_report() call site; write_backtest_report() itself untouched; git diff confined to 30 insertions / 2 deletions (both deletions are the two doc/comment lines being extended, per acceptance criteria)."
  - "Task 2 STOPPED per the plan's own explicit instruction: 'If any hash differs, stop and report ... do not copy anything and do not modify the live directory.' 5 of 7 pre-existing artifacts differ from a from-scratch regeneration at the SHA-256 level. Root-caused to environment-level BLAS floating-point non-associativity (OpenBLAS DYNAMIC_ARCH, confirmed via three independent lines of evidence below) — NOT a logic/metric change introduced by this plan's code diff. No artifact was copied into outputs/reports/platform/; the live directory's seven files are provably untouched (byte-identical before/after all diagnostics, see below)."
  - "A second, independent plan-authoring discrepancy was found during validation: the plan's Task 2 verify script and acceptance criteria assert the filtered_state_probs artifact has exactly 588 rows. The actual per_step_metrics only appends non-degraded steps (backtest/driver.py:452-455), and 118 of 588 walk-forward steps are flagged degraded in this run, so the natural (correct, un-reindexed) artifact has 470 rows — matching report_model_metrics' own 'n_steps: 470' log line and the live model_metrics_brier.parquet's implicit sample size. Per the plan's own action text ('do not reindex, sort, fill, or round anything'), 470 is the CORRECT row count; forcing 588 would require inventing data. This is flagged for human review, not silently patched."

requirements-completed: []

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
    description: "Regenerate into a scratch directory, prove byte-identity of the seven pre-existing artifacts, copy only the two new artifacts into outputs/reports/platform/"
    requirement: "NB-01"
    verification:
      - kind: manual
        ref: "Full re-run of run_full_backtest_evaluation() into a scratch directory (132.9s, 588/588 steps) reproduces every documented reference number exactly (median_sojourn 97.0, median_lag 164.0, ratio 0.5914634146341463, 4 of 6 resolved, Brier 0.208746, strategy terminal log wealth 4.026505, max drawdown -21.2432%) — the plan's own 'backstop' verification passes. SHA-256 byte-identity (the PRIMARY verification) does NOT pass for 5 of 7 pre-existing files; see Execution Blocked."
        status: blocked
    human_judgment: true

actuals:
  tokens: 28000
  tasks: 1
  commits: 1

metrics:
  duration_minutes: 75
  completed_date: 2026-09-09
status: blocked
---

# Phase 6 Plan 02: Persist full_sample_states and filtered_state_probs Summary

Task 1 landed cleanly: `run_full_backtest_evaluation()` now additively persists
`full_sample_states` and `filtered_state_probs` at the existing
`write_backtest_report()` call site, proven by a 4-test round-trip suite and a
`git diff` confined to exactly the authorized scope. Task 2 (regenerate, prove
byte-identity, copy) is **BLOCKED**: a from-scratch regeneration reproduces
every documented reference number exactly but does not reproduce the seven
pre-existing artifacts' SHA-256 hashes bit-for-bit. Investigation (three
independent tests) traces this to environment-level BLAS floating-point
non-associativity, not to this plan's code change — but the plan's own text
is explicit that any hash mismatch is a stop condition, not something to work
around, so no artifact was copied into `outputs/reports/platform/`.

## Task Commits

1. **Task 1: Add full_sample_states and filtered_state_probs to the existing artifacts dict**
   - `4bf37b8` (feat) — `src/trading_crab_lib/platform/evaluation/report.py`,
     `tests/unit/test_platform_evaluation_report.py`

Task 2 produced no commit — no files were modified or created (see Execution
Blocked).

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
- `.venv/bin/python -m pytest tests/ -q` (full suite) → **1458 passed, 3
  skipped, 0 failures** (baseline before this plan was 1454 passed, 3
  skipped — net +4, zero regressions, consistent with the 4 new tests added).

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

## EXECUTION BLOCKED — Task 2

### What was attempted

A one-off script (never `main()`, so nothing could write into
`outputs/reports/platform/` directly) called
`run_full_backtest_evaluation(monthly_features, monthly_raw, cfg,
output_dir=SCRATCH_DIR)` exactly once against the real `monthly_features`/
`monthly_raw` platform checkpoints. Real run, 588/588 walk-forward steps,
132.9s wall clock. Recorded reference numbers:

```
sojourn_lag={'median_sojourn': 97.0, 'median_lag': 164.0,
             'ratio': 0.5914634146341463, 'n_transitions': 6,
             'n_resolved': 4, 'act_threshold': 0.7}
```

This is an **exact** match to `BASELINE-v1-tracer-bullet.md`'s "Current
reference run" table (97.0 / 164.0 / 0.591 / 4 of 6) and to the plan's own
"backstop" `must_haves` truth. `kpi_table` from the scratch run:

| leg | terminal_log_wealth | max_drawdown |
|---|---|---|
| strategy | 4.026505 | -21.2432% |
| no_regime_ablation | 3.647238 | -19.8068% |
| spy_buy_hold | 5.680455 | -48.9475% |
| sixty_forty | 5.047073 | -26.9625% |
| faber_sma | 6.372592 | -18.9421% |

`model_metrics_brier.parquet` → `brier = 0.208746`. All three of the plan's
literal backstop numbers (strategy terminal log wealth 4.0265, max drawdown
-21.24%, Brier 0.2087) reproduce exactly at the documented precision.

### Why it stopped: SHA-256 byte-identity fails for 5 of 7 pre-existing artifacts

Per-file SHA-256 comparison (scratch vs. live `outputs/reports/platform/`,
BEFORE any copy):

| file | match? |
|---|---|
| `backtest_report.md` | **MATCH** |
| `backtest_equity_curve_strategy.parquet` | MISMATCH |
| `backtest_equity_curve_ablation.parquet` | MISMATCH |
| `backtest_kpi_table.parquet` | MISMATCH |
| `model_metrics_brier.parquet` | MISMATCH |
| `model_metrics_calibration.parquet` | MISMATCH |
| `model_metrics_confusion.parquet` | **MATCH** |

Per the plan's explicit Task 2 action text: *"If any hash differs, stop and
report which file and what changed ... do not copy anything and do not
modify the live directory."* Followed literally: **no file was copied**, and
the live directory is confirmed byte-identical before and after every
diagnostic run (`sha256sum` of all seven files recorded before the first
scratch run and again after all diagnostics — identical).

### Characterizing the difference (`pandas.testing.assert_frame_equal`)

With default tolerance (`check_exact=False`), all five mismatching files'
DataFrames are **logically equal** (`assert_frame_equal` passes). Only with
`check_exact=True` do the underlying float64 values differ — at the
**10th–16th significant digit**, e.g.:

```
terminal_log_wealth (strategy leg):
  live:    4.026504994840959
  scratch: 4.026504994840789          # differs at the 13th significant digit
brier:
  live:    0.2087462934432449
  scratch: 0.20874629344339823        # differs at the 15th significant digit
```

Every value that any downstream consumer would actually read (4 decimal
places in the report, 4-6 in the KPI table) is unaffected.

### Root-causing: three independent pieces of evidence rule out the code change

1. **The `report.py` diff is provably additive-only.** `git diff` (shown
   above) touches only two new local variables, two new dict-literal keys,
   and docstring/comment text — zero lines inside any KPI/metric/labeling
   computation.
2. **Two independent from-scratch regenerations of the CURRENT (post-Task-1)
   code are byte-identical to EACH OTHER** but differ from the live
   artifacts (generated in an earlier session, before this plan started).
   This proves the current code+environment pairing is internally
   deterministic — the non-reproducibility is specifically between *this*
   session and the *session that produced the live artifacts*, not a
   run-to-run flakiness introduced by this plan.
3. **Forcing single-threaded BLAS does not change the outcome**
   (`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
   NUMEXPR_NUM_THREADS=1`) — a third regeneration under these constraints
   still mismatches the live artifacts at the same five files, ruling out
   simple thread-count non-determinism as the sole cause and pointing at
   `numpy.show_config()`'s reported `OpenBLAS ... DYNAMIC_ARCH` build (which
   selects a CPU-microarchitecture-specific SIMD kernel at *runtime*,
   per-process) — i.e., a different underlying host CPU between the session
   that produced the live artifacts and this session, which is entirely
   plausible in a container/cloud environment and is outside this plan's
   control.
4. Corroborating evidence from `registry/trials.jsonl` (an unrelated,
   already-tracked file `run_backtest`/`no_regime_ablation` append trial
   rows to): the ORIGINAL session's two ablation-leg trial entries
   (`15:25:48` and `17:06:59`, i.e. two *different* invocations within the
   original session) recorded the **exact same** `terminal_log_wealth:
   3.647237762337868` bit-for-bit — proof that runs *within* a single
   session/environment ARE bit-reproducible. My three regeneration runs in
   *this* session all agree with each other at `3.647237762337872` — a
   different, but internally consistent, bit pattern. This is the clearest
   evidence available that the discrepancy is a property of *which session
   generated the numbers*, not of this plan's code.

**Conclusion: this is environment-level floating-point non-determinism
between two separate session instances (most likely due to
`DYNAMIC_ARCH`-selected differences in the underlying host CPU's BLAS
kernel), not a logic, metric, or hyperparameter change introduced by this
plan.** The plan's own "backstop" verification (must_haves, item 4) exists
for exactly this eventuality and passes cleanly. The primary SHA-256 gate
does not pass, and the plan's text is unambiguous that this specific
trigger is a stop condition requiring investigation and reporting, not an
autonomous work-around — so Task 2 halts here for human review rather than
copying the artifacts unilaterally.

### A second, independent discrepancy found during validation: the "588 rows" assumption

Separately from the byte-identity question, validating the two NEW
artifacts against the plan's own literal `<verify>` script surfaced a second
issue: the script asserts `backtest_filtered_state_probs.parquet` has
exactly 588 rows. The actual artifact (built with zero transformation beyond
the `state_{k}` column rename, per the plan's own "do not reindex, sort,
fill, or round anything" instruction) has **470 rows**, not 588.

This traces to `backtest/driver.py:452-455`: `per_step_metrics["dates"]`
(and `"proba"`, `"classes"`) only receive an `.append()` call when a step is
**not** flagged `degraded` (insufficient class examples for the L2 CV
refit — 118 of the 588 walk-forward decision points are degraded and
excluded in this run). `build_filtered_probs_matrix` consumes
`per_step_metrics` directly, so its natural output has exactly as many rows
as non-degraded steps — 470, matching `report_model_metrics`'s own logged
`n_steps: 470` and consistent with the live `model_metrics_brier.parquet`
(same Brier value implies the same sample size). The plan's "588" figure
appears to conflate the walk-forward's total decision-point schedule (588,
correctly documented in `BASELINE-v1-tracer-bullet.md`'s "Window: 588
monthly steps") with the number of steps that actually land in
`per_step_metrics` after degraded-step exclusion.

**This is flagged, not silently patched**, because forcing the artifact to
588 rows would mean inventing 118 rows of data that do not exist in
`per_step_metrics` — directly contradicting the plan's own "do not reindex"
instruction. 470 is the correct row count for a faithful, untransformed
persistence of `filtered_probs_matrix`.

### Full validation of the two new artifacts (against the SCRATCH copy — not yet in the live directory)

| property | `backtest_full_sample_states.parquet` | `backtest_filtered_state_probs.parquet` |
|---|---|---|
| rows | 695 | 470 (see discrepancy above; NOT 588) |
| columns | `state` (int64) | `state_0` .. `state_4` (float64) |
| index | DatetimeIndex, 1963-02-28 → 2020-12-31 | DatetimeIndex, 1974-02-28 → 2020-12-31 |
| value bounds | states ∈ {0,1,2,3,4} | every value ∈ [0.0, 0.912...] ⊂ [0,1] |
| row-sum invariant | n/a | every row sums to 1.0 within 1e-9 (min 0.9999999999999998, max 1.0000000000000002) |
| occupancy (0/1/2/3/4) | 1.5827 / 13.9568 / 31.9424 / 40.5755 / 11.9424 % (sum 1.0000000000000002) — matches the documented reference 1.6/14.0/31.9/40.6/11.9% within 0.05pp per state | n/a |
| `compute_sojourn_lag_headline` recomputed from this pair | median_sojourn=97.0, median_lag=164.0, ratio=0.5914634146341463, n_resolved=4, n_transitions=6, act_threshold=0.7 — **exact match** to the reference | (same) |

Every band and every headline number this plan set out to prove **is
correct in substance**. Only the SHA-256 byte gate and the literal "588"
row-count assertion do not hold, for the two independent, well-characterized
reasons documented above.

### State of the repository after this plan

- `outputs/reports/platform/` still contains exactly the same **seven**
  files it had before this plan started, confirmed byte-identical
  (`sha256sum` recorded before the first diagnostic run and again after all
  diagnostics — zero diff).
- `registry/trials.jsonl`: my scratch regeneration script did not override
  `registry_path`, so three diagnostic runs appended six trial rows to this
  **live, tracked** file as an unintended side effect. This was caught and
  reverted (`git checkout -- registry/trials.jsonl`) before finishing — the
  file is back to its pre-plan committed state. Recorded here as a
  documented mistake in my own scratch tooling (not a plan defect): any
  future regeneration for this plan must pass `registry_path=<scratch>` to
  `run_full_backtest_evaluation()` to avoid this.
- `outputs/reports/platform/backtest_full_sample_states.parquet` and
  `outputs/reports/platform/backtest_filtered_state_probs.parquet` do **not
  exist** in the live directory. They exist only in ephemeral scratch
  directories under `/tmp` (session-local, not part of the repository), and
  will be lost when this session ends.

### What a human/orchestrator needs to decide before Task 2 can close

1. **Byte-identity policy:** is exact SHA-256 identity across separate
   session/container instances a realistic bar given confirmed
   `DYNAMIC_ARCH` OpenBLAS non-associativity, or should the acceptance
   criterion be relaxed to `pandas.testing.assert_frame_equal` (default
   tolerance) — which DOES pass for all seven files?
2. **Row-count correction:** should the plan's acceptance criteria be
   corrected from 588 to 470 for `backtest_filtered_state_probs.parquet`
   (matching the natural, untransformed `per_step_metrics` output), or is
   there a different, intended source of "588" rows this plan should have
   used instead of `per_step_metrics` directly?
3. Once (1) and (2) are resolved, the two new artifacts (proven correct in
   substance above) can be copied from a fresh scratch regeneration (with
   `registry_path` overridden this time) into `outputs/reports/platform/`
   to complete Task 2.

## Known Stubs

None from Task 1's landed code. Task 2's two artifacts are NOT yet on disk
in the live directory — `platform/plotting/loaders.py`'s
`load_full_sample_states()` and `load_filtered_state_probs()` (built in plan
06-01) will continue to raise `FileNotFoundError` with their documented
actionable message until Task 2 is unblocked and completed.

## Deviations from Plan

### Auto-fixed Issues

None — Rules 1-3 do not apply here. The mismatch is not a bug in the code
this plan wrote (which is provably additive-only and independently
verified as internally deterministic); it is an environmental property of
running numerical code across two different session instances, which no
in-scope code change can fix.

### Rule 4 (architectural / ask) — not formally invoked, but functionally equivalent

This is not a schema/architecture change request. It is reported here as an
**EXECUTION BLOCKED** condition per the plan's own explicit "stop and
report" instruction, because relaxing a stated hard acceptance criterion
(byte-for-byte SHA-256 identity; exactly 588 rows) is a decision the plan
reserved for human judgment, not an in-scope auto-fix.

## Issues Encountered

- Scratch regeneration script did not override `registry_path`, causing
  three diagnostic runs to append six rows to the live, tracked
  `registry/trials.jsonl`. Caught and reverted before this summary was
  written. Any future regeneration attempt for this plan MUST pass an
  explicit scratch `registry_path` to `run_full_backtest_evaluation()`.

## User Setup Required

None. Zero new dependencies. No secrets.

## Self-Check: PASSED (for what was actually delivered)

- `src/trading_crab_lib/platform/evaluation/report.py` — FOUND, diff
  confirmed additive-only (30 insertions / 2 deletions)
- `tests/unit/test_platform_evaluation_report.py` — FOUND,
  `TestNewLabelingArtifacts` present, 4 new tests, all pass
- commit `4bf37b8` — FOUND (`git log --oneline -1` on this branch)
- `outputs/reports/platform/backtest_full_sample_states.parquet` — NOT
  FOUND in the live directory (expected — Task 2 blocked, see above)
- `outputs/reports/platform/backtest_filtered_state_probs.parquet` — NOT
  FOUND in the live directory (expected — Task 2 blocked, see above)
- `outputs/reports/platform/`'s seven pre-existing files — FOUND, confirmed
  byte-identical to their pre-plan SHA-256 values

---
*Phase: 06-platform-notebook-suite*
*Status: BLOCKED (Task 1 complete, Task 2 blocked pending human decision)*
