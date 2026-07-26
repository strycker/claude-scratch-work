---
phase: 05-honest-backtest-evaluation
plan: 06
subsystem: backtest
tags: [platform, evaluation, honesty-framework, tdd, cross-ai-review-fixes, report, capstone]

# Dependency graph
requires:
  - phase: 05-honest-backtest-evaluation
    plan: 02
    provides: "backtest/driver.py run_backtest(monthly_features, asset_returns, cfg, *, cash_returns, use_regime_tilt, registry_path) -> (equity_curve, per_step_metrics)"
  - phase: 05-honest-backtest-evaluation
    plan: 03
    provides: "evaluation/kpis.py (terminal_log_wealth/max_drawdown_and_duration/cvar/crisis_capture_ratio); evaluation/sojourn_lag.py (build_filtered_probs_matrix/compute_sojourn_lag_headline, review F1)"
  - phase: 05-honest-backtest-evaluation
    plan: 04
    provides: "evaluation/model_metrics.py report_model_metrics(per_step_metrics, output_dir) — requires a date-joined per_step_metrics['y_true'] (review F2)"
  - phase: 05-honest-backtest-evaluation
    plan: 05
    provides: "backtest/baselines.py spy_buy_hold/sixty_forty/faber_sma/no_regime_ablation"
provides:
  - "src/trading_crab_lib/platform/evaluation/report.py: assemble_backtest_report, write_backtest_report, run_full_backtest_evaluation, main(argv), __main__ hook"
  - "backtest report markdown + per-leg equity-curve/KPI-table/model-metrics parquet artifacts under OUTPUT_DIR/reports/platform/"
affects: []

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "assemble_backtest_report is a pure markdown builder over precomputed dicts (mirrors report/weekly.py's assemble_weekly_report isolation pattern) — headline-first fixed section order (D-01a): sojourn/lag ratio -> Faber comparison -> no-regime-ablation delta -> smoothed-vs-filtered gap -> baseline gauntlet table -> strategy KPI table + Conventions note"
    - "run_full_backtest_evaluation builds the SMOOTHED reference labeling as ONE full-sample fit_jump_model + canonicalize_states call over the holdout-bounded dev window — a genuinely distinct object/length from the walk-forward driver's per_step_metrics (Pitfall 1), proven by a dedicated integration test"
    - "the headline's filtered-probs input is the MULTICLASS matrix from sojourn_lag.build_filtered_probs_matrix, never a class-agnostic max (review F1, inherited from Plan 03, wired here for the first time against a real driver run)"
    - "y_true for model_metrics is joined by REINDEXING the smoothed reference states onto per_step_metrics['dates'] (review F2) with a hard length-assertion before report_model_metrics is called — never loop-sourced"
    - "cash_ret (splice.build_core_research_series -> assets.returns.compute_monthly_returns) is passed into BOTH run_backtest and no_regime_ablation as cash_returns (review F4) — the strategy's cash residual and every baseline's non-invested leg earn the SAME series"
    - "the investable asset_returns universe fed to run_backtest excludes the 'cash' splice class (FZFXX) entirely — cash is the vol-target residual, never a tilt-into-able risk position"
    - "the smoothed-vs-filtered gap's 'smoothed' half is a hindsight-oracle vol_targeted_tilt driven by the FULL-SAMPLE smoothed state at each real walk-forward decision date (no new allocation math — reuses vol_targeted_tilt/returns_by_regime_stats unchanged) — genuinely distinct from the filtered (actual walk-forward) performance number"
    - "baselines are holdout-bounded by the REPORT LAYER via split_by_holdout_boundary before being passed to spy_buy_hold/sixty_forty/faber_sma, since those functions do not enforce the cutoff internally (per baselines.py's own documented caller-responsibility contract)"

key-files:
  created:
    - src/trading_crab_lib/platform/evaluation/report.py
    - tests/unit/test_platform_evaluation_report.py
    - tests/integration/test_mini_backtest.py
  modified: []

key-decisions:
  - "run_full_backtest_evaluation computes the smoothed-vs-filtered gap via a hindsight-oracle vol_targeted_tilt driven by the full-sample smoothed states at each real walk-forward decision date (never inventing new allocation math) rather than a simpler point-estimate proxy — matches the design's non-causal batch-labeling doctrine and keeps the gap input genuinely distinct from the filtered strategy performance (Pitfall 1)."
  - "The investable asset_returns universe fed to run_backtest excludes the 'cash' splice class (FZFXX) — cash is never tilted into as a risk position; it is the vol-target residual that earns cash_ret directly via run_backtest's cash_returns parameter (review F4)."
  - "The D-04 grep gate against 'deflated|dsr' required rewording the module docstring's own prohibition explanation (it originally spelled out 'deflated Sharpe (DSR)' to explain what must NOT appear) — reworded to 'the design-Phase-6 registry-denominator risk-adjusted statistic' so the docstring can explain the constraint without itself tripping the gate it documents."
  - "The trailing __main__ block was changed from a standalone synthetic self-check (the convention every other evaluation/backtest module in this phase follows) to `raise SystemExit(main())` — mirroring report/weekly.py's own pattern exactly, since main() now requires real Phase 1 checkpoints (monthly_features/monthly_raw) and there is no meaningful synthetic no-checkpoint self-check for the report layer itself; the integration test is the runnable no-network proof for this module instead."

patterns-established:
  - "Pattern: the report-layer module (evaluation/report.py) is the ONLY module in Phase 5 whose __main__ hook calls main() directly rather than running a standalone synthetic self-check — because it is genuinely a checkpoint-consuming orchestrator, not a pure-compute module. Mirrors report/weekly.py exactly."

requirements-completed: [EVAL-01, EVAL-02, EVAL-03, EVAL-04]

coverage:
  - id: D1
    description: "assemble_backtest_report renders the sojourn/detection-lag ratio as the FIRST metrics section, before the Faber comparison and the no-regime-ablation delta (D-01a headline-first ordering)"
    requirement: "EVAL-03"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_evaluation_report.py::TestHeadlineOrdering"
        status: pass
    human_judgment: false
  - id: D2
    description: "the markdown contains a baseline-gauntlet section naming SPY, 60/40, and Faber with both log wealth and max drawdown, plus a no-regime-ablation delta line"
    requirement: "EVAL-02"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_evaluation_report.py::TestBaselineGauntletSection"
        status: pass
    human_judgment: false
  - id: D3
    description: "write_backtest_report writes the markdown to a tmp path and the KPI/equity-curve parquet artifacts round-trip via read_parquet"
    requirement: "EVAL-04"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_evaluation_report.py::TestArtifactsWritten"
        status: pass
    human_judgment: false
  - id: D4
    description: "report.py never computes or displays the deferred risk-adjusted (deflated Sharpe) statistic — the trial registry is written but never read for that denominator this phase (D-04)"
    requirement: "EVAL-01"
    verification:
      - kind: other
        ref: "grep -niE \"deflated|dsr\" src/trading_crab_lib/platform/evaluation/report.py returns nothing"
        status: pass
    human_judgment: false
  - id: D5
    description: "run_full_backtest_evaluation drives the strategy backtest (cash_returns=cash_ret, review F4), the no-regime ablation, and the three price baselines end-to-end on a synthetic monthly frame with no network and no real checkpoints, producing the markdown + all parquet artifacts"
    requirement: "EVAL-01"
    verification:
      - kind: integration
        ref: "tests/integration/test_mini_backtest.py::TestRunFullBacktestEvaluationEndToEnd::test_produces_markdown_and_all_parquet_artifacts"
        status: pass
    human_judgment: false
  - id: D6
    description: "the registry gains exactly 2 trials per full run (strategy + no-regime ablation); the 3 price baselines (spy_buy_hold/sixty_forty/faber_sma) log no trial (review F7)"
    requirement: "EVAL-02"
    verification:
      - kind: integration
        ref: "tests/integration/test_mini_backtest.py::TestRunFullBacktestEvaluationEndToEnd::test_registry_gains_exactly_two_trials_strategy_and_ablation_only"
        status: pass
    human_judgment: false
  - id: D7
    description: "the smoothed reference labeling (one full-sample L1 fit) and the walk-forward per_step_metrics are built as genuinely distinct series/objects (Pitfall 1) — proven by differing lengths (the smoothed reference spans the entire dev window; per_step_metrics only visits post-min_train decision dates)"
    requirement: "EVAL-03"
    verification:
      - kind: integration
        ref: "tests/integration/test_mini_backtest.py::TestRunFullBacktestEvaluationEndToEnd::test_smoothed_and_filtered_series_are_distinct_objects"
        status: pass
    human_judgment: false
  - id: D8
    description: "the model-metrics y_true is joined by date to the smoothed reference state at each walk-forward decision month — every scored label equals the smoothed reference state at that exact date (review F2), never loop-sourced"
    requirement: "EVAL-04"
    verification:
      - kind: integration
        ref: "tests/integration/test_mini_backtest.py::TestYTrueDateAlignment::test_y_true_equals_smoothed_reference_state_at_each_decision_date"
        status: pass
    human_judgment: false

duration: 7min
completed: 2026-07-25
status: complete
---

# Phase 5 Plan 6: Honest Backtest Report Summary

**`report.py` — the EVAL-01..04 capstone wiring `run_full_backtest_evaluation` into a headline-first markdown report: the sojourn/detection-lag ratio renders FIRST (D-01a), followed by the Faber comparison, the no-regime-ablation delta, the smoothed-vs-filtered gap (via a hindsight-oracle tilt, Pitfall 1), the baseline gauntlet table, and the strategy KPI table with a documented cash-return convention note (review F4) — proven end-to-end on a synthetic 48-month frame with no network, no DSR, and a date-joined y_true (review F2).**

## Performance

- **Duration:** 7 min
- **Started:** 2026-07-25T14:47:43Z
- **Completed:** 2026-07-25T14:54:47Z
- **Tasks:** 3 completed
- **Files modified:** 3 (all created)

## Accomplishments

- Implemented `src/trading_crab_lib/platform/evaluation/report.py`:
  - `assemble_backtest_report(*, sojourn_lag, strategy_kpis, ablation_kpis, baseline_kpis, gap) -> str` — a pure markdown builder (no I/O) with a FIXED section order: (1) Headline: sojourn/detection-lag ratio (D-01a, first metrics section), (2) Faber comparison (§23.1, standing target not a gate), (3) no-regime-ablation delta ("does the regime layer pay rent"), (4) smoothed-vs-filtered gap, (5) baseline gauntlet table (SPY/60-40/Faber + strategy + ablation rows, log wealth AND max drawdown), (6) strategy KPI table (terminal log wealth, max drawdown+duration, CVaR(5%), turnover, in-sample crisis capture) plus a documented "Conventions" section explaining the cash-return symmetry (review F4) and the turnover/cost symmetry (A5).
  - `write_backtest_report(markdown, artifacts, *, output_dir=None) -> Path` — always writes `backtest_report.md` (distinct filename from `report/weekly.py`'s `weekly_report.md`, no collision) plus one `backtest_{name}.parquet` per artifact dict entry.
  - `run_full_backtest_evaluation(monthly_features, monthly_raw, cfg, *, registry_path=None, output_dir=None) -> dict` — the full orchestration: `build_core_research_series` + `compute_monthly_returns` for equity/bond/cash return series -> `run_backtest` (strategy, `cash_returns=cash_ret`, review F4) -> `no_regime_ablation` (same cash_ret) -> the three holdout-bounded price baselines -> ONE full-sample `fit_jump_model`/`canonicalize_states` fit for the smoothed reference labeling (Pitfall 1) -> `build_filtered_probs_matrix` + `compute_sojourn_lag_headline` (review F1, multiclass, never a class-agnostic max) -> date-joined `y_true` with a hard length assertion (review F2) -> per-leg KPIs + a hindsight-oracle smoothed-vs-filtered gap -> `report_model_metrics` -> assemble + write.
  - `main(argv)` CLI entry point loading `load_platform_config()` + the platform checkpoint manager's `monthly_features`/`monthly_raw`, plus `if __name__ == "__main__": raise SystemExit(main())`.
- **Review F1 wiring (first real exercise against a driver run):** the headline's filtered-probs input is `sojourn_lag.build_filtered_probs_matrix(per_step_metrics)` — a genuine multiclass matrix — passed to `compute_sojourn_lag_headline`, never a class-agnostic max series.
- **Review F2 wiring:** `per_step_metrics["y_true"]` is set by reindexing the smoothed reference states onto `per_step_metrics["dates"]`; a `ValueError` is raised if any decision date has no corresponding smoothed reference state (should never happen on real data, but the guard is load-bearing), and a hard length assertion runs before `report_model_metrics` is called.
- **Review F4 wiring:** `cash_ret` (from `build_core_research_series` -> `compute_monthly_returns`) is passed as `cash_returns` into BOTH `run_backtest` (strategy) and `no_regime_ablation`; the investable `asset_returns` frame fed to both explicitly EXCLUDES the "cash" splice class (FZFXX) so cash is never tilted into as a risk position — it is purely the vol-target residual.
- **Pitfall 1 wiring:** the smoothed reference labeling is a single full-sample `fit_jump_model` call over the holdout-bounded dev window (non-causal by design, per the labeler's own documented batch behavior) — a genuinely distinct object from the walk-forward driver's `per_step_metrics`, proven by differing lengths in the integration test.
- **D-04 grep gate:** `grep -niE "deflated|dsr" src/trading_crab_lib/platform/evaluation/report.py` returns nothing — required rewording the module docstring's own explanation of the prohibition (see Decisions).
- Full suite: **1130 passed**, no regressions (was 1120 after Plan 05).

## Task Commits

Each task was committed atomically (TDD RED→GREEN for Task 1):

1. **Task 1: Write the report unit test (RED)** — `90003fd` (test) — `TestHeadlineOrdering`, `TestBaselineGauntletSection`, `TestArtifactsWritten` (6 tests) fail at collection because `trading_crab_lib.platform.evaluation.report` did not yet exist.
2. **Task 2: Implement assemble_backtest_report + write_backtest_report** — `511686d` (feat) — all 6 Task-1 tests pass; grep gate for `deflated|dsr` confirmed clean.
3. **Task 3: Wire run_full_backtest_evaluation + main() CLI + synthetic e2e test** — `396a873` (feat) — `tests/integration/test_mini_backtest.py` added (4 tests including `TestYTrueDateAlignment`); full suite green (1130 passed).

## Files Created/Modified

- `src/trading_crab_lib/platform/evaluation/report.py` — new module: `assemble_backtest_report`, `write_backtest_report`, `_leg_kpis`, `_build_kpi_table`, `_smoothed_hindsight_perf`, `run_full_backtest_evaluation`, `main`, `__main__` hook.
- `tests/unit/test_platform_evaluation_report.py` — new test file: 3 classes, 6 tests, synthetic precomputed inputs only.
- `tests/integration/test_mini_backtest.py` — new test file: 2 classes, 4 tests, synthetic 48-month end-to-end run (real jump-model/nowcaster fits, no network, no real checkpoints).

## Decisions Made

- **Smoothed-vs-filtered gap via a hindsight-oracle tilt.** The plan's `<action>` text names `compute_gap(smoothed_perf, filtered_perf)` but doesn't literally specify how `smoothed_perf` is computed. Rather than inventing a new formula, `_smoothed_hindsight_perf` reuses `vol_targeted_tilt`/`returns_by_regime_stats` UNCHANGED, driven by a one-hot probability on the full-sample smoothed state at each of the SAME decision dates the walk-forward driver actually visited — this is the literal "what would the strategy have earned with regime hindsight but no return foresight" comparator §5.4 describes, and it keeps the gap's two inputs genuinely distinct (Pitfall 1) without adding new allocation math (05-RESEARCH.md's "essentially zero new numerical algorithms" doctrine).
- **`asset_returns` fed to `run_backtest` excludes the "cash" splice class.** Cash (FZFXX) is the vol-target residual (`tilt["cash"]`), never a risk position to be tilted into; including it as an investable asset would let the regime-conditional Sharpe ranking "buy" cash as a normal (near-zero-vol, high-Sharpe) asset, corrupting the intended architecture where cash is always the passive 1-scale residual.
- **D-04 grep-gate rewording.** The docstring originally spelled out "deflated Sharpe (DSR)" to document the prohibition — which then tripped its own grep gate. Reworded to "the design-Phase-6 registry-denominator risk-adjusted statistic" (mirrors Plan 02's identical false-positive fix for `get_holdout_checkpoint_manager(`).
- **`__main__` hook calls `main()` directly** (`raise SystemExit(main())`) rather than running a standalone synthetic self-check, mirroring `report/weekly.py`'s pattern exactly — `main()` requires real Phase 1 checkpoints, so there is no meaningful synthetic self-check for the orchestrator itself; the integration test is the runnable no-network proof instead.

## Deviations from Plan

None — the plan's `<action>` blocks were followed as written for all three tasks, including the exact `must_haves.truths` wording for F1/F2/F4 and the headline-first section ordering (D-01a). The only interpretive gap-filling was `_smoothed_hindsight_perf`'s concrete formula (documented above under Decisions), which the plan intentionally left as an orchestration detail rather than a literal formula.

## Issues Encountered

- A manual sanity-check run (outside pytest) of `run_full_backtest_evaluation` against synthetic data was used to validate the full wiring before writing the formal integration test — this was ephemeral (all paths under `tempfile.TemporaryDirectory()`), and `git status --short` was checked afterward to confirm no stray `registry/` directory or other artifact leaked into the repo root (none did).

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- Plan 07 (the final plan in Phase 5) can now build against a real, wired `run_full_backtest_evaluation` — the report layer's `main()` CLI is ready to run against real Phase 1-4 checkpoints (`monthly_features`, `monthly_raw`) once they exist on disk.
- All four EVAL requirements (EVAL-01..04) are now fully wired end-to-end in one orchestration function, closing the honest-backtest-evaluation phase's central integration risk.
- No blockers.

---
*Phase: 05-honest-backtest-evaluation*
*Completed: 2026-07-25*

## Self-Check: PASSED

All 3 created files confirmed present on disk; all 3 commit hashes
(90003fd, 511686d, 396a873) confirmed present in `git log --oneline --all`.
