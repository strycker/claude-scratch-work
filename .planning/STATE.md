---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
current_phase: 05
current_phase_name: honest-backtest-evaluation
status: executing
stopped_at: Phase 5 Plan 5 complete (baseline gauntlet: SPY, 60/40, Faber SMA, no-regime ablation)
last_updated: "2026-07-25T14:37:32.365Z"
last_activity: 2026-07-25
last_activity_desc: Completed 05-05-PLAN.md (baseline gauntlet: SPY, 60/40, Faber SMA, no-regime ablation)
progress:
  total_phases: 6
  completed_phases: 4
  total_plans: 28
  completed_plans: 26
  percent: 93
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-07-09)

**Core value:** Honest, regime-aware weekly guidance that beats buy-and-hold SPY net of
avoided drawdowns — never fooled by its own backtest.
**Current focus:** Phase 05 — honest-backtest-evaluation

## Current Position

Phase: 05 (honest-backtest-evaluation) — EXECUTING
Plan: 6 of 7
Status: Ready to execute
Last activity: 2026-07-25 — Completed 05-05-PLAN.md (baseline gauntlet: SPY, 60/40, Faber SMA, no-regime ablation)

Progress: [█████████░] 93%

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

### Pending Todos

- Verifier informational note: `holdout.py`/`registry.py` use hardcoded constants that
  match `config/platform_settings.yaml` sections rather than reading them — wire to
  config if/when the values ever need to change (future-divergence risk only).

### Blockers/Concerns

- ~~Live 1962+ data run still pending FRED_API_KEY in the claude.ai/code environment
  (Phase 1 human-verification item).~~ **RESOLVED 2026-07-23** — `FRED_API_KEY` is present
  in the environment (32-char key) and functionally verified against the live FRED API
  (authenticated GDP series fetch succeeded). No longer a blocker; do not re-flag.

## Deferred Items

Items acknowledged and carried forward from previous milestone close:

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| *(none — first milestone)* | | | |

## Session Continuity

Last session: 2026-07-25T14:35:23.065Z
Stopped at: Phase 5 Plan 5 complete (baseline gauntlet: SPY, 60/40, Faber SMA, no-regime ablation)
Resume file: None
