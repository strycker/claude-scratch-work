# Phase 2: Honesty Infrastructure - Context

**Gathered:** 2026-07-16
**Status:** Ready for planning

<domain>
## Phase Boundary

The structural honesty guarantees installed before any model is tuned: physical 2021+
holdout carve, trial registry, walk-forward runner, purged/embargoed CV splitter,
causal-feature gating, and the smoothed-vs-filtered gap + detection-lag metrics.
Requirements HON-01…06. No modeling (Phase 3), no report wiring (Phase 4).

</domain>

<decisions>
## Implementation Decisions

### Trial registry (HON-02)
- **D-01:** Append-only JSONL, **committed to git** — an immutable pre-registration ledger.
  One line per evaluated configuration: config hash, feature set, params, metrics, timestamp,
  git SHA of the code that produced it. Survives machine loss; git history is the
  tamper-evidence. Location: a tracked path (e.g. `registry/trials.jsonl`) — NOT under
  gitignored `data/`. This ledger is the multiple-testing denominator for the deflated
  Sharpe at design freeze (design §8.4, §22); losing it invalidates the freeze evaluation.
- **D-02:** A trial = one evaluated configuration (its full walk-forward result), not one
  walk-forward step. All grid cells logged regardless of outcome. No manual bookkeeping —
  the runner logs automatically.

### Holdout carve (HON-01)
- **D-03:** **Ingestion-level split.** Dev checkpoint files literally end at 2020-12-31;
  rows dated 2021+ are written only to a separate `data/holdout/` tree. The default
  pipeline cannot peek because the data is not in its files. Live-scoring mode explicitly
  loads the holdout tree (sanctioned opt-in per the PROJECT.md holdout rule: live refits
  on full history, results firewalled from selection until freeze).
- **D-04:** [informational] The carve applies to the platform (monthly) checkpoint namespace built in
  Phase 1. The frozen quarterly incumbent is exempt (it predates the holdout discipline
  and is baseline-only).

### Gap/lag reporting surface (HON-05)
- **D-05:** CLI run output + persisted artifact (parquet/JSON under
  `outputs/reports/model_metrics/` alongside the salvaged metrics-artifacts pattern) in
  this phase. Weekly-report wiring happens in Phase 4 with the rest of the report — Phase 2
  stays pure infrastructure and does not touch the incumbent's report path.

### Carried forward (locked earlier — do not re-decide)
- Holdout operating rule: dev/tuning use only ≤2020-12 walk-forward results; live weekly
  scoring refits on full history but post-2021 performance is never consulted for model
  selection until design freeze (PROJECT.md Constraints).
- New code self-contained in `src/trading_crab_lib/platform/`; frozen incumbent untouched
  (Phase 1 D-01/D-02).
- Salvage ports: `ideas/gsd-salvage/prediction/feature_gating.py` seeds HON-06 (causal
  gating, loud opt-out); `ideas/gsd-salvage/prediction/model_metrics_artifacts.py` seeds
  the HON-05/EVAL-04 metrics-artifact side. Port into `platform/`, adapt to monthly
  checkpoints — do not import from `ideas/`.
- Walk-forward semantics are design-fixed (§8.1): expanding window, refit on data ≤ t,
  record decisions, step forward. Runner must execute a trivial model end-to-end (Phase 2
  exit); real models arrive in Phase 3.

### Claude's Discretion
- Purged/embargoed CV parameters (purge = label horizon; embargo length) per López de
  Prado ch. 7 — researcher confirms conventions; expose in config.
- Exact registry JSONL schema and module layout inside `platform/`.
- How the trivial model for HON-03's end-to-end proof is defined (e.g. persistence
  classifier).

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Design (authoritative)
- `platform_design/platform_design.md` §8 (evaluation & honesty framework — all six
  mechanisms), §5.4 (gap/lag metric definitions), §6.5 (overlapping-label CV discipline),
  §22 (registry logs every grid cell), D13 (holdout lock).

### Requirements & planning
- `.planning/REQUIREMENTS.md` — HON-01…06.
- `.planning/ROADMAP.md` — Phase 2 goal + 5 success criteria.
- `.planning/phases/01-monthly-data-layer-long-histories/01-CONTEXT.md` — Phase 1 locked
  decisions (architecture, checkpoint namespace).
- `.planning/phases/01-monthly-data-layer-long-histories/01-07-SUMMARY.md` — the
  `monthly_features` table + checkpoint the guards wrap around.

### Salvage (port, don't import)
- `ideas/gsd-salvage/prediction/feature_gating.py` — causal-gating pattern for HON-06.
- `ideas/gsd-salvage/prediction/model_metrics_artifacts.py` — metrics-artifact pattern.
- `ideas/gsd-salvage/README.md` — port verdicts.

### Codebase
- `src/trading_crab_lib/platform/` — checkpoints.py (namespace), config.py,
  transforms_monthly.py (produces what the holdout carve splits).
- `CLAUDE.md` (root) — conventions; pitfall P1 (the sin HON-06 guards against).

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `platform/checkpoints.py`: checkpoint factory — extend with the holdout-split write path
  and the default-loader truncation invariant.
- `platform/transforms_monthly.py`: `monthly_features` assembly — the natural seam where
  the ≤2020-12 / 2021+ split happens at write time.
- Salvaged `feature_gating.py` / `model_metrics_artifacts.py` (see canonical refs).

### Established Patterns
- Config-driven everything (`config/platform_settings.yaml` gets `holdout:`, `registry:`,
  `cv:` sections).
- Tests network-mocked; synthetic frames for unit tests; invariant-style tests (see
  Phase 1's lean-set invariant test) — the holdout guard deserves the same: a test that
  proves the default loader CANNOT return post-2020 rows.

### Integration Points
- Registry + runner live in `platform/` (e.g. `platform/honesty/` or flat modules —
  planner's call).
- Nothing in this phase touches the incumbent pipeline or its reports.

</code_context>

<specifics>
## Specific Ideas

- The registry ledger's credibility IS the product (Core Value: "never fooled by its own
  backtest") — prefer boring, inspectable JSONL over clever storage.
- Holdout guard should fail loudly and unmistakably if code attempts default-path access
  to holdout data (raise, not warn).

</specifics>

<deferred>
## Deferred Ideas

- Weekly-report wiring of gap/lag metrics — Phase 4 (D-05).
- DSR computation against the registry — Phase 6 of the design (freeze), later milestone.

</deferred>

---

*Phase: 2-Honesty Infrastructure*
*Context gathered: 2026-07-16*
