# Phase 3: Regime Labeling & Prediction - Context

**Gathered:** 2026-07-22
**Status:** Ready for planning

<domain>
## Phase Boundary

L1 regime labeling (jump model on the 1962+ lean monthly features) and L2 regime
nowcasting (calibrated probabilities from causal features), plus the label-churn
monitoring metric and the empirical transition matrix diagnostic. Requirements
L1-01…03, L2-01…02. No asset prediction (Phase 4), no allocation/report (Phase 4),
no full backtest (Phase 5).

</domain>

<decisions>
## Implementation Decisions

### Embargo window (L1-02)
- **D-01:** Trailing **12 months** of labels are embargoed from L2 training (the
  conservative end of the 6–12 month requirement range; matches the trailing-24-month
  churn measurement's sensitive edge). Expose in `config/platform_settings.yaml` under
  the existing `cv:`/new `labeling:` section, default 12.

### §4.4 acceptance criteria enforcement (L1-01)
- **D-02:** **Report-only in v1.** The labeler always completes; occupancy, sojourn,
  and per-state stats are computed, logged loudly (WARNING on violation), and persisted
  as a diagnostics artifact. Hard-gating is deferred until (K, λ) grid tuning in v2
  (L1-V2-01) — tracer-bullet spirit: every layer present, every layer naive, never
  blocked on tuning.

### Soft label confidences (L1-02)
- **D-03:** **Softmax over negative squared distances to centroids** (temperature-free
  v1). Cheap, self-contained, no new dependencies; produces the per-month confidence
  vector persisted alongside hard labels. The probabilistically-founded alternative
  (companion HMM forward-backward γ) is explicitly deferred to the v2 t-HMM benchmark
  (L1-V2-01) — do not pull hmmlearn into the platform namespace in v1.

### Regime naming (v1 scope)
- **D-04:** **Numeric labels (0–4) + auto-generated one-line economic profiles** per
  state (mean/sign summary of key features per regime, mirroring the incumbent's
  `suggest_names()` idea but simpler). Human-pinned names deferred to Phase 4 when the
  weekly report needs display names. No `platform_regime_labels.yaml` in this phase.

### Carried forward (locked earlier — do not re-decide)
- Jump model spec is design-fixed (§4.1): k-means + per-jump penalty λ, exact DP decode
  (O(TK²) Viterbi-like), multi-restart, k-means warm start, **default λ, K=5** (L1-01).
  λ/K grid tuning against §4.4 criteria is v2 (L1-V2-01).
- Nowcaster is a **calibrated multinomial logistic** on causal features, returning a
  probability distribution, never argmax (L2-01, §5.1). Recursive prior-state feature
  and γ sample weights are v2 (L2-V2-01).
- Transition machinery in v1 = **empirical transition matrix diagnostic only** (L2-02);
  feature-conditional TVTP model is v2 (L2-V2-02).
- All training/tuning through Phase 2's rails: dev data ≤2020-12 (holdout carve),
  `PurgedEmbargoedKFold` for any CV, `assert_causal_features()` gating on supervised
  paths, walk-forward via the frozen `expanding_steps`/`run_walkforward` interface,
  every evaluated configuration auto-logged to `registry/trials.jsonl`.
- New code self-contained in `src/trading_crab_lib/platform/` (e.g. `platform/labeling/`
  and/or `platform/prediction/` — planner's call); frozen quarterly incumbent untouched.
- Label churn metric (L1-03): fraction of trailing labels revised per refresh, computed
  after each run (design §5.4 defines it over the trailing 24 months).

### Claude's Discretion
- Default λ value and how it is derived (e.g. scan a small range and pick the smallest λ
  satisfying report-only sojourn sanity, or a literature-informed constant) — researcher
  confirms convention; expose λ and K in config.
- Exact DP implementation details (vectorization, restart count, convergence tolerance).
- Calibration method for the nowcaster (isotonic vs sigmoid via
  `CalibratedClassifierCV`) and CV arrangement, provided it uses PurgedEmbargoedKFold.
- Feature standardization/winsorization choices for the labeler (§4.1 robustness note).
- Module layout inside `platform/` and checkpoint names for labels/confidences/matrix.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Design (authoritative)
- `platform_design/platform_design.md` §4.1 (jump model objective + solver), §4.3
  (geometric-duration fix hierarchy — regime-age is v2), §4.4 (acceptance criteria to
  report on), §5.1 (nowcaster spec incl. transition-window headline metric), §5.3
  (anti-flicker — consumed in Phase 4, produced-for here), §5.4 (label churn definition),
  D4/D5 (two-stage labeling/prediction split), R2 (jump model replaces balanced k-means,
  k-means warm start), §14 Phase 1 tracer bullet.

### Requirements & planning
- `.planning/REQUIREMENTS.md` — L1-01…03, L2-01…02 (and v2 deferrals L1-V2-01,
  L2-V2-01/02 that bound this phase).
- `.planning/ROADMAP.md` — Phase 3 goal + 5 success criteria.
- `.planning/phases/02-honesty-infrastructure/02-CONTEXT.md` — honesty rails this phase
  must run inside.
- `.planning/phases/01-monthly-data-layer-long-histories/01-CONTEXT.md` — monthly spine
  + lean feature set decisions.

### Codebase
- `src/trading_crab_lib/platform/honesty/walkforward.py` — the frozen interface Phase 3
  models plug into (`expanding_steps`, `run_walkforward`).
- `src/trading_crab_lib/platform/honesty/{cv,gating,holdout,registry}.py` — CV splitter,
  causal gate, holdout managers, trial ledger.
- `src/trading_crab_lib/platform/transforms_monthly.py` + `taxonomy.py` — the
  `monthly_features` table and `lean_feature_set()` the labeler consumes.
- `src/trading_crab_lib/clustering.py` — incumbent k-means (reference for warm start
  idiom only; do not import incumbent code paths into platform/).
- `CLAUDE.md` (root) — conventions; pitfall P1.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `platform/checkpoints.py` checkpoint factory — labels, confidences, and transition
  matrix persist through the platform namespace (dev tree ends 2020-12 by construction).
- `platform/honesty/walkforward.py` — the nowcaster's walk-forward evaluation runs
  through `run_walkforward` (swap `DummyClassifier` for the calibrated logistic);
  registry logging comes free.
- sklearn already installed: `KMeans` (warm start), `LogisticRegression`,
  `CalibratedClassifierCV` — zero new dependencies expected for this phase.

### Established Patterns
- Config-driven everything: new `labeling:` section (K, lambda, n_restarts, embargo_months)
  in `config/platform_settings.yaml`, read defensively via `.get()` (02-01 pattern —
  don't extend `_REQUIRED_PLATFORM_SECTIONS`).
- Synthetic-frame, no-network unit tests; invariant-style tests (e.g. a test that proves
  the nowcaster's training window excludes embargoed months).
- TDD RED→GREEN commits per task.

### Integration Points
- Labeler input: `monthly_features` checkpoint (lean 1962+ columns via
  `lean_feature_set()`).
- Nowcaster input: causal features through `assert_causal_features()`; targets are the
  labeler's output minus the 12-month embargo.
- Nothing touches the incumbent pipeline or its reports.

</code_context>

<specifics>
## Specific Ideas

- The DP decode is exact — no greedy/heuristic shortcuts; the leakage-free warm start
  and multi-restart determinism should be tested the same way Phase 2 tested the
  leakage invariant (headline invariant tests).
- Nowcaster evaluation must report transition-window accuracy separately from overall
  accuracy (§5.1 headline metric) — overall accuracy alone is the trap the design warns
  about (~90% from persistence).
- Embargo must be structural, not advisory: the training-set builder physically excludes
  the trailing 12 months of labels, with a test proving it.

</specifics>

<deferred>
## Deferred Ideas

- (K, λ) grid tuning against §4.4 acceptance criteria; subsample stability with
  Hungarian matching; t-HMM benchmark — v2 (L1-V2-01).
- Recursive prior-state feature, γ sample weights, transition-window-weighted training —
  v2 (L2-V2-01).
- Feature-conditional TVTP transition model with regime age — v2 (L2-V2-02).
- Human-pinned regime display names — Phase 4 (weekly report).
- Hard-gating §4.4 acceptance criteria — revisit with v2 tuning.

</deferred>

---

*Phase: 3-Regime Labeling & Prediction*
*Context gathered: 2026-07-22*
