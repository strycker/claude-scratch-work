# Roadmap: Trading-Crab

## Overview

Trading-Crab's existing 9-step quarterly pipeline stays frozen as the incumbent baseline
while a new regime-conditional platform is built alongside it in `src/trading_crab_lib/`.
The journey follows design §14's tracer-bullet principle: lay the monthly data foundation,
then install the honesty framework (holdout carve, trial registry, walk-forward runner,
purged CV) *before* any model is tuned, then build one thin end-to-end pass through every
modeling layer (regime labeling → nowcasting → asset prediction → allocation & report),
prove it honestly on a 1972–2020 walk-forward backtest against real baselines, and finally
migrate the validated skeleton to the public `strycker/trading-crab` repo.

## Phases

**Phase Numbering:**

- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Monthly Data Layer & Long Histories** - Monthly-frequency ingestion, spliced 1962+ core-asset histories, ALFRED vintages, and a fast/slow/agency feature taxonomy (completed 2026-07-15)
- [x] **Phase 2: Honesty Infrastructure** - Physical 2021+ holdout carve, trial registry, walk-forward runner, purged/embargoed CV, and causal-feature gating — installed before any model is tuned (completed 2026-07-22)
- [x] **Phase 3: Regime Labeling & Prediction** - Jump-model regime labeler plus calibrated logistic nowcaster, both walk-forward safe (completed 2026-07-22)
- [x] **Phase 4: Asset Prediction & Allocation** - Returns-by-regime tables, EWMA vol, naive vol-targeted allocation, weekly report, and a minimal daily tripwire (completed 2026-07-23)
- [x] **Phase 5: Honest Backtest & Evaluation** - Full 1972–2020 walk-forward backtest vs. baseline gauntlet with first-class honesty metrics (completed 2026-07-27, closed 2026-08-04)
- [x] **Phase 6: Platform Notebook Suite** - Six EDA + human-in-the-loop validation notebooks (P1–P6) covering L0–L4 and evaluation (completed 2026-09-10)
- [ ] **Phase 7: Regime Representation** - Resolve A13/A15 (one feature policy for driver and report), then add an independent leadership-axis classifier on relative/invariant features (absorbs INV-01)
- [ ] **Phase 8: Migration to Public Repo** - Platform decoupled and migrated to `strycker/trading-crab`, tests green in CI, docs updated

## Phase Details

### Phase 1: Monthly Data Layer & Long Histories

**Goal**: The pipeline ingests and transforms monthly data with long spliced histories back
to ~1962, point-in-time vintages where available, and a documented feature taxonomy —
replacing the quarterly-only spine as the foundation for regime modeling.
**Depends on**: Nothing (first phase)
**Requirements**: DATA-01, DATA-02, DATA-03, DATA-04, DATA-05, DATA-06
**Success Criteria** (what must be TRUE):

  1. Running feature engineering produces a monthly-frequency dataset (not quarterly), with
     quarterly agency series correctly lagged/aligned to their monthly publication cadence.

  2. Core asset histories (S&P total return, Treasury total-return synthetic, gold, oil,
     cash) are available back to ~1962, with splicing rules documented per asset.

  3. Agency series (e.g. GDP, CPI) pull ALFRED point-in-time vintages where archives exist,
     with a documented publication-lag-alignment fallback for the pre-vintage era.

  4. Every feature is classified fast/slow/agency in config, and a lean full-history
     (1962+) feature set is defined and usable for labeling.

  5. Satellite ETFs and Glenn's holdings ingest with NULL-tolerant handling for shorter
     histories; paid-provider adapter seams (Norgate/Tiingo/EODHD) are documented as
     placeholders only, with stockcharts.com/finviz.com noted as candidate sources.
**Plans**: 7/7 plans complete
**Wave 1**

- [x] 01-01-PLAN.md — Foundation: platform subpackage scaffold, checkpoint namespace, config loader, fast/slow/agency taxonomy (DATA-04)

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 01-02-PLAN.md — Core-asset splicing engine + long-history synthetics + splicing_rules.md (DATA-02)
- [x] 01-03-PLAN.md — ALFRED point-in-time vintage fetch + reconstruction + fallback doc (DATA-03)
- [x] 01-04-PLAN.md — Paid-provider adapter seams (placeholder stubs + doc) (DATA-06)
- [x] 01-05-PLAN.md — Monthly macro ingestion (FRED/multpl/macrotrends at monthly cadence) (DATA-01)
- [x] 01-06-PLAN.md — Universe daily price ingestion, NULL-tolerant short histories (DATA-05)

**Wave 3** *(blocked on Wave 2 completion)*

- [x] 01-07-PLAN.md — Monthly transforms: agency alignment + lean feature assembly + tagging (DATA-01, DATA-03, DATA-04)

### Phase 2: Honesty Infrastructure

**Goal**: Every subsequent modeling result is protected by structural honesty guarantees —
a physically separate holdout, a trial registry, a walk-forward runner, purged CV, and
causal-feature gating — installed before any model is tuned.
**Depends on**: Phase 1
**Requirements**: HON-01, HON-02, HON-03, HON-04, HON-05, HON-06
**Success Criteria** (what must be TRUE):

  1. Data dated 2021+ lives in separate files/paths the default dev pipeline cannot read;
     live-scoring mode must opt in explicitly to access it.

  2. Every evaluated configuration (features, params, metrics) is automatically logged to a
     trial registry store and is queryable after a run, with no manual bookkeeping.

  3. A walk-forward runner refits on data ≤ t at each step, executes a trivial model
     end-to-end, and records the decision made at each step.

  4. Purged + embargoed CV splitting is available as a drop-in replacement for
     `TimeSeriesSplit` for any supervised component with overlapping labels.

  5. Supervised training paths load causal (not centered/look-ahead) features by default
     with a loud opt-out; smoothed-vs-filtered gap and detection lag are computed and
     reported as first-class run outputs.
**Plans**: 5/5 plans complete

**Wave 1**

- [x] 02-01-PLAN.md — Foundation config sections + honesty package + physical 2021+ holdout carve (HON-01)

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 02-02-PLAN.md — Append-only JSONL trial registry, git-tracked ledger (HON-02)
- [x] 02-03-PLAN.md — PurgedEmbargoedKFold BaseCrossValidator, hand-rolled (HON-04)
- [x] 02-04-PLAN.md — Causal-feature gating guard + smoothed-vs-filtered gap/detection-lag metrics + artifact surface (HON-05, HON-06)

**Wave 3** *(blocked on Wave 2 completion)*

- [x] 02-05-PLAN.md — Expanding-window walk-forward runner with automatic registry logging (HON-03)

### Phase 3: Regime Labeling & Prediction

**Goal**: The system can label historical market regimes with a temporally-persistent
jump model and nowcast today's regime with calibrated probabilities, using only causal
information.
**Depends on**: Phase 1, Phase 2
**Requirements**: L1-01, L1-02, L1-03, L2-01, L2-02
**Success Criteria** (what must be TRUE):

  1. Running the labeler on the 1962+ monthly feature set produces regime labels (default
     λ, K=5) via a k-means-warm-started jump model with per-jump penalty, exact DP decode,
     and multiple restarts.

  2. Labels come with soft confidences and are persisted; the trailing 6–12 months of
     labels are marked/embargoed so L2 training cannot see them.

  3. A label-churn metric (fraction of trailing labels revised on each refresh) is computed
     and available for monitoring after each run.

  4. Given causal features through today, the nowcaster returns a calibrated probability
     distribution over regimes (not a single argmax class).

  5. An empirical transition matrix is available as a diagnostic showing the forward
     regime distribution implied by history.
**Plans**: 4/4 plans complete

**Wave 1**

- [x] 03-01-PLAN.md — Jump-model labeler core: exact DP decode, multi-restart alternation, k-means warm start, canonicalization, soft confidences + labeling config (L1-01)
- [x] 03-03-PLAN.md — Calibrated nowcaster: structural 12-month embargo, CalibratedClassifierCV + PurgedEmbargoedKFold, transition-window accuracy, registry logging (L2-01, L1-02)

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 03-02-PLAN.md — Labeling persistence + label churn + report-only §4.4 diagnostics + auto profiles (L1-02, L1-03)
- [x] 03-04-PLAN.md — Empirical transition matrix diagnostic (L2-02)

### Phase 4: Asset Prediction & Allocation

**Goal**: Glenn can open a weekly report that tells him, per regime, which assets look
favorable, what volatility-targeted portfolio mix to hold, what trades are implied versus
his current holdings, and whether any tripwire condition demands he act sooner.
**Depends on**: Phase 3
**Requirements**: L3-01, L3-02, L4-01, L4-02, L4-03, L4-04
**Success Criteria** (what must be TRUE):

  1. Returns-by-regime tables report historical return/risk stats per asset conditional on
     regime for the v1 asset universe.

  2. EWMA volatility forecasts are computed per asset and available to size positions and
     feed the tripwire.

  3. A naive vol-targeted regime-tilt allocation is produced with hysteresis bands
     (act ~0.7 / unwind ~0.4) so the target mix doesn't flip on every small change.

  4. The weekly report (via existing email machinery) shows current regime distribution +
     trajectory, per-asset signals, and target-vs-current mix with trades implied — with
     current holdings sourced from a manual per-account YAML file (a Fidelity CSV parser
     seam is documented as a placeholder only).

  5. A minimal daily tripwire monitor combines 3 independent signals (e.g. vol spike,
     credit-spread velocity, drawdown-from-peak) with OR-logic into one escalation output:
     none / "run weekly scoring early" / "Tier-1 de-risk review."
**Plans**: 5/5 plans complete

**Wave 1**

- [x] 04-01-PLAN.md — Data gaps + config foundation: SPY into universe, daily DAAA/DBAA ingestion (macro_daily.py), allocation/tripwire/report config sections (L4-04)
- [x] 04-02-PLAN.md — Returns-by-regime tables + EWMA vol forecasts (L3-01, L3-02)

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 04-03-PLAN.md — Vol-targeted regime-tilt allocation + hysteresis state machine (L4-01)
- [x] 04-04-PLAN.md — Daily tripwire monitor: 3-signal OR-logic escalation CLI (L4-04)

**Wave 3** *(blocked on Wave 2 completion)*

- [x] 04-05-PLAN.md — Weekly report (regime dist + trajectory + trades implied) + per-account holdings YAML (L4-02, L4-03)

### Phase 5: Honest Backtest & Evaluation

**Goal**: The full tracer-bullet pipeline (L1–L4) can be evaluated honestly over
1972–2020 walk-forward, against real baselines, with first-class metrics that reveal
whether regime timing is actually worth anything.
**Depends on**: Phase 4
**Requirements**: EVAL-01, EVAL-02, EVAL-03, EVAL-04
**Success Criteria** (what must be TRUE):

  1. Running the walk-forward backtest over 1972–2020 executes all layers (L1 labeler →
     L2 nowcaster → L3 returns/vol → L4 allocation) end-to-end and produces a result,
     without touching the 2021+ holdout.

  2. The backtest report includes a baseline gauntlet — buy-and-hold SPY, 60/40, and
     Faber's 10-month SMA — computed over the same window for direct comparison.

  3. The sojourn/detection-lag ratio is reported prominently as the headline go/no-go
     number for whether regime timing is adding value.

  4. Model metrics artifacts (multiclass Brier score, calibration bins, confusion tables)
     are persisted to disk per run for later inspection and comparison.
**Plans**: 7/7 plans complete

**Wave 1**

- [x] 05-01-PLAN.md — Foundation: backtest: config section + backtest/evaluation package scaffolds + transaction-cost/turnover identity (EVAL-01, D-03)

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 05-02-PLAN.md — Walk-forward backtest driver: refit L1→L4 on data ≤ t, holdout-bounded, one registry trial (EVAL-01)
- [x] 05-03-PLAN.md — Strategy KPIs + sojourn/detection-lag headline orchestration (EVAL-01, EVAL-03)
- [x] 05-04-PLAN.md — Model-metrics artifacts: multiclass Brier, calibration bins, confusion tables (EVAL-04)

**Wave 3** *(blocked on Wave 2 completion)*

- [x] 05-05-PLAN.md — Baseline gauntlet: SPY, 60/40, Faber 10-month SMA + no-regime ablation (EVAL-02)

**Wave 4** *(blocked on Wave 3 completion)*

- [x] 05-06-PLAN.md — Backtest report assembly (headline-first) + synthetic end-to-end integration (EVAL-01, EVAL-02, EVAL-03, EVAL-04)

**Wave 5** *(blocked on Wave 4 completion)*

- [x] 05-07-PLAN.md — Real 1972–2020 run against live checkpoints — blocking human-verify (design §14 Phase 1 exit) (EVAL-01..04)

### Phase 6: Platform Notebook Suite

**Goal**: Every platform layer (L0–L4) and the Phase 5 evaluation have a notebook that
shows the data and renders the diagnostics — a **periodic verification-and-validation
surface** (are the regimes still well-defined? is the data still behaving as historic?),
plus **one cold-start selection gate at P3** where a human signs off on the regime
labeling before it is trusted for live use.
**Depends on**: Phase 5
**Requirements**: NB-01
**Success Criteria** (what must be TRUE):

  1. Six notebooks exist (`P1_data_spine`, `P2_features_taxonomy`, `P3_regime_labeling`,
     `P4_nowcaster`, `P5_assets_allocation`, `P6_backtest_evaluation`) and each runs
     top-to-bottom against real checkpoints without error.

  2. `P3_regime_labeling` ends in a cold-start sign-off cell where the operator records
     date, verdict, and reasoning for the regime labeling. The other five notebooks are
     periodic check surfaces — they state what to look for, but carry no per-run gate.

  2b. Notebooks read the full data span (including post-2020) via the explicit
     `get_holdout_checkpoint_manager()` opt-in, so drift in the live era is visible;
     model-fitting paths remain fenced at 2020-12 via the default dev manager.

  3. All plotting logic lives in `platform/plotting/` library functions called by the
     notebooks — never defined inline (ADR #11 convention, applied to the platform).

  4. `P3_regime_labeling` shows the 5 regimes against dated economic history (recessions,
     inflation eras, credit events) so a human can judge whether the labels are real.

  5. `P6_backtest_evaluation` renders the equity curves, baseline gauntlet, ablation
     delta, calibration, and the sojourn/lag headline with its resolved-transition count.
**Plans**: 7/7 plans executed

- [x] 06-01-PLAN.md — plotting core/loaders/data/drift
- [x] 06-02-PLAN.md — artifact persistence (additive write in evaluation/report.py)
- [x] 06-03-PLAN.md — P3_regime_labeling (+ A13 display)
- [x] 06-04-PLAN.md — P2_features_taxonomy
- [x] 06-05-PLAN.md — P4_nowcaster
- [x] 06-06-PLAN.md — P5_assets_allocation
- [x] 06-07-PLAN.md — P6_backtest_evaluation

Verified 2026-09-10 (`06-VERIFICATION.md`, 5/5 criteria, status `human_needed`); UAT closed
2026-09-10 (`06-UAT.md`, 2/2) — the fresh-run item was settled by executing all six notebooks
from cleared state via `nbclient`, and the P3 cold-start sign-off was recorded by the operator
as **accept-with-caveats** (the regimes are *crisis* regimes, not *allocation* regimes — which
is the finding that motivated Phase 7).

**Why this precedes migration**: the migration's per-step validation gate is "run the
notebook and verify." The platform currently has **zero** notebooks — all 12 in
`notebooks/` drive the legacy quarterly pipeline. Without this phase there is nothing to
validate a migrated step against, and five phases of verified work remain un-inspectable.

### Phase 7: Regime Representation

**Goal**: Decide what the regime labeler should see, prove that decision walk-forward, and
add a **second, independent labeler on relative/leadership features** so the platform can
inform allocation during ordinary markets — not only crisis avoidance.
**Depends on**: Phase 6
**Requirements**: REG-01, INV-01
**Full scope**: `.planning/PROPOSAL-phase-regime-representation.md`

**Why this phase exists**: audit items **A13** (HIGHEST) and **A15** were assigned to no
phase. Phase 6 was scoped only to *display* A13 and did so — the reference and filtered
labelings disagree on **389/470 = 82.8%** of compared months with no diagonal structure —
but nothing was scheduled to resolve it. Separately, the Phase 6 P3 sign-off accepted the
regimes with the caveat that they are *crisis* regimes, not *allocation* regimes. Both trace
to one cause: what the labeler is allowed to see.

**A13's mechanism** (diagnosed 2026-09-10): `backtest/driver.py::_window_active_features`
admits a feature once it has `feature_min_history` (120) months in-window. With staggered
start dates that fully explains the 7 changes — `curve_10y2y` 1976-06→1986-06, `gold`/`oil`
1985-02→1995-02, `fred_vix` 1990-01→2000-01. Meanwhile
`evaluation/report.py::_reference_label_columns` freezes a set at the first decision date.
§5.4 compares two estimators fit on different feature spaces. A policy disagreement, not a defect.

**Wave 1 — resolve A13/A15. This is a gate; wave 2 does not start unless it passes.**
**Wave 2 — the leadership classifier**, fit unsupervised on relative/invariant features.

**Success Criteria** (what must be TRUE):

  1. Driver and report label under **one documented feature policy**; a test fails if they
     diverge. The choice and its rejected alternatives are recorded as an ADR.

  2. The §5.4 sojourn/lag ratio is **interpretable** — computed between two labelings fit on
     the same feature space, shown with its resolved-transition count. The A13 caveat is
     removed only because the cause is fixed, never because the wording was softened.

  3. Post-fix labeling disagreement is measured and reported against the 82.8% pre-fix
     baseline. No target is set — a number, not a goal, to avoid fitting to it.

  4. Classifier #1's ablation delta is re-measured on **both** axes (`wealth_delta` and
     `dd_delta`, currently +0.379267 / −0.014364), each within its `06-VALIDATION.md` band.

  5. Classifier #2 exists, is fit **unsupervised** on a feature set disjoint from classifier
     #1's 13 (a test asserts disjointness), with occupancy summing to 1.0 and no state below
     the §4.4 5% floor left unmarked.

  6. Statistical dependence between the two labelings is measured and reported. High
     dependence is a **failure to add an axis** and is recorded as such.

  7. Joint (#1 × #2) allocation lift is measured walk-forward against #1 alone, every
     configuration logged to the trial registry, deflated-Sharpe applied for the full count.

  8. `platform/` still imports nothing from the legacy library — the import-guard test is
     extended to the new modules.

     > ⚠ **CORRECTION 2026-09-15 — the "Verified 2026-09-10" claim that stood here was FALSE.**
     > It rested on `MIGRATION-PLAN.md`'s exit check, which ended in `| grep -v platform`. Every
     > match line begins with a path containing `platform`, so that filter discarded **every**
     > violation: the check returned nothing whether or not the code was decoupled, and could
     > never fail. An AST scan finds **31 real legacy import sites** in `platform/` — bare
     > `trading_crab_lib` ×16, `.checkpoints` ×7, `.ingestion.*` ×7, `.email` ×1. All predate
     > Phase 7; **wave 1 added none**. Vendoring them is `MIGRATION-PLAN.md` P0 / Phase 8
     > criterion 1.
     >
     > **The "port, don't import" instruction for wave 2 still stands** — but because the
     > coupling must not be *widened*, not because `platform/` is already clean. Now guarded by
     > `tests/unit/test_platform_legacy_import_ratchet.py`, a ratchet that may only decrease.

**Plans**: 12 plans, 8 executed — 4 in phase-wave 1 (criteria 1-4 + ADR-0001, **complete**,
UAT-signed accept-with-caveats 2026-09-15) and 8 in phase-wave 2 (criteria 5-7 + INV-01 +
ADR-0002, planned 2026-09-15 in the second planning pass D-09 called for; 07-05..07-08
executed, 07-09..07-12 outstanding)

> ⚠ **Two senses of "wave" collide in this phase — read carefully.** The phase gates on
> **phase-wave 1 → phase-wave 2** (resolve A13/A15, then the leadership classifier). All four
> plans below are **entirely inside phase-wave 1**. The `exec-wave N` labels are GSD *execution*
> ordering within this pass, not the phase's gate. No plan below touches classifier #2.

Plans (all phase-wave 1):

- [x] 07-01-PLAN.md — Tracer: freeze the L1 feature policy to one computed-once column list shared by driver and reference, with the criterion-1 equivalence test *(exec-wave 1)*
- [x] 07-02-PLAN.md — Recompute `monthly_features` from cached `monthly_raw` (D-02-A, ten-column frozen set) and re-pin the A13 golden constant exactly *(exec-wave 2)*
- [x] 07-03-PLAN.md — Re-measure criteria 2/3/4 on real data against bands that name the values they reject, plus the D-03 logged rejection trial *(exec-wave 3)*
- [x] 07-04-PLAN.md — D-05 three-state pre/post table, D-08 A13 caveat resolution, and the policy ADR *(exec-wave 4)*

Plans (all phase-wave 2 — the leadership classifier; criteria 5, 6, 7 + INV-01):

- [x] 07-05-PLAN.md — Tracer: `canonicalize_states` gains an explicit `sort_column` that raises instead of silently falling back to centroid column 0; `platform/features/relative.py` ports the leadership features at monthly cadence; M2SL/TOTALSL ingestion config *(exec-wave 1)*
- [x] 07-06-PLAN.md — `total_trial_count()` reads the provenance header (38 prior + post-header rows) and `evaluation/deflated_sharpe.py` implements Bailey–López de Prado, with the estimator choice fixed in writing first *(exec-wave 1)*
- [x] 07-07-PLAN.md — INV-01 screening: named candidates, dimensional reduction as a discovery tool only, era-stability assessed on expanding windows, every candidate registry-logged, survivors named *(exec-wave 2)*
- [x] 07-08-PLAN.md — Pin classifier #2's features/K/λ/sort column and criterion 7's measurement routing at a blocking decision, write ADR-0002 (Proposed) with the trial ceiling **before** running, then fit classifier #2 with the criterion-5 disjointness and occupancy tests *(exec-wave 3)*
- [ ] 07-09-PLAN.md — Criterion 6: ARI + NMI + Cramér's V + crosstab with no threshold (D-15), tests that fail on a perfect statistic as well as a wrong one, and a human judgement on whether an axis was added *(exec-wave 4)*
- [ ] 07-10-PLAN.md — `blend_regime_tilts` (new code — `tilt.py` takes one probability input today), and the blocking confirmation of the four now-load-bearing `[ASSUMED]` bands *before* any lift number exists *(exec-wave 4)*
- [ ] 07-11-PLAN.md — Criterion 7: one harness produces both the joint leg and the #1-alone baseline over an identical step sequence; lift reported with its window inline; deflated Sharpe applied for the full live-read count *(exec-wave 5)*
- [ ] 07-12-PLAN.md — ADR-0002 → Accepted with the measured results including the unfavourable ones; all eight probe edges resolved; REG-01 and INV-01 claimed; ratchet re-measured *(exec-wave 6)*

**Explicit non-goals**: no fitting to forward returns; no raising K on classifier #1; no
2021+ holdout use for any selection decision; no migration work.

**Requirement coverage**: phase-wave 1 (plans 07-01…07-04) delivered REG-01 **in part** — the
driver/reference feature policy, §5.4 interpretability, and the ablation re-measurement
clauses — and deferred INV-01 in full by `07-CONTEXT.md` D-09, recorded as a decision in
ADR-0001. Phase-wave 2 (plans 07-05…07-12) **claims both in full**: REG-01's second-classifier,
disjointness, orthogonality and joint-lift clauses, and **INV-01 entire** — candidates
constructed and screened with dimensional reduction as a discovery tool, loading stability
tested across eras, every candidate registry-logged and walk-forward assessed, survivors
admitted as named features and never anonymous principal components (design decision R4).
Recorded in ADR-0002.

**Absorbs INV-01** (formerly Phase 8): named invariant candidates (M2/GDP, market-cap/GDP,
credit/GDP) are wave 2's feature-discovery work, and INV-01's constraint that survivors be
admitted as **named** features — never anonymous principal components, preserving design
decision R4 — is exactly the interpretability requirement a leadership classifier needs.

### Phase 8: Migration to Public Repo

**Goal**: The validated platform lives in `strycker/trading-crab`, the public/PyPI
two-package repo, ready for continued development outside the heavy-dev workbench.
**Depends on**: Phase 7
**Requirements**: MIG-01
**Success Criteria** (what must be TRUE):

  1. `platform/` imports nothing from the legacy library — the four coupling seams
     (CheckpointManager, multpl/macrotrends scraper helpers, email helpers) are vendored
     and an import-guard test enforces it. **Baseline measured 2026-09-15: 31 sites remain**
     (the prior grep-based check was unfalsifiable — see the Phase 7 criterion-8 correction).
     The enforcing test now exists as `tests/unit/test_platform_legacy_import_ratchet.py`;
     this phase's exit is that test passing with `MAX_LEGACY_IMPORT_SITES = 0`.

  2. The two-package layout (`trading-crab` + `trading-crab-lib`) exists in
     `strycker/trading-crab` with the L0–L4 modules migrated.

  3. The test suite passes green in the new repo's CI, not just locally (≥378 platform
     tests).

  4. A real 1972–2020 walk-forward in the new repo reproduces the reference numbers in
     `MIGRATION-PLAN.md` §6.

  5. README and docs in the new repo describe the regime-conditional platform, not the
     legacy quarterly pipeline.
**Plans**: TBD
**Detailed step plan**: `MIGRATION-PLAN.md` (P0–P6)

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8
(Phase 7 gates internally: wave 2 does not start unless wave 1 passes.)

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Monthly Data Layer & Long Histories | 7/7 | Complete   | 2026-07-15 |
| 2. Honesty Infrastructure | 5/5 | Complete   | 2026-07-22 |
| 3. Regime Labeling & Prediction | 4/4 | Complete   | 2026-07-22 |
| 4. Asset Prediction & Allocation | 5/5 | Complete   | 2026-07-23 |
| 5. Honest Backtest & Evaluation | 7/7 | Complete (closed 2026-08-04) | 2026-07-27 |
| 6. Platform Notebook Suite | 7/7 | Executed + verified (5/5 criteria; human_needed) | 2026-09-10 |
| 7. Regime Representation | 8/12 | In progress — wave 1 closed (verified+validated+UAT); 07-05..07-08 executed, ADR-0002 at Proposed; 07-09..07-12 outstanding | wave 1: 2026-09-15; 07-08: 2026-09-17 |
| 8. Migration to Public Repo | 0/TBD | Not started | - |
