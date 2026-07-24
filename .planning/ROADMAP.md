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
- [ ] **Phase 5: Honest Backtest & Evaluation** - Full 1972–2020 walk-forward backtest vs. baseline gauntlet with first-class honesty metrics
- [ ] **Phase 6: Migration to Public Repo** - Validated skeleton migrated to `strycker/trading-crab`, tests green, docs updated

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
**Plans**: 3/7 plans executed

**Wave 1**

- [x] 05-01-PLAN.md — Foundation: backtest: config section + backtest/evaluation package scaffolds + transaction-cost/turnover identity (EVAL-01, D-03)

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 05-02-PLAN.md — Walk-forward backtest driver: refit L1→L4 on data ≤ t, holdout-bounded, one registry trial (EVAL-01)
- [x] 05-03-PLAN.md — Strategy KPIs + sojourn/detection-lag headline orchestration (EVAL-01, EVAL-03)
- [ ] 05-04-PLAN.md — Model-metrics artifacts: multiclass Brier, calibration bins, confusion tables (EVAL-04)

**Wave 3** *(blocked on Wave 2 completion)*

- [ ] 05-05-PLAN.md — Baseline gauntlet: SPY, 60/40, Faber 10-month SMA + no-regime ablation (EVAL-02)

**Wave 4** *(blocked on Wave 3 completion)*

- [ ] 05-06-PLAN.md — Backtest report assembly (headline-first) + synthetic end-to-end integration (EVAL-01, EVAL-02, EVAL-03, EVAL-04)

**Wave 5** *(blocked on Wave 4 completion)*

- [ ] 05-07-PLAN.md — Real 1972–2020 run against live checkpoints — blocking human-verify (design §14 Phase 1 exit) (EVAL-01..04)

### Phase 6: Migration to Public Repo

**Goal**: The validated skeleton lives in `strycker/trading-crab`, the public/PyPI
two-package repo, ready for continued development outside the heavy-dev workbench.
**Depends on**: Phase 5
**Requirements**: MIG-01
**Success Criteria** (what must be TRUE):

  1. The two-package layout (`trading-crab` + `trading-crab-lib`) exists in
     `strycker/trading-crab` with the new L0–L4 modules migrated.

  2. The test suite passes green in the new repo's CI, not just locally.
  3. README and docs in the new repo describe the regime-conditional platform, not just
     the legacy quarterly pipeline.
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5 → 6

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Monthly Data Layer & Long Histories | 7/7 | Complete   | 2026-07-15 |
| 2. Honesty Infrastructure | 5/5 | Complete   | 2026-07-22 |
| 3. Regime Labeling & Prediction | 4/4 | Complete   | 2026-07-22 |
| 4. Asset Prediction & Allocation | 5/5 | Complete   | 2026-07-23 |
| 5. Honest Backtest & Evaluation | 3/7 | In Progress|  |
| 6. Migration to Public Repo | 0/TBD | Not started | - |
