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
- [x] **Phase 7: Regime Representation** - Resolve A13/A15 (one feature policy for driver and report), then add an independent leadership-axis classifier on relative/invariant features (absorbs INV-01). **Closed 2026-09-21: INV-01 delivered in full; REG-01 delivered PARTIALLY — criterion 6's dependence verdict is UNRESOLVED and no independent second axis is established**
- [x] **Phase 8: Regime Persistence & Stability** - The nowcaster carries state memory, hysteresis gates allocation, and §4.4 criterion 3 is actually run (completed 2026-09-30)
- [x] **Phase 08.1: Point-in-Time Data Audit** *(INSERTED 2026-09-29)* - Every feature lagged to its publication date; re-run decision-bearing evaluations under a declared budget (completed 2026-09-30)
- [x] **Phase 08.2: Lean MVP — Simplify, Modularize, Notebook-Gate** *(INSERTED 2026-09-29)* - A usable weekly product first; module map M0–M7; notebook gate per module; lean planning (completed 2026-10-02)
- [x] **Phase 08.3: Regime Rebuild I — month-end returns (E-07) and re-measure** *(INSERTED 2026-10-02)* - Scoreboard re-measured on month-end returns a trader could earn; decides whether MVP-1's core mix stands (completed 2026-10-05)
- [x] **Phase 08.4: Cold-Start Tooling — Archive, Reset, Restore & Fail-Loud Build** *(INSERTED 2026-10-02; re-scoped 2026-10-05)* - Archive/reset/restore/promote state at will, tagged model archives under git; the build fails loud and runs from empty (completed 2026-10-09)
- [ ] **Phase 08.5: Clean-Slate Reset & Learnings Before Migration** *(INSERTED 2026-10-05)* - Use the 8.4 tooling for the production reset (everything generated out of git, registry true-zero epoch), MVP-1 from a fresh clone, learnings and the Phase 9.x rebuild order
- [ ] **Phase 9: Migration to Public Repo** - Platform decoupled and migrated to `strycker/trading-crab`, tests green in CI, docs updated

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

     > ✅ **MET 2026-09-21.** Evidence: `platform_design/adr/0002-l1-second-classifier.md`
     > § RE-PIN 2026-09-18 and § ACCEPTANCE 2026-09-21. Classifier #2 (K = 5, λ = 16.0)
     > occupancy **16.6667 / 22.7011 / 22.4138 / 23.8506 / 14.3678 %** over **696 months,
     > 1963-01-31 → 2020-12-31**, **sum error exactly 0.0**, every state inside §4.4
     > criterion 1's band. Disjointness from classifier #1's lean 13 asserted on the
     > **resolved** frozen eight by
     > `tests/unit/test_platform_features_relative.py::test_resolved_frozen_list_is_disjoint_from_the_lean_set`.
     >
     > ⚠ **The "§4.4 5% floor" in this criterion's wording does not exist and never did.**
     > §4.4 criterion 1 reads "every state ≥ ~8% and ≤ ~35% of months" — a **two-sided** band.
     > The 5% figure was a project-wide misquote corrected 2026-09-17
     > (ADR-0001 § AMENDMENT 2026-09-17); the implementation had no cap check at all, so the
     > occupancy test **could confirm and never fail** — the same evidence shape as criterion
     > 8's `grep -v platform` below. The criterion text is left as written for the record;
     > **it was scored against the real two-sided band, which is stricter in both directions.**
     > Design §4.4 criterion 1 was additionally **AMENDED 2026-09-18** to carry a *recurrence
     > exemption* (at most one state below the ~8% floor, if it recurs in ≥3 named, temporally
     > separated episodes with a coherent profile and low-n discipline, and the ~35% cap is not
     > relaxed). **Classifier #2 does not invoke it.** Classifier #1 does — its crisis state at
     > **5.7554%** over 40 months across nine named episodes, 1970 through 2020 (ADR-0001
     > § RE-PIN 2026-09-18).

  6. Statistical dependence between the two labelings is measured and reported. High
     dependence is a **failure to add an axis** and is recorded as such.

     > ⚠ **MEASURED AND REPORTED; the verdict is UNRESOLVED — the criterion is NOT satisfied.**
     > Evidence: `.planning/phases/07-regime-representation/07-DEPENDENCE.md` § RESULT
     > 2026-09-18. Re-measured after **both** classifiers were re-pinned: ARI **0.354841**,
     > NMI **0.464088**, Cramér's V **0.589748**, `n_compared` = **695 months, 1963-02-28 →
     > 2020-12-31**, K₁ = 6 vs K₂ = 5. Read against a block-permutation control (2000 resamples,
     > seed 20260918), the pre-registered rule committed at `298b1bc` **before the control code
     > existed** returns **INCONCLUSIVE**: observed NMI 0.464088 sits at the **96.60th
     > percentile**, above p95 **0.449202** but not above p99 **0.501195**.
     >
     > **This measurement cannot decide whether classifier #2 added an axis, and nothing else
     > in this phase decides it.** It is neither the "high dependence" failure this criterion
     > anticipated nor a demonstration of an added axis. **No tie-break was run and none may
     > be** — the pre-registration forbids a fourth statistic, a different null and a re-run at
     > another seed. **Criterion 7 is not a tie-break on this.** REG-01 is therefore claimed
     > **PARTIALLY**, with this as the named open item.

  7. Joint (#1 × #2) allocation lift is measured walk-forward against #1 alone, every
     configuration logged to the trial registry, deflated-Sharpe applied for the full count.

     > **RESULT: FAILED — A11 / ADR-0003 quality tier (2026-09-21): neither leg's deflated Sharpe
     > clears the hurdle.** Recorded here per Glenn's ruling 2026-09-29 ("mark both"; 08-A11.md §3.4).
     > Re-measured by 08-10 under the 5pp band: −0.125307 / +0.026164. Both legs are exposed to the
     > CR-03 publication-lag look-ahead; re-run owned by Phase 08.1.
     >
     > ✅ **MET 2026-09-21 as a measurement. The wealth sign is negative and is stated as
     > measured.** Evidence: `.planning/phases/07-regime-representation/07-JOINT-LIFT.md`.
     > `wealth_delta` = **−0.123438 nats** and `dd_delta` = **+0.024084**, both over **588
     > steps, 1972-01-31 → 2020-12-31**, 0 degraded steps on both legs, routing
     > `L1_ONLY_LAST_FILTERED_STATE`. In the units the numbers are actually in: the joint leg
     > ended at ≈ **0.8839×** the #1-alone leg's terminal wealth — an **11.61% shortfall** —
     > while making the worst drawdown **2.41 percentage points shallower** (−25.74% vs
     > −28.14%) and 7 months shorter. **The drawdown improvement does not offset the wealth
     > loss and is not recorded as doing so**; it is further qualified by ADR-0002's named
     > limitation, since this labeling identifies crises *ex post*.
     >
     > Both configurations logged: tags `07-11-c1-alone-L1only` and `07-11-joint-c1xc2-L1only`;
     > `total_trial_count()` 40 → 42, ceiling 44. Deflated Sharpe applied at the full live-read
     > count of **42** for both legs (2.28151 × 10⁻¹² and 1.46904 × 10⁻¹¹): **neither leg
     > clears the multiple-testing hurdle.** D-06 makes measurement the gate, not the sign —
     > which is why this criterion is met and why the number is quoted rather than framed.

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

**Ratchet re-measured at phase close (2026-09-21, plan 07-12), recorded here so criterion 8's
correction block above stays byte-identical:** the AST scan counts **31** legacy import sites —
`trading_crab_lib` ×16, `.checkpoints` ×7, `.ingestion.http` ×3, `.ingestion.browser` ×2,
`.ingestion` ×1, `.ingestion.assets` ×1, `.email` ×1. **Unchanged from the 2026-09-15
measurement; wave 2 added none and removed none.** `MAX_LEGACY_IMPORT_SITES` is therefore left
at **31** and was not edited — the constant may only decrease, and there was no decrease to
record.

**Plans**: 12 plans, **12 executed** — 4 in phase-wave 1 (criteria 1-4 + ADR-0001, **complete**,
UAT-signed accept-with-caveats 2026-09-15) and 8 in phase-wave 2 (criteria 5-7 + INV-01 +
ADR-0002, planned 2026-09-15 in the second planning pass D-09 called for; 07-05..07-08
executed, 07-09..07-11 executed 2026-09-18..2026-09-21, 07-12 executed 2026-09-21)

> ⚠ **Two senses of "wave" collide in this phase — read carefully.** The phase gates on
> **phase-wave 1 → phase-wave 2** (resolve A13/A15, then the leadership classifier). All four
> plans below are **entirely inside phase-wave 1**. The `exec-wave N` labels are GSD *execution*
> ordering within this pass, not the phase's gate. No plan below touches classifier #2.

(Note 2026-09-29: this warning refers to the wave-1 plans 07-01..07-04 only; wave-2 plans 07-05..07-12 follow.)

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
- [x] 07-09-PLAN.md — Criterion 6: ARI + NMI + Cramér's V + crosstab with no threshold (D-15), tests that fail on a perfect statistic as well as a wrong one, and a human judgement on whether an axis was added *(exec-wave 4)*
- [x] 07-10-PLAN.md — `blend_regime_tilts` (new code — `tilt.py` takes one probability input today), and the blocking confirmation of the four now-load-bearing `[ASSUMED]` bands *before* any lift number exists *(exec-wave 4)*
- [x] 07-11-PLAN.md — Criterion 7: one harness produces both the joint leg and the #1-alone baseline over an identical step sequence; lift reported with its window inline; deflated Sharpe applied for the full live-read count *(exec-wave 5)*
- [x] 07-12-PLAN.md — ADR-0002 → Accepted with the measured results including the unfavourable ones; all eight probe edges resolved; REG-01 claimed **partially** (criterion 6 unresolved) and INV-01 in full; ratchet re-measured at 31, unchanged *(exec-wave 6)*

**Explicit non-goals**: no fitting to forward returns; no raising K on classifier #1; no
2021+ holdout use for any selection decision; no migration work.

**Requirement coverage**: phase-wave 1 (plans 07-01…07-04) delivered REG-01 **in part** — the
driver/reference feature policy, §5.4 interpretability, and the ablation re-measurement
clauses — and deferred INV-01 in full by `07-CONTEXT.md` D-09, recorded as a decision in
ADR-0001. Phase-wave 2 (plans 07-05…07-12) claims **INV-01 in full** and **REG-01 only in
part**:

- **INV-01 — IN FULL.** Candidates constructed and screened with dimensional reduction as a
  discovery tool (`pca.transform()` never called), loading stability tested across 10 eras
  ending 1972-01-31 → 2017-01-31 at tolerance `LOADING_STABILITY_TOLERANCE = 0.15`, every
  candidate registry-logged (2 rows, `total_trial_count()` 38 → 40) and walk-forward assessed,
  survivors `m2_gdp` and `credit_gdp` admitted as **named** features and never anonymous
  principal components (design decision R4).

- **REG-01 — PARTIALLY.** The second-classifier, disjointness and joint-lift clauses are
  satisfied. **The orthogonality clause was measured but its verdict is UNRESOLVED** — the
  pre-registered control returned INCONCLUSIVE (criterion 6 above), and no tie-break is
  permitted. **No independent second axis is established by this phase**, and this closure
  does not claim one. Closing REG-01 needs more data or a different instrument, in a phase
  that pre-registers its rule the way this one did.

Both recorded in ADR-0002 § Requirement coverage.

**Open items this phase carries forward rather than closes** (full list in ADR-0002 § Deferrals
and open items at acceptance): criterion 6 unresolved with no tie-break permitted; §4.4
criterion 3's Hungarian subsample-stability test **never run for either classifier**; ADR-0001
condition (iv)'s **covariance clause unimplemented** (no per-regime covariance exists at L4-01 —
it falls to L3, design §6.2), with `vol_targeted_tilt` and `driver.py:497` still
(iv)-non-compliant for consumers other than the joint harness; classifier #1's **filtered**
labeling changing state in **246 of 588 decision months (41.84%)** against a 3.60% full-sample
rate, governed by no band and feeding the tilt directly; classifier #2's §5.4 ratio of **1.074**
(median sojourn 29.0 months against a 27.0-month median detection lag, only 5 of 12 transitions
resolved); the crisis state's **3.0-month** median sojourn sitting exactly on criterion 2's
boundary against a 1–3 month detection lag, so criterion 7's `dd_delta` is **not** evidence
crises are nowcastable in time to act; `DEGENERATE_SHARPE_VARIANCE = 1.0` still a declared
assumption governing every DSR until 20 independent Sharpe-bearing trials exist; the L2
CV-robustness question routed around rather than resolved; and **audit item A11 open by
deliberate choice**.

**Absorbs INV-01** (formerly Phase 8): named invariant candidates (M2/GDP, market-cap/GDP,
credit/GDP) are wave 2's feature-discovery work, and INV-01's constraint that survivors be
admitted as **named** features — never anonymous principal components, preserving design
decision R4 — is exactly the interpretability requirement a leadership classifier needs.

### Phase 8: Regime Persistence & Stability

**Goal**: The real-time (filtered) regime labeling stops flickering, allocation responds through
the anti-flicker machinery the design already specifies, and the one §4.4 acceptance criterion
that has never been run is run.

**Depends on**: Phase 7
**Blocks**: Phase 9 (Migration) — per the Phase 7 wave-2 UAT ruling, 2026-09-21
**Requirements**: PER-01, PER-02, PER-03, PER-04, PER-05, PER-06, PER-07, PER-08, PER-09, PER-10
— assigned at planning 2026-09-21, one per success criterion 0–9 in order.

**Why this phase exists.** Phase 7's UAT measured classifier #1's *filtered* labeling changing
state in **246 of 587 step-pairs (41.91%)** against a full-sample rate of 3.74%. Median filtered
run length is **1.0 month**; 136 of 247 runs are a single month.

> ### CORRECTION 2026-09-21 — the first scoping of this phase had the causal model wrong
>
> These criteria originally assumed the 41.91% churn was L2 nowcaster flicker, fixable by §5.1's
> recursive prior-state feature. **Research and independent verification show it is not.**
>
> - `joint_driver.py:502` sets `state_1 = states_1.iloc[-1]` — the **jump model's** (L1) label.
> - `joint_driver.py:431` — under the decision-bearing `ROUTING_L1_ONLY`,
>   `probs_1 = _last_state_one_hot(states_1)`; **`_refit_l2` is never called.**
> - Classifier #2 runs the **same code path** and churns **4.09%** (24/587).
>
> So a §5.1 fix, which changes L2, **cannot move that number**. The original criterion 2 would
> have reported 246 → 246 bit-for-bit whether the fix worked perfectly or not at all — an
> unfalsifiable criterion, inside a phase written to catch unfalsifiable criteria. Sixth recorded
> instance of this project's signature defect; authored by Claude, caught by its own research.
>
> **There are two distinct problems and they are now separated:**
>
> **A — L1 terminal-month churn (41.91%).** The DP's terminal month is the only month with no
> right neighbour, so deviating there costs **λ** where an interior deviation costs **2λ** — and
> the filtered labeling reads exactly that month, every step. Classifier #1 has **λ/d = 1.0**;
> #2 has **2.0** and churns 10× less. This is a consequence of the λ re-pin of 2026-09-18
> (coefficient 4 → 1).
>
> **B — L2 flat posteriors.** Under `l2` routing `active_regime` changes **462/587 (78.7%)** and
> is all-cash in **387/588** months, implying **≥309/488** non-degraded steps had max calibrated
> probability below the 0.70 act threshold. 0.70 is **4.2× uniform at K=6**. §5.1 addresses this.
>
> **CORRECTED 2026-09-24 (found in 08-09, verified by the orchestrator).** "462/587 (78.7%)" counted
> every consecutive all-cash pair as a change, because `NaN != NaN` evaluates True — **340** of the
> 462 are None→None. Treating None as a state, `active_regime` changes **122/587 (20.8%)**. The
> **387/588** all-cash count stands. The orchestrator relayed 462 during Phase 8 scoping using the
> same comparison; it framed Track B's motivation but bore on no ruling.
>
> **REFUTED 2026-09-23 by criterion 3's own pre-registered rule (08-02).** Churn is flat in k:
> classifier #1 changes 246 / 242 / 245 / 248 / 247 / 249 of 587 pairs for k = 1…6, so the
> terminal-month edge artefact described above is not the cause. The 41.91% is the labeler's own
> path. Not ruled out: edge effects longer than 6 months; λ/d is not isolated from K, features
> and d. See `08-TRACK-A.md`.

**Depends on**: Phase 7
**Blocks**: Phase 9 (Migration) — per the Phase 7 wave-2 UAT ruling, 2026-09-21

**Success Criteria** (what must be TRUE):

  0. **PREREQUISITE — the per-step probability matrix is persisted.** `driver.py:530-532`
     accumulates it and throws it away; only the equity curve is written. Nothing in criteria 1–3
     is measurable without it. Small and boring; it gates everything else.

  1. **A prior-state belief propagates into the nowcaster's output, and it cannot leak.**
     **REWORDED 2026-09-21** from "the feature set includes the prior predicted distribution".
     An explicit Bayes filter — `π_t ∝ [Σ π_{t−1} A] · L_t`, with `A` the existing empirical
     transition matrix — satisfies §5.1's intent ("a discriminative replacement for the HMM
     filter", §5.1's own first sentence) with **zero train/serve skew, zero leakage surface and
     zero free parameters**, because it has no training-time analogue at all. It does not put the
     prior state in the feature set, and the original wording required that; the wording is
     changed deliberately, not reinterpreted.
     The L1 labeler is **non-causal within its window by design** (`jump_model.py:15-18`), so a
     *trained* prior-state column would carry post-t information that no holdout, purge or embargo
     can detect. Eliminating that channel beats guarding it.

  2. **Both churn series are reported, and neither can masquerade as the other.**
     - **A's metric:** `state_1` changes / 587. Currently **246 (41.91%)**. A §5.1-style change
       must NOT move it; if it appears to, something is wired wrong.

     - **B's metric:** `argmax(regime_probs)` churn — **never measured anywhere**, and the object
       criterion 1 actually changes. Requires criterion 0.
     No target is pre-declared for either. Both are reported with their window.

  3. **Track A is diagnosed before it is changed.** The zero-trial diagnostic: record the step-*t*
     fit's label for months *t−1 … t−6* and churn each series across steps. Falling churn in *k*
     confirms the terminal-month edge artefact; flat churn refutes it and λ/d is the whole story.
     **A λ sweep is NOT authorized** — it costs one registry trial per value against 42/44 used,
     and needs its own ruling.

  4. **§5.3's hysteresis gates allocation, closing audit item A7 — and is evaluated only after B.**
     Verified 2026-09-21: `active_regime` is elementwise equal to `state_1` in all 588 months, so
     under the decision-bearing routing the hysteresis is a **provable identity** on a one-hot
     input. It cannot be evaluated until the probability vector stops being degenerate, which is
     why §5.1 comes first. Mechanism (hard gate / bounded turnover / magnitude-scaling) and the
     0.70/0.40 pair are **decision checkpoints for Glenn**, not planner choices.

  5. **§4.4 criterion 3 is RUN for both classifiers**, with **four** subsample schemes: drop first
     decade, drop last decade, circular block bootstrap, **and leave-one-episode-out**. The fourth
     is added because the three named schemes are poorly aimed: classifier #1's **state 2** is
     **one contiguous episode, 1996-07 → 2002-05** — exactly the "20% state appearing once as a
     contiguous block" the §4.4 amendment describes — and **neither decade-drop touches it**.
     Leave-one-episode-out is the design's own "drop 2008-09" generalised; for a one-episode state
     it is degenerate, **and that degeneracy is the answer**, with no threshold invented.
     Use **centroid distance in de-standardized units** (empirical multivariate Wasserstein is
     unusable at n=40: sampling bias 2.88 against a true signal of 0.949) with
     `scipy.optimize.linear_sum_assignment`. Report a **within-state split-half null at the same
     n** as the yardstick — the null is **0.706**, not 0, at n=40. Every row carries subsample
     occupancy, because `_recompute_centroids` freezes zero-occupancy states at their previous
     centroid and would otherwise score an evaporated state as *stable*.

  6. **Criterion 7 is re-measured** under whatever changed, both legs, one harness, window inline.

  7. **A11 is revisited and answered.** Reopened by Glenn 2026-09-21 after being deliberately left
     open on 2026-09-18. Written as the reversal of a prior decision that it is.

  8. **Validation gap G6 is pinned** — a test asserting non-joint consumers of `vol_targeted_tilt`
     (notably `driver.py:497`) receive the **unpooled** estimate. Pin the known **non**-compliance;
     writing it as a compliance assertion produces a test that can only pass.

  9. **Recorded counts match reality.** `CLAUDE.md` ×2 and `README.md`'s badge say **1705**; the
     suite is at **2018**. Also fix F-4: the churn rate's denominator is 587 pairs, not 588 months
     (`246/587 = 41.91%`, recorded as 41.84%) — the fix breaks an existing pin, so do both in one
     commit.

**Plans**: 19 plans (10 across 5 execution waves, planned 2026-09-21, plus 9 gap-closure plans 08-11..08-19; the count was 10 until 2026-09-29). Two blocking
`checkpoint:decision` gates — A11 in wave 1 (deliberately **ahead** of every number, so a gate is
never chosen after seeing the value it judges) and §5.3's mechanism plus the 0.70/0.40 pair in
wave 4.

> ⚠ **The two tracks never share a plan, and that is the phase's core structural decision.**
> Track A is the L1 terminal-month churn (`state_1`, **246/587 = 41.91%**) and Track B is the L2
> filtered posterior (`argmax(regime_probs)`, never measured before this phase). A §5.1-style
> change **cannot** move Track A. Criterion 0 — persisting the per-step probability matrix — gates
> criteria 1, 2 and 4 and is therefore plan 08-01, the phase tracer.

**Wave 1** *(no dependencies; 08-05 blocks on a human decision)*

- [x] 08-01-PLAN.md — **Tracer.** Persist the per-step probability matrix end to end and split criterion 2's churn into its two named series, with the l1only identity pinned and F-4's denominator fixed alongside its pin *(PER-01, PER-03, PER-10)*
- [x] 08-02-PLAN.md — Track A's zero-trial terminal-month diagnostic: churn vs `iloc[-k]` for k = 1…6, both classifiers, anchored elementwise to the tracked curve at k = 1 *(PER-04)*
- [x] 08-03-PLAN.md — §4.4 criterion 3 machinery: Hungarian matching on de-standardized centroids, the within-state split-half null, the evaporation flag, and the four subsample schemes *(PER-06)*
- [x] 08-04-PLAN.md — G6 pinned as the known **non**-compliance across all three `vol_targeted_tilt` consumers, with the both-halves rule *(PER-09)*
- [x] 08-05-PLAN.md — **A11 answered, written as the reversal it is** — decided in wave 1 so the ruling precedes every number it could judge *(PER-08)*

**Wave 2** *(08-06 blocked on 08-01; 08-07 blocked on 08-03)*

- [x] 08-06-PLAN.md — The explicit Bayes filter (no training column at all) plus the **signed** detection offset, and the leakage guard whose substituted arm proves it discriminates *(PER-02)*
- [x] 08-07-PLAN.md — Run §4.4 criterion 3 for both classifiers under all four schemes; write the record with the pre-registered prediction quoted from its commit *(PER-06)*

**Wave 3** *(blocked on 08-06)*

- [x] 08-08-PLAN.md — Wire the filter into both drivers and the weekly report; report **three** churn numbers (A, B0 the control, B1 the new one) with the l1only curve pinned bit-for-bit *(PER-02, PER-03)*

**Wave 4** *(blocked on 08-08)*

- [x] 08-09-PLAN.md — §5.3's hysteresis gates allocation and A7 closes, at two blocking decisions whose registry cost is priced before the choice *(PER-05)*

**Wave 5** *(blocked on 08-02, 08-05, 08-07, 08-09)*

- [x] 08-10-PLAN.md — Criterion 7 re-measured (both legs, one harness, window inline), recorded counts corrected against a live collection, and the phase's closing measurement record *(PER-07, PER-10)*

**Gap closure — UAT 2026-09-28 (G-08-1, G-08-2)** *(4 plans, 3 waves; 08-12 blocks on two decisions for Glenn, run in parallel with 08-11)*

- [x] 08-11-PLAN.md — **G-08-1.** The criterion-7 re-derivation compares floats portably (rel 1e-9, abs 0), proven to reject the band-off record, a 1-ppm error in every field and a zeroed DSR; the same defect class is swept suite-wide and listed *(PER-07, PER-10)* — wave 1
- [x] 08-12-PLAN.md — **G-08-2 decisions.** Measured facts on the serving recipe, then Glenn rules on the ragged edge (the 2026-08-31 row lacks 3 model columns) and on the input-independent served posterior; recorded in 08-SERVING.md *(PER-02, PER-05)* — wave 1, checkpoint
- [x] 08-13-PLAN.md — **G-08-2 build.** `python -m trading_crab_lib.platform.report.serving` builds all three serving artifacts weekly reads (the UAT found one; there are three) through the backtest's own `fit_l2_nowcaster`, under `NO_REGISTRY`; weekly scores the model's own columns; the end-to-end test is proven able to fail *(PER-02, PER-05, PER-10)* — wave 2
- [x] 08-14-PLAN.md — **G-08-2 serve.** Glenn's two 08-12 rulings implemented, then the supported sequence run on the real tracked data in scratch directories, with G-08-2's real-data status stated honestly *(PER-02, PER-05, PER-10)* — wave 3

**Gap closure 2 — code review + verification 2026-09-29 (CR-01, CR-05, CR-06, neutral-posture display)** *(5 plans, 5 sequential waves — every plan moves the recorded test count, so no two share a wave; ADR-0004 budget 0, registry stays 44 byte-identical; Glenn's ruling "fix small ones, then close"; CR-02/CR-03/CR-04 are Phase 8.1)*

- [x] 08-15-PLAN.md — **Display.** Neutral posture no longer lists every (regime, asset) row unlabelled (24 rows, each asset 6×, since `036ae74`); per-asset rows follow the active regime and name it, as the page's own A7 sentence says *(PER-05, PER-10)* — wave 1
- [x] 08-16-PLAN.md — **CR-01 tracer, served path.** `fit_l2_nowcaster` returns the training prior of the rows it fit; serving persists it beside the model; weekly divides by it (π₀ and A stay on the labels, decided and justified); the verified real-data flip (state 3's L 1.96 → 0.63, top belief state 3 → 0) pinned by a test that fails under the old prior *(PER-02, PER-10)* — wave 2
- [x] 08-17-PLAN.md — **CR-01 backtest.** Both drivers divide by the same returned prior (`_refit_l2` → posterior + prior); `likelihood_ratio` refuses the whole-window prior shape; l1only bit-for-bit pins unmodified *(PER-02, PER-03, PER-10)* — wave 3
- [x] 08-18-PLAN.md — **CR-05 + CR-06.** `run_joint_lift` checks a declared ADR-0004 ceiling (`--declared-ceiling`) before any append, with explicit raises that survive `-O`; registry rows record the effective `use_regime_filter`; the joint row names its harness plan instead of claiming `07-11` *(PER-02, PER-07, PER-10)* — wave 4
- [x] 08-19-PLAN.md — **Re-measure and record.** l1only proven byte-unchanged; the l2 leg, B1 and S-1 re-measured under NO_REGISTRY behind controls, every new negative offset adjudicated by truncation; the served belief re-run on real data; old → new amendments in 08-MEASUREMENTS and 08-SERVING, nothing overwritten *(PER-02, PER-03, PER-07, PER-10)* — wave 5

**Explicit non-goals**: no λ sweep; no dependence statistic of any kind; no migration work; no
2021+ holdout use for any selection decision; no re-pin of K or λ; no target pre-declared for
either churn metric; `legacy/` and the reference submodules untouched; the legacy-import ratchet
stays at **31** and may only decrease.

---

### Phase 08.1: Point-in-Time Data Audit (INSERTED)

**Goal:** No feature enters a fit or a score before it could have been published. Found by Phase 8's
code review (CR-03, orchestrator-verified 2026-09-29 against `monthly_raw` built 2026-09-21):
`fred_m2sl` (~1 month lag) and `fred_totalsl` (~2 months) carry `shift: false`, and multpl series
(incl. `div_yield`, 2–3 months, a classifier-#1 lean-set feature) have no lag handling at all — so
every L1 labeling since Phase 1 and both criterion-7 legs are exposed. Scope, to be fixed at
discuss-phase: measure every raw series' publication lag; set the shifts; a test that FAILS when a
feature is used before its publication date; re-run the decision-bearing evaluations under a
declared ADR-0004 trial budget, with earlier positive results labelled possibly-inflated until then.
Look-ahead can only flatter each leg, so the absolute-performance verdicts (DSR does not clear; A11 FAILED) stand; the SIGN of relative numbers (wealth_delta, dd_delta) and the dependence verdict are NOT guaranteed, because the legs are exposed differently (Phase 7 wave-2 verification, 2026-09-29). Criterion 5 is directly exposed: lagging fred_m2sl by 1 month moves classifier #2's occupancy to 3.45/13.79/22.70/21.84/38.22% — outside the 8–35% band at both ends, 17.82% of months keep their state (a 2-month lag gives 98.13%); orchestrator re-ran the diagnostic 2026-09-29.
Candidates to schedule at discuss-phase (Phase 8 named blockers): CR-02 (filter belief carried
across refits whose state ids are not aligned) and the input-independent L2 recipe.
**Order decided (Glenn, 2026-09-30): 08.1 runs BEFORE 08.2** — 8.1's re-measured answer to "does the
regime tilt beat the no-regime ablation on point-in-time data?" shapes 8.2's MVP design.
**Two usability fixes ride along (Glenn, 2026-09-30)**, found in his 2026-09-30 Mac weekly report:

  - **Report always prints the target allocation.** Today the target/executed book appears only
    per configured account; with no account configured the page never says what to hold. Print the
    executed target book unconditionally; per-account trades stay as an extra section.

  - **Staleness cap measured per series, not against the newest raw row.** The 2026-09-30 build
    appended a mostly-empty current-month row (16 series missing), so the scored month (June) sat
    exactly at the 3-month-end cap and next month's run would likely refuse. Base the cap on the
    per-series publication-lag table this phase builds (a series is "stale" only if it is later than
    its own expected lag), so a partial current-month row cannot trip it.
**Usable output of 8.1:** a weekly page that shows a concrete allocation on point-in-time data and
runs reliably month to month; the regime tilt remains unproven until 8.1's re-measurement says otherwise.
**Requirements**: TBD
**Depends on:** Phase 8
**Blocks**: Phase 9 (Migration) — Glenn's ruling, 2026-09-29
**Plans:** 3/3 plans complete (2 waves; lean mode; ADR-0004 budget 2 rows, 44 → 46, spent by 08.1-03 after Glenn's checkpoint "Run"). **Answer E-06: NO** — the regime view stays advisory in 8.2 (G-06).

Plans:

- [x] 08.1-01-PLAN.md — `publication_lags` table for every raw series applied once in `build_monthly_spine`; GDP pre-vintage fallback lag 3; last-complete-month index end; merge-on-save marker guard; PIT truncation test with div_yield mutation proof; D-12 hard-coded-44 sites → 46 — wave 1
- [x] 08.1-02-PLAN.md — weekly: per-series STALE banner replaces the q1-c cap (A-12 supersedes A-07); always-printed target allocation table (A-11); README 3-line accounts how-to — wave 1
- [x] 08.1-03-PLAN.md — D-06 "before" run (NO_REGISTRY) on HEAD data, offline migration of tracked data to point-in-time, test re-pins, checkpoint, D-04 tilt-vs-ablation run (44 → 46) and DECISIONS rows (E-06 answer) — wave 2, checkpoint

### Phase 08.2: Lean MVP — Simplify, Modularize, Notebook-Gate (INSERTED)

**Goal:** A product Glenn can **use** every week, built and reviewed module by module, with planning
kept small. Added 2026-09-29 at Glenn's request: the planning-to-code ratio had reached ~5.5:1
(Phase 8: 17.8k planning lines vs 3.2k source lines changed) and the product is still not usable
with confidence. Plan: `REBUILD-FROM-SCRATCH-GUIDE.md` §3–§5; decisions: `platform_design/DECISIONS.md`
(Lean column).
**Lean mode (applies to this phase and after, unless Glenn says otherwise):** one module per phase,
≤ 2–3 plans, ≤ ~150-line PLANs, ≤ 5 discuss questions, decisions recorded as `DECISIONS.md` rows;
mutation proofs / ruling records / amendments only for **decision-bearing** numbers.
**Candidate success criteria (to be confirmed at discuss-phase):**

  1. **MVP-1 usable:** one command on real data produces the weekly report with a regime-free,
     vol-targeted core allocation that works on its own; the regime view is shown as **advisory**
     (no weight) until it beats the no-regime ablation.

  2. **Module map:** the platform is organised as the guide's modules M0–M7, each with a small stated
     interface — simplification by *not carrying forward* (classifier #2 / joint driver / stability
     suite parked, not deleted).

  3. **Notebook gates:** every module used by MVP-1 has a notebook that runs top-to-bottom on real
     data with one human sign-off cell — including the gaps today: serving/report (N4), the filtered
     belief, and a "does the regime layer pay rent?" scoreboard (N7).

  4. **Builds on 08.1** (which now runs first, Glenn 2026-09-30): point-in-time data, the per-series
     staleness rule and an always-visible target allocation are already in place; 8.1's re-measured
     tilt-vs-ablation answer decides whether MVP-1 shows the regime view as advisory or weighted.

  5. Full suite green on Mac and Linux; no new trial-registry rows (budget 0).

**Requirements**: TBD
**Depends on:** Phase 8
**Order:** after 08.1 (decided by Glenn 2026-09-30). 08.2's modules are the natural units for Phase 9's
module-by-module migration.
**Plans:** 3/3 plans complete (2 waves; lean mode; budget 0, registry stays 46). MVP-1 delivered: weekly page on the no-regime path, regime view suspended, tripwire, static scoreboard; classifier #2 / joint driver / stability parked; MODULE-MAP; notebooks P7–P9 signed off.

Plans:

- [x] 08.2-01-PLAN.md — **Page on the no-regime path.** `report.allocation_mode` (live `no_regime`, code default `regime_tilt`); weekly target = the ablation's own functions (parity with `run_backtest(use_regime_tilt=False)` at rel 1e-9, synthetic + tracked data); regime view suspended (D-03); mode checkpoint + A-15 switch rule (held=None, note); DECISIONS A-13 built, A-15 — wave 1
- [x] 08.2-02-PLAN.md — **Park + module map.** `quality_tier` extracted to `evaluation/deflated_sharpe.py`; classifier2 / joint_driver / stability `git mv`'d to `platform/parked/` (no shims, import-only edits to record tests); boundary test (subprocess `sys.modules` + AST scan); `platform_design/MODULE-MAP.md` (M0–M7, M8, M9+, Parked) — wave 1
- [x] 08.2-03-PLAN.md — **Tripwire, scoreboard, notebooks.** `evaluate_tripwire` (per-signal as-of; STALE after 5 business days; UNAVAILABLE, never green) + `fred_daily_raw` in the build; static scoreboard (`report/scoreboard.py`, common 1972–2020 window, E-07 line); `build_weekly_page`; notebooks P7 (N4) / P8 (N6) / P9 (N7) run headless with pending sign-offs; real-data scratch smoke; DECISIONS A-14 built, G-08 — wave 2

### Phase 08.3: Regime Rebuild I — month-end returns (E-07) and re-measure (INSERTED)

**Goal:** The honest scoreboard is measured on returns a trader could actually earn. Today `equities_tr`
(multpl S&P monthly **average**), `oil` (WTISPLC monthly average) and `long_duration_tr` (GS10 monthly
average) produce next-month returns that include part of the move month-end features already saw
(DECISIONS E-07), which can flatter trend and tilt rules. Rebuild these research series from **month-end**
prices where a free point-in-time source exists, then re-measure the scoreboard (no-regime ablation,
regime tilt, SPY, 60/40, Faber) once, under a declared ADR-0004 budget, before/after side by side.
**Why first (Glenn, 2026-10-02):** first step of the regime rebuild, because its answer can change MVP-1's
core-mix choice (no-regime ablation vs Faber, A-13) and is the baseline every later regime module is judged against.
**Lean mode:** one module (M3 returns + scoreboard), ≤ 3 plans, ≤ 5 discuss questions, decisions as
`DECISIONS.md` rows; rigor (mutation proof, before/after) only for the decision-bearing re-measurement.
**Candidate success criteria (to be confirmed at discuss-phase):**

  1. Equity / oil / long-duration monthly returns are month-end-to-month-end (or documented as a
     bounded exception), applied once in the splice, with a test that fails on an averaged series.

  2. One budgeted re-measurement: before (8.1 numbers) vs after, all five legs, common 1972–2020 window,
     10 bps; DSR at the live count; one-sentence answers to "does the tilt beat the ablation?" and
     "does MVP-1's core mix stand?" recorded in DECISIONS.

  3. The weekly page, scoreboard and P9 reflect the new numbers; the E-07 caveat is removed or narrowed.
  4. Full suite green on Python 3.10 (pandas 2) and 3.11+ (pandas 3); point-in-time test still green.

**Requirements**: TBD
**Depends on:** Phase 08.2
**Later regime-rebuild items:** the L2 recipe with a distinct-output check (M6), CR-02 cross-refit state alignment (M5),
A-03 hysteresis revisit and REG-01 second axis. Phase 08.4's discussion decides which fold into the cold start.
**Plans:** 3/3 plans complete

Plans:

- [x] 08.3-01-PLAN.md — **Month-end P&L, built but off.** ^GSPC / DGS10 / DCOILWTICO month-end columns (P&L-only, lag 0); one `features_from_raw` drop shared by the build and the recompute script, with a mutation-proved no-leak check on `monthly_features` and on the L2 fit; `build_pnl_research_series` + `pnl_splice` overlay (oil ratio-spliced to WTISPLC at 1986-01, D-02); `run_backtest(pnl_returns=)` at the wealth line only, with the ablation getting the same frame; baselines and report wiring (Faber month-end signal and returns, D-08; tilt inputs unchanged, D-09); ceiling test sites 46 → 48. The live P&L stays unchanged until 08.3-02 — wave 1
- [x] 08.3-02-PLAN.md — **Before, migrate, decide, run.** NO_REGISTRY before run on HEAD reproduces 8.1 at rel 1e-9 (`pit_08.3/before/`); additive `monthly_raw` migration (+3 columns; the 47 existing columns exact; merge=False; marker) plus a feature-neutrality check; checkpoint; `pnl_splice` activated and one 2-row run (46 → 48, tag `08.3-monthend-tilt-vs-ablation`); DECISIONS E-07 done, E-08 built, E-12 before|after with DSR at 48 vs 2.2606 and the E-06 / E-09 answers; scoreboard tag and tracked re-pins so the suite ends green — wave 2, checkpoint
- [x] 08.3-03-PLAN.md — **Page and notebooks.** E-07 caveat narrowed to oil before 1986; real-data scratch page (same book, 55 model columns, 08.3 scoreboard); P9 (N7) and P7 re-executed headless; README and recorded counts; full suite on pandas 3 and pandas 2 — wave 3

### Phase 08.4: Cold-Start Tooling — Archive, Reset, Restore & Fail-Loud Build (INSERTED)

**Goal (Glenn, 2026-10-05):** the platform can archive its datasets, outputs and models, and **cold-start or reset at
any time**. Chosen archives can be version-controlled in git as tagged model snapshots.

- **One plain module (`platform/state.py`, KISS K-1..K-11)** provides `archive`, `list`, `reset`, `restore` and
  `promote`.

- **Each archive** is a `tar.gz` plus a manifest:
  - name, note, created time;
  - git commit and dirty flag;
  - registry trial count;
  - per-file sha256.
- **Archives are local and gitignored.** `promote` commits one into git and tags it `model/<name>`, without live
  book state (G-11). It never pushes.

- **`reset --yes`** archives first, then empties `data/` and `outputs/`. It never touches the registry or
  `archives/`.

- **No pickles in any archive;** reset deletes them.
- **`restore`** extracts safely (`filter="data"`) and checks sha256 before moving files into place. There is no
  refit; the no_regime page needs no model.

- **The data build fails loud (D-08).** It runs from an empty `data/`, and a failed FRED daily fetch exits 1. In
  `no_regime` mode the weekly page needs no regime model.

**Context:** `.planning/phases/08.4-cold-start-rebuild-and-learnings/08.4-CONTEXT.md`. Budget: 0 rows.
**Depends on:** Phase 08.3
**Plans:** 2/2 plans complete

Plans:
**Wave 1**

- [x] 08.4-01-PLAN.md — **Fail-loud build and a model-free page (K-9, K-10; D-08, D-T7, D-T8).** `build.fail_loud` (true, pinned) + `build.allow_missing_sources`: a missing source or an unallowed fallback splice raises `BuildFailed` before any write; derived splice columns are replaced on merge (`replace_columns`); a failed FRED daily fetch exits 1 (A1 test flipped, dated); the closing hint names the page; one plain 221defc regression. The no_regime page needs no nowcaster, labels or belief, and shows "Scoreboard: not yet measured" — wave 1

**Wave 2** *(runs after 01, so the D-08-built row and the full-suite run see both plans)*

- [x] 08.4-02-PLAN.md — **State tool and docs (K-2..K-8).** One module, `python -m trading_crab_lib.platform.state {archive,list,reset,restore,promote}`: plain `state.tar.gz` + manifest (per-file sha256, commit, trial count); no pickles anywhere; reset `--yes` auto-archives and never touches registry/ or archives/; restore into empty trees with `filter="data"` and sha checks, no refit; promote drops G-11 live state, commits and tags `model/<name>`, never pushes. `.gitignore` / pre-commit lines; README state section; DECISIONS G-12, D-08 built; full suite. Mac checks go to UAT — wave 1

### Phase 08.5: Clean-Slate Reset & Learnings Before Migration (INSERTED)

**Goal (reframed by Glenn, 2026-10-05): a production clean-slate reset, done with 8.4's tooling.**

- Archive and promote the current state as `model/pre-reset-2026-10`.
- Clear **everything** generated out of git: data (raw included), outputs, labels, models, notebook outputs.
- Reset the trial registry to a **true-zero epoch** (ADR-0005, with the caveat disclosed).
- From a **fresh clone** on Glenn's Mac: build, then the MVP-1 no-regime page, run twice with identical output, then
  the N4 sign-off.

- Fold the learnings into the guide, MODULE-MAP, MIGRATION-PLAN and DECISIONS.
- Restructure Phase 9 into a **module-by-module, human-gated rebuild (9.x: M0→M7) in the target `trading-crab` repo**.

**Not in 8.5:** rebuilding any regime model: E-11 (M3), the L2 recipe (M6), CR-02 (M5), A-03, REG-01.
**Why before Phase 9:** migrating a stack that has only ever run incrementally risks carrying hidden state into the
public repo.
**Context:** `.planning/phases/08.5-clean-slate-reset-learnings-before-migration/08.5-CONTEXT.md` (captured in the 8.4
discussion). Budget: 0 rows of measurement, plus the deliberate epoch reset.
**Depends on:** Phase 08.4
**Blocks:** Phase 9 (Migration): its order and scope come from 8.5's learnings.
**Plans:** 0 plans

Plans:

- [ ] TBD (run /gsd-plan-phase 8.5)

### Phase 9: Migration to Public Repo

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
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 8.1 → 8.2 → 8.3 → 8.4 → 8.5 → 9
(8.3 inserted by Glenn 2026-10-02: the regime rebuild starts with E-07 before migration.)
(8.4 inserted by Glenn 2026-10-02: cold-start rebuild and learnings before any migration.)
(Order confirmed by Glenn 2026-09-30: 8.1 before 8.2 — 8.1's outcome shapes 8.2's design.)
(Phase 7 gates internally: wave 2 does not start unless wave 1 passes.)

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Monthly Data Layer & Long Histories | 7/7 | Complete   | 2026-07-15 |
| 2. Honesty Infrastructure | 5/5 | Complete   | 2026-07-22 |
| 3. Regime Labeling & Prediction | 4/4 | Complete   | 2026-07-22 |
| 4. Asset Prediction & Allocation | 5/5 | Complete   | 2026-07-23 |
| 5. Honest Backtest & Evaluation | 7/7 | Complete (closed 2026-08-04) | 2026-07-27 |
| 6. Platform Notebook Suite | 7/7 | Executed + verified (5/5 criteria; human_needed) | 2026-09-10 |
| 7. Regime Representation | 12/12 | Complete (closed 2026-09-21; ADR-0002 Accepted; criterion 6 UNRESOLVED, no tie-break) | 2026-09-21 |
| 8. Regime Persistence & Stability | 19/19 | Complete (verified 12/12; Mac human checks passed) | 2026-09-30 |
| 08.1 Point-in-Time Data Audit (INSERTED) | 3/3 | Complete (verified; Mac run passed; E-06 NO) | 2026-09-30 |
| 08.2 Lean MVP — Simplify, Modularize, Notebook-Gate (INSERTED) | 3/3 | Complete (verified; Mac run passed; P7–P9 signed off) | 2026-10-02 |
| 08.3 Regime Rebuild I — month-end returns (E-07) (INSERTED) | 3/3 | Complete (verified; Mac run passed; P7–P9 signed off; Mac data commit reverted, D-08) | 2026-10-05 |
| 08.4 Cold-Start Tooling — Archive/Reset/Restore (INSERTED) | 2/2 | Complete (verified 13/13; Mac UAT 4/4 passed; D-09 gold source) | 2026-10-09 |
| 08.5 Clean-Slate Reset & Learnings (INSERTED) | 0/TBD | Not started | - |
| 9. Migration to Public Repo | 0/TBD | Not started | - |
