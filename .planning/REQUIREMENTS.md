# Requirements: Trading-Crab

**Defined:** 2026-07-09
**Core Value:** Honest, regime-aware weekly guidance that beats buy-and-hold SPY net of avoided drawdowns — never fooled by its own backtest.

Scope: v1 = design §14 **Phase 0 (honesty foundations) + Phase 1 (tracer bullet)** + migration
to the public repo. Design references: `platform_design/platform_design.md` v1.7.

## v1 Requirements

### Honesty Infrastructure (HON)

- [x] **HON-01**: 2021+ holdout data is physically carved into separate files/paths that the dev pipeline cannot read by default; live scoring mode opts in explicitly (design §8.5, D13)
- [x] **HON-02**: Trial registry logs every evaluated configuration (config hash → features, params, metrics) to a flat file/SQLite store; all grid cells logged regardless of outcome (design §8.4, §22)
- [x] **HON-03**: Walk-forward runner refits L1 labels and downstream models on data ≤ t at each step, records decisions, and steps forward — runs a trivial model end-to-end (design §8.1, §14 Phase 0 exit)
- [x] **HON-04**: Purged + embargoed CV splitter available for all supervised components with overlapping labels, replacing plain TimeSeriesSplit (design §6.5, R7)
- [x] **HON-05**: Smoothed-vs-filtered gap and detection lag are computed and reported as first-class outputs (design §5.4)
- [x] **HON-06**: Causal-feature gating enforced for supervised training — supervised paths load causal features by default with loud opt-out (salvaged `feature_gating.py`; design R5, pitfall P1)

### Data Layer (DATA)

- [x] **DATA-01**: Monthly data spine — ingestion and transforms produce monthly features; quarterly agency series aligned with publication lags (design R1, D10)
- [x] **DATA-02**: Spliced USD histories back to ~1962 for core assets (S&P total return, Treasury total-return synthetic, gold, oil, cash), free sources, splicing rules documented per asset (design R13, §9)
- [x] **DATA-03**: ALFRED point-in-time vintages for agency series where archives exist, documented publication-lag alignment fallback pre-vintage era (design R6, D12)
- [x] **DATA-04**: Features classified into fast/slow/agency taxonomy in config; lean full-history (1962+) feature set defined for labeling (design §9, D11)
- [x] **DATA-05**: Key satellite ETFs and Glenn's holdings ingested with NULL-tolerant handling for shorter histories (design D11)
- [x] **DATA-06**: Paid-provider adapter seams documented (Norgate, Tiingo, EODHD) — placeholder notes only, no implementation; stockcharts.com/finviz.com noted as candidate feature sources

### Regime Labeling (L1)

- [x] **L1-01**: Jump-model labeler (k-means + per-jump penalty λ, exact DP decode, multi-restart, k-means warm start) with default λ, K=5 (design §4.1, R2)
- [x] **L1-02**: Labels and soft confidences persisted; trailing 6–12 month labels embargoed from L2 training (design §3)
- [x] **L1-03**: Label churn (fraction of trailing labels revised per refresh) tracked as a monitoring metric (design §5.4)

### Regime Prediction (L2)

- [x] **L2-01**: Calibrated logistic nowcaster P(regime | causal features) — probabilities, not argmax (design §5.1, §14 Phase 1)
- [x] **L2-02**: Empirical transition matrix retained as diagnostic input for forward regime distribution (feature-conditional TVTP model deferred to v2; design R3)

### Asset Prediction (L3)

- [x] **L3-01**: Returns-by-regime tables (the trading-crab baseline) for the v1 universe (design §14 Phase 1)
- [x] **L3-02**: EWMA volatility forecasts per asset feeding sizing and the tripwire (design §6.2)

### Allocation & Report (L4)

- [x] **L4-01**: Naive vol-targeted regime-tilt allocation with hysteresis bands (act ~0.7 / unwind ~0.4) (design §5.3, §14 Phase 1)
- [x] **L4-02**: Weekly report: current regime distribution + trajectory, per-asset signals, target mix vs current mix with trades implied — reusing existing report/email machinery (design §7)
- [x] **L4-03**: Current holdings input via manual YAML per account; Fidelity CSV parser seam documented as placeholder (§16.7 accounts)
- [x] **L4-04**: Minimal daily tripwire monitor — 3 signals from independent families (e.g. vol spike, credit-spread velocity, drawdown-from-peak) with OR-logic escalation output: none / "run weekly scoring early" / "Tier-1 de-risk review" (design §23.2, §25)

### Evaluation (EVAL)

- [x] **EVAL-01**: Honest walk-forward backtest 1972–2020 runs end-to-end through all layers (design §14 Phase 1 exit)
- [x] **EVAL-02**: Baseline gauntlet in the backtest report: buy-and-hold SPY, 60/40, Faber 10-month SMA (design §8.7, §23.1)
- [x] **EVAL-03**: Sojourn/detection-lag ratio reported prominently — the go/no-go number for regime timing (design §5.4)
- [x] **EVAL-04**: Model metrics artifacts (multiclass Brier, calibration bins, confusion tables) persisted per run (salvaged `model_metrics_artifacts.py`; design §8.8)

### Notebooks (NB)

- [x] **NB-01**: Six platform notebooks (`P1_data_spine`, `P2_features_taxonomy`, `P3_regime_labeling`, `P4_nowcaster`, `P5_assets_allocation`, `P6_backtest_evaluation`) covering L0–L4 + evaluation, serving as a **periodic V&V surface** (regimes still well-defined? data still behaving as historic?); each runs top-to-bottom against real checkpoints, reads the full span via the explicit holdout opt-in while fitting stays fenced at 2020-12, compares current behavior against the pre-2021 fitted baseline, and calls plotting logic from `platform/plotting/` rather than defining it inline. **`P3_regime_labeling` additionally carries a cold-start sign-off cell**; the other five carry no per-run gate

### Migration (MIG)

- [ ] **MIG-01**: Platform decoupled from the legacy library (4 seams vendored, import-guard test) and migrated to `strycker/trading-crab` (two-package layout), tests green in that repo's CI, real run reproduces the reference numbers, README/docs updated — step plan in `MIGRATION-PLAN.md`

### Regime Persistence & Stability (PER)

Minted at Phase 8 planning, 2026-09-21, one per ROADMAP Phase 8 success criterion 0–9 in order.
The corrected causal model behind them is in `08-CONTEXT.md`'s AMENDMENT: the recorded 41.91%
filtered churn is an **L1** quantity (`joint_driver.py:502`), while design §5.1 changes **L2** and
under the decision-bearing `ROUTING_L1_ONLY` `_refit_l2` is never called
(`joint_driver.py:431`). The two tracks are therefore separate requirements and neither may be
measured by the other's metric.

- [ ] **PER-01**: The per-step probability matrix the walk-forward drivers accumulate
  (`driver.py:530-532`, `joint_driver.py:508-510`) is persisted to disk for both classifiers and
  both routings — the prerequisite that gates PER-02, PER-03 and PER-05 (criterion 0)
- [ ] **PER-02**: A prior-state belief propagates into the nowcaster's consumed output via an
  explicit Bayes filter, `π_t ∝ [Σ π_{t−1} A] · L_t`, with zero train/serve skew, zero leakage
  surface and zero free parameters — and a guard test that **fails** on a smoothed substitution
  (criterion 1, reworded 2026-09-21; design §5.1/§4.2)
- [ ] **PER-03**: Both churn series are reported with their own denominators, windows and degraded
  counts, and neither can masquerade as the other: Track A (`state_1` changes / 587) and Track B
  (`argmax(regime_probs)`), with no target pre-declared for either (criterion 2)
- [ ] **PER-04**: Track A is diagnosed before it is changed — the zero-trial terminal-month
  diagnostic, churning the step-*t* fit's label for months *t−1 … t−6* across steps. A λ sweep is
  **not** authorized (criterion 3)
- [ ] **PER-05**: §5.3's hysteresis gates allocation, closing audit item A7, evaluated only after
  the probability vector stops being degenerate; the mechanism and the 0.70/0.40 pair are human
  decisions, not planner choices (criterion 4; design §5.3)
- [ ] **PER-06**: §4.4 criterion 3 is RUN for both classifiers under four subsample schemes —
  drop first decade, drop last decade, circular block bootstrap, and leave-one-episode-out — using
  centroid distance in de-standardized units with Hungarian matching, a within-state split-half
  null at the same n, and subsample occupancy on every row (criterion 5; design §4.4)
- [ ] **PER-07**: Criterion 7 is re-measured under whatever changed, both legs, one harness,
  window inline; the prior `wealth_delta` −0.123438 / `dd_delta` +0.024084 is a comparison point,
  not a target (criterion 6)
- [ ] **PER-08**: Audit item A11 — *"no gate fails on a bad model"* — is answered and written as
  the reversal of the 2026-09-18 decision to leave it open (criterion 7)
- [ ] **PER-09**: Validation gap G6 is pinned as the known **non**-compliance: non-joint consumers
  of `vol_targeted_tilt` receive the unpooled per-regime estimate (criterion 8)
- [ ] **PER-10**: Recorded counts match measured reality — the four documentation sites, and
  F-4's churn denominator (587 adjacent pairs, not 588 months) with its pin moved in the same
  commit (criterion 9)

### Invariants (INV)

- [~] **REG-01**: One documented feature policy shared by the walk-forward driver and the
  evaluation reference (resolves audit items A13/A15), with the §5.4 sojourn/lag ratio made
  interpretable and classifier #1's ablation delta re-measured on both wealth and drawdown;
  then a **second, independent regime classifier** fit unsupervised on relative/leadership
  features disjoint from classifier #1's, with orthogonality measured (not assumed) and the
  joint allocation lift assessed walk-forward against classifier #1 alone — every
  configuration logged to the trial registry. Scope in
  `.planning/PROPOSAL-phase-regime-representation.md`

  > **DELIVERED PARTIALLY — Phase 7, 2026-09-21. The open item is named, not folded in:
  > criterion 6's dependence verdict is UNRESOLVED.** Wave 1 (ADR-0001) satisfied the
  > feature-policy, §5.4-interpretability and ablation-re-measurement clauses. Wave 2 (ADR-0002)
  > satisfies three of the four remaining clauses — the second independent classifier on a
  > disjoint feature set, the disjointness assertion
  > (`test_resolved_frozen_list_is_disjoint_from_the_lean_set`), and the joint lift assessed
  > walk-forward against classifier #1 alone (`wealth_delta` **−0.123438**, `dd_delta`
  > **+0.024084**, both over **588 steps, 1972-01-31 → 2020-12-31**, tags
  > `07-11-c1-alone-L1only` / `07-11-joint-c1xc2-L1only`). **The orthogonality clause was
  > measured but its verdict is unresolved:** the pre-registered block-permutation control
  > returned **INCONCLUSIVE** — NMI **0.464088** at the **96.60th** percentile against p95
  > 0.449202 and p99 0.501195, `n_compared` = **695** months, **1963-02-28 → 2020-12-31** — and
  > the pre-registration at `298b1bc` forbids a tie-break. **No independent second axis is
  > established by this phase.** Evidence:
  > `platform_design/adr/0002-l1-second-classifier.md` § Requirement coverage and
  > § ACCEPTANCE 2026-09-21; `.planning/phases/07-regime-representation/07-DEPENDENCE.md`,
  > `07-JOINT-LIFT.md`.

- [x] **INV-01**: Named, era-stable invariant candidates (e.g. M2/GDP, market cap/GDP, credit/GDP) constructed and screened using dimensional-reduction techniques as discovery tools; loading stability tested across eras; every candidate logged to the trial registry and assessed walk-forward; survivors admitted as **named** features, never anonymous principal components (preserves design decision R4)

  > **DELIVERED IN FULL — Phase 7, 2026-09-21**, closing the D-09 wave-2 deferral ADR-0001
  > recorded. All four clauses, each with its artifact: (1) candidates constructed and screened
  > with dimensional reduction as a **discovery tool only** — `pca.transform()` is never called
  > in `platform/features/invariants.py`, so no component score was ever computed; (2) loading
  > stability tested across **10 eras ending 1972-01-31 → 2017-01-31** at the named tolerance
  > **`LOADING_STABILITY_TOLERANCE = 0.15`**; (3) every candidate logged to the real trial
  > registry — 2 rows tagged `07-07-inv01-screen`, `total_trial_count()` 38 → 40; (4) survivors
  > admitted as **named** features — `m2_gdp` and `credit_gdp`, of which only `m2_gdp` enters
  > classifier #2's frozen eight (`credit_gdp` dropped for 0.957–0.969 collinearity), R4 held
  > structurally by `test_rejects_integer_or_component_label_index`. **Claimed at the strength
  > of the evidence and no further:** the era screen was *survived*; a PC1 loading of exactly
  > 1/√2 on two standardized candidates is an arithmetic identity, not by itself evidence of
  > five-decade economic stability. Evidence:
  > `platform_design/adr/0002-l1-second-classifier.md` § Requirement coverage;
  > `.planning/phases/07-regime-representation/07-INV01-SCREENING.md`.

## v2 Requirements

Deferred to later milestones. Tracked but not in the current roadmap.

### Regime Quality (design Phase 2)

- **L1-V2-01**: (K, λ) grid against §4.4 acceptance criteria; subsample stability with Hungarian matching; t-HMM benchmark comparison (§22)

### Regime Prediction (design Phase 3)

- **L2-V2-01**: Nowcaster upgrade — recursive prior-state feature, γ sample weights, transition-window metrics
- **L2-V2-02**: Feature-conditional transition model with regime age (TVTP-style)
- **L2-V2-03**: Full tripwire orchestrator with family-independence voting (§25)

### Asset Prediction (design Phase 4)

- **L3-V2-01**: Regime-conditional covariance + Ledoit–Wolf; GARCH layer; DCC option
- **L3-V2-02**: Mixture-of-experts (soft gating, partial pooling) + boosted ceiling model
- **L3-V2-03**: Fair-value gap module with convergence KPI (§6.3)

### Allocation & Tactics (design Phase 5)

- **L4-V2-01**: BL/HRP weights, fractional Kelly, no-trade bands (§21)
- **L4-V2-02**: Model-driven vol-scaled regime-conditional stops + §27 policy stack (thesis-typed risk tools)
- **L4-V2-03**: Crash-probability dashboard + crisis-TYPE conditioning (§23)
- **L4-V2-04**: Tactical sleeve reporting (separate account) with time stops (§16.5)

### Data & Ops

- **DATA-V2-01**: Paid data provider integration (Norgate/Tiingo/EODHD) when breadth features need survivorship-clean constituents
- **DATA-V2-02**: stockcharts.com / finviz.com feature ingestion (existing subscriptions; root ROADMAP 3.6/3.7)
- **OPS-V2-01**: Automated scheduled runs (cron/GitHub Actions) + email delivery of the weekly report
- **L4-V2-05**: Fidelity positions-CSV parser replacing manual YAML upkeep
- **FEAT-V2-01**: Breadth/dispersion/VIX-term-structure fast features (§16.3); options-implied Tier 1 features incl. VRP (§18.1); volume integration (§20); AVWAP module (§17)
- **FEAT-V2-02**: "Old Fool Indicator" — awaiting Glenn's definition from Mike Silva (Figuring Out Money) source material (§26)

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Automated trade execution | Guidance-only philosophy; human executes in Fidelity |
| Sub-monthly decision horizons | Edge thesis lives at monthly+ horizons (D8) |
| Shorts, options trading, leveraged-inverse ETFs | §16.7 resolved; long-only (SH only in future sleeve) |
| Crypto, MLPs | §16.7 exclusions (K-1 avoidance) |
| Deep sequence models / RL for returns or allocation | §15 declined; sample size cannot support them |
| Tax/friction modeling | Operator handles; turnover penalty is for stability only |
| Chain-level options features (GEX, max pain) as model inputs | No free backtestable history — levels only (§18.2) |
| Sticky HDP-HMM | §4.2 declined — hallucinates states at this sample size |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| DATA-01 | Phase 1 | Complete |
| DATA-02 | Phase 1 | Complete |
| DATA-03 | Phase 1 | Complete |
| DATA-04 | Phase 1 | Complete |
| DATA-05 | Phase 1 | Complete |
| DATA-06 | Phase 1 | Complete |
| HON-01 | Phase 2 | Complete |
| HON-02 | Phase 2 | Complete |
| HON-03 | Phase 2 | Complete |
| HON-04 | Phase 2 | Complete |
| HON-05 | Phase 2 | Complete |
| HON-06 | Phase 2 | Complete |
| L1-01 | Phase 3 | Complete |
| L1-02 | Phase 3 | Complete |
| L1-03 | Phase 3 | Complete |
| L2-01 | Phase 3 | Complete |
| L2-02 | Phase 3 | Complete |
| L3-01 | Phase 4 | Complete |
| L3-02 | Phase 4 | Complete |
| L4-01 | Phase 4 | Complete |
| L4-02 | Phase 4 | Complete |
| L4-03 | Phase 4 | Complete |
| L4-04 | Phase 4 | Complete |
| EVAL-01 | Phase 5 | Complete |
| EVAL-02 | Phase 5 | Complete |
| EVAL-03 | Phase 5 | Complete |
| EVAL-04 | Phase 5 | Complete |
| NB-01 | Phase 6 | Complete |
| REG-01 | Phase 7 | **Partial** — criterion 6 (dependence) UNRESOLVED; see ADR-0002 |
| INV-01 | Phase 7 | Complete |
| PER-01 | Phase 8 | Pending |
| PER-02 | Phase 8 | Pending |
| PER-03 | Phase 8 | Pending |
| PER-04 | Phase 8 | Pending |
| PER-05 | Phase 8 | Pending |
| PER-06 | Phase 8 | Pending |
| PER-07 | Phase 8 | Pending |
| PER-08 | Phase 8 | Pending |
| PER-09 | Phase 8 | Pending |
| PER-10 | Phase 8 | Pending |
| MIG-01 | Phase 9 | Pending |

**Coverage:**

- v1 requirements: 31 total
- Mapped to phases: 31
- Unmapped: 0 ✓

---
*Requirements defined: 2026-07-09*
*Last updated: 2026-07-09 after roadmap creation (corrected v1 requirement count from 24 to 28 — recount of the itemized list above)*
