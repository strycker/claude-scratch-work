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
- [ ] **EVAL-02**: Baseline gauntlet in the backtest report: buy-and-hold SPY, 60/40, Faber 10-month SMA (design §8.7, §23.1)
- [ ] **EVAL-03**: Sojourn/detection-lag ratio reported prominently — the go/no-go number for regime timing (design §5.4)
- [ ] **EVAL-04**: Model metrics artifacts (multiclass Brier, calibration bins, confusion tables) persisted per run (salvaged `model_metrics_artifacts.py`; design §8.8)

### Migration (MIG)

- [ ] **MIG-01**: Validated skeleton code migrated to `strycker/trading-crab` (two-package layout), tests green there, README/docs updated

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
| EVAL-02 | Phase 5 | Pending |
| EVAL-03 | Phase 5 | Pending |
| EVAL-04 | Phase 5 | Pending |
| MIG-01 | Phase 6 | Pending |

**Coverage:**

- v1 requirements: 28 total
- Mapped to phases: 28
- Unmapped: 0 ✓

---
*Requirements defined: 2026-07-09*
*Last updated: 2026-07-09 after roadmap creation (corrected v1 requirement count from 24 to 28 — recount of the itemized list above)*
