# Platform Decision Register

**One line per decision, all phases, human-editable.** This is the index. The long-form
reasoning lives in the linked source, and the source wins if the two disagree. Edit a row here
to change a decision before re-planning or rebuilding. Mark it `CHANGED yyyy-mm-dd (who)` and
leave the old text struck through, so the history stays readable.

**Scope:** the monthly L0–L4 platform (`src/trading_crab_lib/platform/`,
`config/platform_settings.yaml`, `notebooks/platform/`). The legacy quarterly pipeline has its
own log in the root `CLAUDE.md` (ADR #1–#12, D1–D50). It is frozen and not covered here.

**Created:** 2026-09-29, from phase CONTEXT files, ADRs 0001–0004 and the Phase 8 ruling
records.

**Lean column.** This records whether a lean rebuild would **KEEP** the decision as is,
**SIMPLIFY** it, or **DEFER** it until the product is usable. It is the orchestrator's
recommendation (2026-09-29) and **not yet ruled on**. Glenn edits it.

---

## Product and scope

| ID | Decision | Why | Source | Lean |
|---|---|---|---|---|
| P-01 | Guidance only: the platform recommends and Glenn trades manually in Fidelity. Long-only; no options, shorts, crypto or MLPs. | Personal investor, not a trading desk | PROJECT.md, design §1 | KEEP |
| P-02 | Benchmark: buy-and-hold SPY **net of avoided drawdowns**. | The reason to exist is sidestepping 2000/2008-class damage | design §1 | KEEP |
| P-03 | Monthly modelling spine, weekly scoring, nothing under 1 month. | The edge thesis lives at monthly+ horizons | design D8, D10 | KEEP |
| P-04 | Free data sources only in v1; paid providers (Norgate, Tiingo, EODHD) are placeholder adapters. | Cost; splice quality is acceptable | Phase 1 | KEEP |
| P-05 | Build in `claude-scratch-work`, then migrate the validated platform to `strycker/trading-crab`. The legacy quarterly pipeline stays behind. | Heavy-dev tooling here, clean public repo there | MIGRATION-PLAN.md, ROADMAP Phase 9 | KEEP |
| P-06 | Tracer-bullet build: every layer present and naive first, then upgrade modules. | A usable skeleton early | design §14 | KEEP (the most important one) |

## L0 — Data

| ID | Decision | Why | Source | Lean |
|---|---|---|---|---|
| D-01 | "Model the index, trade the ETF": one spliced research series per asset class, mapped to a tradable ticker. | Long histories (1962+) that ETFs lack | Phase 1 D-03 | KEEP |
| D-02 | Core 5 classes: equities (SPY), long duration (TLT), gold (IAU), oil (USO), cash. Satellites, holdings and watchlist are scored as well. | Minimal cross-asset spine | Phase 1 D-08..D-13 | KEEP |
| D-03 | Ratio-splice at overlap windows; provenance of every resolved source is logged. | Continuity without level jumps | Phase 1 D-04; docs/splicing_rules.md | KEEP |
| D-04 | ALFRED point-in-time vintages only for revision-heavy agency series; other series get a publication-lag shift. | Revisions leak future information | Phase 1 D-06 | SIMPLIFY (shifts first, vintages later) |
| D-05 | **Every feature must be lagged to its publication date.** Measured gaps (2026-09-29): `fred_m2sl` ~1 month, `fred_totalsl` ~2 months, `div_yield` (multpl) 2–3 months are all unshifted today. | Look-ahead (CR-03) | 08-REVIEW CR-03; Phase 08.1 | KEEP, **open** — **How (Glenn 2026-09-30):** measured `publication_lag_months` table for every series, applied once in ingestion; ALFRED series stay vintage-aligned (08.1-CONTEXT D-01..D-03) |
| D-06 | Frozen 13-column lean taxonomy (fast + slow) for classifier #1. | Few, named, interpretable features | Phase 1 taxonomy | KEEP |

## Honesty rails

| ID | Decision | Why | Source | Lean |
|---|---|---|---|---|
| H-01 | Physical 2021+ holdout carve: the dev checkpoint manager cannot read post-2020 rows. | Unlimited peeking destroys the out-of-sample test | Phase 2 D-03 | KEEP (cheap, already built) |
| H-02 | The fence is on **fitting**, not on **looking**: scoring, monitoring and notebooks may read the full span. A post-2020 observation that changes a decision is recorded with its date. | Must see live decay | Phase 6 D-06/D-07; PROJECT.md | KEEP |
| H-03 | Append-only, git-tracked trial registry; one row per evaluated configuration. | Multiple-testing denominator | Phase 2 D-01/D-02 | SIMPLIFY (keep the ledger; drop per-row ceremony) |
| H-04 | Walk-forward only; purged and embargoed CV; a 12-month label embargo for L2. | No temporal leakage | Phase 2, Phase 3 D-01 | KEEP |
| H-05 | Deflated Sharpe uses the **whole** registry since project start. | Honest selection penalty | Phase 7 D-16 | KEEP |
| H-06 | Per-phase trial **budgets**, declared before the runs and amendable only in advance. Unspent budget expires. There is no standing cap; the 44 was Phase 7–8's closed budget. | Makes silent search-creep visible without a wall | ADR-0004 (2026-09-28) | SIMPLIFY (one line per phase) |
| H-07 | Non-selecting fits (serving refits, observational legs) write `NO_REGISTRY` rows and spend no budget. | They select nothing | ADR-0002 (e); 08-SERVING §0 | KEEP |
| H-08 | Ablation arms are marked `independent_trial: false`: they count toward N but not toward the Sharpe variance. | Otherwise the DSR hurdle collapses | 230c91c; 08-10 | KEEP |
| H-09 | Deflated-Sharpe hurdle is the **one** governing quality gate (A11). The plausibility bands stay non-gating. | "Is the output any good?" needs one honest answer | ADR-0003 | SIMPLIFY (report it, don't block on it pre-MVP) |
| H-10 | Floats compared across platforms at rel 1e-9, abs 0; never bit-exact. | macOS Accelerate vs Linux OpenBLAS differ by 1–32 ULP | 08-11; 2026-09-29 serving-test fix | KEEP |
| H-11 | "A check that can only confirm" is a defect: every new test ships with a mutation that turns it red. | 13 confirm-only checks found in Phase 8 | 08-MEASUREMENTS | SIMPLIFY (for core invariants only) |

## L1 — Regime labeling

| ID | Decision | Why | Source | Lean |
|---|---|---|---|---|
| L1-01 | Statistical jump model is the labeler (t-HMM is only a benchmark). Two-sided labeling is by design; it is offline ground truth. | Out-of-sample label stability; a single persistence knob λ | design D4; Phase 3 | KEEP |
| L1-02 | Classifier #1: **K = 6, λ = 10** (= 1 × 10 fitted columns) on the **10 frozen common-support features**. The walk-forward driver and the evaluation use the same feature space. | K = 5 could not meet the occupancy band at any λ; λ must scale with the fitted count | ADR-0001; config `labeling` | KEEP |
| L1-03 | One sub-floor state is allowed under the recurrence exemption (40 months across 9 crisis episodes). | Crises are rare but real | ADR-0001 | KEEP |
| L1-04 | Classifier #2 (leadership / relative, 8 features, **K = 5, λ = 16**) is pinned by rule, with zero selection trials. | Candidate independent axis | ADR-0002 | **DEFER**: added no lift; independence unproven; criterion 5 breaks under a 1-month M2 lag |
| L1-05 | Occupancy band 8–35% per state; report-only in v1. | Semantic states, not forced balance | design D2; Phase 3 D-02 | KEEP (as a diagnostic) |
| L1-06 | Numeric state ids plus auto-generated one-line economic profiles. State ids are **not** comparable across walk-forward refits. | Canonical ordering is unstable across refits | Phase 3 D-04; 08-MEASUREMENTS §6 | KEEP, **open**: CR-02 needs cross-refit alignment |

## L2 — Regime prediction

| ID | Decision | Why | Source | Lean |
|---|---|---|---|---|
| L2-01 | Calibrated multinomial logistic nowcaster (sigmoid calibration, purged CV, 5 folds) on causal features. | Simple, calibrated, honest baseline | Phase 3 | KEEP |
| L2-02 | Feature admission: at least 120 months of history (`feature_min_history`) and every class ≥ n_splits in the induced block. | CV must not crash on rare classes | driver `_cv_safe_active_features` | **SIMPLIFY**: at full history it admits all 55 features, shrinks training to 153 rows (2007–2019, 3 of 6 regimes) and gives a **constant** posterior |
| L2-03 | Bayes filter π_t ∝ [π_{t−1}A]·L_t with L_t = posterior ÷ **the nowcaster's own training class prior**. The cold start π₀ and A come from the in-window labels. | Real memory without a new trained model | 08-06, 08-16/08-17 (CR-01 fix 2026-09-29) | KEEP |
| L2-04 | Leakage is governed by **causal invariance** (the truncation test): belief at t must not change when rows after t are removed. | The only test that can actually catch look-ahead | S-1 ruling 2026-09-23 | KEEP |
| L2-05 | The serving nowcaster is built by a supported command (`python -m trading_crab_lib.platform.report.serving`) through the backtest's own `fit_l2_nowcaster`. It fits on data ≤ 2020-12 and writes `NO_REGISTRY`. | No train/serve skew | 08-13; 08-SERVING | KEEP |

## L3 / L4 — Assets, allocation, report

| ID | Decision | Why | Source | Lean |
|---|---|---|---|---|
| A-01 | Returns-by-regime tables plus an EWMA vol layer are the L3 baseline. | The naive baseline any mixture-of-experts must beat | Phase 4 | KEEP |
| A-02 | Vol-targeted regime tilt at **10% annual target vol**. | Defensive by design | Phase 4 D-03 | KEEP |
| A-03 | Hysteresis thresholds **0.70 / 0.40**, declared per classifier; the active regime is a reported label and gates no weight. | Stops label flicker from reaching the page | 08-09; 08-A7 | KEEP. **Revisit in 8.1** on the final belief (counts moved 307→273 and 415→408 of 488) |
| A-04 | Bounded turnover via a **5-percentage-point no-trade band**, not swept; the same function in both drivers and in serve. | Filtered churn became trades | 08-09; 08-A7 §7 | KEEP |
| A-05 | 10 bps cost per rebalance on traded notional; turnover reported separately. | Light-touch realism | Phase 5 D-03 | KEEP |
| A-06 | Weekly report always writes markdown; email is opt-in. Holdings are per-account weight YAML. | Human in the loop | Phase 4 D-01/D-02 | KEEP |
| A-07 | Serving scores the **latest month complete in every model column** (q1-c) and refuses if that month is more than **3 month-ends behind the newest data row** (confirmed 2026-09-29). | Publication lag makes the newest row ragged every month | 08-12/08-14; 08-SERVING §2 | KEEP |
| A-08 | The report prints the **count of distinct posterior vectors** (q2-ii), so a constant model can never pass as a live signal. | Honesty on the trading surface | 08-14 | KEEP |
| A-09 | Neutral posture (no active regime): the per-asset section prints one sentence and no rows. Rows that do print name their regime. | 24 unlabelled contradictory rows were misleading | 08-15; ruling 2026-09-29 | KEEP |
| A-10 | Minimal daily tripwire (3 signals). | Crash avoidance is the point | Phase 4 | KEEP |
| A-11 | The weekly report **always prints the target (executed) allocation**; per-account trades are an extra section. | With no account configured the page never said what to hold (Glenn's 2026-09-30 Mac report) | Glenn 2026-09-30; Phase 08.1 | KEEP, **to build** — **Form (Glenn 2026-09-30):** simple table: class, ticker, target %, last week %, change |
| A-12 | Staleness is judged **per series against its own publication lag**, replacing A-07's "3 month-ends behind the newest row" rule. | A mostly-empty current-month row put the scored month exactly at the cap | Glenn 2026-09-30; Phase 08.1 | KEEP, **to build** (supersedes A-07's cap once built) — **Behavior (Glenn 2026-09-30):** stale series → report still runs with a prominent STALE banner naming them (no refusal, no imputation) |

## Evaluation

| ID | Decision | Why | Source | Lean |
|---|---|---|---|---|
| E-01 | Baseline gauntlet: SPY, 60/40, Faber 10-month SMA, plus the no-regime ablation. | "Does the regime layer pay rent?" | Phase 5 D-02 | KEEP (the only question that matters) |
| E-02 | Criterion 7 (joint lift) is routed L1-only and is decision-bearing; L2 legs are observational. | The L2 posterior was too weak to carry decisions | ADR-0002 | KEEP |
| E-03 | Criterion 7 result: **met as a measurement, FAILED as a result** (A11). REG-01 stays **partial, pending 8.1**. | Neither leg clears the DSR hurdle | ROADMAP; REQUIREMENTS (ruling 2026-09-29) | record |
| E-04 | Relative verdicts (the sign of wealth_delta and dd_delta, and the dependence verdict) are **not** guaranteed under the CR-03 look-ahead. Absolute verdicts stand. | Look-ahead flatters each leg, but differently | 07-VERIFICATION-WAVE2 | record |
| E-05 | Superseded numbers are never overwritten: amend with old → new and a dated pointer. | Audit trail | Phase 7 D-05; 08-MEASUREMENTS §11 | SIMPLIFY (git history does most of this) |

## Process

| ID | Decision | Why | Source | Lean |
|---|---|---|---|---|
| G-01 | Human notebooks P1–P6 (one per layer), each with a human sign-off cell in P3. | Human understanding and gating | Phase 6 | KEEP. **Extend**: nothing yet covers classifier #2, the Bayes filter, the no-trade band or serving |
| G-02 | Plotting lives in `platform/plotting/` and notebooks call it; no inline plotting logic. | Reuse and testability | Phase 6 D-01/D-02 | KEEP |
| G-04 | Phase order **8.1 → 8.2**: the point-in-time re-measurement of tilt vs ablation comes before the lean MVP, because its answer shapes the MVP's design. | Glenn, 2026-09-30 | ROADMAP | record |
| G-05 | Phase 8.1 trial budget: **2 rows** (one tilt-vs-ablation run on point-in-time data), opening count 44, ceiling 46; criterion 7 not re-run. | Glenn, 2026-09-30 | 08.1-CONTEXT | record |
| G-03 | The planning volume per phase is out of proportion (see the ratio table in `REBUILD-FROM-SCRATCH-GUIDE.md` §1). | Planning had become coding | Glenn, 2026-09-29 | **CHANGE**: Phase 8.2 lean-MVP mode |
