# Trading-Crab

## What This Is

A regime-conditional investment platform that simulates the decision process of a
hedge fund or expert trader — predicting market conditions (regimes), forecasting
asset behavior conditional on those regimes, and producing weekly portfolio guidance
(target mix, trades implied, stops, crash-risk dashboard) that Glenn executes manually
in Fidelity. Guidance-only by design: the platform recommends, the human trades.

The authoritative design is `platform_design/platform_design.md` (v1.7, 2026-07-08) —
a five-layer architecture (L0 data → L1 regime labeling → L2 regime prediction →
L3 asset prediction → L4 allocation & tactics) wrapped in an honesty framework
(walk-forward everything, purged CV, trial registry, locked 2021+ holdout, deflated
Sharpe). The build philosophy is the tracer bullet (§14): a "fully operational battle
station" — every layer present, every layer naive — usable from the first milestone,
then upgraded module by module against frozen interfaces.

## Core Value

Honest, regime-aware weekly guidance that beats buy-and-hold SPY **net of avoided
drawdowns** — and is never fooled by its own backtest. The honesty framework is not
overhead; it is the product. A beautiful but leaky backtest is worthless.

## Requirements

### Validated

<!-- Inferred from existing codebase (see .planning/codebase/). -->

- ✓ 9-step quarterly pipeline (ingest → features → cluster → regime-label → predict → asset-returns → dashboard → diagnostics → tactics) — existing; becomes the **frozen incumbent** baseline
- ✓ Config-driven feature engineering (cross-ratios, log, Bernstein gap-fill, derivatives, divergence, momentum, yield-curve) — existing
- ✓ Clustering suite (KMeans, balanced KMeans, GMM, HMM, Spectral, DBSCAN) — existing; warm starts and benchmarks for the new labeler
- ✓ Checkpoint system (parquet + manifest, preservation checkpoints) — existing
- ✓ Weekly report + SMTP email machinery — existing; reused by the new platform's report
- ✓ ~769-test pytest suite; two-package layout (`trading-crab-lib` + `trading-crab`); CI + PyPI publish workflows — existing

### Active (v1 = design §14 Phases 0–1 + migration)

- [ ] Honesty infrastructure: physical 2021+ holdout carve, trial registry, walk-forward runner, purged/embargoed CV
- [ ] Monthly data spine with 1962+ spliced core-asset histories (free sources) and ALFRED vintage alignment
- [ ] Jump-model regime labeler (default λ, K=5; k-means warm start; existing HMM as benchmark)
- [ ] Calibrated logistic nowcaster on causal features; smoothed-vs-filtered gap + detection lag reported
- [ ] Returns-by-regime tables + EWMA vol layer for core 5 + key satellites + Glenn's holdings
- [ ] Naive vol-targeted regime-tilt allocation with hysteresis; weekly report with target-vs-current mix (manual YAML holdings)
- [ ] Minimal 3-signal daily tripwire monitor
- [ ] Honest walk-forward backtest 1972–2020 vs baseline gauntlet (SPY, 60/40, Faber 10-mo SMA)
- [ ] Validated skeleton migrated to `strycker/trading-crab` (public/PyPI repo)

### Out of Scope

- Automated trade execution — guidance-only by design; human executes in Fidelity
- Decisions at horizons < 1 month — the edge thesis lives at monthly+ horizons (design D8)
- Shorts, options trading, leveraged-inverse ETFs — §16.7 resolved; long-only (SH permitted only in the future tactical sleeve)
- Cryptocurrencies and MLPs — §16.7 exclusions (K-1 avoidance)
- Deep sequence models / RL for returns or allocation — §15 declined; sample size cannot support them
- Tax/friction modeling — operator handles manually; turnover penalty retained for statistical stability only

## Context

- **Design doc is law:** `platform_design/platform_design.md` v1.7. §11 (R1–R15) maps current-code gaps; §13 verdict: *enhance trading-crab, don't restart* — keep the chassis, rebuild the modeling core; §14 phase plan; §22 (K, λ) selection protocol; §23 crash playbook; §27 stop mechanics.
- **Root `ROADMAP.md` Tier 0** (T0.1–T0.8) is the product-level backlog this GSD milestone draws from. `.planning/ROADMAP.md` (GSD) governs execution phases.
- **Codebase map:** `.planning/codebase/` (7 documents, 2026-07-09).
- **Data:** free sources only for v1 — FRED/ALFRED, Shiller, Yahoo/Stooq, LBMA gold, existing multpl/macrotrends scrapers. **Paid-provider placeholders** to note in the data layer: Norgate, Tiingo, EODHD (cleaner splices; survivorship-clean constituents for future breadth features). Glenn has **stockcharts.com and finviz.com subscriptions** (root ROADMAP items 3.6/3.7) — candidate feature sources, not v1 dependencies.
- **Known accepted compromise:** ALFRED vintages don't reach 1962; pre-vintage-era agency data gets publication-lag alignment only.
- **Repo strategy:** build + validate in `claude-scratch-work` (heavy-dev, GSD-equipped), migrate the validated skeleton to `strycker/trading-crab`, continue development there. `gsd-scratch-work` and `trading-crab-lib` submodules are frozen archives (salvage in `ideas/gsd-salvage/`, incl. `feature_gating.py` and `model_metrics_artifacts.py` to fold into the honesty layer).
- **"Old Fool Indicator"** (design §26): deferred — Glenn will extract the definition from Mike Silva's Figuring Out Money videos; homegrown version planned once defined.
- **AI routing doctrine** (root `CLAUDE.md`): Fable-led sessions, Claude-only; delegate mechanical work to Sonnet subagents; correctness-critical math stays on Fable.

## Constraints

- **Tech stack**: Python 3.10+, existing two-package src layout, config in `settings.yaml`, parquet checkpoints — extend, don't rewrite (design R15)
- **Data**: free sources only in v1; daily USD price series spliced per documented rules; ALFRED for point-in-time agency data
- **Execution venue**: Fidelity, long-only, no options/shorts/crypto/MLPs
- **Honesty discipline**: 2021+ holdout locked — development/tuning/model-selection use only ≤2020-12 walk-forward results; live weekly scoring refits on full history but its post-2021 performance is firewalled from all selection decisions until design freeze; every evaluated configuration goes in the trial registry
- **Cadence**: monthly modeling spine, weekly scoring, manual CLI runs in v1 (automation is a tracked placeholder)

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Enhance trading-crab chassis, rebuild modeling core (§13) | Infra (config/checkpoints/tests/CLI) is sound; the math needs the upgrade | — Pending |
| Jump model as primary labeler; t-HMM as benchmark (D4) | OOS label stability; single interpretable persistence knob λ | — Pending |
| Build here → migrate validated skeleton to trading-crab | Heavy-dev + GSD tooling here; clean public/PyPI repo there | — Pending |
| v1 = Phase 0 + tracer bullet + migration | Battle-station skeleton before module upgrades; platform usable early; fits token/session limits | — Pending |
| Free data sources in v1; paid-provider placeholders | Splice quality acceptable for v1; Norgate/Tiingo/EODHD deferred | — Pending |
| Holdout rule for live use | Dev/tuning ≤2020-12; live scoring refits full history, results firewalled until freeze | — Pending |
| Quarterly pipeline kept as frozen incumbent | It is the baseline the new skeleton must replace; zero new investment | — Pending |
| Minimal daily tripwire ships in v1 | Crash avoidance is the platform's reason to exist; cheap by construction | — Pending |
| Core accounts only in v1 (no sleeve reporting) | Tactical sleeve arrives with the tactics layer in a later milestone | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-07-09 after initialization*
