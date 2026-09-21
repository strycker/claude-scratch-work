# Honest Backtest Report (EVAL-01..04, design §5.4/§8.7-9/§23.1)

## Headline: Sojourn / Detection-Lag Ratio (§5.4)

This is the go/no-go number: median regime sojourn (how long a regime typically lasts) divided by the median real-time detection lag (how long the walk-forward nowcaster took to notice a transition, checked against P(its own target state) — review F1). A high ratio means most of a regime's life is capturable after detection; a ratio near 1 means the lag eats the trade.

- median sojourn (months): 9.5
- median detection lag (months): 111.0
- **ratio: 0.08558558558558559**
- sample: median lag over **15 resolved of 25 transitions** (a transition resolves only when P(target) reaches the 70% action threshold).

## Baseline Comparison: Faber 10-Month SMA (§23.1)

The Faber 10-month SMA is design §23.1's STANDING TARGET for the regime strategy to beat on both log wealth AND max drawdown — this is recorded, not a pass/fail gate (D-01a).

- strategy: terminal log wealth=4.1315, max drawdown=-28.32%
- faber_sma: terminal log wealth=6.3726, max drawdown=-18.94%

## No-Regime-Ablation Delta (Does the Regime Layer Pay Rent?)

The no-regime ablation (design §8.7) is the SAME L1-L4 code path with the regime tilt disabled (backtest/baselines.py::no_regime_ablation) — never a hand-rolled parallel implementation (D-02).

- terminal log wealth delta (strategy - ablation): +0.0269 (strategy=4.1315, ablation=4.1046)
- max drawdown delta (strategy - ablation): -7.98% (strategy=-28.32%, ablation=-20.34%)

## Feature-Policy Pre/Post Comparison (Wave 1, ADR-0001)

**Comparability caveat.** The pre-fix column below reflects BOTH the pre-fix EXPANDING driver feature-admission policy (audit item A13's original asymmetry between `driver.py::_window_active_features` and `report.py::_reference_label_columns`, frozen only by this phase's D-01) AND a stale nine-column `monthly_features` checkpoint whose `oil` column began 1985-02 instead of its full 1962-01 history (corrected by D-02-A, which is why the frozen set below has ten columns, not nine). This run therefore changed TWO things at once, and this record cannot separate how much of any movement below is the driver-freeze policy change versus the feature-space correction — attributing the whole delta to either cause alone would repeat exactly the 'fooled by its own backtest' failure mode `.planning/UAT-AUDIT-2026-09-09.md` documents.

**Sample-comparability note** (read alongside the disagreement and §5.4-ratio rows below): each cell carries its own `n_compared`/`n_resolved` denominator and date window INLINE, on purpose. When the two runs' non-degraded step counts differ, the two percentages compare different-sized, differently-dated populations, and a bare percentage-point delta between them is NOT a same-population improvement — read the denominator and the dates inside each cell before reading the percentage itself.

| Quantity | Pre-fix (superseded, 9-column, stale checkpoint) | Post-fix (frozen, 10-column policy) |
|---|---|---|
| Median regime sojourn (months) | 97.0 | 9.5 |
| Median detection lag (months) | 164.0 | 111.0 |
| §5.4 ratio (`n_resolved` of `n_transitions`) | 0.5910 (4 of 6; 1974-02 -> 2020-12) | 0.0856 (15 of 25, resolved within 1974-02 -> 2020-12) |
| Labeling disagreement (`pct_disagree`, `n_compared`) | 82.77% (389/470; 1974-02 -> 2020-12) | 85.03% (426/501; 1974-02 -> 2020-12) |
| Multiclass Brier | 0.2087 (n_steps not recorded in the pre-fix source) | 0.1892 (n_steps=501) |
| `wealth_delta` (no-regime-ablation, terminal log wealth) | +0.3793 | +0.0269 |
| `dd_delta` (no-regime-ablation, max drawdown) | -1.44% | -7.98% |
| Strategy terminal log wealth | 4.0265 | 4.1315 |
| Strategy max drawdown | -21.24% (33 mo) | -28.32% (32 mo) |
| Ablation terminal log wealth | 3.6472 | 4.1046 |
| Ablation max drawdown | -19.81% | -20.34% (32 mo) |

**Frozen L1 feature columns (all 10, the labeler's full admitted set):** `cape_shiller`, `credit_spread_baa_aaa`, `curve_10y3m`, `div_yield`, `oil`, `real_rate_level`, `realized_vol_1m`, `realized_vol_3m`, `trailing_return_1m`, `trailing_return_3m`.

**Why the Multiclass Brier and confusion tables move.** `full_sample_states` is reindexed onto the walk-forward's decision dates as `y_true` (step (e) of `run_full_backtest_evaluation`). Both the feature-space correction (D-02-A) and the driver freeze (D-01) change WHICH smoothed labeling gets reindexed, so the labels the nowcaster is scored against changed — a Brier movement here is this mechanical relabeling, not because the nowcaster improved.

## Smoothed-vs-Filtered Gap

- gap (smoothed hindsight performance - real-time filtered performance): 0.5276 — the measured hindsight content of the strategy (§5.4). The smoothed reference is ONE full-sample labeler fit; the filtered series is the walk-forward driver's actual per-step decisions — genuinely distinct series (Pitfall 1), never the same object reused.

## Baseline Gauntlet

| Leg | Terminal Log Wealth | Max Drawdown |
|-----|---------------------|--------------|
| SPY Buy & Hold | 5.6805 | -48.95% |
| 60/40 | 5.0471 | -26.96% |
| Faber 10-Month SMA | 6.3726 | -18.94% |
| Strategy (regime tilt) | 4.1315 | -28.32% |
| No-Regime Ablation | 4.1046 | -20.34% |

- no-regime-ablation delta vs. strategy: +0.0269 terminal log wealth (-7.98% max drawdown) — does the regime layer pay rent?

## Strategy KPIs

- terminal log wealth: 4.1315
- max drawdown: -28.32% (32 months underwater)
- CVaR(5%): -0.0371
- turnover (mean monthly): 0.1042
- in-sample crisis capture ratios (down-capture, A6):
  - 1973-74_oil_shock: -0.27
  - 1980-82_volcker_recession: 0.71
  - 2000-02_dotcom_bust: -0.06
  - 2008-09_gfc: 0.31

### Conventions

Cash-return convention (review F4): the strategy's cash residual and every baseline's non-invested leg (60/40's implicit reconstitution carry, Faber's out-of-market months) all earn the SAME `cash_ret` series built by `splice.build_core_research_series` -> `assets.returns.compute_monthly_returns` — so the §23.1 Faber comparison is cost-symmetric, never quietly biased by a strategy that earns 0% cash against a baseline earning the real 1970s-80s double-digit T-bill rate.
Turnover/cost convention (A5): `cost_bps` is applied identically to every rebalancing leg (strategy, 60/40, Faber) via `backtest/costs.py::apply_transaction_cost` — only SPY buy-and-hold is cost-free by construction (no rebalancing ever occurs).
