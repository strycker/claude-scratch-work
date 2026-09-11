# Honest Backtest Report (EVAL-01..04, design §5.4/§8.7-9/§23.1)

## Headline: Sojourn / Detection-Lag Ratio (§5.4)

This is the go/no-go number: median regime sojourn (how long a regime typically lasts) divided by the median real-time detection lag (how long the walk-forward nowcaster took to notice a transition, checked against P(its own target state) — review F1). A high ratio means most of a regime's life is capturable after detection; a ratio near 1 means the lag eats the trade.

- median sojourn (months): 97.0
- median detection lag (months): 402.0
- **ratio: 0.24129353233830847**
- sample: median lag over **3 resolved of 6 transitions** (a transition resolves only when P(target) reaches the 70% action threshold).
- ⚠ **Small sample:** the ratio is a median over very few resolved transitions — treat it as indicative, not a robust go/no-go number. The strategy KPIs and the multiclass Brier are the more reliable signals at this stage.

## Baseline Comparison: Faber 10-Month SMA (§23.1)

The Faber 10-month SMA is design §23.1's STANDING TARGET for the regime strategy to beat on both log wealth AND max drawdown — this is recorded, not a pass/fail gate (D-01a).

- strategy: terminal log wealth=3.9194, max drawdown=-20.99%
- faber_sma: terminal log wealth=6.3726, max drawdown=-18.94%

## No-Regime-Ablation Delta (Does the Regime Layer Pay Rent?)

The no-regime ablation (design §8.7) is the SAME L1-L4 code path with the regime tilt disabled (backtest/baselines.py::no_regime_ablation) — never a hand-rolled parallel implementation (D-02).

- terminal log wealth delta (strategy - ablation): +0.2722 (strategy=3.9194, ablation=3.6472)
- max drawdown delta (strategy - ablation): -1.18% (strategy=-20.99%, ablation=-19.81%)

## Smoothed-vs-Filtered Gap

- gap (smoothed hindsight performance - real-time filtered performance): -0.4983 — the measured hindsight content of the strategy (§5.4). The smoothed reference is ONE full-sample labeler fit; the filtered series is the walk-forward driver's actual per-step decisions — genuinely distinct series (Pitfall 1), never the same object reused.

## Baseline Gauntlet

| Leg | Terminal Log Wealth | Max Drawdown |
|-----|---------------------|--------------|
| SPY Buy & Hold | 5.6805 | -48.95% |
| 60/40 | 5.0471 | -26.96% |
| Faber 10-Month SMA | 6.3726 | -18.94% |
| Strategy (regime tilt) | 3.9194 | -20.99% |
| No-Regime Ablation | 3.6472 | -19.81% |

- no-regime-ablation delta vs. strategy: +0.2722 terminal log wealth (-1.18% max drawdown) — does the regime layer pay rent?

## Strategy KPIs

- terminal log wealth: 3.9194
- max drawdown: -20.99% (39 months underwater)
- CVaR(5%): -0.0448
- turnover (mean monthly): 0.0792
- in-sample crisis capture ratios (down-capture, A6):
  - 1973-74_oil_shock: 0.13
  - 1980-82_volcker_recession: 0.81
  - 2000-02_dotcom_bust: 0.05
  - 2008-09_gfc: 0.10

### Conventions

Cash-return convention (review F4): the strategy's cash residual and every baseline's non-invested leg (60/40's implicit reconstitution carry, Faber's out-of-market months) all earn the SAME `cash_ret` series built by `splice.build_core_research_series` -> `assets.returns.compute_monthly_returns` — so the §23.1 Faber comparison is cost-symmetric, never quietly biased by a strategy that earns 0% cash against a baseline earning the real 1970s-80s double-digit T-bill rate.
Turnover/cost convention (A5): `cost_bps` is applied identically to every rebalancing leg (strategy, 60/40, Faber) via `backtest/costs.py::apply_transaction_cost` — only SPY buy-and-hold is cost-free by construction (no rebalancing ever occurs).
