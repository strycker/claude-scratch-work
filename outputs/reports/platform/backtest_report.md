# Honest Backtest Report (EVAL-01..04, design §5.4/§8.7-9/§23.1)

## Headline: Sojourn / Detection-Lag Ratio (§5.4)

This is the go/no-go number: median regime sojourn (how long a regime typically lasts) divided by the median real-time detection lag (how long the walk-forward nowcaster took to notice a transition, checked against P(its own target state) — review F1). A high ratio means most of a regime's life is capturable after detection; a ratio near 1 means the lag eats the trade.

- median sojourn (months): 84.5
- median detection lag (months): 161.5
- **ratio: 0.5232198142414861**
- sample: median lag over **2 resolved of 7 transitions** (a transition resolves only when P(target) reaches the 70% action threshold).
- ⚠ **Small sample:** the ratio is a median over very few resolved transitions — treat it as indicative, not a robust go/no-go number. The strategy KPIs and the multiclass Brier are the more reliable signals at this stage.

## Baseline Comparison: Faber 10-Month SMA (§23.1)

The Faber 10-month SMA is design §23.1's STANDING TARGET for the regime strategy to beat on both log wealth AND max drawdown — this is recorded, not a pass/fail gate (D-01a).

- strategy: terminal log wealth=111.0627, max drawdown=-66.25%
- faber_sma: terminal log wealth=1.1391, max drawdown=-99.69%

## No-Regime-Ablation Delta (Does the Regime Layer Pay Rent?)

The no-regime ablation (design §8.7) is the SAME L1-L4 code path with the regime tilt disabled (backtest/baselines.py::no_regime_ablation) — never a hand-rolled parallel implementation (D-02).

- terminal log wealth delta (strategy - ablation): -2.8299 (strategy=111.0627, ablation=113.8927)
- max drawdown delta (strategy - ablation): +1.83% (strategy=-66.25%, ablation=-68.08%)

## Smoothed-vs-Filtered Gap

- gap (smoothed hindsight performance - real-time filtered performance): -14.2217 — the measured hindsight content of the strategy (§5.4). The smoothed reference is ONE full-sample labeler fit; the filtered series is the walk-forward driver's actual per-step decisions — genuinely distinct series (Pitfall 1), never the same object reused.

## Baseline Gauntlet

| Leg | Terminal Log Wealth | Max Drawdown |
|-----|---------------------|--------------|
| SPY Buy & Hold | 5.6805 | -48.95% |
| 60/40 | 131.0293 | -2.27% |
| Faber 10-Month SMA | 1.1391 | -99.69% |
| Strategy (regime tilt) | 111.0627 | -66.25% |
| No-Regime Ablation | 113.8927 | -68.08% |

- no-regime-ablation delta vs. strategy: -2.8299 terminal log wealth (+1.83% max drawdown) — does the regime layer pay rent?

## Strategy KPIs

- terminal log wealth: 111.0627
- max drawdown: -66.25% (12 months underwater)
- CVaR(5%): -0.1890
- turnover (mean monthly): 0.0279
- in-sample crisis capture ratios (down-capture, A6):
  - 1973-74_oil_shock: -2334.68
  - 1980-82_volcker_recession: 2216.46
  - 2000-02_dotcom_bust: -1396.66
  - 2008-09_gfc: -22.84

### Conventions

Cash-return convention (review F4): the strategy's cash residual and every baseline's non-invested leg (60/40's implicit reconstitution carry, Faber's out-of-market months) all earn the SAME `cash_ret` series built by `splice.build_core_research_series` -> `assets.returns.compute_monthly_returns` — so the §23.1 Faber comparison is cost-symmetric, never quietly biased by a strategy that earns 0% cash against a baseline earning the real 1970s-80s double-digit T-bill rate.
Turnover/cost convention (A5): `cost_bps` is applied identically to every rebalancing leg (strategy, 60/40, Faber) via `backtest/costs.py::apply_transaction_cost` — only SPY buy-and-hold is cost-free by construction (no rebalancing ever occurs).
**⚠ Excluded assets:** the following research classes were EXCLUDED from this run because their source data was unavailable (e.g. gold when macrotrends is IP-blocked): `gold`. The backtest ran on the remaining assets only — treat cross-run comparisons that include these assets as not directly comparable.
