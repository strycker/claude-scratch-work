# Trading-Crab Platform Weekly Report

## Regime View (suspended)

Regime view: suspended — the served nowcaster is input-independent (fixed in the regime rebuild); the allocation does not use it.

Scored as of 2026-08-31, the latest month observed in all 55 of the nowcaster's model columns (1 month-end behind the newest row). Newer rows lack model columns: 2026-09-30 lacks wti_fred, gold_spot, wti_crude, oil. Nothing is imputed.

1 distinct posterior vector across 233 complete months (2007-04-30 → 2026-08-31): the served model scored on every month observed in all 55 model columns, compared exactly (no rounding, no threshold). The served posterior does not depend on the features: it is the same every week.

## Crash Tripwire (advisory — changes no weight)

- Realized-vol spike (SPY): GREEN — 0.99x short/baseline EWMA vol (trips above 1.50x), as of 2026-10-02
- Credit-spread velocity (BAA-AAA): GREEN — +2 bps widening (trips at +25 bps), as of 2026-10-01
- Drawdown from peak (SPY): GREEN — -0.81% (trips at -10.00%), as of 2026-10-02

Escalation: none (3 of 3 signals current; nothing is imputed)

Run date 2026-10-05. A signal older than 5 business days is STALE and a missing input UNAVAILABLE; neither counts as green. The tripwire changes no weight on this page.

## Scoreboard (static — last budgeted run)

| Leg | TLW 1972–2020 | MDD 1972–2020 |
|---|---|---|
| Regime tilt (strategy) | 3.8863 | -26.64% |
| No-regime ablation | 4.1153 | -20.56% |
| SPY buy & hold | 4.9900 | -51.09% |
| 60/40 | 4.5098 | -29.49% |
| Faber 10-mo SMA | 4.9759 | -23.38% |

Own-span TLW (backtest_kpi_table.parquet): Regime tilt (strategy) 3.8863; No-regime ablation 4.1153; SPY buy & hold 5.7006 (from 1962-02); 60/40 5.1031 (from 1962-02); Faber 10-mo SMA 5.7740 (from 1962-02). The table above puts all five legs on the same months, so its numbers are comparable; the own-span ones are not.

Run 2026-10-04 (registry tag 08.3-monthend-tilt-vs-ablation) · window 1972-01-31 → 2020-12-31 · 588 monthly steps · 10 bps

Caveat (E-07): scoreboard returns are month-end to month-end, except oil before 1986, which uses monthly-average WTI (a bounded exception, E-08; D-02).

## Target vs. Current — Trades Implied

**Allocation mode:** no_regime

**Note:** allocation mode changed to no_regime (was regime_tilt) — this run executes the new target in full, so a large change against last week's book is expected (DECISIONS A-15).

### Target allocation

| Class | Ticker | Target % | Last week % | Change |
|---|---|---|---|---|
| equities | SPY | 34.4% | 34.4% | +0.0 pp |
| long_duration | TLT | 34.4% | 34.4% | +0.0 pp |
| gold | IAU | 19.5% | 19.5% | +0.0 pp |
| oil | USO | 11.7% | 11.7% | +0.0 pp |
| cash | FZFXX | 0.0% | 0.0% | +0.0 pp |

Target is the executed book after the no-trade band; last week is the book this report executed before this run (n/a when there was none).

Targets below are the EXECUTED book after the 5.0% no-trade band (design §5.3 bounded turnover, 08-A7.md): an asset whose target moved by no more than the band from its last executed weight keeps that weight.

_Target allocation cash residual: 0.0%_
