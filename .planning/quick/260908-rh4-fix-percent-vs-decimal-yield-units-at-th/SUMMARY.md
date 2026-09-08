---
id: 260908-rh4
slug: fix-percent-vs-decimal-yield-units-at-th
date: 2026-09-08
status: complete
---

# Summary

Every KPI the platform produced was built on a 100x units error. The report ran,
wrote all its artifacts, registered its trials, and reported a strategy with a
**mean monthly return of +21.9%**.

## Root cause

Raw columns keep their source's native units, and those units differ by source:
FRED's `fred_gs10` is **percent** (4.68), multpl's `div_yield` is a **decimal
fraction** (0.011). `bond_price()` / `monthly_total_return()` are decimal-domain
functions — `pv_face = 1/(1+r)**n` only prices to par in decimals — but
`build_treasury_tr_synthetic()` fed them the percent series unconverted.

`coupon_accrued = yield_t0 / 12` with `yield_t0 = 4.68` accrues a **39% monthly
coupon**. `long_duration_tr` compounded to **2.3e128**.

Separately, `yield_as_return` returned the yield series *as the research series*.
Every consumer runs those through `compute_monthly_returns()` == `pct_change()`,
so "cash return" was the month-over-month **change in the yield**: a T-bill going
0.5% -> 3.0% booked a **+500% month**.

`build_equity_total_return` uses the same `/12` pattern and was **correct** —
because its input is already decimal. That asymmetry is the tell.

## Changes

- `yield_to_decimal()` + `assert_yield_units_plausible()` in `splice.py`. The
  guard is deliberately asymmetric: a converted yield above 100% annualized
  **raises** (no market produces that — it is provably a units error), while a
  suspiciously low median only **warns** (double conversion and a genuine ZIRP
  window are indistinguishable, and failing would reject real 2009-2021 T-bill
  data).
- `yield_units` is per-class config (`percent` for both FRED classes), not a
  blanket rule — precisely because `div_yield` must NOT be converted.
- `build_cash_index()` replaces the raw passthrough: compounds the accrual into
  an index level so `pct_change()` recovers it. Accrual is causal — month `t`
  earns the yield observed at `t-1`, matching `monthly_total_return`.
- `bond_price` docstring now states its decimal domain explicitly.

## Validated against real data

| series | annualized return | annualized vol |
|---|---|---|
| equities_tr | 10.87% | 12.31% |
| long_duration_tr | 5.87% | 6.67% |
| gold | 7.80% | 15.59% |
| oil | 9.78% | 37.79% |
| cash | 4.42% | 0.91% |

Cash compounds $1 (1962) -> $17.24, consistent with ~4.4% over 64 years.

## Test bugs fixed (this is why 1360 tests passed over a broken product)

1. **Every platform fixture used decimal yields** (`fred_gs10: 0.04`) while
   production is percent. `test_mini_pipeline.py` used `uniform(1.0, 16.0)` —
   percent. The suite disagreed with itself about units and no test asserted on
   magnitude, so the defect had nowhere to surface. Fixtures now use production
   units.
2. **`test_chain_works_on_non_single_source_key` asserted the bug**
   (`result["cash"] == [0.02] * 12` — cash IS the raw yield). Rewritten to
   assert an index level.
3. **The flagship integration test exercised none of its own logic.**
   `test_mini_backtest.py` omitted `feature_min_history`, taking the driver's
   production default of **120 months** inside a **48-month** window. No feature
   could ever qualify, so `_window_active_features` returned nothing, L1 got a
   zero-column frame, and **all 24 steps degraded** to "hold previous weights"
   with `n_steps: 0`. It verified plumbing and artifact shapes while running
   zero labeling, prediction, or allocation. Now 16 of 24 steps train.

## Verification

- 12 new tests. `test_percent_yields_produce_a_plausible_treasury_index` gives
  **4.5e62 / 12163% annualized** against pre-change code.
- `TestKpisAreOnAPlausibleScale::test_strategy_monthly_returns_are_a_plausible_magnitude`
  fails end-to-end against pre-change code — but **only after** fix 3 above; on
  the fully-degraded test it passed, which is exactly the false comfort worth
  recording.
- Suite **1372 passed** (was 1360). ruff clean.
