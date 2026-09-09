# Baseline — v1 Tracer Bullet, First Trustworthy Run

**Recorded:** 2026-09-09
**Run:** `python -m trading_crab_lib.platform.evaluation.report`, real data, live FRED
**Code state:** `main` @ `599614c` (after PR #140 — yield-units fix + holdout fence)
**Window:** 588 monthly steps, 1972-01 → 2020-12 (dev only; 2021+ holdout untouched)

This is the **reference point**. Every prior recorded number in the project is void —
they were produced under the percent-vs-decimal yield defect (see
`UAT-AUDIT-2026-09-09.md`). Compare all future work against this row, not against
anything in `05-VERIFICATION.md` or the phase summaries.

> **Caveat, stated up front:** a defect found *during* this audit (`UAT-AUDIT`
> §3a — ALFRED vintage index-base splice corrupting `fred_cpi` and therefore
> `real_rate_level`) was **half-fixed** on 2026-09-09. These numbers are trustworthy
> in the sense that the arithmetic is now possible; they are not a *final*
> characterization of the design.
>
> **SUPERSEDED 2026-09-09** — see the table below. The 1988 rebasing is fixed; the
> 1971 handoff survives (item **A12**) and is proven to distort regime occupancy
> (`UAT-AUDIT` Part III, Q2). Treat the rows below as the current reference and the
> original baseline as history.

## Superseding run — 2026-09-09, after the 1988 rebasing fix

| metric | original baseline | current |
|---|---|---|
| strategy terminal log wealth | 3.7204 | **3.9090** |
| max drawdown | −21.75% (62 mo underwater) | **−21.24% (33 mo)** |
| no-regime-ablation delta | +0.0732 | **+0.2618** |
| median detection lag | 76.0 mo | **61.5 mo** |
| sojourn / lag ratio | 1.25 | **1.545** |
| resolved transitions | 5 of 6 | 4 of 6 |
| multiclass Brier | 0.1816 | 0.1838 |

Faber (6.3726 / −18.94%), SPY (5.6805 / −48.95%) and 60/40 (5.0471 / −26.96%) are
unchanged — none of them touches the agency series. The strategy is **still last of
five legs on log wealth** and Faber still beats it on both §23.1 dimensions.

---

## Baseline gauntlet

| Leg | Terminal log wealth | Max drawdown |
|---|---|---|
| Faber 10-month SMA | **6.3726** | **−18.94%** |
| SPY buy & hold | 5.6805 | −48.95% |
| 60/40 | 5.0471 | −26.96% |
| **Strategy (regime tilt)** | **3.7204** | **−21.75%** |
| No-regime ablation | 3.6472 | −19.81% |

## Strategy KPIs

| Metric | Value |
|---|---|
| Terminal log wealth | 3.7204 |
| Max drawdown | −21.75% (62 months underwater) |
| CVaR(5%) | −0.0469 |
| Mean monthly turnover | 0.0765 |
| Multiclass Brier | 0.1816 over 476 steps |
| Crisis down-capture | 1973-74: 0.13 · 1980-82: 0.91 · 2000-02: 0.12 · 2008-09: 0.17 |

## Headline go/no-go (design §5.4)

| Metric | Value |
|---|---|
| Median sojourn | 95.0 months |
| Median detection lag | **76.0 months** |
| **Ratio** | **1.25** |
| Sample | 5 resolved of 6 transitions |

## Research series (sanity anchor)

| Series | Annualized return | Annualized vol |
|---|---|---|
| equities_tr | 10.87% | 12.31% |
| long_duration_tr | 5.87% | 6.67% |
| gold | 7.80% | 15.59% |
| oil | 9.78% | 37.79% |
| cash | 4.42% | 0.91% |

Cash compounds $1 (1962) → $17.24, consistent with ~4.4% over 64 years.

---

## The honest reading

**The strategy finishes last of five legs on log wealth.** Faber — design §23.1's
standing target — beats it on *both* dimensions (6.3726 vs 3.7204 wealth; −18.94% vs
−21.75% drawdown). SPY and 60/40 also beat it on wealth.

**The regime layer does not pay rent.** It adds +0.0732 terminal log wealth over the
no-regime ablation (≈0.15%/yr) while making max drawdown **1.95pp worse**
(−21.75% vs −19.81%). On a risk-adjusted basis that is arguably negative.

**The headline ratio is at its floor.** 1.25, against a design bar of ~5. A median
detection lag of 76 months means the nowcaster takes over six years to reach 70%
confidence in a transition. §5.4's own words: a ratio near 1 means the lag eats the
trade.

**The labeler is finding epochs, not regimes.** Six transitions in 59 years, median
sojourn 95 months. Whatever this is detecting, it is closer to secular eras than to
anything tradeable at a monthly cadence.

**What genuinely works:** crisis down-capture of 0.13 / 0.12 / 0.17 in three of four
crises — the strategy really is defensive when it matters. And vol targeting is *not*
the drag (mean scale 0.93, median 1.00 — it is essentially fully invested, so the
underperformance is asset-mix and timing, not under-leverage).

This is the expected shape of a §14 tracer bullet: every layer present, every layer
naive, beats nothing yet. The value of this record is that it is the first version of
that statement backed by arithmetic that could actually occur.

---

## Known caveats on this baseline

1. **`real_rate_level` is corrupted** by the ALFRED index-base defect
   (`UAT-AUDIT-2026-09-09.md` §3a) and is a defining feature of 64.3% of regime
   occupancy. Fixing it may move these numbers materially.
2. **19% of steps degraded** (112 of 588 L2 refits held previous weights). Not yet
   investigated.
3. **Design §5.3 hysteresis gates nothing.** `active_regime` is computed, threaded and
   recorded in both the backtest and the live weekly path, but weights come from
   `vol_targeted_tilt(regime_probs, …)` either way. Defensible if probability-weighted
   tilting is meant to replace hard switching — but it means the act/unwind thresholds
   are reporting state, not control, and that should be confirmed as intended.
4. **Gold's history is IAU-backed (2005+)** when macrotrends is unreachable, not spot
   gold back to 1915 — see the splice provenance note.
