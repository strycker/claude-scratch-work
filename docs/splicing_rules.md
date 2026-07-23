# Splicing Rules — Core Research Series (DATA-02 / D-04)

This document records, per D-04, the source(s), date range, join date, and
splice/synthesis method for each of the five core asset-class research series
built by `src/trading_crab_lib/platform/splice.py::build_core_research_series()`.
It is the DATA-02 documentation deliverable: every splice rule must be
reproducible from this file plus `config/platform_settings.yaml`'s `splice:`
block — never inferred, never computed dynamically.

**Target span:** all five series target `data.start_date: 1962-01-01` in
`config/platform_settings.yaml` (D-08). Per-class actual coverage start is
noted below where the underlying free source begins later than 1962; in that
case the series is simply shorter, not backfilled or fabricated.

**Per D-03 ("model the index, trade the ETF"):** each class produces ONE
long-history *research* series used by all models; the weekly report maps
each class to Glenn's actual tradable ticker separately. The research series
and the tradable ETF are never the same instrument pre-ETF-inception.

**Join-date discipline (RESEARCH Pitfall 2):** any splice that stitches two
segments together (`ratio_splice()`) requires a literal `join_date` read from
config or this document — never a value computed at runtime from data
availability. `ratio_splice()` raises `ValueError` if `join_date` is absent
from either input series, precisely to prevent a silent, undocumented seam
from moving between runs.

---

## 1. Equities — `equities_tr` (research) → SPY (tradable)

**Context.** D-03 maps US equities to "S&P 500 total return" as the research
series, tracked against SPY (1993+) as the tradable proxy.

**Decision.** Build total return directly from the incumbent's already-scraped
multpl.com S&P 500 price series (`sp500`) and dividend yield series
(`div_yield`), rather than adding a new Shiller `ie_data.xls`/shillerdata.com
fetcher. `build_equity_total_return()` computes
`monthly_return = price.pct_change() + div_yield / 12`, chained into a
cumulative index starting at 1.0.

- **Source:** `multpl.com` (already scraped by `ingestion/multpl.py`, D-01 —
  not re-fetched or modified by this plan).
- **Date range:** multpl's S&P 500 and dividend-yield tables cover back to
  ~1871 (well past the 1962 target span).
- **Join date:** none — single continuous source, no `ratio_splice` needed.
- **Method:** `total_return_from_price_div` — price return + (annual dividend
  yield / 12), monthly, chained multiplicatively.

**Rationale.** RESEARCH's primary recommendation was the Shiller total-return
column (pre-built TR, no researcher-side math needed) with multpl as a
cross-check only. This was a **[ASSUMED — MEDIUM confidence]** recommendation
(RESEARCH A3): no live fetch of `shillerdata.com`'s actual file structure was
performed during research, and the file's URL has moved historically between
`econ.yale.edu/~shiller` and `shillerdata.com`. Implementing against an
unverified schema risks building on a broken assumption. multpl's price +
dividend-yield series are already integrated, already tested, and require no
new ingestion code (D-01 constraint: `platform/` imports but never modifies
incumbent fetchers) — so this is the Claude's-Discretion source choice for
this plan.

**Tradeoff.** multpl-derived TR is a *constructed* total return (price return
+ yield/12), not a source-native TR column — it will diverge slightly from a
"true" TR index that accounts for exact ex-dividend timing and reinvestment
mechanics. **Upgrade path:** swap in a verified Shiller TR column as a
drop-in replacement for `build_equity_total_return()`'s output once the
shillerdata.com file schema is confirmed against a live fetch; the dispatch
in `build_core_research_series()` would only need a new `method` branch, not
a redesign.

---

## 2. Long duration — `long_duration_tr` (research) → TLT (tradable)

**Context.** D-03 maps long-duration Treasuries to a "Treasury total-return
synthetic from CMT yields (GS10)" as the research series, tracked against TLT
(2002+) as the tradable proxy.

**Decision.** No free source ships a pre-built long-Treasury total-return
index back to 1962, so one is synthesized from FRED's `GS10` constant-maturity
10-year yield series via **par-bond repricing**: each month, assume a
freshly-issued par bond at the prevailing CMT yield; accrue that month's
coupon income; then reprice the (now one-month-seasoned) bond at the *new*
month's yield using standard bond present-value math. Total return for the
month is `(price_t1 - price_t0 + coupon_accrued) / price_t0`.

- **Source:** FRED `GS10` (already available to the incumbent's FRED
  fetcher pattern; `fred_gs10` in `raw`).
- **Date range:** GS10 covers 1953+, comfortably spanning the 1962 target.
- **Join date:** none — single continuous source, no `ratio_splice` needed.
- **Method:** `cmt_par_bond_repricing` — `bond_price()` +
  `monthly_total_return()` chained by `build_treasury_tr_synthetic()`.
  **Compounding convention:** semiannual (`coupon_freq: 2` in
  `config/platform_settings.yaml`'s `splice.long_duration` block), matching
  standard US Treasury coupon-payment conventions — this is the convention
  asserted by `bond_price(0.05, 0.05, 10)` pricing to ~1.0 (par) and by the
  rising/falling-yield sign tests in `tests/unit/test_platform_splice.py`.
  Maturity is pinned at `maturity_years: 10` (matching GS10's own tenor).

**Rationale.** This is the "rolling par bond" / "constant maturity total
return" approximation described in RESEARCH Pattern 3, cited to
portfoliooptimizer.io's summary of the standard fixed-income PV-math method
for simulating constant-maturity government bond ETF returns. It is a
**[ASSUMED — MEDIUM confidence]** methodology choice (RESEARCH A4): the
standard-approximation framing is well established, but this implementation
has not been cross-checked against a published index methodology (e.g.
Bloomberg/ICE long-Treasury index docs) — flagged here explicitly per D-04
rather than silently baked into code.

**Tradeoff — GS10 (~10Y) vs TLT (~20Y+) duration mismatch (RESEARCH Open
Question 2, resolved).** GS10 is a 10-year constant-maturity yield; TLT
tracks Treasuries with 20+ years remaining maturity. The research series and
the tradable ETF therefore have materially different durations and will
diverge in volatility magnitude and sensitivity to yield-curve moves — a
100bp yield move produces a much larger price swing in a 20-year bond than a
10-year one. **This mismatch is accepted, not fixed**, per the locked D-03
mapping (GS10 is the specified source) and the "model the index, trade the
ETF" framing (D-03), which already accepts proxy imperfection as a design
principle. A future reader comparing `long_duration_tr` history against live
TLT returns should expect them to diverge in scale, not treat that as a bug.
**Upgrade path (not built):** blend GS10 with GS20/GS30 where available for a
closer duration match, if a future phase needs tighter tracking.

---

## 3. Gold — `gold` (research) → IAU (tradable)

**Context.** D-03 maps gold to "macrotrends monthly (1915+) / LBMA" as the
research series, tracked against **IAU** (2005+; Glenn holds IAU, not GLD —
D-03, D-10).

**Decision.** Single-source passthrough of the incumbent's already-scraped
macrotrends monthly gold spot series (`gold_spot`) — no splice needed.

- **Source:** `macrotrends.net` (already scraped by
  `ingestion/macrotrends.py`, D-01 — reused as-is, zero new ingestion code).
- **Date range:** macrotrends gold spot covers 1915+, comfortably spanning
  the 1962 target.
- **Join date:** none — single continuous source, no `ratio_splice` needed.
- **Method:** `single_source` — `raw[source_col].dropna()`, renamed to the
  `research_name`.

**Rationale.** RESEARCH identified macrotrends gold as already-scraped and
verified in `ingestion/macrotrends.py`'s `DEFAULT_SERIES` — using it directly
is a "zero new ingestion code" win (D-01: `platform/` imports but never
modifies incumbent fetchers) with no accuracy tradeoff, since it is the
primary free source with the longest history for this asset class.

**Tradeoff.** None identified — macrotrends already exceeds the 1962 target
span with a single continuous series (RESEARCH's Open Question 3 on
multi-segment splices for gold — e.g. macrotrends → LBMA → IAU-implied spot
as three segments — was explicitly deferred: default to the simplest source
that meets the target span, only add a segment if a coverage gap is found;
none was found for gold).

---

## 4. Oil — `oil` (research) → oil proxy ticker (USO or class-level)

**Context.** D-03 maps oil to "WTI spot (macrotrends 1946+ / FRED)" as the
research series, tracked against USO (2006+) or scored at the class level
(D-03 leaves the exact tradable mapping open).

**Decision.** Single-source passthrough of the incumbent's already-scraped
macrotrends monthly WTI crude series (`wti_crude`), with FRED's `WTISPLC`
(`wti_fred` in `raw`) recorded in config as a `cross_check_col` — a
documented validation cross-check, not a splice input.

- **Source:** `macrotrends.net` (primary; already scraped, D-01 reuse).
- **Cross-check source:** FRED `WTISPLC` (1946+) — a single extra FRED
  series call, used to validate macrotrends' scrape independently, per
  RESEARCH's recommendation. Not blended or spliced into the research
  series; `wti_fred` is available in `raw` for out-of-band comparison only.
- **Date range:** macrotrends WTI covers 1946+, comfortably spanning the
  1962 target.
- **Join date:** none — single continuous primary source, no `ratio_splice`
  needed.
- **Method:** `single_source` — `raw[source_col].dropna()`, renamed to the
  `research_name`. (`cross_check_col` is read from config but not consumed
  by `build_core_research_series()`'s dispatch — it documents intent for a
  future validation script, not live logic in this plan.)

**Rationale.** Same reasoning as gold: macrotrends is already integrated and
already exceeds the target span, so it is the primary source with zero new
ingestion code. FRED WTISPLC is cheap to add as a config-documented
cross-check (a single extra FRED series) without complicating the splice
logic itself.

**Tradeoff.** None identified for the primary series. The cross-check column
is documented but not yet wired into an automated divergence check — a
future phase could add one using the same `divergence.py` pattern the
incumbent pipeline already uses for other cross-asset signals.

---

## 5. Cash — `cash` (research) → FZFXX/SPAXX (Fidelity money markets, tradable)

**Context.** D-03/D-12 map cash to "3-month T-bill yield (TB3MS) as return
series", tracked against Fidelity money-market funds FZFXX/SPAXX — which
have NAV ≈ $1 and no meaningful price history to splice (D-12).

**Decision.** Use FRED `TB3MS` directly **as a return series** — no price
history, no splice, no synthesis. This was already specified by D-03/D-08;
this plan implements it without deviation.

- **Source:** FRED `TB3MS` (3-month T-bill secondary market rate;
  `fred_tb3ms` in `raw`).
- **Date range:** TB3MS covers 1934+, comfortably spanning the 1962 target.
- **Join date:** none — single continuous source, no `ratio_splice` needed.
- **Method:** `yield_as_return` — `raw[yield_col].dropna()`, renamed to the
  `research_name`. The yield value itself is used as the periodic return
  (no price-level construction, unlike the long-duration synthetic) because
  a money-market fund's NAV is structurally pinned near $1 and its yield
  *is* its return.

**Rationale.** Money-market NAV stability makes any price-based splice
meaningless — the yield-as-return treatment is both the simplest and the
most accurate representation available, and was already locked in D-03/D-12
rather than left to discretion.

**Tradeoff.** None — this is the correct treatment for a NAV-stable
instrument, not a simplification.

---

## Summary table

| Class | Research series | Source(s) | Coverage start | Join date | Method | Tradable |
|---|---|---|---|---|---|---|
| Equities | `equities_tr` | multpl.com (price + div yield) | ~1871 | none (single source) | `total_return_from_price_div` | SPY |
| Long duration | `long_duration_tr` | FRED `GS10` | 1953 | none (single source) | `cmt_par_bond_repricing` (par-bond repricing, semiannual) | TLT |
| Gold | `gold` | macrotrends | 1915 | none (single source) | `single_source` | IAU |
| Oil | `oil` | macrotrends (primary), FRED `WTISPLC` (cross-check) | 1946 | none (single source) | `single_source` | USO / class-level |
| Cash | `cash` | FRED `TB3MS` | 1934 | none (single source) | `yield_as_return` | FZFXX / SPAXX |

All five sources exceed the `data.start_date: 1962-01-01` target span with a
single continuous free source — none of the five core classes required a
2-segment `ratio_splice()` in this implementation (RESEARCH Open Question 3,
resolved: default to the simplest source per D-03's literal text, add a
segment only if a coverage gap is found; none was found). `ratio_splice()` is
implemented and tested (`tests/unit/test_platform_splice.py`) and available
for any future class or source substitution that does need to stitch two
segments at a literal, documented join date.

---

## 6. Phase 4 additions — SPY dual-purpose, DAAA/DBAA credit source, Fidelity-CSV seam

**SPY is now dual-purpose (Phase 4, RESEARCH Pitfall 1).** SPY was added to
`config/platform_settings.yaml`'s `universe.satellites`, so it is now
ingested as a daily universe ticker via
`platform/ingestion/prices_daily.py::fetch_universe_prices()` into the
`daily_raw` checkpoint. This is in addition to — not a replacement for —
SPY's existing role as the documented **tradable proxy** for the `equities`
research class (`splice.equities.tradable: SPY`, §1 above). The monthly
`equities_tr` splice (multpl-derived total return) is unchanged and remains
the modeling series used by every model; SPY's daily closes exist solely to
feed the Phase 4 tripwire's drawdown-from-peak signal (L4-04, design §23.2)
— they are never used as a substitute for `equities_tr` in any model input.

**DAAA/DBAA — daily credit-spread source (Phase 4, RESEARCH Pitfall 2).**
`fred_monthly`'s `BAA`/`AAA` series (Moody's Seasoned Corporate Bond Yields)
are monthly-frequency only on FRED. The tripwire's credit-spread-velocity
signal needs genuinely daily data, so `platform/ingestion/macro_daily.py`
fetches FRED's daily counterparts `DAAA`/`DBAA` at native frequency (no
resample) and persists them as the `fred_daily_raw` platform checkpoint.
This is a new, separate daily credit source — it does not replace or alter
the monthly `fred_baa`/`fred_aaa` series used elsewhere.

**Fidelity positions-CSV parser — documentation-only placeholder seam
(v2, L4-V2-05).** Holdings in v1 come exclusively from manual per-account
YAML files (`config/accounts/<account>.yaml`, D-01) — ticker → weight
fraction, validated with tolerance against a ~1.0 sum (weights + cash).
No CSV parsing of Fidelity's exported positions report is implemented in
v1; this is deliberately deferred to a future milestone (L4-V2-05). If
implemented later, a CSV-parsing loader would live alongside
`platform/report/holdings.py` and produce the same `{"weights": {...},
"cash": frac}` shape the YAML loader already returns, so downstream
report code would not need to change.
