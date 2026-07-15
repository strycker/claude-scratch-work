# Phase 1: Monthly Data Layer & Long Histories - Research

**Researched:** 2026-07-10
**Domain:** Financial time-series ingestion, splicing, point-in-time (ALFRED) vintage
data, monthly feature-frequency migration
**Confidence:** MEDIUM (existing codebase patterns HIGH; splicing/ALFRED specifics
MEDIUM/LOW — several claims are training-knowledge and need spot-checking against
live FRED/ALFRED responses during implementation)

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Architecture (where the monthly spine lives)**
- **D-01:** New parallel subpackage (e.g. `src/trading_crab_lib/platform/`) with its own
  ingestion/splicing modules and its own checkpoint namespace (`monthly_*` or a separate
  checkpoint subdir). The quarterly incumbent (9-step pipeline, 769 tests) is FROZEN and
  must not be modified. Existing fetchers (fred.py, multpl.py, macrotrends.py, assets.py)
  may be imported/reused from the new package but not edited beyond what import-reuse
  strictly requires.
- **D-02:** The new subpackage is the Phase 6 migration unit — keep its internal imports
  self-contained so it lifts cleanly into `strycker/trading-crab`.

**Splicing policy**
- **D-03:** "Model the index, trade the ETF." Each asset CLASS gets ONE spliced research
  series used by all models; the weekly report maps each class to Glenn's tradable ticker.
  - US equities: S&P 500 total return (Shiller/multpl history → modern index) → SPY
  - Long duration: Treasury total-return synthetic from CMT yields (GS10) → TLT
  - Gold: macrotrends monthly (1915+) / LBMA → **IAU** (Glenn holds IAU, not GLD)
  - Oil: WTI spot (macrotrends 1946+ / FRED) → oil proxy ticker (USO or leave class-level)
  - Cash: 3-month T-bill yield (TB3MS) as return series → **FZFXX/SPAXX**
- **D-04:** Splice joins: ratio-splice at overlap windows (scale older segment to match
  the newer at the seam); every splice rule documented per asset (source, date range,
  join date, method) in a docs file or config block — a DATA-02 deliverable.
- **D-05:** Store daily raw prices where sources provide daily; derive the monthly spine
  by resampling. Daily persistence is deliberate — the Phase 4 tripwire consumes daily data.

**ALFRED scope**
- **D-06:** True point-in-time vintages ONLY for revision-heavy agency series used in
  labeling/features: GDP, CPI (CPIAUCSL), UNRATE, INDPRO, PAYEMS (+GNP if kept). All
  other agency-ish series keep publication-lag shift. Market-observed series (rates,
  spreads, prices) never need vintages. Pre-vintage-era fallback = publication-lag
  alignment, documented as an accepted compromise (ALFRED archives mostly start in the
  1990s).
- **D-07:** ALFRED uses the same `FRED_API_KEY` (already in env) — no new credentials.

**Universe (tickers)**
- **D-08:** Core 5 classes (equities, long duration, gold, oil, cash) via spliced
  research series per D-03.
- **D-09:** Key satellites (NULL-tolerant, shorter histories OK): QQQ, IWM, EFA, IEF,
  SHY, VNQ, VYM, AGG, HYG, LQD, EEM, DBA, MCD, COST, O, TSM, IAU, SLV, GDX.
- **D-10:** Glenn's current holdings (must always be scored): IAU, SLV, GDX, DBA,
  FZFXX, SPAXX.
- **D-11:** Watchlist (not held, score from day one): UNG, UEC.
- **D-12:** FZFXX/SPAXX are Fidelity money-market funds — no price ingestion (NAV ≈ $1);
  map to the cash class whose return series is the 3-month T-bill yield.
- **D-13:** Single names (MCD, COST, O, TSM, UEC) ingest like ETFs (daily adjusted close
  via yfinance chain); no fundamental data in v1.

### Claude's Discretion
- Exact subpackage name and module layout (planner decides; follow existing conventions).
- Checkpoint naming scheme for the monthly namespace.
- Which free source wins per splice segment when several qualify (researcher investigates
  Shiller vs multpl vs Stooq coverage/quality and recommends — see findings below).
- Treasury TR synthetic construction method (standard CMT-yield-to-total-return
  approximation; researcher picks the formula and documents it — see findings below).

### Deferred Ideas (OUT OF SCOPE)
- Fidelity positions-CSV parser (L4-V2-05) — Phase 4 uses manual YAML.
- Survivorship-clean constituent data for breadth features (DATA-V2-01) — needs paid provider.
- stockcharts/finviz feature ingestion (DATA-V2-02).
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| DATA-01 | Monthly data spine — ingestion and transforms produce monthly features; quarterly agency series aligned with publication lags | See "Standard Stack", "Architecture Patterns" — monthly resample (`ME` not `QE`), `fred.py` shift pattern reused, quarterly agency series forward-filled or explicitly flagged low-frequency in taxonomy |
| DATA-02 | Spliced USD histories back to ~1962 for core assets, free sources, splicing rules documented per asset | See "Splicing Sources & Recommendation", "Code Examples" — Shiller vs multpl vs Stooq comparison, ratio-splice formula, per-asset source table |
| DATA-03 | ALFRED point-in-time vintages for agency series, documented fallback pre-vintage era | See "ALFRED Point-in-Time Implementation" — `fredapi.get_series_all_releases()` mechanics, vintage coverage start dates, fallback logic |
| DATA-04 | Features classified fast/slow/agency taxonomy in config; lean full-history (1962+) feature set defined | See "Feature Taxonomy Config Design" |
| DATA-05 | Key satellite ETFs and Glenn's holdings ingested with NULL-tolerant handling for shorter histories | See "Don't Hand-Roll" (reuse `assets.py` fallback chain), "Common Pitfalls" (NULL-tolerant merge) |
| DATA-06 | Paid-provider adapter seams documented (Norgate, Tiingo, EODHD) — placeholder notes only | See "Paid-Provider Adapter Seam Notes" |
</phase_requirements>

## Summary

This phase is a pure **data-engineering** phase: no modeling, no clustering, no ML. The
job is to build a second, parallel ingestion+transform subpackage that (1) resamples to
**monthly** instead of quarterly, (2) splices five core-asset research series back to
~1962 from free sources, (3) adds ALFRED point-in-time vintage pulls for five revision-
heavy agency series, (4) classifies every feature as fast/slow/agency in config, and (5)
ingests satellites/holdings with graceful NULL tolerance. Every one of these five moves
has a close analog already in the codebase — `fred.py`'s per-series `shift` flag becomes
per-series `vintage: true`; `assets.py`'s fallback chain is reused verbatim for satellites;
`macrotrends.py` already returns monthly gold (1915+) and WTI (1946+), which is a free
lunch for two of the five splice segments.

The two genuinely new pieces of domain knowledge this phase needs are (a) how `fredapi`
exposes ALFRED vintages (`get_series_all_releases()` — one call per series, not one call
per historical date), and (b) how to turn constant-maturity Treasury *yields* into a
*total-return* series (the "par-bond repricing" method: hold a fictitious par bond at the
prevailing CMT yield, accrue one month of coupon, then reprice it at next month's yield
using bond present-value math to capture the price/duration effect). Both are documented
below with code-shaped pseudocode.

**Primary recommendation:** Build `src/trading_crab_lib/platform/` as a self-contained
subpackage (own `ingestion/`, `splice.py`, `taxonomy.py`, own checkpoint namespace via a
second `CheckpointManager(base_dir=...)` instance — the existing class already takes a
directory, no need to fork it) that *imports but never edits* the four existing fetchers,
adds one new fetcher module for ALFRED vintages, and stores its own `platform_settings.yaml`
(not `settings.yaml`) so the frozen incumbent's `validate_config()` schema is never touched.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| FRED/ALFRED fetch (agency + market series) | Ingestion (library) | — | Network I/O, no business logic; mirrors existing `ingestion/fred.py` |
| multpl/macrotrends scrape reuse | Ingestion (library) | — | Existing modules imported as-is (D-01) |
| ETF/single-name/satellite price fetch | Ingestion (library) | — | Existing `assets.py` fallback chain reused verbatim |
| Splicing (index→ETF research series) | Transform (library) | — | Pure pandas math on already-fetched series; new `splice.py` module |
| Feature taxonomy classification | Config | Transform (library) | Declarative data in YAML; consumed by transform code, not computed |
| Monthly resample + publication-lag alignment | Transform (library) | — | Same pattern as `fred.py`'s `.resample("QE")` + `.shift()`, generalized to `"ME"` |
| Checkpoint persistence (monthly namespace) | Storage (library) | — | Reuse `CheckpointManager` with a distinct directory, not a new class |
| Paid-provider adapter seam docs | Docs only | — | No runtime code in v1 — explicitly a placeholder (DATA-06) |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `fredapi` | ≥0.5 (already a dep) | FRED + ALFRED (point-in-time vintages) fetch | Already used by `ingestion/fred.py`; the same `Fred` client class exposes `get_series_all_releases`, `get_series_as_of_date`, `get_series_first_release`, `get_series_vintage_dates` [VERIFIED: github.com/mortada/fredapi README] — no new package needed |
| `pandas` | ≥2.0 (already a dep) | Resampling, splice-ratio math, monthly frequency alias `"ME"` | Existing convention throughout the codebase |
| `requests` / `lxml` | already deps | multpl.com / macrotrends.net scraping (reused, not modified) | D-01 |
| `yfinance` | already a dep | ETF/single-name daily prices (satellites, holdings, watchlist) | `assets.py` fallback chain already handles this |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `pandas-datareader` | already an optional extra | Stooq fallback for daily ETF history when yfinance fails | Already wired as Phase 3 of `assets.py`'s fallback chain — no new integration work |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `fredapi.get_series_all_releases()` (bulk vintage pull, 1 call/series) | Looping `get_series_as_of_date()` per historical month | The loop is O(months) API calls per series — for 1962+ monthly that's ~750 calls × 5 series = 3,750 requests. `get_series_all_releases()` returns the full revision history in one call; point-in-time values are then reconstructed locally from `realtime_start`/`realtime_end` columns — [ASSUMED, not verified in this session] confirm the exact DataFrame shape at implementation time with a live call |
| Manual CSV download of Shiller data (`ie_data.xls`) | Scripted `requests.get()` of the same public XLS/CSV URL | Either works; scripted fetch keeps the pipeline hands-off (no manual re-download step), consistent with existing ingestion modules — recommend scripted fetch with a WARNING-level fallback message if Yale's URL changes (it has moved once before; econ.yale.edu vs shillerdata.com) |
| Stooq via `pandas-datareader` for splice segments | Stooq via direct CSV download (bypassing pandas-datareader) | `pandas-datareader`'s `StooqDailyReader` already exists in the codebase's fallback chain (`assets.py` Phase 3); Stooq disabled anonymous automated zip downloads circa Dec 2020 and can intermittently CAPTCHA/rate-limit — [CITED: quantstart.com "An Introduction to Stooq Pricing Data"] treat Stooq as a *fallback*, not a primary splice source, exactly as `assets.py` already does |

**Installation:** No new packages. `pip install -e "src/trading_crab_lib/[ingestion,dev]"` already provides everything (`fredapi`, `requests`, `lxml`, `yfinance`, `pandas-datareader` via extras).

**Version verification:** `fredapi>=0.5` is already pinned in `src/trading_crab_lib/pyproject.toml` and `requirements.txt`; the ALFRED methods used here (`get_series_all_releases`, `get_series_as_of_date`, `get_series_first_release`, `get_series_vintage_dates`) have existed on the `Fred` class since early releases per the current `mortada/fredapi` README [CITED: github.com/mortada/fredapi]. No version bump needed. Confirm at implementation time by running `pip show fredapi` in the dev environment (not runnable in this research session — `fredapi` is not installed in the research sandbox).

## Package Legitimacy Audit

**No new packages are introduced by this phase.** Every ingestion capability needed
(FRED/ALFRED, HTML scraping, ETF prices, Stooq fallback) is served by packages already
declared in `src/trading_crab_lib/pyproject.toml` (`fredapi`, `requests`, `lxml`,
`yfinance`, `pandas-datareader`). The Package Legitimacy Gate protocol is therefore
not applicable — there is nothing new to run `npm view`/`pip index versions` against.

If the planner or a future researcher discovers a need for a genuinely new package
(e.g. a dedicated Shiller-data client, a Norgate/Tiingo/EODHD SDK when those seams are
eventually implemented in a v2 phase), run the Package Legitimacy Gate at that time —
not now, since DATA-06 explicitly scopes those integrations to documentation-only
placeholders in this phase.

**Packages removed due to [SLOP] verdict:** none — no packages evaluated.
**Packages flagged as suspicious [SUS]:** none — no packages evaluated.

## Architecture Patterns

### System Architecture Diagram

```
                    ┌─────────────────────────────────────────────┐
                    │         src/trading_crab_lib/platform/       │
                    │         (NEW subpackage, D-01/D-02)          │
                    └─────────────────────────────────────────────┘
   Existing fetchers (imported, not modified)     New fetchers (this phase)
   ┌─────────────────┐ ┌──────────────┐            ┌────────────────────────┐
   │ ingestion/fred.py│ │multpl.py     │            │ platform/ingestion/    │
   │ ingestion/       │ │macrotrends.py│            │   alfred.py  (NEW)     │
   │  assets.py       │ │              │            │   shiller.py (NEW,     │
   └────────┬─────────┘ └──────┬───────┘            │     opt. — see below)  │
            │                  │                    └───────────┬────────────┘
            ▼                  ▼                                ▼
   ┌────────────────────────────────────────────────────────────────────────┐
   │  platform/ingest_monthly.py — orchestrates fetch calls, merges into    │
   │  ONE wide DataFrame indexed by daily dates (D-05: store daily where    │
   │  available) → data/checkpoints/platform/monthly_raw_daily.parquet     │
   └───────────────────────────────┬────────────────────────────────────────┘
                                    ▼
   ┌────────────────────────────────────────────────────────────────────────┐
   │  platform/splice.py — per-asset-class splice: overlap-window ratio-   │
   │  scale older segment to newer segment at documented join date         │
   │  (D-03/D-04) → produces ONE research series per asset class:          │
   │  equities_tr, long_duration_tr, gold, oil, cash                       │
   └───────────────────────────────┬────────────────────────────────────────┘
                                    ▼
   ┌────────────────────────────────────────────────────────────────────────┐
   │  platform/transforms_monthly.py — resample daily → monthly ("ME"),    │
   │  align quarterly agency series (ffill + publication lag OR ALFRED     │
   │  vintage-correct value as of each month-end) per feature taxonomy     │
   │  (DATA-01, DATA-03, DATA-04)                                          │
   └───────────────────────────────┬────────────────────────────────────────┘
                                    ▼
              data/checkpoints/platform/monthly_features.parquet
                    (fast / slow / agency columns tagged in
                     platform_settings.yaml taxonomy block)
```

### Recommended Project Structure
```
src/trading_crab_lib/platform/
├── __init__.py                # subpackage marker; NO re-exports needed yet
├── ingestion/
│   ├── __init__.py
│   └── alfred.py              # NEW — ALFRED vintage pulls for D-06 series
├── splice.py                  # NEW — per-asset-class splicing (D-03/D-04)
├── taxonomy.py                # NEW — fast/slow/agency classification helpers (DATA-04)
├── transforms_monthly.py      # NEW — monthly resample + alignment orchestration
├── ingest_monthly.py          # NEW — top-level orchestrator (calls existing + new fetchers)
└── checkpoints.py             # THIN wrapper: CheckpointManager pointed at a
                                # separate subdir (data/checkpoints/platform/) —
                                # do NOT subclass, just pass a different base dir
                                # if the class supports it, else instantiate a
                                # second CheckpointManager with a distinct CHECKPOINT_DIR
config/
└── platform_settings.yaml     # NEW — separate from settings.yaml (D-02: keeps the
                                # new subpackage's config independent of the frozen
                                # incumbent's validate_config() schema)
docs/
└── splicing_rules.md          # NEW — DATA-02 deliverable: source, date range,
                                # join date, method per asset (or a config block —
                                # planner's discretion per D-04)
```
**[ASSUMED]** — exact module names/layout are Claude's Discretion per CONTEXT.md; this
is a recommendation following existing `src/trading_crab_lib/` conventions (lowercase,
underscore, one concern per file), not a locked requirement.

### Pattern 1: Reuse existing fetchers unmodified, add a thin orchestrator
**What:** `platform/ingest_monthly.py` calls `trading_crab_lib.ingestion.fred.fetch_all(cfg)`,
`trading_crab_lib.ingestion.macrotrends.fetch_all(cfg)`, `trading_crab_lib.ingestion.assets.fetch_all(cfg)`
with a **platform-specific cfg dict** (different `start_date`, different series list,
`frequency` implicitly not used since these modules resample internally to `"QE"` — see
Pitfall 1 below) and re-resamples the *raw* pulled data to monthly itself, OR — cleaner —
duplicates only the resample line.
**When to use:** For every existing-fetcher reuse in this phase.
**Example:**
```python
# Source: pattern observed in src/trading_crab_lib/ingestion/fred.py:_fetch_one()
# platform/ingest_monthly.py (new file)
from __future__ import annotations
import pandas as pd
from fredapi import Fred

def _fetch_one_monthly(fred: Fred, series_id: str, start: str, end: str) -> pd.Series:
    """Monthly analog of fred.py:_fetch_one() — same client, different resample rule."""
    raw = fred.get_series(series_id, observation_start=start, observation_end=end)
    return raw.resample("ME").last()   # "ME" = month-end, vs incumbent's "QE"
```
**Why not literally call `ingestion.fred.fetch_all()`?** That function hardcodes
`.resample("QE")` inside `_fetch_one()` (see `src/trading_crab_lib/ingestion/fred.py:44`).
Reusing it as-is would give quarterly data, defeating DATA-01. The fetch *client
construction* and *series-list-driven parallel fetch pattern* are reusable; the resample
call is not. Recommend a thin new function that mirrors the pattern (same `Fred(api_key=...)`
construction, same `ThreadPoolExecutor` pattern) rather than importing and monkeypatching
the incumbent's private `_fetch_one`.

### Pattern 2: ALFRED point-in-time reconstruction (bulk pull, not per-date loop)
**What:** For each of the 5 D-06 series, call `get_series_all_releases(series_id)` ONCE.
This returns a long-format DataFrame with columns `date` (the observation's reference
period), `realtime_start`, `realtime_end`, `value` — every revision of every observation.
Point-in-time reconstruction: for a target "as-of" month-end `t`, the value known at `t`
for reference period `date` is the row where `realtime_start <= t < realtime_end`
(the vintage active at time `t`).
**When to use:** GDP, CPI (CPIAUCSL), UNRATE, INDPRO, PAYEMS (+GNP if kept) per D-06.
**Example:**
```python
# Source: fredapi README method list [CITED: github.com/mortada/fredapi]
# platform/ingestion/alfred.py (new file) — illustrative, verify exact column
# names against a live call before relying on this in production code.
from __future__ import annotations
import pandas as pd
from fredapi import Fred

def fetch_vintage_series(fred: Fred, series_id: str) -> pd.DataFrame:
    """One bulk call — NOT one call per historical date (avoids O(months) API load)."""
    return fred.get_series_all_releases(series_id)
    # columns (per fredapi docs): realtime_start, realtime_end, date, value

def value_as_of(all_releases: pd.DataFrame, as_of_date: pd.Timestamp) -> pd.Series:
    """Reconstruct the point-in-time series knowable at `as_of_date`."""
    known = all_releases[all_releases["realtime_start"] <= as_of_date]
    # keep, per reference `date`, the row with the LATEST realtime_start <= as_of_date
    # (i.e. the most recent vintage active at as_of_date)
    known = known.sort_values("realtime_start").groupby("date").tail(1)
    return known.set_index("date")["value"]
```
**Pre-vintage-era fallback (D-06):** ALFRED archives are documented to begin around
2006 for the ALFRED *website* [CITED: fredblog.stlouisfed.org "ALFRED at 15"], but the
underlying vintage data itself often reaches back much further per series — e.g. PAYEMS's
first vintage date is reported as 1955-05-06 [WEBSEARCH — unverified against a live API
call this session; flagged `[ASSUMED]`, confirm via `get_series_vintage_dates("PAYEMS")`
before committing to a specific coverage-start claim in the splice docs]. For dates
before a series' earliest recorded vintage, fall back to the incumbent's publication-lag
shift pattern (`.shift(1)` for a monthly-lag series) exactly as `fred.py` already does
for GDP/GNP — this is the "documented publication-lag-alignment fallback" D-06 calls for.

### Pattern 3: Treasury total-return synthetic from CMT yields ("par-bond repricing")
**What:** Constant Maturity Treasury (CMT) series (e.g. `GS10`) give *yields*, not
*prices*. To approximate a spliceable total-return series for "long duration" (target:
TLT proxy), use the par-bond repricing method: each period, assume you hold a freshly
issued par bond at the prevailing CMT yield; accrue that period's coupon income, then
"reprice" the now-slightly-seasoned bond at the *new* period's yield using standard
bond present-value math (price = PV of remaining coupons + face value, discounted at
the new yield). The month-over-month total return is (coupon accrued + price change) /
starting price.
**When to use:** D-03's "long duration" class, sourced from `GS10` (or a blend of
maturities if a smoother duration match is wanted — Claude's Discretion).
**Example:**
```python
# Source: pattern described in portfoliooptimizer.io "The Mathematics of Bonds:
# Simulating the Returns of Constant Maturity Government Bond ETFs"
# [CITED: portfoliooptimizer.io — method summary, not code; formula below is
# standard fixed-income PV math, adapted to this description]
def bond_price(yield_annual: float, coupon_annual: float, years_to_maturity: float,
                freq: int = 2) -> float:
    """PV of a par-ish bond given a flat yield curve assumption at `yield_annual`."""
    periods = int(round(years_to_maturity * freq))
    coupon = coupon_annual / freq
    r = yield_annual / freq
    pv_coupons = sum(coupon / (1 + r) ** t for t in range(1, periods + 1))
    pv_face = 1.0 / (1 + r) ** periods
    return pv_coupons + pv_face

def monthly_total_return(yield_t0: float, yield_t1: float, maturity_years: float = 10) -> float:
    """One month of total return: issue at yield_t0, reprice one month later at yield_t1."""
    price_t0 = bond_price(yield_t0, yield_t0, maturity_years)          # priced at par by construction
    price_t1 = bond_price(yield_t1, yield_t0, maturity_years - 1/12)   # coupon fixed, time passes, yield moves
    coupon_accrued = yield_t0 / 12
    return (price_t1 - price_t0 + coupon_accrued) / price_t0
```
**[ASSUMED — MEDIUM confidence]** This is the standard textbook approximation
(sometimes called the "rolling par bond" or "constant maturity total return" method);
the exact convention (semiannual vs monthly compounding, whether to also model roll-down
along the yield curve) should be locked down and documented in the splicing-rules doc
per D-04, not silently baked into code without a comment explaining the choice. Treat
this as the discretion-item deliverable, not gospel — cross-check against at least one
published index methodology (e.g. Bloomberg/ICE long-Treasury index docs, or academic
replications) before finalizing, since this directly determines what "long duration"
looked like 1962–1990s when no ETF existed to validate against.

### Anti-Patterns to Avoid
- **Looping `get_series_as_of_date()` per historical month:** O(N) API calls where N =
  number of months × number of vintage series (~750 × 5 = 3,750 calls for 1962+ monthly).
  Use `get_series_all_releases()` once per series and reconstruct locally (Pattern 2).
- **Splicing by chaining absolute price levels:** naively concatenating "series A price"
  then "series B price" at a join date produces a discontinuous jump unless B is
  rescaled to match A's level at the seam. D-04 explicitly requires ratio-splice
  (scale-to-match), not level-concatenation.
- **Modifying `fred.py`/`multpl.py`/`macrotrends.py`/`assets.py` in place:** D-01 forbids
  this. If a genuinely shared helper is needed (e.g. a rate-limit sleep utility), extract
  it to a new shared module the new subpackage imports — don't edit the frozen files
  even for "just adding a parameter."
- **Storing the monthly spine only, discarding daily:** D-05 requires daily persistence
  because the Phase 4 tripwire needs daily granularity. Don't resample-and-discard.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| ALFRED point-in-time reconstruction | A custom scraper of the ALFRED website, or a manual per-date loop against the FRED API | `fredapi.Fred.get_series_all_releases()` / `get_series_vintage_dates()` — already a dependency | Purpose-built for exactly this; one bulk call vs. hundreds of manual date-scoped calls; already imported in `fred.py` so no new install |
| ETF price fetch with fallback (satellites/holdings) | A new fetch-with-retry chain for the platform subpackage | `trading_crab_lib.ingestion.assets.fetch_all()` (existing 5-phase fallback: yfinance batch → yfinance SSL-bypass → Stooq → OpenBB → empty) | D-01 says import, don't rewrite; the existing chain already handles rate limits, SSL quirks (curl_cffi), and graceful NULL-tolerant degradation — re-solving this would duplicate ~500 lines for no benefit |
| HTML scraping of multpl.com / macrotrends.net | New scrapers | `ingestion/multpl.py`, `ingestion/macrotrends.py` (import, call with platform cfg) | Rate limiting (2s/3s) and parsing quirks are already solved and tested; macrotrends already returns monthly gold (1915+) and WTI (1946+) — exactly what D-03 needs, no new scraping code required at all for two of five splice segments |
| NULL-tolerant merge of variable-length histories | A custom "pad and align" utility | `pandas.concat(..., axis=1)` / `DataFrame.join(how="outer")` — the incumbent already relies on this pattern (see `assets.py` producing NaN-padded columns for pre-ETF-launch dates) | Standard pandas outer-join semantics already produce exactly the NULL-tolerant behavior D-05 asks for; downstream trees/naive models already tolerate NaN in this codebase's supervised paths |

**Key insight:** Every "new" capability in this phase (monthly resample, ALFRED vintages,
splicing, NULL-tolerant satellite ingestion) is either a one-line variant of an existing
pattern (resample frequency string) or a documented library feature one call away
(`get_series_all_releases`). The actual net-new code surface for this phase is small:
a splice module and a vintage-reconstruction module. Resist the temptation to build a
generalized "data source adapter framework" for the paid-provider seams (DATA-06) —
that requirement is explicitly documentation-only in v1.

## Common Pitfalls

### Pitfall 1: Reusing `fred.py`/`macrotrends.py`/`assets.py` fetch functions verbatim silently keeps quarterly resampling
**What goes wrong:** `ingestion/fred.py:_fetch_one()` and `ingestion/macrotrends.py`'s
`_scrape_series()` both hardcode `.resample("QE")` internally. Calling `fetch_all(cfg)`
from the new platform code without modification produces quarterly data even though the
phase's entire point is monthly.
**Why it happens:** The resample call is private/internal to the existing functions, not
exposed as a config-driven frequency parameter (there was no need for one before this phase).
**How to avoid:** Don't call the existing `fetch_all()` public functions for the *resampled*
DataFrame. Either (a) write new monthly-resample analogs that reuse the *client construction
+ parallel-fetch* pattern but override the resample rule (Pattern 1 above), or (b) if a
genuinely shared raw-fetch-then-resample helper would reduce duplication meaningfully,
consider (with the user, since this brushes against D-01) whether a *tiny* signature
addition (e.g. `resample_rule: str = "QE"` parameter) to the private helper is worth
requesting — but default to duplication over touching frozen files.
**Warning signs:** Row counts that look like ~260 rows (quarterly since 1950) instead of
~770 rows (monthly since 1962) in the platform's raw checkpoint.

### Pitfall 2: Splicing without documenting the join date silently breaks reproducibility
**What goes wrong:** A ratio-splice needs an explicit anchor date where old-series and
new-series overlap; if that date isn't pinned in config/docs, re-running the pipeline
after a data refresh can shift the seam (e.g. if the "modern" series' start date is
computed dynamically instead of hardcoded).
**Why it happens:** Overlap windows between two sources are often multiple months/years
wide (e.g. multpl S&P history and Shiller S&P history both cover 1950-2026); an
undocumented choice of "first overlapping observation" vs "last" vs "average of overlap
window" produces different splice scale factors on each run if the underlying data range
changes.
**How to avoid:** D-04 already requires this be documented per-asset — treat it as a
literal config value (`join_date: "1988-01-31"`, `method: "ratio_scale_at_join"`), not a
computed default. Write a small unit test asserting the recorded join date and computed
scale factor don't silently change when new raw data arrives.
**Warning signs:** A splice-quality plot showing a visible kink/discontinuity at the seam
that wasn't there in a prior run.

### Pitfall 3: ALFRED vintage data volume and API request cost
**What goes wrong:** `get_series_all_releases()` for a series with 70+ years of monthly
observations and monthly revisions can return a very large DataFrame (potentially tens of
thousands of rows for high-frequency, heavily-revised series). Fetching this for 5-6
series is fine, but doing it inside the parallel `ThreadPoolExecutor` pattern used by
`fred.py` without checking response size/time could stall the "8 workers" cap the
incumbent tuned for quarterly-cadence 16-series fetches.
**Why it happens:** All-releases payloads are much larger than the single-vintage
`get_series()` payloads the incumbent's rate-limit assumptions were tuned against.
**How to avoid:** Fetch the 5-6 ALFRED series serially or with a smaller worker cap than
the incumbent's 8; cache the bulk `all_releases` DataFrame as its own checkpoint (it
rarely changes month to month for old data) so re-runs don't re-fetch the full revision
history every time.
**Warning signs:** Ingestion step 1 for the platform subpackage taking dramatically
longer than the incumbent's ~10 minutes despite fetching fewer nominal series.

### Pitfall 4: Confusing "quarterly agency series aligned with publication lags" (DATA-01) with "ALFRED vintage-corrected" (DATA-03)
**What goes wrong:** These are two *different* fixes for two different problems and the
phase needs both, applied to different feature sets. Publication-lag shift fixes *look-
ahead bias from timing* (you don't know Q1 GDP until ~30 days after Q1 ends). ALFRED
vintage-correction fixes *look-ahead bias from revision* (the GDP number you'd have seen
30 days after Q1 end was later revised, sometimes substantially, and using the *final*
revised number is still look-ahead even with correct timing). Applying only one is not
sufficient for the D-06 series.
**Why it happens:** The incumbent codebase only has the `shift: true` mechanism (ADR #7);
it's tempting to assume "shift" already solves the agency-data problem, since it solved
"good enough" for the quarterly-only prior design (per the design doc's own R6 finding).
**How to avoid:** For the 5-6 D-06 series, apply BOTH: pull the *value known as of* each
monthly refresh date (vintage-correct, handles revision) — this value is inherently
already correctly-timed (a vintage active at date `t` only contains what was published
by `t`), so vintage-correction actually *subsumes* the shift for series where vintages are
available. Only fall back to the plain `shift()` pattern for dates *before* the earliest
available vintage (pre-vintage-era fallback per D-06).
**Warning signs:** A feature taxonomy config that lists a series under `agency:` but
still has a lingering `shift: true` flag copy-pasted from `settings.yaml` without an
accompanying `vintage: true` — the planner/implementer should treat `vintage: true` as
the primary control for D-06 series and `shift` as the explicit fallback-only path.

### Pitfall 5: Multi-frequency merge silently drops rows via inner-join defaults
**What goes wrong:** Monthly market data (daily→monthly resample) joined against
quarterly/less-frequent agency series using `pd.concat(..., axis=1)` with default `join="outer"`
is fine, but the moment anyone reaches for `pd.merge()` (default `how="inner"`) or
`DataFrame.join()` (default `how="left"` on the *calling* frame — order-dependent), the
resulting row count can silently shrink to the intersection or one side's index, dropping
months that only one source series covers.
**Why it happens:** pandas' three join primitives (`concat`, `merge`, `join`) default to
different behaviors, and this project's existing code (`assets.py`, checkpoint merges)
deliberately uses `concat(axis=1)` (outer by default) specifically to preserve NaN-padded
short histories — the exact NULL-tolerant behavior D-05 needs.
**How to avoid:** Standardize on `pd.concat([...], axis=1)` for all platform-subpackage
merges of per-source Series/DataFrames, matching the existing codebase convention; if
`merge()` is used anywhere, explicitly pass `how="outer"`.
**Warning signs:** Satellite ETF columns (short history) causing the merged DataFrame's
overall row count to shrink instead of staying at the full 1962+ monthly range.

## Code Examples

### Ratio-splice at an overlap window (D-04)
```python
# Source: general splicing technique, standard in academic long-history index
# construction (e.g. CRSP/Ibbotson methodology descriptions); implementation is
# original to this project, not copied from a specific library.
import pandas as pd

def ratio_splice(old: pd.Series, new: pd.Series, join_date: pd.Timestamp) -> pd.Series:
    """
    Scale `old` so its value at `join_date` matches `new`'s value at `join_date`,
    then concatenate: old (scaled) before join_date, new (unscaled) at/after.

    `old` and `new` must both have a valid observation at exactly `join_date`
    (or the nearest available date — resolve before calling, don't silently
    interpolate the scale factor).
    """
    scale = new.loc[join_date] / old.loc[join_date]
    scaled_old = old.loc[:join_date] * scale
    spliced = pd.concat([scaled_old.iloc[:-1], new.loc[join_date:]])
    spliced.name = new.name
    return spliced
```

### Monthly resample of daily raw prices, preserving daily checkpoint (D-05)
```python
# Source: adapted from src/trading_crab_lib/ingestion/assets.py resample pattern
# (which uses "QE"); this phase changes the target rule per DATA-01/D-05.
def to_monthly_spine(daily_prices: pd.DataFrame) -> pd.DataFrame:
    """Derive the monthly spine from a daily checkpoint. Daily itself is
    persisted separately (D-05) — this function does not discard it."""
    return daily_prices.resample("ME").last()
```

## State of the Art

| Old Approach (incumbent, frozen) | Current Approach (this phase) | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Quarterly resample (`"QE"`), ~1950 start, ~260 observations | Monthly resample (`"ME"`), ~1962 start, ~770 observations | This phase | 3x observation density; regime labeler (Phase 3) gets far more transitions to learn from without quadrupling detection lag in calendar time (design doc R1) |
| Publication-lag `shift()` only, applied to GDP/GNP | ALFRED vintage-correct values for GDP/CPI/UNRATE/INDPRO/PAYEMS, `shift()` fallback pre-vintage | This phase | Removes revision-driven look-ahead bias, not just timing-driven look-ahead bias (design doc R6/D12) |
| 16-ETF universe, 1993–2006 start (ETF inception) | Spliced research series per asset class back to ~1962; ETFs remain tradable instruments only | This phase | Labeling/features get 30+ extra years of history the ETF-only universe structurally cannot provide (design doc R13) |

**Deprecated/outdated:** The incumbent's `balanced_cluster` / `KMeansConstrained` and PCA(5)
approach is **not** touched by this phase (Phase 3's job) — do not attempt to migrate
clustering logic here; this phase produces the monthly feature table those future phases
will consume.

## Splicing Sources & Recommendation

Per-asset-class free-source comparison (Claude's Discretion item):

| Asset class | Modern (tradable proxy) | Research-series candidates (free) | Recommendation |
|---|---|---|---|
| US equities | SPY (1993+) | (a) multpl.com S&P 500 price + dividend (already scraped, monthly tables, back to ~1871 per multpl's own historical tables) (b) Shiller `ie_data.xls`/`shillerdata.com` (monthly, 1871+, includes an explicit **total-return** variant since Sept 2018) [CITED: shillerdata.com / search summary] | **Shiller total-return column**, not multpl's price-only series — multpl's existing scrape lacks a clean total-return construction (dividends are a separate scraped series requiring the researcher to build the TR math themselves); Shiller ships TR pre-built. Use multpl only as a cross-check/gap-fill source, not primary. |
| Long duration | TLT (2002+) | GS10 (FRED, CMT 10Y yield, 1953+) via par-bond repricing (Pattern 3) | GS10-based synthetic — no free source ships a pre-built long-Treasury *total-return* index back to 1962; this must be constructed. Document the repricing method explicitly (D-04). |
| Gold | IAU (2005+) | macrotrends monthly gold (1915+) — **already scraped by the incumbent** [VERIFIED: src/trading_crab_lib/ingestion/macrotrends.py DEFAULT_SERIES] | macrotrends — zero new ingestion code, just reuse via import (D-01) |
| Oil | USO (2006+) or class-level | macrotrends monthly WTI (1946+) — **already scraped**; FRED `WTISPLC` (1946+) as a cross-check [CITED: fred.stlouisfed.org/series/WTISPLC] | macrotrends primary (already integrated), FRED WTISPLC as a validation cross-check since it's a single extra FRED series call |
| Cash | FZFXX/SPAXX (NAV≈$1, no price history) | TB3MS (FRED, 3-month T-bill, 1934+) | TB3MS directly as the return series (D-03 already specifies this) — no splicing needed, single continuous source |

**[ASSUMED — MEDIUM confidence]** The Shiller total-return recommendation is based on
WebSearch summaries describing the dataset, not a direct inspection of the live
`shillerdata.com`/Yale CSV/XLS file structure in this research session (no network fetch
of the actual spreadsheet was performed). Before implementation, fetch the actual file
and verify: (1) the exact column name for the total-return series, (2) whether it's
already monthly or needs derivation from price+dividend, (3) the file's live URL (it has
moved between `econ.yale.edu/~shiller/data.htm` and `shillerdata.com` historically — treat
both as candidates and check which resolves).

## ALFRED Point-in-Time Implementation

See Architecture Patterns → Pattern 2 above for the mechanics. Summary of what's
[VERIFIED] vs [ASSUMED] in this research:

- **[CITED: github.com/mortada/fredapi README]** `fredapi.Fred` exposes
  `get_series_first_release`, `get_series_latest_release`, `get_series_as_of_date`,
  `get_series_all_releases`, `get_series_vintage_dates` — confirmed by fetching the
  live README during this research session.
- **[ASSUMED]** Exact column names/shape of `get_series_all_releases()`'s returned
  DataFrame (`date`, `realtime_start`, `realtime_end`, `value` per the general FRED API
  convention) — not confirmed by an actual API call in this session (`fredapi` is not
  installed in the research sandbox and `FRED_API_KEY` was not accessed per the
  untrusted-input/secrets boundary). **The planner should add a first task that makes
  ONE live call to `get_series_all_releases("PAYEMS")` and asserts the column names
  before building the rest of the vintage-reconstruction logic on top of an assumption.**
- **[ASSUMED]** Per-series earliest-vintage-date claims (e.g. "PAYEMS's first vintage is
  1955-05-06") — sourced from a WebSearch summary of ALFRED download pages, not a direct
  `get_series_vintage_dates()` call. Confirm per-series at implementation time; do not
  hardcode these dates into the splicing docs without that confirmation.

## Feature Taxonomy Config Design

Design doc §9 defines three tiers:
- **Fast layer** (regime detection; market-observed, unrevised): curve slope (10Y-3M,
  10Y-2Y), credit spreads (BAA-AAA full history; HY OAS 1997+ supplement), realized vol
  (multi-scale), trailing returns/momentum (multi-scale), oil, gold, USD index,
  breakevens (2003+).
- **Slow layer** (strategic tilt): CAPE, Buffett Indicator (market cap/GDP), dividend
  yield vs BAA, real-rate level.
- **Agency layer** (vintage-corrected only): unemployment, CPI, GDP growth — via ALFRED,
  used sparingly in the fast layer.

**Recommendation:** a `taxonomy:` block in the new `platform_settings.yaml`, keyed by
feature name, e.g.:
```yaml
taxonomy:
  fast:
    - curve_10y3m
    - curve_10y2y
    - credit_spread_baa_aaa
    - realized_vol_1m
    - realized_vol_3m
    - trailing_return_1m
    - oil
    - gold
  slow:
    - cape_shiller
    - buffett_indicator
    - div_yield_vs_baa
    - real_rate_level
  agency:
    - fred_unrate      # ALFRED vintage-corrected
    - fred_cpi         # ALFRED vintage-corrected
    - fred_gdp_growth  # ALFRED vintage-corrected
```
This declarative list is what DATA-04 calls "features classified fast/slow/agency
taxonomy in config" and doubles as the "lean full-history (1962+) feature set" — the
lean set is simply the union of `fast` + `slow` (agency-layer features "used sparingly,"
per design §9, and NULL-tolerant/derived-fallback features stay out of the lean labeling
set per the Missing-feature policy D11).

## Paid-Provider Adapter Seam Notes (DATA-06)

Placeholder-only documentation, no implementation, per explicit scope:

| Provider | What it offers (per public pricing/docs pages, not verified via API) | Placeholder seam |
|---|---|---|
| Norgate Data | Survivorship-bias-free US equities/ETF data, end-of-day, subscription desktop-database product (not a typical REST API) | `platform/ingestion/norgate.py` — stub module raising `NotImplementedError` with a docstring pointing to this research doc; no live integration |
| Tiingo | REST API, EOD + some intraday, equities/ETFs/mutual funds/forex/crypto, free tier for basic EOD [CITED: search summary of tiingo.com] | Same stub pattern |
| EODHD | REST API, 60+ exchanges, 150K+ tickers, fundamentals/options/news add-ons, paid plans starting ~€19.99/mo [CITED: eodhd.com/pricing] | Same stub pattern |

`stockcharts.com` and `finviz.com` are explicitly noted (per CONTEXT.md specifics) as
candidate **feature** sources (technical-indicator/screener data Glenn has paid access
to) for a **later milestone** — not price-history sources for this phase. No adapter
seam needed for them yet; a one-line comment in the paid-provider doc noting them as
"future feature source, not price source" satisfies DATA-06's scope.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `get_series_all_releases()` returns columns `date`/`realtime_start`/`realtime_end`/`value` | ALFRED Point-in-Time Implementation | Vintage-reconstruction code built on wrong column assumptions fails at first live run — low risk, caught immediately by the recommended first-task live-call check |
| A2 | ALFRED vintage archives for PAYEMS begin 1955-05-06 (and similar claims for GDP/CPI/UNRATE/INDPRO would need the same per-series check) | ALFRED Point-in-Time Implementation | Splicing-rules doc could understate/overstate true vintage coverage; low risk since D-06 already mandates a documented fallback for the pre-vintage era regardless of the exact date |
| A3 | Shiller's `shillerdata.com`/Yale dataset ships a ready-to-use monthly total-return column | Splicing Sources & Recommendation | If false, the S&P equities splice needs to construct TR itself from price+dividend (multpl already scrapes both), which is more work but not blocked — fallback path exists |
| A4 | Par-bond repricing is an acceptable/standard method for CMT-yield-to-total-return synthesis, semiannual compounding convention | Pattern 3 | If the method is non-standard or the compounding convention is wrong, the "long duration" research series' pre-2002 (pre-TLT) history could be subtly miscalibrated — flagged for explicit documentation and methodology cross-check per D-04, not silent adoption |
| A5 | `fredapi>=0.5` (already pinned) exposes the ALFRED methods listed — confirmed via README, but the exact pinned version range wasn't tested against a live import in this sandbox (fredapi not installed here) | Standard Stack | Low risk — if methods are missing in an older installed version, `pip install -U fredapi` resolves it; no breaking API changes reported in search results |

## Open Questions (RESOLVED)

> All three questions below are addressed in the Phase 1 plans (plan-checker verified,
> 2026-07-10): (1) column schema → defensive `_detect_vintage_columns()` + mocked tests
> in Plan 01-03; (2) GS10 vs TLT duration mismatch → accepted per locked D-03, documented
> in `docs/splicing_rules.md` per Plan 01-02 Task 2; (3) multi-segment splices → 2-segment
> default per D-03, extend only if coverage gaps found (Plan 01-02).


1. **Exact column schema of `fredapi.get_series_all_releases()`**
   - What we know: The method exists and returns "all releases... with dates and
     realtime validity periods" per the library's own README summary.
   - What's unclear: Exact column names/dtypes without a live call (blocked in this
     research session — no `fredapi` install, and touching `.env`/`FRED_API_KEY` is out
     of scope for a research agent per the untrusted-input boundary).
   - Recommendation: First implementation task should be a throwaway script making one
     live call and printing `.columns`/`.dtypes`/`.head()` before writing any
     reconstruction logic against it.

2. **Which maturity(ies) to use for the long-duration synthetic**
   - What we know: D-03 specifies GS10 as the source; TLT (the tradable proxy) tracks
     20+ year Treasuries, not 10-year — there's a duration mismatch between the research
     series (GS10, ~10Y) and the tradable ETF (TLT, ~20Y+).
   - What's unclear: Whether to (a) accept the mismatch since "model the index, trade
     the ETF" already accepts proxy imperfection (D-03's own framing), (b) blend GS10 +
     GS20/GS30 where available, or (c) use a different single maturity closer to TLT's
     effective duration.
   - Recommendation: Accept GS10 per the explicit D-03 mapping (it's a locked decision,
     not open) but flag the duration mismatch explicitly in the splicing-rules doc so a
     future reader isn't confused about why the research series and live TLT diverge in
     volatility magnitude.

3. **Multiple overlapping join dates when more than 2 sources exist for one asset class**
   - What we know: Gold, for example, could have macrotrends (1915+) → LBMA (modern) →
     IAU-implied spot (2005+) as three segments, not two.
   - What's unclear: Whether the phase needs a 2-segment or N-segment splice per asset;
     D-03 only lists macrotrends → IAU explicitly (2 segments), suggesting 2 is sufficient
     for gold specifically.
   - Recommendation: Default to the simplest 2-segment splice matching D-03's literal
     text per asset; only add a third segment if a coverage gap is found during
     implementation (e.g. if macrotrends' scrape has NaN gaps in the 1993–2005 window).

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| `fredapi` | FRED + ALFRED fetch | ✗ (not installed in research sandbox) | — | Install via `pip install -e "src/trading_crab_lib/[ingestion]"` before implementation — already declared in the extras, no fallback needed once installed |
| `FRED_API_KEY` | All FRED/ALFRED calls | Not checked (secrets boundary — research agent does not read `.env`) | — | Existing key already used by incumbent pipeline (D-07: no new credential needed) |
| Network access (FRED, multpl.com, macrotrends.net, Yahoo Finance, Shiller data host) | All ingestion | Not tested (research sandbox network policy; live calls to shillerdata.com/GitHub succeeded via WebFetch/WebSearch during this session, suggesting outbound HTTPS is generally reachable) | — | If any single source is unreachable at implementation time, each existing fetcher module already has graceful WARNING-level degradation; extend the same pattern to new fetchers |
| Python 3.11.15 | Runtime | ✓ | 3.11.15 | — (project supports 3.10–3.13; 3.11 is within range) |

**Missing dependencies with no fallback:** none identified — `fredapi` just needs a
routine `pip install` with an extra already declared in `pyproject.toml`.

**Missing dependencies with fallback:** none beyond the above (all resolvable by
standard `pip install -e ".[ingestion]"`).

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 8.0+ (already the project standard) |
| Config file | root `pyproject.toml` `[tool.pytest.ini_options]` (existing; no new config needed) |
| Quick run command | `pytest tests/unit/test_platform_splice.py tests/unit/test_platform_alfred.py -x` |
| Full suite command | `pytest tests/ -v` (existing 769-test incumbent suite MUST stay green — this phase adds tests, never touches existing ones) |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| DATA-01 | Monthly resample produces ~770 rows (1962-2026 monthly) not ~260 (quarterly) | unit | `pytest tests/unit/test_platform_ingest.py::test_monthly_row_count -x` | ❌ Wave 0 |
| DATA-01 | Quarterly agency series correctly aligned (no look-ahead at month boundaries within a quarter) | unit | `pytest tests/unit/test_platform_ingest.py::test_quarterly_series_alignment -x` | ❌ Wave 0 |
| DATA-02 | `ratio_splice()` produces continuous series at the join date (no discontinuity beyond floating-point tolerance) | unit | `pytest tests/unit/test_platform_splice.py::test_splice_continuity_at_join -x` | ❌ Wave 0 |
| DATA-02 | Each of 5 core asset classes has a documented splice rule (source, date range, join date, method) — presence check against docs/config | unit | `pytest tests/unit/test_platform_splice.py::test_all_asset_classes_documented -x` | ❌ Wave 0 |
| DATA-03 | ALFRED vintage reconstruction returns a value known strictly before its later revision (mocked `get_series_all_releases` response) | unit | `pytest tests/unit/test_platform_alfred.py::test_value_as_of_respects_vintage -x` | ❌ Wave 0 |
| DATA-03 | Pre-vintage-era dates fall back to publication-lag shift, not an error/NaN | unit | `pytest tests/unit/test_platform_alfred.py::test_pre_vintage_fallback -x` | ❌ Wave 0 |
| DATA-04 | Every column in the monthly feature checkpoint has exactly one taxonomy tag (fast/slow/agency) | unit | `pytest tests/unit/test_platform_taxonomy.py::test_every_feature_tagged -x` | ❌ Wave 0 |
| DATA-05 | Satellite ETF with short history produces NaN (not a crash/dropped row) for pre-inception dates in the merged monthly frame | unit | `pytest tests/unit/test_platform_ingest.py::test_short_history_satellite_null_tolerant -x` | ❌ Wave 0 |
| DATA-06 | Paid-provider stub modules import cleanly and raise `NotImplementedError` with a helpful message on call | unit | `pytest tests/unit/test_platform_paid_provider_stubs.py -x` | ❌ Wave 0 |

All tests use synthetic/mocked inputs (no live network calls in the automated suite),
matching the incumbent's existing convention (`tests/unit/test_ingestion.py` mocks
`requests.get`; this phase's new tests should mock `fredapi.Fred` and `requests.get`
identically — do not add tests that hit live FRED/yfinance/macrotrends during CI).

### Sampling Rate
- **Per task commit:** targeted `pytest tests/unit/test_platform_*.py -x`
- **Per wave merge:** `pytest tests/ -v` (full suite — confirms zero regression to the
  frozen incumbent's 769 tests)
- **Phase gate:** Full suite green before `/gsd-verify-work`

### Wave 0 Gaps
- [ ] `tests/unit/test_platform_ingest.py` — covers DATA-01, DATA-05
- [ ] `tests/unit/test_platform_splice.py` — covers DATA-02
- [ ] `tests/unit/test_platform_alfred.py` — covers DATA-03
- [ ] `tests/unit/test_platform_taxonomy.py` — covers DATA-04
- [ ] `tests/unit/test_platform_paid_provider_stubs.py` — covers DATA-06
- [ ] Framework install: none — pytest already configured project-wide; no new fixtures
  needed beyond what `tests/conftest.py` already provides (may add a small
  `platform_monthly_index` fixture analogous to existing `quarterly_index`)

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | N/A — no new auth surface; reuses existing `FRED_API_KEY` env-var pattern |
| V3 Session Management | no | N/A |
| V4 Access Control | no | N/A — local pipeline, no multi-user access control surface |
| V5 Input Validation | yes | Same pattern as incumbent: `validate_config()`-style fail-fast on the new `platform_settings.yaml` schema; parse scraped/API values defensively (existing multpl.py `value_type` conversion pattern already handles this for scraped values) |
| V6 Cryptography | no | N/A — no secrets generated/stored by this phase beyond the existing `FRED_API_KEY` (already handled by the incumbent's `.env` convention, D-07 explicitly reuses it, no new credential) |

### Known Threat Patterns for this stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| SSRF via unvalidated splice/scrape URLs if paid-provider seam docs later gain a config-driven `base_url` | Tampering | Not applicable to this phase (DATA-06 is docs-only, no runtime HTTP client added for paid providers); when implemented in a future v2 phase, hardcode/allowlist base URLs the same way `macrotrends.py`'s `base_url` default works today rather than accepting arbitrary user-supplied URLs |
| Malformed/malicious scraped HTML content (multpl.com/macrotrends.net) causing a crash or (less likely) code execution via `pandas.read_html` | Tampering | Already mitigated by the incumbent's try/except-and-log-WARNING pattern (`macrotrends.py:_scrape_series` wraps `pandas.read_html` fallback in the same broad exception handling as the rest of the module) — this phase's new fetchers should follow the identical pattern, not introduce `eval`/`exec` on scraped content |
| Credential leakage via `FRED_API_KEY` in logs | Information Disclosure | Existing `fred.py` never logs the key itself, only series IDs/friendly names — new `alfred.py` fetcher must follow the same discipline (log series IDs, never the `Fred` client's constructor args) |

## Sources

### Primary (HIGH confidence)
- `src/trading_crab_lib/ingestion/fred.py`, `macrotrends.py`, `assets.py`, `multpl.py` —
  direct code inspection, this session
- `src/trading_crab_lib/checkpoints.py`, `config.py` — direct code inspection, this session
- `config/settings.yaml` — direct inspection, this session
- `.planning/codebase/INTEGRATIONS.md` — existing fetcher inventory doc
- `platform_design/platform_design.md` §2, §9, §11 — direct inspection, this session
- github.com/mortada/fredapi README (fetched live via WebFetch this session) — ALFRED
  method names confirmed directly from source, not from a search summary

### Secondary (MEDIUM confidence)
- WebSearch summary of ALFRED coverage/vintage-date claims (fredblog.stlouisfed.org,
  alfred.stlouisfed.org pages) — not independently re-fetched in full
- WebSearch summary of Shiller dataset structure (shillerdata.com, econ.yale.edu) — not
  independently re-fetched; the actual CSV/XLS was not opened this session
- WebSearch summary of Stooq/pandas-datareader coverage and limitations (quantstart.com,
  pydata.github.io pandas-datareader docs)
- WebSearch summary of par-bond repricing methodology (portfoliooptimizer.io blog post)
- FRED series pages (WTISPLC, DCOILWTICO) — WebSearch summary, series IDs cross-checked
  against multiple independent search result snippets for consistency

### Tertiary (LOW confidence)
- Per-series ALFRED earliest-vintage-date claims (e.g. PAYEMS "1955-05-06") — single
  WebSearch summary, not independently verified via a live `get_series_vintage_dates()`
  call; flagged in Assumptions Log (A2)
- Norgate/Tiingo/EODHD pricing/feature comparisons — WebSearch summary of third-party
  comparison sites, relevant only to the documentation-only DATA-06 deliverable

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — no new packages, all reuse of already-vetted, already-installed
  dependencies with a directly-fetched README confirming the exact methods needed
- Architecture: HIGH — directly follows locked D-01/D-02/D-05 decisions and existing
  codebase conventions inspected first-hand
- Splicing sources/methodology: MEDIUM — asset-class-to-source mapping is grounded in
  locked D-03 decisions; the *exact* file structure of Shiller's dataset and the *exact*
  bond-repricing convention are WebSearch-sourced and need a implementation-time spot-check
- ALFRED mechanics: MEDIUM — method names/existence confirmed via live README fetch
  (HIGH); exact DataFrame schema and per-series vintage-coverage-start dates are
  WebSearch-sourced only (LOW) — flagged explicitly as the top open question
- Pitfalls: HIGH — derived directly from reading the actual incumbent source code
  (hardcoded `"QE"` resample calls, join semantics, etc.), not speculative

**Research date:** 2026-07-10
**Valid until:** ~60 days (2026-09-08) for the codebase-derived findings (stable — the
incumbent pipeline is frozen); ~30 days for the external-source claims (Shiller file
location, ALFRED vintage coverage specifics) since these should be spot-checked against
live sources at implementation time regardless of this document's age.
