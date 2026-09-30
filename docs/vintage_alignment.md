# ALFRED Vintage Alignment (DATA-03)

This document records the scope of ALFRED point-in-time vintage correction, the
distinction between publication-lag *shift* and vintage *correction*, and the
pre-vintage-era fallback policy — matching the behavior implemented in
`src/trading_crab_lib/platform/ingestion/alfred.py`.


> **Known gap (found 2026-09-29, Phase 8 code review CR-03) — closed in code by 08.1-01.**
> Publication lag used to be handled only where `shift: true` was set, and only for FRED series:
> `fred_m2sl` (~1 month) and `fred_totalsl` (~2 months) carried `shift: false`, and multpl
> `div_yield` (2–3 months, in classifier #1's lean set) had no lag handling at all. Every raw series
> now has an entry in the top-level `publication_lags` table of `config/platform_settings.yaml`,
> applied once in `transforms_monthly.build_monthly_spine` (`publication_lags.apply_publication_lags`),
> and `tests/unit/test_platform_point_in_time.py` fails when a feature at t uses a value published
> after t. The `shift` flag is gone from `fred_monthly`. The tracked `monthly_raw` is migrated in 08.1-03.

## D-06 vintage scope

True point-in-time ALFRED vintages are pulled ONLY for the revision-heavy agency
series used in regime labeling/features:

- **GDP** (`GDPC1`)
- **CPI** (`CPIAUCSL`)
- **UNRATE**
- **INDPRO**
- **PAYEMS**
- (+GNP, if kept)

All other agency-ish series keep the existing publication-lag `shift: true`
mechanism (`src/trading_crab_lib/ingestion/fred.py`, ADR #7). **Market-observed
series never need vintages** — rates, spreads, and prices are known in real
time as they're published; there is no revision history to correct for.

## Publication-lag shift vs. vintage correction

These are two different fixes for two different kinds of look-ahead bias, and
the D-06 series need both, applied in the right order:

| | Fixes | Mechanism |
|---|---|---|
| **Publication-lag shift** (`shift: true`) | *Timing* look-ahead — you cannot know Q1 GDP the day Q1 ends | `df[col].shift(+1)` in `fred.py` |
| **Vintage correction** (this module) | *Revision* look-ahead — the number known 30 days after Q1 end was later revised, and using the *final* revised figure is still cheating even with correct timing | `value_as_of(all_releases, as_of_date)` reconstructs the value actually published by `as_of_date` |

**Vintage correction subsumes the shift where vintages exist.** A vintage
active at date `t` only contains observations that had actually been
published by `t` — it is *inherently* correctly-timed as well as
revision-correct, so there is no need to apply an additional `shift()` on
top of a vintage-corrected series. The plain `shift()` pattern is used only
as the fallback for dates before the earliest recorded vintage (below).

Applying only the shift (as the incumbent quarterly pipeline does today,
ADR #7) is **not sufficient** for the five D-06 series — it is still possible
to be scoring against a later-revised figure even though the timing is
correct.

## Pre-vintage-era fallback (accepted compromise)

ALFRED's vintage archives mostly begin decades after each series' raw start
date (often the 1990s, not 1962+). `align_with_fallback()` handles this
explicitly rather than emitting `NaN` or raising:

- For `as_of` dates **before** a series' earliest recorded `realtime_start`,
  the result is the corresponding **publication-lag-shifted** value from the
  caller-supplied `shift_series` — the same value the incumbent `fred.py`
  pipeline already produces.
- For `as_of` dates **at or after** the earliest vintage, the result is the
  vintage-corrected value from `value_as_of()`.

This is stated here explicitly per D-06 ("not silently absorbed"): for the
pre-vintage era, the D-06 series are only as look-ahead-safe as the ordinary
shift mechanism — genuine point-in-time correction is unavailable before
ALFRED's archive begins for that series.

**Fallback lag per series (08.1, ruling 1).** The fallback lag is each series'
`publication_lags.<name>.fallback_months`, applied on the reference-month grid by
`transforms_monthly._shift_fallback_series`. Pre-vintage `fred_gdp` now lags **3**:
GDPC1 is quarterly, dated at the quarter start and forward-filled to monthly, so Q1
(reference January) must wait until April 30 for the ~end-of-April advance estimate.
It used to lag 1, which showed Q1 at February 28 — before the quarter had even ended —
and so made pre-vintage GDP visible about 2 months early before 1991-12 (GDPC1's
earliest vintage is 1991-12-04). CPI's fallback stays at 1: it covers only 1962 to
1972-07 and CPI is released mid-following-month. UNRATE, INDPRO and PAYEMS have
vintages covering the whole spine, so their fallback of 1 is never consulted.

**Per-series earliest-vintage dates are not hardcoded.** RESEARCH Assumption
A2 flags that per-series ALFRED coverage-start claims (e.g. PAYEMS
"1955-05-06") come from a single WebSearch summary, not an independently
verified live call. `align_with_fallback()` derives the cutover point at
runtime from `all_releases[realtime_start].min()`, so it self-corrects
against whatever the live API actually returns. Before the first real run,
confirm coverage with a live `get_series_vintage_dates()` spot-check per
D-06 series — this is a manual, non-blocking follow-up (the defensive column
detection in `_detect_vintage_columns()` means an unexpected schema fails
loudly rather than silently misreconstructing data).

## Credentials

Per D-07, ALFRED reuses the existing `FRED_API_KEY` — no new credentials are
required. `fetch_all_vintages(cfg)` reads the key from
`cfg["fred_vintage"]["api_key"]` (injected by the Plan 01 config loader) and
never logs it; only series IDs and friendly names are logged.
