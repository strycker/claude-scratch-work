# ALFRED Vintage Alignment (DATA-03)

This document records the scope of ALFRED point-in-time vintage correction, the
distinction between publication-lag *shift* and vintage *correction*, and the
pre-vintage-era fallback policy — matching the behavior implemented in
`src/trading_crab_lib/platform/ingestion/alfred.py`.

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
