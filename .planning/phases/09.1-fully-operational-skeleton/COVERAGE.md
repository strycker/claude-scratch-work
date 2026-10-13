# API Coverage — FRED and Tiingo (Phase 9.1)

> Full coverage by default. Opt-outs are explicit, reasoned decisions. The skeleton calls exactly one read endpoint
> per service (09.1-02 `ingest.py`); everything else is a decided gap, most of it owned by a later phase or excluded by
> the product rules (P-01 long-only/no crypto, P-03 nothing under one month, KISS P-07).

## FRED (api.stlouisfed.org/fred)

| capability | decision | reason |
|---|---|---|
| fred: series/observations (monthly values) | INTEGRATE | |
| fred: observations with realtime_start/realtime_end (ALFRED vintages) | OPT-OUT | not needed yet: the skeleton shifts by publication_lags; vintages arrive with the data spine in 9.2 (DECISIONS D-04) |
| fred: series/vintagedates | OPT-OUT | not needed yet: arrives with ALFRED vintages in 9.2 |
| fred: series (metadata) | OPT-OUT | not needed: the four series ids are fixed in config/platform_settings.yaml |
| fred: series/search | OPT-OUT | not needed: no series discovery; ids are chosen in config by a human |
| fred: observations units/frequency/aggregation_method transforms | OPT-OUT | not needed: raw levels only; any transform happens once, causally, in pandas |
| fred: releases, categories, sources, tags, series/updates | OPT-OUT | not needed: catalogue browsing has no use in the platform |
| fred: GeoFRED maps | OPT-OUT | explicitly out of scope: no regional data in the design |

## Tiingo (api.tiingo.com)

| capability | decision | reason |
|---|---|---|
| tiingo: daily end-of-day prices (adjClose) | INTEGRATE | |
| tiingo: daily prices with resampleFreq=monthly | OPT-OUT | not needed: the docs do not state which date a monthly row carries; pandas resamples daily to month-end |
| tiingo: daily ticker metadata | OPT-OUT | not needed: tickers are fixed in config; history start comes from the data |
| tiingo: IEX intraday and realtime quotes | OPT-OUT | explicitly out of scope: nothing under one month (P-03) |
| tiingo: crypto | OPT-OUT | explicitly out of scope: no crypto (P-01) |
| tiingo: forex | OPT-OUT | not needed: USD-listed ETFs only |
| tiingo: fundamentals | OPT-OUT | not needed yet: no fundamental feature in the skeleton; revisit with the 9.3 taxonomy |
| tiingo: news | OPT-OUT | not needed: no text data in the design |
| tiingo: websocket streaming | OPT-OUT | explicitly out of scope: weekly manual CLI runs, no automation in 9.1 |
