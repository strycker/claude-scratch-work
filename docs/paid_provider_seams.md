# Paid Provider Adapter Seams (DATA-06)

**Status: placeholder only.** Nothing in this document is implemented in v1.
The platform data layer (`src/trading_crab_lib/platform/`) uses free sources
exclusively (FRED/ALFRED, multpl.com, macrotrends.net, yfinance, Stooq). This
document reserves three explicit adapter seams for a future v2 milestone and
records the reasoning so no one re-derives it from scratch.

## Why placeholders now

Free-source splicing (see `docs/splicing_rules.md`) gets acceptable quality
for v1's core-5 asset classes back to ~1962. A future milestone may want
survivorship-bias-free equity/ETF constituent data or cleaner intraday/EOD
feeds — that's what these three providers offer. Building real adapters now
would be speculative: no v1 requirement consumes them, and none of the three
have a working REST client wired into this repo. Each stub module raises
`NotImplementedError` so the seam is visible and intentional, not silently
missing.

## Provider seams

### Norgate Data

- **What it offers:** Survivorship-bias-free US equities/ETF end-of-day
  data — useful for breadth features that need historically-accurate index
  constituents (delisted/renamed tickers included), which no free source
  in this repo currently provides.
- **Integration shape:** Subscription desktop database product (Norgate Data
  Updater + local flat-file/API access), not a typical REST API. A future
  adapter would likely read from the local Norgate data directory rather
  than making HTTP calls.
- **Stub module:** `src/trading_crab_lib/platform/ingestion/norgate.py` —
  `fetch_prices(cfg)` raises `NotImplementedError` immediately. No SDK
  import, no network call, no new dependency.

### Tiingo

- **What it offers:** REST API for end-of-day (plus some intraday) equities,
  ETFs, mutual funds, forex, and crypto data. Has a free tier for basic EOD
  access, making it the lowest-friction of the three to eventually pilot.
- **Integration shape:** Standard REST API with an API key, similar in
  shape to the existing `fred.py` fetcher pattern (per-series HTTP GET,
  JSON response).
- **Stub module:** `src/trading_crab_lib/platform/ingestion/tiingo.py` —
  `fetch_prices(cfg)` raises `NotImplementedError` immediately. No SDK
  import, no network call, no new dependency.

### EODHD

- **What it offers:** REST API covering 60+ exchanges and 150K+ tickers,
  with fundamentals/options/news add-ons available on paid plans — the
  broadest coverage of the three if international/exchange breadth is
  ever needed.
- **Integration shape:** Standard REST API with an API key.
- **Stub module:** `src/trading_crab_lib/platform/ingestion/eodhd.py` —
  `fetch_prices(cfg)` raises `NotImplementedError` immediately. No SDK
  import, no network call, no new dependency.

## When a real adapter is eventually built (v2)

- Read this document first and update the "Integration shape" section with
  whatever was actually verified against the live provider (this document
  currently reflects public pricing/docs pages, not a live API inspection).
- Follow the existing ingestion module conventions: config-driven series
  list (no hardcoded tickers/URLs in Python), graceful WARNING-level
  degradation on network failure, `from __future__ import annotations`,
  logging not print.
- Allowlist any provider base URL rather than accepting an arbitrary
  user-supplied URL (SSRF hygiene — see the phase's threat register,
  T-04-01).
- Run the Package Legitimacy Gate on any new SDK dependency before adding
  it (`pip index versions <pkg>`, verify on PyPI) — none of the three
  providers require a new dependency for this placeholder phase, but a
  real REST client library may be worth adding later.

## Feature sources (not price sources) — stockcharts.com / finviz.com

Glenn has existing paid subscriptions to **stockcharts.com** and
**finviz.com**. These are candidate **feature** sources (technical
indicators, screener output) for a later milestone — explicitly **not**
price-history sources for this phase. No adapter seam is reserved for them
here; when that later milestone lands, treat them as a new, separate
integration (not a variant of the norgate/tiingo/eodhd price-provider
pattern above).
