# External Integrations

**Analysis Date:** 2026-03-16

## APIs & External Services

**Macroeconomic data:**
- FRED (Federal Reserve Economic Data) - Quarterly macroeconomic series ingestion
  - SDK/Client: `fredapi` (declared in `pyproject.toml`; used in `src/market_regime/ingestion/fred.py`)
  - Auth: `FRED_API_KEY` (loaded via `src/market_regime/config.py`, required by `src/market_regime/ingestion/fred.py`)

**Web-scraped macro/market series:**
- multpl.com - Scrapes multiple public time series (e.g., S&P 500 valuation and Treasury rates) defined in `config/settings.yaml`
  - SDK/Client: `requests` + `lxml` (used in `src/market_regime/ingestion/multpl.py`)
  - Auth: none (public pages; uses a browser-like User-Agent in `src/market_regime/ingestion/multpl.py`)

**Asset prices (ETF / equity):**
- Yahoo Finance - Primary data source for monthly adjusted-close ETF prices
  - SDK/Client: `yfinance` (used in `src/market_regime/ingestion/assets.py`)
  - Auth: none (public)
  - Notes: uses `curl_cffi` sessions and disables TLS verification for reliability in some network environments (`src/market_regime/ingestion/assets.py`)

**Asset prices (fallback providers):**
- stooq.pl - Fallback historical price source
  - SDK/Client: `pandas-datareader` (optional; used in `src/market_regime/ingestion/assets.py`)
  - Auth: none (public)
- OpenBB - Fallback wrapper that can use multiple providers
  - SDK/Client: `openbb` (optional; used in `src/market_regime/ingestion/assets.py`)
  - Auth: none required for the initial `cboe` provider attempt; other OpenBB providers may require keys (integration points exist in `src/market_regime/ingestion/assets.py`, but specific key env vars are not configured in this repo)

## Data Storage

**Databases:**
- Not detected (no database drivers/ORMs or connection configuration found)

**File Storage:**
- Local filesystem only
  - Parquet artifacts: `pyarrow` via pandas (`data/raw/*.parquet`, `data/processed/*.parquet`, `data/regimes/*.parquet`, managed by `src/market_regime/checkpoints.py` and used in `run_pipeline.py`)
  - Pickle artifacts: Used for models and Grok labels (see `src/market_regime/ingestion/grok.py`, `run_pipeline.py`)

**Caching:**
- Checkpointing layer on local disk (no external cache service)
  - Implementation: `src/market_regime/checkpoints.py` (CheckpointManager, parquet + manifest pattern)

## Authentication & Identity

**Auth Provider:**
- None (no user auth / identity system detected)
- API auth is limited to a single external key for FRED via `FRED_API_KEY` (see `src/market_regime/config.py`, `.env.example`)

## Monitoring & Observability

**Error Tracking:**
- None detected (no Sentry/OTel/Datadog integrations found)

**Logs:**
- Python `logging` configured via `src/market_regime/config.py:setup_logging()` and used throughout `src/market_regime/*`

## CI/CD & Deployment

**Hosting:**
- Not detected (no deployment configuration found)

**CI Pipeline:**
- None detected (no `.github/workflows/*.yml` present)

## Environment Configuration

**Required env vars:**
- `FRED_API_KEY` - required for `src/market_regime/ingestion/fred.py`; loaded/injected by `src/market_regime/config.py`

**Secrets location:**
- Local env vars / `.env` (template in `.env.example`; `.env` expected to exist locally and be gitignored)

## Webhooks & Callbacks

**Incoming:**
- None detected (no HTTP server / webhook endpoints present)

**Outgoing:**
- None detected (no webhook emitters present)

---

*Integration audit: 2026-03-16*
