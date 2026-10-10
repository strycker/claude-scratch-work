# External Integrations

**Analysis Date:** 2026-10-05

> **Update 2026-10-05 (after this map was written; DECISIONS G-13 and P-07).** These were **deleted**:
> - `platform/parked/` (classifier #2, the joint driver, the stability suite);
> - the parked-only helpers `allocation/joint_tilt.py`, `evaluation/dependence.py` and the `features/` package;
> - the `labeling_2` config block;
> - the research scripts `run_joint_lift`, `run_subsample_stability`, `terminal_month_diagnostic`,
>   `joint_lift_diagnostics` and `diagnose_s1_truncation`;
> - the one-off scripts `diagnose_yahoo_tls`, `diagnose_yfinance`, `diagnose_cpi_handoff`, `run_policy_trials`,
>   `smoke_step5.sh` and `egress_test.sh`;
> - the parked-boundary and doc-count tests.
>
> Root `CLAUDE.md` was trimmed; the old text is in `docs/archive/LEGACY-CLAUDE.md`. Wherever this map mentions any
> of these, read it as history.


## APIs & External Services

### FRED (Federal Reserve Economic Data)

- **Service**: Federal Reserve Economic Data API
- **What it's used for**: Macroeconomic time series (GDP, CPI, yields, employment, money supply, etc.)
- **SDK/Client**: `fredapi` (Python wrapper around FRED REST API)
- **Auth**: `FRED_API_KEY` environment variable (free registration at fred.stlouisfed.org)
- **Locations**: 
  - Legacy: `src/trading_crab_lib/ingestion/fred.py` (quarterly resampling)
  - Platform: `src/trading_crab_lib/platform/ingestion/macro_monthly.py` (monthly fetching), `macro_daily.py` (daily fetching), `alfred.py` (point-in-time vintage data)
- **Rate limiting**: Parallel fetch via ThreadPoolExecutor (max 8 workers for quarterly, 3 for ALFRED) to reduce wall-clock time from N×latency to ~1 round-trip
- **Data series**: 14-24 series depending on pipeline (GDP, GNP, BAA, AAA, CPI, yields, VIX, unemployment, M2, etc.)
- **Publication lag shifts**: GDP and GNP (+1 quarter shift) to prevent look-ahead bias (released ~30 days after quarter end)
- **ALFRED (point-in-time vintage)**: Supports `get_series_all_releases()` for revision-historical data on GDP, CPI, UNRATE, INDPRO, PAYEMS

### multpl.com (Web Scraping)

- **Service**: Stock market valuation metrics website (multpl.com)
- **What it's used for**: S&P 500 price, earnings, P/E ratio, dividend yield, dividend growth, Cape Shiller PE, treasury rates, inflation data
- **Method**: CSS selector scraping via lxml (200% faster than BeautifulSoup)
- **Target**: `#datatable` table rows from HTML pages
- **Locations**: 
  - Legacy: `src/trading_crab_lib/ingestion/multpl.py` (quarterly)
  - Platform: `src/trading_crab_lib/platform/ingestion/macro_monthly.py` (monthly scraping via multpl module)
- **Rate limiting**: 2 seconds between requests to avoid bot detection
- **Series count**: 46 datasets from multpl (S&P 500 metrics, treasuries, CPI, GDP, income, population, etc.)
- **User-agent**: Custom user-agent string to bypass basic bot checks
- **Config**: URLs and value parsing rules in `config/settings.yaml` under `multpl.datasets`

### macrotrends.net (Web Scraping)

- **Service**: Historical commodity price and economic data website
- **What it's used for**: Long-history commodity prices (gold back to 1915, WTI crude back to 1946) and other economic indicators
- **Method**: JSON extraction from embedded JavaScript (`<script>var rawData={...}</script>` tags)
- **Locations**: 
  - Legacy: `src/trading_crab_lib/ingestion/macrotrends.py`
  - Platform: `src/trading_crab_lib/platform/ingestion/macro_monthly.py` (via macrotrends module)
- **Rate limiting**: 3 seconds between requests
- **Series**: Gold spot price, WTI crude oil, other commodities
- **Config**: Base URL and series paths in `config/settings.yaml` and `config/platform_settings.yaml` under `macrotrends`
- **Note**: macrotrends has extended history reaching back to 1915 for some series, enabling pre-modern regime analysis

### yfinance (Yahoo Finance)

- **Service**: Yahoo Finance daily adjusted-close prices for equities and ETFs
- **What it's used for**: ETF/equity price history (SPY, TLT, GLD, QQQ, VNQ, AGG, etc.) for regime profiling and tactical asset classification
- **Method**: `yfinance` Python wrapper
- **Locations**: 
  - Legacy: `src/trading_crab_lib/ingestion/assets.py` (quarterly resampling)
  - Platform: `src/trading_crab_lib/platform/ingestion/prices_daily.py` (daily persistence + monthly spine)
- **SSL workaround**: Uses `curl_cffi` backend (direct import in assets.py) with SSL context control for environments with TLS interception
- **Rate limiting**: Chunked fetches (5 tickers per request), exponential backoff on 429 responses (max 3 retries)
- **Fallback chain**: yfinance → Stooq (Playwright/Selenium headless browser fallback) → last resort HTTP retry
- **Auth**: None required (free tier, but rate-limited)
- **Known limitation**: Commonly blocked by bot checks and TLS interception on corporate networks
- **ETF universe**: 16-38 tickers depending on pipeline phase (satellites + holdings + watchlist from `config/platform_settings.yaml`)

### Tiingo (REST API, optional premium)

- **Service**: Tiingo daily price and market data API (free tier: daily EOD for equities/ETFs)
- **What it's used for**: Daily prices as first-choice source in the price-ingestion fallback chain
- **Location**: `src/trading_crab_lib/platform/ingestion/tiingo.py`
- **Auth**: `TIINGO_API_KEY` environment variable (free registration at tiingo.com)
- **Method**: REST API with HTTP Authorization header (never URL-embedded for security)
- **Rate limiting**: 1 second between tickers, exponential backoff on 429 (max 3 retries, base 2 seconds)
- **Credential redaction**: Every occurrence of API key scrubbed from logs via `_redact()` helper
- **Fallback position**: First in the chain (before yfinance) because it doesn't require bot workarounds
- **Config**: Base URL allowlisted to prevent SSRF; API key resolved from env var or `cfg["tiingo"]["api_key"]`

### Browser Automation (Optional Extras)

- **Playwright** (primary engine): Headless browser for JavaScript-gated sources (required `playwright install chromium` post-install)
- **Selenium** (fallback engine): Alternative headless browser (requires ambient Chrome/Chromium + matching driver)
- **Purpose**: Scrape sources that serve JavaScript verification challenges (Stooq, Cloudflare-protected sites)
- **Location**: `src/trading_crab_lib/ingestion/browser.py`
- **Usage in platform**: `src/trading_crab_lib/platform/ingestion/prices_daily.py` falls back to Playwright/Selenium when yfinance is blocked

## Data Storage

**Databases:**
- **None** - No traditional database (SQLite, PostgreSQL, etc.)
- All data stored as **Parquet files** (columnar format with schema) under `data/` and `outputs/`
- **Checkpoint system**: `CheckpointManager` in `src/trading_crab_lib/checkpoints.py` manages save/load/freshness via JSON metadata (timestamps, config hash)

**File Storage:**
- **Local filesystem only** - No cloud storage (S3, GCS, etc.) in v0.1.5
- Directory structure: `data/{raw,processed,regimes,checkpoints}`, `outputs/{models,plots,reports}`
- Parquet files: `macro_raw.parquet`, `features.parquet`, `features_supervised.parquet`, `cluster_labels.parquet`, `profiles.parquet`
- Checkpoints: Timestamped parquet snapshots with JSON manifests
- Preservation checkpoints: `*_secondary` files survive `clear_all()` for audit trails
- Models: `outputs/models/*.pkl` (sklearn models via joblib, not pickle)

**Caching:**
- **None** - No Redis or in-memory cache
- Freshness checks: `CheckpointManager.is_fresh(name, max_age_days=7)` determines whether to recompute
- Default: 7-day cache lifetime for raw data; no expiration for derived checkpoints (user must `--refresh` or `--recompute`)

## Authentication & Identity

**Auth Provider:**
- **None** for core pipeline - All auth is API-key based
- **Email (SMTP)**: Optional; configured via `config/email.yaml` or `TC_SMTP_*` env vars
- **Identity**: No user authentication; single-user service (Glenn runs it manually)

**API Key Management:**
- FRED_API_KEY: Env var, `.env` file, or `cfg["fred"]["api_key"]` from settings
- TIINGO_API_KEY: Env var or `cfg["tiingo"]["api_key"]`
- All keys: Never logged in plaintext; scrubbed from exception messages
- `.env` file: Git-ignored, not committed; template: `.env.example`

**Security Notes:**
- No credentials in `config/settings.yaml` or `config/platform_settings.yaml`
- Tiingo API key redacted in logs via `_redact()` function
- SSL verification: Default enabled; override via `YFINANCE_VERIFY_SSL=false` only on TLS-intercepting networks (rare)
- No public API for external consumers; CLI-only in v0.1.5

## Monitoring & Observability

**Error Tracking:**
- **None** - No Sentry, Datadog, etc.
- Logging to stderr via stdlib `logging` module
- Log level: `INFO` by default, `DEBUG` with `--verbose` flag
- Ingestion failures: Logged at WARNING, pipeline continues with partial data (graceful degradation)

**Logs:**
- **Approach**: stdlib `logging` module, no third-party handlers
- **Format**: `"%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"`
- **Handlers**: Console (stderr) only; no file logging built-in
- **Rotation**: None (logs are ephemeral; each run starts fresh)
- **Verbosity**: Controlled by `RunConfig.apply_logging()` (DEBUG when `--verbose`)

## CI/CD & Deployment

**Hosting:**
- **None in v0.1.5** - Manual CLI runs on Glenn's workstation
- **Target environments**: Bare metal, VPS, Docker container, GitHub Actions runner
- **Automation target**: Cron job on server or GitHub Actions scheduled workflow (not yet live)

**CI Pipeline:**
- **GitHub Actions**: 
  - `.github/workflows/python-package.yml` - Multi-version testing (3.10-3.13), pytest with coverage
  - `.github/workflows/publish-lib.yml` - Publish library to PyPI on `lib-v*` tags
  - `.github/workflows/publish-app.yml` - Publish app to PyPI on `v*` tags
- **Tests**: 556+ test suite (unit + integration), ~769 collected when all optional deps installed
- **Coverage**: Pytest-cov used locally; CI doesn't enforce a minimum coverage threshold yet

**Deployment (Docker):**
- **Image**: Multi-stage Dockerfile with `base` (core libs only) and `pipeline` (full stack) stages
- **Entry point**: `tradingcrab` CLI command
- **Volumes**: `/app/{config,data,outputs}` for bind-mounting host directories
- **Secrets**: All passed via env vars (`FRED_API_KEY`, `TIINGO_API_KEY`, `TC_SMTP_*`, etc.); never baked into image
- **docker-compose.yml**: Three services (weekly-report, pipeline, notebook) for local orchestration
- **Artifact registry**: Docker Hub (future) or GitHub Container Registry

## Webhooks & Callbacks

**Incoming:**
- **None** - No webhook support in v0.1.5
- Pipeline is manual (CLI-driven) or cron-scheduled (automated via GitHub Actions or server cron)

**Outgoing:**
- **Email (SMTP)**: Optional weekly report delivery
  - Recipients: Configurable via `config/email.yaml` or `TC_EMAIL_*` env vars
  - Attachments: Optional inline plot images (multipart/related MIME)
  - Triggers: `--weekly-report --send-email` flags or `scripts/run_weekly_report.py`
  - TLS/SSL: Configurable (STARTTLS port 587 or implicit SSL port 465)

## Environment Configuration

**Required env vars:**
- `FRED_API_KEY` - FRED API key (free at fred.stlouisfed.org)

**Optional env vars:**
- `TIINGO_API_KEY` - Tiingo API key for daily prices (free tier at tiingo.com); omit to fall back to yfinance
- `YFINANCE_VERIFY_SSL` - Set to `false` to disable SSL verification (TLS-intercepting networks only)
- `TC_ROOT_DIR`, `TC_CONFIG_DIR`, `TC_DATA_DIR`, `TC_OUTPUT_DIR` - Override default paths (useful for Docker, CI)
- `TC_SMTP_HOST`, `TC_SMTP_PORT`, `TC_SMTP_USER`, `TC_SMTP_PASSWORD`, `TC_EMAIL_FROM`, `TC_EMAIL_TO`, `TC_EMAIL_USE_TLS`, `TC_EMAIL_USE_SSL` - Email configuration overrides
- `TC_CA_BUNDLE`, `SSL_CERT_FILE`, `REQUESTS_CA_BUNDLE`, `CURL_CA_BUNDLE`, `NODE_EXTRA_CA_CERTS` - CA certificate bundle overrides (TLS-intercepting networks)

**Secrets location:**
- `.env` file (git-ignored) - Primary source for local development
- Environment variables - For Docker, CI/CD, headless servers
- **Never commit** API keys, passwords, email addresses to git

## Data Flow & Service Dependencies

```
┌──────────────────────────────────────────────────────────────────┐
│                       External Services                          │
├─────────────────┬─────────────────┬──────────────┬───────────────┤
│   FRED API      │  multpl.com     │  macrotrends │  yfinance/    │
│   (quarterly,   │  (web scrape)   │  (web scrape)│  Tiingo/Stooq │
│    monthly,     │                 │              │  (daily price)│
│    daily via    │                 │              │               │
│    ALFRED)      │                 │              │               │
└────────┬────────┴────────┬────────┴──────────┬───┴───────────────┘
         │                 │                   │
         ▼                 ▼                   ▼
┌──────────────────────────────────────────────────────────────────┐
│  Ingestion (Step 1)                                              │
│  `platform/ingestion/{macro_monthly,macro_daily,prices_daily}`  │
│  → Merge into macro_raw.parquet + monthly_prices_raw.parquet   │
└────────┬─────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────────┐
│  Feature Engineering (Step 2)                                    │
│  `platform/transforms_monthly.py::features_from_raw`            │
│  → features.parquet + features_supervised.parquet               │
└────────┬─────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────────┐
│  Clustering (Step 3)                                             │
│  `platform/clustering.py`                                        │
│  → cluster_labels.parquet                                       │
└────────┬─────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────────┐
│  Regime Profiling & Prediction (Steps 4-7)                       │
│  → profiles.parquet, models/*.pkl, dashboard outputs            │
└────────┬─────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────────┐
│  Email (Optional)                                                │
│  `src/trading_crab_lib/email.py::send_weekly_email`             │
│  SMTP → weekly_report + attached plots                          │
└──────────────────────────────────────────────────────────────────┘
```

## Integration Resilience

**Graceful Degradation:**
- If any single data source fails (FRED, multpl, yfinance): Logged at WARNING, pipeline continues with whatever data was fetched
- If all price sources fail: Empty prices DataFrame; clustering and regime labeling continue (regimes remain unchanged if market codes available)
- If email config incomplete: Weekly report generated, email send skipped (logged at WARNING)

**Retry Logic:**
- **FRED**: No retry; single failure → WARNING + skip that series
- **yfinance**: Chunked fetches (5 tickers) with exponential backoff (max 3 retries, base 2 seconds, jitter)
- **Tiingo**: Per-ticker retry on 429 with exponential backoff
- **Web scraping** (multpl, macrotrends): Rate-limited delays (2-3 seconds); no automatic retry (single attempt)
- **SMTP**: No retry; failure → exception + stack trace

**Failure Modes:**
- Missing FRED_API_KEY at runtime → OSError raised (fail-fast)
- Missing Tiingo key → Falls back to yfinance
- Network timeouts → Caught and logged; ingestion continues with partial data
- Invalid config → ValueError raised at load time (fail-fast) with complete error list

---

*Integration audit: 2026-10-05*
