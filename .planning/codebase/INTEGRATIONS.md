# External Integrations

**Analysis Date:** 2026-07-09

## APIs & External Services

**Federal Reserve Economic Data (FRED):**
- Service: US Economic Indicators (gdp, gnp, cpi, interest rates, unemployment, etc.)
- SDK/Client: `fredapi >= 0.5` (via `src/trading_crab_lib/ingestion/fred.py`)
- Auth: `FRED_API_KEY` environment variable (free registration at fred.stlouisfed.org/docs/api/api_key.html)
- Implementation: Parallel fetch (ThreadPoolExecutor, 8 workers max) with per-series publication-lag shift capability
- Series count: 16 configured in `config/settings.yaml` (GDP, GNP, BAA, AAA, CPI, GS10, TB3MS, VIXCLS, UNRATE, M2SL, M2NS, GS2, T10Y2Y, T10Y3M, HOUST, UMCSENT, INDPRO, PAYEMS)
- Resampling: Daily/monthly data resampled to quarterly (Q-E, period-end)
- Publication lag: GDP and GNP automatically shifted +1 quarter to prevent look-ahead bias
- Rate limiting: Tolerant of small parallel bursts; max 8 concurrent requests

**multpl.com (Market multiples scraper):**
- Service: Historical market data (S&P 500 levels, dividend yields, price-to-GDP, credit spreads, etc.)
- SDK/Client: `lxml >= 4.9`, `requests >= 2.31` via CSS selectors
- Implementation: HTML scraping with `lxml.html.cssselect("#datatable tr")`, per-request 2-second rate limiting
- Series count: 46 datasets configured in `config/settings.yaml`
- User-Agent: Chrome 63.0.3239.108 on Linux (mimics browser to avoid blocking)
- Rate limiting: 2 seconds per request (RATE_LIMIT_SECONDS constant) — never reduce below 2 without approval
- Parsing: Value type conversions (num → float, percent → float/100, million/trillion → parse suffix)
- Fallback: Silent NaN on parse errors (logged at WARNING, pipeline continues)

**macrotrends.net (Long-history commodity prices):**
- Service: Gold prices (1915+), WTI crude oil (1946+), silver, copper
- SDK/Client: `requests >= 2.31` with JSON regex extraction via `src/trading_crab_lib/ingestion/macrotrends.py`
- Implementation: Extracts embedded JavaScript `var defined_data = [...]` from page source using regex
- Rate limiting: 3 seconds per request (slower than multpl.com, polite scraping)
- Data format: Monthly prices extracted from JSON blob, resampled to quarterly
- Fallback: Empty DataFrame on scrape failure (triggers macro-proxy fallback in downstream code)

**Yahoo Finance (ETF / Equity prices):**
- Service: Historical adjusted-close prices for 16+ ETFs (SPY, TLT, GLD, QQQ, VNQ, AGG, USO, IWM, etc.)
- SDK/Client: `yfinance >= 0.2` via `src/trading_crab_lib/ingestion/assets.py`
- Implementation: Phase 1 — yfinance batch download (single HTTP request for all tickers)
- Fallback chain:
  1. yfinance batch (curl_cffi backend)
  2. yfinance per-ticker with SSL verification disabled
  3. pandas-datareader stooq (if installed)
  4. OpenBB (if installed)
  5. Empty DataFrame (uses macro proxies instead)
- Rate limiting: Batch download in Phase 1 minimizes "Too Many Requests" errors
- SSL verification: Disabled for all yfinance calls (curl_cffi does not respect CURL_CA_BUNDLE)
- Data: Monthly adjusted close, resampled to quarterly (Q-E, period-end)
- Pre-1993 data: Not available; pipeline gracefully falls back to macro proxies for regimes before 1993

## Data Storage

**Checkpoints (Parquet):**
- Location: `data/checkpoints/{name}.parquet` (DataFrames only, not models)
- Types: `macro_raw`, `macro_raw_secondary`, `features`, `features_secondary`, `features_supervised`, `features_supervised_secondary`, `cluster_labels`, `kmeans_scores`, `forward_probabilities`
- Preservation checkpoints: `*_secondary` files survive `clear_all()` to retain full column audit trail
- Manifest: `{name}.manifest.json` alongside each parquet file records creation timestamp, config hash, row/column counts
- Freshness: `CheckpointManager.is_fresh(name, max_age_days=7)` checks wall-clock age (not data age; always use `--refresh` on Fridays for production)
- Clearing: `clear_all()` removes primary checkpoints; `include_preservation=True` also removes `*_secondary` files

**Models (Pickle via joblib):**
- Location: `outputs/models/{name}.pkl` (sklearn models only)
- Examples: `current_regime.pkl` (RandomForestClassifier), `decision_tree.pkl`, `lightgbm_model.pkl`
- Serialization: `joblib.dump()` / `joblib.load()` (more stable than pickle.dump across Python versions)
- Lifecycle: Overwritten on each pipeline step 5 run unless checkpointing prevents re-fit

**Raw Data (Parquet):**
- Location: `data/raw/macro_raw.parquet`, `data/raw/asset_prices.parquet`
- Created: Step 1 (ingestion)
- Retention: Persistent across pipeline runs; re-fetched only with `--refresh`

**Processed Data:**
- Location: `data/processed/features.parquet`, `data/processed/features_supervised.parquet`
- Created: Step 2 (feature engineering)
- Centered vs causal: `features` uses both-sided rolling windows (for clustering); `features_supervised` uses only past data (for model training)

**Regimes:**
- Location: `data/regimes/cluster_labels.parquet`, `data/regimes/regime_profiles.parquet`, `data/regimes/transition_matrix.parquet`
- Created: Steps 3–4
- Invalidation: Changing `clustering_features` in `settings.yaml` requires re-run of steps 3–7

**Outputs:**
- Plots: `outputs/plots/{step}_{description}.png`
- Reports: `outputs/reports/dashboard.csv`, `outputs/reports/weekly_report.md`, `outputs/reports/email_body.txt`
- Diagnostics: `outputs/reports/diagnostics/{rrg.parquet, tactics.parquet}`

**File Storage (Local):**
- No cloud storage integration; all files reside on local filesystem
- Docker: Mounted via `volumes` in docker-compose.yml (host directories)

**Caching:**
- In-memory: pandas/numpy operations during a single pipeline run
- Disk: All intermediate checkpoints via parquet + manifest
- No distributed cache (Memcached, Redis) — single-machine pipeline

## Authentication & Identity

**FRED API:**
- Auth type: API key (free, registered account required)
- Env var: `FRED_API_KEY`
- Setup: Free registration at fred.stlouisfed.org/docs/api/api_key.html
- Scope: Read-only access to all FRED series
- No per-series quotas documented; rate-limit tolerance is unknown but observed to be high

**multpl.com & macrotrends.net:**
- Auth type: None (public HTTP scraping)
- Rate limiting: Polite delays (2-3 seconds per request)

**Email (SMTP):**
- Auth type: Username + password (plaintext in YAML or env vars, transport via TLS/SSL)
- Config file: `config/email.local.yaml` (git-ignored, copy from `config/email.example.yaml`)
- Env var overrides: `TC_SMTP_HOST`, `TC_SMTP_PORT`, `TC_SMTP_USER`, `TC_SMTP_PASSWORD`, `TC_EMAIL_FROM`, `TC_EMAIL_TO`, `TC_EMAIL_USE_TLS`, `TC_EMAIL_USE_SSL`
- TLS/SSL: Configurable per email setup (default TLS on port 587)
- Implementation: `smtplib` (stdlib) with `email.mime` for multipart messages

## Monitoring & Observability

**Error Tracking:**
- None (no external service integration)
- Local handling: Try-catch with logging to stderr/file via `logging` module

**Logs:**
- Approach: Python `logging` module configured in `config.py`
- Log level: INFO by default, DEBUG with `--verbose` flag
- Output: stdout/stderr (no file rotation)
- No external aggregation (Splunk, DataDog, etc.)

**Metrics & Monitoring:**
- Pipeline step timing: Tracked with `time.monotonic()` in `run_pipeline.py`, printed at step completion
- Step validation: `validate_step_output()` in `monitoring.py` checks DataFrame shape and NaN fraction
- CV fold scores: Per-fold accuracy logged as formatted table
- Regime stability: Persistence probabilities + average run length per regime

## CI/CD & Deployment

**Hosting:**
- PyPI (packages published via GitHub Actions)
- GitHub (source repository, releases tagged for publishing)
- Local / Docker (primary deployment target)

**CI Pipeline:**
- Service: GitHub Actions (`.github/workflows/python-package.yml`)
- Triggers: Every push/PR to `main` branch
- Matrix: Python 3.10, 3.11, 3.12, 3.13
- Steps:
  1. Checkout (exclude submodules as they are read-only references)
  2. Set up Python (matrix version)
  3. Install: library + app + all extras (ingestion, plotting, hmm, clustering-extras, boosting)
  4. Lint: flake8 (E9, F63, F7, F82), ruff (E, F, W, I, UP), pylint (informational)
  5. Test: pytest with all optional dependencies
  6. Type-check: mypy (informational, not strict)
- Artifacts: None saved (tests only)

**CI — Package Build Verification:**
- Service: GitHub Actions (`build-pkg` job in `python-package.yml`)
- Builds: Both `trading-crab-lib` (sdist + wheel) and `trading-crab` (sdist + wheel)
- Verification: Both packages build cleanly without warnings

**Publishing:**
- Service: GitHub Actions (`.github/workflows/publish-pypi.yml`)
- Trigger: Git tags (`v*` for app, `lib-v*` for lib, `both-v*` for both)
- Alternative: Manual workflow dispatch (choose `app` / `lib` / `both`)
- Credentials: `PYPI_LIB_TOKEN` and `PYPI_APP_TOKEN` (GitHub repo secrets, scoped to respective packages)
- Upload: `twine upload --skip-existing` (prevents duplicate uploads)
- Matrix: Separate jobs for each package, but only publishes if tag matches

## Webhooks & Callbacks

**Incoming:**
- None (pipeline is a pull system, not push-triggered)

**Outgoing:**
- Email delivery (weekly report via SMTP)
- No GitHub/GitLab webhooks
- No third-party service callbacks

## Environment Configuration

**Required env vars (production pipeline):**
- `FRED_API_KEY` - Must be set to run steps 1 (ingestion fails without it)

**Optional env vars (deployment paths):**
- `TC_ROOT_DIR` - Override repo root path (defaults to package-discovered path)
- `TC_CONFIG_DIR` - Override config directory (defaults to `{TC_ROOT_DIR}/config`)
- `TC_DATA_DIR` - Override data directory (defaults to `{TC_ROOT_DIR}/data`)
- `TC_OUTPUT_DIR` - Override output directory (defaults to `{TC_ROOT_DIR}/outputs`)

**Optional env vars (email delivery):**
- `TC_SMTP_HOST` - Overrides `config/email.yaml` smtp_host
- `TC_SMTP_PORT` - Overrides `config/email.yaml` smtp_port
- `TC_SMTP_USER` - Overrides `config/email.yaml` username
- `TC_SMTP_PASSWORD` - Overrides `config/email.yaml` password
- `TC_EMAIL_FROM` - Overrides `config/email.yaml` from_address
- `TC_EMAIL_TO` - Overrides `config/email.yaml` to_address
- `TC_EMAIL_USE_TLS` - Overrides `config/email.yaml` use_tls (true/false/1/0/yes/no)
- `TC_EMAIL_USE_SSL` - Overrides `config/email.yaml` use_ssl

**Secrets location:**
- `.env` file (git-ignored, never committed)
- `config/email.local.yaml` (git-ignored, copy from `config/email.example.yaml`)
- Docker: All secrets passed via `env_file: .env` in `docker-compose.yml`
- Never hardcode secrets in source code, config files, or Dockerfile

## Docker Integration

**Container Images:**
- Base stage: `trading-crab:base` (library + core deps only, lightweight)
- Pipeline stage: `trading-crab:latest` (full install with all extras, default)

**Image registry:**
- None configured (images built locally, pushed manually if needed)

**Runtime Environment:**
- `PYTHONUNBUFFERED=1` (unbuffered stdout/stderr for real-time logging)
- `PYTHONDONTWRITEBYTECODE=1` (skip .pyc file generation)
- `FRED_API_KEY`, `TC_*`, `TC_SMTP_*`, `TC_EMAIL_*` passed via `env_file: .env`

**Volume Mounts (docker-compose):**
- `config/:/app/config:ro` (read-only, host manages settings.yaml)
- `data/:/app/data` (read-write, pipeline writes checkpoints)
- `outputs/:/app/outputs` (read-write, pipeline writes plots and reports)
- `notebooks/:/app/notebooks` (read-write for notebook service only)

**Services (docker-compose.yml):**
- `weekly-report` — one-shot service for cron scheduling (runs full pipeline + email)
- `pipeline` — interactive runner for ad-hoc step subsets
- `notebook` — JupyterLab server on port 8888

---

*Integration audit: 2026-07-09*
