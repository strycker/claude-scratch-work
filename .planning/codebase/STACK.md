# Technology Stack

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


## Languages

**Primary:**
- Python 3.10+ - Core application language; tested on Python 3.10, 3.11, 3.12, 3.13. All `src/` modules use `from __future__ import annotations` for PEP 563 postponed evaluation.

**Secondary:**
- YAML 1.2 - Configuration files (`config/settings.yaml`, `config/platform_settings.yaml`, `config/email.example.yaml`, `config/regime_labels.yaml`, `config/portfolio.yaml`)
- Shell (Bash) - Setup and utility scripts (`scripts/setup.sh`, `scripts/jupyter_notebook_local.sh`, `scripts/run_weekly_report.py`)
- Dockerfile - Multi-stage container builds (`Dockerfile` with `base` and `pipeline` stages)

## Runtime

**Environment:**
- Python 3.10+ (minimal version, recommended 3.11+)
- pip (standard Python package manager, primary in CI/CD)
- uv (workspace-aware fast package manager, optional, recommended for local dev)
- Poetry (alternative package manager with lock file support, optional)

**Package Manager:**
- pip with `requirements.txt` (pinned minimums for reproducibility)
- `pyproject.toml` (PEP 517/518 build backend via setuptools; also read by Poetry and uv)
- Workspace configuration: uv workspace with `trading-crab-lib` as member; Poetry path dependency
- Lockfile: Optional via `pip-tools` (`requirements-lock.txt` when fully pinned is needed)

## Frameworks

**Core Data & ML:**
- pandas 2.0+ - DataFrames, time-series resampling, data manipulation
- numpy 1.25+ - Numerical arrays, mathematical operations, gradient computation
- scikit-learn 1.4+ - Clustering (KMeans, KMeansConstrained), RandomForest, DecisionTree, StandardScaler, PCA
- scipy 1.11+ - Interpolation (BPoly.from_derivatives for gap-filling), statistics

**Machine Learning (Optional):**
- hmmlearn 0.3+ - Gaussian Hidden Markov Models for regime detection (optional `[hmm]` extra)
- statsmodels 0.14+ - Markov regime-switching models, financial time-series analysis (optional `[hmm]` extra)
- lightgbm 4.0+ - Gradient boosting classifier, alternative to scikit-learn RF (optional `[boosting]` extra)
- hdbscan 0.8+ - Density-based clustering exploration (optional `[clustering-extras]` extra)
- kneed 0.8+ - Knee point detection for clustering k-sweep (optional `[clustering-extras]` extra)

**Visualization:**
- matplotlib 3.8+ - Base plotting library, all figure output (PNG/PDF)
- seaborn 0.13+ - Statistical visualization (heatmaps, violin plots, pair plots)

**Configuration & Environment:**
- PyYAML 6.0+ - YAML settings file parsing (`config/settings.yaml`, `platform_settings.yaml`)
- python-dotenv 1.0+ - Environment variable loading from `.env` file

**Data Serialization:**
- pyarrow 14.0+ - Parquet file I/O for checkpoint DataFrames (columnar compression, typed)
- joblib 1.3+ - Model serialization (sklearn models, replaced pickle for stability)

**Testing:**
- pytest 8.0+ - Test runner and fixtures
- pytest-cov 5.0+ - Coverage reporting

**Code Quality & Type Checking:**
- flake8 - Syntax error checking (E9, F63, F7, F82 only, via pre-commit)
- ruff 0.4+ - Fast Python linter with isort integration; rules: E, F, W, I, UP
- pylint 3.0+ - Additional static analysis (informational, not strict)
- mypy 1.0+ - Type checking (informational mode, not blocking CI yet)

**Interactive Development:**
- jupyterlab 4.0+ - Interactive notebooks for exploration and diagnostics (optional `[dev]` extra)
- ipykernel 6.0+ - Jupyter kernel for Python

**Data Ingestion (Optional):**
- fredapi 0.5+ - FRED API client for macroeconomic data (optional `[ingestion]` extra)
- requests 2.31+ - HTTP client for web requests (optional `[ingestion]` extra)
- lxml 4.9+ - Fast HTML/XML parsing for multpl.com scraping via cssselect (optional `[ingestion]` extra)
- cssselect 1.2+ - CSS-selector support for lxml (optional `[ingestion]` extra)
- beautifulsoup4 4.12+ - HTML parsing fallback (optional `[ingestion]` extra)
- yfinance 0.2+ - Yahoo Finance ETF/equity price history (optional `[ingestion]` extra)
- curl_cffi 0.5+ - HTTPS client for yfinance with SSL context control (optional `[ingestion]` extra)
- html5lib 1.1+ - HTML parser for pandas.read_html fallback (optional `[ingestion]` extra)
- certifi 2024.0+ - CA certificate bundle for SSL verification across all platforms

**Browser Automation (Optional):**
- playwright 1.40+ - Headless browser for JavaScript-gated sources (optional `[browser]` extra; requires `playwright install chromium`)
- selenium 4.15+ - Alternative headless browser fallback (optional `[browser]` extra; requires ambient Chromium + driver)

## Key Dependencies

**Critical (always required):**
- pandas 2.0+ - Time-series resampling, DataFrame operations
- numpy 1.25+ - Numerical computing, array operations
- scikit-learn 1.4+ - ML algorithms (KMeans, RF, PCA, StandardScaler)
- scipy 1.11+ - Interpolation (BPoly gap-filling), statistical distributions
- pyyaml 6.0+ - Configuration file parsing
- pyarrow 14.0+ - Parquet checkpoint persistence
- joblib 1.3+ - sklearn model serialization (replaces pickle)
- python-dotenv 1.0+ - `.env` file loading for secrets

**Ingestion (always required in practice, though technically optional):**
- fredapi 0.5+ - FRED API (macroeconomic data); fail-fast if missing
- requests 2.31+ - HTTP client for web scraping
- lxml 4.9+ - Fast HTML parsing for multpl.com via #datatable CSS selector
- cssselect 1.2+ - CSS-selector support for lxml
- yfinance 0.2+ - ETF/equity prices from Yahoo Finance
- curl_cffi 0.5+ - HTTPS client imported directly by ingestion/assets.py for SSL bypass workaround
- certifi 2024.0+ - CA certificates for SSL verification on all platforms

**Visualization (optional but included in full stack):**
- matplotlib 3.8+ - All matplotlib figures
- seaborn 0.13+ - Statistical visualization (heatmaps, violin plots)

**Machine Learning (optional):**
- hmmlearn 0.3+ - Gaussian HMM (conditional import in hmm.py)
- statsmodels 0.14+ - Markov switching models (conditional import in markov.py)
- lightgbm 4.0+ - LightGBM classifier (conditional import in prediction/gradient_boosting.py)
- hdbscan 0.8+ - HDBSCAN clustering (conditional import in density.py)
- kneed 0.8+ - Knee-point detection (conditional import in clustering.py)

**Why it matters:**
- **fredapi**: Primary macroeconomic data source; no fallback. Missing → ImportError at step 1.
- **scikit-learn, numpy, pandas**: Core data-science stack; everything depends on these.
- **pyarrow**: Checkpoint persistence; parquet is more stable than pickle for DataFrames.
- **joblib**: ML model serialization; used instead of pickle for sklearn objects.
- **FRED API key**: Required at runtime; `.env` file or environment variable.

## Configuration

**Environment:**
- `.env` file (git-ignored, not committed) - Secrets: `FRED_API_KEY`, `TIINGO_API_KEY`, SSL overrides (`YFINANCE_VERIFY_SSL`), email config (`TC_SMTP_*`, `TC_EMAIL_*`), path overrides (`TC_ROOT_DIR`, `TC_CONFIG_DIR`, `TC_DATA_DIR`, `TC_OUTPUT_DIR`)
- No secrets should be hardcoded; all via env vars

**Build & Package Configuration:**
- `pyproject.toml` (root) - Main app package metadata, entry points, dev extras, workspace config
- `src/trading_crab_lib/pyproject.toml` - Library package (independent PyPI publish), core + optional extras (`[ingestion]`, `[plotting]`, `[hmm]`, `[clustering-extras]`, `[boosting]`, `[browser]`, `[all]`, `[dev]`)
- `setup.cfg` / `.toml` - setuptools config via pyproject.toml (PEP 517/518)
- `.pre-commit-config.yaml` - Pre-commit hooks (ruff `--fix`, flake8 syntax, mypy informational)
- `.pylintrc` - pylint configuration (line-length 127, disabled false-positives for ML code)
- `pytest.ini` (via `pyproject.toml [tool.pytest.ini_options]`) - Test discovery, pythonpath, filter warnings

**Runtime Configuration:**
- `config/settings.yaml` - All tuneable parameters: FRED series, multpl.com URLs, macrotrends paths, feature lists, clustering k-sweep bounds, prediction horizons, CV splits, model hyperparameters. **Legacy quarterly pipeline configuration** (frozen).
- `config/platform_settings.yaml` - **Active monthly platform configuration** (Phase 08 development): monthly data layer, FRED monthly/daily series, ALFRED vintage settings, index closes, publication lags, universe holdings/satellites, Tiingo API config.
- `config/regime_labels.yaml` - Manually-pinned regime names (edited by hand after step 3 clustering; overrides auto-suggested names)
- `config/portfolio.yaml` - Asset weights for portfolio recommendations
- `config/email.example.yaml` - Email SMTP configuration template; copy to `email.local.yaml` (git-ignored) to enable weekly email delivery

**Python Logging:**
- Configured in `src/trading_crab_lib/config.py::setup_logging()`
- Format: `"%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"` with `datefmt="%Y-%m-%d %H:%M:%S"`
- Root logger set to `INFO` by default; `DEBUG` when `--verbose` flag passed
- No third-party logging framework (stdlib `logging` only)

**CI/CD Configuration:**
- `.github/workflows/python-package.yml` - Multi-version testing (Python 3.10-3.13), pytest with coverage
- `.github/workflows/python-app.yml` - Legacy single-version CI (deferred for removal)
- `.github/workflows/publish-lib.yml` - Publish library to PyPI on `lib-v*` tags
- `.github/workflows/publish-app.yml` - Publish app to PyPI on `v*` tags
- GitHub branch protections: main requires PR review, CI passing
- Secrets: `PYPI_API_TOKEN` (used by publish workflows)

## Platform Requirements

**Development (Local):**
- Python 3.10+ with pip (or uv, or Poetry)
- Git (for repository cloning and version control)
- C build tools (gcc/clang) for compiling scipy/lxml native extensions (requires `build-essential` on Ubuntu/Debian)
- libxml2-dev, libxslt1-dev, libssl-dev (system libraries for lxml on Linux)
- RAM: 2 GB minimum, 4 GB recommended (Jupyter notebooks + full pipeline)
- Disk: 500 MB for repo + dependencies; 1 GB+ for data/checkpoints/outputs (depends on data range)
- Network: FRED API (fredapi), multpl.com, macrotrends.net, yfinance (required for data ingestion)

**Production (Docker):**
- Docker Engine (any recent version supporting multi-stage builds)
- docker-compose (optional, for orchestrated multi-service runs)
- RAM: 1 GB minimum (pipeline runs sequentially, not memory-intensive)
- Disk: 500 MB total (config + data + outputs)
- Network: FRED API, multpl.com, macrotrends.net, yfinance, SMTP (if email enabled)

**Deployment Target (as of v0.1.5):**
- Bare metal / VPS (cron job or GitHub Actions runner)
- Docker container (single-shot weekly report via cron or GitHub Actions)
- Kubernetes (future; not yet containerized for multi-replica architectures)

## Build & Distribution

**Library Package (`trading-crab-lib`):**
- pip name: `trading-crab-lib`
- Published to PyPI on `lib-v*` git tags via GitHub Actions
- Installable with pip, uv, or Poetry
- Extras: `[ingestion]`, `[plotting]`, `[hmm]`, `[clustering-extras]`, `[boosting]`, `[browser]`, `[all]`, `[dev]`
- Build: `python -m build` or `pip install build && python -m build`
- Wheel + sdist distribution

**App Package (`trading-crab`):**
- pip name: `trading-crab`
- Published to PyPI on `v*` git tags via GitHub Actions
- Depends on `trading-crab-lib>=0.1.5`
- Entry points: `tradingcrab` (main CLI), `tradingcrab-setup` (setup helper), `tradingcrab-publish` (notebook publishing)
- Build: `python -m build` or via Poetry/uv

**Docker Images:**
- `trading-crab:base` - Core library (pandas, sklearn, scipy) only; no ingestion/plotting
- `trading-crab:pipeline` (default) - Full stack (ingestion, plotting, boosting, CLI)
- Multi-stage build reduces image size by separating compile-time deps from runtime
- All secrets passed via env vars at runtime, never baked into image

---

*Stack analysis: 2026-10-05*
