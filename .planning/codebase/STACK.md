# Technology Stack

**Analysis Date:** 2026-07-09

## Languages

**Primary:**
- Python 3.10+ - Core application language; tested on Python 3.10, 3.11, 3.12, 3.13

**Secondary:**
- YAML - Configuration files (`config/settings.yaml`, `config/email.example.yaml`, `config/regime_labels.yaml`)
- Shell (Bash) - Setup and utility scripts (`scripts/setup.sh`, `scripts/jupyter_notebook_local.sh`)

## Runtime

**Environment:**
- Python 3.10+ (minimal version)
- pip (standard Python package manager)
- uv (workspace-aware package manager, optional but recommended)
- Poetry (alternative package manager with lock file support, optional)

**Package Manager:**
- pip (primary, used in CI/CD and Docker)
- Lockfile: `requirements.txt` (pinned minimums), `requirements-dev.txt` (development extras)
- Workspace: `pyproject.toml` with `[tool.uv.workspace]` for multi-package builds
- Poetry: `pyproject.toml` has `[tool.poetry]` section for Poetry users

## Frameworks

**Core ML / Data Science:**
- scikit-learn 1.4+ - Clustering (KMeans, KMeansConstrained), RandomForest, DecisionTree, StandardScaler, PCA
- scipy 1.11+ - Interpolation (BPoly.from_derivatives for gap-filling), statistics
- pandas 2.0+ - DataFrames, time-series resampling, data manipulation
- numpy 1.25+ - Numerical arrays, mathematical operations

**Optional ML Extensions:**
- hmmlearn 0.3+ - Gaussian Hidden Markov Models for regime detection (`src/trading_crab_lib/hmm.py`)
- statsmodels 0.14+ - Markov regime-switching models (`src/trading_crab_lib/markov.py`)
- lightgbm 4.0+ - Gradient boosting classifier (optional alternative to scikit-learn)
- hdbscan 0.8+ - Density-based clustering exploration
- kneed 0.8+ - Knee point detection for clustering k-sweep

**Visualization:**
- matplotlib 3.8+ - Base plotting library, all figure output
- seaborn 0.13+ - Statistical visualization (heatmaps, violin plots, etc.)

**Configuration & Secrets:**
- PyYAML 6.0+ - Settings file parsing (`config/settings.yaml`)
- python-dotenv 1.0+ - Environment variable loading from `.env` file

**Serialization & Checkpointing:**
- pyarrow 14.0+ - Parquet file I/O (DataFrame checkpointing)
- joblib 1.3+ - Model serialization (sklearn models) — switched from pickle for improved stability

**Testing:**
- pytest 8.0+ - Test runner and fixtures
- pytest-cov 5.0+ - Coverage reporting
- flake8 - Syntax error checking (E9, F63, F7, F82 only)
- ruff 0.4+ - Fast Python linter (E, F, W, I, UP rules)
- pylint 3.0+ - Additional static analysis (informational in CI)
- mypy 1.0+ - Type checking (informational, not strict yet)

**Notebooks:**
- jupyterlab 4.0+ - Interactive notebooks for exploration and diagnostics
- ipykernel 6.0+ - Jupyter kernel for Python

## Key Dependencies

**Critical (in core `[project.dependencies]`):**
- trading-crab-lib >= 0.1.2 - Library package (for app); installed via path in `src/trading_crab_lib/`
- pandas >= 2.0 - DataFrames, time-series operations
- numpy >= 1.25 - Numerical computing
- scikit-learn >= 1.4 - ML algorithms (KMeans, RF, PCA, StandardScaler)
- scipy >= 1.11 - Interpolation, statistics

**Data Ingestion (optional, `[ingestion]` extra):**
- fredapi >= 0.5 - Federal Reserve Economic Data (FRED) API client
- requests >= 2.31 - HTTP client for web scraping
- lxml >= 4.9 - Fast HTML/XML parsing (multpl.com scraper)
- beautifulsoup4 >= 4.12 - HTML parsing (alternative/fallback)
- yfinance >= 0.2 - Yahoo Finance price data (ETF/equity prices)
- certifi >= 2024.0 - CA certificate bundle for SSL verification

**Visualization (optional, `[plotting]` extra):**
- matplotlib >= 3.8 - All plotting
- seaborn >= 0.13 - Statistical visualization

**Advanced Clustering (optional, `[clustering-extras]` extra):**
- hdbscan >= 0.8 - Density-based clustering (experimental)
- kneed >= 0.8 - Knee point detection for elbow curves

**Hidden Markov Models (optional, `[hmm]` extra):**
- hmmlearn >= 0.3 - GaussianHMM implementation
- statsmodels >= 0.14 - Markov regime-switching models

**Gradient Boosting (optional, `[boosting]` extra):**
- lightgbm >= 4.0 - LightGBM classifier (alternative to RF)

**Infrastructure:**
- joblib >= 1.3 - Model serialization (replaces pickle for sklearn objects)
- pyarrow >= 14.0 - Parquet file format (checkpoint I/O)
- pyyaml >= 6.0 - YAML config parsing
- python-dotenv >= 1.0 - Environment variable loading

## Configuration

**Environment:**
- `.env` file (not committed) - Secrets: `FRED_API_KEY`, `TC_SMTP_*`, `TC_EMAIL_*`
- Env var overrides: `TC_ROOT_DIR`, `TC_CONFIG_DIR`, `TC_DATA_DIR`, `TC_OUTPUT_DIR`
- Env var email config: `TC_SMTP_HOST`, `TC_SMTP_PORT`, `TC_SMTP_USER`, `TC_SMTP_PASSWORD`, `TC_EMAIL_FROM`, `TC_EMAIL_TO`, `TC_EMAIL_USE_TLS`, `TC_EMAIL_USE_SSL`
- Python logging: configured via `config.py` with `DEBUG` level when `--verbose` is passed

**Build:**
- `pyproject.toml` (main) - Entry points, dev extras, workspace config for app package
- `src/trading_crab_lib/pyproject.toml` (library) - Core + optional extras, independent package metadata
- `setup.cfg` / `.toml` config - setuptools uses PEP 518 build backend
- `Dockerfile` - Multi-stage build (base + pipeline stages), all secrets via env vars
- `docker-compose.yml` - Three services (weekly-report, pipeline, notebook)
- `MANIFEST.in` - Package data includes; Python source files auto-discovered

**Runtime Config:**
- `config/settings.yaml` - All tuneable parameters (FRED series, multpl URLs, feature lists, clustering k-sweep, etc.)
- `config/regime_labels.yaml` - Manually-pinned regime names after clustering (edited by hand post-step-3)
- `config/portfolio.yaml` - Asset weights for portfolio recommendations
- `config/email.example.yaml` - Email SMTP configuration template (copy to `email.local.yaml` to enable)

## Platform Requirements

**Development:**
- Python 3.10+ with pip
- Git (for repository cloning)
- Optional: `build` package for wheel building (`pip install build`)
- Optional: `poetry` for lock file management
- Optional: `uv` for fast package installs and workspace management
- Optional: Docker & docker-compose for containerized runs

**Production / Deployment:**
- Python 3.10+ (tested on 3.10-3.13)
- Parquet support: pyarrow or fastparquet
- Network access: FRED API (fredapi), multpl.com, macrotrends.net, yfinance, SMTP (email delivery)
- Disk space: ~100 MB for raw data + processed checkpoints (depending on data range); ~50 MB for outputs

**Docker Deployment:**
- Docker Engine (any recent version supporting multi-stage builds)
- docker-compose (optional, for orchestrated multi-service runs)
- Memory: 1 GB minimum (2 GB recommended for full pipeline with notebooks)
- Storage: 500 MB total (config + data + outputs, depends on output lifespan)

---

*Stack analysis: 2026-07-09*
