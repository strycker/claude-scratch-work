# Technology Stack

**Analysis Date:** 2026-03-16

## Languages

**Primary:**
- Python (>=3.10) - Core library in `src/market_regime/`, pipeline entrypoints in `run_pipeline.py` and `pipelines/*.py`

**Secondary:**
- Bash - Setup and helper scripts in `scripts/setup.sh` and `scripts/jupyter_notebook_local.sh`
- YAML - Runtime configuration in `config/settings.yaml` and regime naming in `config/regime_labels.yaml`

## Runtime

**Environment:**
- Python (>=3.10) - Declared in `pyproject.toml` (`requires-python = ">=3.10"`)

**Package Manager:**
- pip - Installation instructions and pinned minimum bounds via `requirements.txt` / `requirements-dev.txt`
- Lockfile: missing (no `poetry.lock`, `Pipfile.lock`, or `requirements-lock.txt` detected)

**Build backend / packaging:**
- setuptools - `pyproject.toml` uses `setuptools.backends.legacy:build` and `tool.setuptools.packages.find` with `where = ["src"]`

## Frameworks

**Core:**
- pandas (>=2.0) - Tabular time-series transformations across the pipeline (see `src/market_regime/transforms.py`, `src/market_regime/checkpoints.py`)
- scikit-learn (>=1.4) - PCA + clustering + classifiers (see `src/market_regime/clustering.py`, `src/market_regime/prediction.py`)
- scipy (>=1.11) - Numeric utilities for feature engineering (see `src/market_regime/transforms.py`)

**Testing:**
- pytest (>=8.0) - Test runner configured in `pyproject.toml` under `[tool.pytest.ini_options]`

**Build/Dev:**
- JupyterLab - Notebook environment listed in `pyproject.toml` optional deps `dev` and `requirements-dev.txt`

## Key Dependencies

**Critical:**
- pandas (>=2.0) - Primary DataFrame engine for quarterly resampling, joins, and feature tables (`src/market_regime/*`)
- numpy (>=1.25) - Numeric operations used throughout feature engineering and modeling (`src/market_regime/*`)
- pyarrow (>=14.0) - Parquet I/O for checkpoints and inter-step artifacts (`src/market_regime/checkpoints.py`, `run_pipeline.py`)
- pyyaml (>=6.0) - Settings loading (`src/market_regime/config.py`, `config/settings.yaml`)
- python-dotenv (>=1.0) - Loads `.env` for local development (`src/market_regime/config.py`)
- requests (>=2.31) - HTTP client for web scraping multpl.com (`src/market_regime/ingestion/multpl.py`)
- lxml (>=4.9) - HTML parsing/CSS selectors for multpl.com scraping (`src/market_regime/ingestion/multpl.py`)
- fredapi (>=0.5) - FRED macroeconomic series fetch (`src/market_regime/ingestion/fred.py`)
- yfinance (>=0.2) - ETF price history ingestion (`src/market_regime/ingestion/assets.py`)
- matplotlib (>=3.8), seaborn (>=0.13) - Plot generation (`src/market_regime/plotting.py`)

**Infrastructure:**
- certifi (>=2024.0) - Included to support TLS verification in some environments (declared in `pyproject.toml` / `requirements.txt`)

**Optional extras (declared, not required by default):**
- pandas-datareader (>=0.10) - stooq ETF price fallback (`src/market_regime/ingestion/assets.py`; extra: `data-extras` in `pyproject.toml`)
- openbb (>=4.1) - OpenBB ETF price fallback (`src/market_regime/ingestion/assets.py`; extra: `data-extras` in `pyproject.toml`)
- hdbscan (>=0.8) - Optional clustering backend (extra: `clustering-extras` in `pyproject.toml`)
- kneed (>=0.8) - Optional knee/elbow detection utility (extra: `clustering-extras` in `pyproject.toml`)
- k-means-constrained (optional install prompted by `scripts/setup.sh`) - Balanced-size clustering backend (see `scripts/setup.sh`, and CLI flag `--no-constrained` in `run_pipeline.py`)

## Configuration

**Environment:**
- `.env` loaded at runtime via `dotenv.load_dotenv()` in `src/market_regime/config.py`
- Secrets are expected via env vars (not committed); `.env.example` is the template

**Key configs required:**
- `FRED_API_KEY` - Required for FRED ingestion; injected into loaded config by `src/market_regime/config.py` and enforced by `src/market_regime/ingestion/fred.py`

**Build:**
- Packaging/config: `pyproject.toml`
- Runtime settings: `config/settings.yaml`
- Regime label pinning: `config/regime_labels.yaml`

## Platform Requirements

**Development:**
- Python 3.10+ (validated in `scripts/setup.sh`)
- Shell tooling for setup: bash (`scripts/setup.sh`)

**Production:**
- Not detected (no containerization or deployment manifests found; no `.github/workflows/*` CI config detected)

---

*Stack analysis: 2026-03-16*
