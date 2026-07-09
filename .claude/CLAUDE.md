<!-- GSD:project-start source:PROJECT.md -->

## Project

**Trading-Crab**

A regime-conditional investment platform that simulates the decision process of a
hedge fund or expert trader — predicting market conditions (regimes), forecasting
asset behavior conditional on those regimes, and producing weekly portfolio guidance
(target mix, trades implied, stops, crash-risk dashboard) that Glenn executes manually
in Fidelity. Guidance-only by design: the platform recommends, the human trades.

The authoritative design is `platform_design/platform_design.md` (v1.7, 2026-07-08) —
a five-layer architecture (L0 data → L1 regime labeling → L2 regime prediction →
L3 asset prediction → L4 allocation & tactics) wrapped in an honesty framework
(walk-forward everything, purged CV, trial registry, locked 2021+ holdout, deflated
Sharpe). The build philosophy is the tracer bullet (§14): a "fully operational battle
station" — every layer present, every layer naive — usable from the first milestone,
then upgraded module by module against frozen interfaces.

**Core Value:** Honest, regime-aware weekly guidance that beats buy-and-hold SPY **net of avoided
drawdowns** — and is never fooled by its own backtest. The honesty framework is not
overhead; it is the product. A beautiful but leaky backtest is worthless.

### Constraints

- **Tech stack**: Python 3.10+, existing two-package src layout, config in `settings.yaml`, parquet checkpoints — extend, don't rewrite (design R15)
- **Data**: free sources only in v1; daily USD price series spliced per documented rules; ALFRED for point-in-time agency data
- **Execution venue**: Fidelity, long-only, no options/shorts/crypto/MLPs
- **Honesty discipline**: 2021+ holdout locked — development/tuning/model-selection use only ≤2020-12 walk-forward results; live weekly scoring refits on full history but its post-2021 performance is firewalled from all selection decisions until design freeze; every evaluated configuration goes in the trial registry
- **Cadence**: monthly modeling spine, weekly scoring, manual CLI runs in v1 (automation is a tracked placeholder)

<!-- GSD:project-end -->

<!-- GSD:stack-start source:codebase/STACK.md -->

## Technology Stack

## Languages

- Python 3.10+ - Core application language; tested on Python 3.10, 3.11, 3.12, 3.13
- YAML - Configuration files (`config/settings.yaml`, `config/email.example.yaml`, `config/regime_labels.yaml`)
- Shell (Bash) - Setup and utility scripts (`scripts/setup.sh`, `scripts/jupyter_notebook_local.sh`)

## Runtime

- Python 3.10+ (minimal version)
- pip (standard Python package manager)
- uv (workspace-aware package manager, optional but recommended)
- Poetry (alternative package manager with lock file support, optional)
- pip (primary, used in CI/CD and Docker)
- Lockfile: `requirements.txt` (pinned minimums), `requirements-dev.txt` (development extras)
- Workspace: `pyproject.toml` with `[tool.uv.workspace]` for multi-package builds
- Poetry: `pyproject.toml` has `[tool.poetry]` section for Poetry users

## Frameworks

- scikit-learn 1.4+ - Clustering (KMeans, KMeansConstrained), RandomForest, DecisionTree, StandardScaler, PCA
- scipy 1.11+ - Interpolation (BPoly.from_derivatives for gap-filling), statistics
- pandas 2.0+ - DataFrames, time-series resampling, data manipulation
- numpy 1.25+ - Numerical arrays, mathematical operations
- hmmlearn 0.3+ - Gaussian Hidden Markov Models for regime detection (`src/trading_crab_lib/hmm.py`)
- statsmodels 0.14+ - Markov regime-switching models (`src/trading_crab_lib/markov.py`)
- lightgbm 4.0+ - Gradient boosting classifier (optional alternative to scikit-learn)
- hdbscan 0.8+ - Density-based clustering exploration
- kneed 0.8+ - Knee point detection for clustering k-sweep
- matplotlib 3.8+ - Base plotting library, all figure output
- seaborn 0.13+ - Statistical visualization (heatmaps, violin plots, etc.)
- PyYAML 6.0+ - Settings file parsing (`config/settings.yaml`)
- python-dotenv 1.0+ - Environment variable loading from `.env` file
- pyarrow 14.0+ - Parquet file I/O (DataFrame checkpointing)
- joblib 1.3+ - Model serialization (sklearn models) — switched from pickle for improved stability
- pytest 8.0+ - Test runner and fixtures
- pytest-cov 5.0+ - Coverage reporting
- flake8 - Syntax error checking (E9, F63, F7, F82 only)
- ruff 0.4+ - Fast Python linter (E, F, W, I, UP rules)
- pylint 3.0+ - Additional static analysis (informational in CI)
- mypy 1.0+ - Type checking (informational, not strict yet)
- jupyterlab 4.0+ - Interactive notebooks for exploration and diagnostics
- ipykernel 6.0+ - Jupyter kernel for Python

## Key Dependencies

- trading-crab-lib >= 0.1.2 - Library package (for app); installed via path in `src/trading_crab_lib/`
- pandas >= 2.0 - DataFrames, time-series operations
- numpy >= 1.25 - Numerical computing
- scikit-learn >= 1.4 - ML algorithms (KMeans, RF, PCA, StandardScaler)
- scipy >= 1.11 - Interpolation, statistics
- fredapi >= 0.5 - Federal Reserve Economic Data (FRED) API client
- requests >= 2.31 - HTTP client for web scraping
- lxml >= 4.9 - Fast HTML/XML parsing (multpl.com scraper)
- beautifulsoup4 >= 4.12 - HTML parsing (alternative/fallback)
- yfinance >= 0.2 - Yahoo Finance price data (ETF/equity prices)
- certifi >= 2024.0 - CA certificate bundle for SSL verification
- matplotlib >= 3.8 - All plotting
- seaborn >= 0.13 - Statistical visualization
- hdbscan >= 0.8 - Density-based clustering (experimental)
- kneed >= 0.8 - Knee point detection for elbow curves
- hmmlearn >= 0.3 - GaussianHMM implementation
- statsmodels >= 0.14 - Markov regime-switching models
- lightgbm >= 4.0 - LightGBM classifier (alternative to RF)
- joblib >= 1.3 - Model serialization (replaces pickle for sklearn objects)
- pyarrow >= 14.0 - Parquet file format (checkpoint I/O)
- pyyaml >= 6.0 - YAML config parsing
- python-dotenv >= 1.0 - Environment variable loading

## Configuration

- `.env` file (not committed) - Secrets: `FRED_API_KEY`, `TC_SMTP_*`, `TC_EMAIL_*`
- Env var overrides: `TC_ROOT_DIR`, `TC_CONFIG_DIR`, `TC_DATA_DIR`, `TC_OUTPUT_DIR`
- Env var email config: `TC_SMTP_HOST`, `TC_SMTP_PORT`, `TC_SMTP_USER`, `TC_SMTP_PASSWORD`, `TC_EMAIL_FROM`, `TC_EMAIL_TO`, `TC_EMAIL_USE_TLS`, `TC_EMAIL_USE_SSL`
- Python logging: configured via `config.py` with `DEBUG` level when `--verbose` is passed
- `pyproject.toml` (main) - Entry points, dev extras, workspace config for app package
- `src/trading_crab_lib/pyproject.toml` (library) - Core + optional extras, independent package metadata
- `setup.cfg` / `.toml` config - setuptools uses PEP 518 build backend
- `Dockerfile` - Multi-stage build (base + pipeline stages), all secrets via env vars
- `docker-compose.yml` - Three services (weekly-report, pipeline, notebook)
- `MANIFEST.in` - Package data includes; Python source files auto-discovered
- `config/settings.yaml` - All tuneable parameters (FRED series, multpl URLs, feature lists, clustering k-sweep, etc.)
- `config/regime_labels.yaml` - Manually-pinned regime names after clustering (edited by hand post-step-3)
- `config/portfolio.yaml` - Asset weights for portfolio recommendations
- `config/email.example.yaml` - Email SMTP configuration template (copy to `email.local.yaml` to enable)

## Platform Requirements

- Python 3.10+ with pip
- Git (for repository cloning)
- Optional: `build` package for wheel building (`pip install build`)
- Optional: `poetry` for lock file management
- Optional: `uv` for fast package installs and workspace management
- Optional: Docker & docker-compose for containerized runs
- Python 3.10+ (tested on 3.10-3.13)
- Parquet support: pyarrow or fastparquet
- Network access: FRED API (fredapi), multpl.com, macrotrends.net, yfinance, SMTP (email delivery)
- Disk space: ~100 MB for raw data + processed checkpoints (depending on data range); ~50 MB for outputs
- Docker Engine (any recent version supporting multi-stage builds)
- docker-compose (optional, for orchestrated multi-service runs)
- Memory: 1 GB minimum (2 GB recommended for full pipeline with notebooks)
- Storage: 500 MB total (config + data + outputs, depends on output lifespan)

<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->

## Conventions

## Naming Patterns

- **Library modules** (in `src/trading_crab_lib/`): lowercase with underscores (`transforms.py`, `checkpoints.py`, `regime.py`)
- **Pipeline steps** (in `pipelines/`): numbered prefix `NN_descriptive_name.py` (e.g., `01_ingest.py`, `02_features.py`)
- **Test files**: `test_<module>.py` for unit tests in `tests/unit/`; `test_<feature>.py` for integration tests in `tests/integration/`
- **CLI and app modules** (in `src/trading_crab/`): lowercase with underscores (`cli.py`, `pipeline.py`)
- Verb + noun pattern: `fetch_all()`, `apply_log_transforms()`, `build_profiles()`, `fit_clusters()`
- Helper functions prefixed with underscore: `_fetch_one()`, `_config_hash()`, `_get_nested()`
- Boolean predicates: `is_fresh()`, `should_write()`, `preservation_checkpoint_should_write()`
- Test functions: `test_<behavior>()` with explicit behavior description (e.g., `test_gap_fill_idempotent()`, `test_market_code_not_filled()`)
- DataFrame variables: noun describing contents (`features`, `pca_df`, `clustered`, `returns`, `profiles`)
- Series variables: noun for single values (`labels`, `cluster`, `sp500_prices`, `transitions`)
- Config/dict variables: `cfg`, `meta`, `frames`, `results`
- Loop indices: single letters accepted only for math/ML contexts (`i`, `j`, `k`, `n`, `q`, `v`, `t`, `h`) — never used for meaningful data (see `.pylintrc` `good-names`)
- Temporary/intermediate: typically short-lived (`tmp_path`, `session_dir`, `result`, `df`)
- Type aliases: avoided; use explicit union types (`X | None` instead of `Optional[X]`)
- Type hints on all public functions (enforced by ruff + mypy)
- `cls` in classmethods left unannotated per Python convention
- Generic objects from optional dependencies: `model: object`, `pca_obj: object` (used when type cannot be imported)
- Module-level: `UPPER_CASE` (e.g., `_MAX_WORKERS`, `PRESERVATION_CHECKPOINT_NAMES`, `CHECKPOINT_DIR`)
- Availability flags: `_HMM_AVAILABLE`, `_STATSMODELS_AVAILABLE`, `HAS_LIGHTGBM` assigned in try/except blocks
- Private availability flags allowed by pylint via `.pylintrc` `variable-rgx = (_[A-Z][A-Z0-9_]*$)|...`

## Code Style

- Tool: ruff (linter + auto-fixer) + flake8 (syntax only in pre-commit)
- Line length: 127 characters (set in `pyproject.toml` and `.pylintrc`)
- Imports: ruff's isort integration (handled automatically via `ruff --fix`)
- No trailing whitespace (enforced by pre-commit `trailing-whitespace` hook)
- EOF fixers: all files end with single newline (enforced by pre-commit `end-of-file-fixer` hook)
- Tool: ruff (primary) with selective pylint (secondary, via `.pylintrc`)
- ruff rules: `["E", "F", "W", "I", "UP"]` (pycodestyle errors/warnings, pyflakes, isort, pyupgrade)
- Ruff ignores: `["E741"]` (ambiguous variable names like `l`, `O`, `I` — common in ML code)
- Per-file ignores for tests: `"tests/**/*.py" = ["F811"]` (re-imported/shadowed fixtures are expected)
- Pre-commit hooks: ruff (`--fix` auto-correct), flake8 (syntax only: `E9,F63,F7,F82`), mypy (informational only)
- `.pylintrc` disables false-positives common in ML/data science: documentation checks (C0114/115/116), complexity metrics (R0914/912/915/913), duplication (R0801), broad exception catching (W0718)
- Target: Python 3.10+
- Required: `from __future__ import annotations` at the top of all source files (enables PEP 563 postponed evaluation for 3.10 compatibility)
- Union types: `X | Y` not `Union[X, Y]`; optional: `X | None` not `Optional[X]`
- Pattern matching: `match` statements acceptable but not required

## Import Organization

- None defined at project level (ruff isort does not require aliases)
- Absolute imports preferred for clarity (e.g., `from trading_crab_lib.transforms import engineer_all`)
- Gracefully handled in try/except for optional dependencies (`hmmlearn`, `statsmodels`, `hdbscan`, `lightgbm`, `k-means-constrained`)
- Error message provides install instructions: `"Install with: pip install 'trading-crab-lib[ingestion]'"`
- Availability flags set at module load: `_HMM_AVAILABLE = False` assigned in except block, later checked with `if _HMM_AVAILABLE:` before use
- Tests skip via `pytest.mark.skipif` when optional deps unavailable

## Error Handling

- Specific exception types always caught (no bare `except:`)
- Broad exception catching used only for network ingestion code (marked with `# noqa: BLE001`)
- Fail-fast for config issues: `ValueError` raised immediately with complete error list via `validate_config()`
- Missing/invalid files: `FileNotFoundError` raised with full path included
- Network failures: caught and logged at WARNING, pipeline continues with empty/partial data (graceful degradation)

## Logging

- Each module: `log = logging.getLogger(__name__)` at top (after imports)
- Root logger configured in `config.py`: `setup_logging(level: str = "INFO")`
- Format: `"%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"` with `datefmt="%Y-%m-%d %H:%M:%S"`
- Verbosity control: `RunConfig.apply_logging()` sets root to DEBUG if `verbose=True`
- **DEBUG**: checkpoint freshness checks, detailed step progression (only when `--verbose`)
- **INFO**: normal pipeline progress, checkpoint saves/loads, ingestion completion counts, named regime outputs
- **WARNING**: missing/invalid config, corrupt checkpoint metadata, network failures, stale data
- **ERROR**: not used; critical failures raise exceptions instead

## Comments

- Algorithm intent: explain *why* a non-obvious approach is chosen (e.g., Bernstein gap fill in log space, not linear)
- Complex math: document the formula or reference (e.g., "Derivative of a linear series should be roughly constant")
- Publication-lag shifts: why GDP/GNP are shifted, not other FRED series
- Intentional simplifications (ponytail style): mark with `# ponytail: explanation` comment naming the simplification and upgrade path
- Triple-quoted docstrings on all public functions (enforced at code-review, not lint time)
- Format: brief one-liner, then longer description if needed, Args/Returns/Raises sections
- Example from `regime.py`:
- Always present: explain module purpose, usage example, key concepts
- Rendered as the first triple-quoted string in the file (before imports of `from __future__`)
- Example from `checkpoints.py`: lists usage pattern for save/load/is_fresh/clear
- Horizontal lines using `# ── Name ──` (en-dashes, exactly 2 dashes before and after)
- Used to organize long modules into logical sections
- No other divider styles

## Function Design

- Target: short enough to fit on one screen without scrolling (rarely exceed ~50 lines)
- Longer functions acceptable only for pipelines (step functions inherently complex) and exploratory notebooks
- Library functions broken into helpers when they exceed 40 lines
- Keyword-only arguments (`*`) for optional flags to prevent accidental positional misuse
- Config objects passed whole (`cfg: dict[str, Any]`) rather than unpacked
- RunConfig always passed as positional parameter when needed: `def step_func(df: pd.DataFrame, run_cfg: RunConfig)`
- Examples from registry:
- Predictable types: functions return exactly one type (not `X | None` unless documented)
- DataFrames/Series always indexed consistently (preserved from input where possible)
- Models return as objects, not dicts (flat API in `prediction/__init__.py`); bundle dicts only in `classifier.py` for test support
- Functions do not mutate inputs unless explicitly documented
- `.copy()` used before modifying: `def func(df): df = df.copy(); df["col"] = ...; return df`
- Verified in tests: `def test_does_not_mutate_input(self, raw_macro_df): original_cols = list(raw_macro_df.columns); func(raw_macro_df); assert list(raw_macro_df.columns) == original_cols`

## Module Design

- Public functions: no prefix (e.g., `load()`, `fetch_all()`, `build_profiles()`)
- Private functions/constants: `_prefix` (e.g., `_fetch_one()`, `_MAX_WORKERS`, `_REQUIRED_SECTIONS`)
- Package-level re-exports in `__init__.py` for convenience (e.g., `plotting/__init__.py` re-exports all plot functions)
- Used in `plotting/__init__.py` and `monitoring/__init__.py` to re-export all submodule functions
- Pattern: `from .submodule import *` with explicit `__all__` list (when needed for clarity)
- Enables `from trading_crab_lib.plotting import plot_regime_timeline` without knowing the submodule
- Avoided via explicit imports where possible
- Lazy imports (`from ... import X` inside a function) used only for optional dependencies or `__getattr__` patterns
- No known circular dependency chains in current codebase (see CLAUDE.md)

<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->

## Architecture

## System Overview

```

```

## Component Responsibilities

| Component | Responsibility | File |
|-----------|----------------|------|
| **CLI & Pipeline Orchestration** | Parse args, dispatch 9 steps, manage flow | `src/trading_crab/cli.py`, `src/trading_crab/pipeline.py` |
| **Ingestion** | Fetch FRED, multpl.com, macrotrends, yfinance; merge into `macro_raw.parquet` | `src/trading_crab_lib/ingestion/{fred,multpl,assets,macrotrends,grok}.py` |
| **Feature Engineering** | Cross-ratios, log transforms, gap-fill, derivatives → `features.parquet` | `src/trading_crab_lib/transforms.py`, `divergence.py`, `momentum.py`, `indicators.py` |
| **Clustering** | PCA reduction, KMeans/GMM/HMM/DBSCAN/Spectral, model comparison → `cluster_labels.parquet` | `src/trading_crab_lib/clustering.py`, `gmm.py`, `hmm.py`, `markov.py`, `density.py`, `spectral.py`, `cluster_comparison.py` |
| **Regime Profiling** | Build regime statistics, suggest names, compute transitions → `profiles.parquet` | `src/trading_crab_lib/regime.py` |
| **Supervised Prediction** | Train current-regime + forward classifiers, save models | `src/trading_crab_lib/prediction/__init__.py`, `classifier.py`, `gradient_boosting.py` |
| **Asset Analysis** | Compute quarterly returns by regime, rank assets | `src/trading_crab_lib/asset_returns.py` |
| **Reporting** | Dashboard signals, portfolio construction, email templates | `src/trading_crab_lib/reporting.py`, `email.py` |
| **Diagnostics** | RRG analysis, rolling statistics, ratio tracking | `src/trading_crab_lib/diagnostics.py` |
| **Tactical Classification** | Assign buy_hold/swing/stand_aside per asset | `src/trading_crab_lib/tactics.py` |
| **Visualization** | All matplotlib figures (per-step plotting submodule) | `src/trading_crab_lib/plotting/` (9 files) |
| **Pipeline Monitoring** | Step output validation, CV diagnostics, health summaries | `src/trading_crab_lib/monitoring/` (5 files) |
| **State Persistence** | Checkpoint save/load with parquet + JSON metadata | `src/trading_crab_lib/checkpoints.py` |
| **Configuration** | Load settings.yaml, validate, supply to all steps | `src/trading_crab_lib/config.py` |
| **Runtime Flags** | Dataclass for CLI args (refresh, plots, steps, verbose, etc.) | `src/trading_crab_lib/runtime.py` |

## Pattern Overview

- **Two-stage learning:** unsupervised regime discovery (steps 1-4) → supervised nowcasting/forecasting (steps 5-9)
- **Look-ahead bias guards:** centered features for clustering; causal features for supervised learning (separate checkpoints: `features.parquet` vs `features_supervised.parquet`)
- **Configuration-driven:** all parameters (FRED series, feature lists, clustering k, model depths) live in `config/settings.yaml`, not hardcoded
- **Deterministic:** global random seed at pipeline start; `market_code` column excluded from gap-fill row-validity logic to prevent label-pattern leakage
- **Checkpoint-everything:** every step saves intermediate parquet + metadata; `is_fresh()` checks skip recomputation for data < N days old
- **Three regime sources:** `grok` (AI labels), `clustered` (step 3 output), `predicted` (step 5 output), or none (fully data-driven)

## Layers

- Purpose: Fetch raw time-series data from external sources (FRED API, web scrapers, yfinance)
- Location: `src/trading_crab_lib/ingestion/`
- Contains: HTTP fetchers for FRED, multpl.com, macrotrends.net, yfinance; optional grok label loader
- Depends on: config (series list + URLs), requests/fredapi/lxml/yfinance (optional extras)
- Used by: Pipeline step 1; saved to `data/raw/macro_raw.parquet`
- Key innovation: publication-lag shifts for GDP/GNP (+1 quarter) to prevent look-ahead bias
- Purpose: Transform raw columns into engineered features (ratios, log transforms, gap-fill, derivatives)
- Location: `src/trading_crab_lib/transforms.py`, plus `divergence.py`, `momentum.py`, `indicators.py`, `yield_curve_features.py`
- Contains: `engineer_all()` master function orchestrating 6-step pipeline (see ADR #4 in CLAUDE.md for order)
- Depends on: scipy (BPoly gap-fill), numpy (gradient), raw macro_raw DataFrame
- Used by: Steps 2-3-4-5 (clustering and supervised models)
- Key abstractions: `add_cross_ratios()`, `apply_log_transforms()`, `_fill_column()`, `apply_derivatives()`
- Outputs: Two versions for two purposes:
- Purpose: Find regime structure via clustering; profile and name each regime
- Location: `src/trading_crab_lib/clustering.py`, plus `gmm.py`, `hmm.py`, `markov.py`, `density.py`, `spectral.py`, `cluster_comparison.py`, `regime.py`
- Contains: PCA, KMeans (standard + constrained), GMM, HMM, Markov, DBSCAN, HDBSCAN, Spectral, gap statistic, knee detection
- Key decisions:
- Outputs: `cluster_labels.parquet`, `profiles.parquet`, `regime_labels.yaml` (manual curation after auto-suggest)
- Purpose: Train classifiers to predict current regime and forward transitions from causal features only
- Location: `src/trading_crab_lib/prediction/`
- Contains: Two APIs (ADR #12):
- Also includes: LightGBM wrapper, GradientBoosting, decision-tree training, forward classifiers (h ∈ {1,2,4,8} quarters)
- Key features: TimeSeriesSplit CV (never shuffle), label confidence weighting (γ from regime smoother), calibration curve output
- Outputs: `outputs/models/{current_regime,dt_current,lgbm_current,forward_*}.pkl`
- Purpose: Summarize results for human decision-making (current regime + assets, buy/sell signals, portfolio weighting, diagnostics)
- Location: `src/trading_crab_lib/reporting.py`, `asset_returns.py`, `regime.py`, `diagnostics.py`, `tactics.py`, `email.py`
- Contains:
- Outputs: `outputs/reports/dashboard.csv`, `outputs/reports/weekly_report.md`, `outputs/reports/diagnostics/`
- **RuntimeConfig** (`runtime.py`): Dataclass holding all CLI flags (refresh, plots, verbose, steps, etc.). Passed through every step.
- **CheckpointManager** (`checkpoints.py`): Save/load DataFrames as parquet + JSON metadata; validity checks; clear operations. Prevents re-scraping and re-computing.
- **Config** (`config.py`): Load and validate `config/settings.yaml`; convenience loaders for FRED series, multpl URLs, ETF tickers, etc.
- **Plotting** (`plotting/` package): Visualization package with submodules per pipeline stage. Re-exports all plot functions + color constants. Honors `save_plots` / `show_plots` flags.
- **Monitoring** (`monitoring/` package): Pipeline health checks (step output validation, CV fold accuracy, ingestion completeness, regime stability). Logged at INFO; wired into main loop.

## Data Flow

### Primary Request Path (Full Pipeline)

### Conditional Flows

- If `NAME == "grok"`: Load from `data/grok_*.pickle` (external LLM labels, static)
- If `NAME == "clustered"`: Load from checkpoint `market_code_clustered` (step 3 output, if saved with `--save-market-code`)
- If `NAME == "predicted"`: Load from checkpoint `market_code_predicted` (step 5 auto-save, latest prediction)
- Otherwise: No external label; clustering fully data-driven
- Each step checks `CheckpointManager.is_fresh(name)` before recomputing
- `--refresh` forces step 1-2 re-scraping/recompute; other steps load cached checkpoints if < 7 days old
- `--recompute` forces step 2; steps 3+ still load cached checkpoints unless they depend on step 2
- `--steps 3,4,5` skips steps 1-2, loads their checkpoints; runs only listed steps
- **Global state:** None. RunConfig passed explicitly to every function.
- **Mutable shared state:** DataFrame columns (e.g., market_code) are not used to determine gap-fill validity; only feature columns determine row validity (prevents label-pattern leakage into gap-fill boundaries)
- **Model persistence:** sklearn objects pickled to `outputs/models/`, not kept in memory across steps

## Key Abstractions

- Purpose: Encapsulate all CLI flags (refresh, plots, verbose, steps, market_code, etc.)
- Created once at entry point, passed to every step
- `from_args(Namespace)` factory converts argparse → RunConfig
- Example usage: `if run_cfg.generate_plots: plot_(..., run_cfg.save_plots, run_cfg.show_plots)`
- Purpose: Persistent state — DataFrames as parquet + JSON metadata
- API: `save(df, name)`, `load(name)`, `is_fresh(name, max_age_days)`, `clear(name)`, `clear_all()`
- Metadata tracked: creation timestamp, config hash (settings.yaml MD5), row/col counts
- Special: Preservation checkpoints (`*_secondary`) survive `clear_all()` to audit pre-selection columns
- Example: `if cm.is_fresh("macro_raw", max_age_days=7): df = cm.load(...) else: df = fetch_fresh(...)`
- Purpose: Orchestrate the complete feature pipeline (6 steps in fixed order)
- Takes: raw DataFrame + config + `causal` boolean
- Returns: fully engineered DataFrame (or two versions if called twice with causal=True/False)
- Order guaranteed by design: cross-ratios → log → select → gap-fill → derivatives → select (ADR #4)
- Key invariant: gap-fill happens AFTER log transform, BEFORE derivatives
- Purpose: Load trained model from pickle, score current quarter
- Returns: dict with `{"regime": int, "probabilities": {0: 0.15, 1: 0.70, ...}}`
- Used by: Dashboard (step 7), notebooks, live scoring scripts
- Example: `pred = predict_current(features_row, model_path="outputs/models/current_regime.pkl")`
- Purpose: Compute per-regime statistics (mean, std, Sharpe, transition matrix)
- Takes: features DataFrame + regime labels
- Returns: dict with regime_id → dict of statistics
- Used by: Regime naming heuristics, reporting, diagnostics
- Purpose: All matplotlib figures in one package; consistent palette and save/show behavior
- Submodules: `core.py` (helpers), `ingestion.py`, `features.py`, `clustering.py`, `regime.py`, `prediction.py`, `assets.py`, `diagnostics.py`
- All functions: `plot_X(..., run_cfg: RunConfig)` — honor `save_plots` / `show_plots`
- Color palette: `CUSTOM_COLORS` (5 regimes), `REGIME_CMAP` (matplotlib ListedColormap)
- Example: `from trading_crab_lib import plotting; plotting.plot_regime_timeline(..., run_cfg)`

## Entry Points

- Location: `src/trading_crab/cli.py:run_pipeline()`
- Triggers: `tradingcrab` command (pip-installed entry point)
- Responsibilities: Parse argparse, construct RunConfig, call `trading_crab.pipeline.main()`
- Example: `tradingcrab --refresh --recompute --plots --steps 3,4,5`
- Location: `src/trading_crab/pipeline.py:main()` (can be called as `from trading_crab.pipeline import main; main()`)
- Behavior: Parse sys.argv, construct RunConfig, dispatch step functions
- Location: `run_pipeline.py` (repo root)
- Behavior: Adds repo root to sys.path, imports and calls `trading_crab.pipeline.main()`
- Purpose: Allows `python run_pipeline.py ...` without pip install
- Example: `python run_pipeline.py --refresh --plots`
- Each notebook (01–12) in `notebooks/` imports from `trading_crab_lib` and checkpoint manager
- Example: `from trading_crab_lib.checkpoints import CheckpointManager; cm = CheckpointManager(); features = cm.load("features")`
- Location: `pipelines/{01..09}_*.py`
- Behavior: Simplified entry points (no RunConfig, no checkpoint management, often hardcoded flags)
- Purpose: Legacy compatibility + examples for minimal runs
- Example: `python pipelines/03_cluster.py`

## Architectural Constraints

- **Threading:** Single-threaded event loop. No asyncio or multiprocessing. Network fetches (FRED, yfinance) are serial (rate-limited to 2 sec/request).
- **Global state:** None. All state is passed via function args (RunConfig, config dict, DataFrames).
- **Circular imports:** Prevented by lazy imports in `__getattr__` for convenience re-exports (`trading_crab_lib.RunConfig`, `trading_crab_lib.CheckpointManager`).
- **Module-level code:** Runs at import time for path resolution (`_resolve_dir()` in `__init__.py`), config validation (`validate_config()` called at end of `load()`). All data loads are lazy (inside step functions).
- **Determinism:** Controlled by `cfg["pipeline"]["random_state"]` (default 42). Set once in `main()` via `np.random.seed()` and `random.seed()`. Gap-fill row validity excludes `market_code` to prevent label-pattern interference.

## Anti-Patterns

### Look-Ahead Bias (the #1 trap)

### Ignoring Publication Lag

### Changing Clustering Geometry Without Re-Labeling

### Using Standard KMeans Instead of Balanced KMeans

## Error Handling

- **Config validation:** `validate_config()` (called at end of `load()`) checks all required sections and critical keys. Single `ValueError` lists every issue at once.
- **Missing optional dependencies:** Clustering modules (gmm.py, hmm.py, markov.py, density.py) catch `ImportError` and provide helpful messages with install instructions.
- **Network failures:** Ingestion modules (fred.py, multpl.py, assets.py) catch exceptions, log at WARNING, and return empty DataFrame. Pipeline continues with whatever data was fetched (partial ingestion is acceptable; raises flag in monitoring output).
- **Checkpoint misses:** `CheckpointManager.load()` raises `FileNotFoundError` if checkpoint missing (caller must decide: re-compute or exit).
- **No re-raising:** No bare `except:`. All exception handlers are explicit by type.

## Cross-Cutting Concerns

- Config: `validate_config()` checks settings.yaml at load time
- Data: `validate_step_output()` in monitoring checks DataFrame shape, NaN fractions, dtypes
- Features: `compute_feature_quality()` warns if NaN > 50% in any column
- FRED: API key from `.env` (required; fails if missing)
- multpl.com: Rate-limited by 2-second delays (no auth)
- yfinance: No auth required
- Email: Optional; SMTP config from `config/email.yaml` or `TC_SMTP_*` env vars
- CheckpointManager memoizes via parquet + metadata validation (MD5 of settings.yaml)
- Feature engineering applies `dropna()` on final feature columns (shrinks from 100+ to 70 columns typically)
- PCA is fixed at 5 components (not variance-threshold, which would vary across runs)
- Plotting writes to disk only if `save_plots=True`; `show_plots` controls plt.show() (false in CI)

<!-- GSD:architecture-end -->

<!-- GSD:skills-start source:skills/ -->

## Project Skills

No project skills found. Add skills to any of: `.claude/skills/`, `.agents/skills/`, `.cursor/skills/`, `.github/skills/`, or `.codex/skills/` with a `SKILL.md` index file.
<!-- GSD:skills-end -->

<!-- GSD:workflow-start source:GSD defaults -->

## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:

- `/gsd-quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd-debug` for investigation and bug fixing
- `/gsd-execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->

<!-- GSD:profile-start -->

## Developer Profile

> Profile not yet configured. Run `/gsd-profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
