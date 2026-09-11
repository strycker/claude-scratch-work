# trading-crab-lib

A library of building blocks for market-regime classification and prediction:
feature engineering, unsupervised clustering, regime profiling, supervised
prediction, reporting/diagnostics, plotting, and checkpointing. Pure Python
data-science code — no CLI, no pipeline orchestration, no opinion about how
you wire the pieces together.

If you want a runnable end-to-end pipeline with a CLI, see the sibling
application package, `trading-crab` (`src/trading_crab/` in the same
monorepo), which depends on this library and provides the `tradingcrab`
command plus full pipeline orchestration.

## What it provides

- **Feature engineering** (`transforms.py`) — `engineer_all()` orchestrates
  cross-asset ratios, log transforms, Bernstein-polynomial gap filling, and
  smoothed derivatives, including a yield-curve-spread step
  (`add_yield_curve_features()`). Additional feature families live in
  `divergence.py` (cross-asset divergence z-scores and triggers),
  `momentum.py` (trailing momentum, relative strength, rolling
  cross-correlation), and `indicators.py` (a composite LEI-proxy indicator).
- **Clustering** (`clustering.py`) — PCA reduction, KMeans (standard and
  size-balanced), and clustering-diagnostics helpers (gap statistic, knee
  detection, SVD comparison). Alternative/exploratory clustering methods
  live in their own modules: `gmm.py` (Gaussian Mixture Models), `hmm.py`
  (Gaussian HMM via `hmmlearn`, optional), `markov.py` (Markov
  regime-switching via `statsmodels`, optional), `density.py` (DBSCAN /
  HDBSCAN), and `spectral.py` (spectral clustering). `cluster_comparison.py`
  cross-compares methods (pairwise Rand index, RF feature importances).
- **Regime profiling** (`regime.py`) — `build_profiles()`, `suggest_names()`,
  and transition-matrix / forward-probability helpers built from cluster
  labels.
- **Supervised prediction** (`prediction/`) — two APIs with different
  consumers, kept deliberately separate:
  - `prediction/__init__.py` is the **flat production API**:
    `train_current_regime(X, y, cfg)` returns a plain
    `RandomForestClassifier`, `train_decision_tree(X, y, cfg)` returns a
    `DecisionTreeClassifier`, and `predict_current(model, features_now)`
    returns `{"regime": int, "probabilities": {...}}`.
  - `prediction/classifier.py` is a **bundle API** used for test/analysis
    tooling: its `train_current_regime(X, y, cv_splits=N)` returns a dict of
    fitted models plus per-fold CV reports, and it adds interpretability
    helpers (`extract_top_features()`, `train_interpretability_tree()`).
  - Import from the module that matches what you need — the two return
    different shapes on purpose. `prediction/gradient_boosting.py` backs the
    bundle API's optional GradientBoosting support.
- **Reporting, diagnostics, tactics** — `reporting.py` (dashboard signals,
  portfolio construction), `diagnostics.py` (Relative Rotation Graph
  analysis, rolling z-scores), `tactics.py` (buy_hold / swing / stand_aside
  classification), `email.py` (weekly report email delivery).
- **Plotting** (`plotting/` package) — matplotlib/seaborn visualizations,
  organized per pipeline stage (`ingestion.py`, `features.py`,
  `clustering.py`, `regime.py`, `prediction.py`, `assets.py`,
  `diagnostics.py`), re-exported from `plotting/__init__.py`.
- **Data ingestion** (`ingestion/` package, optional — see the `ingestion`
  extra below) — FRED API (`fred.py`), multpl.com scraper (`multpl.py`),
  yfinance ETF prices (`assets.py`), macrotrends.net scraper
  (`macrotrends.py`), and a headless-browser fallback for JS-gated sources
  (`browser.py`).
- **Checkpointing** (`checkpoints.py`) — `CheckpointManager` saves/loads
  DataFrames as parquet (with a JSON manifest for freshness checks) and
  sklearn models via joblib.

## Install

Base install (core deps only — pandas, numpy, scikit-learn, scipy, pyarrow,
pyyaml, joblib, python-dotenv; no network or plotting dependencies):

```bash
pip install trading-crab-lib
```

Optional extras (each maps to a group in this package's own
`pyproject.toml`):

| Extra | Adds | Purpose |
|---|---|---|
| `ingestion` | fredapi, requests, lxml, cssselect, beautifulsoup4, yfinance, curl_cffi, html5lib, certifi | FRED/multpl.com/yfinance/macrotrends data fetchers |
| `plotting` | matplotlib, seaborn | All visualization functions in `plotting/` |
| `hmm` | hmmlearn, statsmodels | Gaussian HMM and Markov regime-switching clustering |
| `clustering-extras` | hdbscan, kneed | Density-based clustering and knee-point detection |
| `boosting` | lightgbm | LightGBM classifier alternative to RandomForest |
| `browser` | playwright, selenium | Headless-browser fallback for JS-gated ingestion sources — after installing this extra, run `playwright install chromium` once (the wheel ships no browser binaries); Selenium additionally needs a Chrome/Chromium binary + matching driver on `PATH` |
| `all` | everything above | Convenience bundle: `trading-crab-lib[all]` |
| `dev` | pytest, pytest-cov, flake8, ruff | Development/test tooling |

```bash
pip install "trading-crab-lib[ingestion,plotting]"
```

## Quick usage

```python
import trading_crab_lib as tcl
from trading_crab_lib.transforms import engineer_all

# load() accepts None (reads config/settings.yaml from the repo root),
# a path/string to a specific YAML file, or a pre-built dict — the dict
# form bypasses all file I/O, which is the easiest path for a pip-only
# install with no project layout on disk.
cfg = tcl.load({"data": {...}, "features": {...}, ...})

# causal=False: centered smoothing, safe only for clustering/regime
#   labeling (uses future data within each rolling window).
# causal=True: backward-only smoothing, required for supervised learning
#   and live scoring — never train on centered features and score causal
#   ones, or vice versa (look-ahead bias).
features = engineer_all(raw_macro_df, cfg, causal=True)
```

Functions that expect a full project layout on disk (`CheckpointManager`,
the `ingestion/` fetchers) look for `config/`, `data/`, and `outputs/`
directories relative to the detected repo root. Override each with the
`TC_CONFIG_DIR`, `TC_DATA_DIR`, and `TC_OUTPUT_DIR` environment variables
(or `TC_ROOT_DIR` to override the root all three are derived from) — or
avoid the file-layout requirement entirely by passing a config dict
straight to `load()` as shown above.

## Links

- Homepage / source: https://github.com/strycker/trading-crab
- For the CLI and full pipeline orchestration built on top of this library,
  see `src/trading_crab/` (the `trading-crab` package) in the same
  repository.

## License

MIT
