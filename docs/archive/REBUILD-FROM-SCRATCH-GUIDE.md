# REBUILD-FROM-SCRATCH-GUIDE.md

A complete guide for rebuilding the Trading-Crab market regime pipeline from zero.
Assumes only Python 3.10+, pip, and a FRED API key are available.

**Created:** 2026-03-31
**Covers:** Architecture, build order, feature pipeline, clustering, prediction, testing,
packaging, lessons learned, and critical invariants.

---

## 1. What You Are Building

A **weekly automated report** that answers:
> "What macroeconomic regime are we in right now, which asset classes should I hold,
> and what is the probability of transitioning to a different regime in 1–8 quarters?"

The pipeline has three conceptual layers:

1. **Unsupervised labeling** — quarterly macro data (1950–present) is clustered into
   5 "market regimes" (e.g. Stagflation, Growth Boom, Rising-Rate Slowdown).
2. **Supervised prediction** — a Random Forest trained on the cluster labels predicts
   today's regime and forward transition probabilities at horizons of 1, 2, 4, 8 quarters.
3. **Portfolio ranking** — ETF returns are computed per regime; the current regime's
   ranking becomes the buy/hold/sell recommendation.

**End deliverable:** `outputs/reports/weekly_report.md` + optional email with inline plots.

### Data sources
| Source | What | How |
|--------|------|-----|
| multpl.com | 46 macro series (S&P 500, CAPE, CPI, dividends, …) | lxml scraper |
| FRED API | 14 series (GDP, BAA/AAA rates, VIX, M2, yield curve, …) | fredapi |
| macrotrends.net | Gold spot, WTI crude oil (historical depth) | HTML JSON extraction |
| yfinance | 38 ETF price histories (SPY, TLT, GLD, …) | yfinance |

---

## 2. Two-Package Architecture

**Do not ship a monolith.** The project is split into two independent PyPI packages
from day one. This is the most important architectural decision.

```
trading-crab-lib   (pip name)   src/trading_crab_lib/
trading-crab       (pip name)   src/trading_crab/        + root pyproject.toml
```

### Why two packages?

| Problem with monolith | Two-package solution |
|-----------------------|---------------------|
| `pip install trading-crab` forces matplotlib on a headless server | `pip install trading-crab-lib` pulls only pandas/numpy/sklearn/scipy |
| Notebooks can't import pipeline without CLI overhead | Library is importable standalone |
| CI tests the full CLI even for pure library changes | Library and app test independently |
| Any change touches the single installable unit | Library API is stable; app evolves faster |

### Package responsibilities

**`trading-crab-lib`** — pure library, no CLI, no config files:
- All transforms, clustering, prediction, reporting, plotting, ingestion modules
- Optional extras: `[ingestion]`, `[plotting]`, `[hmm]`, `[clustering-extras]`, `[boosting]`
- Core dependencies: `pandas`, `numpy`, `pyarrow`, `scikit-learn`, `scipy`, `pyyaml`, `joblib`
- Ships `py.typed` marker for type-checking support

**`trading-crab`** — application layer:
- `cli.py`: entry points (`tradingcrab`, `tradingcrab-setup`, `tradingcrab-publish`)
- `pipeline.py`: 1400-line orchestration (9 pipeline steps, argparse, logging, timing)
- Depends on `trading-crab-lib>=0.1.2`
- Ships `settings.example.yaml` as package data via `tradingcrab-setup`

### uv workspace (recommended for development)

Put `[tool.uv.workspace] members = ["src/trading_crab_lib"]` in the root
`pyproject.toml`. Then `uv sync` installs both packages in editable mode automatically.
Without uv: `pip install -e src/trading_crab_lib/[all,dev] && pip install -e .[dev]`.

---

## 3. Repository Layout

Build the repository in this exact structure from day one. Retrofitting is painful.

```
trading-crab/
├── CLAUDE.md                    ← all dev docs (architecture, ADRs, pitfalls)
├── README.md                    ← user-facing overview
├── ROADMAP.md                   ← prioritized backlog
├── STATE.md                     ← current pipeline status, test counts
├── pyproject.toml               ← trading-crab application package
├── run_pipeline.py              ← backward-compat shim → trading_crab.pipeline.main()
├── Makefile                     ← dev shortcuts
├── .env.example                 ← FRED_API_KEY=your_key_here (never .env)
│
├── config/
│   ├── settings.yaml            ← ALL tunable parameters (no hardcoded constants)
│   └── regime_labels.yaml       ← human-edited regime names after clustering
│
├── legacy/                      ← original 1249-line monolith (read-only reference)
│   └── unified_script.py
│
├── src/
│   ├── trading_crab/            ← app package
│   │   ├── __init__.py
│   │   ├── cli.py
│   │   └── pipeline.py
│   └── trading_crab_lib/        ← library package
│       ├── pyproject.toml       ← independent pyproject for lib
│       ├── __init__.py          ← ROOT, CONFIG_DIR, DATA_DIR, OUTPUT_DIR + env overrides
│       ├── config.py
│       ├── runtime.py           ← RunConfig dataclass
│       ├── checkpoints.py       ← CheckpointManager (parquet + manifest)
│       ├── transforms.py        ← entire feature engineering pipeline
│       ├── clustering.py
│       ├── regime.py
│       ├── asset_returns.py
│       ├── reporting.py
│       ├── plotting.py          ← ALL visualization (never inline in notebooks)
│       ├── monitoring.py        ← pipeline health checks
│       ├── diagnostics.py       ← RRG analysis
│       ├── tactics.py
│       ├── email.py
│       ├── divergence.py
│       ├── momentum.py
│       ├── indicators.py
│       ├── gmm.py
│       ├── hmm.py
│       ├── markov.py
│       ├── density.py
│       ├── spectral.py
│       ├── cluster_comparison.py
│       ├── ingestion/
│       │   ├── __init__.py      ← ingestion_completeness_report()
│       │   ├── multpl.py
│       │   ├── fred.py
│       │   ├── assets.py
│       │   ├── macrotrends.py
│       │   └── grok.py
│       └── prediction/
│           ├── __init__.py      ← FLAT API (production)
│           ├── classifier.py    ← BUNDLE API (tests only)
│           └── gradient_boosting.py
│
├── pipelines/                   ← standalone step scripts (01–09)
├── notebooks/                   ← one per step (01–12) + exploration
├── scripts/
│   ├── setup.sh
│   └── run_weekly_report.py
├── tests/
│   ├── conftest.py
│   ├── unit/                    ← one file per library module
│   └── integration/             ← end-to-end tests (synthetic data)
├── data/                        ← gitignored; created at runtime
└── outputs/                     ← gitignored; created at runtime
```

### Critical layout rules

1. `data/` and `outputs/` must be in `.gitignore` from day one — never commit data files.
2. `legacy/` is read-only. Every algorithm must be traceable back to `unified_script.py`.
3. All config in `config/settings.yaml` — never hardcode URLs, feature lists, or parameters
   in Python source.
4. Notebooks call functions from `plotting.py`; they never define plotting logic inline.

---

## 4. Build Order

Build bottom-up: stable primitives first, orchestration last.
Each layer should have tests before the next layer uses it.

### Layer 0 — Configuration & Runtime (build first, never changes)

**Files:** `config.py`, `runtime.py`, `__init__.py`

`config.py` — single `load()` function that reads `settings.yaml` and returns a dict.
`load_portfolio()` reads an optional `portfolio.local.yaml`. Nothing else.

`runtime.py` — `RunConfig` dataclass:
```python
@dataclass
class RunConfig:
    verbose: bool = False
    generate_plots: bool = False
    refresh_source_datasets: bool = False
    recompute_derived_datasets: bool = False
    save_plots: bool = True
    show_plots: bool = False        # always False in CI

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "RunConfig": ...
```
Pass this object through every pipeline step. Never use global variables for runtime flags.

`__init__.py` — defines `ROOT`, `CONFIG_DIR`, `DATA_DIR`, `OUTPUT_DIR` as `Path` objects,
resolved from env vars (`TC_ROOT_DIR`, `TC_DATA_DIR`, etc.) with repo-relative defaults.
Also lazy-re-exports `RunConfig` and `CheckpointManager` for convenience.

**Tests:** `test_config.py`, `test_runtime.py`, `test_init_module.py`

---

### Layer 1 — Checkpoint System (build second)

**File:** `checkpoints.py`

`CheckpointManager` wraps all parquet/joblib I/O:
```python
cm = CheckpointManager()           # uses DATA_DIR/checkpoints/
cm.save("macro_raw", df)           # saves .parquet + manifest entry
df = cm.load("macro_raw")          # loads .parquet
cm.is_fresh("macro_raw", max_age_days=7)   # wall-clock freshness check
cm.save_model("current_regime", model)     # joblib.dump
model = cm.load_model("current_regime")   # joblib.load
cm.list()                          # all checkpoints + creation times
cm.clear("macro_raw")              # delete one
cm.clear_all()                     # delete all except preservation checkpoints
```

**Key decisions:**
- Parquet for DataFrames (typed, compressed, inspectable)
- joblib for sklearn models (NOT pickle — avoids arbitrary code execution risk)
- Manifest is a JSON sidecar per checkpoint (not one global manifest — avoids lock contention)
- "Preservation checkpoints" (`macro_raw_secondary`, `features_secondary`) survive `clear_all()`
  — they retain all columns before `dropna()` narrows the DataFrame

**Tests:** `test_checkpoints.py` — test save/load roundtrip, freshness logic,
preservation skip, corrupt manifest graceful degradation.

---

### Layer 2 — Ingestion (build third)

**Files:** `ingestion/multpl.py`, `ingestion/fred.py`, `ingestion/assets.py`,
`ingestion/macrotrends.py`, `ingestion/__init__.py`

Build each ingestion module independently; merge in `pipeline.py` step 1.

**multpl.py** — lxml cssselect scraper:
```python
RATE_LIMIT_SECONDS = 2             # never reduce below 2
def fetch_series(url, value_type) -> pd.Series: ...
def fetch_all(cfg, run_cfg) -> pd.DataFrame: ...  # 46 series → quarterly DataFrame
```
All URLs in `config/settings.yaml`. Never hardcode URLs in Python.

**fred.py** — fredapi wrapper:
```python
def fetch_all(cfg, run_cfg) -> pd.DataFrame: ...   # 14 series
```
Apply `shift: true` per-series from config to prevent look-ahead bias (GDP, GNP released
~30 days after quarter end). Resample with `.last()` to get end-of-quarter values.

**assets.py** — yfinance ETF prices. Key pitfall: set `CURL_CA_BUNDLE` and `SSL_CERT_FILE`
to `certifi.where()` at module import time to avoid SSL errors.

**`ingestion/__init__.py`** — `ingestion_completeness_report()` checks expected columns
(derived from config) vs actual columns in `macro_raw`. Returns a `CompletenessReport`
dataclass. Wire into step 1 to catch partial ingestion failures early.

**Testing:** Mock all HTTP calls. Use `responses` or `unittest.mock.patch` for requests.
Never make real network calls in tests.

---

### Layer 3 — Feature Engineering (build fourth, most complex)

**File:** `transforms.py`

See Section 5 for the full pipeline deep-dive.
The key rule: build and test each transform function in isolation, then wire them
together in `engineer_all()`. Test `engineer_all()` end-to-end with synthetic data.

---

### Layer 4 — Clustering (build fifth)

**File:** `clustering.py`

See Section 6 for details. Build `reduce_pca()` → `evaluate_kmeans()` → `pick_best_k()`
→ `fit_clusters()`. Add alternative methods (GMM, DBSCAN, HMM, Spectral) in separate
modules that are optional-dependency gated.

---

### Layer 5 — Regime Profiling + Prediction

**Files:** `regime.py`, `prediction/__init__.py`, `prediction/classifier.py`

See Section 6 for the prediction API split (flat vs bundle).

---

### Layer 6 — Reporting, Plotting, Diagnostics, Email

**Files:** `asset_returns.py`, `reporting.py`, `plotting.py`, `monitoring.py`,
`diagnostics.py`, `tactics.py`, `email.py`

Build plotting last because it depends on all other modules. Keep all visualization
in `plotting.py` — never define plot logic in notebooks or pipeline scripts.

---

### Layer 7 — Pipeline Orchestration (build last)

**File:** `src/trading_crab/pipeline.py`

Wire all library modules into 9 pipeline steps behind a single `main()` entry point.
Each step function:
1. Checks checkpoint freshness (skip if fresh)
2. Loads its inputs from checkpoints
3. Calls library functions
4. Saves outputs to checkpoints
5. Optionally generates plots (`if run_cfg.generate_plots`)
6. Logs timing

The pipeline is the thinnest possible layer over the library. It contains no algorithm
logic — only orchestration, checkpoint I/O, and argparse.

---

## 5. Feature Engineering Pipeline (Deep Dive)

This is the most important and most error-prone part of the project.
The order is sacred and must match `legacy/unified_script.py` exactly.

### The 6-step transform order

```
raw DataFrame
  │
  ├─ 1. add_cross_ratios()        10 derived columns (price/GDP, credit spread, etc.)
  ├─ 2. add_yield_curve_features() yield curve spreads from FRED rate series
  ├─ 3. add_divergence_features()  cross-asset correlation divergence (level space)
  ├─ 4. add_momentum_features()    trailing returns, relative strength, CPI acceleration
  ├─ 5. apply_log_transforms()     np.log(col.clip(lower=1e-9)) for 30+ columns
  ├─ 6. select_initial_features()  narrow to ~36 columns (config: initial_features)
  ├─ 7. apply_gap_fill()           Bernstein polynomial interpolation (interior NaNs)
  │                                Taylor extrapolation (edge NaNs)
  ├─ 8. apply_derivatives()        np.gradient on matplotlib day-number axis
  │                                centered rolling mean window=5 before AND after gradient
  │                                produces d1, d2, d3 per column
  ├─ 9. add_divergence_features()  cross-asset correlation divergence (derivative space)
  └─ 10. select_clustering_features() narrow to ~69 columns (config: clustering_features)
```

### Why log before gap fill?

Raw series like S&P 500 are exponential. Interpolating between 1000 and 2000 linearly
overshoots (midpoint = 1500 linear vs 1414 = exp((log1000+log2000)/2) in log space).
**Invariant:** gap fill always happens after log transform.

### Gap fill algorithm

Uses `scipy.interpolate.BPoly.from_derivatives` with 4 boundary conditions per side
(value + d1 + d2 + d3). For edge gaps (start/end of series), uses Taylor extrapolation:
`f(x+h) ≈ f(x) + h·f'(x) + (h²/2)·f''(x) + (h³/6)·f'''(x)`.

```python
def _fill_column(series: pd.Series) -> pd.Series:
    # find contiguous NaN gaps; classify as interior vs edge
    # interior: use BPoly with derivatives at both boundary points
    # edge: use Taylor from the nearest known point outward
```

**Critical bug to avoid:** Do NOT use `market_code` to determine valid rows in gap fill.
`market_code` is a label column whose NaN pattern varies depending on which label source
was loaded. Using it as a validity mask makes gap-fill non-deterministic across runs.
Only use the feature column itself to identify valid (non-NaN) rows.

### Causal vs centered smoothing (two feature files)

Step 2 produces **two separate parquet files**:

| File | Smoothing | Used by |
|------|-----------|---------|
| `features.parquet` | `causal=False` (centered, uses future) | Clustering (steps 3–4) |
| `features_supervised.parquet` | `causal=True` (past only) | Supervised learning (steps 5–7) |

Training on centered features and scoring with causal features is **look-ahead bias** —
the single most dangerous mistake in this pipeline. The column names are identical in
both files (intentional), so always pass the right file to the right step.

### Derivative computation

```python
day_nums = df.index.map(matplotlib.dates.date2num).values
d1 = np.gradient(rolling_mean(series), day_nums)
d2 = np.gradient(rolling_mean(d1), day_nums)
d3 = np.gradient(rolling_mean(d2), day_nums)
```

The rolling mean (window=5, centered) is applied both BEFORE each gradient call (to smooth
the input) and implicitly via gradient's own finite-difference stencil. This matches the
legacy script exactly.

### Feature list ownership

`initial_features` and `clustering_features` live in `config/settings.yaml`, not Python.
Changing `clustering_features` changes clustering geometry and invalidates `regime_labels.yaml`.
After any change: delete cluster checkpoints, re-run steps 3–4, update YAML, commit.

---

## 6. Clustering & Prediction

### PCA + KMeans pipeline

```
features DataFrame (69 cols)
  │
  ├─ StandardScaler                # unit variance before PCA
  ├─ PCA(n_components=5)           # fixed at 5; never variance-threshold
  ├─ StandardScaler                # re-scale PCA components before KMeans
  │
  ├─ k-sweep (k=2..12)             # silhouette + CH + DB per k
  │   → best_k from silhouette (capped at k_cap=5)
  │
  ├─ KMeans(k=best_k, n_init=50)   → cluster labels
  └─ KMeansConstrained(k=5, size_min=bucket-2, size_max=bucket+2)
                                   → balanced_cluster labels (PRIMARY)
```

**Why balanced clustering?** With only ~300 quarters of data, standard KMeans often
puts 70% in one cluster. At k=5, `KMeansConstrained` ensures ~60 quarters per regime,
giving reliable mean/std for all downstream statistics. Use `balanced_cluster` everywhere
downstream; `cluster` is kept for geometric reference only.

**Why 5 PCA components?** Established by the legacy analysis. Changing this changes cluster
geometry and invalidates any manually-pinned regime names. Benchmark scree plots before
changing.

### Alternative clustering methods

Implement in separate modules with optional-dependency guards:

| Module | Method | Key advantage |
|--------|--------|---------------|
| `gmm.py` | GaussianMM | Soft probabilities; BIC model selection |
| `hmm.py` | GaussianHMM | Models temporal autocorrelation natively |
| `markov.py` | MarkovRegression | Univariate recession/expansion classification |
| `density.py` | DBSCAN / HDBSCAN | No k required; handles noise points |
| `spectral.py` | SpectralClustering | Graph-based; handles non-convex clusters |

These are analysis tools for notebooks. The production pipeline uses KMeans only.

### Prediction API — two modules, one consumer each

This split is essential; conflating them causes bugs.

**`prediction/__init__.py` — flat API (production):**
```python
def train_current_regime(X, y, cfg) -> RandomForestClassifier: ...
def train_decision_tree(X, y, cfg) -> DecisionTreeClassifier: ...
def train_forward_classifiers(X, y, cfg) -> dict[int, RandomForestClassifier]: ...
def predict_current(model, X_latest) -> dict: ...  # {"regime": int, "probabilities": {...}}
```
`outputs/models/current_regime.pkl` always contains a **bare** `RandomForestClassifier`.
`pipeline.py` step 7 (`dashboard`) loads this file directly with `joblib.load()`.
Never change this to a dict-wrapped model without updating all three consumers.

**`prediction/classifier.py` — bundle API (tests only):**
```python
def train_current_regime(X, y, cv_splits=5) -> dict:
    # returns {"models": {"rf": ..., "dt": ..., "gb": ...},
    #          "cv_reports": {"rf": [FoldReport, ...], ...},
    #          "labels": [...]}
```
Used exclusively by `test_models_regime.py` and `test_models_reporting.py` to assert on
per-fold CV metrics. Never import from pipeline code.

### Feature alignment for supervised learning

Always align features and labels by index, never by position:
```python
common = features_supervised.index.intersection(labels.index)
X = features_supervised.loc[common].drop(columns=["market_code"], errors="ignore")
X = X.dropna(axis=1, how="any")   # log which cols are dropped
y = labels.loc[common]
```
Using `iloc[:len(labels)]` silently misaligns when clustering drops leading NaN rows.

### Cross-validation

Always `TimeSeriesSplit(n_splits=5)` — never `train_test_split`. Shuffled CV on time-series
data produces wildly optimistic accuracy that collapses in production.

---

## 7. Regime Profiling & Reporting

### Building regime profiles

After clustering, `regime.py:build_profiles()` computes mean and std of every feature
per cluster label. These profiles are the basis for human-readable regime naming.

`suggest_names()` applies simple heuristics: high CPI derivative → inflationary,
high credit spread → recession risk, high S&P momentum → growth boom. The suggested names
are written to `config/regime_labels.yaml` on first run; the human edits them to finalize.

### Transition matrix

```python
def build_transition_matrix(labels: pd.Series) -> pd.DataFrame:
    # counts transitions from regime i → regime j in adjacent quarters
    # normalizes rows to probabilities
    # diagonal = probability of staying in same regime
```

For forward-horizon transition matrices (1Q, 2Q, 4Q, 8Q), use `compute_forward_probabilities()`
which computes empirical P(regime at t+h | regime at t) across all quarters.

### Dashboard output

`reporting.py:print_dashboard()` currently uses `print()` directly (12 calls) — acceptable
for a user-facing terminal function, but consider refactoring to return a string so callers
can log, email, or display it without monkey-patching stdout.

`save_dashboard_csv()` writes `outputs/reports/dashboard.csv` with regime, probabilities,
asset signals, and forward transition probabilities. This CSV is the input to the email
report generator.

---

## 8. Testing Strategy

### Philosophy

- Every library module gets a corresponding `tests/unit/test_{module}.py`.
- No test makes real network calls — mock everything at the boundary.
- Tests must pass with core-only dependencies; optional-dep tests skip gracefully.
- Pipeline step tests redirect all file I/O to `tmp_path` via `monkeypatch`.

### Test file mapping

| Module | Test file | Key fixtures |
|--------|-----------|-------------|
| `config.py` | `test_config.py` | temp settings YAML |
| `runtime.py` | `test_runtime.py` | mock argparse Namespace |
| `checkpoints.py` | `test_checkpoints.py` | `tmp_path` CheckpointManager |
| `transforms.py` | `test_transforms.py` | `quarterly_index`, `raw_macro_df` (conftest) |
| `clustering.py` | `test_clustering.py` | synthetic PCA-ready arrays |
| `regime.py` | `test_regime.py` | synthetic label Series |
| `prediction/__init__.py` | `test_prediction_flat.py` | synthetic X, y |
| `prediction/classifier.py` | `test_models_regime.py` | synthetic X, y |
| `reporting.py` | `test_reporting.py` | mock dashboard CSV |
| `plotting.py` | `test_plotting.py` | `matplotlib.use("Agg")` backend |
| `monitoring.py` | `test_monitoring.py` | synthetic DataFrames |
| `ingestion/multpl.py` | `test_ingestion.py` | `responses` mock or `requests_mock` |
| `ingestion/fred.py` | `test_ingestion.py` | mock fredapi.Fred |
| `ingestion/assets.py` | `test_ingestion.py` | mock yfinance.download |
| `diagnostics.py` | `test_diagnostics_rrg.py` | synthetic price DataFrame |
| `tactics.py` | `test_tactics.py` | synthetic returns DataFrame |
| `email.py` | `test_email_weekly.py` | mock smtplib.SMTP |
| `gmm.py` | `test_gmm.py` | synthetic 2D array |
| `hmm.py` | `test_hmm.py` | `pytest.importorskip("hmmlearn")` |
| `markov.py` | `test_markov.py` | `pytest.importorskip("statsmodels")` |
| `density.py` | `test_density.py` | `pytest.importorskip("hdbscan")` |
| `spectral.py` | `test_spectral.py` | synthetic 2D array |
| `divergence.py` | `test_divergence.py` | synthetic price pairs |
| `momentum.py` | `test_momentum.py` | synthetic macro DataFrame |

### `conftest.py` — shared fixtures (build early)

```python
@pytest.fixture
def quarterly_index():
    return pd.date_range("1980-01-01", periods=80, freq="QE")

@pytest.fixture
def raw_macro_df(quarterly_index):
    # synthetic DataFrame with all columns needed by transforms
    # use np.random.default_rng(42) for reproducibility
    ...
```

Build these fixtures before writing any transform tests. Synthetic data must have the
right column names (matching settings.yaml) or tests will fail silently with KeyError.

### Optional-dependency test skipping

```python
# At module level in test file:
pytest.importorskip("hmmlearn")
# OR per test:
@pytest.mark.skipif(not HAS_HDBSCAN, reason="hdbscan not installed")
```

Never use `try/except ImportError` in tests — use `pytest.importorskip` or `skipif`.
Document which optional deps are needed for full (zero-skip) test runs.

### Pipeline step tests

Redirect all file I/O using monkeypatch:
```python
def test_step1_ingest(monkeypatch, tmp_path, mocker):
    import pipelines.step_01_ingest as step
    monkeypatch.setattr(step, "DATA_DIR", tmp_path)
    mocker.patch("trading_crab_lib.ingestion.multpl.fetch_all", return_value=mock_df)
    mocker.patch("trading_crab_lib.ingestion.fred.fetch_all", return_value=mock_df)
    step.main()
    assert (tmp_path / "raw" / "macro_raw.parquet").exists()
```
This pattern prevents tests from corrupting production checkpoints.

### Plotting tests

```python
import matplotlib
matplotlib.use("Agg")  # non-interactive backend; must be set before pyplot import

def test_plot_something(tmp_path):
    run_cfg = RunConfig(save_plots=True, show_plots=False, generate_plots=True)
    run_cfg.output_dir = tmp_path
    fig = plot_something(df, run_cfg=run_cfg)
    assert (tmp_path / "03_something.png").exists()
```

Test that plot functions: (a) don't raise, (b) produce the expected output file,
(c) handle empty DataFrames gracefully.

### Determinism tests

```python
def test_engineer_all_deterministic(raw_macro_df, cfg):
    result1 = engineer_all(raw_macro_df.copy(), cfg)
    result2 = engineer_all(raw_macro_df.copy(), cfg)
    pd.testing.assert_frame_equal(result1, result2)
```

Run this test whenever transforms.py changes. Non-determinism in feature engineering
is the hardest bug to debug because symptoms appear in the email report, not in unit tests.

---

## 9. Packaging & Distribution

### Two pyproject.toml files

**Root `pyproject.toml`** — the `trading-crab` application package:
- `[tool.setuptools.packages.find] where=["src"] include=["trading_crab"]`
- `[project.scripts]` with three entry points
- `[tool.uv.workspace] members=["src/trading_crab_lib"]`
- `[tool.pytest.ini_options] testpaths=["tests"] pythonpath=["src"]`

**`src/trading_crab_lib/pyproject.toml`** — the `trading-crab-lib` library:
- `[tool.setuptools.packages.find] where=[".."] include=["trading_crab_lib*"]`
  (note: `where` is the parent of `pyproject.toml`, i.e. `src/`)
- Core deps: pandas, numpy, pyarrow, scikit-learn, scipy, pyyaml, joblib, python-dotenv
- Optional extras: ingestion, plotting, hmm, clustering-extras, boosting, all, dev
- `[tool.setuptools.package-data] trading_crab_lib=["py.typed"]`

### uv workspace (strongly recommended)

```toml
# root pyproject.toml
[tool.uv.workspace]
members = ["src/trading_crab_lib"]
```

`uv sync` installs both packages in editable mode from one command. Without this,
developers must remember to `pip install -e src/trading_crab_lib/` separately.

### CI/CD workflows

Use separate GitHub Actions workflows for library vs application releases:

| Workflow | Trigger | Publishes |
|----------|---------|-----------|
| `publish-lib.yml` | tag `lib-v*` | `trading-crab-lib` to PyPI |
| `publish-app.yml` | tag `v*` | `trading-crab` to PyPI |
| `python-package.yml` | push/PR to main | runs test suite |

Avoid a single `publish.yml` that tries to publish both — it creates ambiguity about
which version changed and which PyPI token to use.

### Settings.yaml is NOT shipped with the library

`trading-crab-lib` has no config file. The library accepts a config dict at runtime.
`trading-crab` (app) ships `settings.example.yaml` as package data and `tradingcrab-setup`
generates `config/settings.yaml` from it interactively.

Long-term: library should accept `config: dict | Path | None = None` on all public
functions, defaulting to the standard file path only when not provided. This enables
clean `pip install trading-crab-lib` usage without a git clone.

### Environment variables for Docker/CI

```bash
TC_ROOT_DIR=/app          # override repo root detection
TC_CONFIG_DIR=/app/config
TC_DATA_DIR=/data          # mount a volume here
TC_OUTPUT_DIR=/outputs
FRED_API_KEY=xxx           # loaded from .env via python-dotenv
```

`__init__.py` reads these at import time. This makes the library fully relocatable
without changing any Python code — essential for Docker deployments.

### MANIFEST.in

Library sdist needs only Python files and `py.typed`:
```
include src/trading_crab_lib/py.typed
prune tests
prune notebooks
prune trading-crab-lib   # exclude submodule
```

Application sdist needs config examples:
```
include config/settings.example.yaml
include config/email.example.yaml
include .env.example
```

---

## 10. Lessons Learned

These are mistakes made during the original development that would be avoided
in a rebuild. Read this section before writing any code.

### L1. Build the checkpoint system before writing any pipeline code

The temptation is to build the algorithm first and add persistence later.
This leads to: monolithic scripts that re-run everything on each execution,
manual parquet saves with inconsistent naming, no freshness checking, and
test suites that corrupt production data.

`CheckpointManager` should be the second thing you build (after `config.py`).
Wire every step through it from the start.

### L2. Two feature files from day one

Starting with one `features.parquet` and splitting later required changing
every downstream reference and re-validating that centered/causal were not mixed.
Create `features.parquet` (centered) and `features_supervised.parquet` (causal)
from the very first working version of step 2.

### L3. Config in YAML, not Python constants

When `initial_features` and `clustering_features` were Python lists, every change
required editing Python source, running tests, and committing. Moving them to
`settings.yaml` made experimentation instant. Do this from day one.

### L4. The two-prediction-API split prevents a subtle breakage

If the bundle-API dict `{"models": {"rf": ...}}` were ever pickled as
`current_regime.pkl`, then `pipelines/07_dashboard.py` would fail at:
```python
model = joblib.load("current_regime.pkl")
model.predict(X)   # AttributeError: dict has no attribute 'predict'
```
Keep `prediction/__init__.py` (flat API) and `prediction/classifier.py` (bundle API)
strictly separated. Document which tests import from which module.

### L5. Global numpy seed must be set at pipeline entry, not in each module

Setting `random_state=42` in every sklearn model call is not sufficient for full
reproducibility. `np.random.seed(42)` must be called once at `pipeline.py:main()`
before any library code runs. Without this, any code path that uses `np.random`
without an explicit seed will produce different results across runs.

### L6. `market_code` in gap-fill causes non-determinism

The gap-fill function originally used `df[[col, "market_code"]].dropna()` to find
valid (non-NaN) rows. Since `market_code` has different NaN patterns depending on
which label source was loaded (`--market-code grok` vs `--market-code clustered`),
the gap-fill boundaries change, altering all derivative values, which changes the
model. Remove `market_code` from any valid-row calculation.

### L7. `dropna(axis=1)` in step 5 silently drops columns

The line `X = features.dropna(axis=1, how="any")` is fragile: a single NaN in any
column removes it entirely from the feature set. Different pipeline runs (with different
`market_code` sources or gap-fill results) produce different feature sets, giving the
RF different inputs. Fix: either log every dropped column at WARNING, or pin the column
list explicitly in `settings.yaml`.

### L8. ETF derivatives can't go in `clustering_features`

ETF price history starts in 1993–2004. The macro clustering dataset covers 1950–present
(~305 quarters). Adding ETF derivatives to `clustering_features` forces `dropna()` to
discard all pre-1993 rows, losing 55 years of regime history. Keep ETF derivatives in
`initial_features` only (for supervised learning). Use gold and oil from macrotrends
(history back to 1915/1946) for clustering.

### L9. Seaborn pairplot is unusably slow on 69 features

69×69 = 4761 subplots. Disable `generate_pairplot` by default. Gate it behind
`--pairplot` flag and document the runtime (~10 minutes on a laptop).

### L10. The pipeline monolith grows to 1400+ lines fast

Even with all algorithm logic in library modules, the orchestration for 9 steps
accumulates quickly: argparse, step dispatch, checkpoint logic, monitoring hooks,
plot generation, timing. Keep it organized with clear step boundaries and a single
`run_step(name, fn, args...)` wrapper that handles timing, logging, and error recovery.

### L11. Email config keys must match between load function and YAML example

`load_email_config()` expects `from_address` / `to_address`. If the YAML example
uses `sender` / `recipients` (a name imported from another project), every user
who copies the example gets a KeyError at send time. Keep key names consistent
across code, YAML example, and documentation.

### L12. Submodule references accumulate and confuse new developers

Three read-only submodules (`gsd-scratch-work`, `trading-crab`, `trading-crab-lib`)
exist as historical references. In a rebuild, decide upfront whether you need
submodules at all. If the reference implementation is a single script, copy it into
`legacy/` and don't add submodules. Submodules add `git submodule update` overhead
to every fresh clone.

### L13. Test count is not the same as coverage quality

571 tests were collected but `src/trading_crab/pipeline.py` (1374 lines) had zero
test coverage. A large test suite with gaps in critical orchestration code is worse
than a smaller suite with complete coverage of the most important paths. Prioritize
testing the pipeline's step dispatch, argument parsing, and checkpoint handoff.

### L14. CI workflow proliferation

6 workflow files accumulated over time (`publish.yml`, `publish-app.yml`,
`publish-lib.yml`, `python-app.yml`, `python-package.yml`, `python-publish.yml`).
Some were duplicates from before the 2-package split. Start with 3: test, publish-lib,
publish-app. Add more only when needed.

---

## 11. Critical Invariants (Never Break These)

These are the rules that, if violated, produce silent wrong answers rather than errors.
They are the hardest class of bug to diagnose.

### I1. Feature pipeline order is fixed

```
cross-ratios → yield curve → divergence (level) → momentum → log → select →
gap-fill → derivatives → divergence (derivative) → select
```
Specifically: **gap fill must happen after log transform** (Bernstein interpolates
in log space). Changing this order will produce different results that look plausible
but are wrong.

### I2. `balanced_cluster` is the primary regime assignment

Use `balanced_cluster` (from `KMeansConstrained`) for all downstream steps.
`cluster` (standard KMeans best-k) is kept for geometric reference only.
Never swap these labels without explicit benchmarking and YAML update.

### I3. GDP and GNP are always shifted +1 quarter

`shift: true` in `settings.yaml` for `fred_gdp` and `fred_gnp`. The BEA releases
the advance estimate ~30 days after quarter end; you cannot know Q1 GDP at Q1 end.
Not shifting introduces look-ahead bias that inflates model accuracy by ~10–15%.

### I4. PCA = 5 components (fixed, not variance-threshold)

Variance-threshold PCA shifts as new data arrives, making results non-reproducible.
5 components were established by scree plot analysis on the actual data. Benchmark
silhouette scores at 3, 5, 7, 10 before changing.

### I5. `current_regime.pkl` always contains a bare `RandomForestClassifier`

`pipeline.py` step 7 does `model.predict(X_latest)` directly on the loaded object.
If the bundle-API dict is ever saved here, the dashboard silently crashes.
The flat API `train_current_regime()` returns a bare classifier; always use it
for the production model file.

### I6. `features_supervised.parquet` for supervised steps; `features.parquet` for clustering

Never swap. Centered smoothing (features.parquet) uses 2 future quarters in rolling
windows — a model trained on it cannot be scored on real-time causal data.

### I7. Feature and label alignment by index intersection, not position

```python
common = features.index.intersection(labels.index)
X = features.loc[common]
y = labels.loc[common]
```
`iloc[:len(labels)]` silently misaligns if any rows were dropped by `dropna()`.

### I8. No committed secrets

Never commit `.env`, `email.yaml`, `portfolio.local.yaml`, or any API key.
Add them to `.gitignore` before writing the files.

### I9. No network calls in tests

Mock all HTTP clients (`requests`, `fredapi`, `yfinance`). Tests must pass offline,
in CI, and on any machine without API keys.

### I10. Submodules are read-only

`gsd-scratch-work/`, `trading-crab-lib/`, `trading-crab/` are references only.
`git pull` and `git submodule update` are allowed. Modifying or pushing to them
corrupts the historical record.

---

## Appendix: Quick Reference Commands

```bash
# Install both packages for development
pip install -e "src/trading_crab_lib/[all,dev]"
pip install -e ".[dev]"
# OR: uv sync

# Full pipeline run (re-scrape + recompute + plots)
tradingcrab --refresh --recompute --plots

# Run specific steps only
tradingcrab --steps 3,4,5 --plots

# Run tests (requires all optional deps for zero skips)
pytest tests/ -v

# Check checkpoint state
python -c "from trading_crab_lib.checkpoints import CheckpointManager; print(CheckpointManager().list())"

# Verify feature pipeline determinism
python -c "
from trading_crab_lib.transforms import engineer_all
from trading_crab_lib.config import load
import pandas as pd, numpy as np
cfg = load()
# load your features and verify engineer_all(df) == engineer_all(df)
"

# Build distribution packages
python -m build src/trading_crab_lib/   # library wheel + sdist
python -m build .                        # app wheel + sdist
```

---

*This guide was generated on 2026-03-31 from the Trading-Crab codebase at
commit on branch `claude/review-meta-plan-Sqot2`. Refer to `CLAUDE.md` for
the full architecture decision record and `STATE.md` for current pipeline status.*
