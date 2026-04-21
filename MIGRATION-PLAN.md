# MIGRATION-PLAN.md — Incremental Migration from claude-scratch-work to trading-crab

**Purpose:** Step-by-step guide for migrating validated code from `strycker/claude-scratch-work`
to the public repo `strycker/trading-crab`, one pipeline stage at a time.

**Target repo:** `https://github.com/strycker/trading-crab`
**Structure:** Identical two-package layout (`trading-crab` + `trading-crab-lib`) to this repo.
**Delivery format:** For any step tag (e.g. `S1.2`), ask: _"Show me the code changes for S1.2"_
and the exact file contents will be generated for copy-paste into the target repo.

---

## Phase Status Summary

| Phase | Status | Notes |
|-------|--------|-------|
| Q0 — Foundation | ✅ Complete | S0.1–S0.7 done; plus S9.3/S9.5 work done early |
| Q1 — Data Ingestion | 🔲 Next | Begin with S1.1 |
| Q2 — Feature Engineering | 🔲 | |
| Q3 — Clustering | 🔲 | |
| Q4 — Regime Labeling | 🔲 | |
| Q5 — Prediction | 🔲 | |
| Q6 — Asset Returns & Dashboard | 🔲 | |
| Q7 — Diagnostics & Tactics | 🔲 | |
| Q8 — Pipeline Orchestration & CLI | 🔲 | Stubs exist (cli.py, pipeline.py) |
| Q9 — DevOps & Packaging | 🟡 Partial | CI/publish/pylint/ruff/mypy/Poetry done |

---

## How to Use This Plan

1. Work through Q-phases in order (Q0 → Q9).
2. Within each Q-phase, implement S-steps in order.
3. After each S-step: run `pytest` on the target repo, open the notebook if one is included,
   and verify manually before moving to the next step.
4. Commit to `trading-crab` after each S-step passes your human validation.
5. Never skip ahead — later steps depend on earlier ones compiling and passing tests.

### Tag format
- `S0.1` = Q-phase 0, step 1
- Ask: _"Show me the code changes for S3.4"_ → get exact file contents

---

## Q0 — Foundation: Repo Skeleton & Core Infrastructure ✅ COMPLETE

**Goal:** An installable but empty shell. Both packages install, imports work, tests run (0 tests pass, 0 fail).

| Tag | Status | Description | Files Created/Modified |
|-----|--------|-------------|------------------------|
| S0.1 | ✅ | Root repo files | `README.md`, `.gitignore`, `.env.example`, `Makefile`, `pyproject.toml` (root/app) |
| S0.2 | ✅ | Library pyproject + package skeleton | `src/trading_crab_lib/pyproject.toml`, `src/trading_crab_lib/__init__.py`, `src/trading_crab_lib/py.typed` |
| S0.3 | ✅ | App package skeleton | `src/trading_crab/__init__.py`, `src/trading_crab/cli.py` (stub), `src/trading_crab/pipeline.py` (stub) |
| S0.4 | ✅ | Core library modules | `src/trading_crab_lib/runtime.py`, `src/trading_crab_lib/config.py` (load skeleton only), `src/trading_crab_lib/checkpoints.py` |
| S0.5 | ✅ | Plotting & monitoring package skeletons | `src/trading_crab_lib/plotting/__init__.py`, `src/trading_crab_lib/plotting/core.py`, `src/trading_crab_lib/monitoring/__init__.py` |
| S0.6 | ✅ | Test infrastructure | `tests/__init__.py`, `tests/conftest.py`, `tests/unit/__init__.py`, `tests/integration/__init__.py` |
| S0.7 | ✅ | Settings skeleton + regime labels | `config/settings.yaml` (empty sections only), `config/regime_labels.yaml` (empty) |

**Validation after Q0:**
```bash
pip install -e "src/trading_crab_lib/[dev]" && pip install -e ".[dev]"
python -c "from trading_crab_lib import RunConfig, CheckpointManager; print('OK')"
pytest tests/ -v  # 0 tests, should collect with no errors
```

---

## Q1 — Step 1: Data Ingestion

**Goal:** `pipelines/01_ingest.py` runs end-to-end (with real or mocked API keys),
producing `data/checkpoints/macro_raw.parquet` and `data/raw/asset_prices.parquet`.
Notebook 01 renders fully.

**Config additions (settings.yaml):** `data:`, `fred.series:`, `multpl.datasets:`, `macrotrends:` sections.

| Tag | Description | Files Created/Modified |
|-----|-------------|------------------------|
| S1.1 | settings.yaml — data + fred sections | `config/settings.yaml` (data, fred.series: GDP/GNP/BAA/AAA/CPI/GS10/TB3MS/VIXCLS/UNRATE/M2SL/M2NS/GS2/T10Y2Y/T10Y3M) |
| S1.2 | settings.yaml — multpl + macrotrends sections | `config/settings.yaml` (multpl.datasets: all 46 series; macrotrends: gold/oil/silver URLs) |
| S1.3 | Ingestion: multpl scraper | `src/trading_crab_lib/ingestion/__init__.py` (stub), `src/trading_crab_lib/ingestion/multpl.py`, `src/trading_crab_lib/ingestion/grok.py` |
| S1.4 | Ingestion: FRED fetcher | `src/trading_crab_lib/ingestion/fred.py` |
| S1.5 | Ingestion: assets + macrotrends | `src/trading_crab_lib/ingestion/assets.py`, `src/trading_crab_lib/ingestion/macrotrends.py` |
| S1.6 | Ingestion package init (completeness report) | `src/trading_crab_lib/ingestion/__init__.py` (full: `CompletenessReport`, `ingestion_completeness_report()`) |
| S1.7 | config.py finalized for step 1 | `src/trading_crab_lib/config.py` (full `load()`, `validate_config()`, `load_portfolio()`, `setup_logging()`) |
| S1.8 | Monitoring & plotting for step 1 | `src/trading_crab_lib/monitoring/ingestion.py`, `src/trading_crab_lib/plotting/ingestion.py` |
| S1.9 | Pipeline script | `pipelines/01_ingest.py` |
| S1.10 | Notebook 01 | `notebooks/01_ingestion.ipynb` |
| S1.11 | Tests for step 1 | `tests/unit/test_ingestion.py`, `tests/unit/test_macrotrends.py`, `tests/unit/test_checkpoints.py`, `tests/unit/test_config.py`, `tests/unit/test_runtime.py`, `tests/unit/test_ingestion_completeness.py`, `tests/unit/test_fred_series_config.py` |

**Validation after Q1:**
```bash
pytest tests/unit/test_ingestion.py tests/unit/test_checkpoints.py tests/unit/test_config.py -v
# Set FRED_API_KEY in .env, then:
python pipelines/01_ingest.py
ls data/checkpoints/macro_raw.parquet  # should exist
jupyter lab notebooks/01_ingestion.ipynb  # run all cells
```

---

## Q2 — Step 2: Feature Engineering

**Goal:** `pipelines/02_features.py` runs, producing `data/checkpoints/features.parquet`
and `data/checkpoints/features_supervised.parquet`. Notebooks 02 and 08 render fully.

**Config additions:** `features:` section (log_columns, initial_features, clustering_features,
derivative_window, divergence pairs, asset_price_columns).

| Tag | Description | Files Created/Modified |
|-----|-------------|------------------------|
| S2.1 | settings.yaml — features section | `config/settings.yaml` (log_columns, initial_features, clustering_features, derivative_window) |
| S2.2 | settings.yaml — divergence + asset_price_columns | `config/settings.yaml` (features.divergence pairs/windows, features.asset_price_columns) |
| S2.3 | Core transforms module | `src/trading_crab_lib/transforms.py` (cross_ratios, log, gap-fill, derivatives, `engineer_all`) |
| S2.4 | Yield curve features | `src/trading_crab_lib/yield_curve_features.py` |
| S2.5 | Divergence features | `src/trading_crab_lib/divergence.py` |
| S2.6 | Momentum features | `src/trading_crab_lib/momentum.py` |
| S2.7 | Composite indicators | `src/trading_crab_lib/indicators.py` |
| S2.8 | Monitoring & plotting for step 2 | `src/trading_crab_lib/monitoring/features.py`, `src/trading_crab_lib/plotting/features.py` |
| S2.9 | Pipeline script | `pipelines/02_features.py` |
| S2.10 | Notebooks 02 + 08 | `notebooks/02_features.ipynb`, `notebooks/08_raw_series.ipynb` |
| S2.11 | Tests for step 2 | `tests/unit/test_transforms.py`, `tests/unit/test_yield_curve_features.py`, `tests/unit/test_divergence.py`, `tests/unit/test_momentum.py`, `tests/unit/test_indicators.py` |

**Validation after Q2:**
```bash
pytest tests/unit/test_transforms.py tests/unit/test_divergence.py tests/unit/test_momentum.py -v
python pipelines/02_features.py
python -c "import pandas as pd; df=pd.read_parquet('data/checkpoints/features.parquet'); print(df.shape)"
jupyter lab notebooks/02_features.ipynb  # run all cells, inspect gap-fill plots
```

---

## Q3 — Step 3: Clustering

**Goal:** `pipelines/03_cluster.py` runs, producing `data/checkpoints/cluster_labels.parquet`.
Notebooks 03 and 10 render fully with PCA scatter, silhouette, GMM BIC surface.

**Config additions:** `clustering:` section (n_pca_components, n_clusters_search, k_cap, balanced_k).

| Tag | Description | Files Created/Modified |
|-----|-------------|------------------------|
| S3.1 | settings.yaml — clustering section | `config/settings.yaml` (clustering: n_pca_components=5, n_clusters_search, k_cap, balanced_k) |
| S3.2 | Core clustering module | `src/trading_crab_lib/clustering.py` (reduce_pca, evaluate_kmeans, pick_best_k, fit_clusters, gap statistic, SVD) |
| S3.3 | GMM module | `src/trading_crab_lib/gmm.py` |
| S3.4 | Density clustering (DBSCAN/HDBSCAN) | `src/trading_crab_lib/density.py` |
| S3.5 | Spectral clustering | `src/trading_crab_lib/spectral.py` |
| S3.6 | HMM module | `src/trading_crab_lib/hmm.py` |
| S3.7 | Markov regime-switching | `src/trading_crab_lib/markov.py` |
| S3.8 | Cluster comparison utilities | `src/trading_crab_lib/cluster_comparison.py` |
| S3.9 | Monitoring & plotting for step 3 | `src/trading_crab_lib/monitoring/clustering.py`, `src/trading_crab_lib/plotting/clustering.py` |
| S3.10 | Pipeline script | `pipelines/03_cluster.py` |
| S3.11 | Notebooks 03 + 10 | `notebooks/03_clustering.ipynb`, `notebooks/10_model_comparison.ipynb` |
| S3.12 | Tests for step 3 | `tests/unit/test_clustering.py`, `tests/unit/test_clustering_exploration.py`, `tests/unit/test_gmm.py`, `tests/unit/test_density.py`, `tests/unit/test_spectral.py`, `tests/unit/test_hmm.py`, `tests/unit/test_markov.py`, `tests/unit/test_cluster_comparison.py` |

**Validation after Q3:**
```bash
pytest tests/unit/test_clustering.py tests/unit/test_gmm.py -v
python pipelines/03_cluster.py
jupyter lab notebooks/03_clustering.ipynb  # verify PCA scatter colored by regime
```

---

## Q4 — Step 4: Regime Labeling

**Goal:** `pipelines/04_regime_label.py` runs, producing named regime profiles and transition matrix.
Notebooks 04 and 07 render fully.

**Config additions:** `config/regime_labels.yaml` (populated after first clustering run).

| Tag | Description | Files Created/Modified |
|-----|-------------|------------------------|
| S4.1 | Regime module | `src/trading_crab_lib/regime.py` (build_profiles, suggest_names, build_transition_matrix, compute_forward_probabilities) |
| S4.2 | Regime labels config | `config/regime_labels.yaml` (skeleton — fill in after running clustering) |
| S4.3 | Plotting for step 4 | `src/trading_crab_lib/plotting/regime.py` |
| S4.4 | Pipeline script | `pipelines/04_regime_label.py` |
| S4.5 | Notebooks 04 + 07 | `notebooks/04_regimes.ipynb`, `notebooks/07_pairplot.ipynb` |
| S4.6 | Tests for step 4 | `tests/unit/test_regime.py`, `tests/unit/test_forward_probabilities.py` |

**Validation after Q4:**
```bash
pytest tests/unit/test_regime.py tests/unit/test_forward_probabilities.py -v
python pipelines/04_regime_label.py
jupyter lab notebooks/04_regimes.ipynb  # verify timeline + transition matrix heatmap
# Edit config/regime_labels.yaml with human-chosen names before proceeding
```

---

## Q5 — Step 5: Prediction

**Goal:** `pipelines/05_predict.py` runs, producing `outputs/models/current_regime.pkl`
(bare RandomForestClassifier). Notebooks 05 and 11 render fully.

**Config additions:** `prediction:` section (forward_horizons_quarters, cv_splits, dt_max_depth, rf_max_depth).

| Tag | Description | Files Created/Modified |
|-----|-------------|------------------------|
| S5.1 | settings.yaml — prediction section | `config/settings.yaml` (prediction: cv_splits=5, dt_max_depth=8, rf_max_depth=12, forward_horizons_quarters) |
| S5.2 | Flat prediction API | `src/trading_crab_lib/prediction/__init__.py` (train_current_regime, train_decision_tree, train_lightgbm, train_forward_classifiers, predict_current) |
| S5.3 | Bundle API (for tests) | `src/trading_crab_lib/prediction/classifier.py`, `src/trading_crab_lib/prediction/gradient_boosting.py` |
| S5.4 | Monitoring & plotting for step 5 | `src/trading_crab_lib/monitoring/prediction.py`, `src/trading_crab_lib/plotting/prediction.py` |
| S5.5 | Pipeline script | `pipelines/05_predict.py` |
| S5.6 | Notebooks 05 + 11 | `notebooks/05_prediction.ipynb`, `notebooks/11_feature_selection.ipynb` |
| S5.7 | Tests for step 5 | `tests/unit/test_prediction_flat.py`, `tests/unit/test_lightgbm.py`, `tests/test_models_regime.py`, `tests/test_models_boosting.py`, `tests/test_models_interpret_tree.py`, `tests/test_models_behavior.py` |

**Validation after Q5:**
```bash
pytest tests/unit/test_prediction_flat.py tests/test_models_regime.py -v
python pipelines/05_predict.py
ls outputs/models/current_regime.pkl  # should be a bare RandomForestClassifier
jupyter lab notebooks/05_prediction.ipynb  # verify CV fold chart + decision tree plot
```

---

## Q6 — Steps 6–7: Asset Returns, Dashboard & Email

**Goal:** `pipelines/06_asset_returns.py` and `07_dashboard.py` run end-to-end,
producing `outputs/reports/dashboard.csv` and `outputs/reports/weekly_report.md`.
Notebook 06 renders fully.

**Config additions:** `assets:` section (ETF universe), `dashboard:` section, `pipeline.random_state`.

| Tag | Description | Files Created/Modified |
|-----|-------------|------------------------|
| S6.1 | settings.yaml — assets + dashboard sections | `config/settings.yaml` (assets: etf tickers list; dashboard: top_n, etc.) |
| S6.2 | Asset returns module | `src/trading_crab_lib/asset_returns.py` |
| S6.3 | Reporting module | `src/trading_crab_lib/reporting.py` |
| S6.4 | Email module | `src/trading_crab_lib/email.py` |
| S6.5 | Email config example | `config/email.example.yaml` |
| S6.6 | Monitoring for steps 6-7 | `src/trading_crab_lib/monitoring/pipeline.py` |
| S6.7 | Plotting for step 6 | `src/trading_crab_lib/plotting/assets.py` |
| S6.8 | Pipeline scripts | `pipelines/06_asset_returns.py`, `pipelines/07_dashboard.py` |
| S6.9 | Notebook 06 | `notebooks/06_assets.ipynb` |
| S6.10 | Tests for steps 6-7 | `tests/unit/test_returns.py`, `tests/test_models_reporting.py`, `tests/unit/test_reporting.py`, `tests/test_email_weekly.py` |

**Validation after Q6:**
```bash
pytest tests/unit/test_returns.py tests/unit/test_reporting.py tests/test_email_weekly.py -v
python pipelines/06_asset_returns.py && python pipelines/07_dashboard.py
cat outputs/reports/weekly_report.md  # should show current regime + asset rankings
jupyter lab notebooks/06_assets.ipynb  # verify violin plots + Sharpe table
```

---

## Q7 — Steps 8–9: Diagnostics & Tactics

**Goal:** `pipelines/08_diagnostics.py` and `09_tactics.py` run end-to-end.
Notebooks 09 and 12 render fully.

**Config additions:** `tactics:` section (vol_threshold, trend_threshold, benchmark).

| Tag | Description | Files Created/Modified |
|-----|-------------|------------------------|
| S7.1 | settings.yaml — tactics section | `config/settings.yaml` (tactics: thresholds, benchmark ticker) |
| S7.2 | Diagnostics module | `src/trading_crab_lib/diagnostics.py` (rolling_zscore, percentile_rank, normalize_100, compute_rrg) |
| S7.3 | Tactics module | `src/trading_crab_lib/tactics.py` (compute_tactics_metrics, classify_tactics) |
| S7.4 | Plotting for steps 8-9 | `src/trading_crab_lib/plotting/diagnostics.py` (plot_rrg_scatter, plot_divergence_timeseries) |
| S7.5 | Pipeline scripts | `pipelines/08_diagnostics.py`, `pipelines/09_tactics.py` |
| S7.6 | Notebooks 09 + 12 | `notebooks/09_diagnostics.ipynb`, `notebooks/12_divergence_momentum.ipynb` |
| S7.7 | Tests for steps 8-9 | `tests/unit/test_diagnostics_rrg.py`, `tests/unit/test_tactics.py` |

**Validation after Q7:**
```bash
pytest tests/unit/test_diagnostics_rrg.py tests/unit/test_tactics.py -v
python pipelines/08_diagnostics.py && python pipelines/09_tactics.py
jupyter lab notebooks/09_diagnostics.ipynb  # verify RRG 4-quadrant scatter
```

---

## Q8 — Pipeline Orchestration & CLI

**Goal:** `tradingcrab --help` works; `run_pipeline.py --steps 1,2,3` runs the full
orchestrated pipeline with checkpointing, logging, timing, and `--plots` support.
All integration + smoke tests pass.

| Tag | Description | Files Created/Modified |
|-----|-------------|------------------------|
| S8.1 | settings.yaml — pipeline section | `config/settings.yaml` (pipeline: random_state=42) |
| S8.2 | Full pipeline orchestrator | `src/trading_crab/pipeline.py` (all 9 steps, STEPS dict, main(), argparse) |
| S8.3 | CLI entry points | `src/trading_crab/cli.py` (tradingcrab, tradingcrab-setup, tradingcrab-publish) |
| S8.4 | Backward-compat shim | `run_pipeline.py` |
| S8.5 | Weekly report script | `scripts/run_weekly_report.py`, `scripts/setup.sh`, `scripts/jupyter_notebook_local.sh` |
| S8.6 | Remaining constraints tests | `tests/test_constraints_etf_universe.py`, `tests/test_constraints_frequency.py` |
| S8.7 | Pipeline + CLI smoke tests | `tests/test_pipeline_smoke.py`, `tests/test_cli_smoke.py`, `tests/test_scripts_weekly_report.py` |
| S8.8 | Mini-pipeline integration test | `tests/integration/test_mini_pipeline.py` |

**Validation after Q8:**
```bash
pytest tests/ -v --tb=short  # full suite should pass
tradingcrab --help
python run_pipeline.py --steps 3,4,5 --plots --verbose
# Verify outputs/plots/ is populated
```

---

## Q9 — Remaining Tests, DevOps & Packaging 🟡 PARTIAL

**Goal:** Full test suite green; Docker build succeeds; GitHub Actions CI passes on push;
`pip install trading-crab-lib` from PyPI (test index) works cleanly.

> **Already done in `trading-crab` ahead of schedule:**
> - S9.3 — `ci.yml` (matrix CI) and `publish-pypi.yml` (consolidated) are live
> - S9.5 — ruff and mypy wired into CI; `.pylintrc` present; `poetry.toml` sections added
>
> Items below are what remains.

| Tag | Status | Description | Files Created/Modified |
|-----|--------|-------------|------------------------|
| S9.1 | 🔲 | Remaining unit tests | `tests/unit/test_init_module.py`, `tests/unit/test_plotting.py`, `tests/unit/test_monitoring.py`, `tests/unit/test_confusion_matrix_plot.py` |
| S9.2 | 🔲 | Constraint + legacy tests | `tests/test_pipelines_ingest_features.py`, `tests/unit/test_cluster_comparison.py` |
| S9.3 | ✅ | GitHub Actions workflows | `ci.yml` (matrix CI), `publish-pypi.yml` (consolidated both packages) |
| S9.4 | 🔲 | Docker support | `Dockerfile`, `docker-compose.yml`, `.dockerignore` |
| S9.5 | 🟡 | Pre-commit + linters + Poetry | `.pylintrc` ✅, ruff in CI ✅, mypy in CI ✅, Poetry `[tool.poetry]` sections ✅, `.pre-commit-config.yaml` 🔲 |
| S9.6 | 🔲 | CLAUDE.md + STATE.md for target repo | `CLAUDE.md` (adapted for trading-crab), `STATE.md`, `ROADMAP.md` |

**Validation after Q9:**
```bash
pytest tests/ -v  # full suite, all passing
poetry install && poetry run pytest tests/ -v  # verify Poetry path works
docker build --target pipeline -t trading-crab:local .
docker run --rm trading-crab:local tradingcrab --help
pip install -e "src/trading_crab_lib/[all,dev]" && pip install -e ".[dev]"
# Submit to test.pypi.org, verify installability from there
```

---

## Cross-Cutting Notes

### settings.yaml build-up order
Each Q-phase adds clearly delimited sections. Use comments to mark which phase added what:
```yaml
# ── Q1: Data Ingestion ─────────────────────────────────────
data:
  start_date: "1950-01-01"
  ...

# ── Q2: Feature Engineering ────────────────────────────────
features:
  log_columns: [...]
  ...
```

### What migrates with each S-step
Every S-step produces a self-contained set of changes. When you ask "Show me S2.3", you'll receive:
- Full file contents for each new file
- Diff/additions for any modified files (e.g. `settings.yaml`)
- Exact pytest commands to validate

### Dependencies between Q-phases
```
Q0 (foundation) → Q1 (ingest) → Q2 (features) → Q3 (cluster)
                                                      ↓
                                               Q4 (regimes)
                                                      ↓
                                               Q5 (predict)
                                                      ↓
                                          Q6 (assets/dashboard)
                                                      ↓
                                          Q7 (diagnostics/tactics)
                                                      ↓
                                          Q8 (CLI/orchestration)
                                                      ↓
                                            Q9 (devops/packaging)
```
Do not start Q*n+1* until Q*n* is fully human-validated in the target repo.

### What is NOT migrated
- `legacy/` — reference only; stays in claude-scratch-work
- `gsd-scratch-work/` and `trading-crab-lib/` submodules — not applicable in target repo
- `data/` and `outputs/` — gitignored, never committed
- Any pickle snapshots in `data/*.pickle` — not migrated

### Notebook validation checklist (per Q-phase)
For each notebook migrated:
- [ ] All cells run top-to-bottom without errors
- [ ] Plots display correctly (no blank figures)
- [ ] Data shapes printed match expected values
- [ ] No hardcoded paths that break outside claude-scratch-work
