# Trading-Crab — Known Pitfalls and Gotchas

A collection of traps, anti-patterns, and non-obvious failures discovered
during development. Read before making changes.

---

## Look-ahead Bias (the #1 Financial ML Sin)

### P1. Using centered rolling windows for supervised learning

**Symptom:** Model accuracy looks great but real-time predictions are wrong.

**Cause:** `rolling(window=5, center=True)` for derivatives uses 2 future quarters
in every window. A model trained on centered features can only be scored on
centered features — which requires knowing the future.

**Fix:** Always use `features_supervised.parquet` (causal=True) for steps 5-7.
`features.parquet` (causal=False) is for clustering steps 3-4 only. Never swap them.

**Detection:** if step 5 loads `features.parquet` instead of `features_supervised.parquet`,
the training pipeline will "work" but the model will be biased toward future data.

---

### P2. Not applying publication-lag shifts to GDP/GNP

**Symptom:** Model learns to use Q1 GDP to predict Q1 regime label.

**Cause:** BEA releases Q1 GDP approximately 30-90 days after Q1 ends.
The quarterly-resampled value in FRED appears as of Q1 end, but it wasn't
available until Q2.

**Fix:** `shift: true` in `config/settings.yaml` for `fred_gdp` and `fred_gnp`.
This shifts those columns forward by one quarter before feature engineering.

**Rule:** any FRED series with significant revision history and a release lag
longer than one quarter must be shifted. Check BEA release calendar.

---

### P3. Using clustering labels as supervised training targets without alignment

**Symptom:** `X` and `y` have different lengths; `.dropna()` removes extra rows silently.

**Cause:** clustering runs on `features.dropna()`, which may drop leading rows
(before FRED data starts, ~1947). Supervised training must align `X` and `y` on
the same index, not just the same length.

**Fix:** always use index intersection:
```python
common = features.index.intersection(labels.index)
X = features.loc[common].drop(columns=["market_code"], errors="ignore").dropna(axis=1, how="any")
y = labels.loc[common]
```
Never use `iloc[:len(labels)]` — this silently misaligns if any rows were dropped.

---

## Temporal Leakage in Cross-Validation

### P4. Using `train_test_split` (shuffled) for time-series data

**Symptom:** CV accuracy is 95%; production accuracy is 60%.

**Cause:** shuffling destroys temporal order. A training set of "2010, 2015, 2020"
tested on "2011, 2014" leaks information about the future through the training set.

**Fix:** always use `TimeSeriesSplit(n_splits=5)`. The final model is fitted on
all available data — this is correct because you want to use the most recent data
for scoring.

**Invariant:** `shuffle=False` is not enough. You need `TimeSeriesSplit` which
ensures all training data precedes all test data in each fold.

---

### P5. Forward-looking binary classifiers: label alignment

**Cause:** `y_future = y.shift(-h)` for horizon h introduces NaN at the end.
The `.dropna()` drops trailing rows, but `X_aligned = X.loc[y_future.index]`
must be re-aligned accordingly.

**Symptom:** `ValueError: X and y have inconsistent lengths`.

**Fix:** current code does `y_future = y.shift(-h).dropna()` and then
`X_aligned = X.loc[y_future.index]`. This is correct. Do not simplify to
`X.iloc[:len(y_future)]`.

---

## SSL and Network Issues

### P6. yfinance "self signed certificate in chain" error

**Symptom:** all tickers fail with SSL error; "possibly delisted" for every ticker.

**Cause:** yfinance uses `curl_cffi` as its HTTP backend on newer versions.
`curl_cffi` does not automatically use Python's `certifi` certificate bundle
on macOS and some Linux environments.

**Fix:** set `CURL_CA_BUNDLE` and `SSL_CERT_FILE` to `certifi.where()` before
importing yfinance. This is done at module load in `src/market_regime/ingestion/assets.py`.
Do not remove those lines.

**Note:** if the user has set these env vars already, `os.environ.setdefault()` is a
no-op — it never overrides user-set values.

---

### P7. multpl.com rate limiting

**Symptom:** some series return empty HTML or 429 Too Many Requests.

**Cause:** scraping 46 URLs too quickly. The `RATE_LIMIT_SECONDS = 2` in
`ingestion/multpl.py` is the minimum safe delay.

**Fix:** never reduce below 2 seconds. If you're seeing 429s, increase to 3-4s.
The `--refresh` flag should only be used when genuinely needed (full pipeline run,
not development iteration). Use checkpoints for development.

---

## Python Version and Dependency Issues

### P8. `X | Y` union type syntax on Python < 3.10

**Symptom:** `TypeError: unsupported operand type(s) for |: 'type' and 'type'`

**Cause:** `X | Y` for type unions was introduced in Python 3.10 (PEP 604).
When used in function annotations at runtime, it fails on 3.9.

**Fix:** add `from __future__ import annotations` at the top of every module
that uses `X | Y` syntax. This makes all annotations lazy strings, bypassing
the runtime evaluation. All `src/market_regime/` files should have this.

**Check:** `grep -r "X | Y\|: str | " src/ | grep -v "from __future__"` — if
any file matches, it needs the guard.

---

### P9. `contourpy` and other transitive deps failing on Python 3.10

**Symptom:** `pip install` fails with "Requires-Python >= 3.11".

**Cause:** `requirements.txt` was generated with `pip-compile` on Python 3.11.
Exact-pinned transitive deps may not have wheels for 3.10.

**Fix:** `requirements.txt` uses `>=` minimum bounds (not exact pins) for all
direct dependencies only. Never regenerate it with `pip-compile --generate-hashes`
— use manual `>=` bounds matching `pyproject.toml`.

---

### P10. `k-means-constrained` compilation on some platforms

**Symptom:** `pip install k-means-constrained` fails with Cython/C++ compile errors.

**Cause:** the package requires compilation and may not have pre-built wheels for
all Python/platform combinations.

**Fix:** the pipeline has a `--no-constrained` flag that falls back to standard
KMeans when this package is unavailable. The `setup.sh` script prompts the user
before attempting to install it.

---

## Data and Config Pitfalls

### P11. Changing `clustering_features` invalidates `regime_labels.yaml`

**Symptom:** after adding a new feature, the named regimes in `regime_labels.yaml`
no longer match the cluster assignments.

**Cause:** changing the feature set changes the PCA geometry, which changes the
cluster assignments. Cluster 0 may now be what was cluster 2.

**Fix:** after any change to `clustering_features` in `settings.yaml`:
1. Delete `data/checkpoints/cluster_labels*` and `data/regimes/cluster_labels.parquet`
2. Re-run steps 3-4
3. Inspect the new regime profiles and update `config/regime_labels.yaml` accordingly
4. Commit the new `regime_labels.yaml`

---

### P12. `end_date: "2025-09-30"` in settings.yaml is hardcoded

**Symptom:** pipeline silently ignores data after 2025-09-30.

**Cause:** the `end_date` in `config/settings.yaml` was set during initial development.

**Fix:** change to `null` (YAML null) and handle in `ingestion/fred.py` and
`ingestion/multpl.py` by using `datetime.today()` when the config value is null.
This is item #8 on the CLAUDE.md Planned backlog.

---

### P13. Checkpoint freshness check uses wall-clock time, not data time

**Symptom:** `cm.is_fresh("macro_raw", max_age_days=7)` returns True even though
FRED released new data yesterday.

**Cause:** freshness is based on when the checkpoint was created, not on whether
upstream data has changed.

**Fix:** for development iteration, this is acceptable. For production, always
run with `--refresh` on Fridays to ensure latest FRED data is pulled. The weekly
cron job (Tier 3 roadmap) should always pass `--refresh`.

---

## Clustering Pitfalls

### P14. Silhouette score selects k=2 when data is bimodal

**Symptom:** `best_k = 2` from the k-sweep even though you want 5 regimes.

**Cause:** silhouette score favors compact, well-separated clusters. Real macro
data often has two dominant modes (growth vs recession) that score highest at k=2.

**Fix:** `k_cap: 5` in `settings.yaml` caps the accepted k at 5. The `balanced_k: 5`
parameter forces 5 balanced clusters regardless of the silhouette result.
If you want to explore k=2, inspect the elbow curve plot manually.

---

### P15. PCA re-scaling before KMeans

**Symptom:** KMeans converges to wrong clusters after PCA.

**Cause:** PCA components are not unit-variance. `StandardScaler` must be applied
AFTER PCA and BEFORE feeding to KMeans. The pipeline does this correctly in
`clustering/kmeans.py` and again in `run_pipeline.py step3`.

**Invariant:** the transform chain is:
`features → StandardScaler → PCA(5) → StandardScaler → KMeans`.
The second StandardScaler ensures all PCA components have unit variance for the
distance calculations in KMeans.

---

## Plotting Pitfalls

### P16. `plt.show()` in headless environments

**Symptom:** pipeline hangs or crashes when run on a server without a display.

**Fix:** `run_cfg.show_plots = False` by default. Only set `True` via `--show-plots`
when running locally with a display. CI/CD pipelines should never pass `--show-plots`.

---

### P17. Seaborn pairplot is very slow on large feature sets

**Symptom:** step 2 `--plots` takes 10+ minutes.

**Cause:** pairplot with 69 features generates 69×69 = 4761 subplots.

**Fix:** pairplot is disabled by default (`generate_pairplot: False` in RunConfig).
Enable only when specifically investigating feature relationships.

---

## Portfolio Construction Pitfalls

### P18. `generate_recommendation()` parameter order differs from legacy

**Symptom:** BUY/SELL signals are reversed.

**Cause:** `legacy/portfolio.py` has `generate_recommendation(current_weights, target_weights)`
while `src/market_regime/reporting/portfolio.py` has `generate_recommendation(target_weights, current_weights=None)`.

**Fix:** always call with keyword arguments:
```python
generate_recommendation(target_weights=blended, current_weights=None)
```
Never rely on positional argument order for this function.

---

### P19. `blended_regime_portfolio()` probabilities must sum to ~1.0

**Symptom:** portfolio weights don't sum to 1.0 after blending.

**Cause:** `prediction["probabilities"]` from `predict_current()` returns RF class
probabilities from `predict_proba()` which sum to exactly 1.0. However, if you
manually construct a probability dict (e.g., from forward classifiers), the
individual binary classifier probabilities do NOT sum to 1.0 — they're independent.

**Fix:** only use `prediction["probabilities"]` (from the multi-class RF) as input
to `blended_regime_portfolio()`. Forward classifier probabilities (binary, one
per regime) are not valid inputs to the blending function.

---

## Test Suite Pitfalls

### P20. ~~Running `pytest` corrupts the `macro_raw` checkpoint~~ — FIXED

**Was:** `tests/test_pipelines_ingest_features.py` wrote synthetic 4-row data to
`data/checkpoints/macro_raw.parquet` (and `features_causal`, `features_noncausal`)
as a side effect of the pipeline smoke tests.

**Fix applied (March 2026):** both smoke tests now use `monkeypatch.setattr(step, "DATA_DIR", tmp_path)`
to redirect all file I/O to pytest's temporary directory. No production checkpoint files
are touched during `pytest`. The `--recompute` workaround is no longer needed after a test run.

---

## Behavior Model Pitfalls

### P21. `make_behavior_labels` uses strict inequalities — exactly-at-threshold is "flat"

**Symptom:** a return of exactly `0.0` is classified as `"flat"`, not `"up"`,
even when `up_threshold=0.0`.

**Cause:** the threshold checks use `r > up_threshold` and `r < down_threshold`
(strict), not `>=` / `<=`. With both thresholds at `0.0` this means:
- `r > 0`: "up"
- `r < 0`: "down"
- `r == 0`: "flat"

**This is intentional.** The thresholds define the exclusive boundaries of the
"flat" zone. A return right at the boundary is ambiguous and classified as flat.
Do not change to `>=` / `<=` — the test suite verifies the strict behavior.

---

## Tech Debt and Security Pitfalls

### P22. SSL verification is disabled in `ingestion/assets.py`

**Symptom:** no warning; yfinance fetches succeed silently even on untrusted networks.

**Cause:** `assets.py` uses `curl_cffi.requests.Session(verify=False, impersonate="chrome")`
unconditionally and suppresses `InsecureRequestWarning`. This was the simplest fix for
the curl_cffi CA-store problem on macOS/proxy environments.

**Risk:** susceptible to MITM attacks on price data. Applies even on "trusted" networks
(corporate proxies, captive portals).

**Planned fix (ROADMAP Tier 1):** add a `RunConfig` / settings flag to control SSL
verification, defaulting to secure. Make the insecure path opt-in with an explicit
warning log line.

---

### P23. Partial ingestion silently produces plausible-looking outputs

**Symptom:** pipeline runs successfully but regime analysis is based on fewer series
than expected (e.g., 30 of 46 multpl series, or 5 of 7 FRED series).

**Cause:** ingestion failures are caught and logged at WARNING level, but the pipeline
continues with whatever data was successfully fetched. Missing columns are silently
absent from `macro_raw.parquet`.

**Detection:** check `macro_raw.parquet` column count after ingestion; should be ~53
columns. Fewer columns means some series failed to fetch.

**Fix (ROADMAP Tier 1):** add an ingestion completeness report (expected vs. actual
series count) and optionally hard-fail below a minimum coverage threshold.

---

### P24. `CheckpointManager.list()` silently ignores corrupt metadata files

**Symptom:** `cm.list()` returns fewer checkpoints than expected; no error.

**Cause:** `CheckpointManager.list()` catches all JSON parse errors and continues
without logging which file failed.

**Detection:** manually inspect `data/checkpoints/*.meta.json` for malformed files.

**Fix:** the method should log at WARNING which file failed to parse before continuing.

---

### P25. Committed data artifacts in `data/` can create stale-data bugs

**Files present in repo:** `data/fred_api_datasets_snapshot_20260216.pickle`,
`data/multpl_datasets_snapshot_20260216.pickle`,
`data/grok_quarter_classifications_*.pickle`, `data/grok_quarter_classifications_*.xlsx`

**Risk:** if the pipeline accidentally loads these instead of freshly-fetched data,
results are silently based on Feb 2026 snapshots. Also increases repo size and carries
unclear licensing for scraped data.

**Planned fix:** move to `data/archives/` with explicit documentation, or move small
fixtures to `tests/fixtures/` and gitignore the rest.

---

### P26. FRED ingestion hard-fails when `FRED_API_KEY` is missing

**Symptom:** `pipelines/01_ingest.py` raises `EnvironmentError("FRED_API_KEY is not set")`
even though `config.py` only logs a WARNING about the missing key.

**Cause:** `fred.py`'s `fetch_all()` calls `fredapi.Fred(api_key=...)` which raises
if the key is None. The config loader deliberately defers validation.

**Fix:** copy `.env.example` to `.env` and add your free key from fred.stlouisfed.org.
Alternatively, run steps 3-7 only if you already have a `macro_raw.parquet` checkpoint.

---

### P27. Pickle files are an arbitrary-code-execution risk

**Affected files:** `outputs/models/current_regime.pkl`, `outputs/models/decision_tree.pkl`,
`outputs/models/forward_classifiers.pkl`, `data/grok_quarter_classifications_*.pickle`,
checkpoint model files in `data/checkpoints/`.

**Risk:** Python's `pickle.load()` executes arbitrary code embedded in the file. If any of
these files come from an untrusted source (shared drive, downloaded artifact), loading
them is equivalent to running unknown code.

**Current mitigation:** none — all pickles are assumed to be locally generated.
Never load a pickle file whose provenance you cannot verify.

**Planned fix:** migrate sklearn model serialization to `joblib.dump` / `joblib.load`
(marginally more stable across Python versions, same security profile). For data
exchange, prefer parquet or CSV. See ARCHITECTURE.md ADR #6.
