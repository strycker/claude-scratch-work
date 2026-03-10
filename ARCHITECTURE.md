# Trading-Crab — Architecture Decision Record

Documents the "why" behind key design decisions so future contributors
don't accidentally break invariants that look arbitrary.

---

## 1. Two Feature Files: `features.parquet` and `features_supervised.parquet`

**Decision:** Step 2 produces two separate parquet files from the same raw data.

**Why:**
- **Centered smoothing** (`causal=False`) uses both past and future neighbors in
  each rolling window. This is mathematically superior for interpolating genuinely
  missing historical data and for characterizing what a regime "looks like" across
  its full span. Used for: clustering (step 3), regime profiling (step 4).
- **Causal smoothing** (`causal=False`) uses only past data in every rolling
  window. This exactly replicates what you could compute at the end of a quarter
  with only information available at that moment. Used for: supervised learning
  (step 5), live scoring (steps 5-7).
- **The critical invariant:** if a feature is computed with a centered window,
  it contains information from the future. Training a supervised model on
  centered features and then trying to score "today's" data is look-ahead bias —
  the model learned patterns that cannot be reproduced in real-time.

**Alternative considered:** single file with a flag column — rejected because it
leads to accidental mixing of centered and causal features when steps share files.

**Column names are identical** in both files. This is intentional: steps 5-7
import the causal file as a drop-in replacement. The checkpoint manager uses
`"features"` vs `"features_supervised"` keys to distinguish them.

---

## 2. Five PCA Components (Fixed, Not Variance-Threshold)

**Decision:** `n_pca_components: 5` in `settings.yaml`. Not "keep 90% variance".

**Why:**
- The legacy script established 5 components as the working baseline after
  experimenting with scree plots on the actual 69-column feature matrix.
- More importantly: changing the number of PCA components changes the clustering
  geometry, which changes the cluster assignments, which invalidates any manually
  pinned regime names in `config/regime_labels.yaml`.
- Variance-threshold PCA is non-deterministic across data updates (as more data
  arrives the cumulative variance curve shifts). Fixed components are reproducible.

**When to revisit:** if the feature set changes substantially (e.g., adding 20 new
FRED series), re-run the scree plot and benchmark silhouette scores for 3, 5, 7, 10
components before changing. Document the new choice here.

---

## 3. Balanced KMeans as the Primary Regime Assignment

**Decision:** We use `balanced_cluster` (from `KMeansConstrained`) for all downstream
steps, not `cluster` (from standard KMeans with best-k from silhouette).

**Why:**
- Per-regime statistics require sufficient samples to be meaningful. Standard KMeans
  often produces clusters of wildly different sizes (e.g., 70% in one cluster).
- With only ~300 quarters, a cluster of 10 quarters has unreliable mean/std estimates.
- `KMeansConstrained(size_min=bucket-2, size_max=bucket+2)` ensures each regime
  has ~60 quarters at k=5, giving reliable statistics for all downstream computations.

**Tradeoff:** balanced clustering slightly distorts cluster geometry — some quarters
near a boundary get assigned to a less-natural regime to meet the size constraint.
This is acceptable: the goal is interpretable regimes with robust statistics, not
geometrically pure clusters.

**Alternative considered:** hierarchical clustering — rejected because it doesn't
produce equal-size clusters and has no clear stopping rule for k.

---

## 4. Bernstein Polynomial Gap Fill in Log Space

**Decision:** Gap fill happens AFTER log transform.

**Why:**
- The raw series (e.g., S&P 500, GDP) are exponential-looking. Interpolating
  between two values of 1000 and 2000 in linear space overshoots the midpoint.
  In log space, the midpoint of [log(1000), log(2000)] = log(1414) is correct.
- Bernstein polynomials require 4 boundary conditions per side (value, d1, d2, d3).
  All three derivatives must also be computed in log space for consistency.
- **Invariant:** the order is always: cross-ratios → log → select → gap-fill → derivatives → select.
  Do not move gap fill before log transform.

**Why Bernstein polynomials (not linear/cubic spline)?**
- `BPoly.from_derivatives` produces a polynomial that exactly matches value +
  first 3 derivatives at both endpoints. This makes the interpolation smooth and
  compatible with the derivative features we compute afterward.
- Cubic splines minimize curvature globally; Bernstein interpolates boundary
  conditions locally. For gap filling (usually 1-4 quarters), local is better.

---

## 5. Taylor Extrapolation for Edge Gaps

**Decision:** Use Taylor expansion (not Bernstein) for leading and trailing edge gaps.

**Why:**
- Bernstein requires boundary conditions on both sides. For edge gaps (missing
  data at the start or end of the time series), one side has no neighbors.
- Taylor extrapolation uses value + d1 + d2 + d3 at the known edge to project
  outward: `f(x+h) ≈ f(x) + h·f'(x) + (h²/2)·f''(x) + ...`
- This is mathematically consistent with the interior Bernstein approach.

---

## 6. CheckpointManager: Parquet for DataFrames, Pickle for Models

**Decision:** DataFrames always use parquet; sklearn models use pickle.

**Why (parquet):**
- Smaller files (columnar compression)
- Typed (dtypes are preserved)
- Human-inspectable (duckdb, pandas, parquet-viewer all read it)
- No Python version lock-in (unlike pickle for DataFrames)

**Why (pickle for models):**
- sklearn's model serialization format is pickle. There is no parquet-serializable
  alternative for a fitted `RandomForestClassifier`.
- Risk: pickle files are Python-version-sensitive. If you retrain on Python 3.11
  and score on Python 3.10, the pickle may fail. Mitigation: use `joblib.dump`
  which is slightly more stable than raw pickle for sklearn objects. (TODO: migrate
  from `pickle.dump` to `joblib.dump` in `pipelines/05_predict.py`)

---

## 7. Publication-Lag Shifts for GDP and GNP

**Decision:** `fred_gdp` and `fred_gnp` are shifted +1 quarter.

**Why:**
- The BEA releases the "advance estimate" of GDP approximately 30 days after
  quarter end. The "third estimate" (most revised) comes ~90 days later.
- At the end of Q1 you cannot know Q1 GDP. The first GDP reading you have at
  the end of Q1 is the Q4 (previous year) third estimate.
- Not shifting these series introduces look-ahead bias: the model learns to use
  "future" GDP data as if it were available today.
- This is set in `config/settings.yaml` as `shift: true` per series.
- **Invariant:** all FRED series that are subject to significant revision and
  have a publication lag longer than one quarter should be shifted.

---

## 8. Runtime Flags via `RunConfig` Dataclass

**Decision:** All runtime behavior is controlled by a single `RunConfig` object
passed through the pipeline, not by global variables or config file values.

**Why:**
- Avoids action-at-a-distance bugs where a deeply nested module checks a global
  flag that was set somewhere else
- Makes the pipeline deterministic and testable: pass `RunConfig(generate_plots=False)`
  in tests to skip all matplotlib code without monkeypatching globals
- The dataclass `from_args()` factory method converts argparse `Namespace` to
  `RunConfig` in `run_pipeline.py` — the only place argparse is used

---

## 9. Two-Clustering Architecture (Standard + Constrained)

**Decision:** Always produce both `cluster` and `balanced_cluster`, even though
only `balanced_cluster` is used downstream.

**Why:**
- `cluster` (unconstrained, best-k from silhouette) serves as a geometric
  reference: if `balanced_cluster` looks very different from `cluster`, it means
  the size constraint is distorting the natural clusters significantly.
- Having both lets you visually compare in notebooks without re-running clustering.
- The k-sweep silhouette scores that determine `best_k` for `cluster` are also
  saved (`data/regimes/kmeans_scores.parquet`) for elbow-curve visualization.

---

## 10. Config-Driven Feature Lists

**Decision:** `initial_features` and `clustering_features` lists live in
`config/settings.yaml`, not hardcoded in Python.

**Why:**
- These lists were analytically determined by examining which series have coverage
  back to ~1950 and which derivatives are informative for clustering.
- Putting them in YAML lets you experiment without touching Python source code.
- **Invariant:** changing `clustering_features` changes the clustering geometry and
  invalidates any manually pinned `regime_labels.yaml`. Update the YAML file, delete
  the old `cluster_labels.parquet` checkpoint, and re-run steps 3-7 before committing.

---

## 11. All Visualization in `plotting.py` — Never Inline in Notebooks

**Decision:** Notebooks call functions from `src/market_regime/plotting.py`; they
do not define plotting logic inline.

**Why:**
- Reusability: the same plot is needed in the notebook AND in the CLI `--plots` mode.
- Testability: plotting functions can be tested by mocking `matplotlib.pyplot`.
- Consistency: all plots use the same `CUSTOM_COLORS` palette and output naming scheme.
- DRY: prevents three slightly-different versions of the same chart drifting apart.

**Consequence:** if you need a new plot, add it to `plotting.py` first, then call
it from the notebook. Never define a `fig, ax = plt.subplots()` block directly in
a notebook cell that isn't backed by a function.
