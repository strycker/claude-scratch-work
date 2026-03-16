# Phase 2: Regime Clustering & Interpretation - Research

**Researched:** 2026-03-16  
**Domain:** Unsupervised market regime clustering & profiling  
**Confidence:** HIGH

## Summary

Phase 2 should harden and productize the existing PCA + clustering + profiling pipeline so that it produces a small, stable, and interpretable set of market regimes that downstream models and humans can trust. The reference logic in `legacy/unified_script.py` and the modular `src/market_regime/clustering.py`, `pipelines/03_cluster.py`, and `market_regime.regime` + `pipelines/04_regime_label.py` already implement the core algorithms; this phase is mainly about wiring, configuration discipline, naming conventions, and validation around those pieces. All work must sit on top of the Phase 1 feature pipeline, treat regimes as quarter-level labels, and avoid any changes that would silently invalidate prior analyses without explicit versioning.

**Primary recommendation:** Use the existing PCA(5) → KMeans(k-sweep) → balanced KMeans pipeline and `regime` profiling module as the authoritative stack, focusing Phase 2 work on configuration, naming stability (`regime_labels.yaml`), artifacts under `data/regimes/`, and tests/plots that make regime behavior and stability inspectable.

<phase_requirements>
## Phase Requirements

| ID        | Description                                                  | Research Support |
|-----------|--------------------------------------------------------------|------------------|
| REGIME-01 | Cluster quarters into a small, interpretable set of regimes. | Use `pipelines/03_cluster.py` + `market_regime.clustering.reduce_pca/evaluate_kmeans/pick_best_k/fit_clusters` with config-driven `n_pca_components`, `n_clusters_search`, `k_cap`, and `balanced_k` to produce 4–7 balanced regimes stored in `data/regimes/cluster_labels.parquet`. |
| REGIME-02 | Profile each regime with reproducible descriptive statistics. | Use `pipelines/04_regime_label.py` + `market_regime.regime.build_profiles` to compute per-cluster statistics over the Phase 1 feature set (and later ETF returns), writing `data/regimes/profiles.parquet` and ensuring the features align with `config/settings.yaml`’s `clustering_features`. |
| REGIME-03 | Maintain a stable mapping from cluster IDs to regime names.   | Use `market_regime.regime.suggest_names` + `load_name_overrides(CONFIG_DIR)` together with `config/regime_labels.yaml` and `data/regimes/regime_names_suggested.yaml` to pin human-readable names, ensure deterministic ID ordering via `_canonicalize_cluster_col`, and document any renames via config/git history. |
</phase_requirements>

## Standard Stack

### Core

| Library / Module                 | Version / Source          | Purpose                                                 | Why Standard |
|----------------------------------|---------------------------|---------------------------------------------------------|-------------|
| `market_regime.clustering`      | Local package (`src/`)    | PCA reduction, k-sweep scoring, clustering, k selection | Mirrors legacy script with cleaner API; tested in `tests/unit/test_clustering.py`. |
| `pipelines/03_cluster.py`       | Local pipeline            | Orchestrates PCA + clustering and writes regime artifacts | Provides CLI-friendly, checkpoint-aware clustering step wired to `config/settings.yaml`. |
| `market_regime.regime`          | Local package (`src/`)    | Regime profiling, heuristic naming, transition matrix   | Encapsulates interpretation logic and naming heuristics in one place. |
| `pipelines/04_regime_label.py`  | Local pipeline            | Runs profiling + naming and writes profiles and transitions | Produces reproducible `profiles.parquet`, `transition_matrix.parquet`, and suggested name YAMLs. |
| `config/settings.yaml`          | YAML config               | Controls PCA dimension, k-search bounds, balanced k, feature lists | Single source of truth for clustering parameters and feature schema. |
| `config/regime_labels.yaml`     | YAML config (to use)      | Manually pinned mapping from cluster IDs to human names | Satisfies REGIME-03 via version-controlled label mapping. |
| `pandas`, `numpy`, `sklearn`    | Python deps (per project) | DataFrames, PCA, KMeans, metrics                       | Already used across `legacy/` and `src/`; well-tested. |
| `k_means_constrained`           | Optional Python dep       | Balanced KMeans clustering with size constraints       | Provides equal-ish cluster sizes needed for robust per-regime statistics. |

### Supporting

| Library / Module                 | Version / Source          | Purpose                              | When to Use |
|----------------------------------|---------------------------|--------------------------------------|------------|
| `market_regime.features.transforms` | Local package           | Provides engineered quarterly features | Always use as upstream source; do not recompute ad hoc in Phase 2. |
| `market_regime.io.checkpoints`  | Local package             | Checkpoint manager (if used by steps) | When reading/writing pre-clustered features or regimes via checkpoints. |
| `notebooks/03_clustering.ipynb` | Notebook                  | Exploratory clustering visualization and investigation | For manual sanity checks / tuning, not for production logic. |
| `tests/unit/test_clustering.py` | Test module               | Validates PCA + KMeans helpers       | Use as reference when extending tests for new Phase 2 behavior. |

### Alternatives Considered

| Instead of                        | Could Use                     | Tradeoff |
|-----------------------------------|-------------------------------|---------|
| `market_regime.clustering` PCA(5) | Adaptive n-components search via `optimize_n_components` | Good for investigation, but v1 should stick to 5 PCs (per `CLAUDE.md`) for stability and comparability with legacy. |
| KMeans (+ constrained KMeans)     | GMM / DBSCAN / Spectral (see `gmm.py`, `density.py`, `spectral.py`) | Useful for research notebooks and sensitivity checks; not standard for v1 regimes to avoid confusion and label instability. |
| Manual ad-hoc naming in notebooks | Central `regime_labels.yaml` | Notebooks are too easy to drift; config + code-based suggestions give traceable, reproducible naming. |

**Installation (for constrained clustering, if not already present):**

```bash
pip install k-means-constrained
```

## Architecture Patterns

### Recommended Project Structure (Regime Layer)

```text
data/
  processed/
    features.parquet              # Phase 1 output: engineered quarterly features
  regimes/
    cluster_labels.parquet        # quarter → cluster, balanced_cluster, market_code
    pca_components.parquet        # quarter → PC1…PCn
    kmeans_scores.parquet         # k, inertia, silhouette, calinski, davies_bouldin
    profiles.parquet              # cluster × (stat, feature)
    transition_matrix.parquet     # empirical P(next=j | current=i)
    regime_names_suggested.yaml   # auto-suggested names (incl. overrides)

config/
  settings.yaml                   # clustering.n_pca_components, n_clusters_search, k_cap, balanced_k
  regime_labels.yaml              # manually pinned regime names (for REGIME-03)

pipelines/
  03_cluster.py                   # features → PCA + clustering → regimes artifacts
  04_regime_label.py              # regimes + features → profiles + names + transitions
```

### Pattern 1: PCA → StandardScale → KMeans Sweep → Best-k Selection

**What:** Use `reduce_pca()` to project the Phase 1 feature matrix into a fixed 5-D PC space, then run `evaluate_kmeans()` over k=2..`n_clusters_search`, and select `best_k` via `pick_best_k(scores, k_cap)`.  
**When to use:** Every time we recompute regimes from features; never hand-code your own PCA or k-sweep in pipeline steps.

Example (as in `pipelines/03_cluster.py`):

```python
from market_regime.clustering import reduce_pca, evaluate_kmeans, pick_best_k
from sklearn.preprocessing import StandardScaler

features = pd.read_parquet(DATA_DIR / "processed" / "features.parquet")
X = features.drop(columns=["market_code"], errors="ignore")

pca_df, pca_model, scaler = reduce_pca(
    X,
    n_components=clust_cfg["n_pca_components"],
    random_state=clust_cfg["random_state"],
)

X_scaled = StandardScaler().fit_transform(pca_df.values)
scores = evaluate_kmeans(
    X_scaled,
    k_range=range(2, clust_cfg["n_clusters_search"] + 1),
    random_state=clust_cfg["random_state"],
)
best_k = pick_best_k(scores, k_cap=clust_cfg["k_cap"])
```

### Pattern 2: Dual Clustering Outputs with Deterministic IDs

**What:** Fit both a “natural” KMeans (`cluster`) and a size-constrained/balanced clustering (`balanced_cluster`), then canonicalize labels so cluster 0 always has the lowest mean PC1, etc.  
**When to use:** Whenever exposing regime IDs to downstream code or users; never rely on raw KMeans label permutations.

Key pattern (`market_regime.clustering.fit_clusters`):

```python
from market_regime.clustering import fit_clusters

clustered = fit_clusters(
    pca_df,
    best_k=best_k,
    balanced_k=clust_cfg["balanced_k"],
    random_state=clust_cfg["random_state"],
)
```

Internally, `_canonicalize_cluster_col` ensures stable label ordering based on mean PC1, which is essential for REGIME-03 (names tied to IDs).

### Pattern 3: Regime Profiling & Naming

**What:** Use `build_profiles()` to compute per-cluster statistics on the original feature space and `suggest_names()` + `load_name_overrides()` to derive human-readable regime names, persisted in YAML.  
**When to use:** Every time we (re)compute regimes or change clustering config; use this to regenerate interpretable, reproducible profiles.

Pattern from `pipelines/04_regime_label.py`:

```python
from market_regime.regime import (
    build_profiles,
    suggest_names,
    build_transition_matrix,
    load_name_overrides,
)

features = pd.read_parquet(DATA_DIR / "processed" / "features.parquet")
labels = pd.read_parquet(DATA_DIR / "regimes" / "cluster_labels.parquet")["balanced_cluster"]

common = features.index.intersection(labels.index)
features = features.loc[common]
labels = labels.loc[common]

profile = build_profiles(features, labels)
profile.to_parquet(DATA_DIR / "regimes" / "profiles.parquet")

auto_names = suggest_names(features, labels)
overrides = load_name_overrides(CONFIG_DIR)
regime_names = {**auto_names, **overrides}
```

### Anti-Patterns to Avoid

- **Ad-hoc PCA/clustering code in notebooks or new scripts:** Always route through `market_regime.clustering` to maintain consistency and test coverage.
- **Hardcoding k or PCA dimension in code:** Use `config/settings.yaml` (`clustering.n_pca_components`, `n_clusters_search`, `k_cap`, `balanced_k`) instead of literals.
- **Manual, non-config-based naming:** Do not embed name mappings in code or notebooks; use `regime_labels.yaml` + YAML suggestions so names are version-controlled.
- **Using different feature sets for profiling vs clustering without tracking:** Profiles should be based on the same or a superset of features used for clustering, and any change belongs in `settings.yaml`.

## Don't Hand-Roll

| Problem                              | Don't Build                         | Use Instead                           | Why |
|--------------------------------------|-------------------------------------|----------------------------------------|-----|
| PCA + clustering from raw features   | Custom StandardScaler/PCA/KMeans loop | `market_regime.clustering.reduce_pca/evaluate_kmeans/pick_best_k/fit_clusters` | Already match legacy behavior, are unit-tested, and centralize logging and config usage. |
| Balanced regime sizes                | Home-grown cluster-size constraints | `k_means_constrained.KMeansConstrained` via `fit_clusters` | Edge cases around empty clusters and size bounds are non-trivial; library handles them. |
| Regime naming heuristics             | Free-form string building per regime | `market_regime.regime.suggest_names` + `NAMING_HEURISTICS` and `regime_labels.yaml` | Centralizes domain knowledge about inflation/growth/rates, avoids silent drift across scripts. |
| Regime transition probabilities      | Custom next-step counting logic     | `market_regime.regime.build_transition_matrix` | Correctly normalizes row probabilities and handles sparse transitions. |

**Key insight:** The clustering and profiling logic is already factored into well-tested, reusable modules; Phase 2 should orchestrate and configure these, not re-implement algorithms.

## Common Pitfalls

### Pitfall 1: Unstable Cluster IDs Across Runs

**What goes wrong:** KMeans label assignments are arbitrary; without canonicalization and a stable mapping, “Regime 0” today might be “Regime 3” tomorrow.  
**Why it happens:** Raw KMeans labels depend on initialization and library version; if you treat IDs as meaningful without post-processing, names and downstream models drift.  
**How to avoid:** Always call `fit_clusters()` (which uses `_canonicalize_cluster_col`) and then pin names via `regime_labels.yaml`. Never expose raw KMeans labels to downstream code.  
**Warning signs:** Regime name suggestions or profiles appear to “jump” between runs even when config and data haven’t changed.

### Pitfall 2: Feature Schema Drift

**What goes wrong:** Changing the `clustering_features` or features used for profiling without updating config and documentation makes historical regimes incomparable and can invalidate `regime_labels.yaml`.  
**Why it happens:** Ad-hoc tweaks in notebooks or code to “try a new feature” that never make it into `settings.yaml` or commit messages.  
**How to avoid:** Treat `config/settings.yaml` as the only source of truth for feature lists; any schema change should be deliberate, documented, and accompanied by a note about its impact on existing labels.  
**Warning signs:** Profiles show unexpected variables; tests referencing `clustering_features` fail; saved profiles no longer match expectations from `legacy/unified_script.py`.

### Pitfall 3: Overfitting k to Historical Data

**What goes wrong:** Picking a high k because silhouette/CH favors it slightly, leading to too many regimes that are hard to interpret and sparsely populated.  
**Why it happens:** Blindly optimizing cluster score metrics without honoring the design target of ~4–7 regimes.  
**How to avoid:** Use `pick_best_k(scores, k_cap=cfg["clustering"]["k_cap"])` with `k_cap` set around 5 (per `CLAUDE.md`), and inspect cluster profiles qualitatively to ensure interpretability.  
**Warning signs:** Many regimes have very few quarters; regime names become highly granular or redundant.

### Pitfall 4: Misaligned Index Between Features and Labels

**What goes wrong:** Profiles or transition matrices are computed on misaligned DataFrames (e.g., features and labels with different date indexes), causing silent dropping of rows or incorrect statistics.  
**Why it happens:** Features may extend beyond the range where balanced clusters are defined; `build_profiles` expects aligned indices.  
**How to avoid:** Always intersect indices before profiling (`common = features.index.intersection(labels.index)` as in `pipelines/04_regime_label.py`).  
**Warning signs:** Fewer rows than expected in `profiles.parquet`; transition matrix missing expected regimes.

## Code Examples

### Computing and Saving Regime Profiles

```python
from market_regime import DATA_DIR, CONFIG_DIR
from market_regime.regime import (
    build_profiles,
    suggest_names,
    build_transition_matrix,
    load_name_overrides,
)
import pandas as pd
import yaml

features = pd.read_parquet(DATA_DIR / "processed" / "features.parquet")
labels = pd.read_parquet(DATA_DIR / "regimes" / "cluster_labels.parquet")["balanced_cluster"]

common = features.index.intersection(labels.index)
features = features.loc[common]
labels = labels.loc[common]

profile = build_profiles(features, labels)
profile.to_parquet(DATA_DIR / "regimes" / "profiles.parquet")

auto_names = suggest_names(features, labels)
overrides = load_name_overrides(CONFIG_DIR)
regime_names = {**auto_names, **overrides}

with open(DATA_DIR / "regimes" / "regime_names_suggested.yaml", "w") as f:
    yaml.dump(regime_names, f, default_flow_style=False)

tm = build_transition_matrix(labels)
tm.to_parquet(DATA_DIR / "regimes" / "transition_matrix.parquet")
```

### Running the Clustering Pipeline Step

```bash
python pipelines/03_cluster.py
python pipelines/04_regime_label.py
```

These steps should be treated as the canonical way to recompute regimes from updated features.

## State of the Art

| Old Approach                            | Current Approach                                             | When Changed         | Impact |
|-----------------------------------------|--------------------------------------------------------------|----------------------|--------|
| Monolithic script `legacy/unified_script.py` with inline PCA + KMeans + plotting | Modular `src/market_regime/clustering.py` + `pipelines/03_cluster.py` + `regime.py` + `pipelines/04_regime_label.py` | Refactor completed by March 2026 | Easier to test, configure, and reuse; aligns with pipeline steps in `run_pipeline.py`. |
| Arbitrary KMeans label IDs             | Canonicalized IDs via `_canonicalize_cluster_col` and overrides via `regime_labels.yaml` | Introduced in `clustering.py`/`regime.py` | Stabilizes regime identity across runs and refits. |
| Manual exploratory profiling in notebooks only | Code-driven `build_profiles` + saved `profiles.parquet`      | Added in `regime.py` and pipeline 4 | Allows reproducible profiles and downstream automated use. |

**Deprecated/outdated:**

- Directly running `legacy/unified_script.py` for new work; it remains a reference but not the primary API for v1.
- Hand-tuned k per run; `evaluate_kmeans` + `pick_best_k` with `k_cap` is the standard selection mechanism.

## Open Questions

1. **How often should clustering be recomputed as new data arrives?**
   - What we know: The pipeline can recompute regimes from fresh features quickly, and labels are deterministic given config and data.  
   - What's unclear: Whether v1 should treat regimes as fixed over the historical sample (re-cluster only rarely) or allow them to shift more frequently as data grows.  
   - Recommendation: For v1, treat regimes as fixed for a chosen historical window and only re-run clustering when intentionally changing the feature schema or date range; codify this as a config/versioning decision rather than an automated weekly step.

2. **How aggressively should naming heuristics influence final regime names vs manual overrides?**
   - What we know: `suggest_names` uses heuristic tags based on medians of key macro variables relative to global medians, and overrides in `regime_labels.yaml` can replace them.  
   - What's unclear: The exact balance between automated tags vs curated, narrative-friendly names.  
   - Recommendation: Use auto-suggestions as a starting point, then manually curate `regime_labels.yaml` once regimes are inspected; avoid frequent renames so supervised models and users can rely on stable semantics.

3. **Should Phase 2 also incorporate empirical forward probabilities as in legacy `compute_forward_probabilities`?**
   - What we know: Legacy analysis computes empirical probabilities of reaching regimes within N quarters; `regime.py` already has `build_transition_matrix`, but not the full forward-window probabilities.  
   - What's unclear: Whether this belongs in Phase 2 (unsupervised regime analysis) or Phase 3 (supervised transitions).  
   - Recommendation: Leave full forward-window probability computation to Phase 3, but ensure Phase 2 exposes clean one-step transition matrices suitable for comparison.

## Validation Architecture

### Test Framework

| Property         | Value                    |
|------------------|--------------------------|
| Framework        | pytest                   |
| Config file      | `pyproject.toml` (`[tool.pytest.ini_options]`) |
| Quick run command | `pytest tests/unit/test_clustering.py -q` |
| Full suite command | `pytest tests/ -v` or `make test` |

### Phase Requirements → Test Map

| Req ID    | Behavior                                                                                          | Test Type  | Automated Command                                             | File Exists? |
|-----------|---------------------------------------------------------------------------------------------------|------------|----------------------------------------------------------------|-------------|
| REGIME-01 | PCA + clustering with k-sweep and deterministic best-k selection produce stable regime labels.    | unit       | `pytest tests/unit/test_clustering.py -q`                     | ✅ |
| REGIME-02 | Regime profiles compute per-cluster statistics over features without index misalignment or NaNs.  | unit       | `pytest tests/unit/test_regime.py::test_build_profiles` (to add) | ❌ Wave 0 |
| REGIME-03 | Regime naming + overrides yield stable, deterministic name mappings for canonicalized cluster IDs. | unit       | `pytest tests/unit/test_regime.py::test_suggest_names_overrides` (to add) | ❌ Wave 0 |

### Sampling Rate

- **Per Phase 2 commit touching clustering code or config:** Run `pytest tests/unit/test_clustering.py -q`.  
- **Per change to regime profiling/naming logic or `regime_labels.yaml`:** Run `pytest tests/unit/test_clustering.py -q tests/unit/test_regime.py -q` (once `test_regime.py` exists).  
- **Phase gate:** Before declaring Phase 2 complete, run `pytest tests/ -v` and ensure all clustering- and regime-related tests are green.

### Wave 0 Gaps

- [ ] Create `tests/unit/test_regime.py` covering `build_profiles`, `suggest_names`, `build_transition_matrix`, and `load_name_overrides`.  
- [ ] Optional: Add an integration-style test that runs `pipelines/03_cluster.py` and `pipelines/04_regime_label.py` on a small synthetic dataset (or using checkpoints) to assert expected shapes and presence of output files under `data/regimes/`.  
- [ ] Document how changes to `config/settings.yaml` and `config/regime_labels.yaml` are validated (e.g., a short checklist in `ROADMAP.md` or Phase 2 PLAN).

## Sources

### Primary (HIGH confidence)

- `CLAUDE.md` — project-wide design for clustering (fixed 5 PCA components, two clusterings, checkpoints, etc.).  
- `legacy/unified_script.py` — reference implementation for PCA, clustering, and Bernstein gap fill; mirrored by `src/market_regime` modules.  
- `src/market_regime/clustering.py` — authoritative implementation for PCA/k-sweep/cluster fitting and canonicalization.  
- `src/market_regime/regime.py` — authoritative implementation for profiles, naming heuristics, and transition matrix.  
- `pipelines/03_cluster.py`, `pipelines/04_regime_label.py` — current pipeline wiring for Steps 3–4.  
- `.planning/ROADMAP.md`, `.planning/REQUIREMENTS.md`, `.planning/PROJECT.md`, `.planning/STATE.md` — define Phase 2 goals and REGIME-* requirements.  
- `.planning/codebase/TESTING.md` — description of pytest-based testing patterns and existing clustering tests.

### Secondary (MEDIUM confidence)

- None needed; all relevant logic is in-repo and aligned between `legacy/` and `src/`.

### Tertiary (LOW confidence)

- None relied on.

## Metadata

**Confidence breakdown:**

| Area           | Level | Reason |
|----------------|-------|--------|
| Standard Stack | HIGH  | All clustering/profiling code is present in `src/` and verified against `legacy/unified_script.py`, with unit tests in place for clustering. |
| Architecture   | HIGH  | Pipelines 3–4 already encode the desired architecture; Phase 2 mainly needs planning around configuration, naming, and artifacts. |
| Pitfalls       | MEDIUM | Pitfalls are inferred from legacy comments, current code structure, and typical clustering issues; they should be validated during implementation and notebook exploration. |

**Research date:** 2026-03-16  
**Valid until:** 2026-04-15 (low-churn domain; revise sooner if clustering config or feature schema changes materially).

