---
phase: 02-regime-clustering-interpretation
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - pipelines/03_cluster.py
  - src/market_regime/clustering.py
  - config/settings.yaml
  - tests/unit/test_clustering.py
autonomous: true
requirements:
  - REGIME-01
must_haves:
  truths:
    - Historical quarters are assigned to a manageable number of regimes (target ~4–7) using the Phase 1 feature set and PCA + clustering.
    - Re-running the clustering step with the same configuration and input features produces identical regime labels for all quarters.
    - The clustering step writes regime artifacts under data/regimes/ that downstream steps and notebooks can reload without ad-hoc logic.
  artifacts:
    - path: data/regimes/cluster_labels.parquet
      provides: "Quarter-level regime labels with both cluster and balanced_cluster and optional market_code."
    - path: data/regimes/pca_components.parquet
      provides: "Quarter-level PCA components (PC1…PCn) used for clustering and diagnostics."
    - path: data/regimes/kmeans_scores.parquet
      provides: "k-sweep evaluation table (k, inertia, silhouette, calinski, davies_bouldin) used to select best_k."
  key_links:
    - from: pipelines/03_cluster.py
      to: src/market_regime/clustering.py
      via: "reduce_pca, evaluate_kmeans, pick_best_k, fit_clusters"
      pattern: "from market_regime.clustering import"
    - from: pipelines/03_cluster.py
      to: config/settings.yaml
      via: "clustering.* configuration (n_pca_components, n_clusters_search, k_cap, balanced_k, random_state)"
      pattern: "clust_cfg = cfg[\"clustering\"]"
    - from: tests/unit/test_clustering.py
      to: src/market_regime/clustering.py
      via: "direct imports of reduce_pca/evaluate_kmeans/fit_clusters and synthetic data fixtures"
      pattern: "from market_regime.clustering import"
---

<objective>
Harden the unsupervised clustering step so that it produces deterministic, reproducible regime labels and core regime artifacts under data/regimes/, using the standardized PCA + KMeans pipeline and config-driven parameters.

Purpose: Satisfy REGIME-01 by ensuring that quarter-level regime assignments are stable, interpretable, and wired through the canonical clustering stack rather than ad-hoc scripts or notebooks.
Output: Updated clustering pipeline and tests that reliably write cluster_labels.parquet, pca_components.parquet, and kmeans_scores.parquet for downstream profiling, models, and notebooks.
</objective>

<execution_context>
@.planning/ROADMAP.md
@.planning/REQUIREMENTS.md
@.planning/phases/02-regime-clustering-interpretation/02-RESEARCH.md
@CLAUDE.md
</execution_context>

<context>
@pipelines/03_cluster.py
@src/market_regime/clustering.py
@config/settings.yaml
@tests/unit/test_clustering.py
</context>

<tasks>

<task type="auto">
  <name>Task 1: Ensure deterministic clustering artifacts for regimes</name>
  <files>pipelines/03_cluster.py, src/market_regime/clustering.py, config/settings.yaml, tests/unit/test_clustering.py</files>
  <action>
  Align the clustering step with the Phase 2 architecture so that PCA + KMeans and the k-sweep are always driven by config and produce the standard regime artifacts with deterministic label IDs.

  - In pipelines/03_cluster.py, verify that:
    - It reads the engineered quarterly features from data/processed/features.parquet and drops only market_code (if present) before clustering.
    - It calls reduce_pca with n_components, random_state taken from cfg["clustering"] (no hardcoded values).
    - It runs evaluate_kmeans on a StandardScaled version of the PCA components over k in range(2, n_clusters_search + 1) with random_state from config.
    - It selects best_k via pick_best_k(scores, k_cap=cfg["clustering"]["k_cap"]) and prints/logs the chosen k.
    - It calls fit_clusters(pca_df, best_k, balanced_k=cfg["clustering"]["balanced_k"], random_state=cfg["clustering"]["random_state"]) and relies on it for both cluster and balanced_cluster label logic, including canonicalization.
    - It restores market_code into the clustered DataFrame when present, and writes:
      - data/regimes/cluster_labels.parquet with columns ["cluster", "balanced_cluster"] and optional "market_code".
      - data/regimes/pca_components.parquet with only PC1…PCn columns (no label columns).
      - data/regimes/kmeans_scores.parquet with the full k-sweep table and k as a column (not index-only).

  - In src/market_regime/clustering.py, confirm that:
    - fit_clusters re-scales the PCA components prior to clustering and logs size summaries for both cluster and balanced_cluster.
    - _canonicalize_cluster_col is applied to both cluster and balanced_cluster so that label 0 always corresponds to the lowest mean PC1, ensuring deterministic IDs given the same PCA projection.
    - Any logging messages are clear enough to diagnose cluster size distributions and chosen k, but do not hardcode project-specific assumptions beyond what REGIME-01 requires.

  - In config/settings.yaml, review the clustering.* section and, if needed, adjust or document:
    - n_pca_components (kept at 5 per CLAUDE.md).
    - n_clusters_search (upper bound for k sweep, typically 12).
    - k_cap (maximum accepted k for silhouette-based selection, typically 5).
    - balanced_k (target number of balanced regimes, typically 5).
    - random_state (for reproducibility across runs).
    Ensure no clustering parameters are duplicated elsewhere in code.

  - In tests/unit/test_clustering.py, strengthen or extend tests (without overfitting to implementation details) so that:
    - reduce_pca on synthetic numeric data produces the requested number of components with finite values and logs variance coverage.
    - evaluate_kmeans returns a DataFrame that always includes k, inertia, silhouette, calinski, and davies_bouldin columns and is well-formed for pick_best_k.
    - fit_clusters on a small synthetic PCA DataFrame:
      - Produces both cluster and balanced_cluster columns.
      - Uses _canonicalize_cluster_col so that relabeling is deterministic (e.g. cluster IDs are in contiguous [0, k-1] order and change when PC1 order is reversed in a controlled fixture).
    - Any optional dependency fallbacks (e.g. missing k-means-constrained) are covered so balanced_cluster still exists and logs a clear warning.

  Keep changes focused on wiring, determinism, and artifact shapes rather than altering the underlying clustering algorithms or PCA dimension choice (those are fixed by earlier design decisions).
  </action>
  <verify>
  - Automated: pytest tests/unit/test_clustering.py -q
  - Automated: python pipelines/03_cluster.py  # should complete without error and (re)write regime artifacts under data/regimes/
  </verify>
  <done>
  - Running python pipelines/03_cluster.py on a valid features.parquet writes cluster_labels.parquet, pca_components.parquet, and kmeans_scores.parquet under data/regimes/.
  - Re-running the step with identical inputs and config yields the same cluster and balanced_cluster label assignments (up to file ordering), thanks to canonicalization and fixed random_state.
  - pytest tests/unit/test_clustering.py -q passes and explicitly covers reduce_pca, evaluate_kmeans, pick_best_k, and fit_clusters behavior needed for REGIME-01.
  </done>
</task>

</tasks>

<verification>
- Running pytest tests/unit/test_clustering.py -q passes, confirming that PCA reduction, k-sweep scoring, and clustering functions behave as expected for synthetic data and edge cases.
- Running python pipelines/03_cluster.py on the Phase 1 feature checkpoints completes without error and produces cluster_labels.parquet, pca_components.parquet, and kmeans_scores.parquet in data/regimes/.
- Inspecting cluster_labels.parquet shows contiguous integer IDs for cluster and balanced_cluster, with reasonable regime counts (~4–7) and no missing labels.
</verification>

<success_criteria>
- REGIME-01 is supported by a deterministic, config-driven clustering pipeline that can be re-run safely as data or configuration evolves.
- All downstream steps that rely on data/regimes/cluster_labels.parquet, data/regimes/pca_components.parquet, and data/regimes/kmeans_scores.parquet can treat these files as stable interfaces.
- Clustering behavior (number of regimes, label stability, artifact shapes) is guarded by unit tests and a simple pipeline run rather than implicit assumptions in notebooks.
</success_criteria>

<output>
After completion, leave this plan file in place and rely on /gsd:execute-phase 2 to run it as part of the Phase 2 execution wave. No additional SUMMARY file is required beyond the standard execute-phase summaries.
</output>

