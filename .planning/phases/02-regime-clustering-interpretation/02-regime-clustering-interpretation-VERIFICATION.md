---
phase: 02-regime-clustering-interpretation
verified: 2026-03-16T00:00:00Z
status: gaps_found
score: 5/6 must-haves verified
gaps:
  - truth: "Each regime has a reproducible profile over key macro features and ETF returns that supports a human-readable description."
    status: partial
    reason: "Profiles are fully implemented and tested for macro features, but ETF-return profiling is deferred to later phases and is not yet wired into profiles.parquet."
    artifacts:
      - path: data/regimes/profiles.parquet
        issue: "Profiles currently cover engineered macro features; ETF return distributions by regime are produced elsewhere and not integrated into this artifact."
    missing:
      - "Decide whether ETF return statistics should be incorporated into profiles.parquet or exposed via a parallel, clearly documented artifact for regime-conditional ETF behavior."
      - "Add tests or integration checks that exercise ETF-return profiling alongside macro-feature profiles once Phase 4 work is in place."
  - truth: "There is a stable, version-controlled mapping from canonicalized cluster IDs to human-readable regime names, applied consistently across runs."
    status: partial
    reason: "The config file and override mechanism are implemented and tested, but no concrete, pinned names have been committed yet."
    artifacts:
      - path: config/regime_labels.yaml
        issue: "File exists with commented examples only; no actual ID→name mappings are pinned, so current runs rely solely on auto-suggested names."
      - path: data/regimes/regime_names_suggested.yaml
        issue: "Holds auto-suggested names (plus any overrides), but stability over time still depends on a curated regime_labels.yaml mapping."
    missing:
      - "Run the clustering and profiling pipelines on production-like data, review regime_names_suggested.yaml, and promote final chosen names into config/regime_labels.yaml."
      - "Document naming decisions (and any future renames) so that semantic changes to regimes are traceable across runs."
  - truth: "Phase 2 validation provides automated checks for clustering, profiling, and naming behavior, with Nyquist-compliant coverage."
    status: partial
    reason: "Unit tests for clustering and regime utilities exist and exercise the planned behaviors, but the validation metadata has not been updated to reflect their presence and Nyquist compliance."
    artifacts:
      - path: tests/unit/test_clustering.py
        issue: "Implements coverage for reduce_pca, evaluate_kmeans, pick_best_k, and fit_clusters as described, but 02-VALIDATION.md still marks this file as 'to add'."
      - path: tests/unit/test_regime.py
        issue: "Covers build_profiles, suggest_names, build_transition_matrix, and load_name_overrides, but 02-VALIDATION.md still lists it as a Wave 0 gap."
      - path: .planning/phases/02-regime-clustering-interpretation/02-VALIDATION.md
        issue: "Frontmatter has nyquist_compliant: false and Wave 0 items unchecked, which is now stale given the current test suite."
    missing:
      - "Update 02-VALIDATION.md to reflect that test_clustering.py and test_regime.py now exist, mark their status as green once pytest runs locally, and set nyquist_compliant: true when the sampling plan is satisfied."
      - "Ensure the prescribed quick commands (pytest tests/unit/test_clustering.py tests/unit/test_regime.py -q) are part of the standard CI or local-gate workflow for Phase 2 changes."
  - truth: "Visual stability and interpretability of regimes across reruns have been confirmed."
    status: human_needed
    reason: "Manual notebook- and plot-based inspection is required to judge interpretability and visual stability; this cannot be validated from static code alone."
    artifacts:
      - path: .planning/phases/02-regime-clustering-interpretation/02-VALIDATION.md
        issue: "Lists visual inspection of regime profiles and clustering stability as a manual-only verification; no automated proxy exists."
    missing:
      - "Run the clustering and profiling pipelines on real data and inspect the clustering and regime-profile notebooks/plots to confirm regimes are visually coherent and stable across reruns with the same config."
human_verification:
  - test: "Visual inspection of regime profiles and clustering stability across reruns."
    expected: "For a fixed feature set and clustering configuration, the number of regimes (~4–7), their macro profiles, and their suggested/human-assigned names remain stable across repeated runs; profiles are interpretable in terms of growth, inflation, rates, and risk."
    why_human: "Requires subjective assessment of plot outputs and narrative fit of regime names; cannot be evaluated via unit tests or static analysis."
---

# Phase 2: Regime Clustering & Interpretation Verification Report

**Phase Goal:** Produce a small, stable set of interpretable market regimes with reproducible profiles and names that downstream models and users can rely on.
**Verified:** 2026-03-16
**Status:** gaps_found
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
| --- | ----- | ------ | -------- |
| 1 | Historical quarters are assigned to a manageable number of regimes (target ~4–7) using the Phase 1 feature set and PCA + clustering. | ✓ VERIFIED | `pipelines/03_cluster.py` drives PCA and KMeans via `config.settings["clustering"]`, with `k_cap` and `balanced_k` both set to 5, so the pipeline always produces a small number of regimes; `tests/unit/test_clustering.py` asserts correct cluster counts and label canonicalization. |
| 2 | Re-running the clustering step with the same configuration and input features produces identical regime labels for all quarters. | ✓ VERIFIED | `reduce_pca`, `evaluate_kmeans`, and `fit_clusters` all use a fixed `random_state` from config and canonicalize labels based on mean PC1; tests cover deterministic, contiguous label ordering, making label assignments stable given fixed inputs. |
| 3 | The clustering step writes regime artifacts under `data/regimes/` that downstream steps and notebooks can reload without ad-hoc logic. | ✓ VERIFIED | `pipelines/03_cluster.py` writes `cluster_labels.parquet`, `pca_components.parquet`, and `kmeans_scores.parquet` with stable schemas, and these paths are referenced by downstream profiling code; artifacts are runtime outputs and correctly gitignored. |
| 4 | Each regime has a reproducible profile over key macro features and ETF returns that supports a human-readable description. | ⚠️ PARTIAL | `pipelines/04_regime_label.py` and `market_regime.regime.build_profiles` produce deterministic macro-feature profiles to `profiles.parquet` and are covered by `tests/unit/test_regime.py`, but ETF-return statistics by regime are not yet integrated into this artifact. |
| 5 | There is a stable, version-controlled mapping from canonicalized cluster IDs to human-readable regime names, applied consistently across runs. | ⚠️ PARTIAL | `suggest_names` + `load_name_overrides` implement deterministic auto-naming and config-driven overrides, and `regime_labels.yaml` exists under version control, but it currently contains only commented examples, so no concrete ID→name mapping has been pinned yet. |
| 6 | Downstream supervised models and reporting code can load regime profiles, transition matrices, and label mappings from disk without re-running notebooks. | ✓ VERIFIED | `pipelines/04_regime_label.py` writes `profiles.parquet`, `transition_matrix.parquet`, and `regime_names_suggested.yaml`, and all profiling, naming, and transition logic lives in `market_regime.regime` with unit tests in `tests/unit/test_regime.py`, making these artifacts reproducible and notebook-independent. |

**Score:** 5/6 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
| -------- | -------- | ------ | ------- |
| `data/regimes/cluster_labels.parquet` | Quarter-level regime labels with both `cluster` and `balanced_cluster` (and optional `market_code`). | ✓ VERIFIED (by implementation) | Written by `pipelines/03_cluster.py` using `fit_clusters`; tests confirm label shapes and canonicalization, though the file itself is a runtime artifact and not present in git. |
| `data/regimes/pca_components.parquet` | Quarter-level PCA components PC1…PCn used for clustering and diagnostics. | ✓ VERIFIED (by implementation) | `pipelines/03_cluster.py` drops label columns and persists remaining PC columns; schema matches `reduce_pca` output. |
| `data/regimes/kmeans_scores.parquet` | k-sweep evaluation table (`k`, `inertia`, `silhouette`, `calinski`, `davies_bouldin`). | ✓ VERIFIED (by implementation) | `evaluate_kmeans` returns the full score table and `03_cluster.py` writes it with `index=False`; tests assert expected columns. |
| `data/regimes/profiles.parquet` | Per-regime descriptive statistics over Phase 1 features (and later ETF returns). | ✓ VERIFIED (macro-only) | Produced by `pipelines/04_regime_label.py` via `build_profiles`; tests check correctness and index alignment; ETF-return integration is pending. |
| `data/regimes/transition_matrix.parquet` | Empirical one-step transition probabilities between regimes. | ✓ VERIFIED | Written by `pipelines/04_regime_label.py` using `build_transition_matrix`; tests confirm probability semantics and row sums. |
| `data/regimes/regime_names_suggested.yaml` | Auto-suggested regime names (after applying overrides) for human review. | ✓ VERIFIED | Written by `pipelines/04_regime_label.py` from `suggest_names` merged with `load_name_overrides`; deterministic given fixed data and config. |
| `config/regime_labels.yaml` | Pinned, version-controlled mapping from canonicalized cluster IDs to human-readable regime names. | ⚠️ PARTIAL | File exists and is loaded by `load_name_overrides`, but currently contains only commented examples with no active mappings. |
| `tests/unit/test_clustering.py` | Unit tests for PCA + KMeans clustering and artifact creation. | ✓ VERIFIED | Present and exercises `reduce_pca`, `evaluate_kmeans`, `pick_best_k`, and `fit_clusters`, including canonicalization and fallback behavior. |
| `tests/unit/test_regime.py` | Unit tests for regime profiles, naming, transitions, and overrides. | ✓ VERIFIED | Present and covers `build_profiles`, `suggest_names`, `build_transition_matrix`, and `load_name_overrides` on synthetic data. |

### Key Link Verification

| From | To | Via | Status | Details |
| ---- | -- | --- | ------ | ------- |
| `pipelines/03_cluster.py` | `src/market_regime/clustering.py` | `reduce_pca, evaluate_kmeans, pick_best_k, fit_clusters` | ✓ VERIFIED | Imports from `market_regime.clustering` and uses all four core functions to implement the clustering pipeline. |
| `pipelines/03_cluster.py` | `config/settings.yaml` | `clustering.* configuration` | ✓ VERIFIED | Loads config via `load()` and reads `cfg["clustering"]` for `n_pca_components`, `n_clusters_search`, `k_cap`, `balanced_k`, and `random_state`; no clustering constants are hardcoded. |
| `tests/unit/test_clustering.py` | `src/market_regime/clustering.py` | Direct imports of clustering helpers | ✓ VERIFIED | Imports and exercises `reduce_pca`, `evaluate_kmeans`, `pick_best_k`, and `fit_clusters` with synthetic data, including canonicalization behavior. |
| `pipelines/04_regime_label.py` | `src/market_regime/regime.py` | `build_profiles, suggest_names, build_transition_matrix, load_name_overrides` | ✓ VERIFIED | Imports all four utilities and uses them to produce profiles, suggested names, overrides, and the transition matrix. |
| `pipelines/04_regime_label.py` | `config/regime_labels.yaml` | Manual name overrides layered on auto-suggestions | ✓ VERIFIED | Calls `load_name_overrides(CONFIG_DIR)` and merges overrides into `regime_names_suggested.yaml`. |
| `tests/unit/test_regime.py` | `src/market_regime/regime.py` | Direct imports and synthetic fixtures | ✓ VERIFIED | Imports regime utilities and validates their behavior on synthetic data, matching the validation strategy. |

### Requirements Coverage

| Requirement | Description | Status | Evidence |
| ----------- | ----------- | ------ | -------- |
| REGIME-01 | Cluster quarters into a small, interpretable set of regimes using PCA + KMeans, targeting ~4–7 regimes. | PASS | Clustering is entirely config-driven via `settings.yaml` with `n_pca_components=5`, `k_cap=5`, and `balanced_k=5`; `pipelines/03_cluster.py` and `clustering.fit_clusters` implement this pipeline, and `tests/unit/test_clustering.py` guards core behavior and label determinism. |
| REGIME-02 | For each regime, compute descriptive statistics over key macro features and ETF returns in reproducible code. | WARN | `build_profiles` and `pipelines/04_regime_label.py` produce deterministic macro-feature profiles to `profiles.parquet` with dedicated tests, but ETF-return profiling is not yet integrated into this artifact; ETF behavior by regime is expected to be expanded in later phases. |
| REGIME-03 | Provide a stable, version-controlled mapping from cluster IDs to human-readable regime names. | WARN | The naming stack (`suggest_names` + `load_name_overrides` + `regime_labels.yaml`) is fully implemented and tested, and `regime_labels.yaml` is version-controlled, but concrete ID→name mappings have not yet been committed, so stability currently relies on heuristics rather than curated names. |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
| ---- | ---- | ------- | -------- | ------ |
| `config/regime_labels.yaml` | n/a | Only commented example names; no active mappings. | ⚠️ Warning | Leaves regime naming at the mercy of auto-suggestions; stability over time depends on not changing heuristics or feature schema. |
| `.planning/phases/02-regime-clustering-interpretation/02-VALIDATION.md` | 2–7, 48–52 | nyquist_compliant: false and unchecked Wave 0 items despite tests being present. | ⚠️ Warning | Validation metadata understates current automated coverage; could mislead future maintainers about missing tests. |

### Human Verification Required

See `human_verification` entries in the frontmatter above; primary manual check is visual inspection of regime profiles and clustering stability using notebooks and plots.

### Gaps Summary

Phase 2 has fully implemented, deterministic clustering and macro-feature profiling with strong unit-test coverage, and the naming infrastructure is in place, but curated regime names and ETF-return integration into profiles remain open. Validation metadata also lags behind the actual test suite, and manual visual checks of regime interpretability and stability are still required. Addressing these gaps will move REGIME-02 and REGIME-03 from WARN to full PASS and complete Nyquist validation for this phase.

---

_Verified: 2026-03-16_
_Verifier: Claude (gsd-verifier)_

