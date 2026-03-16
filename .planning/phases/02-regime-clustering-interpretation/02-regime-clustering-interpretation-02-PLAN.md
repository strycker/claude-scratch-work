---
phase: 02-regime-clustering-interpretation
plan: 02
type: execute
wave: 2
depends_on:
  - 02-regime-clustering-interpretation-01
files_modified:
  - pipelines/04_regime_label.py
  - src/market_regime/regime.py
  - config/regime_labels.yaml
  - tests/unit/test_regime.py
autonomous: true
requirements:
  - REGIME-01
  - REGIME-02
  - REGIME-03
must_haves:
  truths:
    - Each regime has a reproducible profile over key macro features and ETF returns that supports a human-readable description.
    - There is a stable, version-controlled mapping from canonicalized cluster IDs to human-readable regime names, applied consistently across runs.
    - Downstream supervised models and reporting code can load regime profiles, transition matrices, and label mappings from disk without re-running notebooks.
  artifacts:
    - path: data/regimes/profiles.parquet
      provides: "Per-regime descriptive statistics over the Phase 1 feature set (and, when available, ETF returns) used for human-readable regime descriptions."
    - path: data/regimes/transition_matrix.parquet
      provides: "Empirical one-step transition probabilities P(next=j | current=i) between regimes."
    - path: data/regimes/regime_names_suggested.yaml
      provides: "Auto-suggested regime names (after applying overrides) for human review."
    - path: config/regime_labels.yaml
      provides: "Pinned, version-controlled mapping from canonicalized cluster IDs to human-readable regime names."
    - path: tests/unit/test_regime.py
      provides: "Unit tests covering build_profiles, suggest_names, build_transition_matrix, and load_name_overrides behavior required for REGIME-02 and REGIME-03."
  key_links:
    - from: pipelines/04_regime_label.py
      to: src/market_regime/regime.py
      via: "build_profiles, suggest_names, build_transition_matrix, load_name_overrides"
      pattern: "from market_regime.regime import"
    - from: pipelines/04_regime_label.py
      to: config/regime_labels.yaml
      via: "manual name overrides layered on top of auto-suggested names"
      pattern: "load_name_overrides(CONFIG_DIR)"
    - from: tests/unit/test_regime.py
      to: src/market_regime/regime.py
      via: "direct imports and synthetic data fixtures for profiles, names, and transitions"
      pattern: "from market_regime.regime import"
---

<objective>
Implement and validate regime profiling, naming, and transition-matrix logic so that each regime has a reproducible profile and stable human-readable name, backed by unit tests and config-driven overrides.

Purpose: Satisfy REGIME-02 and REGIME-03 (and complement REGIME-01) by turning raw cluster labels into interpretable, stable regimes with machine- and human-consumable artifacts under data/regimes/ and config/regime_labels.yaml.
Output: A profiling and labeling pipeline (pipelines/04_regime_label.py + market_regime.regime) plus tests and configs that produce profiles.parquet, transition_matrix.parquet, and a deterministic mapping from cluster IDs to names.
</objective>

<execution_context>
@.planning/ROADMAP.md
@.planning/REQUIREMENTS.md
@.planning/phases/02-regime-clustering-interpretation/02-RESEARCH.md
@CLAUDE.md
</execution_context>

<context>
@pipelines/04_regime_label.py
@src/market_regime/regime.py
@config/regime_labels.yaml
@data/regimes/cluster_labels.parquet
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Add unit tests for regime profiles, names, and transitions</name>
  <files>tests/unit/test_regime.py</files>
  <behavior>
  - Test build_profiles:
    - Given a small synthetic features DataFrame (e.g., quarterly index with 2–3 numeric columns) and an aligned labels Series with a few regimes, build_profiles returns a DataFrame indexed by cluster ID whose statistics (e.g., means) match manual calculations for each regime.
    - When features and labels have non-identical indexes, build_profiles correctly aligns on the intersection without raising and does not produce NaNs in basic statistics for regimes that have at least one observation.
  - Test suggest_names + load_name_overrides:
    - On a synthetic features + labels pair where one regime clearly has higher “inflation-like” or “growth-like” values than another, suggest_names returns distinct, deterministic names across runs with the same input.
    - When config/regime_labels.yaml provides an override for a particular cluster ID, load_name_overrides applies it and the merged mapping prefers manual overrides over auto-suggestions.
  - Test build_transition_matrix:
    - For a short, known sequence of labels (e.g., [0, 0, 1, 1, 0]), build_transition_matrix returns a square DataFrame whose rows sum to 1 (within numerical tolerance) and whose entries match hand-computed transition probabilities.
    - The transition matrix includes all regimes present in the input labels, even if some have no outgoing transitions (rows should still exist, possibly as zeros).
  </behavior>
  <action>
  Create tests/unit/test_regime.py to exercise the core behaviors of the regime profiling and naming utilities so that REGIME-02 and REGIME-03 are guarded by automated tests.

  - Define synthetic fixtures (either inline or via simple helper functions) that:
    - Create a small quarterly DatetimeIndex (e.g., 6–8 periods) with numeric columns representing stylized macro features (e.g., "inflation", "growth") with clearly separated regime behavior.
    - Create a pandas Series of integer regime labels aligned to the index, with at least two distinct regimes and at least one transition between them.

  - Write tests for build_profiles that:
    - Call build_profiles(features, labels) and verify:
      - The index of the returned profile includes all regimes present in labels.
      - The mean of each feature per regime in the profile matches the expected mean computed manually from the synthetic features subset.
    - Include a case where features has extra rows or labels has fewer rows, and assert that build_profiles uses the intersection of indexes (no misalignment bugs).

  - Write tests for suggest_names and load_name_overrides that:
    - Call suggest_names(features, labels) and assert:
      - It returns a mapping from regime ID to non-empty string name for each regime in labels.
      - The mapping is deterministic across repeated calls given the same input (e.g., by comparing two calls within the same test).
    - Use a temporary YAML override file (e.g., via tmp_path fixture) to simulate config/regime_labels.yaml:
      - Write a simple override mapping (e.g., {0: "Custom Regime A"}) and point load_name_overrides at that path to verify that:
        - Overrides are read correctly.
        - When combined with suggest_names, manual overrides replace the corresponding auto-suggested entries.

  - Write tests for build_transition_matrix that:
    - Construct a short label sequence with known transitions (e.g., [0, 0, 1, 0, 1]) and call build_transition_matrix(labels).
    - Assert that:
      - The matrix has one row/column per regime ID in the labels.
      - Each row sums to 1 (within a small epsilon).
      - Specific entries match hand-computed probabilities (e.g., P(next=1 | current=0)).

  Keep the tests self-contained, using only synthetic data and temporary paths; do not rely on actual project data files. Follow existing testing patterns from tests/unit/test_clustering.py and tests/conftest.py.
  </action>
  <verify>
  - Automated: pytest tests/unit/test_regime.py -q
  </verify>
  <done>
  - tests/unit/test_regime.py exists and contains tests for build_profiles, suggest_names, build_transition_matrix, and load_name_overrides.
  - pytest tests/unit/test_regime.py -q passes locally without requiring any external data or network access.
  - Failing behaviors for REGIME-02 or REGIME-03 (e.g., misaligned profiles, unstable names, broken overrides, or invalid transition matrices) would now be caught by these tests.
  </done>
</task>

<task type="auto">
  <name>Task 2: Wire profiling, naming, and transition artifacts into the pipeline</name>
  <files>pipelines/04_regime_label.py, src/market_regime/regime.py, config/regime_labels.yaml</files>
  <action>
  Ensure the regime profiling and labeling pipeline produces the expected artifacts under data/regimes/ and respects manual name overrides from config/regime_labels.yaml while using the canonicalized cluster IDs from Step 3.

  - In pipelines/04_regime_label.py, confirm and, if needed, refine that:
    - It reads features from data/processed/features.parquet and balanced_cluster labels from data/regimes/cluster_labels.parquet, then aligns them via a common index intersection before profiling.
    - It calls build_profiles(features, labels) and writes the result to data/regimes/profiles.parquet with a stable schema (index = regime IDs, columns = feature/statistics as defined in build_profiles).
    - It calls suggest_names(features, labels) to obtain auto-suggested names, then load_name_overrides(CONFIG_DIR) to obtain manual overrides from config/regime_labels.yaml, and merges them with overrides taking precedence.
    - It writes the merged mapping to data/regimes/regime_names_suggested.yaml so that the user can inspect the final name assignments for the current configuration.
    - It calls build_transition_matrix(labels) and writes the resulting DataFrame to data/regimes/transition_matrix.parquet.
    - It prints/logs a concise summary of regime IDs, names, and quarter counts, plus a rounded transition matrix for quick manual inspection.

  - In src/market_regime/regime.py, review build_profiles, suggest_names, build_transition_matrix, and load_name_overrides to ensure:
    - They treat labels as already canonicalized IDs (from Step 3) and do not perform extra relabeling that could drift from cluster_labels.parquet.
    - build_profiles cleanly handles NaNs and index alignment (only intersection is used), and produces interpretable statistics (e.g., means/medians) that are stable across runs with the same inputs.
    - suggest_names bases its heuristics on well-defined comparisons of feature medians or other stable aggregates, so name suggestions are deterministic and not sensitive to row ordering.
    - load_name_overrides reads config/regime_labels.yaml from CONFIG_DIR, gracefully handles missing or partial mappings, and returns a plain dict[int, str] suitable for overlaying onto auto-suggestions.

  - In config/regime_labels.yaml, create or update a minimal initial mapping (even if provisional), for example:
    - Map known regime IDs to placeholder but human-readable names (e.g., "Regime 0: Placeholder Growth", etc.).
    - Include comments where appropriate to document that these names are curated and should only change intentionally after reviewing profiles and suggested names.

  Keep the behavioral contract simple: pipelines/03_cluster.py defines canonical regime IDs via balanced_cluster; pipelines/04_regime_label.py adds human meaning and transition structure, drawing only on the stable utilities in market_regime.regime and overrides from config/regime_labels.yaml.
  </action>
  <verify>
  - Automated: python pipelines/03_cluster.py &amp;&amp; python pipelines/04_regime_label.py
  - Automated: pytest tests/unit/test_regime.py -q
  </verify>
  <done>
  - Running python pipelines/03_cluster.py &amp;&amp; python pipelines/04_regime_label.py on the project data completes without error and produces profiles.parquet, transition_matrix.parquet, and regime_names_suggested.yaml under data/regimes/.
  - config/regime_labels.yaml exists, is loaded by load_name_overrides, and its overrides are reflected in the final regime_names_suggested.yaml mapping.
  - pytest tests/unit/test_regime.py -q passes and exercises the core profiling, naming, and transition-matrix behaviors required for REGIME-02 and REGIME-03.
  </done>
</task>

</tasks>

<verification>
- pytest tests/unit/test_regime.py -q passes, providing coverage for build_profiles, suggest_names, build_transition_matrix, and load_name_overrides on synthetic data.
- Running python pipelines/03_cluster.py &amp;&amp; python pipelines/04_regime_label.py on the Phase 1 outputs writes profiles.parquet, transition_matrix.parquet, and regime_names_suggested.yaml under data/regimes/ without errors.
- Inspecting regime_names_suggested.yaml and config/regime_labels.yaml shows a stable mapping from canonical regime IDs to human-readable names, and transition_matrix.parquet rows sum to 1 within numerical tolerance.
</verification>

<success_criteria>
- REGIME-02 is satisfied: every regime has a reproducible profile and transition structure encoded in profiles.parquet and transition_matrix.parquet, suitable for human-readable descriptions and downstream modeling.
- REGIME-03 is satisfied: a version-controlled config/regime_labels.yaml pins human-readable names to canonical regime IDs, and the combined naming logic is deterministic across runs with the same data and configuration.
- Downstream phases (supervised models, portfolio analysis, recommendations) can rely on these artifacts as stable interfaces to regime semantics.
</success_criteria>

<output>
After completion, this plan will be executed as part of /gsd:execute-phase 2, producing regime profiles, transitions, and a stable naming configuration. No additional documentation beyond the standard Phase 2 SUMMARY files is required unless new gaps are discovered during execution.
</output>

