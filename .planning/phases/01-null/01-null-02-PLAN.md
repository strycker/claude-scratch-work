---
phase: 01-null
plan: 02
type: execute
wave: 2
depends_on:
  - 01-null-01
files_modified:
  - config/settings.yaml
  - pipelines/02_features.py
  - src/market_regime/features/transforms.py
  - src/market_regime/io/checkpoints.py
  - notebooks/02_features.ipynb
autonomous: true
requirements:
  - DATA-02
  - DATA-03
must_haves:
  truths:
    - "A stable, documented feature set is computed end-to-end from checkpointed raw data."
    - "Both non-causal and causal feature variants are produced as separate, clearly named artifacts."
    - "Downstream phases can rely on a documented feature contract (columns, index, and frequency) without re-deriving it from notebooks."
    - "Supervised training code can unambiguously load causal features with no look-ahead leakage."
  artifacts:
    - path: "config/settings.yaml"
      provides: "Config-driven definitions for feature lists, causal flags, and any additional yield-curve/feature settings."
    - path: "pipelines/02_features.py"
      provides: "Step-02 feature orchestration that reads from checkpoints and writes dual feature artifacts."
    - path: "src/market_regime/features/transforms.py"
      provides: "Implementation of engineer_all(causal=...) and the ordered feature pipeline."
    - path: "data/checkpoints/features_noncausal.parquet"
      provides: "Non-causal feature set for clustering and exploratory work."
      min_lines: 1
    - path: "data/checkpoints/features_causal.parquet"
      provides: "Causal feature set suitable for supervised learning without look-ahead bias."
      min_lines: 1
  key_links:
    - from: "pipelines/02_features.py"
      to: "src/market_regime/features.transforms.engineer_all"
      via: "function calls for causal=False and causal=True variants"
    - from: "pipelines/02_features.py"
      to: "src/market_regime/io.checkpoints.CheckpointManager"
      via: "save_parquet and is_fresh calls for feature checkpoints"
    - from: "pipelines/02_features.py"
      to: "data/checkpoints/macro_raw.parquet"
      via: "loading raw macro data as input to feature engineering"
    - from: "pipelines/02_features.py"
      to: "data/checkpoints/asset_prices.parquet"
      via: "loading ETF prices as input to feature engineering"
---

<objective>
Standardize the feature engineering step so it produces documented, dual (non-causal and causal) feature artifacts from checkpointed raw data, with clear contracts for downstream clustering and supervised models.

Purpose: Turn `engineer_all(causal=...)` and its config into a stable feature contract that satisfies DATA-02/03 and prevents look-ahead leakage.
Output: Updated feature orchestration, documented feature artifacts, and clear separation between causal and non-causal feature sets.
</objective>

<execution_context>
@.planning/PROJECT.md
@.planning/REQUIREMENTS.md
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/phases/01-null/01-RESEARCH.md
@.planning/phases/01-null/01-null-01-PLAN.md
</execution_context>

<context>
@CLAUDE.md
@config/settings.yaml
@pipelines/02_features.py
@src/market_regime/features/transforms.py
@src/market_regime/io/checkpoints.py
@notebooks/02_features.ipynb
</context>

<tasks>

<task type="auto">
  <name>Task 1: Ensure config-driven feature lists and causal flags are explicit</name>
  <files>
config/settings.yaml
src/market_regime/features/transforms.py
  </files>
  <action>
- Review `config/settings.yaml` and ensure that:
  - `features.initial_features` and `features.clustering_features` reflect the intended feature lists for Phase 1 and are documented with comments where helpful.
  - Any additional yield-curve or feature settings called out in `CLAUDE.md` and Phase 1 RESEARCH are present under appropriate keys (e.g. in `features.*` or `data.*`), not hard-coded in Python.
- In `src/market_regime/features/transforms.py`, confirm that `engineer_all(cfg, causal: bool)`:
  - Implements the feature pipeline order exactly as documented in `CLAUDE.md` (ratios → log → select → gap-fill → derivatives → select).
  - Uses the config-driven feature lists rather than in-function constants.
  - When `causal=True`, applies any required causal windowing/lagging behavior so that no future information leaks into the resulting features.
- If needed, add docstrings or comments focused on non-obvious intent (e.g. why certain series are shifted or why causal mode differs) without restating obvious code.
  </action>
  <verify>
- Run `pytest tests/ -k "features" -v` to ensure existing feature tests pass with the clarified config-driven behavior.
  </verify>
  <done>
- Feature engineering is fully driven by `config/settings.yaml`, and `engineer_all(causal=...)` reflects the documented pipeline and causal behavior.
- There are no hard-coded feature lists or magic constants in the feature code that contradict the config.
  </done>
</task>

<task type="auto">
  <name>Task 2: Wire Step 02 features to produce dual checkpoints from raw data</name>
  <files>
pipelines/02_features.py
src/market_regime/io/checkpoints.py
  </files>
  <action>
- In `pipelines/02_features.py`, ensure a `main(run_cfg: RunConfig)` function:
  - Loads config via `market_regime.config.load()` and uses `CheckpointManager` to read `macro_raw` and `asset_prices` checkpoints produced by Step 1.
  - Calls `engineer_all(cfg, causal=False)` to produce the non-causal feature DataFrame and `engineer_all(cfg, causal=True)` to produce the causal variant.
  - Saves the resulting DataFrames as distinct checkpoints (e.g. `features_noncausal` and `features_causal`) via `cm.save_parquet`, with names matching Phase 1 RESEARCH.
  - Optionally respects a freshness check (e.g. `is_fresh("features_noncausal")`) when deciding whether to recompute features, guided by `run_cfg.recompute_derived_datasets`.
- Make sure the step uses only the quarterly (or monthly→quarterly) index from the raw checkpoints and does not introduce sub-daily frequencies.
- Confirm that logs clearly indicate when features are recomputed vs loaded from checkpoints, and which artifact names are being written.
  </action>
  <verify>
- Run `python pipelines/02_features.py` to confirm it completes successfully and writes `features_noncausal` and `features_causal` checkpoints to `data/checkpoints/`.
- Run `pytest tests/ -k "features" -v` again to confirm tests still pass with the dual-artifact behavior.
  </verify>
  <done>
- Running Step 02 via `pipelines/02_features.py` produces both non-causal and causal feature checkpoints from `macro_raw` and `asset_prices`.
- Feature artifacts are named consistently and can be loaded unambiguously by downstream steps.
  </done>
</task>

<task type="auto">
  <name>Task 3: Document the feature contract for downstream phases</name>
  <files>
notebooks/02_features.ipynb
README.md
  </files>
  <action>
- Update `notebooks/02_features.ipynb` (or an appropriate markdown/README section if better) to briefly describe:
  - The two feature artifacts (`features_noncausal` and `features_causal`), including their intended use (clustering vs supervised models).
  - The expected index frequency (quarterly) and high-level column groups (ratios, logs, derivatives, etc.).
  - Any key caveats around causal features (e.g. publication-lag shifts, rolling windows).
- In `README.md` or a concise docs section, add a short Phase 1–level description of the feature pipeline and outputs so users know:
  - Which command(s) to run to regenerate features.
  - Which parquet files to load for clustering vs supervised learning.
  - How this relates to the Phase 1 requirements DATA-02 and DATA-03.
  </action>
  <verify>
- Manually open the updated notebook/documentation and confirm the feature artifacts, their intended uses, and the regeneration commands are clearly explained.
  </verify>
  <done>
- Downstream work in Phases 2–3 has a clear, documented description of the feature artifacts and how to regenerate them, without needing to inspect the code.
  </done>
</task>

</tasks>

<verification>
- `pytest tests/ -k "features" -v` passes, validating the config-driven feature pipeline and dual artifacts.
- `python pipelines/02_features.py` successfully reads raw checkpoints and produces `features_noncausal` and `features_causal` checkpoints.
- Documentation (notebook and/or README) clearly describes the feature artifacts, their uses, and how to regenerate them.
</verification>

<success_criteria>
- The feature engineering step is reproducible, config-driven, and produces separate non-causal and causal feature artifacts from checkpointed raw data.
- Downstream code can rely on these artifacts without ambiguity about which features are causal and without risk of look-ahead leakage.
- Users can easily understand and regenerate the feature artifacts with documented commands.
</success_criteria>

<output>
After completion, ensure `.planning/ROADMAP.md` Phase 1 references the dual feature artifacts and that future plans in Phases 2–3 can point to `features_noncausal` and `features_causal` as stable inputs.
</output>

