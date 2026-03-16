---
phase: 03-supervised-regime-behavior-models
plan: 02
type: execute
wave: 2
depends_on:
  - 03-01
files_modified:
  - src/market_regime/prediction/classifier.py
  - tests/test_models_regime.py
  - tests/test_models_reporting.py
autonomous: true
requirements:
  - MODEL-01
  - MODEL-02
  - MODEL-04
user_setup: []
must_haves:
  truths:
    - "The system can train current-regime classifiers on causal features with walk-forward, time-series aware validation."
    - "Forward-horizon models emit regime transition probabilities for at least one quarter ahead in an interpretable format."
    - "Model performance for current and forward regimes is summarized with transparent, per-regime metrics suitable for later reporting."
  artifacts:
    - path: "src/market_regime/prediction/classifier.py"
      provides: "Fully implemented training and evaluation helpers for current and forward regime classifiers."
      min_lines: 200
    - path: "tests/test_models_regime.py"
      provides: "Detailed tests covering current and forward regime models, including probability shapes and leakage guards."
      min_lines: 120
    - path: "tests/test_models_reporting.py"
      provides: "Tests ensuring regime model metrics and summaries are time-series aware and human-inspectable."
      min_lines: 80
  key_links:
    - from: "src/market_regime/prediction/classifier.py"
      to: "src/market_regime/io/checkpoints.py"
      via: "CheckpointManager used to persist trained regime models and metrics under outputs/models"
      pattern: "CheckpointManager"
    - from: "tests/test_models_regime.py"
      to: "src/market_regime/prediction/classifier.py"
      via: "import train_current_regime and train_forward_classifiers"
      pattern: "from market_regime\\.prediction\\.classifier import train_current_regime, train_forward_classifiers"
    - from: "tests/test_models_reporting.py"
      to: "src/market_regime/prediction/classifier.py"
      via: "import model_metrics_summary for regime models"
      pattern: "model_metrics_summary"
---

<objective>
Implement end-to-end current and forward regime classifiers using causal features, walk-forward validation, and standardized metrics so Phase 3 can expose reliable regime signals and performance summaries.

Purpose: Turn Phase 2 regime labels and Phase 1 causal features into supervised models that predict the current regime and near-term transitions with honest time-series evaluation.
Output: Working implementations of `train_current_regime`, `train_forward_classifiers`, and regime-focused reporting helpers with passing tests and persisted model/metrics checkpoints.
</objective>

<execution_context>
@.planning/PROJECT.md
@.planning/REQUIREMENTS.md
@.planning/ROADMAP.md
@.planning/phases/03-supervised-regime-behavior-models/03-RESEARCH.md
@.planning/phases/03-supervised-regime-behavior-models/03-supervised-regime-behavior-models-01-SUMMARY.md
</execution_context>

<context>
@CLAUDE.md
@src/market_regime/__init__.py
@src/market_regime/io/checkpoints.py
@src/market_regime/regime/profiler.py

<interfaces>
<!-- Interfaces from the classifier scaffold that this plan must fully implement. -->

From src/market_regime/prediction/classifier.py:
```python
def train_current_regime(
    features: pd.DataFrame,
    labels: pd.Series,
    cv_splits: int = 5,
) -> dict: ...

def train_forward_classifiers(
    features: pd.DataFrame,
    regimes: pd.Series,
    horizons: list[int],
) -> dict[int, dict]: ...

def model_metrics_summary(results: dict) -> dict: ...
```
</interfaces>
</context>

<tasks>

<task type="auto">
  <name>Task 1: Implement current-regime classifier with walk-forward CV and metrics</name>
  <files>src/market_regime/prediction/classifier.py, tests/test_models_regime.py</files>
  <action>
    - Flesh out `train_current_regime` so that it:
      - Accepts a feature matrix of causal quarterly features (index aligned with regime labels) and a `labels` Series containing integer or named regime IDs from Phase 2.
      - Uses an internal `_tscv_scores` helper built on `TimeSeriesSplit` (no shuffling) to compute per-split classification reports via `sklearn.metrics.classification_report(output_dict=True)`.
      - Trains at least two models: a `DecisionTreeClassifier(max_depth=8, random_state=42)` and a `RandomForestClassifier(max_depth=12, n_estimators=200, random_state=42)`.
      - Returns a dict containing:
        - `"models"`: mapping short model names (e.g. `"dt"`, `"rf"`) to fitted estimators trained on the full dataset.
        - `"cv_reports"`: list of per-split reports for each model.
        - `"labels"`: the fitted label index / classes, to help downstream consumers interpret probabilities.
    - Update `tests/test_models_regime.py` to:
      - Use realistic synthetic data with at least 30 time steps and 3+ regimes.
      - Assert that:
        - Each CV split’s test indices come strictly after train indices.
        - The returned `"models"` and `"cv_reports"` have entries for both `"dt"` and `"rf"`.
        - A simple end-to-end call to `train_current_regime` produces probability outputs (`predict_proba`) for all regimes.
  </action>
  <verify>
    <automated>pytest -q -k "models_regime and current_regime"</automated>
  </verify>
  <done>
    - `train_current_regime` trains DecisionTree and RandomForest models without leakage, verified by `TimeSeriesSplit`-based tests.
    - Tests confirm that CV splits respect temporal ordering and that predictions/probabilities are defined for all regimes.
  </done>
</task>

<task type="auto">
  <name>Task 2: Implement forward regime transition classifiers for 1+ horizons</name>
  <files>src/market_regime/prediction/classifier.py, tests/test_models_regime.py</files>
  <action>
    - Implement `train_forward_classifiers` to:
      - Take `features` and `regimes` aligned at time \(t\), plus a list of integer horizons (e.g. `[1, 2, 4, 8]`, eventually driven by config but hard-coded defaults are acceptable here).
      - For each horizon `h`, construct targets `y_{t+h}` using `regimes.shift(-h)` and drop rows with missing targets, ensuring that:
        - The last `h` rows are excluded from both training and validation for that horizon.
      - For each horizon, train models analogous to `train_current_regime` (DecisionTree + RandomForest) with `TimeSeriesSplit` CV, returning:
        - Per-horizon `"models"` (fitted on available data).
        - `"cv_reports"` per horizon and per model.
        - Optional `"class_order"` listing regimes for that horizon.
      - Ensure outputs can be used to compute empirical transition probabilities \(P(\text{regime}_{t+h} = j \mid X_t, \text{regime}_t)\), even if this function only returns standard multi-class probabilities.
    - Extend `tests/test_models_regime.py` with:
      - Synthetic tests for at least horizon `h = 1`, verifying:
        - Targets are correctly shifted (e.g. by comparing to a manually constructed Series).
        - No sample uses features from time \(t+h\) to predict \(\text{regime}_{t+h}\).
        - Probability outputs for each horizon sum to 1 across regimes for each sample.
  </action>
  <verify>
    <automated>pytest -q -k "models_regime and forward_regime"</automated>
  </verify>
  <done>
    - `train_forward_classifiers` returns a non-empty dict keyed by horizon with trained models and CV metrics.
    - Tests validate correct target shifting, exclusion of trailing quarters, and well-formed probability vectors.
  </done>
</task>

<task type="auto">
  <name>Task 3: Implement regime model metrics summary and reporting helpers</name>
  <files>src/market_regime/prediction/classifier.py, tests/test_models_reporting.py</files>
  <action>
    - Implement `model_metrics_summary` (and any small helpers it needs) to:
      - Accept the raw results dicts from `train_current_regime` and `train_forward_classifiers`.
      - Aggregate per-split `classification_report` dicts into:
        - Overall accuracy, macro-F1, and weighted-F1 per model (and per horizon where applicable).
        - Per-class metrics (precision, recall, F1, support) for each regime.
      - Return a compact, JSON-serializable structure that later phases can write to disk alongside models.
    - Update `tests/test_models_reporting.py` to:
      - Construct representative fake CV report data (mimicking `classification_report(output_dict=True)` output).
      - Assert that `model_metrics_summary`:
        - Preserves all regime labels/classes present in the input.
        - Produces sensible aggregate metrics (e.g. accuracy between 0 and 1, non-negative supports).
        - Is stable under repeated calls and does not mutate the input data structure.
  </action>
  <verify>
    <automated>pytest -q -k "models_reporting and regime"</automated>
  </verify>
  <done>
    - Regime model metrics can be summarized in a standardized dict suitable for serialization and later reporting.
    - Reporting tests pass and guard against regressions in metric aggregation logic.
  </done>
</task>

</tasks>

<verification>
- Run `pytest -q -k "models_regime or models_reporting"` and confirm all tests pass.
- Optionally run a local script or notebook cell that uses checkpointed feature/label slices to smoke-test `train_current_regime` and `train_forward_classifiers` on real data (no network).
</verification>

<success_criteria>
- Current-regime and forward-regime classifiers train successfully on causal features with TimeSeriesSplit validation and clear metrics, satisfying MODEL-01 and MODEL-02.
- Regime model performance is summarized through `model_metrics_summary`, providing transparent, per-regime metrics in support of MODEL-04.
</success_criteria>

<output>
After completion, ensure `.planning/phases/03-supervised-regime-behavior-models/03-supervised-regime-behavior-models-02-SUMMARY.md` is created by the executor to record the implemented regime models and metrics.
</output>

