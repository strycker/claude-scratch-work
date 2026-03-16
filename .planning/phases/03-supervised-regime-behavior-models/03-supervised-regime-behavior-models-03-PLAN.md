---
phase: 03-supervised-regime-behavior-models
plan: 03
type: execute
wave: 3
depends_on:
  - 03-01
  - 03-02
files_modified:
  - src/market_regime/prediction/classifier.py
  - tests/test_models_behavior.py
  - tests/test_models_reporting.py
autonomous: true
requirements:
  - MODEL-03
  - MODEL-04
user_setup: []
must_haves:
  truths:
    - "The system can derive simple directional labels (e.g. up/flat/down) for ETF and portfolio returns over the next quarter from returns data."
    - "Forward behavior models map current features and regimes to probabilistic directional outcomes for ETFs or candidate portfolios."
    - "Behavior model performance is evaluated with time-series aware metrics and surfaced alongside regime model metrics."
  artifacts:
    - path: "src/market_regime/prediction/classifier.py"
      provides: "Helpers to construct forward behavior labels and train directional classifiers for ETFs/portfolios."
      min_lines: 260
    - path: "tests/test_models_behavior.py"
      provides: "Behavior-focused tests verifying label construction, model training, and directional outputs."
      min_lines: 100
    - path: "tests/test_models_reporting.py"
      provides: "Additional tests verifying behavior model metrics and joint reporting with regime models."
      min_lines: 100
  key_links:
    - from: "src/market_regime/prediction/classifier.py"
      to: "src/market_regime/assets/returns.py"
      via: "Functions that accept quarterly ETF/portfolio returns DataFrames produced by the returns pipeline"
      pattern: "returns"
    - from: "tests/test_models_behavior.py"
      to: "src/market_regime/prediction/classifier.py"
      via: "import make_behavior_labels and train_forward_behavior_models"
      pattern: "train_forward_behavior_models"
---

<objective>
Implement forward-looking ETF and portfolio behavior models that convert returns into directional labels and probabilistic predictions, with tests and metrics integrated into the supervised modeling layer.

Purpose: Connect regimes and causal features to actionable, up/flat/down-style expectations for ETFs and simple portfolios over the next quarter, evaluated with honest time-series metrics.
Output: Behavior label construction helpers, `train_forward_behavior_models` implementations, and extended reporting/tests that cover MODEL-03 and its contribution to MODEL-04.
</objective>

<execution_context>
@.planning/PROJECT.md
@.planning/REQUIREMENTS.md
@.planning/ROADMAP.md
@.planning/phases/03-supervised-regime-behavior-models/03-RESEARCH.md
@.planning/phases/03-supervised-regime-behavior-models/03-supervised-regime-behavior-models-01-SUMMARY.md
@.planning/phases/03-supervised-regime-behavior-models/03-supervised-regime-behavior-models-02-SUMMARY.md
</execution_context>

<context>
@CLAUDE.md
@src/market_regime/__init__.py
@src/market_regime/io/checkpoints.py
@src/market_regime/regime/profiler.py

<interfaces>
<!-- Behavior-related interfaces to implement on top of the classifier scaffold. -->

From src/market_regime/prediction/classifier.py:
```python
def make_behavior_labels(
    returns: pd.Series | pd.DataFrame,
    horizon: int,
    up_threshold: float,
    down_threshold: float,
) -> pd.Series: ...

def train_forward_behavior_models(
    features: pd.DataFrame,
    regimes: pd.Series,
    returns: pd.DataFrame,
    horizons: list[int],
) -> dict: ...
```
</interfaces>
</context>

<tasks>

<task type="auto">
  <name>Task 1: Implement behavior label construction helpers</name>
  <files>src/market_regime/prediction/classifier.py, tests/test_models_behavior.py</files>
  <action>
    - Add a `make_behavior_labels` helper that:
      - Accepts a quarterly returns Series for a single ETF (or a column of a DataFrame), a forward horizon `h`, and thresholds `up_threshold` and `down_threshold` expressed as decimal returns (e.g. `0.02` for +2%).
      - Constructs labels aligned with features at time \(t\) by shifting returns with `returns.shift(-h)` and:
        - Assigning `"up"` when `r_{t+h} &gt;= up_threshold`.
        - Assigning `"down"` when `r_{t+h} &lt;= down_threshold`.
        - Assigning `"flat"` otherwise.
      - Drops rows with missing future returns (trailing `h` quarters).
      - Optionally supports vectorized operation over a returns DataFrame by returning a dict of label Series keyed by column name.
    - Extend `tests/test_models_behavior.py` to:
      - Create small synthetic returns Series with known values and assert that:
        - Labels are correctly assigned for up/flat/down categories given specified thresholds.
        - The last `h` periods are excluded from the labeled index.
        - No labels are produced using features from the future relative to prediction time.
  </action>
  <verify>
    <automated>pytest -q -k "models_behavior and make_behavior_labels"</automated>
  </verify>
  <done>
    - `make_behavior_labels` produces correctly aligned up/flat/down labels for ETF returns over horizon `h`.
    - Tests validate threshold behavior and exclusion of trailing periods with missing targets.
  </done>
</task>

<task type="auto">
  <name>Task 2: Implement forward behavior models for ETFs and simple portfolios</name>
  <files>src/market_regime/prediction/classifier.py, tests/test_models_behavior.py</files>
  <action>
    - Complete `train_forward_behavior_models` to:
      - Accept causal feature matrix `features`, regime labels `regimes`, a returns DataFrame (with columns per ETF or portfolio), and a list of horizons.
      - For each ETF/portfolio column and each horizon `h`:
        - Use `make_behavior_labels` to construct directional labels.
        - Join labels with features and regimes on index intersections, discarding rows without labels.
        - Train a simple classifier (e.g. `RandomForestClassifier(max_depth=8, n_estimators=100, class_weight="balanced")`) with `TimeSeriesSplit` CV over time.
        - Store per-horizon, per-asset model objects and CV reports in a nested results dict.
      - Ensure outputs include:
        - `"models"`: nested mapping `{asset_name: {horizon: estimator}}`.
        - `"cv_reports"`: nested CV metrics dicts for MODEL-03 and MODEL-04.
        - Optional `"label_mapping"` documenting which string labels map to which classes.
    - Enhance `tests/test_models_behavior.py` to:
      - Build synthetic features, regimes, and returns for 2–3 ETFs over at least 20 quarters.
      - Verify that for a chosen ETF and horizon:
        - Models are trained and can output `predict_proba` for each behavior class.
        - Probability vectors sum to 1 and include all up/flat/down classes present in the data.
  </action>
  <verify>
    <automated>pytest -q -k "models_behavior and forward_behavior"</automated>
  </verify>
  <done>
    - `train_forward_behavior_models` returns trained directional models and CV metrics for at least one ETF and horizon on synthetic data.
    - Behavior tests guard against misaligned labels and missing probability classes.
  </done>
</task>

<task type="auto">
  <name>Task 3: Extend reporting to include behavior models alongside regime models</name>
  <files>src/market_regime/prediction/classifier.py, tests/test_models_reporting.py</files>
  <action>
    - Extend `model_metrics_summary` (or add a thin wrapper) so it can:
      - Accept combined results from regime and behavior training functions.
      - Tag metrics with model family (regime vs behavior), asset/portfolio name, and horizon.
      - Produce compact summaries suitable for dashboards and later recommendation logic (e.g. per-ETF confusion-style metrics for up/flat/down).
    - Update `tests/test_models_reporting.py` to:
      - Include fake behavior-model CV report data and verify:
        - Behavior entries are surfaced alongside regime entries without collisions in keys.
        - Metrics can be filtered by asset name and class (e.g. `("ETF1", "up")`).
  </action>
  <verify>
    <automated>pytest -q -k "models_reporting and behavior"</automated>
  </verify>
  <done>
    - Behavior model metrics are summarized with the same time-series aware rigor as regime models.
    - Reporting tests confirm that regime and behavior metrics can be consumed together by downstream phases.
  </done>
</task>

</tasks>

<verification>
- Run `pytest -q -k "models_behavior or models_reporting"` and confirm all behavior-related tests pass.
- Optionally run a small local script that constructs synthetic features/regimes/returns and calls `train_forward_behavior_models` to inspect predicted behavior probabilities.
</verification>

<success_criteria>
- Directional behavior labels are constructed correctly from returns, and classifiers map current features/regimes to up/flat/down with time-series aware validation, satisfying MODEL-03.
- Behavior metrics are integrated into the shared reporting structure, extending MODEL-04 coverage to asset/portfolio behavior models.
</success_criteria>

<output>
After completion, ensure `.planning/phases/03-supervised-regime-behavior-models/03-supervised-regime-behavior-models-03-SUMMARY.md` is created by the executor to record behavior model implementations and tests.
</output>

