---
phase: 03-supervised-regime-behavior-models
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - src/market_regime/prediction/classifier.py
  - tests/test_models_regime.py
  - tests/test_models_behavior.py
  - tests/test_models_reporting.py
autonomous: true
requirements:
  - MODEL-01
  - MODEL-02
  - MODEL-03
  - MODEL-04
user_setup: []
must_haves:
  truths:
    - "There is a clear, centralized API for training current-regime, forward-regime, and behavior models using causal features and Phase 2 regime labels."
    - "Model-related tests exist and can be run in isolation for Phase 3 behaviors."
    - "Supervised model training never uses non-time-series-aware validation or forward-looking features."
  artifacts:
    - path: "src/market_regime/prediction/classifier.py"
      provides: "Training and evaluation entry points for current and forward regime/behavior models."
      min_lines: 120
    - path: "tests/test_models_regime.py"
      provides: "Tests for current and forward regime classifiers, including leakage and CV checks."
      min_lines: 80
    - path: "tests/test_models_behavior.py"
      provides: "Tests for ETF/portfolio behavior label construction and directional models."
      min_lines: 60
    - path: "tests/test_models_reporting.py"
      provides: "Tests for model metrics, reporting helpers, and time-series aware evaluation."
      min_lines: 60
  key_links:
    - from: "tests/test_models_regime.py"
      to: "src/market_regime/prediction/classifier.py"
      via: "import train_current_regime and train_forward_classifiers APIs"
      pattern: "from market_regime\\.prediction\\.classifier import"
    - from: "tests/test_models_behavior.py"
      to: "src/market_regime/prediction/classifier.py"
      via: "import train_forward_behavior_models or equivalent behavior helpers"
      pattern: "train_forward_behavior_models"
    - from: "tests/test_models_reporting.py"
      to: "src/market_regime/prediction/classifier.py"
      via: "import model metrics/reporting helpers"
      pattern: "model_metrics"
---

<objective>
Bootstrap the supervised modeling layer with a centralized classifier module and dedicated test files for regimes, behavior, and reporting so later plans can focus on richer modeling logic rather than scaffolding.

Purpose: Establish clear, testable APIs and fixtures for current-regime, forward-regime, and behavior models that respect causal features and time-series validation.
Output: A new `prediction/classifier.py` module with stubbed training functions and three pytest modules covering regime models, behavior labels/models, and reporting/metrics.
</objective>

<execution_context>
@.planning/PROJECT.md
@.planning/REQUIREMENTS.md
@.planning/ROADMAP.md
@.planning/phases/03-supervised-regime-behavior-models/03-RESEARCH.md
</execution_context>

<context>
@CLAUDE.md
@src/market_regime/__init__.py
@src/market_regime/io/checkpoints.py
@src/market_regime/regime/profiler.py

<interfaces>
<!-- Key interfaces this plan should introduce for downstream plans. Exact signatures can be refined in implementation but must follow this shape. -->

From src/market_regime/prediction/classifier.py (to be created):
```python
def train_current_regime(features, labels, cv_splits: int = 5) -> dict: ...

def train_forward_classifiers(features, regimes, horizons) -> dict[int, dict]: ...

def train_forward_behavior_models(features, regimes, returns, horizons) -> dict: ...

def model_metrics_summary(results: dict) -> dict: ...
```
</interfaces>
</context>

<tasks>

<task type="auto">
  <name>Task 1: Create classifier module scaffold with supervised training APIs</name>
  <files>src/market_regime/prediction/classifier.py</files>
  <action>
    - Create a new `classifier.py` module under `src/market_regime/prediction/` that defines:
      - `train_current_regime(features: pd.DataFrame, labels: pd.Series, cv_splits: int = 5) -> dict`
      - `train_forward_classifiers(features: pd.DataFrame, regimes: pd.Series, horizons: list[int]) -> dict[int, dict]`
      - `train_forward_behavior_models(features: pd.DataFrame, regimes: pd.Series, returns: pd.DataFrame, horizons: list[int]) -> dict`
      - An internal helper for time-series CV (e.g. `_tscv_scores(model, X, y, n_splits: int) -> dict`) using `sklearn.model_selection.TimeSeriesSplit`.
      - A small `model_metrics_summary(results: dict) -> dict` helper that standardizes metric keys (accuracy, macro/micro F1, per-class metrics) for MODEL-04.
    - In this plan, keep implementations minimal but correct:
      - Use simple `DecisionTreeClassifier(max_depth=8)` and `RandomForestClassifier(max_depth=12, n_estimators=100, random_state=42)` as baselines.
      - For forward models, construct shifted targets with `regimes.shift(-h)` or returns shifted similarly, dropping rows without valid targets.
      - Centralize all TimeSeriesSplit usage in `_tscv_scores` to avoid duplication in later plans.
    - Include docstrings that clearly describe:
      - Inputs (must be causal feature sets produced by Phase 1, with publication lags honored).
      - Outputs (trained models, CV metrics, and optional holdout metrics).
    - Wire basic logging via the project logging conventions (use `logging.getLogger(__name__)`) instead of `print`.
  </action>
  <verify>
    <automated>pytest -q -k "import and has train_current_regime and train_forward_classifiers" || python -c "from market_regime.prediction.classifier import train_current_regime, train_forward_classifiers, train_forward_behavior_models, model_metrics_summary; print('ok')"</automated>
  </verify>
  <done>
    - `src/market_regime/prediction/classifier.py` exists and can be imported without errors.
    - The four core functions are present with the expected signatures and basic docstrings.
    - `_tscv_scores` uses `TimeSeriesSplit` and never shuffles data.
  </done>
</task>

<task type="auto">
  <name>Task 2: Create regime and behavior model test modules with initial fixtures</name>
  <files>tests/test_models_regime.py, tests/test_models_behavior.py</files>
  <action>
    - Add `tests/test_models_regime.py` that:
      - Imports `train_current_regime` and `train_forward_classifiers` from `market_regime.prediction.classifier`.
      - Builds a small synthetic quarterly DataFrame of causal-style features and integer regime labels (e.g. 5–10 quarters, 3 regimes) with clear ordering.
      - Asserts that:
        - `train_current_regime` returns a dict containing `"models"` and `"cv_scores"` keys.
        - Each model in `"models"` has a `predict` method and can be fit on the synthetic data without raising.
        - CV splits respect temporal ordering (e.g. by checking that the max train index is always less than the min test index).
      - Includes at least one test that would fail if a non-time-series CV (e.g. shuffled KFold) were used.
    - Add `tests/test_models_behavior.py` that:
      - Imports `train_forward_behavior_models`.
      - Builds synthetic ETF returns for 1–3 ETFs across 8–12 quarters and a corresponding regime series.
      - Demonstrates a simple label binning scheme for up/flat/down or similar directional categories and ensures the helper correctly handles missing final horizons.
      - Asserts that models can be trained on this synthetic data and return probability outputs for each class without errors.
  </action>
  <verify>
    <automated>pytest -q -k "models_regime or models_behavior"</automated>
  </verify>
  <done>
    - `tests/test_models_regime.py` and `tests/test_models_behavior.py` exist and run without import errors.
    - At least one test guards against non-time-series-aware CV usage.
    - Synthetic fixtures allow fast, network-free execution.
  </done>
</task>

<task type="auto">
  <name>Task 3: Create reporting and metrics tests for supervised models</name>
  <files>tests/test_models_reporting.py</files>
  <action>
    - Create `tests/test_models_reporting.py` that:
      - Imports `model_metrics_summary` (and any related reporting helpers) from `market_regime.prediction.classifier`.
      - Constructs a small fake `results` dict shaped like the output of `train_current_regime` / `train_forward_classifiers` (e.g. containing per-split classification reports from `sklearn.metrics.classification_report`).
      - Asserts that `model_metrics_summary`:
        - Returns a dict with top-level keys for each model/horizon and nested keys for accuracy, macro-F1, and per-class metrics.
        - Does not require access to real checkpoints or network resources.
      - Optionally includes a smoke test that integrates with `pytest` markers to allow future selective runs (e.g. `@pytest.mark.models`).
  </action>
  <verify>
    <automated>pytest -q -k "models_reporting"</automated>
  </verify>
  <done>
    - `tests/test_models_reporting.py` exists and can be run independently.
    - `model_metrics_summary` is exercised with representative input and produces stable, inspectable output.
  </done>
</task>

</tasks>

<verification>
- Run `pytest -q -k "models_regime or models_behavior or models_reporting"` and confirm all tests pass on synthetic data.
- Confirm `python -c "from market_regime.prediction.classifier import train_current_regime, train_forward_classifiers, train_forward_behavior_models, model_metrics_summary; print('ok')"` runs without error.
</verification>

<success_criteria>
- Classifier module and test files exist and import cleanly, establishing the API surface for Phase 3 supervised models.
- Time-series-aware CV is centralized and guarded by tests to prevent leakage regressions.
- MODEL-01–MODEL-04 have initial, traceable test coverage hooks ready for richer implementations in subsequent plans.
</success_criteria>

<output>
After completion, ensure `.planning/phases/03-supervised-regime-behavior-models/03-supervised-regime-behavior-models-01-SUMMARY.md` is created by the executor to record what was implemented and which tests were added.
</output>

