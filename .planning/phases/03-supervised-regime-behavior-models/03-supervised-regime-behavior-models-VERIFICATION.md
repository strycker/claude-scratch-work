---
phase: 03-supervised-regime-behavior-models
verified: 2026-03-16T00:00:00Z
status: human_needed
score: 3/4 must-haves verified
gaps:
  - truth: "Model-related tests exist and can be run in isolation for Phase 3 behaviors."
    status: partial
    reason: "Tests are implemented and aligned with the actual code layout, but the phase validation plan still points at non-existent test paths and marks Wave 0 as pending."
    artifacts:
      - path: ".planning/phases/03-supervised-regime-behavior-models/03-VALIDATION.md"
        issue: "References `tests/unit/test_classifier.py` and specific test names that do not exist; frontmatter has nyquist_compliant: false and wave_0_complete: false."
      - path: "tests/test_models_regime.py"
        issue: "Implements the regime tests that VALIDATION.md intends but under a different filename and without updating the validation matrix."
      - path: "tests/test_models_behavior.py"
        issue: "Implements behavior-model tests that are not wired into the current validation matrix."
      - path: "tests/test_models_reporting.py"
        issue: "Covers regime metrics aggregation only; behavior metrics summarisation is implemented separately in `market_regime.prediction` but not yet under test."
    missing:
      - "Update `03-VALIDATION.md` to reference `tests/test_models_regime.py`, `tests/test_models_behavior.py`, and `tests/test_models_reporting.py` with accurate test names and statuses, and flip nyquist_compliant to true once Wave 0 is explicitly mapped."
      - "Add at least one automated test that exercises the behavior-model metrics/reporting path (e.g. via `market_regime.prediction.model_metrics_summary`) to close the MODEL-04 Nyquist gap for behavior models."
  - truth: "Supervised model training never uses non-time-series-aware validation or forward-looking features."
    status: partial
    reason: "Implemented code consistently uses TimeSeriesSplit and causal feature checkpoints, but there is an explicit fallback to potentially non-causal features and no automated guard that prevents training when only non-causal features are available."
    artifacts:
      - path: "pipelines/05_predict.py"
        issue: "Prefers `features_supervised.parquet` (causal) but falls back silently—with only a warning—to `features.parquet` if the supervised file is missing, which may reintroduce leakage depending on how `features.parquet` was produced."
      - path: "src/market_regime/prediction/classifier.py"
        issue: "All CV helpers use `TimeSeriesSplit` with no shuffling and construct forward targets via `regimes.shift(-h)`, which is correct; there is no assertion that the features passed in are from the causal checkpoint."
      - path: "src/market_regime/prediction.py"
        issue: "Behavior-model helpers also use TimeSeriesSplit and shifted returns, but similarly assume the caller has supplied causal features."
    missing:
      - "Harden `pipelines/05_predict.py` so that supervised training fails fast (or is explicitly gated) when `features_supervised.parquet` is missing, instead of silently falling back to `features.parquet`."
      - "Add a small assertion or configuration check in the supervised training entry points (or in a thin wrapper) that they are being fed the causal supervised feature checkpoint, and cover this with a unit test."
human_verification:
  - test: "Inspect end-to-end supervised model performance and confusion-style summaries on real quarterly data."
    expected: "Current-regime and forward-regime models show plausible, stable accuracy/F1 across TimeSeriesSplit folds, and behavior models produce reasonable up/flat/down distributions for major ETFs and simple portfolios."
    why_human: "Judging whether numerical performance is 'good enough' for live use, and whether confusion reports and behavior distributions align with investor intuition, requires domain judgment beyond what automated tests provide."
---

# Phase 3: Supervised Regime & Behavior Models Verification Report

**Phase Goal:** Train, validate, and report on supervised models that turn regimes into real-time and forward-looking signals for regimes and ETF/portfolio behavior.
**Verified:** 2026-03-16T00:00:00Z  
**Status:** human_needed  
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth                                                                                                      | Status        | Evidence |
|---|------------------------------------------------------------------------------------------------------------|--------------|----------|
| 1 | There is a clear, centralized API for training current-regime, forward-regime, and behavior models using causal features and Phase 2 regime labels. | ✓ VERIFIED    | `src/market_regime/prediction/classifier.py` exposes `train_current_regime`, `train_forward_classifiers`, `train_forward_behavior_models`, and `model_metrics_summary`; `src/market_regime/prediction.py` exposes behavior helpers; `pipelines/05_predict.py` wires the current and forward regime helpers into step 5. |
| 2 | Model-related tests exist and can be run in isolation for Phase 3 behaviors.                              | ⚠️ PARTIAL    | `tests/test_models_regime.py`, `tests/test_models_behavior.py`, and `tests/test_models_reporting.py` provide synthetic, network-free tests, but `03-VALIDATION.md` still points at non-existent `tests/unit/test_classifier.py` tests and marks Wave 0 as pending. |
| 3 | Supervised model training never uses non-time-series-aware validation or forward-looking features.        | ⚠️ PARTIAL    | All supervised helpers use `TimeSeriesSplit` and construct forward targets via `shift(-h)`; step 5 prefers `features_supervised.parquet` but can still be run against `features.parquet` without an automated guard, leaving a residual leakage risk if the causal checkpoint is missing. |
| 4 | Forward regime and behavior models are wired into the prediction layer in a way that later phases can consume. | ✓ VERIFIED (regime), ⚠️ PARTIAL (behavior) | `pipelines/05_predict.py` consumes `train_current_regime` and `train_forward_classifiers` and persists model bundles under `outputs/models/`; behavior helpers live in `market_regime.prediction`/`prediction.classifier` with tests but are not yet invoked from a pipeline step, leaving wiring to later phases. |

**Score:** 3/4 must-haves verified (one truth partial on tests/validation, one partial on leakage safeguards).

### Required Artifacts

| Artifact                                | Expected                                                                                  | Status     | Details |
|-----------------------------------------|------------------------------------------------------------------------------------------|-----------|---------|
| `src/market_regime/prediction/classifier.py` | Centralized supervised regime and (classifier-based) behavior model helpers with TimeSeriesSplit CV and metric aggregation. | ✓ VERIFIED | File exists (~400+ lines); defines `_tscv_reports`, `FoldReport`, `train_current_regime`, `train_forward_classifiers`, `make_behavior_labels`, `train_forward_behavior_models`, and `model_metrics_summary`, all using walk-forward CV and shifted targets. |
| `src/market_regime/prediction.py`      | Higher-level supervised helpers, including behavior models and a generic metrics flattener. | ✓ VERIFIED | File exists and provides `make_behavior_labels`, `train_forward_behavior_models`, and a row-oriented `model_metrics_summary` used by behavior-focused tests. |
| `tests/test_models_regime.py`          | Tests for current and forward regime classifiers, including leakage/CV ordering checks and probability sanity checks. | ✓ VERIFIED | Synthetic tests assert TimeSeriesSplit ordering (train indices < test indices), presence of `FoldReport`, and well-formed probability outputs for current and forward regime bundles. |
| `tests/test_models_behavior.py`        | Tests for ETF/portfolio behavior label construction and directional behavior models.      | ✓ VERIFIED | Synthetic tests cover up/flat/down labelling semantics, dropping of trailing periods, and per-asset behavior models with probability-normalisation checks. |
| `tests/test_models_reporting.py`       | Tests for model metrics and reporting helpers.                                            | ✓ VERIFIED (regime), ⚠️ PARTIAL (behavior) | Exercises `classifier.model_metrics_summary` for current and forward regime bundles; behavior metrics summarisation in `prediction.model_metrics_summary` is implemented but currently untested. |

### Key Link Verification

| From                            | To                                         | Via / Pattern                                                       | Status   | Details |
|---------------------------------|--------------------------------------------|---------------------------------------------------------------------|----------|---------|
| `pipelines/05_predict.py`       | `src/market_regime/prediction/classifier.py` | `from market_regime.prediction.classifier import train_current_regime, train_forward_classifiers` | ✓ WIRED  | Step 5 uses the centralized helpers on causal features (when available) and persists model bundles to `outputs/models/`. |
| `tests/test_models_regime.py`   | `src/market_regime/prediction/classifier.py` | `from market_regime.prediction.classifier import FoldReport, train_current_regime, train_forward_classifiers` | ✓ WIRED  | Regime tests import and exercise the supervised helpers directly on synthetic data. |
| `tests/test_models_behavior.py` | `src/market_regime/prediction.py`          | `from market_regime.prediction import make_behavior_labels, train_forward_behavior_models` | ✓ WIRED  | Behavior tests exercise the behavior helpers exposed from the prediction module. |
| `tests/test_models_reporting.py`| `src/market_regime/prediction/classifier.py` | `from market_regime.prediction.classifier import model_metrics_summary` | ✓ WIRED  | Reporting tests cover regime metrics aggregation; behavior metrics flattener lives in `prediction.py` and is not yet under test. |

### Requirements Coverage (MODEL-01 – MODEL-04)

| Requirement | Description (short)                                  | Status | Evidence |
|------------|------------------------------------------------------|--------|----------|
| MODEL-01   | Current-regime classifier with causal features and time-series aware validation. | PASS   | `train_current_regime` uses `TimeSeriesSplit` via `_tscv_reports`; step 5 preferentially consumes `features_supervised.parquet`; `tests/test_models_regime.py` asserts temporal ordering and probability shape, and `pipelines/05_predict.py` demonstrates end-to-end integration and model persistence. |
| MODEL-02   | Forward regime transition models with probabilities for ≥1 quarter ahead. | PASS   | `train_forward_classifiers` builds shifted targets via `regimes.shift(-h)`, drops trailing quarters, and trains DT/RF bundles per horizon; `pipelines/05_predict.py` calls this helper for configured horizons and saves the resulting bundles; tests verify correct shifting, exclusion of trailing samples, and probability normalisation for `h=1`. |
| MODEL-03   | Forward ETF/portfolio behavior models (directional up/flat/down). | WARN   | `market_regime.prediction` and `prediction/classifier.py` both implement `make_behavior_labels` and `train_forward_behavior_models` using shifted returns and TimeSeriesSplit CV; `tests/test_models_behavior.py` validates label semantics and per-asset, per-horizon models on synthetic data. However, no pipeline step currently consumes these behavior models, and behavior metrics are not yet aggregated alongside regime metrics, leaving wiring/reporting to later phases. |
| MODEL-04   | Transparent, time-series aware evaluation and reporting for supervised models. | WARN   | Regime models: CV metrics are captured via `FoldReport` objects and aggregated into overall/per-class metrics by `classifier.model_metrics_summary`, with good unit coverage in `tests/test_models_reporting.py`. Behavior models: CV scores and basic summaries exist in `prediction.py` but are not yet surfaced through a shared reporting layer or covered by tests; `03-VALIDATION.md` still marks Nyquist compliance as false and uses outdated test paths. |

### Anti-Patterns and Nyquist Validation Gaps

| File                                       | Line(s) (approx) | Pattern / Issue                                           | Severity | Impact |
|--------------------------------------------|------------------|-----------------------------------------------------------|----------|--------|
| `.planning/phases/03-supervised-regime-behavior-models/03-VALIDATION.md` | frontmatter & table | Validation matrix references non-existent `tests/unit/test_classifier.py` and marks all Phase 3 tasks as pending; `nyquist_compliant: false`. | WARN     | Documentation and Nyquist tracking are out of sync with the implemented `tests/test_models_*.py`, making it harder to see test coverage at a glance. |
| `pipelines/05_predict.py`                 | feature load block | Silent (warning-only) fallback from `features_supervised.parquet` to `features.parquet`. | WARN     | Allows supervised training to proceed without the guaranteed-causal feature checkpoint, potentially reintroducing leakage if `features.parquet` is ever non-causal. |

### Human Verification Required

1. **Model performance sanity on real data**
   - **Test:** Run `pipelines/05_predict.py` against real checkpoints, inspect printed current-regime probabilities and any downstream plots/notebooks that use the supervised bundles.
   - **Expected:** Probabilities and predicted regimes should look plausible across history (no degenerate always-one-regime behaviour; CV metrics should be in a reasonable range for quarterly macro data).
   - **Why human:** Determining whether the achieved metrics are acceptable for your decision-making tolerance is a judgment call, not a binary pass/fail.

2. **Behavior model usefulness**
   - **Test:** On a sample of ETFs/portfolios and horizons, inspect behavior model up/flat/down predictions in a notebook (e.g. overlay against realised returns).
   - **Expected:** Predicted directional labels should correlate with realised returns better than chance and be stable enough to inform regime-aware allocations.
   - **Why human:** Assessing whether the directional signals are economically meaningful (vs. noisy but statistically non-zero) requires domain expertise.

### Gaps Summary

Phase 3’s supervised regime and behavior models are **functionally implemented and reasonably well tested**, and regime models are fully wired into the prediction pipeline with TimeSeriesSplit-based validation and metrics aggregation. The main gaps are **process and safety-oriented** rather than algorithmic: the validation document has not been updated to reflect the new `tests/test_models_*.py` layout or to mark Nyquist compliance, behavior-model metrics are not yet integrated into the shared reporting surface, and the supervised pipeline still permits a potentially non-causal feature fallback. Addressing these gaps will tighten leakage guarantees and bring the validation artefacts back in line with the actual implementation and tests.

---

_Verified: 2026-03-16T00:00:00Z_  
_Verifier: Claude (gsd-verifier)_

