# Phase 3: Supervised Regime & Behavior Models - Research

**Researched:** 2026-03-16  
**Domain:** Time-series supervised learning for market regimes and ETF/portfolio behavior  
**Confidence:** MEDIUM-HIGH

<user_constraints>
## User Constraints (from CONTEXT.md or global project docs)

### Locked Decisions
- Operate on an ETF-only universe for v1; no single stocks or direct crypto, only ETF wrappers (CONSTR-01).
- No intraday or auto-trading; cadence is weekly / quarterly, with recommendations and reports only (CONSTR-02).
- Regime definitions and clustering are established in Phase 2 and must be treated as the canonical unsupervised labels for this phase.
- Publication-lag handling and causal feature variants from Phase 1 must be respected; no look-ahead leakage in supervised training.

### Claude's Discretion
- Choice of specific model families within scikit-learn (e.g. RandomForest vs Gradient Boosting vs simple baselines) as long as they are time-series aware and interpretable.
- Exact way to encode regime labels and forward horizons (e.g. one-vs-rest per regime vs multi-class directly), provided outputs remain interpretable probabilities.
- Concrete thresholds and binning schemes for “up/flat/down” style ETF/portfolio directional labels.

### Deferred Ideas (OUT OF SCOPE)
- Any move beyond ETFs in the investable universe (single stocks, options, direct crypto).
- Intraday prediction or trade execution logic.
- Heavyweight deep-learning architectures; keep Phase 3 focused on classical, explainable models.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID       | Description                                                                                             | Research Support                                                                                           |
|----------|---------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------|
| MODEL-01 | Current-regime classifier using causal features and time-series aware validation.                      | Defines feature pipeline usage, target labels, scikit-learn stack, and TimeSeriesSplit-based evaluation. |
| MODEL-02 | Forward-horizon regime transition probability models (≥1 quarter ahead).                               | Specifies horizon encoding, target construction from regime sequences, and probabilistic model outputs.  |
| MODEL-03 | Forward-looking ETF/portfolio directional behavior models (e.g. up/flat/down next quarter).           | Outlines label construction from quarterly ETF/portfolio returns and suitable interpretable classifiers. |
| MODEL-04 | Transparent, time-series aware evaluation and reporting of all supervised models (train/test metrics). | Recommends metrics, backtesting protocol, and confusion-style summaries aligned with project constraints.|
</phase_requirements>

## Summary

Phase 3 turns the unsupervised regime labels and ETF returns into deployable, supervised signals: (a) “what regime are we in now?”, (b) “where are regimes likely to move next?”, and (c) “what does that imply for ETF and portfolio behavior over the next quarter (and beyond)?”. The implementation should treat the clustering output from Phase 2 and the causal feature variants from Phase 1 as fixed inputs and build a thin, testable modeling layer on top, using classical scikit-learn models with walk-forward validation and explicit publication-lag handling.

The standard stack is `pandas`/`numpy` for data manipulation, `scikit-learn` for classifiers and time-series cross-validation, and the existing `CheckpointManager` for persisting model artifacts under `outputs/models/`. Architecturally, the phase should expose high-level functions like `train_current_regime()`, `train_forward_regime_models()`, and `train_forward_behavior_models()` that operate on checkpointed feature and label DataFrames, return both fitted estimators and performance summaries, and write model + metrics checkpoints consumable by later phases (portfolio behavior, recommendations, and the weekly report).

**Primary recommendation:** Use scikit-learn classifiers (DecisionTree + RandomForest as the baseline ensemble) with `TimeSeriesSplit`-style walk-forward evaluation on causal features, wrapping them in thin training/evaluation helpers that output calibrated regime and directional behavior probabilities plus clear, per-class metrics.

## Standard Stack

### Core

| Library      | Version (approx) | Purpose                                                       | Why Standard                                                                                 |
|-------------|------------------|---------------------------------------------------------------|----------------------------------------------------------------------------------------------|
| `pandas`    | 2.x              | Time-series tabular manipulation for features/labels.        | Already used throughout the project; ideal for quarterly panels and label construction.     |
| `numpy`     | 1.26+            | Numeric operations, arrays, basic math for labels/metrics.   | Underpins pandas and scikit-learn; already in use for feature engineering.                  |
| `scikit-learn` | 1.3+         | Classifiers, `TimeSeriesSplit`, metrics, model persistence.  | Project already depends on it; well-suited for interpretable, non-DL time-series models.    |
| `joblib` or `pickle` (via sklearn) | builtin / dependency | Persisting trained estimators under `outputs/models/`. | Standard for serializing sklearn models; lightweight, no new infra needed.                  |

### Supporting

| Library     | Version (approx) | Purpose                                                 | When to Use                                                                 |
|------------|------------------|---------------------------------------------------------|-----------------------------------------------------------------------------|
| `matplotlib` / `seaborn` | existing | Visualizing confusion matrices, calibration, and time-series backtests. | For MODEL-04 plots and diagnostics, integrated with existing plotting code. |
| `scipy`    | existing         | Metrics / utilities if needed (e.g. calibration), but mostly already used. | Only if sklearn wrappers are insufficient; keep usage minimal.             |

### Alternatives Considered

| Instead of                             | Could Use                  | Tradeoff                                                                                           |
|----------------------------------------|----------------------------|----------------------------------------------------------------------------------------------------|
| Pure RandomForest/DecisionTree models  | Gradient Boosting / XGBoost / LightGBM | Higher raw accuracy possible but more complex, harder to interpret, and adds dependencies.       |
| Custom CV/backtest loops               | `TimeSeriesSplit` + simple walk-forward wrappers | Hand-rolled loops are easy to get subtly wrong; `TimeSeriesSplit` encodes correct ordering.      |
| Deep learning (RNNs, Transformers)     | Remain with classical ML   | Overkill for quarterly data and complicates interpretability + infra; could be a future phase.   |

**Installation (already covered by project):**

```bash
pip install -e ".[dev]"
```

## Architecture Patterns

### Recommended Project Structure (Phase 3–relevant)

```text
src/market_regime/
├── prediction/
│   └── classifier.py        # Current & forward regime + behavior model training/eval API
├── regime/
│   └── profiler.py          # Regime labels, profiles, transition matrices (Phase 2)
├── assets/
│   └── returns.py           # Quarterly ETF returns, behavior summaries (Phase 4 focus)
├── io/
│   └── checkpoints.py       # CheckpointManager for features, labels, and models
└── ...
```

Even if `prediction/classifier.py` or `assets/returns.py` are missing or partially implemented in this workspace snapshot, Phase 3 should target this structure, matching `CLAUDE.md` and the legacy modular design (where `supervised.py` and `asset_returns.py` play analogous roles).

### Pattern 1: Current-Regime Classifier

**What:** Train a multi-class classifier that maps causal features for quarter \(t\) to the regime label (e.g. `balanced_cluster` or named regime) for quarter \(t\).

**When to use:** Always, to satisfy MODEL-01. This is the canonical “current regime from today’s data” signal.

**Shape:**
- **Inputs:** Causal feature matrix \(X_t\) at quarterly frequency, one row per quarter in the historical window, excluding any features that depend on future information (publication lags already enforced in ingestion).
- **Target:** Regime label \(y_t\) from Phase 2 (balanced or final named regimes), aligned by quarter.
- **Model:** Start with a `DecisionTreeClassifier` for interpretability, then a `RandomForestClassifier` baseline; keep both for diagnostics.
- **Validation:** Use `TimeSeriesSplit` or equivalent walk-forward splits (e.g. expanding window: train on early quarters, test on later ones).

Conceptual API:

```python
def train_current_regime(
    features: pd.DataFrame,
    labels: pd.Series,
    cv_splits: int = 5,
) -> dict:
    """
    Train current-regime classifiers with walk-forward validation.

    Returns:
        {
          "models": {"rf": rf_model, "dt": dt_model},
          "cv_scores": {...},   # per-split metrics
          "holdout_scores": {...},
        }
    """
```

### Pattern 2: Forward Regime Transition Models

**What:** For each forward horizon \(h \in \{1,2,4,8\}\) quarters (config-driven), train models that estimate \(P(\text{regime}_{t+h} = j \mid X_t, \text{regime}_t)\).

**When to use:** To satisfy MODEL-02 and provide multi-step transition probabilities for downstream portfolio and recommendation logic.

**Shape:**
- **Inputs:** Same causal features \(X_t\) plus possibly one-hot or integer encoding of \(\text{regime}_t\).
- **Targets:** Either:
  - Multi-class regime labels \(y_{t+h} \in \{0,\dots,k-1\}\) directly, or
  - Binary labels per “target regime” (one-vs-rest) if you need sharper per-regime signals.
- **Models:** RandomForestClassifier (for probabilities + feature importances) and a shallow DecisionTreeClassifier (for explanatory tree diagrams if desired).
- **Validation:** Same time-series CV procedure, but ensure the last \(h\) quarters of data (which lack \(y_{t+h}\)) are excluded from training and evaluation.

Conceptual API:

```python
def train_forward_classifiers(
    features: pd.DataFrame,
    regimes: pd.Series,
    horizons: list[int],
) -> dict[int, dict]:
    """
    For each horizon h, construct shifted targets y_{t+h} and train classifiers.

    Returns dict[horizon] = {"models": {...}, "cv_scores": {...}, "holdout_scores": {...}}.
    """
```

### Pattern 3: Forward ETF / Portfolio Behavior Models

**What:** Supervised models that map current features and regimes to directional outcomes (“up/flat/down” or similar) of ETF or portfolio returns over the next quarter (or horizon \(h\)).

**When to use:** To satisfy MODEL-03 and expose stoplight-style behavior summaries that later phases can connect to recommendations.

**Shape:**
- **Targets:**
  - Define quarterly ETF returns \(r_{t+1}\) from the returns pipeline.
  - Bin them into categories, e.g.:
    - `up`: \(r_{t+1} \geq +X\%\)
    - `down`: \(r_{t+1} \leq -Y\%\)
    - `flat`: otherwise
  - Alternatively, for portfolios, use weighted combinations of ETF returns.
- **Inputs:** The same causal features \(X_t\) plus possibly regime one-hot indicators; each ETF or portfolio can have its own model, or you can include an ETF identifier in the feature space if designed carefully.
- **Models:** Small RandomForest/DecisionTree classifiers per ETF or portfolio template; keep models simple and well-regularized to avoid overfitting noisy returns.

Conceptual API:

```python
def train_forward_behavior_models(
    features: pd.DataFrame,
    regimes: pd.Series,
    returns: pd.DataFrame,
    horizons: list[int],
) -> dict:
    """
    Train directional (up/flat/down) classifiers for each ETF / portfolio template.
    """
```

### Anti-Patterns to Avoid

- **Mixing non-causal and causal features:** Do not train supervised models on any feature representation that includes forward-looking information relative to the prediction time (e.g. unrevised GDP, unshifted FRED series). Always use the causal/shifted variants and respect `RunConfig` flags and settings.
- **Random train/test splits:** Never use random or shuffling-based CV for these time-series models; it will leak future information into the training set.
- **Overfitting with too many complex models:** Avoid stacking many high-capacity models; prioritize a small, interpretable set (DecisionTree + RandomForest) with honest time-based validation.

## Don't Hand-Roll

| Problem                                      | Don't Build                                           | Use Instead                                     | Why                                                                                  |
|---------------------------------------------|-------------------------------------------------------|-------------------------------------------------|--------------------------------------------------------------------------------------|
| Time-series cross-validation                | Custom sliding/expanding CV loops                     | `sklearn.model_selection.TimeSeriesSplit`       | Encodes ordering constraints correctly; reduces subtle leakage bugs.                |
| Classification metrics & reports            | Manual confusion matrices / precision/recall calcs    | `sklearn.metrics.classification_report`, confusion matrix helpers | Battle-tested implementations; easier to keep consistent across models. |
| Model persistence                           | Ad-hoc pickle paths, manual version naming            | Project `CheckpointManager` + sklearn `joblib` or `.pkl` models in `outputs/models/` | Keeps artifacts discoverable, versioned, and consistent with rest of pipeline.      |
| Probability calibration                     | Custom sigmoid or Platt scaling code                  | `sklearn.calibration.CalibratedClassifierCV` if needed | Properly implemented calibration is non-trivial; use established tools if calibration is required. |

**Key insight:** The hard part here is *framing* the prediction problems (targets, horizons, causal features, validation) and wiring them into the existing checkpoint and reporting stack — not inventing new CV, metrics, or model-storage mechanisms.

## Common Pitfalls

### Pitfall 1: Label Leakage via Target Construction

**What goes wrong:** When building forward labels (e.g. regime at \(t+1\) or ETF returns at \(t+1\)), it is easy to accidentally align features from \(t+1\) instead of \(t\), or to include features that rely on unreleased data (e.g. unshifted GDP).

**Why it happens:** Pandas `shift` and reindex operations are error-prone; publication lag rules are subtle and already encoded at ingestion, but can be accidentally bypassed if raw columns are used.

**How to avoid:**
- Always construct targets using explicit `shift(-h)` on the *label/returns series*, not the features.
- Use only the causal feature parquet/checkpoint from Phase 1 intended for supervised tasks.
- Add assertions that the last \(h\) quarters (which lack valid targets) are excluded from model training and evaluation.

**Warning signs:** Unrealistically high out-of-sample metrics, particularly on the most recent quarters; models relying heavily on macro series known to be slow-published.

### Pitfall 2: Non Time-Series-Aware Cross-Validation

**What goes wrong:** Using `KFold(shuffle=True)` or train/test splits that mix early and late quarters destroys temporal ordering and inflates performance.

**Why it happens:** Generic sklearn examples default to IID assumptions; copying them directly into a time-series project introduces hidden leakage.

**How to avoid:**
- Use `TimeSeriesSplit` or custom walk-forward splits only.
- Keep validation logic centralized in helper functions (e.g. `_tscv_scores()` style) rather than reimplemented per model.

**Warning signs:** Train/Test metrics that are nearly identical, or better test performance than training on later regimes with very small sample sizes.

### Pitfall 3: Over-Complex Models on Small Samples

**What goes wrong:** Quarterly data (~300 quarters) per regime/horizon is limited; deep trees or large forests with many features can overfit easily.

**Why it happens:** Copying default hyperparameters from generic tabular classification examples without adjusting for sample size and temporal dependence.

**How to avoid:**
- Cap tree depth (e.g. `max_depth=8` as per project notes).
- Restrict features to a curated subset when training supervised models; consider dropping very noisy derivatives or strongly collinear variables.
- Prefer simple baselines (logistic regression or shallow trees) as reference models and check ensembles against them.

**Warning signs:** Huge gaps between training and validation performance; wildly fluctuating feature importances across CV folds.

### Pitfall 4: Uninterpretable Outputs

**What goes wrong:** Models emit probabilities or logits that are hard to relate to regimes or behavior categories, making it difficult to trust recommendations later.

**Why it happens:** Focusing solely on global accuracy metrics and ignoring per-class breakdowns or confusion matrices.

**How to avoid:**
- Always produce per-regime and per-direction-class metrics (precision, recall, F1, support).
- Keep a DecisionTree baseline and visualize its structure (even if not used in production) to sanity-check splits and important variables.
- Store and surface confusion matrices and calibration plots consumed later by Phase 5/6 reporting.

## Code Examples (Conceptual)

### Current-Regime Classifier with TimeSeriesSplit

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report

def ts_cv_scores(model, X, y, n_splits: int = 5) -> dict:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    reports = []
    for train_idx, test_idx in tscv.split(X):
        model_ = model.__class__(**model.get_params())
        model_.fit(X.iloc[train_idx], y.iloc[train_idx])
        y_pred = model_.predict(X.iloc[test_idx])
        reports.append(classification_report(y.iloc[test_idx], y_pred, output_dict=True))
    return {"reports": reports}
```

### Forward-Horizon Target Construction

```python
def make_forward_labels(regimes: pd.Series, horizon: int) -> pd.Series:
    # y_{t+h} aligned with features at time t
    return regimes.shift(-horizon).dropna()
```

## State of the Art (Within This Project)

| Old Approach                               | Current Approach                                  | When Changed      | Impact                                                                 |
|-------------------------------------------|---------------------------------------------------|-------------------|------------------------------------------------------------------------|
| Ad-hoc scripts (`legacy/supervised.py`)   | Modular `src/market_regime/prediction/classifier.py` (planned/aligned) | V1 refactor (design) | Centralizes model training, CV, and persistence; easier to test.      |
| Simple train/test split without CV        | `TimeSeriesSplit` walk-forward CV                | V1 design         | More realistic performance estimates for regime shifts.                |
| Implicit regime transitions               | Explicit forward-horizon classifiers + transition matrices | V1 design with Phases 2–3 | Makes transition probabilities first-class artifacts.                  |

**Deprecated/outdated (for Phase 3 design):**
- Relying purely on unsupervised regimes without current / forward supervised signals.
- Any cross-validation method that shuffles or mixes future quarters into training folds.

## Validation Architecture

Nyquist-style validation is **enabled** (`nyquist_validation: true` in `.planning/config.json`), so Phase 3 must plan for automated tests around supervised models.

### Test Framework

| Property        | Value                          |
|-----------------|--------------------------------|
| Framework       | `pytest` (project standard)    |
| Config file     | `pyproject.toml` (pytest section) |
| Quick run cmd   | `pytest tests -k "model or regime" -q` (approximate; refine in planning) |
| Full suite cmd  | `pytest -v`                    |

Given the current workspace snapshot does not expose specific model test files, assume we will create dedicated tests for Phase 3 behavior.

### Phase Requirements → Test Map (Planned)

| Req ID   | Behavior                                                             | Test Type   | Automated Command (sketch)                                             | File Exists? |
|----------|----------------------------------------------------------------------|------------|-------------------------------------------------------------------------|-------------|
| MODEL-01 | Current-regime classifier trains and predicts without leakage.      | unit/integration | `pytest tests/test_models_regime.py::test_current_regime_classifier`    | ❌ Wave 0   |
| MODEL-02 | Forward regime transition models emit valid probability vectors.    | unit/integration | `pytest tests/test_models_regime.py::test_forward_regime_transitions`   | ❌ Wave 0   |
| MODEL-03 | ETF/portfolio behavior models produce up/flat/down labels correctly.| unit        | `pytest tests/test_models_behavior.py::test_behavior_labels_and_models` | ❌ Wave 0   |
| MODEL-04 | Metrics and reports are time-series aware and correctly summarized. | unit/integration | `pytest tests/test_models_reporting.py::test_model_metrics_reporting`   | ❌ Wave 0   |

### Sampling Rate

- **Per implementation commit (Phase 3 files):** Run focused tests, e.g. `pytest tests/test_models_regime.py -q`.
- **Per wave / branch merge:** Run the full model-related subset, e.g. `pytest tests -k "model or regime or behavior"`.
- **Phase 3 gate:** Full project test suite green (`pytest -v`) with stable metrics artifacts regenerated successfully.

### Wave 0 Gaps

- [ ] Create `tests/test_models_regime.py` to cover current-regime and forward-regime classifiers (MODEL-01, MODEL-02).
- [ ] Create `tests/test_models_behavior.py` to cover ETF/portfolio behavior label construction and models (MODEL-03).
- [ ] Create `tests/test_models_reporting.py` to verify metrics/summary generation and guard against non-time-series CV usage (MODEL-04).
- [ ] Add lightweight fixtures for sample feature/label DataFrames derived from small checkpoint slices (no network access).

## Sources

### Primary (HIGH confidence)

- `CLAUDE.md` — project-wide architecture, stack, and remaining gaps (noting desired classifier module, TimeSeriesSplit, and DecisionTree/RandomForest use).
- `legacy/unified_script.py` — end-to-end unsupervised pipeline and regime feature engineering; confirms PCA, clustering, and data preparation patterns this phase must sit on.
- `ROADMAP.md` and `REQUIREMENTS.md` — authoritative definitions of MODEL-01–MODEL-04 and their success criteria.

### Secondary (MEDIUM confidence)

- Implicit design from `regime/profiler.py` (transition matrix construction) regarding how regime sequences are handled and how empirical transitions are used alongside supervised models.

### Tertiary (LOW confidence)

- Assumptions about missing `src/market_regime/prediction/classifier.py` and `assets/returns.py` in this snapshot; design here follows documented intent but may require adjustment once those files are fully visible.

## Metadata

**Confidence breakdown:**

| Area             | Level       | Reason                                                                                  |
|------------------|------------|-----------------------------------------------------------------------------------------|
| Standard stack   | HIGH       | Fully aligned with project docs; scikit-learn and pandas are already core dependencies.|
| Architecture     | MEDIUM-HIGH| Based on documented modular design; minor uncertainty about exact current code layout. |
| Pitfalls         | MEDIUM     | Grounded in general time-series ML practice and project constraints, not code inspection.|

**Research date:** 2026-03-16  
**Valid until:** 2026-04-15 (Phase 3 planning should revisit if stack or requirements change materially)

## RESEARCH COMPLETE

**Phase:** 3 - Supervised Regime & Behavior Models  
**Confidence:** MEDIUM-HIGH

### Key Findings

- Phase 3 should expose a small set of supervised training APIs (current regime, forward regimes, and ETF/portfolio behavior) that operate on causal features and fixed regime labels from earlier phases.
- `scikit-learn` with `TimeSeriesSplit` and simple classifiers (DecisionTree + RandomForest) provides a solid, interpretable baseline stack that matches project conventions.
- Correct target construction (shifted regimes and returns) and strict time-series validation are critical to avoid leakage and unrealistic performance.
- Model outputs must include interpretable probabilities and confusion-style metrics so Phase 5/6 can judge whether recommendations are trustworthy.

### File Created

`.planning/phases/03-supervised-regime-behavior-models/03-RESEARCH.md`

### Confidence Assessment

| Area           | Level       | Reason                                                                                   |
|----------------|------------|------------------------------------------------------------------------------------------|
| Standard Stack | HIGH       | Directly supported by `CLAUDE.md` and existing dependencies.                            |
| Architecture   | MEDIUM-HIGH| Consistent with project-level modularization; minor uncertainty about current src state.|
| Pitfalls       | MEDIUM     | Based on domain best practices plus project constraints, not exhaustive code analysis.  |

### Open Questions

- Exact current implementation and tests for `prediction/classifier.py` and ETF returns modules may require minor adjustments to the proposed APIs.
- Specific thresholds and binning for ETF/portfolio directional labels (“up/flat/down”) should be tuned empirically during planning.

### Ready for Planning

Research complete. Planner can now create detailed PLAN.md files for Phase 3 tasks.

