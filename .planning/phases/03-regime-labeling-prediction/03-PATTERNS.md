# Phase 3: Regime Labeling & Prediction - Pattern Map

**Mapped:** 2026-07-22
**Files analyzed:** 9 (7 new modules + 2 test files; config edit not counted as a file)
**Analogs found:** 9 / 9

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|---|---|---|---|---|
| `src/trading_crab_lib/platform/labeling/__init__.py` | package init | — | `src/trading_crab_lib/platform/honesty/__init__.py` | exact (empty re-export init) |
| `src/trading_crab_lib/platform/labeling/jump_model.py` | service (fit/transform) | batch/transform | `src/trading_crab_lib/platform/honesty/cv.py` (module docstring/class style) + incumbent `clustering.py` (warm-start idiom, reference only) | role-match |
| `src/trading_crab_lib/platform/labeling/diagnostics.py` | service (report/CLI artifact) | batch + file-I/O | `src/trading_crab_lib/platform/honesty/gap_lag.py` | exact |
| `src/trading_crab_lib/platform/prediction/__init__.py` | package init | — | `src/trading_crab_lib/platform/honesty/__init__.py` | exact |
| `src/trading_crab_lib/platform/prediction/nowcaster.py` | service (fit/predict) | batch/transform | `src/trading_crab_lib/platform/honesty/walkforward.py` (clone-per-fold, registry logging) + `cv.py` (PurgedEmbargoedKFold usage) | role-match |
| `src/trading_crab_lib/platform/prediction/transition_matrix.py` | utility (pure transform) | transform | `src/trading_crab_lib/platform/honesty/gap_lag.py` (`compute_*` pure functions) | role-match |
| `config/platform_settings.yaml` (edit: add `labeling:` section) | config | — | existing `cv:`/`holdout:`/`registry:` sections in same file | exact |
| `tests/unit/test_platform_labeling.py` | test | invariant/property | `tests/unit/test_platform_walkforward.py` + `tests/unit/test_platform_cv.py` | exact |
| `tests/unit/test_platform_nowcaster.py` | test | invariant/property | `tests/unit/test_platform_cv.py` (parametrized leakage sweep) + `tests/unit/test_platform_walkforward.py` (registry) | exact |

## Pattern Assignments

### `src/trading_crab_lib/platform/labeling/jump_model.py` (service, batch/transform)

**Analog:** `src/trading_crab_lib/platform/honesty/cv.py` for module/docstring/class conventions; incumbent `src/trading_crab_lib/clustering.py` for the k-means warm-start idiom (**reference only — do not import**).

**Module docstring + imports pattern** (from `cv.py` lines 1-32):
```python
"""
<One-line summary> (REQ-ID, design §X.Y).

<Why this exists, what algorithm, what guarantee it provides>. <Any
"no silent default" or "deliberate" callouts, per project convention of
explaining non-obvious design choices in the docstring itself.>

Usage::

    from trading_crab_lib.platform.labeling.jump_model import fit_jump_model
    result = fit_jump_model(X, K=5, lam=52.0, n_restarts=10)
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
```

**Core pattern — RESEARCH.md's verified Patterns 1-4** (already vetted against installed sklearn 1.9.0/numpy 2.4.6 signatures in 03-RESEARCH.md — copy verbatim, do not re-derive):
- `decode_states_dp(d, lam)` — O(TK) vectorized DP with min/second-min trick (03-RESEARCH.md Pattern 1)
- `fit_jump_model(X, K, lam, *, n_restarts=10, max_iter=50, random_state=42)` — alternation + multi-restart, `KMeans(n_clusters=K, n_init=1, init="k-means++", random_state=random_state + r)` per restart (Pattern 2)
- `canonicalize_states(states, centroids, feature_names)` — economic sort on `trailing_return_1m` (Pattern 3)
- `soft_confidences(d)` — temperature-free softmax over `-d` (Pattern 4)
- Degenerate-empty-state freeze-on-empty guard inside the centroid recompute loop (see Pitfall 2 in RESEARCH.md) — mandatory, has a documented failure mode (NaN propagation)

**Keyword-only optional args convention** (CLAUDE.md Function Design): `n_restarts`, `max_iter`, `random_state` are keyword-only after `K`, `lam` — mirrors `PurgedEmbargoedKFold.__init__(self, n_splits=5, *, label_horizon, embargo)` in `cv.py` line 50.

**Config read pattern** (from `platform/config.py` + CONTEXT D-01/discretion — read defensively via `.get()`, do not extend `_REQUIRED_PLATFORM_SECTIONS`):
```python
labeling_cfg = cfg.get("labeling", {})
K = labeling_cfg.get("K", 5)
lam = labeling_cfg.get("lambda", 52.0)
n_restarts = labeling_cfg.get("n_restarts", 10)
embargo_months = labeling_cfg.get("embargo_months", 12)
```

**Lean feature pull** (RESEARCH.md "Code Examples" section, sourced from `taxonomy.py` + `checkpoints.py` + `config.py`):
```python
from trading_crab_lib.platform.checkpoints import get_platform_checkpoint_manager
from trading_crab_lib.platform.taxonomy import lean_feature_set
from trading_crab_lib.platform.config import load_platform_config

cfg = load_platform_config()
monthly_features = get_platform_checkpoint_manager().load("monthly_features")
lean_cols = sorted(lean_feature_set(cfg) & set(monthly_features.columns))
X_df = monthly_features[lean_cols].dropna()
```

**What NOT to do:** do not `import trading_crab_lib.clustering` (incumbent quarterly module) — CONTEXT/RESEARCH both explicitly forbid pulling incumbent code paths into `platform/`. Use `sklearn.cluster.KMeans` directly, same as the incumbent does, but as an independent import.

---

### `src/trading_crab_lib/platform/labeling/diagnostics.py` (service, batch + file-I/O)

**Analog:** `src/trading_crab_lib/platform/honesty/gap_lag.py` (full file — this is the closest structural analog in the entire codebase: pure `compute_*` functions + one `report_*` function that prints + persists a parquet artifact + a `__main__` synthetic self-check).

**Module docstring pattern** (`gap_lag.py` lines 1-19): explain what each function computes, cross-reference the design section, and note explicitly what data does/doesn't exist yet if relevant (not applicable here since Phase 3 has real inputs, unlike Phase 2's gap_lag stub-ahead pattern).

**Schema-stable artifact columns pattern** (`gap_lag.py` lines 33-36, 130-134):
```python
_ARTIFACT_COLUMNS = ["state", "occupancy_pct", "median_sojourn_months", "profile"]
# ...
rows = [...] if metrics else []
df = pd.DataFrame(rows)
if df.empty:
    df = pd.DataFrame(columns=_ARTIFACT_COLUMNS)
df.to_parquet(artifact_path, index=False)
```

**Report-only WARNING-on-violation pattern** (D-02: labeler always completes; violations logged loudly, never gate/raise):
```python
log = logging.getLogger(__name__)
if occupancy_pct < min_occupancy_threshold:
    log.warning("State %d occupancy %.1f%% below §4.4 sanity threshold — report-only, not blocking (D-02)", state, occupancy_pct)
```

**Persist-and-print pattern** (`gap_lag.py::report_gap_lag` lines 109-146):
```python
def report_labeling_diagnostics(metrics: dict, *, output_dir: Path | None = None) -> Path:
    target_dir = Path(output_dir) if output_dir is not None else OUTPUT_DIR / "reports" / "model_metrics"
    target_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = target_dir / "labeling_diagnostics.parquet"
    ...
    log.info(summary)
    print(summary)  # noqa: T201 — first-class CLI run output, not debug noise
    return artifact_path
```

**Load-before-save churn pattern** (RESEARCH.md Pitfall 3 — NOT in any existing file verbatim, but the `CheckpointManager.load()` → `FileNotFoundError` contract is documented in `platform/checkpoints.py` and the incumbent `CLAUDE.md` "Error Handling" section: "Checkpoint misses: `CheckpointManager.load()` raises `FileNotFoundError` if checkpoint missing (caller must decide)"):
```python
try:
    previous = get_platform_checkpoint_manager().load("regime_labels")
    churn = label_churn(previous["state"], new_states, trailing_months=24)
except FileNotFoundError:
    log.info("No previous regime_labels checkpoint — churn metric unavailable on first run")
    churn = float("nan")
# ... only now save the new labels, overwriting old
get_platform_checkpoint_manager().save(labels_df, "regime_labels")
```

**`__main__` synthetic self-check pattern** (`gap_lag.py` lines 149-174, mandated by ponytail rules — "non-trivial logic leaves one runnable check"):
```python
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # synthetic states array, no network, no Phase-1-real-data dependency
    ...
    report_labeling_diagnostics({...})
```

---

### `src/trading_crab_lib/platform/prediction/nowcaster.py` (service, fit/predict)

**Analog:** `src/trading_crab_lib/platform/honesty/walkforward.py` for the clone-per-fold + automatic registry-logging idiom; `src/trading_crab_lib/platform/honesty/cv.py` for the `PurgedEmbargoedKFold` usage this module must plug into.

**Structural embargo pattern** (RESEARCH.md Pattern 5, verbatim-ready):
```python
def build_nowcaster_training_set(features_df: pd.DataFrame, labels: pd.Series, *,
                                  embargo_months: int = 12) -> tuple[pd.DataFrame, pd.Series]:
    if embargo_months < 0:
        raise ValueError("embargo_months must be >= 0")
    cutoff = labels.index.max() - pd.DateOffset(months=embargo_months)
    eligible_labels = labels.loc[labels.index <= cutoff]
    common = features_df.index.intersection(eligible_labels.index)
    X = features_df.loc[common]
    y = eligible_labels.loc[common]
    return X, y
```
Note the `ValueError` guard is new (not in any existing file) — added per RESEARCH.md's Security Domain V5 recommendation ("training-set builder should assert `embargo_months >= 0`").

**Calibrated fit through the frozen CV interface** (RESEARCH.md Pattern 6, using `cv.py` directly — no new CV code, reuse verbatim per "Don't Hand-Roll" table):
```python
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from trading_crab_lib.platform.honesty.cv import PurgedEmbargoedKFold

def fit_nowcaster(X, y, *, label_horizon: int = 12, embargo: int = 1, n_splits: int = 5,
                   random_state: int = 42) -> CalibratedClassifierCV:
    cv = PurgedEmbargoedKFold(n_splits=n_splits, label_horizon=label_horizon, embargo=embargo)
    model = CalibratedClassifierCV(
        LogisticRegression(max_iter=1000, random_state=random_state),
        method="sigmoid",
        cv=cv,
    )
    model.fit(X, y)
    return model
```

**Never-argmax output contract** (L2-01, CONTEXT D-carried-forward): callers must always receive `predict_proba()`, never `predict()`, mirrored from the incumbent's `prediction/__init__.py::predict_current()` dict-with-`probabilities` return shape (see incumbent CLAUDE.md "Key Abstractions" — `predict_current` returns `{"regime": int, "probabilities": {...}}`). Do not use the incumbent module; replicate the *shape convention* only.

**Transition-window accuracy** (RESEARCH.md Pattern 7 — copy verbatim; always returns all three keys together, per the design's explicit anti-"persistence-trap" warning):
```python
def transition_window_accuracy(y_true, y_pred, *, window_months: int = 3) -> dict:
    ...
    return {"overall_accuracy": ..., "transition_accuracy": ..., "steady_state_accuracy": ...}
```

**Registry logging:** if this module's evaluation runs through `run_walkforward` (CONTEXT explicitly recommends this — "nowcaster's walk-forward evaluation runs through `run_walkforward`, swap `DummyClassifier` for the calibrated logistic"), no new registry code is needed — call `run_walkforward(features_df, target_series, model=fit_nowcaster_estimator, min_train=..., registry_path=..., config={...})` exactly per `walkforward.py` lines 52-106 and `test_platform_walkforward.py`'s usage.

**Model persistence:** `joblib.dump`/`joblib.load` (RESEARCH.md Security Domain table cites incumbent pitfall P27 explicitly) — no existing platform-namespace model-persistence helper exists yet; add a small `joblib.dump(model, path)` call inline, following the incumbent's D14 migration convention (see incumbent `CLAUDE.md` ADR/D14: "sklearn model serialization uses joblib, not pickle").

---

### `src/trading_crab_lib/platform/prediction/transition_matrix.py` (utility, pure transform)

**Analog:** `src/trading_crab_lib/platform/honesty/gap_lag.py`'s `compute_gap`/`compute_detection_lag`/`sojourn_lag_ratio` — small, pure, independently-testable `compute_*` functions with no I/O, matching this module's single responsibility.

**Core pattern** (RESEARCH.md Pattern 8, copy verbatim):
```python
def empirical_transition_matrix(states: pd.Series) -> pd.DataFrame:
    """Row-normalized K x K count table: P(next state = j | current state = i)."""
    pairs = pd.DataFrame({"from": states.iloc[:-1].values, "to": states.iloc[1:].values})
    counts = pd.crosstab(pairs["from"], pairs["to"])
    return counts.div(counts.sum(axis=1), axis=0)
```
Function-design convention from `gap_lag.py`: no class wrapper, plain function, full docstring, type hints on both param and return (CLAUDE.md "Type hints on all public functions").

---

## Shared Patterns

### Module docstring style (apply to all 4 new `.py` files)
**Source:** `src/trading_crab_lib/platform/honesty/cv.py` lines 1-27, `gap_lag.py` lines 1-19
Triple-quoted module docstring immediately after `from __future__ import annotations`... actually **before** it (docstring is always the very first statement). Structure: one-line summary + `(REQ-ID, design §section)`, then explain the "why" (non-obvious algorithmic choice), then a `Usage::` block with a runnable import + call example.

### `from __future__ import annotations` + logger setup
**Source:** every `platform/honesty/*.py` file; enforced by CLAUDE.md ("Required: `from __future__ import annotations` at the top of all source files")
```python
from __future__ import annotations

import logging
# stdlib, then third-party, then trading_crab_lib imports (ruff isort order)

log = logging.getLogger(__name__)
```

### Config access — defensive `.get()`, no schema extension
**Source:** `platform/config.py` `_REQUIRED_PLATFORM_SECTIONS` list + CONTEXT's explicit instruction "don't extend `_REQUIRED_PLATFORM_SECTIONS`"
**Apply to:** `jump_model.py`, `nowcaster.py`, and the new `config/platform_settings.yaml` `labeling:` section — add the YAML block (mirroring the existing `cv:`/`holdout:`/`registry:` sections' comment-header style, lines 219-227, 203-217 of `platform_settings.yaml`), but do NOT add `"labeling"` to `_REQUIRED_PLATFORM_SECTIONS` in `config.py`.

Suggested new YAML block (style-matched to existing `cv:` section):
```yaml
# ── Jump-model labeling (L1-01/L1-02) ───────────────────────────────────────────
# λ = 4 × len(lean_feature_set) is a feature-count-scaled starting default
# (RESEARCH.md Pitfall 5) — re-derive, don't reuse the literal, if the lean
# feature set's column count changes.
labeling:
  K: 5
  lambda: 52.0
  n_restarts: 10
  embargo_months: 12
```

### Checkpoint persistence — reuse verbatim, no subclassing
**Source:** `platform/checkpoints.py::get_platform_checkpoint_manager()`
**Apply to:** `jump_model.py`/`diagnostics.py` (checkpoints: `regime_labels`, `regime_confidences`, `regime_diagnostics`, `regime_profiles`) and `nowcaster.py`/`transition_matrix.py` (checkpoints: nowcaster model artifact via joblib, `transition_matrix`).
```python
from trading_crab_lib.platform.checkpoints import get_platform_checkpoint_manager
cm = get_platform_checkpoint_manager()
cm.save(labels_df, "regime_labels")
```

### Causal gating — only for the nowcaster's feature input, never the labeler
**Source:** `platform/honesty/gating.py::assert_causal_features`
**Apply to:** `nowcaster.py` only. Per RESEARCH.md's explicit Anti-Pattern warning, do NOT wrap the labeler's `monthly_features` load with this gate expecting non-causal columns to exist — the labeler's *procedure* is intentionally globally-optimizing (non-causal at the batch level) even though its *input columns* are causal; only the nowcaster's *training target* embargo (D-01) is this phase's new causal-safety concern, and `assert_causal_features` already passes trivially on `monthly_features` per gating.py's own docstring (lines 4-9).

### Test file structure — class-per-behavior, parametrized invariant sweeps
**Source:** `tests/unit/test_platform_cv.py` (parametrized leakage sweep, `TestNoLeakageAcrossPurgeEmbargoWindow`) and `tests/unit/test_platform_walkforward.py` (`_make_synthetic_monthly()` local helper, `if __name__ == "__main__": pytest.main([__file__, "-x", "-q"])` footer)
**Apply to:** `test_platform_labeling.py`, `test_platform_nowcaster.py`
```python
"""Unit tests for trading_crab_lib.platform.labeling.jump_model (L1-01...03).

Synthetic monthly DataFrame, no network — mirrors
tests/unit/test_platform_walkforward.py's synthetic-frame convention.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest

def _make_synthetic_monthly(n_months: int = 120, seed: int = 42) -> pd.DataFrame:
    ...

class TestDPDecodeExact:
    def test_matches_brute_force_on_small_TK(self):
        ...

if __name__ == "__main__":
    pytest.main([__file__, "-x", "-q"])
```
Two-run invariant test for the churn metric (RESEARCH.md Pitfall 3 warning sign) must run the labeler/diagnostics pipeline **twice** in the same test against the same `tmp_path`-scoped checkpoint dir and assert churn == 0.0 on the second run — a single-run test cannot catch a broken load-before-save ordering.

## No Analog Found

None — every file in scope has a strong analog (module-docstring/class-shape analogs from `platform/honesty/`, or RESEARCH.md's already-verified-against-installed-signatures code patterns for the DP/jump-model/nowcaster math itself, which has no existing analog anywhere in the codebase because it's genuinely new algorithmic territory — RESEARCH.md's own "Don't Hand-Roll" table already confirms nothing off-the-shelf covers the DP decode).

## Metadata

**Analog search scope:** `src/trading_crab_lib/platform/` (all subpackages), `config/platform_settings.yaml`, `tests/unit/test_platform_*.py`; incumbent `src/trading_crab_lib/clustering.py` consulted for warm-start idiom only (not importable).
**Files scanned:** `platform/honesty/{cv,walkforward,gap_lag,gating,registry}.py`, `platform/checkpoints.py`, `platform/config.py`, `platform/taxonomy.py`, `config/platform_settings.yaml`, `tests/unit/test_platform_{walkforward,cv}.py`.
**Pattern extraction date:** 2026-07-22
