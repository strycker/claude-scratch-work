# Phase 2: Honesty Infrastructure - Pattern Map

**Mapped:** 2026-07-16
**Files analyzed:** 12 (6 modules + 6 test files; config change is an edit, not a new file)
**Analogs found:** 12 / 12

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|---|---|---|---|---|
| `src/trading_crab_lib/platform/honesty/__init__.py` | config/barrel | n/a | `src/trading_crab_lib/platform/__init__.py` (if barrel exists) / `plotting/__init__.py` re-export pattern | role-match |
| `src/trading_crab_lib/platform/honesty/holdout.py` | service (checkpoint factory + guard) | file-I/O | `src/trading_crab_lib/platform/checkpoints.py` | exact |
| `src/trading_crab_lib/platform/honesty/registry.py` | service (append-only ledger) | event-driven / file-I/O | `src/trading_crab_lib/platform/checkpoints.py` (I/O conventions) + RESEARCH.md Pattern 2 (no direct codebase analog for JSONL) | partial |
| `src/trading_crab_lib/platform/honesty/cv.py` | utility (sklearn extension) | transform | none in-repo (new algorithm class); RESEARCH.md Pattern 3 is the spec | no analog |
| `src/trading_crab_lib/platform/honesty/walkforward.py` | service (orchestration loop) | batch | `src/trading_crab_lib/monitoring/prediction.py::compute_cv_fold_scores` (clone-per-fold pattern) | role-match |
| `src/trading_crab_lib/platform/honesty/gating.py` | utility (guard function) | request-response | `ideas/gsd-salvage/prediction/feature_gating.py` | exact (port) |
| `src/trading_crab_lib/platform/honesty/gap_lag.py` | utility (pure compute) + service (artifact writer) | transform + file-I/O | `ideas/gsd-salvage/prediction/model_metrics_artifacts.py` | role-match |
| `config/platform_settings.yaml` (edit — add `holdout:`, `registry:`, `cv:` sections) | config | n/a | existing sections in same file (`data:`, `fred_monthly:`, `taxonomy:`) | exact |
| `tests/unit/test_platform_holdout.py` | test | invariant | `tests/unit/test_platform_taxonomy.py` | exact |
| `tests/unit/test_platform_registry.py` | test | unit | `tests/unit/test_platform_taxonomy.py` | role-match |
| `tests/unit/test_platform_cv.py` | test | unit (contract test) | `tests/unit/test_platform_transforms.py` (synthetic-frame style) | role-match |
| `tests/unit/test_platform_walkforward.py` | test | integration-lite (synthetic, no network) | `tests/integration/test_mini_pipeline.py` | exact |
| `tests/unit/test_platform_gating.py` | test | unit | `tests/unit/test_platform_taxonomy.py` | exact |
| `tests/unit/test_platform_gap_lag.py` | test | unit | `tests/unit/test_platform_taxonomy.py` | exact |

## Pattern Assignments

### `src/trading_crab_lib/platform/honesty/holdout.py` (service, file-I/O)

**Analog:** `src/trading_crab_lib/platform/checkpoints.py` (full file, 29 lines — read in full)

**Whole-file pattern to copy (module docstring + constant + factory function):**
```python
"""
Platform checkpoint-namespace factory.

Reuses the incumbent :class:`~trading_crab_lib.checkpoints.CheckpointManager`
verbatim (D-01: never subclass or reimplement save/load/is_fresh) pointed at a
separate directory, ``data/checkpoints/platform/``.
"""

from __future__ import annotations

from trading_crab_lib import DATA_DIR
from trading_crab_lib.checkpoints import CheckpointManager

PLATFORM_CHECKPOINT_DIR = DATA_DIR / "checkpoints" / "platform"


def get_platform_checkpoint_manager() -> CheckpointManager:
    """Return a :class:`CheckpointManager` scoped to the platform checkpoint namespace."""
    return CheckpointManager(checkpoint_dir=PLATFORM_CHECKPOINT_DIR)
```

**What to change for `holdout.py`:**
- Add a second constant/factory pair for `HOLDOUT_CHECKPOINT_DIR = DATA_DIR / "holdout"` and
  `get_holdout_checkpoint_manager()`, same shape as above (see RESEARCH.md Pattern 1, already
  written out nearly verbatim — copy it directly).
- Add a `split_by_holdout_boundary(df, cutoff="2020-12-31")` function performing
  `df.loc[:cutoff]` / `df.loc[cutoff_next:]` — this is new logic, no direct in-repo analog;
  write it as a plain pandas slice, no class needed (ponytail: one function, not a class).
- **Critical (D-03/Pitfall 1):** `get_platform_checkpoint_manager()`'s default path must never
  be able to reach `data/holdout/`. Do not add any `try/except FileNotFoundError → fall back to
  holdout` logic anywhere. No test double-duty: the invariant test (below) must assert this by
  attempting to load a holdout-dated row through the default manager and confirming absence /
  raise, not by monkeypatching.

### `src/trading_crab_lib/platform/honesty/gating.py` (utility, request-response)

**Analog:** `ideas/gsd-salvage/prediction/feature_gating.py` (full file, 65 lines — read in full above)

**Direct port — copy verbatim, adapt names only.** Full analog file content is in
`code_context` above. Key excerpt (signature + error message shape, lines 22-64):
```python
def select_step5_feature_path(
    processed_dir: Path,
    *,
    allow_noncausal_features: bool,
) -> tuple[Path, str, bool]:
    supervised_path = processed_dir / FEATURES_SUPERVISED_FILENAME
    if supervised_path.exists():
        return supervised_path, "features_supervised", False

    noncausal_path = processed_dir / FEATURES_NONCAUSAL_FILENAME
    if not allow_noncausal_features:
        raise FileNotFoundError(
            f"{supervised_path} not found.\n"
            "Step 5 leakage guardrail requires causal features.\n\n"
            "If you intentionally want to run with non-causal features, re-run with:\n"
            "  --allow-noncausal-features\n\n"
            f"Fallback file would be: {noncausal_path}"
        )
    ...
    log.warning("NONCAUSAL_USED=true — step 5 falling back from %s to %s.", ...)
    return noncausal_path, "features", True
```

**IMPORTANT — resolved Open Question 1 from RESEARCH.md:** grepped
`src/trading_crab_lib/platform/transforms_monthly.py` for `center=True`/`centered`/
`rolling(` — only hit is `equity_returns.rolling(3).std(ddof=0)` (trailing, not centered).
`monthly_features` is causal-by-construction (no `*_supervised` companion file exists or is
needed). **Therefore `gating.py`'s job is NOT a two-file selector.** Reshape the port to:
`select_platform_feature_path(checkpoint_dir, *, allow_noncausal=False)` that (a) loads
`monthly_features` via `get_platform_checkpoint_manager()`, (b) asserts no forbidden
centered-window column name pattern is present (e.g., no columns produced with
`rolling(..., center=True)` — grep the taxonomy/transform code for any tagged column names
if such a list exists; otherwise assert structurally), (c) raises loudly only if such a column
is found and `allow_noncausal=False`. Keep the same raise-loud / warn-on-opt-out shape as the
analog. Do not invent a `monthly_features_supervised.parquet` path — it does not exist and
should not be created.

### `src/trading_crab_lib/platform/honesty/gap_lag.py` (utility + service)

**Analog:** `ideas/gsd-salvage/prediction/model_metrics_artifacts.py` (parquet-writer half only —
the FoldReport-specific aggregation logic does NOT transfer; only the I/O shape does)

**Pattern to copy — "empty-safe DataFrame + to_parquet" idiom** (lines 424-439, 447-462,
464-484 — same idiom repeated 3x in the analog, use once):
```python
df = pd.DataFrame(rows)
if df.empty:
    df = pd.DataFrame(columns=[...])  # explicit column list keeps schema stable when empty
df.to_parquet(output_path, index=False)
```
Apply this idiom to `gap_lag.py`'s artifact writer, writing under
`outputs/reports/model_metrics/` per D-05 (same directory the analog already targets — no new
directory convention needed).

**Pure compute functions — no analog, write from RESEARCH.md spec directly** (already fully
specified in RESEARCH.md Code Examples — copy `sojourn_lag_ratio()` verbatim; write
`compute_gap()` / `compute_detection_lag()` following the same style: plain functions over
pandas Series, explicit `ValueError` on invalid input, no classes).

### `src/trading_crab_lib/platform/honesty/registry.py` (service, event-driven)

**Analog:** no direct in-repo analog for JSONL — closest I/O-convention analog is
`src/trading_crab_lib/checkpoints.py` (manifest-as-JSON pattern, lines 45-59 imports:
`import json`, `from pathlib import Path`) for "always open with explicit encoding, mkdir
parents=True" conventions. RESEARCH.md Pattern 2 already contains the exact target
implementation (`append_trial()` with `hashlib.md5` config hash, `subprocess.run(["git",
"rev-parse", "HEAD"])` git SHA, JSONL append `"a"` mode) — use it as the source, not
`checkpoints.py` structure.

**Error handling / logging convention to copy from `checkpoints.py`:** `log = logging.getLogger(__name__)`
at module top, after imports — every module in this package must follow this (see CLAUDE.md
Logging conventions).

### `src/trading_crab_lib/platform/honesty/cv.py` (utility, transform)

**Analog:** none in-repo. `sklearn.model_selection.BaseCrossValidator` is the contract;
RESEARCH.md Pattern 3 is the complete target implementation (`PurgedEmbargoedKFold`, ~35 lines,
already fully written in RESEARCH.md — copy that class body directly, no modification needed
except wiring `label_horizon`/`embargo` as required (no-default) kwargs per Open Question 2).

**Import convention to match repo style** (`from __future__ import annotations` first, per
CLAUDE.md/root convention — confirm this line is present; RESEARCH.md's example already has it).

### `src/trading_crab_lib/platform/honesty/walkforward.py` (service, batch)

**Analog:** `src/trading_crab_lib/monitoring/prediction.py::compute_cv_fold_scores` — read
via Grep-located function (clone-per-fold idiom, referenced in ADR D26). Confirmed pattern
(not re-read here since RESEARCH.md Pattern 4 already extracted the relevant idiom):
```python
from sklearn.base import clone
m = clone(model)
m.fit(X_train, y_train)
```
Full target implementation already written in RESEARCH.md Pattern 4
(`expanding_steps()` generator + `run_walkforward()`) — copy directly; wire
`registry.append_trial()` call at the end of `run_walkforward()` per Pitfall 3 (must call
inside the runner, not left to callers).

## Shared Patterns

### Module docstring + `from __future__ import annotations` + logger setup
**Source:** `src/trading_crab_lib/platform/checkpoints.py` lines 1-24, 46-59;
`src/trading_crab_lib/platform/config.py` lines 1-24
**Apply to:** every new file in `platform/honesty/`
```python
"""One-line purpose statement.

Longer description of what this module does and why.
"""

from __future__ import annotations

import logging
...

log = logging.getLogger(__name__)
```

### Config-driven, no hardcoded params
**Source:** `src/trading_crab_lib/platform/config.py` — `load_platform_config()`,
`validate_platform_config()` with `_REQUIRED_PLATFORM_SECTIONS` list and collect-all-errors
raise
**Apply to:** `config/platform_settings.yaml` edit — add `holdout:`, `registry:`, `cv:` to
whatever list mirrors `_REQUIRED_PLATFORM_SECTIONS` if these become required sections (planner
call: CONTEXT.md leaves exact schema to discretion — if made required, add to that list and to
`_well_formed_cfg()`-style test fixtures).

### CheckpointManager reuse — never subclass
**Source:** `src/trading_crab_lib/platform/checkpoints.py` (D-01 comment, line 4: "never
subclass or reimplement save/load/is_fresh")
**Apply to:** `holdout.py` — both `get_platform_checkpoint_manager()` and
`get_holdout_checkpoint_manager()` must construct `CheckpointManager(checkpoint_dir=...)`
instances, never a subclass.

### Test file structure — class-per-function, docstring-per-test explaining the behavior
**Source:** `tests/unit/test_platform_taxonomy.py` (full file read, 80+ lines shown)
```python
"""Unit tests for trading_crab_lib.platform.<module> (<REQ-ID>).

Follows the incumbent tests/unit/test_config.py structure: ...
"""

from __future__ import annotations

import pytest

from trading_crab_lib.platform.<module> import <functions>


def _well_formed_cfg() -> dict:
    """A <fixture> where <invariant holds>."""
    return {...}


class Test<Behavior>:
    def test_<specific_behavior>(self):
        """<one-line description of what's being verified>."""
        ...
```
**Apply to:** all 6 new test files. For `test_platform_holdout.py`, model the invariant-style
assertion after this class-based structure but the assertion itself (default manager cannot
reach holdout data) is new — no existing invariant test to copy verbatim; write it as
`class TestHoldoutBoundary: def test_default_manager_cannot_load_post_2020_rows(self): ...`.

### Synthetic-frame, no-network test data
**Source:** `tests/integration/test_mini_pipeline.py` (`_make_synthetic_macro(n_quarters=80)`
pattern per CLAUDE.md D44) — apply the same idea at monthly cadence for
`test_platform_walkforward.py` and `test_platform_cv.py` (build a synthetic monthly
DataFrame with a DatetimeIndex, no file I/O, no network).

## No Analog Found

| File | Role | Data Flow | Reason |
|---|---|---|---|
| `src/trading_crab_lib/platform/honesty/cv.py` | utility | transform | No purged/embargoed CV splitter exists anywhere in this codebase or its history — genuinely new algorithm class; RESEARCH.md Pattern 3 is the authoritative spec to implement against (not a codebase analog) |
| `src/trading_crab_lib/platform/honesty/registry.py` (JSONL append logic specifically) | service | event-driven | No JSONL ledger exists in the codebase; nearest relative is `per_fold.jsonl` writing in `model_metrics_artifacts.py` (lines 441-445: `with per_fold_path.open("w", ...)` — note this analog uses `"w"` not `"a"`, which is the OPPOSITE of what D-01 requires for the trial registry — do not copy the open-mode, only the `json.dump(row, f, default=str); f.write("\n")` idiom) |

## Metadata

**Analog search scope:** `src/trading_crab_lib/platform/`, `src/trading_crab_lib/checkpoints.py`,
`src/trading_crab_lib/monitoring/prediction.py`, `ideas/gsd-salvage/prediction/`,
`tests/unit/test_platform_*.py`, `tests/integration/test_mini_pipeline.py`
**Files scanned:** 8 read in full/targeted, 1 grepped for centered-window transforms
**Pattern extraction date:** 2026-07-16
