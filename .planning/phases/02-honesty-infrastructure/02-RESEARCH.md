# Phase 2: Honesty Infrastructure - Research

**Researched:** 2026-07-16
**Domain:** Financial-ML evaluation hygiene infrastructure (holdout isolation, purged/embargoed CV, trial ledger, walk-forward harness, causal-feature gating)
**Confidence:** MEDIUM-HIGH

## Summary

This phase builds six structural guarantees around the Phase-1 monthly platform
(`src/trading_crab_lib/platform/`) — no new modeling, only scaffolding every future
model must run through. Five of the six mechanisms (holdout carve, registry, walk-forward
runner skeleton, causal-feature gate, gap/lag reporting) are straightforward — they extend
patterns already proven in this exact codebase (`CheckpointManager(checkpoint_dir=...)`,
config-driven validation, salvaged `feature_gating.py`/`model_metrics_artifacts.py`). The
one piece with real design risk is HON-04 (purged/embargoed CV): there is no maintained,
trustworthy PyPI package for it — `mlfinlab` (the canonical reference implementation) went
closed-source/paid, and the only free alternative (`timeseriescv`) is unmaintained since
2018 with unknown download volume (flagged `SUS` by the legitimacy gate). scikit-learn ships
`TimeSeriesSplit(gap=...)` but that is a single fixed gap, not label-aware purging plus a
fractional embargo — insufficient per López de Prado ch. 7 and per design §6.5's explicit
callout. **Hand-roll a `PurgedEmbargoedKFold` implementing sklearn's `BaseCrossValidator`
API** (~100 lines, well-documented algorithm, correctness-critical — stays on Fable per root
CLAUDE.md's AI routing rule, not delegated).

The other five mechanisms are almost entirely "wire existing patterns into a new namespace."
`CheckpointManager` already accepts a custom `checkpoint_dir`, which is the exact seam
needed for the holdout split (`platform/checkpoints.py` → default manager reads
`data/checkpoints/platform/` truncated ≤2020-12; a second manager reads `data/holdout/`
and is never imported by default code paths). The trial registry is a ~50-line
append-only-JSONL writer/reader with no external dependency (JSONL + schema validation at
write time is the documented industry pattern for exactly this use case). The walk-forward
runner is a simple expanding-window loop around `sklearn.dummy.DummyClassifier` as the
"trivial model" proving the refit→record→step-forward cycle end to end, with real models
(jump-model labeler, nowcaster) arriving in Phase 3 against the same interface. Causal
gating and metrics-artifact persistence port the two salvaged modules near-verbatim, adapted
from the frozen quarterly `features_supervised.parquet` concept to the platform's
`monthly_features` checkpoint distinction.

**Primary recommendation:** Build `src/trading_crab_lib/platform/honesty/` as a five-module
package (`holdout.py`, `registry.py`, `cv.py`, `walkforward.py`, `gating.py` +
`metrics_artifacts.py`), each independently testable with synthetic monthly data, wired
together only by the walk-forward runner. Hand-roll the purged/embargoed splitter; do not
add a new PyPI dependency for it.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Holdout carve (physical file split) | Data / Storage (checkpoints) | — | `CheckpointManager` already owns file I/O; a second checkpoint-dir instance is the natural seam, not a new subsystem |
| Trial registry | Data / Storage (flat file) | — | Append-only JSONL under a tracked git path; no query engine, no DB — read via pandas at analysis time |
| Purged/embargoed CV splitter | Library (sklearn-compatible utility) | — | A stateless `BaseCrossValidator` subclass consumed by any future supervised trainer (Phase 3+); no I/O, no config beyond purge/embargo params |
| Walk-forward runner | Orchestration (pipeline-adjacent) | Data / Storage (writes registry rows per step) | Drives the expanding-window loop and calls into registry + holdout-boundary checks; it is the integration point, not a leaf module |
| Causal-feature gating | Library (guard function) | — | A pure decision function (`select_platform_feature_path`) called by any training entry point; no state |
| Gap/lag metrics | Library (compute) | Reporting (CLI + artifact) | Computation is a pure function over two label/probability series; the CLI+artifact surface (D-05) is the reporting half, deferred report *wiring* is Phase 4 |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| scikit-learn | 1.9.0 (installed; project floor `>=1.4`) | `BaseCrossValidator` base class, `DummyClassifier`/`DummyRegressor` for the trivial model | Already a core dependency; `BaseCrossValidator` is the documented extension point for custom splitters — every downstream trainer that accepts `cv=` expects this interface `[VERIFIED: local install]` |
| pandas | 2.0+ (already core dep) | JSONL registry read/write, monthly index slicing for the holdout boundary | Already core; `pd.read_json(lines=True)` / manual line-append covers the registry without a new dependency |
| pyarrow | 14.0+ (already core dep) | Metrics-artifact parquet output (mirrors salvaged `model_metrics_artifacts.py`) | Already core; consistent with every other checkpoint/report file in the repo |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| joblib | 1.3+ (already core dep) | Optional: hash config dicts for the registry's config-hash column | Only if `hashlib.md5(json.dumps(cfg, sort_keys=True))` proves insufficient (it won't — see Code Examples) |
| GitPython | not installed | Programmatic git SHA capture for registry rows | **Do not add.** `subprocess.run(["git", "rev-parse", "HEAD"])` is one line and avoids a new dependency (ponytail rung 3: stdlib does it) |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Hand-rolled `PurgedEmbargoedKFold` | `mlfinlab` | Canonical reference implementation, but relicensed closed-source/paid — cannot be a dependency `[CITED: web search — mlfinlab relicensing]` |
| Hand-rolled `PurgedEmbargoedKFold` | `timeseriescv` (PyPI 0.2) | Free, has a GitHub repo, but unmaintained since 2018, `weeklyDownloads: null` (unknown), flagged `SUS` by the legitimacy gate — not worth the dependency risk for ~100 lines of well-specified algorithm `[VERIFIED: package-legitimacy check]` |
| JSONL flat-file registry | SQLite (`registry.db`) | HON-02 explicitly allows either ("flat file/SQLite store" — REQUIREMENTS.md), but CONTEXT.md D-01 **locks JSONL-in-git** for tamper-evidence via git history; SQLite binary diffs are not human-reviewable in PRs |
| `sklearn.dummy.DummyClassifier` as trivial model | Hand-rolled persistence classifier | `DummyClassifier(strategy="prior")`/`(strategy="most_frequent")` already implements "predict the class seen most/last" with zero new code — reach for it before hand-rolling (ponytail rung 5) |

**Installation:** No new packages required — every library above is already a project
dependency (`pyproject.toml` / `requirements.txt`). This phase adds zero new entries to
either file.

**Version verification:** `scikit-learn` 1.9.0 confirmed installed locally (`python3 -c
"import sklearn; print(sklearn.__version__)"` → `1.9.0`); project floor is `>=1.4` in
CLAUDE.md's Frameworks table — `BaseCrossValidator.split(X, y=None, groups=None)` and
`get_n_splits()` signatures are stable across that whole range `[VERIFIED: local
introspection]`.

## Package Legitimacy Audit

| Package | Registry | Age | Downloads | Source Repo | Verdict | Disposition |
|---------|----------|-----|-----------|--------------|---------|-------------|
| timeseriescv | PyPI | ~8 yrs (published 2018-09-07) | unknown (`weeklyDownloads: null`) | github.com/sam31415/timeseriescv | SUS | **Not adopted** — hand-roll instead |
| mlfinlab | PyPI | n/a | n/a | n/a | N/A (not found on PyPI under this constraint) | **Not adopted** — relicensed closed-source; excluded on policy grounds regardless of registry status |

**Packages removed due to [SLOP] verdict:** none.
**Packages flagged as suspicious [SUS]:** `timeseriescv` — evaluated and explicitly rejected
in favor of hand-rolling (see Alternatives Considered); no install is planned, so no
`checkpoint:human-verify` task is needed — this row exists to document the evaluation trail.

*No new packages are recommended for installation in this phase — nothing here needs a
`checkpoint:human-verify` gate.*

## Architecture Patterns

### System Architecture Diagram

```
                    ┌─────────────────────────────────────────┐
                    │  Phase 1 output (already built)          │
                    │  platform/checkpoints.py                 │
                    │  → data/checkpoints/platform/             │
                    │     monthly_features.parquet (1962–today) │
                    └───────────────────┬───────────────────────┘
                                         │
                     ┌───────────────────┴────────────────────┐
                     │  HON-01: holdout.py                     │
                     │  split_by_holdout_boundary(df, cutoff)  │
                     │  writes TWO trees:                      │
                     └──┬───────────────────────────────────┬──┘
                        │                                   │
                        ▼                                   ▼
        ┌───────────────────────────┐        ┌──────────────────────────────┐
        │ data/checkpoints/platform/  │        │ data/holdout/                 │
        │ (dev default, ≤2020-12)     │        │ (2021+, opt-in only)          │
        │ get_platform_checkpoint_    │        │ get_holdout_checkpoint_       │
        │ manager() — DEFAULT PATH    │        │ manager() — explicit call,    │
        └──────────────┬──────────────┘        │ never on the default path     │
                        │                        └────────────────────────────┘
                        ▼
        ┌────────────────────────────────────────────────────────┐
        │  HON-06: gating.py                                       │
        │  select_platform_feature_path(..., allow_noncausal=False)│
        │  loud FileNotFoundError unless opted out                 │
        └──────────────────────┬─────────────────────────────────┘
                                ▼
        ┌────────────────────────────────────────────────────────┐
        │  HON-03: walkforward.py                                  │
        │  for t in expanding_steps(index, step_months=1):         │
        │      train = data[:t]  (never t+1..)                     │
        │      model = clone(trivial_model).fit(train)             │
        │      decision = model.predict(data.loc[[t]])              │
        │      record_step(t, decision, params) → registry row      │
        └──────────────────────┬─────────────────────────────────┘
                                │  each config's aggregate result
                                ▼
        ┌────────────────────────────────────────────────────────┐
        │  HON-02: registry.py                                     │
        │  append_trial(config_hash, features, params, metrics,    │
        │               git_sha, timestamp) → registry/trials.jsonl │
        │  (tracked in git — NOT under data/)                       │
        └────────────────────────────────────────────────────────┘

        ┌────────────────────────────────────────────────────────┐
        │  HON-04: cv.py — PurgedEmbargoedKFold(BaseCrossValidator)│
        │  used BY any future Phase 3+ supervised trainer,          │
        │  no dependency on the above four modules                  │
        └────────────────────────────────────────────────────────┘

        ┌────────────────────────────────────────────────────────┐
        │  HON-05: gap_lag.py                                       │
        │  compute_gap(smoothed_perf, filtered_perf)                │
        │  compute_detection_lag(transitions, filtered_probs, thr)  │
        │  → CLI print + outputs/reports/model_metrics/*.parquet    │
        │  (generic; Phase 3 plugs in real jump-model + nowcaster)  │
        └────────────────────────────────────────────────────────┘
```

### Recommended Project Structure
```
src/trading_crab_lib/platform/
├── honesty/
│   ├── __init__.py         # re-exports (mirrors platform/__init__.py convention)
│   ├── holdout.py          # HON-01: split + two CheckpointManager factories + guard
│   ├── registry.py         # HON-02: append_trial(), read_trials(), config hashing
│   ├── cv.py                # HON-04: PurgedEmbargoedKFold(BaseCrossValidator)
│   ├── walkforward.py       # HON-03: expanding_steps(), WalkForwardRunner
│   ├── gating.py            # HON-06: select_platform_feature_path() (ported feature_gating.py)
│   └── gap_lag.py           # HON-05: compute_gap(), compute_detection_lag(), sojourn_lag_ratio()
config/
└── platform_settings.yaml   # add `holdout:`, `registry:`, `cv:` sections (per CONTEXT.md discretion)
tests/unit/
├── test_platform_holdout.py
├── test_platform_registry.py
├── test_platform_cv.py
├── test_platform_walkforward.py
├── test_platform_gating.py
└── test_platform_gap_lag.py
```

### Pattern 1: Holdout carve as a second `CheckpointManager` instance, not a new I/O layer
**What:** Reuse `CheckpointManager(checkpoint_dir=...)` — already parameterized — pointed at
`data/holdout/` for 2021+ rows, and the existing `PLATFORM_CHECKPOINT_DIR` for ≤2020-12 rows.
**When to use:** Any place `monthly_features` (or any platform checkpoint) is written or read.
**Example:**
```python
# Source: src/trading_crab_lib/platform/checkpoints.py (existing, D-01 pattern)
from trading_crab_lib import DATA_DIR
from trading_crab_lib.checkpoints import CheckpointManager

PLATFORM_CHECKPOINT_DIR = DATA_DIR / "checkpoints" / "platform"   # ≤2020-12, DEFAULT
HOLDOUT_CHECKPOINT_DIR = DATA_DIR / "holdout"                      # 2021+, OPT-IN ONLY

def get_platform_checkpoint_manager() -> CheckpointManager:
    return CheckpointManager(checkpoint_dir=PLATFORM_CHECKPOINT_DIR)

def get_holdout_checkpoint_manager() -> CheckpointManager:
    """Never called by default code paths — live-scoring mode calls this explicitly."""
    return CheckpointManager(checkpoint_dir=HOLDOUT_CHECKPOINT_DIR)
```
The split itself is one `df.loc[:cutoff]` / `df.loc[cutoff + offset:]` at write time inside
`build_monthly_spine()`'s caller (or a thin wrapper around it) — do not touch
`transforms_monthly.py` itself; the split is a *write-path* concern, not a *feature-
computation* concern.

### Pattern 2: JSONL trial registry — append-only, schema-validated at write time
**What:** One JSON object per line, opened in `"a"` mode, never rewritten.
**When to use:** Every evaluated configuration in every future modeling phase.
**Example:**
```python
# Source: pattern confirmed by web research (JSONL append-only + write-time schema
# validation is the documented industry convention for experiment ledgers)
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REGISTRY_PATH = Path("registry/trials.jsonl")  # tracked in git — NOT under data/

def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()

def append_trial(*, config: dict, features: list[str], metrics: dict) -> dict:
    row = {
        "config_hash": hashlib.md5(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest()[:12],
        "config": config,
        "features": features,
        "metrics": metrics,
        "git_sha": _git_sha(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REGISTRY_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")
    return row
```
Read back with `pd.read_json(REGISTRY_PATH, lines=True)` for analysis — no query engine
needed at this scale (dozens–hundreds of trials, not millions).

### Pattern 3: `BaseCrossValidator` subclass for purged + embargoed K-fold
**What:** A drop-in replacement for `TimeSeriesSplit` implementing López de Prado ch. 7:
purge overlapping-label training rows around each test fold, then embargo a further window
immediately after the test fold.
**When to use:** Any supervised component (Phase 3+) with overlapping labels (forward-return
targets, forward-regime-transition targets).
**Example:**
```python
# Source: sklearn BaseCrossValidator extension pattern (VERIFIED via local introspection);
# purge/embargo algorithm per López de Prado, Advances in Financial ML, ch. 7 [ASSUMED —
# book content from training knowledge, not re-verified against the physical text this
# session; conventions cross-checked against web search results describing the same
# purge/embargo definitions].
from __future__ import annotations

import numpy as np
from sklearn.model_selection import BaseCrossValidator


class PurgedEmbargoedKFold(BaseCrossValidator):
    """K-fold CV with purging (drop train rows whose label window overlaps the test
    fold) and embargo (drop a further window of train rows immediately after the
    test fold, to block leakage through slow-moving features/autocorrelation).

    Args:
        n_splits: number of folds.
        label_horizon: purge window, in index positions — rows whose label
            "resolves" up to `label_horizon` steps after their own index are
            purged if that resolution window overlaps the test fold.
        embargo: embargo window, in index positions, applied immediately after
            each test fold.
    """

    def __init__(self, n_splits: int = 5, *, label_horizon: int = 1, embargo: int = 0) -> None:
        self.n_splits = n_splits
        self.label_horizon = label_horizon
        self.embargo = embargo

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def split(self, X, y=None, groups=None):
        n = len(X)
        indices = np.arange(n)
        fold_bounds = np.array_split(indices, self.n_splits)
        for test_idx in fold_bounds:
            test_start, test_end = test_idx[0], test_idx[-1]
            purge_start = max(0, test_start - self.label_horizon)
            embargo_end = min(n, test_end + 1 + self.embargo)
            train_mask = np.ones(n, dtype=bool)
            train_mask[purge_start:embargo_end] = False
            yield indices[train_mask], test_idx
```
**Config exposure (Claude's Discretion, CONTEXT.md):** add a `cv:` section to
`platform_settings.yaml` — `label_horizon_months` defaults to the longest forward horizon in
use (design D8: 1m/3m decision horizons, 12m strategic input → default purge = 12 to be safe
until a specific model's horizon is known; embargo defaults to a small fixed window, e.g. 1
month, per López de Prado's convention of "a few days to a few percent of sample size" scaled
to monthly data `[ASSUMED — convention transferred from daily-bar literature to monthly
cadence; flag for confirmation, see Assumptions Log]`.

### Pattern 4: Walk-forward runner as an expanding-window generator + trivial model
**What:** A generator yielding `(t, train_slice, test_row)` tuples; the runner clones and
refits a model at each step, using `sklearn.base.clone()` (the exact pattern already used in
the incumbent's `monitoring/prediction.py::compute_cv_fold_scores` per ADR/D26).
**When to use:** HON-03's Phase-0 exit criterion — "walk-forward runner executes a trivial
model end-to-end."
**Example:**
```python
# Source: pattern mirrors src/trading_crab_lib/monitoring/prediction.py's
# compute_cv_fold_scores() clone-per-fold convention (existing codebase pattern, D26)
from sklearn.base import clone
from sklearn.dummy import DummyClassifier


def expanding_steps(index, *, min_train_months: int = 60, step_months: int = 1):
    """Yield (t, train_index, test_index) for an expanding window, refitting every
    step_months, honoring the ≤2020-12 holdout boundary (caller must pre-slice)."""
    for i in range(min_train_months, len(index), step_months):
        yield index[i], index[:i], index[i : i + 1]


def run_walkforward(features_df, target_series, *, model=None):
    model = model or DummyClassifier(strategy="prior")  # trivial model, Phase 2 default
    decisions = []
    for t, train_idx, test_idx in expanding_steps(features_df.index):
        m = clone(model)
        m.fit(features_df.loc[train_idx], target_series.loc[train_idx])
        pred = m.predict(features_df.loc[test_idx])
        decisions.append({"t": str(t), "prediction": pred.tolist(), "params": m.get_params()})
    return decisions
```
`DummyClassifier(strategy="prior")` (predicts the class prior, i.e., most frequent class
weighted by frequency) is the ponytail-correct choice for "trivial model" — zero hand-rolled
persistence logic, already in the core sklearn dependency. A reasonable placeholder target
for Phase 2's proof run: sign of next-month return on the `equities_tr` research series from
`monthly_features` (real jump-model regime labels arrive in Phase 3 and replace this target
without changing the runner's interface).

### Pattern 5: Causal-feature gating — direct port of salvaged `feature_gating.py`
**What:** A guard function returning `(path, source_name, noncausal_used)`, raising loudly
unless explicitly opted out.
**When to use:** Every supervised-training entry point in the platform namespace.
**Example:**
```python
# Source: ideas/gsd-salvage/prediction/feature_gating.py, adapted to platform checkpoint
# names (monthly_features vs monthly_features_supervised — see Open Questions: Phase 1 did
# not produce a *_supervised split; this phase may need to add one, or gate on causal-only
# transform flags within monthly_features itself — planner must resolve, see below)
def select_platform_feature_path(checkpoint_dir, *, allow_noncausal: bool) -> tuple[Path, str, bool]:
    supervised = checkpoint_dir / "monthly_features_supervised.parquet"
    if supervised.exists():
        return supervised, "monthly_features_supervised", False
    noncausal = checkpoint_dir / "monthly_features.parquet"
    if not allow_noncausal:
        raise FileNotFoundError(
            f"{supervised} not found. Supervised training requires causal features.\n"
            "Pass allow_noncausal=True to intentionally opt out (loud, logged)."
        )
    log.warning("NONCAUSAL_USED=true — falling back to %s", noncausal.name)
    return noncausal, "monthly_features", True
```

### Anti-Patterns to Avoid
- **Reusing `TimeSeriesSplit(gap=N)` as "good enough" purging:** it applies a single fixed
  gap between train and test, not label-horizon-aware purging plus a separate embargo — the
  design (§6.5) explicitly calls this insufficient for overlapping labels.
- **Subclassing or monkey-patching `CheckpointManager`** for the holdout split — D-01 (Phase
  1) already establishes "reuse verbatim, never subclass"; a second instance with a different
  `checkpoint_dir` is the correct pattern, already proven.
- **Writing the registry as a dict overwritten each run** — defeats the "immutable
  pre-registration ledger" purpose (D-01); always open in append mode, never `"w"`.
- **Computing gap/lag metrics only at report-generation time in Phase 4** — HON-05 requires
  them as first-class *run* outputs now; Phase 4 only adds weekly-report *wiring* (D-05), the
  computation must already exist and be tested in this phase.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| sklearn-compatible CV splitter shape | A bespoke train/test-index generator with its own calling convention | Subclass `sklearn.model_selection.BaseCrossValidator` | Every future trainer (`cross_val_score`, `GridSearchCV`, or the codebase's own manual clone-per-fold loop) expects `.split(X)` yielding index arrays and `.get_n_splits()` — matching the interface costs nothing and buys drop-in compatibility |
| "Trivial model" for the walk-forward proof | A custom `PersistenceClassifier` class | `sklearn.dummy.DummyClassifier(strategy="prior")` | Already does exactly this; zero new code, zero new tests needed for the model itself (only for the runner around it) |
| Config hashing for the registry | A custom canonicalization/hashing scheme | `hashlib.md5(json.dumps(cfg, sort_keys=True).encode())` | `sort_keys=True` makes JSON serialization deterministic for the plain nested dicts this project uses; no need for a canonical-JSON library |
| Git SHA capture | GitPython or a hand-rolled `.git/` parser | `subprocess.run(["git", "rev-parse", "HEAD"])` | One line, no new dependency, git is always present in this dev environment |

**Key insight:** Every mechanism in this phase except the purged/embargoed CV splitter has
either (a) a direct existing-codebase pattern to extend (`CheckpointManager`, salvaged
modules) or (b) a one-line stdlib/sklearn answer. The only place custom code is justified is
the CV splitter, because the free ecosystem genuinely has no maintained implementation of
this specific, well-published algorithm.

## Common Pitfalls

### Pitfall 1: Holdout guard that "warns" instead of "raises"
**What goes wrong:** A WARNING-level log on default-path holdout access is easy to miss in a
long pipeline run; the whole point of D13/HON-01 is that development *cannot* accidentally
see 2021+ data.
**Why it happens:** The codebase's existing convention favors graceful degradation
(WARNING + continue) for network/data-quality issues (CLAUDE.md Error Handling section) —
that convention does not apply here.
**How to avoid:** CONTEXT.md's own `<specifics>` section is explicit: "Holdout guard should
fail loudly and unmistakably if code attempts default-path access to holdout data (raise, not
warn)." Implement as a hard `FileNotFoundError`/`RuntimeError` from
`get_platform_checkpoint_manager()`'s default path — there should be no code path by which
`data/holdout/` content becomes reachable without calling
`get_holdout_checkpoint_manager()` explicitly.
**Warning signs:** Any `try/except FileNotFoundError: fall back to holdout dir` pattern
anywhere in the codebase — this must never exist.

### Pitfall 2: Purge window sized in the wrong units
**What goes wrong:** López de Prado's canonical examples use daily bars; naively copying "a
few days" of purge/embargo onto a **monthly** spine (design D10) purges/embargoes essentially
nothing (< 1 row) — the guard becomes a no-op.
**Why it happens:** Literature and blog examples are almost all daily-frequency; this
project's spine is monthly (12 rows/year, not 252).
**How to avoid:** Size `label_horizon` and `embargo` in **months**, driven by the actual
label/target horizon (design D8: 1m/3m decision horizons, 12m strategic tilt) — not a fixed
day count. Default purge = the horizon of whatever forward-return/forward-regime target is in
use; make it a required, explicit parameter (no silent default of 0) so Phase 3+ trainers
must think about it per-target.
**Warning signs:** A CV test asserting `n_purged_rows > 0` for realistic monthly horizon
values — if this test can't be written honestly, the sizing is wrong.

### Pitfall 3: Registry logs metrics but not the *config that produced them*
**What goes wrong:** A registry row with only `accuracy: 0.62` and no reconstruction of what
features/params/CV-fold-boundaries produced it is worthless as the "multiple-testing
denominator" the deflated Sharpe needs (design §8.4/§22) — you can count trials but not
distinguish them.
**Why it happens:** Easy to add registry logging as an afterthought bolted onto a training
script, capturing only the final score.
**How to avoid:** `append_trial()` must be called with the *full* evaluated configuration
(feature list, model class + hyperparameters, CV scheme params, resulting metrics) as a
single atomic write — D-02 is explicit: "A trial = one evaluated configuration (its full
walk-forward result)... All grid cells logged regardless of outcome. No manual bookkeeping —
the runner logs automatically." Wire `append_trial()` calls into the walk-forward runner
itself, not into ad-hoc analysis scripts.
**Warning signs:** Any script that computes a metric and prints it without also calling
`append_trial()` — that's a decision made outside the ledger, exactly what D-01 exists to
prevent.

### Pitfall 4: Gap/lag metrics computed against a target that doesn't exist yet
**What goes wrong:** Design §5.4's gap/lag metrics fundamentally compare *smoothed* (two-
sided) regime labels against *filtered* (real-time nowcast) probabilities — neither exists
until Phase 3 (jump-model labeler + logistic nowcaster). Attempting to compute "the real
metric" in Phase 2 either blocks on Phase 3 or produces meaningless numbers.
**Why it happens:** HON-05's phrasing ("computed and reported as first-class outputs") reads
as if real numbers are expected now.
**How to avoid:** Per design §14's Phase 0/Phase 1 split (Phase 0 = infrastructure only;
Phase 1 = "smoothed-vs-filtered gap and detection lag reported" as an *exit criterion of the
next phase*), this phase's job is the **generic, tested computation functions** —
`compute_gap(smoothed_perf, filtered_perf)`, `compute_detection_lag(transitions,
filtered_probs, threshold)` — proven against synthetic label/probability series in unit
tests, with the CLI+artifact surface (D-05) wired and working end-to-end on that synthetic
data. Phase 3 plugs in the real jump-model output without changing this module's interface.
**Warning signs:** A HON-05 task that depends on Phase 3 artifacts existing — that's a phase-
boundary violation; flag it back to the planner rather than silently deferring the whole
requirement.

### Pitfall 5: `monthly_features_supervised` split doesn't exist yet (unlike the incumbent's `features_supervised.parquet`)
**What goes wrong:** The salvaged `feature_gating.py` ports cleanly *in shape* but assumes a
causal-vs-noncausal checkpoint pair already exists (`features_supervised.parquet` vs
`features.parquet`, per ADR #1 in the incumbent). Phase 1's `transforms_monthly.py` (per its
summary) produces `daily_raw` / `monthly_raw` / `monthly_features` — **no supervised/causal
split is mentioned in the Phase 1 summary.**
**Why it happens:** Phase 1 focused on the monthly spine and lean feature taxonomy, not the
centered-vs-causal smoothing distinction — design D12/§9 says agency features use ALFRED
point-in-time alignment (which is already causal by construction) and the fast/slow layer
uses "one-sided EMAs... backward derivatives" (§9 Transforms, already causal by construction)
— so `monthly_features` **may already be entirely causal**, unlike the incumbent's centered-
smoothing clustering features.
**How to avoid:** Before implementing HON-06's gating exactly like the salvaged module,
**verify with the planner/Phase-1 code** whether `monthly_features` has any non-causal
(centered) transform anywhere in it. If it is already fully causal (plausible, per D12/§9), the
gating function's job shifts from "pick between two files" to "assert the loaded checkpoint
contains no centered-window columns" — a different, simpler guard. This is flagged as an Open
Question below; the planner should resolve it by reading `transforms_monthly.py` directly
rather than assuming the incumbent's two-file pattern transfers unchanged.
**Warning signs:** A HON-06 task that references `monthly_features_supervised.parquet` when
no code in Phase 1 ever writes that file.

## Code Examples

### Config-hash for a registry row (deterministic, no new dependency)
```python
# Source: stdlib pattern, no external reference needed
import hashlib
import json

def config_hash(cfg: dict) -> str:
    return hashlib.md5(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:12]
```

### Reading the registry back for analysis
```python
# Source: pandas built-in JSONL support
import pandas as pd
trials = pd.read_json("registry/trials.jsonl", lines=True)
trials.groupby("config_hash")["metrics"].apply(lambda s: s)  # inspect per-config history
```

### Sojourn/detection-lag ratio (generic, testable without real regime labels)
```python
# Source: design §5.4 definition, implemented as a pure function over synthetic inputs
def sojourn_lag_ratio(median_sojourn_months: float, median_detection_lag_months: float) -> float:
    """§5.4: ratio largely determines whether regime timing can work.
    e.g. sojourn≈18m, lag≈2m → 9.0 (most of the regime captured).
    e.g. sojourn≈5m, lag≈2m → 2.5 (lag eats the trade)."""
    if median_detection_lag_months <= 0:
        raise ValueError("detection lag must be positive")
    return median_sojourn_months / median_detection_lag_months
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|---------------|--------|
| `mlfinlab` as the standard purged-CV library | Hand-rolled `BaseCrossValidator` subclass, or `skfolio.model_selection.CombinatorialPurgedCV` for portfolio-specific combinatorial variants | mlfinlab relicensing (date not pinned this session) `[CITED: web search result]` | Free-tier financial-ML projects must implement purging themselves or use narrower-scope libraries (`skfolio` is portfolio-optimization-focused, not a general drop-in) |
| `pickle` for model serialization | `joblib` | Already migrated in this repo (D14, P27 fix) | Not directly relevant to Phase 2, but any trivial-model checkpoint persisted by the walk-forward runner should use `CheckpointManager.save_model()` (joblib-backed), matching existing convention |

**Deprecated/outdated:**
- `mlfinlab` as a pip dependency: no longer a viable open-source install path; treat any
  tutorial/blog code importing it as reference-only, re-implement locally.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | López de Prado ch. 7's purge/embargo definitions and sizing conventions ("purge = label horizon", "embargo = small fraction of sample") — recalled from training knowledge, not re-verified against the physical book text this session (only cross-checked against web-search summaries of the same concepts, which agree directionally) | Standard Stack, Pattern 3, Pitfall 2 | If the exact formula/sizing convention differs from what's implemented, purged CV could under- or over-purge; mitigated by making purge/embargo explicit config parameters (not hardcoded) and requiring a test asserting non-trivial purge counts at realistic monthly horizons |
| A2 | `monthly_features` (Phase 1 output) contains no non-causal (centered-window) transforms, making the causal/noncausal split narrower in scope than the incumbent's `features.parquet`/`features_supervised.parquet` pair | Pitfall 5 | If Phase 1 actually did produce some centered feature, HON-06 as scoped here would fail to gate it — planner must verify by reading `transforms_monthly.py` before finalizing the gating module's exact behavior |
| A3 | "Embargo sized in months, driven by the target's forward horizon" is the correct monthly-cadence analog of López de Prado's daily-bar embargo convention | Pattern 3 | If wrong, embargo could be too aggressive (starves training data at K=5 monthly folds) or too weak (leaks autocorrelated features); exposed as a config knob specifically so this can be tuned without code changes |
| A4 | `DummyClassifier(strategy="prior")` predicting sign-of-next-month equities_tr return is an acceptable "trivial model" for HON-03's Phase-0 exit proof | Pattern 4 | Low risk — CONTEXT.md explicitly leaves this to Claude's discretion; any reasonable trivial model satisfies the exit criterion ("executes a trivial model end-to-end"), this is a recommendation not a hard requirement |

**If this table is empty:** N/A — see rows above; all four require light confirmation but
none blocks planning (each has an explicit config knob or fallback noted).

## Open Questions

1. **Does `monthly_features` contain any centered/non-causal transform?**
   - What we know: Phase 1's summary describes only one-sided constructs (ALFRED
     point-in-time alignment, backward derivatives implied by design §9's "one-sided EMAs...
     no centered/zero-phase smoothing"), and `realized_vol_1m` is explicitly documented as a
     naive trailing `|1-month-return|` proxy (D30/Phase-1 decisions) — no mention of any
     `center=True` rolling window anywhere in the Phase 1 summary.
   - What's unclear: whether this is a deliberate design guarantee (§9's explicit prohibition
     on centered smoothing in labeling features) that Phase 1 fully honored, or simply "not
     mentioned because not tested for."
   - Recommendation: planner has the executor read `transforms_monthly.py` directly
     (already-open in this research session, ~confirmed no `center=True` calls) before
     writing the HON-06 gating task — if truly causal-by-construction, HON-06's job is an
     *assertion/guard* ("no forbidden centered-window columns present"), not a *file-path
     selector* between two checkpoint variants. This changes the gating module's shape
     materially from a direct salvaged-module port.

2. **Exact purge/embargo default values for `cv:` config section.**
   - What we know: design D8 fixes decision horizons at 1m/3m with 12m as a strategic input;
     purge should equal "the label horizon" per López de Prado.
   - What's unclear: whether Phase 2 should ship a single global default (e.g., purge=12,
     embargo=1, sized for the longest horizon in use) or require every caller to specify
     purge explicitly with no default.
   - Recommendation: no silent default — `PurgedEmbargoedKFold(label_horizon=..., embargo=...)`
     should require both as explicit constructor args (no defaults) so every Phase 3+ caller
     is forced to think about its own target's horizon; document the convention in the
     module docstring rather than baking in one number.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| scikit-learn | HON-04 (`BaseCrossValidator`), HON-03 (`DummyClassifier`, `clone`) | ✓ | 1.9.0 | — |
| pandas / pyarrow | HON-02 (registry read), HON-05 (metrics artifacts) | ✓ | already core deps | — |
| git CLI | HON-02 (git SHA capture) | ✓ (repo is git-managed) | n/a | — |
| Network / FRED API | none — this phase operates entirely on already-ingested `monthly_features` checkpoints | n/a | — | — |

**Missing dependencies with no fallback:** none.
**Missing dependencies with fallback:** none — this phase adds zero new external
dependencies.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 8.0+ (already project standard) |
| Config file | `pyproject.toml` (`[tool.pytest.ini_options]`, existing) |
| Quick run command | `pytest tests/unit/test_platform_honesty*.py -x` (or per-module files listed below) |
| Full suite command | `pytest tests/ -v` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|---------------------|--------------|
| HON-01 | Default `get_platform_checkpoint_manager()` cannot read 2021+ rows; `get_holdout_checkpoint_manager()` can, only when called explicitly | unit (invariant-style, mirrors Phase 1's lean-set invariant test) | `pytest tests/unit/test_platform_holdout.py -x` | ❌ Wave 0 |
| HON-02 | `append_trial()` writes one JSON line per call, never truncates, includes config_hash/git_sha/timestamp | unit | `pytest tests/unit/test_platform_registry.py -x` | ❌ Wave 0 |
| HON-03 | `run_walkforward()` refits at each step on data strictly ≤ t, records a decision per step, and completes end-to-end on synthetic monthly data with a `DummyClassifier` | unit (synthetic frame, no network — mirrors `tests/integration/test_mini_pipeline.py` convention) | `pytest tests/unit/test_platform_walkforward.py -x` | ❌ Wave 0 |
| HON-04 | `PurgedEmbargoedKFold` purges label-overlapping rows and embargoes the post-test window; `get_n_splits()`/`.split()` match `BaseCrossValidator` contract; verified against `sklearn.utils.estimator_checks` or a manual contract test | unit | `pytest tests/unit/test_platform_cv.py -x` | ❌ Wave 0 |
| HON-05 | `compute_gap()`/`compute_detection_lag()`/`sojourn_lag_ratio()` produce correct values on synthetic smoothed/filtered label series; CLI prints a summary; artifact parquet/JSON written under `outputs/reports/model_metrics/` | unit | `pytest tests/unit/test_platform_gap_lag.py -x` | ❌ Wave 0 |
| HON-06 | Default training path refuses non-causal features with a loud, informative error; `allow_noncausal=True` opts out with a WARNING log | unit (mirrors salvaged module's own implicit test shape) | `pytest tests/unit/test_platform_gating.py -x` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** targeted module test file (`pytest tests/unit/test_platform_<module>.py -x`)
- **Per wave merge:** `pytest tests/unit/test_platform_*.py -v`
- **Phase gate:** Full suite green (`pytest tests/ -v`) before `/gsd-verify-work`

### Wave 0 Gaps
- [ ] `tests/unit/test_platform_holdout.py` — covers HON-01
- [ ] `tests/unit/test_platform_registry.py` — covers HON-02
- [ ] `tests/unit/test_platform_cv.py` — covers HON-04
- [ ] `tests/unit/test_platform_walkforward.py` — covers HON-03
- [ ] `tests/unit/test_platform_gating.py` — covers HON-06
- [ ] `tests/unit/test_platform_gap_lag.py` — covers HON-05
- [ ] No new framework install needed — pytest already fully configured for this project

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-------------------|
| V2 Authentication | no | n/a — no auth surface in this phase |
| V3 Session Management | no | n/a |
| V4 Access Control | yes (data-access control, not user auth) | The holdout guard *is* the access-control mechanism: default checkpoint manager physically cannot read `data/holdout/`; enforced by directory separation, not a runtime flag that could be silently flipped |
| V5 Input Validation | yes | `validate_platform_config()`-style collect-all-errors validation extended to the new `holdout:`/`registry:`/`cv:` config sections (mirrors existing `validate_platform_config` pattern in `platform/config.py`) |
| V6 Cryptography | no | n/a — `hashlib.md5` used here is for *content-addressing* (config dedup), not security; no attacker model requires collision-resistance for this use case |

### Known Threat Patterns for this stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|-----------------------|
| Silent holdout leakage (dev code accidentally reads 2021+ data) | Information Disclosure (of future data into a decision process — the domain-specific "threat" here is look-ahead bias, not a security breach) | Physical directory separation + hard-raise guard (Pitfall 1); no runtime toggle that bypasses the split |
| Registry tampering (retroactively editing a trial's logged metrics to look better) | Tampering | Append-only JSONL + git history as the tamper-evidence layer (D-01) — never open the file in `"w"` mode; a later git commit rewriting history would be visible in the git log itself |
| Purge/embargo misconfiguration silently defeating the leakage guard | Tampering (of the evaluation process, not data) | No silent defaults (Open Question 2) — force explicit `label_horizon`/`embargo` per caller; add a unit test asserting non-trivial purge counts at realistic values (Pitfall 2) |

## Sources

### Primary (HIGH confidence)
- `src/trading_crab_lib/checkpoints.py` — read directly, confirms `CheckpointManager(checkpoint_dir=...)` constructor signature and save/load/is_fresh/clear API `[VERIFIED: local file read]`
- `src/trading_crab_lib/platform/checkpoints.py`, `config.py`, `taxonomy.py` — read directly, confirm existing platform namespace conventions `[VERIFIED: local file read]`
- `ideas/gsd-salvage/prediction/feature_gating.py`, `model_metrics_artifacts.py`, `README.md` — read directly `[VERIFIED: local file read]`
- `platform_design/platform_design.md` §2 (D13), §5.4, §8, §9, §14, §22 — read directly `[VERIFIED: local file read]`
- Local `python3 -c "import sklearn; print(sklearn.__version__)"` → `1.9.0`; `BaseCrossValidator` MRO and `TimeSeriesSplit.__init__`/`.split()` signatures introspected directly `[VERIFIED: local introspection]`
- `gsd-tools query package-legitimacy check --ecosystem pypi timeseriescv` → `SUS` verdict, `weeklyDownloads: null`, published 2018-09-07 `[VERIFIED: package-legitimacy check]`

### Secondary (MEDIUM confidence)
- WebSearch: "purged and embargoed cross-validation overlapping labels Lopez de Prado implementation sklearn splitter" — confirms mlfinlab relicensing, confirms sklearn ships no native purge/embargo splitter (only `TimeSeriesSplit(gap=...)`) `[CITED: web search, multiple corroborating results including skfolio docs and GitHub issue threads describing the same gap]`
- WebSearch: "experiment trial registry JSONL append-only ledger schema machine learning config hash git sha" — confirms JSONL append-only + write-time schema validation is the documented pattern for experiment ledgers `[CITED: web search]`

### Tertiary (LOW confidence)
- López de Prado ch. 7's specific purge/embargo sizing formulas — recalled from training
  knowledge, not re-verified against the physical book text this session (see Assumptions
  Log A1) `[ASSUMED]`

## Metadata

**Confidence breakdown:**
- Standard stack (no new deps, reuse existing patterns): HIGH — every recommendation is
  either already installed or directly introspected this session
- Purged/embargoed CV design: MEDIUM — algorithm shape verified against sklearn's extension
  contract and cross-checked via web search, but exact purge/embargo sizing convention from
  López de Prado ch. 7 is `[ASSUMED]` (not re-read from source this session)
- Holdout/registry/gating/walk-forward architecture: HIGH — directly extends proven,
  already-reviewed code in this exact repo (`CheckpointManager`, salvaged modules)
- Gap/lag metrics scope for this phase: MEDIUM — the phase-boundary interpretation (generic
  functions now, real data in Phase 3) is a reasoned inference from design §14's Phase 0/1
  split, not an explicit CONTEXT.md statement; flagged as Pitfall 4 for planner attention

**Research date:** 2026-07-16
**Valid until:** 30 days (stable domain — no fast-moving external APIs; the one time-
sensitive claim, mlfinlab's licensing status, should be re-checked if this research is reused
after a long gap)
