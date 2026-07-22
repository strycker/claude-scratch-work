# Phase 3: Regime Labeling & Prediction - Research

**Researched:** 2026-07-22
**Domain:** Statistical jump-model regime labeling (Bemporad-Boyd / Nystrup) + calibrated
multinomial nowcasting on a monthly macro/asset feature spine
**Confidence:** MEDIUM (algorithm mechanics HIGH; default λ value and restart-count
conventions MEDIUM/LOW — literature gives ranges and heuristics, not a single
universally-correct number for this feature set)

## Summary

Phase 3 builds two new platform subsystems, both self-contained in
`src/trading_crab_lib/platform/`, both zero-new-dependency (numpy/pandas/scikit-learn
already installed at 1.9.0/2.4.6/3.0.3), both plugging into Phase 2's frozen honesty
interfaces rather than reinventing them.

**L1 labeler** (`platform/labeling/`): a statistical jump model — k-means clustering plus
a per-jump penalty λ solved by coordinate-descent alternation (exact DP for the state
sequence given fixed centroids; cluster means for centroids given the state sequence),
multi-restart, k-means warm start. Input is the lean (fast ∪ slow taxonomy tier, 13
columns) `monthly_features` checkpoint from Phase 1. Output is a hard state label 0..4 per
month, a softmax-over-negative-squared-distance soft confidence vector per D-03, an
occupancy/sojourn diagnostics artifact (report-only per D-02), an auto-generated one-line
economic profile per state (D-04), and a label-churn metric computed against the
*previous* run's persisted labels (L1-03/§5.4).

**L2 nowcaster** (`platform/prediction/`): a `CalibratedClassifierCV`-wrapped multinomial
`LogisticRegression`, trained on causal `monthly_features` columns against the labeler's
output with the trailing 12 months of labels structurally excluded (D-01 embargo), using
`PurgedEmbargoedKFold` as the calibration CV splitter. Evaluation reports transition-window
accuracy (±3 months around ex-post label transitions) separately from overall accuracy —
the headline metric per design §5.1, because a trivial persistence classifier scores ~90%
overall and hides at exactly the moments that matter. An empirical transition matrix
(row-normalized K×K count table over the full smoothed label sequence) is computed
alongside as the L2-02 diagnostic.

The labeler is intentionally **non-causal at the batch level** (the DP jointly optimizes
over the full time axis, so a label at month t is influenced by the *global* fit including
months after t) — this is correct and by design for L1 ground-truth labeling (design §14
D4/D5: two-stage split), exactly mirroring the incumbent's centered-vs-causal feature split
(ADR #1) but applied to *labels* instead of *features*. The nowcaster, by contrast, must be
strictly causal both in its feature inputs (already guaranteed — Phase 1/2's
`monthly_features` has no centered columns; `assert_causal_features()` passes trivially)
and in its training targets (guaranteed only by the *structural* 12-month embargo this
phase must implement — CONTEXT specifics call this out explicitly).

**Primary recommendation:** implement the DP decode as a vectorized O(TK) numpy routine
(the min/second-min trick — trivial extra code, and future-proofs against v2's larger
(K,λ) grid sweeps even though at K=5, T≈770 the naive O(TK²) loop is already sub-millisecond
and would work fine too); default `λ = 4 × len(lean_feature_set)` (≈52 for the current
13-feature lean set, expressed as a per-feature multiplier so it self-scales if the lean
set changes, per design §22's "λ normalized by feature dimension" convention) as a
literature-informed starting constant, with a documented one-time human sanity check
against the real report-only diagnostics rather than a runtime auto-search (matches D-02's
tracer-bullet "always completes" spirit and CONTEXT's discretion note); `n_restarts=10`
(a conventional order-of-magnitude for jump-model multi-start, and explicitly enough
per Nystrup et al.'s reported robustness at this problem scale); `method="sigmoid"` for
`CalibratedClassifierCV` (isotonic needs ≳1000 samples per class to avoid overfitting the
calibration map; the dev-window sample here is ~700 months over 5 classes, well under that
bar).

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Jump-model DP decode + alternation | Library (`platform/labeling/`) | — | Pure numpy/pandas compute, no I/O boundary |
| Regime label + confidence persistence | Library / checkpoint layer (`platform/checkpoints.py`) | — | Reuses existing platform `CheckpointManager` verbatim, no new persistence code |
| Label-churn + acceptance-criteria diagnostics | Library (`platform/labeling/diagnostics.py`) | Reporting (future weekly report, Phase 4) | Computed here; consumed downstream, not rendered here |
| Nowcaster training-set builder (embargo) | Library (`platform/prediction/`) | Honesty rails (`platform/honesty/cv.py`) | Embargo logic lives with the nowcaster; CV mechanics reused from Phase 2 |
| Calibrated nowcaster fit/predict | Library (`platform/prediction/nowcaster.py`) | — | sklearn `Pipeline`-free composition (Calibrated wraps LogisticRegression directly) |
| Empirical transition matrix | Library (`platform/prediction/transition_matrix.py`) | — | Pure pandas groupby/crosstab, no model state |
| Trial registry logging | Honesty rails (`platform/honesty/registry.py`) | — | Reused as-is; nowcaster walk-forward runs through `run_walkforward` per CONTEXT |

## Standard Stack

### Core

| Library | Version (installed) | Purpose | Why Standard |
|---------|---------|---------|--------------|
| scikit-learn | 1.9.0 `[VERIFIED: pip in this environment]` | `KMeans` (warm start), `LogisticRegression` (multinomial by default for `lbfgs`), `CalibratedClassifierCV`, `StandardScaler` | Already a project dependency (root `pyproject.toml`); zero-new-dependency phase |
| numpy | 2.4.6 `[VERIFIED: pip in this environment]` | DP decode, squared-distance broadcasting, softmax | Already a dependency |
| pandas | 3.0.3 `[VERIFIED: pip in this environment]` | Label/confidence Series/DataFrame, transition-matrix crosstab | Already a dependency |
| scipy | 1.17.1 `[VERIFIED: pip in this environment]` | Not required for the DP itself (plain numpy broadcasting suffices at K=5); available if winsorization via `scipy.stats.mstats.winsorize` is preferred over a `.clip(quantile)` one-liner | Already a dependency; not adding new usage is fine (ponytail rung 3: `.clip()` + `.quantile()` on a DataFrame is stdlib-pandas, no scipy needed) |

**Note on `LogisticRegression`:** the installed sklearn 1.9.0 signature
(`[VERIFIED: inspect.signature in this environment]`) has **no `multi_class` parameter** —
it was removed after deprecation; multinomial behavior is now automatic for solvers that
support it (`lbfgs`, the default). Do not write `LogisticRegression(multi_class="multinomial")`
— it will raise `TypeError: unexpected keyword argument`. Just
`LogisticRegression(max_iter=..., random_state=...)` with the default `lbfgs` solver is
correct and already multinomial.

**Note on `CalibratedClassifierCV`:** installed signature
(`[VERIFIED: inspect.signature in this environment]`) is
`CalibratedClassifierCV(estimator=None, *, method='sigmoid', cv=None, n_jobs=None, ensemble='auto')`
— the constructor param is `estimator`, not the older `base_estimator` name from
pre-1.2 sklearn. `cv` accepts any `BaseCrossValidator`-compatible splitter (int, generator,
iterable, or an object exposing `.split(X, y)`), so `PurgedEmbargoedKFold` plugs in directly:
`CalibratedClassifierCV(LogisticRegression(...), method="sigmoid", cv=PurgedEmbargoedKFold(n_splits=5, label_horizon=12, embargo=1))`.

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `sklearn.cluster.KMeans` | 1.9.0 | k-means warm start for the jump model's centroid initialization | Once per restart, `n_init=1` with a distinct `random_state` per restart (the outer multi-restart loop supplies the diversity; `KMeans`'s own internal `n_init` search is redundant work here) |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Multinomial `LogisticRegression` (calibrated) | Gradient-boosted trees (design §5.1 mentions this as a valid discriminative nowcaster) | Design explicitly names logistic regression first and CONTEXT locks "calibrated multinomial logistic" as the v1 choice — GBT is not in scope; would also need its own calibration and complicate feature-importance interpretability at ~700 samples |
| Softmax-over-negative-squared-distance confidences (D-03) | Companion HMM forward-backward γ | D-03 explicitly defers this to v2 (t-HMM benchmark, L1-V2-01) — do not import `hmmlearn` into the platform namespace in v1 |
| O(TK) vectorized DP (min/second-min trick) | Naive O(TK²) nested loop | At K=5, T≈770 both run in well under a second — either is fine; O(TK) is barely more code and is the right habit if K ever grows in the v2 (K,λ) grid (§22: K ∈ {4,5,6,7}) |
| Economic-sort state canonicalization (see Pitfalls) | Hungarian matching (`scipy.optimize.linear_sum_assignment`) against the previous run's centroids | Hungarian matching is explicitly named in design §22 for *subsample-stability* testing (v2, L1-V2-01) — a different concern from same-K refresh-to-refresh churn measurement. Economic-sort is cheaper, zero-new-usage-of-scipy.optimize, and sufficient for v1's report-only churn metric; document as the chosen convention so it doesn't get conflated with the deferred v2 Hungarian-matching feature |

**Installation:** none — no new packages. `pip show scikit-learn numpy pandas scipy` all
resolve inside the existing environment.

## Package Legitimacy Audit

**No new external packages required for this phase** — labeling and nowcasting are built
entirely on scikit-learn, numpy, and pandas, all already installed and already declared in
the project's `pyproject.toml`/`requirements.txt`. The Package Legitimacy Gate is not
applicable; skipping the verdict table per the gate's own scope ("every phase that installs
external packages").

**Packages removed due to [SLOP] verdict:** none (no new packages evaluated).
**Packages flagged as suspicious [SUS]:** none.

## Architecture Patterns

### System Architecture Diagram

```
Phase 1 checkpoint                    Phase 3: L1 Labeler                          Phase 3: L2 Nowcaster
┌─────────────────────┐        ┌──────────────────────────────┐         ┌──────────────────────────────────┐
│ monthly_features     │        │ platform/labeling/            │         │ platform/prediction/               │
│ (platform checkpoint │───────▶│                                │         │                                    │
│  namespace)          │ lean_  │ 1. select lean_feature_set()  │         │ 1. build_nowcaster_training_set() │
│                       │ feature│    columns (13 cols, taxonomy)│         │    — X = causal monthly_features   │
└─────────────────────┘ _set() │ 2. winsorize + StandardScaler │         │      cols (unchanged from L1's    │
                                │ 3. multi-restart alternation: │         │      lean_feature_set() input)    │
        ┌───────────────────┐  │    k-means warm start          │         │    — y = labeler's hard state     │
        │ previous           │◀─┤    → exact DP decode (O(TK))  │         │      series MINUS trailing        │
        │ regime_labels      │  │    → recompute centroids       │         │      12 months (D-01 embargo,     │
        │ checkpoint (if any)│  │    → repeat to convergence     │         │      applied ONCE here, not a     │
        └───────────────────┘  │ 4. pick best-J restart          │         │      per-fold CV thing)           │
                 │              │ 5. canonicalize state order    │         │ 2. CalibratedClassifierCV(         │
                 │ churn        │    (economic sort)              │         │      LogisticRegression(...),     │
                 │ comparison   │ 6. softmax(-dist²) confidences │         │      method="sigmoid",            │
                 │              │    (D-03, temperature-free)    │         │      cv=PurgedEmbargoedKFold(...))│
                 ▼              │ 7. occupancy/sojourn diagnostics│         │      .fit(X, y)                   │
        ┌───────────────────┐  │    (D-02: WARNING-only, always │         │ 3. transition_window_accuracy()   │
        │ label_churn metric │  │    completes) + auto profiles  │         │    on a held-out split (±3mo      │
        │ (L1-03/§5.4)        │  │    (D-04)                       │         │    around ex-post transitions)    │
        └───────────────────┘  │ 8. label_churn vs previous      │         │ 4. empirical_transition_matrix()  │
                                │    checkpoint (LOAD BEFORE SAVE)│         │    from full labeler output       │
                                └──────────────┬─────────────────┘         │    (diagnostic, not embargoed)    │
                                               │                            └──────────────┬─────────────────┘
                                               ▼                                            ▼
                        platform checkpoints: regime_labels,              platform checkpoints: nowcaster
                        regime_confidences, regime_diagnostics,           model artifact, transition_matrix,
                        regime_profiles, transition matrix input          transition-window accuracy report
                                               │                                            │
                                               └──────────────── consumed by ───────────────┘
                                                        Phase 4 (asset prediction,
                                                        allocation, weekly report)
                                                        — NOT built in Phase 3
```

### Recommended Project Structure

```
src/trading_crab_lib/platform/
├── labeling/
│   ├── __init__.py
│   ├── jump_model.py       # standardize_features, decode_states_dp,
│   │                        # fit_jump_model (alternation + multi-restart),
│   │                        # canonicalize_states, soft_confidences
│   └── diagnostics.py       # occupancy_and_sojourns, label_churn,
│                             # auto_profile (D-04), report_labeling_diagnostics
│                             # (mirrors honesty/gap_lag.py's report_* pattern)
└── prediction/
    ├── __init__.py
    ├── nowcaster.py         # build_nowcaster_training_set (embargo),
    │                        # fit_nowcaster, transition_window_accuracy
    └── transition_matrix.py # empirical_transition_matrix
```

This mirrors the existing `platform/honesty/` and `platform/ingestion/` convention of one
small, single-purpose file per concern rather than one large module (see `CLAUDE.md`
Function Design: "Library functions broken into helpers when they exceed 40 lines").

### Pattern 1: Exact DP decode given fixed centroids (O(TK) vectorized)

**What:** Given per-timestep squared distances `d[t, k] = ||x_t - μ_k||²` (shape `(T, K)`,
computed once via numpy broadcasting), find the state sequence minimizing
`Σ_t d[t, s_t] + λ · Σ_t 1[s_t ≠ s_{t-1}]` exactly.

**When to use:** Inside the alternation loop, every iteration, given the current centroids.

**Example (numpy, no new dependency):**
```python
# Source: recurrence derived from Bemporad & Boyd (2018) "Fitting Jump Models"
# (Automatica 96) — the discrete step is "standard discrete dynamic programming"
# per the paper; this is the min/second-min O(TK) specialization exploiting that
# the jump penalty is a flat constant independent of (i, j) state pair.
import numpy as np


def decode_states_dp(d: np.ndarray, lam: float) -> tuple[np.ndarray, float]:
    """d: (T, K) squared distances. Returns (states, total_cost)."""
    T, K = d.shape
    cost = np.empty((T, K))
    backptr = np.empty((T, K), dtype=int)
    cost[0] = d[0]
    backptr[0] = -1  # no predecessor
    for t in range(1, T):
        prev = cost[t - 1]
        idx1 = int(np.argmin(prev))
        min1 = prev[idx1]
        # second-best (any index != idx1) — needed so "stay in idx1" doesn't
        # illegally use its own value as the "cheapest jump-from" option.
        masked = prev.copy()
        masked[idx1] = np.inf
        idx2 = int(np.argmin(masked))
        min2 = masked[idx2]
        for k in range(K):
            stay_cost = prev[k]
            jump_from = idx2 if k == idx1 else idx1
            jump_cost = (min2 if k == idx1 else min1) + lam
            if stay_cost <= jump_cost:
                cost[t, k] = d[t, k] + stay_cost
                backptr[t, k] = k
            else:
                cost[t, k] = d[t, k] + jump_cost
                backptr[t, k] = jump_from
    states = np.empty(T, dtype=int)
    states[-1] = int(np.argmin(cost[-1]))
    total_cost = float(cost[-1, states[-1]])
    for t in range(T - 2, -1, -1):
        states[t] = backptr[t + 1, states[t + 1]]
    return states, total_cost
```
The inner `for k in range(K)` loop is left unvectorized deliberately — at K=5 it is 5
iterations of scalar numpy ops per timestep, already sub-millisecond for T≈770, and keeping
it as an explicit loop is far easier to verify against the recurrence by eye than a
fully-vectorized K-dimensional numpy expression would be (ponytail rung 7: correctness-
readable beats maximally-vectorized when the constant factor doesn't matter).

**Degenerate empty-state handling (pitfall):** after `decode_states_dp` returns, some state
`k` may have zero assigned timesteps in `states`. The centroid-recompute step
(`μ_k = mean(x_t for t where states[t]==k)`) must **not** call `.mean()` on an empty
selection (produces `NaN`, silently poisons every subsequent DP iteration's distance
column for that state). Freeze that state's centroid at its previous value instead:
```python
for k in range(K):
    mask = states == k
    if mask.any():
        centroids[k] = X[mask].mean(axis=0)
    # else: keep previous centroids[k] unchanged (ponytail: frozen-centroid
    # fallback, simplest fix that keeps the alternation loop well-defined;
    # an empty state usually self-heals or gets absorbed by a subsequent
    # iteration once neighboring centroids move)
```

### Pattern 2: Alternation loop with multi-restart

**What:** Given `K`, `λ`, standardized features `X`, repeat "warm-start k-means → DP decode
→ recompute centroids from DP output" until the state sequence stops changing; repeat the
whole thing `n_restarts` times from different k-means seeds; keep the lowest-cost result.

**Example:**
```python
# Source: alternation structure per Bemporad & Boyd (2018) §2-3 (coordinate
# descent: convex parameter-fit step + discrete DP step); multi-restart +
# k-means warm start per design §4.1 and CONTEXT's locked spec.
from sklearn.cluster import KMeans


def fit_jump_model(X: np.ndarray, K: int, lam: float, *,
                    n_restarts: int = 10, max_iter: int = 50,
                    random_state: int = 42) -> dict:
    best = None
    for r in range(n_restarts):
        km = KMeans(n_clusters=K, n_init=1, init="k-means++",
                     random_state=random_state + r).fit(X)
        centroids = km.cluster_centers_.copy()
        prev_states = None
        states, total_cost = None, None
        for _ in range(max_iter):
            d = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
            states, total_cost = decode_states_dp(d, lam)
            if prev_states is not None and np.array_equal(states, prev_states):
                break  # converged
            for k in range(K):
                mask = states == k
                if mask.any():
                    centroids[k] = X[mask].mean(axis=0)
            prev_states = states
        if best is None or total_cost < best["total_cost"]:
            best = {"states": states, "centroids": centroids,
                    "total_cost": total_cost, "restart": r}
    return best
```
**Convergence criterion:** the state sequence is unchanged between successive alternation
iterations (`np.array_equal`) — standard for k-means-style coordinate descent; falls back
to `max_iter` as a hard cap (50 is generous for K=5 at this sample size; the alternation
typically converges in single digits of iterations in the k-means literature this pattern
descends from).

### Pattern 3: State canonicalization (fixes label-switching)

**What:** After picking the best restart, and again on every refresh, sort state indices
into a fixed, economically meaningful order so state "2" means the same thing across
restarts and across refreshes — otherwise the churn metric (L1-03) and the D-04 auto
profiles are meaningless (k-means/jump-model cluster indices are arbitrary permutations by
construction).

**When to use:** Immediately after selecting the best-restart result, before persisting.

**Example:**
```python
# Convention chosen for v1: sort states by ascending mean of the standardized
# trailing_return_1m centroid coordinate — an economically monotonic axis
# (bear -> bull), cheap, deterministic, no new dependency. This is NOT the
# same mechanism as design §22's Hungarian-matching subsample-stability test
# (deferred to v2, L1-V2-01) — that compares labels ACROSS different training
# subsamples for stability scoring; this sorts labels WITHIN one fit for a
# stable numbering convention.
def canonicalize_states(states: np.ndarray, centroids: np.ndarray,
                         feature_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    sort_col = feature_names.index("trailing_return_1m")
    order = np.argsort(centroids[:, sort_col])
    remap = {old: new for new, old in enumerate(order)}
    new_states = np.array([remap[s] for s in states])
    return new_states, centroids[order]
```

### Pattern 4: Soft confidences (D-03, temperature-free softmax)

```python
def soft_confidences(d: np.ndarray) -> np.ndarray:
    """d: (T, K) squared distances to canonicalized centroids.
    Returns (T, K) row-stochastic confidence matrix."""
    neg_d = -d
    neg_d = neg_d - neg_d.max(axis=1, keepdims=True)  # numeric stability only
    exp = np.exp(neg_d)
    return exp / exp.sum(axis=1, keepdims=True)
```
No temperature hyperparameter per D-03 ("temperature-free v1") — do not add a `/T` scaling
knob; that is explicitly out of scope for this phase.

### Pattern 5: Structural embargo for the nowcaster training set (D-01)

**What:** The trailing 12 months of labels are physically excluded from ever being a
training target — applied **once**, when building `(X, y)`, not per-CV-fold.

```python
def build_nowcaster_training_set(features_df: pd.DataFrame, labels: pd.Series, *,
                                  embargo_months: int = 12) -> tuple[pd.DataFrame, pd.Series]:
    cutoff = labels.index.max() - pd.DateOffset(months=embargo_months)
    eligible_labels = labels.loc[labels.index <= cutoff]
    common = features_df.index.intersection(eligible_labels.index)
    X = features_df.loc[common]
    y = eligible_labels.loc[common]
    return X, y
```
The invariant test (per CONTEXT specifics: "with a test proving it") asserts
`y.index.max() <= labels.index.max() - pd.DateOffset(months=embargo_months)` — a real
assertion on real output, not a mock of the embargo boundary.

### Pattern 6: Calibrated nowcaster fit through `PurgedEmbargoedKFold`

```python
# Source: sklearn.calibration.CalibratedClassifierCV signature verified in
# this environment (sklearn 1.9.0) — `cv` accepts any object exposing
# .split(X, y); PurgedEmbargoedKFold (platform/honesty/cv.py) already
# implements that interface (subclasses BaseCrossValidator).
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

from trading_crab_lib.platform.honesty.cv import PurgedEmbargoedKFold

cv = PurgedEmbargoedKFold(n_splits=5, label_horizon=12, embargo=1)  # reuses
# platform_settings.yaml's existing cv.default_label_horizon_months /
# cv.default_embargo_months (Phase 2) — a DIFFERENT concern from D-01's
# 12-month L1->L2 embargo: this purges CV folds for potential feature-window
# overlap WITHIN the already-embargoed training set, applied every fold.
model = CalibratedClassifierCV(
    LogisticRegression(max_iter=1000, random_state=42),
    method="sigmoid",
    cv=cv,
)
model.fit(X_causal, y_embargoed)
proba = model.predict_proba(X_today)  # shape (1, K) — never argmax downstream
```

### Pattern 7: Transition-window accuracy (§5.1 headline metric)

```python
def transition_window_accuracy(y_true: pd.Series, y_pred: pd.Series, *,
                                window_months: int = 3) -> dict:
    """Returns {'transition_accuracy': ..., 'steady_state_accuracy': ...,
    'overall_accuracy': ...} — always report all three together so the
    persistence-baseline trap (~90% overall) is never presented alone."""
    transitions = y_true.index[y_true.ne(y_true.shift(1))][1:]  # skip t=0 (no prior)
    near_transition = pd.Series(False, index=y_true.index)
    for t in transitions:
        lo, hi = t - pd.DateOffset(months=window_months), t + pd.DateOffset(months=window_months)
        near_transition |= (y_true.index >= lo) & (y_true.index <= hi)
    correct = y_true.eq(y_pred)
    return {
        "overall_accuracy": float(correct.mean()),
        "transition_accuracy": float(correct[near_transition].mean()) if near_transition.any() else float("nan"),
        "steady_state_accuracy": float(correct[~near_transition].mean()) if (~near_transition).any() else float("nan"),
    }
```

### Pattern 8: Empirical transition matrix (L2-02)

```python
def empirical_transition_matrix(states: pd.Series) -> pd.DataFrame:
    """Row-normalized K x K count table: P(next state = j | current state = i)."""
    pairs = pd.DataFrame({"from": states.iloc[:-1].values, "to": states.iloc[1:].values})
    counts = pd.crosstab(pairs["from"], pairs["to"])
    return counts.div(counts.sum(axis=1), axis=0)
```

### Anti-Patterns to Avoid

- **Reusing `market_code`-style label-in-gap-fill leakage (P1 analog):** do not let the
  jump-model's OWN state sequence feed back into feature computation for the SAME fit
  (e.g., a "regime age" feature computed from this run's own labels used as an INPUT to
  this run's labeling). Design §4.3's regime-age feature is explicitly a *nowcaster/
  transition-model* input (§5.2), and even there it is v2-deferred (L2-V2-02) — do not
  pull it into the v1 labeler or nowcaster.
- **Treating the labeler's full-history batch fit as if it were causal:** it is not, by
  design (D4/D5 two-stage split) — never gate `monthly_features` → labeler input through
  `assert_causal_features(..., allow_noncausal=False)` expecting it to fail; the FEATURES
  are causal (Phase 1 guarantee), the LABELING PROCEDURE (global DP) is intentionally
  non-causal. Only the L2 nowcaster's *training set* needs the structural embargo.
- **`multi_class="multinomial"` kwarg:** removed from `LogisticRegression` in the installed
  sklearn version — will raise `TypeError`. See Standard Stack note above.
- **Isotonic calibration at this sample size:** `CalibratedClassifierCV(method="isotonic")`
  overfits the calibration map badly under ~1000 calibration samples per class
  `[ASSUMED — well-established sklearn community guidance; not independently re-verified
  via docs fetch this session, see Assumptions Log A1]`. At ~700 monthly dev-window rows
  split 5 ways and further split by CV folds, isotonic is the wrong default; use `sigmoid`.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| K-means clustering / warm start | Custom Lloyd's-algorithm loop | `sklearn.cluster.KMeans` | Already installed, battle-tested, `k-means++` init built in |
| Multinomial logistic regression | Custom softmax-regression gradient descent | `sklearn.linear_model.LogisticRegression` | Already installed; `lbfgs` solver is multinomial by default |
| Probability calibration | Custom Platt-scaling or isotonic-regression fit | `sklearn.calibration.CalibratedClassifierCV` | Already installed; handles the CV-internal fit/calibrate split correctly, which is easy to get subtly wrong by hand (calibrating on the same data used to fit the base estimator overstates calibration quality) |
| Purged/embargoed CV | New CV splitter for this phase | `platform/honesty/cv.py::PurgedEmbargoedKFold` | Built and tested in Phase 2 for exactly this purpose; CONTEXT explicitly requires reusing it |
| Exact discrete-state optimal path given a penalty | Greedy/heuristic assignment | The DP recurrence in Pattern 1 | Design §4.1 mandates "exact DP decode" — CONTEXT specifics explicitly forbid greedy/heuristic shortcuts |

**Key insight:** every piece of this phase already exists as a well-tested primitive in
scikit-learn except the DP decode itself (which has no off-the-shelf scikit-learn
equivalent — jump models are not a built-in sklearn estimator family) and the
embargo/canonicalization/churn glue code, which is inherently project-specific.

## Common Pitfalls

### Pitfall 1: Label switching across restarts and refreshes

**What goes wrong:** Two runs of the same jump model on nearly-identical data can produce
labels where "state 0" in run A corresponds to "state 3" in run B, purely because k-means-
style cluster indices are an arbitrary permutation of the optimization's internal
bookkeeping — not because the underlying regimes changed.
**Why it happens:** Neither k-means nor the DP decode has any notion of a canonical state
ordering; `argmin` ties/near-ties can flip which physical cluster gets which integer index.
**How to avoid:** Canonicalize immediately after selecting the best restart, using a fixed,
economically-motivated sort key (Pattern 3) — applied identically on every refresh so state
numbering is stable over time, not just within one fit.
**Warning signs:** Label-churn metric (L1-03) reads suspiciously high (e.g., >50% of
trailing months "changed") even when a visual/economic inspection shows the underlying
regime pattern is basically unchanged — that's the signature of un-canonicalized label
switching, not real churn.

### Pitfall 2: Degenerate empty states inside the DP alternation

**What goes wrong:** A centroid-recompute step divides by zero / calls `.mean()` on an
empty selection when a state has no assigned timesteps mid-alternation, producing `NaN`
centroids that then poison every subsequent DP iteration.
**Why it happens:** Especially likely with K=5 on a 13-dimensional feature space and a
large λ (which suppresses jumps, potentially starving a state entirely if its k-means-warm-
start seed happened not to land near a real cluster).
**How to avoid:** Freeze-on-empty (Pattern 1) — keep the previous iteration's centroid for
any state with zero assigned points that iteration, rather than crashing or propagating
`NaN`.
**Warning signs:** `fit_jump_model` silently returns fewer than K distinct states in the
final `states` array, or diagnostics show a state with occupancy exactly 0%.

### Pitfall 3: Churn metric with no previous-run artifact to compare against

**What goes wrong:** The very first run of the labeler (or any run against a fresh/cleared
checkpoint directory) has no prior `regime_labels` checkpoint to diff against — a naive
implementation either crashes on `FileNotFoundError` or silently reports a meaningless 0%
churn.
**Why it happens:** `CheckpointManager.load()` raises `FileNotFoundError` if the checkpoint
is missing (documented incumbent convention) — this is the FIRST time this checkpoint name
will ever exist in the platform namespace.
**How to avoid:** Load-before-save ordering is mandatory: attempt to load the *previous*
`regime_labels` checkpoint, catch `FileNotFoundError` explicitly, and if caught, log
`INFO: no previous regime_labels checkpoint — churn metric unavailable on first run` and
report churn as `NaN` (not 0.0 — 0.0 would misleadingly read as "perfectly stable"). Only
**after** computing churn against the (possibly absent) previous artifact does the new fit
get saved, overwriting the old one.
**Warning signs:** A test that runs the labeler twice in a row on the same data and asserts
churn == 0.0 the second time will catch a broken load-before-save ordering; a test that
only runs the labeler once cannot catch this pitfall at all — the plan must include a
two-run invariant test.

### Pitfall 4: Confusing the two different "embargo" concepts in this phase

**What goes wrong:** D-01's 12-month L1→L2 label-freshness embargo (applied once, when
building the nowcaster's training set) gets conflated with `PurgedEmbargoedKFold`'s
per-fold purge/embargo (applied every CV fold, inside `CalibratedClassifierCV`) — leading
to either double-embargoing (wasting data) or, worse, implementing only one and believing
both are covered.
**Why it happens:** Both use the word "embargo" and both exist in the same phase; the
platform_settings.yaml `cv:` section (Phase 2) already has `default_embargo_months: 1`
which is a DIFFERENT number from D-01's `labeling.embargo_months: 12`.
**How to avoid:** Two config keys, two purposes, both present: `labeling.embargo_months`
(D-01, structural, applied once) and `cv.default_label_horizon_months` /
`cv.default_embargo_months` (Phase 2, per-fold, reused as-is inside
`PurgedEmbargoedKFold(...)` for the nowcaster's internal CV). Name them distinctly in code
and comments; do not introduce a third "embargo" concept.
**Warning signs:** A single `embargo_months` config value being read by both the training-
set builder and the CV splitter constructor is the tell that the two concerns got merged.

### Pitfall 5: λ scale is feature-set-dependent, not portable across phases

**What goes wrong:** Hardcoding a raw numeric λ (e.g., copying the "λ=80" example found in
the literature search for a different S&P-500-only feature set) without accounting for the
13-dimensional lean feature set's different scale — squared-distance terms scale roughly
linearly with feature count for standardized features, so a λ tuned for a 2-3 feature model
will under-penalize (too many jumps) on a 13-feature model.
**Why it happens:** Published examples use different feature counts and different
standardization conventions; λ is not a universal constant.
**How to avoid:** Express the config default as `4 × len(lean_feature_set(cfg))` in the
derivation comment (even if the config stores the resolved number, e.g. `52`), so if the
lean feature set grows or shrinks in a later phase, the next engineer knows to re-derive
rather than reuse the stale literal.
**Warning signs:** Occupancy/sojourn diagnostics (D-02, report-only) show either near-100%-
occupancy-in-one-state (λ far too high) or monthly flip-flopping with median sojourn ~1
month (λ far too low, effectively degenerating toward plain k-means) — either is a strong
signal to hand-adjust the config default before trusting downstream results, even though
D-02 makes this non-blocking.

## Code Examples

Verified patterns from official sources (installed-environment signatures verified this
session via `inspect.signature`; DP recurrence derived from Bemporad & Boyd 2018 as
summarized via WebSearch — see Sources):

### Full lean-feature pull for labeling

```python
# Source: existing project pattern, platform/honesty/gating.py + taxonomy.py
from trading_crab_lib.platform.checkpoints import get_platform_checkpoint_manager
from trading_crab_lib.platform.taxonomy import lean_feature_set
from trading_crab_lib.platform.config import load_platform_config

cfg = load_platform_config()
monthly_features = get_platform_checkpoint_manager().load("monthly_features")
lean_cols = sorted(lean_feature_set(cfg) & set(monthly_features.columns))
X_df = monthly_features[lean_cols].dropna()  # jump model needs no NaNs
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|---------------|--------|
| Balanced KMeans (incumbent quarterly pipeline, `KMeansConstrained`) | Statistical jump model (k-means + persistence penalty, exact DP) | This phase (design R2) | Adds temporal persistence natively; removes the incumbent's forced equal-cluster-size constraint (which was a workaround for small-sample regime statistics, not a modeling choice) |
| HMM Viterbi decoding + Baum-Welch EM (design §4.2, benchmark only) | Jump model coordinate descent (this phase's production path) | Design decision, not a historical trend | Jump models are reported empirically more stable out-of-sample (less refresh churn) than HMM labels per design §4.1 — directly mitigates the embargo/churn concern this phase must report on |

**Deprecated/outdated:**
- `LogisticRegression(multi_class=...)`: removed from the sklearn API surface installed in
  this environment (1.9.0) — see Standard Stack note.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Isotonic calibration needs ≳1000 samples/class to avoid overfitting vs sigmoid | Standard Stack note, Anti-Patterns, Summary | If wrong and isotonic is actually fine at ~700 rows/5 classes, the recommendation to default to `sigmoid` is merely suboptimal, not incorrect — sigmoid is a safe, non-harmful default either way. Low risk; direct sklearn-docs confirmation attempt this session returned HTTP 403 (proxy-blocked), so this is carried from training knowledge, not re-verified in-session. |
| A2 | `λ = 4 × len(lean_feature_set)` (≈52) is a reasonable literature-informed starting default for this feature set | Summary, Pattern in Pitfall 5 | Medium risk — the one concrete published λ value found (≈80, S&P-500-only 2000-2015 study) used a different feature count/scale, so it is not directly transferable. D-02 makes this non-blocking (report-only diagnostics + human hand-adjustment), so a wrong default is cheap to fix, not a correctness bug. |
| A3 | `n_restarts=10` is a conventional default for jump-model multi-restart | Summary, Standard Stack | Low risk — CONTEXT's own research-questions text already cites "Nystrup et al. use ~10" as a starting assumption; this session's WebSearch found jump models described as "robust to initialization" but did not find a specific restart count in the sources actually fetched. Increasing to 20-30 costs only runtime (trivial at this problem size), so erring low is cheap to correct. |
| A4 | The lean feature set (13 columns) has no NaN gaps across the full 1962+ span once Phase 1's live FRED ingestion run completes | Environment Availability, Code Examples | Medium risk — Phase 1's STATE.md lists "live 1962+ run pending FRED_API_KEY" as an outstanding human-verification item; if agency-tier-adjacent features (`real_rate_level`, needing `fred_cpi`) have pre-vintage gaps, `.dropna()` on the lean set could truncate the effective labeling window well after 1962. The plan should include a check/report of the actual non-null date range once real data is ingested, not assume 1962-01 coverage. |

## Open Questions

1. **Does the real 1962+ `monthly_features` checkpoint already exist on disk?**
   - What we know: Phase 1's STATE.md marks the live 1962+ ingestion run as a pending
     human-verification item (requires `FRED_API_KEY`); this session found no
     `data/checkpoints/platform/` directory present.
   - What's unclear: Whether the plan should assume real data is available by execution
     time, or whether Phase 3's tasks must be developed and tested purely against
     synthetic monthly frames (as Phase 2's tests already do) with a separate
     `checkpoint:human-verify` task for the real 1962+ run.
   - Recommendation: follow the established Phase 1/2 pattern — build and test against
     synthetic monthly DataFrames (no network), and add one `checkpoint:human-verify` task
     for running the labeler + nowcaster against real ingested data once `FRED_API_KEY` is
     confirmed set, exactly mirroring how Phase 1 handled its own live-run gap.

2. **Exact numeric λ to hardcode as the config default.**
   - What we know: the design gives the interpretive framework (§4.1, §22) and one
     external numeric example (λ≈80) for a differently-scaled feature set; CONTEXT gives
     Claude explicit discretion here.
   - What's unclear: the precise "right" number for this project's 13-feature lean set
     without actually running the labeler against real standardized 1962+ data and
     eyeballing the report-only diagnostics.
   - Recommendation: ship `λ = 52` (`= 4 × 13`) as the initial config default with the
     derivation formula documented in a code/config comment, and treat the first real run
     against ingested data as the sanity-check point (D-02's report-only diagnostics are
     designed exactly for this) — adjust by hand if occupancy/sojourn look clearly broken,
     no runtime auto-search needed.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| scikit-learn | Jump-model KMeans warm start, LogisticRegression, CalibratedClassifierCV | ✓ | 1.9.0 | — |
| numpy | DP decode, softmax confidences | ✓ | 2.4.6 | — |
| pandas | Label/confidence/transition-matrix DataFrames | ✓ | 3.0.3 | — |
| `monthly_features` platform checkpoint (real 1962+ data) | End-to-end real-data run of the labeler/nowcaster | ✗ (not yet ingested — no `data/checkpoints/platform/` directory found this session) | — | Develop/test against synthetic monthly DataFrames (Phase 1/2 established pattern); gate the real-data run behind a `checkpoint:human-verify` task pending `FRED_API_KEY` per Phase 1's outstanding human-verification item |

**Missing dependencies with no fallback:** none — every code-level dependency is already
installed.

**Missing dependencies with fallback:** the real ingested 1962+ `monthly_features`
checkpoint (fallback: synthetic-frame development/testing, real run deferred to a
`checkpoint:human-verify` task).

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest 8.0+ (`[tool.pytest.ini_options]` in root `pyproject.toml`) |
| Config file | `pyproject.toml` (existing; no new config needed) |
| Quick run command | `pytest tests/unit/test_platform_labeling.py tests/unit/test_platform_nowcaster.py -x` |
| Full suite command | `pytest tests/ -v` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| L1-01 | Jump-model labeler produces K=5 states via k-means-warm-started DP with per-jump penalty, multi-restart | unit | `pytest tests/unit/test_platform_labeling.py::TestFitJumpModel -x` | ❌ Wave 0 |
| L1-01 | DP decode is exact — invariant test: brute-force enumeration matches DP output on a small synthetic T,K | unit | `pytest tests/unit/test_platform_labeling.py::TestDPDecodeExact -x` | ❌ Wave 0 |
| L1-02 | Labels persisted with soft confidences; trailing 12 months embargoed from L2 training (structural, real assertion) | unit | `pytest tests/unit/test_platform_nowcaster.py::TestEmbargoInvariant -x` | ❌ Wave 0 |
| L1-03 | Label-churn metric computed against previous run; first-run-no-prior handled without crash | unit | `pytest tests/unit/test_platform_labeling.py::TestLabelChurn -x` | ❌ Wave 0 |
| L2-01 | Nowcaster returns calibrated probability distribution (not argmax) summing to ~1.0 | unit | `pytest tests/unit/test_platform_nowcaster.py::TestNowcasterCalibratedOutput -x` | ❌ Wave 0 |
| L2-01 | Transition-window accuracy reported separately from overall accuracy | unit | `pytest tests/unit/test_platform_nowcaster.py::TestTransitionWindowAccuracy -x` | ❌ Wave 0 |
| L2-02 | Empirical transition matrix rows sum to 1.0 (or 0 for a never-observed state) | unit | `pytest tests/unit/test_platform_nowcaster.py::TestTransitionMatrix -x` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest tests/unit/test_platform_labeling.py tests/unit/test_platform_nowcaster.py -x` (fast, synthetic-frame, no network)
- **Per wave merge:** `pytest tests/ -v`
- **Phase gate:** Full suite green before `/gsd-verify-work`

### Wave 0 Gaps
- [ ] `tests/unit/test_platform_labeling.py` — covers L1-01, L1-02 (label side), L1-03
- [ ] `tests/unit/test_platform_nowcaster.py` — covers L1-02 (embargo boundary), L2-01, L2-02
- [ ] No new fixtures needed — follow `tests/unit/test_platform_walkforward.py`'s synthetic
      monthly DataFrame convention (`_make_synthetic_monthly()`-style local helper per test
      file; no shared conftest fixture required for a 13-column synthetic feature frame)

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | This phase has no auth surface — local batch compute on checkpointed parquet files |
| V3 Session Management | no | No session/request handling in this phase |
| V4 Access Control | no | Single-user local CLI/library code, no multi-tenant access boundary |
| V5 Input Validation | yes | Config values (`labeling.K`, `labeling.lambda`, `labeling.n_restarts`, `labeling.embargo_months`) read defensively via `.get()` with sane defaults per the established `platform/config.py` pattern; no schema validation added to `_REQUIRED_PLATFORM_SECTIONS` per CONTEXT's explicit "don't extend" instruction — malformed values would surface as a `KeyError`/`TypeError` at fit time rather than a validated pre-flight error, which is an accepted v1 tradeoff already established by the 02-01 pattern this phase reuses |
| V6 Cryptography | no | No secrets, tokens, or cryptographic material touched by this phase — reuses the existing `FRED_API_KEY` handling from Phase 1 unchanged (this phase does not call FRED itself; it reads the already-persisted `monthly_features` checkpoint) |

### Known Threat Patterns for this stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Pickle/joblib deserialization of the persisted nowcaster model artifact | Tampering | The incumbent's known pitfall P27 (`CLAUDE.md`) applies equally here: use `joblib.dump`/`joblib.load` (not raw `pickle`) for the `CalibratedClassifierCV` artifact, consistent with the incumbent's D14 migration; never load a model artifact of unknown provenance |
| Config-driven `λ`/`K`/`embargo_months` read via unchecked `.get()` | Tampering / Denial of Service (degenerate fit) | Not a security boundary in the traditional sense (single-user local project, config file is git-tracked and human-edited) — but a malformed `embargo_months` (e.g., negative) could silently produce an empty or inverted training set. Recommend the training-set builder assert `embargo_months >= 0` and raise `ValueError` rather than silently producing nonsense — cheap defensive check, not full schema validation |

## Sources

### Primary (HIGH confidence)
- Installed-environment signatures verified via `python3 -c "import inspect; ..."` this
  session: `sklearn.calibration.CalibratedClassifierCV.__init__`,
  `sklearn.linear_model.LogisticRegression.__init__` (sklearn 1.9.0)
- `src/trading_crab_lib/platform/honesty/{cv,walkforward,gating,holdout,registry,gap_lag}.py`,
  `platform/checkpoints.py`, `platform/transforms_monthly.py`, `platform/taxonomy.py`,
  `platform/config.py` — read directly this session
- `config/platform_settings.yaml` — read directly this session
- `.planning/phases/03-regime-labeling-prediction/03-CONTEXT.md` — read directly this session
- `platform_design/platform_design.md` §4.1, §4.3, §4.4, §5.1-5.4, §14, §22 — read directly
  this session

### Secondary (MEDIUM confidence)
- Bemporad & Boyd (2018), "Fitting Jump Models," *Automatica* 96 — summarized via WebSearch
  (algorithm structure: coordinate descent, convex parameter step + DP discrete step,
  multi-restart with warm start for the mode-assignment task) — full PDF fetch blocked by
  HTTP 403 this session, so the recurrence in Pattern 1 is this researcher's derivation
  from the stated objective function, not a verbatim transcription of the paper's equations
- WebSearch summaries of Nystrup et al. jump-model literature (regime persistence via λ,
  k-means-based local submodels, one published λ≈80 example for a differently-scaled
  S&P-500-only feature set) — see Assumptions Log A2

### Tertiary (LOW confidence)
- Isotonic-vs-sigmoid calibration sample-size guidance (Assumptions Log A1) — carried from
  training knowledge; a direct sklearn-docs fetch attempt this session was blocked
  (HTTP 403 via the environment's outbound proxy) and not re-attempted through an
  authenticated MCP tool

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — every library is already installed and its API verified live in
  this environment
- Architecture (module layout, embargo structure, DP mechanics): HIGH — directly derives
  from the design doc's explicit objective function and CONTEXT's locked decisions, plus
  established platform/ code conventions
- Default λ value: MEDIUM/LOW — literature-informed but not precisely transferable; D-02's
  report-only acceptance criteria make this self-correcting rather than blocking
- Pitfalls: HIGH — derived from direct code reading (CheckpointManager's
  `FileNotFoundError`-on-missing behavior, installed sklearn's removed `multi_class` param)
  plus first-principles reasoning about k-means/DP label-switching, which is well-
  established general knowledge about this model family

**Research date:** 2026-07-22
**Valid until:** 30 days (stable library APIs; the one time-sensitive fact — installed
package versions — should be re-verified if the environment's dependency lockfile changes)
