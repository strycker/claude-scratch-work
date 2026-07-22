"""
Statistical jump-model regime labeler — exact DP decode (L1-01, design §4.1, R2).

A jump model is k-means clustering plus a per-jump penalty λ that discourages
state changes, solved by coordinate-descent alternation: an exact dynamic-
program finds the globally optimal state sequence given fixed centroids, then
centroids are recomputed as per-state means given the state sequence, repeated
to convergence with k-means warm start and multiple restarts.

The DP decode is provably exact — design §4.1 mandates "exact DP decode" and
this phase's CONTEXT specifics forbid greedy/heuristic shortcuts. See
tests/unit/test_platform_labeling.py::TestDPDecodeExact for the brute-force
enumeration invariant that proves this.

The labeler is intentionally non-causal at the batch level: the DP jointly
optimizes over the full time axis, so a label at month t is influenced by the
global fit including months after t. This is correct and by design for
ground-truth L1 labeling (design §14 D4/D5 two-stage split) — do not gate
labeler input through assert_causal_features() expecting failure; the
FEATURES are causal, the LABELING PROCEDURE is intentionally non-causal.

Usage::

    from trading_crab_lib.platform.labeling.jump_model import fit_jump_model
    result = fit_jump_model(X, K=5, lam=52.0, n_restarts=10)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)


def decode_states_dp(d: np.ndarray, lam: float) -> tuple[np.ndarray, float]:
    """Exact DP decode of the jump-model state sequence given fixed centroids.

    Args:
        d: (T, K) squared distances, d[t, k] = ||x_t - centroid_k||^2.
        lam: scalar per-jump penalty.

    Returns:
        (states, total_cost) where states is a length-T int array minimizing
        Σ_t d[t, s_t] + λ · Σ_t 1[s_t != s_{t-1}] exactly, and total_cost is
        that minimum value.

    Source: recurrence derived from Bemporad & Boyd (2018) "Fitting Jump
    Models" (Automatica 96) — the O(TK) min/second-min specialization
    exploiting that the jump penalty is a flat constant independent of the
    (i, j) state pair (see 03-RESEARCH.md Pattern 1).
    """
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
        # Inner loop left explicit (not vectorized): at K=5 this is 5 scalar
        # numpy ops per timestep, already sub-millisecond for T~770, and an
        # explicit loop is far easier to verify against the recurrence by eye
        # than a fully-vectorized K-dimensional expression (ponytail rung 7).
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


def soft_confidences(d: np.ndarray) -> np.ndarray:
    """Temperature-free softmax over negative squared distance (D-03).

    Args:
        d: (T, K) squared distances to canonicalized centroids.

    Returns:
        (T, K) row-stochastic confidence matrix (each row sums to 1.0).

    No temperature hyperparameter per D-03 ("temperature-free v1") — do not
    add a `/T` scaling knob; that is explicitly out of scope for this phase.
    """
    neg_d = -d
    neg_d = neg_d - neg_d.max(axis=1, keepdims=True)  # numeric stability only
    exp = np.exp(neg_d)
    return exp / exp.sum(axis=1, keepdims=True)


def standardize_features(X: pd.DataFrame) -> np.ndarray:
    """Winsorize to [1%, 99%] per column, then zero-mean/unit-variance scale.

    Args:
        X: lean feature DataFrame (columns in a fixed, caller-determined order).

    Returns:
        (T, d) numpy array, winsorized then StandardScaler-transformed. Column
        order is preserved from X (caller retains X.columns for canonicalization).
    """
    winsorized = X.clip(lower=X.quantile(0.01), upper=X.quantile(0.99), axis=1)
    return StandardScaler().fit_transform(winsorized)


def _recompute_centroids(
    X: np.ndarray, states: np.ndarray, K: int, prev_centroids: np.ndarray
) -> np.ndarray:
    """Per-state mean of X, freezing any zero-occupancy state at its previous
    centroid (Pitfall 2 — prevents NaN from poisoning subsequent DP iterations)."""
    centroids = prev_centroids.copy()
    for k in range(K):
        mask = states == k
        if mask.any():
            centroids[k] = X[mask].mean(axis=0)
        # else: keep previous centroids[k] unchanged (frozen-centroid fallback)
    return centroids


def fit_jump_model(
    X: np.ndarray,
    K: int,
    lam: float,
    *,
    n_restarts: int = 10,
    max_iter: int = 50,
    random_state: int = 42,
) -> dict:
    """Multi-restart k-means-warm-started jump-model alternation (design §4.1).

    Per restart: k-means warm start -> exact DP decode -> recompute centroids
    (freeze-on-empty) -> repeat until the state sequence stops changing or
    max_iter is hit. The lowest-total_cost restart is kept.

    Args:
        X: (T, d) standardized feature array (see standardize_features).
        K: number of states.
        lam: per-jump penalty.
        n_restarts: number of independent k-means-warm-started attempts.
        max_iter: hard cap on alternation iterations per restart.
        random_state: base seed; restart r uses random_state + r.

    Returns:
        dict with keys ``states`` (len-T int array), ``centroids`` ((K, d)
        array, no NaN even if a state was ever empty mid-alternation),
        ``total_cost`` (float), ``restart`` (int, winning restart index).

    Determinism: two calls with the same random_state return
    np.array_equal-identical states (KMeans n_init=1 + fixed seed per restart,
    deterministic DP decode, deterministic alternation order).
    """
    best: dict | None = None
    for r in range(n_restarts):
        km = KMeans(n_clusters=K, n_init=1, init="k-means++", random_state=random_state + r).fit(X)
        centroids = km.cluster_centers_.copy()
        prev_states = None
        states, total_cost = None, None
        for _ in range(max_iter):
            d = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
            states, total_cost = decode_states_dp(d, lam)
            if prev_states is not None and np.array_equal(states, prev_states):
                break  # converged
            centroids = _recompute_centroids(X, states, K, centroids)
            prev_states = states
        if best is None or total_cost < best["total_cost"]:
            best = {"states": states, "centroids": centroids, "total_cost": total_cost, "restart": r}
    return best


def canonicalize_states(
    states: np.ndarray, centroids: np.ndarray, feature_names: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Relabel state indices into a fixed, economically meaningful order.

    Sorts states by ascending centroid coordinate of trailing_return_1m (bear
    -> bull) so state numbering is stable across restarts and refreshes —
    otherwise the label-churn metric (L1-03) and D-04 auto profiles are
    meaningless, since k-means/jump-model cluster indices are arbitrary
    permutations by construction. Applying this twice is idempotent (the
    second call is already sorted, so order == identity).

    Falls back to centroid column 0 with a WARNING if trailing_return_1m is
    absent from feature_names (e.g. a caller running on a reduced feature set).

    Args:
        states: length-T int array of raw (pre-canonicalization) state labels.
        centroids: (K, d) array in the same raw state-index order as states.
        feature_names: column names matching centroids' second axis, in order.

    Returns:
        (new_states, new_centroids) — states relabeled 0..K-1 by ascending
        sort_col, centroids reordered to match.
    """
    if "trailing_return_1m" in feature_names:
        sort_col = feature_names.index("trailing_return_1m")
    else:
        sort_col = 0
        log.warning(
            "trailing_return_1m not in feature_names — falling back to centroid "
            "column 0 for canonicalization sort order"
        )
    order = np.argsort(centroids[:, sort_col])
    remap = {old: new for new, old in enumerate(order)}
    new_states = np.array([remap[s] for s in states])
    return new_states, centroids[order]
