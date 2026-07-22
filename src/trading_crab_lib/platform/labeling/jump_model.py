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
