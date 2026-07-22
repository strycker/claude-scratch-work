"""Unit tests for trading_crab_lib.platform.labeling.jump_model (L1-01...03).

Synthetic monthly DataFrame, no network — mirrors
tests/unit/test_platform_walkforward.py's synthetic-frame convention.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from trading_crab_lib.platform.config import load_platform_config
from trading_crab_lib.platform.labeling.jump_model import (
    canonicalize_states,
    decode_states_dp,
    fit_jump_model,
    soft_confidences,
    standardize_features,
)

N_MONTHS = 120


def _make_synthetic_monthly(n_months: int = N_MONTHS, seed: int = 42) -> pd.DataFrame:
    """Return a lean-shaped (13-column) synthetic monthly features DataFrame.

    Column names mirror the real taxonomy fast+slow tiers (config/platform_settings.yaml)
    so canonicalize_states can sort on trailing_return_1m exactly as it would on real data.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2010-01-31", periods=n_months, freq="ME")
    columns = [
        "curve_10y3m",
        "curve_10y2y",
        "credit_spread_baa_aaa",
        "fred_vix",
        "gold",
        "oil",
        "trailing_return_1m",
        "trailing_return_3m",
        "realized_vol_1m",
        "realized_vol_3m",
        "cape_shiller",
        "div_yield",
        "real_rate_level",
    ]
    data = {col: rng.normal(0, 1, n_months) for col in columns}
    return pd.DataFrame(data, index=idx)


def _brute_force_decode(d: np.ndarray, lam: float) -> tuple[np.ndarray, float]:
    """Reference implementation: enumerate all K^T state sequences exhaustively.

    Only tractable for small (T, K) — used solely as the RED-gate correctness
    oracle for decode_states_dp, never in production code.
    """
    T, K = d.shape
    best_cost = None
    best_seq = None
    for seq in itertools.product(range(K), repeat=T):
        cost = sum(d[t, seq[t]] for t in range(T))
        cost += lam * sum(1 for t in range(1, T) if seq[t] != seq[t - 1])
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_seq = seq
    return np.array(best_seq), float(best_cost)


# ── decode_states_dp: exact against brute force ─────────────────────────────


class TestDPDecodeExact:
    @pytest.mark.parametrize("T,K,lam,seed", [
        (4, 2, 0.0, 1),
        (4, 2, 0.5, 2),
        (4, 2, 5.0, 3),
        (5, 2, 1.0, 4),
        (4, 3, 0.0, 5),
        (4, 3, 0.8, 6),
        (5, 3, 2.0, 7),
    ])
    def test_matches_brute_force_on_small_TK(self, T, K, lam, seed):
        rng = np.random.default_rng(seed)
        d = rng.uniform(0.0, 10.0, size=(T, K))
        states, total_cost = decode_states_dp(d, lam)
        brute_states, brute_cost = _brute_force_decode(d, lam)
        assert total_cost == pytest.approx(brute_cost)
        assert np.array_equal(states, brute_states)

    def test_large_lambda_gives_constant_state(self):
        rng = np.random.default_rng(99)
        T, K = 6, 4
        d = rng.uniform(0.0, 10.0, size=(T, K))
        lam = 1000.0  # forbids all jumps
        states, _ = decode_states_dp(d, lam)
        assert len(np.unique(states)) == 1
        expected_state = int(np.argmin(d.sum(axis=0)))
        assert states[0] == expected_state


# ── soft_confidences: row-stochastic, temperature-free ──────────────────────


class TestSoftConfidences:
    def test_rows_sum_to_one(self):
        rng = np.random.default_rng(11)
        d = rng.uniform(0.0, 10.0, size=(20, 5))
        conf = soft_confidences(d)
        assert conf.shape == d.shape
        np.testing.assert_allclose(conf.sum(axis=1), 1.0)
        assert (conf >= 0).all() and (conf <= 1).all()

    def test_argmax_equals_argmin_distance(self):
        rng = np.random.default_rng(12)
        d = rng.uniform(0.0, 10.0, size=(20, 5))
        conf = soft_confidences(d)
        assert np.array_equal(conf.argmax(axis=1), d.argmin(axis=1))

    def test_shift_invariant(self):
        """Adding a per-row constant to -d (numeric-stability shift) must not
        change the resulting row-stochastic distribution."""
        rng = np.random.default_rng(13)
        d = rng.uniform(0.0, 10.0, size=(20, 5))
        shift = rng.uniform(-50, 50, size=(20, 1))
        conf_unshifted = soft_confidences(d)
        conf_shifted = soft_confidences(d - shift)
        np.testing.assert_allclose(conf_unshifted, conf_shifted, atol=1e-10)


# ── standardize_features: winsorize + zero-mean/unit-variance ───────────────


class TestStandardizeFeatures:
    def test_zero_mean_unit_variance_and_column_order(self):
        df = _make_synthetic_monthly(n_months=60, seed=21)
        X = standardize_features(df)
        assert X.shape == df.shape
        np.testing.assert_allclose(X.mean(axis=0), 0.0, atol=1e-8)
        np.testing.assert_allclose(X.std(axis=0), 1.0, atol=1e-8)


# ── fit_jump_model: determinism, K=5 separable blobs, empty-state safety ────


def _make_blob_array(centers: np.ndarray, n_per_block: int, *, seed: int, spread: float = 0.5) -> np.ndarray:
    """Contiguous blocks of tightly-clustered points around each center —
    a jump model with a modest lambda should recover exactly len(centers) states."""
    rng = np.random.default_rng(seed)
    return np.vstack([rng.normal(c, spread, size=(n_per_block, centers.shape[1])) for c in centers])


class TestFitJumpModel:
    def test_k5_separable_blobs_returns_5_states(self):
        centers = np.array([[0] * 13, [30] * 13, [60] * 13, [90] * 13, [120] * 13], dtype=float)
        X = _make_blob_array(centers, n_per_block=20, seed=1)
        result = fit_jump_model(X, K=5, lam=5.0, n_restarts=5, max_iter=50, random_state=1)
        assert len(np.unique(result["states"])) == 5

    def test_determinism(self):
        centers = np.array([[0] * 5, [10] * 5, [20] * 5], dtype=float)
        X = _make_blob_array(centers, n_per_block=15, seed=0)
        r1 = fit_jump_model(X, K=6, lam=200.0, n_restarts=3, max_iter=50, random_state=42)
        r2 = fit_jump_model(X, K=6, lam=200.0, n_restarts=3, max_iter=50, random_state=42)
        assert np.array_equal(r1["states"], r2["states"])

    def test_empty_state_freeze_produces_no_nan(self):
        """K=6 requested but only 3 real clusters in the data — several states
        end up with zero occupancy every alternation iteration. The freeze-
        on-empty guard must keep centroids finite (Pitfall 2)."""
        centers = np.array([[0] * 5, [10] * 5, [20] * 5], dtype=float)
        X = _make_blob_array(centers, n_per_block=15, seed=0)
        result = fit_jump_model(X, K=6, lam=200.0, n_restarts=3, max_iter=50, random_state=42)
        assert np.isnan(result["centroids"]).any() == False  # noqa: E712 — explicit per acceptance criteria
        assert len(np.unique(result["states"])) < 6  # confirms an empty state actually occurred


# ── canonicalize_states: economic sort, idempotent, permutation-stable ──────


class TestCanonicalize:
    FEATURE_NAMES = ["gold", "trailing_return_1m", "oil"]

    def test_orders_by_trailing_return_1m(self):
        states = np.array([0, 1, 2, 1, 0])
        # centroid col 1 (trailing_return_1m) descending by raw state index
        centroids = np.array([[0.0, 5.0, 0.0], [0.0, -3.0, 0.0], [0.0, 1.0, 0.0]])
        new_states, new_centroids = canonicalize_states(states, centroids, self.FEATURE_NAMES)
        # ascending trailing_return_1m order should be: raw state 1 (-3) -> 0,
        # raw state 2 (1) -> 1, raw state 0 (5) -> 2
        assert np.array_equal(new_centroids[:, 1], np.array([-3.0, 1.0, 5.0]))
        expected_remap = {1: 0, 2: 1, 0: 2}
        expected_states = np.array([expected_remap[s] for s in states])
        assert np.array_equal(new_states, expected_states)

    def test_idempotent(self):
        rng = np.random.default_rng(5)
        states = rng.integers(0, 3, size=10)
        centroids = rng.normal(0, 1, size=(3, 3))
        once_states, once_centroids = canonicalize_states(states, centroids, self.FEATURE_NAMES)
        twice_states, twice_centroids = canonicalize_states(once_states, once_centroids, self.FEATURE_NAMES)
        assert np.array_equal(once_states, twice_states)
        np.testing.assert_allclose(once_centroids, twice_centroids)

    def test_permutation_stability(self):
        """Two fits that are permutations of each other (same physical
        clusters, different arbitrary integer labels) canonicalize identically."""
        rng = np.random.default_rng(6)
        base_states = rng.integers(0, 3, size=10)
        base_centroids = rng.normal(0, 1, size=(3, 3))

        perm = {0: 2, 1: 0, 2: 1}
        permuted_states = np.array([perm[s] for s in base_states])
        permuted_centroids = np.empty_like(base_centroids)
        for old, new in perm.items():
            permuted_centroids[new] = base_centroids[old]

        canon_a, centroids_a = canonicalize_states(base_states, base_centroids, self.FEATURE_NAMES)
        canon_b, centroids_b = canonicalize_states(permuted_states, permuted_centroids, self.FEATURE_NAMES)
        assert np.array_equal(canon_a, canon_b)
        np.testing.assert_allclose(centroids_a, centroids_b)


# ── labeling config: defensive .get(), not a required section ───────────────


class TestLabelingConfig:
    def test_config_exposes_labeling_defaults_via_get(self):
        cfg = load_platform_config()
        labeling_cfg = cfg.get("labeling", {})
        assert labeling_cfg.get("K", 5) == 5
        assert labeling_cfg.get("lambda", 52.0) == 52.0
        assert labeling_cfg.get("n_restarts", 10) == 10
        assert labeling_cfg.get("embargo_months", 12) == 12

    def test_labeling_not_a_required_section(self):
        from trading_crab_lib.platform.config import _REQUIRED_PLATFORM_SECTIONS

        assert "labeling" not in _REQUIRED_PLATFORM_SECTIONS
        # Config with no "labeling" key at all must still validate fine.
        minimal_cfg = {section: {} for section in _REQUIRED_PLATFORM_SECTIONS}
        load_platform_config(minimal_cfg)  # must not raise


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-q"])
