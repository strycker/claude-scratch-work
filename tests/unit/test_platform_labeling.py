"""Unit tests for trading_crab_lib.platform.labeling.jump_model (L1-01...03).

Synthetic monthly DataFrame, no network — mirrors
tests/unit/test_platform_walkforward.py's synthetic-frame convention.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from trading_crab_lib.platform.labeling.jump_model import decode_states_dp, soft_confidences

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


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-q"])
