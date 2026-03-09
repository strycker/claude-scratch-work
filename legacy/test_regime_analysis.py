"""Tests for regime_analysis module — transition matrices and profiling."""

import numpy as np
import pandas as pd
import pytest

from trading_crab.regime_analysis import (
    compute_forward_probabilities,
    compute_transition_matrix,
)


class TestTransitionMatrix:
    """Test empirical transition matrix computation."""

    def test_rows_sum_to_one(self):
        """Each row of the transition matrix should sum to 1."""
        labels = np.array([0, 1, 0, 1, 0, 2, 1, 2, 0, 1])
        matrix = compute_transition_matrix(labels)
        row_sums = matrix.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, 1.0)

    def test_correct_shape(self):
        """Matrix should be n_regimes × n_regimes."""
        labels = np.array([0, 1, 2, 0, 1, 2])
        matrix = compute_transition_matrix(labels)
        assert matrix.shape == (3, 3)

    def test_deterministic_sequence(self):
        """A perfectly repeating sequence should have known transitions."""
        # 0 → 1 → 2 → 0 → 1 → 2 → ...
        labels = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        matrix = compute_transition_matrix(labels)
        # 0 always transitions to 1
        assert matrix.loc[0, 1] == 1.0
        assert matrix.loc[0, 0] == 0.0
        assert matrix.loc[0, 2] == 0.0

    def test_single_regime(self):
        """A constant sequence: regime stays in itself with probability 1."""
        labels = np.array([0, 0, 0, 0, 0])
        matrix = compute_transition_matrix(labels)
        assert matrix.loc[0, 0] == 1.0

    def test_two_element_sequence(self):
        """Smallest valid input: one transition."""
        labels = np.array([0, 1])
        matrix = compute_transition_matrix(labels)
        assert matrix.loc[0, 1] == 1.0


class TestForwardProbabilities:
    """Test forward-looking regime probabilities."""

    def test_returns_dict_of_dataframes(self):
        """Should return one DataFrame per horizon."""
        labels = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
        result = compute_forward_probabilities(labels, horizons=[1, 2])
        assert isinstance(result, dict)
        assert set(result.keys()) == {1, 2}
        for h, df in result.items():
            assert isinstance(df, pd.DataFrame)

    def test_horizon_1_matches_transition_matrix(self):
        """At horizon=1, forward probs should equal the transition matrix."""
        labels = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        fwd = compute_forward_probabilities(labels, horizons=[1])
        trans = compute_transition_matrix(labels)
        pd.testing.assert_frame_equal(fwd[1], trans, atol=1e-10)

    def test_probabilities_increase_with_horizon(self):
        """Longer horizons should generally have higher probabilities
        of encountering any given regime."""
        labels = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
        fwd = compute_forward_probabilities(labels, horizons=[1, 4])
        # For a cycling sequence, longer horizon → higher prob of seeing each regime
        for regime in [0, 1, 2]:
            for from_regime in [0, 1, 2]:
                assert fwd[4].loc[from_regime, regime] >= fwd[1].loc[from_regime, regime] - 0.01

    def test_probabilities_bounded(self):
        """All probabilities should be between 0 and 1."""
        labels = np.random.RandomState(42).randint(0, 3, size=50)
        fwd = compute_forward_probabilities(labels, horizons=[1, 4, 8])
        for h, df in fwd.items():
            assert (df >= 0).all().all(), f"Negative probability at horizon {h}"
            assert (df <= 1).all().all(), f"Probability > 1 at horizon {h}"
