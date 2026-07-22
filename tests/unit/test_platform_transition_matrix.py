"""Tests for the L2-02 empirical transition matrix diagnostic (design §5.1/R3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_crab_lib.platform.prediction.transition_matrix import (
    empirical_transition_matrix,
)


class TestEmpiricalTransitionMatrix:
    def test_occupied_rows_sum_to_one(self):
        """Row-normalized: every occupied 'from' row sums to 1.0."""
        rng = np.random.default_rng(42)
        states = pd.Series(rng.integers(0, 4, size=200))

        matrix = empirical_transition_matrix(states)

        row_sums = matrix.sum(axis=1)
        for total in row_sums:
            assert total == pytest.approx(1.0)

    def test_hand_computed_known_counts(self):
        """Matches hand-computed row-normalized counts on a known sequence."""
        # Transitions (from -> to): 0->1, 1->0, 0->1, 1->1, 1->0
        # from=0: {1: 2}                -> row sums to 1.0, all mass on to=1
        # from=1: {0: 2, 1: 1}          -> P(0)=2/3, P(1)=1/3
        states = pd.Series([0, 1, 0, 1, 1, 0])

        matrix = empirical_transition_matrix(states)

        assert matrix.loc[0, 1] == pytest.approx(1.0)
        assert matrix.loc[1, 0] == pytest.approx(2 / 3)
        assert matrix.loc[1, 1] == pytest.approx(1 / 3)

    def test_never_a_from_state_does_not_crash(self):
        """A state that only ever appears as the final label (never a 'from')
        does not raise ZeroDivision/NaN crash — the row is simply absent."""
        # State 2 only appears at the very end, so it is never a 'from' state.
        states = pd.Series([0, 1, 0, 1, 2])

        matrix = empirical_transition_matrix(states)

        # No crash, and state 2 has no 'from' row (or, if present via crosstab
        # column inclusion, must not contain NaN).
        if 2 in matrix.index:
            assert not matrix.loc[2].isna().any()

    def test_pure_function_no_io(self):
        """Import-and-call on a plain Series works with no I/O or model state."""
        states = pd.Series([0, 0, 1, 1, 0])
        matrix = empirical_transition_matrix(states)
        assert isinstance(matrix, pd.DataFrame)


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-q"])
