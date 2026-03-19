"""Tests for regime.compute_forward_probabilities()."""

import numpy as np
import pandas as pd

from trading_crab_lib.regime import compute_forward_probabilities


def _make_labels():
    """Repeating 0,0,1,1 pattern — 12 quarters."""
    return pd.Series(
        [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
        name="balanced_cluster",
    )


def test_forward_prob_returns_dict_of_dataframes():
    labels = _make_labels()
    result = compute_forward_probabilities(labels, horizons=[1, 4])

    assert isinstance(result, dict)
    assert set(result.keys()) == {1, 4}
    for h, df in result.items():
        assert isinstance(df, pd.DataFrame)
        assert list(df.index) == [0, 1]
        assert list(df.columns) == [0, 1]


def test_forward_prob_rows_sum_to_one_or_less():
    labels = _make_labels()
    result = compute_forward_probabilities(labels, horizons=[1, 4, 8])

    for h, df in result.items():
        row_sums = df.sum(axis=1)
        # Each row should sum to <= 1.0 (it normalizes by total count)
        assert (row_sums <= 1.0 + 1e-9).all(), f"h={h}: row sums exceed 1.0"


def test_forward_prob_h1_matches_transition_matrix():
    """At horizon=1, forward probs should match the 1-step transition matrix."""
    labels = pd.Series([0, 0, 1, 0, 1])
    result = compute_forward_probabilities(labels, horizons=[1])

    from trading_crab_lib.regime import build_transition_matrix
    tm = build_transition_matrix(labels)

    # At h=1 the forward probability (reach j within 1 quarter) is the same
    # as the 1-step transition probability since the window is just [t+1].
    pd.testing.assert_frame_equal(result[1], tm, atol=1e-9)


def test_forward_prob_longer_horizon_increases_reach():
    """Longer horizons should generally increase the probability of reaching a regime."""
    labels = pd.Series([0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0])
    result = compute_forward_probabilities(labels, horizons=[1, 4])

    # P(reach regime 1 | currently 0) should be >= at h=4 vs h=1
    prob_h1 = result[1].loc[0, 1]
    prob_h4 = result[4].loc[0, 1]
    assert prob_h4 >= prob_h1 - 1e-9


def test_forward_prob_default_horizons():
    labels = _make_labels()
    result = compute_forward_probabilities(labels)
    assert set(result.keys()) == {1, 4, 8}


def test_forward_prob_handles_nan_labels():
    labels = pd.Series([0, np.nan, 1, 0, 1, np.nan, 0])
    result = compute_forward_probabilities(labels, horizons=[1])
    # NaN labels are dropped — should still produce valid output
    assert set(result[1].index) == {0, 1}
