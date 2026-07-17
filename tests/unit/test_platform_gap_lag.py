"""Unit tests for trading_crab_lib.platform.honesty.gap_lag (HON-05).

Follows the incumbent tests/unit/test_platform_taxonomy.py structure: a
class per behavior, docstring per test. All computation runs against
hand-constructed synthetic series — the real smoothed/filtered regime
series arrive in Phase 3 (RESEARCH.md Pitfall 4); this module's job is the
generic, tested compute + reporting surface Phase 3 plugs into unchanged.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from trading_crab_lib.platform.honesty.gap_lag import (
    compute_detection_lag,
    compute_gap,
    sojourn_lag_ratio,
)

# ── compute_gap ──────────────────────────────────────────────────────────────────


class TestComputeGap:
    def test_compute_gap_value(self):
        """gap = smoothed_perf - filtered_perf (§5.4: 'measured hindsight content')."""
        assert compute_gap(0.9, 0.6) == pytest.approx(0.3)

    def test_compute_gap_can_be_negative(self):
        """Filtered outperforming smoothed is a valid (negative) gap, not an error."""
        assert compute_gap(0.4, 0.6) == pytest.approx(-0.2)


# ── compute_detection_lag ────────────────────────────────────────────────────────


class TestComputeDetectionLag:
    def test_known_construction_single_transition(self):
        """A transition at position p with the filtered-prob series first crossing
        the threshold at p+k returns lag == k."""
        probs = pd.Series([0.1, 0.2, 0.3, 0.4, 0.7, 0.8])
        result = compute_detection_lag([1], probs, threshold=0.5)
        assert result["lags"] == [3.0]
        assert result["median"] == pytest.approx(3.0)

    def test_immediate_crossing_at_transition_is_zero_lag(self):
        """The threshold crossing AT the transition position itself is lag 0."""
        probs = pd.Series([0.1, 0.9, 0.9])
        result = compute_detection_lag([1], probs, threshold=0.5)
        assert result["lags"] == [0.0]

    def test_median_across_multiple_transitions(self):
        """Multiple transitions each get their own lag; median is computed over all resolved lags."""
        probs = pd.Series([0.9, 0.1, 0.2, 0.9, 0.1, 0.2, 0.3, 0.9])
        result = compute_detection_lag([0, 4], probs, threshold=0.8)
        assert result["lags"] == [0.0, 3.0]
        assert result["median"] == pytest.approx(1.5)

    def test_unresolved_transition_is_nan_and_excluded_from_median(self):
        """A transition whose probability never crosses the threshold is NaN, not
        silently dropped or defaulted to 0 — and is excluded from the median."""
        probs = pd.Series([0.9, 0.1, 0.2, 0.9, 0.1, 0.2, 0.3])
        result = compute_detection_lag([0, 4], probs, threshold=0.8)
        assert result["lags"][0] == pytest.approx(0.0)
        assert math.isnan(result["lags"][1])
        assert result["median"] == pytest.approx(0.0)

    def test_all_transitions_unresolved_median_is_nan(self):
        probs = pd.Series([0.1, 0.2, 0.3])
        result = compute_detection_lag([0], probs, threshold=0.9)
        assert math.isnan(result["lags"][0])
        assert math.isnan(result["median"])

    def test_empty_transitions_raises(self):
        probs = pd.Series([0.1, 0.2])
        with pytest.raises(ValueError):
            compute_detection_lag([], probs, threshold=0.5)

    def test_empty_filtered_probs_raises(self):
        with pytest.raises(ValueError):
            compute_detection_lag([0], pd.Series([], dtype=float), threshold=0.5)

    def test_out_of_bounds_transition_raises(self):
        """A transition position beyond filtered_probs' length is malformed input, not a silent NaN."""
        probs = pd.Series([0.1, 0.2, 0.3])
        with pytest.raises(ValueError):
            compute_detection_lag([10], probs, threshold=0.5)

    def test_negative_transition_raises(self):
        probs = pd.Series([0.1, 0.2, 0.3])
        with pytest.raises(ValueError):
            compute_detection_lag([-1], probs, threshold=0.5)


# ── sojourn_lag_ratio ──────────────────────────────────────────────────────────────


class TestSojournLagRatio:
    def test_typical_values(self):
        """§5.4 examples: sojourn=18/lag=2 -> 9.0 (most of the regime captured);
        sojourn=5/lag=2 -> 2.5 (lag eats the trade)."""
        assert sojourn_lag_ratio(18, 2) == pytest.approx(9.0)
        assert sojourn_lag_ratio(5, 2) == pytest.approx(2.5)

    def test_non_positive_lag_raises(self):
        with pytest.raises(ValueError):
            sojourn_lag_ratio(10, 0)
        with pytest.raises(ValueError):
            sojourn_lag_ratio(10, -1)
