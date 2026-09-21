"""Unit tests for trading_crab_lib.platform.evaluation.sojourn_lag (EVAL-03,
design §5.4 headline number).

Follows the incumbent test_platform_gap_lag.py structure: a class per
behavior, docstring per test. All computation runs against hand-built
synthetic series (no network, no checkpoint dependency).

TestPerTransitionTargetStateProbs is the load-bearing test for cross-AI
review finding F1: it proves ``compute_sojourn_lag_headline`` converts a
MULTICLASS filtered-probs matrix into per-transition lags by checking each
transition against P(its OWN target state) — never a class-agnostic
max-across-classes series, which would systematically understate detection
lag (the "fooled by its own backtest" failure).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_crab_lib.platform.evaluation.sojourn_lag import (
    build_filtered_probs_matrix,
    compute_sojourn_lag_headline,
)
from trading_crab_lib.platform.honesty.gap_lag import compute_detection_lag, sojourn_lag_ratio

# ── compute_sojourn_lag_headline ──────────────────────────────────────────────────


class TestSojournLagHeadline:
    def test_known_construction_matches_sojourn_lag_ratio(self):
        """Synthetic full_sample_states run-length series with a known median
        sojourn (runs of 4, 3, 6, 2 -> median 3.5) and a synthetic
        filtered-probs matrix whose target-state columns cross the threshold
        at known, distinct lags per transition (2, 1, 0 -> pooled median
        1.0). The returned ratio must equal
        gap_lag.sojourn_lag_ratio(3.5, 1.0) directly.

        states:  [0]*4 + [1]*3 + [0]*6 + [2]*2  (length 15)
        runs:    4, 3, 6, 2  -> median sojourn 3.5
        transitions: position 4 -> state 1; position 7 -> state 0;
                     position 13 -> state 2
        """
        full_index = pd.date_range("2000-01-31", periods=15, freq="ME")
        states = [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 2, 2]
        full_sample_states = pd.Series(states, index=full_index)

        # col0 -> target state 0's own transition is at position 7: crosses at
        # position 8 (lag 1).
        col0 = [0.0] * 7 + [0.2, 0.85] + [0.85] * 6
        # col1 -> target state 1's own transition is at position 4: crosses at
        # position 6 (lag 2).
        col1 = [0.1] * 6 + [0.75] * 9
        # col2 -> target state 2's own transition is at position 13: crosses
        # immediately (lag 0).
        col2 = [0.0] * 13 + [0.9, 0.9]
        filtered_probs_matrix = pd.DataFrame({0: col0, 1: col1, 2: col2}, index=full_index)

        result = compute_sojourn_lag_headline(full_sample_states, filtered_probs_matrix, act_threshold=0.7)

        assert result["median_sojourn"] == pytest.approx(3.5)
        assert result["median_lag"] == pytest.approx(1.0)
        assert result["ratio"] == pytest.approx(sojourn_lag_ratio(3.5, 1.0))

    def test_reports_transition_and_resolved_counts(self):
        # 3 transitions; state 1's probability never crosses 0.7 → 1 unresolved,
        # so n_transitions=3 but n_resolved=2. Surfaces the headline's sample size.
        # The two resolved transitions cross one period late (lag 1) so the
        # pooled median lag is strictly positive (sojourn_lag_ratio rejects 0).
        full_index = pd.date_range("2000-01-31", periods=15, freq="ME")
        states = [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 2, 2]
        full_sample_states = pd.Series(states, index=full_index)
        col0 = [0.0] * 8 + [0.85] * 7          # state-0 transition @7 crosses @8 (lag 1)
        col1 = [0.1] * 15                        # state-1 transition @4 NEVER crosses → unresolved
        col2 = [0.0] * 14 + [0.9]                # state-2 transition @13 crosses @14 (lag 1)
        fpm = pd.DataFrame({0: col0, 1: col1, 2: col2}, index=full_index)

        result = compute_sojourn_lag_headline(full_sample_states, fpm, act_threshold=0.7)

        assert result["n_transitions"] == 3
        assert result["n_resolved"] == 2
        assert result["act_threshold"] == pytest.approx(0.7)


class TestDistinctSmoothedFiltered:
    def test_smoothed_and_filtered_are_separate_inputs(self):
        """The function accepts the smoothed full-sample state series and the
        walk-forward filtered-probs matrix as SEPARATE arguments (forces the
        Pitfall-1 distinction) and derives transitions purely from the
        smoothed states' change points. A transition that occurs BEFORE the
        filtered-probs matrix's date range even starts (simulating the
        walk-forward warmup period) resolves to NaN (unresolved) rather than
        crashing or silently matching against out-of-range data — proving
        the two series are never conflated.

        states: [0]*3 + [1]*3 + [0]*2 + [2]*4  (length 12)
        runs: 3, 3, 2, 4 -> median sojourn 3.0
        transitions: position 3 -> state 1 (BEFORE filtered range, NaN);
                     position 6 -> state 0 (lag 0);
                     position 8 -> state 2 (lag 1)
        """
        full_index = pd.date_range("2000-01-31", periods=12, freq="ME")
        states = [0, 0, 0, 1, 1, 1, 0, 0, 2, 2, 2, 2]
        full_sample_states = pd.Series(states, index=full_index)

        # filtered_probs_matrix only covers positions 6-11 (the walk-forward
        # decisions only start after warmup) -- position 3's transition into
        # state 1 has no corresponding filtered-probs row at all.
        sub_index = full_index[6:12]
        col0 = [0.9] * 6  # crosses immediately at position 6 -> lag 0
        col1 = [0.0] * 6  # never used (transition 3 is out of range)
        col2 = [0.0, 0.0, 0.3, 0.85, 0.85, 0.85]  # crosses at position 9 -> lag 1
        filtered_probs_matrix = pd.DataFrame({0: col0, 1: col1, 2: col2}, index=sub_index)

        result = compute_sojourn_lag_headline(full_sample_states, filtered_probs_matrix, act_threshold=0.7)

        assert result["median_sojourn"] == pytest.approx(3.0)
        # resolved lags: state0 -> 0, state2 -> 1 (state1's transition is
        # unresolved/NaN and excluded) -> median 0.5
        assert result["median_lag"] == pytest.approx(0.5)
        assert result["ratio"] == pytest.approx(sojourn_lag_ratio(3.0, 0.5))


class TestPerTransitionTargetStateProbs:
    """Review F1 — the REAL multiclass->single-series conversion.

    A K=5 filtered-probs matrix where transition A (into state 2) and
    transition B (into state 4) each have their OWN target-state column
    cross the threshold at a distinct, known lag, while a class-AGNOSTIC
    row-max (max across all 5 columns per row) would cross EARLIER for both
    transitions due to spurious spikes in unrelated classes. Proves
    ``compute_sojourn_lag_headline`` pools PER-TARGET-STATE lags (checking
    each transition against its own target-state column), not a
    class-agnostic max — the honest conversion, not a wiring stub over a
    pre-built single series.
    """

    def test_per_target_state_median_exceeds_class_agnostic_max(self):
        """states: [0]*5 + [2]*4 + [4]*4 (length 13).
        transitions: position 5 -> state 2; position 9 -> state 4.

        Own-column (per-target-state) crossings:
          - state 2 (col 2): 0.10, 0.20, 0.30, 0.85 (positions 5-8) -> crosses
            at position 8 -> lag 3.
          - state 4 (col 4): 0.20, 0.75 (positions 9-10) -> crosses at
            position 10 -> lag 1.
          pooled per-target-state lags = [3, 1] -> median 2.0

        Class-agnostic row-max crossings (spurious spikes in unrelated
        columns 3 and 0):
          - row 6 (col 3 = 0.75) makes row-max cross EARLY relative to the
            position-5 transition -> lag 1 (not 3).
          - row 9 (col 0 = 0.80) makes row-max cross IMMEDIATELY relative to
            the position-9 transition -> lag 0 (not 1).
          pooled class-agnostic lags = [1, 0] -> median 0.5

        2.0 (per-target-state) is STRICTLY GREATER than 0.5 (class-agnostic).
        """
        full_index = pd.date_range("2000-01-31", periods=13, freq="ME")
        states = [0, 0, 0, 0, 0, 2, 2, 2, 2, 4, 4, 4, 4]
        full_sample_states = pd.Series(states, index=full_index)

        base = 0.05
        col0 = [base] * 13
        col0[9] = 0.80  # spurious spike unrelated to state 2's or state 4's own column
        col1 = [base] * 13
        col2 = [base, base, base, base, base, 0.10, 0.20, 0.30, 0.85, base, base, base, base]
        col3 = [base] * 13
        col3[6] = 0.75  # spurious spike unrelated to state 2's own column
        col4 = [base] * 13
        col4[9] = 0.20
        col4[10] = 0.75

        filtered_probs_matrix = pd.DataFrame(
            {0: col0, 1: col1, 2: col2, 3: col3, 4: col4}, index=full_index
        )

        result = compute_sojourn_lag_headline(full_sample_states, filtered_probs_matrix, act_threshold=0.7)

        # Per-target-state (own-column) construction, computed independently
        # here to prove the function's internal pooling matches.
        own_lag_state2 = compute_detection_lag([5], filtered_probs_matrix[2].reset_index(drop=True), threshold=0.7)
        own_lag_state4 = compute_detection_lag([9], filtered_probs_matrix[4].reset_index(drop=True), threshold=0.7)
        expected_target_state_median = float(np.median([own_lag_state2["median"], own_lag_state4["median"]]))
        assert expected_target_state_median == pytest.approx(2.0)
        assert result["median_lag"] == pytest.approx(expected_target_state_median)

        # Class-agnostic max-across-classes construction — built here purely
        # for comparison, never inside the production function.
        row_max = filtered_probs_matrix.max(axis=1).reset_index(drop=True)
        class_agnostic_lags = compute_detection_lag([5, 9], row_max, threshold=0.7)
        assert class_agnostic_lags["median"] == pytest.approx(0.5)

        assert result["median_lag"] > class_agnostic_lags["median"]
        assert result["median_lag"] == pytest.approx(2.0)


# ── build_filtered_probs_matrix ────────────────────────────────────────────────────


class TestBuildFilteredProbsMatrix:
    def test_pads_missing_states_with_zero(self):
        """A step observing only 3 of the 5 union classes gets its two
        missing state columns padded with 0.0; the returned frame is
        (n_steps, K) indexed by the driver's decision dates (union-of-classes
        reconciliation, review F1/F3)."""
        dates = [pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-29"), pd.Timestamp("2020-03-31")]
        per_step_metrics = {
            "dates": dates,
            "proba": [
                np.array([0.10, 0.20, 0.30, 0.15, 0.25]),  # classes 0,1,2,3,4 (all 5)
                np.array([0.50, 0.30, 0.20]),  # classes 0,2,4 (only 3 of 5)
                np.array([0.05, 0.15, 0.20, 0.30, 0.30]),  # classes 0,1,2,3,4 (all 5)
            ],
            "classes": [
                [0, 1, 2, 3, 4],
                [0, 2, 4],
                [0, 1, 2, 3, 4],
            ],
        }

        result = build_filtered_probs_matrix(per_step_metrics)

        assert result.shape == (3, 5)
        assert list(result.columns) == [0, 1, 2, 3, 4]
        assert list(result.index) == dates

        # Step index 1 only observed classes [0, 2, 4]; columns 1 and 3 must
        # be padded with 0.0, not dropped or NaN.
        row1 = result.loc[dates[1]]
        assert row1[0] == pytest.approx(0.50)
        assert row1[1] == pytest.approx(0.0)
        assert row1[2] == pytest.approx(0.30)
        assert row1[3] == pytest.approx(0.0)
        assert row1[4] == pytest.approx(0.20)

    def test_full_class_step_has_no_padding(self):
        """A step observing all 5 union classes has every column populated
        from its own proba values, none padded."""
        dates = [pd.Timestamp("2020-01-31")]
        per_step_metrics = {
            "dates": dates,
            "proba": [np.array([0.1, 0.2, 0.3, 0.15, 0.25])],
            "classes": [[0, 1, 2, 3, 4]],
        }

        result = build_filtered_probs_matrix(per_step_metrics)

        row0 = result.loc[dates[0]]
        assert row0[0] == pytest.approx(0.1)
        assert row0[1] == pytest.approx(0.2)
        assert row0[2] == pytest.approx(0.3)
        assert row0[3] == pytest.approx(0.15)
        assert row0[4] == pytest.approx(0.25)


# ── T0.12: the silent zero on a wrong column shape ──────────────────────────
#
# compute_sojourn_lag_headline returned n_resolved=0, median_lag=NaN, ratio=NaN
# with NO exception and NO warning when filtered_probs_matrix carried
# "state_{k}" STRING columns. target_state is an int, so `0 not in
# ["state_0", ...]` was True for every state and every transition fell into the
# no-column branch. It read as the substantive finding "real-time detection
# never happened" when the cause was a wrong matrix shape. Plan 07-11's first
# draft reported 0 of 25 transitions resolved for exactly this reason.
#
# The discriminator is column TYPE, not overlap: a genuinely absent state is a
# real unresolved transition and must still be allowed.


class TestT012WrongColumnShapeRaises:
    @staticmethod
    def _ref():
        idx = pd.date_range("1972-01-31", periods=60, freq="ME")
        return pd.Series([0] * 30 + [1] * 30, index=idx, name="state"), idx

    def test_state_k_string_columns_raise_instead_of_reporting_zero(self):
        """The defect itself. Deleting the guard makes this return n_resolved=0."""
        ref, idx = self._ref()
        probs = pd.DataFrame(
            {f"state_{i}": [0.0] * 32 + [1.0] * 28 for i in range(2)}, index=idx
        )
        with pytest.raises(ValueError, match="CANONICAL INTEGER state labels"):
            compute_sojourn_lag_headline(ref, probs)

    def test_the_message_names_the_offending_columns(self):
        """A shape error that does not say which columns are wrong sends the
        caller back to guessing — the same cost as the silent zero."""
        ref, idx = self._ref()
        probs = pd.DataFrame({"state_0": [1.0] * 60, "state_1": [0.0] * 60}, index=idx)
        with pytest.raises(ValueError) as exc:
            compute_sojourn_lag_headline(ref, probs)
        assert "state_0" in str(exc.value)

    def test_integer_columns_still_compute_a_real_headline(self):
        """The guard must not be a one-way refusal — otherwise it could only
        confirm. Probability for state 1 crosses 0.70 two months late."""
        ref, idx = self._ref()
        p1 = [0.0] * 32 + [1.0] * 28
        probs = pd.DataFrame({0: [1.0 - x for x in p1], 1: p1}, index=idx)
        out = compute_sojourn_lag_headline(ref, probs)
        assert out["n_transitions"] == 1
        assert out["n_resolved"] == 1
        assert out["median_lag"] == pytest.approx(2.0)

    def test_a_genuinely_absent_state_column_is_NOT_an_error(self):
        """Overlap is the WRONG discriminator. The only transition here targets
        state 1, which has no column — zero overlap, but legitimate: that
        transition is truly unresolved and keeps the NaN convention. A guard
        keyed on overlap would wrongly raise here."""
        ref, idx = self._ref()
        out = compute_sojourn_lag_headline(ref, pd.DataFrame({0: [1.0] * 60}, index=idx))
        assert out["n_transitions"] == 1
        assert out["n_resolved"] == 0
        assert np.isnan(out["median_lag"])

    def test_bool_columns_are_rejected_too(self):
        """bool is an int subclass in Python; a True/False-keyed matrix is not a
        state labeling and must not slip through the isinstance check."""
        ref, idx = self._ref()
        probs = pd.DataFrame({True: [1.0] * 60, False: [0.0] * 60}, index=idx)
        with pytest.raises(ValueError, match="CANONICAL INTEGER state labels"):
            compute_sojourn_lag_headline(ref, probs)
