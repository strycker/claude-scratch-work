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

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import trading_crab_lib.platform.evaluation.sojourn_lag as sojourn_lag_module
from trading_crab_lib.platform.evaluation.sojourn_lag import (
    build_filtered_probs_matrix,
    classify_negative_offsets,
    compute_signed_detection_offsets,
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


# ── Plan 08-06 Task 2: the extraction of _transitions_by_state moved nothing ───────
#
# These read the REAL dev inputs from tracked files, not the committed JSON alone: a
# pin that only re-read diagnostics_l1only.json would pass whatever the refactor did
# to the function, because the JSON was written before it. Both inputs are tracked
# (data/checkpoints/platform/regime_labels.parquet and the l1only joint curve), so
# nothing here skips in CI. The reconstruction mirrors
# scripts/joint_lift_diagnostics.py::diagnose exactly: dev split of the full-sample
# L1 labels, and a one-hot of the walk-forward state_1 column (exact, not an
# approximation, under ROUTING_L1_ONLY).

_JOINT_LIFT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "reports" / "platform" / "joint_lift"


def _real_dev_inputs() -> tuple[pd.Series, pd.DataFrame]:
    from trading_crab_lib.platform.checkpoints import get_platform_checkpoint_manager
    from trading_crab_lib.platform.honesty.holdout import DEFAULT_HOLDOUT_CUTOFF, split_by_holdout_boundary

    full = get_platform_checkpoint_manager().load("regime_labels")["state"]
    dev, _ = split_by_holdout_boundary(full.to_frame("state"), cutoff=DEFAULT_HOLDOUT_CUTOFF)
    filtered = pd.read_parquet(_JOINT_LIFT_DIR / "joint_lift_joint_l1only.parquet")["state_1"].dropna().astype(int)
    probs = pd.DataFrame({int(k): (filtered == k).astype(float) for k in sorted(filtered.unique())}, index=filtered.index)
    return dev["state"], probs


class TestHeadlinePinOnRealDevInputs:
    def test_classifier_1_headline_is_9_5_over_4_0_with_25_of_25(self):
        states, probs = _real_dev_inputs()
        assert states.index.max() <= pd.Timestamp("2020-12-31")  # holdout never read
        out = compute_sojourn_lag_headline(states, probs, act_threshold=0.70)
        assert (out["median_sojourn"], out["median_lag"], out["ratio"]) == (9.5, 4.0, 2.375)
        assert (out["n_transitions"], out["n_resolved"]) == (25, 25)

    def test_recomputation_equals_the_committed_record(self):
        record = json.loads((_JOINT_LIFT_DIR / "diagnostics_l1only.json").read_text())["classifier_1"]["sojourn_lag"]
        states, probs = _real_dev_inputs()
        out = compute_sojourn_lag_headline(states, probs, act_threshold=record["act_threshold"])
        for key in ("median_sojourn", "median_lag", "ratio", "n_transitions", "n_resolved"):
            assert out[key] == record[key], key


# ── Plan 08-06 Task 2: compute_signed_detection_offsets ────────────────────────────


def _single_transition_fixture(col_values: list[float], *, i: int = 10, n: int = 20):
    """States 0 until position i, then state 1. Column 1 is ``col_values``."""
    idx = pd.date_range("1990-01-31", periods=n, freq="ME")
    states = pd.Series([0] * i + [1] * (n - i), index=idx)
    probs = pd.DataFrame({0: [1.0 - v for v in col_values], 1: col_values}, index=idx)
    return states, probs


class TestSignedOffsetAgreesWithDetectionLag:
    def test_elementwise_equal_where_no_belief_leads(self):
        """Four transitions, none led: forward lags 2, 1, 3 and one unresolved.
        Every signed offset must equal compute_detection_lag's answer for the same
        transition, NaN for NaN. Fails on any divergence in the forward branch."""
        idx = pd.date_range("1990-01-31", periods=26, freq="ME")
        # transitions: @6 -> 1, @11 -> 0, @15 -> 2, @20 -> 1
        states = pd.Series([0] * 6 + [1] * 5 + [0] * 4 + [2] * 5 + [1] * 6, index=idx)
        col0 = [0.9] * 6 + [0.1] * 6 + [0.8] * 3 + [0.1] * 11  # @11 crosses @12: lag 1
        col1 = [0.0] * 8 + [0.75] * 3 + [0.0] * 15              # @6 crosses @8: lag 2; @20 never: NaN
        col2 = [0.0] * 18 + [0.9] * 8                            # @15 crosses @18: lag 3
        probs = pd.DataFrame({0: col0, 1: col1, 2: col2}, index=idx)

        out = compute_signed_detection_offsets(states, probs, act_threshold=0.7)

        expected = [
            compute_detection_lag([pos], probs[s].reindex(states.index), threshold=0.7)["lags"][0]
            for pos, s in zip(out["positions"], out["target_states"])
        ]
        assert out["positions"] == [6, 11, 15, 20]
        np.testing.assert_array_equal(np.array(out["offsets"]), np.array(expected))
        np.testing.assert_array_equal(np.array(out["offsets"]), np.array([2.0, 1.0, 3.0, np.nan]))
        assert out["n_negative"] == 0 and out["n_zero_or_negative"] == 0
        assert out["min_offset"] == 1.0

    def test_agrees_on_every_transition_of_the_real_dev_inputs(self):
        """The real l1only inputs: wherever compute_detection_lag is not 0, the signed
        offset must equal it exactly; where it is 0, the signed offset must be <= 0.
        Measured 2026-09-23: all 25 transitions have lag >= 1, so all 25 are equal and
        none leads (min_offset 1.0, median 4.0 — the headline's own median)."""
        states, probs = _real_dev_inputs()
        out = compute_signed_detection_offsets(states, probs, act_threshold=0.70)
        assert out["n_transitions"] == 25
        for pos, s, off in zip(out["positions"], out["target_states"], out["offsets"]):
            lag = compute_detection_lag([pos], probs[s].reindex(states.index), threshold=0.70)["lags"][0]
            if lag == 0:
                assert off <= 0, (pos, s, off)
            elif np.isnan(lag):
                assert np.isnan(off), (pos, s, off)
            else:
                assert off == lag, (pos, s, off, lag)
        assert out["median_offset"] == compute_sojourn_lag_headline(states, probs)["median_lag"] == 4.0


class TestNegativeOffsetsAreReachable:
    def test_a_three_month_lead_is_exactly_minus_3(self):
        """Column 1 is above threshold from i-3 onward; below at i-4. The offset is -3.
        compute_detection_lag on the same input says 0 — the floor this function exists
        to remove. Fails if the backward walk is missing (that would read 0 too)."""
        col = [0.1] * 7 + [0.8] * 13  # i = 10; run starts at 7
        states, probs = _single_transition_fixture(col)
        out = compute_signed_detection_offsets(states, probs, act_threshold=0.7)
        assert out["offsets"] == [-3.0]
        assert out["n_negative"] == 1 and out["n_zero_or_negative"] == 1
        assert out["min_offset"] == -3.0
        assert compute_detection_lag([10], probs[1], threshold=0.7)["lags"] == [0.0]

    def test_crossing_exactly_at_the_transition_is_zero_not_negative(self):
        col = [0.1] * 10 + [0.8] * 10
        states, probs = _single_transition_fixture(col)
        out = compute_signed_detection_offsets(states, probs, act_threshold=0.7)
        assert out["offsets"] == [0.0]
        assert out["n_negative"] == 0 and out["n_zero_or_negative"] == 1

    def test_the_run_boundary_is_respected(self):
        """Above at i and i-1, BELOW at i-2, above again at i-5..i-3. The contiguous
        run containing i starts at i-1, so the offset is -1 — not -5, which a naive
        'first crossing anywhere before' search would return."""
        col = [0.1] * 5 + [0.8] * 3 + [0.1] + [0.8] * 11  # i=10: 9,10 above; 8 below; 5..7 above
        states, probs = _single_transition_fixture(col)
        out = compute_signed_detection_offsets(states, probs, act_threshold=0.7)
        assert out["offsets"] == [-1.0]

    def test_nan_before_the_first_decision_stops_the_walk(self):
        """Pre-warmup months carry NaN (no filtered row); NaN is never 'at or above'."""
        idx = pd.date_range("1990-01-31", periods=20, freq="ME")
        states = pd.Series([0] * 10 + [1] * 10, index=idx)
        probs = pd.DataFrame({0: [0.1] * 14, 1: [0.9] * 14}, index=idx[6:])  # first row at position 6
        out = compute_signed_detection_offsets(states, probs, act_threshold=0.7)
        assert out["offsets"] == [-4.0]


class TestUnresolvedStaysNaN:
    def test_never_crossing_is_nan_counted_but_excluded_from_the_median(self):
        idx = pd.date_range("1990-01-31", periods=20, freq="ME")
        states = pd.Series([0] * 5 + [1] * 5 + [2] * 10, index=idx)
        probs = pd.DataFrame(
            {0: [0.9] * 5 + [0.0] * 15, 1: [0.0] * 7 + [0.8] * 3 + [0.0] * 10, 2: [0.3] * 20}, index=idx
        )
        out = compute_signed_detection_offsets(states, probs, act_threshold=0.7)
        assert out["offsets"][0] == 2.0 and np.isnan(out["offsets"][1])
        assert out["n_transitions"] == 2 and out["n_resolved"] == 1
        assert out["median_offset"] == 2.0 and out["min_offset"] == 2.0

    def test_a_target_state_with_no_column_is_nan(self):
        states, probs = _single_transition_fixture([0.1] * 20)
        out = compute_signed_detection_offsets(states, probs[[0]], act_threshold=0.7)
        assert np.isnan(out["offsets"][0]) and out["n_resolved"] == 0
        assert np.isnan(out["median_offset"]) and np.isnan(out["min_offset"])


class TestSignedOffsetReusesTheT012Guard:
    def test_string_columns_raise_naming_this_function(self):
        states, probs = _single_transition_fixture([0.1] * 20)
        probs.columns = ["state_0", "state_1"]
        with pytest.raises(ValueError, match="compute_signed_detection_offsets: .*CANONICAL INTEGER"):
            compute_signed_detection_offsets(states, probs)


# ── classify_negative_offsets: the held-through-return rule (plan 08-08, AMENDED 2026-09-23) ──
#
# Pinned against 08-06's synthetic arms BEFORE any real data is read. The prototype
# results recorded in 08-08-PLAN.md (measured 2026-09-23 against 08-06's committed
# fixtures) are reproduced exactly: caveat -> leads [], misses [45]; Arm 2 -> leads
# [14, 45, 69], misses []; Arm 1 -> no negative offsets; clause (iii) dropped entirely
# -> Arm 2 leads [14, 69], misses [45].
#
# Clause (iii)'s two halves (belief[s] >= act AND belief[r] < act) are REDUNDANT with
# each other while act_threshold > 0.5 (belief[s] >= 0.70 forces belief[r] <= 0.30).
# Both are kept because 08-09 may re-pin the threshold relative to 1/K, possibly below
# 0.5, where they stop being redundant. Nobody should later "simplify" one away. For
# the same reason there is deliberately NO mutation arm that drops only one half: at
# act = 0.70 it cannot misclassify anything (measured 2026-09-23), so such an arm would
# demand a failure that cannot occur.


def _recursion_world():
    """08-06's fixture and arms, reused rather than re-derived (one world, one set of numbers)."""
    import test_platform_nowcaster_recursion as rec  # tests/unit is on sys.path under pytest's prepend mode

    return rec


def _caveat_arm():
    rec = _recursion_world()
    states, evidence, _, _ = rec._world()
    a_emp = rec.transition_matrix_for(states, state_index=rec.IDX)
    a_stickier = pd.DataFrame(0.999 * np.eye(len(rec.IDX)), index=rec.IDX, columns=rec.IDX) + 0.001 * a_emp
    return states, rec._run_filter(states, evidence, a_stickier)


def _arm2():
    rec = _recursion_world()
    states, evidence, _, _ = rec._world()
    a = rec.transition_matrix_for(states, state_index=rec.IDX)
    prior = rec.unconditional_belief(states, state_index=rec.IDX)
    return states, rec._one_hot(rec._viterbi(evidence, a, prior), states.index)


def _arm1():
    rec = _recursion_world()
    states, evidence, _, _ = rec._world()
    a = rec.transition_matrix_for(states, state_index=rec.IDX)
    return states, rec._run_filter(states, evidence, a)


def _classify(states, belief, act=0.70):
    offsets = compute_signed_detection_offsets(states, belief, act_threshold=act)
    return offsets, classify_negative_offsets(states, belief, offsets, act)


class TestHeldThroughReturnRuleOnTheSyntheticArms:
    def test_caveat_fixture_honest_miss_is_exempted(self):
        states, belief = _caveat_arm()
        offsets, out = _classify(states, belief)
        assert offsets["n_negative"] == 1
        assert out["lead_positions"] == []
        assert out["held_through_miss_positions"] == [45]
        assert out["n_lead"] == 0 and out["n_held_through_miss"] == 1
        (d,) = out["details"]
        assert d["offset"] == -14.0 and d["state"] == 2 and d["preceding_state"] == 0
        assert d["preceding_run"] == (36, 44)

    def test_arm2_leak_still_halts_three_leads_zero_misses(self):
        """Without this arm the amendment would be a relaxation with no evidence it
        left the guard intact."""
        states, belief = _arm2()
        _, out = _classify(states, belief)
        assert out["lead_positions"] == [14, 45, 69]
        assert out["held_through_miss_positions"] == []
        assert out["n_lead"] == 3 and out["n_held_through_miss"] == 0

    def test_arm1_honest_filter_has_no_negative_offsets_to_classify(self):
        states, belief = _arm1()
        offsets, out = _classify(states, belief)
        assert offsets["n_negative"] == 0
        assert out["n_negative"] == 0 and out["n_lead"] == 0 and out["n_held_through_miss"] == 0

    def test_mutation_dropping_clause_iii_entirely_swallows_the_leak_at_45(self, monkeypatch):
        """Position 45 is the one led turn that is ALSO an s -> r -> s return, so it is
        exactly the case the exemption must not swallow. With clause (iii) gone it is
        misclassified as a miss (n_lead 3 -> 2) — proving (iii) is what keeps it a lead."""
        monkeypatch.setattr(sojourn_lag_module, "_belief_held_through", lambda *a, **k: True)
        states, belief = _arm2()
        _, out = _classify(states, belief)
        assert out["lead_positions"] == [14, 69]
        assert out["held_through_miss_positions"] == [45]
        assert out["n_lead"] == 2


class TestHeldThroughReturnRuleClauses:
    @staticmethod
    def _frame(states: list[int], s_col: list[float], k: int = 3) -> tuple[pd.Series, pd.DataFrame]:
        idx = pd.date_range("1990-01-31", periods=len(states), freq="ME")
        cols = {j: [0.0] * len(states) for j in range(k)}
        cols[states[-1]] = s_col
        others = [j for j in range(k) if j != states[-1]]
        for j in others:
            cols[j] = [(1.0 - v) / len(others) for v in s_col]
        return pd.Series(states, index=idx), pd.DataFrame(cols, index=idx)

    def test_run_at_the_start_of_the_series_is_a_lead_not_a_return(self):
        """Clause (ii)'s q-1 >= 0 guard. The series ENDS in s, so an unguarded
        states[q-1] with q = 0 would wrap to states[-1] == s and call this a return."""
        states, belief = self._frame([1] * 5 + [2] * 5, [0.9] * 10)
        offsets, out = _classify(states, belief)
        assert offsets["offsets"] == [-5.0]
        assert out["lead_positions"] == [5] and out["held_through_miss_positions"] == []

    def test_a_non_return_t_r_s_is_a_lead(self):
        states, belief = self._frame([0] * 4 + [1] * 4 + [2] * 4, [0.9] * 12)
        offsets, out = _classify(states, belief)
        neg = [p for p, o in zip(offsets["positions"], offsets["offsets"]) if o < 0]
        assert neg == [8]
        assert out["lead_positions"] == [8]

    def test_belief_registering_r_inside_the_run_is_a_lead(self):
        """s -> r -> s where the belief drops below act on s for one month inside r and
        comes back early: the leak pattern. (iii) fails -> LEAD."""
        states = [2] * 4 + [0] * 4 + [2] * 4
        s_col = [0.9] * 5 + [0.05] + [0.9] * 6  # leaves s at position 5, back above act from 6
        st, belief = self._frame(states, s_col)
        belief.loc[belief.index[5], 0] = 0.9  # ...and registers r = 0 there
        belief.loc[belief.index[5], 1] = 0.05
        offsets, out = _classify(st, belief)
        by_pos = dict(zip(offsets["positions"], offsets["offsets"]))
        assert by_pos[8] == -2.0
        assert out["lead_positions"] == [8] and out["held_through_miss_positions"] == []

    def test_a_missing_belief_month_never_exempts(self):
        states = [2] * 4 + [0] * 4 + [2] * 4
        st, belief = self._frame(states, [0.9] * 12)
        _, full = _classify(st, belief)
        assert full["held_through_miss_positions"] == [8]
        _, gappy = _classify(st, belief.iloc[3:])  # belief starts exactly at q-1 = 3: still a miss
        assert gappy["held_through_miss_positions"] == [8]
        _, later = _classify(st, belief.iloc[4:])  # q-1 = 3 now unobserved (NaN)
        assert later["held_through_miss_positions"] == [] and later["lead_positions"] == [8]

    def test_threshold_mismatch_raises(self):
        states, belief = _caveat_arm()
        offsets = compute_signed_detection_offsets(states, belief, act_threshold=0.70)
        with pytest.raises(ValueError, match="act_threshold"):
            classify_negative_offsets(states, belief, offsets, 0.60)
