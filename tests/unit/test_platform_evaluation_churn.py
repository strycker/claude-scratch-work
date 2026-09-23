"""Guards for ``evaluation/churn.py`` — the two churn series, kept apart (plan 08-01).

This phase exists because the recorded 41.91% filtered churn is an **L1**
quantity (``joint_driver.py:502``, ``state_1 = states_1.iloc[-1]``) while the
design change it was attributed to touches **L2**. Under the decision-bearing
``ROUTING_L1_ONLY`` the tilt is fed ``_last_state_one_hot(states_1)``
(``joint_driver.py:431``), so the two series coincide *by construction* and a
criterion that re-measures the L1 series after an L2 fix reports 246 -> 246
whether the fix worked perfectly or not at all.

The tests below are therefore shaped to FAIL on the specific ways the two
series could quietly become one object, or on the ways a rate could be quoted
against the wrong denominator:

- the ``state_{k}``-string round trip must restore **integer** columns, because
  a string-keyed matrix scores every transition unresolved inside
  ``compute_sojourn_lag_headline`` and returns ``n_resolved = 0`` that reads as
  a finding (T0.12);
- ``read_probability_matrix`` must RAISE on a column that cannot denote a
  canonical integer state, rather than passing it onward;
- ``churn_rate`` must divide by adjacent PAIRS, and the test asserts the
  month-denominated value is *not* returned (F-4).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from trading_crab_lib.platform.evaluation.churn import (
    argmax_churn,
    churn_rate,
    read_probability_matrix,
    state_change_count,
    write_probability_matrix,
)
from trading_crab_lib.platform.evaluation.sojourn_lag import build_filtered_probs_matrix

#: F-4's two candidate denominators over the live 588-step window.
_N_CHANGES = 246
_N_ROWS = 588


def _per_step_metrics(n_steps: int = 6, *, classes=(0, 1, 2)) -> dict:
    """A driver-shaped ``per_step_metrics`` bucket, built by hand.

    Mirrors ``joint_driver.py:508-510``'s three parallel lists exactly: one
    ``date``, one ``proba`` array and one ``classes`` list per NON-degraded step.
    """
    dates = list(pd.date_range("1972-01-31", periods=n_steps, freq="ME"))
    rng = np.linspace(0.1, 0.9, n_steps)
    proba, class_lists = [], []
    for i in range(n_steps):
        row = np.full(len(classes), (1.0 - rng[i]) / max(1, len(classes) - 1))
        row[i % len(classes)] = rng[i]
        proba.append(row / row.sum())
        class_lists.append(list(classes))
    return {"dates": dates, "proba": proba, "classes": class_lists}


# ── state_change_count: ONE definition of a state change for this phase ─────


class TestStateChangeCount:
    def test_matches_the_scripts_dropna_then_compare_convention(self):
        """The same count ``run_joint_lift.py::_n_transitions`` produces."""
        states = pd.Series([0, 0, 1, 1, 1, 2])
        assert state_change_count(states) == 2

    def test_counts_a_change_on_the_very_first_pair(self):
        """A series whose first adjacent pair already differs.

        Fails if the implementation skipped the opening pair (an off-by-one that
        would silently under-report every churn number in this phase).
        """
        assert state_change_count(pd.Series([0, 1, 1, 1])) == 1

    def test_all_nan_series_is_zero_not_an_error(self):
        assert state_change_count(pd.Series([np.nan, np.nan, np.nan])) == 0

    def test_single_row_series_has_no_pairs_and_therefore_no_changes(self):
        assert state_change_count(pd.Series([3])) == 0

    def test_empty_series(self):
        assert state_change_count(pd.Series(dtype=float)) == 0

    def test_nan_rows_are_dropped_before_comparing_not_treated_as_a_state(self):
        """``[0, NaN, 0]`` is ONE run, not two changes."""
        assert state_change_count(pd.Series([0, np.nan, 0])) == 0


# ── churn_rate: F-4's denominator, pinned against its own predecessor ───────


class TestChurnRateIsDenominatedInPairs:
    def test_246_over_588_rows_is_246_over_587(self):
        """The recorded 41.84% divided by MONTHS; 587 adjacent PAIRS exist.

        The two values differ in the 4th decimal (0.418980 vs 0.418367), so an
        assertion on the correct one alone would pass under a sloppy tolerance.
        The second assertion rejects the old denominator explicitly.
        """
        rate = churn_rate(_N_CHANGES, _N_ROWS)
        assert rate == pytest.approx(_N_CHANGES / (_N_ROWS - 1), abs=1e-12)
        assert rate != pytest.approx(_N_CHANGES / _N_ROWS, abs=1e-9), (
            "churn_rate returned the MONTH-denominated value (F-4's off-by-one)"
        )

    def test_rate_over_fewer_than_two_rows_raises_rather_than_returning_zero(self):
        """A 0.0 here would be a number where no rate exists."""
        for n_rows in (0, 1):
            with pytest.raises(ValueError, match="at least two rows"):
                churn_rate(0, n_rows)

    def test_two_rows_is_a_single_pair(self):
        assert churn_rate(1, 2) == 1.0


# ── the parquet round trip: string on disk, INTEGER in memory (T0.12) ───────


class TestProbabilityMatrixRoundTrip:
    def test_written_then_read_is_identical_including_the_column_index_dtype(self, tmp_path):
        """Fails if the ``state_{k}`` rename is one-way.

        ``check_column_type=True`` is the load-bearing argument: an object-dtype
        column index holding ints would satisfy a value-only comparison and
        would still be the shape ``compute_sojourn_lag_headline`` refuses.
        """
        original = build_filtered_probs_matrix(_per_step_metrics())
        info = write_probability_matrix(_per_step_metrics(), tmp_path / "probs.parquet")
        restored = read_probability_matrix(tmp_path / "probs.parquet")
        assert_frame_equal(original, restored, check_column_type=True)
        assert info["n_rows"] == len(original)
        assert info["states"] == [0, 1, 2]

    def test_the_on_disk_columns_are_state_k_strings(self, tmp_path):
        """Parquet column names must be strings; match ``report.py:1038``."""
        write_probability_matrix(_per_step_metrics(), tmp_path / "probs.parquet")
        raw = pd.read_parquet(tmp_path / "probs.parquet")
        assert list(raw.columns) == ["state_0", "state_1", "state_2"]

    def test_row_count_is_the_number_of_non_degraded_steps(self, tmp_path):
        info = write_probability_matrix(_per_step_metrics(n_steps=9), tmp_path / "p.parquet")
        assert info["n_rows"] == 9

    def test_read_raises_on_columns_that_cannot_denote_a_state(self, tmp_path):
        path = tmp_path / "bad.parquet"
        pd.DataFrame({"a": [0.5], "b": [0.5]}).to_parquet(path)
        with pytest.raises(ValueError, match="'a'"):
            read_probability_matrix(path)

    def test_read_raises_on_bare_integer_strings_without_the_prefix(self, tmp_path):
        """``["0", "1"]`` is the near-miss a permissive parser would accept."""
        path = tmp_path / "bare.parquet"
        pd.DataFrame({"0": [0.5], "1": [0.5]}).to_parquet(path)
        with pytest.raises(ValueError, match="'0'"):
            read_probability_matrix(path)

    def test_read_raises_on_a_state_prefix_with_a_non_integer_suffix(self, tmp_path):
        path = tmp_path / "suffix.parquet"
        pd.DataFrame({"state_0": [0.5], "state_x": [0.5]}).to_parquet(path)
        with pytest.raises(ValueError, match="state_x"):
            read_probability_matrix(path)

    def test_restored_matrix_is_accepted_by_the_T0_12_guard(self, tmp_path):
        """The round trip's whole point: the result must survive the guard that
        raises on ``state_{k}`` strings, which is the second line of defence."""
        from trading_crab_lib.platform.evaluation.sojourn_lag import compute_sojourn_lag_headline

        write_probability_matrix(_per_step_metrics(n_steps=12), tmp_path / "p.parquet")
        restored = read_probability_matrix(tmp_path / "p.parquet")
        states = pd.Series([0] * 4 + [1] * 4 + [2] * 4, index=restored.index)
        headline = compute_sojourn_lag_headline(states, restored, act_threshold=0.5)
        assert headline["n_transitions"] == 2


# ── argmax_churn: Track B's measurement, carrying its own window ────────────


class TestArgmaxChurn:
    def test_reports_changes_pairs_rate_and_window(self):
        index = pd.date_range("1972-01-31", periods=4, freq="ME")
        matrix = pd.DataFrame(
            {0: [0.9, 0.9, 0.1, 0.1], 1: [0.1, 0.1, 0.9, 0.9]}, index=index
        )
        out = argmax_churn(matrix)
        assert out["n_changes"] == 1
        assert out["n_rows"] == 4
        assert out["n_pairs"] == 3
        assert out["rate"] == pytest.approx(1 / 3)
        assert out["first_date"] == "1972-01-31"
        assert out["last_date"] == "1972-04-30"

    def test_rate_is_denominated_in_pairs_not_rows(self):
        index = pd.date_range("1972-01-31", periods=5, freq="ME")
        matrix = pd.DataFrame(
            {0: [0.9, 0.1, 0.9, 0.1, 0.9], 1: [0.1, 0.9, 0.1, 0.9, 0.1]}, index=index
        )
        out = argmax_churn(matrix)
        assert out["n_changes"] == 4
        assert out["rate"] == pytest.approx(4 / 4)
        assert out["rate"] != pytest.approx(4 / 5, abs=1e-9)

    def test_a_constant_argmax_is_zero_churn_and_still_a_measurement(self):
        index = pd.date_range("1972-01-31", periods=3, freq="ME")
        matrix = pd.DataFrame({0: [0.9, 0.8, 0.7], 1: [0.1, 0.2, 0.3]}, index=index)
        assert argmax_churn(matrix)["n_changes"] == 0

    def test_single_row_matrix_raises_rather_than_reporting_a_rate(self):
        index = pd.date_range("1972-01-31", periods=1, freq="ME")
        with pytest.raises(ValueError, match="at least two rows"):
            argmax_churn(pd.DataFrame({0: [1.0]}, index=index))
