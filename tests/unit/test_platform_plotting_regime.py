"""Tests for src/trading_crab_lib/platform/plotting/regime.py.

No test in this module performs network I/O, fits a model on real data, or
runs a walk-forward. The one real-data test reads the ``monthly_features``
parquet directly from the repository path (skipped when absent) and only
replays the cheap per-window active-feature rule.
"""

from __future__ import annotations

from pathlib import Path

# matplotlib.use("Agg") must precede pyplot import — import order is intentional.
# pylint: disable=wrong-import-position,wrong-import-order
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from trading_crab_lib.platform.backtest import driver as pdriver  # noqa: E402
from trading_crab_lib.platform.plotting import history as phist  # noqa: E402
from trading_crab_lib.platform.plotting import regime as pregime  # noqa: E402
from trading_crab_lib.platform.prediction.transition_matrix import (  # noqa: E402
    empirical_transition_matrix,
)

# pylint: enable=wrong-import-position,wrong-import-order

REAL_MONTHLY_FEATURES = Path("data/checkpoints/platform/monthly_features.parquet")

# The seven documented A13 change points (audit finding, reproduced in
# CONTEXT.md Amendment 3 item H).
EXPECTED_CHANGE_POINTS = [
    ("1972-01-31", 4),
    ("1972-02-29", 6),
    ("1972-04-30", 8),
    ("1973-02-28", 9),
    ("1986-06-30", 10),
    ("1995-02-28", 12),
    ("2000-01-31", 13),
]


@pytest.fixture
def monthly_index() -> pd.DatetimeIndex:
    return pd.date_range("1970-01-31", periods=360, freq="ME")


@pytest.fixture
def states(monthly_index) -> pd.Series:
    """Persistent synthetic labeling — long runs, all five states occupied."""
    rng = np.random.default_rng(7)
    values: list[int] = []
    state = 0
    while len(values) < len(monthly_index):
        run = int(rng.integers(6, 40))
        values.extend([state] * run)
        state = (state + int(rng.integers(1, 5))) % 5
    return pd.Series(values[: len(monthly_index)], index=monthly_index, name="state")


@pytest.fixture
def confidences(monthly_index) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    raw = rng.random((len(monthly_index), 5))
    normalized = raw / raw.sum(axis=1, keepdims=True)
    return pd.DataFrame(normalized, index=monthly_index, columns=[f"state_{k}" for k in range(5)])


class TestPlotRegimeTimeline:
    def test_does_not_crash(self, states, tmp_path):
        save_path = tmp_path / "timeline.png"
        fig = pregime.plot_regime_timeline(
            states, events=phist.ECONOMIC_EVENTS, title="timeline", save_path=save_path
        )
        assert isinstance(fig, plt.Figure)
        assert save_path.exists()

    def test_with_no_recessions(self, states):
        fig = pregime.plot_regime_timeline(states, recessions=None)
        assert isinstance(fig, plt.Figure)

    def test_with_one_recession_block(self, states):
        usrec = pd.Series(0, index=states.index)
        usrec.iloc[40:56] = 1
        fig = pregime.plot_regime_timeline(
            states, recessions=phist.recession_periods(usrec), events=phist.ECONOMIC_EVENTS
        )
        assert isinstance(fig, plt.Figure)

    def test_empty_input(self):
        assert isinstance(pregime.plot_regime_timeline(pd.Series(dtype=float)), plt.Figure)


class TestPlotOccupancyAndSojourn:
    def test_does_not_crash(self, states, tmp_path):
        save_path = tmp_path / "occ.png"
        fig = pregime.plot_occupancy_and_sojourn(states, save_path=save_path)
        assert isinstance(fig, plt.Figure)
        assert save_path.exists()

    def test_never_occupied_state_does_not_crash(self, monthly_index):
        constant = pd.Series(2, index=monthly_index)
        assert isinstance(pregime.plot_occupancy_and_sojourn(constant, n_states=5), plt.Figure)

    def test_empty_input(self):
        assert isinstance(pregime.plot_occupancy_and_sojourn(pd.Series(dtype=float)), plt.Figure)


class TestPlotTransitionMatrix:
    def test_does_not_crash(self, states, tmp_path):
        matrix = empirical_transition_matrix(states)
        save_path = tmp_path / "trans.png"
        fig = pregime.plot_transition_matrix(matrix, save_path=save_path)
        assert isinstance(fig, plt.Figure)
        assert save_path.exists()

    def test_empty_input(self):
        assert isinstance(pregime.plot_transition_matrix(pd.DataFrame()), plt.Figure)


class TestPlotSoftConfidences:
    def test_does_not_crash(self, confidences):
        assert isinstance(pregime.plot_soft_confidences(confidences), plt.Figure)

    def test_integer_columns_accepted(self, confidences):
        renamed = confidences.rename(columns={f"state_{k}": k for k in range(5)})
        assert isinstance(pregime.plot_soft_confidences(renamed), plt.Figure)

    def test_empty_input(self):
        assert isinstance(pregime.plot_soft_confidences(pd.DataFrame()), plt.Figure)


class TestPlotRegimeProfiles:
    def test_does_not_crash(self, tmp_path):
        profiles = pd.DataFrame(
            {"state": [0, 1, 2], "profile": ["state 0: high vol", "state 1: low vol", "state 2: wide spread"]}
        )
        save_path = tmp_path / "profiles.png"
        fig = pregime.plot_regime_profiles(profiles, save_path=save_path)
        assert isinstance(fig, plt.Figure)
        assert save_path.exists()

    def test_empty_input(self):
        assert isinstance(pregime.plot_regime_profiles(pd.DataFrame()), plt.Figure)


class TestPlotSojournDistribution:
    def test_does_not_crash(self, states):
        assert isinstance(pregime.plot_sojourn_distribution(states), plt.Figure)

    def test_single_state_series(self, monthly_index):
        constant = pd.Series(0, index=monthly_index)
        assert isinstance(pregime.plot_sojourn_distribution(constant), plt.Figure)

    def test_empty_input(self):
        assert isinstance(pregime.plot_sojourn_distribution(pd.Series(dtype=float)), plt.Figure)


class TestActiveFeatureCountTimeline:
    def test_uses_the_drivers_own_rule_object(self):
        """The A13 reconstruction must not reimplement `_window_active_features`."""
        assert pregime._window_active_features is pdriver._window_active_features

    def test_late_starting_column_steps_the_count_up_on_the_expected_date(self):
        idx = pd.date_range("2000-01-31", periods=60, freq="ME")
        frame = pd.DataFrame(
            {
                "early_a": np.arange(60, dtype=float),
                "early_b": np.arange(60, dtype=float),
                "late_c": np.concatenate([np.full(20, np.nan), np.arange(40, dtype=float)]),
            },
            index=idx,
        )
        timeline = pregime.active_feature_count_timeline(
            frame, ["early_a", "early_b", "late_c"], min_history=10, min_train=12
        )
        # `late_c` first has data at position 20, so it clears 10 months of
        # history once the training block (index[:i], strictly before t) covers
        # position 29 — i.e. i == 30, decision date idx[30].
        assert int(timeline["n_active"].iloc[0]) == 2
        assert int(timeline.loc[idx[29], "n_active"]) == 2
        assert int(timeline.loc[idx[30], "n_active"]) == 3
        assert int(timeline["n_active"].iloc[-1]) == 3

    def test_active_column_lists_are_sorted(self):
        idx = pd.date_range("2000-01-31", periods=40, freq="ME")
        frame = pd.DataFrame({"z": np.arange(40.0), "a": np.arange(40.0)}, index=idx)
        timeline = pregime.active_feature_count_timeline(frame, ["z", "a"], min_history=5, min_train=10)
        assert timeline["active"].iloc[0] == ["a", "z"]

    def test_index_shorter_than_min_train_returns_empty_frame(self):
        idx = pd.date_range("2000-01-31", periods=5, freq="ME")
        frame = pd.DataFrame({"a": np.arange(5.0)}, index=idx)
        timeline = pregime.active_feature_count_timeline(frame, ["a"], min_history=2, min_train=10)
        assert timeline.empty
        assert list(timeline.columns) == ["n_active", "active"]

    @pytest.mark.skipif(
        not REAL_MONTHLY_FEATURES.exists(),
        reason="real platform monthly_features checkpoint not present",
    )
    def test_real_dev_features_reproduce_the_seven_a13_change_points(self):
        from trading_crab_lib.platform.config import load_platform_config
        from trading_crab_lib.platform.taxonomy import lean_feature_set

        cfg = load_platform_config()
        backtest_cfg = cfg["backtest"]
        features = pd.read_parquet(REAL_MONTHLY_FEATURES)
        cols = sorted(lean_feature_set(cfg) & set(features.columns))

        timeline = pregime.active_feature_count_timeline(
            features,
            cols,
            min_history=backtest_cfg["feature_min_history"],
            min_train=backtest_cfg["min_train_months"],
        )

        assert len(timeline) == 588
        assert int(timeline["n_active"].min()) >= 4
        assert int(timeline["n_active"].max()) <= 13

        change_dates = pregime.feature_set_change_dates(timeline)
        observed = [(str(d.date()), int(timeline.loc[d, "n_active"])) for d in change_dates]
        assert observed == EXPECTED_CHANGE_POINTS


class TestFeatureSetChangeDates:
    def test_constant_count_returns_empty_list(self):
        idx = pd.date_range("2000-01-31", periods=20, freq="ME")
        timeline = pd.DataFrame({"n_active": [5] * 20, "active": [["a"]] * 20}, index=idx)
        assert pregime.feature_set_change_dates(timeline) == []

    def test_includes_the_first_step_when_the_count_later_changes(self):
        idx = pd.date_range("2000-01-31", periods=6, freq="ME")
        timeline = pd.DataFrame({"n_active": [2, 2, 3, 3, 4, 4]}, index=idx)
        dates = pregime.feature_set_change_dates(timeline)
        assert dates == [idx[0], idx[2], idx[4]]

    def test_empty_input(self):
        assert pregime.feature_set_change_dates(pd.DataFrame()) == []


class TestPlotActiveFeatureCount:
    def test_does_not_crash(self, tmp_path):
        idx = pd.date_range("2000-01-31", periods=30, freq="ME")
        timeline = pd.DataFrame({"n_active": [4] * 10 + [6] * 10 + [9] * 10}, index=idx)
        save_path = tmp_path / "active.png"
        fig = pregime.plot_active_feature_count(timeline, save_path=save_path)
        assert isinstance(fig, plt.Figure)
        assert save_path.exists()

    def test_empty_input(self):
        assert isinstance(pregime.plot_active_feature_count(pd.DataFrame()), plt.Figure)


class TestLabelDisagreement:
    def test_identical_series_report_zero(self, states):
        result = pregime.label_disagreement(states, states)
        assert result["pct_disagree"] == 0.0
        assert result["n_compared"] == len(states)
        assert result["n_disagree"] == 0

    def test_half_disagreement_reports_one_half(self, monthly_index):
        reference = pd.Series([0] * len(monthly_index), index=monthly_index)
        comparison = reference.copy()
        comparison.iloc[: len(monthly_index) // 2] = 1
        result = pregime.label_disagreement(reference, comparison)
        assert result["n_compared"] == len(monthly_index)
        assert result["pct_disagree"] == pytest.approx(0.5)

    def test_disjoint_indexes_report_zero_compared_without_raising(self):
        left = pd.Series([0, 1, 2], index=pd.date_range("1970-01-31", periods=3, freq="ME"))
        right = pd.Series([0, 1, 2], index=pd.date_range("2010-01-31", periods=3, freq="ME"))
        result = pregime.label_disagreement(left, right)
        assert result["n_compared"] == 0
        assert result["pct_disagree"] == 0.0
        assert result["first_common_date"] is None
        assert result["per_state_confusion"].empty

    def test_partial_overlap_uses_only_the_intersection(self, monthly_index):
        reference = pd.Series(0, index=monthly_index)
        comparison = pd.Series(1, index=monthly_index[100:])
        result = pregime.label_disagreement(reference, comparison)
        assert result["n_compared"] == len(monthly_index) - 100
        assert result["pct_disagree"] == 1.0
        assert result["first_common_date"] == monthly_index[100]
        assert result["last_common_date"] == monthly_index[-1]

    def test_pct_disagree_is_always_a_proportion(self, states):
        shuffled = states.sample(frac=1.0, random_state=3).sort_index()
        result = pregime.label_disagreement(states, shuffled)
        assert 0.0 <= result["pct_disagree"] <= 1.0

    def test_confusion_frame_counts_sum_to_n_compared(self, states, monthly_index):
        comparison = pd.Series((states.to_numpy() + 1) % 5, index=monthly_index)
        result = pregime.label_disagreement(states, comparison)
        assert int(result["per_state_confusion"].to_numpy().sum()) == result["n_compared"]

    def test_empty_input(self):
        result = pregime.label_disagreement(pd.Series(dtype=float), pd.Series(dtype=float))
        assert result["n_compared"] == 0


class TestPlotLabelComparison:
    def test_two_series_with_different_spans_and_lengths(self, states, tmp_path):
        shorter = states.iloc[120:].copy()
        shorter.iloc[:] = (shorter.to_numpy() + 2) % 5
        save_path = tmp_path / "compare.png"
        fig = pregime.plot_label_comparison(
            {"reference (1970-)": states, "filtered (1980-)": shorter},
            change_dates=[states.index[50], states.index[200]],
            save_path=save_path,
        )
        assert isinstance(fig, plt.Figure)
        assert save_path.exists()

    def test_three_labelings(self, states):
        fig = pregime.plot_label_comparison(
            {
                "reference": states,
                "filtered": states.iloc[60:],
                "shipped": states.iloc[240:],
            }
        )
        assert isinstance(fig, plt.Figure)

    def test_disjoint_labelings_do_not_crash(self):
        left = pd.Series([0, 1, 2], index=pd.date_range("1970-01-31", periods=3, freq="ME"))
        right = pd.Series([3, 4, 0], index=pd.date_range("2010-01-31", periods=3, freq="ME"))
        assert isinstance(pregime.plot_label_comparison({"a": left, "b": right}), plt.Figure)

    def test_empty_input(self):
        assert isinstance(pregime.plot_label_comparison({}), plt.Figure)

    def test_all_series_empty(self):
        fig = pregime.plot_label_comparison({"a": pd.Series(dtype=float)})
        assert isinstance(fig, plt.Figure)
