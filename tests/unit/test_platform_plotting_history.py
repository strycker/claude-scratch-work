"""Tests for src/trading_crab_lib/platform/plotting/history.py.

No test in this module performs network I/O — the only outbound-call path
(:func:`load_usrec_or_warn`) is exercised exclusively through a monkeypatched
``fredapi.Fred`` that raises, and through a config with no API key.
"""

from __future__ import annotations

import logging

# matplotlib.use("Agg") must precede pyplot import — import order is intentional.
# pylint: disable=wrong-import-position,wrong-import-order
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from trading_crab_lib.platform.plotting import history as phist  # noqa: E402

# pylint: enable=wrong-import-position,wrong-import-order


@pytest.fixture
def monthly_index() -> pd.DatetimeIndex:
    return pd.date_range("1970-01-31", periods=600, freq="ME")


@pytest.fixture
def states(monthly_index) -> pd.Series:
    rng = np.random.default_rng(42)
    return pd.Series(rng.integers(0, 5, len(monthly_index)), index=monthly_index, name="state")


@pytest.fixture
def usrec(monthly_index) -> pd.Series:
    """One contiguous recession block over 1973-12 .. 1975-03."""
    series = pd.Series(0, index=monthly_index, name="usrec")
    block = (series.index >= pd.Timestamp("1973-12-31")) & (series.index <= pd.Timestamp("1975-03-31"))
    series.loc[block] = 1
    return series


class TestEconomicEvents:
    def test_exactly_six_entries(self):
        assert len(phist.ECONOMIC_EVENTS) == 6

    def test_every_entry_parses_with_start_before_end(self):
        for start, end, label in phist.ECONOMIC_EVENTS:
            start_ts = pd.Timestamp(start)
            end_ts = pd.Timestamp(end)
            assert start_ts < end_ts, f"{label}: {start} is not strictly before {end}"
            assert isinstance(label, str) and label

    def test_labels_are_unique(self):
        labels = [label for _, _, label in phist.ECONOMIC_EVENTS]
        assert len(set(labels)) == len(labels)


class TestLoadUsrecOrWarn:
    def test_missing_api_key_returns_none_and_warns(self, caplog):
        caplog.set_level(logging.WARNING, logger="trading_crab_lib.platform.plotting.history")
        result = phist.load_usrec_or_warn({"fred_monthly": {"api_key": None}})
        assert result is None
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        message = warnings[0].getMessage()
        assert "USREC" in message
        assert "FRED_API_KEY" in message

    def test_raising_client_returns_none_and_warns(self, caplog, monkeypatch):
        import fredapi

        class _RaisingFred:
            def __init__(self, *args, **kwargs):
                raise RuntimeError("simulated FRED outage")

        monkeypatch.setattr(fredapi, "Fred", _RaisingFred)
        caplog.set_level(logging.WARNING, logger="trading_crab_lib.platform.plotting.history")

        result = phist.load_usrec_or_warn({"fred_monthly": {"api_key": "not-a-real-key"}})

        assert result is None
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        message = warnings[0].getMessage()
        assert "USREC" in message
        assert "FRED_API_KEY" in message
        # D-10: the warning must name the consequence, not just the failure.
        assert "OMITTED" in message

    def test_empty_config_does_not_raise(self):
        assert phist.load_usrec_or_warn({}) is None

    def test_successful_fetch_returns_zero_one_month_end_series(self, monkeypatch):
        import fredapi

        daily = pd.Series(
            [0.0] * 40 + [1.0] * 40,
            index=pd.date_range("1973-01-01", periods=80, freq="D"),
        )

        class _StubFred:
            def __init__(self, *args, **kwargs):
                pass

            def get_series(self, series_id, observation_start=None, observation_end=None):
                assert series_id == "USREC"
                return daily

        monkeypatch.setattr(fredapi, "Fred", _StubFred)
        result = phist.load_usrec_or_warn({"fred_monthly": {"api_key": "k"}})

        assert result is not None
        assert set(result.unique()) <= {0, 1}
        assert str(result.dtype).startswith("int")


class TestRecessionPeriods:
    def test_one_contiguous_block(self, usrec):
        periods = phist.recession_periods(usrec)
        assert len(periods) == 1
        start, end = periods[0]
        assert start == pd.Timestamp("1973-12-31")
        assert end == pd.Timestamp("1975-03-31")

    def test_all_zero_returns_empty(self, monthly_index):
        assert phist.recession_periods(pd.Series(0, index=monthly_index)) == []

    def test_empty_input_returns_empty(self):
        assert phist.recession_periods(pd.Series(dtype=float)) == []

    def test_two_blocks(self, monthly_index):
        series = pd.Series(0, index=monthly_index)
        series.iloc[3:6] = 1
        series.iloc[20:22] = 1
        periods = phist.recession_periods(series)
        assert len(periods) == 2
        assert periods[0] == (monthly_index[3], monthly_index[5])
        assert periods[1] == (monthly_index[20], monthly_index[21])

    def test_block_running_to_the_final_month_is_closed(self, monthly_index):
        series = pd.Series(0, index=monthly_index)
        series.iloc[-4:] = 1
        periods = phist.recession_periods(series)
        assert periods == [(monthly_index[-4], monthly_index[-1])]


class TestRegimeEraContingency:
    def test_rows_sum_to_one_and_values_in_unit_interval(self, states, usrec):
        table = phist.regime_era_contingency(states, usrec=usrec)
        assert not table.empty
        row_sums = table.sum(axis=1)
        assert np.allclose(row_sums.to_numpy(), 1.0, atol=1e-9)
        values = table.to_numpy()
        assert values.min() >= 0.0
        assert values.max() <= 1.0

    def test_has_recession_row_when_usrec_supplied(self, states, usrec):
        table = phist.regime_era_contingency(states, usrec=usrec)
        assert phist.RECESSION_ERA_LABEL in table.index
        assert phist.BASELINE_ERA_LABEL in table.index

    def test_omits_recession_row_when_usrec_is_none(self, states):
        table = phist.regime_era_contingency(states, usrec=None)
        assert phist.RECESSION_ERA_LABEL not in table.index
        assert phist.BASELINE_ERA_LABEL in table.index
        assert np.allclose(table.sum(axis=1).to_numpy(), 1.0, atol=1e-9)

    def test_columns_are_state_ids(self, states):
        table = phist.regime_era_contingency(states, n_states=5)
        assert list(table.columns) == [0, 1, 2, 3, 4]

    def test_era_with_no_overlapping_months_is_omitted(self, monthly_index):
        # States covering only the 2010s — every pre-2000 event has zero overlap.
        idx = pd.date_range("2010-01-31", periods=60, freq="ME")
        short_states = pd.Series(0, index=idx)
        table = phist.regime_era_contingency(short_states)
        assert "1973 oil shock" not in table.index
        assert np.allclose(table.sum(axis=1).to_numpy(), 1.0, atol=1e-9)

    def test_empty_input(self):
        table = phist.regime_era_contingency(pd.Series(dtype=float))
        assert table.empty


class TestRegimeEraMarginals:
    def test_baseline_row_is_all_ones(self, states, usrec):
        table = phist.regime_era_marginals(states, usrec=usrec)
        baseline = table.loc[phist.BASELINE_ERA_LABEL].to_numpy()
        assert np.allclose(baseline, 1.0, atol=1e-9)

    def test_values_in_unit_interval(self, states, usrec):
        table = phist.regime_era_marginals(states, usrec=usrec)
        values = table.to_numpy()
        assert values.min() >= 0.0
        assert values.max() <= 1.0 + 1e-9

    def test_never_occupied_state_is_zero_not_nan(self, monthly_index):
        constant = pd.Series(1, index=monthly_index)
        table = phist.regime_era_marginals(constant, n_states=5)
        assert not table.isna().any().any()
        assert table[0].sum() == 0.0

    def test_empty_input(self):
        assert phist.regime_era_marginals(pd.Series(dtype=float)).empty


class TestPlotEraContingency:
    def test_does_not_crash(self, states, usrec, tmp_path):
        table = phist.regime_era_contingency(states, usrec=usrec)
        save_path = tmp_path / "contingency.png"
        fig = phist.plot_era_contingency(table, title="regime x era", save_path=save_path)
        assert isinstance(fig, plt.Figure)
        assert save_path.exists()

    def test_marginals_frame_also_renders(self, states, usrec):
        fig = phist.plot_era_contingency(phist.regime_era_marginals(states, usrec=usrec))
        assert isinstance(fig, plt.Figure)

    def test_empty_input(self):
        fig = phist.plot_era_contingency(pd.DataFrame())
        assert isinstance(fig, plt.Figure)
