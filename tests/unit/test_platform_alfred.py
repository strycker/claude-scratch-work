"""Mocked unit tests for platform/ingestion/alfred.py — ALFRED point-in-time
vintage ingestion (DATA-03).

All network access is mocked — no real HTTP/FRED calls are made.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


def _make_three_column_all_releases_df() -> pd.DataFrame:
    """Synthetic get_series_all_releases() response in the REAL fredapi shape
    — realtime_start / date / value only, no realtime_end column.

    Same two-revision structure as :func:`_make_all_releases_df`:

    - 2020-01-01: two revisions — an original vintage (value=100, known from
      2020-02-01) and a later revision (value=150, known from 2020-05-01).
    - 2020-06-01: a single vintage (value=200, known from 2020-06-15).
    """
    return pd.DataFrame(
        {
            "realtime_start": pd.to_datetime(
                ["2020-02-01", "2020-05-01", "2020-06-15"]
            ),
            "date": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-06-01"]),
            "value": [100.0, 150.0, 200.0],
        }
    )


def _make_all_releases_df() -> pd.DataFrame:
    """Synthetic get_series_all_releases() response with two reference periods:

    - 2020-01-01: two revisions — an original vintage (value=100, known from
      2020-02-01) and a later revision (value=150, known from 2020-05-01).
    - 2020-06-01: a single vintage (value=200, known from 2020-06-15) — used
      to exercise the "at/after the earliest vintage" branch of
      align_with_fallback with a matching reference period.
    """
    return pd.DataFrame(
        {
            "realtime_start": pd.to_datetime(
                ["2020-02-01", "2020-05-01", "2020-06-15"]
            ),
            "realtime_end": pd.to_datetime(
                ["2020-05-01", "2099-12-31", "2099-12-31"]
            ),
            "date": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-06-01"]),
            "value": [100.0, 150.0, 200.0],
        }
    )


# ── _detect_vintage_columns ─────────────────────────────────────────────────


def test_detect_vintage_columns_case_insensitive():
    from trading_crab_lib.platform.ingestion.alfred import _detect_vintage_columns

    df = pd.DataFrame(
        columns=["Realtime_Start", "REALTIME_END", "Date", "VALUE"]
    )
    mapping = _detect_vintage_columns(df)
    assert mapping == {
        "realtime_start": "Realtime_Start",
        "realtime_end": "REALTIME_END",
        "date": "Date",
        "value": "VALUE",
    }


def test_detect_vintage_columns_raises_on_missing_role():
    from trading_crab_lib.platform.ingestion.alfred import _detect_vintage_columns

    df = pd.DataFrame(columns=["realtime_start", "date"])  # no value — genuinely required
    with pytest.raises(ValueError, match="value"):
        _detect_vintage_columns(df)


def test_detect_vintage_columns_accepts_real_fredapi_three_column_shape():
    from trading_crab_lib.platform.ingestion.alfred import _detect_vintage_columns

    df = _make_three_column_all_releases_df()
    mapping = _detect_vintage_columns(df)
    assert set(mapping) == {"realtime_start", "date", "value"}
    assert "realtime_end" not in mapping


def test_detect_vintage_columns_maps_realtime_end_when_present():
    from trading_crab_lib.platform.ingestion.alfred import _detect_vintage_columns

    df = _make_all_releases_df()  # four-column frame, carries realtime_end
    mapping = _detect_vintage_columns(df)
    assert mapping["realtime_end"] == "realtime_end"


# ── value_as_of ──────────────────────────────────────────────────────────────


def test_value_as_of_respects_vintage():
    from trading_crab_lib.platform.ingestion.alfred import value_as_of

    all_releases = _make_all_releases_df()

    # Between the two revisions: only the first vintage (100) was known.
    between = value_as_of(all_releases, pd.Timestamp("2020-03-01"))
    assert between.loc[pd.Timestamp("2020-01-01")] == pytest.approx(100.0)

    # After the second revision: the later value (150) is now known.
    after = value_as_of(all_releases, pd.Timestamp("2020-06-01"))
    assert after.loc[pd.Timestamp("2020-01-01")] == pytest.approx(150.0)


def test_value_as_of_ignores_future_revisions():
    from trading_crab_lib.platform.ingestion.alfred import value_as_of

    all_releases = _make_all_releases_df()

    # Before the first vintage was even published: nothing is known yet.
    before = value_as_of(all_releases, pd.Timestamp("2020-01-15"))
    assert before.empty


def test_value_as_of_on_three_column_frame():
    """Proves reconstruction genuinely works on the real (three-column)
    fredapi shape, not only on the padded four-column fixture."""
    from trading_crab_lib.platform.ingestion.alfred import value_as_of

    all_releases = _make_three_column_all_releases_df()

    # Between the two revisions: only the first vintage (100) was known.
    between = value_as_of(all_releases, pd.Timestamp("2020-03-01"))
    assert between.loc[pd.Timestamp("2020-01-01")] == pytest.approx(100.0)

    # After the second revision: the later value (150) is now known.
    after = value_as_of(all_releases, pd.Timestamp("2020-06-01"))
    assert after.loc[pd.Timestamp("2020-01-01")] == pytest.approx(150.0)


# ── align_with_fallback ──────────────────────────────────────────────────────


def test_pre_vintage_fallback():
    from trading_crab_lib.platform.ingestion.alfred import align_with_fallback

    all_releases = _make_all_releases_df()  # earliest vintage: 2020-02-01
    # Second as-of date matches the 2020-06-01 reference period, which was
    # first published (realtime_start) on 2020-06-15 — already known by
    # 2020-06-20.
    as_of_dates = pd.DatetimeIndex(["2019-01-01", "2020-06-20"])
    shift_series = pd.Series(
        {pd.Timestamp("2019-01-01"): 42.0, pd.Timestamp("2020-06-20"): 999.0}
    )

    result = align_with_fallback(all_releases, as_of_dates, shift_series)

    # Before the earliest recorded vintage: falls back to the shift value —
    # never NaN, never a raised error (D-06).
    assert result.loc[pd.Timestamp("2019-01-01")] == pytest.approx(42.0)
    # At/after the vintage era: uses the reconstructed point-in-time value
    # for the matching reference period, not the fallback.
    assert result.loc[pd.Timestamp("2020-06-20")] == pytest.approx(200.0)


# ── fetch_vintage_series ──────────────────────────────────────────────────────


@patch("trading_crab_lib.platform.ingestion.alfred.Fred")
def test_fetch_vintage_series_calls_bulk_endpoint_once(mock_fred_cls):
    from trading_crab_lib.platform.ingestion.alfred import fetch_vintage_series

    mock_fred = MagicMock()
    mock_fred.get_series_all_releases.return_value = _make_all_releases_df()

    result = fetch_vintage_series(mock_fred, "PAYEMS")

    mock_fred.get_series_all_releases.assert_called_once_with("PAYEMS")
    assert isinstance(result, pd.DataFrame)


@patch("trading_crab_lib.platform.ingestion.alfred.Fred")
def test_fetch_vintage_series_accepts_three_column_response(mock_fred_cls):
    """The real fredapi response (three columns, no realtime_end) must not
    raise — this is exactly the shape returned by every live call."""
    from trading_crab_lib.platform.ingestion.alfred import fetch_vintage_series

    mock_fred = MagicMock()
    mock_fred.get_series_all_releases.return_value = _make_three_column_all_releases_df()

    result = fetch_vintage_series(mock_fred, "PAYEMS")

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["realtime_start", "date", "value"]


# ── fetch_all_vintages ────────────────────────────────────────────────────────


def _make_vintage_cfg():
    return {
        "fred_vintage": {
            "api_key": "fake_key_for_testing",
            "series": {
                "GDPC1": {"name": "fred_gdp", "tier": "agency"},
                "CPIAUCSL": {"name": "fred_cpi", "tier": "agency"},
            },
        }
    }


@patch("trading_crab_lib.platform.ingestion.alfred.Fred")
def test_fetch_all_vintages_basic(mock_fred_cls):
    from trading_crab_lib.platform.ingestion.alfred import fetch_all_vintages

    mock_fred = MagicMock()
    mock_fred.get_series_all_releases.return_value = _make_all_releases_df()
    mock_fred_cls.return_value = mock_fred

    result = fetch_all_vintages(_make_vintage_cfg())
    assert "fred_gdp" in result
    assert "fred_cpi" in result


@patch("trading_crab_lib.platform.ingestion.alfred.Fred")
def test_fetch_all_vintages_handles_single_series_failure(mock_fred_cls):
    from trading_crab_lib.platform.ingestion.alfred import fetch_all_vintages

    mock_fred = MagicMock()

    def _side_effect(series_id):
        if series_id == "GDPC1":
            raise OSError("API rate limit")
        return _make_all_releases_df()

    mock_fred.get_series_all_releases.side_effect = _side_effect
    mock_fred_cls.return_value = mock_fred

    result = fetch_all_vintages(_make_vintage_cfg())
    assert "fred_cpi" in result
    assert "fred_gdp" not in result


def test_fetch_all_vintages_missing_api_key_raises():
    from trading_crab_lib.platform.ingestion.alfred import fetch_all_vintages

    cfg = {"fred_vintage": {"api_key": None, "series": {}}}
    with pytest.raises(OSError, match="FRED_API_KEY"):
        fetch_all_vintages(cfg)
