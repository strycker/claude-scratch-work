"""Mocked unit tests for platform/ingestion/alfred.py — ALFRED point-in-time
vintage ingestion (DATA-03).

All network access is mocked — no real HTTP/FRED calls are made.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


def _make_all_releases_df() -> pd.DataFrame:
    """Synthetic get_series_all_releases() response for one reference period
    (2020-01-01) with two revisions: an original vintage (value=100, known
    from 2020-02-01) and a later revision (value=150, known from 2020-05-01).
    """
    return pd.DataFrame(
        {
            "realtime_start": pd.to_datetime(["2020-02-01", "2020-05-01"]),
            "realtime_end": pd.to_datetime(["2020-05-01", "2099-12-31"]),
            "date": pd.to_datetime(["2020-01-01", "2020-01-01"]),
            "value": [100.0, 150.0],
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

    df = pd.DataFrame(columns=["realtime_start", "date", "value"])  # no realtime_end
    with pytest.raises(ValueError, match="realtime_end"):
        _detect_vintage_columns(df)


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


# ── align_with_fallback ──────────────────────────────────────────────────────


def test_pre_vintage_fallback():
    from trading_crab_lib.platform.ingestion.alfred import align_with_fallback

    all_releases = _make_all_releases_df()  # earliest vintage: 2020-02-01
    as_of_dates = pd.DatetimeIndex(["2019-01-01", "2020-06-01"])
    shift_series = pd.Series(
        {pd.Timestamp("2019-01-01"): 42.0, pd.Timestamp("2020-06-01"): 999.0}
    )

    result = align_with_fallback(all_releases, as_of_dates, shift_series)

    # Before the earliest recorded vintage: falls back to the shift value —
    # never NaN, never a raised error (D-06).
    assert result.loc[pd.Timestamp("2019-01-01")] == pytest.approx(42.0)
    # At/after the vintage era: uses the reconstructed point-in-time value,
    # not the fallback.
    assert result.loc[pd.Timestamp("2020-06-01")] == pytest.approx(150.0)


# ── fetch_vintage_series ──────────────────────────────────────────────────────


@patch("trading_crab_lib.platform.ingestion.alfred.Fred")
def test_fetch_vintage_series_calls_bulk_endpoint_once(mock_fred_cls):
    from trading_crab_lib.platform.ingestion.alfred import fetch_vintage_series

    mock_fred = MagicMock()
    mock_fred.get_series_all_releases.return_value = _make_all_releases_df()

    result = fetch_vintage_series(mock_fred, "PAYEMS")

    mock_fred.get_series_all_releases.assert_called_once_with("PAYEMS")
    assert isinstance(result, pd.DataFrame)


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
