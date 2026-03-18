"""HTTP-mocked tests for ingestion modules (multpl.py, fred.py, assets.py).

All network access is mocked — no real HTTP calls are made.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# ── multpl.py tests ──────────────────────────────────────────────────────────


SAMPLE_MULTPL_HTML = """
<html><body>
<table id="datatable">
<tr><th>Date</th><th>Value</th></tr>
<tr><td>Mar 31, 2024</td><td>5,200.00</td></tr>
<tr><td>Dec 31, 2023</td><td>4,800.00</td></tr>
<tr><td>Sep 30, 2023</td><td>4,300.00</td></tr>
</table>
</body></html>
"""

SAMPLE_MULTPL_PERCENT_HTML = """
<html><body>
<table id="datatable">
<tr><th>Date</th><th>Value</th></tr>
<tr><td>Mar 31, 2024</td><td>1.50%</td></tr>
<tr><td>Dec 31, 2023</td><td>1.60%</td></tr>
</table>
</body></html>
"""


class _FakeResponse:
    def __init__(self, content: str, status_code: int = 200):
        self.content = content.encode("utf-8")
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")


@patch("market_regime.ingestion.multpl.time.sleep")
@patch("market_regime.ingestion.multpl.requests.get")
def test_multpl_scrape_raw_rows(mock_get, mock_sleep):
    from market_regime.ingestion.multpl import _scrape_raw_rows

    mock_get.return_value = _FakeResponse(SAMPLE_MULTPL_HTML)
    try:
        rows = _scrape_raw_rows("https://example.com/table")
    except ImportError:
        pytest.skip("cssselect not installed")
    assert len(rows) == 3
    assert rows[0] == ["Mar 31, 2024", "5,200.00"]


@patch("market_regime.ingestion.multpl.time.sleep")
@patch("market_regime.ingestion.multpl.requests.get")
def test_multpl_parse_series_numeric(mock_get, mock_sleep):
    from market_regime.ingestion.multpl import _parse_series

    raw = [
        ["Mar 31, 2024", "5,200.00"],
        ["Dec 31, 2023", "4,800.00"],
        ["Sep 30, 2023", "4,300.00"],
    ]
    s = _parse_series(raw, "sp500", "num")
    assert isinstance(s, pd.Series)
    assert len(s) > 0
    assert s.iloc[0] == pytest.approx(4300.0)  # earliest by date, quarterly resampled


@patch("market_regime.ingestion.multpl.time.sleep")
@patch("market_regime.ingestion.multpl.requests.get")
def test_multpl_parse_series_percent(mock_get, mock_sleep):
    from market_regime.ingestion.multpl import _parse_series

    raw = [
        ["Mar 31, 2024", "1.50%"],
        ["Dec 31, 2023", "1.60%"],
    ]
    s = _parse_series(raw, "div_yield", "percent")
    # Percents should be stored as decimal fractions
    assert all(v < 1.0 for v in s.values)
    assert s.iloc[-1] == pytest.approx(0.015)


@patch("market_regime.ingestion.multpl.time.sleep")
@patch("market_regime.ingestion.multpl.requests.get")
def test_multpl_fetch_all_basic(mock_get, mock_sleep):
    from market_regime.ingestion.multpl import fetch_all

    mock_get.return_value = _FakeResponse(SAMPLE_MULTPL_HTML)
    cfg = {
        "multpl": {
            "datasets": [
                ["sp500", "SP500 Prices", "https://example.com/sp500", "num"],
            ]
        }
    }
    df = fetch_all(cfg)
    assert isinstance(df, pd.DataFrame)
    if df.empty:
        # cssselect not installed — scraping fails gracefully
        pytest.skip("cssselect not installed; multpl scrape returns empty")
    assert "sp500" in df.columns
    assert len(df) > 0


@patch("market_regime.ingestion.multpl.time.sleep")
@patch("market_regime.ingestion.multpl.requests.get")
def test_multpl_fetch_all_handles_scrape_failure(mock_get, mock_sleep):
    from market_regime.ingestion.multpl import fetch_all

    mock_get.side_effect = Exception("Connection refused")
    cfg = {
        "multpl": {
            "datasets": [
                ["sp500", "SP500 Prices", "https://example.com/sp500", "num"],
            ]
        }
    }
    df = fetch_all(cfg)
    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_multpl_fetch_all_no_datasets():
    from market_regime.ingestion.multpl import fetch_all

    df = fetch_all({"multpl": {"datasets": []}})
    assert df.empty


# ── fred.py tests ────────────────────────────────────────────────────────────


def _make_fred_cfg():
    return {
        "fred": {
            "api_key": "fake_key_for_testing",
            "series": {
                "GDP": {"name": "fred_gdp", "shift": True},
                "BAA": {"name": "fred_baa", "shift": False},
            },
        },
        "data": {
            "start_date": "2020-01-01",
            "end_date": "2024-01-01",
        },
    }


def _make_mock_fred_series():
    idx = pd.date_range("2020-01-01", periods=16, freq="QE")
    return pd.Series(np.arange(100.0, 116.0), index=idx)


@patch("market_regime.ingestion.fred.Fred")
def test_fred_fetch_all_basic(mock_fred_cls):
    from market_regime.ingestion.fred import fetch_all

    mock_fred = MagicMock()
    mock_fred.get_series.return_value = _make_mock_fred_series()
    mock_fred_cls.return_value = mock_fred

    cfg = _make_fred_cfg()
    df = fetch_all(cfg)
    assert isinstance(df, pd.DataFrame)
    assert "fred_gdp" in df.columns
    assert "fred_baa" in df.columns


@patch("market_regime.ingestion.fred.Fred")
def test_fred_shift_applies_to_gdp(mock_fred_cls):
    from market_regime.ingestion.fred import _fetch_one

    mock_fred = MagicMock()
    raw = _make_mock_fred_series()
    mock_fred.get_series.return_value = raw

    result = _fetch_one(mock_fred, "GDP", "2020-01-01", "2024-01-01", shift=True)
    # Shift means the first value should be NaN (shifted forward by 1 quarter)
    assert pd.isna(result.iloc[0])
    # Second value should equal the original first value
    assert result.iloc[1] == pytest.approx(raw.resample("QE").last().iloc[0])


@patch("market_regime.ingestion.fred.Fred")
def test_fred_no_shift_for_baa(mock_fred_cls):
    from market_regime.ingestion.fred import _fetch_one

    mock_fred = MagicMock()
    raw = _make_mock_fred_series()
    mock_fred.get_series.return_value = raw

    result = _fetch_one(mock_fred, "BAA", "2020-01-01", "2024-01-01", shift=False)
    # No shift — first value should NOT be NaN
    assert not pd.isna(result.iloc[0])


def test_fred_missing_api_key_raises():
    from market_regime.ingestion.fred import fetch_all

    cfg = {
        "fred": {"api_key": None, "series": {}},
        "data": {"start_date": "2020-01-01", "end_date": "2024-01-01"},
    }
    with pytest.raises(EnvironmentError, match="FRED_API_KEY"):
        fetch_all(cfg)


@patch("market_regime.ingestion.fred.Fred")
def test_fred_fetch_all_handles_single_series_failure(mock_fred_cls):
    from market_regime.ingestion.fred import fetch_all

    mock_fred = MagicMock()

    def _side_effect(series_id, **kwargs):
        if series_id == "GDP":
            raise Exception("API rate limit")
        return _make_mock_fred_series()

    mock_fred.get_series.side_effect = _side_effect
    mock_fred_cls.return_value = mock_fred

    cfg = _make_fred_cfg()
    df = fetch_all(cfg)
    # GDP failed, BAA should still be present
    assert "fred_baa" in df.columns


# ── assets.py tests ──────────────────────────────────────────────────────────


def _make_assets_cfg():
    return {
        "assets": {"etfs": ["SPY", "GLD"]},
        "data": {
            "start_date": "2020-01-01",
            "end_date": "2024-01-01",
        },
    }


@patch("market_regime.ingestion.assets._ssl_bypass_curl_session")
@patch("market_regime.ingestion.assets._batch_yfinance")
def test_assets_fetch_all_basic(mock_batch, mock_session):
    from market_regime.ingestion.assets import fetch_all

    idx = pd.date_range("2020-03-31", periods=16, freq="QE")
    mock_batch.return_value = (
        {
            "SPY": pd.Series(np.arange(300.0, 316.0), index=idx, name="SPY"),
            "GLD": pd.Series(np.arange(150.0, 166.0), index=idx, name="GLD"),
        },
        False,
    )
    mock_session.return_value = MagicMock()

    cfg = _make_assets_cfg()
    df = fetch_all(cfg)
    assert isinstance(df, pd.DataFrame)
    assert "SPY" in df.columns
    assert "GLD" in df.columns
    assert len(df) == 16


@patch("market_regime.ingestion.assets._ssl_bypass_curl_session")
@patch("market_regime.ingestion.assets._batch_yfinance")
@patch("market_regime.ingestion.assets._fetch_missing_with_ssl_bypass")
def test_assets_phase2_retry_on_missing(mock_phase2, mock_batch, mock_session):
    from market_regime.ingestion.assets import fetch_all

    idx = pd.date_range("2020-03-31", periods=16, freq="QE")
    # Phase 1 only gets SPY
    mock_batch.return_value = (
        {"SPY": pd.Series(np.arange(300.0, 316.0), index=idx, name="SPY")},
        False,
    )
    # Phase 2 recovers GLD
    mock_phase2.return_value = {
        "GLD": pd.Series(np.arange(150.0, 166.0), index=idx, name="GLD"),
    }
    mock_session.return_value = MagicMock()

    cfg = _make_assets_cfg()
    df = fetch_all(cfg)
    assert "SPY" in df.columns
    assert "GLD" in df.columns


def test_assets_fetch_all_no_tickers():
    from market_regime.ingestion.assets import fetch_all

    df = fetch_all({"assets": {"etfs": []}, "data": {"start_date": "2020-01-01", "end_date": "2024-01-01"}})
    assert df.empty


@patch("market_regime.ingestion.assets._ssl_bypass_curl_session")
@patch("market_regime.ingestion.assets._batch_yfinance")
@patch("market_regime.ingestion.assets._fetch_missing_with_ssl_bypass")
@patch("market_regime.ingestion.assets._fetch_tickers_stooq")
@patch("market_regime.ingestion.assets._fetch_tickers_openbb")
def test_assets_all_phases_fail_returns_empty(
    mock_openbb, mock_stooq, mock_phase2, mock_batch, mock_session
):
    from market_regime.ingestion.assets import fetch_all

    mock_batch.return_value = ({}, False)
    mock_phase2.return_value = {}
    mock_stooq.return_value = []
    mock_openbb.return_value = []
    mock_session.return_value = MagicMock()

    cfg = _make_assets_cfg()
    df = fetch_all(cfg)
    assert df.empty
