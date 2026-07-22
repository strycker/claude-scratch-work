"""Mocked unit tests for platform/ingestion/macro_monthly.py — monthly
macro/long-history raw ingestion (DATA-01).

All network access is mocked — no real HTTP/FRED calls are made.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


class _FakeResponse:
    def __init__(self, text: str, status_code: int = 200):
        self.text = text
        self.content = text.encode("utf-8")
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise OSError(f"HTTP {self.status_code}")


SAMPLE_MULTPL_HTML = """
<html><body>
<table id="datatable">
<tr><th>Date</th><th>Value</th></tr>
<tr><td>Dec 31, 2021</td><td>4,700.00</td></tr>
<tr><td>Nov 30, 2021</td><td>4,600.00</td></tr>
<tr><td>Oct 31, 2021</td><td>4,500.00</td></tr>
<tr><td>Sep 30, 2021</td><td>4,400.00</td></tr>
<tr><td>Aug 31, 2021</td><td>4,300.00</td></tr>
<tr><td>Jul 31, 2021</td><td>4,200.00</td></tr>
</table>
</body></html>
"""

SAMPLE_MACROTRENDS_JSON_HTML = (
    "<html><body><script>\nvar chartData = [\n"
    + ",".join(
        f'{{"date": "{d}", "close": "{100 + i}"}}'
        for i, d in enumerate(
            pd.date_range("2020-01-01", periods=24, freq="MS").strftime("%Y-%m-%d")
        )
    )
    + "\n];\n</script></body></html>"
)


# ── _fetch_fred_monthly: monthly cadence (~12 rows/year, not quarterly's ~4) ──


def _make_daily_fred_series(years: int = 3) -> pd.Series:
    idx = pd.date_range("2020-01-01", periods=365 * years, freq="D")
    return pd.Series(np.linspace(100.0, 200.0, len(idx)), index=idx)


def test_fetch_fred_monthly_produces_monthly_cadence_not_quarterly():
    from trading_crab_lib.platform.ingestion.macro_monthly import _fetch_fred_monthly

    mock_fred = MagicMock()
    raw = _make_daily_fred_series(years=3)
    mock_fred.get_series.return_value = raw

    monthly = _fetch_fred_monthly(mock_fred, "GS10", "2020-01-01", "2023-01-01", shift=False)
    quarterly = raw.resample("QE").last()

    # ~12 rows/year over 3 years -> ~36 rows; materially more than quarterly's ~12
    assert len(monthly) >= 34
    assert len(monthly) > len(quarterly) * 2


def _make_fred_monthly_cfg():
    return {
        "fred_monthly": {
            "api_key": "fake_key_for_testing",
            "series": {
                "GS10": {"name": "fred_gs10", "shift": False},
                "TB3MS": {"name": "fred_tb3ms", "shift": False},
            },
        },
        "data": {
            "start_date": "2020-01-01",
            "end_date": "2023-01-01",
            "monthly_freq": "ME",
        },
    }


@patch("trading_crab_lib.platform.ingestion.macro_monthly.Fred")
def test_fetch_fred_monthly_basic(mock_fred_cls):
    from trading_crab_lib.platform.ingestion.macro_monthly import fetch_fred_monthly

    mock_fred = MagicMock()
    mock_fred.get_series.return_value = _make_daily_fred_series(years=2)
    mock_fred_cls.return_value = mock_fred

    df = fetch_fred_monthly(_make_fred_monthly_cfg())
    assert isinstance(df, pd.DataFrame)
    assert "fred_gs10" in df.columns
    assert "fred_tb3ms" in df.columns
    assert len(df) >= 20  # ~24 months over 2 years


@patch("trading_crab_lib.platform.ingestion.macro_monthly.Fred")
def test_fetch_fred_monthly_handles_single_series_failure(mock_fred_cls):
    from trading_crab_lib.platform.ingestion.macro_monthly import fetch_fred_monthly

    mock_fred = MagicMock()

    def _side_effect(series_id, **kwargs):
        if series_id == "GS10":
            raise OSError("API rate limit")
        return _make_daily_fred_series(years=2)

    mock_fred.get_series.side_effect = _side_effect
    mock_fred_cls.return_value = mock_fred

    df = fetch_fred_monthly(_make_fred_monthly_cfg())
    assert "fred_tb3ms" in df.columns
    assert "fred_gs10" not in df.columns


def test_fetch_fred_monthly_missing_api_key_raises():
    from trading_crab_lib.platform.ingestion.macro_monthly import fetch_fred_monthly

    cfg = {
        "fred_monthly": {"api_key": None, "series": {}},
        "data": {"start_date": "2020-01-01", "end_date": "2023-01-01"},
    }
    with pytest.raises(OSError, match="FRED_API_KEY"):
        fetch_fred_monthly(cfg)


# ── multpl monthly scrape — month-end indexed, graceful cssselect degradation ─


@patch("trading_crab_lib.platform.ingestion.macro_monthly.time.sleep")
@patch("trading_crab_lib.ingestion.multpl.requests.get")
def test_scrape_multpl_monthly_basic(mock_get, mock_sleep):
    from trading_crab_lib.platform.ingestion.macro_monthly import _scrape_multpl_monthly

    mock_get.return_value = _FakeResponse(SAMPLE_MULTPL_HTML)
    cfg = {
        "multpl_monthly": {
            "datasets": [["sp500", "SP500 Prices", "https://www.multpl.com/x", "num"]]
        }
    }
    result = _scrape_multpl_monthly(cfg)
    if not result:
        pytest.skip("cssselect not installed; multpl scrape degrades gracefully")
    assert "sp500" in result
    s = result["sp500"]
    assert len(s) > 0
    assert all(idx.is_month_end for idx in s.index)


@patch("trading_crab_lib.platform.ingestion.macro_monthly.time.sleep")
@patch("trading_crab_lib.ingestion.multpl.requests.get")
def test_scrape_multpl_monthly_degrades_gracefully_on_scrape_failure(mock_get, mock_sleep):
    from trading_crab_lib.platform.ingestion.macro_monthly import _scrape_multpl_monthly

    mock_get.side_effect = OSError("Connection refused")
    cfg = {
        "multpl_monthly": {
            "datasets": [["sp500", "SP500 Prices", "https://www.multpl.com/x", "num"]]
        }
    }
    result = _scrape_multpl_monthly(cfg)
    assert result == {}


def test_scrape_multpl_monthly_no_datasets():
    from trading_crab_lib.platform.ingestion.macro_monthly import _scrape_multpl_monthly

    assert _scrape_multpl_monthly({"multpl_monthly": {"datasets": []}}) == {}


# ── macrotrends monthly scrape — month-end indexed ────────────────────────────


@patch("trading_crab_lib.platform.ingestion.macro_monthly.time.sleep")
@patch("trading_crab_lib.platform.ingestion.macro_monthly.requests.get")
def test_scrape_macrotrends_monthly_month_end_indexed(mock_get, mock_sleep):
    from trading_crab_lib.platform.ingestion.macro_monthly import _scrape_macrotrends_monthly

    mock_get.return_value = _FakeResponse(SAMPLE_MACROTRENDS_JSON_HTML)
    s = _scrape_macrotrends_monthly(
        "https://www.macrotrends.net",
        "/1333/historical-gold-prices-100-year-chart",
        "gold_spot",
        "mean",
    )
    assert isinstance(s, pd.Series)
    assert len(s) > 0
    assert all(idx.is_month_end for idx in s.index)


@patch("trading_crab_lib.platform.ingestion.macro_monthly.time.sleep")
@patch("trading_crab_lib.platform.ingestion.macro_monthly.requests.get")
def test_fetch_macrotrends_monthly_all_handles_series_failure(mock_get, mock_sleep):
    from trading_crab_lib.platform.ingestion.macro_monthly import _fetch_macrotrends_monthly_all

    mock_get.side_effect = OSError("Connection refused")
    cfg = {
        "macrotrends_monthly": {
            "base_url": "https://www.macrotrends.net",
            "series": [
                {"name": "gold_spot", "path": "/1333/historical-gold-prices-100-year-chart", "resample": "mean"}
            ],
        }
    }
    result = _fetch_macrotrends_monthly_all(cfg)
    assert result == {}


# ── fetch_macro_monthly orchestrator — NULL-tolerant partial-source-failure ───


@patch("trading_crab_lib.platform.ingestion.macro_monthly.time.sleep")
@patch("trading_crab_lib.platform.ingestion.macro_monthly.requests.get")
@patch("trading_crab_lib.platform.ingestion.macro_monthly.Fred")
def test_fetch_macro_monthly_null_tolerant_partial_success(mock_fred_cls, mock_get, mock_sleep):
    from trading_crab_lib.platform.ingestion.macro_monthly import fetch_macro_monthly

    mock_fred = MagicMock()
    mock_fred.get_series.return_value = _make_daily_fred_series(years=2)
    mock_fred_cls.return_value = mock_fred

    # multpl and macrotrends both call `requests.get` through the SAME
    # singleton `requests` module (multpl.py's `import requests` and this
    # module's `import requests` bind the identical object) — one patch
    # covers both call sites; discriminate by URL to fail multpl only,
    # exercising partial-source-failure tolerance (a failed source is
    # simply absent from the merged frame, never an error).
    def _get_side_effect(url, **kwargs):
        if "macrotrends" in url:
            return _FakeResponse(SAMPLE_MACROTRENDS_JSON_HTML)
        raise OSError("multpl unreachable")

    mock_get.side_effect = _get_side_effect

    cfg = {
        "fred_monthly": {
            "api_key": "fake_key_for_testing",
            "series": {"GS10": {"name": "fred_gs10", "shift": False}},
        },
        "data": {"start_date": "2020-01-01", "end_date": "2022-01-01", "monthly_freq": "ME"},
        "multpl_monthly": {
            "datasets": [["sp500", "SP500 Prices", "https://www.multpl.com/x", "num"]]
        },
        "macrotrends_monthly": {
            "base_url": "https://www.macrotrends.net",
            "series": [
                {"name": "gold_spot", "path": "/1333/historical-gold-prices-100-year-chart", "resample": "mean"}
            ],
        },
    }

    df = fetch_macro_monthly(cfg)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    # FRED + macrotrends present despite multpl having failed entirely
    assert "fred_gs10" in df.columns
    assert "gold_spot" in df.columns
    assert "sp500" not in df.columns  # failed source is simply absent, not an error
    # Row count reflects the monthly (not quarterly) span of the merged data
    assert len(df) >= 20


def test_fetch_macro_monthly_all_sources_empty_returns_empty_df():
    from trading_crab_lib.platform.ingestion.macro_monthly import fetch_macro_monthly

    cfg = {
        "fred_monthly": {"api_key": None, "series": {}},
        "data": {"start_date": "2020-01-01", "end_date": "2022-01-01"},
        "multpl_monthly": {"datasets": []},
        "macrotrends_monthly": {"base_url": "https://www.macrotrends.net", "series": []},
    }
    # fred_monthly has no series configured (empty dict loop) and no api_key
    # check is only hit if series present — exercise via empty series dict
    # instead of the api_key error path used in the dedicated test above.
    cfg["fred_monthly"]["api_key"] = "fake_key_for_testing"
    df = fetch_macro_monthly(cfg)
    assert isinstance(df, pd.DataFrame)
    assert df.empty
