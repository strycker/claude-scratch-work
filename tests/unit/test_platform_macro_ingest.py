"""Mocked unit tests for platform/ingestion/macro_monthly.py — monthly
macro/long-history raw ingestion (DATA-01).

All network access is mocked — no real HTTP/FRED calls are made.
"""
from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from trading_crab_lib.platform.ingestion import macro_monthly


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

# A "Just a moment..."-style Cloudflare interstitial: no embedded JSON, no
# <table> — macrotrends._html_yields_data must return False for this.
SAMPLE_MACROTRENDS_INTERSTITIAL_HTML = """
<html><body>
<div class="cf-browser-verification">Just a moment...</div>
<div id="challenge-running"></div>
</body></html>
"""

# Regression fixture for the 2026-08-05 residential diagnostic against the
# live macrotrends page: value column FIRST with a squashed multi-line
# header, date column SECOND titled plainly "Month" — exercises the "month"
# keyword match (and its value/date collision guard) rather than the
# df.columns[0] fallback silently doing the right thing by accident.
SAMPLE_MACROTRENDS_MERGED_HEADER_TABLE_HTML = """
<html><body>
<table class="table">
<thead><tr><th>Gold PricesMonthly Closing Price</th><th>Month</th></tr></thead>
<tbody>
<tr><td>1,560.00</td><td>2020-01-31</td></tr>
<tr><td>1,600.00</td><td>2020-02-29</td></tr>
<tr><td>1,650.00</td><td>2020-03-31</td></tr>
</tbody>
</table>
</body></html>
"""


# ── _fetch_fred_monthly: monthly cadence (~12 rows/year, not quarterly's ~4) ──


def _make_daily_fred_series(years: int = 3) -> pd.Series:
    idx = pd.date_range("2020-01-01", periods=365 * years, freq="D")
    return pd.Series(np.linspace(100.0, 200.0, len(idx)), index=idx)


def test_fetch_fred_monthly_produces_monthly_cadence_not_quarterly():
    from trading_crab_lib.platform.ingestion.macro_monthly import _fetch_fred_monthly

    mock_fred = MagicMock()
    raw = _make_daily_fred_series(years=3)
    mock_fred.get_series.return_value = raw

    monthly = _fetch_fred_monthly(mock_fred, "GS10", "2020-01-01", "2023-01-01")
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
@patch("trading_crab_lib.platform.ingestion.macro_monthly.http_get")
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


# ── macrotrends monthly scrape — browser fallback ─────────────────────────────
# playwright IS importable in this environment — every test that can reach
# _scrape_macrotrends_monthly patches fetch_page_html, or an unpatched
# fallback would launch a real browser.


@patch("trading_crab_lib.platform.ingestion.macro_monthly.fetch_page_html")
@patch("trading_crab_lib.platform.ingestion.macro_monthly.http_get")
def test_scrape_macrotrends_monthly_json_body_never_calls_fetch_page_html(mock_get, mock_fetch_page_html):
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
    mock_fetch_page_html.assert_not_called()


@patch("trading_crab_lib.platform.ingestion.macro_monthly.fetch_page_html")
@patch("trading_crab_lib.platform.ingestion.macro_monthly.http_get")
def test_scrape_macrotrends_monthly_table_body_never_calls_fetch_page_html(mock_get, mock_fetch_page_html):
    from trading_crab_lib.platform.ingestion.macro_monthly import _scrape_macrotrends_monthly

    mock_get.return_value = _FakeResponse(SAMPLE_MACROTRENDS_MERGED_HEADER_TABLE_HTML)

    s = _scrape_macrotrends_monthly(
        "https://www.macrotrends.net",
        "/1333/historical-gold-prices-100-year-chart",
        "gold_spot",
        "mean",
    )

    assert isinstance(s, pd.Series)
    assert len(s) > 0
    mock_fetch_page_html.assert_not_called()


@patch("trading_crab_lib.platform.ingestion.macro_monthly.fetch_page_html")
@patch("trading_crab_lib.platform.ingestion.macro_monthly.http_get")
def test_scrape_macrotrends_monthly_interstitial_falls_back_to_browser_render(mock_get, mock_fetch_page_html):
    from trading_crab_lib.platform.ingestion.macro_monthly import _scrape_macrotrends_monthly

    mock_get.return_value = _FakeResponse(SAMPLE_MACROTRENDS_INTERSTITIAL_HTML)
    mock_fetch_page_html.return_value = SAMPLE_MACROTRENDS_JSON_HTML

    s = _scrape_macrotrends_monthly(
        "https://www.macrotrends.net",
        "/1333/historical-gold-prices-100-year-chart",
        "gold_spot",
        "mean",
    )

    assert isinstance(s, pd.Series)
    assert len(s) > 0
    mock_fetch_page_html.assert_called_once()
    call_args, call_kwargs = mock_fetch_page_html.call_args
    assert call_args[0] == "https://www.macrotrends.net/1333/historical-gold-prices-100-year-chart"
    assert call_kwargs.get("require_selector") is False


@patch("trading_crab_lib.platform.ingestion.macro_monthly.fetch_page_html")
@patch("trading_crab_lib.platform.ingestion.macro_monthly.http_get")
def test_scrape_macrotrends_monthly_interstitial_and_browser_fallback_none_raises(mock_get, mock_fetch_page_html):
    from trading_crab_lib.platform.ingestion.macro_monthly import _scrape_macrotrends_monthly

    mock_get.return_value = _FakeResponse(SAMPLE_MACROTRENDS_INTERSTITIAL_HTML)
    mock_fetch_page_html.return_value = None

    with pytest.raises(ValueError):
        _scrape_macrotrends_monthly(
            "https://www.macrotrends.net",
            "/1333/historical-gold-prices-100-year-chart",
            "gold_spot",
            "mean",
        )


@patch("trading_crab_lib.platform.ingestion.macro_monthly.fetch_page_html")
@patch("trading_crab_lib.platform.ingestion.macro_monthly.http_get")
def test_scrape_macrotrends_monthly_handles_merged_header_and_month_column(mock_get, mock_fetch_page_html):
    """Regression for the 2026-08-05 live diagnostic, monthly analog: a
    'Month' date column that is not first, plus a squashed value header.
    Must not silently return an empty Series."""
    from trading_crab_lib.platform.ingestion.macro_monthly import _scrape_macrotrends_monthly

    mock_get.return_value = _FakeResponse(SAMPLE_MACROTRENDS_MERGED_HEADER_TABLE_HTML)

    s = _scrape_macrotrends_monthly(
        "https://www.macrotrends.net",
        "/1333/historical-gold-prices-100-year-chart",
        "gold_spot",
        "mean",
    )

    assert isinstance(s, pd.Series)
    assert len(s) > 0
    assert not s.isna().all()
    assert all(idx.is_month_end for idx in s.index)


@patch("trading_crab_lib.platform.ingestion.macro_monthly.time.sleep")
@patch("trading_crab_lib.platform.ingestion.macro_monthly.browser_session")
@patch("trading_crab_lib.platform.ingestion.macro_monthly.http_get")
def test_fetch_macrotrends_monthly_all_handles_series_failure(mock_get, mock_session, mock_sleep):
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
@patch("trading_crab_lib.platform.ingestion.macro_monthly.browser_session")
@patch("trading_crab_lib.platform.ingestion.macro_monthly.http_get")
@patch("trading_crab_lib.ingestion.multpl.requests.get")
@patch("trading_crab_lib.platform.ingestion.macro_monthly.Fred")
def test_fetch_macro_monthly_null_tolerant_partial_success(
    mock_fred_cls, mock_multpl_get, mock_http_get, mock_session, mock_sleep
):
    from trading_crab_lib.platform.ingestion.macro_monthly import fetch_macro_monthly

    mock_fred = MagicMock()
    mock_fred.get_series.return_value = _make_daily_fred_series(years=2)
    mock_fred_cls.return_value = mock_fred

    # multpl and macrotrends no longer share a transport: macrotrends fetches
    # through the browser-impersonating client (its Cloudflare bot check
    # rejects a plain-requests TLS fingerprint), while multpl still uses
    # requests. So they are patched separately now — macrotrends succeeds and
    # multpl fails outright, exercising partial-source-failure tolerance
    # (a failed source is simply absent from the merged frame, never an error).
    mock_http_get.return_value = _FakeResponse(SAMPLE_MACROTRENDS_JSON_HTML)
    mock_multpl_get.side_effect = OSError("multpl unreachable")

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


# ── TestInvariantSeriesIngestion — INV-01/D-12: M2SL + TOTALSL ──────────────
# M2SL and TOTALSL are natively monthly and ingested through the existing
# config-driven fred_monthly.series path. (The classifier-#2 invariant ratios
# they once fed were retired with the parked research code, 2026-10-05.)


class TestInvariantSeriesIngestion:
    def test_platform_config_maps_new_series_with_lags_in_the_publication_lags_table(self):
        """08.1: lags moved out of fred_monthly (no `shift` key anywhere) into the
        top-level publication_lags table, which carries the measured H.6 / G.19
        release delays."""
        from trading_crab_lib.platform.config import load_platform_config
        from trading_crab_lib.platform.ingestion.publication_lags import lag_table

        cfg = load_platform_config()
        series = cfg["fred_monthly"]["series"]
        assert series["M2SL"]["name"] == "fred_m2sl"
        assert series["TOTALSL"]["name"] == "fred_totalsl"
        assert [sid for sid, meta in series.items() if "shift" in meta] == []
        table = lag_table(cfg)
        assert table["fred_m2sl"] == 1
        assert table["fred_totalsl"] == 2

    @patch("trading_crab_lib.platform.ingestion.macro_monthly.Fred")
    def test_mocked_fetch_returns_both_renamed_columns(self, mock_fred_cls):
        from trading_crab_lib.platform.ingestion.macro_monthly import fetch_fred_monthly

        mock_fred = MagicMock()
        mock_fred.get_series.return_value = _make_daily_fred_series(years=2)
        mock_fred_cls.return_value = mock_fred

        cfg = {
            "fred_monthly": {
                "api_key": "fake_key_for_testing",
                "series": {
                    "M2SL": {"name": "fred_m2sl", "shift": False},
                    "TOTALSL": {"name": "fred_totalsl", "shift": False},
                },
            },
            "data": {"start_date": "2020-01-01", "end_date": "2022-01-01", "monthly_freq": "ME"},
        }
        df = fetch_fred_monthly(cfg)
        assert "fred_m2sl" in df.columns
        assert "fred_totalsl" in df.columns


# ── World Bank monthly commodity prices (08.4 UAT, 2026-10-09) ───────────────


def _pink_sheet_bytes() -> bytes:
    """A workbook laid out like the World Bank's: title rows, a header row, a units row, then
    one row per month labelled 1960M01, with "…" for a missing value."""
    import io

    rows = [
        ["World Bank Commodity Price Data (The Pink Sheet)", None, None],
        ["monthly prices in nominal US dollars, 1960 to present", None, None],
        ["Updated on October 02, 2026", None, None],
        [None, None, None],
        [None, "Crude oil, WTI", "Gold"],
        [None, "($/bbl)", "($/troy oz)"],
        ["1960M01", "…", 35],
        ["1960M02", "…", 35],
        ["1968M03", 3.1, 37.5],
        ["2026M09", 64.2, 4319],
    ]
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        pd.DataFrame(rows).to_excel(writer, sheet_name="Monthly Prices", header=False, index=False)
    return buffer.getvalue()


class _Response:
    def __init__(self, *, text: str = "", content: bytes = b""):
        self.text, self.content = text, content

    def raise_for_status(self) -> None:
        return None


_WB_CFG = {
    "worldbank_monthly": {
        "page_url": "https://www.worldbank.org/en/research/commodity-markets",
        "series": [{"name": "gold_wb", "column": "Gold"}],
    }
}
_WB_LINK = "https://thedocs.worldbank.org/en/doc/abc-0050012026/related/CMO-Historical-Data-Monthly.xlsx"


def test_parse_worldbank_monthly_reads_month_end_values_by_column_name():
    pytest.importorskip("openpyxl")
    parsed = macro_monthly.parse_worldbank_monthly(_pink_sheet_bytes(), ["Gold", "Crude oil, WTI"])

    gold = parsed["Gold"]
    assert list(gold.index) == list(pd.to_datetime(["1960-01-31", "1960-02-29", "1968-03-31", "2026-09-30"]))
    assert gold.tolist() == [35.0, 35.0, 37.5, 4319.0]
    assert parsed["Crude oil, WTI"].index.min() == pd.Timestamp("1968-03-31")  # "…" months dropped


def test_fetch_worldbank_monthly_follows_the_current_link_and_renames(monkeypatch):
    pytest.importorskip("openpyxl")
    calls = []

    def fake_get(url, **kwargs):
        calls.append(url)
        if url == _WB_CFG["worldbank_monthly"]["page_url"]:
            return _Response(text=f'<a href="{_WB_LINK}">Monthly prices</a>')
        return _Response(content=_pink_sheet_bytes())

    monkeypatch.setattr(macro_monthly, "http_get", fake_get)
    out = macro_monthly._fetch_worldbank_monthly(_WB_CFG)

    assert calls == [_WB_CFG["worldbank_monthly"]["page_url"], _WB_LINK]
    assert out["gold_wb"].name == "gold_wb" and out["gold_wb"].iloc[-1] == 4319.0


def test_fetch_worldbank_monthly_logs_and_returns_nothing_when_the_link_is_gone(monkeypatch, caplog):
    monkeypatch.setattr(macro_monthly, "http_get", lambda url, **kwargs: _Response(text="<html>moved</html>"))
    with caplog.at_level(logging.WARNING):
        assert macro_monthly._fetch_worldbank_monthly(_WB_CFG) == {}
    assert "CMO-Historical-Data-Monthly.xlsx" in caplog.text
