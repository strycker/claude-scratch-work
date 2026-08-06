"""HTTP-mocked tests for ingestion.macrotrends scraper.

All network access is mocked — no real HTTP calls are made.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from trading_crab_lib.ingestion.macrotrends import (
    _extract_json_data,
    _scrape_series,
    fetch_all,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

SAMPLE_JSON_DATA = [
    {"date": "2020-01-01", "close": "1550.50"},
    {"date": "2020-04-01", "close": "1700.00"},
    {"date": "2020-07-01", "close": "1950.25"},
    {"date": "2020-10-01", "close": "1880.00"},
    {"date": "2021-01-01", "close": "1830.00"},
    {"date": "2021-04-01", "close": "1770.00"},
]

SAMPLE_PAGE_HTML = f"""
<html><head><script>
var defined_data = {json.dumps(SAMPLE_JSON_DATA)};
</script></head><body></body></html>
"""

SAMPLE_TABLE_HTML = """
<html><body>
<table class="historical_data_table">
<tr><th>Date</th><th>Value</th></tr>
<tr><td>2020-03-31</td><td>1,550.50</td></tr>
<tr><td>2020-06-30</td><td>1,700.00</td></tr>
<tr><td>2020-09-30</td><td>1,950.25</td></tr>
</table>
</body></html>
"""

# A "Just a moment..."-style Cloudflare interstitial: no embedded JSON, no
# <table> — _html_yields_data must return False for this.
SAMPLE_INTERSTITIAL_HTML = """
<html><body>
<div class="cf-browser-verification">Just a moment...</div>
<div id="challenge-running"></div>
</body></html>
"""

# Regression fixture for the 2026-08-05 residential diagnostic against the
# live macrotrends page: a real headless browser reaches a table with NO
# distinguishing class (plain class="table", not "historical_data_table"),
# a date column literally titled "Month" (not "Date"/"Year"), and a value
# column with a squashed multi-line header ("Gold PricesMonthly Closing
# Price"). The value column is placed FIRST and the date column SECOND so
# this fixture actually exercises the "month" keyword match rather than the
# df.columns[0] fallback silently doing the right thing by accident.
SAMPLE_MERGED_HEADER_TABLE_HTML = """
<html><body>
<table class="table">
<thead><tr><th>Gold PricesMonthly Closing Price</th><th>Month</th></tr></thead>
<tbody>
<tr><td>1,560.00</td><td>2020-01-01</td></tr>
<tr><td>1,600.00</td><td>2020-02-01</td></tr>
<tr><td>1,650.00</td><td>2020-03-01</td></tr>
</tbody>
</table>
</body></html>
"""


class _FakeResponse:
    def __init__(self, text: str, status_code: int = 200):
        self.text = text
        self.content = text.encode("utf-8")
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise OSError(f"HTTP {self.status_code}")


# ── _extract_json_data tests ─────────────────────────────────────────────────

def test_extract_json_data_finds_embedded_array():
    result = _extract_json_data(SAMPLE_PAGE_HTML)
    assert result is not None
    assert len(result) == 6
    assert result[0]["date"] == "2020-01-01"


def test_extract_json_data_returns_none_for_no_data():
    result = _extract_json_data("<html><body>no data here</body></html>")
    assert result is None


# ── _scrape_series tests ─────────────────────────────────────────────────────

@patch("trading_crab_lib.ingestion.macrotrends.http_get")
def test_scrape_series_from_json_embed(mock_get):
    mock_get.return_value = _FakeResponse(SAMPLE_PAGE_HTML)

    s = _scrape_series("https://www.macrotrends.net", "/test", "gold_spot", "mean")

    assert isinstance(s, pd.Series)
    assert s.name == "gold_spot"
    assert len(s) > 0
    # Should be quarterly resampled
    assert hasattr(s.index, "freqstr") or len(s) <= 6


@patch("trading_crab_lib.ingestion.macrotrends.http_get")
def test_scrape_series_html_table_fallback(mock_get):
    """When no embedded JSON, falls back to pandas.read_html."""
    mock_get.return_value = _FakeResponse(SAMPLE_TABLE_HTML)

    s = _scrape_series("https://www.macrotrends.net", "/test", "gold_spot", "mean")

    assert isinstance(s, pd.Series)
    assert len(s) > 0


@patch("trading_crab_lib.ingestion.macrotrends.http_get")
def test_scrape_series_uses_impersonating_client(mock_get):
    mock_get.return_value = _FakeResponse(SAMPLE_PAGE_HTML)

    _scrape_series("https://www.macrotrends.net", "/1333/historical-gold-prices-100-year-chart", "gold_spot", "mean")

    mock_get.assert_called_once()
    called_url = mock_get.call_args.args[0] if mock_get.call_args.args else mock_get.call_args.kwargs.get("url")
    assert called_url == "https://www.macrotrends.net/1333/historical-gold-prices-100-year-chart"


# ── fetch_all tests ──────────────────────────────────────────────────────────

@patch("trading_crab_lib.ingestion.macrotrends.time.sleep")
@patch("trading_crab_lib.ingestion.macrotrends.browser_session")
@patch("trading_crab_lib.ingestion.macrotrends.http_get")
def test_fetch_all_default_series(mock_get, mock_browser_session, mock_sleep):
    mock_get.return_value = _FakeResponse(SAMPLE_PAGE_HTML)
    mock_browser_session.return_value = MagicMock()

    cfg = {}  # uses DEFAULT_SERIES
    df = fetch_all(cfg)

    assert isinstance(df, pd.DataFrame)
    assert "gold_spot" in df.columns
    assert "wti_crude" in df.columns
    assert len(df) > 0


@patch("trading_crab_lib.ingestion.macrotrends.time.sleep")
@patch("trading_crab_lib.ingestion.macrotrends.browser_session")
@patch("trading_crab_lib.ingestion.macrotrends.http_get")
def test_fetch_all_custom_config(mock_get, mock_browser_session, mock_sleep):
    mock_get.return_value = _FakeResponse(SAMPLE_PAGE_HTML)
    mock_browser_session.return_value = MagicMock()

    cfg = {
        "macrotrends": {
            "base_url": "https://www.macrotrends.net",
            "series": [
                {"name": "test_gold", "path": "/1333/gold", "resample": "mean"},
            ],
        }
    }
    df = fetch_all(cfg)

    assert isinstance(df, pd.DataFrame)
    assert "test_gold" in df.columns


@patch("trading_crab_lib.ingestion.macrotrends.time.sleep")
@patch("trading_crab_lib.ingestion.macrotrends.browser_session")
@patch("trading_crab_lib.ingestion.macrotrends.http_get")
def test_fetch_all_handles_failure_gracefully(mock_get, mock_browser_session, mock_sleep):
    mock_get.side_effect = Exception("Network error")
    mock_browser_session.return_value = MagicMock()

    cfg = {}
    df = fetch_all(cfg)

    assert isinstance(df, pd.DataFrame)
    assert df.empty


# ═══════════════════════════════════════════════════════════════════════════
# Browser fallback (playwright IS importable in this environment — every
# test that can reach _scrape_series patches fetch_page_html, or an
# unpatched fallback would launch a real browser).
# ═══════════════════════════════════════════════════════════════════════════


@patch("trading_crab_lib.ingestion.macrotrends.fetch_page_html")
@patch("trading_crab_lib.ingestion.macrotrends.http_get")
def test_scrape_series_json_body_never_calls_fetch_page_html(mock_get, mock_fetch_page_html):
    """HTTP is tried first: a body with embedded JSON must not touch the browser."""
    mock_get.return_value = _FakeResponse(SAMPLE_PAGE_HTML)

    s = _scrape_series("https://www.macrotrends.net", "/test", "gold_spot", "mean")

    assert isinstance(s, pd.Series)
    assert len(s) > 0
    mock_fetch_page_html.assert_not_called()


@patch("trading_crab_lib.ingestion.macrotrends.fetch_page_html")
@patch("trading_crab_lib.ingestion.macrotrends.http_get")
def test_scrape_series_table_body_never_calls_fetch_page_html(mock_get, mock_fetch_page_html):
    """HTTP is tried first: a body with a parseable table must not touch the browser."""
    mock_get.return_value = _FakeResponse(SAMPLE_TABLE_HTML)

    s = _scrape_series("https://www.macrotrends.net", "/test", "gold_spot", "mean")

    assert isinstance(s, pd.Series)
    assert len(s) > 0
    mock_fetch_page_html.assert_not_called()


@patch("trading_crab_lib.ingestion.macrotrends.fetch_page_html")
@patch("trading_crab_lib.ingestion.macrotrends.http_get")
def test_scrape_series_interstitial_falls_back_to_browser_render(mock_get, mock_fetch_page_html):
    mock_get.return_value = _FakeResponse(SAMPLE_INTERSTITIAL_HTML)
    mock_fetch_page_html.return_value = SAMPLE_PAGE_HTML

    s = _scrape_series(
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


@patch("trading_crab_lib.ingestion.macrotrends.fetch_page_html")
@patch("trading_crab_lib.ingestion.macrotrends.http_get")
def test_scrape_series_interstitial_and_browser_fallback_none_raises(mock_get, mock_fetch_page_html):
    mock_get.return_value = _FakeResponse(SAMPLE_INTERSTITIAL_HTML)
    mock_fetch_page_html.return_value = None

    with pytest.raises(ValueError):
        _scrape_series("https://www.macrotrends.net", "/test", "gold_spot", "mean")


@patch("trading_crab_lib.ingestion.macrotrends.fetch_page_html")
@patch("trading_crab_lib.ingestion.macrotrends.time.sleep")
@patch("trading_crab_lib.ingestion.macrotrends.browser_session")
@patch("trading_crab_lib.ingestion.macrotrends.http_get")
def test_fetch_all_degrades_to_empty_df_when_interstitial_and_browser_unavailable(
    mock_get, mock_browser_session, mock_sleep, mock_fetch_page_html
):
    """_scrape_series raising (interstitial + no browser recovery) must
    degrade fetch_all to an empty DataFrame with a WARNING, not propagate."""
    mock_get.return_value = _FakeResponse(SAMPLE_INTERSTITIAL_HTML)
    mock_browser_session.return_value = MagicMock()
    mock_fetch_page_html.return_value = None

    df = fetch_all({})

    assert isinstance(df, pd.DataFrame)
    assert df.empty


@patch("trading_crab_lib.ingestion.macrotrends.fetch_page_html")
@patch("trading_crab_lib.ingestion.macrotrends.http_get")
def test_scrape_series_quarter_end_indexed(mock_get, mock_fetch_page_html):
    """Frozen-resample regression guard (D-01) — quarterly stays QE."""
    mock_get.return_value = _FakeResponse(SAMPLE_PAGE_HTML)

    s = _scrape_series("https://www.macrotrends.net", "/test", "gold_spot", "mean")

    assert all(idx.is_quarter_end for idx in s.index)


@patch("trading_crab_lib.ingestion.macrotrends.fetch_page_html")
@patch("trading_crab_lib.ingestion.macrotrends.http_get")
def test_scrape_series_html_table_handles_merged_header_and_month_column(mock_get, mock_fetch_page_html):
    """Regression for the 2026-08-05 live diagnostic: the real macrotrends
    table has a 'Month' date column (not 'Date'/'Year', and not first) and a
    squashed value header ('Gold PricesMonthly Closing Price'). Must not
    silently return an empty Series."""
    mock_get.return_value = _FakeResponse(SAMPLE_MERGED_HEADER_TABLE_HTML)

    s = _scrape_series("https://www.macrotrends.net", "/test", "gold_spot", "mean")

    assert isinstance(s, pd.Series)
    assert len(s) > 0
    assert not s.isna().all()
    mock_fetch_page_html.assert_not_called()  # HTTP body already had a parseable table


# ── column detection on the REAL live table shape (2026-08-06) ──────────────


# What pandas.read_html actually produces for the live macrotrends page: BOTH
# columns carry the SAME squashed header (pandas only de-duplicates the second
# with ".1"), dates are strings, and values are $-prefixed with separators.
SAMPLE_LIVE_IDENTICAL_HEADERS_HTML = """
<html><body>
<table class="table">
<thead><tr><th>Gold PricesMonthly Closing Price</th><th>Gold PricesMonthly Closing Price</th></tr></thead>
<tbody>
<tr><td>2026-08-01</td><td>$4,245.30</td></tr>
<tr><td>2026-07-01</td><td>$4,041.70</td></tr>
<tr><td>2026-06-01</td><td>$4,006.70</td></tr>
</tbody>
</table>
</body></html>
"""


def test_detect_columns_when_both_headers_are_identical():
    """No header keyword can separate these — both contain 'price' AND 'month'.
    Detection must go by content."""
    from io import StringIO

    import pandas as pd

    from trading_crab_lib.ingestion.macrotrends import _detect_date_and_value_columns

    df = pd.read_html(StringIO(SAMPLE_LIVE_IDENTICAL_HEADERS_HTML))[0]
    date_col, value_col = _detect_date_and_value_columns(df, "gold_spot")

    assert date_col == df.columns[0]
    assert value_col == df.columns[1]
    assert date_col != value_col


def test_numeric_column_is_never_chosen_as_the_date_column():
    """read_html types a clean price column as float64, and pd.to_datetime
    reads floats as nanosecond epochs — scoring a perfect 1.0 as a date
    candidate and winning the tie against the real date column."""
    import pandas as pd

    from trading_crab_lib.ingestion.macrotrends import _detect_date_and_value_columns

    df = pd.DataFrame({"Gold PricesMonthly Closing Price": [1560.0, 1600.0], "Month": ["2020-01-01", "2020-02-01"]})
    date_col, value_col = _detect_date_and_value_columns(df, "gold_spot")

    assert date_col == "Month"
    assert value_col == "Gold PricesMonthly Closing Price"


def test_currency_symbols_are_stripped_before_numeric_conversion():
    """'$4,245.30' — the comma was handled, the dollar sign was not, so every
    row became NaN even once the right column was picked."""
    from trading_crab_lib.ingestion.macrotrends import _scrape_series_html_table

    s = _scrape_series_html_table(SAMPLE_LIVE_IDENTICAL_HEADERS_HTML, "gold_spot", "mean")

    assert not s.empty
    assert s.loc["2026-06-30"] == pytest.approx(4006.70)
    # Q3 averages July and August under the "mean" resample.
    assert s.loc["2026-09-30"] == pytest.approx((4041.70 + 4245.30) / 2)
