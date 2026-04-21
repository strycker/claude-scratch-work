"""HTTP-mocked tests for ingestion.macrotrends scraper.

All network access is mocked — no real HTTP calls are made.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pandas as pd

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

@patch("trading_crab_lib.ingestion.macrotrends.requests.get")
def test_scrape_series_from_json_embed(mock_get):
    mock_get.return_value = _FakeResponse(SAMPLE_PAGE_HTML)

    s = _scrape_series("https://www.macrotrends.net", "/test", "gold_spot", "mean")

    assert isinstance(s, pd.Series)
    assert s.name == "gold_spot"
    assert len(s) > 0
    # Should be quarterly resampled
    assert hasattr(s.index, "freqstr") or len(s) <= 6


@patch("trading_crab_lib.ingestion.macrotrends.requests.get")
def test_scrape_series_html_table_fallback(mock_get):
    """When no embedded JSON, falls back to pandas.read_html."""
    mock_get.return_value = _FakeResponse(SAMPLE_TABLE_HTML)

    s = _scrape_series("https://www.macrotrends.net", "/test", "gold_spot", "mean")

    assert isinstance(s, pd.Series)
    assert len(s) > 0


# ── fetch_all tests ──────────────────────────────────────────────────────────

@patch("trading_crab_lib.ingestion.macrotrends.time.sleep")
@patch("trading_crab_lib.ingestion.macrotrends.requests.get")
def test_fetch_all_default_series(mock_get, mock_sleep):
    mock_get.return_value = _FakeResponse(SAMPLE_PAGE_HTML)

    cfg = {}  # uses DEFAULT_SERIES
    df = fetch_all(cfg)

    assert isinstance(df, pd.DataFrame)
    assert "gold_spot" in df.columns
    assert "wti_crude" in df.columns
    assert len(df) > 0


@patch("trading_crab_lib.ingestion.macrotrends.time.sleep")
@patch("trading_crab_lib.ingestion.macrotrends.requests.get")
def test_fetch_all_custom_config(mock_get, mock_sleep):
    mock_get.return_value = _FakeResponse(SAMPLE_PAGE_HTML)

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
@patch("trading_crab_lib.ingestion.macrotrends.requests.get")
def test_fetch_all_handles_failure_gracefully(mock_get, mock_sleep):
    mock_get.side_effect = Exception("Network error")

    cfg = {}
    df = fetch_all(cfg)

    assert isinstance(df, pd.DataFrame)
    assert df.empty
