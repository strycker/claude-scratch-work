"""HTTP-mocked tests for the platform daily universe price ingestion module
(trading_crab_lib.platform.ingestion.prices_daily).

All network access is mocked — no real HTTP/yfinance calls are made.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from trading_crab_lib.platform.ingestion import prices_daily
from trading_crab_lib.platform.ingestion.prices_daily import (
    _batch_stooq_daily,
    _batch_yfinance_daily,
    fetch_universe_prices,
    to_monthly_spine,
    universe_fetch_tickers,
)


def _make_yf_frame(tickers: list[str]) -> pd.DataFrame:
    """A minimal yf.download()-shaped MultiIndex frame (level-0=metric,
    level-1=ticker) with non-NaN Close values for the given tickers."""
    idx = pd.date_range("2020-01-02", periods=3, freq="B")
    columns = pd.MultiIndex.from_product([["Close"], tickers])
    data = np.tile(np.arange(100.0, 103.0), (len(tickers), 1)).T
    return pd.DataFrame(data, index=idx, columns=columns)


def _make_universe_cfg():
    return {
        "universe": {
            "satellites": ["QQQ", "IWM"],
            "holdings": ["IAU", "SLV", "GDX", "DBA", "FZFXX", "SPAXX"],
            "watchlist": ["UNG", "UEC"],
            "no_price_ingest": ["FZFXX", "SPAXX"],
        },
        "data": {
            "start_date": "2020-01-01",
            "end_date": "2024-01-01",
            "monthly_freq": "ME",
        },
    }


# ── universe_fetch_tickers ──────────────────────────────────────────────────


def test_universe_fetch_tickers_excludes_money_market_funds():
    cfg = _make_universe_cfg()
    tickers = universe_fetch_tickers(cfg)
    assert "FZFXX" not in tickers
    assert "SPAXX" not in tickers


def test_universe_fetch_tickers_includes_holdings_satellites_watchlist():
    cfg = _make_universe_cfg()
    tickers = universe_fetch_tickers(cfg)
    # Holdings (D-10) always included.
    for t in ("IAU", "SLV", "GDX", "DBA"):
        assert t in tickers
    # Satellites (D-09).
    for t in ("QQQ", "IWM"):
        assert t in tickers
    # Watchlist (D-11).
    for t in ("UNG", "UEC"):
        assert t in tickers


# ── fetch_universe_prices: NULL-tolerant merge (DATA-05) ───────────────────


@patch("trading_crab_lib.platform.ingestion.prices_daily._ssl_bypass_curl_session")
@patch("trading_crab_lib.platform.ingestion.prices_daily._batch_yfinance_daily")
def test_short_history_ticker_becomes_nan_padded_not_dropped(mock_batch, mock_session):
    cfg = _make_universe_cfg()

    full_idx = pd.date_range("2020-01-02", periods=250, freq="B")
    short_idx = full_idx[-30:]  # short-history ticker only has recent dates

    mock_batch.return_value = {
        "QQQ": pd.Series(np.arange(300.0, 300.0 + len(full_idx)), index=full_idx, name="QQQ"),
        "IAU": pd.Series(np.arange(30.0, 30.0 + len(short_idx)), index=short_idx, name="IAU"),
    }
    mock_session.return_value = MagicMock()

    daily, monthly = fetch_universe_prices(cfg)

    assert isinstance(daily, pd.DataFrame)
    assert isinstance(monthly, pd.DataFrame)
    # No dropped rows: full row count preserved (outer join, not inner).
    assert len(daily) == len(full_idx)
    # Short-history ticker is NaN-padded for pre-inception dates, not absent.
    assert "IAU" in daily.columns
    pre_inception = daily.loc[full_idx[0], "IAU"]
    assert pd.isna(pre_inception)
    # Post-inception dates are populated.
    assert daily.loc[short_idx[-1], "IAU"] == 59.0


@patch("trading_crab_lib.platform.ingestion.prices_daily._ssl_bypass_curl_session")
@patch("trading_crab_lib.platform.ingestion.prices_daily._batch_yfinance_daily")
def test_fetch_universe_prices_no_crash_on_short_history(mock_batch, mock_session):
    """Merging variable-length histories never raises."""
    cfg = _make_universe_cfg()
    idx_a = pd.date_range("2020-01-02", periods=100, freq="B")
    idx_b = pd.date_range("2023-06-01", periods=10, freq="B")

    mock_batch.return_value = {
        "QQQ": pd.Series(np.arange(100.0), index=idx_a, name="QQQ"),
        "UEC": pd.Series(np.arange(10.0), index=idx_b, name="UEC"),
    }
    mock_session.return_value = MagicMock()

    daily, monthly = fetch_universe_prices(cfg)
    assert not daily.empty
    assert not monthly.empty


@patch("trading_crab_lib.platform.ingestion.prices_daily._batch_stooq_daily")
@patch("trading_crab_lib.platform.ingestion.prices_daily._ssl_bypass_curl_session")
@patch("trading_crab_lib.platform.ingestion.prices_daily._batch_yfinance_daily")
def test_fetch_universe_prices_all_fail_returns_empty(mock_batch, mock_session, mock_stooq):
    cfg = _make_universe_cfg()
    mock_batch.return_value = {}       # yfinance yields nothing
    mock_stooq.return_value = {}       # Stooq fallback also yields nothing
    mock_session.return_value = MagicMock()

    daily, monthly = fetch_universe_prices(cfg)
    assert daily.empty
    assert monthly.empty
    mock_stooq.assert_called_once()  # fallback WAS attempted after yfinance failed


@patch("trading_crab_lib.platform.ingestion.prices_daily._batch_stooq_daily")
@patch("trading_crab_lib.platform.ingestion.prices_daily._ssl_bypass_curl_session")
@patch("trading_crab_lib.platform.ingestion.prices_daily._batch_yfinance_daily")
def test_fetch_universe_prices_falls_back_to_stooq_when_yfinance_empty(mock_batch, mock_session, mock_stooq):
    cfg = _make_universe_cfg()
    mock_batch.return_value = {}  # yfinance rate-limited
    mock_session.return_value = MagicMock()
    idx = pd.date_range("2020-01-02", periods=5, freq="D")
    mock_stooq.return_value = {"QQQ": pd.Series(range(5), index=idx, name="QQQ", dtype=float)}

    daily, monthly = fetch_universe_prices(cfg)
    assert not daily.empty
    assert "QQQ" in daily.columns
    mock_stooq.assert_called_once()


@patch("trading_crab_lib.platform.ingestion.prices_daily.time.sleep")
@patch("trading_crab_lib.platform.ingestion.prices_daily.browser_session")
@patch("trading_crab_lib.platform.ingestion.prices_daily.http_get")
def test_batch_stooq_daily_parses_csv_close(mock_http_get, mock_browser_session, mock_sleep):
    csv = "Date,Open,High,Low,Close,Volume\n2020-01-02,100,101,99,100.5,1000\n2020-01-03,100.5,102,100,101.2,1200\n"
    mock_http_get.return_value = MagicMock(text=csv)
    mock_browser_session.return_value = MagicMock()

    out = _batch_stooq_daily(["SPY"], "2020-01-01", "2020-01-10")
    assert "SPY" in out
    assert list(out["SPY"].values) == [100.5, 101.2]
    assert str(out["SPY"].index[0].date()) == "2020-01-02"


@patch("trading_crab_lib.platform.ingestion.prices_daily.time.sleep")
@patch("trading_crab_lib.platform.ingestion.prices_daily.playwright_available")
@patch("trading_crab_lib.platform.ingestion.prices_daily.browser_session")
@patch("trading_crab_lib.platform.ingestion.prices_daily.http_get")
def test_batch_stooq_daily_skips_js_challenge_page(mock_http_get, mock_browser_session, mock_playwright_available, mock_sleep):
    # Stooq's anti-bot page (served to some datacenter/VPN IPs) — must NOT be parsed as data.
    # playwright_available patched False: this test is specifically about the HTTP-path guard,
    # not the browser fallback (that is covered separately below) — no real browser launch here.
    challenge = "<!DOCTYPE html><html><head></head><body><noscript>This site requires JavaScript</noscript></body></html>"
    mock_http_get.return_value = MagicMock(text=challenge)
    mock_browser_session.return_value = MagicMock()
    mock_playwright_available.return_value = False

    out = _batch_stooq_daily(["SPY"], "2020-01-01", "2020-01-10")
    assert out == {}


@patch("trading_crab_lib.platform.ingestion.prices_daily.time.sleep")
@patch("trading_crab_lib.platform.ingestion.prices_daily.browser_session")
@patch("trading_crab_lib.platform.ingestion.prices_daily.http_get")
@patch("requests.get")
def test_batch_stooq_daily_uses_impersonating_client_not_plain_requests(
    mock_requests_get, mock_http_get, mock_browser_session, mock_sleep
):
    csv = "Date,Open,High,Low,Close,Volume\n2020-01-02,100,101,99,100.5,1000\n"
    mock_http_get.return_value = MagicMock(text=csv)
    mock_browser_session.return_value = MagicMock()

    out = _batch_stooq_daily(["SPY"], "2020-01-01", "2020-01-10")

    assert "SPY" in out
    mock_http_get.assert_called()
    mock_requests_get.assert_not_called()


@patch("trading_crab_lib.platform.ingestion.prices_daily.time.sleep")
@patch("trading_crab_lib.platform.ingestion.prices_daily.browser_session")
@patch("trading_crab_lib.platform.ingestion.prices_daily.http_get")
def test_batch_stooq_daily_rate_limits_between_tickers(mock_http_get, mock_browser_session, mock_sleep):
    csv = "Date,Open,High,Low,Close,Volume\n2020-01-02,100,101,99,100.5,1000\n"
    mock_http_get.return_value = MagicMock(text=csv)
    mock_browser_session.return_value = MagicMock()

    out = _batch_stooq_daily(["SPY", "QQQ", "IWM"], "2020-01-01", "2020-01-10")

    assert len(out) == 3
    assert mock_sleep.call_count >= 2


# ── _parse_stooq_csv: single shared CSV-header guard ────────────────────────


def test_parse_stooq_csv_returns_series_for_valid_csv():
    csv = "Date,Open,High,Low,Close,Volume\n2020-01-02,100,101,99,100.5,1000\n2020-01-03,100.5,102,100,101.2,1200\n"
    s = prices_daily._parse_stooq_csv("SPY", csv)
    assert s is not None
    assert list(s.values) == [100.5, 101.2]
    assert s.name == "SPY"


def test_parse_stooq_csv_returns_none_for_challenge_page():
    challenge = "<!DOCTYPE html><html><head></head><body><noscript>This site requires JavaScript</noscript></body></html>"
    assert prices_daily._parse_stooq_csv("SPY", challenge) is None


# ── _batch_stooq_daily: headless-browser fallback wiring ───────────────────


@patch("trading_crab_lib.platform.ingestion.prices_daily.time.sleep")
@patch("trading_crab_lib.platform.ingestion.prices_daily.fetch_stooq_csvs")
@patch("trading_crab_lib.platform.ingestion.prices_daily.browser_session")
@patch("trading_crab_lib.platform.ingestion.prices_daily.http_get")
def test_batch_stooq_daily_never_calls_browser_fallback_when_http_recovers_a_ticker(
    mock_http_get, mock_browser_session, mock_fetch_browser, mock_sleep
):
    csv = "Date,Open,High,Low,Close,Volume\n2020-01-02,100,101,99,100.5,1000\n"
    mock_http_get.return_value = MagicMock(text=csv)
    mock_browser_session.return_value = MagicMock()

    out = _batch_stooq_daily(["SPY"], "2020-01-01", "2020-01-10")

    assert "SPY" in out
    mock_fetch_browser.assert_not_called()


@patch("trading_crab_lib.platform.ingestion.prices_daily.time.sleep")
@patch("trading_crab_lib.platform.ingestion.prices_daily.playwright_available")
@patch("trading_crab_lib.platform.ingestion.prices_daily.fetch_stooq_csvs")
@patch("trading_crab_lib.platform.ingestion.prices_daily.browser_session")
@patch("trading_crab_lib.platform.ingestion.prices_daily.http_get")
def test_batch_stooq_daily_falls_back_to_browser_when_http_recovers_nothing(
    mock_http_get, mock_browser_session, mock_fetch_browser, mock_playwright_available, mock_sleep
):
    challenge = "<!DOCTYPE html><html><body>JS challenge</body></html>"
    mock_http_get.return_value = MagicMock(text=challenge)
    mock_browser_session.return_value = MagicMock()
    mock_playwright_available.return_value = True

    browser_csv = "Date,Open,High,Low,Close,Volume\n2020-01-02,100,101,99,100.5,1000\n"
    mock_fetch_browser.return_value = {"SPY": browser_csv}

    out = _batch_stooq_daily(["SPY"], "2020-01-01", "2020-01-10")

    assert "SPY" in out
    mock_fetch_browser.assert_called_once()
    called_urls = mock_fetch_browser.call_args.args[0]
    assert "SPY" in called_urls
    assert called_urls["SPY"].startswith("https://stooq.com/q/d/l/?s=spy.us")


@patch("trading_crab_lib.platform.ingestion.prices_daily.time.sleep")
@patch("trading_crab_lib.platform.ingestion.prices_daily.playwright_available")
@patch("trading_crab_lib.platform.ingestion.prices_daily.fetch_stooq_csvs")
@patch("trading_crab_lib.platform.ingestion.prices_daily.browser_session")
@patch("trading_crab_lib.platform.ingestion.prices_daily.http_get")
def test_batch_stooq_daily_skips_browser_fallback_when_playwright_unavailable(
    mock_http_get, mock_browser_session, mock_fetch_browser, mock_playwright_available, mock_sleep
):
    challenge = "<!DOCTYPE html><html><body>JS challenge</body></html>"
    mock_http_get.return_value = MagicMock(text=challenge)
    mock_browser_session.return_value = MagicMock()
    mock_playwright_available.return_value = False

    out = _batch_stooq_daily(["SPY"], "2020-01-01", "2020-01-10")

    assert out == {}
    mock_fetch_browser.assert_not_called()


@patch("trading_crab_lib.platform.ingestion.prices_daily.time.sleep")
@patch("trading_crab_lib.platform.ingestion.prices_daily.playwright_available")
@patch("trading_crab_lib.platform.ingestion.prices_daily.fetch_stooq_csvs")
@patch("trading_crab_lib.platform.ingestion.prices_daily.browser_session")
@patch("trading_crab_lib.platform.ingestion.prices_daily.http_get")
def test_batch_stooq_daily_browser_challenge_page_rejected_by_shared_guard(
    mock_http_get, mock_browser_session, mock_fetch_browser, mock_playwright_available, mock_sleep
):
    challenge = "<!DOCTYPE html><html><body>JS challenge</body></html>"
    mock_http_get.return_value = MagicMock(text=challenge)
    mock_browser_session.return_value = MagicMock()
    mock_playwright_available.return_value = True
    mock_fetch_browser.return_value = {"SPY": challenge}  # browser ALSO got a challenge page

    out = _batch_stooq_daily(["SPY"], "2020-01-01", "2020-01-10")

    assert out == {}


@patch("trading_crab_lib.platform.ingestion.prices_daily.time.sleep")
@patch("trading_crab_lib.platform.ingestion.prices_daily.playwright_available")
@patch("trading_crab_lib.platform.ingestion.prices_daily.fetch_stooq_csvs")
@patch("trading_crab_lib.platform.ingestion.prices_daily.browser_session")
@patch("trading_crab_lib.platform.ingestion.prices_daily.http_get")
def test_batch_stooq_daily_browser_fallback_raising_does_not_propagate(
    mock_http_get, mock_browser_session, mock_fetch_browser, mock_playwright_available, mock_sleep
):
    challenge = "<!DOCTYPE html><html><body>JS challenge</body></html>"
    mock_http_get.return_value = MagicMock(text=challenge)
    mock_browser_session.return_value = MagicMock()
    mock_playwright_available.return_value = True
    mock_fetch_browser.side_effect = RuntimeError("browser launch failed")

    out = _batch_stooq_daily(["SPY"], "2020-01-01", "2020-01-10")

    assert out == {}


@patch("trading_crab_lib.platform.ingestion.prices_daily.time.sleep")
@patch("trading_crab_lib.platform.ingestion.prices_daily.playwright_available")
@patch("trading_crab_lib.platform.ingestion.prices_daily.fetch_stooq_csvs")
@patch("trading_crab_lib.platform.ingestion.prices_daily.browser_session")
@patch("trading_crab_lib.platform.ingestion.prices_daily.http_get")
@patch("trading_crab_lib.platform.ingestion.prices_daily._parse_stooq_csv")
def test_batch_stooq_daily_both_paths_route_through_shared_parse_helper(
    mock_parse, mock_http_get, mock_browser_session, mock_fetch_browser, mock_playwright_available, mock_sleep
):
    # Force the HTTP path's parse to reject, so the browser fallback fires too —
    # patching this ONE helper changes the outcome of BOTH paths, proving the
    # guard was factored out rather than duplicated.
    mock_parse.return_value = None
    mock_http_get.return_value = MagicMock(text="http-response-text")
    mock_browser_session.return_value = MagicMock()
    mock_playwright_available.return_value = True
    mock_fetch_browser.return_value = {"SPY": "browser-response-text"}

    _batch_stooq_daily(["SPY"], "2020-01-01", "2020-01-10")

    assert mock_parse.call_count == 2
    texts_seen = [c.args[1] for c in mock_parse.call_args_list]
    assert "http-response-text" in texts_seen
    assert "browser-response-text" in texts_seen


def test_fetch_universe_prices_no_tickers_returns_empty():
    cfg = {
        "universe": {
            "satellites": [],
            "holdings": [],
            "watchlist": [],
            "no_price_ingest": [],
        },
        "data": {"start_date": "2020-01-01", "end_date": "2024-01-01", "monthly_freq": "ME"},
    }
    daily, monthly = fetch_universe_prices(cfg)
    assert daily.empty
    assert monthly.empty


# ── _batch_yfinance_daily: chunking + retry/backoff ─────────────────────────


@patch("trading_crab_lib.platform.ingestion.prices_daily.time.sleep")
@patch("trading_crab_lib.platform.ingestion.prices_daily._YFINANCE_AVAILABLE", True)
@patch("trading_crab_lib.platform.ingestion.prices_daily.yf")
def test_yfinance_daily_fetches_in_small_chunks(mock_yf, mock_sleep):
    tickers = [f"T{i}" for i in range(12)]

    def _download(**kwargs):
        chunk = kwargs["tickers"]
        assert len(chunk) <= prices_daily._YF_CHUNK_SIZE
        return _make_yf_frame(chunk)

    mock_yf.download.side_effect = _download

    out = _batch_yfinance_daily(tickers, "2020-01-01", "2020-02-01")

    assert mock_yf.download.call_count > 1
    assert len(out) == 12


@patch("trading_crab_lib.platform.ingestion.prices_daily.time.sleep")
@patch("trading_crab_lib.platform.ingestion.prices_daily._YFINANCE_AVAILABLE", True)
@patch("trading_crab_lib.platform.ingestion.prices_daily.yf")
def test_yfinance_daily_keeps_partial_results_when_a_chunk_exhausts_retries(mock_yf, mock_sleep):
    tickers = [f"T{i}" for i in range(6)]  # chunk 1: T0-T4 (5 tickers), chunk 2: T5

    def _download(**kwargs):
        chunk = kwargs["tickers"]
        if chunk == tickers[:5]:
            return _make_yf_frame(chunk)
        raise RuntimeError("429 Too Many Requests")

    mock_yf.download.side_effect = _download

    out = _batch_yfinance_daily(tickers, "2020-01-01", "2020-02-01")

    assert out != {}
    assert set(out.keys()) == set(tickers[:5])


@patch("trading_crab_lib.platform.ingestion.prices_daily.time.sleep")
@patch("trading_crab_lib.platform.ingestion.prices_daily._YFINANCE_AVAILABLE", True)
@patch("trading_crab_lib.platform.ingestion.prices_daily.yf")
def test_yfinance_daily_retries_rate_limited_chunk_with_growing_backoff(mock_yf, mock_sleep):
    tickers = ["T0", "T1", "T2"]  # single chunk (<= _YF_CHUNK_SIZE)
    call_count = {"n": 0}

    def _download(**kwargs):
        call_count["n"] += 1
        if call_count["n"] < 3:
            raise RuntimeError("Rate limit exceeded (429)")
        return _make_yf_frame(kwargs["tickers"])

    mock_yf.download.side_effect = _download

    out = _batch_yfinance_daily(tickers, "2020-01-01", "2020-02-01")

    assert mock_yf.download.call_count == 3
    assert len(out) == 3
    sleep_intervals = [c.args[0] for c in mock_sleep.call_args_list]
    assert len(sleep_intervals) >= 2
    assert sleep_intervals[0] < sleep_intervals[1]


# ── to_monthly_spine (D-05: daily -> monthly, daily preserved) ─────────────


def test_to_monthly_spine_yields_month_end_frequency():
    idx = pd.date_range("2020-01-02", periods=90, freq="B")  # ~3 months of business days
    daily_df = pd.DataFrame({"QQQ": np.arange(90.0)}, index=idx)

    monthly = to_monthly_spine(daily_df, monthly_freq="ME")

    assert isinstance(monthly, pd.DataFrame)
    # Every index entry lands on a month-end date.
    assert all(monthly.index == monthly.index.to_series().apply(lambda d: d + pd.offsets.MonthEnd(0)))
    # Row count collapses from 90 daily rows to one row per calendar month spanned.
    assert len(monthly) < len(daily_df)
    assert len(monthly) == idx.to_series().dt.to_period("M").nunique()


def test_to_monthly_spine_empty_input_returns_empty():
    empty = pd.DataFrame()
    result = to_monthly_spine(empty)
    assert result.empty
