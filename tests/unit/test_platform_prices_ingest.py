"""HTTP-mocked tests for the platform daily universe price ingestion module
(trading_crab_lib.platform.ingestion.prices_daily).

All network access is mocked — no real HTTP/yfinance calls are made.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from trading_crab_lib.platform.ingestion.prices_daily import (
    fetch_universe_prices,
    to_monthly_spine,
    universe_fetch_tickers,
)


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


@patch("trading_crab_lib.platform.ingestion.prices_daily._ssl_bypass_curl_session")
@patch("trading_crab_lib.platform.ingestion.prices_daily._batch_yfinance_daily")
def test_fetch_universe_prices_all_fail_returns_empty(mock_batch, mock_session):
    cfg = _make_universe_cfg()
    mock_batch.return_value = {}
    mock_session.return_value = MagicMock()

    daily, monthly = fetch_universe_prices(cfg)
    assert daily.empty
    assert monthly.empty


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
