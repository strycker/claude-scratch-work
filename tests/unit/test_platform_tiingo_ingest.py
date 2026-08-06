"""Unit tests for the live Tiingo daily-price adapter
(trading_crab_lib.platform.ingestion.tiingo).

All network access is mocked — no real HTTP calls, and no API key is needed
to run these.
"""
from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from trading_crab_lib.platform.ingestion import tiingo

_KEY = "sk-tiingo-supersecret-0123456789"


def _rows(*, adj: bool = True) -> list[dict]:
    """Two EOD rows, with or without the adjClose field."""
    out = []
    for day, close, adj_close in (("2024-01-02", 100.0, 90.0), ("2024-01-03", 101.0, 91.0)):
        row = {"date": f"{day}T00:00:00.000Z", "close": close}
        if adj:
            row["adjClose"] = adj_close
        out.append(row)
    return out


def _response(payload: object, status: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = payload
    return resp


# ── adjClose preference ─────────────────────────────────────────────────────


@patch("trading_crab_lib.platform.ingestion.tiingo.time.sleep")
@patch("trading_crab_lib.platform.ingestion.tiingo.plain_session")
@patch("trading_crab_lib.platform.ingestion.tiingo.http_get")
def test_prefers_adjclose_over_close(mock_get, mock_session, mock_sleep):
    """Mixing adjusted and unadjusted prices would silently corrupt returns at
    every dividend and split, and nothing downstream would flag it."""
    mock_get.return_value = _response(_rows(adj=True))

    out = tiingo.fetch_daily_prices(["SPY"], "2024-01-01", "2024-01-05", api_key=_KEY)

    assert list(out["SPY"].to_numpy()) == [90.0, 91.0]  # adjClose, not close


@patch("trading_crab_lib.platform.ingestion.tiingo.time.sleep")
@patch("trading_crab_lib.platform.ingestion.tiingo.plain_session")
@patch("trading_crab_lib.platform.ingestion.tiingo.http_get")
def test_falls_back_to_close_when_adjclose_absent(mock_get, mock_session, mock_sleep):
    mock_get.return_value = _response(_rows(adj=False))

    out = tiingo.fetch_daily_prices(["SPY"], "2024-01-01", "2024-01-05", api_key=_KEY)

    assert list(out["SPY"].to_numpy()) == [100.0, 101.0]


@patch("trading_crab_lib.platform.ingestion.tiingo.time.sleep")
@patch("trading_crab_lib.platform.ingestion.tiingo.plain_session")
@patch("trading_crab_lib.platform.ingestion.tiingo.http_get")
def test_series_is_daily_and_tz_naive(mock_get, mock_session, mock_sleep):
    mock_get.return_value = _response(_rows())

    out = tiingo.fetch_daily_prices(["SPY"], "2024-01-01", "2024-01-05", api_key=_KEY)

    series = out["SPY"]
    assert isinstance(series.index, pd.DatetimeIndex)
    assert series.index.tz is None
    assert series.name == "SPY"
    assert list(series.index) == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]


# ── credential handling ─────────────────────────────────────────────────────


@patch("trading_crab_lib.platform.ingestion.tiingo.time.sleep")
@patch("trading_crab_lib.platform.ingestion.tiingo.plain_session")
@patch("trading_crab_lib.platform.ingestion.tiingo.http_get")
def test_api_key_travels_in_header_never_in_url(mock_get, mock_session, mock_sleep):
    mock_get.return_value = _response(_rows())

    tiingo.fetch_daily_prices(["SPY"], "2024-01-01", "2024-01-05", api_key=_KEY)

    args, kwargs = mock_get.call_args
    url = args[0] if args else kwargs["url"]
    assert _KEY not in url
    assert "token" not in url.lower()
    assert kwargs["headers"]["Authorization"] == f"Token {_KEY}"


@patch("trading_crab_lib.platform.ingestion.tiingo.time.sleep")
@patch("trading_crab_lib.platform.ingestion.tiingo.plain_session")
@patch("trading_crab_lib.platform.ingestion.tiingo.http_get")
def test_api_key_never_reaches_the_logs_via_an_exception(mock_get, mock_session, mock_sleep, caplog):
    """The realistic leak path: the transport echoes request detail — including
    the Authorization header — into an exception message, which is then logged.
    Raise exactly that and assert no record carries the key."""
    mock_get.side_effect = OSError(f"connection failed for GET /prices with Authorization: Token {_KEY}")

    with caplog.at_level(logging.DEBUG):
        out = tiingo.fetch_daily_prices(["SPY"], "2024-01-01", "2024-01-05", api_key=_KEY)

    assert out == {}
    assert caplog.records, "expected the failure to be logged at all"
    for record in caplog.records:
        assert _KEY not in record.getMessage()
    assert any("REDACTED" in rec.getMessage() for rec in caplog.records)


def test_missing_api_key_returns_empty_without_raising_or_requesting(monkeypatch, caplog):
    monkeypatch.delenv("TIINGO_API_KEY", raising=False)

    with patch("trading_crab_lib.platform.ingestion.tiingo.http_get") as mock_get, \
         patch("trading_crab_lib.platform.ingestion.tiingo.plain_session") as mock_session, \
         caplog.at_level(logging.INFO):
        out = tiingo.fetch_daily_prices(["SPY"], "2024-01-01", "2024-01-05")

    assert out == {}
    mock_get.assert_not_called()
    mock_session.assert_not_called()  # no session built either
    # INFO, not WARNING — an unconfigured optional source is not a fault.
    assert any(rec.levelno == logging.INFO and "no API key" in rec.getMessage() for rec in caplog.records)


def test_resolve_api_key_precedence_and_blank_handling(monkeypatch):
    monkeypatch.setenv("TIINGO_API_KEY", "from-env")
    assert tiingo.resolve_api_key(None, "explicit") == "explicit"
    assert tiingo.resolve_api_key(None, None) == "from-env"
    assert tiingo.resolve_api_key(None, "   ") == "from-env"  # blank arg ignored

    monkeypatch.delenv("TIINGO_API_KEY", raising=False)
    assert tiingo.resolve_api_key({"tiingo": {"api_key": "from-cfg"}}) == "from-cfg"
    assert tiingo.resolve_api_key({"tiingo": {"api_key": "  "}}) is None
    assert tiingo.resolve_api_key({}) is None


# ── degradation and rate limiting ───────────────────────────────────────────


@patch("trading_crab_lib.platform.ingestion.tiingo.time.sleep")
@patch("trading_crab_lib.platform.ingestion.tiingo.plain_session")
@patch("trading_crab_lib.platform.ingestion.tiingo.http_get")
def test_one_failing_ticker_does_not_abort_the_batch(mock_get, mock_session, mock_sleep):
    def _side_effect(url, **_kwargs):
        if "/QQQ/" in url:
            raise OSError("boom")
        return _response(_rows())

    mock_get.side_effect = _side_effect

    out = tiingo.fetch_daily_prices(["SPY", "QQQ", "IWM"], "2024-01-01", "2024-01-05", api_key=_KEY)

    assert set(out) == {"SPY", "IWM"}


@patch("trading_crab_lib.platform.ingestion.tiingo.time.sleep")
@patch("trading_crab_lib.platform.ingestion.tiingo.plain_session")
@patch("trading_crab_lib.platform.ingestion.tiingo.http_get")
def test_429_is_retried_with_bounded_backoff_then_succeeds(mock_get, mock_session, mock_sleep):
    mock_get.side_effect = [
        _response(None, status=429),
        _response(None, status=429),
        _response(_rows()),
    ]

    out = tiingo.fetch_daily_prices(["SPY"], "2024-01-01", "2024-01-05", api_key=_KEY)

    assert "SPY" in out
    assert mock_get.call_count == 3
    # Backoff grew between retries rather than spinning at a fixed interval.
    backoffs = [c.args[0] for c in mock_sleep.call_args_list if c.args and c.args[0] >= 2.0]
    assert backoffs == sorted(backoffs) and len(backoffs) >= 2


@patch("trading_crab_lib.platform.ingestion.tiingo.time.sleep")
@patch("trading_crab_lib.platform.ingestion.tiingo.plain_session")
@patch("trading_crab_lib.platform.ingestion.tiingo.http_get")
def test_429_retry_budget_is_bounded(mock_get, mock_session, mock_sleep):
    mock_get.return_value = _response(None, status=429)

    out = tiingo.fetch_daily_prices(["SPY"], "2024-01-01", "2024-01-05", api_key=_KEY)

    assert out == {}
    assert mock_get.call_count == tiingo._MAX_RETRIES  # bounded, not unbounded


@patch("trading_crab_lib.platform.ingestion.tiingo.time.sleep")
@patch("trading_crab_lib.platform.ingestion.tiingo.plain_session")
@patch("trading_crab_lib.platform.ingestion.tiingo.http_get")
@pytest.mark.parametrize("status", [401, 403, 404, 500])
def test_non_200_non_429_is_not_retried(mock_get, mock_session, mock_sleep, status):
    mock_get.return_value = _response(None, status=status)

    out = tiingo.fetch_daily_prices(["SPY"], "2024-01-01", "2024-01-05", api_key=_KEY)

    assert out == {}
    assert mock_get.call_count == 1  # an auth failure must not burn the retry budget


@patch("trading_crab_lib.platform.ingestion.tiingo.time.sleep")
@patch("trading_crab_lib.platform.ingestion.tiingo.plain_session")
@patch("trading_crab_lib.platform.ingestion.tiingo.http_get")
def test_unparseable_payload_degrades_to_skip(mock_get, mock_session, mock_sleep):
    mock_get.return_value = _response({"detail": "not a list"})

    out = tiingo.fetch_daily_prices(["SPY"], "2024-01-01", "2024-01-05", api_key=_KEY)

    assert out == {}


# ── transport choice ────────────────────────────────────────────────────────


@patch("trading_crab_lib.platform.ingestion.tiingo.time.sleep")
@patch("trading_crab_lib.platform.ingestion.tiingo.http_get")
@patch("trading_crab_lib.platform.ingestion.tiingo.plain_session")
def test_uses_plain_session_not_the_impersonating_client(mock_plain, mock_get, mock_sleep):
    """Tiingo is a keyed REST API with no bot check, so impersonation defeats
    nothing and only adds failure modes — curl_cffi ships its own CA store and
    its own transport, and was observed being reset against this very endpoint
    while plain requests got HTTP 200."""
    mock_get.return_value = _response(_rows())

    tiingo.fetch_daily_prices(["SPY"], "2024-01-01", "2024-01-05", api_key=_KEY)

    mock_plain.assert_called_once()
    assert not hasattr(tiingo, "browser_session"), "the impersonating client must not be imported here"
