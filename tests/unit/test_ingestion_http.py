"""Unit tests for trading_crab_lib.ingestion.http — the shared
browser-impersonating HTTP client used by Stooq and macrotrends.

All network access is mocked — no real HTTP calls are made.
"""
from __future__ import annotations

import inspect

import requests

from trading_crab_lib.ingestion import http


class _FakeCurlSession:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeCurlModule:
    """Records every Session(**kwargs) construction call."""

    def __init__(self):
        self.calls: list[dict] = []

    def Session(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeCurlSession(**kwargs)


class _RecordingSession:
    def __init__(self):
        self.calls: list[tuple] = []

    def get(self, url, headers=None, timeout=None):
        self.calls.append((url, headers, timeout))
        return "response"


# ── browser_session ─────────────────────────────────────────────────────────


def test_browser_session_requests_chrome_impersonation_with_tls_verification(monkeypatch):
    fake_curl = _FakeCurlModule()
    monkeypatch.setattr(http, "_curl_requests", fake_curl)
    monkeypatch.setattr(http, "_CURL_CFFI_AVAILABLE", True)

    http.browser_session()

    assert len(fake_curl.calls) == 1
    recorded_kwargs = fake_curl.calls[0]
    assert recorded_kwargs["impersonate"] == "chrome"
    assert recorded_kwargs["verify"]  # truthy — TLS verification requested


def test_browser_session_falls_back_to_requests_without_curl_cffi(monkeypatch):
    monkeypatch.setattr(http, "_CURL_CFFI_AVAILABLE", False)
    monkeypatch.setattr(http, "_REQUESTS_AVAILABLE", True)
    monkeypatch.setattr(http, "_requests", requests)

    session = http.browser_session()

    assert isinstance(session, requests.Session)
    assert session.headers.get("User-Agent") == http.BROWSER_HEADERS["User-Agent"]


def test_browser_session_returns_none_without_any_client(monkeypatch):
    monkeypatch.setattr(http, "_CURL_CFFI_AVAILABLE", False)
    monkeypatch.setattr(http, "_REQUESTS_AVAILABLE", False)

    assert http.browser_session() is None


# ── http_get ─────────────────────────────────────────────────────────────────


def test_http_get_reuses_supplied_session(monkeypatch):
    session = _RecordingSession()
    browser_session_calls: list[dict] = []

    def _fake_browser_session(**kwargs):
        browser_session_calls.append(kwargs)
        return None

    monkeypatch.setattr(http, "browser_session", _fake_browser_session)

    result = http.http_get("https://example.com/data", session=session)

    assert result == "response"
    assert len(session.calls) == 1
    url, headers, timeout = session.calls[0]
    assert url == "https://example.com/data"
    assert timeout == http.DEFAULT_TIMEOUT
    assert headers is not None
    # No new session was constructed — the supplied one was reused.
    assert browser_session_calls == []


# ── module hygiene ───────────────────────────────────────────────────────────


def test_module_does_not_import_the_ssl_bypass_factory():
    src = inspect.getsource(http)
    assert "import ingestion.assets" not in src
    assert "from trading_crab_lib.ingestion.assets" not in src
    assert "from trading_crab_lib.ingestion import assets" not in src
    assert "_ssl_bypass_curl_session" not in src
