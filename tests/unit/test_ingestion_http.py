"""Unit tests for trading_crab_lib.ingestion.http — the shared
browser-impersonating HTTP client used by Stooq and macrotrends.

All network access is mocked — no real HTTP calls are made.
"""
from __future__ import annotations

import inspect
import logging

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


_NOT_PASSED = object()


class _RecordingCurlSession:
    """Stands in for a real curl_cffi Session.

    Records whether ``headers`` was passed at all — distinguishing "omitted"
    from "passed as None" matters, because only omitting it leaves curl_cffi's
    own impersonated header set untouched.
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls: list[tuple] = []

    def get(self, url, headers=_NOT_PASSED, timeout=None):
        self.calls.append((url, headers, timeout))
        return "response"


class _ClassBasedCurlModule:
    """curl_cffi stand-in whose ``Session`` is a real class, so
    ``isinstance(session, module.Session)`` works as it does for the genuine
    module."""

    Session = _RecordingCurlSession


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


# ── header handling: impersonating path vs plain-requests fallback ──────────


def _impersonating(monkeypatch):
    """Point http at the class-based curl_cffi stand-in and return a session."""
    monkeypatch.setattr(http, "_curl_requests", _ClassBasedCurlModule())
    monkeypatch.setattr(http, "_CURL_CFFI_AVAILABLE", True)
    return _RecordingCurlSession()


def test_is_impersonating_session_true_for_curl_session(monkeypatch):
    session = _impersonating(monkeypatch)
    assert http.is_impersonating_session(session) is True


def test_is_impersonating_session_false_for_plain_requests_session(monkeypatch):
    _impersonating(monkeypatch)
    assert http.is_impersonating_session(requests.Session()) is False


def test_http_get_does_not_clobber_impersonated_headers(monkeypatch):
    """BROWSER_HEADERS must NOT be applied to a curl_cffi session — doing so
    replaces its matched Chrome header set with a hand-written subset, leaving
    the User-Agent disagreeing with the TLS fingerprint."""
    session = _impersonating(monkeypatch)

    http.http_get("https://stooq.com/q/d/l/?s=spy.us&i=d", session=session)

    assert len(session.calls) == 1
    _url, headers, _timeout = session.calls[0]
    # headers was omitted entirely, not passed as None or as BROWSER_HEADERS.
    assert headers is _NOT_PASSED


def test_http_get_passes_only_caller_headers_on_impersonating_path(monkeypatch):
    session = _impersonating(monkeypatch)

    http.http_get("https://example.com/x", session=session, headers={"X-Test": "1"})

    _url, headers, _timeout = session.calls[0]
    assert headers == {"X-Test": "1"}
    # No BROWSER_HEADERS keys leaked in alongside the caller's.
    assert "User-Agent" not in headers


def test_http_get_applies_browser_headers_on_plain_requests_fallback(monkeypatch):
    """Regression guard (passes before and after): the fallback path has no
    impersonation to contradict, so it still needs browser-like headers."""
    _impersonating(monkeypatch)
    session = _RecordingSession()

    http.http_get("https://example.com/x", session=session)

    _url, headers, _timeout = session.calls[0]
    assert headers["User-Agent"] == http.BROWSER_HEADERS["User-Agent"]


# ── TC_CA_BUNDLE override (TLS-intercepting networks) ───────────────────────


def test_ca_bundle_override_returns_none_when_unset(monkeypatch):
    monkeypatch.delenv("TC_CA_BUNDLE", raising=False)
    assert http.ca_bundle_override() is None


def test_ca_bundle_override_treats_blank_as_unset(monkeypatch):
    """An exported-but-empty variable must not be handed to curl_cffi as a
    CA path — that would break verification rather than fix it."""
    monkeypatch.setenv("TC_CA_BUNDLE", "   ")
    assert http.ca_bundle_override() is None


def test_browser_session_verifies_against_ca_bundle_when_set(monkeypatch, tmp_path):
    bundle = tmp_path / "combined.pem"
    bundle.write_text("-----BEGIN CERTIFICATE-----\n")
    monkeypatch.setenv("TC_CA_BUNDLE", str(bundle))
    fake_curl = _FakeCurlModule()
    monkeypatch.setattr(http, "_curl_requests", fake_curl)
    monkeypatch.setattr(http, "_CURL_CFFI_AVAILABLE", True)

    http.browser_session()

    assert fake_curl.calls[0]["verify"] == str(bundle)


def test_ca_bundle_override_ignores_a_path_that_does_not_exist(monkeypatch, caplog):
    """The live run set a RELATIVE path (a .env cannot expand ~ or $HOME) and it
    was handed straight to curl_cffi, failing deep in the TLS stack instead of
    at its cause."""
    monkeypatch.setenv("TC_CA_BUNDLE", "/nonexistent/combined.pem")
    with caplog.at_level(logging.WARNING):
        assert http.ca_bundle_override() is None
    assert any("no such file exists" in rec.message for rec in caplog.records)


def test_ca_bundle_override_expands_user_home(monkeypatch, tmp_path):
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    bundle = fake_home / ".trading-crab-ca.pem"
    bundle.write_text("-----BEGIN CERTIFICATE-----\n")
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("TC_CA_BUNDLE", "~/.trading-crab-ca.pem")

    assert http.ca_bundle_override() == str(bundle)


def test_browser_session_explicit_verify_false_is_not_overridden(monkeypatch, tmp_path):
    """An explicit opt-out stays an opt-out — the bundle upgrades the default,
    it does not re-enable verification a caller deliberately disabled."""
    bundle = tmp_path / "combined.pem"
    bundle.write_text("-----BEGIN CERTIFICATE-----\n")
    monkeypatch.setenv("TC_CA_BUNDLE", str(bundle))
    fake_curl = _FakeCurlModule()
    monkeypatch.setattr(http, "_curl_requests", fake_curl)
    monkeypatch.setattr(http, "_CURL_CFFI_AVAILABLE", True)

    http.browser_session(verify=False)

    assert fake_curl.calls[0]["verify"] is False
