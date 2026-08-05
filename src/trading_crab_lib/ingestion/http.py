"""
Shared browser-impersonating HTTP client for public data sources that
bot-block plain ``requests`` by TLS fingerprint.

Stooq serves a JavaScript browser-verification page (not CSV) to plain
``requests`` clients; macrotrends.net fronts its pages with a Cloudflare
"Just a moment..." interstitial. Both are TLS-fingerprint bot checks, not
credential problems — ``curl_cffi``'s ``impersonate="chrome"`` reproduces a
real Chrome TLS ClientHello and JA3 fingerprint, which plain ``requests``
cannot do.

This module leaves TLS certificate verification ENABLED by default
(``verify=True``). It is deliberately NOT the SSL-bypass session-factory
helper in ``ingestion/assets.py``, which sets ``verify=False``
unconditionally (tracked tech debt, pitfall P22). This module must never
import from that assets ingestion module and must never widen that
SSL-bypass blast radius.

Usage:
    from trading_crab_lib.ingestion.http import http_get, browser_session

    session = browser_session()
    resp = http_get("https://stooq.com/q/d/l/?s=spy.us&i=d", session=session)
"""

from __future__ import annotations

import logging
import os
from typing import Any

try:
    from curl_cffi import requests as _curl_requests  # type: ignore[import]
    _CURL_CFFI_AVAILABLE = True
except ImportError:
    _curl_requests = None  # type: ignore[assignment]
    _CURL_CFFI_AVAILABLE = False

try:
    import requests as _requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _requests = None  # type: ignore[assignment]
    _REQUESTS_AVAILABLE = False

log = logging.getLogger(__name__)

BROWSER_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

DEFAULT_IMPERSONATE = "chrome"
DEFAULT_TIMEOUT = 30.0

# Operator escape hatch for TLS-intercepting networks. curl_cffi uses its own
# bundled CA store and ignores the macOS Keychain, the system trust store, and
# the SSL_CERT_FILE / REQUESTS_CA_BUNDLE environment variables — so on a
# network that re-signs TLS (corporate proxy, filtering appliance, some VPNs)
# verification fails with "self signed certificate in certificate chain" even
# though browsers are fine. Point this at a PEM bundle containing both the
# public roots and the intercepting CA (``scripts/fix_local_ca.sh`` builds one)
# to keep verification ON rather than disabling it.
_CA_BUNDLE_ENV = "TC_CA_BUNDLE"


def ca_bundle_override() -> str | None:
    """Return the operator-supplied CA bundle path, or None when unset.

    Empty/whitespace values are treated as unset so an exported-but-blank
    variable cannot silently break verification.
    """
    value = (os.environ.get(_CA_BUNDLE_ENV) or "").strip()
    return value or None

# Transport exception classes call sites should catch. Both curl_cffi's and
# requests' RequestException subclass OSError, so OSError alone is a safe
# catch-all even when only one client (or neither) is installed.
_error_types: list[type[BaseException]] = [OSError]
if _CURL_CFFI_AVAILABLE and _curl_requests is not None:
    _error_types.append(_curl_requests.exceptions.RequestException)
if _REQUESTS_AVAILABLE and _requests is not None:
    _error_types.append(_requests.exceptions.RequestException)
HTTP_ERRORS: tuple[type[BaseException], ...] = tuple(dict.fromkeys(_error_types))


def impersonation_available() -> bool:
    """Return True when curl_cffi (browser TLS impersonation) is importable."""
    return _CURL_CFFI_AVAILABLE


def is_impersonating_session(session: Any) -> bool:
    """Return True when *session* is a curl_cffi session presenting a browser
    TLS fingerprint.

    Used to decide whether :data:`BROWSER_HEADERS` should be applied. It must
    NOT be applied to an impersonating session: curl_cffi already supplies a
    complete, internally-consistent browser header set (correct ``sec-ch-ua``
    and ``sec-fetch-*`` client hints, header ordering, and a User-Agent that
    matches the TLS fingerprint it presents). Overriding that with a
    hand-written subset yields a request whose fingerprint and User-Agent
    disagree and which is missing client hints every real browser sends —
    itself a strong bot signal, and worse than sending nothing.

    Identity is checked against the curl_cffi Session class rather than
    duck-typed, so a plain ``requests.Session`` can never be mistaken for one.
    """
    if not _CURL_CFFI_AVAILABLE or _curl_requests is None:
        return False
    return isinstance(session, _curl_requests.Session)


def browser_session(*, impersonate: str = DEFAULT_IMPERSONATE, verify: bool = True) -> Any | None:
    """Construct a browser-impersonating HTTP session.

    Prefers a ``curl_cffi.requests.Session`` with TLS fingerprint
    impersonation (defeats TLS-fingerprint bot checks like Stooq's JS
    challenge and macrotrends' Cloudflare interstitial). Falls back to a
    plain ``requests.Session`` carrying browser-like headers when curl_cffi
    is not installed. Returns None only when neither HTTP client is
    importable.

    ``verify`` defaults to True (TLS verification enabled) and is only ever
    set from this explicit caller argument — no code path here widens it on
    its own initiative. This is intentionally NOT the SSL-bypass factory in
    the assets ingestion module.
    """
    # A caller that explicitly asked for verify=False keeps that; otherwise an
    # operator-supplied bundle replaces the boolean, so verification stays ON
    # even on a TLS-intercepting network.
    effective_verify: bool | str = verify
    if verify:
        bundle = ca_bundle_override()
        if bundle:
            effective_verify = bundle
            log.debug("Using CA bundle from %s: %s", _CA_BUNDLE_ENV, bundle)

    if _CURL_CFFI_AVAILABLE and _curl_requests is not None:
        try:
            return _curl_requests.Session(impersonate=impersonate, verify=effective_verify)
        except TypeError:
            log.debug(
                "curl_cffi Session does not support impersonate= in this version — "
                "falling back to a session without impersonation"
            )
            return _curl_requests.Session(verify=effective_verify)

    if _REQUESTS_AVAILABLE and _requests is not None:
        log.debug("curl_cffi not importable — falling back to plain requests.Session (no TLS impersonation)")
        session = _requests.Session()
        session.headers.update(BROWSER_HEADERS)
        session.verify = effective_verify
        return session

    log.warning("Neither curl_cffi nor requests is importable — cannot construct an HTTP session.")
    return None


def http_get(
    url: str,
    *,
    session: Any = None,
    headers: dict[str, str] | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    impersonate: str = DEFAULT_IMPERSONATE,
    verify: bool = True,
) -> Any:
    """Issue a GET request through a browser-impersonating session.

    Reuses *session* when supplied (preferred — avoids re-negotiating TLS per
    request); otherwise builds a one-shot session via :func:`browser_session`.
    Raises OSError if no HTTP client is available at all.

    :data:`BROWSER_HEADERS` is applied ONLY on the plain-``requests`` fallback
    path, where there is no impersonation for it to contradict. On the
    curl_cffi path only caller-supplied *headers* are passed through, and the
    kwarg is omitted entirely when the caller supplied none — see
    :func:`is_impersonating_session` for why clobbering curl_cffi's own header
    set defeats the impersonation it was chosen for.
    """
    own_session = session
    if own_session is None:
        own_session = browser_session(impersonate=impersonate, verify=verify)
    if own_session is None:
        raise OSError(
            "No HTTP client available (neither curl_cffi nor requests is importable) — "
            "cannot issue request to " + url
        )

    if is_impersonating_session(own_session):
        # Let curl_cffi's matched browser header set stand. Passing headers=None
        # would still override it in some versions, so omit the kwarg outright.
        if headers:
            return own_session.get(url, headers=dict(headers), timeout=timeout)
        return own_session.get(url, timeout=timeout)

    merged_headers = dict(BROWSER_HEADERS)
    if headers:
        merged_headers.update(headers)

    return own_session.get(url, headers=merged_headers, timeout=timeout)
