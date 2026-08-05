"""
Headless-Chromium fallback for fetching Stooq CSVs when TLS impersonation fails.

Verdict this module exists because of: Stooq serves a 796-byte JavaScript
browser-verification page (not CSV) to the caller. TLS-fingerprint
impersonation via curl_cffi (``ingestion/http.py``) was tested exhaustively on
a residential connection — ``impersonate="chrome"`` and ``impersonate="safari"``,
each with and without a hand-written header override, each with and without a
same-session quote-page warm-up — and every single combination returned the
identical challenge page and set zero cookies. The challenge requires
executing JavaScript, not presenting a convincing TLS handshake; no
fingerprint can satisfy it. This module executes that JavaScript in a real
headless Chromium instead of trying to impersonate it at the transport layer.

This module is a SIBLING of ``ingestion/http.py``, not a replacement. The
plain (cheap, TLS-impersonating) HTTP path in ``http.py`` stays the first
attempt everywhere it is used; this module is only reached as a fallback when
that path recovers nothing.

Playwright is an OPTIONAL dependency (the ``[browser]`` packaging extra).
When it is not installed, :func:`fetch_stooq_csvs` degrades to an empty
result with a logged warning — it never raises.

Usage:
    from trading_crab_lib.ingestion.browser import fetch_stooq_csvs, playwright_available

    if playwright_available():
        csvs = fetch_stooq_csvs({"SPY": "https://stooq.com/q/d/l/?s=spy.us&i=d"})
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

log = logging.getLogger(__name__)

_PLAYWRIGHT_AVAILABLE = False
try:
    from playwright.sync_api import sync_playwright
    _PLAYWRIGHT_AVAILABLE = True
except ImportError:
    sync_playwright = None  # type: ignore[assignment]

# A normal Stooq HTML page. Navigating here first lets the challenge's own
# JavaScript execute and commit its verification cookie into the browser
# context before any CSV endpoint is requested.
_STOOQ_WARMUP_URL = "https://stooq.com/q/?s=spy.us"

# Matches prices_daily._STOOQ_RATE_LIMIT_SECONDS — rapid-fire per-ticker
# requests is itself a bot signal.
_BROWSER_RATE_LIMIT_SECONDS = 1.5

# Operator escape hatch: an ambient Chromium build can mismatch the installed
# playwright package's expected build (this container is exactly such a
# case). When set, it is passed as chromium.launch(executable_path=...).
_CHROMIUM_PATH_ENV = "TC_CHROMIUM_PATH"

_NAV_TIMEOUT_MS = 30_000
_REQUEST_TIMEOUT_MS = 30_000
# Short post-navigation wait letting the challenge JS run and commit its
# cookie before the first CSV request is issued.
_CHALLENGE_SETTLE_MS = 2_000


def playwright_available() -> bool:
    """Return True when the ``playwright`` package is importable."""
    return _PLAYWRIGHT_AVAILABLE


def fetch_stooq_csvs(
    urls: dict[str, str],
    *,
    warmup_url: str = _STOOQ_WARMUP_URL,
    rate_limit_seconds: float = _BROWSER_RATE_LIMIT_SECONDS,
) -> dict[str, str]:
    """Fetch each Stooq CSV URL through one headless-Chromium session.

    Launches exactly one Chromium instance, navigates once to *warmup_url*
    so the JS challenge executes and sets its verification cookie, then
    issues one ``context.request.get`` per URL in *urls* — the browser
    CONTEXT's own request API, never a harvested-cookie replay through a
    different HTTP client. A cookie replayed outside the browser that
    produced it may be bound to that browser's TLS fingerprint and identity,
    which would reintroduce exactly the fingerprint/identity mismatch this
    module exists to remove.

    Returns plain ``{ticker: csv_text}`` for whatever was recovered before
    any failure. This function performs NO parsing, validation, or
    interpretation of the CSV text — that is the caller's job via the single
    shared CSV-header guard. It never raises: a missing playwright install,
    a failed launch, a failed navigation, or a failed per-ticker fetch all
    degrade to returning whatever was already accumulated, with a WARNING
    logged for each cause.
    """
    if not urls:
        return {}

    if not _PLAYWRIGHT_AVAILABLE:
        log.warning(
            "playwright is not installed — Stooq headless-browser fallback skipped. "
            "Install with: pip install 'playwright>=1.40'  then: playwright install chromium"
        )
        return {}

    results: dict[str, str] = {}
    try:
        with sync_playwright() as pw:
            launch_kwargs: dict[str, Any] = {"headless": True}
            chromium_path = os.environ.get(_CHROMIUM_PATH_ENV)
            if chromium_path:
                launch_kwargs["executable_path"] = chromium_path

            browser = pw.chromium.launch(**launch_kwargs)
            try:
                context = browser.new_context()
                page = context.new_page()
                page.goto(warmup_url, timeout=_NAV_TIMEOUT_MS)
                page.wait_for_timeout(_CHALLENGE_SETTLE_MS)

                for i, (ticker, url) in enumerate(urls.items()):
                    if i > 0:
                        time.sleep(rate_limit_seconds)
                    try:
                        resp = context.request.get(url, timeout=_REQUEST_TIMEOUT_MS)
                        if not resp.ok:
                            log.warning("Stooq browser fetch non-OK response for %s", ticker)
                            continue
                        results[ticker] = resp.text()
                    except Exception as exc:  # noqa: BLE001 — one bad ticker must not abort the batch
                        log.warning("Stooq browser fetch failed for %s: %s", ticker, exc)
                        continue
            finally:
                # Must survive a mid-batch failure — this is the only reason
                # the try/finally exists; do not collapse it.
                browser.close()
    except Exception as exc:  # noqa: BLE001 — launch/navigation degradation, never raise to caller
        log.warning("Stooq headless-browser fallback failed: %s", exc)

    return results
