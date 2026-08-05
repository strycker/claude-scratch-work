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

Second verdict, from a five-variant diagnostic on the same residential
connection: inside a real browser, Stooq's CSV endpoint distinguishes a
NAVIGATION from an API request. Playwright's ``context.request.get`` is
answered with the body ``"Access denied"`` (HTTP 200), and stealth launch
args plus a ``navigator.webdriver`` override do not change that. A genuine
``page.goto`` to the same URL is served normally — as a file attachment,
which Playwright surfaces by raising ``Error: Download is starting``. That
raise is the SUCCESS path here, not a failure. Meanwhile an ordinary Stooq
HTML page loads fine (263 KB) in the same headless browser, so this is
neither an IP block nor automation detection. Hence: navigate and take the
download, rather than ask the request API.

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

import contextlib
import logging
import os
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_PLAYWRIGHT_AVAILABLE = False
try:
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright
    _PLAYWRIGHT_AVAILABLE = True
except ImportError:
    sync_playwright = None  # type: ignore[assignment]
    PlaywrightError = Exception  # type: ignore[assignment,misc]

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
_DOWNLOAD_TIMEOUT_MS = 30_000

# Cheap reduction of the automation signature on the navigation path we now
# depend on. NOTE: these did NOT lift the "Access denied" the request API
# received — switching to a real navigation did. They are kept because they
# are harmless and the navigation path is the one that matters now.
_STEALTH_LAUNCH_ARGS = ["--disable-blink-features=AutomationControlled"]
_STEALTH_INIT_SCRIPT = "Object.defineProperty(navigator, 'webdriver', {get: () => undefined});"
# Short post-navigation wait letting the challenge JS run and commit its
# cookie before the first CSV request is issued.
_CHALLENGE_SETTLE_MS = 2_000


def playwright_available() -> bool:
    """Return True when the ``playwright`` package is importable."""
    return _PLAYWRIGHT_AVAILABLE


def _fetch_one_via_download(page: Any, url: str) -> str | None:
    """Navigate to *url* and read the file Stooq serves back as a download.

    The CSV endpoint responds with a file attachment, so ``page.goto`` raises
    ``Error: Download is starting`` instead of completing a navigation. That
    raise is EXPECTED and is swallowed deliberately — the download itself is
    the payload. Returns None if no download materialises in time.
    """
    with page.expect_download(timeout=_DOWNLOAD_TIMEOUT_MS) as download_info:
        with contextlib.suppress(PlaywrightError):
            page.goto(url, timeout=_NAV_TIMEOUT_MS)

    download = download_info.value
    path = download.path()
    if path is None:
        return None
    try:
        return Path(path).read_text()
    finally:
        # The temp file is ours once read; failing to clean it up would leak
        # one file per ticker per run.
        with contextlib.suppress(OSError):
            download.delete()


def _fetch_one_via_page_fetch(page: Any, url: str) -> str | None:
    """Fallback: same-origin ``fetch()`` executed inside the page.

    Carries the page's cookies, origin and referer, which a detached request
    client does not. Only tried when the download path yields nothing.
    """
    text = page.evaluate("async (u) => (await fetch(u)).text()", url)
    return text if text else None


def fetch_stooq_csvs(
    urls: dict[str, str],
    *,
    warmup_url: str = _STOOQ_WARMUP_URL,
    rate_limit_seconds: float = _BROWSER_RATE_LIMIT_SECONDS,
) -> dict[str, str]:
    """Fetch each Stooq CSV URL through one headless-Chromium session.

    Launches exactly one Chromium instance and navigates once to *warmup_url*
    so the JS challenge executes and sets its verification cookie. Each URL is
    then fetched by NAVIGATING to it and taking the file Stooq serves back as
    a download (see :func:`_fetch_one_via_download`), falling back to an
    in-page ``fetch()`` when no download materialises.

    Everything stays inside the browser that solved the challenge — no cookie
    is harvested and replayed through a different HTTP client, because a
    cookie may be bound to the identity that earned it, which would
    reintroduce exactly the fingerprint/identity mismatch this module exists
    to remove. Note that Playwright's own ``context.request.get`` is NOT such
    a client but is still refused by this endpoint with "Access denied"; only
    a real navigation is served.

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
            launch_kwargs: dict[str, Any] = {"headless": True, "args": list(_STEALTH_LAUNCH_ARGS)}
            chromium_path = os.environ.get(_CHROMIUM_PATH_ENV)
            if chromium_path:
                launch_kwargs["executable_path"] = chromium_path

            browser = pw.chromium.launch(**launch_kwargs)
            try:
                # accept_downloads is required — the CSV endpoint answers a
                # navigation with a file attachment, not a page.
                context = browser.new_context(accept_downloads=True)
                context.add_init_script(_STEALTH_INIT_SCRIPT)
                page = context.new_page()
                page.goto(warmup_url, timeout=_NAV_TIMEOUT_MS)
                page.wait_for_timeout(_CHALLENGE_SETTLE_MS)

                for i, (ticker, url) in enumerate(urls.items()):
                    if i > 0:
                        time.sleep(rate_limit_seconds)
                    try:
                        text = _fetch_one_via_download(page, url)
                        if not text:
                            log.debug("No download for %s — trying in-page fetch", ticker)
                            text = _fetch_one_via_page_fetch(page, url)
                        if not text:
                            log.warning("Stooq browser fetch yielded nothing for %s", ticker)
                            continue
                        results[ticker] = text
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
