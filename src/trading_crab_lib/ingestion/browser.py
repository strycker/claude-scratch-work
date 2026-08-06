"""
Headless-Chromium browser-rendering capability for sources gated behind
JavaScript execution.

This module renders pages in a real (headless) browser for ANY source whose
content requires JavaScript to execute before it appears — it is not shaped
around any one site. It stays a SIBLING of ``ingestion/http.py``: the cheap,
TLS-impersonating HTTP path in ``http.py`` is always tried FIRST by every
caller, and this module is only reached as a fallback when that path
recovers nothing.

Two verdicts recorded here, both earned on a residential connection (this
module cannot be exercised against the live internet from inside the
container that develops it — see the per-caller ``<human-check>`` steps):

1. Stooq serves a 796-byte JavaScript browser-verification page (not CSV) to
   every TLS-impersonating client tried (``curl_cffi`` with
   ``impersonate="chrome"`` and ``impersonate="safari"``, each with and
   without a hand-written header override, each with and without a
   same-session quote-page warm-up). Every combination returned the
   identical challenge page and set zero cookies. The challenge requires
   executing JavaScript, not presenting a convincing TLS handshake; no
   fingerprint can satisfy it.

2. A five-variant diagnostic on the same connection found that, inside a
   real browser, Stooq's CSV endpoint distinguishes a NAVIGATION from an API
   request. Playwright's ``context.request.get`` is answered with the body
   ``"Access denied"`` (HTTP 200), and stealth launch args plus a
   ``navigator.webdriver`` override do not change that. A genuine
   ``page.goto`` to the same URL is served normally — as a file attachment,
   which Playwright surfaces by raising ``Error: Download is starting``.
   That raise is the SUCCESS path, not a failure. Meanwhile an ordinary
   Stooq HTML page loads fine (263 KB) in the same headless browser, so this
   is neither an IP block nor automation detection. Hence: navigate and take
   the download for that endpoint, rather than ask the request API.
   ``fetch_stooq_csvs`` is kept as a named wrapper because this
   download-interception flow is endpoint-specific, not general.

Two engines are available (Playwright primary, Selenium as a fallback — see
the "why two engines" note further down); both are OPTIONAL dependencies
(the ``[browser]`` packaging extra). When neither is installed,
:func:`fetch_page_html` and :func:`fetch_urls_as_text` degrade to an empty
result with a logged warning — they never raise.

Usage:
    from trading_crab_lib.ingestion.browser import fetch_page_html, playwright_available

    if playwright_available():
        html = fetch_page_html("https://example.com/some-js-rendered-page")

    from trading_crab_lib.ingestion.browser import fetch_stooq_csvs
    csvs = fetch_stooq_csvs({"SPY": "https://stooq.com/q/d/l/?s=spy.us&i=d"})

Why a second engine (Selenium) exists — read before assuming it is a way
around a block Playwright cannot pass:

- Selenium is NOT bot-evasion redundancy. Both engines drive a real Chromium
  and present near-identical automation signals to a remote page.
- ChromeDriver historically leaks ``$cdc_`` properties into the page that
  fingerprinting scripts specifically look for, which makes Selenium if
  anything MORE detectable than Playwright, not less.
- Its value is ENGINE redundancy — a way to still render a page when
  Playwright is not installed, or when the installed ``playwright`` package
  and the ambient Chromium build do not match (a failure mode this repo has
  already hit, which is why ``TC_CHROMIUM_PATH`` exists).
- It is not a second attempt at a block Playwright could not pass.
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

_SELENIUM_AVAILABLE = False
try:
    from selenium import webdriver
    from selenium.common.exceptions import TimeoutException as SeleniumTimeout
    from selenium.webdriver.chrome.options import Options as ChromeOptions
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.support.ui import WebDriverWait
    _SELENIUM_AVAILABLE = True
except ImportError:
    # Every name bound here so tests can patch them even when selenium is not
    # installed (it is not, in this repo's dev/CI containers).
    webdriver = None  # type: ignore[assignment]
    SeleniumTimeout = Exception  # type: ignore[assignment,misc]
    ChromeOptions = None  # type: ignore[assignment,misc]
    By = None  # type: ignore[assignment,misc]
    EC = None  # type: ignore[assignment,misc]
    WebDriverWait = None  # type: ignore[assignment,misc]

# A normal Stooq HTML page. Navigating here first (fetch_urls_as_text's
# warmup_url) lets the challenge's own JavaScript execute and commit its
# verification cookie into the browser context before any CSV endpoint is
# requested. Stooq-specific; fetch_page_html takes no warm-up.
_STOOQ_WARMUP_URL = "https://stooq.com/q/?s=spy.us"

# Matches prices_daily._STOOQ_RATE_LIMIT_SECONDS — rapid-fire per-URL
# requests is itself a bot signal.
_BROWSER_RATE_LIMIT_SECONDS = 1.5

# Operator escape hatch: an ambient Chromium build can mismatch the installed
# playwright package's expected build (this container is exactly such a
# case). When set, it is passed as chromium.launch(executable_path=...) on
# the Playwright path and as options.binary_location on the Selenium path.
_CHROMIUM_PATH_ENV = "TC_CHROMIUM_PATH"

_NAV_TIMEOUT_MS = 30_000
_REQUEST_TIMEOUT_MS = 30_000
# Short on purpose. A CSV attachment either starts arriving immediately or is
# not coming at all, and this timeout is paid PER URL — at 30s across a
# 22-ticker universe a blocked run burned ~11 minutes before giving up.
_DOWNLOAD_TIMEOUT_MS = 8_000

# Stooq blocks all-or-nothing. If the first few tickers come back as
# something other than CSV, the rest will too — bail instead of grinding
# through the whole universe collecting the same challenge page. This is a
# Stooq-specific opt-in (see fetch_urls_as_text's expected_prefix), not
# baked into the generic path.
_ABORT_AFTER_CONSECUTIVE_NON_CSV = 3

# Cheap reduction of the automation signature on the navigation path.
# NOTE: these did NOT lift the "Access denied" the request API received —
# switching to a real navigation did. They are kept because they are
# harmless and the navigation path is the one that matters.
_STEALTH_LAUNCH_ARGS = ["--disable-blink-features=AutomationControlled"]
_STEALTH_INIT_SCRIPT = "Object.defineProperty(navigator, 'webdriver', {get: () => undefined});"
# Short post-navigation wait letting page JS (e.g. a challenge) run and
# settle before the page is read or a follow-up request is issued.
_CHALLENGE_SETTLE_MS = 2_000

# Explicit viewport for both fetch paths. A default or absent viewport is
# itself a distinguishing signal, and some layouts render differently
# without one — set it explicitly rather than rely on Playwright's default.
_VIEWPORT = {"width": 1280, "height": 800}

# Navigation wait predicate for fetch_page_html. FORBIDDEN alternative:
# "networkidle", which waits for 500ms of zero network traffic — pages
# carrying analytics, ads, or polling never reach that quiet window, so it
# times out on pages that loaded perfectly fine. A grep gate in this repo's
# tests enforces that "networkidle" never appears in this file.
_WAIT_UNTIL = "domcontentloaded"

# Engine selection env var, checked by _resolve_engine when the caller does
# not pass an explicit engine= argument. Accepted values: "playwright",
# "selenium", "auto" (default).
_BROWSER_ENGINE_ENV = "TC_BROWSER_ENGINE"

# Pinned in code (not read from the ambient browser) so the value is visible
# and stable across whatever Chrome build happens to be installed. This is
# hygiene — a UA string any real Chrome install would also send — not an
# evasion technique.
_SELENIUM_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# Applied to the PLAYWRIGHT context, which otherwise inherits Chromium's own
# headless User-Agent. Observed on 2026-08-05: third-party requests from a page
# we loaded carried `m_ch_ua=..."HeadlessChrome"|v="151"` — the browser was
# announcing itself as headless in client-hint headers.
#
# NOTE this is the OPPOSITE call to the one made for the curl_cffi client in
# ingestion/http.py, and deliberately so. There, curl_cffi supplies a complete
# header set matched to the TLS fingerprint it presents, so overriding the UA
# CREATES an inconsistency. Here, headless Chromium's own default is ALREADY
# inconsistent with what a real Chrome sends, so overriding it removes one.
# The rule in both cases is the same: make the UA agree with everything else
# the client presents.
_PLAYWRIGHT_USER_AGENT = _SELENIUM_USER_AGENT

# Operator toggle: set TC_BROWSER_HEADLESS=false to drive a VISIBLE browser.
# Headless leaves signals no flag fully suppresses; a real windowed session
# does not. Only usable on a machine with a display, which is why headless
# stays the default.
_HEADLESS_ENV = "TC_BROWSER_HEADLESS"

# Window size mirrors _VIEWPORT so both engines render at the same size.
_SELENIUM_ARGS = [
    "--headless=new",
    "--disable-blink-features=AutomationControlled",
    "--window-size=1280,800",
    f"--user-agent={_SELENIUM_USER_AGENT}",
]


def playwright_available() -> bool:
    """Return True when the ``playwright`` package is importable."""
    return _PLAYWRIGHT_AVAILABLE


def selenium_available() -> bool:
    """Return True when the ``selenium`` package is importable."""
    return _SELENIUM_AVAILABLE


def _launch_hint(exc: BaseException) -> str:
    """Append an actionable remedy to browser-launch failures we recognise.

    A missing shared library buries the only useful line ("error while loading
    shared libraries: libnspr4.so") inside hundreds of lines of Chromium
    command-line flags. The browser downloaded fine; the HOST lacks Chromium's
    system dependencies — a distinct failure from "playwright not installed",
    and it needs a distinct fix.
    """
    text = str(exc)
    if "error while loading shared libraries" in text or "cannot open shared object file" in text:
        return (
            f"{text}\n"
            "  >> Chromium is installed but the HOST is missing its system libraries. Fix with:\n"
            "       sudo $(which playwright) install-deps chromium\n"
            "     or on Debian/Ubuntu install libnspr4, libnss3, libasound2t64, libatk1.0-0, "
            "libatk-bridge2.0-0, libcups2, libdrm2, libxkbcommon0, libxcomposite1, libxdamage1, "
            "libxfixes3, libxrandr2, libgbm1, libpango-1.0-0, libcairo2."
        )
    return text


def headless_mode() -> bool:
    """Return False only when TC_BROWSER_HEADLESS is explicitly falsy.

    Defaults to headless because a visible window needs a display, which CI
    and most servers lack. Set TC_BROWSER_HEADLESS=false on a desktop to drive
    a real windowed browser, which emits none of the headless-specific signals.
    """
    raw = (os.environ.get(_HEADLESS_ENV) or "").strip().lower()
    return raw not in {"false", "0", "no", "off"}


def _launch_kwargs() -> dict[str, Any]:
    """Shared ``chromium.launch()`` kwargs for both fetch paths.

    ``TC_CHROMIUM_PATH`` is an operator escape hatch for when the ambient
    Chromium build mismatches the installed playwright package's expected
    build; the key is left ABSENT (not set to None) when unset so Playwright
    uses its own bundled browser.
    """
    kwargs: dict[str, Any] = {"headless": headless_mode(), "args": list(_STEALTH_LAUNCH_ARGS)}
    chromium_path = os.environ.get(_CHROMIUM_PATH_ENV)
    if chromium_path:
        kwargs["executable_path"] = chromium_path
    return kwargs


def _resolve_engine(engine: str | None) -> str | None:
    """Resolve which browser engine :func:`fetch_page_html` should use.

    Precedence: explicit *engine* argument, then ``TC_BROWSER_ENGINE``, then
    ``"auto"`` (Playwright preferred, Selenium as fallback). An explicit
    request for an engine that is not available is honored as a failure —
    it does NOT silently fall through to the other engine, because the
    operator asked for something specific and not getting it is exactly what
    they need to see.
    """
    requested = (engine or os.environ.get(_BROWSER_ENGINE_ENV) or "auto").strip().lower()

    if requested == "playwright":
        if _PLAYWRIGHT_AVAILABLE:
            return "playwright"
        log.warning(
            "TC_BROWSER_ENGINE/engine explicitly requested 'playwright' but it is not "
            "installed. Install with: pip install 'playwright>=1.40'  then: playwright "
            "install chromium"
        )
        return None

    if requested == "selenium":
        if _SELENIUM_AVAILABLE:
            return "selenium"
        log.warning(
            "TC_BROWSER_ENGINE/engine explicitly requested 'selenium' but it is not "
            "installed. Install with: pip install 'selenium>=4.15'  plus a Chrome/Chromium "
            "binary."
        )
        return None

    if requested != "auto":
        log.warning(
            "Unrecognised browser engine %r (accepted: 'playwright', 'selenium', 'auto') — "
            "treating as 'auto'.", requested,
        )

    if _PLAYWRIGHT_AVAILABLE:
        return "playwright"
    if _SELENIUM_AVAILABLE:
        return "selenium"
    log.warning(
        "No browser engine available. Install one of: "
        "pip install 'playwright>=1.40'  then: playwright install chromium   —or—   "
        "pip install 'selenium>=4.15'  plus a Chrome/Chromium binary"
    )
    return None


def _fetch_page_html_playwright(
    url: str,
    *,
    wait_for_selector: str | None,
    require_selector: bool,
    settle_ms: int,
    timeout_ms: int,
) -> str | None:
    """Render *url* with Playwright and return its HTML, or None. Never raises."""
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(**_launch_kwargs())
            try:
                context = browser.new_context(viewport=dict(_VIEWPORT), user_agent=_PLAYWRIGHT_USER_AGENT)
                context.add_init_script(_STEALTH_INIT_SCRIPT)
                page = context.new_page()
                page.goto(url, timeout=timeout_ms, wait_until=_WAIT_UNTIL)

                if wait_for_selector:
                    try:
                        # state="attached", NOT Playwright's default "visible".
                        # We want the element in the DOM so the HTML we return
                        # carries it — not on screen. macrotrends' data table is
                        # present but hidden behind a chart/table toggle, so a
                        # visibility wait times out (30s per series) on a page
                        # whose table was there the whole time.
                        page.wait_for_selector(wait_for_selector, timeout=timeout_ms, state="attached")
                    except PlaywrightError as exc:
                        if require_selector:
                            log.warning(
                                "Selector %r never appeared for %s (%s) — required, returning None",
                                wait_for_selector, url, str(exc)[:80],
                            )
                            return None
                        log.warning(
                            "Selector %r never appeared for %s (%s) — not required, returning "
                            "rendered HTML anyway",
                            wait_for_selector, url, str(exc)[:80],
                        )

                if settle_ms:
                    page.wait_for_timeout(settle_ms)
                return page.content() or None
            finally:
                browser.close()
    except Exception as exc:  # noqa: BLE001 — launch/navigation degradation, never raise to caller
        log.warning("Playwright fetch_page_html failed for %s: %s", url, _launch_hint(exc))
        return None


def _fetch_page_html_selenium(
    url: str,
    *,
    wait_for_selector: str | None,
    require_selector: bool,
    settle_ms: int,
    timeout_ms: int,
) -> str | None:
    """Render *url* with Selenium and return its HTML, or None. Never raises.

    Selenium implements fetch_page_html ONLY — it has no download-
    interception equivalent, so fetch_urls_as_text stays Playwright-only.
    """
    driver = None
    try:
        options = ChromeOptions()
        for arg in _SELENIUM_ARGS:
            options.add_argument(arg)
        chromium_path = os.environ.get(_CHROMIUM_PATH_ENV)
        if chromium_path:
            options.binary_location = chromium_path

        driver = webdriver.Chrome(options=options)
        driver.set_page_load_timeout(timeout_ms / 1000.0)
        driver.get(url)

        if wait_for_selector:
            try:
                WebDriverWait(driver, timeout_ms / 1000.0).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, wait_for_selector))
                )
            except SeleniumTimeout as exc:
                if require_selector:
                    log.warning(
                        "Selector %r never appeared for %s (%s) — required, returning None",
                        wait_for_selector, url, str(exc)[:80],
                    )
                    return None
                log.warning(
                    "Selector %r never appeared for %s (%s) — not required, returning "
                    "rendered HTML anyway",
                    wait_for_selector, url, str(exc)[:80],
                )

        if settle_ms:
            time.sleep(settle_ms / 1000.0)
        return driver.page_source or None
    except Exception as exc:  # noqa: BLE001 — launch/navigation degradation, never raise to caller
        log.warning("Selenium fetch_page_html failed for %s: %s", url, exc)
        return None
    finally:
        if driver is not None:
            # An orphaned chromedriver process per failed run is worse than a
            # swallowed teardown error.
            with contextlib.suppress(Exception):
                driver.quit()


def fetch_page_html(
    url: str,
    *,
    wait_for_selector: str | None = None,
    require_selector: bool = True,
    settle_ms: int = _CHALLENGE_SETTLE_MS,
    timeout_ms: int = _NAV_TIMEOUT_MS,
    engine: str | None = None,
) -> str | None:
    """Render *url* in a real browser and return its HTML, or None.

    ``wait_for_selector`` is a GUESS at an element that signals the page is
    ready. ``require_selector`` controls what happens when that guess is
    wrong:

    - ``True`` (default): a selector that never appears is treated as a real
      failure — the caller asked for a specific element and not getting it
      means the page did not render what was expected. Returns None.
    - ``False``: a selector miss is logged at WARNING and whatever rendered
      is returned anyway, so an unconfirmed selector cannot throw away a
      page that may still carry the data (e.g. in a ``<script>`` tag) even
      though no matching element exists.

    ``engine`` selects the browser engine via :func:`_resolve_engine`
    (argument, then ``TC_BROWSER_ENGINE``, then Playwright-first ``auto``).
    Never raises: returns None and logs a WARNING when no engine is
    available, when the launch fails, or when navigation fails.
    """
    resolved = _resolve_engine(engine)
    if resolved == "playwright":
        return _fetch_page_html_playwright(
            url,
            wait_for_selector=wait_for_selector,
            require_selector=require_selector,
            settle_ms=settle_ms,
            timeout_ms=timeout_ms,
        )
    if resolved == "selenium":
        return _fetch_page_html_selenium(
            url,
            wait_for_selector=wait_for_selector,
            require_selector=require_selector,
            settle_ms=settle_ms,
            timeout_ms=timeout_ms,
        )
    return None


def _fetch_one_via_download(page: Any, url: str) -> str | None:
    """Navigate to *url* and read the file the endpoint serves back as a download.

    Some endpoints (e.g. Stooq's CSV download) respond with a file
    attachment, so ``page.goto`` raises ``Error: Download is starting``
    instead of completing a navigation. That raise is EXPECTED and is
    swallowed deliberately — the download itself is the payload. Returns
    None if no download materialises in time.
    """
    try:
        with page.expect_download(timeout=_DOWNLOAD_TIMEOUT_MS) as download_info:
            with contextlib.suppress(PlaywrightError):
                page.goto(url, timeout=_NAV_TIMEOUT_MS)
        download = download_info.value
    except PlaywrightError as exc:
        # expect_download RAISES when no download arrives. That must return
        # None so the in-page fetch fallback still gets its turn — letting it
        # propagate would skip the fallback entirely for this URL.
        log.debug("No download event for %s (%s)", url, str(exc)[:80])
        return None

    path = download.path()
    if path is None:
        return None
    try:
        return Path(path).read_text()
    finally:
        # The temp file is ours once read; failing to clean it up would leak
        # one file per URL per run.
        with contextlib.suppress(OSError):
            download.delete()


def _fetch_one_via_page_fetch(page: Any, url: str) -> str | None:
    """Fallback: same-origin ``fetch()`` executed inside the page.

    Carries the page's cookies, origin and referer, which a detached request
    client does not. Only tried when the download path yields nothing.
    """
    text = page.evaluate("async (u) => (await fetch(u)).text()", url)
    return text if text else None


def fetch_urls_as_text(
    urls: dict[str, str],
    *,
    warmup_url: str | None = None,
    rate_limit_seconds: float = _BROWSER_RATE_LIMIT_SECONDS,
    expected_prefix: str | None = None,
    abort_after_consecutive_mismatch: int = _ABORT_AFTER_CONSECUTIVE_NON_CSV,
) -> dict[str, str]:
    """Fetch each URL through one headless-Chromium session, as raw text.

    This is the Playwright download-interception flow originally built for
    Stooq (:func:`fetch_stooq_csvs` is now a thin wrapper over this
    function), generalised off Stooq's shape. PLAYWRIGHT-ONLY: there is no
    Selenium equivalent of Playwright's download interception
    (``page.expect_download``), so this function does not accept an
    ``engine`` argument and never falls back to Selenium even when Selenium
    is the only engine installed — no faked Selenium download path exists.

    Launches exactly one Chromium instance. When *warmup_url* is given,
    navigates there once first so any challenge JS executes and sets its
    verification cookie before any target URL is fetched; when None (the
    generic case) no warm-up navigation happens. Each URL is then fetched by
    NAVIGATING to it and taking the file the endpoint serves back as a
    download (see :func:`_fetch_one_via_download`), falling back to an
    in-page ``fetch()`` when no download materialises.

    Everything stays inside the browser that solved any challenge — no
    cookie is harvested and replayed through a different HTTP client,
    because a cookie may be bound to the identity that earned it.

    ``expected_prefix`` is a Stooq-specific opt-in: when set, a body that
    does not start with it (case-insensitively, after stripping leading
    whitespace) counts toward ``abort_after_consecutive_mismatch``
    consecutive misses, after which the remaining URLs are abandoned rather
    than working through the whole batch collecting the same unwanted body.
    When ``expected_prefix`` is None (the default / generic case), the
    counter never increments and EVERY url is attempted.

    Returns plain ``{key: text}`` for whatever was recovered before any
    failure. This function performs NO parsing, validation, or
    interpretation of the returned text — that is the caller's job. It
    never raises: a missing playwright install, a failed launch, a failed
    navigation, or a failed per-URL fetch all degrade to returning whatever
    was already accumulated, with a WARNING logged for each cause.
    """
    if not urls:
        return {}

    if not _PLAYWRIGHT_AVAILABLE:
        log.warning(
            "playwright is not installed — headless-browser fallback skipped. This download-"
            "interception flow is Playwright-specific, so Selenium cannot serve this call even "
            "when it is installed. Install with: pip install 'playwright>=1.40'  then: "
            "playwright install chromium"
        )
        return {}

    results: dict[str, str] = {}
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(**_launch_kwargs())
            try:
                # accept_downloads is required — some endpoints answer a
                # navigation with a file attachment, not a page.
                context = browser.new_context(
                    accept_downloads=True, viewport=dict(_VIEWPORT), user_agent=_PLAYWRIGHT_USER_AGENT
                )
                context.add_init_script(_STEALTH_INIT_SCRIPT)
                page = context.new_page()
                if warmup_url:
                    page.goto(warmup_url, timeout=_NAV_TIMEOUT_MS)
                    page.wait_for_timeout(_CHALLENGE_SETTLE_MS)

                consecutive_mismatch = 0
                for i, (key, url) in enumerate(urls.items()):
                    if i > 0:
                        time.sleep(rate_limit_seconds)
                    try:
                        text = _fetch_one_via_download(page, url)
                        if not text:
                            log.debug("No download for %s — trying in-page fetch", key)
                            text = _fetch_one_via_page_fetch(page, url)
                        if not text:
                            log.warning("Browser fetch yielded nothing for %s", key)
                            continue
                        results[key] = text
                    except Exception as exc:  # noqa: BLE001 — one bad url must not abort the batch
                        log.warning("Browser fetch failed for %s: %s", key, exc)
                        text = None

                    # Stooq-specific opt-in — see docstring. The generic path
                    # (expected_prefix=None) never increments this counter.
                    if expected_prefix is None:
                        continue
                    if text and text.lstrip().lower().startswith(expected_prefix):
                        consecutive_mismatch = 0
                    else:
                        consecutive_mismatch += 1
                        if consecutive_mismatch >= abort_after_consecutive_mismatch:
                            log.warning(
                                "Got unexpected content for %d consecutive URLs — abandoning the "
                                "browser path rather than working through the remaining %d.",
                                consecutive_mismatch, len(urls) - i - 1,
                            )
                            break
            finally:
                # Must survive a mid-batch failure — this is the only reason
                # the try/finally exists; do not collapse it.
                browser.close()
    except Exception as exc:  # noqa: BLE001 — launch/navigation degradation, never raise to caller
        log.warning("Headless-browser fetch_urls_as_text failed: %s", exc)

    return results


def fetch_stooq_csvs(
    urls: dict[str, str],
    *,
    warmup_url: str = _STOOQ_WARMUP_URL,
    rate_limit_seconds: float = _BROWSER_RATE_LIMIT_SECONDS,
) -> dict[str, str]:
    """Fetch each Stooq CSV URL through one headless-Chromium session.

    A thin wrapper over :func:`fetch_urls_as_text` with Stooq's warm-up URL
    and its ``"date,"`` CSV-header opt-in for the consecutive-mismatch
    early-bail. Kept as a named, backwards-compatible entry point because
    ``prices_daily.py`` imports it directly.

    Note that Playwright's own ``context.request.get`` is NOT refused
    because it lacks cookies — it is refused by this endpoint with "Access
    denied" even inside the solved browser context; only a real navigation
    is served. See the module docstring's second verdict for why this
    module navigates and takes the download rather than asking the request
    API.

    Returns plain ``{ticker: csv_text}`` for whatever was recovered before
    any failure — see :func:`fetch_urls_as_text` for the full contract.
    """
    return fetch_urls_as_text(
        urls,
        warmup_url=warmup_url,
        rate_limit_seconds=rate_limit_seconds,
        expected_prefix="date,",
    )
