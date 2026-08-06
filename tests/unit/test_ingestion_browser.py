"""Unit tests for the headless-Chromium Stooq CSV fetcher
(trading_crab_lib.ingestion.browser).

Playwright is mocked ENTIRELY in every test — no browser launch, no network.

The fakes model the real control flow the module depends on: Stooq's CSV
endpoint answers a NAVIGATION with a file attachment, so ``page.goto`` raises
"Download is starting" and the payload arrives via ``page.expect_download``.
Modelling that ordering (expect_download entered first, goto inside it) is the
point — a looser mock would pass even if the module used the request API that
Stooq refuses with "Access denied".
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from trading_crab_lib.ingestion import browser as browser_mod


class _FakeDownload:
    def __init__(self, path_str: str | None) -> None:
        self._path = path_str
        self.deleted = False

    def path(self) -> str | None:
        return self._path

    def delete(self) -> None:
        self.deleted = True


class _FakeDownloadInfo:
    def __init__(self, page: _FakePage) -> None:
        self._page = page

    @property
    def value(self) -> _FakeDownload:
        # Resolved at access time — after the `with` body has run goto.
        return self._page.pending_download


class _FakeExpectDownloadCM:
    def __init__(self, page: _FakePage) -> None:
        self._page = page

    def __enter__(self) -> _FakeDownloadInfo:
        return _FakeDownloadInfo(self._page)

    def __exit__(self, *_exc: object) -> bool:
        return False


class _FakePage:
    """Fake Playwright Page.

    ``download_behavior`` maps url -> csv text (a download is produced),
    an Exception (goto raises it), or None (no download materialises).
    ``fetch_behavior`` maps url -> text returned by the in-page fetch fallback.
    """

    def __init__(
        self,
        download_behavior: dict[str, object],
        fetch_behavior: dict[str, str] | None = None,
        tmpdir: str | None = None,
    ) -> None:
        self.download_behavior = download_behavior
        self.fetch_behavior = fetch_behavior or {}
        self.tmpdir = tmpdir or tempfile.mkdtemp()
        self.goto_urls: list[str] = []
        self.evaluate_urls: list[str] = []
        self.expect_download_count = 0
        self.downloads: list[_FakeDownload] = []
        self.pending_download = _FakeDownload(None)

    def goto(self, url: str, timeout: float | None = None) -> None:
        self.goto_urls.append(url)
        behavior = self.download_behavior.get(url)
        if isinstance(behavior, Exception):
            raise behavior
        if isinstance(behavior, str):
            target = Path(self.tmpdir) / f"dl{len(self.goto_urls)}.csv"
            target.write_text(behavior)
            self.pending_download = _FakeDownload(str(target))
            self.downloads.append(self.pending_download)

    def expect_download(self, timeout: float | None = None) -> _FakeExpectDownloadCM:
        self.expect_download_count += 1
        self.pending_download = _FakeDownload(None)  # reset; goto fills it in
        return _FakeExpectDownloadCM(self)

    def wait_for_timeout(self, _ms: float) -> None:
        return None

    def evaluate(self, _script: str, url: str) -> str | None:
        self.evaluate_urls.append(url)
        return self.fetch_behavior.get(url)


def _build_fake_playwright(page: _FakePage):
    """Assemble a fake sync_playwright() chain around *page*."""
    pw_cm = MagicMock()
    pw_obj = MagicMock()
    pw_cm.__enter__.return_value = pw_obj
    pw_cm.__exit__.return_value = False

    browser_obj = MagicMock()
    pw_obj.chromium.launch.return_value = browser_obj

    context_obj = MagicMock()
    browser_obj.new_context.return_value = context_obj
    context_obj.new_page.return_value = page

    return pw_cm, pw_obj, browser_obj, context_obj


# ── playwright_available() ──────────────────────────────────────────────────


@patch("trading_crab_lib.ingestion.browser._PLAYWRIGHT_AVAILABLE", True)
def test_playwright_available_reflects_true_flag():
    assert browser_mod.playwright_available() is True


@patch("trading_crab_lib.ingestion.browser._PLAYWRIGHT_AVAILABLE", False)
def test_playwright_available_reflects_false_flag():
    assert browser_mod.playwright_available() is False


# ── fetch_stooq_csvs: happy path via the download flow ──────────────────────


@patch("trading_crab_lib.ingestion.browser._PLAYWRIGHT_AVAILABLE", True)
@patch("trading_crab_lib.ingestion.browser.time.sleep")
@patch("trading_crab_lib.ingestion.browser.sync_playwright")
def test_fetch_stooq_csvs_one_browser_one_warmup_n_downloads(mock_sync_pw, mock_sleep):
    urls = {
        "SPY": "https://stooq.com/q/d/l/?s=spy.us",
        "QQQ": "https://stooq.com/q/d/l/?s=qqq.us",
        "IWM": "https://stooq.com/q/d/l/?s=iwm.us",
    }
    page = _FakePage({
        urls["SPY"]: "date,close\nspy-data",
        urls["QQQ"]: "date,close\nqqq-data",
        urls["IWM"]: "date,close\niwm-data",
    })
    pw_cm, pw_obj, browser_obj, _context = _build_fake_playwright(page)
    mock_sync_pw.return_value = pw_cm

    out = browser_mod.fetch_stooq_csvs(urls)

    assert out == {
        "SPY": "date,close\nspy-data",
        "QQQ": "date,close\nqqq-data",
        "IWM": "date,close\niwm-data",
    }
    pw_obj.chromium.launch.assert_called_once()
    # One warm-up navigation + one navigation per ticker.
    assert len(page.goto_urls) == 4
    assert page.expect_download_count == 3
    browser_obj.close.assert_called_once()
    assert mock_sleep.call_count >= 2


@patch("trading_crab_lib.ingestion.browser._PLAYWRIGHT_AVAILABLE", True)
@patch("trading_crab_lib.ingestion.browser.time.sleep")
@patch("trading_crab_lib.ingestion.browser.sync_playwright")
def test_fetch_stooq_csvs_context_accepts_downloads(mock_sync_pw, mock_sleep):
    """Without accept_downloads the attachment is discarded and every ticker
    silently yields nothing."""
    urls = {"SPY": "https://stooq.com/q/d/l/?s=spy.us"}
    page = _FakePage({urls["SPY"]: "date,close\nspy-data"})
    pw_cm, _pw, browser_obj, _context = _build_fake_playwright(page)
    mock_sync_pw.return_value = pw_cm

    browser_mod.fetch_stooq_csvs(urls)

    _, kwargs = browser_obj.new_context.call_args
    assert kwargs.get("accept_downloads") is True


@patch("trading_crab_lib.ingestion.browser._PLAYWRIGHT_AVAILABLE", True)
@patch("trading_crab_lib.ingestion.browser.time.sleep")
@patch("trading_crab_lib.ingestion.browser.sync_playwright")
def test_fetch_stooq_csvs_goto_raising_download_starting_is_not_a_failure(mock_sync_pw, mock_sleep):
    """`page.goto` raising "Download is starting" is the EXPECTED success path
    for this endpoint — it must be swallowed and the download still read."""
    urls = {"SPY": "https://stooq.com/q/d/l/?s=spy.us"}
    tmpdir = tempfile.mkdtemp()
    csv_file = Path(tmpdir) / "spy.csv"
    csv_file.write_text("date,close\nspy-data")

    page = _FakePage({}, tmpdir=tmpdir)

    def _goto(url: str, timeout: float | None = None) -> None:
        page.goto_urls.append(url)
        if url in urls.values():
            page.pending_download = _FakeDownload(str(csv_file))
            raise browser_mod.PlaywrightError("Page.goto: Download is starting")

    page.goto = _goto  # type: ignore[method-assign]
    pw_cm, _pw, browser_obj, _context = _build_fake_playwright(page)
    mock_sync_pw.return_value = pw_cm

    out = browser_mod.fetch_stooq_csvs(urls)

    assert out == {"SPY": "date,close\nspy-data"}
    browser_obj.close.assert_called_once()


@patch("trading_crab_lib.ingestion.browser._PLAYWRIGHT_AVAILABLE", True)
@patch("trading_crab_lib.ingestion.browser.time.sleep")
@patch("trading_crab_lib.ingestion.browser.sync_playwright")
def test_fetch_stooq_csvs_falls_back_to_in_page_fetch_when_no_download(mock_sync_pw, mock_sleep):
    urls = {"SPY": "https://stooq.com/q/d/l/?s=spy.us"}
    # No download materialises; the in-page fetch supplies the text.
    page = _FakePage({urls["SPY"]: None}, fetch_behavior={urls["SPY"]: "date,close\nvia-fetch"})
    pw_cm, _pw, _browser, _context = _build_fake_playwright(page)
    mock_sync_pw.return_value = pw_cm

    out = browser_mod.fetch_stooq_csvs(urls)

    assert out == {"SPY": "date,close\nvia-fetch"}
    assert page.evaluate_urls == [urls["SPY"]]


@patch("trading_crab_lib.ingestion.browser._PLAYWRIGHT_AVAILABLE", True)
@patch("trading_crab_lib.ingestion.browser.time.sleep")
@patch("trading_crab_lib.ingestion.browser.sync_playwright")
def test_fetch_stooq_csvs_does_not_use_in_page_fetch_when_download_succeeds(mock_sync_pw, mock_sleep):
    urls = {"SPY": "https://stooq.com/q/d/l/?s=spy.us"}
    page = _FakePage(
        {urls["SPY"]: "date,close\nvia-download"},
        fetch_behavior={urls["SPY"]: "date,close\nSHOULD-NOT-BE-USED"},
    )
    pw_cm, _pw, _browser, _context = _build_fake_playwright(page)
    mock_sync_pw.return_value = pw_cm

    out = browser_mod.fetch_stooq_csvs(urls)

    assert out == {"SPY": "date,close\nvia-download"}
    assert page.evaluate_urls == []


@patch("trading_crab_lib.ingestion.browser._PLAYWRIGHT_AVAILABLE", True)
@patch("trading_crab_lib.ingestion.browser.time.sleep")
@patch("trading_crab_lib.ingestion.browser.sync_playwright")
def test_fetch_stooq_csvs_deletes_downloaded_temp_file(mock_sync_pw, mock_sleep):
    """One leaked temp file per ticker per run would accumulate silently."""
    urls = {"SPY": "https://stooq.com/q/d/l/?s=spy.us"}
    page = _FakePage({urls["SPY"]: "date,close\nspy-data"})
    pw_cm, _pw, _browser, _context = _build_fake_playwright(page)
    mock_sync_pw.return_value = pw_cm

    browser_mod.fetch_stooq_csvs(urls)

    assert page.downloads and all(d.deleted for d in page.downloads)


# ── degradation ─────────────────────────────────────────────────────────────


@patch("trading_crab_lib.ingestion.browser._PLAYWRIGHT_AVAILABLE", False)
@patch("trading_crab_lib.ingestion.browser.sync_playwright")
def test_fetch_stooq_csvs_returns_empty_and_warns_when_playwright_missing(mock_sync_pw, caplog):
    with caplog.at_level(logging.WARNING):
        out = browser_mod.fetch_stooq_csvs({"SPY": "https://stooq.com/q/d/l/?s=spy.us"})

    assert out == {}
    mock_sync_pw.assert_not_called()
    assert any("pip install" in rec.message for rec in caplog.records)


@patch("trading_crab_lib.ingestion.browser._PLAYWRIGHT_AVAILABLE", True)
@patch("trading_crab_lib.ingestion.browser.sync_playwright")
def test_fetch_stooq_csvs_empty_urls_returns_empty_without_launching(mock_sync_pw):
    out = browser_mod.fetch_stooq_csvs({})
    assert out == {}
    mock_sync_pw.assert_not_called()


@patch("trading_crab_lib.ingestion.browser._PLAYWRIGHT_AVAILABLE", True)
@patch("trading_crab_lib.ingestion.browser.time.sleep")
@patch("trading_crab_lib.ingestion.browser.sync_playwright")
def test_fetch_stooq_csvs_partial_failure_keeps_prior_results_and_closes_browser(mock_sync_pw, mock_sleep):
    urls = {
        "SPY": "https://stooq.com/q/d/l/?s=spy.us",
        "QQQ": "https://stooq.com/q/d/l/?s=qqq.us",
        "IWM": "https://stooq.com/q/d/l/?s=iwm.us",
    }
    page = _FakePage({
        urls["SPY"]: "date,close\nspy-data",
        urls["QQQ"]: RuntimeError("boom mid-batch"),
        urls["IWM"]: "date,close\niwm-data",
    })
    pw_cm, _pw, browser_obj, _context = _build_fake_playwright(page)
    mock_sync_pw.return_value = pw_cm

    out = browser_mod.fetch_stooq_csvs(urls)

    assert out == {"SPY": "date,close\nspy-data", "IWM": "date,close\niwm-data"}
    assert "QQQ" not in out
    browser_obj.close.assert_called_once()


# ── TC_CHROMIUM_PATH escape hatch ───────────────────────────────────────────


@patch("trading_crab_lib.ingestion.browser._PLAYWRIGHT_AVAILABLE", True)
@patch("trading_crab_lib.ingestion.browser.time.sleep")
@patch("trading_crab_lib.ingestion.browser.sync_playwright")
def test_fetch_stooq_csvs_passes_chromium_path_env_var_to_launch(mock_sync_pw, mock_sleep, monkeypatch):
    monkeypatch.setenv("TC_CHROMIUM_PATH", "/custom/chrome-linux/chrome")
    urls = {"SPY": "https://stooq.com/q/d/l/?s=spy.us"}
    page = _FakePage({urls["SPY"]: "date,close\nspy-data"})
    pw_cm, pw_obj, _browser, _context = _build_fake_playwright(page)
    mock_sync_pw.return_value = pw_cm

    browser_mod.fetch_stooq_csvs(urls)

    _, kwargs = pw_obj.chromium.launch.call_args
    assert kwargs.get("executable_path") == "/custom/chrome-linux/chrome"


@patch("trading_crab_lib.ingestion.browser._PLAYWRIGHT_AVAILABLE", True)
@patch("trading_crab_lib.ingestion.browser.time.sleep")
@patch("trading_crab_lib.ingestion.browser.sync_playwright")
def test_fetch_stooq_csvs_omits_executable_path_key_when_env_unset(mock_sync_pw, mock_sleep, monkeypatch):
    monkeypatch.delenv("TC_CHROMIUM_PATH", raising=False)
    urls = {"SPY": "https://stooq.com/q/d/l/?s=spy.us"}
    page = _FakePage({urls["SPY"]: "date,close\nspy-data"})
    pw_cm, pw_obj, _browser, _context = _build_fake_playwright(page)
    mock_sync_pw.return_value = pw_cm

    browser_mod.fetch_stooq_csvs(urls)

    _, kwargs = pw_obj.chromium.launch.call_args
    assert "executable_path" not in kwargs


# ── regressions from the 2026-08-05 live run ────────────────────────────────


@patch("trading_crab_lib.ingestion.browser._PLAYWRIGHT_AVAILABLE", True)
@patch("trading_crab_lib.ingestion.browser.time.sleep")
@patch("trading_crab_lib.ingestion.browser.sync_playwright")
def test_download_timeout_still_falls_back_to_in_page_fetch(mock_sync_pw, mock_sleep):
    """expect_download RAISES on timeout. If that propagates, the in-page fetch
    fallback is skipped entirely — which is what happened on the live run."""
    urls = {"SPY": "https://stooq.com/q/d/l/?s=spy.us"}
    page = _FakePage({}, fetch_behavior={urls["SPY"]: "date,close\nvia-fetch"})

    def _expect_download_timeout(timeout=None):
        raise browser_mod.PlaywrightError('Timeout 8000ms exceeded while waiting for event "download"')

    page.expect_download = _expect_download_timeout  # type: ignore[method-assign]
    pw_cm, _pw, _browser, _context = _build_fake_playwright(page)
    mock_sync_pw.return_value = pw_cm

    out = browser_mod.fetch_stooq_csvs(urls)

    assert out == {"SPY": "date,close\nvia-fetch"}
    assert page.evaluate_urls == [urls["SPY"]]


@patch("trading_crab_lib.ingestion.browser._PLAYWRIGHT_AVAILABLE", True)
@patch("trading_crab_lib.ingestion.browser.time.sleep")
@patch("trading_crab_lib.ingestion.browser.sync_playwright")
def test_abandons_browser_path_after_consecutive_non_csv(mock_sync_pw, mock_sleep):
    """Stooq blocks all-or-nothing — do not spend a per-ticker timeout on all
    22 collecting the same challenge page."""
    urls = {f"T{i}": f"https://stooq.com/q/d/l/?s=t{i}.us" for i in range(10)}
    challenge = "<!DOCTYPE html><html><head>challenge"
    page = _FakePage({u: challenge for u in urls.values()})
    pw_cm, _pw, _browser, _context = _build_fake_playwright(page)
    mock_sync_pw.return_value = pw_cm

    browser_mod.fetch_stooq_csvs(urls)

    assert page.expect_download_count == browser_mod._ABORT_AFTER_CONSECUTIVE_NON_CSV
    assert page.expect_download_count < len(urls)


@patch("trading_crab_lib.ingestion.browser._PLAYWRIGHT_AVAILABLE", True)
@patch("trading_crab_lib.ingestion.browser.time.sleep")
@patch("trading_crab_lib.ingestion.browser.sync_playwright")
def test_valid_csv_resets_the_consecutive_non_csv_counter(mock_sync_pw, mock_sleep):
    """A single bad ticker in the middle must not abort an otherwise fine run."""
    urls = {f"T{i}": f"https://stooq.com/q/d/l/?s=t{i}.us" for i in range(6)}
    behavior = {u: "date,close\nok" for u in urls.values()}
    behavior[urls["T2"]] = "<!DOCTYPE html>nope"
    page = _FakePage(behavior)
    pw_cm, _pw, _browser, _context = _build_fake_playwright(page)
    mock_sync_pw.return_value = pw_cm

    out = browser_mod.fetch_stooq_csvs(urls)

    assert page.expect_download_count == len(urls)
    assert len(out) == len(urls)


# ═══════════════════════════════════════════════════════════════════════════
# Task 1 additions: fetch_page_html + fetch_urls_as_text generalization
# ═══════════════════════════════════════════════════════════════════════════


class _FakeHtmlPage:
    """Fake Playwright Page for the fetch_page_html path.

    Deliberately exposes goto(url, timeout=None, wait_until=None) — a wider
    signature than _FakePage, because fetch_page_html's navigation (unlike
    fetch_urls_as_text's) is allowed to pass wait_until.
    """

    def __init__(self, content_text: str = "<html>ok</html>", selector_raises: bool = False) -> None:
        self._content_text = content_text
        self._selector_raises = selector_raises
        self.goto_calls: list[tuple[str, float | None, str | None]] = []
        self.wait_for_selector_calls: list[tuple[str, float | None, str | None]] = []
        self.wait_for_timeout_calls: list[float] = []

    def goto(self, url: str, timeout: float | None = None, wait_until: str | None = None) -> None:
        self.goto_calls.append((url, timeout, wait_until))

    def wait_for_selector(self, selector: str, timeout: float | None = None, state: str | None = None) -> None:
        self.wait_for_selector_calls.append((selector, timeout, state))
        if self._selector_raises:
            raise browser_mod.PlaywrightError("Timeout waiting for selector")

    def wait_for_timeout(self, ms: float) -> None:
        self.wait_for_timeout_calls.append(ms)

    def content(self) -> str:
        return self._content_text


@patch("trading_crab_lib.ingestion.browser._PLAYWRIGHT_AVAILABLE", True)
@patch("trading_crab_lib.ingestion.browser.sync_playwright")
def test_fetch_page_html_returns_content_and_closes_browser(mock_sync_pw):
    page = _FakeHtmlPage(content_text="<html>hello</html>")
    pw_cm, pw_obj, browser_obj, _context = _build_fake_playwright(page)
    mock_sync_pw.return_value = pw_cm

    html = browser_mod.fetch_page_html("https://example.com")

    assert html == "<html>hello</html>"
    pw_obj.chromium.launch.assert_called_once()
    browser_obj.close.assert_called_once()


@patch("trading_crab_lib.ingestion.browser._PLAYWRIGHT_AVAILABLE", True)
@patch("trading_crab_lib.ingestion.browser.sync_playwright")
def test_fetch_page_html_calls_wait_for_selector_before_content(mock_sync_pw):
    page = _FakeHtmlPage(content_text="<html>table here</html>")
    pw_cm, _pw, _browser, _context = _build_fake_playwright(page)
    mock_sync_pw.return_value = pw_cm

    html = browser_mod.fetch_page_html("https://example.com", wait_for_selector="table.x")

    assert html == "<html>table here</html>"
    # state="attached", not Playwright's default "visible": we need the element
    # in the DOM so the returned HTML carries it, not rendered on screen. A
    # visibility wait times out on an element hidden behind a UI toggle even
    # though it was in the DOM the whole time.
    assert page.wait_for_selector_calls == [("table.x", browser_mod._NAV_TIMEOUT_MS, "attached")]


@patch("trading_crab_lib.ingestion.browser._PLAYWRIGHT_AVAILABLE", True)
@patch("trading_crab_lib.ingestion.browser.sync_playwright")
def test_fetch_page_html_required_selector_missing_returns_none(mock_sync_pw, caplog):
    page = _FakeHtmlPage(content_text="<html>partial</html>", selector_raises=True)
    pw_cm, _pw, browser_obj, _context = _build_fake_playwright(page)
    mock_sync_pw.return_value = pw_cm

    with caplog.at_level(logging.WARNING):
        html = browser_mod.fetch_page_html("https://example.com", wait_for_selector="table.x")

    assert html is None
    browser_obj.close.assert_called_once()


@patch("trading_crab_lib.ingestion.browser._PLAYWRIGHT_AVAILABLE", True)
@patch("trading_crab_lib.ingestion.browser.sync_playwright")
def test_fetch_page_html_optional_selector_missing_returns_html_anyway(mock_sync_pw, caplog):
    page = _FakeHtmlPage(content_text="<html>partial but usable</html>", selector_raises=True)
    pw_cm, _pw, _browser, _context = _build_fake_playwright(page)
    mock_sync_pw.return_value = pw_cm

    with caplog.at_level(logging.WARNING):
        html = browser_mod.fetch_page_html(
            "https://example.com", wait_for_selector="table.x", require_selector=False
        )

    assert html == "<html>partial but usable</html>"
    assert any("never appeared" in rec.message for rec in caplog.records)


@patch("trading_crab_lib.ingestion.browser._SELENIUM_AVAILABLE", False)
@patch("trading_crab_lib.ingestion.browser._PLAYWRIGHT_AVAILABLE", False)
@patch("trading_crab_lib.ingestion.browser.sync_playwright")
def test_fetch_page_html_no_engine_available_returns_none(mock_sync_pw, caplog):
    """NO engine available means BOTH flags off. Patching only the Playwright
    flag let this fall through to Selenium and launch a real Chrome — it passed
    only because selenium happened to be uninstalled when it was written."""
    with caplog.at_level(logging.WARNING):
        html = browser_mod.fetch_page_html("https://example.com")

    assert html is None
    mock_sync_pw.assert_not_called()
    assert any("pip install" in rec.message for rec in caplog.records)


@patch("trading_crab_lib.ingestion.browser._PLAYWRIGHT_AVAILABLE", True)
@patch("trading_crab_lib.ingestion.browser.sync_playwright")
def test_fetch_page_html_launch_failure_returns_none(mock_sync_pw, caplog):
    pw_cm = MagicMock()
    pw_obj = MagicMock()
    pw_cm.__enter__.return_value = pw_obj
    pw_cm.__exit__.return_value = False
    pw_obj.chromium.launch.side_effect = RuntimeError("boom")
    mock_sync_pw.return_value = pw_cm

    with caplog.at_level(logging.WARNING):
        html = browser_mod.fetch_page_html("https://example.com")

    assert html is None


@patch("trading_crab_lib.ingestion.browser._PLAYWRIGHT_AVAILABLE", True)
@patch("trading_crab_lib.ingestion.browser.sync_playwright")
def test_fetch_page_html_sets_explicit_viewport(mock_sync_pw):
    page = _FakeHtmlPage()
    pw_cm, _pw, browser_obj, _context = _build_fake_playwright(page)
    mock_sync_pw.return_value = pw_cm

    browser_mod.fetch_page_html("https://example.com")

    _, kwargs = browser_obj.new_context.call_args
    assert kwargs.get("viewport") == {"width": 1280, "height": 800}


@patch("trading_crab_lib.ingestion.browser._PLAYWRIGHT_AVAILABLE", True)
@patch("trading_crab_lib.ingestion.browser.time.sleep")
@patch("trading_crab_lib.ingestion.browser.sync_playwright")
def test_fetch_urls_as_text_sets_explicit_viewport(mock_sync_pw, mock_sleep):
    urls = {"K": "https://example.com/k"}
    page = _FakePage({urls["K"]: "some text"})
    pw_cm, _pw, browser_obj, _context = _build_fake_playwright(page)
    mock_sync_pw.return_value = pw_cm

    browser_mod.fetch_urls_as_text(urls)

    _, kwargs = browser_obj.new_context.call_args
    assert kwargs.get("viewport") == {"width": 1280, "height": 800}


@patch("trading_crab_lib.ingestion.browser._PLAYWRIGHT_AVAILABLE", True)
@patch("trading_crab_lib.ingestion.browser.time.sleep")
@patch("trading_crab_lib.ingestion.browser.sync_playwright")
def test_fetch_urls_as_text_generic_path_attempts_every_url_without_early_bail(mock_sync_pw, mock_sleep):
    """Contrast with test_abandons_browser_path_after_consecutive_non_csv,
    which proves the Stooq wrapper opts IN to the early-bail via
    expected_prefix. The generic path (no expected_prefix) must not."""
    urls = {f"K{i}": f"https://example.com/{i}" for i in range(10)}
    unrecognised = "<!DOCTYPE html><html><head>not what we expect"
    page = _FakePage({u: unrecognised for u in urls.values()})
    pw_cm, _pw, _browser, _context = _build_fake_playwright(page)
    mock_sync_pw.return_value = pw_cm

    out = browser_mod.fetch_urls_as_text(urls)

    assert page.expect_download_count == len(urls)
    assert len(out) == len(urls)


@patch("trading_crab_lib.ingestion.browser._PLAYWRIGHT_AVAILABLE", True)
@patch("trading_crab_lib.ingestion.browser.time.sleep")
@patch("trading_crab_lib.ingestion.browser.sync_playwright")
def test_fetch_urls_as_text_skips_warmup_navigation_when_warmup_url_none(mock_sync_pw, mock_sleep):
    urls = {"K": "https://example.com/k"}
    page = _FakePage({urls["K"]: "some text"})
    pw_cm, _pw, _browser, _context = _build_fake_playwright(page)
    mock_sync_pw.return_value = pw_cm

    browser_mod.fetch_urls_as_text(urls, warmup_url=None)

    # Only the per-url navigation, no separate warm-up goto.
    assert page.goto_urls == [urls["K"]]


# ═══════════════════════════════════════════════════════════════════════════
# Task 2 additions: Selenium as a second engine
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture(autouse=True)
def _clear_browser_engine_env(monkeypatch):
    """Every test in this section that does not explicitly set these deletes
    them, so a stray export from another test cannot flip engine selection."""
    monkeypatch.delenv("TC_BROWSER_ENGINE", raising=False)
    monkeypatch.delenv("TC_CHROMIUM_PATH", raising=False)


class _FakeSeleniumTimeoutError(Exception):
    """Stand-in for selenium.common.exceptions.TimeoutException."""


class _FakeSeleniumOptions:
    """Fake selenium.webdriver.chrome.options.Options — records args."""

    def __init__(self) -> None:
        self.arguments: list[str] = []
        self.binary_location: str | None = None

    def add_argument(self, arg: str) -> None:
        self.arguments.append(arg)


class _FakeSeleniumDriver:
    """Fake selenium.webdriver.Chrome instance."""

    def __init__(
        self,
        page_source: str = "<html>selenium-rendered</html>",
        get_raises: Exception | None = None,
    ) -> None:
        self.page_source = page_source
        self.get_raises = get_raises
        self.get_urls: list[str] = []
        self.page_load_timeout: float | None = None
        self.quit_called = False

    def set_page_load_timeout(self, seconds: float) -> None:
        self.page_load_timeout = seconds

    def get(self, url: str) -> None:
        self.get_urls.append(url)
        if self.get_raises is not None:
            raise self.get_raises

    def quit(self) -> None:
        self.quit_called = True


def _patch_selenium(monkeypatch, driver, *, available: bool = True, selector_raises: bool = False):
    """Patch every selenium-facing name in browser_mod ENTIRELY — mirroring
    the module's own ImportError-branch binding so these names are patchable
    even though selenium is not installed in this environment."""
    fake_webdriver = MagicMock()
    fake_webdriver.Chrome.return_value = driver
    monkeypatch.setattr(browser_mod, "_SELENIUM_AVAILABLE", available)
    monkeypatch.setattr(browser_mod, "webdriver", fake_webdriver)
    monkeypatch.setattr(browser_mod, "ChromeOptions", _FakeSeleniumOptions)
    monkeypatch.setattr(browser_mod, "SeleniumTimeout", _FakeSeleniumTimeoutError)
    monkeypatch.setattr(browser_mod, "By", MagicMock())
    monkeypatch.setattr(browser_mod, "EC", MagicMock())

    fake_wait_instance = MagicMock()
    if selector_raises:
        fake_wait_instance.until.side_effect = _FakeSeleniumTimeoutError("selector timeout")
    fake_wait_cls = MagicMock(return_value=fake_wait_instance)
    monkeypatch.setattr(browser_mod, "WebDriverWait", fake_wait_cls)
    # The Selenium path's settle_ms uses time.sleep (unlike Playwright's
    # page.wait_for_timeout) — mock it so tests don't pay the real 2s delay.
    monkeypatch.setattr(browser_mod.time, "sleep", lambda *_a, **_k: None)
    return fake_webdriver, fake_wait_cls


# ── selenium_available() ─────────────────────────────────────────────────────


@patch("trading_crab_lib.ingestion.browser._SELENIUM_AVAILABLE", True)
def test_selenium_available_reflects_true_flag():
    assert browser_mod.selenium_available() is True


@patch("trading_crab_lib.ingestion.browser._SELENIUM_AVAILABLE", False)
def test_selenium_available_reflects_false_flag():
    assert browser_mod.selenium_available() is False


# ── engine selection precedence ──────────────────────────────────────────────


@patch("trading_crab_lib.ingestion.browser._PLAYWRIGHT_AVAILABLE", True)
@patch("trading_crab_lib.ingestion.browser.sync_playwright")
def test_fetch_page_html_auto_prefers_playwright_when_both_available(mock_sync_pw, monkeypatch):
    driver = _FakeSeleniumDriver()
    fake_webdriver, _wait_cls = _patch_selenium(monkeypatch, driver)

    page = _FakeHtmlPage(content_text="<html>via-playwright</html>")
    pw_cm, pw_obj, _browser, _context = _build_fake_playwright(page)
    mock_sync_pw.return_value = pw_cm

    html = browser_mod.fetch_page_html("https://example.com")

    assert html == "<html>via-playwright</html>"
    pw_obj.chromium.launch.assert_called_once()
    fake_webdriver.Chrome.assert_not_called()


@patch("trading_crab_lib.ingestion.browser._PLAYWRIGHT_AVAILABLE", False)
@patch("trading_crab_lib.ingestion.browser.sync_playwright")
def test_fetch_page_html_auto_falls_back_to_selenium_when_playwright_unavailable(mock_sync_pw, monkeypatch):
    driver = _FakeSeleniumDriver(page_source="<html>via-selenium</html>")
    fake_webdriver, _wait_cls = _patch_selenium(monkeypatch, driver)

    html = browser_mod.fetch_page_html("https://example.com")

    assert html == "<html>via-selenium</html>"
    mock_sync_pw.assert_not_called()
    fake_webdriver.Chrome.assert_called_once()


@patch("trading_crab_lib.ingestion.browser._PLAYWRIGHT_AVAILABLE", True)
@patch("trading_crab_lib.ingestion.browser.sync_playwright")
def test_fetch_page_html_env_selenium_overrides_auto(mock_sync_pw, monkeypatch):
    monkeypatch.setenv("TC_BROWSER_ENGINE", "selenium")
    driver = _FakeSeleniumDriver(page_source="<html>via-selenium</html>")
    _patch_selenium(monkeypatch, driver)

    html = browser_mod.fetch_page_html("https://example.com")

    assert html == "<html>via-selenium</html>"
    mock_sync_pw.assert_not_called()


@patch("trading_crab_lib.ingestion.browser._PLAYWRIGHT_AVAILABLE", True)
@patch("trading_crab_lib.ingestion.browser.sync_playwright")
def test_fetch_page_html_env_playwright_overrides_auto(mock_sync_pw, monkeypatch):
    monkeypatch.setenv("TC_BROWSER_ENGINE", "playwright")
    driver = _FakeSeleniumDriver()
    fake_webdriver, _wait_cls = _patch_selenium(monkeypatch, driver)

    page = _FakeHtmlPage(content_text="<html>via-playwright</html>")
    pw_cm, _pw, _browser, _context = _build_fake_playwright(page)
    mock_sync_pw.return_value = pw_cm

    html = browser_mod.fetch_page_html("https://example.com")

    assert html == "<html>via-playwright</html>"
    fake_webdriver.Chrome.assert_not_called()


@patch("trading_crab_lib.ingestion.browser._PLAYWRIGHT_AVAILABLE", False)
@patch("trading_crab_lib.ingestion.browser.sync_playwright")
def test_fetch_page_html_explicit_playwright_request_not_honored_does_not_fall_through(
    mock_sync_pw, monkeypatch, caplog
):
    """An explicit request for an engine that cannot be honored is a
    failure the operator must see — it must NOT silently substitute
    Selenium even though Selenium is available."""
    monkeypatch.setenv("TC_BROWSER_ENGINE", "playwright")
    driver = _FakeSeleniumDriver()
    fake_webdriver, _wait_cls = _patch_selenium(monkeypatch, driver)

    with caplog.at_level(logging.WARNING):
        html = browser_mod.fetch_page_html("https://example.com")

    assert html is None
    fake_webdriver.Chrome.assert_not_called()
    assert any("playwright" in rec.message for rec in caplog.records)


@patch("trading_crab_lib.ingestion.browser._PLAYWRIGHT_AVAILABLE", True)
@patch("trading_crab_lib.ingestion.browser.sync_playwright")
def test_fetch_page_html_explicit_engine_arg_beats_env_var(mock_sync_pw, monkeypatch):
    monkeypatch.setenv("TC_BROWSER_ENGINE", "playwright")
    driver = _FakeSeleniumDriver(page_source="<html>via-selenium</html>")
    _patch_selenium(monkeypatch, driver)

    html = browser_mod.fetch_page_html("https://example.com", engine="selenium")

    assert html == "<html>via-selenium</html>"
    mock_sync_pw.assert_not_called()


@patch("trading_crab_lib.ingestion.browser._PLAYWRIGHT_AVAILABLE", True)
@patch("trading_crab_lib.ingestion.browser.sync_playwright")
def test_fetch_page_html_unrecognised_env_value_warns_and_behaves_as_auto(mock_sync_pw, monkeypatch, caplog):
    monkeypatch.setenv("TC_BROWSER_ENGINE", "bogus-engine")
    page = _FakeHtmlPage(content_text="<html>via-playwright</html>")
    pw_cm, pw_obj, _browser, _context = _build_fake_playwright(page)
    mock_sync_pw.return_value = pw_cm

    with caplog.at_level(logging.WARNING):
        html = browser_mod.fetch_page_html("https://example.com")

    assert html == "<html>via-playwright</html>"
    pw_obj.chromium.launch.assert_called_once()
    assert any("Unrecognised" in rec.message for rec in caplog.records)


@patch("trading_crab_lib.ingestion.browser._PLAYWRIGHT_AVAILABLE", False)
@patch("trading_crab_lib.ingestion.browser.sync_playwright")
def test_fetch_page_html_neither_engine_available_names_both_install_paths(mock_sync_pw, monkeypatch, caplog):
    monkeypatch.setattr(browser_mod, "_SELENIUM_AVAILABLE", False)

    with caplog.at_level(logging.WARNING):
        html = browser_mod.fetch_page_html("https://example.com")

    assert html is None
    mock_sync_pw.assert_not_called()
    combined = " ".join(rec.message for rec in caplog.records)
    assert "playwright" in combined
    assert "selenium" in combined


# ── Selenium engine internals ────────────────────────────────────────────────


@patch("trading_crab_lib.ingestion.browser._PLAYWRIGHT_AVAILABLE", False)
def test_fetch_page_html_selenium_sets_hardened_options(monkeypatch):
    driver = _FakeSeleniumDriver()
    fake_webdriver, _wait_cls = _patch_selenium(monkeypatch, driver)

    browser_mod.fetch_page_html("https://example.com")

    _, kwargs = fake_webdriver.Chrome.call_args
    options = kwargs["options"]
    assert "--headless=new" in options.arguments
    assert "--disable-blink-features=AutomationControlled" in options.arguments
    assert "--window-size=1280,800" in options.arguments
    assert any(arg.startswith("--user-agent=") for arg in options.arguments)


@patch("trading_crab_lib.ingestion.browser._PLAYWRIGHT_AVAILABLE", False)
def test_fetch_page_html_selenium_waits_for_selector_via_webdriverwait(monkeypatch):
    driver = _FakeSeleniumDriver(page_source="<html>table here</html>")
    _fake_webdriver, fake_wait_cls = _patch_selenium(monkeypatch, driver)

    html = browser_mod.fetch_page_html("https://example.com", wait_for_selector="table.x")

    assert html == "<html>table here</html>"
    fake_wait_cls.assert_called_once()
    fake_wait_cls.return_value.until.assert_called_once()


@patch("trading_crab_lib.ingestion.browser._PLAYWRIGHT_AVAILABLE", False)
def test_fetch_page_html_selenium_required_selector_missing_returns_none(monkeypatch, caplog):
    driver = _FakeSeleniumDriver(page_source="<html>partial</html>")
    _patch_selenium(monkeypatch, driver, selector_raises=True)

    with caplog.at_level(logging.WARNING):
        html = browser_mod.fetch_page_html("https://example.com", wait_for_selector="table.x")

    assert html is None
    assert driver.quit_called is True


@patch("trading_crab_lib.ingestion.browser._PLAYWRIGHT_AVAILABLE", False)
def test_fetch_page_html_selenium_optional_selector_missing_returns_page_source(monkeypatch, caplog):
    driver = _FakeSeleniumDriver(page_source="<html>partial but usable</html>")
    _patch_selenium(monkeypatch, driver, selector_raises=True)

    with caplog.at_level(logging.WARNING):
        html = browser_mod.fetch_page_html(
            "https://example.com", wait_for_selector="table.x", require_selector=False
        )

    assert html == "<html>partial but usable</html>"
    assert any("never appeared" in rec.message for rec in caplog.records)


@patch("trading_crab_lib.ingestion.browser._PLAYWRIGHT_AVAILABLE", False)
def test_fetch_page_html_selenium_quits_driver_and_swallows_get_failure(monkeypatch, caplog):
    driver = _FakeSeleniumDriver(get_raises=RuntimeError("navigation exploded"))
    _patch_selenium(monkeypatch, driver)

    with caplog.at_level(logging.WARNING):
        html = browser_mod.fetch_page_html("https://example.com")

    assert html is None
    assert driver.quit_called is True


@patch("trading_crab_lib.ingestion.browser._PLAYWRIGHT_AVAILABLE", False)
def test_fetch_page_html_selenium_sets_binary_location_from_env(monkeypatch):
    monkeypatch.setenv("TC_CHROMIUM_PATH", "/custom/chrome-linux/chrome")
    driver = _FakeSeleniumDriver()
    fake_webdriver, _wait_cls = _patch_selenium(monkeypatch, driver)

    browser_mod.fetch_page_html("https://example.com")

    _, kwargs = fake_webdriver.Chrome.call_args
    options = kwargs["options"]
    assert options.binary_location == "/custom/chrome-linux/chrome"


# ── fetch_urls_as_text stays Playwright-only ─────────────────────────────────


@patch("trading_crab_lib.ingestion.browser._PLAYWRIGHT_AVAILABLE", False)
def test_fetch_urls_as_text_selenium_only_returns_empty_and_warns(monkeypatch, caplog):
    driver = _FakeSeleniumDriver()
    fake_webdriver, _wait_cls = _patch_selenium(monkeypatch, driver)

    with caplog.at_level(logging.WARNING):
        out = browser_mod.fetch_urls_as_text({"K": "https://example.com/k"})

    assert out == {}
    fake_webdriver.Chrome.assert_not_called()
    assert any("pip install" in rec.message for rec in caplog.records)


# ── headless signal hygiene (2026-08-05 escalation) ─────────────────────────


def test_headless_mode_defaults_true(monkeypatch):
    monkeypatch.delenv("TC_BROWSER_HEADLESS", raising=False)
    assert browser_mod.headless_mode() is True


@pytest.mark.parametrize("value", ["false", "False", "0", "no", "off", " FALSE "])
def test_headless_mode_false_only_when_explicitly_disabled(monkeypatch, value):
    monkeypatch.setenv("TC_BROWSER_HEADLESS", value)
    assert browser_mod.headless_mode() is False


@pytest.mark.parametrize("value", ["true", "1", "", "yes", "anything-else"])
def test_headless_mode_stays_true_for_anything_else(monkeypatch, value):
    monkeypatch.setenv("TC_BROWSER_HEADLESS", value)
    assert browser_mod.headless_mode() is True


@patch("trading_crab_lib.ingestion.browser._PLAYWRIGHT_AVAILABLE", True)
@patch("trading_crab_lib.ingestion.browser.sync_playwright")
def test_playwright_context_overrides_the_headless_user_agent(mock_sync_pw):
    """Headless Chromium announces itself as "HeadlessChrome" in its UA and
    client hints — observed leaking to third parties as
    m_ch_ua=..."HeadlessChrome"|v="151". The context must override it."""
    page = _FakePage({})
    pw_cm, _pw, browser_obj, _context = _build_fake_playwright(page)
    mock_sync_pw.return_value = pw_cm

    browser_mod.fetch_page_html("https://example.com")

    _, kwargs = browser_obj.new_context.call_args
    ua = kwargs.get("user_agent")
    assert ua, "no user_agent set on the Playwright context"
    assert "Headless" not in ua


@patch("trading_crab_lib.ingestion.browser._PLAYWRIGHT_AVAILABLE", True)
@patch("trading_crab_lib.ingestion.browser.time.sleep")
@patch("trading_crab_lib.ingestion.browser.sync_playwright")
def test_download_path_context_also_overrides_the_user_agent(mock_sync_pw, mock_sleep):
    """Both context construction sites, not just the HTML one."""
    urls = {"SPY": "https://stooq.com/q/d/l/?s=spy.us"}
    page = _FakePage({urls["SPY"]: "date,close\nspy-data"})
    pw_cm, _pw, browser_obj, _context = _build_fake_playwright(page)
    mock_sync_pw.return_value = pw_cm

    browser_mod.fetch_stooq_csvs(urls)

    _, kwargs = browser_obj.new_context.call_args
    assert "Headless" not in (kwargs.get("user_agent") or "")


@patch("trading_crab_lib.ingestion.browser._PLAYWRIGHT_AVAILABLE", True)
@patch("trading_crab_lib.ingestion.browser.sync_playwright")
def test_headless_env_toggle_reaches_chromium_launch(mock_sync_pw, monkeypatch):
    monkeypatch.setenv("TC_BROWSER_HEADLESS", "false")
    page = _FakePage({})
    pw_cm, pw_obj, _browser, _context = _build_fake_playwright(page)
    mock_sync_pw.return_value = pw_cm

    browser_mod.fetch_page_html("https://example.com")

    _, kwargs = pw_obj.chromium.launch.call_args
    assert kwargs.get("headless") is False
