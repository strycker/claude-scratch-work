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
