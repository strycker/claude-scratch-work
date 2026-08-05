"""Unit tests for the headless-Chromium Stooq CSV fetcher
(trading_crab_lib.ingestion.browser).

Playwright is mocked ENTIRELY in every test — no browser launch, no network.
"""
from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

from trading_crab_lib.ingestion import browser as browser_mod


def _fake_response(text: str, ok: bool = True) -> MagicMock:
    resp = MagicMock()
    resp.ok = ok
    resp.text.return_value = text
    return resp


def _build_fake_playwright(get_side_effect):
    """Build a fake sync_playwright() call chain recording real control flow.

    Returns (pw_context_manager, pw_obj, browser_obj, context_obj, page_obj)
    so assertions can inspect launch kwargs, navigation calls, and per-ticker
    request calls made by the code under test.
    """
    pw_cm = MagicMock()
    pw_obj = MagicMock()
    pw_cm.__enter__.return_value = pw_obj
    pw_cm.__exit__.return_value = False

    browser_obj = MagicMock()
    pw_obj.chromium.launch.return_value = browser_obj

    context_obj = MagicMock()
    browser_obj.new_context.return_value = context_obj

    page_obj = MagicMock()
    context_obj.new_page.return_value = page_obj

    context_obj.request.get.side_effect = get_side_effect

    return pw_cm, pw_obj, browser_obj, context_obj, page_obj


# ── playwright_available() ──────────────────────────────────────────────────


@patch("trading_crab_lib.ingestion.browser._PLAYWRIGHT_AVAILABLE", True)
def test_playwright_available_reflects_true_flag():
    assert browser_mod.playwright_available() is True


@patch("trading_crab_lib.ingestion.browser._PLAYWRIGHT_AVAILABLE", False)
def test_playwright_available_reflects_false_flag():
    assert browser_mod.playwright_available() is False


# ── fetch_stooq_csvs: happy path ────────────────────────────────────────────


@patch("trading_crab_lib.ingestion.browser._PLAYWRIGHT_AVAILABLE", True)
@patch("trading_crab_lib.ingestion.browser.time.sleep")
@patch("trading_crab_lib.ingestion.browser.sync_playwright")
def test_fetch_stooq_csvs_one_browser_one_warmup_n_requests(mock_sync_pw, mock_sleep):
    urls = {
        "SPY": "https://stooq.com/q/d/l/?s=spy.us",
        "QQQ": "https://stooq.com/q/d/l/?s=qqq.us",
        "IWM": "https://stooq.com/q/d/l/?s=iwm.us",
    }
    responses_by_url = {
        urls["SPY"]: _fake_response("date,close\nspy-data"),
        urls["QQQ"]: _fake_response("date,close\nqqq-data"),
        urls["IWM"]: _fake_response("date,close\niwm-data"),
    }

    pw_cm, pw_obj, browser_obj, context_obj, page_obj = _build_fake_playwright(
        lambda url, timeout=None: responses_by_url[url]
    )
    mock_sync_pw.return_value = pw_cm

    out = browser_mod.fetch_stooq_csvs(urls)

    assert out == {"SPY": "date,close\nspy-data", "QQQ": "date,close\nqqq-data", "IWM": "date,close\niwm-data"}
    pw_obj.chromium.launch.assert_called_once()
    page_obj.goto.assert_called_once()
    assert context_obj.request.get.call_count == 3
    browser_obj.close.assert_called_once()
    assert mock_sleep.call_count >= 2


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

    def get_side_effect(url, timeout=None):
        if url == urls["SPY"]:
            return _fake_response("date,close\nspy-data")
        if url == urls["QQQ"]:
            raise RuntimeError("boom mid-batch")
        return _fake_response("date,close\niwm-data")

    pw_cm, pw_obj, browser_obj, context_obj, page_obj = _build_fake_playwright(get_side_effect)
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

    pw_cm, pw_obj, browser_obj, context_obj, page_obj = _build_fake_playwright(
        lambda url, timeout=None: _fake_response("date,close\nspy-data")
    )
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

    pw_cm, pw_obj, browser_obj, context_obj, page_obj = _build_fake_playwright(
        lambda url, timeout=None: _fake_response("date,close\nspy-data")
    )
    mock_sync_pw.return_value = pw_cm

    browser_mod.fetch_stooq_csvs(urls)

    _, kwargs = pw_obj.chromium.launch.call_args
    assert "executable_path" not in kwargs
