"""
Tiingo daily-price adapter — live REST integration, first link in the
daily-price fallback chain (DATA-06).

Every unauthenticated, bot-gated source reachable from this platform's
operator network is currently blocked: Yahoo Finance is TLS-intercepted and
answers with a middlebox 429, Stooq serves a JavaScript verification
challenge to both plain HTTP and headless Chromium, and macrotrends sits
behind Cloudflare. A keyed REST API is the reliable path, so Tiingo is wired
ahead of yfinance and both Stooq paths in
``platform.ingestion.prices_daily.fetch_universe_prices`` — see that module
for the fallback chain, and ``docs/paid_provider_seams.md`` for the fuller
provider write-up and the not-yet-live-verified caveat.

Tiingo's free tier carries hourly/daily request caps, so requests are paced
between tickers and a 429 response is retried with bounded exponential
backoff before the ticker is given up on. The API key travels ONLY in an
``Authorization`` request header (never in the URL — a URL-embedded
credential leaks into proxy access logs and exception messages far more
easily than a header does) and is redacted from every log record, including
ones derived from exception messages, via :func:`_redact`.

Usage:
    from trading_crab_lib.platform.ingestion.tiingo import fetch_daily_prices

    prices = fetch_daily_prices(["SPY", "QQQ"], "2024-01-01", "2024-02-01")
"""

from __future__ import annotations

import logging
import os
import time
from datetime import date
from typing import Any
from urllib.parse import urlencode

import pandas as pd

from trading_crab_lib.ingestion.http import HTTP_ERRORS, http_get, plain_session

log = logging.getLogger(__name__)

# Allowlisted provider host — not an arbitrary caller-supplied URL (SSRF
# hygiene, T-04-01 in the phase threat register).
_DEFAULT_BASE_URL = "https://api.tiingo.com/tiingo/daily"
_API_KEY_ENV = "TIINGO_API_KEY"

# Free-tier hourly/daily request caps are real — pace the batch and bound
# the 429 backoff rather than hammering the endpoint.
_RATE_LIMIT_SECONDS = 1.0
_MAX_RETRIES = 3
_BACKOFF_BASE_SECONDS = 2.0
_RATE_LIMITED_STATUS = 429

_REDACTED = "***REDACTED***"


def _redact(text: str, api_key: str | None) -> str:
    """Scrub every occurrence of *api_key* from *text*.

    This is the single mechanism keeping the credential out of logs. The
    transport library (curl_cffi / requests) can echo request detail —
    including the Authorization header — into an exception's message, which
    is precisely why scrubbing is centralized here rather than relied upon
    per-call-site. Returns *text* unchanged when *api_key* is falsy.
    """
    if not api_key:
        return text
    return text.replace(api_key, _REDACTED)


def resolve_api_key(cfg: dict[str, Any] | None = None, api_key: str | None = None) -> str | None:
    """Resolve the Tiingo API key.

    Precedence: an explicit *api_key* argument, then the ``TIINGO_API_KEY``
    environment variable, then ``cfg["tiingo"]["api_key"]``. Whitespace-only
    values are treated as unset. Returns None when nothing is configured.
    Never logs the resolved value.
    """
    if api_key and api_key.strip():
        return api_key.strip()

    env_key = os.environ.get(_API_KEY_ENV)
    if env_key and env_key.strip():
        return env_key.strip()

    if cfg:
        cfg_key = cfg.get("tiingo", {}).get("api_key")
        if cfg_key and str(cfg_key).strip():
            return str(cfg_key).strip()

    return None


def _parse_rows(ticker: str, rows: list[dict[str, Any]]) -> pd.Series | None:
    """Parse a Tiingo EOD JSON row list into a tz-naive daily Series.

    Prefers ``adjClose``; falls back to ``close`` only when ``adjClose`` is
    absent or None, logging one DEBUG record per ticker on the FIRST such
    fallback (not one per row). The adjusted series matches the
    ``auto_adjust=True`` semantics the yfinance path already produces, which
    is what keeps the two sources interchangeable downstream — mixing
    adjusted and unadjusted prices would silently corrupt returns at every
    dividend and split.

    Builds a tz-naive, daily-granularity index with NO resampling (D-05:
    daily is what gets persisted). Returns None (never an empty Series) when
    nothing parsed, so callers have one falsy check.
    """
    if not isinstance(rows, list):
        return None

    dates: list[Any] = []
    values: list[float] = []
    fallback_logged = False

    for row in rows:
        if not isinstance(row, dict):
            continue
        row_date = row.get("date")
        if row_date is None:
            continue

        adj_close = row.get("adjClose")
        if adj_close is not None:
            value = adj_close
        else:
            close = row.get("close")
            if close is None:
                continue
            value = close
            if not fallback_logged:
                log.debug("Tiingo: %s row missing adjClose — falling back to close", ticker)
                fallback_logged = True

        dates.append(row_date)
        values.append(value)

    if not dates:
        return None

    index = pd.to_datetime(dates, utc=True).tz_localize(None).normalize()
    series = pd.Series(values, index=index, name=ticker, dtype=float)
    series = series.sort_index()
    series = series[~series.index.duplicated(keep="last")]
    series = series.dropna()
    return series if not series.empty else None


def _fetch_one(
    ticker: str,
    start: str,
    end: str,
    api_key: str,
    session: Any,
    base_url: str,
    *,
    max_retries: int = _MAX_RETRIES,
    backoff_base_seconds: float = _BACKOFF_BASE_SECONDS,
) -> pd.Series | None:
    """Fetch one ticker's EOD history, retrying a 429 with bounded backoff.

    The credential is supplied ONLY as an ``Authorization: Token <key>``
    header — never as a query parameter — and the date range is the only
    thing in the URL.
    """
    query = urlencode({"startDate": start, "endDate": end})
    url = f"{base_url}/{ticker}/prices?{query}"
    headers = {"Authorization": f"Token {api_key}"}

    for attempt in range(max_retries):
        try:
            resp = http_get(url, session=session, headers=headers, timeout=30)
        except HTTP_ERRORS as exc:  # noqa: BLE001 — network degradation, never abort the batch
            log.warning("Tiingo fetch failed for %s: %s", ticker, _redact(str(exc), api_key))
            return None

        status = resp.status_code
        if status == _RATE_LIMITED_STATUS:
            log.warning(
                "Tiingo rate-limited for %s (attempt %d/%d) — backing off",
                ticker, attempt + 1, max_retries,
            )
            if attempt < max_retries - 1:
                time.sleep(backoff_base_seconds * (2 ** attempt))
            continue

        if status != 200:
            log.warning("Tiingo returned status %d for %s — skipping, no retry", status, ticker)
            return None

        try:
            payload = resp.json()
        except (ValueError, TypeError) as exc:
            log.warning("Tiingo response JSON parse failed for %s: %s", ticker, _redact(str(exc), api_key))
            return None

        return _parse_rows(ticker, payload)

    log.warning("Tiingo retry budget exhausted for %s (%d attempts) — giving up", ticker, max_retries)
    return None


def fetch_daily_prices(
    tickers: list[str],
    start: str,
    end: str,
    *,
    api_key: str | None = None,
    cfg: dict[str, Any] | None = None,
) -> dict[str, pd.Series]:
    """Fetch daily adjusted-close prices for *tickers* from Tiingo.

    Returns a dict of ticker -> daily Series (only successfully fetched
    tickers). One failing ticker never aborts the batch — mirrors
    ``macro_monthly.fetch_fred_monthly``'s per-series graceful degradation
    exactly. Never raises.

    A missing API key is a configuration state, not a fault: it is logged at
    INFO (not WARNING), an empty dict is returned immediately, and no HTTP
    session is constructed and no request is issued.
    """
    resolved_key = resolve_api_key(cfg, api_key)
    if resolved_key is None:
        log.info(
            "Tiingo is skipped — no API key configured (set the %s environment "
            "variable to enable this source)",
            _API_KEY_ENV,
        )
        return {}

    tiingo_cfg = (cfg or {}).get("tiingo", {})
    base_url = tiingo_cfg.get("base_url", _DEFAULT_BASE_URL)
    rate_limit_seconds = tiingo_cfg.get("rate_limit_seconds", _RATE_LIMIT_SECONDS)
    max_retries = tiingo_cfg.get("max_retries", _MAX_RETRIES)

    # Plain requests, NOT the impersonating client. Tiingo is a keyed REST API
    # with no bot check, so impersonation defeats nothing and only adds
    # failure modes: curl_cffi carries its own CA store and its own transport,
    # and has been observed failing where plain requests succeeded (a proxy
    # reset it outright against this very endpoint while requests got HTTP 200).
    session = plain_session()
    if session is None:
        log.warning("No HTTP client available — Tiingo fetch skipped.")
        return {}

    log.info("Tiingo: fetching %d tickers from %s to %s ...", len(tickers), start, end)

    results: dict[str, pd.Series] = {}
    for i, ticker in enumerate(tickers):
        if i > 0:
            time.sleep(rate_limit_seconds)
        series = _fetch_one(ticker, start, end, resolved_key, session, base_url, max_retries=max_retries)
        if series is not None:
            results[ticker] = series

    log.info("Tiingo daily batch recovered %d/%d tickers", len(results), len(tickers))
    return results


def fetch_prices(cfg: dict[str, Any]) -> pd.DataFrame:
    """Config-driven entry point: fetch the platform's tradable universe from Tiingo.

    Resolves the ticker set via
    ``platform.ingestion.prices_daily.universe_fetch_tickers`` (imported
    inside the function body — a deferred import avoids a module-level
    cycle, because ``prices_daily`` imports this module's
    :func:`fetch_daily_prices` at module level). Returns an empty DataFrame
    when there are no tickers or nothing was recovered. Never raises on an
    empty or minimal *cfg*.
    """
    from trading_crab_lib.platform.ingestion.prices_daily import universe_fetch_tickers

    tickers = universe_fetch_tickers(cfg)
    if not tickers:
        return pd.DataFrame()

    data_cfg = cfg.get("data", {})
    start = data_cfg.get("start_date", "1962-01-01")
    end = data_cfg.get("end_date") or str(date.today())

    results = fetch_daily_prices(tickers, start, end, cfg=cfg)
    if not results:
        return pd.DataFrame()

    # Outer, NULL-tolerant join — a short-history ticker becomes a
    # NaN-padded column rather than dropping rows.
    df = pd.concat(results.values(), axis=1)
    df.index.name = "date"
    return df
