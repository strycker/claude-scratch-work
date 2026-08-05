"""
Daily universe price ingestion — tradable-universe satellites, Glenn's holdings,
and watchlist (DATA-05, D-05).

Unlike the incumbent's ``ingestion/assets.py`` (which resamples to *quarterly*
for the frozen 9-step pipeline), this module persists DAILY adjusted-close
prices and derives a MONTHLY spine separately. Daily persistence is deliberate
— the Phase 4 tripwire consumes daily granularity.

Reuses ``assets._ssl_bypass_curl_session`` (a pure importable session factory)
for the SSL-bypass Yahoo Finance workaround, but does NOT reuse
``assets._batch_yfinance`` — that helper hardcodes ``interval="1mo"`` and an
internal ``.resample("QE")`` which would defeat daily persistence.

Fetch ticker set = union(satellites, holdings, watchlist) minus
``no_price_ingest`` (money-market funds, D-12 — NAV≈$1, mapped to the cash
class instead of price-ingested).

Usage:
    from trading_crab_lib.platform.ingestion.prices_daily import fetch_universe_prices

    daily, monthly = fetch_universe_prices(cfg)
"""

from __future__ import annotations

import logging
import time
from datetime import date
from io import StringIO
from typing import Any

import pandas as pd

from trading_crab_lib.ingestion.assets import _ssl_bypass_curl_session
from trading_crab_lib.ingestion.browser import fetch_stooq_csvs, playwright_available
from trading_crab_lib.ingestion.http import HTTP_ERRORS, browser_session, http_get

try:
    import yfinance as yf  # type: ignore[import]
    _YFINANCE_AVAILABLE = True
except ImportError:
    yf = None  # type: ignore[assignment]
    _YFINANCE_AVAILABLE = False

log = logging.getLogger(__name__)

# 22 rapid-fire per-ticker requests is itself a bot signal — pace Stooq fetches.
_STOOQ_RATE_LIMIT_SECONDS = 1.5

# yfinance chunking + backoff. Defined locally (not imported from ingestion/assets.py's
# private constant) — a single oversized yf.download() call for the whole universe is
# what triggers Yahoo's rate limit and discards everything on failure.
_YF_CHUNK_SIZE = 5
_YF_MAX_RETRIES = 3
_YF_BACKOFF_BASE_SECONDS = 2.0
_YF_CHUNK_SLEEP_SECONDS = 1.0
_RATE_LIMIT_SIGNATURES = (
    "Too Many Requests",
    "Rate limit",
    "rate limited",
    "429",
)


# ── universe ticker set ─────────────────────────────────────────────────────

def universe_fetch_tickers(cfg: dict[str, Any]) -> list[str]:
    """Return the tradable-universe ticker set for price ingestion.

    union(satellites, holdings, watchlist) minus no_price_ingest — preserves
    Glenn's holdings (D-10) and excludes money-market funds FZFXX/SPAXX (D-12).
    """
    universe = cfg.get("universe", {})
    satellites = universe.get("satellites", [])
    holdings = universe.get("holdings", [])
    watchlist = universe.get("watchlist", [])
    no_price_ingest = set(universe.get("no_price_ingest", []))

    tickers = set(satellites) | set(holdings) | set(watchlist)
    tickers -= no_price_ingest
    return sorted(tickers)


# ── daily batch fetch (no internal resample — daily preserved) ─────────────

def _extract_close_series(raw: pd.DataFrame, tickers: list[str]) -> dict[str, pd.Series]:
    """Extract per-ticker Close Series from one yf.download() result frame.

    Handles both column shapes yfinance returns: a MultiIndex (level-0=metric,
    level-1=ticker) for a multi-ticker download, or flat columns for a
    single-ticker download. No resample — daily data is preserved for D-05
    persistence. Returns a dict of ticker -> daily Close Series (only
    successfully fetched, non-all-NaN tickers).
    """
    if isinstance(raw.columns, pd.MultiIndex):
        levels = raw.columns.get_level_values(0)
        if "Close" not in levels:
            log.warning("'Close' missing from batch download MultiIndex: %s", levels[:6].tolist())
            return {}
        close_df = raw["Close"]
    else:
        if "Close" not in raw.columns:
            log.warning("'Close' missing from single-ticker download columns: %s", list(raw.columns))
            return {}
        t = tickers[0] if len(tickers) == 1 else "Close"
        close_df = raw[["Close"]].rename(columns={"Close": t})

    results: dict[str, pd.Series] = {}
    for ticker in tickers:
        if ticker not in close_df.columns:
            log.debug("Ticker %s absent from batch results", ticker)
            continue
        s = close_df[ticker].dropna()
        if s.empty:
            log.warning("All-NaN Close data for %s in daily batch download", ticker)
            continue
        s = s.copy()
        s.name = ticker
        s.index = pd.to_datetime(s.index)
        if hasattr(s.index, "tz") and s.index.tz is not None:
            s.index = s.index.tz_localize(None)
        results[ticker] = s  # no resample — daily preserved

    return results


def _download_yfinance_chunk(
    chunk: list[str], start: str, end: str, session: Any = None
) -> dict[str, pd.Series]:
    """Download one chunk of tickers, retrying on rate-limit/transport errors
    or an empty result with exponential backoff.

    Treats both a raised transport/rate-limit error and an empty/None result
    as retryable. Returns {} only after exhausting ``_YF_MAX_RETRIES``
    attempts — the caller is responsible for keeping any results already
    accumulated from other chunks (partial results are never discarded).
    """
    kwargs: dict[str, Any] = {
        "tickers": chunk,
        "start": start,
        "end": end,
        "interval": "1d",
        "auto_adjust": True,
        "progress": False,
    }
    if session is not None:
        kwargs["session"] = session

    for attempt in range(_YF_MAX_RETRIES):
        raw = None
        try:
            raw = yf.download(**kwargs)
        except (OSError, RuntimeError, ValueError, TypeError) as exc:  # noqa: BLE001 — network degradation
            is_rate_limited = any(sig in str(exc) for sig in _RATE_LIMIT_SIGNATURES)
            log.warning(
                "yfinance chunk download failed for %s (attempt %d/%d)%s: %s",
                chunk, attempt + 1, _YF_MAX_RETRIES,
                " [rate-limited]" if is_rate_limited else "", exc,
            )

        if raw is not None and not raw.empty:
            return _extract_close_series(raw, chunk)

        if attempt < _YF_MAX_RETRIES - 1:
            sleep_seconds = _YF_BACKOFF_BASE_SECONDS * (2 ** attempt)
            time.sleep(sleep_seconds)

    log.warning("Exhausted %d retries for yfinance chunk %s — skipping, keeping other chunks' results", _YF_MAX_RETRIES, chunk)
    return {}


def _batch_yfinance_daily(
    tickers: list[str], start: str, end: str, session: Any = None
) -> dict[str, pd.Series]:
    """Fetch all tickers at daily interval, chunked with retry + backoff.

    Mirrors ``assets._batch_yfinance`` but requests interval="1d" and performs
    NO internal resample — daily data is preserved for D-05 persistence.

    Splits the ticker universe into chunks of ``_YF_CHUNK_SIZE`` (a single
    oversized ``yf.download()`` call for the whole universe is a common
    trigger for Yahoo's rate limit). Each chunk is retried up to
    ``_YF_MAX_RETRIES`` times with exponential backoff; a chunk that
    exhausts its retries is skipped rather than discarding results already
    recovered from other chunks. Returns a dict of ticker -> daily Close
    Series (only successfully fetched tickers).
    """
    if not _YFINANCE_AVAILABLE:
        log.warning(
            "yfinance is not installed — daily batch download skipped. "
            "Run: pip install yfinance"
        )
        return {}

    log.info(
        "Batch-fetching %d tickers (daily) from yfinance in chunks of %d ...",
        len(tickers), _YF_CHUNK_SIZE,
    )

    results: dict[str, pd.Series] = {}
    chunks = [tickers[i:i + _YF_CHUNK_SIZE] for i in range(0, len(tickers), _YF_CHUNK_SIZE)]
    for i, chunk in enumerate(chunks):
        if i > 0:
            time.sleep(_YF_CHUNK_SLEEP_SECONDS)
        results.update(_download_yfinance_chunk(chunk, start, end, session=session))

    log.info("yfinance daily batch recovered %d/%d tickers", len(results), len(tickers))
    return results


# ── Stooq daily fallback (no API key; used when yfinance yields nothing) ─────

def _parse_stooq_csv(ticker: str, text: str) -> pd.Series | None:
    """Parse one Stooq CSV response body into a named daily Close Series.

    The single guard standing between a challenge page and it being parsed
    as price data: a genuine CSV starts with the ``date,`` header. Anything
    else (an HTML challenge, an empty body, "No data") is rejected here and
    ONLY here — both the plain-HTTP path and the headless-browser fallback
    route their response text through this one function, so the guard can
    never drift out of sync between the two paths.

    Returns None (never an empty Series) on any rejection, so callers have
    exactly one falsy check. Every rejection path logs a WARNING.
    """
    if not text.lstrip().lower().startswith("date,"):
        log.warning("Stooq returned non-CSV for %s (anti-bot challenge or no data) — skipping", ticker)
        return None

    try:
        df = pd.read_csv(StringIO(text))
    except (ValueError, pd.errors.ParserError) as exc:
        log.warning("Stooq CSV parse failed for %s: %s", ticker, exc)
        return None

    if df.empty or "Close" not in df.columns or "Date" not in df.columns:
        return None

    s = pd.Series(
        df["Close"].to_numpy(),
        index=pd.to_datetime(df["Date"]),
        name=ticker,
    ).dropna()
    return s if not s.empty else None


def _batch_stooq_daily(
    tickers: list[str], start: str, end: str
) -> dict[str, pd.Series]:
    """Fallback daily fetch from Stooq's per-ticker CSV endpoint.

    Fetches ``https://stooq.com/q/d/l/?s=<sym>.us&i=d`` for each ticker and
    returns a dict of ticker -> daily Close Series (successfully fetched only).
    No API key required.

    Two paths, in order:

    1. Plain HTTP through the shared browser-impersonating client
       (``ingestion.http``) — cheap, correct in spirit, and the one that
       stops being needed the day Stooq drops its challenge. Tried first,
       always.
    2. Headless-Chromium fallback (``ingestion.browser``), gated on the HTTP
       path recovering ZERO tickers AND playwright being installed. Stooq
       serves a JavaScript browser-verification *challenge page* (not CSV)
       to plain ``requests``/curl_cffi clients — the challenge must be
       EXECUTED, and only a real browser can do that. This path costs a
       browser launch per run, so it is a fallback, not the default.

    Both paths route their response text through the single
    :func:`_parse_stooq_csv` guard — a challenge page cannot become price
    data via either path. This function never raises; total failure
    degrades to ``{}``.
    """
    urls: dict[str, str] = {}
    d1 = pd.Timestamp(start).strftime("%Y%m%d")
    d2 = pd.Timestamp(end).strftime("%Y%m%d")
    for ticker in tickers:
        sym = f"{ticker.lower().replace('.', '-')}.us"
        urls[ticker] = f"https://stooq.com/q/d/l/?s={sym}&i=d&d1={d1}&d2={d2}"

    results: dict[str, pd.Series] = {}
    recovered_via = "HTTP"

    session = browser_session()
    if session is None:
        log.warning("No HTTP client available — Stooq HTTP path skipped.")
    else:
        log.info("Stooq daily fallback: fetching %d tickers via HTTP ...", len(tickers))
        for i, ticker in enumerate(tickers):
            if i > 0:
                time.sleep(_STOOQ_RATE_LIMIT_SECONDS)
            url = urls[ticker]
            try:
                # No headers= on purpose — a self-identifying "trading-crab/1.0" UA
                # used to override the impersonating client's Chrome header set,
                # advertising the scraper to the very bot check we are evading.
                resp = http_get(url, session=session, timeout=30)
                text = resp.text
            except HTTP_ERRORS as exc:  # noqa: BLE001 — network degradation
                log.warning("Stooq fetch failed for %s: %s", ticker, exc)
                continue

            s = _parse_stooq_csv(ticker, text)
            if s is not None:
                results[ticker] = s

    # Fallback, gated on BOTH conditions: the HTTP path recovered nothing AND
    # the headless-browser challenge solver is actually available. The browser
    # is never launched while the cheap HTTP path still works.
    if not results and playwright_available():
        recovered_via = "headless-browser fallback"
        log.warning(
            "Stooq HTTP path recovered 0/%d tickers — trying headless-browser challenge solver ...",
            len(tickers),
        )
        try:
            csvs = fetch_stooq_csvs(urls)
        except Exception as exc:  # noqa: BLE001 — never propagate a browser-path failure
            log.warning("Stooq headless-browser fallback raised: %s", exc)
            csvs = {}
        for ticker, text in csvs.items():
            s = _parse_stooq_csv(ticker, text)
            if s is not None:
                results[ticker] = s

    log.info(
        "Stooq daily fallback (%s): recovered %d/%d tickers",
        recovered_via if results else "no path succeeded",
        len(results), len(tickers),
    )
    return results


# ── monthly spine derivation ────────────────────────────────────────────────

def to_monthly_spine(daily_df: pd.DataFrame, monthly_freq: str = "ME") -> pd.DataFrame:
    """Derive the monthly spine from daily prices via month-end `.last()`.

    Daily itself is persisted separately by the caller (D-05) — this function
    does not discard it.
    """
    if daily_df.empty:
        return daily_df
    return daily_df.resample(monthly_freq).last()


# ── public API ───────────────────────────────────────────────────────────────

def fetch_universe_prices(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch daily universe prices and derive a monthly spine.

    Batch-fetches daily adjusted-close prices for the tradable universe
    (satellites, holdings, watchlist, minus no_price_ingest), merges every
    ticker's Series with ``pd.concat([...], axis=1)`` (outer join — short
    histories become NaN-padded columns, never dropped rows), and derives the
    monthly spine via :func:`to_monthly_spine`.

    On total fetch failure, degrades gracefully to an empty/NaN frame with a
    WARNING (matching ``assets.py``'s graceful terminal fallback) — does NOT
    rebuild the full Stooq/OpenBB daily chain (documented future fallback).

    Returns:
        (daily, monthly) tuple of DataFrames indexed by date.
    """
    tickers = universe_fetch_tickers(cfg)
    if not tickers:
        log.warning("No universe tickers configured — skipping price fetch")
        empty = pd.DataFrame()
        return empty, empty

    start = cfg["data"]["start_date"]
    end = cfg["data"]["end_date"] or str(date.today())
    monthly_freq = cfg["data"].get("monthly_freq", "ME")

    log.info(
        "Fetching daily universe prices for %d tickers from %s to %s: %s",
        len(tickers), start, end, ", ".join(tickers),
    )

    session = _ssl_bypass_curl_session()
    results = _batch_yfinance_daily(tickers, start, end, session=session)

    if not results:
        log.warning("yfinance universe fetch returned nothing (rate-limit/block?) — trying Stooq daily fallback ...")
        results = _batch_stooq_daily(tickers, start, end)

    if not results:
        log.warning(
            "All universe price fetches failed (yfinance + Stooq) — degrading to empty/NaN frame. "
            "Both sources are commonly blocked on datacenter / VPN IPs; retry from a residential "
            "connection or wait out the yfinance rate-limit."
        )
        empty = pd.DataFrame()
        return empty, empty

    # NULL-tolerant merge: outer concat, never drop rows for short-history tickers.
    daily = pd.concat(results.values(), axis=1)
    daily.index.name = "date"

    monthly = to_monthly_spine(daily, monthly_freq)

    log.info(
        "Universe daily prices fetched: %d days, %d/%d tickers, %.1f%% coverage",
        len(daily), len(daily.columns), len(tickers),
        100 * daily.notna().mean().mean(),
    )
    return daily, monthly
