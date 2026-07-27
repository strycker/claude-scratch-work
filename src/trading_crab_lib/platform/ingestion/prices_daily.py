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
from datetime import date
from typing import Any

import pandas as pd

from trading_crab_lib.ingestion.assets import _ssl_bypass_curl_session

try:
    import yfinance as yf  # type: ignore[import]
    _YFINANCE_AVAILABLE = True
except ImportError:
    yf = None  # type: ignore[assignment]
    _YFINANCE_AVAILABLE = False

log = logging.getLogger(__name__)


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

def _batch_yfinance_daily(
    tickers: list[str], start: str, end: str, session: Any = None
) -> dict[str, pd.Series]:
    """Fetch all tickers in ONE yf.download() call at daily interval.

    Mirrors ``assets._batch_yfinance`` but requests interval="1d" and performs
    NO internal resample — daily data is preserved for D-05 persistence.
    Returns a dict of ticker -> daily Close Series (only successfully fetched
    tickers).
    """
    if not _YFINANCE_AVAILABLE:
        log.warning(
            "yfinance is not installed — daily batch download skipped. "
            "Run: pip install yfinance"
        )
        return {}

    log.info("Batch-fetching %d tickers (daily) from yfinance ...", len(tickers))

    kwargs: dict[str, Any] = {
        "tickers": tickers,
        "start": start,
        "end": end,
        "interval": "1d",
        "auto_adjust": True,
        "progress": False,
    }
    if session is not None:
        kwargs["session"] = session

    try:
        raw = yf.download(**kwargs)
    except (OSError, RuntimeError, ValueError, TypeError) as exc:
        log.warning("Batch yfinance daily download failed: %s", exc)
        return {}

    if raw is None or raw.empty:
        return {}

    # Multiple tickers -> MultiIndex columns (level-0=metric, level-1=ticker).
    # Single ticker -> flat columns.
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


# ── Stooq daily fallback (no API key; used when yfinance yields nothing) ─────

def _batch_stooq_daily(
    tickers: list[str], start: str, end: str
) -> dict[str, pd.Series]:
    """Fallback daily fetch from Stooq's per-ticker CSV endpoint.

    Fetches ``https://stooq.com/q/d/l/?s=<sym>.us&i=d`` for each ticker and
    returns a dict of ticker -> daily Close Series (successfully fetched only).
    No API key required.

    Stooq serves a JavaScript anti-bot *challenge page* (not CSV) to some
    datacenter / VPN IPs. A real CSV response begins with the ``Date,`` header;
    anything else (an HTML challenge, an empty body, an error) is detected and
    skipped rather than parsed as data — so this degrades cleanly to ``{}``
    when Stooq is unreachable or challenges the caller.
    """
    try:
        import requests  # noqa: PLC0415 — optional-at-call-time network dep
    except ImportError:
        log.warning("requests is not installed — Stooq daily fallback skipped.")
        return {}

    d1 = pd.Timestamp(start).strftime("%Y%m%d")
    d2 = pd.Timestamp(end).strftime("%Y%m%d")
    headers = {"User-Agent": "Mozilla/5.0 (compatible; trading-crab/1.0)"}

    log.info("Stooq daily fallback: fetching %d tickers ...", len(tickers))
    results: dict[str, pd.Series] = {}
    for ticker in tickers:
        sym = f"{ticker.lower().replace('.', '-')}.us"
        url = f"https://stooq.com/q/d/l/?s={sym}&i=d&d1={d1}&d2={d2}"
        try:
            resp = requests.get(url, headers=headers, timeout=30)
            text = resp.text
        except requests.RequestException as exc:  # noqa: BLE001 — network degradation
            log.warning("Stooq fetch failed for %s: %s", ticker, exc)
            continue

        # A genuine CSV starts with the 'Date,' header. Anything else is Stooq's
        # JS challenge page, an empty body, or 'No data' — skip, don't parse.
        if not text.lstrip().lower().startswith("date,"):
            log.warning("Stooq returned non-CSV for %s (anti-bot challenge or no data) — skipping", ticker)
            continue

        try:
            from io import StringIO

            df = pd.read_csv(StringIO(text))
        except (ValueError, pd.errors.ParserError) as exc:
            log.warning("Stooq CSV parse failed for %s: %s", ticker, exc)
            continue

        if df.empty or "Close" not in df.columns or "Date" not in df.columns:
            continue
        s = pd.Series(
            df["Close"].to_numpy(),
            index=pd.to_datetime(df["Date"]),
            name=ticker,
        ).dropna()
        if not s.empty:
            results[ticker] = s

    log.info("Stooq daily fallback: recovered %d/%d tickers", len(results), len(tickers))
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
