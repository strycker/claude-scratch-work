"""
ETF / equity price ingestion — yfinance with multi-source fallback chain.

Downloads monthly adjusted-close prices for the configured asset tickers,
resamples to quarterly (period-end), and returns a wide DataFrame indexed
by quarter-end dates — ready to join against cluster labels.

Fallback chain (tried in order, stopping at first success)
----------------------------------------------------------
1. yfinance batch (curl_cffi)         — one HTTP request for all tickers
2. yfinance per-ticker, SSL bypass    — curl_cffi Session(verify=False)
3. stooq via pandas-datareader        — free, no API key, same data
4. OpenBB (cboe provider, then default) — optional install, multiple backends
5. Empty DataFrame                    — triggers macro-proxy fallback in
                                        asset_returns.compute_proxy_returns()

Install optional fallback libraries
-------------------------------------
    pip install pandas-datareader     # enables Phase 3 (stooq)
    pip install openbb                # enables Phase 4 (OpenBB)
    # or together:
    pip install "market-regime[data-extras]"

Notes on sources that are NOT suitable for historical ETF prices
----------------------------------------------------------------
- Finviz / Finviz Elite: stock screener only — no historical OHLCV data.
- StockCharts.com: chart-rendering service, no data export API.

Pre-1993 ETF data does not exist in any source above; for that era the code
falls back gracefully (partial NaN rows) rather than raising an error.

SSL certificate verification
----------------------------
SSL verification is permanently disabled for all yfinance calls.

curl_cffi (yfinance's HTTP backend) uses its own bundled libcurl with its own
CA store.  It does NOT check the macOS Keychain, system trust store, certifi,
or any CURL_CA_BUNDLE env var.  There is no reliable runtime way to inject a
custom CA into curl_cffi's store, so verification is always skipped via
curl_cffi.requests.Session(verify=False).

Rate limiting
-------------
Phase 1 fetches all tickers in a single yf.download() batch call, which
makes one HTTP request instead of one per ticker.  This dramatically reduces
the chance of hitting Yahoo Finance's "Too Many Requests" rate limit.

Usage:
    from market_regime.ingestion.assets import fetch_all
    prices = fetch_all(cfg)   # returns DataFrame of quarterly adj-close prices
"""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd

log = logging.getLogger(__name__)

# ── Error signature detection ──────────────────────────────────────────────────

_SSL_ERROR_SIGNATURES = (
    "SSL",
    "certificate",
    "curl: (60)",
    "CertificateVerify",
    "CERTIFICATE_VERIFY",
    "self signed",
    "CERTIFICATE_VERIFY_FAILED",
)

_RATE_LIMIT_SIGNATURES = (
    "Too Many Requests",
    "Rate limit",
    "rate limited",
    "429",
)


def _ssl_bypass_curl_session():
    """
    Create a curl_cffi.requests.Session with SSL verification completely disabled.

    SSL is disabled unconditionally — this is intentional.  curl_cffi uses its
    own bundled CA store and ignores the macOS Keychain, system trust store, and
    certifi.  There is no reliable way to inject a custom CA at runtime; disabling
    verification is the only approach that works in all network environments.

    We try impersonate="chrome" first (needed for Yahoo Finance anti-bot detection).
    If that raises (unsupported in the installed curl_cffi version), we fall back to
    no impersonation with verify=False.
    """
    try:
        from curl_cffi import requests as curl_requests
    except ImportError:
        log.warning(
            "curl_cffi not importable — cannot create SSL-bypass session. "
            "Try: pip install curl_cffi"
        )
        return None

    try:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    except ImportError:
        pass

    # Try with browser impersonation first (avoids Yahoo anti-bot 401).
    # Some older curl_cffi builds don't accept impersonate= on Session();
    # fall back to a plain verify=False session in that case.
    try:
        return curl_requests.Session(verify=False, impersonate="chrome")
    except TypeError:
        log.debug("curl_cffi Session does not support impersonate= — using plain verify=False")
        return curl_requests.Session(verify=False)


# ── Phase 1: batch yfinance download ──────────────────────────────────────────

def _batch_yfinance(
    tickers: list[str], start: str, end: str, session=None
) -> tuple[dict[str, pd.Series], bool]:
    """
    Fetch all tickers in ONE yf.download() call.

    Returns (results, ssl_error_seen).
    results maps ticker → quarterly Close Series (only successfully fetched tickers).
    ssl_error_seen is True when an exception message contained an SSL keyword.

    Passing session=None uses yfinance's default curl_cffi behaviour.
    Passing a curl_cffi Session overrides SSL settings for that session.
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance is not installed.  Run: pip install yfinance")

    log.info("Batch-fetching %d tickers from yfinance ...", len(tickers))

    kwargs: dict = dict(
        tickers=tickers,
        start=start,
        end=end,
        interval="1mo",
        auto_adjust=True,
        progress=False,
    )
    if session is not None:
        kwargs["session"] = session

    ssl_seen = False
    try:
        raw = yf.download(**kwargs)
    except Exception as exc:
        msg = str(exc)
        if any(sig in msg for sig in _SSL_ERROR_SIGNATURES):
            ssl_seen = True
            log.warning("SSL error in batch yfinance download: %s", exc)
        elif any(sig in msg for sig in _RATE_LIMIT_SIGNATURES):
            log.warning("Rate limit hit in batch yfinance download: %s", exc)
        else:
            log.warning("Batch yfinance download failed: %s", exc)
        return {}, ssl_seen

    if raw is None or raw.empty:
        return {}, ssl_seen

    # Extract Close column(s).
    # Multiple tickers → MultiIndex columns: level-0 = metric, level-1 = ticker.
    # Single ticker   → flat columns: ['Open', 'High', 'Low', 'Close', 'Volume'].
    if isinstance(raw.columns, pd.MultiIndex):
        levels = raw.columns.get_level_values(0)
        if "Close" not in levels:
            log.warning("'Close' missing from batch download MultiIndex: %s", levels[:6].tolist())
            return {}, ssl_seen
        close_df = raw["Close"]  # DataFrame: index=date, columns=tickers
    else:
        if "Close" not in raw.columns:
            log.warning("'Close' missing from single-ticker download columns: %s", list(raw.columns))
            return {}, ssl_seen
        t = tickers[0] if len(tickers) == 1 else "Close"
        close_df = raw[["Close"]].rename(columns={"Close": t})

    results: dict[str, pd.Series] = {}
    for ticker in tickers:
        if ticker not in close_df.columns:
            log.debug("Ticker %s absent from batch results", ticker)
            continue
        s = close_df[ticker].dropna()
        if s.empty:
            log.warning("All-NaN Close data for %s in batch download", ticker)
            continue
        s = s.copy()
        s.name = ticker
        s.index = pd.to_datetime(s.index)
        if hasattr(s.index, "tz") and s.index.tz is not None:
            s.index = s.index.tz_localize(None)
        results[ticker] = s.resample("QE").last()

    return results, ssl_seen


# ── Phase 2: per-ticker SSL bypass ────────────────────────────────────────────

def _fetch_ticker_with_session(ticker: str, start: str, end: str, session) -> pd.Series:
    """
    Fetch one ticker via yf.Ticker(session=session).history().

    session must be a curl_cffi.requests.Session — yfinance ≥ 0.2 rejects
    a plain requests.Session with "Yahoo API requires curl_cffi session".
    """
    import yfinance as yf
    log.info("Fetching %s via curl_cffi session (SSL bypass) ...", ticker)
    raw = yf.Ticker(ticker, session=session).history(
        start=start, end=end, interval="1mo", auto_adjust=True
    )
    if raw.empty:
        log.warning("No data returned for %s", ticker)
        return pd.Series(name=ticker, dtype=float)
    close = raw["Close"].copy()
    close.name = ticker
    close.index = pd.to_datetime(close.index)
    if hasattr(close.index, "tz") and close.index.tz is not None:
        close.index = close.index.tz_localize(None)
    return close.resample("QE").last()


def _fetch_missing_with_ssl_bypass(
    missing: list[str], start: str, end: str
) -> dict[str, pd.Series]:
    """
    Retry each missing ticker using a curl_cffi Session(verify=False).
    Returns dict of successfully fetched tickers.
    """
    session = _ssl_bypass_curl_session()
    if session is None:
        return {}

    results: dict[str, pd.Series] = {}
    for ticker in missing:
        try:
            s = _fetch_ticker_with_session(ticker, start, end, session)
            if not s.empty:
                results[ticker] = s
        except Exception as exc:
            log.warning("SSL bypass also failed for %s: %s", ticker, exc)
    return results


# ── Phase 3: stooq fallback (pandas-datareader) ────────────────────────────────

def _fetch_ticker_stooq(ticker: str, start: str, end: str) -> pd.Series:
    """
    Fetch one ticker from stooq.pl via pandas-datareader.

    Stooq provides free daily OHLCV data for major US ETFs going back to
    their inception.  No API key required.  Install: pip install pandas-datareader

    Raises ImportError if pandas-datareader is not installed (caller breaks loop).
    """
    from pandas_datareader import data as pdr  # ImportError propagates to caller

    log.info("Fetching %s from stooq (fallback) ...", ticker)
    symbol = f"{ticker}.US"  # stooq convention for US-listed securities
    try:
        raw = pdr.get_data_stooq(symbol, start=start, end=end)
    except Exception as exc:
        log.warning("stooq returned error for %s (%s): %s", ticker, symbol, exc)
        return pd.Series(name=ticker, dtype=float)

    if raw is None or raw.empty:
        log.warning("No stooq data for %s", ticker)
        return pd.Series(name=ticker, dtype=float)

    close = raw["Close"].rename(ticker)
    close.index = pd.to_datetime(close.index)
    # stooq returns descending order — sort before resample
    return close.sort_index().resample("QE").last()


def _fetch_tickers_stooq(tickers: list[str], start: str, end: str) -> list[pd.Series]:
    """Fetch all tickers via stooq; return list of non-empty Series."""
    results: list[pd.Series] = []
    for ticker in tickers:
        try:
            s = _fetch_ticker_stooq(ticker, start, end)
            if not s.empty:
                results.append(s)
        except ImportError:
            log.warning(
                "pandas-datareader not installed — stooq fallback unavailable.  "
                "Run: pip install pandas-datareader"
            )
            break  # no point trying more tickers
        except Exception as exc:
            log.warning("stooq failed for %s: %s", ticker, exc)
    return results


# ── Phase 4: OpenBB fallback ───────────────────────────────────────────────────

def _fetch_ticker_openbb(ticker: str, start: str, end: str) -> pd.Series:
    """
    Fetch one ticker via OpenBB.

    Tries the 'cboe' provider first (free, no API key).  Falls back to the
    OpenBB default provider if cboe does not have the ticker.

    Install: pip install openbb
    Raises ImportError if openbb is not installed (caller breaks loop).
    """
    from openbb import obb  # ImportError propagates to caller

    log.info("Fetching %s via OpenBB (fallback) ...", ticker)

    df: pd.DataFrame | None = None
    for provider in ("cboe", None):  # None → OpenBB picks default
        try:
            kwargs: dict = dict(symbol=ticker, start_date=start, end_date=end)
            if provider is not None:
                kwargs["provider"] = provider
            result = obb.equity.price.historical(**kwargs)
            df = result.to_df()
            if not df.empty:
                break
        except Exception as exc:
            log.debug("OpenBB provider=%s failed for %s: %s", provider, ticker, exc)

    if df is None or df.empty:
        log.warning("OpenBB returned no data for %s", ticker)
        return pd.Series(name=ticker, dtype=float)

    # OpenBB column names vary by provider — find the close column
    close_col = next(
        (c for c in df.columns if c.lower() in ("close", "adj_close", "adjusted_close")),
        None,
    )
    if close_col is None:
        log.warning("OpenBB: could not find close column for %s (columns: %s)", ticker, list(df.columns))
        return pd.Series(name=ticker, dtype=float)

    close = df[close_col].rename(ticker)
    close.index = pd.to_datetime(close.index)
    if hasattr(close.index, "tz") and close.index.tz is not None:
        close.index = close.index.tz_localize(None)
    return close.sort_index().resample("QE").last()


def _fetch_tickers_openbb(tickers: list[str], start: str, end: str) -> list[pd.Series]:
    """Fetch all tickers via OpenBB; return list of non-empty Series."""
    results: list[pd.Series] = []
    for ticker in tickers:
        try:
            s = _fetch_ticker_openbb(ticker, start, end)
            if not s.empty:
                results.append(s)
        except ImportError:
            log.warning(
                "openbb not installed — OpenBB fallback unavailable.  "
                "Run: pip install openbb"
            )
            break  # no point trying more tickers
        except Exception as exc:
            log.warning("OpenBB failed for %s: %s", ticker, exc)
    return results


# ── public API ─────────────────────────────────────────────────────────────────

def fetch_all(cfg: dict) -> pd.DataFrame:
    """
    Fetch quarterly adjusted-close prices for all tickers in cfg["assets"]["etfs"].

    SSL certificate verification is disabled for all yfinance calls.  curl_cffi
    (yfinance's HTTP backend) uses its own bundled CA store and does not check the
    macOS Keychain or system trust store; there is no reliable runtime way to inject
    a custom CA, so verification is always skipped.

    Phase 1  — batch yf.download() with verify=False curl_cffi session
               (one HTTP request for all tickers — avoids rate limiting)
    Phase 2  — per-ticker yf.Ticker(session=...).history() with same session,
               for any tickers missing from the batch result
    Phase 3  — stooq via pandas-datareader
    Phase 4  — OpenBB
    Phase 5  — empty DataFrame → caller uses macro-proxy returns

    Returns:
        DataFrame indexed by quarter-end dates, one column per ticker.
        Columns contain NaN where data is unavailable (pre-ETF-launch dates).
        Returns an empty DataFrame if all fetches fail.
    """
    tickers: list[str] = cfg.get("assets", {}).get("etfs", [])
    if not tickers:
        log.warning("No asset tickers configured — skipping")
        return pd.DataFrame()

    start = cfg["data"]["start_date"]
    end = cfg["data"]["end_date"] or str(date.today())

    # Always use verify=False — see module docstring for rationale.
    session = _ssl_bypass_curl_session()

    # ── Phase 1: batch download ────────────────────────────────────────────────
    results, _ = _batch_yfinance(tickers, start, end, session=session)

    # ── Phase 2: per-ticker retry for any missing tickers ─────────────────────
    missing = [t for t in tickers if t not in results]
    if missing:
        log.info(
            "Phase 2: fetching %d/%d missing ticker(s) individually ...",
            len(missing), len(tickers),
        )
        recovered = _fetch_missing_with_ssl_bypass(missing, start, end)
        results.update(recovered)
        still_missing = [t for t in missing if t not in recovered]
        if still_missing:
            log.warning("Could not fetch: %s", still_missing)

    series_list = list(results.values())

    # ── Phase 3: stooq via pandas-datareader ──────────────────────────────────
    if not series_list:
        log.info("Trying stooq fallback (free, no API key) ...")
        series_list = _fetch_tickers_stooq(tickers, start, end)
        if series_list:
            log.info("stooq fallback succeeded (%d/%d tickers)", len(series_list), len(tickers))
        else:
            log.warning(
                "stooq returned no data.  "
                "Install pandas-datareader to enable this fallback: pip install pandas-datareader"
            )

    # ── Phase 4: OpenBB ───────────────────────────────────────────────────────
    if not series_list:
        log.info("Trying OpenBB fallback ...")
        series_list = _fetch_tickers_openbb(tickers, start, end)
        if series_list:
            log.info("OpenBB fallback succeeded (%d/%d tickers)", len(series_list), len(tickers))
        else:
            log.warning(
                "OpenBB returned no data.  "
                "Install openbb to enable this fallback: pip install openbb\n"
                "Falling back to macro-data proxy returns (see asset_returns.compute_proxy_returns)."
            )

    # ── Phase 5: empty → macro proxy returns computed by caller ───────────────
    if not series_list:
        log.error(
            "All price data sources failed.  "
            "The pipeline will use macro-data proxy returns (compute_proxy_returns) instead.  "
            "To reuse a previously fetched checkpoint, run without --refresh-assets:\n"
            "  python run_pipeline.py --steps 6"
        )
        return pd.DataFrame()

    df = pd.concat(series_list, axis=1)
    df.index.name = "date"

    log.info(
        "Asset prices fetched: %d quarters, %d/%d tickers, %.1f%% coverage",
        len(df),
        len(df.columns),
        len(tickers),
        100 * df.notna().mean().mean(),
    )
    return df
