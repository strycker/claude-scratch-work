"""
ETF / equity price ingestion — yfinance with multi-source fallback chain.

Downloads monthly adjusted-close prices for the configured asset tickers,
resamples to quarterly (period-end), and returns a wide DataFrame indexed
by quarter-end dates — ready to join against cluster labels.

Fallback chain (tried in order, stopping at first success)
----------------------------------------------------------
1. yfinance (curl_cffi path)          — fast, default
2. yfinance (env-var SSL bypass)      — clears CURL_CA_BUNDLE for corp proxies
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
  Good for current fundamental/technical signals; use for notebook QA overlays
  once the regime pipeline has already run (see ROADMAP 2.8).
- StockCharts.com: chart-rendering service, no data export API.
  Scraping is possible but complex and subject to ToS review.
  Tracked in ROADMAP 3.4 as a potential raw-data source for technical indicators.

Pre-1993 ETF data does not exist in any source above; for that era the code
falls back gracefully (partial NaN rows) rather than raising an error.
A future macrotrends.net scraper can backfill gold, oil, and bond history.

SSL / corporate-firewall note
------------------------------
If you are behind a corporate proxy that inspects HTTPS traffic (self-signed
cert in the chain), yfinance will log SSL errors and return empty data.
This module detects that situation automatically and retries by temporarily
clearing CURL_CA_BUNDLE / REQUESTS_CA_BUNDLE so that curl_cffi (yfinance's
HTTP backend) skips certificate verification.

To avoid the automatic retry delay on every run, set these environment
variables in your shell *before* starting Python:

    export CURL_CA_BUNDLE=""
    export REQUESTS_CA_BUNDLE=""

For a more secure permanent fix, ask your IT department for the corporate
root CA certificate bundle and point to it instead of using an empty string:

    export CURL_CA_BUNDLE=/path/to/corp-ca-bundle.pem

Note on SSL bypass: yfinance ≥ 0.2 uses curl_cffi internally and does NOT
accept a requests.Session.  The SSL bypass path creates a curl_cffi session
with verify=False and impersonate="chrome", which yfinance accepts.  Env vars
(CURL_CA_BUNDLE="") are also cleared as a belt-and-suspenders measure, but
the curl_cffi session is the primary bypass mechanism.

Usage:
    from market_regime.ingestion.assets import fetch_all
    prices = fetch_all(cfg)   # returns DataFrame of quarterly adj-close prices
"""

from __future__ import annotations

import logging
import os
from datetime import date

import pandas as pd

log = logging.getLogger(__name__)

# curl_cffi (yfinance's HTTP backend) maintains its own certificate store and
# does not automatically use Python's certifi bundle on macOS or in corporate
# proxy environments.  Setting these env vars (only when not already set by the
# user) is the first-line fix for non-MITM SSL environments.
try:
    import certifi as _certifi
    os.environ.setdefault("CURL_CA_BUNDLE", _certifi.where())
    os.environ.setdefault("SSL_CERT_FILE", _certifi.where())
except ImportError:
    pass


# ── SSL error detection ────────────────────────────────────────────────────────

_SSL_ERROR_SIGNATURES = (
    "SSL",
    "certificate",
    "curl: (60)",
    "CertificateVerify",
    "CERTIFICATE_VERIFY",
    "self signed",
)

_SSL_HELP_MESSAGE = """\

╔══════════════════════════════════════════════════════════════════════════╗
║   SSL Certificate Error — yfinance blocked by firewall/proxy            ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  Cause: A corporate proxy or VPN is doing HTTPS inspection using a       ║
║  self-signed certificate that is NOT trusted by Python or libcurl.       ║
║                                                                          ║
║  ▶ Retrying automatically with SSL verification disabled ...             ║
║                                                                          ║
║  To avoid this retry delay on future runs, set before starting:          ║
║    export CURL_CA_BUNDLE=""                                              ║
║    export REQUESTS_CA_BUNDLE=""                                          ║
║                                                                          ║
║  ⚠  "export NODE_EXTRA_CA_CERTS=0" only affects Node.js — it does       ║
║     NOTHING for Python. Do not rely on it here.                          ║
║                                                                          ║
║  For a more secure fix, point to your corporate CA bundle instead:       ║
║    export CURL_CA_BUNDLE=/path/to/corp-ca-bundle.pem                    ║
║                                                                          ║
║  To skip yfinance and use a previously saved checkpoint:                 ║
║    python run_pipeline.py --steps 6   (without --refresh-assets)        ║
╚══════════════════════════════════════════════════════════════════════════╝
"""


class _SSLErrorDetector(logging.Handler):
    """Temporary log handler attached to the yfinance logger to sniff SSL errors."""

    def __init__(self) -> None:
        super().__init__()
        self.ssl_detected = False

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        if any(sig in msg for sig in _SSL_ERROR_SIGNATURES):
            self.ssl_detected = True


def _ssl_bypass_curl_session():
    """
    Create a curl_cffi.requests.Session with SSL verification disabled.

    This is the correct SSL bypass for yfinance ≥ 0.2, which requires a
    curl_cffi session (not a requests.Session).  curl_cffi is a required
    yfinance dependency so it should always be importable.

    Returns None only if curl_cffi is somehow absent (yfinance < 0.2).
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

    # impersonate="chrome" is required by yfinance to pass its browser-check;
    # verify=False disables certificate validation for the self-signed cert.
    return curl_requests.Session(verify=False, impersonate="chrome")


def _apply_ssl_env_bypass() -> dict[str, str | None]:
    """
    Clear SSL certificate env vars as a belt-and-suspenders measure alongside
    the curl_cffi session bypass.  Returns saved values for restoration.
    """
    saved: dict[str, str | None] = {}
    for var in ("CURL_CA_BUNDLE", "REQUESTS_CA_BUNDLE", "SSL_CERT_FILE"):
        saved[var] = os.environ.get(var)
        os.environ[var] = ""
    return saved


def _restore_ssl_env(saved: dict[str, str | None]) -> None:
    """Restore env vars previously saved by _apply_ssl_env_bypass()."""
    for var, val in saved.items():
        if val is None:
            os.environ.pop(var, None)
        else:
            os.environ[var] = val


# ── per-ticker fetch ───────────────────────────────────────────────────────────

def _fetch_ticker(ticker: str, start: str, end: str, session=None) -> pd.Series:
    """
    Download monthly adjusted close for one ticker and resample to quarterly.

    When session is None (default), uses yf.download() with curl_cffi's default
    SSL settings.

    When session is a curl_cffi.requests.Session (SSL-bypass path), uses
    yf.Ticker(session=session).history() which honours session.verify=False.
    A curl_cffi session is required — passing a requests.Session raises
    "Yahoo API requires curl_cffi session" in yfinance ≥ 0.2.
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance is not installed. Run: pip install yfinance")

    log.info("Fetching %s from yfinance ...", ticker)

    if session is not None:
        # curl_cffi session path — used for SSL bypass; verify=False set on session
        raw = yf.Ticker(ticker, session=session).history(
            start=start, end=end, interval="1mo", auto_adjust=True
        )
    else:
        raw = yf.download(
            ticker,
            start=start,
            end=end,
            interval="1mo",
            auto_adjust=True,
            progress=False,
        )

    if raw.empty:
        log.warning("No data returned for %s", ticker)
        return pd.Series(name=ticker, dtype=float)

    # yf.download() can return MultiIndex columns for a single ticker
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"][ticker]
    else:
        close = raw["Close"]

    close = close.copy()
    close.name = ticker
    close.index = pd.to_datetime(close.index)

    # yf.Ticker.history() returns a timezone-aware index; strip tz for consistency
    if hasattr(close.index, "tz") and close.index.tz is not None:
        close.index = close.index.tz_localize(None)

    return close.resample("QE").last()


def _fetch_tickers(
    tickers: list[str], start: str, end: str, session=None
) -> tuple[list[pd.Series], bool]:
    """
    Fetch all tickers via yfinance.

    Returns:
        (series_list, ssl_error_seen) — series_list contains non-empty results;
        ssl_error_seen is True if any exception message matched an SSL signature.
    """
    results: list[pd.Series] = []
    ssl_seen = False
    for ticker in tickers:
        try:
            s = _fetch_ticker(ticker, start, end, session=session)
            if not s.empty:
                results.append(s)
        except Exception as exc:
            if any(sig in str(exc) for sig in _SSL_ERROR_SIGNATURES):
                ssl_seen = True
                log.warning(
                    "SSL error fetching %s (will retry with SSL bypass): %s",
                    ticker, exc,
                )
            else:
                log.warning("Failed to fetch %s: %s", ticker, exc)
    return results, ssl_seen


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

    Automatic SSL recovery
    ----------------------
    If the first fetch attempt returns no data and the yfinance logger emitted
    SSL-related errors, this function logs a diagnostic help message and retries
    by temporarily clearing CURL_CA_BUNDLE / REQUESTS_CA_BUNDLE so that
    curl_cffi (yfinance's HTTP backend) skips certificate verification.
    This handles corporate proxies with self-signed certificates.
    (A requests.Session is NOT used — yfinance ≥ 0.2 rejects it.)

    Returns:
        DataFrame indexed by quarter-end dates, one column per ticker.
        Columns contain NaN where data is unavailable (pre-ETF-launch dates).
        Returns an empty DataFrame if all fetches fail (proxy returns from macro
        data are computed by the caller as a fallback).
    """
    tickers: list[str] = cfg.get("assets", {}).get("etfs", [])
    if not tickers:
        log.warning("No asset tickers configured — skipping")
        return pd.DataFrame()

    start = cfg["data"]["start_date"]
    end = cfg["data"]["end_date"] or str(date.today())

    # ── Phase 1: normal fetch ──────────────────────────────────────────────────
    # Attach a temporary handler to the yfinance logger so we can detect SSL
    # errors that yfinance swallows internally (it logs them but doesn't raise).
    detector = _SSLErrorDetector()
    yf_logger = logging.getLogger("yfinance")
    yf_logger.addHandler(detector)

    try:
        series_list, ssl_from_exc = _fetch_tickers(tickers, start, end)
    finally:
        yf_logger.removeHandler(detector)

    # ── Phase 2: SSL bypass retry ──────────────────────────────────────────────
    # Trigger when ANY SSL error was detected — either from the yfinance logger
    # (detector.ssl_detected) or from exception messages (ssl_from_exc).
    # Only retry tickers that are still missing so we don't re-fetch successes.
    # Clears CURL_CA_BUNDLE / REQUESTS_CA_BUNDLE so curl_cffi skips cert
    # verification.  Does NOT pass a requests.Session — yfinance ≥ 0.2 rejects
    # that with "Yahoo API requires curl_cffi session".
    ssl_detected = detector.ssl_detected or ssl_from_exc
    fetched_names = {s.name for s in series_list}
    missing = [t for t in tickers if t not in fetched_names]

    if missing and ssl_detected:
        log.warning(_SSL_HELP_MESSAGE)
        log.info(
            "SSL bypass active — retrying %d/%d missing tickers "
            "with curl_cffi session (verify=False)",
            len(missing), len(tickers),
        )
        curl_session = _ssl_bypass_curl_session()
        saved_env = _apply_ssl_env_bypass()  # belt-and-suspenders: also clear env vars
        try:
            retry_list, _ = _fetch_tickers(missing, start, end, session=curl_session)
        finally:
            _restore_ssl_env(saved_env)
        series_list.extend(retry_list)

        if retry_list:
            log.warning(
                "SSL bypass recovered %d/%d tickers.  "
                "Add 'export CURL_CA_BUNDLE=\"\"' to your shell profile to "
                "skip the retry delay on future runs.",
                len(retry_list), len(missing),
            )
        else:
            log.error(
                "SSL bypass also returned no data.  "
                "Check your network connection or run without --refresh-assets "
                "to load from a saved checkpoint."
            )

    # ── Phase 3: stooq via pandas-datareader ─────────────────────────────────
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

    # ── Phase 5: empty → macro proxy returns computed by caller ──────────────
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
        "Asset prices fetched: %d quarters, %d tickers, %.1f%% coverage",
        len(df),
        len(df.columns),
        100 * df.notna().mean().mean(),
    )
    return df
