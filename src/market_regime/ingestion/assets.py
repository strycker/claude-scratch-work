"""
ETF / equity price ingestion via yfinance.

Downloads monthly adjusted-close prices for the configured asset tickers,
resamples to quarterly (period-end), and returns a wide DataFrame indexed
by quarter-end dates — ready to join against cluster labels.

Pre-1993 ETF data does not exist in Yahoo Finance; for that era the code
falls back gracefully (partial NaN rows) rather than raising an error.
A future Macrotrends scraper can backfill gold, oil, and bond history.

SSL / corporate-firewall note
------------------------------
If you are behind a corporate proxy that inspects HTTPS traffic (self-signed
cert in the chain), yfinance will log SSL errors and return empty data.
This module detects that situation automatically and retries with SSL
verification disabled (using a requests.Session(verify=False)).

To avoid the automatic retry delay on every run, set these environment
variables in your shell *before* starting Python:

    export CURL_CA_BUNDLE=""
    export REQUESTS_CA_BUNDLE=""

⚠  NOTE: "export NODE_EXTRA_CA_CERTS=0" affects Node.js only — it has
   NO effect on Python or libcurl. Do not rely on it here.

For a more secure permanent fix, ask your IT department for the corporate
root CA certificate bundle and point to it instead of using an empty string:

    export CURL_CA_BUNDLE=/path/to/corp-ca-bundle.pem

To skip the yfinance fetch entirely and reuse a previously saved checkpoint,
run step 6 without --refresh-assets:

    python run_pipeline.py --steps 6   # uses data/raw/asset_prices.parquet

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


def _ssl_bypass_session():
    """Return a requests.Session with SSL verification disabled, or None on failure."""
    try:
        import requests
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        session = requests.Session()
        session.verify = False
        return session
    except ImportError:
        log.error("Cannot create SSL bypass session — requests not installed")
        return None


# ── per-ticker fetch ───────────────────────────────────────────────────────────

def _fetch_ticker(ticker: str, start: str, end: str, session=None) -> pd.Series:
    """
    Download monthly adjusted close for one ticker and resample to quarterly.

    When session is None (default), uses yf.download() which internally uses
    curl_cffi for fast parallel fetching.

    When session is a requests.Session (SSL bypass path), uses
    yf.Ticker(session=session).history() which routes through the requests
    library, honouring session.verify=False.
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance is not installed. Run: pip install yfinance")

    log.info("Fetching %s from yfinance ...", ticker)

    if session is not None:
        # requests-based path — used for SSL bypass
        t = yf.Ticker(ticker, session=session)
        raw = t.history(start=start, end=end, interval="1mo", auto_adjust=True)
    else:
        # curl_cffi-based path — default, faster
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
) -> list[pd.Series]:
    """Fetch all tickers; return list of non-empty Series."""
    results = []
    for ticker in tickers:
        try:
            s = _fetch_ticker(ticker, start, end, session=session)
            if not s.empty:
                results.append(s)
        except Exception as exc:
            log.warning("Failed to fetch %s: %s", ticker, exc)
    return results


# ── public API ─────────────────────────────────────────────────────────────────

def fetch_all(cfg: dict) -> pd.DataFrame:
    """
    Fetch quarterly adjusted-close prices for all tickers in cfg["assets"]["etfs"].

    Automatic SSL recovery
    ----------------------
    If the first fetch attempt returns no data and the yfinance logger emitted
    SSL-related errors, this function logs a diagnostic help message and retries
    using a requests.Session(verify=False).  This handles corporate proxies that
    perform HTTPS inspection with a self-signed certificate.

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
        series_list = _fetch_tickers(tickers, start, end)
    finally:
        yf_logger.removeHandler(detector)

    # ── Phase 2: SSL bypass retry ──────────────────────────────────────────────
    if not series_list and detector.ssl_detected:
        log.warning(_SSL_HELP_MESSAGE)
        session = _ssl_bypass_session()
        if session is not None:
            log.info(
                "SSL bypass active — fetching %d tickers via requests.Session(verify=False)",
                len(tickers),
            )
            series_list = _fetch_tickers(tickers, start, end, session=session)
            if series_list:
                log.warning(
                    "SSL bypass succeeded (%d tickers fetched).  "
                    "Add 'export CURL_CA_BUNDLE=\"\"' to your shell profile to "
                    "skip the retry delay on future runs.",
                    len(series_list),
                )
            else:
                log.error(
                    "SSL bypass also returned no data.  "
                    "Check your network connection or run without --refresh-assets "
                    "to load from a saved checkpoint."
                )

    if not series_list:
        if not detector.ssl_detected:
            log.error("No asset price data retrieved")
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
