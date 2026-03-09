"""
ETF / equity price ingestion via yfinance.

Downloads monthly adjusted-close prices for the configured asset tickers,
resamples to quarterly (period-end), and returns a wide DataFrame indexed
by quarter-end dates — ready to join against cluster labels.

Pre-1993 ETF data does not exist in Yahoo Finance; for that era the code
falls back gracefully (partial NaN rows) rather than raising an error.
A future Macrotrends scraper can backfill gold, oil, and bond history.

Usage:
    from market_regime.ingestion.assets import fetch_all
    prices = fetch_all(cfg)   # returns DataFrame of quarterly adj-close prices
"""

import logging
from datetime import date

import pandas as pd

log = logging.getLogger(__name__)


def _fetch_ticker(ticker: str, start: str, end: str) -> pd.Series:
    """Download monthly adjusted close for one ticker and resample to quarterly."""
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError(
            "yfinance is not installed. Run: pip install yfinance"
        )

    log.info("Fetching %s from yfinance ...", ticker)
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

    # yfinance returns a MultiIndex column when downloading one ticker
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"][ticker]
    else:
        close = raw["Close"]

    close.name = ticker
    close.index = pd.to_datetime(close.index)
    quarterly = close.resample("QE").last()
    return quarterly


def fetch_all(cfg: dict) -> pd.DataFrame:
    """
    Fetch quarterly adjusted-close prices for all tickers in cfg["assets"]["etfs"].

    Returns:
        DataFrame indexed by quarter-end dates, one column per ticker.
        Columns contain NaN where data is unavailable (pre-ETF-launch dates).
    """
    tickers: list[str] = cfg.get("assets", {}).get("etfs", [])
    if not tickers:
        log.warning("No asset tickers configured — skipping")
        return pd.DataFrame()

    start = cfg["data"]["start_date"]
    end = cfg["data"]["end_date"] or str(date.today())

    series_list: list[pd.Series] = []
    for ticker in tickers:
        try:
            s = _fetch_ticker(ticker, start, end)
            if not s.empty:
                series_list.append(s)
        except Exception as exc:
            log.warning("Failed to fetch %s: %s", ticker, exc)

    if not series_list:
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


