"""
FRED API ingestion.

Fetches each series defined in config["fred"]["series"], resamples to
quarterly frequency (period-end), and returns a single wide DataFrame
with one column per series.

Usage:
    from market_regime.ingestion.fred import fetch_all
    df = fetch_all(cfg)
"""

import logging
from datetime import date

import pandas as pd
from fredapi import Fred

log = logging.getLogger(__name__)


def fetch_series(fred: Fred, series_id: str, start: str, end: str) -> pd.Series:
    """Pull one FRED series and resample to quarterly end."""
    raw = fred.get_series(series_id, observation_start=start, observation_end=end)
    # Resample to quarter-end; forward-fill within the quarter
    quarterly = raw.resample("QE").last().ffill()
    quarterly.name = series_id
    return quarterly


def fetch_all(cfg: dict) -> pd.DataFrame:
    """
    Fetch every series in cfg["fred"]["series"] and join into one DataFrame.

    Returns:
        DataFrame indexed by quarter-end dates, columns = friendly names.
    """
    api_key = cfg["fred"]["api_key"]
    if not api_key:
        raise EnvironmentError("FRED_API_KEY is not set")

    fred = Fred(api_key=api_key)

    start = cfg["data"]["start_date"]
    end = cfg["data"]["end_date"] or str(date.today())

    series: dict[str, str] = cfg["fred"]["series"]
    frames: dict[str, pd.Series] = {}

    for name, series_id in series.items():
        log.info("Fetching FRED series %s → %s", series_id, name)
        try:
            frames[name] = fetch_series(fred, series_id, start, end)
        except Exception as exc:
            log.warning("Failed to fetch %s (%s): %s", name, series_id, exc)

    df = pd.DataFrame(frames)
    df.index.name = "date"
    log.info("FRED fetch complete: %d quarters, %d series", len(df), len(df.columns))
    return df
