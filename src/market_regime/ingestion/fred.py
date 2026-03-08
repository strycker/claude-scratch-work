"""
FRED API ingestion.

Fetches each series defined in cfg["fred"]["series"], resamples to
quarterly frequency (period-end), and returns a single wide DataFrame.

Publication-lag shift:
  GDP and GNP figures are released after the quarter closes.  Series marked
  shift:true are shifted forward by one quarter so the value aligns with
  the quarter in which it would actually have been known.  This is critical
  for preventing look-ahead bias in supervised models.
"""

import logging
from datetime import date

import pandas as pd
from fredapi import Fred

log = logging.getLogger(__name__)


def _fetch_one(fred: Fred, series_id: str, start: str, end: str, shift: bool) -> pd.Series:
    """Pull one FRED series, resample to QE, optionally apply publication lag."""
    raw = fred.get_series(series_id, observation_start=start, observation_end=end)
    quarterly = raw.resample("QE").last()
    if shift:
        quarterly = quarterly.shift(1)  # lag one quarter — data known next quarter
    return quarterly


def fetch_all(cfg: dict) -> pd.DataFrame:
    """
    Fetch every series in cfg["fred"]["series"] and join into one DataFrame.

    Config shape expected:
        fred:
          series:
            GDP:
              name:  "fred_gdp"
              shift: true
            BAA:
              name:  "fred_baa"
              shift: false

    Returns:
        DataFrame indexed by quarter-end dates, columns = friendly names.
    """
    api_key = cfg["fred"].get("api_key")
    if not api_key:
        raise EnvironmentError("FRED_API_KEY is not set")

    fred = Fred(api_key=api_key)

    start = cfg["data"]["start_date"]
    end = cfg["data"]["end_date"] or str(date.today())

    series_cfg: dict = cfg["fred"]["series"]
    frames: dict[str, pd.Series] = {}

    for series_id, meta in series_cfg.items():
        friendly_name = meta["name"]
        shift = meta.get("shift", False)
        lag_note = " (shifted +1Q for publication lag)" if shift else ""
        log.info("Fetching FRED %-10s → %s%s", series_id, friendly_name, lag_note)
        try:
            s = _fetch_one(fred, series_id, start, end, shift)
            s.name = friendly_name
            frames[friendly_name] = s
        except Exception as exc:
            log.warning("Failed to fetch %s (%s): %s", friendly_name, series_id, exc)

    df = pd.DataFrame(frames)
    df.index.name = "date"
    log.info("FRED fetch complete: %d quarters, %d series", len(df), len(df.columns))
    return df
