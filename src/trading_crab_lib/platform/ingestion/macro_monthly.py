"""
Monthly macro/long-history raw ingestion (DATA-01).

The incumbent quarterly pipeline's fetchers (``ingestion/fred.py``,
``ingestion/multpl.py``, ``ingestion/macrotrends.py``) all hardcode a
period-end quarterly resample rule internally — reusing them verbatim would
silently keep quarterly cadence and defeat this phase's entire purpose
(RESEARCH Pitfall 1).
This module writes thin monthly analogs that reuse the same client
construction / parallel-fetch / scrape-and-parse patterns but target
``"ME"`` (month-end) instead, without editing any frozen incumbent file
(D-01).

FRED market/fast-layer series (``fetch_fred_monthly``) reuse the
``fred.py::_fetch_one`` client-construction + ``ThreadPoolExecutor`` +
try/except-WARNING skeleton (this task). multpl valuation anchors and
macrotrends long-history commodities are added by a later task in this same
module, ultimately merged by ``fetch_macro_monthly`` via
``pd.concat([...], axis=1)`` — an outer, NULL-tolerant join (RESEARCH
Pitfall 5).

Usage:
    from trading_crab_lib.platform.ingestion.macro_monthly import fetch_fred_monthly
    from trading_crab_lib.platform.config import load_platform_config

    cfg = load_platform_config()
    fred_monthly = fetch_fred_monthly(cfg)
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from typing import Any

import pandas as pd

try:
    from fredapi import Fred
except ImportError as _err:
    raise ImportError(
        "fredapi is required for FRED data ingestion. "
        "Install with: pip install 'trading-crab-lib[ingestion]'"
    ) from _err

log = logging.getLogger(__name__)

# FRED is tolerant of small parallel bursts — same cap as the incumbent's
# fred.py (single-vintage get_series() calls, not the larger all-releases
# payloads ALFRED ingestion uses).
_MAX_WORKERS = 8


# ── FRED monthly ────────────────────────────────────────────────────────────


def _fetch_fred_monthly(
    fred: Fred,
    series_id: str,
    start: str,
    end: str,
    shift: bool,
    monthly_freq: str = "ME",
) -> pd.Series:
    """Pull one FRED series, resample to month-end, optionally apply publication lag.

    Monthly analog of ``fred.py::_fetch_one`` — same client and pull, but
    resamples to ``monthly_freq`` (month-end, ``"ME"``) instead of the
    incumbent's hardcoded quarterly period-end rule (RESEARCH Pitfall 1).
    """
    raw = fred.get_series(series_id, observation_start=start, observation_end=end)
    monthly = raw.resample(monthly_freq).last()
    if shift:
        monthly = monthly.shift(1)  # lag one period — data known next period
    return monthly


def fetch_fred_monthly(cfg: dict[str, Any]) -> pd.DataFrame:
    """
    Fetch every series in cfg["fred_monthly"]["series"] and join into one
    monthly DataFrame.

    Mirrors ``fred.py::fetch_all``'s client-construction + ThreadPoolExecutor
    + try/except-log-WARNING-return-None skeleton (graceful degradation on a
    single-series failure) — never imports or modifies ``ingestion/fred.py``.

    Config shape expected:
        fred_monthly:
          series:
            GS10:
              name:  "fred_gs10"
              shift: false
        data:
          start_date:   "1962-01-01"
          end_date:     null
          monthly_freq: "ME"

    Returns:
        DataFrame indexed by month-end dates, columns = friendly names.
    """
    api_key = cfg["fred_monthly"].get("api_key")
    if not api_key:
        raise OSError("FRED_API_KEY is not set")

    fred = Fred(api_key=api_key)

    start = cfg["data"]["start_date"]
    end = cfg["data"]["end_date"] or str(date.today())
    monthly_freq = cfg["data"].get("monthly_freq", "ME")

    series_cfg: dict = cfg["fred_monthly"]["series"]

    def _fetch_task(series_id: str, meta: dict) -> tuple[str, pd.Series | None]:
        friendly_name = meta["name"]
        shift = meta.get("shift", False)
        lag_note = " (shifted +1)" if shift else ""
        log.info("Fetching FRED (monthly) %-10s → %s%s", series_id, friendly_name, lag_note)
        try:
            s = _fetch_fred_monthly(fred, series_id, start, end, shift, monthly_freq)
            s.name = friendly_name
            return friendly_name, s
        except Exception as exc:  # noqa: BLE001 — fredapi raises various types
            log.warning("Failed to fetch %s (%s): %s", friendly_name, series_id, exc)
            return friendly_name, None

    frames: dict[str, pd.Series] = {}
    with ThreadPoolExecutor(max_workers=min(_MAX_WORKERS, max(len(series_cfg), 1))) as pool:
        futures = {
            pool.submit(_fetch_task, sid, meta): sid
            for sid, meta in series_cfg.items()
        }
        for future in as_completed(futures):
            friendly_name, series = future.result()
            if series is not None:
                frames[friendly_name] = series

    df = pd.DataFrame(frames)
    df.index.name = "date"
    log.info("FRED monthly fetch complete: %d months, %d series", len(df), len(df.columns))
    return df
