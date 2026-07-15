"""
ALFRED point-in-time vintage ingestion (D-06).

Publication-lag `shift` alone (see ``trading_crab_lib.ingestion.fred``) fixes
*timing* look-ahead — you can't know Q1 GDP the day Q1 ends — but not
*revision* look-ahead: using the final, fully-revised GDP figure is still
cheating even when correctly dated. ALFRED vintages give the value that was
actually published as of a given date, for the five revision-heavy agency
series in scope (GDP, CPI, UNRATE, INDPRO, PAYEMS).

Bulk fetch, never per-date loop:
  ``fetch_vintage_series()`` calls ``fredapi.Fred.get_series_all_releases()``
  exactly ONCE per series — it returns every historical revision of every
  observation in a single request. Looping ``get_series_as_of_date()`` per
  historical month would be O(months) API calls (~750/series for 1962+
  monthly) and is an explicit anti-pattern (RESEARCH Pattern 2 / Anti-Pattern).

Pre-vintage-era fallback (D-06):
  ALFRED archives mostly begin decades after 1962. For dates earlier than a
  series' earliest recorded vintage, ``align_with_fallback()`` falls back to
  the caller-supplied publication-lag-shifted value — an accepted, documented
  compromise (see ``docs/vintage_alignment.md``), never a silent NaN/error.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import pandas as pd

try:
    from fredapi import Fred
except ImportError as _err:
    raise ImportError(
        "fredapi is required for ALFRED vintage ingestion. "
        "Install with: pip install 'trading-crab-lib[ingestion]'"
    ) from _err

log = logging.getLogger(__name__)

# All-releases payloads are much larger than fred.py's single-vintage
# get_series() calls (RESEARCH Pitfall 3) — keep the worker cap well below
# the incumbent's 8 so we don't stall the pool on oversized responses.
_MAX_WORKERS = 3

_REQUIRED_VINTAGE_ROLES = ("realtime_start", "realtime_end", "date", "value")


def _detect_vintage_columns(df: pd.DataFrame) -> dict[str, str]:
    """Locate realtime_start/realtime_end/date/value columns case-insensitively.

    Guards the [ASSUMED] all-releases column schema (RESEARCH A1) without a
    live call: returns a mapping from canonical role name to the actual
    column name found in *df*.

    Raises:
        ValueError: naming every required role that could not be located.
    """
    lower_to_actual = {str(c).lower(): c for c in df.columns}
    mapping: dict[str, str] = {}
    missing: list[str] = []
    for role in _REQUIRED_VINTAGE_ROLES:
        if role in lower_to_actual:
            mapping[role] = lower_to_actual[role]
        else:
            missing.append(role)
    if missing:
        raise ValueError(
            "get_series_all_releases() response is missing required column(s): "
            f"{', '.join(missing)}. Found columns: {list(df.columns)}"
        )
    return mapping


def fetch_vintage_series(fred: Fred, series_id: str) -> pd.DataFrame:
    """Pull the full revision history for one series in ONE bulk call.

    Never loop ``get_series_as_of_date()`` per historical date — this is the
    documented anti-pattern (RESEARCH Pattern 2). Column schema is validated
    defensively (A1) before returning.
    """
    all_releases = fred.get_series_all_releases(series_id)
    _detect_vintage_columns(all_releases)  # fail fast on an unexpected schema
    return all_releases


def value_as_of(all_releases: pd.DataFrame, as_of_date: pd.Timestamp) -> pd.Series:
    """Reconstruct the point-in-time series knowable at *as_of_date*.

    For each reference period (``date``), keeps the vintage with the latest
    ``realtime_start`` that is still ``<= as_of_date`` — the value actually
    published by that date. Rows with ``realtime_start > as_of_date`` are
    invisible, so a later revision can never leak backward in time.
    """
    cols = _detect_vintage_columns(all_releases)
    rs_col, date_col, value_col = cols["realtime_start"], cols["date"], cols["value"]

    known = all_releases[all_releases[rs_col] <= as_of_date]
    if known.empty:
        return pd.Series(dtype=float, name=value_col)

    known = known.sort_values(rs_col).groupby(date_col).tail(1)
    result = known.set_index(date_col)[value_col]
    result.index.name = "date"
    return result


def align_with_fallback(
    all_releases: pd.DataFrame,
    as_of_dates: pd.DatetimeIndex,
    shift_series: pd.Series,
) -> pd.Series:
    """Point-in-time value per as-of date, with the D-06 pre-vintage fallback.

    For each date in *as_of_dates*: if it precedes the series' earliest
    recorded vintage, returns the corresponding value from the publication-
    lag-shifted *shift_series* (D-06's documented accepted compromise — never
    NaN, never a raised error). Otherwise the value is the latest reference
    period known by that as-of date (per :func:`value_as_of`) — the most
    recently *published* observation, correctly vintage-corrected — mirroring
    what a `shift()`-aligned series does for un-vintaged series (Pitfall 4:
    vintage-correction subsumes the shift once vintages exist), falling back
    to *shift_series* only if no reference period is known at all yet.
    """
    cols = _detect_vintage_columns(all_releases)
    rs_col = cols["realtime_start"]
    earliest_vintage = pd.to_datetime(all_releases[rs_col]).min()

    values: dict[pd.Timestamp, float] = {}
    for as_of in pd.DatetimeIndex(as_of_dates):
        if as_of < earliest_vintage:
            values[as_of] = shift_series.get(as_of, float("nan"))
            continue
        known = value_as_of(all_releases, as_of)
        if known.empty:
            values[as_of] = shift_series.get(as_of, float("nan"))
        else:
            values[as_of] = known.loc[known.index.max()]

    result = pd.Series(values)
    result.index.name = "date"
    result.name = cols["value"]
    return result


def fetch_all_vintages(cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Fetch bulk all-releases DataFrames for every configured D-06 series.

    Config shape expected (mirrors ``fred.py::fetch_all``'s series-loop
    pattern, config-driven per ``cfg["fred_vintage"]["series"]``):
        fred_vintage:
          series:
            GDPC1:
              name: fred_gdp
              tier: agency

    Graceful degradation mirrors ``fred.py::_fetch_task`` — a failed series
    is logged at WARNING and simply absent from the result, never raises.
    Never logs the API key itself, only series IDs/friendly names (RESEARCH
    Security Domain).
    """
    api_key = cfg["fred_vintage"].get("api_key")
    if not api_key:
        raise OSError("FRED_API_KEY is not set")

    fred = Fred(api_key=api_key)
    series_cfg: dict = cfg["fred_vintage"]["series"]

    def _fetch_task(series_id: str, meta: dict) -> tuple[str, pd.DataFrame | None]:
        friendly_name = meta["name"]
        log.info("Fetching ALFRED all-releases for %-10s → %s", series_id, friendly_name)
        try:
            return friendly_name, fetch_vintage_series(fred, series_id)
        except Exception as exc:  # noqa: BLE001 — fredapi raises various types
            log.warning("Failed to fetch vintage history for %s (%s): %s", friendly_name, series_id, exc)
            return friendly_name, None

    results: dict[str, pd.DataFrame] = {}
    with ThreadPoolExecutor(max_workers=min(_MAX_WORKERS, max(len(series_cfg), 1))) as pool:
        futures = {pool.submit(_fetch_task, sid, meta): sid for sid, meta in series_cfg.items()}
        for future in as_completed(futures):
            friendly_name, df = future.result()
            if df is not None:
                results[friendly_name] = df

    log.info("ALFRED vintage fetch complete: %d/%d series", len(results), len(series_cfg))
    return results
