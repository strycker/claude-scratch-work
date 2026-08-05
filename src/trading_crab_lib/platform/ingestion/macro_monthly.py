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
try/except-WARNING skeleton. multpl valuation anchors
(``_scrape_multpl_monthly``) reuse ``multpl.py``'s importable raw-row
scraper and value-parsing constants. macrotrends long-history commodities
(``_scrape_macrotrends_monthly``) reuse ``macrotrends.py``'s importable
``_extract_json_data`` JSON extractor. ``fetch_macro_monthly`` merges every
source into ONE wide DataFrame via ``pd.concat([...], axis=1)`` — an outer,
NULL-tolerant join (RESEARCH Pitfall 5) — never ``pd.merge``/``.join``
defaults, which would silently drop months only one source covers.

Usage:
    from trading_crab_lib.platform.ingestion.macro_monthly import fetch_macro_monthly
    from trading_crab_lib.platform.config import load_platform_config

    cfg = load_platform_config()
    macro = fetch_macro_monthly(cfg)
"""

from __future__ import annotations

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from io import StringIO
from typing import Any

import numpy as np
import pandas as pd

try:
    from fredapi import Fred
except ImportError as _err:
    raise ImportError(
        "fredapi is required for FRED data ingestion. "
        "Install with: pip install 'trading-crab-lib[ingestion]'"
    ) from _err

from trading_crab_lib.ingestion import macrotrends, multpl
from trading_crab_lib.ingestion.http import browser_session, http_get

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


# ── multpl monthly ───────────────────────────────────────────────────────────


def _parse_multpl_series_monthly(raw_rows: list, short_name: str, value_type: str) -> pd.Series:
    """Monthly analog of ``multpl.py::_parse_series`` — same value parsing
    (suffix stripping, comma removal, percent-to-decimal), but resamples to
    ``"ME"`` instead of the incumbent's hardcoded quarterly period-end rule.
    Reuses ``multpl._SUFFIX_MAP`` rather than duplicating the constant.
    """
    df = pd.DataFrame(raw_rows, columns=["date", short_name])
    df["date"] = pd.to_datetime(df["date"], format="%b %d, %Y")

    suffix = multpl._SUFFIX_MAP.get(value_type)
    if suffix:
        df[short_name] = df[short_name].str.replace(suffix, "", regex=False)

    df[short_name] = (
        df[short_name]
        .replace("", np.nan)
        .str.replace(",", "", regex=False)
        .astype(float)
    )

    if value_type == "percent":
        df[short_name] /= 100.0

    return (
        df.dropna()
        .set_index("date")[short_name]
        .resample("ME")
        .last()
    )


def _scrape_multpl_monthly(cfg: dict[str, Any]) -> dict[str, pd.Series]:
    """
    Scrape every dataset in cfg["multpl_monthly"]["datasets"] at monthly
    cadence, reusing ``multpl.py``'s importable raw-row scraper
    (``_scrape_raw_rows``) and value-parsing convention (``_SUFFIX_MAP``)
    — never editing the frozen incumbent module.

    Degrades gracefully (WARNING + skip) when ``cssselect`` — multpl's
    optional dependency — is unavailable, matching the incumbent multpl
    test's skip behavior. Keeps multpl's 2s rate-limit sleep between requests.

    Returns:
        dict mapping short_name -> monthly pd.Series (missing entries on
        scrape/parse failure — callers merge with a NULL-tolerant concat).
    """
    datasets: list = cfg.get("multpl_monthly", {}).get("datasets", [])
    if not datasets:
        log.warning("No multpl_monthly datasets configured — skipping")
        return {}

    results: dict[str, pd.Series] = {}
    for entry in datasets:
        short_name, _desc, url, value_type = entry
        log.info("Scraping multpl (monthly) %-24s  %s", short_name, url)
        try:
            raw_rows = multpl._scrape_raw_rows(url)
            s = _parse_multpl_series_monthly(raw_rows, short_name, value_type)
            s.name = short_name
            results[short_name] = s
        except ImportError as exc:
            log.warning("multpl scrape unavailable for %s (cssselect missing?): %s", short_name, exc)
        except Exception as exc:  # noqa: BLE001 — network/parsing libraries raise various types
            log.warning("Failed to scrape multpl %s: %s", short_name, exc)
        time.sleep(multpl.RATE_LIMIT_SECONDS)

    return results


# ── macrotrends monthly ──────────────────────────────────────────────────────


def _scrape_macrotrends_html_table_monthly(
    html: str,
    column_name: str,
    resample_method: str,
) -> pd.Series:
    """Monthly analog of ``macrotrends.py::_scrape_series_html_table`` — same
    ``pandas.read_html`` fallback parsing, but resamples to ``"ME"`` instead
    of the incumbent's hardcoded quarterly period-end rule. The incumbent
    function itself cannot be reused directly since its resample rule is
    baked in (D-01: frozen module, no edits)."""
    tables = pd.read_html(StringIO(html))
    if not tables:
        raise ValueError(f"No HTML tables found for {column_name}")

    df = max(tables, key=len)

    date_col = next(
        (c for c in df.columns if "date" in str(c).lower() or "year" in str(c).lower()),
        df.columns[0],
    )
    value_col = next(
        (c for c in df.columns if "value" in str(c).lower() or "price" in str(c).lower() or "close" in str(c).lower()),
        df.columns[-1],
    )

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[value_col] = pd.to_numeric(
        df[value_col].astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    )
    df = df.dropna(subset=[date_col, value_col])

    s = df.set_index(date_col)[value_col].sort_index()
    s.name = column_name
    s = s[~s.index.duplicated(keep="last")]

    if resample_method == "last":
        return s.resample("ME").last()
    return s.resample("ME").mean()


def _scrape_macrotrends_monthly(
    base_url: str,
    path: str,
    column_name: str,
    resample_method: str,
    session: Any = None,
) -> pd.Series:
    """
    Scrape a single macrotrends series at monthly cadence.

    Reuses ``macrotrends._extract_json_data`` (import, not copy) to parse
    the embedded JSON blob macrotrends pages ship; the parse loop and HTML
    table fallback are duplicated here (not imported) because the
    incumbent's ``_scrape_series``/``_scrape_series_html_table`` bake in a
    hardcoded quarterly resample that must stay frozen (D-01) — this analog
    resamples to ``"ME"`` instead.

    Fetches through ``ingestion/http.py``'s browser-impersonating client, not
    plain ``requests``: macrotrends fronts its pages with a Cloudflare bot
    check that answers a plain-``requests`` TLS fingerprint with an
    interstitial. That interstitial carries no ``<table>`` and no embedded
    JSON, so it surfaces here as a "No tables found" parse failure rather
    than as an HTTP error — the request succeeded, the page just was not the
    data. *session* is threaded in by the caller so one impersonating session
    is reused across every series.
    """
    url = f"{base_url}{path}"
    log.info("Scraping macrotrends (monthly): %s → %s", column_name, url)

    # No headers= — macrotrends.HEADERS carries a hardcoded Chrome/120 UA that
    # would override the impersonating client's own matched header set.
    resp = http_get(url, session=session, timeout=30)
    resp.raise_for_status()

    data = macrotrends._extract_json_data(resp.text)
    if data is None or len(data) == 0:
        log.debug("No embedded JSON found — trying pandas.read_html for %s", column_name)
        return _scrape_macrotrends_html_table_monthly(resp.text, column_name, resample_method)

    sample = data[0]
    date_key = next((k for k in sample if "date" in k.lower()), None)
    value_key = next(
        (k for k in sample if k.lower() in ("close", "value", "v1", "v2")),
        None,
    )
    if date_key is None or value_key is None:
        keys = list(sample.keys())
        if len(keys) >= 2:
            date_key, value_key = keys[0], keys[1]
        else:
            raise ValueError(f"Cannot identify date/value keys in macrotrends JSON: {keys}")

    dates = []
    values = []
    for row in data:
        d = row.get(date_key)
        v = row.get(value_key)
        if d is None or v is None:
            continue
        try:
            if isinstance(v, str):
                v = re.sub(r"<[^>]+>", "", v).replace(",", "")
            values.append(float(v))
            dates.append(pd.Timestamp(d))
        except (ValueError, TypeError):
            continue

    if not dates:
        raise ValueError(f"No parseable data found for {column_name}")

    s = pd.Series(values, index=pd.DatetimeIndex(dates), name=column_name)
    s = s.sort_index()
    s = s[~s.index.duplicated(keep="last")]

    if resample_method == "last":
        return s.resample("ME").last()
    return s.resample("ME").mean()


def _fetch_macrotrends_monthly_all(cfg: dict[str, Any]) -> dict[str, pd.Series]:
    """
    Scrape every series in cfg["macrotrends_monthly"]["series"] at monthly
    cadence. Keeps macrotrends' 3s rate-limit sleep between requests.

    Returns:
        dict mapping series name -> monthly pd.Series (missing entries on
        scrape/parse failure — callers merge with a NULL-tolerant concat).
    """
    mt_cfg = cfg.get("macrotrends_monthly", {})
    base_url = mt_cfg.get("base_url", "https://www.macrotrends.net")
    series_list_cfg = mt_cfg.get("series", [])

    # One impersonating session reused across every series (see
    # _scrape_macrotrends_monthly for why plain requests is bot-blocked here).
    session = browser_session()

    results: dict[str, pd.Series] = {}
    for entry in series_list_cfg:
        name = entry["name"]
        path = entry["path"]
        resample_method = entry.get("resample", "mean")
        try:
            s = _scrape_macrotrends_monthly(base_url, path, name, resample_method, session=session)
            if not s.empty:
                results[name] = s
        except Exception as exc:  # noqa: BLE001 — network libraries raise various types
            log.warning("Failed to scrape macrotrends %s: %s", name, exc)
        time.sleep(macrotrends.RATE_LIMIT_SECONDS)

    return results


# ── Orchestrator ─────────────────────────────────────────────────────────────


def fetch_macro_monthly(cfg: dict[str, Any]) -> pd.DataFrame:
    """
    Fetch FRED monthly market series, multpl valuation anchors, and
    macrotrends long-history commodities, then merge ALL series into ONE
    wide monthly DataFrame.

    Merges exclusively via ``pd.concat([...], axis=1)`` (outer join —
    NULL-tolerant, RESEARCH Pitfall 5) — never ``pd.merge``/``DataFrame.join``
    defaults, which would silently drop months only one source covers.

    Returns:
        DataFrame indexed by month-end dates, one column per series. Empty
        DataFrame if every source failed.
    """
    frames: list[pd.Series] = []

    fred_df = fetch_fred_monthly(cfg)
    if not fred_df.empty:
        frames.extend(fred_df[col] for col in fred_df.columns)

    frames.extend(_scrape_multpl_monthly(cfg).values())
    frames.extend(_fetch_macrotrends_monthly_all(cfg).values())

    if not frames:
        log.warning("macro_monthly: no series fetched successfully")
        return pd.DataFrame()

    df = pd.concat(frames, axis=1)
    df.index.name = "date"
    log.info(
        "macro_monthly fetch complete: %d months, %d series",
        len(df), len(df.columns),
    )
    return df
