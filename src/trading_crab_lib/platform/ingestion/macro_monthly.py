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
scraper and value-parsing constants. World Bank monthly commodity prices
(``_fetch_worldbank_monthly``, gold) read the "Pink Sheet" workbook. macrotrends commodities
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
from io import BytesIO, StringIO
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
from trading_crab_lib.ingestion.browser import fetch_page_html
from trading_crab_lib.ingestion.http import browser_session, http_get
from trading_crab_lib.platform.ingestion import prices_daily

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
    monthly_freq: str = "ME",
) -> pd.Series:
    """Pull one FRED series and resample it to month-end.

    Monthly analog of ``fred.py::_fetch_one`` — same client and pull, but
    resamples to ``monthly_freq`` (month-end, ``"ME"``) instead of the
    incumbent's hardcoded quarterly period-end rule (RESEARCH Pitfall 1).
    No publication lag here: lags are applied once, downstream, from
    ``cfg['publication_lags']`` (``publication_lags.apply_publication_lags``).
    """
    raw = fred.get_series(series_id, observation_start=start, observation_end=end)
    return raw.resample(monthly_freq).last()


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
        log.info("Fetching FRED (monthly) %-10s → %s", series_id, friendly_name)
        try:
            s = _fetch_fred_monthly(fred, series_id, start, end, monthly_freq)
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
    baked in (D-01: frozen module, no edits).

    Date-column matching includes "month" alongside "date"/"year" — a
    2026-08-05 residential diagnostic against the live page found a "Month"
    date column, which neither of the other two keywords catches, leaving
    only the ``df.columns[0]`` fallback (a silent trap if the date column is
    ever not first). Value column is detected FIRST and excluded from the
    date search: a squashed header like "Gold PricesMonthly Closing Price"
    contains "month" as a substring of "Monthly", which would otherwise
    misidentify the value column as the date column."""
    tables = pd.read_html(StringIO(html))
    if not tables:
        raise ValueError(f"No HTML tables found for {column_name}")

    df = max(tables, key=len)

    # Content-based detection, imported (not copied) from macrotrends.py — the
    # live table reads as TWO IDENTICALLY-NAMED columns, so no header keyword
    # can distinguish them. See that helper for the full explanation.
    date_col, value_col = macrotrends._detect_date_and_value_columns(df, column_name)

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[value_col] = pd.to_numeric(macrotrends._clean_numeric(df[value_col]), errors="coerce")

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

    When the HTTP body carries neither the embedded JSON array nor a
    parseable table (``macrotrends._html_yields_data`` returns False —
    imported, not copied, per D-01), falls back to rendering the page in a
    real browser (``ingestion.browser.fetch_page_html``,
    ``require_selector=False`` since ``macrotrends.BROWSER_WAIT_SELECTOR`` is
    an unconfirmed candidate selector). The rendered HTML then goes through
    the exact same JSON-then-table parse chain as the HTTP body — a rendered
    page gets no parsing privilege the HTTP path lacks. Whether a real
    browser gets past this Cloudflare deployment is confirmed (2026-08-05
    residential diagnostic, HTTP 200 on the real page); whether the
    end-to-end parse of THIS series produces correct data is NOT — this
    module's unit tests mock the browser call entirely.
    """
    url = f"{base_url}{path}"
    log.info("Scraping macrotrends (monthly): %s → %s", column_name, url)

    # No headers= — macrotrends.HEADERS carries a hardcoded Chrome/120 UA that
    # would override the impersonating client's own matched header set.
    resp = http_get(url, session=session, timeout=30)
    resp.raise_for_status()

    html = resp.text
    if not macrotrends._html_yields_data(html):
        log.warning(
            "macrotrends HTTP response for %s carried neither embedded JSON nor a parseable "
            "table (bot interstitial?) — falling back to a rendered page", column_name,
        )
        rendered = fetch_page_html(
            url, wait_for_selector=macrotrends.BROWSER_WAIT_SELECTOR, require_selector=False
        )
        if rendered:
            html = rendered

    data = macrotrends._extract_json_data(html)
    if data is None or len(data) == 0:
        log.debug("No embedded JSON found — trying pandas.read_html for %s", column_name)
        return _scrape_macrotrends_html_table_monthly(html, column_name, resample_method)

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
            log.warning("Failed to scrape macrotrends %s: %s%s", name, exc, network_hint(exc))
        time.sleep(macrotrends.RATE_LIMIT_SECONDS)

    return results


def network_hint(exc: BaseException) -> str:
    """A plain-English cause for a blocked or throttled fetch, or ``""`` (08.4 UAT, corporate VPN)."""
    text = f"{type(exc).__name__}: {exc}"
    if "403" in text:
        return (
            " — blocked (HTTP 403): the site refuses this network, as it does on many corporate "
            "VPNs, firewalls and cloud hosts. Run the build off the VPN."
        )
    if "RateLimit" in text or "Too Many Requests" in text or "429" in text:
        return " — rate-limited: wait an hour and retry, or run off the VPN (a shared exit IP is throttled)."
    return ""


# ── World Bank monthly commodity prices (08.4 UAT, 2026-10-09) ───────────────
#
# The "Pink Sheet": monthly averages of daily prices in nominal USD, 1960-01 onward, free and
# keyless, re-issued early each month. Its download URL changes with each issue, so the
# commodity-markets page is read for the current link first. It is reachable where macrotrends
# (HTTP 403) is not: corporate VPNs and cloud hosts.

_WORLDBANK_XLSX = re.compile(r"https://thedocs\.worldbank\.org/[^\"'\s]*CMO-Historical-Data-Monthly\.xlsx")


def parse_worldbank_monthly(content: bytes, columns: list[str]) -> dict[str, pd.Series]:
    """One month-end Series per named column of the Pink Sheet's "Monthly Prices" sheet.

    The sheet has title rows, a header row naming each commodity (``Gold``), a units row,
    then one row per month labelled ``1960M01``. Missing values are written "…" and become NaN.

    Raises:
        ValueError: if no header row names any of ``columns``.
    """
    sheet = pd.read_excel(BytesIO(content), sheet_name="Monthly Prices", header=None)
    for row in range(min(len(sheet), 12)):
        header = sheet.iloc[row].tolist()
        if any(name in header for name in columns):
            break
    else:
        raise ValueError(f"World Bank 'Monthly Prices' sheet: no header row names any of {columns}")
    body = sheet.iloc[row + 1:]
    body = body[body.iloc[:, 0].astype(str).str.fullmatch(r"\d{4}M\d{2}").to_numpy()]
    index = pd.DatetimeIndex(
        pd.to_datetime(body.iloc[:, 0].str.replace("M", "-"), format="%Y-%m") + pd.offsets.MonthEnd(0), name="date"
    )
    results: dict[str, pd.Series] = {}
    for name in columns:
        if name not in header:
            log.warning("World Bank 'Monthly Prices' sheet has no column %r — skipped", name)
            continue
        values = pd.to_numeric(body.iloc[:, header.index(name)], errors="coerce").to_numpy(dtype=float)
        results[name] = pd.Series(values, index=index).dropna()
    return results


def _fetch_worldbank_monthly(cfg: dict[str, Any]) -> dict[str, pd.Series]:
    """Fetch every ``cfg['worldbank_monthly']['series']`` entry, renamed to its ``name``.

    A failed fetch logs a WARNING and returns what it has (nothing), as for every other source;
    the build's fail-loud gate then names the missing column.
    """
    wb_cfg = cfg.get("worldbank_monthly") or {}
    series_cfg: list = wb_cfg.get("series", [])
    if not series_cfg:
        return {}
    page_url = wb_cfg["page_url"]
    try:
        page = http_get(page_url)
        page.raise_for_status()
        link = _WORLDBANK_XLSX.search(page.text)
        if link is None:
            raise ValueError(f"no CMO-Historical-Data-Monthly.xlsx link found on {page_url}")
        log.info("Fetching World Bank (monthly) %s", link.group(0))
        book = http_get(link.group(0), timeout=120.0)
        book.raise_for_status()
        parsed = parse_worldbank_monthly(book.content, [entry["column"] for entry in series_cfg])
    except ImportError as exc:
        log.warning("World Bank prices need openpyxl (pip install openpyxl): %s", exc)
        return {}
    except Exception as exc:  # noqa: BLE001 — network/parsing libraries raise various types
        log.warning("Failed to fetch World Bank monthly prices: %s%s", exc, network_hint(exc))
        return {}
    return {
        entry["name"]: parsed[entry["column"]].rename(entry["name"])
        for entry in series_cfg
        if entry["column"] in parsed
    }


# ── Index month-end closes (08.3, P&L only) ────────────────────────────────


def _fetch_index_month_end(cfg: dict[str, Any]) -> dict[str, pd.Series]:
    """Fetch every ticker in ``cfg['index_monthly']`` as a month-end close, renamed.

    A no-op when the block is absent. A failed or empty fetch logs a WARNING and
    the column is simply absent, as for every other source. These columns are
    P&L only (``pnl_only: true``): ``transforms_monthly.features_from_raw`` drops
    them before ``monthly_features`` is assembled.
    """
    index_cfg: dict = cfg.get("index_monthly") or {}
    if not index_cfg:
        return {}
    start = cfg["data"]["start_date"]
    end = cfg["data"].get("end_date") or str(date.today())
    monthly_freq = cfg["data"].get("monthly_freq", "ME")
    try:
        monthly = prices_daily.fetch_yfinance_month_end(list(index_cfg), start, end, monthly_freq)
    except Exception as exc:  # noqa: BLE001 — network libraries raise various types
        log.warning("Failed to fetch index month-end closes %s: %s", list(index_cfg), exc)
        return {}

    results: dict[str, pd.Series] = {}
    for ticker, meta in index_cfg.items():
        name = meta["name"]
        if ticker not in monthly.columns or monthly[ticker].dropna().empty:
            log.warning(
                "Index month-end fetch returned nothing for %s (%s) — column absent. Yahoo rate-limits or "
                "blocks many shared networks (corporate VPNs among them): retry later, or off the VPN.",
                name,
                ticker,
            )
            continue
        results[name] = monthly[ticker].rename(name)
    return results


# ── Orchestrator ─────────────────────────────────────────────────────────────


def fetch_macro_monthly(cfg: dict[str, Any]) -> pd.DataFrame:
    """
    Fetch FRED monthly market series, multpl valuation anchors, macrotrends
    commodities, World Bank monthly commodity prices and index month-end closes (``index_monthly``),
    then merge ALL series into ONE wide monthly DataFrame.

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
    frames.extend(_fetch_worldbank_monthly(cfg).values())
    frames.extend(_fetch_index_month_end(cfg).values())

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
