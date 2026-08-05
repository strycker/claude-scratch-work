"""
macrotrends.net historical price scraper.

Fetches long-history monthly price data for commodities and rates that
pre-date ETF inception (gold back to 1915, WTI crude oil back to 1946, etc.).

macrotrends embeds price data as a JavaScript variable in the page source:
    var defined_data = [ ... ];
rather than serving it via a separate API endpoint.  This module extracts
that JSON blob using a regex, parses it into a DataFrame, and resamples to
quarterly frequency.

macrotrends.net fronts its pages with a Cloudflare bot check (a "Just a
moment..." interstitial returned as HTTP 403 to plain ``requests`` clients,
detected by TLS fingerprint). Fetches go through the shared
browser-impersonating client (``trading_crab_lib.ingestion.http``) to defeat
that check.

Two-step fetch per series: the plain HTTP path above is tried FIRST — it is
cheap, and becomes correct again the day Cloudflare stops interfering. Only
when that HTTP body carries neither the embedded JSON array nor a parseable
HTML table (i.e. it was an interstitial, not the page) does this module fall
back to rendering the page in a real headless browser
(``trading_crab_lib.ingestion.browser.fetch_page_html``). Whether a real
browser gets past THIS PARTICULAR Cloudflare deployment on THIS PARTICULAR
page is not confirmed by this module's unit tests, which mock the browser
call entirely — see the ``<human-check>`` in the quick task that added this
fallback for the only way to answer that.

A residential-connection diagnostic (2026-08-05) confirmed a real headless
browser DOES reach this page (HTTP 200, real content, no interstitial), and
also confirmed two things the fallback design must account for: (1) the
embedded-JSON regex (``_DATA_PATTERN``) does NOT match on the rendered page
— the series lives inside a Highcharts closure, not a named ``window``
global — so the rendered HTML is expected to fall through to the
``pandas.read_html`` table path, not the JSON path; (2) the page's only
``<table>`` has no distinguishing class (plain ``class="table"``, not
``historical_data_table`` as an earlier third-party snippet assumed) and its
header cell text can be a squashed multi-line label — see
``BROWSER_WAIT_SELECTOR`` and ``_scrape_series_html_table``'s column
detection below.

Rate-limited to 3 seconds per request (polite scraping).

Usage:
    from trading_crab_lib.ingestion.macrotrends import fetch_all
    prices = fetch_all(cfg)
"""

from __future__ import annotations

import json
import logging
import re
import time
from io import StringIO
from typing import Any

import pandas as pd

from trading_crab_lib.ingestion.browser import fetch_page_html
from trading_crab_lib.ingestion.http import browser_session, http_get

log = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}
RATE_LIMIT_SECONDS = 3.0

# Selector passed to the browser fallback with require_selector=False (never
# as a hard requirement). Updated 2026-08-05 from a residential-connection
# diagnostic against the live page: a third-party snippet had previously
# assumed ``table.historical_data_table``, but that class does NOT exist on
# the rendered page — the actual DOM has a plain, undistinguished
# ``<table class="table">``. require_selector=False means a miss (e.g. a
# genuine interstitial, or a macrotrends page type with no table at all)
# still returns the rendered HTML rather than discarding it, since the data
# may be reachable some other way even without a matching selector.
BROWSER_WAIT_SELECTOR = "table"

# Regex to find the embedded data array in the page source.
# macrotrends stores chart data as:  var defined_data = [{...}, ...];
# Some pages use slightly different variable names, so we match broadly.
_DATA_PATTERN = re.compile(
    r'var\s+\w*[Dd]ata\s*=\s*(\[\s*\{.*?\}\s*\])\s*;',
    re.DOTALL,
)


# Default series configuration.  Each entry: (column_name, url_path, resample_method).
# resample_method: "mean" for prices (average monthly price), "last" for rates.
DEFAULT_SERIES: list[tuple[str, str, str]] = [
    (
        "gold_spot",
        "/1333/historical-gold-prices-100-year-chart",
        "mean",
    ),
    (
        "wti_crude",
        "/1369/crude-oil-price-history-chart",
        "mean",
    ),
]


def _extract_json_data(html: str) -> list[dict] | None:
    """Extract embedded JSON data array from macrotrends page source."""
    match = _DATA_PATTERN.search(html)
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return None


def _html_yields_data(html: str) -> bool:
    """True when *html* carries either the embedded JSON array or a table
    ``pandas.read_html`` can parse.

    A Cloudflare interstitial carries NEITHER — which is exactly why a
    browser render is worth its cost as a fallback, and exactly why a
    response that DOES carry either must never pay that cost. Imported by
    ``platform/ingestion/macro_monthly.py`` (not copied) so both modules
    share this single decision.
    """
    if _extract_json_data(html):
        return True
    try:
        tables = pd.read_html(StringIO(html))
    except (ValueError, ImportError):
        return False
    return bool(tables)


def _scrape_series(
    base_url: str,
    url_path: str,
    column_name: str,
    resample_method: str,
    session: Any = None,
) -> pd.Series:
    """
    Scrape a single macrotrends series and return quarterly data.

    The embedded JSON typically has keys like "date" and "value" (or "v1", "v2").
    We try common key patterns to find the date and close/value columns.
    """
    url = f"{base_url}{url_path}"
    log.info("Scraping macrotrends: %s → %s", column_name, url)

    # No headers= here on purpose: HEADERS carries a hardcoded Chrome/120 UA
    # that would override the impersonating client's own matched header set and
    # reintroduce a UA/TLS-fingerprint mismatch. http_get applies browser
    # headers itself on the plain-requests fallback path.
    resp = http_get(url, session=session, timeout=30)
    resp.raise_for_status()

    html = resp.text
    if not _html_yields_data(html):
        log.warning(
            "macrotrends HTTP response for %s carried neither embedded JSON nor a parseable "
            "table (bot interstitial?) — falling back to a rendered page", column_name,
        )
        rendered = fetch_page_html(url, wait_for_selector=BROWSER_WAIT_SELECTOR, require_selector=False)
        if rendered:
            html = rendered

    data = _extract_json_data(html)
    if data is None or len(data) == 0:
        # Fallback: try pandas.read_html for table-based pages
        log.debug("No embedded JSON found — trying pandas.read_html for %s", column_name)
        return _scrape_series_html_table(html, column_name, resample_method)

    # Detect column keys — macrotrends uses varying schemas
    sample = data[0]
    date_key = next((k for k in sample if "date" in k.lower()), None)
    value_key = next(
        (k for k in sample if k.lower() in ("close", "value", "v1", "v2")),
        None,
    )
    if date_key is None or value_key is None:
        # Fall back to positional: first key = date, second = value
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
            # Some values have HTML tags or commas
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
        return s.resample("QE").last()
    return s.resample("QE").mean()


def _scrape_series_html_table(
    html: str,
    column_name: str,
    resample_method: str,
) -> pd.Series:
    """Fallback: parse an HTML table from the page using pandas.read_html.

    Column header text is matched by substring, not exact name, because
    macrotrends header phrasing varies by page (and can be a squashed
    multi-line label per a 2026-08-05 residential diagnostic, e.g. a value
    column literally titled "Gold PricesMonthly Closing Price" — still
    caught by the "price" substring). "month" is matched alongside
    "date"/"year" because some macrotrends tables title their date column
    plainly "Month", which neither of those catches — a column that only
    matches by falling back to ``df.columns[0]`` is a silent trap if the
    date column is ever NOT first.
    """
    tables = pd.read_html(StringIO(html))
    if not tables:
        raise ValueError(f"No HTML tables found for {column_name}")

    # Use the largest table (most likely the data table)
    df = max(tables, key=len)

    # Find value column FIRST, then date column excluding it — a squashed
    # header like "Gold PricesMonthly Closing Price" contains "month" as a
    # substring of "Monthly", so if date detection ran first (or considered
    # this column a candidate) it would misidentify the VALUE column as the
    # date column. Excluding value_col from the date search resolves that
    # collision without narrowing either keyword set.
    value_col = next(
        (c for c in df.columns if "value" in str(c).lower() or "price" in str(c).lower() or "close" in str(c).lower()),
        df.columns[-1],
    )
    date_col = next(
        (
            c for c in df.columns
            if c != value_col
            and ("date" in str(c).lower() or "year" in str(c).lower() or "month" in str(c).lower())
        ),
        next((c for c in df.columns if c != value_col), df.columns[0]),
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
        return s.resample("QE").last()
    return s.resample("QE").mean()


def fetch_all(cfg: dict[str, Any]) -> pd.DataFrame:
    """
    Scrape all configured macrotrends series and return a quarterly DataFrame.

    Config shape (optional — uses DEFAULT_SERIES if not present):
        macrotrends:
          base_url: "https://www.macrotrends.net"
          series:
            - name: gold_spot
              path: "/1333/historical-gold-prices-100-year-chart"
              resample: mean
            - name: wti_crude
              path: "/1369/crude-oil-price-history-chart"
              resample: mean

    Returns:
        DataFrame indexed by quarter-end dates, one column per series.
    """
    mt_cfg = cfg.get("macrotrends", {})
    base_url = mt_cfg.get("base_url", "https://www.macrotrends.net")

    series_list_cfg = mt_cfg.get("series")
    if series_list_cfg:
        series_defs = [
            (s["name"], s["path"], s.get("resample", "mean"))
            for s in series_list_cfg
        ]
    else:
        series_defs = DEFAULT_SERIES

    frames: list[pd.Series] = []
    session = browser_session()

    for column_name, url_path, resample_method in series_defs:
        try:
            s = _scrape_series(base_url, url_path, column_name, resample_method, session=session)
            if not s.empty:
                frames.append(s)
                log.info(
                    "macrotrends %s: %d quarters (%s → %s)",
                    column_name, len(s),
                    s.index[0].strftime("%Y-Q%q") if len(s) > 0 else "?",
                    s.index[-1].strftime("%Y-Q%q") if len(s) > 0 else "?",
                )
        except Exception as exc:  # noqa: BLE001 — network libraries raise various types
            log.warning("Failed to scrape macrotrends %s: %s", column_name, exc)
        time.sleep(RATE_LIMIT_SECONDS)

    if not frames:
        log.warning("macrotrends: no series fetched successfully")
        return pd.DataFrame()

    df = pd.concat(frames, axis=1)
    df.index.name = "date"
    log.info(
        "macrotrends fetch complete: %d quarters, %d series",
        len(df), len(df.columns),
    )
    return df
