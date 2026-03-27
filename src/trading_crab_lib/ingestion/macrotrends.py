"""
macrotrends.net historical price scraper.

Fetches long-history monthly price data for commodities and rates that
pre-date ETF inception (gold back to 1915, WTI crude oil back to 1946, etc.).

macrotrends embeds price data as a JavaScript variable in the page source:
    var defined_data = [ ... ];
rather than serving it via a separate API endpoint.  This module extracts
that JSON blob using a regex, parses it into a DataFrame, and resamples to
quarterly frequency.

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

import pandas as pd
import requests

log = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}
RATE_LIMIT_SECONDS = 3.0

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


def _scrape_series(
    base_url: str,
    url_path: str,
    column_name: str,
    resample_method: str,
) -> pd.Series:
    """
    Scrape a single macrotrends series and return quarterly data.

    The embedded JSON typically has keys like "date" and "value" (or "v1", "v2").
    We try common key patterns to find the date and close/value columns.
    """
    url = f"{base_url}{url_path}"
    log.info("Scraping macrotrends: %s → %s", column_name, url)

    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()

    data = _extract_json_data(resp.text)
    if data is None or len(data) == 0:
        # Fallback: try pandas.read_html for table-based pages
        log.debug("No embedded JSON found — trying pandas.read_html for %s", column_name)
        return _scrape_series_html_table(resp.text, column_name, resample_method)

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
    """Fallback: parse an HTML table from the page using pandas.read_html."""
    from io import StringIO
    tables = pd.read_html(StringIO(html))
    if not tables:
        raise ValueError(f"No HTML tables found for {column_name}")

    # Use the largest table (most likely the data table)
    df = max(tables, key=len)

    # Find date and value columns
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
        return s.resample("QE").last()
    return s.resample("QE").mean()


def fetch_all(cfg: dict) -> pd.DataFrame:
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

    for column_name, url_path, resample_method in series_defs:
        try:
            s = _scrape_series(base_url, url_path, column_name, resample_method)
            if not s.empty:
                frames.append(s)
                log.info(
                    "macrotrends %s: %d quarters (%s → %s)",
                    column_name, len(s),
                    s.index[0].strftime("%Y-Q%q") if len(s) > 0 else "?",
                    s.index[-1].strftime("%Y-Q%q") if len(s) > 0 else "?",
                )
        except Exception as exc:
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
