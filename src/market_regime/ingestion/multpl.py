"""
multpl.com scraper.

Each page hosts a table of historical values (monthly or quarterly).
We scrape, parse, resample to quarterly, and return a Series per table.

Usage:
    from market_regime.ingestion.multpl import fetch_all
    df = fetch_all(cfg)
"""

import logging
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup

log = logging.getLogger(__name__)

BASE_URL = "https://www.multpl.com/{slug}/table/by-quarter"
HEADERS = {"User-Agent": "market-regime-research/0.1 (educational use)"}
RATE_LIMIT_SECONDS = 2.0  # be polite


def fetch_table(slug: str) -> pd.Series:
    """Scrape a single multpl.com table page and return a quarterly Series."""
    url = BASE_URL.format(slug=slug)
    log.info("Scraping %s", url)
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "lxml")
    table = soup.find("table", {"id": "datatable"})
    if table is None:
        raise ValueError(f"No #datatable found at {url}")

    rows = []
    for tr in table.find_all("tr")[1:]:  # skip header
        cols = tr.find_all("td")
        if len(cols) < 2:
            continue
        date_str = cols[0].get_text(strip=True)
        val_str = cols[1].get_text(strip=True).replace(",", "").replace("%", "")
        try:
            rows.append((pd.to_datetime(date_str), float(val_str)))
        except (ValueError, TypeError):
            continue

    if not rows:
        raise ValueError(f"Parsed 0 rows from {url}")

    s = pd.Series(dict(rows)).sort_index()
    s = s.resample("QE").last()
    return s


def fetch_all(cfg: dict) -> pd.DataFrame:
    """
    Scrape every table in cfg["multpl"]["tables"].

    Returns:
        DataFrame indexed by quarter-end dates, columns = friendly names.
    """
    tables: dict[str, str] = cfg.get("multpl", {}).get("tables", {})
    if not tables:
        log.warning("No multpl tables configured — skipping")
        return pd.DataFrame()

    frames: dict[str, pd.Series] = {}
    for name, slug in tables.items():
        try:
            frames[name] = fetch_table(slug)
            frames[name].name = name
        except Exception as exc:
            log.warning("Failed to scrape multpl/%s: %s", slug, exc)
        time.sleep(RATE_LIMIT_SECONDS)

    df = pd.DataFrame(frames)
    df.index.name = "date"
    log.info("multpl fetch complete: %d quarters, %d series", len(df), len(df.columns))
    return df
