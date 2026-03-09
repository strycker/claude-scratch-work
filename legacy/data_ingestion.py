"""
Data Ingestion: scrape multpl.com, pull FRED API data, load Grok labels.

Each function returns a clean quarterly DataFrame indexed by date strings.
All three sources are merged into a single DataFrame by `merge_all_sources()`.
"""

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from lxml import html as HTMLParser

try:
    from dotenv import load_dotenv
    from fredapi import Fred
except ImportError:
    load_dotenv = None
    Fred = None

from trading_crab.config import (
    FRED_SERIES,
    GROK_EXCEL_DATE,
    MULTPL_DATASETS,
    START_DATE,
    END_DATE,
    TODAY_STR,
)

log = logging.getLogger(__name__)


# ===================================================================
# Multpl.com scraping
# ===================================================================

def scrape_multpl_table(url: str) -> list[list[str]]:
    """Scrape a two-column HTML data table from multpl.com."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/63.0.3239.108 Safari/537.36"
        )
    }
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    parsed = HTMLParser.fromstring(response.content.decode("utf-8"))
    rows = parsed.cssselect("#datatable tr")
    return [[td.text.strip() for td in row.cssselect("td")] for row in rows[1:]]


def parse_multpl_series(
    raw_rows: list[list[str]], short_name: str, value_type: str
) -> pd.Series:
    """
    Convert raw [date_str, value_str] rows into a quarterly pd.Series.

    Handles percent signs, unit suffixes, commas, and empty strings.
    Percents are stored in decimal form (e.g. 5% → 0.05).
    """
    df = pd.DataFrame(raw_rows, columns=["date", short_name])
    df["date"] = pd.to_datetime(df["date"], format="%b %d, %Y")

    # Strip unit suffixes before numeric conversion
    suffix_map = {"percent": "%", "million": " million", "trillion": " trillion"}
    if value_type in suffix_map:
        df[short_name] = df[short_name].str.replace(
            suffix_map[value_type], "", regex=False
        )

    df[short_name] = (
        df[short_name]
        .replace("", np.nan)
        .str.replace(",", "", regex=False)
        .astype(float)
    )

    if value_type == "percent":
        df[short_name] /= 100.0

    return df.dropna().set_index("date")[short_name].resample("QE").last()


def fetch_multpl_data() -> pd.DataFrame:
    """Scrape all datasets from multpl.com and return a quarterly DataFrame."""
    log.info("Scraping %d datasets from multpl.com...", len(MULTPL_DATASETS))
    series_list = []
    for short_name, desc, url, value_type in MULTPL_DATASETS:
        log.debug("  %s (%s)", short_name, desc)
        raw = scrape_multpl_table(url)
        series_list.append(parse_multpl_series(raw, short_name, value_type))

    df = pd.concat(series_list, axis=1)
    df.columns = [row[0] for row in MULTPL_DATASETS]
    df.index = df.index.strftime("%Y-%m-%d")
    log.info("  multpl.com: %d rows × %d cols", *df.shape)
    return df


# ===================================================================
# FRED API
# ===================================================================

def fetch_fred_data() -> pd.DataFrame:
    """Pull all configured series from the FRED API."""
    if load_dotenv is None or Fred is None:
        raise ImportError("fredapi and python-dotenv are required. "
                          "Install with: pip install fredapi python-dotenv")
    load_dotenv()
    api_key = os.getenv("FRED_API_KEY")
    if api_key is None:
        log.warning("FRED_API_KEY not found in environment variables.")
    fred = Fred(api_key=api_key)

    log.info("Fetching %d series from FRED API...", len(FRED_SERIES))
    columns = []
    for series_id, meta in FRED_SERIES.items():
        log.debug("  %s → %s", series_id, meta["name"])
        s = fred.get_series(series_id)
        if meta["shift"]:
            s = s.shift(1)
        columns.append(s.resample("QE").last().rename(meta["name"]))

    df = pd.concat(columns, axis=1)
    df.index = df.index.strftime("%Y-%m-%d")
    log.info("  FRED: %d rows × %d cols", *df.shape)
    return df


# ===================================================================
# Grok regime classifications
# ===================================================================

def load_grok_labels(data_dir: Path) -> pd.DataFrame:
    """Load Grok quarter classifications from an Excel file."""
    filepath = data_dir / f"grok_quarter_classifications_{GROK_EXCEL_DATE}.xlsx"
    log.info("Loading Grok labels from %s", filepath)
    df = pd.read_excel(filepath)
    df.columns = df.columns.str.strip()
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
    df["market_code"] = df["primary_class"].astype("category").cat.codes
    df = df.rename(columns={"quarter": "date"}).set_index("date")
    df.index = df.index.astype(str)
    log.info("  Grok: %d rows, %d unique regimes", len(df), df["market_code"].nunique())
    return df


# ===================================================================
# Merge + derived ratios
# ===================================================================

def merge_all_sources(
    multpl_df: pd.DataFrame, fred_df: pd.DataFrame, grok_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge multpl, FRED, and Grok data into one quarterly DataFrame.

    Also computes derived cross-asset ratios (credit spread, price/GDP, etc.)
    and filters to the configured date range.
    """
    log.info("Merging datasets...")
    df = pd.concat([multpl_df, fred_df, grok_df[["market_code"]]], axis=1)
    df["market_code"] = df["market_code"].fillna(-1).astype(int)
    df = df.loc[START_DATE:END_DATE]

    # Derived cross-asset ratios
    df["div_yield2"]       = df["dividend"] / df["sp500"]
    df["price_div"]        = df["sp500"] / df["dividend"]
    df["price_gdp"]        = df["sp500"] / df["gdp"]
    df["price_gdp2"]       = df["sp500"] / df["fred_gdp"]
    df["price_gnp2"]       = df["sp500"] / df["fred_gnp"]
    df["div_minus_baa"]    = df["div_yield"] - df["fred_baa"] / 100.0
    df["credit_spread"]    = (df["fred_baa"] - df["fred_aaa"]) / 100.0
    df["real_price2"]      = df["sp500"] / df["cpi"]
    df["real_price3"]      = df["sp500"] / df["fred_cpi"]
    df["real_price_gdp2"]  = df["sp500_adj"] / df["gdp"]

    log.info("  Merged: %d rows × %d cols (date range %s to %s)",
             *df.shape, df.index[0], df.index[-1])
    return df


# ===================================================================
# Checkpoint helpers
# ===================================================================

def save_checkpoint(df: pd.DataFrame, data_dir: Path, name: str) -> None:
    """Save a DataFrame as both CSV and pickle."""
    csv_path = data_dir / f"{name}_{TODAY_STR}.csv"
    pkl_path = data_dir / f"{name}_{TODAY_STR}.pickle"
    df.to_csv(csv_path)
    df.to_pickle(pkl_path)
    log.info("  Saved checkpoint: %s (.csv + .pickle)", name)


def load_checkpoint(data_dir: Path, name: str, snapshot_date: str) -> pd.DataFrame:
    """Load a DataFrame from a pickle checkpoint."""
    pkl_path = data_dir / f"{name}_{snapshot_date}.pickle"
    log.info("  Loading checkpoint: %s", pkl_path)
    return pd.read_pickle(pkl_path)
