"""
Configuration: constants, feature lists, and CLI argument parsing.

All magic numbers, dataset definitions, and feature selections live here
so that every other module imports from a single source of truth.
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging setup — every module calls `logging.getLogger(__name__)`
# ---------------------------------------------------------------------------
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_DATE_FORMAT = "%H:%M:%S"


def configure_logging(verbose: bool = False) -> None:
    """Set root logger to DEBUG (verbose) or INFO."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the pipeline."""
    p = argparse.ArgumentParser(
        prog="trading_crab",
        description="Trading-Crab: market regime classification pipeline",
    )
    p.add_argument(
        "--refresh", action="store_true",
        help="Re-scrape source data from multpl.com and FRED API",
    )
    p.add_argument(
        "--recompute", action="store_true",
        help="Recompute derived features, log transforms, and derivatives",
    )
    p.add_argument(
        "--plots", action="store_true",
        help="Generate matplotlib plots at each stage",
    )
    p.add_argument(
        "--data-dir", type=str, default="data",
        help="Path to the data directory for checkpoints (default: data/)",
    )
    p.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug-level logging",
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Dates and paths
# ---------------------------------------------------------------------------
TODAY_STR = datetime.today().strftime("%Y%m%d")
START_DATE = "1950-01-01"
END_DATE = "2025-09-30"

# Snapshot dates for loading pre-saved pickle files
SNAPSHOT_DATE_MULTPL = "20260216"
SNAPSHOT_DATE_FRED = "20260216"
SNAPSHOT_DATE_GROK = "20260216"
SNAPSHOT_DATE_PREPARED = "20260301"
GROK_EXCEL_DATE = "20260201"


def data_path(data_dir: str = "data") -> Path:
    """Return a resolved Path to the data directory, creating it if needed."""
    p = Path(data_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
CUSTOM_COLORS = [
    "#0000d0",  # vivid blue
    "#d00000",  # vivid red
    "#f48c06",  # bright orange
    "#8338ec",  # vivid purple
    "#50a000",  # dark yellow-green
]

N_PCA_COMPONENTS = 5

# ---------------------------------------------------------------------------
# Multpl.com dataset definitions
# Each row: [short_name, description, url, value_type]
# value_type controls parsing:
#   'num'      → strip commas, convert to float
#   'percent'  → strip '%', convert, divide by 100
#   'million'  → strip ' million', convert
#   'trillion' → strip ' trillion', convert
# ---------------------------------------------------------------------------
MULTPL_DATASETS = [
    ["book",                 "SP500 Book Value Per Share",        "https://www.multpl.com/s-p-500-book-value/table/by-quarter",                             "num"],
    ["dividend",             "SP500 Dividend",                    "https://www.multpl.com/s-p-500-dividend/table/by-month",                                 "num"],
    ["div_growth",           "SP500 Dividend Growth",             "https://www.multpl.com/s-p-500-dividend-growth/table/by-quarter",                        "percent"],
    ["div_yield",            "SP500 Dividend Yield",              "https://www.multpl.com/s-p-500-dividend-yield/table/by-month",                           "percent"],
    ["earn",                 "SP500 Earnings",                    "https://www.multpl.com/s-p-500-earnings/table/by-month",                                 "num"],
    ["earn_yld",             "SP500 Earnings Yield",              "https://www.multpl.com/s-p-500-earnings-yield/table/by-month",                           "percent"],
    ["earn_growth",          "SP500 Earnings Growth",             "https://www.multpl.com/s-p-500-earnings-growth/table/by-quarter",                        "percent"],
    ["real_earn_growth",     "SP500 Real Earnings Growth",        "https://www.multpl.com/s-p-500-real-earnings-growth/table/by-quarter",                   "percent"],
    ["sp500_pe",             "SP500 PE Ratio",                    "https://www.multpl.com/s-p-500-pe-ratio/table/by-month",                                 "num"],
    ["sp500",                "SP500 Historical Prices",           "https://www.multpl.com/s-p-500-historical-prices/table/by-month",                        "num"],
    ["sp500_adj",            "SP500 Inflation Adjusted Prices",   "https://www.multpl.com/inflation-adjusted-s-p-500/table/by-month",                       "num"],
    ["price_book",           "SP500 Price to Book Value",         "https://www.multpl.com/s-p-500-price-to-book/table/by-quarter",                          "num"],
    ["price_sales",          "SP500 Price to Sales Ratio",        "https://www.multpl.com/s-p-500-price-to-sales/table/by-quarter",                         "num"],
    ["cape_shiller",         "Shiller PE 10 Ratio",               "https://www.multpl.com/shiller-pe/table/by-month",                                       "num"],
    ["sales",                "SP500 Sales Per Share",              "https://www.multpl.com/s-p-500-sales/table/by-quarter",                                  "num"],
    ["sales_growth",         "SP500 Sales Per Share Growth",       "https://www.multpl.com/s-p-500-sales-growth/table/by-quarter",                           "percent"],
    ["real_sales",           "SP500 Real Sales Per Share",         "https://www.multpl.com/s-p-500-real-sales/table/by-quarter",                             "num"],
    ["real_sales_growth",    "SP500 Real Sales Per Share Growth",  "https://www.multpl.com/s-p-500-real-sales-growth/table/by-quarter",                      "percent"],
    ["1mo_ustreas",          "US 1 Month Treasury Rate",           "https://www.multpl.com/1-month-treasury-rate/table/by-month",                            "percent"],
    ["6mo_ustreas",          "US 6 Month Treasury Rate",           "https://www.multpl.com/6-month-treasury-rate/table/by-month",                            "percent"],
    ["1yr_ustreas",          "US 1 Year Treasury Rate",            "https://www.multpl.com/1-year-treasury-rate/table/by-month",                             "percent"],
    ["2yr_ustreas",          "US 2 Year Treasury Rate",            "https://www.multpl.com/2-year-treasury-rate/table/by-month",                             "percent"],
    ["3yr_ustreas",          "US 3 Year Treasury Rate",            "https://www.multpl.com/3-year-treasury-rate/table/by-month",                             "percent"],
    ["5yr_ustreas",          "US 5 Year Treasury Rate",            "https://www.multpl.com/5-year-treasury-rate/table/by-month",                             "percent"],
    ["10yr_ustreas",         "US 10 Year Treasury Rate",           "https://www.multpl.com/10-year-treasury-rate/table/by-month",                            "percent"],
    ["20yr_ustreas",         "US 20 Year Treasury Rate",           "https://www.multpl.com/20-year-treasury-rate/table/by-month",                            "percent"],
    ["30yr_ustreas",         "US 30 Year Treasury Rate",           "https://www.multpl.com/30-year-treasury-rate/table/by-month",                            "percent"],
    ["5yr_real_int",         "US 5 Year Real Interest Rate",       "https://www.multpl.com/5-year-real-interest-rate/table/by-month",                        "percent"],
    ["10yr_real_int",        "US 10 Year Real Interest Rate",      "https://www.multpl.com/10-year-real-interest-rate/table/by-month",                       "percent"],
    ["20yr_real_int",        "US 20 Year Real Interest Rate",      "https://www.multpl.com/20-year-real-interest-rate/table/by-month",                       "percent"],
    ["30yr_real_int",        "US 30 Year Real Interest Rate",      "https://www.multpl.com/30-year-real-interest-rate/table/by-month",                       "percent"],
    ["cpi",                  "US Consumer Price Index (CPI)",      "https://www.multpl.com/cpi/table/by-month",                                              "num"],
    ["fed_debt",             "US Federal Debt Percent",            "https://www.multpl.com/u-s-federal-debt-percent/table/by-year",                          "percent"],
    ["gdp",                  "US GDP",                             "https://www.multpl.com/us-gdp/table/by-year",                                            "trillion"],
    ["gdp_growth",           "US GDP Growth Rate",                 "https://www.multpl.com/us-gdp-growth-rate/table/by-quarter",                             "percent"],
    ["real_gdp",             "US Real GDP",                        "https://www.multpl.com/us-gdp-inflation-adjusted/table/by-quarter",                      "trillion"],
    ["real_gdp_growth",      "US Real GDP Growth Rate",            "https://www.multpl.com/us-real-gdp-growth-rate/table/by-quarter",                        "percent"],
    ["real_gdp_per_cap",     "US Real GDP Per Capita",             "https://www.multpl.com/us-real-gdp-per-capita/table/by-quarter",                         "num"],
    ["us_home_prices",       "US Home Prices",                     "https://www.multpl.com/case-shiller-home-price-index-inflation-adjusted/table/by-month", "num"],
    ["us_avg_income",        "US Average Income",                  "https://www.multpl.com/us-average-income/table/by-year",                                 "num"],
    ["us_med_income",        "US Median Income",                   "https://www.multpl.com/us-median-income/table/by-year",                                  "num"],
    ["us_med_income_growth", "US Median Income Growth",            "https://www.multpl.com/us-median-income-growth/table/by-year",                           "percent"],
    ["us_med_real_income",   "US Median Real Income",              "https://www.multpl.com/us-median-real-income/table/by-year",                             "num"],
    ["us_infl",              "US Inflation Rate",                  "https://www.multpl.com/inflation/table/by-month",                                        "percent"],
    ["us_pop",               "US Population",                      "https://www.multpl.com/united-states-population/table/by-month",                         "million"],
    ["us_pop_growth",        "US Population Growth Rate",          "https://www.multpl.com/us-population-growth-rate/table/by-month",                        "percent"],
]

# FRED series IDs and the column names we give them.
# GDP/GNP are shifted by one period to reflect publication lag.
FRED_SERIES = {
    "GDP":      {"name": "fred_gdp",   "shift": True},
    "GNP":      {"name": "fred_gnp",   "shift": True},
    "BAA":      {"name": "fred_baa",   "shift": False},
    "AAA":      {"name": "fred_aaa",   "shift": False},
    "CPIAUCSL": {"name": "fred_cpi",   "shift": False},
    "GS10":     {"name": "fred_gs10",  "shift": False},
    "TB3MS":    {"name": "fred_tb3ms", "shift": False},
}

# ---------------------------------------------------------------------------
# Feature lists
# ---------------------------------------------------------------------------

# Columns to log-transform (exponential growth → linear scale)
LOG_COLUMNS = [
    "cape_shiller", "cpi", "div_yield", "div_yield2", "dividend", "earn",
    "earn_yld", "fed_debt", "fred_cpi", "fred_gdp", "fred_gnp", "gdp",
    "price_div", "price_gdp", "price_gdp2", "price_gnp2", "real_gdp",
    "real_price2", "real_price3", "real_price_gdp2", "sp500", "sp500_adj",
    "us_pop",
]

# Features kept after log transforms (input to derivative computation)
INITIAL_FEATURES = [
    "10yr_ustreas", "credit_spread", "div_minus_baa",
    "fred_aaa", "fred_baa", "fred_gs10", "fred_tb3ms",
    "gdp_growth", "log_cape_shiller", "log_cpi", "log_div_yield",
    "log_div_yield2", "log_dividend", "log_earn", "log_earn_yld",
    "log_fed_debt", "log_fred_cpi", "log_fred_gdp", "log_fred_gnp",
    "log_gdp", "log_price_div", "log_price_gdp", "log_price_gdp2",
    "log_price_gnp2", "log_real_gdp", "log_real_price2", "log_real_price3",
    "log_real_price_gdp2", "log_sp500", "log_sp500_adj", "log_us_pop",
    "real_gdp_growth", "real_gdp_per_cap", "sp500_pe", "us_infl",
    "us_pop_growth",
]

# Features selected for PCA / clustering (mostly derivatives)
CLUSTERING_FEATURES = [
    "10yr_ustreas_d1", "10yr_ustreas_d2",
    "credit_spread", "credit_spread_d1",
    "div_minus_baa_d1", "div_minus_baa_d2",
    "fred_aaa_d1", "fred_aaa_d2",
    "fred_baa_d1", "fred_baa_d2",
    "fred_gs10_d1", "fred_gs10_d2",
    "fred_tb3ms_d1", "fred_tb3ms_d2",
    "gdp_growth", "gdp_growth_d1",
    "log_cape_shiller_d1", "log_cape_shiller_d2",
    "log_cpi_d1", "log_cpi_d2",
    "log_div_yield_d1", "log_div_yield_d2",
    "log_div_yield2_d1", "log_div_yield2_d2",
    "log_dividend_d1", "log_dividend_d2",
    "log_earn_d1", "log_earn_d2",
    "log_earn_yld_d1", "log_earn_yld_d2",
    "log_fed_debt_d1", "log_fed_debt_d2",
    "log_fred_cpi_d1", "log_fred_cpi_d2",
    "log_fred_gdp_d1", "log_fred_gdp_d2",
    "log_fred_gnp_d1", "log_fred_gnp_d2",
    "log_gdp_d1", "log_gdp_d2",
    "log_price_div_d1", "log_price_div_d2",
    "log_price_gdp_d1", "log_price_gdp_d2",
    "log_price_gdp2_d1", "log_price_gdp2_d2",
    "log_price_gnp2_d1", "log_price_gnp2_d2",
    "log_real_gdp_d1", "log_real_gdp_d2",
    "log_real_price2_d1", "log_real_price2_d2",
    "log_real_price3_d1", "log_real_price3_d2",
    "log_real_price_gdp2_d1", "log_real_price_gdp2_d2",
    "log_sp500_d1", "log_sp500_d2", "log_sp500_d3",
    "log_sp500_adj_d1", "log_sp500_adj_d2", "log_sp500_adj_d3",
    "real_gdp_growth", "real_gdp_growth_d1",
    "real_gdp_per_cap_d1", "real_gdp_per_cap_d2",
    "sp500_pe", "sp500_pe_d1",
    "us_infl", "us_infl_d1",
]

# Key features used for regime profiling/interpretation
REGIME_PROFILE_FEATURES = [
    "gdp_growth", "real_gdp_growth", "us_infl", "sp500_pe",
    "credit_spread", "10yr_ustreas_d1", "log_sp500_d1",
    "log_cpi_d1", "log_earn_d1",
]

# ---------------------------------------------------------------------------
# Clustering hyperparameters
# ---------------------------------------------------------------------------
MAX_K_SEARCH = 12       # upper bound for KMeans k sweep
BALANCED_K = 5          # number of balanced-size clusters
K_CAP = 5               # cap on best-k from silhouette search
SMOOTHING_WINDOW = 5    # rolling window for derivative smoothing
