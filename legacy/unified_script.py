#!/usr/bin/env python
# coding: utf-8


"""
Trading-Crab: Market Regime Classification & Prediction Pipeline
================================================================

Predicts market conditions, optimal portfolios, and stock picks by:

1. SCRAPING macro financial data from multpl.com and the FRED API over a
   multi-decade horizon (quarterly resolution).
2. ENGINEERING features — log transforms, smoothed derivatives (1st–3rd order),
   and cross-asset ratios (e.g. S&P priced in gold, credit spreads).
3. FILLING GAPS via Bernstein-polynomial interpolation informed by local
   derivatives, so every quarter has a complete feature vector.
4. REDUCING dimensionality with PCA (5 components).
5. CLUSTERING quarters into market-regime labels using both standard KMeans
   and size-constrained KMeans for balanced regimes.
6. EVALUATING cluster quality with silhouette, Calinski-Harabasz, and
   Davies-Bouldin scores across a range of k values.

The resulting regime labels feed downstream supervised models (not in this
script) that predict current/future regimes and rank asset-class performance.
"""


# # Trading-Crab

# ## Predict market conditions, best portfolios, and stock picks

# # Concepts / Main Approach Outline:
# - Scrape public datasets and use free APIs to obtain macro financial data over a 50-year period, ensuring these metrics are still available today if I had to score a model now
# - Assumption: one of the most predictive features in any financial model will be the market conditions... are we in a recession?  A market boom?  A bubble?  A slowly forming top?  High/Low inflation?  Stagflation?  Therefore we want to CLASSIFY (apply unsupervised learning) to our time series datasets on the order of quarters.  Idea would be to get roughly equally-sized clusters that have distinct behaviors
# - Once we have the time-series classified according to variance techniques, we want to PREDICT today's classification using data available to us TODAY.  This means we want to construct a SUPERVISED learning model that, given features known only at that time &mdash; nothing forward-looking or revised &mdash; we have a notion of what market condition regime we are in
# - Even more powerfully, we can also construct supervised learning models to predict whether certain classifications will occur in the next quarter, next year, next 2 years, etc.  For example, if we are in a boom period, what are the chances that we'll experience a recession in the next 2 years?
# - Once we have good predictions for market conditions and some rough models for predicting future conditions, we can then try to predict the value of various asset classes (or ETFs), either each relative to cash (USD) or relative to each other (e.g. S&P500 priced in $Gold, or TLT bonds priced in USO oil prices).  This will give us an idea of what assets do best in each PREDICTED market regime (that is, you should be able to rank the assets according to which out-perform or under-perform the others, including cash).  We can use these relative performance models to construct rough portfolio mixes.
# - Putting it all together (Part I):  modeling individual asset performance
#   - Using predicted market current market conditions, future market conditions, and all historic data and derived data (e.g., smoothed first derivative of oil prices measured in gold, etc.), predict the likelihood of whether a given ETF will be +X% at Y quarters in the future.
#   - For example, we might be interested in the likelihood that the S&P will at some point in the next 2 years crash 20%, or separately, be 20% higher.
#   - Note that these models are somewhat independent, particularly in volatile markets.  Models need not sum up to 100% &mdash; you could simultaneously predict that the S&P500 will crash with 80% probability AND with 80% probability rebound to +20% (actually you won't know the order... it might have a blow-off top and THEN crash).
#   - Use these models to build a "stoplight" dashboard... for every asset, what are the probabilities of the asset going up or down as measured in dollars (or relative to another asset)
# - Putting it all together (Part II):  Final project conclusion = actual trading recommendations
#   - Given a portfolio of X assets at Y percentages, the market condition regime, the recommended portfolio mix, the projected performance of each asset (which indicators have recently turned on warning lights), should you buy, sell, or hold that asset?
#   - Send a weekly email (can use AI for this part!) with the final recommendations on portfolio changes &mdash; what assets need traded, bought, or sold THIS WEEK?

# # To Do:
# - Reduce number of rows in the initial dataset, as many do not span the right time range
# - Add historic Gold, Oil, TLT, etc. to datasets &mdash; see https://www.macrotrends.net/
# - Standardize the time range (1950-2025?), infer missing data, throw away or fix anything looking odd
# - Change all variables that are exponential-looking into something normalized and predictive of regime, like taking a logrithm and/or using the 1st, 2nd, and 3rd derivative... likely CHANGE in a variable, or change RELATIVE to another variable is what will be predictive.  For example, the S&P500 itself is not a good signal of regime, but the S&P priced in gold or oil might be.
#   - Consider using smoothed variables / polynomial fits / or other kinds of parameterized versions of the variable features as needed, as some might be too volatile over even quarterly time series
# - For the initial unsupervised clustering phase of the project, can consider using adjusted / revised since I'm only trying to get regimes, so backward-looking features MIGHT be ok
# - Once we have all quarters CLASSIFIED according to k-means or whatever, NOW we can look into
#   - For each regime, find what assets CONSISTENTLY grew, e.g., for every quarter labeled for class X, asset Y always grew each quarter, no negative quarters.  There will likely be noise, but see if you can relax the criteria or find a cleaner signal, then find the right ETFs for the right asset classes or sectors (e.g., stagflation might mean gold is best, growth might mean tech and small caps best, etc.).
#   - SUPERVISED predictive modeling using other features.  At this point you CANNOT use any variable that was revised or otherwise had forward-knowledge of the current or future state.  All features must be values known at the moment we would have been choosing a portfolio.
# - During the supervised learning phase, good to first find feature importance, then reduce the number of features, then try running a single Decision Tree just to get most of the explanatory power before running a Random Forest / XGBoost or whatever is the best final model





# ---------------------------------------------------------------------------
# Imports — stdlib, then third-party, then project-local (PEP 8 order)
# ---------------------------------------------------------------------------
import os
from datetime import datetime

import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix

from dotenv import load_dotenv
from fredapi import Fred
from k_means_constrained import KMeansConstrained
from lxml import html as HTMLParser
from scipy.interpolate import BPoly
import requests
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import ListedColormap





# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

VERBOSE                                 = True
GENERATE_PLOTS                          = False
GENERATE_OPTIONAL_SNS_PAIRPLOT          = False
GENERATE_OPTIONAL_SCATTER_MATRIX_PLOT   = False
REFRESH_SOURCE_DATASETS                 = False
RECOMPUTE_DERIVED_DATASETS              = False

TODAY_STR = datetime.today().strftime("%Y%m%d")
START_DATE = "1950-01-01"
END_DATE = "2025-09-30"

CUSTOM_COLORS = [
    "#0000d0",  # vivid blue
    "#d00000",  # vivid red
    "#f48c06",  # bright orange
    "#8338ec",  # vivid purple
    "#50a000",  # dark yellow-green
]

MY_CMAP = ListedColormap(CUSTOM_COLORS)

DATA_PATH = "../data/"
# DATA_PATH = "../../data/"



# ---------------------------------------------------------------------------
# Multpl.com dataset definitions
# Each entry: [short_name, description, url, value_type]
# value_type controls how the raw string is parsed:
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
# GDP and GNP are shifted by one period so that values reflect the
# information available *at* the quarter end (publication lag).
FRED_SERIES = {
    "GDP":      {"name": "fred_gdp",   "shift": True},
    "GNP":      {"name": "fred_gnp",   "shift": True},
    "BAA":      {"name": "fred_baa",   "shift": False},
    "AAA":      {"name": "fred_aaa",   "shift": False},
    "CPIAUCSL": {"name": "fred_cpi",   "shift": False},
    "GS10":     {"name": "fred_gs10",  "shift": False},
    "TB3MS":    {"name": "fred_tb3ms", "shift": False},
}

# Columns selected for the initial log-transform + feature-selection stage.
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

# Columns selected for PCA / clustering (derivatives + a few levels).
# These were chosen after exploratory analysis; d3 and most raw levels
# were dropped to reduce noise.
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

N_PCA_COMPONENTS = 5
MAX_K_SEARCH = 12       # upper bound for KMeans k sweep
BALANCED_K = 5          # number of balanced-size clusters
K_CAP = 5               # cap on best-k from silhouette search





# ===================================================================
# Helper functions
# ===================================================================

def scrape_multpl_table(url: str) -> list[list[str]]:
    """Scrape a two-column HTML table from multpl.com and return raw rows."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/63.0.3239.108 Safari/537.36"
        )
    }
    response = requests.get(url, headers=headers)
    parsed = HTMLParser.fromstring(response.content.decode("utf-8"))
    rows = parsed.cssselect("#datatable tr")
    return [[td.text.strip() for td in row.cssselect("td")] for row in rows[1:]]


def parse_multpl_series(raw_rows: list, short_name: str, value_type: str) -> pd.Series:
    """
    Convert raw [date_str, value_str] rows into a clean, quarterly pd.Series.

    Handles percent signs, unit suffixes ('million', 'trillion'), commas,
    and empty strings.  Percents are divided by 100 so they're in decimal form.
    """
    df = pd.DataFrame(raw_rows, columns=["date", short_name])
    df["date"] = pd.to_datetime(df["date"], format="%b %d, %Y")

    # Strip unit suffixes before numeric conversion
    suffix_map = {"percent": "%", "million": " million", "trillion": " trillion"}
    if value_type in suffix_map:
        df[short_name] = df[short_name].str.replace(suffix_map[value_type], "", regex=False)

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
        .resample("QE")
        .last()
    )


def compute_smoothed_derivatives(
    series: pd.Series, dates: list, window: int = 5
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Compute smoothed 1st, 2nd, and 3rd derivatives of a time series.

    Uses centered rolling means to suppress noise, and np.gradient for
    numeric differentiation against a day-number x-axis.
    """
    smoothed = series.rolling(window=window, min_periods=1, center=True).mean()
    x = mdates.date2num(dates) - mdates.date2num(dates).min()

    d1 = pd.Series(np.gradient(smoothed, x), index=dates).rolling(
        window=window, min_periods=1, center=True
    ).mean()
    d2 = pd.Series(np.gradient(d1, x), index=dates).rolling(
        window=window, min_periods=1, center=True
    ).mean()
    d3 = pd.Series(np.gradient(d2, x), index=dates).rolling(
        window=window, min_periods=1, center=True
    ).mean()

    return d1, d2, d3


def interpolate_gaps_with_bernstein(
    quarterly_df: pd.DataFrame, col_name: str
) -> pd.Series:
    """
    Fill interior NaN gaps using Bernstein polynomial interpolation.

    For each contiguous block of NaNs bounded by valid data on both sides,
    a BPoly is fitted through the boundary values and their derivatives.
    Edge gaps (leading/trailing) are extrapolated with a Taylor expansion.
    Returns the filled column as a pd.Series.
    """
    all_dates_str = quarterly_df.index.values
    all_dates = [datetime.strptime(str(d), "%Y-%m-%d").date() for d in all_dates_str]

    valid = quarterly_df[[col_name, "market_code"]].dropna()
    valid_dates = [datetime.strptime(str(d), "%Y-%m-%d").date() for d in valid.index.values]

    d1, d2, d3 = compute_smoothed_derivatives(valid[col_name], valid_dates)

    # Build numeric time axis for interpolation
    t = (mdates.date2num(all_dates_str) - mdates.date2num(all_dates_str).min()).astype("int64")

    y = quarterly_df[col_name].reindex(all_dates_str).copy()
    d1_full = d1.reindex(all_dates).copy()
    d2_full = d2.reindex(all_dates).copy()
    d3_full = d3.reindex(all_dates).copy()

    # Locate contiguous NaN blocks as (start, end) pairs
    mask = y.isna().values
    edges = np.flatnonzero(np.diff(np.r_[0, mask, 0])).reshape(-1, 2)

    for start, end in edges:
        left, right = start - 1, end

        if left >= 0 and right < len(y):
            # Interior gap — Bernstein polynomial through both boundaries
            poly = BPoly.from_derivatives(
                [t[left], t[right]],
                [
                    [y.iloc[left],  d1_full.iloc[left],  d2_full.iloc[left],  d3_full.iloc[left]],
                    [y.iloc[right], d1_full.iloc[right], d2_full.iloc[right], d3_full.iloc[right]],
                ],
            )
            y.iloc[start:right] = poly(t[start:right])

        elif right < len(y):
            # Leading gap — backward Taylor expansion from right boundary
            dt = t[start:right] - t[right]
            y.iloc[start:right] = (
                y.iloc[right]
                + d1_full.iloc[right] * dt
                + d2_full.iloc[right] * dt**2 / 2
                + d3_full.iloc[right] * dt**3 / 6
            )
        else:
            # Trailing gap — forward Taylor expansion from left boundary
            dt = t[left + 1 :] - t[left]
            y.iloc[left + 1 :] = (
                y.iloc[left]
                + d1_full.iloc[left] * dt
                + d2_full.iloc[left] * dt**2 / 2
                + d3_full.iloc[left] * dt**3 / 6
            )

    return y


def evaluate_kmeans(X: np.ndarray, k_range: range) -> pd.DataFrame:
    """
    Run KMeans for each k in k_range and return a DataFrame of scores:
    inertia, silhouette, Calinski-Harabasz, and Davies-Bouldin.
    """
    results = []
    for k in k_range:
        labels = KMeans(n_clusters=k, n_init=50, random_state=0).fit_predict(X)
        results.append({
            "k": k,
            "inertia": KMeans(n_clusters=k, n_init=50, random_state=0).fit(X).inertia_,
            "silhouette": silhouette_score(X, labels),
            "calinski": calinski_harabasz_score(X, labels),
            "davies_bouldin": davies_bouldin_score(X, labels),
        })
    return pd.DataFrame(results)





# ===================================================================
# 1. Scrape multpl.com datasets
# ===================================================================

if REFRESH_SOURCE_DATASETS and RECOMPUTE_DERIVED_DATASETS:
    if VERBOSE:
        print("Scraping multpl.com datasets... ", end="")
    multpl_series = []
    for short_name, _desc, url, value_type in MULTPL_DATASETS:
        raw = scrape_multpl_table(url)
        multpl_series.append(parse_multpl_series(raw, short_name, value_type))

    multpl_df = pd.concat(multpl_series, axis=1)
    multpl_df.columns = [row[0] for row in MULTPL_DATASETS]
    multpl_df.index = multpl_df.index.strftime("%Y-%m-%d")
    multpl_df.to_csv(DATA_PATH + "multpl_datasets_snapshot_" + TODAY_STR + ".csv")
    multpl_df.to_pickle(DATA_PATH + "multpl_datasets_snapshot_" + TODAY_STR + ".pickle")

    if VERBOSE:
        print("done.")

elif RECOMPUTE_DERIVED_DATASETS:

    if VERBOSE:
        print("Loading the multpl.com datasets from pickle file... ", end="")

    multpl_df = pd.read_pickle(DATA_PATH + "multpl_datasets_snapshot_20260216.pickle")

    if VERBOSE:
        print("done.")
        print()

    if GENERATE_PLOTS:
        if VERBOSE:
            print("Creating plots of the multpl.com datasets...")
        for idx, col_name in enumerate(multpl_df.columns):
            fig, ax = plt.subplots(figsize=(8, 5))
            date_strings = multpl_df.index.values
            dates = [datetime.strptime(str(d), '%Y-%m-%d').date() for d in date_strings]
            values = multpl_df[col_name].values
            plt.plot(dates, values, color='blue', linestyle='solid', marker='o') # Plot the data
            formatter = mdates.DateFormatter('%Y-%m-%d')
            ax.xaxis.set_major_formatter(formatter)
            fig.autofmt_xdate()
            # plt.title(multpl_data_desc[idx][1])
            plt.title(col_name)
            plt.grid(True)
            plt.show()

        if VERBOSE:
            print()





# ===================================================================
# 2. Pull FRED API data
# ===================================================================

if REFRESH_SOURCE_DATASETS and RECOMPUTE_DERIVED_DATASETS:
    if VERBOSE:
        print("Fetching FRED API data... ", end="")
    load_dotenv()
    fred_api_key = os.getenv("FRED_API_KEY")
    if fred_api_key is None:
        if VERBOSE:
            print("  ERROR: FRED_API_KEY not found in environment variables.")
            # sys.exit()

    fred = Fred(api_key=fred_api_key)

    fred_columns = []
    for series_id, meta in FRED_SERIES.items():
        s = fred.get_series(series_id)
        if meta["shift"]:
            s = s.shift(1)
        fred_columns.append(s.resample("QE").last().rename(meta["name"]))

    fred_df = pd.concat(fred_columns, axis=1)
    fred_df.index = fred_df.index.strftime("%Y-%m-%d")
    fred_df.to_csv(DATA_PATH + "fred_api_datasets_snapshot_" + TODAY_STR + ".csv")
    fred_df.to_pickle(DATA_PATH + "fred_api_datasets_snapshot_" + TODAY_STR + ".pickle")

    if VERBOSE:
        print("done.")
        print()

elif RECOMPUTE_DERIVED_DATASETS:

    if VERBOSE:
        print("Loading the FRED datasets from pickle file... ", end="")

    fred_df = pd.read_pickle(DATA_PATH + "fred_api_datasets_snapshot_20260216.pickle")

    if VERBOSE:
        print("done.")
        print()

    if GENERATE_PLOTS:
        if VERBOSE:
            print("Creating plots of the FRED datasets...")
        for col_name in fred_df.columns:
            fig, ax = plt.subplots(figsize=(8, 5))
            date_strings = fred_df.index.values
            dates = [datetime.strptime(str(d), '%Y-%m-%d').date() for d in date_strings]
            values = fred_df[col_name].values
            plt.plot(dates, values, color='blue', linestyle='solid', marker='o') # Plot the data
            formatter = mdates.DateFormatter('%Y-%m-%d')
            ax.xaxis.set_major_formatter(formatter)
            fig.autofmt_xdate()
            plt.title(col_name)
            plt.grid(True)
            plt.show()

        if VERBOSE:
            print()




# ===================================================================
# 3. Load Grok regime classifications (external labeling baseline)
# ===================================================================

if REFRESH_SOURCE_DATASETS and RECOMPUTE_DERIVED_DATASETS:
    if VERBOSE:
        print("Loading Grok quarter classifications... ", end="")
    grok_df = pd.read_excel(DATA_PATH + "grok_quarter_classifications_20260201.xlsx")
    grok_df.columns = grok_df.columns.str.strip()
    grok_df = grok_df.map(lambda x: x.strip() if isinstance(x, str) else x)
    grok_df["market_code"] = grok_df["primary_class"].astype("category").cat.codes
    grok_df = grok_df.rename(columns={"quarter": "date"}).set_index("date")
    grok_df.index = grok_df.index.astype(str)
    grok_df.to_csv(DATA_PATH + "grok_quarter_classifications_" + TODAY_STR + ".csv")
    grok_df.to_pickle(DATA_PATH + "grok_quarter_classifications_" + TODAY_STR + ".pickle")

    if VERBOSE:
        print("done.")
        print()

elif RECOMPUTE_DERIVED_DATASETS:

    if VERBOSE:
        print("Loading the Grok quarter classifications from pickle file... ", end="")

    grok_df = pd.read_pickle(DATA_PATH + "grok_quarter_classifications_20260216.pickle")

    if VERBOSE:
        print("done.")
        print()





# ===================================================================
# 4. Merge all sources into a single quarterly DataFrame
# ===================================================================

if RECOMPUTE_DERIVED_DATASETS:

    if VERBOSE:
        print("Merging datasets... ", end="")

    quarterly_df = pd.concat([multpl_df, fred_df, grok_df[["market_code"]]], axis=1)
    quarterly_df["market_code"] = quarterly_df["market_code"].fillna(-1).astype(int)
    quarterly_df = quarterly_df.loc[START_DATE:END_DATE]

    # Derived cross-asset ratios
    quarterly_df["div_yield2"]     = quarterly_df["dividend"] / quarterly_df["sp500"]
    quarterly_df["price_div"]      = quarterly_df["sp500"] / quarterly_df["dividend"]
    quarterly_df["price_gdp"]      = quarterly_df["sp500"] / quarterly_df["gdp"]
    quarterly_df["price_gdp2"]     = quarterly_df["sp500"] / quarterly_df["fred_gdp"]
    quarterly_df["price_gnp2"]     = quarterly_df["sp500"] / quarterly_df["fred_gnp"]
    quarterly_df["div_minus_baa"]  = quarterly_df["div_yield"] - quarterly_df["fred_baa"] / 100.0
    quarterly_df["credit_spread"]  = (quarterly_df["fred_baa"] - quarterly_df["fred_aaa"]) / 100.0
    quarterly_df["real_price2"]    = quarterly_df["sp500"] / quarterly_df["cpi"]
    quarterly_df["real_price3"]    = quarterly_df["sp500"] / quarterly_df["fred_cpi"]
    quarterly_df["real_price_gdp2"] = quarterly_df["sp500_adj"] / quarterly_df["gdp"]

    if VERBOSE:
        print("done.")
        print()





# ===================================================================
# 5. Log-transform exponential columns to stabilize variance
# ===================================================================

if RECOMPUTE_DERIVED_DATASETS:

    if VERBOSE:
        print("Applying log transforms... ", end="")

    LOG_COLUMNS = [
        "cape_shiller", "cpi", "div_yield", "div_yield2", "dividend", "earn",
        "earn_yld", "fed_debt", "fred_cpi", "fred_gdp", "fred_gnp", "gdp",
        "price_div", "price_gdp", "price_gdp2", "price_gnp2", "real_gdp",
        "real_price2", "real_price3", "real_price_gdp2", "sp500", "sp500_adj",
        "us_pop",
    ]
    for col in LOG_COLUMNS:
        quarterly_df[f"log_{col}"] = np.log(quarterly_df[col])

    # Keep only the selected features plus the regime label
    quarterly_df = quarterly_df[INITIAL_FEATURES + ["market_code"]].copy()

    if VERBOSE:
        print("done.")
        print()





# ===================================================================
# 6. Interpolate NaN gaps using Bernstein polynomial fitting
# ===================================================================

if RECOMPUTE_DERIVED_DATASETS:

    if VERBOSE:
        print("Interpolating missing values... ")
        orig_dates_string = [datetime.strptime(str(d), '%Y-%m-%d').date() for d in quarterly_df.index.values]
        # print("min date = " + str(orig_dates_string[0]))
        # print("max date = " + str(orig_dates_string[-1]))

    feature_cols = [c for c in quarterly_df.columns if c != "market_code"]

    for col_name in feature_cols:
        # if VERBOSE:
        #     print(col_name)
        #     print("num values = " + str(int(quarterly_df[col_name].notna().sum())) )
        #     print("num nulls " + str(int(quarterly_df[col_name].isna().sum())) )
        #     print()

        if GENERATE_PLOTS:
            orig_values = quarterly_df[col_name].values
            regimes = quarterly_df["market_code"].values

        if quarterly_df[col_name].isna().any():
            quarterly_df[col_name] = interpolate_gaps_with_bernstein(quarterly_df, col_name)

        if GENERATE_PLOTS:
            # fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(5, 12), sharex=True)
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 8), sharex=True)
            fig.suptitle(col_name)

            ax1.scatter(
                x=orig_dates_string, y=orig_values,
                c=regimes, cmap=MY_CMAP,
                s=10,
                alpha=0.7,
                linestyle='solid', marker='o')
            formatter = mdates.DateFormatter('%Y-%m-%d')
            ax1.xaxis.set_major_formatter(formatter)
            fig.autofmt_xdate()

            # ax2.scatter(
            #     x=dates, y=derivative_smoothed,
            #     c=regimes, cmap=MY_CMAP,
            #     s=10,
            #     alpha=0.7,
            #     linestyle='solid', marker='o')

            # ax3.scatter(
            #     x=dates, y=second_der_smoothed,
            #     c=regimes, cmap=MY_CMAP,
            #     s=10,
            #     alpha=0.7,
            #     linestyle='solid', marker='o')

            # ax4.scatter(
            #     x=dates, y=third_der_smoothed,
            #     c=regimes, cmap=MY_CMAP,
            #     s=10,
            #     alpha=0.7,
            #     linestyle='solid', marker='o')

            # ax5.scatter(
            ax2.scatter(
                x=orig_dates_string, y=quarterly_df[col_name],
                c=quarterly_df['market_code'].values, cmap=MY_CMAP,
                s=10,
                alpha=0.7,
                linestyle='solid', marker='o')

            plt.tight_layout()
            plt.show()

    if VERBOSE:
        print("...done interpolating missing values.")
        print()





# ===================================================================
# 7. Compute smoothed derivatives (d1, d2, d3) for every feature
# ===================================================================

if RECOMPUTE_DERIVED_DATASETS:

    if VERBOSE:
        print("Computing smoothed derivatives...", end="")

    for col_name in list(feature_cols):
        valid = quarterly_df[[col_name, "market_code"]].dropna()
        dates = [datetime.strptime(str(d), "%Y-%m-%d").date() for d in valid.index.values]

        d1, d2, d3 = compute_smoothed_derivatives(valid[col_name], dates)

        quarterly_df[f"{col_name}_d1"] = d1.values
        quarterly_df[f"{col_name}_d2"] = d2.values
        quarterly_df[f"{col_name}_d3"] = d3.values

        # Consolidate fragmented DataFrame periodically
        quarterly_df = quarterly_df.copy()

        if GENERATE_PLOTS:
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(5, 12), sharex=True)
            fig.suptitle(col_name)

            ax1.scatter(
                x=dates, y=quarterly_df[col_name],
                c=quarterly_df["market_code"].values, cmap=MY_CMAP,
                s=10,
                alpha=0.7,
                linestyle='solid', marker='o')
            formatter = mdates.DateFormatter('%Y-%m-%d')
            ax1.xaxis.set_major_formatter(formatter)
            fig.autofmt_xdate()

            ax2.scatter(
                x=dates, y=d1.values,
                c=quarterly_df["market_code"].values, cmap=MY_CMAP,
                s=10,
                alpha=0.7,
                linestyle='solid', marker='o')

            ax3.scatter(
                x=dates, y=d2.values,
                c=quarterly_df["market_code"].values, cmap=MY_CMAP,
                s=10,
                alpha=0.7,
                linestyle='solid', marker='o')

            ax4.scatter(
                x=dates, y=d3.values,
                c=quarterly_df["market_code"].values, cmap=MY_CMAP,
                s=10,
                alpha=0.7,
                linestyle='solid', marker='o')

            plt.tight_layout()
            plt.show()

    # Finalize the prepared dataset with clustering features
    quarterly_df = quarterly_df[CLUSTERING_FEATURES + ["market_code"]].copy()
    quarterly_df.to_csv(DATA_PATH + "prepared_quarterly_data_smoothed_" + TODAY_STR + ".csv")
    quarterly_df.to_pickle(DATA_PATH + "prepared_quarterly_data_smoothed_" + TODAY_STR + ".pickle")

    if VERBOSE:
        print("done.")
        print()

else:
    if VERBOSE:
        print("Loading the prepared quarterly datasets from pickle file... ", end="")

    quarterly_df = pd.read_pickle(DATA_PATH + "prepared_quarterly_data_smoothed_20260301.pickle")

    if VERBOSE:
        print("done.")
        print()





# ===================================================================
# 8. OPTIONAL ANOVA Plots for check correlations
# ===================================================================

if GENERATE_PLOTS and GENERATE_OPTIONAL_SNS_PAIRPLOT:

    if VERBOSE:
        print("Creating an sns.pairplot... ", end="")

    sns.pairplot(
        data = quarterly_df,
        vars = CLUSTERING_FEATURES,
        hue  = "market_code",
        # optional but very useful tweaks:
        # diag_kind = "kde",      # or "hist", "auto"
        # kind      = "scatter",  # default anyway
        # palette   = "Set1",     # or "tab10", "husl", your custom list, etc.
        # height    = 2.5,        # size per panel — keep small if many variables
        # aspect    = 1.1,
        # corner    = True,       # triangular = less redundancy (often cleaner)
        # plot_kws  = {"s": 40, "alpha": 0.7, "edgecolor": "none"},
        #diag_kws  = {"alpha": 0.6}
    );
    plt.show()

    if VERBOSE:
        print("done.")
        print()





# ===================================================================
# 9. OPTIONAL Scatter Plot Matrix for checking correlations
# ===================================================================

if GENERATE_PLOTS and GENERATE_OPTIONAL_SCATTER_MATRIX_PLOT:

    if VERBOSE:
        print("Creating a scatter_matrix...", end="")

    scatter_matrix(
        quarterly_df[CLUSTERING_FEATURES],
        figsize   = (11, 11),
        alpha     = 0.6,
        c         = quarterly_df['market_code'].values,
        cmap      = MY_CMAP,
        diagonal  = 'kde',
        marker    = 'o',
        s         = 25,                       # point size
        hist_kwds = {'bins': 20, 'alpha': 0.7},
        range_padding = 0.05
    )

    # Add a legend manually (scatter_matrix doesn't do it automatically)
    # from matplotlib.lines import Line2D
    # legend_elements = [
    #     Line2D([0], [0], marker='o', color='w', label=label,
    #            markerfacecolor=plt.cm.Set1(i/2), markersize=8)
    #     for i, label in enumerate(['control', 'treatment_A', 'treatment_B'])
    # ]

    # plt.legend(handles=legend_elements, title='Condition',
    #            loc='upper right', bbox_to_anchor=(1.25, 1))

    plt.suptitle("Pairwise relationships colored by experimental condition",
        y=0.95, fontsize=14)
    plt.tight_layout()
    plt.show()

    if VERBOSE:
        print("done.")
        print()





# ===================================================================
# 10. PCA dimensionality reduction
# ===================================================================

if VERBOSE:
    print("Running PCA... ", end="")

X_clean_with_market_code = quarterly_df[CLUSTERING_FEATURES + ['market_code']].dropna()
X_clean = quarterly_df[CLUSTERING_FEATURES].dropna().values
X_scaled = StandardScaler().fit_transform(X_clean)

pca = PCA(n_components=N_PCA_COMPONENTS)
X_reduced = pca.fit_transform(X_scaled)
reduced_df = pd.DataFrame(
    X_reduced,
    columns=[f"PC{i+1}" for i in range(N_PCA_COMPONENTS)],
)

if VERBOSE:
    print("done.")
    print()

if GENERATE_PLOTS:

    fig, ax = plt.subplots(figsize=(4, 3))
    shorter_regimes = X_clean_with_market_code['market_code'].values
    plt.scatter(
        x=reduced_df['PC1'], y=reduced_df['PC2'],
        c=shorter_regimes, cmap=MY_CMAP,
        s=10,
        alpha=0.7,
        linestyle='solid', marker='o')
    plt.grid(True)
    plt.show()

    fig, ax = plt.subplots(figsize=(4, 3))
    plt.scatter(
        x=reduced_df['PC1'], y=reduced_df['PC3'],
        c=shorter_regimes, cmap=MY_CMAP,
        s=10,
        alpha=0.7,
        linestyle='solid', marker='o')
    plt.grid(True)
    plt.show()

    fig, ax = plt.subplots(figsize=(4, 3))
    plt.scatter(
        x=reduced_df['PC1'], y=reduced_df['PC4'],
        c=shorter_regimes, cmap=MY_CMAP,
        s=10,
        alpha=0.7,
        linestyle='solid', marker='o')
    plt.grid(True)
    plt.show()

    fig, ax = plt.subplots(figsize=(4, 3))
    plt.scatter(
        x=reduced_df['PC1'], y=reduced_df['PC5'],
        c=shorter_regimes, cmap=MY_CMAP,
        s=10,
        alpha=0.7,
        linestyle='solid', marker='o')
    plt.grid(True)
    plt.show()

    fig, ax = plt.subplots(figsize=(4, 3))
    plt.scatter(
        x=reduced_df['PC2'], y=reduced_df['PC3'],
        c=shorter_regimes, cmap=MY_CMAP,
        s=10,
        alpha=0.7,
        linestyle='solid', marker='o')
    plt.grid(True)
    plt.show()

    fig, ax = plt.subplots(figsize=(4, 3))
    plt.scatter(
        x=reduced_df['PC2'], y=reduced_df['PC4'],
        c=shorter_regimes, cmap=MY_CMAP,
        s=10,
        alpha=0.7,
        linestyle='solid', marker='o')
    plt.grid(True)
    plt.show()

    fig, ax = plt.subplots(figsize=(4, 3))
    plt.scatter(
        x=reduced_df['PC2'], y=reduced_df['PC5'],
        c=shorter_regimes, cmap=MY_CMAP,
        s=10,
        alpha=0.7,
        linestyle='solid', marker='o')
    plt.grid(True)
    plt.show()

    fig, ax = plt.subplots(figsize=(4, 3))
    plt.scatter(
        x=reduced_df['PC3'], y=reduced_df['PC4'],
        c=shorter_regimes, cmap=MY_CMAP,
        s=10,
        alpha=0.7,
        linestyle='solid', marker='o')
    plt.grid(True)
    plt.show()

    fig, ax = plt.subplots(figsize=(4, 3))
    plt.scatter(
        x=reduced_df['PC3'], y=reduced_df['PC5'],
        c=shorter_regimes, cmap=MY_CMAP,
        s=10,
        alpha=0.7,
        linestyle='solid', marker='o')
    plt.grid(True)
    plt.show()

    fig, ax = plt.subplots(figsize=(4, 3))
    plt.scatter(
        x=reduced_df['PC4'], y=reduced_df['PC5'],
        c=shorter_regimes, cmap=MY_CMAP,
        s=10,
        alpha=0.7,
        linestyle='solid', marker='o')
    plt.grid(True)
    plt.show()

    if VERBOSE:
        print()




# ===================================================================
# 11. KMeans clustering — sweep k, pick best, then balanced version
# ===================================================================

if VERBOSE:
    print("Evaluating cluster counts... ", end="")
X_cluster = StandardScaler().fit_transform(reduced_df.values)

# silhouette_scores = []

# for k in range(2, 12):
#     km = KMeans(n_clusters=k, n_init=50, random_state=0)
#     labels = km.fit_predict(X_cluster)

#     silhouette_scores.append({
#         "k": k,
#         "inertia": km.inertia_,
#         "silhouette": silhouette_score(X_cluster, labels),
#         "calinski": calinski_harabasz_score(X_cluster, labels),
#         "davies_bouldin": davies_bouldin_score(X_cluster, labels)
#     })

# silhouette_score_results = pd.DataFrame(silhouette_scores)

scores = evaluate_kmeans(X_cluster, range(2, MAX_K_SEARCH))
best_k = int(min(K_CAP, scores.loc[scores["silhouette"].idxmax(), "k"]))

# Standard KMeans with the chosen k
reduced_df["cluster"] = KMeans(
    n_clusters=best_k, n_init=100, random_state=42
).fit_predict(X_cluster)

# Size-constrained KMeans for roughly equal-sized regime buckets
n = len(X_cluster)
bucket_size = n // BALANCED_K
balanced_model = KMeansConstrained(
    n_clusters=BALANCED_K,
    size_min=bucket_size - 2,
    size_max=bucket_size + 2,
    random_state=0,
)
reduced_df["balanced_cluster"] = balanced_model.fit_predict(X_cluster)

# Attach the original Grok regime labels for comparison
shorter_regimes = (
    quarterly_df[CLUSTERING_FEATURES + ["market_code"]].dropna()
)["market_code"].values

if VERBOSE:
    print("done.")
    print()
    print("Silhouette scores:")
    # print(silhouette_score_results.sort_values("silhouette", ascending=False))
    print(scores.sort_values("silhouette", ascending=False))
    print()
    print(f"{len(reduced_df)} quarters clustered into {best_k} regimes "
          f"(balanced into {BALANCED_K}).")
    print()
    print(f"PCA explained variance ratios: {pca.explained_variance_ratio_.round(3)}")
    print()

if GENERATE_PLOTS:

    fig, ax = plt.subplots()
    ax.plot(scores.k, scores.silhouette, label='silhouette')
    ax.plot(scores.k, scores.calinski / scores.calinski.max(), label='calinski (scaled)')
    ax.plot(scores.k, 1 / scores.davies_bouldin, label='1/db')
    ax.set_xlabel("k")
    ax.legend()
    plt.show()

    fig, ax = plt.subplots(figsize=(4, 3))
    plt.scatter(
        x=reduced_df['PC1'], y=reduced_df['PC2'],
        c=shorter_regimes, cmap=MY_CMAP,
        s=10,
        alpha=0.7,
        linestyle='solid', marker='o')
    plt.grid(True)
    plt.show()

    fig, ax = plt.subplots(figsize=(4, 3))
    plt.scatter(
        x=reduced_df['PC1'], y=reduced_df['PC2'],
        c=reduced_df['balanced_cluster'], cmap=MY_CMAP,
        s=10,
        alpha=0.7,
        linestyle='solid', marker='o')
    plt.grid(True)
    plt.show()

    fig, ax = plt.subplots(figsize=(4, 3))
    plt.scatter(
        x=reduced_df['PC1'], y=reduced_df['PC3'],
        c=reduced_df['balanced_cluster'], cmap=MY_CMAP,
        s=10,
        alpha=0.7,
        linestyle='solid', marker='o')
    plt.grid(True)
    plt.show()

    fig, ax = plt.subplots(figsize=(4, 3))
    plt.scatter(
        x=reduced_df['PC2'], y=reduced_df['PC3'],
        c=reduced_df['balanced_cluster'], cmap=MY_CMAP,
        s=10,
        alpha=0.7,
        linestyle='solid', marker='o')
    plt.grid(True)
    plt.show()

    fig, ax = plt.subplots(figsize=(4, 3))
    plt.scatter(
        x=reduced_df['PC1'], y=reduced_df['PC4'],
        c=reduced_df['balanced_cluster'], cmap=MY_CMAP,
        s=10,
        alpha=0.7,
        linestyle='solid', marker='o')
    plt.grid(True)
    plt.show()

    fig, ax = plt.subplots(figsize=(4, 3))
    plt.scatter(
        x=reduced_df['PC2'], y=reduced_df['PC4'],
        c=reduced_df['balanced_cluster'], cmap=MY_CMAP,
        s=10,
        alpha=0.7,
        linestyle='solid', marker='o')
    plt.grid(True)
    plt.show()

    fig, ax = plt.subplots(figsize=(4, 3))
    plt.scatter(
        x=reduced_df['PC3'], y=reduced_df['PC4'],
        c=reduced_df['balanced_cluster'], cmap=MY_CMAP,
        s=10,
        alpha=0.7,
        linestyle='solid', marker='o')
    plt.grid(True)
    plt.show()

    fig, ax = plt.subplots(figsize=(4, 3))
    plt.scatter(
        x=reduced_df['PC1'], y=reduced_df['PC5'],
        c=reduced_df['balanced_cluster'], cmap=MY_CMAP,
        s=10,
        alpha=0.7,
        linestyle='solid', marker='o')
    plt.grid(True)
    plt.show()

    fig, ax = plt.subplots(figsize=(4, 3))
    plt.scatter(
        x=reduced_df['PC2'], y=reduced_df['PC5'],
        c=reduced_df['balanced_cluster'], cmap=MY_CMAP,
        s=10,
        alpha=0.7,
        linestyle='solid', marker='o')
    plt.grid(True)
    plt.show()

    fig, ax = plt.subplots(figsize=(4, 3))
    plt.scatter(
        x=reduced_df['PC3'], y=reduced_df['PC5'],
        c=reduced_df['balanced_cluster'], cmap=MY_CMAP,
        s=10,
        alpha=0.7,
        linestyle='solid', marker='o')
    plt.grid(True)
    plt.show()

    fig, ax = plt.subplots(figsize=(4, 3))
    plt.scatter(
        x=reduced_df['PC4'], y=reduced_df['PC5'],
        c=reduced_df['balanced_cluster'], cmap=MY_CMAP,
        s=10,
        alpha=0.7,
        linestyle='solid', marker='o')
    plt.grid(True)
    plt.show()



    for col_name in quarterly_df.columns:
        if col_name == 'market_code':
            continue

        # if VERBOSE:
        #     print(col_name)

        orig_all_dates = quarterly_df.index.values
        orig_all_date_string = [datetime.strptime(str(d), '%Y-%m-%d').date() for d in orig_all_dates]
        values = quarterly_df[col_name].values
        regimes = quarterly_df['market_code'].values
        num_nulls = int(quarterly_df[col_name].isna().sum())


        # if VERBOSE:
        #     print("min date = " + str(orig_all_date_string[0]))
        #     print("max date = " + str(orig_all_date_string[-1]))
        #     print("num nulls " + str(num_nulls) )

        fig, ax = plt.subplots(figsize=(4, 3))
        plt.scatter(
            x=orig_all_date_string, y=values,
            c=reduced_df['balanced_cluster'], cmap=MY_CMAP, # good maps include 'viridis' 'Set1' 'tab10' 'Accent' 'rainbow' 'turbo'
            s=10, # use 20 for a 8x5 plot, but 10 is better for smaller plots
            alpha=0.7,
            linestyle='solid', marker='o')
        formatter = mdates.DateFormatter('%Y-%m-%d')
        ax.xaxis.set_major_formatter(formatter)
        fig.autofmt_xdate()
        plt.title(col_name)
        plt.grid(True)
        plt.show()

        if VERBOSE:
            print()
            print()



if VERBOSE:
    print("Script complete.")
    print()
    print()


