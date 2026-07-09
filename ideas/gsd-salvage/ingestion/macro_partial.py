"""
Fill missing macro_raw columns by fetching only the FRED or multpl side needed.

When ``data/raw/macro_raw.parquet`` is stale (e.g. FRED-only snapshot) the pipeline
used to discard it and re-fetch everything.  This module merges missing columns
onto the existing frame so a partial network call can satisfy ``required_for_step2``
without ``--refresh``.
"""

from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger(__name__)

# Columns required for step 2 transforms (cross-ratios, etc.) — keep in sync with
# ``run_pipeline.step1_ingest`` / ``_repair_macro_raw_missing_columns``.
REQUIRED_MACRO_RAW_FOR_STEP2: frozenset[str] = frozenset(
    {
        "dividend",
        "sp500",
        "gdp",
        "fred_gdp",
        "fred_gnp",
        "div_yield",
        "fred_baa",
        "fred_aaa",
        "cpi",
        "fred_cpi",
        "sp500_adj",
    }
)


def fred_column_names(cfg: dict) -> set[str]:
    """Return FRED series column names from ``cfg["fred"]["series"]``."""
    return {meta["name"] for meta in cfg["fred"]["series"].values()}


def multpl_column_names(cfg: dict) -> set[str]:
    """Return multpl short names (first column of each dataset row) from config."""
    return {str(row[0]) for row in cfg["multpl"]["datasets"]}


def merge_missing_macro_columns(
    combined: pd.DataFrame,
    missing: set[str],
    cfg: dict,
) -> pd.DataFrame:
    """
    For each name in ``missing`` that exists in config, pull that series from a
    targeted ``fetch_all`` and align it to ``combined.index``.

    * multpl gaps → one multpl scrape (full table, but no FRED round-trip when
      only multpl columns are missing).
    * FRED gaps → one FRED fetch when only FRED columns are missing.
    * Both → multpl first, then FRED.
    """
    if not missing:
        return combined
    fred_n = fred_column_names(cfg)
    multpl_n = multpl_column_names(cfg)
    unknown = missing - fred_n - multpl_n
    if unknown:
        raise ValueError(
            "merge_missing_macro_columns: columns not defined in fred.series or multpl.datasets: "
            f"{sorted(unknown)}"
        )

    out = combined.copy()
    start = pd.Timestamp(cfg["data"]["start_date"])
    need_multpl = missing & multpl_n
    need_fred = missing & fred_n

    # Network-minimizing: fetch only the missing multpl or FRED columns, not the full universe unless needed.
    if need_multpl:
        from trading_crab_lib.ingestion import multpl as multpl_module

        log.info(
            "Partial macro ingest: scraping multpl for %d missing column(s) only (not full universe)",
            len(need_multpl),
        )
        mdf = multpl_module.fetch_all(cfg, columns=set(need_multpl))
        for col in sorted(need_multpl):
            if col not in mdf.columns:
                log.warning("multpl scrape did not produce expected column %r", col)
                continue
            out[col] = mdf[col].reindex(out.index)

    if need_fred:
        from trading_crab_lib.ingestion import fred as fred_module

        log.info(
            "Partial macro ingest: fetching FRED (%d missing column names)",
            len(need_fred),
        )
        fdf = fred_module.fetch_all(cfg)
        for col in sorted(need_fred):
            if col not in fdf.columns:
                log.warning("FRED ingest did not produce expected column %r", col)
                continue
            out[col] = fdf[col].reindex(out.index)

    out = out[out.index >= start]
    return out
