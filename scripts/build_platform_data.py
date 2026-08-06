#!/usr/bin/env python
"""build_platform_data.py — Build the Phase 1 monthly data checkpoints.

The platform backtest/report machinery (Phase 5) reads checkpoints the
Phase 1 data layer produces. Only ``data/raw/`` is gitignored — the platform
checkpoints under ``data/checkpoints/platform/`` ARE tracked in git — but a
fresh clone's tracked copy can still be stale or empty (e.g. a degraded
source wrote a 0-row frame before Task 1's never-lose-coverage protection
existed). Re-running this script refreshes them, so
``python -m trading_crab_lib.platform.evaluation.report`` can run end-to-end.

    FileNotFoundError: .../data/checkpoints/platform/monthly_features.parquet

This script runs ``build_monthly_spine()`` once to fetch the real sources and
write ``daily_raw``, ``monthly_raw``, and ``monthly_features`` into the platform
checkpoint namespace, so the report CLI can then run end-to-end.

Data sources (all free; only FRED needs a key):
  - FRED           (needs FRED_API_KEY in your environment / .env)
  - multpl.com     (public scrape — S&P valuation anchors)
  - macrotrends.net (public scrape — long-history gold/oil)
  - Yahoo Finance  (yfinance — daily universe ETF/equity prices, no key)

Requires outbound network access to those hosts. Run it from an environment with
normal internet (a laptop), NOT a locked-down CI/sandbox that blocks Yahoo/macrotrends.

Usage:
    python scripts/build_platform_data.py
"""

from __future__ import annotations

import logging
import os
import sys

import pandas as pd


def check_price_coverage(daily_raw: pd.DataFrame | None) -> str | None:
    """Return None when *daily_raw* has both rows and columns, else a
    specific failure message naming what was wrong.

    Pure — no I/O. This is the assertion the build script uses to decide
    whether a price universe actually came through, replacing a substring
    heuristic that matched almost any column name and let the build print
    "BUILD OK" over a 0x0 frame.
    """
    if daily_raw is None:
        return "daily_raw checkpoint is absent (never written — the fetch never ran or failed before saving)"
    if len(daily_raw) == 0:
        return "daily_raw has zero rows — the price fetch returned nothing"
    if len(daily_raw.columns) == 0:
        return "daily_raw has zero columns — the price fetch returned nothing"
    return None


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("build_platform_data")

    # Load .env so FRED_API_KEY (and any other secrets) are available before we
    # check for them — mirrors load_platform_config()/config.py. Without this the
    # check below runs before python-dotenv has populated os.environ, so a key
    # that IS present in .env looks missing. Env vars already set win over .env.
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass  # python-dotenv is a core dep; if absent, fall back to real os.environ

    if not os.environ.get("FRED_API_KEY"):
        log.error(
            "FRED_API_KEY is not set. Add it to your environment or .env "
            "(free key: https://fred.stlouisfed.org/docs/api/api_key.html) and re-run."
        )
        return 2

    from trading_crab_lib.platform.checkpoints import get_platform_checkpoint_manager
    from trading_crab_lib.platform.config import load_platform_config
    from trading_crab_lib.platform.transforms_monthly import build_monthly_spine

    cfg = load_platform_config()
    start = cfg["data"]["start_date"]
    end = cfg["data"].get("end_date") or "today"
    log.info("Building monthly spine %s → %s (fetching FRED + multpl + macrotrends + yfinance)...", start, end)

    monthly_features = build_monthly_spine(cfg)

    cm = get_platform_checkpoint_manager()

    from trading_crab_lib.platform.snapshots import DEFAULT_SNAPSHOT_NAMES, is_snapshot_backed

    for snapshot_name in DEFAULT_SNAPSHOT_NAMES:
        if is_snapshot_backed(cm, snapshot_name):
            log.warning(
                "build_platform_data: checkpoint '%s' is STILL SNAPSHOT-BACKED (offline dev data, "
                "not live) after this build — either its live source never wrote fresh data, or "
                "merge-on-save (Task 1) refused an empty fetch and preserved the snapshot "
                "untouched. Run `python scripts/platform_snapshot.py list` for the capture date.",
                snapshot_name,
            )

    checkpoint_dir = cm.checkpoint_dir if hasattr(cm, "checkpoint_dir") else "data/checkpoints/platform/"
    log.info("Wrote platform checkpoints to: %s", checkpoint_dir)
    log.info(
        "monthly_features: %d months, %d columns, %s → %s",
        len(monthly_features),
        len(monthly_features.columns),
        monthly_features.index.min().date() if len(monthly_features) else "n/a",
        monthly_features.index.max().date() if len(monthly_features) else "n/a",
    )

    # Sanity: the report needs actual price coverage. A substring heuristic used to
    # scan column names for price-like tokens, but "tr" matches almost any column
    # name and "equit" matches the multpl-derived equities_tr column (not produced
    # by price ingestion at all) — that let the build print BUILD OK over a 0x0
    # daily_raw. Load the real daily_raw checkpoint and assert it has rows+columns.
    try:
        daily_raw = cm.load("daily_raw")
    except FileNotFoundError:
        daily_raw = None

    daily_raw_failure = check_price_coverage(daily_raw)
    monthly_features_failure = check_price_coverage(monthly_features)
    if daily_raw_failure is not None or monthly_features_failure is not None:
        for msg in (daily_raw_failure, monthly_features_failure):
            if msg is not None:
                log.error(msg)
        log.error(
            "a data source was unreachable — re-run from a machine with normal internet."
        )
        return 1

    print("\nBUILD OK — you can now run the real backtest:")
    print("    python -m trading_crab_lib.platform.evaluation.report")
    return 0


if __name__ == "__main__":
    sys.exit(main())
