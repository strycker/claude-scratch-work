#!/usr/bin/env python
"""build_platform_data.py — Build the Phase 1 monthly data checkpoints.

The platform backtest/report machinery (Phase 5) reads two checkpoints that the
Phase 1 data layer produces but that are NOT committed to git (``data/`` is
gitignored). On a fresh clone they are absent, so
``python -m trading_crab_lib.platform.evaluation.report`` fails with:

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
    log.info("Wrote platform checkpoints to: %s", cm.checkpoint_dir if hasattr(cm, "checkpoint_dir") else "data/checkpoints/platform/")
    log.info(
        "monthly_features: %d months, %d columns, %s → %s",
        len(monthly_features),
        len(monthly_features.columns),
        monthly_features.index.min().date() if len(monthly_features) else "n/a",
        monthly_features.index.max().date() if len(monthly_features) else "n/a",
    )

    # Sanity: the report needs price/return coverage; warn loudly if the price
    # fetch degraded to empty (the usual failure mode behind a blocked network).
    price_like = [c for c in monthly_features.columns if any(t in c.lower() for t in ("spy", "equit", "price", "return", "tr"))]
    if len(monthly_features) == 0 or not price_like:
        log.warning(
            "monthly_features looks sparse (no price/return-like columns). This usually means a data "
            "source was unreachable (Yahoo/macrotrends). Re-run from a machine with normal internet."
        )
        return 1

    print("\nBUILD OK — you can now run the real backtest:")
    print("    python -m trading_crab_lib.platform.evaluation.report")
    return 0


if __name__ == "__main__":
    sys.exit(main())
