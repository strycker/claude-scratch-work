#!/usr/bin/env python
"""build_platform_data.py — Build the platform's monthly data checkpoints.

Runs ``build_monthly_spine()`` once to fetch the real sources and write ``daily_raw``,
``monthly_raw`` and ``monthly_features`` (dev and holdout) into the platform checkpoint
namespace, then ``fetch_fred_daily()`` to write ``fred_daily_raw`` (DAAA/DBAA, the weekly
tripwire's credit signal). It works from an empty ``data/``.

Under ``build.fail_loud`` (true in config/platform_settings.yaml) a lost source stops the build
before anything is written, and a failed DAAA/DBAA fetch fails the build after the other
checkpoints are written. Either way the script exits 1 and names the source and the retry
command (run this script again).

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
    from trading_crab_lib.platform.honesty.holdout import (
        DEFAULT_HOLDOUT_CUTOFF,
        assert_dev_checkpoint_within_boundary,
    )
    from trading_crab_lib.platform.transforms_monthly import BuildFailed, build_monthly_spine

    cfg = load_platform_config()
    start = cfg["data"]["start_date"]
    end = cfg["data"].get("end_date") or "today"
    log.info("Building monthly spine %s → %s (fetching FRED + multpl + macrotrends + yfinance)...", start, end)

    try:
        monthly_features = build_monthly_spine(cfg)
    except BuildFailed as exc:
        log.error("%s", exc)
        return 1

    # The weekly page's crash tripwire reads fred_daily_raw (DAAA/DBAA) for its credit signal
    # (plan 08.2-03). Under build.fail_loud a failed or empty fetch fails the build (08.4 D-T8,
    # superseding ruling A1) once the other checks have run; with the gate off it only warns.
    from trading_crab_lib.platform.ingestion.macro_daily import fetch_fred_daily

    fail_loud = bool(cfg.get("build", {}).get("fail_loud"))
    retry = "python scripts/build_platform_data.py"
    fred_daily_failed = False
    try:
        fred_daily = fetch_fred_daily(cfg)
    except Exception as exc:  # noqa: BLE001 — network ingestion; fredapi raises various types
        problem = f"fred_daily_raw fetch (DAAA/DBAA) failed ({exc})"
    else:
        problem = "fred_daily_raw: no series fetched (DAAA/DBAA)" if fred_daily.empty else None
        if problem is None:
            log.info("fred_daily_raw: %d days, last date %s", len(fred_daily), fred_daily.index.max().date())
    if problem is not None:
        if fail_loud:
            fred_daily_failed = True
            log.error("%s. The build fails after the other checks; retry: %s", problem, retry)
        else:
            log.warning(
                "%s: the weekly tripwire's credit signal will read UNAVAILABLE (no checkpoint) or STALE "
                "(an old one). The rest of the build is unaffected.",
                problem,
            )

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

    # HON-01 fence, verified rather than assumed. build_monthly_spine carves
    # monthly_features at the 2020-12 boundary; this proves the dev-namespace
    # checkpoint on disk actually honours it. A build that silently leaves
    # post-cutoff rows where fitting code reads them is a failed build — every
    # fit after it would train on holdout data.
    try:
        assert_dev_checkpoint_within_boundary("monthly_features")
        log.info("Holdout fence OK: dev monthly_features ends on or before %s", DEFAULT_HOLDOUT_CUTOFF)
    except RuntimeError as exc:
        log.error("HOLDOUT FENCE VIOLATED: %s", exc)
        return 1
    except FileNotFoundError:
        log.error("HOLDOUT FENCE UNVERIFIABLE: dev monthly_features checkpoint is missing")
        return 1

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

    if fred_daily_failed:
        return 1

    print("\nBUILD OK — now write the weekly page:")
    print("    python -m trading_crab_lib.platform.report.weekly")
    # allocation_mode absent or null means regime_tilt (weekly.allocation_mode_from_config)
    if cfg.get("report", {}).get("allocation_mode") in (None, "regime_tilt"):
        print("regime_tilt mode also needs the served regime model first:")
        print("    python -m trading_crab_lib.platform.report.serving")
    return 0


if __name__ == "__main__":
    sys.exit(main())
