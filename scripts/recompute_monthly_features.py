#!/usr/bin/env python
"""recompute_monthly_features.py — offline, network-free recompute of the dev
``monthly_features`` platform checkpoint (D-02-A, ``07-CONTEXT.md``).

**This is NOT a rebuild.** It is a pure function of the already-CACHED
``monthly_raw`` platform checkpoint — it never calls
``transforms_monthly.build_monthly_spine()`` and never touches the network.
It exists because ``compute_lean_features()`` assigns ``features["oil"] =
monthly_raw["oil"]`` as an unwindowed passthrough, yet the on-disk
``monthly_features.oil`` predates a ``monthly_raw`` rebuild and is stuck at
its stale 1985-02 start while ``monthly_raw.oil`` runs the full 1962-01+
history. Recomputing from the cached raw spine gives every passthrough
column (``oil``, ``gold``, ``fred_vix``, ``cape_shiller``, ``div_yield``) its
current, correct coverage without re-scraping anything.

Do NOT extend this script to call ``build_monthly_spine()``: that function
always re-fetches from FRED, multpl.com, macrotrends.net and yfinance, and
macrotrends/stooq egress is blocked in this container — a re-ingest would not
merely be slow here, it would fail or silently degrade (ROADMAP tech-debt
item R1). ``scripts/build_platform_data.py`` remains the sanctioned full
rebuild entry point, used only from a machine with normal internet access.

Usage:
    python scripts/recompute_monthly_features.py --dry-run   # inspect the delta, write nothing
    python scripts/recompute_monthly_features.py             # write the recomputed checkpoint
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any

import pandas as pd

from trading_crab_lib.platform.checkpoints import get_platform_checkpoint_manager
from trading_crab_lib.platform.config import load_platform_config
from trading_crab_lib.platform.honesty.holdout import (
    DEFAULT_HOLDOUT_CUTOFF,
    assert_dev_checkpoint_within_boundary,
    split_by_holdout_boundary,
    write_monthly_features_split,
)
from trading_crab_lib.platform.transforms_monthly import (
    compute_lean_features,
    tag_feature_columns,
)

log = logging.getLogger(__name__)


def rebuild_monthly_features(monthly_raw: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Reproduce ``build_monthly_spine``'s tail EXACTLY, from a cached ``monthly_raw``.

    Computes the lean feature set, concatenates it onto ``monthly_raw``, and
    dedupes duplicate column labels keeping the lean copy (``keep="last"``) —
    the identical assembly ``build_monthly_spine`` performs before writing
    ``monthly_features``. Any divergence from this exact sequence means the
    recomputed checkpoint would not be shape-identical to what a genuine full
    rebuild would produce.

    Does NOT invoke ``compute_lean_features()``'s output alone: that would
    silently drop every raw column ``compute_lean_features`` does not itself
    derive (40 of the real checkpoint's 53 columns) — the exact bug this
    script exists to avoid.
    """
    lean = compute_lean_features(monthly_raw, cfg)
    tag_feature_columns(lean, cfg)  # WARNING-only defensive taxonomy-coverage check

    monthly_features = pd.concat([monthly_raw, lean], axis=1)
    # Passthrough lean columns (gold/oil/fred_vix/cape_shiller/div_yield) are
    # identical to their monthly_raw source — dedupe, keeping the lean copy.
    monthly_features = monthly_features.loc[:, ~monthly_features.columns.duplicated(keep="last")]
    monthly_features.index.name = "date"
    return monthly_features


def _non_nan_counts(df: pd.DataFrame) -> dict[str, int]:
    return {col: int(df[col].notna().sum()) for col in df.columns}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute the dev monthly_features checkpoint from the cached monthly_raw "
            "checkpoint, with no network access (D-02-A)."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="compute and log the per-column non-NaN delta without writing anything",
    )
    args = parser.parse_args(argv)

    cfg = load_platform_config()
    cm = get_platform_checkpoint_manager()

    monthly_raw = cm.load("monthly_raw")
    log.info(
        "recompute_monthly_features: loaded cached monthly_raw %d rows x %d cols (%s -> %s)",
        len(monthly_raw), len(monthly_raw.columns),
        monthly_raw.index.min().date() if len(monthly_raw) else "n/a",
        monthly_raw.index.max().date() if len(monthly_raw) else "n/a",
    )

    try:
        existing_dev = cm.load("monthly_features")
    except FileNotFoundError:
        log.warning("recompute_monthly_features: no existing dev monthly_features checkpoint found")
        existing_dev = pd.DataFrame()

    rebuilt = rebuild_monthly_features(monthly_raw, cfg)
    dev_df, holdout_df = split_by_holdout_boundary(rebuilt, cutoff=DEFAULT_HOLDOUT_CUTOFF)

    missing_cols = sorted(set(existing_dev.columns) - set(dev_df.columns))
    if missing_cols:
        log.error(
            "recompute_monthly_features: rebuilt dev frame is NARROWER than the existing "
            "checkpoint — refusing to write. Missing column(s): %s",
            missing_cols,
        )
        return 1

    old_counts = _non_nan_counts(existing_dev)
    new_counts = _non_nan_counts(dev_df)
    changed_cols = sorted(
        col for col in new_counts
        if new_counts[col] != old_counts.get(col, 0)
    )

    log.info(
        "recompute_monthly_features: existing dev shape=%s -> rebuilt dev shape=%s",
        existing_dev.shape, dev_df.shape,
    )
    if changed_cols:
        for col in changed_cols:
            log.info(
                "  %s: %d -> %d non-NaN months (delta %+d)",
                col, old_counts.get(col, 0), new_counts[col], new_counts[col] - old_counts.get(col, 0),
            )
    else:
        log.info("  no column's non-NaN count changed")

    if args.dry_run:
        log.info("--dry-run: not writing anything")
        return 0

    write_monthly_features_split(rebuilt, "monthly_features", cutoff=DEFAULT_HOLDOUT_CUTOFF)
    assert_dev_checkpoint_within_boundary("monthly_features", cutoff=DEFAULT_HOLDOUT_CUTOFF)
    log.info(
        "recompute_monthly_features: OK — dev %d rows x %d cols (<= %s), holdout %d rows (> %s)",
        len(dev_df), len(dev_df.columns), DEFAULT_HOLDOUT_CUTOFF, len(holdout_df), DEFAULT_HOLDOUT_CUTOFF,
    )
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    sys.exit(main())
