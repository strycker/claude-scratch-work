#!/usr/bin/env python
"""migrate_month_end_columns.py — one-off, additive migration of the tracked
``monthly_raw`` checkpoint: append the three P&L-only month-end columns
(08.3 D-06; DECISIONS E-08).

Why this exists
---------------
08.3 moves strategy, ablation and baseline P&L onto month-end-to-month-end
returns. The P&L builder reads three new raw columns — ``sp500_close_me``
(yfinance ``^GSPC``), ``dgs10_me`` (FRED DGS10) and ``wti_me`` (FRED
DCOILWTICO) — that the tracked ``monthly_raw`` predates. A full network
rebuild would also move every other column (new months, revisions), so the
before/after measurement would no longer isolate the return change. This
script fetches ONLY the three columns, through the build's own fetchers and
the build's own ``apply_publication_lags``, and appends them to the rows
already on disk.

What it does
------------
1. Refuses to run if any ``splice.pnl_only_columns(cfg)`` column is already in
   ``monthly_raw`` (it runs once; nothing is appended twice).
2. Fetches the ``pnl_only`` FRED series with ``macro_monthly.fetch_fred_monthly``
   on a cfg narrowed to them, and the ``index_monthly`` closes with
   ``macro_monthly._fetch_index_month_end``; lags them with
   ``apply_publication_lags`` (all lag 0); reindexes onto the existing month-end
   index (never adds a row: a new row would pad every existing column).
3. Appends the columns after the existing ones and raises before saving unless:
     * the existing columns are exactly unchanged (index, column-order prefix,
       values, ``check_exact=True``);
     * ``sp500_close_me`` and ``dgs10_me`` are non-NaN on every row, and
       ``wti_me`` is NaN exactly before 1986-01-31 and non-NaN from it on;
     * ``dgs10_me`` passes the yield-units guard as a percent series, and every
       price is > 0;
     * corr(``sp500_close_me`` returns, SPY returns), 2000-2020, >= 0.99; level
       corr(``dgs10_me``, ``fred_gs10``) and corr(``wti_me``, ``wti_fred``),
       1986+, >= 0.98.
4. Saves with ``merge=False`` (merge-on-save reorders columns, new first) and
   rewrites the ``publication_lags.json`` marker so the next build may merge.
   ``monthly_features``, the holdout, labels and ``splice_provenance.json`` are
   not touched: D-01 leaves the features unchanged and ``features_from_raw``
   drops these columns (the L2 leak guard).

The FRED key is read from the environment by ``load_platform_config`` and is
never logged.

Usage:
    python scripts/migrate_month_end_columns.py --dry-run   # fetch + all checks, write nothing
    python scripts/migrate_month_end_columns.py             # fetch, checks, then write
"""

from __future__ import annotations

import argparse
import copy
import logging
import sys
from typing import Any

import numpy as np
import pandas as pd

from trading_crab_lib.platform import splice
from trading_crab_lib.platform.checkpoints import get_platform_checkpoint_manager
from trading_crab_lib.platform.config import load_platform_config
from trading_crab_lib.platform.ingestion import macro_monthly
from trading_crab_lib.platform.ingestion.publication_lags import (
    LAG_MARKER_FILENAME,
    apply_publication_lags,
    lag_marker_matches,
    write_lag_marker,
)

log = logging.getLogger(__name__)

#: Appended in this order, after the existing columns.
NEW_COLUMNS = ["sp500_close_me", "dgs10_me", "wti_me"]
WTI_FIRST = pd.Timestamp("1986-01-31")  # DCOILWTICO starts 1986-01-02
SPY_WINDOW = ("2000-01-31", "2020-12-31")
MIN_RETURN_CORR = 0.99
MIN_LEVEL_CORR = 0.98


# ── Fetch ───────────────────────────────────────────────────────────────────


def _narrow_fred_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    """``cfg`` with ``fred_monthly.series`` narrowed to the ``pnl_only`` series."""
    narrow = copy.deepcopy(cfg)
    series = cfg["fred_monthly"]["series"]
    narrow["fred_monthly"]["series"] = {sid: meta for sid, meta in series.items() if meta.get("pnl_only", False)}
    return narrow


def fetch_month_end_columns(cfg: dict[str, Any], index: pd.DatetimeIndex) -> pd.DataFrame:
    """The three month-end columns, lagged by the build's own table, on *index*."""
    fred_df = macro_monthly.fetch_fred_monthly(_narrow_fred_cfg(cfg))
    index_cols = macro_monthly._fetch_index_month_end(cfg)
    frames = [fred_df[c] for c in fred_df.columns] + list(index_cols.values())
    fetched = pd.concat(frames, axis=1) if frames else pd.DataFrame()
    missing = [c for c in NEW_COLUMNS if c not in fetched.columns]
    if missing:
        raise RuntimeError(f"month-end fetch returned nothing for {missing}; refusing to migrate")
    extra = sorted(set(fetched.columns) - set(NEW_COLUMNS))
    if extra:
        raise RuntimeError(f"month-end fetch returned unexpected columns {extra}; refusing to migrate")
    lagged = apply_publication_lags(fetched[NEW_COLUMNS], cfg)
    out = lagged.reindex(index)
    out.index.name = index.name
    return out


# ── Checks ──────────────────────────────────────────────────────────────────


def _corr(a: pd.Series, b: pd.Series) -> float:
    both = pd.concat([a, b], axis=1).dropna()
    return float(both.iloc[:, 0].corr(both.iloc[:, 1]))


def check_migrated(new: pd.DataFrame, existing: pd.DataFrame) -> dict[str, float]:
    """Raise unless every invariant holds; return the measured correlations."""
    if not new.index.equals(existing.index):
        raise AssertionError("migrated monthly_raw changed the index")
    if list(new.columns) != list(existing.columns) + NEW_COLUMNS:
        raise AssertionError(f"column order is not existing + {NEW_COLUMNS}: {list(new.columns)[-5:]}")
    pd.testing.assert_frame_equal(new[existing.columns], existing, check_exact=True)

    for col in ("sp500_close_me", "dgs10_me"):
        n_nan = int(new[col].isna().sum())
        if n_nan:
            raise AssertionError(f"{col} has {n_nan} NaN rows; expected full coverage")
    wti = new["wti_me"]
    if wti[wti.index < WTI_FIRST].notna().any():
        raise AssertionError("wti_me has values before 1986-01-31")
    if wti[wti.index >= WTI_FIRST].isna().any():
        raise AssertionError("wti_me has NaN on or after 1986-01-31")

    dgs10 = new["dgs10_me"]
    splice.assert_yield_units_plausible(dgs10 / 100.0, source="dgs10_me (percent)")
    if float(dgs10.median()) < 1.0:
        raise AssertionError(f"dgs10_me median {dgs10.median():.4g} is not a percent yield")
    for col in ("sp500_close_me", "wti_me"):
        if (new[col].dropna() <= 0).any():
            raise AssertionError(f"{col} has a non-positive price")

    window = slice(*SPY_WINDOW)
    r_me = new["sp500_close_me"].pct_change(fill_method=None).loc[window]
    r_spy = new["SPY"].pct_change(fill_method=None).loc[window]
    post = new.index >= WTI_FIRST
    corrs = {
        "ret sp500_close_me vs SPY 2000-2020": _corr(r_me, r_spy),
        "level dgs10_me vs fred_gs10 1986+": _corr(new.loc[post, "dgs10_me"], new.loc[post, "fred_gs10"]),
        "level wti_me vs wti_fred 1986+": _corr(new.loc[post, "wti_me"], new.loc[post, "wti_fred"]),
    }
    floors = [MIN_RETURN_CORR, MIN_LEVEL_CORR, MIN_LEVEL_CORR]
    for (name, value), floor in zip(corrs.items(), floors, strict=True):
        if not np.isfinite(value) or value < floor:
            raise AssertionError(f"{name} = {value:.4f} < {floor}")
        log.info("%s = %.4f (>= %s)", name, value, floor)
    return corrs


# ── Migration ───────────────────────────────────────────────────────────────


def migrate(cfg: dict[str, Any], *, dry_run: bool) -> pd.DataFrame:
    cm = get_platform_checkpoint_manager()
    existing = cm.load("monthly_raw")
    present = sorted(splice.pnl_only_columns(cfg) & set(existing.columns))
    if present:
        raise SystemExit(
            f"monthly_raw already holds P&L-only column(s) {present}: this migration runs once; refusing."
        )
    if set(splice.pnl_only_columns(cfg)) != set(NEW_COLUMNS):
        raise AssertionError(f"pnl_only_columns(cfg) = {sorted(splice.pnl_only_columns(cfg))}, expected {NEW_COLUMNS}")
    expected_idx = pd.date_range(existing.index[0], existing.index[-1], freq=cfg["data"].get("monthly_freq", "ME"))
    if not existing.index.equals(expected_idx):
        raise AssertionError("monthly_raw index is not a contiguous month-end range")
    log.info("tracked monthly_raw: %d rows x %d cols (%s .. %s)", len(existing), len(existing.columns),
             existing.index[0].date(), existing.index[-1].date())

    added = fetch_month_end_columns(cfg, existing.index)
    new = pd.concat([existing, added], axis=1)
    new.index.name = existing.index.name
    check_migrated(new, existing)
    log.info("all checks passed: %d rows x %d cols", len(new), len(new.columns))

    if dry_run:
        log.info("--dry-run: nothing written")
        return new

    path = cm.save(new, "monthly_raw", merge=False, source="migrate_month_end_columns (08.3 additive migration)")
    saved = pd.read_parquet(path)
    pd.testing.assert_frame_equal(saved, new, check_exact=True, check_freq=False)
    marker = cm.dir / LAG_MARKER_FILENAME
    write_lag_marker(marker, cfg)
    if not lag_marker_matches(marker, cfg):
        raise AssertionError(f"{marker} does not match the live lag table after writing")
    log.info("wrote %s and %s", path, marker.name)
    return new


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Append the 08.3 month-end P&L columns to the tracked monthly_raw.")
    parser.add_argument("--dry-run", action="store_true", help="fetch and run every check, write nothing")
    args = parser.parse_args(argv)
    migrate(load_platform_config(), dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    sys.exit(main())
