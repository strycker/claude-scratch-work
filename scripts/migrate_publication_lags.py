#!/usr/bin/env python
"""migrate_publication_lags.py — one-off, offline migration of the tracked
``monthly_raw`` checkpoint to the publication-lag table (08.1 D-01, D-02).

Why this exists
---------------
``monthly_raw`` was built before ``publication_lags`` existed. A fresh network
rebuild would change more than the lags (new months, revisions), so the phase's
before/after comparison would no longer isolate them. This script applies the
lag table to the data already on disk, using the same
``apply_publication_lags`` that ``build_monthly_spine`` uses, so the result is
what a fresh build over the same inputs would produce.

What it does
------------
1. Refuses to run if ``publication_lags.json`` already exists (nothing is ever
   lagged twice).
2. Lags every int-entry column with ``apply_publication_lags`` and reindexes
   onto the existing month-end index (column order kept).
3. Rebuilds the splice research series with ``build_core_research_series`` on
   the lagged frame. A research column with no lagged input keeps HEAD's values
   (it may carry merge-preserved history a rebuild cannot reproduce); any
   difference is logged. A research column WITH a lagged input must rebuild to
   HEAD exactly from the unlagged frame, otherwise the script refuses.
4. Re-runs the live ``align_agency_monthly`` (ALFRED) and checks it against the
   HEAD-derived expectation:
     * pre-vintage ``fred_gdp`` (as-of before the first GDP release in the
       fetch) equals HEAD's value two month-ends earlier (fallback lag 1 -> 3);
     * post-vintage ``fred_gdp`` equals HEAD times one constant: the vintage
       chain is ratio-spliced onto the fallback's value at the first vintage
       as-of (``alfred.align_with_fallback``), and that join value moves with
       the fallback lag;
     * every other agency cell equals HEAD.
   Any other difference is revision drift since HEAD was built; the
   HEAD-derived value is kept and the drift is logged, so before and after
   differ only by the lags.
5. Raises unless every lag-0 column equals HEAD exactly and the 2026-08-31 row
   holds HEAD's div_yield@2026-05-31, fred_m2sl@2026-07-31 and
   fred_totalsl@2026-06-30.
6. Saves with ``merge=False`` (merge-on-save would refill the lagged head rows
   with unlagged values), then writes the splice provenance and the marker.

Usage:
    python scripts/migrate_publication_lags.py --dry-run   # all checks, write nothing
    python scripts/migrate_publication_lags.py             # checks, then write
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any

import numpy as np
import pandas as pd

from trading_crab_lib.platform import splice
from trading_crab_lib.platform.checkpoints import get_platform_checkpoint_manager
from trading_crab_lib.platform.config import load_platform_config
from trading_crab_lib.platform.ingestion import alfred
from trading_crab_lib.platform.ingestion.publication_lags import (
    DERIVED,
    LAG_MARKER_FILENAME,
    apply_publication_lags,
    lag_table,
    write_lag_marker,
)
from trading_crab_lib.platform.transforms_monthly import align_agency_monthly

log = logging.getLogger(__name__)

RTOL = 1e-9  # H-10: floats compared at rel 1e-9, abs 0
CHECK_ROW = pd.Timestamp("2026-08-31")
CHECK_SOURCES = {  # research validation 2: the value each lagged column must carry at CHECK_ROW
    "div_yield": pd.Timestamp("2026-05-31"),
    "fred_m2sl": pd.Timestamp("2026-07-31"),
    "fred_totalsl": pd.Timestamp("2026-06-30"),
}
GDP = "fred_gdp"


# ── Comparison helpers ──────────────────────────────────────────────────────


def _mismatch(a: pd.Series, b: pd.Series, *, exact: bool = False) -> pd.Series:
    """Boolean mask of cells that differ (NaN == NaN; rel 1e-9, abs 0 unless *exact*)."""
    a, b = a.astype(float), b.astype(float)
    both_nan = a.isna() & b.isna()
    if exact:
        same = (a == b) | both_nan
    else:
        same = pd.Series(np.isclose(a, b, rtol=RTOL, atol=0.0), index=a.index) | both_nan
    return ~same


def _first_valid(s: pd.Series) -> str:
    fv = s.first_valid_index()
    return fv.date().isoformat() if fv is not None else "none"


# ── Steps ───────────────────────────────────────────────────────────────────


def _lag_columns(head: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, int]]:
    table = lag_table(cfg)
    unlisted = [c for c in head.columns if c not in table]
    if unlisted:
        raise ValueError(f"monthly_raw columns without a publication_lags entry: {unlisted}")
    int_cols = [c for c in head.columns if isinstance(table[c], int)]
    lagged = apply_publication_lags(head[int_cols], cfg).reindex(head.index)
    lagged.index.name = head.index.name
    return lagged, {c: table[c] for c in int_cols}


def _rebuild_lagged_level(
    col: str,
    head: pd.DataFrame,
    rebuilt_unlagged: pd.Series,
    rebuilt_lagged: pd.Series,
) -> pd.Series:
    """A research level with a lagged input (equities_tr <- div_yield), rebuilt on HEAD's base.

    HEAD's level can sit on a merge-preserved base: equities_tr is chained from
    1.0 at the first price, but the tracked copy reads ~1790 there. So HEAD is
    only required to equal the unlagged rebuild times ONE constant k (rel 1e-9
    across every valid cell, identical NaN mask). That proves the rebuild
    reproduces HEAD's returns exactly; the lagged rebuild is then put on the
    same base (x k). Every consumer reads returns (pct_change), which k cancels
    out of, so the result equals a fresh build's returns while the level stays
    comparable to HEAD. Anything else and the script refuses.
    """
    h = head[col].astype(float)
    ru = rebuilt_unlagged.astype(float)
    if not h.isna().equals(ru.isna()):
        raise AssertionError(f"research {col}: unlagged rebuild's NaN mask differs from HEAD; refusing")
    ratio = (h / ru).dropna()
    if ratio.empty or not np.isfinite(ratio).all():
        raise AssertionError(f"research {col}: no finite HEAD/rebuild ratio; refusing")
    k = float(ratio.iloc[0])
    if not np.allclose(ratio.to_numpy(), k, rtol=RTOL, atol=0.0):
        raise AssertionError(
            f"research {col}: HEAD is not the unlagged rebuild times one constant "
            f"(ratio range {ratio.min()!r}..{ratio.max()!r}); refusing"
        )
    out = rebuilt_lagged.astype(float) * k
    log.info("research %s: HEAD = unlagged rebuild x %.12g (merge-preserved base); lagged rebuild put on that base",
             col, k)
    return out.rename(col)


def _research(
    head: pd.DataFrame, lagged: pd.DataFrame, lags: dict[str, int], cfg: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Rebuild the research series from the lagged frame; see module docstring step 3."""
    table = lag_table(cfg)
    research_cols = [c for c in head.columns if table[c] == DERIVED]
    int_cols = list(lags)

    rebuilt_unlagged = splice.build_core_research_series(head[int_cols], cfg).reindex(head.index)
    rebuilt_lagged = splice.build_core_research_series(lagged, cfg).reindex(head.index)
    provenance = rebuilt_lagged.attrs.get("splice_provenance") or {}

    out = pd.DataFrame(index=head.index)
    for col in research_cols:
        sources = [
            src.get("resolved")
            for src in (provenance.get(col, {}).get("sources") or {}).values()
            if src.get("resolved")
        ]
        lagged_inputs = [s for s in sources if lags.get(s, 0) > 0]
        if lagged_inputs:
            out[col] = _rebuild_lagged_level(col, head, rebuilt_unlagged[col], rebuilt_lagged[col])
            log.info(
                "research %s: lagged input(s) %s -> rebuilt; %d cells changed, first valid %s -> %s",
                col, lagged_inputs, int(_mismatch(out[col], head[col]).sum()),
                _first_valid(head[col]), _first_valid(out[col]),
            )
        else:
            drift = _mismatch(rebuilt_lagged[col], head[col])
            if drift.any():
                log.warning(
                    "research %s: no lagged input; rebuild differs from HEAD in %d cells "
                    "(merge-preserved history) -> keeping HEAD values",
                    col, int(drift.sum()),
                )
            else:
                log.info("research %s: no lagged input; rebuild equals HEAD", col)
            out[col] = head[col]
    return out, provenance


def _expected_agency(head: pd.DataFrame, agency_cols: list[str], gdp_boundary: pd.Timestamp) -> pd.DataFrame:
    """The agency frame HEAD implies under the new GDP fallback lag (1 -> 3)."""
    idx = head.index
    expected = head[agency_cols].copy()
    if GDP not in agency_cols:
        return expected
    pre = idx < gdp_boundary
    # Positional shift is a month-end shift: the index is a contiguous month-end range (asserted).
    expected.loc[pre, GDP] = head[GDP].shift(2)[pre]
    post_idx = idx[~pre]
    if len(post_idx):
        t0 = post_idx[0]
        t0_minus_2 = idx[idx.get_loc(t0) - 2]
        scale = head.at[t0_minus_2, GDP] / head.at[t0, GDP]
        expected.loc[~pre, GDP] = head.loc[~pre, GDP] * scale
        log.info(
            "agency fred_gdp: vintage chain re-anchored at %s on the fallback value of %s; "
            "post-vintage scale factor %.12f",
            t0.date(), t0_minus_2.date(), scale,
        )
    return expected


def _agency(head: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, int]]:
    table = lag_table(cfg)
    agency_cols = [c for c in head.columns if isinstance(table[c], dict)]

    vintages = alfred.fetch_all_vintages(cfg)
    if GDP not in vintages:
        raise RuntimeError("ALFRED fetch returned no fred_gdp releases; cannot place the vintage boundary")
    rs_col = alfred._detect_vintage_columns(vintages[GDP])["realtime_start"]
    gdp_boundary = pd.to_datetime(vintages[GDP][rs_col]).min()
    log.info("agency fred_gdp: first release in the fetch %s (pre-vintage rows < this)", gdp_boundary.date())

    live = align_agency_monthly(head.index, cfg).reindex(head.index)
    missing = [c for c in agency_cols if c not in live.columns]
    if missing:
        raise RuntimeError(f"align_agency_monthly returned no column for {missing}")

    expected = _expected_agency(head, agency_cols, gdp_boundary)
    drift_counts: dict[str, int] = {}
    for col in agency_cols:
        drift = _mismatch(live[col], expected[col])
        drift_counts[col] = int(drift.sum())
        if drift.any():
            where = drift[drift].index
            pre = int((where < gdp_boundary).sum()) if col == GDP else 0
            log.warning(
                "agency %s: live ALFRED differs from the HEAD-derived expectation in %d cells "
                "(%s .. %s%s) -> revision drift, keeping the HEAD-derived values",
                col, len(where), where[0].date(), where[-1].date(),
                f"; {pre} pre-vintage" if col == GDP else "",
            )
        else:
            log.info("agency %s: live ALFRED equals the HEAD-derived expectation", col)
    return expected, drift_counts


def _check(new: pd.DataFrame, head: pd.DataFrame, lags: dict[str, int]) -> None:
    """Raise unless every invariant of the migration holds."""
    if list(new.columns) != list(head.columns) or not new.index.equals(head.index):
        raise AssertionError("migrated monthly_raw changed the column list/order or the index")
    for col, lag in lags.items():
        want = head[col].shift(lag) if lag else head[col]
        bad = _mismatch(new[col], want, exact=True)
        if bad.any():
            raise AssertionError(f"{col} (lag {lag}) is not HEAD shifted by {lag}: {int(bad.sum())} cells")
    for col, src in CHECK_SOURCES.items():
        got, want = new.at[CHECK_ROW, col], head.at[src, col]
        if pd.isna(want) or not got == want:
            raise AssertionError(
                f"{CHECK_ROW.date()} {col} = {got!r}, expected HEAD's {src.date()} value {want!r}"
            )
    log.info(
        "check row %s: div_yield=%r (HEAD 2026-05-31), fred_m2sl=%r (HEAD 2026-07-31), "
        "fred_totalsl=%r (HEAD 2026-06-30) — exact",
        CHECK_ROW.date(), new.at[CHECK_ROW, "div_yield"], new.at[CHECK_ROW, "fred_m2sl"],
        new.at[CHECK_ROW, "fred_totalsl"],
    )


def migrate(cfg: dict[str, Any], *, dry_run: bool) -> pd.DataFrame:
    cm = get_platform_checkpoint_manager()
    marker = cm.dir / LAG_MARKER_FILENAME
    if marker.exists():
        raise SystemExit(
            f"{marker} already exists: monthly_raw is already built under a lag table. "
            "This migration runs once; refusing so nothing is lagged twice."
        )

    head = cm.load("monthly_raw")
    expected_idx = pd.date_range(head.index[0], head.index[-1], freq=cfg["data"].get("monthly_freq", "ME"))
    if not head.index.equals(expected_idx):
        raise AssertionError("monthly_raw index is not a contiguous month-end range")
    log.info("HEAD monthly_raw: %d rows x %d cols (%s .. %s)", len(head), len(head.columns),
             head.index[0].date(), head.index[-1].date())

    lagged, lags = _lag_columns(head, cfg)
    research, provenance = _research(head, lagged, lags, cfg)
    agency, drift = _agency(head, cfg)

    new = pd.concat([lagged, research, agency], axis=1)[list(head.columns)].reindex(head.index)
    new.index.name = head.index.name
    _check(new, head, lags)

    for col in head.columns:
        n = int(_mismatch(new[col], head[col], exact=True).sum())
        if n:
            log.info("changed %-16s %4d cells; first valid %s -> %s",
                     col, n, _first_valid(head[col]), _first_valid(new[col]))
    log.info("revision drift kept at HEAD-derived values: %s", {k: v for k, v in drift.items() if v} or "none")

    if dry_run:
        log.info("--dry-run: all checks passed; nothing written")
        return new

    path = cm.save(new, "monthly_raw", merge=False, source="migrate_publication_lags (08.1 offline migration)")
    saved = pd.read_parquet(path)
    for col in new.columns:
        if _mismatch(saved[col].reindex(new.index), new[col], exact=True).any():
            raise AssertionError(f"saved monthly_raw differs from the migrated frame in {col}")
    if provenance:
        splice.write_splice_provenance(provenance, cm.dir / "splice_provenance.json")
    write_lag_marker(marker, cfg)
    log.info("wrote %s, splice_provenance.json and %s", path, marker.name)
    return new


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Migrate the tracked monthly_raw to the publication-lag table (08.1).")
    parser.add_argument("--dry-run", action="store_true", help="run every check, write nothing")
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
