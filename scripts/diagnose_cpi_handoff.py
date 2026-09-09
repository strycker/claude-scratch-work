#!/usr/bin/env python
"""Diagnose the surviving 1971-01 discontinuity in ``fred_cpi`` (audit item A4).

The within-vintage chaining fixed the 1988 rebasing (345.94 -> 347.14, smooth).
The shift_series -> vintage handoff at 1971-01 did NOT get fixed: the series
still jumps 39.60 -> 119.03, a factor of 3.0058, and then carries the 1967=100
base forward (2020 reads ~769 instead of ~258).

``align_with_fallback`` is supposed to ratio-splice the vintage segment onto
shift_series' base at that join, which would make the step continuous. It
evidently did not fire, and the reason is not determinable without seeing the
actual ALFRED frame — which needs a live FRED_API_KEY.

Run this and paste the output:

    python scripts/diagnose_cpi_handoff.py

It makes ONE ALFRED call (CPIAUCSL) and prints only derived diagnostics — no
key material, no full series dumps.
"""
from __future__ import annotations

import logging
import os

import pandas as pd

from trading_crab_lib.platform.config import load_platform_config
from trading_crab_lib.platform.ingestion import alfred
from trading_crab_lib.platform.transforms_monthly import _shift_fallback_series

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

BREAK = pd.Timestamp("1971-01-31")
PRIOR = pd.Timestamp("1970-12-31")


def main() -> int:
    if not os.environ.get("FRED_API_KEY"):
        print("FRED_API_KEY not set — this diagnostic needs a live key.")  # noqa: T201
        return 1

    cfg = load_platform_config()
    fred = alfred.Fred(api_key=os.environ["FRED_API_KEY"])
    releases = alfred.fetch_vintage_series(fred, "CPIAUCSL")

    cols = alfred._detect_vintage_columns(releases)
    rs, dt, val = cols["realtime_start"], cols["date"], cols["value"]
    releases[rs] = pd.to_datetime(releases[rs])
    releases[dt] = pd.to_datetime(releases[dt])

    print("\n=== 1. all-releases frame ===")  # noqa: T201
    print(f"rows={len(releases)}  columns={list(releases.columns)}")  # noqa: T201
    print(f"realtime_start range : {releases[rs].min().date()} -> {releases[rs].max().date()}")  # noqa: T201
    print(f"reference date range : {releases[dt].min().date()} -> {releases[dt].max().date()}")  # noqa: T201
    earliest = releases[rs].min()
    print(f"earliest_vintage     : {earliest.date()}   (handoff happens at the first as_of >= this)")  # noqa: T201

    print("\n=== 2. release rows for reference periods 1970-11 .. 1971-02 ===")  # noqa: T201
    window = releases[(releases[dt] >= "1970-11-01") & (releases[dt] <= "1971-02-28")]
    print(window.sort_values([dt, rs]).head(20).to_string(index=False))  # noqa: T201

    print("\n=== 3. earliest release rows in the whole frame (by realtime_start) ===")  # noqa: T201
    print(releases.nsmallest(8, rs)[[rs, dt, val]].to_string(index=False))  # noqa: T201

    monthly_index = pd.date_range("1962-01-31", "2020-12-31", freq="ME")
    shift_series = _shift_fallback_series(releases, monthly_index, "ME")
    print("\n=== 4. shift_series (the fallback) around the break ===")  # noqa: T201
    print(shift_series.loc["1970-09-30":"1971-04-30"].round(4).to_string())  # noqa: T201

    print("\n=== 5. what value_as_of sees on each side of the break ===")  # noqa: T201
    for as_of in (PRIOR, BREAK):
        known = alfred.value_as_of(releases, as_of)
        if known.empty:
            print(f"as_of {as_of.date()}: known is EMPTY -> falls back to shift_series")  # noqa: T201
            continue
        latest = known.index.max()
        print(  # noqa: T201
            f"as_of {as_of.date()}: {len(known)} ref periods known, "
            f"latest_ref={latest.date()}, value={known.loc[latest]:.4f}"
        )

    print("\n=== 6. the base_scale align_with_fallback would compute ===")  # noqa: T201
    known = alfred.value_as_of(releases, BREAK)
    if known.empty:
        print("known empty at the break — the handoff is NOT here.")  # noqa: T201
    else:
        raw = known.loc[known.index.max()]
        join = shift_series.get(BREAK, float("nan"))
        print(f"raw (published level at join) = {raw:.4f}")  # noqa: T201
        print(f"join_value (shift_series)     = {join}")  # noqa: T201
        print(f"pd.notna(join_value)          = {pd.notna(join)}")  # noqa: T201
        if pd.notna(join) and raw:
            print(f"=> base_scale would be {join / raw:.6f} (1.0 means the splice did NOT fire)")  # noqa: T201

    print("\n=== 7. aligned output around the break (end to end) ===")  # noqa: T201
    aligned = alfred.align_with_fallback(releases, monthly_index, shift_series)
    print(aligned.loc["1970-09-30":"1971-04-30"].round(4).to_string())  # noqa: T201
    print(f"\naligned 2020-08-31 = {aligned.get(pd.Timestamp('2020-08-31'))}   (~258 correct, ~769 = 1967 base)")  # noqa: T201
    _ = cfg
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
