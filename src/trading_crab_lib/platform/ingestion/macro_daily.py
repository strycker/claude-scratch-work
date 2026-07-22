"""
Daily FRED credit ingestion — DAAA/DBAA at native daily frequency (L4-04,
Phase 4 RESEARCH Pitfall 2).

``fred_monthly``'s ``BAA``/``AAA`` series are monthly-only on FRED — the
tripwire's credit-spread-velocity signal (design §23.2, D-04 signal 2) needs
genuinely daily data. This module reuses ``macro_monthly.py``'s
``_fetch_fred_monthly`` client-construction + ``ThreadPoolExecutor`` +
try/except-WARNING skeleton, MINUS the resample — DAAA/DBAA are already
published at daily frequency, so no resample step is needed (mirrors
``prices_daily.py``'s no-internal-resample doctrine).

Usage:
    from trading_crab_lib.platform.ingestion.macro_daily import fetch_fred_daily
    from trading_crab_lib.platform.config import load_platform_config

    cfg = load_platform_config()
    daily_credit = fetch_fred_daily(cfg)
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from fredapi import Fred
except ImportError as _err:
    raise ImportError(
        "fredapi is required for FRED data ingestion. "
        "Install with: pip install 'trading-crab-lib[ingestion]'"
    ) from _err

from trading_crab_lib import OUTPUT_DIR
from trading_crab_lib.platform.checkpoints import get_platform_checkpoint_manager

log = logging.getLogger(__name__)

# Same parallel-fetch cap as macro_monthly.py — FRED tolerates small bursts
# of single-vintage get_series() calls.
_MAX_WORKERS = 8

_ARTIFACT_COLUMNS = ["date"]


# ── FRED daily (no resample — native daily frequency preserved) ────────────


def _fetch_fred_daily(fred: Fred, series_id: str, start: str, end: str) -> pd.Series:
    """Pull one FRED series at native (daily) frequency — no resample, unlike
    ``macro_monthly.py::_fetch_fred_monthly``. DAAA/DBAA are already daily,
    so resampling would only discard rows (mirrors ``prices_daily.py``'s
    no-internal-resample doctrine)."""
    return fred.get_series(series_id, observation_start=start, observation_end=end)


def assemble_fred_daily(series_map: dict[str, pd.Series]) -> pd.DataFrame:
    """Assemble a dict of {friendly_name: pd.Series} into one wide daily
    DataFrame.

    PURE function — no I/O, no network — the unit-testable seam. Merges via
    ``pd.concat([...], axis=1)`` (outer join, NULL-tolerant) — never
    ``pd.merge``/``.join`` defaults, which would silently drop dates only one
    series covers. Returns an empty DataFrame if ``series_map`` is empty.
    """
    if not series_map:
        return pd.DataFrame()
    frames = []
    for name, s in series_map.items():
        s = s.copy()
        s.name = name
        frames.append(s)
    df = pd.concat(frames, axis=1)
    df.index.name = "date"
    return df


def fetch_fred_daily(cfg: dict[str, Any]) -> pd.DataFrame:
    """
    Fetch every series in ``cfg["fred_daily"]["series"]`` at native daily
    frequency and persist as the ``"fred_daily_raw"`` platform checkpoint.

    Mirrors ``macro_monthly.py::fetch_fred_monthly``'s client-construction +
    ThreadPoolExecutor + try/except-log-WARNING-skip skeleton (graceful
    degradation on a single-series failure) — never imports or modifies
    ``ingestion/fred.py`` or ``macro_monthly.py``.

    Config shape expected:
        fred_daily:
          series:
            DAAA:
              name:  "fred_daaa"
              shift: false
        data:
          start_date: "1962-01-01"
          end_date:   null

    On total failure (no series fetched successfully), returns an empty
    DataFrame with a WARNING — graceful degradation, matching
    ``prices_daily.py``'s pattern.

    Returns:
        DataFrame indexed by date, columns = friendly names.
    """
    fred_daily_cfg = cfg.get("fred_daily", {})
    api_key = fred_daily_cfg.get("api_key") or cfg.get("fred_monthly", {}).get("api_key")
    if not api_key:
        log.warning("FRED_API_KEY is not set — fred_daily ingestion skipped")
        return pd.DataFrame()

    series_cfg: dict = fred_daily_cfg.get("series", {})
    if not series_cfg:
        log.warning("No fred_daily series configured — skipping")
        return pd.DataFrame()

    fred = Fred(api_key=api_key)
    start = cfg.get("data", {}).get("start_date", "1962-01-01")
    end = cfg.get("data", {}).get("end_date") or str(date.today())

    def _fetch_task(series_id: str, meta: dict) -> tuple[str, pd.Series | None]:
        friendly_name = meta["name"]
        log.info("Fetching FRED (daily) %-10s → %s", series_id, friendly_name)
        try:
            s = _fetch_fred_daily(fred, series_id, start, end)
            s.name = friendly_name
            return friendly_name, s
        except Exception as exc:  # noqa: BLE001 — fredapi raises various types
            log.warning("Failed to fetch %s (%s): %s", friendly_name, series_id, exc)
            return friendly_name, None

    series_map: dict[str, pd.Series] = {}
    with ThreadPoolExecutor(max_workers=min(_MAX_WORKERS, max(len(series_cfg), 1))) as pool:
        futures = {
            pool.submit(_fetch_task, sid, meta): sid
            for sid, meta in series_cfg.items()
        }
        for future in as_completed(futures):
            friendly_name, s = future.result()
            if s is not None:
                series_map[friendly_name] = s

    df = assemble_fred_daily(series_map)
    if df.empty:
        log.warning("fred_daily: no series fetched successfully")
        return df

    get_platform_checkpoint_manager().save(df, "fred_daily_raw")
    log.info("fred_daily fetch complete: %d days, %d series", len(df), len(df.columns))
    return df


# ── report / persist ────────────────────────────────────────────────────────


def report_fred_daily(frame: pd.DataFrame, *, output_dir: Path | None = None) -> Path:
    """Print a human-readable summary and persist a schema-stable parquet
    artifact (``gap_lag.py`` shape).

    Args:
        frame: the assembled daily DataFrame (index = date, one column per series).
        output_dir: overrides the default artifact directory (for tests).

    Returns:
        Path: the written parquet artifact path.
    """
    target_dir = Path(output_dir) if output_dir is not None else OUTPUT_DIR / "reports" / "platform"
    target_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = target_dir / "fred_daily.parquet"

    df = frame.reset_index() if not frame.empty else pd.DataFrame(columns=_ARTIFACT_COLUMNS)
    df.to_parquet(artifact_path, index=False)

    summary = (
        "FRED Daily Credit Ingestion (L4-04)\n"
        f"  rows:     {len(frame)}\n"
        f"  series:   {list(frame.columns)}\n"
        f"  artifact: {artifact_path}"
    )
    log.info(summary)
    print(summary)  # noqa: T201 — first-class CLI run output, not debug noise

    return artifact_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Synthetic self-check — proves the assembly path with no network and no
    # FRED_API_KEY (RESEARCH Environment Availability: FRED_API_KEY not set
    # in this environment). Real DAAA/DBAA wiring is a live-data human
    # verification item, same as Phase 1.
    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    synthetic_daaa = pd.Series(range(10), index=idx, dtype=float)
    synthetic_dbaa = pd.Series(range(10, 20), index=idx, dtype=float)
    demo_map = {"fred_daaa": synthetic_daaa, "fred_dbaa": synthetic_dbaa}
    demo_frame = assemble_fred_daily(demo_map)
    print(f"self-check: assembled fred_daily frame shape = {demo_frame.shape}")  # noqa: T201
