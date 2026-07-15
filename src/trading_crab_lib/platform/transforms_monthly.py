"""
Monthly feature-table assembly — the phase's vertical join point (DATA-01, DATA-03, DATA-04).

Orchestrates the ingestion + splice + vintage layers built by Plans 01-01..01-06 into
ONE monthly feature table indexed by month-end back to ~1962:

  1. Fetch monthly macro/long-history raw data (``macro_monthly.fetch_macro_monthly``).
  2. Fetch daily universe prices + derive a monthly spine (``prices_daily.fetch_universe_prices``).
  3. Build the 5 core research series via ratio-splice/synthesis
     (``splice.build_core_research_series``).
  4. Point-in-time-align the D-06 agency series (``align_agency_monthly``) — value_as_of
     where ALFRED vintages exist, publication-lag shift fallback before the earliest
     recorded vintage (RESEARCH Pitfall 4: vintage-correction subsumes the shift once
     vintages exist).
  5. Merge everything NULL-tolerantly via ``pd.concat([...], axis=1)`` (RESEARCH
     Pitfall 5 — never ``pd.merge``/``.join`` defaults) onto a canonical month-end index
     spanning ``cfg['data']['start_date']`` forward.
  6. Compute the lean fast/slow taxonomy-tagged feature set (``compute_lean_features`` +
     ``tag_feature_columns``) and persist ``daily_raw``, ``monthly_raw``, and
     ``monthly_features`` checkpoints in the platform namespace.

Mirrors the incumbent ``transforms.py::engineer_all``'s "fixed step order, each step a
named helper" structure (frozen, not edited — D-01). Imports the Plan 02/03/05/06
modules by absolute path; never edits any frozen incumbent module.

Usage:
    from trading_crab_lib.platform.transforms_monthly import build_monthly_spine
    from trading_crab_lib.platform.config import load_platform_config

    cfg = load_platform_config()
    monthly_features = build_monthly_spine(cfg)
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

import pandas as pd

from trading_crab_lib.platform import splice, taxonomy
from trading_crab_lib.platform.checkpoints import get_platform_checkpoint_manager
from trading_crab_lib.platform.ingestion import alfred, macro_monthly, prices_daily

log = logging.getLogger(__name__)


# ── Point-in-time agency alignment (DATA-03 runtime) ────────────────────────


def _shift_fallback_series(
    all_releases: pd.DataFrame, monthly_index: pd.DatetimeIndex, monthly_freq: str
) -> pd.Series:
    """Build the D-06 pre-vintage-era fallback: each reference period's
    first-published value, resampled onto the monthly spine and shifted by
    one period (mirrors the incumbent's publication-lag ``shift()``
    convention — ``ingestion/fred.py`` ADR #7).
    """
    cols = alfred._detect_vintage_columns(all_releases)
    date_col, rs_col, value_col = cols["date"], cols["realtime_start"], cols["value"]

    first = (
        all_releases.sort_values(rs_col)
        .groupby(date_col)
        .head(1)
        .set_index(date_col)[value_col]
        .sort_index()
    )
    first.index = pd.to_datetime(first.index)

    monthly = first.resample(monthly_freq).last().reindex(monthly_index).ffill()
    return monthly.shift(1)


def align_agency_monthly(
    monthly_index: pd.DatetimeIndex, cfg: dict[str, Any], fred_client: Any | None = None
) -> pd.DataFrame:
    """Point-in-time-align every ``cfg['fred_vintage']['series']`` column onto ``monthly_index``.

    Uses ``alfred.fetch_all_vintages`` (bulk, one call per series) then
    ``alfred.align_with_fallback`` per series: point-in-time reconstruction via
    ``value_as_of`` where ALFRED vintages exist, publication-lag-shift fallback
    before the earliest recorded vintage — no revision or timing look-ahead
    (DATA-03 runtime application; RESEARCH Pitfall 4).

    Args:
        monthly_index: canonical month-end spine to align every series onto.
        cfg: platform config (``cfg['fred_vintage']['series']``, ``cfg['data']``).
        fred_client: optional pre-built ``fredapi.Fred`` client. When given,
            each series is fetched individually via ``alfred.fetch_vintage_series``
            (bypassing ``fetch_all_vintages``' own internal client construction) —
            useful for callers that already hold a client. When omitted (the
            default), ``alfred.fetch_all_vintages(cfg)`` builds its own client.
    """
    monthly_freq = cfg["data"].get("monthly_freq", "ME")

    if fred_client is not None:
        series_cfg: dict = cfg["fred_vintage"]["series"]
        all_vintages: dict[str, pd.DataFrame] = {}
        for series_id, meta in series_cfg.items():
            friendly_name = meta["name"]
            try:
                all_vintages[friendly_name] = alfred.fetch_vintage_series(fred_client, series_id)
            except Exception as exc:  # noqa: BLE001 — fredapi raises various types
                log.warning("Failed to fetch vintage history for %s (%s): %s", friendly_name, series_id, exc)
    else:
        all_vintages = alfred.fetch_all_vintages(cfg)

    if not all_vintages:
        log.warning("align_agency_monthly: no vintage series fetched")
        return pd.DataFrame(index=monthly_index)

    columns: dict[str, pd.Series] = {}
    for name, releases in all_vintages.items():
        shift_series = _shift_fallback_series(releases, monthly_index, monthly_freq)
        aligned = alfred.align_with_fallback(releases, monthly_index, shift_series)
        aligned.name = name
        columns[name] = aligned

    df = pd.concat(columns, axis=1)
    df.index.name = "date"
    log.info("align_agency_monthly: aligned %d agency series onto %d months", len(columns), len(monthly_index))
    return df


# ── Orchestrator ─────────────────────────────────────────────────────────────


def build_monthly_spine(cfg: dict[str, Any]) -> pd.DataFrame:
    """Assemble the monthly feature table (DATA-01, DATA-03 runtime, DATA-04).

    Orchestrates, in order: monthly macro ingestion, daily universe prices +
    monthly spine, core research series (splice), and point-in-time agency
    alignment — merged NULL-tolerantly via ``pd.concat([...], axis=1)`` onto a
    canonical month-end index spanning ``cfg['data']['start_date']`` forward.
    Persists ``daily_raw`` and ``monthly_raw`` checkpoints in the platform
    namespace and returns the merged monthly frame.
    """
    monthly_freq = cfg["data"].get("monthly_freq", "ME")
    start = cfg["data"]["start_date"]
    end = cfg["data"]["end_date"] or str(date.today())
    monthly_index = pd.date_range(start=start, end=end, freq=monthly_freq)

    macro = macro_monthly.fetch_macro_monthly(cfg)
    daily, monthly_prices = prices_daily.fetch_universe_prices(cfg)
    research = splice.build_core_research_series(macro, cfg) if not macro.empty else pd.DataFrame()
    agency = align_agency_monthly(monthly_index, cfg)

    frames = [f for f in (macro, monthly_prices, research, agency) if not f.empty]
    monthly_raw = pd.concat(frames, axis=1) if frames else pd.DataFrame(index=monthly_index)
    monthly_raw = monthly_raw.reindex(monthly_index)
    monthly_raw.index.name = "date"

    cm = get_platform_checkpoint_manager()
    cm.save(daily, "daily_raw")
    cm.save(monthly_raw, "monthly_raw")

    log.info(
        "build_monthly_spine: assembled %d months, %d columns (monthly_raw)",
        len(monthly_raw), len(monthly_raw.columns),
    )
    return monthly_raw
