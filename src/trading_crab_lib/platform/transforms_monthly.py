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


# ── Lean feature computation + taxonomy tagging (DATA-04) ──────────────────


def compute_lean_features(monthly_raw: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Compute the taxonomy-tagged full-history (1962+) lean feature set (DATA-04).

    Simple, single-purpose derivations only — column names match
    ``cfg['taxonomy']`` exactly, no speculative multi-scale expansion beyond
    the concrete taxonomy list (design §9). A source column missing from
    ``monthly_raw`` (e.g. a failed ingestion source) simply skips the
    dependent feature rather than raising — mirrors the incumbent's
    graceful-degradation convention.
    """
    cols = set(monthly_raw.columns)
    features: dict[str, pd.Series] = {}

    if {"fred_gs10", "fred_tb3ms"} <= cols:
        features["curve_10y3m"] = monthly_raw["fred_gs10"] - monthly_raw["fred_tb3ms"]
    if "fred_t10y2y" in cols:
        # Already the 10Y-2Y spread as published by FRED — passthrough, no
        # re-derivation (no GS2 series is ingested separately).
        features["curve_10y2y"] = monthly_raw["fred_t10y2y"]
    if {"fred_baa", "fred_aaa"} <= cols:
        features["credit_spread_baa_aaa"] = monthly_raw["fred_baa"] - monthly_raw["fred_aaa"]
    if "fred_vix" in cols:
        features["fred_vix"] = monthly_raw["fred_vix"]
    if "gold" in cols:
        features["gold"] = monthly_raw["gold"]
    if "oil" in cols:
        features["oil"] = monthly_raw["oil"]

    if "equities_tr" in cols:
        equity_returns = monthly_raw["equities_tr"].pct_change()
        features["trailing_return_1m"] = equity_returns
        features["trailing_return_3m"] = monthly_raw["equities_tr"].pct_change(3)
        # ponytail: naive single-period vol proxy (|1m return|) for
        # realized_vol_1m — true intra-period vol needs daily equity prices,
        # not available for the multpl-derived monthly equities_tr research
        # series. Upgrade if/when a daily total-return series exists.
        features["realized_vol_1m"] = equity_returns.abs()
        features["realized_vol_3m"] = equity_returns.rolling(3).std(ddof=0)

    if "cape_shiller" in cols:
        features["cape_shiller"] = monthly_raw["cape_shiller"]
    if "div_yield" in cols:
        features["div_yield"] = monthly_raw["div_yield"]
    if {"fred_gs10", "fred_cpi"} <= cols:
        # Real rate = nominal 10Y yield minus trailing-12-month CPI inflation
        # (YoY % change, percentage points) — both series are 1962+ and
        # already present in monthly_raw (fred_gs10 fast-layer; fred_cpi via
        # align_agency_monthly). No free 1962+ market-cap source exists yet
        # for buffett_indicator (see taxonomy.slow comment in
        # config/platform_settings.yaml) — real_rate_level has no such gap.
        cpi_yoy_pct = monthly_raw["fred_cpi"].pct_change(periods=12) * 100.0
        features["real_rate_level"] = monthly_raw["fred_gs10"] - cpi_yoy_pct

    if not features:
        return pd.DataFrame(index=monthly_raw.index)
    return pd.concat(features, axis=1)


def tag_feature_columns(features_df: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, str]:
    """Map every produced feature column to its taxonomy tier (DATA-04).

    Logs a WARNING listing any untagged column via
    ``taxonomy.check_columns_tagged`` — ``compute_lean_features`` is designed
    to only ever emit taxonomy-listed column names, so this is a defensive
    gate, not expected to fire in normal operation.
    """
    columns = list(features_df.columns)
    untagged = taxonomy.check_columns_tagged(columns, cfg)
    if untagged:
        log.warning("Untagged lean feature column(s) (DATA-04 gap): %s", untagged)
    return {col: taxonomy.classify_feature(col, cfg) for col in columns}


# ── Orchestrator ─────────────────────────────────────────────────────────────


def build_monthly_spine(cfg: dict[str, Any]) -> pd.DataFrame:
    """Assemble the monthly feature table (DATA-01, DATA-03 runtime, DATA-04).

    Orchestrates, in order: monthly macro ingestion, daily universe prices +
    monthly spine, core research series (splice), point-in-time agency
    alignment, and lean feature computation — merged NULL-tolerantly via
    ``pd.concat([...], axis=1)`` onto a canonical month-end index spanning
    ``cfg['data']['start_date']`` forward. Persists ``daily_raw``,
    ``monthly_raw``, and ``monthly_features`` checkpoints in the platform
    namespace and returns the monthly_features frame.
    """
    monthly_freq = cfg["data"].get("monthly_freq", "ME")
    start = cfg["data"]["start_date"]
    end = cfg["data"]["end_date"] or str(date.today())
    monthly_index = pd.date_range(start=start, end=end, freq=monthly_freq)

    macro = macro_monthly.fetch_macro_monthly(cfg)
    daily, monthly_prices = prices_daily.fetch_universe_prices(cfg)

    # The splice input MUST include the monthly ticker columns (e.g. IAU), not
    # just macro-only columns — otherwise a fallback chain naming a tradable
    # ETF (gold: [gold_spot, IAU]) could never resolve, even though the ETF
    # data is right there in monthly_prices. Guarded only for the both-empty
    # case: if either macro or monthly_prices has data, splice still runs on
    # whatever is available (a class missing its required macro columns then
    # raises its own actionable preflight error, rather than silently
    # skipping the whole splice step).
    splice_input_frames = [f for f in (macro, monthly_prices) if not f.empty]
    if splice_input_frames:
        splice_input = pd.concat(splice_input_frames, axis=1)
        research = splice.build_core_research_series(splice_input, cfg)
    else:
        research = pd.DataFrame()

    agency = align_agency_monthly(monthly_index, cfg)

    frames = [f for f in (macro, monthly_prices, research, agency) if not f.empty]
    monthly_raw = pd.concat(frames, axis=1) if frames else pd.DataFrame(index=monthly_index)
    monthly_raw = monthly_raw.reindex(monthly_index)
    monthly_raw.index.name = "date"

    cm = get_platform_checkpoint_manager()
    cm.save(daily, "daily_raw", source="prices_daily.fetch_universe_prices (universe price chain)")
    cm.save(monthly_raw, "monthly_raw", source="build_monthly_spine (combined monthly ingest: macro+prices+research+agency)")

    splice_provenance = research.attrs.get("splice_provenance") if not research.empty else None
    if splice_provenance:
        splice.write_splice_provenance(splice_provenance, cm.dir / "splice_provenance.json")

    lean = compute_lean_features(monthly_raw, cfg)
    tag_feature_columns(lean, cfg)  # WARNING-only defensive taxonomy-coverage check

    monthly_features = pd.concat([monthly_raw, lean], axis=1)
    # Passthrough lean columns (gold/oil/fred_vix/cape_shiller/div_yield) are
    # identical to their monthly_raw source — dedupe, keeping the lean copy.
    monthly_features = monthly_features.loc[:, ~monthly_features.columns.duplicated(keep="last")]
    monthly_features.index.name = "date"

    cm.save(monthly_features, "monthly_features")

    log.info(
        "build_monthly_spine: assembled %d months, %d columns (monthly_features, %d lean)",
        len(monthly_features), len(monthly_features.columns), len(lean.columns),
    )
    return monthly_features
