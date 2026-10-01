"""
Daily tripwire monitor — 3 independent-family risk signals combined by
OR-logic into a single escalation enum (L4-04, design §23.2/§25).

v1 is the minimal 3-signal subset per D-04 (04-CONTEXT.md): realized-vol
spike (vol family), credit-spread velocity BAA-AAA widening (credit family),
and SPY drawdown-from-peak (price family). The full family-independence
orchestrator (§25) is v2 (L2-V2-03) — this module implements only the
count-driven OR-logic escalation: 0 triggers -> NONE, 1 ->
RUN_WEEKLY_SCORING_EARLY, 2 or 3 -> TIER1_DERISK_REVIEW, regardless of WHICH
signal(s) fired (T-04-11: escalation is a pure, fully-enumerated function of
the trigger COUNT, never signal identity).

All numeric thresholds below are PROVISIONAL-UNTIL-PHASE-5-BACKTEST
(RESEARCH.md A3: no literature-sourced numeric anchor exists for these
specific values) — config-overridable via ``cfg["tripwire"]``
(``config/platform_settings.yaml``); Phase 5's honest backtest is the
actual validation mechanism, not this v1 guess.

Usage::

    from trading_crab_lib.platform.tripwire.monitor import run_tripwire
    from trading_crab_lib.platform.config import load_platform_config

    cfg = load_platform_config()
    escalation = run_tripwire(cfg)
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from trading_crab_lib.platform.assets.vol import DAILY_ANNUALIZATION, ewma_vol
from trading_crab_lib.platform.checkpoints import get_platform_checkpoint_manager

log = logging.getLogger(__name__)

# Defaults mirror config/platform_settings.yaml's `tripwire:` section — used
# only when cfg omits a key. Provisional-until-Phase-5-backtest (A3).
_DEFAULT_VOL_HALFLIFE_DAYS = 11.2
_DEFAULT_VOL_SHORT_WINDOW_DAYS = 21
_DEFAULT_VOL_BASELINE_DAYS = 63
_DEFAULT_VOL_SPIKE_RATIO = 1.5
_DEFAULT_CREDIT_LOOKBACK_DAYS = 5
_DEFAULT_CREDIT_VELOCITY_BPS = 25
_DEFAULT_SPY_DRAWDOWN_PCT = 0.10


class TripwireEscalation(str, Enum):
    """Escalation tier — count-driven, never signal-identity-driven (T-04-11)."""

    NONE = "none"
    RUN_WEEKLY_SCORING_EARLY = "run weekly scoring early"
    TIER1_DERISK_REVIEW = "Tier-1 de-risk review"


def escalate(vol_spike: bool, credit_velocity: bool, spy_drawdown: bool) -> TripwireEscalation:
    """OR-logic escalation (D-04, design §23.2): the COUNT of triggered
    signals determines the tier, never WHICH signal(s) fired
    (family-independence headline invariant, T-04-11)."""
    n_triggered = sum([vol_spike, credit_velocity, spy_drawdown])
    if n_triggered >= 2:
        return TripwireEscalation.TIER1_DERISK_REVIEW
    if n_triggered == 1:
        return TripwireEscalation.RUN_WEEKLY_SCORING_EARLY
    return TripwireEscalation.NONE


def realized_vol_spike(
    daily_returns: pd.Series,
    *,
    short_window: int,
    baseline_window: int,
    ratio_threshold: float,
    halflife: float,
) -> bool:
    """Vol family: recent short-window EWMA vol vs its trailing baseline
    EWMA vol. Reuses ``ewma_vol`` (platform/assets/vol.py) for both windows
    — do not duplicate the decay math here."""
    recent = daily_returns.tail(short_window)
    baseline = daily_returns.iloc[-(short_window + baseline_window) : -short_window]
    if len(recent) < 2 or len(baseline) < 2:
        return False
    recent_vol = ewma_vol(recent, halflife=halflife, annualization_factor=DAILY_ANNUALIZATION).iloc[-1]
    baseline_vol = ewma_vol(baseline, halflife=halflife, annualization_factor=DAILY_ANNUALIZATION).iloc[-1]
    if not np.isfinite(baseline_vol) or baseline_vol == 0 or not np.isfinite(recent_vol):
        return False
    return bool((recent_vol / baseline_vol) > ratio_threshold)


def credit_spread_velocity(
    daaa: pd.Series,
    dbaa: pd.Series,
    *,
    lookback_days: int,
    bps_threshold: float,
) -> bool:
    """Credit family: BAA-minus-AAA spread widening over ``lookback_days``,
    in bps (FRED yields are in percentage points -> x100 converts to bps)."""
    spread_bps = (dbaa - daaa) * 100.0
    if len(spread_bps) <= lookback_days:
        return False
    widening = spread_bps.iloc[-1] - spread_bps.iloc[-1 - lookback_days]
    return bool(widening >= bps_threshold)


def spy_drawdown_from_peak(spy_prices: pd.Series, *, drawdown_threshold: float) -> bool:
    """Price family: current SPY price vs its running peak (``cummax``)."""
    running_peak = spy_prices.cummax()
    latest_price = spy_prices.iloc[-1]
    latest_peak = running_peak.iloc[-1]
    if latest_peak == 0:
        return False
    drawdown = (latest_price - latest_peak) / latest_peak
    return bool(drawdown <= -drawdown_threshold)


def run_tripwire(
    cfg: dict[str, Any],
    cm: Any = None,
    *,
    daily_returns: pd.Series | None = None,
    daaa: pd.Series | None = None,
    dbaa: pd.Series | None = None,
    spy_prices: pd.Series | None = None,
) -> TripwireEscalation:
    """Orchestrator: compute the 3 signals and escalate.

    When ``spy_prices``/``daaa``/``dbaa`` are None, loads ``daily_raw["SPY"]``
    and ``fred_daily_raw[["fred_daaa", "fred_dbaa"]]`` from
    ``get_platform_checkpoint_manager()`` (the live path, Plan 01's daily
    ingestion). A missing checkpoint surfaces a clear error at the daily run
    (T-04-12) — not a silent all-clear.
    """
    tripwire_cfg = cfg.get("tripwire", {})
    halflife = tripwire_cfg.get("vol_halflife_days", _DEFAULT_VOL_HALFLIFE_DAYS)
    short_window = tripwire_cfg.get("vol_spike_short_window_days", _DEFAULT_VOL_SHORT_WINDOW_DAYS)
    baseline_window = tripwire_cfg.get("vol_spike_baseline_days", _DEFAULT_VOL_BASELINE_DAYS)
    vol_ratio = tripwire_cfg.get("vol_spike_ratio", _DEFAULT_VOL_SPIKE_RATIO)
    credit_lookback = tripwire_cfg.get("credit_velocity_lookback_days", _DEFAULT_CREDIT_LOOKBACK_DAYS)
    credit_bps = tripwire_cfg.get("credit_velocity_bps", _DEFAULT_CREDIT_VELOCITY_BPS)
    drawdown_pct = tripwire_cfg.get("spy_drawdown_pct", _DEFAULT_SPY_DRAWDOWN_PCT)

    if spy_prices is None or daaa is None or dbaa is None:
        cm = cm or get_platform_checkpoint_manager()
        if spy_prices is None:
            spy_prices = cm.load("daily_raw")["SPY"]
        if daaa is None or dbaa is None:
            fred_daily_raw = cm.load("fred_daily_raw")
            daaa = daaa if daaa is not None else fred_daily_raw["fred_daaa"]
            dbaa = dbaa if dbaa is not None else fred_daily_raw["fred_dbaa"]
    if daily_returns is None:
        daily_returns = spy_prices.pct_change().dropna()

    vol_spike = realized_vol_spike(
        daily_returns,
        short_window=short_window,
        baseline_window=baseline_window,
        ratio_threshold=vol_ratio,
        halflife=halflife,
    )
    credit_velocity = credit_spread_velocity(daaa, dbaa, lookback_days=credit_lookback, bps_threshold=credit_bps)
    spy_drawdown = spy_drawdown_from_peak(spy_prices, drawdown_threshold=drawdown_pct)

    escalation = escalate(vol_spike, credit_velocity, spy_drawdown)
    log.info(
        "Tripwire: vol_spike=%s credit_velocity=%s spy_drawdown=%s -> %s",
        vol_spike, credit_velocity, spy_drawdown, escalation.value,
    )
    print(f"Tripwire escalation: {escalation.value}")  # noqa: T201 — first-class daily-run output

    return escalation


# ── The weekly page's per-signal view (plan 08.2-03, D-04) ───────────────────
#
# ``run_tripwire`` above returns only the enum. The page needs, per signal, the value, the
# threshold and the date it is computed as of, and it must never read a missing or old input as
# green. The helpers below return the NUMBERS behind the bool functions (NaN when undefined);
# ``evaluate_tripwire`` takes ``triggered`` from the bool functions themselves, on the same
# cleaned inputs, so the page and ``run_tripwire`` cannot disagree on a current signal.

# A signal whose last observation is older than this many business days before the run date is
# STALE. Provisional (ruling A1, Glenn 2026-10-01); config key ``tripwire.stale_business_days``.
_DEFAULT_STALE_BUSINESS_DAYS = 5

_SIGNAL_LABELS = {
    "vol_spike": "Realized-vol spike (SPY)",
    "credit_velocity": "Credit-spread velocity (BAA-AAA)",
    "spy_drawdown": "Drawdown from peak (SPY)",
}


def vol_spike_ratio(
    daily_returns: pd.Series, *, short_window: int, baseline_window: int, halflife: float
) -> float:
    """Recent short-window EWMA vol over its trailing baseline EWMA vol: the number
    ``realized_vol_spike`` compares with its ratio threshold. NaN when undefined."""
    recent = daily_returns.tail(short_window)
    baseline = daily_returns.iloc[-(short_window + baseline_window) : -short_window]
    if len(recent) < 2 or len(baseline) < 2:
        return float("nan")
    recent_vol = ewma_vol(recent, halflife=halflife, annualization_factor=DAILY_ANNUALIZATION).iloc[-1]
    baseline_vol = ewma_vol(baseline, halflife=halflife, annualization_factor=DAILY_ANNUALIZATION).iloc[-1]
    if not np.isfinite(baseline_vol) or baseline_vol == 0 or not np.isfinite(recent_vol):
        return float("nan")
    return float(recent_vol / baseline_vol)


def credit_widening_bps(spread_bps: pd.Series, *, lookback_days: int) -> float:
    """BAA-AAA spread change in bps over the last ``lookback_days`` observations: the number
    ``credit_spread_velocity`` compares with its bps threshold. NaN when undefined."""
    if len(spread_bps) <= lookback_days:
        return float("nan")
    return float(spread_bps.iloc[-1] - spread_bps.iloc[-1 - lookback_days])


def drawdown_from_peak(prices: pd.Series) -> float:
    """Latest price against its running peak, as a (non-positive) fraction: the number
    ``spy_drawdown_from_peak`` compares with its threshold. NaN when undefined."""
    if len(prices) == 0:
        return float("nan")
    peak = float(prices.cummax().iloc[-1])
    if peak == 0 or not np.isfinite(peak):
        return float("nan")
    return float((prices.iloc[-1] - peak) / peak)


def _unavailable(name: str, source: str, threshold: float, reason: str) -> dict[str, Any]:
    return {
        "label": _SIGNAL_LABELS[name], "state": "unavailable", "triggered": None, "value": float("nan"),
        "threshold": float(threshold), "as_of": None, "source": source, "reason": reason,
    }


def _reading(
    name: str, source: str, *, value: float, threshold: float, triggered: bool, as_of: pd.Timestamp,
    stale_before: pd.Timestamp,
) -> dict[str, Any]:
    """One signal's row. A value that cannot be computed is UNAVAILABLE (never green); an
    as-of date before ``stale_before`` is STALE whatever the value says."""
    if not np.isfinite(value):
        return _unavailable(name, source, threshold, f"too little data in '{source}' to compute it")
    if as_of < stale_before:
        state = "stale"
    else:
        state = "red" if triggered else "green"
    return {
        "label": _SIGNAL_LABELS[name], "state": state, "triggered": bool(triggered), "value": float(value),
        "threshold": float(threshold), "as_of": pd.Timestamp(as_of), "source": source, "reason": None,
    }


def evaluate_tripwire(cfg: dict[str, Any], cm: Any = None, *, run_date: pd.Timestamp) -> dict[str, Any]:
    """The three signals as the weekly page shows them: state, value, threshold, as-of date.

    Inputs are the checkpoints ``run_tripwire`` reads, ``daily_raw["SPY"]`` and
    ``fred_daily_raw[["fred_daaa", "fred_dbaa"]]``, each loaded on its own: a missing checkpoint
    or column (FileNotFoundError, KeyError) makes the signals that need it UNAVAILABLE and never
    raises. NaNs are dropped per series and DAAA/DBAA are aligned on the days both exist, so a
    NaN tail can never read as "not triggered" (T-08.2-08); a signal's ``as_of`` is the last day
    it actually used. Nothing is imputed.

    Per signal (``vol_spike``, ``credit_velocity``, ``spy_drawdown``, in that order)::

        {"label", "state", "triggered", "value", "threshold", "as_of", "source", "reason"}

    - ``state``: ``red`` / ``green`` (current), ``stale`` (``as_of`` older than
      ``tripwire.stale_business_days`` business days before ``run_date``, default 5, ruling A1)
      or ``unavailable`` (missing input, or too little data to compute the value);
    - ``triggered``: the existing bool function's answer on the cleaned inputs (None when
      unavailable); a stale signal keeps it, so the page can say what it would read;
    - ``value`` / ``threshold`` in the same units: the vol ratio (trips when ``value >
      threshold``), the widening in bps (``>=``), the drawdown as a negative fraction (``<=``).

    ``escalation`` is ``escalate()`` over the CURRENT signals only (stale and unavailable count
    as not triggered), returned with ``n_current``; whether that may be shown as an all-clear is
    the page's call (it may not unless all three are current). Thresholds come from
    ``cfg["tripwire"]`` with this module's defaults, exactly as in ``run_tripwire``.
    """
    tripwire_cfg = cfg.get("tripwire", {})
    halflife = tripwire_cfg.get("vol_halflife_days", _DEFAULT_VOL_HALFLIFE_DAYS)
    short_window = tripwire_cfg.get("vol_spike_short_window_days", _DEFAULT_VOL_SHORT_WINDOW_DAYS)
    baseline_window = tripwire_cfg.get("vol_spike_baseline_days", _DEFAULT_VOL_BASELINE_DAYS)
    vol_ratio = tripwire_cfg.get("vol_spike_ratio", _DEFAULT_VOL_SPIKE_RATIO)
    credit_lookback = tripwire_cfg.get("credit_velocity_lookback_days", _DEFAULT_CREDIT_LOOKBACK_DAYS)
    credit_bps = tripwire_cfg.get("credit_velocity_bps", _DEFAULT_CREDIT_VELOCITY_BPS)
    drawdown_pct = tripwire_cfg.get("spy_drawdown_pct", _DEFAULT_SPY_DRAWDOWN_PCT)
    stale_days = int(tripwire_cfg.get("stale_business_days", _DEFAULT_STALE_BUSINESS_DAYS))

    run_date = pd.Timestamp(run_date).normalize()
    stale_before = run_date - pd.offsets.BDay(stale_days)
    cm = cm or get_platform_checkpoint_manager()
    signals: dict[str, dict[str, Any]] = {}

    # Price family (vol spike + drawdown): daily_raw["SPY"], NaNs dropped.
    spy: pd.Series | None = None
    spy_reason = ""
    try:
        spy = cm.load("daily_raw")["SPY"].dropna().sort_index()
    except FileNotFoundError:
        spy_reason = "checkpoint 'daily_raw' is missing"
    except KeyError:
        spy_reason = "column 'SPY' is missing from 'daily_raw'"
    if spy is not None and spy.empty:
        spy, spy_reason = None, "'daily_raw' has no SPY observation"

    if spy is None:
        signals["vol_spike"] = _unavailable("vol_spike", "daily_raw", vol_ratio, spy_reason)
    else:
        daily_returns = spy.pct_change().dropna()
        signals["vol_spike"] = _reading(
            "vol_spike", "daily_raw",
            value=vol_spike_ratio(daily_returns, short_window=short_window, baseline_window=baseline_window,
                                  halflife=halflife),
            threshold=vol_ratio,
            triggered=realized_vol_spike(daily_returns, short_window=short_window, baseline_window=baseline_window,
                                         ratio_threshold=vol_ratio, halflife=halflife),
            as_of=pd.Timestamp(spy.index[-1]),
            stale_before=stale_before,
        )

    # Credit family: DAAA and DBAA aligned on the days both exist.
    aligned: pd.DataFrame | None = None
    credit_reason = ""
    try:
        frame = cm.load("fred_daily_raw")
        aligned = pd.concat([frame["fred_daaa"], frame["fred_dbaa"]], axis=1).dropna().sort_index()
    except FileNotFoundError:
        credit_reason = "checkpoint 'fred_daily_raw' is missing"
    except KeyError as exc:
        credit_reason = f"column {exc} is missing from 'fred_daily_raw'"
    if aligned is not None and aligned.empty:
        aligned, credit_reason = None, "'fred_daily_raw' has no day with both DAAA and DBAA"

    if aligned is None:
        signals["credit_velocity"] = _unavailable("credit_velocity", "fred_daily_raw", credit_bps, credit_reason)
    else:
        daaa, dbaa = aligned["fred_daaa"], aligned["fred_dbaa"]
        signals["credit_velocity"] = _reading(
            "credit_velocity", "fred_daily_raw",
            value=credit_widening_bps((dbaa - daaa) * 100.0, lookback_days=credit_lookback),
            threshold=credit_bps,
            triggered=credit_spread_velocity(daaa, dbaa, lookback_days=credit_lookback, bps_threshold=credit_bps),
            as_of=pd.Timestamp(aligned.index[-1]),
            stale_before=stale_before,
        )

    if spy is None:
        signals["spy_drawdown"] = _unavailable("spy_drawdown", "daily_raw", -drawdown_pct, spy_reason)
    else:
        signals["spy_drawdown"] = _reading(
            "spy_drawdown", "daily_raw",
            value=drawdown_from_peak(spy),
            threshold=-drawdown_pct,
            triggered=spy_drawdown_from_peak(spy, drawdown_threshold=drawdown_pct),
            as_of=pd.Timestamp(spy.index[-1]),
            stale_before=stale_before,
        )

    current = {name: sig["state"] in ("red", "green") for name, sig in signals.items()}
    escalation = escalate(*(current[n] and bool(signals[n]["triggered"]) for n in _SIGNAL_LABELS))
    result = {
        "run_date": run_date,
        "stale_business_days": stale_days,
        "signals": signals,
        "escalation": escalation,
        "n_current": int(sum(current.values())),
    }
    log.info(
        "Tripwire (page): %s; escalation over %d current signal(s) -> %s",
        {n: s["state"] for n, s in signals.items()}, result["n_current"], escalation.value,
    )
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Synthetic self-check — no network, no checkpoint dependency (mirrors
    # gap_lag.py / vol.py __main__ footers). Live checkpoint wiring is a
    # human-verification item, same as Phase 1 (RESEARCH.md Environment
    # Availability: FRED_API_KEY / yfinance pending).
    rng = np.random.default_rng(42)
    calm = rng.normal(0, 0.005, 80)
    spike = rng.normal(0, 0.03, 21)
    synthetic_returns = pd.Series(np.concatenate([calm, spike]))

    _idx = pd.bdate_range("2024-01-01", periods=30)
    synthetic_daaa = pd.Series(np.full(30, 4.5), index=_idx)
    synthetic_dbaa = pd.Series(np.linspace(5.2, 5.5, 30), index=_idx)

    synthetic_spy = pd.Series(np.concatenate([np.linspace(400, 450, 60), np.linspace(450, 400, 40)]))

    _escalation = run_tripwire(
        {},
        daily_returns=synthetic_returns,
        daaa=synthetic_daaa,
        dbaa=synthetic_dbaa,
        spy_prices=synthetic_spy,
    )
    print(_escalation.value)  # noqa: T201 — last line: the escalation enum value
