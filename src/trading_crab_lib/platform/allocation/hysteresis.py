"""
Hysteresis state machine — the L4-01 anti-flicker Schmitt trigger (design §5.3).

``update_active_regime()`` tracks P(the CURRENTLY-HELD regime's OWN
probability), never argmax(P) — a competitor regime spiking above the act
threshold must NOT steal the tilt while the held regime's own confidence is
still above the unwind floor (RESEARCH Pitfall 3). This is what keeps the
report from churning Glenn's Fidelity account on every noisy month.

State is persisted load-BEFORE-save via ``load_active_regime()`` /
``save_active_regime()``, mirroring ``labeling/diagnostics.py``'s
``_churn_against_previous()`` ordering (same Pitfall 3 concern: reading the
checkpoint after writing it would silently return the just-written value).

Usage::

    from trading_crab_lib.platform.allocation.hysteresis import (
        load_active_regime, save_active_regime, update_active_regime,
    )

    prev_active = load_active_regime()  # load BEFORE any save this run
    new_active = update_active_regime(probs, prev_active)
    save_active_regime(new_active)

**What gates allocation (plan 08-09, audit item A7).** Design §5.3 reads *"hysteresis
bands (act when P crosses ~0.7; unwind below ~0.4) **and/or** smoothed allocation
response with bounded turnover"*. Glenn ruled the second arm on 2026-09-24
(``08-A7.md``): a **5-percentage-point no-trade band**, not swept.
``execute_rebalance()`` is that band, and it is the ONE implementation all three call
sites (``backtest/driver.py``, ``backtest/joint_driver.py``, ``report/weekly.py``) call.
``active_regime`` still gates nothing — A7 closed by rewording, knowingly. Under the
decision-bearing l1only routing ``update_active_regime`` is provably the identity on
argmax (a one-hot always clears any admissible act threshold); the band is the only
§5.3 mechanism that moves that leg.
"""

from __future__ import annotations

import logging

import pandas as pd

from trading_crab_lib.checkpoints import CheckpointManager
from trading_crab_lib.platform.checkpoints import get_platform_checkpoint_manager

log = logging.getLogger(__name__)


def load_active_regime(cm: CheckpointManager | None = None) -> int | None:
    """Load the previously-acted-on regime. Cold start (no checkpoint yet,
    or a persisted null) returns None."""
    cm = cm or get_platform_checkpoint_manager()
    try:
        state = cm.load("hysteresis_state")
    except FileNotFoundError:
        return None
    value = state["active_regime"].iloc[0]
    return None if pd.isna(value) else int(value)


def save_active_regime(active_regime: int | None, cm: CheckpointManager | None = None) -> None:
    """Persist the acted-on regime (or None for the neutral/cash-heavy posture)."""
    cm = cm or get_platform_checkpoint_manager()
    cm.save(pd.DataFrame([{"active_regime": active_regime}]), "hysteresis_state")


def update_active_regime(
    probs: pd.Series | dict,
    prev_active,
    *,
    act_threshold: float = 0.70,
    unwind_threshold: float = 0.40,
):
    """Schmitt-trigger regime switch — the pure state-transition function.

    Args:
        probs: regime id -> probability (pd.Series or dict).
        prev_active: the previously-held regime id, or None (cold start).
        act_threshold: probability a regime must reach to become/remain active (design §5.3, ~0.7).
        unwind_threshold: probability floor below which the held regime is
            considered to have collapsed (design §5.3, ~0.4).

    Returns:
        The new active regime id, or None (neutral/cash-heavy posture).

    Cold-start rule [ASSUMED — A1, not explicitly specified in design/CONTEXT,
    flagged for the weekly report]: with no prior state, act immediately on
    argmax(probs) if it already clears act_threshold; otherwise stay neutral
    until some regime first crosses the act threshold.
    """
    probs = pd.Series(probs)

    if prev_active is None:
        top_regime = probs.idxmax()
        return top_regime if probs[top_regime] >= act_threshold else None

    # Branch 2: the held regime's own probability is still high enough — HOLD,
    # regardless of any competitor's probability (Pitfall 3: never argmax(P)).
    if probs.get(prev_active, 0.0) >= unwind_threshold:
        return prev_active

    # Branch 3: the held regime has collapsed — switch only to a competitor
    # that has itself crossed the act threshold; otherwise go neutral.
    qualifying = probs[probs >= act_threshold]
    return qualifying.idxmax() if not qualifying.empty else None


# ── thresholds (Task 2 ruling: keep-absolute) ─────────────────────────────────

#: Glenn, 2026-09-24 (``08-A7.md``): act 0.70 / unwind 0.40 for BOTH K=6 and K=5.
DEFAULT_ACT_THRESHOLD: float = 0.70
DEFAULT_UNWIND_THRESHOLD: float = 0.40


def hysteresis_thresholds(cfg: dict) -> tuple[float, float]:
    """``(act_threshold, unwind_threshold)`` from ``cfg["allocation"]["hysteresis"]``.

    An absent key means the ruled value (0.70 / 0.40). A PRESENT value that breaks
    ``0 < unwind <= act <= 1.0`` raises — never a silent fallback — because that
    invariant is what the one-hot identity proof in ``08-A7.md`` rests on.
    """
    block = cfg.get("allocation", {}).get("hysteresis", {}) or {}
    act = block.get("act_threshold", DEFAULT_ACT_THRESHOLD)
    unwind = block.get("unwind_threshold", DEFAULT_UNWIND_THRESHOLD)
    for name, value in (("act_threshold", act), ("unwind_threshold", unwind)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"allocation.hysteresis.{name} must be a number, got {value!r}")
    if not 0.0 < unwind <= act <= 1.0:
        raise ValueError(
            f"allocation.hysteresis thresholds must satisfy 0 < unwind <= act <= 1.0; got "
            f"act={act}, unwind={unwind}. The one-hot identity (08-A7.md) depends on it."
        )
    return float(act), float(unwind)


# ── the no-trade band (Task 1 ruling: b-bounded-turnover) ─────────────────────


def no_trade_band_from_config(cfg: dict) -> float | None:
    """The band half-width from ``cfg["allocation"]["no_trade_band"]``, or None.

    ``None`` — key absent or explicitly null — means NO band: the executed book is the
    target, byte for byte, which is how every pre-08-09 curve was produced and how
    synthetic test configs run. The live config carries 0.05, pinned by a test, so
    deleting the key goes red rather than silently disabling the band. A present value
    outside ``(0, 1)`` raises.
    """
    value = cfg.get("allocation", {}).get("no_trade_band")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 < value < 1.0:
        raise ValueError(f"allocation.no_trade_band must be a number in (0, 1) or null, got {value!r}")
    return float(value)


def execute_rebalance(
    target_weights: pd.Series,
    target_cash: float,
    held_weights: pd.Series | None,
    *,
    band: float | None,
) -> dict:
    """The EXECUTED book for one rebalance: the target, passed through the no-trade band.

    Semantics are Glenn's ruling of 2026-09-24 verbatim (``08-09-PLAN.md`` <rulings>),
    fixed before any number was read:

    - per risky asset ``a``: if ``|target_a - held_a| <= band`` hold ``held_a``,
      otherwise trade to ``target_a``. ``<=``: a move of exactly ``band`` is NOT traded
      (plain floating-point comparison, no tolerance).
    - ``held`` is the last EXECUTED weight (this function's previous output), not a
      drifted one. An asset absent from either side counts as 0 there, so a zero target
      is traded to zero only if its held weight exceeds ``band``.
    - cash is the residual ``1 - sum(risky)`` and is never itself banded.
    - if that residual would be negative, the assets TRADED this step are scaled down
      pro rata until the book sums to 1; held assets are not touched.
    - no ``held`` (``None``: nothing executed yet) trades to target in full.

    ``band=None`` returns the target and ``target_cash`` unchanged (the same objects) —
    the band-disabled path is byte-identical to the pre-08-09 allocator.

    Returns:
        dict with ``weights`` (executed risky weights), ``cash`` (the residual),
        ``held_assets`` / ``traded_assets`` (the band's decision per asset) and
        ``scale_down`` (1.0 unless the negative-residual branch fired).
    """
    if band is None:
        return {
            "weights": target_weights,
            "cash": target_cash,
            "held_assets": [],
            "traded_assets": list(target_weights.index),
            "scale_down": 1.0,
        }
    target = pd.Series(target_weights, dtype=float)
    if held_weights is None:
        weights = target.copy()
        return {
            "weights": weights,
            "cash": 1.0 - float(weights.sum()),
            "held_assets": [],
            "traded_assets": list(weights.index),
            "scale_down": 1.0,
        }

    held = pd.Series(held_weights, dtype=float)
    assets = held.index.union(target.index)
    t = target.reindex(assets, fill_value=0.0)
    h = held.reindex(assets, fill_value=0.0)
    hold = (t - h).abs() <= band
    weights = t.where(~hold, h)

    scale_down = 1.0
    held_sum = float(h[hold].sum())
    traded_sum = float(t[~hold].sum())
    if held_sum + traded_sum > 1.0 and traded_sum > 0.0:
        scale_down = max(0.0, 1.0 - held_sum) / traded_sum
        weights[~hold] = t[~hold] * scale_down
        log.debug("no-trade band: residual would be negative; traded assets scaled by %.6f", scale_down)

    return {
        "weights": weights,
        "cash": 1.0 - float(weights.sum()),
        "held_assets": [a for a in assets if hold[a]],
        "traded_assets": [a for a in assets if not hold[a]],
        "scale_down": scale_down,
    }
