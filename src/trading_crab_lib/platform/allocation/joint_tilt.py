"""
Joint (classifier #1 + classifier #2) allocation blend — D-14, plan 07-10.

**This module is genuinely new code, not a parameterization of an existing
function.** Every function in ``allocation/tilt.py`` accepts exactly ONE
probability input: ``regime_tilt_weights(regime, returns_by_regime, probs,
...)`` and ``vol_targeted_tilt(regime_or_probs, ...)``. D-14 requires the two
labelings' probability vectors to reach the allocation tilt as two SEPARATE
inputs, and there is no existing signature that can express that. Only the
pre-scaling weight blend is new; ``regime_tilt_weights``, ``portfolio_vol`` and
``vol_target_scale`` are composed unmodified.

**D-14's prohibition, stated plainly: the blend happens at the WEIGHT level and
a product ``(state_1, state_2)`` state space is never formed.** A product space
would thin badly — K1 x K2 cells over roughly 590 decision months, with
occupancy never uniform, puts rare cells far below design §4.4 criterion 1's
~8% floor. Nothing in this module ever indexes by a state pair; the two
``returns_by_regime`` tables are consumed independently and only the resulting
weight Series are combined.

**ADR-0001 § RE-PIN 2026-09-18, recurrence-exemption condition (iv).**
Classifier #1's crisis state occupies 5.7554% of months — below §4.4's ~8%
floor — and is admitted under the recurrence exemption. Condition (iv) makes
that admission conditional:

    every downstream statistic computed on it carries an explicit low-n flag,
    and it is *not* used for unshrunk point estimates — no per-regime Sharpe,
    no per-regime covariance — without partial pooling toward the all-history
    model (design §6.1 mitigation 2).

``regime_tilt_weights`` consumes exactly such an estimate: the per-regime
annualized Sharpe. So this module does two things before any tilt is computed:

1. **Flags** every regime below the floor (``low_n_regime_flags``), at WARNING,
   naming the regime, its occupancy, the floor and the credibility applied — so
   a caller who never read the ADR still sees it.
2. **Partially pools** that regime's per-regime Sharpe toward the all-history
   estimate (``pool_low_n_regime_sharpe``), with credibility
   ``min(1, occupancy / floor)``. The weight is derived from the criterion
   being exempted rather than tuned: the floor is §4.4's proxy for "enough
   observations to trust", so a state gets exactly the credibility its
   occupancy earns against that floor, and a state that MEETS the floor is
   left untouched — the pooling is a strict no-op for a compliant labeling.

This is adjacent to, and deliberately not a reuse of, ``tilt.py``'s existing
``min_obs_flag`` seam. That seam flags a per-CELL short asset history (D11) at
WARNING and changes no estimate; condition (iv) is about per-REGIME sample
size and requires the estimate itself to move. ``min_obs_flag`` is threaded
through unchanged so both flags fire.

Usage::

    from trading_crab_lib.platform.allocation.joint_tilt import (
        blend_regime_tilts, blend_weight_from_config,
    )

    result = blend_regime_tilts(
        probs_1, returns_by_regime_1,
        probs_2, returns_by_regime_2,
        asset_returns,
        weight_1=blend_weight_from_config(cfg),
        target_vol_annual=0.10, halflife=6,
    )
    # result == {"weights": pd.Series, "cash": float, "scale": float, "portfolio_vol": float}
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd

from trading_crab_lib.platform.allocation.tilt import (
    portfolio_vol,
    regime_tilt_weights,
    vol_target_scale,
)

log = logging.getLogger(__name__)

#: ADR-0002 decision (f) / D-14. The equal-weight no-information prior.
DEFAULT_BLEND_WEIGHT_1 = 0.50

#: Design §4.4 criterion 1 (as amended 2026-09-18): every state >= ~8% of
#: months, except one state per labeling under the recurrence exemption.
OCCUPANCY_FLOOR = 0.08

_SHARPE_COL = "sharpe_annualized"
_MOMENT_COLS = ("mean_monthly_return", "std_monthly_return")


# ── config: the blend weight is pre-declared and never swept ────────────────


def blend_weight_from_config(cfg: dict) -> float:
    """Read ``allocation.blend_weight_1`` defensively, ADR-0002's value as fallback.

    **This constant is pre-declared and NEVER swept.** Sweeping it would be an
    unregistered selection dimension the trial ceiling in ADR-0001 does not
    budget for, and D-13 spends zero selection trials on classifier #2's
    constants. It becomes tunable only at design freeze, at which point the
    blend earns its own ADR.

    A missing key logs at WARNING rather than defaulting silently — a config
    that lost the key should be visible, not absorbed.
    """
    allocation = cfg.get("allocation", {}) if isinstance(cfg, dict) else {}
    if "blend_weight_1" not in allocation:
        log.warning(
            "allocation.blend_weight_1 absent from config — falling back to the "
            "ADR-0002-pinned %.2f. This value is pre-declared and never swept.",
            DEFAULT_BLEND_WEIGHT_1,
        )
        return DEFAULT_BLEND_WEIGHT_1
    value = float(allocation["blend_weight_1"])
    _validate_unit_interval(value, name="blend_weight_1")
    return value


def _validate_unit_interval(value: float, *, name: str) -> float:
    """Reject anything outside the closed unit interval, by name.

    A blend weight above one or below zero produces a negative contribution
    from the complementary leg — a short position, silently violating the
    long-only constraint the rest of the stack assumes cannot be violated
    (threat T-07-22).
    """
    numeric = float(value)
    if math.isnan(numeric) or numeric < 0.0 or numeric > 1.0:
        raise ValueError(
            f"{name} must lie in the closed interval [0, 1]; got {value!r}. "
            "A value outside it would produce a negative (short) weight."
        )
    return numeric


# ── ADR-0001 condition (iv): occupancy, flagging, partial pooling ───────────


def regime_occupancy(returns_by_regime: pd.DataFrame) -> pd.Series:
    """Per-regime share of months, estimated from the long-format stats table.

    A regime's month count is estimated as the MAX ``n_obs`` across its assets:
    within one regime, the longest-history asset is present in every month the
    regime occupies, while short-history assets (D11) under-count it. This is
    an estimate, not the label counts themselves — callers that hold the actual
    state Series (the walk-forward driver does) should pass ``occupancy``
    explicitly to the functions below rather than rely on it.
    """
    if returns_by_regime is None or returns_by_regime.empty:
        return pd.Series(dtype=float)
    months = returns_by_regime.groupby("regime")["n_obs"].max().astype(float)
    total = months.sum()
    if total <= 0:
        return pd.Series(0.0, index=months.index, dtype=float)
    return months / total


def low_n_regime_flags(
    returns_by_regime: pd.DataFrame,
    *,
    occupancy_floor: float = OCCUPANCY_FLOOR,
    occupancy: pd.Series | dict | None = None,
) -> pd.DataFrame:
    """One row per regime: occupancy, floor, credibility and the low-n flag.

    ``credibility = min(1, occupancy / occupancy_floor)`` — the weight the
    regime's own estimate carries in ``pool_low_n_regime_sharpe``. A regime at
    or above the floor scores 1.0 and is untouched; the exempted sub-floor
    state scores its occupancy's share of the floor.

    This is ADR-0001 condition (iv)'s "explicit low-n flag", in a form a caller
    can read, log or persist without having read the ADR.
    """
    if returns_by_regime is None or returns_by_regime.empty:
        return pd.DataFrame(columns=["regime", "occupancy", "floor", "credibility", "low_n"])

    shares = pd.Series(occupancy, dtype=float) if occupancy is not None else regime_occupancy(returns_by_regime)
    rows = []
    for regime_id in returns_by_regime["regime"].drop_duplicates():
        share = float(shares.get(regime_id, float("nan")))
        credibility = 1.0 if not np.isfinite(share) else min(1.0, share / occupancy_floor)
        rows.append(
            {
                "regime": regime_id,
                "occupancy": share,
                "floor": float(occupancy_floor),
                "credibility": float(credibility),
                "low_n": bool(np.isfinite(share) and share < occupancy_floor),
            }
        )
    return pd.DataFrame(rows)


def _all_history_sharpe(returns_by_regime: pd.DataFrame) -> pd.Series:
    """Per-asset all-history annualized Sharpe, pooled from the per-regime cells.

    When the table carries the per-cell moments (``returns_by_regime_stats``
    produces them), the full-sample mean and standard deviation are recovered
    EXACTLY from the per-regime sufficient statistics — the regimes partition
    the asset's observed months, so::

        N   = sum(n_i)
        M   = sum(n_i * m_i) / N
        TSS = sum((n_i - 1) * s_i**2 + n_i * (m_i - M)**2)
        S   = sqrt(TSS / (N - 1))

    reproduces ``returns.mean()`` and ``returns.std()`` over the whole history.
    This is the "all-history model" design §6.1 mitigation 2 shrinks toward.

    A hand-built table carrying only ``sharpe_annualized`` and ``n_obs`` (the
    shape several unit fixtures use) cannot support that reconstruction; it
    falls back to the ``n_obs``-weighted mean Sharpe, which is the same
    quantity to first order and is documented as an approximation.
    """
    has_moments = all(col in returns_by_regime.columns for col in _MOMENT_COLS)
    pooled: dict = {}
    for asset, group in returns_by_regime.groupby("asset"):
        n = group["n_obs"].astype(float)
        total_n = float(n.sum())
        if total_n <= 1:
            pooled[asset] = float("nan")
            continue
        if has_moments and group[list(_MOMENT_COLS)].notna().all().all():
            means = group["mean_monthly_return"].astype(float)
            stds = group["std_monthly_return"].astype(float)
            grand_mean = float((n * means).sum() / total_n)
            tss = float((((n - 1.0) * stds.pow(2)) + (n * (means - grand_mean).pow(2))).sum())
            pooled_std = math.sqrt(tss / (total_n - 1.0)) if tss > 0 else 0.0
            pooled[asset] = (grand_mean / pooled_std) * math.sqrt(12) if pooled_std > 0 else float("nan")
        else:
            sharpe = group[_SHARPE_COL].astype(float).fillna(0.0)
            pooled[asset] = float((n * sharpe).sum() / total_n)
    return pd.Series(pooled, dtype=float)


def pool_low_n_regime_sharpe(
    returns_by_regime: pd.DataFrame,
    *,
    occupancy_floor: float = OCCUPANCY_FLOOR,
    occupancy: pd.Series | dict | None = None,
    return_pooled: bool = False,
):
    """Partially pool sub-floor regimes' Sharpes toward the all-history model.

    Implements ADR-0001 § RE-PIN 2026-09-18 recurrence-exemption condition
    (iv) / design §6.1 mitigation 2. For every (regime, asset) cell whose
    regime occupies less than ``occupancy_floor`` of months::

        sharpe' = credibility * sharpe + (1 - credibility) * all_history_sharpe

    with ``credibility = occupancy / occupancy_floor``. Regimes at or above the
    floor are returned unchanged, so this is a strict no-op for a labeling that
    satisfies §4.4 criterion 1 without the exemption.

    Returns ``(pooled_table, flags)``, or ``(all_history_sharpe, flags)`` when
    ``return_pooled`` is True — the latter exposes the shrinkage TARGET so it
    can be checked against an independently computed full-sample Sharpe.
    """
    flags = low_n_regime_flags(
        returns_by_regime, occupancy_floor=occupancy_floor, occupancy=occupancy
    )
    if returns_by_regime is None or returns_by_regime.empty:
        return (pd.Series(dtype=float) if return_pooled else returns_by_regime), flags

    pooled_sharpe = _all_history_sharpe(returns_by_regime)
    if return_pooled:
        return pooled_sharpe, flags

    low_n = flags[flags["low_n"]]
    if low_n.empty:
        return returns_by_regime, flags

    table = returns_by_regime.copy()
    for row in low_n.itertuples():
        credibility = float(row.credibility)
        mask = table["regime"] == row.regime
        target = table.loc[mask, "asset"].map(pooled_sharpe)
        raw = table.loc[mask, _SHARPE_COL].astype(float)
        # An asset with no poolable all-history estimate keeps its own value —
        # shrinking toward NaN would erase the cell entirely.
        table.loc[mask, _SHARPE_COL] = np.where(
            target.isna(), raw, credibility * raw + (1.0 - credibility) * target.fillna(0.0)
        )
        log.warning(
            "ADR-0001 condition (iv): regime %s is low-n — occupancy %.4f below design "
            "§4.4 criterion 1's ~%.0f%% floor. Its per-regime Sharpe is partially pooled "
            "toward the all-history estimate at credibility %.4f (design §6.1 mitigation 2); "
            "it is NOT an unshrunk point estimate.",
            row.regime, row.occupancy, occupancy_floor * 100, credibility,
        )
    return table, flags


# ── the blend itself ────────────────────────────────────────────────────────


def _degenerate_result() -> dict:
    """Byte-for-byte ``vol_targeted_tilt``'s own degenerate branch."""
    return {"weights": pd.Series(dtype=float), "cash": 1.0, "scale": 0.0, "portfolio_vol": float("nan")}


def _single_classifier_tilt(
    probs,
    returns_by_regime: pd.DataFrame,
    *,
    min_obs_flag: int,
    occupancy_floor: float,
    occupancy,
) -> pd.Series:
    """One classifier's pre-scaling tilt, after condition (iv) pooling."""
    probs_series = pd.Series(probs, dtype=float) if probs is not None else pd.Series(dtype=float)
    probs_series = probs_series[probs_series > 0]
    if probs_series.empty or returns_by_regime is None or returns_by_regime.empty:
        return pd.Series(dtype=float)

    pooled_table, _ = pool_low_n_regime_sharpe(
        returns_by_regime, occupancy_floor=occupancy_floor, occupancy=occupancy
    )
    regime = probs_series.idxmax()
    return regime_tilt_weights(regime, pooled_table, probs_series, min_obs_flag=min_obs_flag)


def blend_regime_tilts(
    probs_1,
    returns_by_regime_1: pd.DataFrame,
    probs_2,
    returns_by_regime_2: pd.DataFrame,
    asset_returns: pd.DataFrame,
    *,
    weight_1: float = DEFAULT_BLEND_WEIGHT_1,
    target_vol_annual: float = 0.10,
    halflife: float,
    min_obs: int = 12,
    min_obs_flag: int = 6,
    occupancy_floor: float = OCCUPANCY_FLOOR,
    occupancy_1: pd.Series | dict | None = None,
    occupancy_2: pd.Series | dict | None = None,
) -> dict:
    """Blend two labelings' tilts at the weight level (D-14), preserving the contract.

    The two probability vectors are consumed INDEPENDENTLY — each through its
    own ``regime_tilt_weights`` call against its own ``returns_by_regime``
    table — and only the two resulting weight Series are combined, with
    ``weight_1`` on classifier #1 and its complement on classifier #2. No
    ``(state_1, state_2)`` product index is ever formed.

    Returns the SAME four keys ``vol_targeted_tilt`` returns — ``weights``,
    ``cash``, ``scale``, ``portfolio_vol`` — so no downstream consumer
    special-cases joint versus single. At ``weight_1`` of 1.0 the result is
    element-wise identical to ``vol_targeted_tilt(probs_1,
    returns_by_regime_1, asset_returns, ...)``, and at 0.0 to the
    classifier-#2-only path, **provided neither labeling has a sub-floor
    regime**. When one does, the blend intentionally diverges: it applies
    condition (iv)'s partial pooling, which ``vol_targeted_tilt`` does not.
    That divergence is the point, and it is logged at WARNING.

    Raises:
        ValueError: if ``weight_1`` lies outside the closed unit interval.
    """
    weight_1 = _validate_unit_interval(weight_1, name="weight_1")

    tilt_1 = _single_classifier_tilt(
        probs_1, returns_by_regime_1,
        min_obs_flag=min_obs_flag, occupancy_floor=occupancy_floor, occupancy=occupancy_1,
    )
    tilt_2 = _single_classifier_tilt(
        probs_2, returns_by_regime_2,
        min_obs_flag=min_obs_flag, occupancy_floor=occupancy_floor, occupancy=occupancy_2,
    )

    legs = [
        (weight_1, tilt_1, "classifier #1"),
        (1.0 - weight_1, tilt_2, "classifier #2"),
    ]
    usable = [(w, t, name) for w, t, name in legs if not t.empty and t.sum() > 0]
    if not usable:
        return _degenerate_result()

    if len(usable) < len(legs):
        log.warning(
            "joint tilt: only %s contributed — the other labeling produced no usable tilt. "
            "Degrading to the surviving classifier rather than to cash.",
            ", ".join(name for _, _, name in usable),
        )

    if sum(w for w, _, _ in usable) <= 0:
        # The only surviving leg carries zero blend weight; use it rather than
        # returning all-cash, which would silently discard a real signal.
        usable = [(1.0 / len(usable), t, name) for _, t, name in usable]

    contributing = [(w, t) for w, t, _ in usable if w > 0]
    if len(contributing) == 1:
        # One leg: scaling then renormalizing is the identity, and skipping it
        # keeps the endpoint reduction EXACT rather than exact-to-float-noise.
        blended = contributing[0][1]
    else:
        blended = pd.Series(dtype=float)
        for weight, tilt in contributing:
            blended = blended.add(tilt * weight, fill_value=0.0)
        total = blended.sum()
        if blended.empty or total <= 0:
            return _degenerate_result()
        blended = blended / total

    if blended.empty or blended.sum() <= 0:
        return _degenerate_result()

    port_vol = portfolio_vol(blended, asset_returns, halflife=halflife, min_obs=min_obs)
    scale = vol_target_scale(target_vol_annual, port_vol)
    return {
        "weights": blended * scale,
        "cash": 1.0 - scale,
        "scale": scale,
        "portfolio_vol": port_vol,
    }
