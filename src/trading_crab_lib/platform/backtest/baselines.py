"""Baseline gauntlet — SPY buy-and-hold, 60/40, Faber SMA, no-regime ablation (EVAL-02).

Four comparison legs, all sharing the SAME spliced core research series
(``splice.build_core_research_series`` — ``equities_tr`` / ``long_duration_tr``
/ ``cash``, docs/splicing_rules.md) and the SAME cost convention, so the
design §23.1 Faber comparison and the "does the regime layer pay rent" delta
(design §8.7) are apples-to-apples:

- :func:`spy_buy_hold` — a single cost-free purchase; the equity total-return
  series' monthly returns, unchanged.
- :func:`sixty_forty` — a fixed 60/40 equity/bond mix, reconstituted every
  month back to target (Claude's-discretion cadence, 05-CONTEXT.md), with the
  documented monthly-reconstitution turnover costed when ``cost_bps > 0``.
- :func:`faber_sma` — Meb Faber's 10-month SMA timing rule (D-01a standing
  target), with a strict 1-step decision lag (A4): the position for month
  ``t+1`` is decided from the SMA signal computed through month ``t`` only —
  never the same month's own level (no look-ahead).
- :func:`no_regime_ablation` — design §8.7's "the regime layer must pay rent"
  ablation. This is NOT a hand-rolled parallel equal-weight implementation
  (RESEARCH Anti-Pattern; D-02) — it is a one-line delegation to
  ``backtest.driver.run_backtest(..., use_regime_tilt=False)``, the SAME
  audited L1-L4 code path with the regime tilt disabled. When
  ``cfg["backtest"].get("skip_l1l2_for_ablation", True)`` (review F5), the
  driver additionally skips the discarded L1/L2 refits for that path, so
  building this ablation does not double the ~588-refit walk-forward
  gauntlet's cost, while the equity curve stays byte-identical to the
  vol-target baseline either way.

Cost convention (A5, review F4): ``cost_bps`` and the 60/40 rebalance cadence
are read from ``cfg["backtest"]`` by the CALL SITE (the report layer, Plan
06) — never hard-coded here — so ``apply_cost_to_baselines`` toggles the
SAME bps haircut on every rebalancing baseline as the strategy leg. The cash
sleeve these baselines earn on their non-invested legs (``cash_ret`` /
``cash_returns``) is the SAME series ``run_backtest``'s strategy cash
residual now earns (review F4) — the cash-return convention is symmetric
across the strategy and the baselines, so the Faber comparison is not
quietly biased over the double-digit-yield 1970s-80s.

Trial-registry scope (review F7, resolving RESEARCH.md Open Question 2,
never locked in CONTEXT.md): only the regime-tilt strategy and
``no_regime_ablation`` (via its delegation to ``run_backtest``) log a
registry trial — both are genuinely evaluated configurations with a fitted
(possibly degenerate) model path. ``spy_buy_hold``, ``sixty_forty``, and
``faber_sma`` are deterministic price arithmetic with no tunable parameters;
they log NO trial — they are report-only comparisons, not part of the
multiple-testing surface the registry bounds.

Callers (the report layer) are responsible for supplying already
holdout-bounded (<=2020-12-31) series to ``spy_buy_hold``/``sixty_forty``/
``faber_sma`` — the same discipline ``run_backtest`` already enforces
internally for ``no_regime_ablation`` via ``split_by_holdout_boundary``.

Usage::

    from trading_crab_lib.platform.backtest.baselines import (
        faber_sma, no_regime_ablation, sixty_forty, spy_buy_hold,
    )

    spy_ret = spy_buy_hold(research["equities_tr"].pct_change())
    sixty_forty_ret = sixty_forty(
        equity_ret, bond_ret, rebalance=cfg["backtest"]["sixty_forty_rebalance"],
        cost_bps=cfg["backtest"]["cost_bps"] if cfg["backtest"]["apply_cost_to_baselines"] else 0.0,
    )
    faber_ret = faber_sma(research["equities_tr"], cash_ret, cost_bps=cost_bps)
    equity_curve, per_step_metrics = no_regime_ablation(
        monthly_features, asset_returns, cfg, cash_returns=cash_returns,
    )
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from trading_crab_lib.platform.backtest.costs import apply_transaction_cost, compute_turnover
from trading_crab_lib.platform.backtest.driver import run_backtest

_SUPPORTED_REBALANCE_CONVENTIONS = ("monthly",)


# ── SPY buy-and-hold ──────────────────────────────────────────────────────────


def spy_buy_hold(equity_ret: pd.Series) -> pd.Series:
    """A single cost-free purchase — the equity total-return series' monthly
    returns, unchanged.

    No rebalancing ever occurs (buy once, hold forever), so there is no
    turnover to cost — this is the only baseline leg that is cost-free by
    construction, not by a ``cost_bps=0`` parameter.
    """
    return equity_ret


# ── 60/40 (fixed-mix, monthly reconstitution) ────────────────────────────────


def sixty_forty(
    equity_ret: pd.Series,
    bond_ret: pd.Series,
    *,
    rebalance: str = "monthly",
    cost_bps: float = 0.0,
) -> pd.Series:
    """Hold 60% equity / 40% bond, reconstituted back to target every month.

    Convention (Claude's discretion, 05-CONTEXT.md; ``cfg["backtest"]
    ["sixty_forty_rebalance"]``): each month's GROSS return is the
    0.6/0.4-weighted blend of that month's equity/bond returns (this is
    exact, because the portfolio started the month AT the 60/40 target —
    it was reconstituted back to target at the end of the prior month).
    That same month's return realization then drifts the mix away from
    60/40; the turnover required to reconstitute back to target for the
    NEXT month is computed and costed against the CURRENT month's return
    (the trade happens at month-end, same convention as
    ``backtest/costs.py``'s target-vs-target turnover).

    Args:
        equity_ret: monthly equity-leg returns.
        bond_ret: monthly bond-leg returns, aligned by index to ``equity_ret``.
        rebalance: reconstitution cadence — only ``"monthly"`` is implemented.
        cost_bps: bps-of-turnover haircut (A5) — 0.0 (default) is frictionless.

    Returns:
        pd.Series: net-of-cost monthly returns, indexed like the intersection
        of ``equity_ret``/``bond_ret``.

    Raises:
        ValueError: if ``rebalance`` is not a supported convention.
    """
    if rebalance not in _SUPPORTED_REBALANCE_CONVENTIONS:
        raise ValueError(
            f"Unsupported rebalance convention {rebalance!r} — only "
            f"{_SUPPORTED_REBALANCE_CONVENTIONS} implemented."
        )

    common = equity_ret.index.intersection(bond_ret.index)
    equity_ret = equity_ret.loc[common].sort_index()
    bond_ret = bond_ret.loc[common].sort_index()

    target = pd.Series({"equity": 0.6, "bond": 0.4})
    net_returns: list[float] = []
    for date in equity_ret.index:
        e = float(equity_ret.loc[date])
        b = float(bond_ret.loc[date])
        gross = 0.6 * e + 0.4 * b

        # Post-return drift away from the 60/40 target this month —
        # the turnover to reconstitute back to target for next month.
        equity_val = 0.6 * (1.0 + e)
        bond_val = 0.4 * (1.0 + b)
        total = equity_val + bond_val
        drifted = pd.Series({"equity": equity_val / total, "bond": bond_val / total}) if total != 0 else target
        turnover = compute_turnover(drifted, target)

        net_returns.append(apply_transaction_cost(gross, turnover, cost_bps))

    return pd.Series(net_returns, index=equity_ret.index, name="sixty_forty")


# ── Faber 10-month SMA (1-step decision lag, A4) ─────────────────────────────


def _faber_position(equity_level: pd.Series, window: int = 10) -> pd.Series:
    """Boolean in-market position with a strict 1-step decision lag (A4).

    The SMA signal at month ``t`` (``equity_level.loc[t] > sma.loc[t]``) uses
    data available AT THE CLOSE of month ``t`` (the level itself is known at
    close of ``t``) — this is a legitimate end-of-month decision, never a
    same-month look-ahead. ``.shift(1)`` then ACTS on that decision starting
    month ``t+1``: the position at ``t+1`` reflects the signal decided at the
    close of ``t``, never ``t+1``'s own level. The first month has no prior
    decision to act on and defaults to out-of-market (``False``).
    """
    sma = equity_level.rolling(window).mean()
    raw_signal = equity_level > sma
    position = raw_signal.shift(1).fillna(False).astype(bool)
    position.name = "faber_position"
    return position


def faber_sma(
    equity_level: pd.Series,
    cash_ret: pd.Series,
    *,
    window: int = 10,
    cost_bps: float = 0.0,
) -> pd.Series:
    """Meb Faber's N-month SMA timing rule (D-01a standing target).

    In-market months earn the equity leg's realized monthly return
    (computed from ``equity_level.pct_change()``); out-of-market months earn
    ``cash_ret`` — the SAME cash-return series the strategy's cash residual
    now earns (review F4), so the comparison is symmetric. A full switch
    (entering or exiting the market) costs ``cost_bps`` against that month's
    realized return; a month that holds its prior position costs nothing.

    Args:
        equity_level: monthly equity-leg total-return LEVEL series (e.g.
            ``research["equities_tr"]`` from ``splice.py``) — a level, not a
            return, because the SMA signal operates on levels.
        cash_ret: monthly cash-leg returns, aligned by index.
        window: SMA lookback in months (Faber's standing default: 10).
        cost_bps: bps-of-turnover haircut (A5) charged only on switch months.

    Returns:
        pd.Series: net-of-cost monthly returns.
    """
    position = _faber_position(equity_level, window=window)
    equity_ret = equity_level.pct_change()

    common = equity_ret.index.intersection(cash_ret.index).intersection(position.index)
    position = position.loc[common]
    equity_ret = equity_ret.loc[common]
    cash_ret = cash_ret.loc[common]

    gross = pd.Series(
        np.where(position.to_numpy(), equity_ret.to_numpy(), cash_ret.to_numpy()),
        index=common,
    )
    switches = position.astype(float).diff().abs().fillna(0.0)
    net = apply_transaction_cost(gross, switches, cost_bps)
    net.name = "faber_sma"
    return net


# ── No-regime ablation (design §8.7, F5, D-02) ───────────────────────────────


def no_regime_ablation(
    monthly_features: pd.DataFrame,
    asset_returns: pd.DataFrame,
    cfg: dict[str, Any],
    *,
    cash_returns: pd.Series | None = None,
    registry_path: Any = None,
) -> tuple[pd.DataFrame, dict[str, list]]:
    """The no-regime ablation — a ONE-LINE delegation to the tilt-off driver path.

    This is the SAME ``run_backtest`` code path with ``use_regime_tilt=False``
    — NOT a hand-rolled parallel equal-weight implementation (RESEARCH
    Anti-Pattern; D-02). No allocation math is reimplemented here; that is
    exactly what makes the "ablation reproduces the vol-target baseline"
    invariant hold (``TestNoRegimeAblationInvariant``).

    ``cash_returns`` is passed straight through so the ablation's cash
    residual earns the SAME series the strategy leg does (review F4) —
    omitting it would silently default the ablation's cash sleeve to 0%,
    which would break the cash-return symmetry this whole plan documents.

    When ``cfg["backtest"].get("skip_l1l2_for_ablation", True)`` (review F5),
    ``run_backtest`` skips the discarded L1 jump-model + L2 nowcaster
    refits for this tilt-off path (their output is unused when the tilt is
    off), so building this ablation does NOT double the ~588-refit
    walk-forward gauntlet's cost — and the ablation equity curve remains
    byte-identical to the vol-target baseline either way (the invariant
    still holds regardless of the skip).

    Trial-registry scope (review F7): unlike ``spy_buy_hold``/
    ``sixty_forty``/``faber_sma`` (deterministic price arithmetic, no
    tunable parameters, no registry trial logged), this ablation DOES log
    its own registry trial via ``run_backtest``'s single ``append_trial``
    call — it is a genuine evaluated configuration with a fitted (here,
    degenerate single-state) model path, and therefore is part of the
    multiple-testing surface the registry bounds.

    Returns:
        tuple[pd.DataFrame, dict[str, list]]: the SAME
        ``(equity_curve, per_step_metrics)`` contract as ``run_backtest``.
    """
    return run_backtest(
        monthly_features,
        asset_returns,
        cfg,
        cash_returns=cash_returns,
        use_regime_tilt=False,
        registry_path=registry_path,
    )


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)

    # Synthetic self-check — no network, no checkpoint dependency.
    _rng = np.random.default_rng(42)
    _idx = pd.date_range("2010-01-31", periods=48, freq="ME")
    _equity_level = pd.Series(100 * (1 + _rng.normal(0.006, 0.03, 48)).cumprod(), index=_idx)
    _bond_level = pd.Series(100 * (1 + _rng.normal(0.001, 0.01, 48)).cumprod(), index=_idx)
    _equity_ret = _equity_level.pct_change()
    _bond_ret = _bond_level.pct_change()
    _cash_ret = pd.Series(_rng.normal(0.001, 0.0005, 48), index=_idx)

    _spy = spy_buy_hold(_equity_ret)
    _sixty_forty = sixty_forty(_equity_ret, _bond_ret, rebalance="monthly", cost_bps=10)
    _faber = faber_sma(_equity_level, _cash_ret, window=10, cost_bps=10)

    print("spy_buy_hold terminal wealth:  ", float((1 + _spy.dropna()).prod()))  # noqa: T201
    print("sixty_forty terminal wealth:   ", float((1 + _sixty_forty.dropna()).prod()))  # noqa: T201
    print("faber_sma terminal wealth:     ", float((1 + _faber.dropna()).prod()))  # noqa: T201

    _lean_cols = [
        "curve_10y3m", "curve_10y2y", "credit_spread_baa_aaa", "fred_vix", "gold", "oil",
        "trailing_return_1m", "trailing_return_3m", "realized_vol_1m", "realized_vol_3m",
        "cape_shiller", "div_yield", "real_rate_level",
    ]
    _monthly_features = pd.DataFrame({c: _rng.normal(0, 1, 48) for c in _lean_cols}, index=_idx)
    _asset_returns = pd.DataFrame(
        {"SPY": _rng.normal(0.006, 0.03, 48), "TLT": _rng.normal(0.001, 0.02, 48)}, index=_idx
    )
    _cfg = {
        "taxonomy": {"fast": _lean_cols[:10], "slow": _lean_cols[10:], "agency": []},
        "labeling": {"K": 2, "lambda": 5.0, "n_restarts": 2, "embargo_months": 3},
        "allocation": {
            "target_vol_annual": 0.10, "ewma_halflife_months": 6, "portfolio_vol_min_obs": 3,
            "hysteresis": {"act_threshold": 0.70, "unwind_threshold": 0.40},
        },
        "backtest": {"cost_bps": 10, "min_train_months": 24, "skip_l1l2_for_ablation": True},
    }
    _ablation_equity, _ = no_regime_ablation(_monthly_features, _asset_returns, _cfg, cash_returns=_cash_ret)
    print("no_regime_ablation equity curve:\n", _ablation_equity)  # noqa: T201
