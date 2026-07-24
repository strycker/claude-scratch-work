"""
Turnover + transaction-cost accounting (D-03).

A monthly regime tilt must not look free. ``compute_turnover`` measures how
much the target weight vector moved between two consecutive rebalances;
``apply_transaction_cost`` applies a config-driven bps haircut proportional
to that turnover. Together they prove the honesty invariant this phase
tests: a frictionless run (``cost_bps=0``) and a costed run differ by
exactly ``turnover * cost_bps / 1e4`` for the same weight sequence.

Turnover convention (Pitfall 3): this is TARGET-vs-TARGET turnover — the
absolute change between the previous rebalance's target weights and the new
rebalance's target weights. It does NOT drift-adjust the previous weights
forward by realized asset returns before diffing (that would require the
asset-return path between rebalances, which the cost module does not own).
This is a defensible, documented simplification, not an oversight: for a
monthly-rebalance, hysteresis-damped, low-turnover tilt the drift between
rebalances is small relative to the tilt's own reallocation, so target-vs-
target is a conservative-enough proxy for actual traded notional.

Usage::

    from trading_crab_lib.platform.backtest.costs import (
        apply_transaction_cost, compute_turnover,
    )

    turnover = compute_turnover(prev_weights, new_weights)
    net_return = apply_transaction_cost(gross_return, turnover, cost_bps=10)
"""

from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger(__name__)


def compute_turnover(prev_weights: pd.Series, new_weights: pd.Series) -> float:
    """Sum of absolute per-asset weight changes over the union of both indices.

    An asset present in only one Series is treated as weight 0.0 in the
    other (index-union alignment via ``reindex(..., fill_value=0.0)`` —
    never positional diffing, which would silently misalign differently-
    ordered or differently-sized weight vectors). An empty ``prev_weights``
    (cold start) yields ``turnover == new_weights.abs().sum()``.

    Args:
        prev_weights: Target weights from the previous rebalance (may be
            empty on a cold start).
        new_weights: Target weights from the current rebalance.

    Returns:
        Non-negative float turnover, e.g. 0.5 means half the portfolio
        (by weight) was reallocated at this rebalance.
    """
    union_index = prev_weights.index.union(new_weights.index)
    prev_aligned = prev_weights.reindex(union_index, fill_value=0.0)
    new_aligned = new_weights.reindex(union_index, fill_value=0.0)
    return float((new_aligned - prev_aligned).abs().sum())


def apply_transaction_cost(gross_return: float, turnover: float, cost_bps: float) -> float:
    """Subtract a bps-of-turnover haircut from a gross return (D-03 identity).

    ``net_return == gross_return - turnover * cost_bps / 1e4``. With
    ``cost_bps=0`` this returns ``gross_return`` unchanged — the
    frictionless case used as the honesty-invariant baseline.

    Args:
        gross_return: The period's return before transaction costs.
        turnover: Non-negative turnover for this rebalance (see
            :func:`compute_turnover`).
        cost_bps: Cost in basis points charged per unit of turnover
            (config ``backtest.cost_bps``, default 10).

    Returns:
        Net-of-cost return as a float.
    """
    return gross_return - turnover * cost_bps / 1e4


if __name__ == "__main__":
    # Synthetic self-check (no network, no checkpoint) — mirrors the
    # honesty/gap_lag.py module shape's __main__ demonstration.
    _prev = pd.Series({"SPY": 0.5, "TLT": 0.3, "cash": 0.2})
    _new = pd.Series({"SPY": 0.3, "TLT": 0.4, "cash": 0.3})
    _turnover = compute_turnover(_prev, _new)
    _gross = 0.015
    _cost_bps = 10
    _net = apply_transaction_cost(_gross, _turnover, _cost_bps)
    print(f"turnover={_turnover:.4f} gross={_gross:.4f} cost_bps={_cost_bps} net={_net:.6f}")
