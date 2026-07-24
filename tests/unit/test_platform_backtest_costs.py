"""Unit tests for trading_crab_lib.platform.backtest.costs (D-03).

Follows the incumbent tests/unit/test_platform_gap_lag.py structure: a
class per behavior, docstring per test. All computation runs against
hand-built synthetic weight Series/floats — no network, no checkpoints.
"""

from __future__ import annotations

import pandas as pd
import pytest

from trading_crab_lib.platform.backtest.costs import (
    apply_transaction_cost,
    compute_turnover,
)

# ── compute_turnover ──────────────────────────────────────────────────────────


class TestTurnover:
    def test_turnover_is_sum_of_absolute_changes(self):
        """turnover == sum(|new - prev|) over the union of both weight indices."""
        prev = pd.Series({"SPY": 0.5, "TLT": 0.3, "cash": 0.2})
        new = pd.Series({"SPY": 0.4, "TLT": 0.4, "cash": 0.2})
        assert compute_turnover(prev, new) == pytest.approx(0.2)

    def test_asset_present_in_only_new_is_treated_as_zero_in_prev(self):
        """An asset with no prior position contributes its full new weight to turnover."""
        prev = pd.Series({"SPY": 0.5})
        new = pd.Series({"SPY": 0.3, "GLD": 0.2})
        # |0.3 - 0.5| + |0.2 - 0.0| == 0.2 + 0.2 == 0.4
        assert compute_turnover(prev, new) == pytest.approx(0.4)

    def test_asset_present_in_only_prev_is_treated_as_zero_in_new(self):
        """An asset dropped entirely contributes its full prior weight to turnover."""
        prev = pd.Series({"SPY": 0.5, "USO": 0.1})
        new = pd.Series({"SPY": 0.5})
        # |0.5-0.5| + |0.0-0.1| == 0.1
        assert compute_turnover(prev, new) == pytest.approx(0.1)

    def test_cold_start_empty_prev_yields_new_abs_sum(self):
        """An empty prev Series (cold start) yields turnover == new.abs().sum()."""
        prev = pd.Series(dtype=float)
        new = pd.Series({"SPY": 0.4, "TLT": 0.3, "cash": 0.3})
        assert compute_turnover(prev, new) == pytest.approx(new.abs().sum())

    def test_identical_weights_yield_zero_turnover(self):
        prev = pd.Series({"SPY": 0.6, "cash": 0.4})
        new = pd.Series({"SPY": 0.6, "cash": 0.4})
        assert compute_turnover(prev, new) == pytest.approx(0.0)


# ── apply_transaction_cost ────────────────────────────────────────────────────


class TestCostIdentity:
    def test_cost_identity_known_answer(self):
        """apply_transaction_cost(gross, turnover, cost_bps) == gross - turnover*cost_bps/1e4."""
        gross = 0.02
        turnover = 0.5
        cost_bps = 10
        expected_cost = 0.5 * 10 / 1e4  # == 0.0005
        assert expected_cost == pytest.approx(0.0005)
        assert apply_transaction_cost(gross, turnover, cost_bps) == pytest.approx(gross - expected_cost)

    def test_zero_cost_bps_returns_gross_unchanged(self):
        assert apply_transaction_cost(0.02, 0.5, 0.0) == pytest.approx(0.02)

    def test_frictionless_vs_costed_delta_equals_turnover_times_bps(self):
        """For the same weight sequence, (frictionless_net - costed_net) == turnover * cost_bps / 1e4
        to full float precision — the honesty invariant named in 05-CONTEXT code_context."""
        prev = pd.Series({"SPY": 0.5, "TLT": 0.3, "cash": 0.2})
        new = pd.Series({"SPY": 0.3, "TLT": 0.4, "cash": 0.3})
        turnover = compute_turnover(prev, new)
        gross_return = 0.015
        cost_bps = 12.5

        frictionless_net = apply_transaction_cost(gross_return, turnover, cost_bps=0.0)
        costed_net = apply_transaction_cost(gross_return, turnover, cost_bps=cost_bps)

        assert frictionless_net - costed_net == pytest.approx(turnover * cost_bps / 1e4)
