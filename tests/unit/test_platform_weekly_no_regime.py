"""
Tests for the weekly page on the no-regime path (plan 08.2-01, D-01 / D-03, DECISIONS A-13, A-15).

``report.allocation_mode: no_regime`` puts the weekly target on the measured no-regime ablation
leg: a constant one-state belief, ``returns_by_regime_stats`` over the asset returns up to the
scored month, the same ``vol_targeted_tilt`` and the same 5 pp no-trade band. The regime
sections of the page collapse into one suspended block. A mode switch executes the new target in
full and says so (A-15).

Every test runs on the synthetic serving world of ``test_platform_report_serving.py`` (reused,
never edited) in per-test tmp dirs, except the tracked-data parity test, which copies the
tracked platform checkpoints into a tmp dir and passes ``NO_REGISTRY`` to every refit.
"""

from __future__ import annotations

import re

import pandas as pd
import pytest

from test_platform_report_serving import _serving_world  # noqa: E402  (tests/unit is on sys.path)

from trading_crab_lib.platform.config import load_platform_config
from trading_crab_lib.platform.report import weekly

_D03_SENTENCE = (
    "Regime view: suspended — the served nowcaster is input-independent (fixed in the regime "
    "rebuild); the allocation does not use it."
)
_REGIME_LINES = ("- regime ", "Filtered Regime Belief", "Active Regime", "Trajectory", "Per-Asset Signals")


def _report(world: dict) -> str:
    return (world["out_dir"] / "reports" / "platform" / "weekly_report.md").read_text()


def _executed(cm) -> tuple[pd.Series, float]:
    """The executed book (risky weights) and its cash residual from the checkpoint."""
    frame = cm.load("executed_weights")
    rows = frame[(frame["basis"] == "executed") & frame["asset"].notna()]
    weights = pd.Series(rows["weight"].to_numpy(dtype=float), index=[str(a) for a in rows["asset"]], dtype=float)
    return weights, 1.0 - float(weights.sum())


def _same(a: pd.Series, b: pd.Series) -> None:
    pd.testing.assert_series_equal(
        pd.Series(a, dtype=float).sort_index(), pd.Series(b, dtype=float).sort_index(),
        rtol=1e-9, atol=0, check_names=False,
    )


# ── the config key (D-01, ruling A2) ─────────────────────────────────────────


class TestAllocationModeFromConfig:
    def test_absent_key_is_regime_tilt(self):
        assert weekly.allocation_mode_from_config({}) == "regime_tilt"
        assert weekly.allocation_mode_from_config({"report": {}}) == "regime_tilt"
        assert weekly.allocation_mode_from_config({"report": {"allocation_mode": None}}) == "regime_tilt"

    @pytest.mark.parametrize("mode", ["no_regime", "regime_tilt"])
    def test_known_modes_pass(self, mode):
        assert weekly.allocation_mode_from_config({"report": {"allocation_mode": mode}}) == mode

    @pytest.mark.parametrize("bad", ["tilt", "NO_REGIME", 1, True])
    def test_unknown_values_raise_naming_the_value(self, bad):
        with pytest.raises(ValueError, match=re.escape(repr(bad))):
            weekly.allocation_mode_from_config({"report": {"allocation_mode": bad}})

    def test_the_live_config_runs_no_regime(self):
        """Pinned: deleting the key would silently put Glenn back on the tilt (G-06)."""
        assert load_platform_config()["report"]["allocation_mode"] == "no_regime"


# ── the tracer: serving.main then weekly.main in no_regime mode ──────────────


class TestNoRegimePageEndToEnd:
    def test_no_regime_page_is_suspended_and_trades_the_no_regime_target(self, tmp_path, monkeypatch):
        from trading_crab_lib.platform.checkpoints import get_platform_checkpoint_manager
        from trading_crab_lib.platform.report import serving

        world = _serving_world(tmp_path, monkeypatch)
        world["cfg"]["report"]["allocation_mode"] = "no_regime"
        world["cfg"]["report"]["accounts"] = ["no_regime_no_holdings_file"]
        assert serving.main([]) == 0
        assert weekly.main([]) == 0

        page = _report(world)
        assert "## Regime View (suspended)" in page
        assert _D03_SENTENCE in page
        assert "Scored as of 2021-06-30" in page
        assert "**Allocation mode:** no_regime" in page
        for line in _REGIME_LINES:
            assert line not in page, line
        assert "## Current Regime Distribution" not in page
        # The A-08 evidence stays: the distinct-posterior count, inside the suspended block.
        assert "distinct posterior vector" in page
        assert page.index("## Regime View (suspended)") < page.index("distinct posterior vector")
        assert page.index("distinct posterior vector") < page.index("## Target vs. Current — Trades Implied")
        # Per-account lines name the core mix, not a regime.
        assert "(no-regime core mix)" in page
        assert "neutral posture)" not in page
        # The mode line sits directly under the trades heading, before the table.
        trades = page.index("## Target vs. Current — Trades Implied")
        assert trades < page.index("**Allocation mode:** no_regime") < page.index("### Target allocation")

        cm = get_platform_checkpoint_manager()
        target = weekly.no_regime_target(cm.load("asset_returns"), pd.Timestamp("2021-06-30"), world["cfg"])
        weights, cash = _executed(cm)
        _same(weights, target["weights"])  # first run: held=None, the target in full
        assert cash == pytest.approx(target["cash"], rel=1e-9, abs=0)

    def test_without_the_key_the_page_keeps_the_regime_tilt_layout(self, tmp_path, monkeypatch):
        from trading_crab_lib.platform.report import serving

        world = _serving_world(tmp_path, monkeypatch)
        assert "allocation_mode" not in world["cfg"]["report"]
        assert serving.main([]) == 0
        assert weekly.main([]) == 0

        page = _report(world)
        assert "## Current Regime Distribution" in page
        assert "## Active Regime" in page
        assert "suspended" not in page
        assert "**Allocation mode:** regime_tilt" in page


class TestNoRegimeTarget:
    def test_is_the_one_state_tilt_over_returns_up_to_as_of(self):
        """The driver's tilt-off arithmetic by its own functions, on the window ending at as_of;
        rows after as_of never enter (served asset_returns run past the scored month)."""
        from trading_crab_lib.platform.allocation.tilt import vol_targeted_tilt
        from trading_crab_lib.platform.assets.returns import returns_by_regime_stats

        idx = pd.date_range("2000-01-31", periods=60, freq="ME")
        rng = pd.Series(range(60), index=idx, dtype=float)
        returns = pd.DataFrame({"SPY": 0.01 + 0.02 * ((rng % 5) - 2) / 2, "TLT": 0.004 - 0.01 * ((rng % 3) - 1)})
        cfg = {"allocation": {"target_vol_annual": 0.10, "ewma_halflife_months": 6, "portfolio_vol_min_obs": 12}}
        as_of = idx[47]

        got = weekly.no_regime_target(returns, as_of, cfg)
        window = returns.loc[:as_of]
        want = vol_targeted_tilt(
            pd.Series({0: 1.0}), returns_by_regime_stats(window, pd.Series(0, index=window.index)), window,
            target_vol_annual=0.10, halflife=6, min_obs=12,
        )
        _same(got["weights"], want["weights"])
        assert got["cash"] == want["cash"]

        # A shock after as_of changes nothing.
        shocked = returns.copy()
        shocked.loc[shocked.index > as_of, "SPY"] = 0.5
        _same(weekly.no_regime_target(shocked, as_of, cfg)["weights"], got["weights"])
