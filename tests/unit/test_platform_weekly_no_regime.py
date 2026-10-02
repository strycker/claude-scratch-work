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

import hashlib
import re
from pathlib import Path

import pandas as pd
import pytest
from test_platform_report_serving import _serving_world  # tests/unit is on sys.path (prepend mode)

from trading_crab_lib.platform.config import load_platform_config
from trading_crab_lib.platform.report import weekly

_REPO_ROOT = Path(__file__).resolve().parents[2]
_REAL_REGISTRY = _REPO_ROOT / "registry" / "trials.jsonl"
_TRACKED_PLATFORM = _REPO_ROOT / "data" / "checkpoints" / "platform"
_TRACKED_OUTPUTS = _REPO_ROOT / "outputs" / "reports" / "platform"
_DEV_END = pd.Timestamp("2020-12-31")

_D03_SENTENCE = (
    "Regime view: suspended — the served nowcaster is input-independent (fixed in the regime "
    "rebuild); the allocation does not use it."
)
_REGIME_LINES = ("- regime ", "Filtered Regime Belief", "Active Regime", "Trajectory", "Per-Asset Signals")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
        assert "distribution above" not in page  # nothing above it is a distribution any more
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


# ── A-15: a mode switch executes the new target in full and says so ──────────

_NOTE = "allocation mode changed to no_regime (was regime_tilt)"


def _run_at(world: dict, monkeypatch, month: str, mode: str | None) -> str:
    """One weekly.main run scoring ``month`` (the full span cut there) in ``mode`` (None = no key)."""
    cut = world["full"].loc[:pd.Timestamp(month)]
    monkeypatch.setattr(weekly, "load_full_span", lambda name: cut)
    if mode is None:
        world["cfg"]["report"].pop("allocation_mode", None)
    else:
        world["cfg"]["report"]["allocation_mode"] = mode
    assert weekly.main([]) == 0
    return _report(world)


class TestModeSwitch:
    def test_switch_rerun_and_next_month(self, tmp_path, monkeypatch):
        from test_platform_report_serving import _allocation_rows

        from trading_crab_lib.platform.allocation.hysteresis import execute_rebalance
        from trading_crab_lib.platform.checkpoints import get_platform_checkpoint_manager
        from trading_crab_lib.platform.report import serving

        world = _serving_world(tmp_path, monkeypatch)
        assert serving.main([]) == 0
        cm = get_platform_checkpoint_manager()
        returns = cm.load("asset_returns")
        may, june = pd.Timestamp("2021-05-31"), pd.Timestamp("2021-06-30")

        # Run 1: regime_tilt (no key) scores May.
        page1 = _run_at(world, monkeypatch, "2021-05-31", None)
        assert "allocation mode changed" not in page1
        tilt_book, _ = _executed(cm)

        # Run 2: no_regime, same month. The full target, not a band against the tilt book.
        page2 = _run_at(world, monkeypatch, "2021-05-31", "no_regime")
        target = weekly.no_regime_target(returns, may, world["cfg"])
        book2, cash2 = _executed(cm)
        _same(book2, target["weights"])
        assert cash2 == pytest.approx(target["cash"], rel=1e-9, abs=0)
        banded = execute_rebalance(target["weights"], target["cash"], tilt_book, band=0.05)["weights"]
        assert not book2.sort_index().equals(banded.sort_index()), "precondition: the band would bite here"
        assert page2.count(_NOTE) == 1
        assert "DECISIONS A-15" in page2
        rows = _allocation_rows(page2)
        for ticker, weight in tilt_book.items():
            assert rows[ticker][3] == f"{weight:.1%}", rows[ticker]  # last week = run 1's book
        heading = page2.index("## Target vs. Current — Trades Implied")
        assert heading < page2.index(_NOTE) < page2.index("### Target allocation")
        frame2 = cm.load("executed_weights")

        # Run 3: a same-month re-run reproduces the book and keeps the note.
        page3 = _run_at(world, monkeypatch, "2021-05-31", "no_regime")
        pd.testing.assert_frame_equal(cm.load("executed_weights"), frame2)
        assert page3.count(_NOTE) == 1

        # Run 4: the next month bands normally against run 3's book, with no note.
        page4 = _run_at(world, monkeypatch, "2021-06-30", "no_regime")
        assert "allocation mode changed" not in page4
        nxt = weekly.no_regime_target(returns, june, world["cfg"])
        want = execute_rebalance(nxt["weights"], nxt["cash"], book2, band=0.05)
        book4, cash4 = _executed(cm)
        _same(book4, want["weights"])
        assert cash4 == pytest.approx(want["cash"], rel=1e-9, abs=0)
        record = cm.load("allocation_mode")
        assert list(record.columns) == ["mode", "as_of", "changed_from"]
        assert record["mode"].tolist() == ["no_regime"]
        assert pd.Timestamp(record["as_of"].iloc[0]) == june
        assert record["changed_from"].isna().all()

    def test_an_executed_book_without_a_mode_record_counts_as_regime_tilt(self, tmp_path, monkeypatch):
        from trading_crab_lib.platform.checkpoints import get_platform_checkpoint_manager
        from trading_crab_lib.platform.report import serving

        world = _serving_world(tmp_path, monkeypatch)
        assert serving.main([]) == 0
        cm = get_platform_checkpoint_manager()
        _run_at(world, monkeypatch, "2021-05-31", None)
        cm.clear("allocation_mode")  # a pre-8.2 checkpoint dir: executed_weights only
        assert not (world["platform_dir"] / "allocation_mode.parquet").exists()

        page = _run_at(world, monkeypatch, "2021-05-31", "no_regime")
        assert page.count(_NOTE) == 1
        book, _ = _executed(cm)
        _same(book, weekly.no_regime_target(cm.load("asset_returns"), pd.Timestamp("2021-05-31"), world["cfg"])["weights"])

    def test_with_neither_checkpoint_there_is_no_note(self, tmp_path, monkeypatch):
        from trading_crab_lib.platform.report import serving

        world = _serving_world(tmp_path, monkeypatch)
        assert serving.main([]) == 0
        page = _run_at(world, monkeypatch, "2021-05-31", "no_regime")
        assert "allocation mode changed" not in page
        assert (world["platform_dir"] / "allocation_mode.parquet").exists()

    def test_a_band_free_config_writes_no_mode_record(self, tmp_path, monkeypatch):
        from trading_crab_lib.platform.report import serving

        world = _serving_world(tmp_path, monkeypatch)
        del world["cfg"]["allocation"]["no_trade_band"]
        assert serving.main([]) == 0
        _run_at(world, monkeypatch, "2021-05-31", "no_regime")
        assert not (world["platform_dir"] / "allocation_mode.parquet").exists()
        assert not (world["platform_dir"] / "executed_weights.parquet").exists()


# ── parity with the measured ablation leg (D-01) ─────────────────────────────


def _spy(monkeypatch, module, name: str) -> list:
    """Wrap ``module.name`` so every call's return value is recorded, in order."""
    real = getattr(module, name)
    calls: list = []

    def wrapper(*args, **kwargs):
        out = real(*args, **kwargs)
        calls.append(out)
        return out

    monkeypatch.setattr(module, name, wrapper)
    return calls


class TestParityWithTheAblationLeg:
    def test_synthetic_parity_weekly_steps_equal_the_ablation_driver(self, tmp_path, monkeypatch):
        """Weekly, driven once per walk-forward step in order, holds what the tilt-off driver
        holds: pre-band target == the driver's tilt, executed book == the driver's rebalance.
        Fails if weekly tilts on the belief, conditions on the regime table, or reads served
        returns past the scored month (they run to 2021-06 here)."""
        from trading_crab_lib.platform.backtest import driver
        from trading_crab_lib.platform.honesty.registry import NO_REGISTRY
        from trading_crab_lib.platform.honesty.walkforward import expanding_steps
        from trading_crab_lib.platform.report import serving

        world = _serving_world(tmp_path, monkeypatch)
        cfg = world["cfg"]
        assert serving.main([]) == 0
        from trading_crab_lib.platform.checkpoints import get_platform_checkpoint_manager

        asset_returns = get_platform_checkpoint_manager().load("asset_returns")
        assert asset_returns.index.max() > _DEV_END, "precondition: served returns run past the dev window"
        dev = world["dev"]
        min_train = len(dev) - 8
        cfg["backtest"].update({"min_train_months": min_train, "skip_l1l2_for_ablation": True, "cost_bps": 10})
        cfg["report"]["allocation_mode"] = "no_regime"

        tilts = _spy(monkeypatch, driver, "vol_targeted_tilt")
        rebalances = _spy(monkeypatch, driver, "execute_rebalance")
        curve, _ = driver.run_backtest(dev, asset_returns, cfg, use_regime_tilt=False, registry_path=NO_REGISTRY)
        steps = list(expanding_steps(dev.index, min_train=min_train))
        assert len(curve) == len(steps) == len(tilts) == len(rebalances) == 8
        assert not curve["degraded"].any()

        for (_t, train_index, _test), tilt, rebalance in zip(steps, tilts, rebalances):
            as_of = pd.Timestamp(train_index[-1])
            cut = world["full"].loc[:as_of]
            monkeypatch.setattr(weekly, "load_full_span", lambda name, cut=cut: cut)
            inputs = weekly._build_report_inputs(cfg)
            assert inputs["allocation_mode"] == "no_regime"
            assert inputs["mode_note"] is None
            _same(inputs["pre_band_target_weights"], tilt["weights"])
            assert inputs["pre_band_cash"] == pytest.approx(tilt["cash"], rel=1e-9, abs=0)
            _same(inputs["target_weights"], rebalance["weights"])
            assert inputs["cash"] == pytest.approx(rebalance["cash"], rel=1e-9, abs=0)

    def test_tracked_data_parity_ablation_rerun_matches_the_tracked_curve_and_weekly(self, tmp_path, monkeypatch):
        """On the tracked data: a NO_REGISTRY ablation re-run reproduces the tracked ablation curve
        (``return``, ``scale``) and its final-step tilt is the weekly no-regime target at 2020-11-30.
        Also pins that D-05's tracked ablation numbers still match the code and the data; a later
        rebuild that changes rows <= 2020-12 turns this red, and that is informative."""
        import shutil

        import trading_crab_lib.platform.assets.returns as returns_mod
        from trading_crab_lib.checkpoints import CheckpointManager
        from trading_crab_lib.platform.assets.returns import compute_monthly_returns, tradable_asset_returns
        from trading_crab_lib.platform.backtest import driver
        from trading_crab_lib.platform.backtest.baselines import no_regime_ablation
        from trading_crab_lib.platform.honesty import registry
        from trading_crab_lib.platform.report import serving
        from trading_crab_lib.platform.splice import build_core_research_series

        sha_before = _sha256(_REAL_REGISTRY)
        count_before = registry.total_trial_count(_REAL_REGISTRY)
        assert count_before == 46

        tmp_ckpt = tmp_path / "platform"
        tmp_ckpt.mkdir()
        for name in ("monthly_features", "monthly_raw", "regime_labels"):
            for suffix in (".parquet", ".meta.json"):
                shutil.copy2(_TRACKED_PLATFORM / f"{name}{suffix}", tmp_ckpt / f"{name}{suffix}")
        monkeypatch.setattr(returns_mod, "OUTPUT_DIR", tmp_path / "out")
        cm = CheckpointManager(checkpoint_dir=tmp_ckpt)
        cfg = load_platform_config()
        assert cfg["backtest"].get("skip_l1l2_for_ablation", True) is True, "precondition: the measured leg skipped L1/L2"
        serving.build_serving_artifacts(cfg, cm=cm, output_dir=tmp_path / "out")

        # The ablation inputs exactly as run_full_backtest_evaluation builds them.
        returns = compute_monthly_returns(build_core_research_series(cm.load("monthly_raw"), cfg))
        asset_returns = tradable_asset_returns(returns, cfg["splice"])
        cash_ret = returns[cfg["splice"]["cash"]["research_name"]]
        tilts = _spy(monkeypatch, driver, "vol_targeted_tilt")
        curve, _ = no_regime_ablation(
            cm.load("monthly_features"), asset_returns, cfg, cash_returns=cash_ret, registry_path=registry.NO_REGISTRY,
        )

        tracked = pd.read_parquet(_TRACKED_OUTPUTS / "backtest_equity_curve_ablation.parquet")
        assert list(curve.index) == list(tracked.index)
        # Datetime resolution differs by pandas major (2.x reads ns, 3.x keeps us); the dates
        # themselves are pinned above, so compare at one resolution.
        curve = curve.set_axis(curve.index.as_unit("ns"))
        tracked = tracked.set_axis(tracked.index.as_unit("ns"))
        for column in ("return", "scale"):
            pd.testing.assert_series_equal(curve[column], tracked[column], rtol=1e-9, atol=0, check_freq=False)

        weekly_target = weekly.no_regime_target(cm.load("asset_returns"), pd.Timestamp("2020-11-30"), cfg)
        _same(weekly_target["weights"], tilts[-1]["weights"])
        assert weekly_target["cash"] == pytest.approx(tilts[-1]["cash"], rel=1e-9, abs=0)

        assert _sha256(_REAL_REGISTRY) == sha_before
        assert registry.total_trial_count(_REAL_REGISTRY) == count_before == 46


class TestSuspendedView:
    def test_the_constant_posterior_sentence_does_not_point_at_a_missing_distribution(self):
        note = "1 distinct posterior vector across 3 complete months (…). " + weekly._CONSTANT_POSTERIOR_SENTENCE
        md = weekly.assemble_weekly_report(
            regime_probs={0: 1.0}, transition_matrix=pd.DataFrame(), returns_by_regime=pd.DataFrame(),
            target_weights=pd.Series({"SPY": 0.6}), accounts=[], active_regime=None,
            input_sensitivity_note=note, regime_view_suspended=True, allocation_mode="no_regime",
        )
        assert "The served posterior does not depend on the features: it is the same every week." in md
        assert "distribution above" not in md
        assert "1 distinct posterior vector across 3 complete months" in md
