"""
Tests for the weekly page's crash tripwire section and ``build_weekly_page`` (plan 08.2-03, D-04).

The tripwire is advisory: it prints each of the three signals red or green with its value,
threshold and as-of date, and an escalation over the CURRENT signals only. A reading older than
``tripwire.stale_business_days`` (5, ruling A1) is STALE and a missing input UNAVAILABLE; neither
is ever green, and nothing is imputed (T-08.2-08). The executed book is the same whatever the
tripwire says (T-08.2-09).

``evaluate_tripwire`` runs on synthetic daily series behind a dict-backed checkpoint stand-in.
``build_weekly_page`` and ``serving.build_scratch_serving`` run on the synthetic serving world of
``test_platform_report_serving.py`` (reused, never edited) in per-test tmp dirs.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from test_platform_report_serving import _serving_world  # tests/unit is on sys.path (prepend mode)

from trading_crab_lib.platform.tripwire import monitor
from trading_crab_lib.platform.tripwire.monitor import TripwireEscalation

_RUN = pd.Timestamp("2026-09-30")  # a Wednesday
_TRADES = "## Target vs. Current — Trades Implied"
_TRIPWIRE = "## Crash Tripwire (advisory — changes no weight)"
_N_DAYS = 300


class _Cm:
    """Checkpoint stand-in: ``load(name)`` returns a frame or raises FileNotFoundError."""

    def __init__(self, **frames: pd.DataFrame) -> None:
        self.frames = frames
        self.loaded: list[str] = []

    def load(self, name: str) -> pd.DataFrame:
        self.loaded.append(name)
        if name not in self.frames:
            raise FileNotFoundError(f"no checkpoint {name}")
        return self.frames[name].copy()


def _days(end: pd.Timestamp = _RUN, n: int = _N_DAYS) -> pd.DatetimeIndex:
    return pd.bdate_range(end=end, periods=n)


def _calm_spy(idx: pd.DatetimeIndex, seed: int = 7) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(100.0 * np.cumprod(1 + rng.normal(0.0004, 0.004, len(idx))), index=idx)


def _drawdown_spy(idx: pd.DatetimeIndex, seed: int = 5) -> pd.Series:
    """Up 100 -> 110, down to 0.88 x 110 over 40 days, then flat with tiny noise: a 12%
    drawdown long settled, so the recent and baseline vol windows are both calm."""
    rng = np.random.default_rng(seed)
    n = len(idx)
    up = np.linspace(100.0, 110.0, 100)
    down = np.linspace(110.0, 96.8, 41)[1:]
    flat = 96.8 * np.cumprod(1 + rng.normal(0.0, 0.0005, n - 140))
    return pd.Series(np.concatenate([up, down, flat]), index=idx)


def _credit(idx: pd.DatetimeIndex, *, widen_last_bps: float = 0.0) -> pd.DataFrame:
    dbaa = np.full(len(idx), 5.5)
    dbaa[-1] += widen_last_bps / 100.0
    return pd.DataFrame({"fred_daaa": np.full(len(idx), 4.5), "fred_dbaa": dbaa}, index=idx)


def _cm(spy: pd.Series | None, credit: pd.DataFrame | None) -> _Cm:
    frames = {}
    if spy is not None:
        frames["daily_raw"] = pd.DataFrame({"SPY": spy, "QQQ": spy * 2})
    if credit is not None:
        frames["fred_daily_raw"] = credit
    return _Cm(**frames)


def _cfg(**tripwire) -> dict:
    return {"tripwire": dict(tripwire)}


def _bools(spy: pd.Series, credit: pd.DataFrame | None) -> dict[str, bool]:
    """The existing bool functions on the cleaned inputs (the oracle for ``triggered``)."""
    spy = spy.dropna()
    out = {
        "vol_spike": monitor.realized_vol_spike(
            spy.pct_change().dropna(), short_window=21, baseline_window=63, ratio_threshold=1.5, halflife=11.2
        ),
        "spy_drawdown": monitor.spy_drawdown_from_peak(spy, drawdown_threshold=0.10),
    }
    if credit is not None:
        aligned = credit[["fred_daaa", "fred_dbaa"]].dropna()
        out["credit_velocity"] = monitor.credit_spread_velocity(
            aligned["fred_daaa"], aligned["fred_dbaa"], lookback_days=5, bps_threshold=25
        )
    return out


# ── evaluate_tripwire ────────────────────────────────────────────────────────


class TestEvaluateTripwire:
    def test_calm_fresh_data_is_three_green(self):
        idx = _days()
        spy, credit = _calm_spy(idx), _credit(idx)
        result = monitor.evaluate_tripwire(_cfg(), _cm(spy, credit), run_date=_RUN)

        assert [s["state"] for s in result["signals"].values()] == ["green", "green", "green"]
        assert list(result["signals"]) == ["vol_spike", "credit_velocity", "spy_drawdown"]
        assert result["escalation"] is TripwireEscalation.NONE
        assert result["n_current"] == 3
        for name, sig in result["signals"].items():
            assert sig["as_of"] == idx[-1], name
            assert np.isfinite(sig["value"]), name
        assert result["signals"]["credit_velocity"]["value"] == pytest.approx(0.0, abs=1e-9)

    def test_spy_twelve_percent_off_peak_is_red_with_value_and_date(self):
        idx = _days()
        spy = _drawdown_spy(idx)
        result = monitor.evaluate_tripwire(_cfg(), _cm(spy, _credit(idx)), run_date=_RUN)

        dd = result["signals"]["spy_drawdown"]
        assert dd["state"] == "red"
        assert dd["triggered"] is True
        want = float(spy.iloc[-1] / spy.cummax().iloc[-1] - 1.0)
        assert dd["value"] == pytest.approx(want, rel=1e-9, abs=0)
        assert dd["value"] == pytest.approx(-0.12, abs=0.01)
        assert dd["threshold"] == pytest.approx(-0.10, rel=1e-9)
        assert dd["as_of"] == idx[-1]
        assert result["signals"]["vol_spike"]["state"] == "green"
        assert result["escalation"] is TripwireEscalation.RUN_WEEKLY_SCORING_EARLY

    def test_nan_tail_on_dbaa_dates_credit_at_the_last_aligned_day_never_green_from_nan(self):
        """The naive read is ``NaN >= 25`` -> False -> green. Aligned, the last day both
        series exist carries a real 40 bps widening, so the signal is red, dated there."""
        idx = _days()
        credit = _credit(idx)
        credit.iloc[-4, credit.columns.get_loc("fred_dbaa")] += 0.40
        credit.iloc[-3:, credit.columns.get_loc("fred_dbaa")] = np.nan
        # The pre-8.2 naive arithmetic reads the NaN tail as "not triggered":
        assert monitor.credit_spread_velocity(credit["fred_daaa"], credit["fred_dbaa"], lookback_days=5,
                                              bps_threshold=25) is False

        result = monitor.evaluate_tripwire(_cfg(), _cm(_calm_spy(idx), credit), run_date=_RUN)
        sig = result["signals"]["credit_velocity"]
        assert sig["as_of"] == idx[-4]
        assert sig["value"] == pytest.approx(40.0, rel=1e-9)
        assert sig["state"] == "red"
        assert sig["triggered"] is True

    def test_data_nine_business_days_old_is_stale_with_its_value(self):
        end = _RUN - pd.offsets.BDay(9)
        idx = _days(end=end)
        result = monitor.evaluate_tripwire(_cfg(), _cm(_calm_spy(idx), _credit(idx)), run_date=_RUN)

        for name, sig in result["signals"].items():
            assert sig["state"] == "stale", name
            assert sig["as_of"] == end, name
            assert np.isfinite(sig["value"]), name
        assert result["n_current"] == 0

    def test_five_business_days_is_still_current_six_is_stale(self):
        for lag, state in ((5, "green"), (6, "stale")):
            idx = _days(end=_RUN - pd.offsets.BDay(lag))
            result = monitor.evaluate_tripwire(_cfg(), _cm(_calm_spy(idx), _credit(idx)), run_date=_RUN)
            assert result["signals"]["spy_drawdown"]["state"] == state, lag

    def test_stale_threshold_comes_from_config(self):
        idx = _days(end=_RUN - pd.offsets.BDay(9))
        result = monitor.evaluate_tripwire(
            {"tripwire": {"stale_business_days": 10}}, _cm(_calm_spy(idx), _credit(idx)), run_date=_RUN
        )
        assert result["signals"]["spy_drawdown"]["state"] == "green"
        assert result["stale_business_days"] == 10

    def test_missing_fred_daily_raw_is_credit_unavailable(self):
        idx = _days()
        result = monitor.evaluate_tripwire(_cfg(), _cm(_calm_spy(idx), None), run_date=_RUN)
        sig = result["signals"]["credit_velocity"]
        assert sig["state"] == "unavailable"
        assert sig["triggered"] is None
        assert sig["as_of"] is None
        assert "fred_daily_raw" in sig["reason"]
        assert result["signals"]["spy_drawdown"]["state"] == "green"
        assert result["n_current"] == 2

    def test_missing_daily_raw_is_vol_and_drawdown_unavailable(self):
        idx = _days()
        result = monitor.evaluate_tripwire(_cfg(), _cm(None, _credit(idx)), run_date=_RUN)
        for name in ("vol_spike", "spy_drawdown"):
            assert result["signals"][name]["state"] == "unavailable", name
            assert "daily_raw" in result["signals"][name]["reason"], name
        assert result["signals"]["credit_velocity"]["state"] == "green"

    def test_missing_column_is_unavailable_not_a_raise(self):
        idx = _days()
        cm = _Cm(daily_raw=pd.DataFrame({"QQQ": _calm_spy(idx)}), fred_daily_raw=_credit(idx)[["fred_daaa"]])
        result = monitor.evaluate_tripwire(_cfg(), cm, run_date=_RUN)
        assert {s["state"] for s in result["signals"].values()} == {"unavailable"}

    def test_triggered_equals_the_existing_bool_functions_on_cleaned_inputs(self):
        idx = _days()
        cases = [
            (_calm_spy(idx), _credit(idx)),
            (_drawdown_spy(idx), _credit(idx, widen_last_bps=30.0)),
        ]
        spiky = _calm_spy(idx).copy()
        rng = np.random.default_rng(1)
        spiky.iloc[-21:] = spiky.iloc[-22] * np.cumprod(1 + rng.normal(0, 0.03, 21))
        spiky.iloc[:40] = np.nan  # a leading NaN block (pre-inception), dropped first
        cases.append((spiky, _credit(idx)))
        seen_true = set()
        for spy, credit in cases:
            result = monitor.evaluate_tripwire(_cfg(), _cm(spy, credit), run_date=_RUN)
            oracle = _bools(spy, credit)
            for name, want in oracle.items():
                assert result["signals"][name]["triggered"] is want, name
                if want:
                    seen_true.add(name)
        assert seen_true == {"vol_spike", "credit_velocity", "spy_drawdown"}  # every signal can fire

    def test_escalation_counts_current_signals_only(self):
        """A stale credit widening that would trigger does not raise the tier."""
        idx = _days()
        old = _days(end=_RUN - pd.offsets.BDay(9))
        result = monitor.evaluate_tripwire(
            _cfg(), _cm(_drawdown_spy(idx), _credit(old, widen_last_bps=30.0)), run_date=_RUN
        )
        assert result["signals"]["credit_velocity"]["state"] == "stale"
        assert result["signals"]["credit_velocity"]["triggered"] is True
        assert result["signals"]["spy_drawdown"]["state"] == "red"
        assert result["n_current"] == 2
        assert result["escalation"] is TripwireEscalation.RUN_WEEKLY_SCORING_EARLY

    def test_run_tripwire_is_unchanged(self):
        """The Phase 4 orchestrator still raises on a missing checkpoint (T-04-12)."""
        idx = _days()
        with pytest.raises(FileNotFoundError):
            monitor.run_tripwire(_cfg(), _cm(_calm_spy(idx), None))


class TestValueHelpers:
    def test_undefined_values_are_nan(self):
        assert np.isnan(monitor.vol_spike_ratio(pd.Series([0.01, 0.02]), short_window=21, baseline_window=63,
                                                halflife=11.2))
        assert np.isnan(monitor.credit_widening_bps(pd.Series([10.0, 11.0]), lookback_days=5))
        assert np.isnan(monitor.drawdown_from_peak(pd.Series(dtype=float)))

    def test_values_agree_with_the_bool_functions(self):
        idx = _days()
        spy = _drawdown_spy(idx)
        ret = spy.pct_change().dropna()
        ratio = monitor.vol_spike_ratio(ret, short_window=21, baseline_window=63, halflife=11.2)
        assert (ratio > 1.5) is monitor.realized_vol_spike(
            ret, short_window=21, baseline_window=63, ratio_threshold=1.5, halflife=11.2
        )
        assert monitor.drawdown_from_peak(spy) == pytest.approx(float(spy.iloc[-1] / spy.max() - 1), rel=1e-12)
        spread = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 140.0])
        assert monitor.credit_widening_bps(spread, lookback_days=5) == pytest.approx(39.0)


# ── the page section ─────────────────────────────────────────────────────────


def _assemble(tripwire=None, **kw) -> str:
    from trading_crab_lib.platform.report import weekly

    return weekly.assemble_weekly_report(
        regime_probs={0: 1.0},
        transition_matrix=pd.DataFrame({0: [1.0]}, index=[0]),
        returns_by_regime=pd.DataFrame(columns=["regime", "asset", "mean_monthly_return", "sharpe_annualized",
                                                "n_obs"]),
        target_weights=pd.Series({"SPY": 0.6}),
        cash=0.4,
        accounts=[],
        active_regime=0,
        tripwire=tripwire,
        **kw,
    )


def _section(page: str, heading: str) -> str:
    start = page.index(heading)
    nxt = page.find("\n## ", start + 1)
    return page[start:] if nxt < 0 else page[start:nxt]


class TestTripwireSection:
    def test_section_sits_before_the_trades_heading_with_one_bullet_per_signal(self):
        idx = _days()
        result = monitor.evaluate_tripwire(_cfg(), _cm(_drawdown_spy(idx), _credit(idx)), run_date=_RUN)
        page = _assemble(result)
        assert page.index(_TRIPWIRE) < page.index(_TRADES)
        section = _section(page, _TRIPWIRE)
        bullets = [line for line in section.splitlines() if line.startswith("- ")]
        assert len(bullets) == 3
        assert any("Drawdown from peak (SPY): RED" in b and "as of 2026-09-30" in b for b in bullets)
        assert any("(trips at -10.00%)" in b for b in bullets)
        assert "Escalation: run weekly scoring early (3 of 3 signals current; nothing is imputed)" in section

    def test_all_green_and_current_is_the_only_way_to_read_none(self):
        idx = _days()
        result = monitor.evaluate_tripwire(_cfg(), _cm(_calm_spy(idx), _credit(idx)), run_date=_RUN)
        section = _section(_assemble(result), _TRIPWIRE)
        assert "Escalation: none (3 of 3 signals current; nothing is imputed)" in section
        assert section.count(": GREEN") == 3

    def test_all_unavailable_reads_unknown_never_green(self):
        result = monitor.evaluate_tripwire(_cfg(), _Cm(), run_date=_RUN)
        section = _section(_assemble(result), _TRIPWIRE)
        assert "Escalation: UNKNOWN (0 of 3 signals current — not green)" in section
        assert "Escalation: none" not in section
        assert "GREEN" not in section
        assert section.count(": UNAVAILABLE") == 3
        assert "python scripts/build_platform_data.py" in section

    def test_all_stale_reads_unknown_and_a_stale_trigger_would_be_red(self):
        old = _days(end=_RUN - pd.offsets.BDay(9))
        result = monitor.evaluate_tripwire(_cfg(), _cm(_drawdown_spy(old), _credit(old)), run_date=_RUN)
        section = _section(_assemble(result), _TRIPWIRE)
        assert "Escalation: UNKNOWN (0 of 3 signals current — not green)" in section
        assert "Escalation: none" not in section
        assert "GREEN" not in section
        assert "Drawdown from peak (SPY): STALE (would be RED)" in section
        assert section.count(": STALE") == 3
        assert "as of 2026-09-17" in section
        assert "python scripts/build_platform_data.py" in section

    def test_without_tripwire_the_page_is_unchanged(self):
        assert _assemble(None) == _assemble()
        assert "Crash Tripwire" not in _assemble(None)


# ── build_weekly_page on the serving world ───────────────────────────────────


def _signal(state: str, triggered: bool | None) -> dict:
    return {"label": "x", "state": state, "triggered": triggered, "value": 1.0, "threshold": 1.0,
            "as_of": pd.Timestamp("2021-07-07"), "source": "daily_raw", "reason": None}


def _stub(state: str) -> dict:
    red = state == "red"
    names = ("vol_spike", "credit_velocity", "spy_drawdown")
    return {
        "run_date": pd.Timestamp("2021-07-08"),
        "stale_business_days": 5,
        "signals": {n: {**_signal(state, red), "label": n} for n in names},
        "escalation": TripwireEscalation.TIER1_DERISK_REVIEW if red else TripwireEscalation.NONE,
        "n_current": 3,
    }


def _no_regime_world(tmp_path: Path, monkeypatch) -> dict:
    from trading_crab_lib.platform.report import serving

    world = _serving_world(tmp_path, monkeypatch)
    world["cfg"]["report"]["allocation_mode"] = "no_regime"
    serving.build_serving_artifacts(world["cfg"])
    return world


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TestBuildWeeklyPage:
    def test_writes_into_output_dir_with_the_tripwire_before_the_trades(self, tmp_path, monkeypatch):
        from trading_crab_lib.platform.checkpoints import get_platform_checkpoint_manager
        from trading_crab_lib.platform.report import weekly

        world = _no_regime_world(tmp_path / "w", monkeypatch)
        seen = {}
        real = weekly.evaluate_tripwire

        def spy(cfg, cm=None, *, run_date):
            seen["run_date"] = run_date
            return real(cfg, cm, run_date=run_date)

        monkeypatch.setattr(weekly, "evaluate_tripwire", spy)
        out = tmp_path / "page"
        markdown, path = weekly.build_weekly_page(world["cfg"], get_platform_checkpoint_manager(), output_dir=out)

        assert path == out / "weekly_report.md"
        assert path.read_text(encoding="utf-8") == markdown
        assert seen["run_date"] == pd.Timestamp("2021-07-08")  # weekly._run_date(), pinned by the world
        assert _TRIPWIRE in markdown
        assert markdown.index("## Regime View (suspended)") < markdown.index(_TRIPWIRE) < markdown.index(_TRADES)
        # The world has no daily data: every signal is UNAVAILABLE and the escalation UNKNOWN.
        assert "Escalation: UNKNOWN (0 of 3 signals current — not green)" in markdown
        assert not (world["out_dir"] / "reports" / "platform" / "weekly_report.md").exists()

    def test_the_executed_book_ignores_the_tripwire(self, tmp_path, monkeypatch):
        from trading_crab_lib.platform.checkpoints import get_platform_checkpoint_manager
        from trading_crab_lib.platform.report import weekly

        pages, books = {}, {}
        for state in ("red", "green"):
            world = _no_regime_world(tmp_path / state, monkeypatch)
            monkeypatch.setattr(weekly, "evaluate_tripwire", lambda cfg, cm=None, *, run_date, s=state: _stub(s))
            cm = get_platform_checkpoint_manager()
            pages[state], _ = weekly.build_weekly_page(world["cfg"], cm, output_dir=tmp_path / state / "page")
            books[state] = cm.load("executed_weights")

        pd.testing.assert_frame_equal(books["red"], books["green"])
        assert "RED" in _section(pages["red"], _TRIPWIRE)
        assert "RED" not in _section(pages["green"], _TRIPWIRE)
        strip = {s: pages[s].replace(_section(pages[s], _TRIPWIRE), "") for s in pages}
        assert strip["red"] == strip["green"]

    def test_main_writes_the_same_page_build_weekly_page_returns(self, tmp_path, monkeypatch):
        from trading_crab_lib.platform.report import weekly

        world = _no_regime_world(tmp_path / "w", monkeypatch)
        captured = {}
        real = weekly.build_weekly_page

        def spy(cfg, cm=None, *, output_dir=None):
            captured["page"] = real(cfg, cm, output_dir=output_dir)
            return captured["page"]

        monkeypatch.setattr(weekly, "build_weekly_page", spy)
        assert weekly.main([]) == 0
        markdown, path = captured["page"]
        assert path == world["out_dir"] / "reports" / "platform" / "weekly_report.md"
        assert path.read_text(encoding="utf-8") == markdown


class TestBuildScratchServing:
    def test_two_cold_copies_give_byte_identical_pages_and_never_write_the_source(self, tmp_path, monkeypatch):
        from trading_crab_lib.platform.report import serving, weekly

        world = _serving_world(tmp_path / "w", monkeypatch)
        world["cfg"]["report"]["allocation_mode"] = "no_regime"
        source = world["platform_dir"]
        before = {p.name: _sha(p) for p in source.iterdir()}

        pages = []
        for name in ("a", "b"):
            scratch = tmp_path / name
            cm = serving.build_scratch_serving(world["cfg"], scratch, source_dir=source)
            assert Path(cm.dir) == scratch / "checkpoints"
            for artifact in ("monthly_features", "monthly_raw", "regime_labels"):
                assert (scratch / "checkpoints" / f"{artifact}.parquet").exists(), artifact
            assert (scratch / "checkpoints" / "nowcaster.pkl").exists()
            markdown, path = weekly.build_weekly_page(world["cfg"], cm, output_dir=scratch / "out")
            assert path.parent == scratch / "out"
            pages.append(markdown)

        assert pages[0] == pages[1]
        assert {p.name: _sha(p) for p in source.iterdir()} == before

    def test_copies_daily_inputs_when_present(self, tmp_path, monkeypatch):
        from trading_crab_lib.platform.checkpoints import get_platform_checkpoint_manager
        from trading_crab_lib.platform.report import serving

        world = _serving_world(tmp_path / "w", monkeypatch)
        idx = _days(end=pd.Timestamp("2021-07-07"))
        src_cm = get_platform_checkpoint_manager()
        src_cm.save(pd.DataFrame({"SPY": _calm_spy(idx)}), "daily_raw")
        src_cm.save(_credit(idx), "fred_daily_raw")

        cm = serving.build_scratch_serving(world["cfg"], tmp_path / "s", source_dir=world["platform_dir"])
        pd.testing.assert_frame_equal(cm.load("fred_daily_raw"), src_cm.load("fred_daily_raw"))
        pd.testing.assert_frame_equal(cm.load("daily_raw"), src_cm.load("daily_raw"))


def test_live_config_carries_the_stale_threshold_and_scoreboard_tag():
    from trading_crab_lib.platform.config import load_platform_config

    cfg = load_platform_config()
    assert cfg["tripwire"]["stale_business_days"] == 5
    # 08.3 (2026-10-04): "08.1-pit-tilt-vs-ablation" -> "08.3-monthend-tilt-vs-ablation"
    assert cfg["report"]["scoreboard_trial_tag"] == "08.3-monthend-tilt-vs-ablation"


class TestServedPosteriorPath:
    def test_is_the_posterior_over_every_complete_month_the_page_counts(self, tmp_path, monkeypatch):
        import re

        from trading_crab_lib.platform.checkpoints import get_platform_checkpoint_manager
        from trading_crab_lib.platform.report import weekly

        world = _no_regime_world(tmp_path / "w", monkeypatch)
        cm = get_platform_checkpoint_manager()
        dates, proba, classes = weekly.served_posterior_path(cm)

        model = cm.load_model("nowcaster")
        cols = [str(c) for c in model.feature_names_in_]
        frame = world["full"][cols].dropna(how="any")
        assert dates.equals(pd.DatetimeIndex(frame.index))
        np.testing.assert_array_equal(proba, model.predict_proba(frame))
        assert classes == [int(c) for c in model.classes_]
        note = weekly._input_sensitivity_note(model, world["full"], cols)
        n_distinct = int(re.match(r"(\d+) distinct", note).group(1))
        assert np.unique(proba, axis=0).shape[0] == n_distinct
