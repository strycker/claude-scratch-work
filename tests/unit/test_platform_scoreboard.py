"""
Tests for the static scoreboard (plan 08.2-03, D-05, ruling A3) and the Sharpe-vs-hurdle figure.

The scoreboard reads the last budgeted run's tracked outputs and never recomputes a fitted leg:
the strategy and ablation curves come from their parquets, the three price baselines from the
deterministic ``backtest/baselines.py::baseline_curves``. The headline columns put all five legs
on the strategy curve's window (1972-2020 in the tracked run); the own-span terminal log wealth
is a footnote. Every number reconciles with ``backtest_kpi_table.parquet`` at rel 1e-9, or the
common columns are withheld with a reason (T-08.2-12). The run date comes from the registry rows
carrying the configured trial tag.

Synthetic tests build the reports dir, ``monthly_raw`` and a registry in tmp. The tracked test
only reads (``outputs/reports/platform``, ``data/checkpoints/platform``, ``registry/``) and
checks the registry is byte-identical afterwards.
"""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from test_platform_plotting_backtest import CFG  # tests/unit is on sys.path (prepend mode)

from trading_crab_lib.platform.evaluation.kpis import max_drawdown_and_duration, terminal_log_wealth
from trading_crab_lib.platform.plotting import backtest as pbacktest
from trading_crab_lib.platform.report import scoreboard

_REPO = Path(__file__).resolve().parents[2]
_TAG = "08.1-pit-tilt-vs-ablation"
_LEGS = ["strategy", "no_regime_ablation", "spy_buy_hold", "sixty_forty", "faber_sma"]


class _Cm:
    def __init__(self, **frames: pd.DataFrame) -> None:
        self.frames = frames

    def load(self, name: str) -> pd.DataFrame:
        if name not in self.frames:
            raise FileNotFoundError(f"no checkpoint {name}")
        return self.frames[name].copy()


def _monthly_raw(seed: int = 42) -> pd.DataFrame:
    n = 730
    idx = pd.date_range("1962-01-31", periods=n, freq="ME")
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "sp500": 100 * np.cumprod(1 + rng.normal(0.006, 0.03, n)),
            "div_yield": np.full(n, 0.03),
            "fred_gs10": 0.04 + 0.00002 * np.arange(n),
            "gold_spot": 35.0 + np.arange(n),
            "wti_crude": 3.0 + 0.01 * np.arange(n),
            "fred_tb3ms": np.full(n, 0.02),
        },
        index=idx,
    )


def _kpi(series: pd.Series) -> tuple[float, float]:
    clean = series.dropna()
    return terminal_log_wealth(clean), max_drawdown_and_duration(clean)["max_drawdown"]


def _write_registry(path: Path) -> None:
    rows = [
        {"config": {"trial_tag": "something-else"}, "timestamp": "2026-10-01T09:00:00+00:00", "git_sha": "ffff"},
        {"config": {"trial_tag": _TAG, "leg": "tilt"}, "timestamp": "2026-09-30T21:33:46+00:00", "git_sha": "74207efa"},
        {"config": {"trial_tag": _TAG, "leg": "abl"}, "timestamp": "2026-09-30T21:33:56+00:00", "git_sha": "74207efa"},
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _world(tmp_path: Path) -> dict:
    """reports dir (KPI table + the two ML curves), monthly_raw, registry and cfg."""
    raw = _monthly_raw()
    baselines = pbacktest.recompute_baseline_curves(raw, CFG)
    window = pd.date_range("1972-01-31", "2020-12-31", freq="ME")
    rng = np.random.default_rng(9)
    curves = {
        leg: pd.DataFrame({"return": rng.normal(0.005, 0.025, len(window)), "turnover": 0.1}, index=window)
        for leg in ("strategy", "ablation")
    }
    reports = tmp_path / "reports"
    reports.mkdir(parents=True)
    for leg, frame in curves.items():
        frame.to_parquet(reports / f"backtest_equity_curve_{leg}.parquet")
    series = {"strategy": curves["strategy"]["return"], "no_regime_ablation": curves["ablation"]["return"], **baselines}
    kpi = pd.DataFrame(
        [{"leg": leg, "terminal_log_wealth": _kpi(series[leg])[0], "max_drawdown": _kpi(series[leg])[1]}
         for leg in _LEGS]
    )
    kpi.to_parquet(reports / "backtest_kpi_table.parquet")
    registry_path = tmp_path / "registry" / "trials.jsonl"
    _write_registry(registry_path)
    cfg = {**CFG, "report": {"scoreboard_trial_tag": _TAG}}
    return {"raw": raw, "series": series, "window": window, "reports": reports, "registry": registry_path,
            "cfg": cfg, "kpi": kpi.set_index("leg")}


def _board(world: dict, raw: pd.DataFrame | None = None) -> dict:
    cm = _Cm(monthly_raw=world["raw"] if raw is None else raw)
    return scoreboard.scoreboard_table(world["cfg"], cm, reports_dir=world["reports"], registry_path=world["registry"])


def _legs(board: dict) -> dict[str, dict]:
    return {row["leg"]: row for row in board["legs"]}


class TestScoreboardTable:
    def test_common_window_numbers_are_the_kpis_on_the_window_slice(self, tmp_path):
        world = _world(tmp_path)
        board = _board(world)
        assert board["available"] is True
        assert board["reconciled"] is True, board["reason"]
        assert [row["leg"] for row in board["legs"]] == _LEGS
        window = world["window"]
        for leg, series in world["series"].items():
            sliced = series.loc[(series.index >= window[0]) & (series.index <= window[-1])]
            tlw, mdd = _kpi(sliced)
            assert _legs(board)[leg]["tlw_common"] == pytest.approx(tlw, rel=1e-9, abs=0), leg
            assert _legs(board)[leg]["mdd_common"] == pytest.approx(mdd, rel=1e-9, abs=0), leg
        # The baselines start in 1962, so their common-window numbers differ from their own span.
        assert _legs(board)["spy_buy_hold"]["tlw_common"] != pytest.approx(world["kpi"].loc["spy_buy_hold",
                                                                                           "terminal_log_wealth"])

    def test_own_span_is_the_kpi_table(self, tmp_path):
        world = _world(tmp_path)
        board = _board(world)
        for leg in _LEGS:
            assert _legs(board)[leg]["tlw_own"] == pytest.approx(
                world["kpi"].loc[leg, "terminal_log_wealth"], rel=1e-9, abs=0
            ), leg
        assert _legs(board)["spy_buy_hold"]["own_start"] == world["series"]["spy_buy_hold"].dropna().index[0]

    def test_window_is_the_strategy_curve(self, tmp_path):
        board = _board(_world(tmp_path))
        assert board["window_start"] == pd.Timestamp("1972-01-31")
        assert board["window_end"] == pd.Timestamp("2020-12-31")
        assert board["n_steps"] == 588
        assert board["cost_bps"] == 10

    def test_run_date_and_sha_come_from_the_tagged_rows_only(self, tmp_path):
        board = _board(_world(tmp_path))
        assert board["run_date"].date() == pd.Timestamp("2026-09-30").date()
        assert board["git_sha"] == "74207efa"
        assert board["trial_tag"] == _TAG
        assert board["n_tagged_rows"] == 2

    def test_perturbed_baseline_inputs_withhold_the_common_columns(self, tmp_path):
        world = _world(tmp_path)
        raw = world["raw"].copy()
        raw["sp500"] = raw["sp500"] * (1 + 0.001 * np.sin(np.arange(len(raw))))
        board = _board(world, raw=raw)
        assert board["available"] is True
        assert board["reconciled"] is False
        assert "spy_buy_hold" in board["reason"]
        lines = scoreboard.format_scoreboard(board)
        text = "\n".join(lines)
        assert "common-window columns withheld" in text
        assert "TLW 1972" not in text
        assert f"{world['kpi'].loc['faber_sma', 'terminal_log_wealth']:.4f}" in text  # own span still shown
        assert scoreboard.E07_CAVEAT in text

    def test_missing_monthly_raw_withholds_not_raises(self, tmp_path):
        world = _world(tmp_path)
        board = scoreboard.scoreboard_table(world["cfg"], _Cm(), reports_dir=world["reports"],
                                            registry_path=world["registry"])
        assert board["available"] is True
        assert board["reconciled"] is False
        assert "monthly_raw" in board["reason"]

    def test_missing_kpi_parquet_is_an_unavailable_line(self, tmp_path):
        world = _world(tmp_path)
        (world["reports"] / "backtest_kpi_table.parquet").unlink()
        board = _board(world)
        assert board["available"] is False
        text = "\n".join(scoreboard.format_scoreboard(board))
        assert "## Scoreboard (static — last budgeted run)" in text
        assert "unavailable" in text
        assert "python -m trading_crab_lib.platform.evaluation.report" in text
        assert scoreboard.E07_CAVEAT in text

    def test_missing_registry_rows_leave_the_run_date_unknown(self, tmp_path):
        world = _world(tmp_path)
        world["registry"].write_text("", encoding="utf-8")
        board = _board(world)
        assert board["run_date"] is None
        assert "run date unknown" in "\n".join(scoreboard.format_scoreboard(board)).lower()


class TestFormatScoreboard:
    def test_lines(self, tmp_path):
        world = _world(tmp_path)
        board = _board(world)
        lines = scoreboard.format_scoreboard(board)
        text = "\n".join(lines)
        assert lines[0] == "## Scoreboard (static — last budgeted run)"
        assert "| Leg | TLW 1972–2020 | MDD 1972–2020 |" in text
        tilt = _legs(board)["strategy"]
        assert f"| Regime tilt (strategy) | {tilt['tlw_common']:.4f} | {tilt['mdd_common']:.2%} |" in text
        for label in ("No-regime ablation", "SPY buy & hold", "60/40", "Faber 10-mo SMA"):
            assert f"| {label} |" in text, label
        assert "(from 1962-0" in text  # own-span start months in the footnote
        assert (
            "Run 2026-09-30 (registry tag 08.1-pit-tilt-vs-ablation) · window 1972-01-31 → 2020-12-31 · "
            "588 monthly steps · 10 bps"
        ) in text
        assert lines[-2] == scoreboard.E07_CAVEAT or scoreboard.E07_CAVEAT in lines

    def test_e07_caveat_is_the_d02_sentence(self):
        assert scoreboard.E07_CAVEAT == (
            "Caveat (E-07): scoreboard returns use monthly-average prices for equities, oil and long "
            "duration, which may flatter trend and tilt rules."
        )

    def test_page_text_avoids_regime_section_markers(self, tmp_path):
        """The scoreboard shares the no_regime page, whose tests forbid these strings."""
        text = "\n".join(scoreboard.format_scoreboard(_board(_world(tmp_path))))
        for marker in ("- regime ", "suspended", "Active Regime", "Trajectory", "distribution above"):
            assert marker not in text, marker


class TestBeforeAfterTable:
    def test_deltas_per_leg(self):
        before = pd.DataFrame({"leg": ["strategy", "spy_buy_hold"], "terminal_log_wealth": [4.0, 5.0],
                               "max_drawdown": [-0.2, -0.5]})
        after = pd.DataFrame({"leg": ["strategy", "spy_buy_hold"], "terminal_log_wealth": [3.5, 5.0],
                              "max_drawdown": [-0.25, -0.5]})
        table = scoreboard.before_after_table(before, after)
        row = table.set_index("leg").loc["strategy"]
        assert row["tlw_before"] == 4.0 and row["tlw_after"] == 3.5
        assert row["tlw_delta"] == pytest.approx(-0.5)
        assert row["mdd_delta"] == pytest.approx(-0.05)
        assert list(table["leg"]) == ["strategy", "spy_buy_hold"]


class TestBaselineCurves:
    def test_plotting_delegate_returns_the_same_series(self):
        from trading_crab_lib.platform.backtest.baselines import baseline_curves

        raw = _monthly_raw()
        got, want = pbacktest.recompute_baseline_curves(raw, CFG), baseline_curves(raw, CFG)
        assert set(got) == set(want) == {"spy_buy_hold", "sixty_forty", "faber_sma"}
        for name in want:
            pd.testing.assert_series_equal(got[name], want[name])

    def test_plotting_function_is_a_one_line_delegate(self):
        src = Path(pbacktest.__file__).read_text(encoding="utf-8")
        fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef)
                  and n.name == "recompute_baseline_curves")
        body = [n for n in fn.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
        assert len(body) == 1 and isinstance(body[0], ast.Return)


class TestSharpeVsHurdle:
    def test_one_bar_per_leg_and_a_hurdle_line(self):
        fig = pbacktest.plot_sharpe_vs_hurdle({"strategy": 0.6, "spy_buy_hold": 0.5, "faber_sma": 0.7}, 2.24,
                                              n_trials=46)
        ax = fig.axes[0]
        assert len(ax.patches) == 3
        assert any(np.allclose(line.get_ydata(), 2.24) for line in ax.get_lines())
        assert "46" in ax.get_title()


class TestBoundary:
    def test_scoreboard_imports_no_plotting(self):
        tree = ast.parse(Path(scoreboard.__file__).read_text(encoding="utf-8"))
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.add(node.module)
        assert not {n for n in names if n.startswith(("matplotlib", "seaborn")) or ".plotting" in n}, names

    def test_importing_the_page_does_not_load_matplotlib(self):
        code = (
            "import sys; import trading_crab_lib.platform.report.scoreboard, trading_crab_lib.platform.report.weekly; "
            "print('matplotlib' in sys.modules)"
        )
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                             env={"PYTHONPATH": str(_REPO / "src"), "PATH": ""}, cwd=_REPO)
        assert out.stdout.strip() == "False", out.stderr


# ── the tracked outputs (read-only) ──────────────────────────────────────────


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TestTrackedScoreboard:
    def test_tracked_numbers(self):
        from trading_crab_lib.checkpoints import CheckpointManager
        from trading_crab_lib.platform.config import load_platform_config

        registry_path = _REPO / "registry" / "trials.jsonl"
        reports = _REPO / "outputs" / "reports" / "platform"
        before = _sha(registry_path)
        board = scoreboard.scoreboard_table(
            load_platform_config(), CheckpointManager(checkpoint_dir=_REPO / "data" / "checkpoints" / "platform"),
            reports_dir=reports, registry_path=registry_path,
        )
        assert board["reconciled"] is True, board["reason"]
        common = [row["tlw_common"] for row in board["legs"]]
        assert common == pytest.approx([3.8909, 4.0424, 5.0022, 4.4799, 5.5451], rel=1e-4)
        kpi = pd.read_parquet(reports / "backtest_kpi_table.parquet").set_index("leg")
        for row in board["legs"]:
            assert row["tlw_own"] == pytest.approx(kpi.loc[row["leg"], "terminal_log_wealth"], rel=1e-9, abs=0)
        assert (board["window_start"], board["window_end"], board["n_steps"]) == (
            pd.Timestamp("1972-01-31"), pd.Timestamp("2020-12-31"), 588)
        assert board["run_date"].date().isoformat() == "2026-09-30"
        assert _sha(registry_path) == before


# ── on the weekly page ───────────────────────────────────────────────────────


class TestOnThePage:
    def test_between_the_tripwire_and_the_trades_heading(self, tmp_path):
        from test_platform_weekly_page import _assemble

        from trading_crab_lib.platform.tripwire import monitor

        tripwire = monitor.evaluate_tripwire({}, _Cm(), run_date=pd.Timestamp("2026-09-30"))
        lines = scoreboard.format_scoreboard(_board(_world(tmp_path)))
        page = _assemble(tripwire, scoreboard=lines)
        assert (
            page.index("## Crash Tripwire")
            < page.index("## Scoreboard (static — last budgeted run)")
            < page.index("## Target vs. Current — Trades Implied")
        )
        assert _assemble(tripwire, scoreboard=None) == _assemble(tripwire)
        assert "Scoreboard" not in _assemble(tripwire)

    def test_build_weekly_page_renders_the_scoreboard_from_its_cm(self, tmp_path, monkeypatch):
        from test_platform_weekly_page import _no_regime_world

        from trading_crab_lib.platform.checkpoints import get_platform_checkpoint_manager
        from trading_crab_lib.platform.report import weekly

        board_world = _world(tmp_path / "board")
        world = _no_regime_world(tmp_path / "w", monkeypatch)
        seen = []

        def table(cfg, cm=None, **kw):
            seen.append(cm)
            return _board(board_world)

        monkeypatch.setattr(weekly, "scoreboard_table", table)
        cm = get_platform_checkpoint_manager()
        markdown, _ = weekly.build_weekly_page(world["cfg"], cm, output_dir=tmp_path / "page")
        assert seen == [cm]
        assert "## Scoreboard (static — last budgeted run)" in markdown
        assert scoreboard.E07_CAVEAT in markdown
        assert markdown.index("## Scoreboard") < markdown.index("## Target vs. Current — Trades Implied")

    def test_the_world_page_degrades_gracefully_without_a_scoreboard_source(self, tmp_path, monkeypatch):
        """With no KPI table under the reports dir the scoreboard reads unavailable (naming the
        command that writes it) and the page still renders."""
        from test_platform_weekly_page import _no_regime_world

        from trading_crab_lib.platform.checkpoints import get_platform_checkpoint_manager
        from trading_crab_lib.platform.report import weekly

        world = _no_regime_world(tmp_path / "w", monkeypatch)
        board_world = _world(tmp_path / "board")
        monkeypatch.setattr(scoreboard, "OUTPUT_DIR", board_world["reports"].parent / "out_unused")
        markdown, _ = weekly.build_weekly_page(world["cfg"], get_platform_checkpoint_manager(),
                                               output_dir=tmp_path / "page")
        assert "## Scoreboard (static — last budgeted run)" in markdown
        assert "unavailable" in markdown
