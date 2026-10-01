"""
Static scoreboard for the weekly page and notebook N7 (plan 08.2-03, D-05, DECISIONS A-14).

One table from the LAST BUDGETED RUN's tracked outputs (8.1's D-04 run): the regime tilt, the
no-regime ablation, SPY buy-and-hold, 60/40 and Faber's 10-month SMA, each with terminal log
wealth (TLW) and max drawdown (MDD). It changes only when a budgeted run rewrites those outputs;
nothing here fits, tunes, selects or writes, and no registry row is appended.

**Where each number comes from.**

- The strategy and ablation legs: ``backtest_equity_curve_{strategy,ablation}.parquet``'s
  ``return`` column, verbatim.
- The three baselines: ``backtest/baselines.py::baseline_curves`` on ``monthly_raw``
  (deterministic price arithmetic, no registry trial), the derivation the KPI table used.
- The own-span TLW and MDD: ``backtest_kpi_table.parquet``.
- The run date and git sha: the registry rows whose ``config.trial_tag`` is
  ``report.scoreboard_trial_tag`` (``08.1-pit-tilt-vs-ablation``); no parquet carries a date.

**The common window (ruling A3).** The KPI table is not like-for-like: the baselines start in
1962, the walk-forward legs in 1972. The headline columns therefore put all five legs on the
strategy curve's own window (1972-01 to 2020-12, 588 steps); the own-span TLW goes in a
footnote with each baseline's start month.

**Reconciliation (T-08.2-12).** Each leg's own-span TLW and MDD, recomputed here, must equal the
KPI table at rel 1e-9; the baselines must also cover every month of the window. If anything
fails (or ``monthly_raw`` cannot be read) the common columns are withheld with the reason and
the page shows the KPI table's own-span values. A missing KPI table or curve is an
``unavailable`` scoreboard that names the command that writes it, never an exception.

**E-07 (D-02).** Every rendering carries ``E07_CAVEAT``: the returns are built from
monthly-average prices, which may flatter trend and tilt rules. Fixing that is deferred to the
regime-rebuild phase; nothing here changes how returns are built.

No matplotlib and no ``platform.plotting`` import: the weekly page must not need the optional
plotting extra (the Sharpe-vs-hurdle figure lives in ``plotting/backtest.py``, G-02).
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import pandas as pd

from trading_crab_lib.platform import checkpoints as platform_checkpoints
from trading_crab_lib.platform.backtest.baselines import baseline_curves
from trading_crab_lib.platform.checkpoints import get_platform_checkpoint_manager
from trading_crab_lib.platform.evaluation.kpis import max_drawdown_and_duration, terminal_log_wealth
from trading_crab_lib.platform.honesty import registry

log = logging.getLogger(__name__)

#: D-02's one-line caveat, shared by the page and N7.
E07_CAVEAT = (
    "Caveat (E-07): scoreboard returns use monthly-average prices for equities, oil and long "
    "duration, which may flatter trend and tilt rules."
)

#: The budgeted run that writes every number the scoreboard reads. Never run it to refresh the page.
REPORT_BUILD_COMMAND = "python -m trading_crab_lib.platform.evaluation.report"

#: ``report.scoreboard_trial_tag`` default: 8.1's D-04 run.
DEFAULT_SCOREBOARD_TRIAL_TAG = "08.1-pit-tilt-vs-ablation"

#: (leg id in the KPI table, page label), in page order.
LEG_LABELS: tuple[tuple[str, str], ...] = (
    ("strategy", "Regime tilt (strategy)"),
    ("no_regime_ablation", "No-regime ablation"),
    ("spy_buy_hold", "SPY buy & hold"),
    ("sixty_forty", "60/40"),
    ("faber_sma", "Faber 10-mo SMA"),
)
_BASELINES = ("spy_buy_hold", "sixty_forty", "faber_sma")
_CURVE_FILES = {"strategy": "backtest_equity_curve_strategy.parquet",
                "no_regime_ablation": "backtest_equity_curve_ablation.parquet"}
_KPI_FILE = "backtest_kpi_table.parquet"
_REL = 1e-9
_HEADING = "## Scoreboard (static — last budgeted run)"


def _kpis(returns: pd.Series) -> tuple[float, float]:
    """(TLW, MDD) over the non-null months, as ``evaluation/report.py`` computes them."""
    clean = returns.dropna()
    return terminal_log_wealth(clean), float(max_drawdown_and_duration(clean)["max_drawdown"])


def _close(a: float, b: float) -> bool:
    return math.isclose(float(a), float(b), rel_tol=_REL, abs_tol=0.0)


def _run_facts(tag: str, registry_path: Path | str | None) -> dict[str, Any]:
    """The run date (latest tagged timestamp) and git sha from the tagged registry rows."""
    trials = registry.read_trials(registry_path)
    if trials.empty or "config" not in trials.columns:
        return {"run_date": None, "git_sha": None, "n_tagged_rows": 0}
    tags = trials["config"].apply(lambda c: c.get("trial_tag") if isinstance(c, dict) else None)
    tagged = trials[tags == tag]
    if tagged.empty:
        return {"run_date": None, "git_sha": None, "n_tagged_rows": 0}
    stamps = pd.to_datetime(tagged["timestamp"], utc=True)
    last = stamps.idxmax()
    sha = tagged.loc[last, "git_sha"] if "git_sha" in tagged.columns else None
    return {"run_date": stamps.loc[last], "git_sha": None if pd.isna(sha) else str(sha),
            "n_tagged_rows": int(len(tagged))}


def scoreboard_table(
    cfg: dict[str, Any],
    cm: Any = None,
    *,
    reports_dir: Path | None = None,
    registry_path: Path | str | None = None,
) -> dict[str, Any]:
    """Read the last budgeted run's numbers, put the five legs on one window, reconcile.

    Args:
        cfg: platform config (``splice`` and ``backtest`` for the baselines; ``report.
            scoreboard_trial_tag``).
        cm: checkpoint manager holding ``monthly_raw`` (default: the platform namespace).
        reports_dir: where the KPI table and curves live (default ``PLATFORM_REPORT_DIR``,
            i.e. ``OUTPUT_DIR/reports/platform``: the tracked outputs, read at call time).
        registry_path: the trial registry (default: the live ledger), read only.

    Returns a dict: ``available`` / ``reason``; ``legs`` (one dict per leg in page order:
    ``leg``, ``label``, ``tlw_common``, ``mdd_common`` (None when withheld), ``tlw_own``,
    ``mdd_own``, ``own_start``); ``window_start``, ``window_end``, ``n_steps``, ``cost_bps``;
    ``reconciled``; ``run_date``, ``git_sha``, ``trial_tag``, ``n_tagged_rows``.
    """
    reports = Path(reports_dir) if reports_dir is not None else platform_checkpoints.PLATFORM_REPORT_DIR
    tag = str(cfg.get("report", {}).get("scoreboard_trial_tag", DEFAULT_SCOREBOARD_TRIAL_TAG))
    board: dict[str, Any] = {
        "available": False, "reason": None, "legs": [], "window_start": None, "window_end": None,
        "n_steps": 0, "cost_bps": cfg.get("backtest", {}).get("cost_bps", 10), "reconciled": False,
        "trial_tag": tag, **_run_facts(tag, registry_path),
    }

    try:
        kpi = pd.read_parquet(reports / _KPI_FILE).set_index("leg")
        curves = {leg: pd.read_parquet(reports / name)["return"] for leg, name in _CURVE_FILES.items()}
        own = {leg: (float(kpi.loc[leg, "terminal_log_wealth"]), float(kpi.loc[leg, "max_drawdown"]))
               for leg, _ in LEG_LABELS}
    except (FileNotFoundError, KeyError) as exc:
        board["reason"] = f"the tracked backtest outputs under {reports} cannot be read ({exc!r})"
        log.warning("scoreboard unavailable: %s", board["reason"])
        return board

    window = pd.DatetimeIndex(curves["strategy"].index)
    board.update(available=True, window_start=pd.Timestamp(window[0]), window_end=pd.Timestamp(window[-1]),
                 n_steps=int(len(window)))
    series: dict[str, pd.Series] = dict(curves)
    problems: list[str] = []
    try:
        series.update(baseline_curves(cm_or_default(cm).load("monthly_raw"), cfg))
    except (FileNotFoundError, KeyError, ValueError) as exc:
        problems.append(f"the baselines could not be rebuilt from monthly_raw ({exc!r})")

    for leg, label in LEG_LABELS:
        row: dict[str, Any] = {"leg": leg, "label": label, "tlw_own": own[leg][0], "mdd_own": own[leg][1],
                               "own_start": None, "tlw_common": None, "mdd_common": None}
        if leg in series:
            s = series[leg]
            clean = s.dropna()
            row["own_start"] = pd.Timestamp(clean.index[0]) if len(clean) else None
            tlw, mdd = _kpis(s)
            if not (_close(tlw, own[leg][0]) and _close(mdd, own[leg][1])):
                problems.append(
                    f"{leg} does not reconcile with {_KPI_FILE} (TLW {tlw!r} vs {own[leg][0]!r}, "
                    f"MDD {mdd!r} vs {own[leg][1]!r})"
                )
            sliced = s.loc[(s.index >= window[0]) & (s.index <= window[-1])].dropna()
            if not sliced.index.equals(window):
                problems.append(f"{leg} covers {len(sliced)} of the window's {len(window)} months")
            row["tlw_common"], row["mdd_common"] = _kpis(sliced)
        board["legs"].append(row)

    board["reconciled"] = not problems
    if problems:
        board["reason"] = "; ".join(problems)
        for row in board["legs"]:
            row["tlw_common"] = row["mdd_common"] = None
        log.warning("scoreboard common-window columns withheld: %s", board["reason"])
    return board


def cm_or_default(cm: Any) -> Any:
    """``cm``, or the platform checkpoint manager when None (resolved at call time)."""
    return cm if cm is not None else get_platform_checkpoint_manager()


def _run_line(board: dict[str, Any]) -> str:
    if board["run_date"] is None:
        run = f"Run date unknown (no registry row is tagged {board['trial_tag']})"
    else:
        run = f"Run {pd.Timestamp(board['run_date']).date().isoformat()} (registry tag {board['trial_tag']})"
    if board["window_start"] is None:
        return run
    return (
        f"{run} · window {board['window_start'].date().isoformat()} → {board['window_end'].date().isoformat()} · "
        f"{board['n_steps']} monthly steps · {board['cost_bps']:g} bps"
    )


def format_scoreboard(board: dict[str, Any]) -> list[str]:
    """The page block: heading, table, own-span footnote, run line, E-07 line.

    Reconciled: ``| Leg | TLW <y0>–<y1> | MDD <y0>–<y1> |`` on the common window, then a footnote
    with every leg's own-span TLW (and the baselines' start months). Not reconciled: the KPI
    table's own-span columns and a "common-window columns withheld" line with the reason.
    Unavailable: one line naming ``REPORT_BUILD_COMMAND``. Always the E-07 caveat.
    """
    lines = [_HEADING, ""]
    if not board["available"]:
        lines.append(
            f"Scoreboard unavailable: {board['reason']}. It is written by the budgeted run "
            f"`{REPORT_BUILD_COMMAND}` (do not run it just to refresh this page)."
        )
        lines.extend(["", E07_CAVEAT, ""])
        return lines

    if board["reconciled"]:
        years = f"{board['window_start'].year}–{board['window_end'].year}"
        lines.append(f"| Leg | TLW {years} | MDD {years} |")
        lines.append("|---|---|---|")
        for row in board["legs"]:
            lines.append(f"| {row['label']} | {row['tlw_common']:.4f} | {row['mdd_common']:.2%} |")
        lines.append("")
        own = []
        for row in board["legs"]:
            start = row["own_start"]
            late = start is not None and row["leg"] in _BASELINES
            own.append(f"{row['label']} {row['tlw_own']:.4f}" + (f" (from {start:%Y-%m})" if late else ""))
        lines.append(
            "Own-span TLW (backtest_kpi_table.parquet): " + "; ".join(own) + ". The table above puts all "
            "five legs on the same months, so its numbers are comparable; the own-span ones are not."
        )
    else:
        lines.append("| Leg | TLW (own span) | MDD (own span) |")
        lines.append("|---|---|---|")
        for row in board["legs"]:
            lines.append(f"| {row['label']} | {row['tlw_own']:.4f} | {row['mdd_own']:.2%} |")
        lines.append("")
        lines.append(
            f"The common-window columns withheld: {board['reason']}. The own-span values above come "
            "from different start months and are not like-for-like."
        )
    lines.extend(["", _run_line(board), "", E07_CAVEAT, ""])
    return lines


def before_after_table(before_kpi: pd.DataFrame, after_kpi: pd.DataFrame) -> pd.DataFrame:
    """One row per leg (``after_kpi``'s order, then any leg only ``before_kpi`` has): TLW and MDD
    before, after and the change (after minus before). N7's point-in-time table (E-06)."""
    before = before_kpi.set_index("leg")
    after = after_kpi.set_index("leg")
    legs = list(after.index) + [leg for leg in before.index if leg not in after.index]
    rows = []
    for leg in legs:
        tb = float(before.loc[leg, "terminal_log_wealth"]) if leg in before.index else float("nan")
        ta = float(after.loc[leg, "terminal_log_wealth"]) if leg in after.index else float("nan")
        mb = float(before.loc[leg, "max_drawdown"]) if leg in before.index else float("nan")
        ma = float(after.loc[leg, "max_drawdown"]) if leg in after.index else float("nan")
        rows.append({"leg": leg, "tlw_before": tb, "tlw_after": ta, "tlw_delta": ta - tb,
                     "mdd_before": mb, "mdd_after": ma, "mdd_delta": ma - mb})
    return pd.DataFrame(rows, columns=["leg", "tlw_before", "tlw_after", "tlw_delta", "mdd_before", "mdd_after",
                                       "mdd_delta"])
