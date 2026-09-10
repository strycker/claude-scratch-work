"""
platform/plotting/backtest.py — P6 backtest-evaluation panels (the phase's
evaluation capstone).

What this module renders
------------------------
1. The three **deterministic baseline legs**
   (:func:`recompute_baseline_curves`) — SPY buy-and-hold, 60/40 and Faber's
   10-month SMA, re-derived read-only from the same spliced core research
   series and the same cost convention ``evaluation/report.py`` uses, so the
   baseline gauntlet P6 draws is provably the one the KPI table summarizes
   rather than an approximation.
2. All **five equity curves on one chart**
   (:func:`plot_equity_curves`) — the two ML legs loaded verbatim from
   Phase 5's persisted artifacts, the three baselines from (1).
3. The **KPI bars** (:func:`plot_kpi_table_bars`) and the design §8.7
   **"does the regime layer pay rent" ablation delta**
   (:func:`compute_ablation_delta`).
4. A one-glance **calibration summary** (:func:`plot_calibration_summary`).
5. The design §5.4 **sojourn / detection-lag headline**
   (:func:`plot_sojourn_lag_headline`), which ALWAYS renders
   :data:`trading_crab_lib.platform.plotting.core.A13_CAVEAT` and the
   resolved-of-total transition count alongside the ratio — unconditionally,
   never gated on the ratio's value.

What this module deliberately does NOT do
-----------------------------------------
* It never invokes the full walk-forward entrypoints. Every ML-strategy
  number comes from a persisted parquet artifact; only the three
  deterministic, no-tunable-parameter price-arithmetic baselines are
  recomputed here, and ``backtest/baselines.py``'s own documented scope says
  those three log NO registry trial (they are report-only comparisons, not
  part of the multiple-testing surface).
* It resolves nothing about audit item A13. Rendering the sojourn/lag ratio
  with its caveat and its resolved-transition count makes A13 *inspectable*;
  no plausibility band around that number settles it.
* It imports no ``CheckpointManager`` and calls ``.save`` nowhere. Every
  function is pure: DataFrame/Series in, DataFrame/Figure/dict out. Opening
  P6 cannot overwrite a production checkpoint or a report artifact.

Fresh-package boundary (D-01): this module imports nothing from the legacy
``trading_crab_lib.plotting`` package, and reaches matplotlib only through
:mod:`trading_crab_lib.platform.plotting.core`, which owns the Agg-backend
guard.
"""

from __future__ import annotations

import logging
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

from trading_crab_lib.platform.backtest.baselines import faber_sma, sixty_forty, spy_buy_hold
from trading_crab_lib.platform.honesty.holdout import DEFAULT_HOLDOUT_CUTOFF, split_by_holdout_boundary
from trading_crab_lib.platform.plotting import core

log = logging.getLogger(__name__)

_STRATEGY_LEG = "strategy"
_ABLATION_LEG = "no_regime_ablation"

# Per-leg line styling. Deliberately explicit rather than a cycler: five
# overlaid curves are the figure most prone to indistinguishable colors, and
# the two ML legs must read differently from the three price baselines at a
# glance. Colors are the platform palette plus one neutral.
_LEG_STYLE: dict[str, dict] = {
    _STRATEGY_LEG: {"color": "#0000d0", "linestyle": "-", "linewidth": 2.4},
    _ABLATION_LEG: {"color": "#8338ec", "linestyle": "--", "linewidth": 1.8},
    "spy_buy_hold": {"color": "#d00000", "linestyle": "-", "linewidth": 1.3},
    "sixty_forty": {"color": "#f48c06", "linestyle": "-.", "linewidth": 1.3},
    "faber_sma": {"color": "#50a000", "linestyle": ":", "linewidth": 2.0},
}

_KPI_METRICS: tuple[str, ...] = ("terminal_log_wealth", "max_drawdown")

# Largest scatter area (points^2) for the biggest calibration bin, so a bin
# with a single observation is still visible rather than zero-sized.
_CALIB_MAX_MARKER_AREA = 320.0
_CALIB_MIN_MARKER_AREA = 18.0


def _no_data_figure(title: str, *, save_path: Path | None, show: bool) -> core.plt.Figure:
    """Annotated placeholder for an empty/degenerate input (never a raise)."""
    fig, ax = core.plt.subplots(figsize=(8, 2))
    ax.axis("off")
    ax.annotate("no data", xy=(0.5, 0.5), xycoords="axes fraction", ha="center", va="center")
    ax.set_title(title)
    return core._save_or_show(fig, save_path=save_path, show=show)


# ── The three deterministic baseline legs (read-only re-derivation) ──────────


def recompute_baseline_curves(monthly_raw: pd.DataFrame, cfg: dict) -> dict[str, pd.Series]:
    """Re-derive the three deterministic baseline return series, dev-bounded.

    This is a **read-only re-derivation** of the baseline construction inside
    ``platform/evaluation/report.py::run_full_backtest_evaluation`` — never an
    import of that private logic. It follows the same data flow using only
    public functions:

    ``build_core_research_series`` -> ``compute_monthly_returns`` -> resolve
    the equity / bond / cash research names from ``cfg["splice"]`` ->
    ``split_by_holdout_boundary(..., cutoff=DEFAULT_HOLDOUT_CUTOFF)`` on each
    -> ``spy_buy_hold`` / ``sixty_forty`` / ``faber_sma`` with the SAME
    ``cost_bps`` and rebalance conventions read from ``cfg["backtest"]``.

    All three legs are deterministic price arithmetic with no tunable
    parameter, and per ``backtest/baselines.py``'s own documented
    trial-registry scope they log **no** registry trial — which is exactly
    why they are cheap and safe to recompute in a viewing notebook, while the
    two ML legs must be loaded from Phase 5's persisted artifacts instead.

    Args:
        monthly_raw: the Phase-1 raw monthly ingest frame (dev namespace).
        cfg: platform config (``load_platform_config()`` output).

    Returns:
        dict mapping ``"spy_buy_hold"`` / ``"sixty_forty"`` / ``"faber_sma"``
        to that leg's monthly net-of-cost return ``pd.Series``, each bounded
        at or before the holdout cutoff.
    """
    # Imported at call time so this module's import graph stays shallow for
    # the D-01 boundary scan and so a notebook that never plots baselines
    # never pays for the splice/returns import chain.
    from trading_crab_lib.platform.assets.returns import compute_monthly_returns
    from trading_crab_lib.platform.splice import build_core_research_series

    splice_cfg = cfg["splice"]
    research = build_core_research_series(monthly_raw, cfg)
    returns = compute_monthly_returns(research)

    equity_name = splice_cfg["equities"]["research_name"]
    bond_name = splice_cfg["long_duration"]["research_name"]
    cash_name = splice_cfg["cash"]["research_name"]

    equity_ret = returns[equity_name]
    bond_ret = returns[bond_name]
    cash_ret = returns[cash_name]
    equity_level = research[equity_name]

    backtest_cfg = cfg.get("backtest", {})
    rebalance = backtest_cfg.get("sixty_forty_rebalance", "monthly")
    cost_bps = backtest_cfg.get("cost_bps", 10)
    baseline_cost_bps = cost_bps if backtest_cfg.get("apply_cost_to_baselines", True) else 0.0

    dev_equity_ret, _ = split_by_holdout_boundary(equity_ret, cutoff=DEFAULT_HOLDOUT_CUTOFF)
    dev_bond_ret, _ = split_by_holdout_boundary(bond_ret, cutoff=DEFAULT_HOLDOUT_CUTOFF)
    dev_cash_ret, _ = split_by_holdout_boundary(cash_ret, cutoff=DEFAULT_HOLDOUT_CUTOFF)
    dev_equity_level, _ = split_by_holdout_boundary(equity_level, cutoff=DEFAULT_HOLDOUT_CUTOFF)

    curves = {
        "spy_buy_hold": spy_buy_hold(dev_equity_ret),
        "sixty_forty": sixty_forty(
            dev_equity_ret, dev_bond_ret, rebalance=rebalance, cost_bps=baseline_cost_bps
        ),
        "faber_sma": faber_sma(dev_equity_level, dev_cash_ret, cost_bps=baseline_cost_bps),
    }
    log.info(
        "Re-derived %d deterministic baseline legs (cost_bps=%s, rebalance=%s), dev-bounded at %s",
        len(curves),
        baseline_cost_bps,
        rebalance,
        DEFAULT_HOLDOUT_CUTOFF,
    )
    return curves


# ── Equity curves — all five legs on one chart ───────────────────────────────


def plot_equity_curves(
    curves: dict[str, pd.Series],
    *,
    title: str = "Equity curves (cumulative log wealth)",
    save_path: Path | None = None,
    show: bool = False,
) -> core.plt.Figure:
    """Overlay one cumulative-log-wealth line per entry of *curves*.

    Each series is a monthly RETURN series; the plotted line is
    ``np.log1p(series.dropna()).cumsum()``, so a line's final value is that
    leg's terminal log wealth over its own observed span. Legs may start at
    different dates (the walk-forward legs begin after their training warmup
    while the price baselines begin at the spine's start) — that is not
    normalized away here, because differing spans are a real property of the
    comparison and hiding it would misrepresent the KPI table.

    Args:
        curves: mapping of leg name to that leg's monthly return series.
        title: axes title.
        save_path: written via :func:`core._save_or_show` when not None.
        show: forwarded to :func:`core._save_or_show`.

    Returns:
        The unclosed :class:`matplotlib.figure.Figure`.
    """
    drawable = {name: s.dropna() for name, s in curves.items() if s is not None and not s.dropna().empty}
    if not drawable:
        return _no_data_figure(title, save_path=save_path, show=show)

    fig, ax = core.plt.subplots(figsize=(12, 6))
    for idx, (name, series) in enumerate(drawable.items()):
        style = dict(_LEG_STYLE.get(name, {}))
        style.setdefault("color", core._regime_color(idx))
        style.setdefault("linestyle", "-")
        style.setdefault("linewidth", 1.4)
        cum = np.log1p(series.sort_index()).cumsum()
        ax.plot(cum.index, cum.to_numpy(), label=f"{name} ({cum.iloc[-1]:.2f})", **style)

    ax.axhline(0.0, color="#666666", linewidth=0.8, linestyle="-", zorder=0)
    ax.set_title(title)
    ax.set_ylabel("cumulative log wealth")
    ax.set_xlabel("date")
    ax.grid(alpha=0.25)
    # Legend below the axes: five overlaid curves fill the frame, and an
    # in-axes legend occludes exactly the lines it labels (06-05's finding).
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=min(len(drawable), 3),
        frameon=False,
        fontsize=9,
    )
    fig.tight_layout()
    return core._save_or_show(fig, save_path=save_path, show=show)


# ── KPI table: the ablation delta and the bar chart ──────────────────────────


def compute_ablation_delta(kpi_table: pd.DataFrame) -> dict[str, float]:
    """Design §8.7's "does the regime layer pay rent" delta.

    Args:
        kpi_table: the persisted ``backtest_kpi_table.parquet`` frame —
            columns ``leg``, ``terminal_log_wealth``, ``max_drawdown``.

    Returns:
        ``{"wealth_delta": strategy - ablation, "dd_delta": strategy - ablation}``
        for ``terminal_log_wealth`` and ``max_drawdown`` respectively. A
        positive ``wealth_delta`` means the regime layer earned its keep on
        terminal wealth; a positive ``dd_delta`` means the strategy drew down
        LESS deeply than the ablation (drawdowns are negative numbers).

    Raises:
        ValueError: naming the missing leg, if either row is absent.
    """
    indexed = kpi_table.set_index("leg")
    for leg in (_STRATEGY_LEG, _ABLATION_LEG):
        if leg not in indexed.index:
            raise ValueError(
                f"kpi_table has no {leg!r} row — cannot compute the design §8.7 ablation delta. "
                f"Legs present: {sorted(indexed.index.astype(str))}"
            )
    strategy = indexed.loc[_STRATEGY_LEG]
    ablation = indexed.loc[_ABLATION_LEG]
    return {
        "wealth_delta": float(strategy["terminal_log_wealth"]) - float(ablation["terminal_log_wealth"]),
        "dd_delta": float(strategy["max_drawdown"]) - float(ablation["max_drawdown"]),
    }


def plot_kpi_table_bars(
    kpi_table: pd.DataFrame,
    *,
    title: str = "Backtest KPIs by leg (read live from backtest_kpi_table.parquet)",
    save_path: Path | None = None,
    show: bool = False,
) -> core.plt.Figure:
    """Grouped bars — one group per leg, one panel per KPI metric.

    ``terminal_log_wealth`` and ``max_drawdown`` live on incompatible scales
    (a positive log wealth of ~5 against a drawdown of ~-0.5), so they are
    drawn in two side-by-side panels sharing the leg axis rather than as two
    bars in one group, where the drawdown bars would collapse onto zero.

    Args:
        kpi_table: the persisted ``backtest_kpi_table.parquet`` frame.
        title: figure suptitle.
        save_path: written via :func:`core._save_or_show` when not None.
        show: forwarded to :func:`core._save_or_show`.

    Returns:
        The unclosed :class:`matplotlib.figure.Figure`.
    """
    if kpi_table is None or kpi_table.empty:
        return _no_data_figure(title, save_path=save_path, show=show)

    legs = kpi_table["leg"].astype(str).tolist()
    positions = np.arange(len(legs))
    colors = [_LEG_STYLE.get(leg, {}).get("color", core._regime_color(i)) for i, leg in enumerate(legs)]

    fig, axes = core.plt.subplots(1, len(_KPI_METRICS), figsize=(13, 4.6))
    for ax, metric in zip(np.atleast_1d(axes), _KPI_METRICS):
        values = kpi_table[metric].astype(float).to_numpy() if metric in kpi_table.columns else np.zeros(len(legs))
        ax.bar(positions, values, color=colors, edgecolor="#333333", linewidth=0.6)
        ax.set_xticks(positions)
        ax.set_xticklabels(legs, rotation=30, ha="right", fontsize=9)
        ax.set_title(metric)
        ax.axhline(0.0, color="#333333", linewidth=0.8)
        ax.grid(axis="y", alpha=0.25)
        span = float(np.nanmax(np.abs(values))) if len(values) else 1.0
        pad = 0.20 * (span if span > 0 else 1.0)
        for pos, value in zip(positions, values):
            above = value >= 0
            ax.annotate(
                f"{value:.4f}",
                xy=(pos, value),
                xytext=(0, 4 if above else -12),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
        ax.margins(y=0.18)
        ax.set_ylim(min(0.0, float(np.nanmin(values)) - pad), max(0.0, float(np.nanmax(values)) + pad))

    fig.suptitle(title)
    fig.tight_layout()
    return core._save_or_show(fig, save_path=save_path, show=show)


# ── Calibration summary (all classes at a glance) ────────────────────────────


def plot_calibration_summary(
    calibration_df: pd.DataFrame,
    *,
    title: str = "Calibration summary (walk-forward, all classes)",
    save_path: Path | None = None,
    show: bool = False,
) -> core.plt.Figure:
    """One scatter point per (class, bin) of the persisted calibration artifact.

    A lighter, all-classes-at-a-glance companion to P4's per-class
    ``nowcaster.plot_calibration_curve``. This is **deliberately independent
    code** — ``backtest.py`` and ``nowcaster.py`` import nothing from each
    other, because the two notebooks are built in the same wave and must not
    depend on each other's completion.

    Point area is scaled by ``n_in_bin`` (the largest bin gets a fixed
    maximum area, the smallest a visible floor — never zero), so a bin
    carrying three observations cannot masquerade as evidence.

    Args:
        calibration_df: ``model_metrics_calibration.parquet`` — columns
            ``class_label``, ``bin``, ``bin_low``, ``bin_high``,
            ``predicted_prob_mean``, ``observed_freq``, ``n_in_bin``.
        title: axes title.
        save_path: written via :func:`core._save_or_show` when not None.
        show: forwarded to :func:`core._save_or_show`.

    Returns:
        The unclosed :class:`matplotlib.figure.Figure`.
    """
    if calibration_df is None or calibration_df.empty:
        return _no_data_figure(title, save_path=save_path, show=show)

    counts = calibration_df["n_in_bin"].astype(float).to_numpy()
    largest = float(np.nanmax(counts)) if len(counts) else 0.0
    if largest > 0:
        areas = _CALIB_MIN_MARKER_AREA + (counts / largest) * (_CALIB_MAX_MARKER_AREA - _CALIB_MIN_MARKER_AREA)
    else:
        areas = np.full(len(counts), _CALIB_MIN_MARKER_AREA)

    fig, ax = core.plt.subplots(figsize=(7.5, 7))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#666666", linewidth=1.0, label="perfectly calibrated")

    for class_label, group in calibration_df.groupby("class_label", sort=True):
        mask = calibration_df["class_label"].to_numpy() == class_label
        ax.scatter(
            group["predicted_prob_mean"].astype(float).to_numpy(),
            group["observed_freq"].astype(float).to_numpy(),
            s=areas[mask],
            color=core._regime_color(int(class_label)),
            alpha=0.75,
            edgecolor="#222222",
            linewidth=0.6,
            label=f"state {int(class_label)}",
        )

    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed frequency")
    ax.set_title(title)
    # A hair of padding on both axes: bins at exactly 0.0 / 1.0 are otherwise
    # half-clipped by the frame (06-05's finding).
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.25)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=3, frameon=False, fontsize=9)
    ax.annotate(
        "point area ∝ n_in_bin",
        xy=(0.02, 0.97),
        xycoords="axes fraction",
        ha="left",
        va="top",
        fontsize=8,
        color="#444444",
    )
    fig.tight_layout()
    return core._save_or_show(fig, save_path=save_path, show=show)


# ── The §5.4 sojourn / detection-lag headline (A13 discipline) ───────────────


def plot_sojourn_lag_headline(
    headline: dict,
    *,
    caveat: str | None = None,
    title: str = "Median sojourn vs median detection lag (design §5.4)",
    save_path: Path | None = None,
    show: bool = False,
) -> core.plt.Figure:
    """Two bars (median sojourn, median detection lag) with the A13 caveat.

    The caveat is rendered **unconditionally** — never behind a flag, never
    only when a threshold trips — and the resolved-of-total transition count
    is always printed alongside the ratio, because that ratio is a median
    over a handful of resolved transitions. A NaN ratio (every transition
    unresolved) renders as ``"n/a (unresolved)"`` rather than raising.

    Nothing here resolves audit item A13. The two series being compared are
    fit on different, time-varying feature sets over different spans, so
    their gap is not established to be a detection lag at all.

    Args:
        headline: ``evaluation.sojourn_lag.compute_sojourn_lag_headline``
            output — keys ``median_sojourn``, ``median_lag``, ``ratio``,
            ``n_transitions``, ``n_resolved``, ``act_threshold``.
        caveat: overrides :data:`core.A13_CAVEAT` (defaults to it when None).
        title: axes title.
        save_path: written via :func:`core._save_or_show` when not None.
        show: forwarded to :func:`core._save_or_show`.

    Returns:
        The unclosed :class:`matplotlib.figure.Figure`.
    """
    caveat_text = core.A13_CAVEAT if caveat is None else caveat

    def _as_float(key: str) -> float:
        value = headline.get(key, float("nan"))
        return float("nan") if value is None else float(value)

    sojourn = _as_float("median_sojourn")
    lag = _as_float("median_lag")
    ratio = _as_float("ratio")
    n_resolved = headline.get("n_resolved", 0)
    n_transitions = headline.get("n_transitions", 0)
    act_threshold = headline.get("act_threshold", float("nan"))

    ratio_text = "n/a (unresolved)" if not np.isfinite(ratio) else f"{ratio:.4f}"
    resolved_text = f"{n_resolved} of {n_transitions} transitions resolved"

    labels = ["median sojourn", "median detection lag"]
    values = [0.0 if not np.isfinite(sojourn) else sojourn, 0.0 if not np.isfinite(lag) else lag]

    fig, ax = core.plt.subplots(figsize=(10, 5.6))
    bars = ax.bar(
        labels,
        values,
        color=[core.CUSTOM_COLORS[4], core.CUSTOM_COLORS[1]],
        edgecolor="#333333",
        linewidth=0.7,
        width=0.5,
    )
    for bar, raw in zip(bars, [sojourn, lag]):
        text = "n/a" if not np.isfinite(raw) else f"{raw:.1f} mo"
        ax.annotate(
            text,
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=10,
        )

    ax.set_ylabel("months")
    # `pad` reserves vertical room for the bold summary line annotated just
    # above the axes — without it the title and the summary overprint each
    # other into an unreadable smear.
    ax.set_title(title, pad=30)
    ax.grid(axis="y", alpha=0.25)
    # Headroom for the bar value labels inside the axes.
    ax.margins(y=0.22)
    ax.annotate(
        f"sojourn/lag ratio = {ratio_text}   |   {resolved_text}"
        f"   |   act_threshold = {act_threshold}",
        xy=(0.5, 1.005),
        xycoords="axes fraction",
        ha="center",
        va="bottom",
        fontsize=10.5,
        fontweight="bold",
    )

    wrapped = "\n".join(textwrap.wrap(caveat_text, width=105))
    # Room BELOW the axes for the caveat box, which must never sit on top of
    # the bars and must never be so far away it reads as unrelated.
    fig.subplots_adjust(bottom=0.30, top=0.86)
    fig.text(
        0.5,
        0.035,
        wrapped,
        ha="center",
        va="bottom",
        fontsize=8.5,
        wrap=True,
        bbox={"boxstyle": "round", "facecolor": "#fff3cd", "edgecolor": "#d39e00"},
    )
    return core._save_or_show(fig, save_path=save_path, show=show)
