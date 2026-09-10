"""
platform/plotting/allocation.py — P5 assets (L3) and allocation (L4) panels.

What this module renders
------------------------
1. The **real investable universe** the backtest actually tilts across
   (:func:`investable_asset_returns`) — a read-only re-derivation of
   ``platform/evaluation/report.py``'s asset-universe construction, so P5 can
   never quietly show a wider or narrower set of tickers than
   ``run_backtest`` trades.
2. **Regime-conditional return statistics** for that universe
   (:func:`plot_returns_by_regime_heatmap`), pivoted from
   :func:`trading_crab_lib.platform.assets.returns.returns_by_regime_stats` —
   never a hand-rolled parallel statistic.
3. Each asset's **trailing annualized EWMA volatility**
   (:func:`plot_ewma_vol_timeline`), so a vol-regime shift is visible by eye.
4. How the naive **vol-targeted regime tilt**'s weights evolve when driven by
   the full-sample smoothed labeling
   (:func:`compute_smoothed_tilt_weights_over_time`,
   :func:`plot_tilt_weights_over_time`).

What this module deliberately does NOT do
-----------------------------------------
* It calls neither ``run_backtest`` nor ``run_full_backtest_evaluation``. The
  weights-over-time panel is a cheap smoothed-labeling illustration, not the
  real walk-forward (that is P6's, and it is a different and far more
  expensive computation).
* It re-derives no allocation math. ``vol_targeted_tilt`` and
  ``returns_by_regime_stats`` are imported and called verbatim; the only thing
  written here is the per-decision-date orchestration around them.
* It imports no ``CheckpointManager`` and calls ``.save`` nowhere. Every
  function is pure: DataFrame in, DataFrame/Figure out. Opening P5 cannot
  overwrite ``hysteresis_state``, ``returns_by_regime``, or any other
  production checkpoint.

Fresh-package boundary (D-01): this module imports nothing from the legacy
``trading_crab_lib.plotting`` package, and reaches matplotlib only through
:mod:`trading_crab_lib.platform.plotting.core`, which owns the Agg-backend guard.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from trading_crab_lib.platform.assets.returns import compute_monthly_returns
from trading_crab_lib.platform.assets.vol import MONTHLY_ANNUALIZATION, ewma_vol
from trading_crab_lib.platform.plotting import core
from trading_crab_lib.platform.splice import build_core_research_series

log = logging.getLogger(__name__)

# Neutral gray for the cash sleeve — deliberately outside core.CUSTOM_COLORS so
# cash never reads as one of the five regime/asset bands.
_CASH_COLOR = "#9e9e9e"

_CASH_COLUMN = "cash"


def _no_data_figure(title: str, *, save_path: Path | None, show: bool) -> core.plt.Figure:
    """Annotated placeholder for an empty/degenerate input (never a raise)."""
    fig, ax = core.plt.subplots(figsize=(8, 2))
    ax.axis("off")
    ax.annotate("no data", xy=(0.5, 0.5), xycoords="axes fraction", ha="center", va="center")
    ax.set_title(title)
    return core._save_or_show(fig, save_path=save_path, show=show)


# ── The real investable universe (read-only re-derivation) ───────────────────


def investable_asset_returns(monthly_raw: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.Series]:
    """The 4-asset investable universe ``run_backtest`` tilts across, plus cash.

    This is a **read-only re-derivation** of the asset-universe construction in
    ``platform/evaluation/report.py::run_full_backtest_evaluation`` — not an
    import of that private construction, and not an approximation of it. Both
    sites cite the same ``cfg["splice"]`` dispatch rule, which is what keeps
    them in sync:

    * every non-``cash`` splice class contributes one column, keyed by that
      class's ``tradable`` ticker and valued by ``returns[research_name]``;
    * a class whose ``research_name`` is absent from the computed returns (an
      ``optional: true`` class such as gold, when neither of its candidate
      sources resolved) is skipped with a WARNING naming it, mirroring
      ``report.py``'s own ``_excluded`` handling;
    * ``cash`` is **excluded from the tilt by design**. It is never a tilted
      risk position — it is the vol-target residual that earns ``cash_ret``.

    Only public functions are used: :func:`build_core_research_series` and
    :func:`compute_monthly_returns`.

    Args:
        monthly_raw: the ``monthly_raw`` platform checkpoint frame.
        cfg: platform config (``load_platform_config()`` output).

    Returns:
        tuple: ``(asset_returns, cash_ret)`` — a DataFrame with one column per
        tradable ticker, and the cash return Series on the same index.
    """
    splice_cfg = cfg["splice"]
    research = build_core_research_series(monthly_raw, cfg)
    returns = compute_monthly_returns(research)

    asset_returns = pd.DataFrame(
        {
            params["tradable"]: returns[params["research_name"]]
            for name, params in splice_cfg.items()
            if name != _CASH_COLUMN and params["research_name"] in returns.columns
        },
        index=returns.index,
    )
    excluded = [
        params["research_name"]
        for name, params in splice_cfg.items()
        if name != _CASH_COLUMN and params["research_name"] not in returns.columns
    ]
    if excluded:
        log.warning("Investable universe EXCLUDES unavailable research classes: %s", excluded)

    cash_ret = returns[splice_cfg[_CASH_COLUMN]["research_name"]]
    return asset_returns, cash_ret


# ── L3: regime-conditional statistics and volatility ─────────────────────────


def plot_returns_by_regime_heatmap(
    stats_df: pd.DataFrame,
    *,
    metric: str = "sharpe_annualized",
    title: str | None = None,
    save_path: Path | None = None,
    show: bool = False,
) -> core.plt.Figure:
    """Annotated regime x asset heatmap of one ``returns_by_regime_stats`` metric.

    The color scale is diverging and symmetric about zero, so the sign of a
    regime-conditional Sharpe (or mean return) is readable at a glance rather
    than inferred from a sequential ramp.

    Args:
        stats_df: a :func:`returns_by_regime_stats`-shaped long frame.
        metric: which of its value columns to render.

    Returns:
        matplotlib Figure — "no data" annotated for an empty frame or a
        *metric* the frame does not carry.
    """
    heading = title or f"{metric} by regime x asset"
    if stats_df.empty or metric not in stats_df.columns:
        return _no_data_figure(heading, save_path=save_path, show=show)

    matrix = stats_df.pivot(index="regime", columns="asset", values=metric).sort_index().sort_index(axis=1)
    values = matrix.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    limit = float(np.abs(finite).max()) if finite.size else 1.0
    limit = limit if limit > 0 else 1.0

    fig, ax = core.plt.subplots(figsize=(1.6 * max(3, values.shape[1]) + 2, 1.0 * max(3, values.shape[0]) + 2))
    image = ax.imshow(values, cmap="RdYlGn", aspect="auto", vmin=-limit, vmax=limit)
    fig.colorbar(image, ax=ax, label=metric)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            cell = values[i, j]
            # White text on the saturated ends of RdYlGn — black on dark green
            # (or dark red) is legible only just, and this table is read for
            # its extremes.
            deep = np.isfinite(cell) and abs(cell) > 0.6 * limit
            ax.text(
                j,
                i,
                "n/a" if not np.isfinite(cell) else f"{cell:.3f}",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if deep else "black",
            )

    ax.set_xticks(np.arange(values.shape[1]))
    ax.set_xticklabels([str(c) for c in matrix.columns])
    ax.set_yticks(np.arange(values.shape[0]))
    ax.set_yticklabels([f"state {r}" for r in matrix.index])
    ax.set_xlabel("asset")
    ax.set_ylabel("regime")
    ax.set_title(heading)
    fig.tight_layout()
    return core._save_or_show(fig, save_path=save_path, show=show)


def plot_ewma_vol_timeline(
    asset_returns: pd.DataFrame,
    *,
    halflife: float = 6,
    title: str = "Trailing annualized EWMA volatility",
    save_path: Path | None = None,
    show: bool = False,
) -> core.plt.Figure:
    """One trailing annualized EWMA-vol line per column of *asset_returns*.

    Each column's vol is computed on that column's own ``dropna()``'d history
    (so a late-inception ticker is not penalized by leading NaNs) and then
    reindexed back onto ``asset_returns.index`` for a shared x-axis.

    The decay math is :func:`trading_crab_lib.platform.assets.vol.ewma_vol` —
    the same function the allocation tilt sizes positions with. It is not
    re-implemented here.

    Returns:
        matplotlib Figure — "no data" annotated for an empty or all-NaN input.
    """
    if asset_returns.empty or not len(asset_returns.columns) or not bool(asset_returns.notna().any().any()):
        return _no_data_figure(title, save_path=save_path, show=show)

    fig, ax = core.plt.subplots(figsize=(12, 4.5))
    drawn = 0
    for position, column in enumerate(asset_returns.columns):
        clean = asset_returns[column].dropna()
        if len(clean) < 2:
            log.warning("EWMA vol not estimable for %s (%d non-NaN observations)", column, len(clean))
            continue
        vol = ewma_vol(clean, halflife=halflife, annualization_factor=MONTHLY_ANNUALIZATION)
        ax.plot(
            asset_returns.index,
            vol.reindex(asset_returns.index).to_numpy(dtype=float),
            label=str(column),
            color=core.CUSTOM_COLORS[position % len(core.CUSTOM_COLORS)],
            linewidth=1.2,
        )
        drawn += 1

    if not drawn:
        core.plt.close(fig)
        return _no_data_figure(title, save_path=save_path, show=show)

    ax.set_ylabel("annualized vol")
    ax.set_ylim(bottom=0.0)
    ax.set_title(f"{title} (halflife={halflife} months)")
    ax.grid(alpha=0.25)
    # Legend below the axes: vol spikes reach the top-left and top-right of the
    # frame at exactly the crisis months an operator most wants to read.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=min(6, drawn), fontsize=8, frameon=False)
    fig.tight_layout()
    return core._save_or_show(fig, save_path=save_path, show=show)
