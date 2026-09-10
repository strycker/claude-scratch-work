"""
platform/plotting/history.py — Economic-history overlay for P3 (Phase 6, D-12/D-13).

Two independent history sources, deliberately kept separate:

* :data:`ECONOMIC_EVENTS` — a plain module constant holding six well-known
  dated ranges (oil shock, Volcker, 1987, LTCM, dot-com, GFC). D-12 forbids a
  dedicated events file, a per-entry provenance schema, and any migration
  obligation for six well-known date ranges, so this stays a module constant
  and this docstring says so on purpose.
* :func:`load_usrec_or_warn` — NBER recession months from FRED's ``USREC``
  series. This is the phase's only outbound network call. Per D-10 its
  failure path is loud and actionable rather than silent: the caller gets
  ``None`` and a WARNING naming the failure, the consequence, and the fix.

Keeping the two separate is itself a mitigation (T-06-10): a substituted or
malformed USREC response produces shading that visibly contradicts the
independent event overlay rather than silently redefining history.

D-13's contingency tables (:func:`regime_era_contingency` and
:func:`regime_era_marginals`) are purely descriptive — both directions of the
same crosstab, and no significance test. A p-value here would invite treating
"the regimes are real" as a passed test, and any number that informs a choice
belongs in the trial registry.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from trading_crab_lib.platform.plotting.core import _save_or_show

log = logging.getLogger(__name__)

# ── D-12: the dated event list, as a module constant (no events file) ────────
#
# (ISO start, ISO end, label). Six well-known ranges; start is strictly before
# end in every entry. These are month-end aligned so they overlay cleanly on a
# monthly regime timeline.
ECONOMIC_EVENTS: tuple[tuple[str, str, str], ...] = (
    ("1973-10-31", "1974-12-31", "1973 oil shock"),
    ("1979-10-31", "1982-11-30", "Volcker disinflation"),
    ("1987-10-31", "1987-12-31", "1987 crash"),
    ("1998-08-31", "1998-10-31", "LTCM and Russia default"),
    ("2000-03-31", "2002-10-31", "dot-com bust"),
    ("2007-12-31", "2009-06-30", "global financial crisis"),
)

#: Row label used for the NBER recession era when a USREC series is supplied.
RECESSION_ERA_LABEL: str = "NBER recession"

#: Row label for the unconditional baseline era (every month in the input).
BASELINE_ERA_LABEL: str = "all months"

_USREC_SERIES_ID = "USREC"


def load_usrec_or_warn(
    cfg: dict[str, Any], *, start: str = "1962-01-01", end: str | None = None
) -> pd.Series | None:
    """Fetch the FRED ``USREC`` recession indicator, or warn loudly and return ``None``.

    Reuses the platform's existing FRED credential path
    (``cfg["fred_monthly"]["api_key"]``, injected by
    :func:`~trading_crab_lib.platform.config.load_platform_config`) — no new
    secret is introduced.

    The response is treated strictly as data: it is coerced to numeric,
    clipped to ``{0, 1}``, and never evaluated (T-06-10).

    Args:
        cfg: platform config dict.
        start: ISO observation start date.
        end: ISO observation end date; defaults to today.

    Returns:
        pd.Series | None: month-end-indexed 0/1 integer Series, or ``None``
        when the fetch fails for any reason (including a missing API key).
        Failure is logged once at WARNING naming what broke, that recession
        shading is omitted while the :data:`ECONOMIC_EVENTS` overlay still
        renders, and what to check (D-10).
    """
    try:
        from fredapi import Fred  # imported lazily: optional [ingestion] extra

        api_key = (cfg.get("fred_monthly") or {}).get("api_key")
        if not api_key:
            raise OSError("FRED_API_KEY is not set")

        fred = Fred(api_key=api_key)
        raw = fred.get_series(
            _USREC_SERIES_ID,
            observation_start=start,
            observation_end=end or str(date.today()),
        )
        # Parse strictly as data — coerce, clip, cast. Never eval, never exec.
        monthly = pd.Series(raw).resample("ME").last()
        monthly = pd.to_numeric(monthly, errors="coerce").fillna(0.0).clip(0.0, 1.0).round()
        monthly = monthly.astype(int)
        monthly.name = "usrec"
        log.info("Fetched FRED %s: %d months %s -> %s", _USREC_SERIES_ID, len(monthly),
                 monthly.index.min(), monthly.index.max())
        return monthly
    except Exception as exc:  # noqa: BLE001 — fredapi/network raise many types; degrade loudly
        log.warning(
            "FRED %s fetch FAILED (%s: %s). CONSEQUENCE: NBER recession shading is OMITTED from "
            "the regime timeline; the dated ECONOMIC_EVENTS overlay still renders, so the chart "
            "is incomplete but not blank. CHECK: FRED_API_KEY is set in .env, and that "
            "api.stlouisfed.org is reachable from this machine.",
            _USREC_SERIES_ID,
            type(exc).__name__,
            exc,
        )
        return None


def recession_periods(usrec: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Collapse runs of 1 in a USREC-shaped Series into ``(start, end)`` pairs.

    Args:
        usrec: 0/1 indicator Series indexed by date. NaNs are treated as 0.

    Returns:
        list of ``(start, end)`` timestamps, one per contiguous block of 1s.
        Empty list for an empty or all-zero input.
    """
    if usrec is None or len(usrec) == 0:
        return []

    flags = pd.to_numeric(usrec, errors="coerce").fillna(0.0) > 0.5
    periods: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    run_start: pd.Timestamp | None = None
    prev: pd.Timestamp | None = None

    for stamp, flag in flags.items():
        if flag and run_start is None:
            run_start = stamp
        elif not flag and run_start is not None:
            periods.append((run_start, prev))
            run_start = None
        prev = stamp

    if run_start is not None and prev is not None:
        periods.append((run_start, prev))
    return periods


def _era_masks(
    index: pd.Index,
    *,
    usrec: pd.Series | None,
    events: tuple[tuple[str, str, str], ...],
) -> dict[str, np.ndarray]:
    """Boolean membership mask per era, in display order (events, recession, baseline)."""
    stamps = pd.DatetimeIndex(index)
    masks: dict[str, np.ndarray] = {}

    for start, end, label in events:
        masks[label] = np.asarray((stamps >= pd.Timestamp(start)) & (stamps <= pd.Timestamp(end)))

    if usrec is not None and len(usrec) > 0:
        flags = pd.to_numeric(usrec, errors="coerce").fillna(0.0) > 0.5
        aligned = flags.reindex(stamps).fillna(False).to_numpy(dtype=bool)
        masks[RECESSION_ERA_LABEL] = aligned

    masks[BASELINE_ERA_LABEL] = np.ones(len(stamps), dtype=bool)
    return masks


def regime_era_contingency(
    states: pd.Series,
    *,
    usrec: pd.Series | None = None,
    events: tuple[tuple[str, str, str], ...] = ECONOMIC_EVENTS,
    n_states: int = 5,
) -> pd.DataFrame:
    """Share of each era's months falling in each state (D-13, era-conditional reading).

    Rows are eras — one per entry in *events*, plus a ``NBER recession`` row
    when *usrec* is supplied, plus an ``all months`` baseline row. Columns are
    state ids ``0 .. n_states - 1``. Every row sums to 1.0.

    Eras with no overlapping months are omitted rather than emitted as an
    all-zero row, so the "rows sum to 1.0" invariant holds unconditionally.

    This is descriptive only. No significance test is computed (D-13).
    """
    if states is None or len(states) == 0:
        return pd.DataFrame(columns=list(range(n_states)))

    values = pd.to_numeric(states, errors="coerce").to_numpy()
    masks = _era_masks(states.index, usrec=usrec, events=events)

    rows: dict[str, list[float]] = {}
    for label, mask in masks.items():
        era_values = values[mask]
        era_values = era_values[~np.isnan(era_values)]
        if era_values.size == 0:
            continue
        rows[label] = [float((era_values == k).sum()) / era_values.size for k in range(n_states)]

    return pd.DataFrame.from_dict(rows, orient="index", columns=list(range(n_states)))


def regime_era_marginals(
    states: pd.Series,
    *,
    usrec: pd.Series | None = None,
    events: tuple[tuple[str, str, str], ...] = ECONOMIC_EVENTS,
    n_states: int = 5,
) -> pd.DataFrame:
    """Share of each state's months falling inside each era (D-13, the other direction).

    Same row/column layout as :func:`regime_era_contingency`, but each value is
    ``months of state k inside era e / total months of state k``. Rows do NOT
    sum to 1.0 and columns need not either — the named eras overlap the NBER
    recession row by construction.

    D-13 asks for both readings because one alone invites seeing a pattern that
    is not there: a state can dominate a short era simply by being the most
    common state overall, which only the marginal view exposes.
    """
    if states is None or len(states) == 0:
        return pd.DataFrame(columns=list(range(n_states)))

    values = pd.to_numeric(states, errors="coerce").to_numpy()
    masks = _era_masks(states.index, usrec=usrec, events=events)
    state_totals = {k: float((values == k).sum()) for k in range(n_states)}

    rows: dict[str, list[float]] = {}
    for label, mask in masks.items():
        era_values = values[mask]
        era_values = era_values[~np.isnan(era_values)]
        if era_values.size == 0:
            continue
        rows[label] = [
            (float((era_values == k).sum()) / state_totals[k]) if state_totals[k] > 0 else 0.0
            for k in range(n_states)
        ]

    return pd.DataFrame.from_dict(rows, orient="index", columns=list(range(n_states)))


def plot_era_contingency(
    contingency: pd.DataFrame,
    *,
    title: str = "Regime x era contingency",
    save_path: Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """Render a contingency/marginal frame as an annotated heatmap.

    An empty frame does not raise — it returns a figure carrying a "no data"
    annotation, matching :func:`~trading_crab_lib.platform.plotting.data.plot_coverage_timeline`.
    """
    if contingency is None or contingency.empty:
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        ax.set_title(title)
        return _save_or_show(fig, save_path=save_path, show=show)

    grid = contingency.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(1.4 * grid.shape[1] + 4, 0.55 * grid.shape[0] + 2))
    image = ax.imshow(grid, aspect="auto", cmap="viridis", vmin=0.0, vmax=max(1e-9, float(np.nanmax(grid))))

    ax.set_xticks(np.arange(grid.shape[1]))
    ax.set_xticklabels([f"state {c}" for c in contingency.columns])
    ax.set_yticks(np.arange(grid.shape[0]))
    ax.set_yticklabels(list(contingency.index), fontsize=8)

    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            value = grid[i, j]
            ax.text(
                j,
                i,
                "—" if np.isnan(value) else f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if (not np.isnan(value) and value > 0.5) else "black",
                fontsize=8,
            )

    fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    ax.set_title(title)
    fig.tight_layout()
    return _save_or_show(fig, save_path=save_path, show=show)
