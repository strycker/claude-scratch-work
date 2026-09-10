"""
platform/plotting/data.py — P1 data-spine plots (Phase 6).

Analogous in role to the legacy ``trading_crab_lib/plotting/ingestion.py``
(coverage plots for raw data), rebuilt against the platform's monthly-only
shape and the D-02 no-``RunConfig`` signature.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from trading_crab_lib.platform.plotting.core import _save_or_show

log = logging.getLogger(__name__)


def plot_coverage_timeline(
    df: pd.DataFrame,
    *,
    title: str = "Column coverage",
    max_columns: int = 60,
    save_path: Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """Render a binary column-availability grid over *df*'s DatetimeIndex.

    Rows (one per column, truncated to *max_columns*) are sorted by
    first-valid date and annotated with the first-valid year, so an operator
    can see at a glance when each series starts and whether any series stops
    updating before the frame's last date.

    An empty DataFrame does not raise — it returns a figure carrying a
    "no data" annotation.
    """
    if df.empty or df.shape[1] == 0:
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        ax.set_title(title)
        return _save_or_show(fig, save_path=save_path, show=show)

    columns = list(df.columns)[:max_columns]
    sub = df[columns]
    first_valid = {col: sub[col].first_valid_index() for col in columns}
    ordered = sorted(
        columns,
        key=lambda c: (first_valid[c] is None, first_valid[c] if first_valid[c] is not None else pd.Timestamp.max),
    )

    grid = np.array([sub[col].notna().to_numpy(dtype=float) for col in ordered])

    fig, ax = plt.subplots(figsize=(12, max(3, 0.3 * len(ordered))))
    ax.imshow(grid, aspect="auto", cmap="Greens", interpolation="nearest", vmin=0, vmax=1)

    ax.set_yticks(np.arange(len(ordered)))
    row_labels = []
    for col in ordered:
        fv = first_valid[col]
        row_labels.append(f"{col} ({fv.year})" if fv is not None else f"{col} (never)")
    ax.set_yticklabels(row_labels, fontsize=6)

    n = len(sub.index)
    n_ticks = min(10, n)
    if n_ticks > 0:
        tick_positions = np.linspace(0, n - 1, n_ticks).astype(int)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(
            [sub.index[i].strftime("%Y-%m") for i in tick_positions],
            rotation=45,
            ha="right",
        )
    ax.set_title(title)
    fig.tight_layout()
    return _save_or_show(fig, save_path=save_path, show=show)
