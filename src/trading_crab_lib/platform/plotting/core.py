"""
platform/plotting/core.py — Shared save/show, palette, and regime-coloring
helpers for the platform plotting spine (Phase 6, D-02/D-04).

Fresh-package boundary (D-01): this module imports nothing from the legacy
``trading_crab_lib.plotting`` package. The Agg-backend guard, palette, and
``_save_or_show`` idiom are copied from ``trading_crab_lib/plotting/core.py``
because they have zero platform-specific coupling — only the import is
forbidden, not the pattern.

Signature convention (D-02): ``platform/config.py`` returns a plain ``dict``
with no ``RunConfig``-shaped object to bind to, so every plot function in
every platform plotting submodule follows::

    def plot_x(data, *, save_path: Path | None = None, show: bool = False) -> plt.Figure:
        ...
        return _save_or_show(fig, save_path=save_path, show=show)

No run-configuration object is invented, and the legacy staleness-cache
helpers (``_plot_is_fresh``, ``load_or_generate``, ``list_available_plots``)
are deliberately not ported — they exist to skip re-running expensive
``RunConfig``-driven pipeline steps and have no platform equivalent.
"""

from __future__ import annotations

import logging
from pathlib import Path

from trading_crab_lib import OUTPUT_DIR

log = logging.getLogger(__name__)


def _in_jupyter() -> bool:
    try:
        from IPython import get_ipython  # type: ignore[import]

        return get_ipython() is not None
    except ImportError:
        return False


try:
    import matplotlib

    # Only force the Agg (headless) backend when NOT running inside
    # Jupyter/IPython. In Jupyter, %matplotlib inline has already configured
    # the inline backend, and calling matplotlib.use("Agg") afterward would
    # break inline display.
    if not _in_jupyter():
        matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
except ImportError as _matplotlib_err:
    raise ImportError(
        "matplotlib is required for platform plotting functions. "
        "Install with: pip install 'trading-crab-lib[plotting]'"
    ) from _matplotlib_err


# ── Color palette (five states — matches platform's configured labeling.K) ────
CUSTOM_COLORS: list[str] = ["#0000d0", "#d00000", "#f48c06", "#8338ec", "#50a000"]
REGIME_CMAP = mcolors.ListedColormap(CUSTOM_COLORS)

PLATFORM_PLOT_DIR: Path = OUTPUT_DIR / "plots" / "platform"

# Single-source caveat string for every module and notebook that displays the
# §5.4 detection-lag / sojourn-ratio headline. Audit item A13 found that the
# "smoothed reference" and the walk-forward's "filtered" path are computed
# from different and time-varying feature sets, so their disagreement is not
# purely detection delay — the ratio is not interpretable until A13 is
# resolved, and no plausibility band around it changes that (06-VALIDATION.md).
A13_CAVEAT: str = (
    "NOT INTERPRETABLE (audit item A13): this ratio compares a fixed-feature "
    "'smoothed reference' labeling against the walk-forward's own per-window "
    "'filtered' labeling, whose active feature set changes 7 times across the "
    "backtest (4 -> 6 -> 8 -> 9 -> 10 -> 12 -> 13 features). Their disagreement "
    "is therefore not purely detection delay, and no plausibility band around "
    "this number resolves that until A13 is settled."
)


def _save_or_show(fig: plt.Figure, *, save_path: Path | None, show: bool) -> plt.Figure:
    """Finalize a figure per D-02: caller decides save/show explicitly.

    Unlike the legacy ``_save_or_show``, this never calls ``plt.close(fig)`` —
    the caller (a notebook cell, or a test asserting on the returned object)
    needs the live ``Figure`` back.
    """
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        log.info("Saved plot: %s", save_path)
    if show or _in_jupyter():
        plt.show()
    return fig


def _regime_color(state_id: int) -> str:
    """Return the palette color for *state_id*, wrapping modulo the palette length."""
    return CUSTOM_COLORS[state_id % len(CUSTOM_COLORS)]
