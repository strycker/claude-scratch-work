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
# §5.4 detection-lag / sojourn-ratio headline.
#
# UPDATED 2026-09-14 (Phase 7, D-01/D-02-A/D-08, ADR-0001): audit item A13
# found that the "smoothed reference" and the walk-forward's "filtered" path
# were computed from DIFFERENT and time-varying feature sets, so their
# disagreement was not purely detection delay. That cause is now FIXED, not
# merely reworded: Phase 7's D-01 freezes the walk-forward driver's L1
# labeler to the SAME feature space `report.py::_reference_label_columns`
# already computes for the smoothed reference (D-02-A: ten columns, `oil`
# included). The caveat below is retired to a resolution narrative on the
# strength of D-08's two licensing artifacts TOGETHER — never because the
# wording was softened: (a) `TestFrozenPolicyEquivalence`
# (tests/unit/test_platform_backtest_driver.py), which proves the driver and
# the reference resolve to IDENTICAL column sets at every sampled decision
# date, and (b) the ratio is published together with its resolved-transition
# denominator (see the caption's own "N of M" line), so a reader can judge
# sample size directly. See `platform_design/adr/0001-l1-feature-policy.md`
# for the full policy record, including the pre-fix seven-change
# (4 -> 6 -> 8 -> 9 -> 10 -> 12 -> 13 features) history this caption used to
# carry — that history now belongs in the ADR, not in a caption describing
# the CURRENT, resolved state.
A13_CAVEAT: str = (
    "RESOLVED (audit item A13, Phase 7 D-01/D-02-A/D-08): this ratio compares "
    "the walk-forward's per-window 'filtered' labeling against a fixed-feature "
    "'smoothed reference' labeling, both now fit on ONE shared, frozen feature "
    "space. The driver's L1 labeler and the evaluation's smoothed reference "
    "resolve their columns from a single shared computation, proved identical "
    "at every sampled decision date by TestFrozenPolicyEquivalence. The ratio "
    "is published together with its resolved-transition denominator (the "
    "'N resolved of M transitions' line above), so a reader can judge whether "
    "it rests on a handful of transitions or on many. It remains a "
    "small-sample INDICATIVE number, never a robust go/no-go figure on its "
    "own — see platform_design/adr/0001-l1-feature-policy.md for the full "
    "policy record."
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
