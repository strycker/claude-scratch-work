"""
plotting.py — Shared visualization helpers for all pipeline stages.

All plot functions:
  - Accept run_cfg: RunConfig and honour save_plots / show_plots
  - Save to outputs/plots/{step}_{description}.png when save_plots=True
  - Are importable by notebooks without side-effects

Custom 5-regime color palette (from legacy/unified_script.py):
    CUSTOM_COLORS = ["#0000d0","#d00000","#f48c06","#8338ec","#50a000"]

Usage:
    from trading_crab_lib import plotting
    from trading_crab_lib.runtime import RunConfig
    run_cfg = RunConfig(generate_plots=True, save_plots=True)
    plotting.plot_pca_scatter(pca_df, labels, regime_names, run_cfg)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from trading_crab_lib import OUTPUT_DIR
from trading_crab_lib import checkpoints as _checkpoints_mod
from trading_crab_lib.runtime import RunConfig


def _in_jupyter() -> bool:
    try:
        from IPython import get_ipython  # type: ignore[import]
        return get_ipython() is not None
    except ImportError:
        return False


try:
    import matplotlib
    # Only force the Agg (headless) backend when NOT running inside Jupyter/IPython.
    # In Jupyter, %matplotlib inline has already configured the inline backend and
    # calling matplotlib.use("Agg") after that would break inline display and cause
    # "FigureCanvasAgg is non-interactive" warnings when plt.show() is called.
    if not _in_jupyter():
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
except ImportError as _matplotlib_err:
    raise ImportError(
        "matplotlib is required for plotting functions. "
        "Install with: pip install 'trading-crab-lib[plotting]'"
    ) from _matplotlib_err

log = logging.getLogger(__name__)

# ── Color palette ──────────────────────────────────────────────────────────────
CUSTOM_COLORS: list[str] = ["#0000d0", "#d00000", "#f48c06", "#8338ec", "#50a000"]
REGIME_CMAP = mcolors.ListedColormap(CUSTOM_COLORS)

PLOT_DIR = OUTPUT_DIR / "plots"


def _save_or_show(fig: plt.Figure, filename: str, run_cfg: RunConfig) -> None:
    """Finalize a figure: save to disk and/or display according to run_cfg.

    In Jupyter notebooks, plt.show() is always called so the figure appears
    inline — regardless of show_plots — because the inline backend handles
    display cleanly and plt.close() would otherwise prevent any inline output.
    """
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    if run_cfg.save_plots:
        out = PLOT_DIR / filename
        fig.savefig(out, dpi=150, bbox_inches="tight")
        log.info("Saved plot: %s", out)
    if run_cfg.show_plots or _in_jupyter():
        plt.show()
    plt.close(fig)


def _regime_color(cluster_id: int) -> str:
    return CUSTOM_COLORS[cluster_id % len(CUSTOM_COLORS)]


# ── Plot Reuse Infrastructure ─────────────────────────────────────────────────


def _plot_is_fresh(filename: str, checkpoint_name: str | None = None) -> bool:
    """Return True if the PNG exists and is newer than the relevant checkpoint.

    If *checkpoint_name* is None, only checks that the PNG exists (always fresh).
    If a checkpoint name is given, compares the PNG's mtime against the checkpoint's
    meta.json ``created`` timestamp — stale if the checkpoint was regenerated after
    the plot was last saved.
    """
    png_path = PLOT_DIR / filename
    if not png_path.exists():
        return False

    if checkpoint_name is None:
        return True

    meta_path = _checkpoints_mod.CHECKPOINT_DIR / f"{checkpoint_name}.meta.json"
    if not meta_path.exists():
        # No checkpoint to compare against — treat PNG as fresh
        return True

    try:
        meta = json.loads(meta_path.read_text())
        ckpt_created = datetime.fromisoformat(meta["created"])
    except (json.JSONDecodeError, KeyError, ValueError):
        return True  # corrupt metadata — don't force regeneration

    png_mtime = datetime.fromtimestamp(png_path.stat().st_mtime)
    return png_mtime >= ckpt_created


def load_or_generate(
    plot_func: callable,
    *args: object,
    filename: str,
    run_cfg: RunConfig,
    checkpoint_name: str | None = None,
    **kwargs: object,
) -> None:
    """Show a cached PNG if fresh, otherwise call *plot_func* to regenerate.

    Designed for use in Jupyter notebooks to avoid re-running expensive plot
    functions when the underlying data hasn't changed.

    Args:
        plot_func: any ``plotting.plot_*`` function.
        *args: positional arguments forwarded to *plot_func*.
        filename: the PNG filename (e.g. ``"03_pca_scatter.png"``).
        run_cfg: runtime configuration.
        checkpoint_name: optional checkpoint name (e.g. ``"features"``) used
            to determine freshness.  If the checkpoint was regenerated after
            the PNG was last saved, the plot is considered stale.
        **kwargs: keyword arguments forwarded to *plot_func*.

    Example (in a Jupyter notebook)::

        from trading_crab_lib import plotting
        plotting.load_or_generate(
            plotting.plot_pca_scatter,
            pca_df, labels, regime_names,
            filename="03_pca_scatter.png",
            run_cfg=run_cfg,
            checkpoint_name="cluster_labels",
        )
    """
    if _plot_is_fresh(filename, checkpoint_name):
        png_path = PLOT_DIR / filename
        log.info("Plot fresh — loading cached: %s", png_path)
        if _in_jupyter():
            try:
                from IPython.display import Image, display  # type: ignore[import]
                display(Image(filename=str(png_path)))
            except ImportError:
                log.warning("IPython not available — regenerating plot")
                plot_func(*args, run_cfg=run_cfg, filename=filename, **kwargs)
        else:
            log.info("Cached plot available at: %s", png_path)
        return

    log.info("Plot stale or missing — regenerating: %s", filename)
    plot_func(*args, run_cfg=run_cfg, filename=filename, **kwargs)


def list_available_plots(plot_dir: Path | None = None) -> str:
    """Return a formatted table of all PNGs in the plot directory.

    Useful in notebooks and CLI to see what plots are available without
    opening a file browser.

    Returns:
        Human-readable table string with filename, modification time, and
        file size for each PNG found.  Returns a "no plots" message if the
        directory is empty or doesn't exist.
    """
    d = plot_dir or PLOT_DIR
    if not d.exists():
        return "No plots directory found."

    pngs = sorted(d.glob("*.png"))
    if not pngs:
        return f"No plots found in {d}"

    lines = [
        f"{'Filename':<50} {'Modified':<22} {'Size':>10}",
        "-" * 84,
    ]

    for p in pngs:
        stat = p.stat()
        mtime = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        size_kb = stat.st_size / 1024
        if size_kb >= 1024:
            size_str = f"{size_kb / 1024:.1f} MB"
        else:
            size_str = f"{size_kb:.1f} KB"
        lines.append(f"{p.name:<50} {mtime:<22} {size_str:>10}")

    lines.append(f"\nTotal: {len(pngs)} plots")
    return "\n".join(lines)
