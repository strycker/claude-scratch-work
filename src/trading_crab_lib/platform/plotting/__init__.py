"""
platform/plotting — Visualization spine for the platform notebook suite
(Phase 6, NB-01).

Re-exports ONLY the shared constants and loaders from :mod:`core` and
:mod:`loaders`. Per-layer plot functions (``data.py``, and the ``features``,
``regime``, ``nowcaster``, ``allocation``, ``backtest`` submodules added by
later plans) are deliberately NOT barrelled here — notebooks and tests import
them by submodule path, e.g.::

    from trading_crab_lib.platform.plotting import data as pdata
    pdata.plot_coverage_timeline(monthly_raw)

This diverges from the legacy barrel in
``src/trading_crab_lib/plotting/__init__.py``, which re-exports every plot
function from every submodule. The reason is concurrency: five downstream
plans (P2-P6) each add one plotting submodule in parallel, and if they all
had to also edit this one file to add their own exports, every plan would
contend for the same lines of the same file. Importing by submodule path
means each plan only ever touches its own new file.

Fresh-package boundary (D-01): nothing under ``platform/plotting/`` imports
from the legacy ``trading_crab_lib.plotting`` package — that package is a
pattern source only.
"""

from __future__ import annotations

from trading_crab_lib.platform.plotting.core import (
    A13_CAVEAT,
    CUSTOM_COLORS,
    PLATFORM_PLOT_DIR,
    REGIME_CMAP,
)
from trading_crab_lib.platform.plotting.loaders import (
    NOTEBOOK_SCRATCH_DIR,
    compute_regime_labeling,
    load_filtered_state_probs,
    load_full_sample_states,
    load_full_span_checkpoint,
    load_platform_checkpoint,
    load_report_artifact,
    redacted_config,
)

__all__ = [
    "A13_CAVEAT",
    "CUSTOM_COLORS",
    "PLATFORM_PLOT_DIR",
    "REGIME_CMAP",
    "NOTEBOOK_SCRATCH_DIR",
    "compute_regime_labeling",
    "load_filtered_state_probs",
    "load_full_sample_states",
    "load_full_span_checkpoint",
    "load_platform_checkpoint",
    "load_report_artifact",
    "redacted_config",
]
