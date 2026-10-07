"""
Platform checkpoint-namespace factory.

Reuses the incumbent :class:`~trading_crab_lib.checkpoints.CheckpointManager`
verbatim (D-01: never subclass or reimplement save/load/is_fresh) pointed at a
separate directory, ``data/checkpoints/platform/``. This keeps the new monthly
data layer's checkpoints (splices, ALFRED vintages, monthly features) fully
isolated from the frozen incumbent quarterly pipeline's checkpoint namespace.

Usage:
    from trading_crab_lib.platform.checkpoints import get_platform_checkpoint_manager

    cm = get_platform_checkpoint_manager()
    cm.save(df, "monthly_raw_daily")
    df = cm.load("monthly_raw_daily")
"""

from __future__ import annotations

from trading_crab_lib import DATA_DIR, OUTPUT_DIR, ROOT
from trading_crab_lib.checkpoints import CheckpointManager

#: The repo root and the two generated-state trees, exposed so ``platform/state.py`` (archive / reset /
#: restore / promote) needs no import of the legacy package root (the legacy-import ratchet).
PLATFORM_ROOT_DIR = ROOT
PLATFORM_DATA_DIR = DATA_DIR
PLATFORM_OUTPUT_DIR = OUTPUT_DIR

PLATFORM_CHECKPOINT_DIR = DATA_DIR / "checkpoints" / "platform"

#: Where the platform's report artifacts live (the tracked backtest outputs among them). Exposed
#: here so platform modules that only READ them (``report/scoreboard.py``) need no import of
#: the legacy package root (the legacy-import ratchet, criterion 8).
PLATFORM_REPORT_DIR = OUTPUT_DIR / "reports" / "platform"


def get_platform_checkpoint_manager() -> CheckpointManager:
    """Return a :class:`CheckpointManager` scoped to the platform checkpoint namespace."""
    return CheckpointManager(checkpoint_dir=PLATFORM_CHECKPOINT_DIR)
