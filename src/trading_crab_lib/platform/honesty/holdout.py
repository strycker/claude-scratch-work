"""
Holdout carve — physical 2021+ boundary (HON-01/D13).

Development, tuning, and model-selection may use only rows dated on or before
``DEFAULT_HOLDOUT_CUTOFF``. Rows dated after the cutoff are written exclusively
to ``HOLDOUT_CHECKPOINT_DIR`` — a tree the default platform checkpoint manager
(:func:`~trading_crab_lib.platform.checkpoints.get_platform_checkpoint_manager`)
never reads. Live-scoring mode opts in explicitly via
:func:`get_holdout_checkpoint_manager`.

There is no fallback code path from the default (dev) manager to the holdout
tree: a checkpoint that exists only in ``data/holdout/`` raises
``FileNotFoundError`` when loaded through the default manager. This absence
of a fallback IS the guarantee (D-03) — the invariant test proves it by real
load, not by monkeypatching.
"""

from __future__ import annotations

import logging

import pandas as pd

from trading_crab_lib import DATA_DIR
from trading_crab_lib.checkpoints import CheckpointManager
from trading_crab_lib.platform.checkpoints import get_platform_checkpoint_manager

log = logging.getLogger(__name__)

HOLDOUT_CHECKPOINT_DIR = DATA_DIR / "holdout"
DEFAULT_HOLDOUT_CUTOFF = "2020-12-31"


def get_holdout_checkpoint_manager() -> CheckpointManager:
    """Return a :class:`CheckpointManager` scoped to the holdout checkpoint namespace.

    Never called by default code paths — live-scoring mode calls this explicitly.
    """
    return CheckpointManager(checkpoint_dir=HOLDOUT_CHECKPOINT_DIR)


def split_by_holdout_boundary(
    df: pd.DataFrame,
    cutoff: str = DEFAULT_HOLDOUT_CUTOFF,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split *df* into (dev_df, holdout_df) at *cutoff* on a DatetimeIndex.

    dev_df holds rows with index <= cutoff; holdout_df holds rows with index >
    cutoff. The two frames are disjoint and their row counts sum to len(df).
    """
    cutoff_ts = pd.Timestamp(cutoff)
    dev_df = df.loc[:cutoff_ts]
    holdout_df = df.loc[df.index > cutoff_ts]
    return dev_df, holdout_df


def write_monthly_features_split(
    df: pd.DataFrame,
    name: str = "monthly_features",
    cutoff: str = DEFAULT_HOLDOUT_CUTOFF,
) -> None:
    """Split *df* at *cutoff* and write each side to its own checkpoint tree.

    Dev rows (<= cutoff) go to the default platform checkpoint manager; holdout
    rows (> cutoff) go to the holdout checkpoint manager. Reloading *name* from
    either manager returns only that manager's side of the boundary.
    """
    dev_df, holdout_df = split_by_holdout_boundary(df, cutoff=cutoff)
    get_platform_checkpoint_manager().save(dev_df, name)
    get_holdout_checkpoint_manager().save(holdout_df, name)
    log.info(
        "write_monthly_features_split: %s -> dev=%d rows (<=%s), holdout=%d rows (>%s)",
        name, len(dev_df), cutoff, len(holdout_df), cutoff,
    )


def assert_dev_checkpoint_within_boundary(
    name: str,
    cutoff: str = DEFAULT_HOLDOUT_CUTOFF,
) -> None:
    """Raise loudly if the dev-namespace checkpoint *name* holds any post-cutoff row.

    Raises:
        RuntimeError: naming the offending max date if any row in the dev
            checkpoint is dated after *cutoff*.
    """
    cutoff_ts = pd.Timestamp(cutoff)
    df = get_platform_checkpoint_manager().load(name)
    if len(df) == 0:
        return
    max_date = df.index.max()
    if max_date > cutoff_ts:
        raise RuntimeError(
            f"Dev checkpoint '{name}' contains a row dated {max_date}, which is "
            f"after the holdout cutoff {cutoff_ts}. This violates HON-01: "
            "development must never see 2021+ data."
        )
