"""
Norgate Data adapter seam — v2 placeholder (DATA-06).

Norgate ships survivorship-bias-free US equities/ETF end-of-day data as a
subscription desktop database product (not a typical REST API). This module
reserves the import seam for a future Norgate adapter; it performs no I/O
and adds no runtime dependency.

See docs/paid_provider_seams.md for the full provider comparison, the
usage pattern once implemented, and integration notes. Do not implement a
live adapter here without updating that document first.
"""

from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger(__name__)


def fetch_prices(cfg: dict) -> pd.DataFrame:
    """Placeholder entry point for the Norgate Data adapter seam.

    Args:
        cfg: Platform config dict (unused — no implementation exists yet).

    Raises:
        NotImplementedError: Always. Norgate is a documented v2 seam with
            no live integration in v1. See docs/paid_provider_seams.md.
    """
    raise NotImplementedError(
        "Norgate Data is a documented v2 adapter seam with no implementation "
        "in v1. See docs/paid_provider_seams.md for provider details and the "
        "adoption plan before building a live integration."
    )
