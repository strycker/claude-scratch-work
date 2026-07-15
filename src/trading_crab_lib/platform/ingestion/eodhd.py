"""
EODHD adapter seam — v2 placeholder (DATA-06).

EODHD offers a REST API covering 60+ exchanges and 150K+ tickers, with
fundamentals/options/news add-ons on paid plans. This module reserves the
import seam for a future EODHD adapter; it performs no I/O and adds no
runtime dependency.

See docs/paid_provider_seams.md for the full provider comparison, the
usage pattern once implemented, and integration notes. Do not implement a
live adapter here without updating that document first.
"""

from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger(__name__)


def fetch_prices(cfg: dict) -> pd.DataFrame:
    """Placeholder entry point for the EODHD adapter seam.

    Args:
        cfg: Platform config dict (unused — no implementation exists yet).

    Raises:
        NotImplementedError: Always. EODHD is a documented v2 seam with
            no live integration in v1. See docs/paid_provider_seams.md.
    """
    raise NotImplementedError(
        "EODHD is a documented v2 adapter seam with no implementation in "
        "v1. See docs/paid_provider_seams.md for provider details and the "
        "adoption plan before building a live integration."
    )
