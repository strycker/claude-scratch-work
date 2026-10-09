"""The config the tracked 8.3 record was built under (helper for tests, not collected).

On 2026-10-09 gold moved from macrotrends ``gold_spot`` (1985-02+, sampled near month-end) to
the World Bank monthly average ``gold_wb`` (1960+), with a month-end P&L overlay on IAU from
2005-02. The tracked ``data/`` and ``outputs/`` were built before that, so a test that re-derives
the tracked record must read it under the gold source it was built with. Phase 08.5 takes the
tracked data out of git; this helper goes with it.
"""

from __future__ import annotations

import copy
from typing import Any

_GOLD_SPOT_FETCH = {"name": "gold_spot", "path": "/1333/historical-gold-prices-100-year-chart", "resample": "mean"}


def tracked_record_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    """``cfg`` with only the gold source rolled back to the 8.3 record's; ``cfg`` is not edited."""
    cfg = copy.deepcopy(cfg)
    cfg.pop("worldbank_monthly", None)
    cfg["macrotrends_monthly"]["series"].insert(0, dict(_GOLD_SPOT_FETCH))
    lags = cfg["publication_lags"]
    lags.pop("gold_wb")
    lags["gold_spot"] = 0
    cfg["splice"]["gold"]["source_col"] = ["gold_spot", "IAU"]
    cfg["pnl_splice"].pop("gold")
    return cfg
