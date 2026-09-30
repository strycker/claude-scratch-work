"""
Publication lags — when each raw monthly series could first have been known (D-01..D-03).

A value for reference month ``m`` is only usable by a decision taken at the end of
month ``m + L``, where ``L`` is that series' measured publication lag. The table lives
in ``cfg['publication_lags']`` (``config/platform_settings.yaml``); this module
validates it and applies it, **once**, to the monthly ingest frame inside
``transforms_monthly.build_monthly_spine`` — so every downstream consumer (the
research-series splice, ``monthly_raw``, ``monthly_features``, labels, backtests,
serving) sees the lagged series and nothing re-derives an unlagged copy.

Entry forms:

* ``int >= 0`` — shift the column's timestamps forward by that many month-ends.
* ``derived`` — a splice research series (``equities_tr``, ``gold``, ...). It has no
  lag of its own: it inherits the lag of its inputs, which are lagged *before* the
  splice. Shifting it again would double-lag its lagged input and misalign its
  un-lagged one.
* ``{vintage: true, fallback_months: N}`` — an ALFRED vintage-aligned agency series
  (D-02). ``value_as_of`` already makes it point-in-time; ``fallback_months`` is the
  lag applied only before the series' earliest recorded vintage
  (``transforms_monthly._shift_fallback_series``).

``apply_publication_lags`` refuses a derived or vintage column and any column missing
from the table, so a column can be neither lagged twice nor silently left unlagged.

The marker (``publication_lags.json`` next to ``monthly_raw``) records the table a
``monthly_raw`` checkpoint was built under. ``monthly_raw`` is merge-on-save, and
merging a lagged frame onto an unlagged disk copy would refill the first ``L`` rows of
every lagged column with their unlagged values — so the build refuses to merge onto a
checkpoint whose marker is missing or different.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)

DERIVED = "derived"
LAG_MARKER_FILENAME = "publication_lags.json"

LagEntry = int | str | dict[str, Any]


# ── Table validation ────────────────────────────────────────────────────────


def _validate_entry(name: str, entry: Any) -> LagEntry:
    # bool is an int subclass — `true` is not a lag.
    if isinstance(entry, int) and not isinstance(entry, bool):
        if entry < 0:
            raise ValueError(f"publication_lags.{name}: lag must be >= 0, got {entry}")
        return entry
    if entry == DERIVED:
        return DERIVED
    if isinstance(entry, dict):
        fallback = entry.get("fallback_months")
        if (
            entry.get("vintage") is True
            and set(entry) == {"vintage", "fallback_months"}
            and isinstance(fallback, int)
            and not isinstance(fallback, bool)
            and fallback >= 1
        ):
            return {"vintage": True, "fallback_months": fallback}
    raise ValueError(
        f"publication_lags.{name}: invalid entry {entry!r}. Expected an int >= 0, "
        f"'{DERIVED}', or {{vintage: true, fallback_months: int >= 1}}."
    )


def lag_table(cfg: dict[str, Any]) -> dict[str, LagEntry]:
    """Validated ``cfg['publication_lags']``.

    Raises:
        ValueError: naming the column of any malformed entry, or if any
            ``fred_monthly.series`` entry still carries a truthy ``shift`` (a lag
            there would be applied on top of this table's).
    """
    shifted = [
        meta.get("name", series_id)
        for series_id, meta in cfg.get("fred_monthly", {}).get("series", {}).items()
        if meta.get("shift")
    ]
    if shifted:
        raise ValueError(
            f"fred_monthly.series has shift: true for {shifted}. Publication lags live only in "
            "`publication_lags`; remove the shift flag so the lag is not applied twice."
        )
    raw = cfg.get("publication_lags") or {}
    return {str(name): _validate_entry(str(name), entry) for name, entry in raw.items()}


# ── Application ─────────────────────────────────────────────────────────────


def apply_publication_lags(frame: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Return a copy of *frame* with every column moved forward by its lag.

    Each column is shifted with ``shift(L, freq=<monthly_freq>)`` — timestamps move,
    not positions, so gaps in the source are handled correctly. Values shifted past
    the frame's last timestamp extend the index; the caller reindexes onto its
    canonical spine. *frame* is never mutated.

    Raises:
        ValueError: for a column not in the table (the runtime completeness check),
            and for a ``derived`` or vintage column (those must never be lagged here).
    """
    table = lag_table(cfg)
    freq = cfg.get("data", {}).get("monthly_freq", "ME")

    unlisted = [c for c in frame.columns if c not in table]
    if unlisted:
        raise ValueError(
            f"apply_publication_lags: no publication_lags entry for {unlisted}. Every ingested "
            "column needs a measured lag in config/platform_settings.yaml before it can enter "
            "monthly_raw."
        )
    not_lag = [c for c in frame.columns if not isinstance(table[c], int)]
    if not_lag:
        raise ValueError(
            f"apply_publication_lags: {not_lag} are derived or vintage-aligned and must not "
            "pass through the lag (derived series inherit their inputs' lags; ALFRED series "
            "are point-in-time by vintage)."
        )

    shifted = {c: frame[c].shift(table[c], freq=freq) if table[c] else frame[c].copy() for c in frame.columns}
    if not shifted:
        return frame.copy()
    out = pd.concat(shifted, axis=1, sort=True)
    out.index.name = frame.index.name
    lagged = {c: table[c] for c in frame.columns if table[c]}
    log.info("apply_publication_lags: lagged %d of %d columns %s", len(lagged), len(frame.columns), lagged)
    return out


# ── Marker ──────────────────────────────────────────────────────────────────


def _marker_payload(cfg: dict[str, Any]) -> dict[str, LagEntry]:
    # JSON round-trip so an in-memory table compares equal to one read from disk.
    return json.loads(json.dumps(lag_table(cfg), sort_keys=True))


def write_lag_marker(path: Path, cfg: dict[str, Any]) -> Path:
    """Write the validated lag table to *path* (normally ``cm.dir / LAG_MARKER_FILENAME``)."""
    path = Path(path)
    payload = {
        "publication_lags": _marker_payload(cfg),
        "written_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def lag_marker_matches(path: Path, cfg: dict[str, Any]) -> bool:
    """True iff the marker at *path* exists and records exactly ``lag_table(cfg)``."""
    path = Path(path)
    if not path.is_file():
        return False
    try:
        recorded = json.loads(path.read_text(encoding="utf-8")).get("publication_lags")
    except (json.JSONDecodeError, AttributeError) as exc:
        log.warning("lag_marker_matches: unreadable marker %s: %s", path, exc)
        return False
    return recorded == _marker_payload(cfg)
