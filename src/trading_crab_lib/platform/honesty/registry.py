"""
Trial registry — append-only JSONL ledger (HON-02/D-01/D-02).

Every evaluated configuration is logged as one immutable JSON line: the
ledger is the multiple-testing denominator for the deflated Sharpe at
design freeze (design §8.4/§22). It is committed to git — never written
under gitignored ``data/`` — so git history is the tamper-evidence layer.

``append_trial()`` opens the ledger in append mode only, never write/truncate
mode: rewriting or truncating the file would destroy pre-registration
records and invalidate the freeze evaluation.

Usage:
    from trading_crab_lib.platform.honesty.registry import append_trial, read_trials

    row = append_trial(config={"model": "rf"}, features=["a", "b"], metrics={"sharpe": 0.4})
    trials = read_trials()  # DataFrame, one row per logged trial
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import pandas as pd

from trading_crab_lib import ROOT

log = logging.getLogger(__name__)

DEFAULT_REGISTRY_PATH = ROOT / "registry" / "trials.jsonl"


def _resolve_registry_path(path: Path | str | None) -> Path:
    """Return *path* as a :class:`Path`, or :data:`DEFAULT_REGISTRY_PATH` if None."""
    return Path(path) if path is not None else DEFAULT_REGISTRY_PATH


def config_hash(config: dict[str, Any]) -> str:
    """Deterministic 12-char hex hash of *config* (order-independent)."""
    return hashlib.md5(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:12]


def _git_sha() -> str:
    """Return the current git HEAD SHA, or "unknown" if git is unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        log.warning("Could not determine git SHA for registry row: %s", exc)
        return "unknown"


#: Sentinel for ``append_trial(path=...)`` meaning "this is NOT a trial — build the
#: row but do not write it." Use it for smoke tests, wiring-verification runs, and any
#: invocation whose purpose is to check that code runs rather than to evaluate a
#: configuration. Phase 7 wave 1 appended four untagged rows from exactly such runs;
#: because D-16 deflates Sharpe over the whole registry, they inflated the trial count
#: against which later searches are deflated. A run that could never have been selected
#: on is not a trial and must not be logged as one.
NO_REGISTRY: Final[str] = "__no_registry__"


def append_trial(
    *,
    config: dict[str, Any],
    features: list[str],
    metrics: dict[str, Any],
    path: Path | str | None = None,
) -> dict[str, Any]:
    """Append one evaluated configuration to the ledger and return the row written.

    Opens the ledger in append ("a") mode only — never truncates or rewrites
    existing lines (D-01).

    Every persisted row MUST carry a non-empty ``config["trial_tag"]`` naming what was
    evaluated. An untagged row is unattributable after the fact: the ledger cannot say
    whether it was a real evaluated configuration or an incidental run, and D-16 counts
    it either way. Callers that are not evaluating anything pass ``path=NO_REGISTRY``.

    Args:
        config: the evaluated configuration. MUST contain a non-empty string
            ``trial_tag``.
        features: feature columns the trial used.
        metrics: the trial's measured outcome.
        path: ledger path, ``None`` for the default, or :data:`NO_REGISTRY` to build the
            row and skip the write entirely (smoke / wiring-verification runs).

    Returns:
        dict[str, Any]: the row (written, unless ``path`` is :data:`NO_REGISTRY`).

    Raises:
        ValueError: if ``config["trial_tag"]`` is missing or blank and ``path`` is not
            :data:`NO_REGISTRY`.
    """
    if path == NO_REGISTRY:
        log.info(
            "Registry append SKIPPED (NO_REGISTRY): this run is not an evaluated "
            "configuration and must not count toward D-16's trial total."
        )
        return {
            "config_hash": config_hash(config),
            "config": config,
            "features": features,
            "metrics": metrics,
            "git_sha": _git_sha(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "written": False,
        }

    tag = config.get("trial_tag") if isinstance(config, dict) else None
    if not isinstance(tag, str) or not tag.strip():
        raise ValueError(
            "append_trial requires a non-empty config['trial_tag'] naming what was "
            "evaluated — an untagged row cannot be attributed later, and D-16 deflates "
            "Sharpe over every row in the ledger. If this run is a smoke test or a "
            "wiring check rather than an evaluated configuration, pass "
            "path=NO_REGISTRY instead of logging it as a trial."
        )

    row = {
        "config_hash": config_hash(config),
        "config": config,
        "features": features,
        "metrics": metrics,
        "git_sha": _git_sha(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    registry_path = _resolve_registry_path(path)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with registry_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")
    return row


def read_trials(path: Path | str | None = None) -> pd.DataFrame:
    """Read the ledger back as a DataFrame, one row per logged trial.

    Returns an empty DataFrame (does not raise) if the ledger is missing or empty.
    """
    registry_path = _resolve_registry_path(path)
    if not registry_path.exists() or registry_path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_json(registry_path, lines=True)
