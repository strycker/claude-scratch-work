"""
platform/plotting/loaders.py — Checkpoint- and report-artifact loaders for
the platform plotting spine and its notebooks (Phase 6, D-06/D-10).

Every loader here follows the same "load-or-raise-actionably" contract
(D-10): a missing artifact stops the caller with a message naming the
checkpoint/artifact, the directory it should live in, and the exact command
that rebuilds it — never a bare ``FileNotFoundError``. Recompute-if-missing
and degrade-and-continue were both rejected in ``06-CONTEXT.md`` because a
notebook that silently renders on nothing would make "runs top-to-bottom"
a meaningless verification criterion.

``load_full_span_checkpoint`` is the ONLY full-span (post-2020) read path
notebooks use (D-06): it wraps
:func:`trading_crab_lib.platform.honesty.holdout.load_full_span` rather than
opening the dev and holdout checkpoint managers by hand, so the fitting fence
stays intact while notebooks can still look at post-cutoff data.

``compute_regime_labeling`` routes ``labeling.diagnostics.label_regimes`` at
the non-production scratch namespace ``NOTEBOOK_SCRATCH_DIR`` rather than the
platform checkpoint namespace (D-10, mirroring the legacy pitfall P20/D5
where running the pipeline from a notebook/test corrupted a production
checkpoint) — no notebook may write ``data/checkpoints/platform/``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from trading_crab_lib import DATA_DIR, OUTPUT_DIR
from trading_crab_lib.checkpoints import CheckpointManager
from trading_crab_lib.platform.checkpoints import get_platform_checkpoint_manager
from trading_crab_lib.platform.honesty.holdout import (
    DEFAULT_HOLDOUT_CUTOFF,
    load_full_span,
)
from trading_crab_lib.platform.labeling.diagnostics import label_regimes

log = logging.getLogger(__name__)

# Scratch namespace for notebook-triggered labeling runs. Deliberately
# distinct from `trading_crab_lib.platform.checkpoints.PLATFORM_CHECKPOINT_DIR`
# so a notebook can never write the production platform checkpoint tree.
NOTEBOOK_SCRATCH_DIR: Path = DATA_DIR / "checkpoints" / "platform_notebook"

_REDACTED_VALUE = "<redacted>"

_REPORT_ARTIFACT_REBUILD_HINT = "python -m trading_crab_lib.platform.evaluation.report"


def load_platform_checkpoint(name: str, *, rebuild_hint: str) -> pd.DataFrame:
    """Load *name* from the platform checkpoint namespace, or raise actionably.

    Args:
        name: checkpoint name (e.g. ``"monthly_raw"``).
        rebuild_hint: the exact command an operator should run to produce
            the missing checkpoint (D-10).

    Raises:
        FileNotFoundError: naming *name*, the checkpoint directory, and
            *rebuild_hint*.
    """
    try:
        return get_platform_checkpoint_manager().load(name)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Platform checkpoint '{name}' not found under "
            f"data/checkpoints/platform/. Rebuild it with: {rebuild_hint}"
        ) from exc


def load_full_span_checkpoint(name: str, *, rebuild_hint: str) -> pd.DataFrame:
    """Load *name* across the full span (dev + holdout), the D-06 opt-in.

    Delegates to :func:`trading_crab_lib.platform.honesty.holdout.load_full_span`
    rather than opening two checkpoint managers by hand — the fitting fence
    (dev manager cannot reach the holdout tree) is never touched by this
    call; it only adds the explicit, visible "looking" opt-in.

    Any row dated after :data:`DEFAULT_HOLDOUT_CUTOFF` is logged at WARNING,
    pointing at ``.planning/POST-2020-OBSERVATIONS.md`` (D-07) — a post-2020
    observation that changes a decision must be recorded there.

    Raises:
        FileNotFoundError: naming *name*, the checkpoint directory, and
            *rebuild_hint*, on a missing dev checkpoint.
    """
    try:
        full_df = load_full_span(name)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Platform checkpoint '{name}' not found under "
            f"data/checkpoints/platform/ (full-span load, D-06). "
            f"Rebuild it with: {rebuild_hint}"
        ) from exc

    cutoff_ts = pd.Timestamp(DEFAULT_HOLDOUT_CUTOFF)
    if isinstance(full_df.index, pd.DatetimeIndex):
        post_cutoff = int((full_df.index > cutoff_ts).sum())
        if post_cutoff:
            log.warning(
                "load_full_span_checkpoint: %s carries %d row(s) dated after "
                "the %s holdout cutoff (D-06 opt-in — fitting stays fenced, "
                "looking does not). Any post-2020 observation that changes a "
                "decision must be recorded in .planning/POST-2020-OBSERVATIONS.md "
                "(D-07).",
                name, post_cutoff, DEFAULT_HOLDOUT_CUTOFF,
            )
    return full_df


def load_report_artifact(filename: str, *, rebuild_hint: str) -> pd.DataFrame:
    """Load *filename* from ``outputs/reports/platform/``, or raise actionably.

    Args:
        filename: report artifact filename (e.g. ``"backtest_kpi_table.parquet"``).
        rebuild_hint: the exact command an operator should run to produce
            the missing artifact (D-10).

    Raises:
        FileNotFoundError: naming *filename*, the report directory, and
            *rebuild_hint*.
    """
    path = OUTPUT_DIR / "reports" / "platform" / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Report artifact '{filename}' not found under "
            f"outputs/reports/platform/. Rebuild it with: {rebuild_hint}"
        )
    return pd.read_parquet(path)


def load_full_sample_states(
    *, rebuild_hint: str = _REPORT_ARTIFACT_REBUILD_HINT
) -> pd.Series:
    """Load the persisted full-sample smoothed state path (Amendment 3 item H).

    Returns:
        pd.Series: int-valued states indexed by date, from
        ``backtest_full_sample_states.parquet``'s ``state`` column.

    Raises:
        FileNotFoundError: with *rebuild_hint*, if the artifact is absent
            (it is written by plan 06-02, not this plan).
    """
    df = load_report_artifact("backtest_full_sample_states.parquet", rebuild_hint=rebuild_hint)
    return df["state"].astype(int)


def load_filtered_state_probs(
    *, rebuild_hint: str = _REPORT_ARTIFACT_REBUILD_HINT
) -> pd.DataFrame:
    """Load the persisted per-date filtered-state probability path.

    The parquet-safe column names (``state_0``, ``state_1``, ...) are
    renamed back to integer state ids, because
    ``evaluation.sojourn_lag.compute_sojourn_lag_headline`` looks columns up
    by integer state id.

    Raises:
        FileNotFoundError: with *rebuild_hint*, if the artifact is absent
            (it is written by plan 06-02, not this plan).
    """
    df = load_report_artifact("backtest_filtered_state_probs.parquet", rebuild_hint=rebuild_hint)
    rename_map = {
        col: int(col.rsplit("_", 1)[-1])
        for col in df.columns
        if isinstance(col, str) and col.startswith("state_") and col.rsplit("_", 1)[-1].isdigit()
    }
    return df.rename(columns=rename_map)


def redacted_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Deep copy of *cfg* with every value under a ``*api_key`` key redacted.

    Notebooks call this before displaying config so a committed cell output
    can never carry a live FRED credential (T-06-01).
    """

    def _redact(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {
                key: (_REDACTED_VALUE if isinstance(key, str) and key.endswith("api_key") else _redact(value))
                for key, value in obj.items()
            }
        if isinstance(obj, list):
            return [_redact(item) for item in obj]
        return obj

    return _redact(cfg)


def compute_regime_labeling(monthly_features: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    """Run ``label_regimes`` at the notebook scratch namespace (D-10).

    Routing at :data:`NOTEBOOK_SCRATCH_DIR`, rather than the production
    platform checkpoint namespace, is what keeps a notebook from ever
    writing ``data/checkpoints/platform/`` (D-10; legacy pitfall P20/D5 —
    running code from a notebook/test corrupting a production checkpoint).

    Returns:
        dict: ``label_regimes``'s own return dict (``states``, ``confidences``,
        ``churn``, ``diagnostics_path``) plus ``regime_labels``,
        ``regime_confidences``, ``regime_profiles`` — the frames read back
        from the scratch namespace.
    """
    result = dict(label_regimes(monthly_features, cfg, checkpoint_dir=NOTEBOOK_SCRATCH_DIR))
    cm = CheckpointManager(checkpoint_dir=NOTEBOOK_SCRATCH_DIR)
    result["regime_labels"] = cm.load("regime_labels")
    result["regime_confidences"] = cm.load("regime_confidences")
    result["regime_profiles"] = cm.load("regime_profiles")
    return result
