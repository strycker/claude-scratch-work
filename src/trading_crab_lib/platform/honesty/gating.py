"""
Causal-feature gating guard (HON-06, design R5, pitfall P1).

RESEARCH.md Open Question 1 is RESOLVED by 02-PATTERNS.md: `monthly_features`
(Phase 1, ``transforms_monthly.py``) is causal-by-construction — the only
rolling window anywhere in the platform namespace is a trailing
``equity_returns.rolling(3).std(ddof=0)``; no ``center=True`` smoothing
exists, and no causal/noncausal companion checkpoint file exists or should
be created.

This guard is therefore NOT a two-file selector, unlike the incumbent's
``features.parquet`` vs ``features_supervised.parquet`` split (ADR #1) or the
salvaged ``feature_gating.py`` it is ported from. It is an assertion guard:
scan the columns of the loaded ``monthly_features`` checkpoint for any
forbidden centered-window suffix and refuse loudly by default, keeping the
salvaged module's raise-loud / warn-on-opt-out shape.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pyarrow.parquet as pq

from trading_crab_lib.platform.checkpoints import get_platform_checkpoint_manager

log = logging.getLogger(__name__)

# Column-name suffixes that, if present, indicate a centered (two-sided /
# zero-phase) window was used to compute the column — forbidden in any
# feature set fed to supervised training (P1: look-ahead bias).
FORBIDDEN_CENTERED_SUFFIXES: tuple[str, ...] = ("_centered", "_c5", "_zerophase")


def assert_causal_features(columns, *, allow_noncausal: bool = False) -> bool:
    """Scan *columns* for any forbidden centered-window suffix.

    Returns:
        bool: ``noncausal_used`` — ``True`` only when a forbidden column was
            found AND ``allow_noncausal=True`` was passed to opt out.

    Raises:
        ValueError: if a forbidden column is found and ``allow_noncausal`` is
            ``False`` (the default) — names the offending column(s) and
            instructs how to opt out.
    """
    offending = [str(c) for c in columns if str(c).endswith(FORBIDDEN_CENTERED_SUFFIXES)]
    if not offending:
        return False

    if not allow_noncausal:
        raise ValueError(
            f"Forbidden non-causal (centered-window) column(s) found: {offending}.\n"
            "Supervised training paths require causal-only features (HON-06) — "
            "training on centered/two-sided windows inflates accuracy and cannot "
            "be reproduced in real-time scoring (pitfall P1).\n\n"
            "If you intentionally want to allow non-causal columns, re-run with:\n"
            "  allow_noncausal=True"
        )

    log.warning(
        "NONCAUSAL_USED=true — non-causal column(s) allowed by explicit opt-out: %s",
        offending,
    )
    return True


def select_platform_feature_path(
    checkpoint_dir: Path | None = None,
    *,
    allow_noncausal: bool = False,
) -> tuple[Path, str, bool]:
    """Resolve the ``monthly_features.parquet`` path and gate its columns.

    Loads the checkpoint's column names cheaply (parquet schema only, no
    row data) via the platform checkpoint manager's directory
    (``checkpoint_dir`` overrides for tests), then runs
    :func:`assert_causal_features` against them.

    Args:
        checkpoint_dir: overrides the default platform checkpoint directory
            (for tests). When omitted, uses
            ``get_platform_checkpoint_manager().dir``.
        allow_noncausal: opt-out flag forwarded to
            :func:`assert_causal_features`.

    Returns:
        tuple[Path, str, bool]: ``(path, "monthly_features", noncausal_used)``.

    Raises:
        FileNotFoundError: if the ``monthly_features`` checkpoint does not exist.
        ValueError: if a forbidden centered-window column is present and
            ``allow_noncausal`` is ``False``.
    """
    directory = checkpoint_dir if checkpoint_dir is not None else get_platform_checkpoint_manager().dir
    path = Path(directory) / "monthly_features.parquet"
    if not path.exists():
        raise FileNotFoundError(f"monthly_features checkpoint not found: {path}")

    columns = pq.ParquetFile(path).schema_arrow.names
    noncausal_used = assert_causal_features(columns, allow_noncausal=allow_noncausal)
    return path, "monthly_features", noncausal_used
