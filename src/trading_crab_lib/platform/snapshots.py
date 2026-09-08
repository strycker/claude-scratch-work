"""
Offline dev snapshots — platform checkpoint export/restore/list (Task 3, 260806-u89).

DEV/OFFLINE CONVENIENCE ONLY. A snapshot is a frozen copy of one or more
platform checkpoints, captured at a stated moment. It is NEVER live data, it
is NEVER loaded automatically anywhere in this library, and every read of
one is announced at WARNING carrying the capture timestamp. Loading is
strictly opt-in via ``TC_USE_SNAPSHOTS=1`` (see :func:`snapshots_enabled`) —
there is no automatic fallback from a failed live fetch to a snapshot
anywhere in this codebase.

Snapshots live at a committed, repo-relative location
(``data/snapshots/platform/``) so a developer with no network can restore
them and run the platform report. Restored checkpoints carry a persistent
``"source": "offline-snapshot"`` marker in their meta json
(:func:`is_snapshot_backed`), so a snapshot can never be silently mistaken
for live data even after it has been restored.

Usage:
    from trading_crab_lib.platform.snapshots import export_snapshot, restore_snapshot

    # After a successful build, on a machine with normal internet:
    export_snapshot()

    # Later, on a machine with no network:
    restore_snapshot()

Or via the CLI: ``python scripts/platform_snapshot.py export|restore|list``.
See ``docs/offline_snapshots.md`` for the full guide.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from trading_crab_lib import ROOT
from trading_crab_lib.checkpoints import CheckpointManager, _config_hash
from trading_crab_lib.platform.checkpoints import get_platform_checkpoint_manager

log = logging.getLogger(__name__)

# Repo-relative (not DATA_DIR-relative) so the files are committable and so
# TC_ROOT_DIR still works for anyone who has relocated their data directory.
SNAPSHOT_DIR: Path = ROOT / "data" / "snapshots" / "platform"

DEFAULT_SNAPSHOT_NAMES: tuple[str, ...] = ("daily_raw", "monthly_raw", "monthly_features")

_ENV_TRUTHY = {"1", "true", "yes"}

# Meta-json marker written by restore_snapshot() and read by is_snapshot_backed().
_SNAPSHOT_MARKER_KEY = "source"
_SNAPSHOT_MARKER_VALUE = "offline-snapshot"
_SNAPSHOT_CAPTURED_AT_KEY = "snapshot_captured_at"


def snapshots_enabled() -> bool:
    """True only when ``TC_USE_SNAPSHOTS`` is set to a truthy value
    ("1"/"true"/"yes", case-insensitive). Anything else, including unset, is
    disabled. Nothing in this library reads a snapshot unless something
    explicitly asks it to — this flag exists for callers that want to opt an
    entire run into snapshot-backed data, not for any automatic fallback.
    """
    return os.environ.get("TC_USE_SNAPSHOTS", "").strip().lower() in _ENV_TRUTHY


def _git_commit() -> str | None:
    """Best-effort current git commit hash. Returns None (never raises) if
    git is unavailable or the working tree is not a git repo — a missing
    git must not break an export."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT, capture_output=True, text=True, timeout=10, check=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning("snapshots: could not determine git commit (%s) — manifest will omit it.", exc)
        return None


def _frame_meta(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "col_names": list(df.columns),
        "index_start": str(df.index[0]) if len(df) else None,
        "index_end": str(df.index[-1]) if len(df) else None,
    }


def read_manifest(snapshot_dir: Path | None = None) -> dict[str, Any]:
    """Read and return ``manifest.json`` from *snapshot_dir* (default
    `SNAPSHOT_DIR`). Returns ``{}`` if the manifest is absent or
    unparseable — never raises."""
    in_dir = Path(snapshot_dir) if snapshot_dir is not None else SNAPSHOT_DIR
    manifest_path = in_dir / "manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        return json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("read_manifest: could not read %s: %s", manifest_path, exc)
        return {}


def export_snapshot(
    names: list[str] | None = None,
    *,
    cm: CheckpointManager | None = None,
    snapshot_dir: Path | None = None,
    max_mb: float = 10.0,
    daily_start: str | None = None,
    compression: str = "zstd",
) -> Path:
    """Export *names* (default `DEFAULT_SNAPSHOT_NAMES`) from the platform
    checkpoint manager to *snapshot_dir* (default `SNAPSHOT_DIR`) as one
    parquet per checkpoint plus a single `manifest.json`.

    A name with no checkpoint currently on disk is skipped with a WARNING —
    the other names still export and the call does not raise. When
    *daily_start* is given, the `daily_raw` frame (only) is truncated to
    that lower date bound before writing — the size lever for a smaller
    committed snapshot, off by default. Refuses to write a snapshot set
    whose total size exceeds *max_mb*, naming the file that blew the
    budget, and deletes whatever it had already written for this call.

    Returns the snapshot directory (`snapshot_dir` or `SNAPSHOT_DIR`).
    """
    names = list(names) if names is not None else list(DEFAULT_SNAPSHOT_NAMES)
    cm = cm if cm is not None else get_platform_checkpoint_manager()
    out_dir = Path(snapshot_dir) if snapshot_dir is not None else SNAPSHOT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    written_paths: list[Path] = []
    per_name_meta: dict[str, Any] = {}

    for name in names:
        try:
            df = cm.load(name)
        except FileNotFoundError:
            log.warning("export_snapshot: no checkpoint '%s' on disk — skipping.", name)
            continue

        if daily_start is not None and name == "daily_raw" and len(df):
            df = df.loc[pd.Timestamp(daily_start):]

        out_path = out_dir / f"{name}.parquet"
        try:
            df.to_parquet(out_path, compression=compression)
        except (ValueError, ImportError) as exc:
            log.warning(
                "export_snapshot: compression '%s' unavailable for '%s' (%s) — falling back to snappy.",
                compression, name, exc,
            )
            df.to_parquet(out_path, compression="snappy")

        written_paths.append(out_path)
        per_name_meta[name] = _frame_meta(df)

    total_bytes = sum(p.stat().st_size for p in written_paths)
    max_bytes = max_mb * 1024 * 1024
    if total_bytes > max_bytes:
        largest = max(written_paths, key=lambda p: p.stat().st_size)
        largest_mb = largest.stat().st_size / 1e6
        for p in written_paths:
            p.unlink(missing_ok=True)
        raise ValueError(
            f"export_snapshot: total snapshot size {total_bytes / 1e6:.2f} MB exceeds "
            f"max_mb={max_mb:.2f} MB — largest file was '{largest.name}' ({largest_mb:.2f} MB). "
            "Nothing was written; narrow `names`, lower `daily_start`, or raise `max_mb`."
        )

    provenance_path = cm.dir / "splice_provenance.json"
    source_provenance: Any | None = None
    if provenance_path.exists():
        try:
            source_provenance = json.loads(provenance_path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("export_snapshot: could not read splice provenance %s: %s", provenance_path, exc)

    manifest: dict[str, Any] = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "names": per_name_meta,
    }
    if source_provenance is not None:
        manifest["source_provenance"] = source_provenance

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info(
        "export_snapshot: wrote %d checkpoint(s) + manifest to %s (%.2f MB total)",
        len(written_paths), out_dir, total_bytes / 1e6,
    )
    return out_dir


def load_snapshot(name: str, *, snapshot_dir: Path | None = None) -> pd.DataFrame:
    """Load one checkpoint's snapshot parquet as a DataFrame.

    Always logs a WARNING naming the checkpoint, the manifest's
    `captured_at`, and that this is frozen offline data rather than a live
    fetch — every read of a snapshot is announced, never silent.
    """
    in_dir = Path(snapshot_dir) if snapshot_dir is not None else SNAPSHOT_DIR
    parquet_path = in_dir / f"{name}.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Snapshot not found: {parquet_path}")

    manifest = read_manifest(snapshot_dir=in_dir)
    captured_at = manifest.get("captured_at", "unknown capture time")
    log.warning(
        "load_snapshot: loading FROZEN OFFLINE snapshot '%s' captured at %s — this is NOT live data.",
        name, captured_at,
    )
    return pd.read_parquet(parquet_path)


def restore_snapshot(
    names: list[str] | None = None,
    *,
    cm: CheckpointManager | None = None,
    snapshot_dir: Path | None = None,
) -> list[str]:
    """Copy each snapshot parquet in *names* (default `DEFAULT_SNAPSHOT_NAMES`)
    into the platform checkpoint namespace, writing an accompanying meta
    json in the existing meta schema plus two extra keys marking it as
    snapshot-derived and carrying the capture timestamp.

    A name with no matching snapshot on disk is skipped with a WARNING.
    Logs one WARNING banner naming every checkpoint actually replaced.
    Returns the list of checkpoint names actually restored.
    """
    names = list(names) if names is not None else list(DEFAULT_SNAPSHOT_NAMES)
    cm = cm if cm is not None else get_platform_checkpoint_manager()
    in_dir = Path(snapshot_dir) if snapshot_dir is not None else SNAPSHOT_DIR

    manifest = read_manifest(snapshot_dir=in_dir)
    captured_at = manifest.get("captured_at", "unknown capture time")

    restored: list[str] = []
    for name in names:
        parquet_path = in_dir / f"{name}.parquet"
        if not parquet_path.exists():
            log.warning("restore_snapshot: no snapshot for '%s' at %s — skipping.", name, parquet_path)
            continue

        df = pd.read_parquet(parquet_path)
        df.to_parquet(cm.dir / f"{name}.parquet")

        meta: dict[str, Any] = {
            "name": name,
            "created": datetime.now().isoformat(),
            "config_hash": _config_hash(),
            "rows": len(df),
            "columns": len(df.columns),
            "col_names": list(df.columns),
            "index_start": str(df.index[0]) if len(df) else None,
            "index_end": str(df.index[-1]) if len(df) else None,
            _SNAPSHOT_MARKER_KEY: _SNAPSHOT_MARKER_VALUE,
            _SNAPSHOT_CAPTURED_AT_KEY: captured_at,
        }
        (cm.dir / f"{name}.meta.json").write_text(json.dumps(meta, indent=2))
        restored.append(name)

    if restored:
        log.warning(
            "restore_snapshot: replaced checkpoint(s) %s with OFFLINE SNAPSHOT data captured %s "
            "— this is NOT live data.",
            restored, captured_at,
        )
    return restored


def is_snapshot_backed(cm: CheckpointManager, name: str) -> bool:
    """True if checkpoint *name* under *cm* was written by
    :func:`restore_snapshot` (its meta json carries the snapshot marker).

    A missing or unparseable meta file returns False rather than raising —
    absence of the marker is the correct default for an ordinary
    `CheckpointManager.save()`-written checkpoint.
    """
    meta_path = cm.dir / f"{name}.meta.json"
    if not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text())
    except (json.JSONDecodeError, OSError, KeyError):
        return False
    return meta.get(_SNAPSHOT_MARKER_KEY) == _SNAPSHOT_MARKER_VALUE
