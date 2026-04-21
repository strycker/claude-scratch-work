"""
CheckpointManager — save, load, and validity-check intermediate DataFrames.

Each checkpoint is a parquet file stored under data/checkpoints/.
A JSON manifest alongside each file records:
  - creation timestamp
  - source config hash (settings.yaml)
  - row/column counts for quick sanity-checking

Why parquet over pickle?
  - Typed, smaller, readable by other tools
  - Survives Python/pandas version upgrades
  - Fast I/O even for 300+ column DataFrames

Models (sklearn objects) are stored as pickle alongside, since they have
no parquet-compatible serialization.

Usage:
    from trading_crab_lib.checkpoints import CheckpointManager
    cm = CheckpointManager()

    # Save
    cm.save(df, "macro_raw")
    cm.save(features, "features")

    # Load (raises FileNotFoundError if missing)
    df = cm.load("macro_raw")

    # Conditional — only recompute if checkpoint is stale or missing
    if cm.is_fresh("macro_raw", max_age_days=7):
        df = cm.load("macro_raw")
    else:
        df = expensive_computation()
        cm.save(df, "macro_raw")

    # List all checkpoints
    print(cm.list())

    # Clear one
    cm.clear("macro_raw")

    # Clear all
    cm.clear_all()
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from trading_crab_lib import CONFIG_DIR, DATA_DIR

log = logging.getLogger(__name__)

CHECKPOINT_DIR = DATA_DIR / "checkpoints"

# ── Preservation checkpoints ─────────────────────────────────────────
# Wide parquet snapshots that survive clear_all().  Purpose: downstream
# steps drop sparse columns via dropna(axis=1), erasing the full column
# audit trail.  Preservation checkpoints retain every column so you can
# always inspect what was available before narrowing.
PRESERVATION_CHECKPOINT_NAMES: frozenset[str] = frozenset({
    "macro_raw_secondary",
    "features_secondary",
    "features_supervised_secondary",
})


def preservation_checkpoint_should_write(
    name: str,
    cm: CheckpointManager,
    *,
    force: bool = False,
) -> bool:
    """Return True if preservation checkpoint *name* should be (re)written.

    Rules:
    - If *force* is True (``--refresh-preservation``), always write.
    - Otherwise write only when the checkpoint does not yet exist.
    """
    if name not in PRESERVATION_CHECKPOINT_NAMES:
        return False
    if force:
        return True
    parquet_path = cm.dir / f"{name}.parquet"
    return not parquet_path.exists()


def _config_hash() -> str:
    """MD5 of settings.yaml — used to detect config changes that invalidate checkpoints."""
    path = CONFIG_DIR / "settings.yaml"
    if not path.exists():
        return "no-config"
    return hashlib.md5(path.read_bytes()).hexdigest()[:8]


class CheckpointManager:
    """
    Manages parquet checkpoints for DataFrames and pickle checkpoints for models.

    All files live under data/checkpoints/.  Each checkpoint pair:
      {name}.parquet  — the DataFrame
      {name}.meta.json — metadata (timestamp, config hash, shape)
    """

    def __init__(self, checkpoint_dir: Path | None = None) -> None:
        if checkpoint_dir is not None:
            self.dir = checkpoint_dir
        else:
            env_override = os.environ.get("TC_CHECKPOINT_DIR")
            self.dir = Path(env_override) if env_override else CHECKPOINT_DIR
        self.dir.mkdir(parents=True, exist_ok=True)

    # ── DataFrame checkpoints ─────────────────────────────────────────────

    def save(self, df: pd.DataFrame, name: str) -> Path:
        """Persist a DataFrame to {name}.parquet and write metadata."""
        parquet_path = self.dir / f"{name}.parquet"
        meta_path = self.dir / f"{name}.meta.json"

        df.to_parquet(parquet_path)

        meta = {
            "name": name,
            "created": datetime.now().isoformat(),
            "config_hash": _config_hash(),
            "rows": len(df),
            "columns": len(df.columns),
            "col_names": list(df.columns),
            "index_start": str(df.index[0]) if len(df) else None,
            "index_end": str(df.index[-1]) if len(df) else None,
        }
        meta_path.write_text(json.dumps(meta, indent=2))

        log.info(
            "Checkpoint saved: %s  (%d rows × %d cols)",
            name, len(df), len(df.columns),
        )
        return parquet_path

    def load(self, name: str) -> pd.DataFrame:
        """Load a DataFrame checkpoint.  Raises FileNotFoundError if missing."""
        parquet_path = self.dir / f"{name}.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {parquet_path}")
        df = pd.read_parquet(parquet_path)
        log.info(
            "Checkpoint loaded: %s  (%d rows × %d cols)",
            name, len(df), len(df.columns),
        )
        return df

    def is_fresh(
        self,
        name: str,
        max_age_days: float = 7.0,
        require_config_match: bool = False,
    ) -> bool:
        """
        Return True if a fresh, valid checkpoint exists.

        Args:
            name: checkpoint name
            max_age_days: reject checkpoints older than this
            require_config_match: if True, also reject if settings.yaml changed
        """
        parquet_path = self.dir / f"{name}.parquet"
        meta_path = self.dir / f"{name}.meta.json"

        if not parquet_path.exists() or not meta_path.exists():
            log.debug("Checkpoint missing: %s", name)
            return False

        try:
            meta = json.loads(meta_path.read_text())
            created = datetime.fromisoformat(meta["created"])
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            log.warning("Corrupt checkpoint metadata %s: %s", meta_path.name, exc)
            return False

        age = datetime.now() - created

        if age > timedelta(days=max_age_days):
            log.info(
                "Checkpoint stale: %s (%.1f days old, max %.1f)",
                name, age.total_seconds() / 86400, max_age_days,
            )
            return False

        if require_config_match and meta.get("config_hash") != _config_hash():
            log.info("Checkpoint config mismatch: %s — settings.yaml changed", name)
            return False

        log.debug(
            "Checkpoint fresh: %s (%.1f days old, %d×%d)",
            name, age.total_seconds() / 86400, meta["rows"], meta["columns"],
        )
        return True

    def clear(self, name: str) -> None:
        """Delete a single checkpoint (parquet + meta)."""
        for suffix in [".parquet", ".meta.json"]:
            p = self.dir / f"{name}{suffix}"
            if p.exists():
                p.unlink()
                log.info("Cleared checkpoint: %s", p.name)

    def clear_all(self, *, include_preservation: bool = False) -> None:
        """Delete all checkpoints in the checkpoint directory.

        Preservation checkpoints (``*_secondary``) are kept by default.
        Pass ``include_preservation=True`` to remove them as well.
        """
        for f in self.dir.iterdir():
            if not include_preservation:
                stem = f.stem
                # .meta.json has stem like "foo.meta" — strip the ".meta" part
                if stem.endswith(".meta"):
                    stem = stem[: -len(".meta")]
                if stem in PRESERVATION_CHECKPOINT_NAMES:
                    log.debug("clear_all: preserving %s", f.name)
                    continue
            f.unlink()
        log.info(
            "All checkpoints cleared%s",
            "" if include_preservation else " (preservation checkpoints kept)",
        )

    def list(self) -> list[dict]:
        """Return a list of checkpoint metadata dicts, sorted by creation time."""
        entries = []
        for meta_path in sorted(self.dir.glob("*.meta.json")):
            try:
                meta = json.loads(meta_path.read_text())
                entries.append(meta)
            except (json.JSONDecodeError, OSError, ValueError) as exc:
                log.warning("Failed to parse checkpoint metadata %s: %s", meta_path.name, exc)
        entries.sort(key=lambda m: m.get("created", ""))
        return entries

    def summary(self) -> str:
        """Human-readable table of all checkpoints."""
        entries = self.list()
        if not entries:
            return "No checkpoints found."
        lines = [f"{'Name':<30} {'Created':<25} {'Shape':<12} {'Config'}", "-" * 80]
        for m in entries:
            shape = f"{m.get('rows','?')}×{m.get('columns','?')}"
            lines.append(
                f"{m['name']:<30} {m['created'][:19]:<25} {shape:<12} {m.get('config_hash','?')}"
            )
        return "\n".join(lines)

    # ── Model (pickle) checkpoints ─────────────────────────────────────────

    def save_model(self, model: Any, name: str) -> Path:
        """Serialize a sklearn model to {name}.pkl using joblib."""
        pkl_path = self.dir / f"{name}.pkl"
        joblib.dump(model, pkl_path)
        log.info("Model checkpoint saved: %s", name)
        return pkl_path

    def load_model(self, name: str) -> Any:
        """Load a serialized model.  Raises FileNotFoundError if missing."""
        pkl_path = self.dir / f"{name}.pkl"
        if not pkl_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {pkl_path}")
        model = joblib.load(pkl_path)
        log.info("Model checkpoint loaded: %s", name)
        return model

    def model_exists(self, name: str) -> bool:
        """Return True if a pickled model checkpoint exists for *name*."""
        return (self.dir / f"{name}.pkl").exists()
