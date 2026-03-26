"""
trading_crab_lib — Market Regime Classification & Prediction Pipeline

Env var overrides (optional):
    TC_ROOT_DIR   — override the auto-detected repo root
    TC_CONFIG_DIR — override config/ directory
    TC_DATA_DIR   — override data/ directory
    TC_OUTPUT_DIR — override outputs/ directory
"""

from __future__ import annotations

import os
from pathlib import Path


def _resolve_dir(env_var: str, default: Path) -> Path:
    """Return Path from *env_var* if set, otherwise *default*."""
    val = os.environ.get(env_var)
    if val:
        return Path(val)
    return default


_DEFAULT_ROOT = Path(__file__).parent.parent.parent  # repo root

ROOT = _resolve_dir("TC_ROOT_DIR", _DEFAULT_ROOT)
CONFIG_DIR = _resolve_dir("TC_CONFIG_DIR", ROOT / "config")
DATA_DIR = _resolve_dir("TC_DATA_DIR", ROOT / "data")
OUTPUT_DIR = _resolve_dir("TC_OUTPUT_DIR", ROOT / "outputs")


# ── Convenience re-exports ──────────────────────────────────────────
# Enables:  import trading_crab_lib as tcl; cfg = tcl.load()

def load(*args, **kwargs):
    """Shortcut for :func:`trading_crab_lib.config.load`."""
    from trading_crab_lib.config import load as _load
    return _load(*args, **kwargs)


def load_portfolio(*args, **kwargs):
    """Shortcut for :func:`trading_crab_lib.config.load_portfolio`."""
    from trading_crab_lib.config import load_portfolio as _load_portfolio
    return _load_portfolio(*args, **kwargs)


# Lazy imports to avoid circular dependencies at module load time
def __getattr__(name: str):
    if name == "RunConfig":
        from trading_crab_lib.runtime import RunConfig
        return RunConfig
    if name == "CheckpointManager":
        from trading_crab_lib.checkpoints import CheckpointManager
        return CheckpointManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
