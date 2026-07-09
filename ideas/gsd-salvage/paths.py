"""
Workspace path resolution for ``trading_crab_lib``.

After a plain ``pip install``, the package lives under ``site-packages``; the
historic ``Path(__file__).parent.parent.parent`` trick points at garbage paths.

**Resolution order**

1. **``TRADING_CRAB_ROOT``** — If set, this directory is treated as the project
   tree: ``config/``, ``data/``, ``outputs/`` under it (same layouts as a repo
   checkout). Optional overrides: ``TRADING_CRAB_CONFIG``, ``TRADING_CRAB_DATA``,
   ``TRADING_CRAB_OUTPUT`` may each replace the corresponding subdirectory path.

2. **Granular dirs (no root)** — If **any** of ``TRADING_CRAB_CONFIG``,
   ``TRADING_CRAB_DATA``, or ``TRADING_CRAB_OUTPUT`` is set, **all three** must
   be set: each is a **directory** path (``TRADING_CRAB_CONFIG`` is the directory
   that contains ``settings.yaml``, i.e. the same as ``CONFIG_DIR`` used by
   ``load()``). ``root`` is inferred as ``config_dir.parent``.

3. **Repo / editable detection** — Walk parents of the ``trading_crab_lib``
   package directory for a tree containing ``config/settings.yaml``.

4. **Otherwise** — ``RuntimeError`` with instructions to set ``TRADING_CRAB_ROOT``
   (typical ``pip install`` consumer).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LibraryPaths:
    """Resolved workspace locations for config, data, and outputs.

    Attributes:
        root: Project root when inferrable (e.g. parent of ``config/``).
        config_dir: Directory containing ``settings.yaml``.
        data_dir: ``data/`` tree (checkpoints, raw, processed).
        output_dir: ``outputs/`` tree (plots, models, reports).
    """

    root: Path
    config_dir: Path
    data_dir: Path
    output_dir: Path


_MAX_WALK = 48


def resolve_library_paths(*, package_file: Path | None = None) -> LibraryPaths:
    """
    Resolve :class:`LibraryPaths` from environment variables or repo layout.

    Parameters
    ----------
    package_file :
        Path to this module's file (``paths.py``). Defaults to ``Path(__file__)``.
        Tests may pass a synthetic path under a fake package layout.
    """
    pkg_file = (package_file or Path(__file__)).resolve()
    package_dir = pkg_file.parent

    # Env-first: supports CI, Docker, and "pip install then point at a project tree"
    # without editable installs. Economists can treat TRADING_CRAB_ROOT as "where
    # my config and data live" on any machine.
    root_s = os.environ.get("TRADING_CRAB_ROOT", "").strip()
    cfg_s = os.environ.get("TRADING_CRAB_CONFIG", "").strip()
    data_s = os.environ.get("TRADING_CRAB_DATA", "").strip()
    out_s = os.environ.get("TRADING_CRAB_OUTPUT", "").strip()
    granular = (cfg_s, data_s, out_s)
    any_granular = any(granular)
    all_granular = all(granular)

    if root_s:
        # Whole-tree override: one variable, default subdirs config/data/outputs.
        root = Path(root_s).expanduser().resolve()
        config_dir = Path(cfg_s).expanduser().resolve() if cfg_s else root / "config"
        data_dir = Path(data_s).expanduser().resolve() if data_s else root / "data"
        output_dir = Path(out_s).expanduser().resolve() if out_s else root / "outputs"
        return LibraryPaths(
            root=root,
            config_dir=config_dir,
            data_dir=data_dir,
            output_dir=output_dir,
        )

    if any_granular:
        # Partial granular paths are forbidden — avoids ambiguous "half outside" layouts.
        if not all_granular:
            raise RuntimeError(
                "Partial TRADING_CRAB_CONFIG / TRADING_CRAB_DATA / TRADING_CRAB_OUTPUT "
                "is not supported. Set TRADING_CRAB_ROOT, or set all three directory "
                "paths. After pip install, set TRADING_CRAB_ROOT (recommended). "
                "See README.md — Library-only install."
            )
        config_dir = Path(cfg_s).expanduser().resolve()
        data_dir = Path(data_s).expanduser().resolve()
        output_dir = Path(out_s).expanduser().resolve()
        root = config_dir.parent
        return LibraryPaths(
            root=root,
            config_dir=config_dir,
            data_dir=data_dir,
            output_dir=output_dir,
        )

    # Editable install / repo checkout: walk upward until we see config/settings.yaml.
    # This ties the installed package back to a checkout that still has config/ at repo root.
    p = package_dir
    for _ in range(_MAX_WALK):
        settings_yaml = p / "config" / "settings.yaml"
        if settings_yaml.is_file():
            root = p
            return LibraryPaths(
                root=root,
                config_dir=root / "config",
                data_dir=root / "data",
                output_dir=root / "outputs",
            )
        parent = p.parent
        if parent == p:
            break
        p = parent

    raise RuntimeError(
        "trading_crab_lib could not find config/settings.yaml above the package "
        "directory and no TRADING_CRAB_* environment variables are set. "
        "After pip install trading-crab-lib, set TRADING_CRAB_ROOT to your project "
        "tree (with config/, data/, outputs/) or set TRADING_CRAB_CONFIG, "
        "TRADING_CRAB_DATA, and TRADING_CRAB_OUTPUT. "
        "See README.md — Library-only install."
    )
