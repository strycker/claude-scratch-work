"""Which code is running: git branch, short commit and checkout path, for every CLI's first line.

Usage::

    from trading_crab_lib.platform.code_version import code_version
    print(code_version())
    # Code: claude/phase-8.5-clean-slate @ 903c3d6 — /Users/me/claude-scratch-work

Git runs in this file's own directory, so the line names the checkout whose code is executing
(an editable install of another clone shows up here), not the shell's working directory.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def _git(*args: str) -> str | None:
    try:
        done = subprocess.run(["git", *args], cwd=_HERE, capture_output=True, text=True, timeout=10)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def code_version() -> str:
    """One line: ``Code: <branch> @ <short sha>[ + uncommitted changes] — <checkout path>``.

    A detached HEAD reads ``(detached)``; outside a git checkout the line says so and names the
    package directory instead.
    """
    sha = _git("rev-parse", "--short", "HEAD")
    if sha is None:
        return f"Code: not a git checkout — {_HERE}"
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    branch = "(detached)" if branch in (None, "HEAD") else branch
    dirty = " + uncommitted changes" if _git("status", "--porcelain", "--untracked-files=no") else ""
    root = _git("rev-parse", "--show-toplevel") or str(_HERE)
    return f"Code: {branch} @ {sha}{dirty} — {root}"
