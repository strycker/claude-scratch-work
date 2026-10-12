"""Which code is running: git branch, short commit and checkout path, for every CLI's first line.

Usage::

    from trading_crab_lib.platform.code_version import code_version
    print(code_version())
    # Code: claude/phase-8.5-clean-slate @ 903c3d6 — /Users/me/claude-scratch-work

Git runs in this file's own directory, so the line names the checkout whose code is executing
(an editable install of another clone shows up here), not the shell's working directory. A copy
installed into a folder inside a checkout (say a non-editable install in the repo's .venv) is not
tracked there and says so instead of borrowing the checkout's commit.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_NAME = Path(__file__).name
#: Where an edit counts as "uncommitted changes" to the code that runs: tracked or new files only
#: (ignored files never show), submodules excluded.
_CODE_PATHS = ("src", "scripts", "config")


def _git(*args: str, cwd: Path | None = None) -> str | None:
    """Stdout of a git command, or None when git is missing, hangs or fails."""
    try:
        done = subprocess.run(["git", *args], cwd=cwd or _HERE, capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError, UnicodeDecodeError):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def code_version() -> str:
    """One line: ``Code: <branch> @ <short sha>[ + uncommitted changes] — <checkout path>``.

    A detached HEAD reads ``(detached)``. When git cannot tell whether the code has uncommitted
    changes, the line says so rather than implying a clean tree.
    """
    sha = _git("rev-parse", "--short", "HEAD")
    if sha is None:
        return f"Code: not a git checkout (or git unavailable) — {_HERE}"
    if _git("ls-files", "--error-unmatch", _NAME) is None:
        return f"Code: an installed copy, not tracked by the checkout around it — {_HERE}"
    root = _git("rev-parse", "--show-toplevel")
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    branch = "(detached)" if branch in (None, "HEAD") else branch
    status = None if root is None else _git(
        "status", "--porcelain", "--ignore-submodules", "--", *_CODE_PATHS, cwd=Path(root)
    )
    if status is None:
        dirty = " (uncommitted changes unknown)"
    else:
        dirty = " + uncommitted changes" if status else ""
    return f"Code: {branch} @ {sha}{dirty} — {root or _HERE}"
