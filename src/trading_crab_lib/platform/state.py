"""
Start over at will: archive, list, reset, restore and promote the platform's generated state (08.4, K-1..K-8).

    python -m trading_crab_lib.platform.state archive NAME [--note TEXT]   # data/ + outputs/ -> archives/NAME/
    python -m trading_crab_lib.platform.state list                         # what is archived / promoted
    python -m trading_crab_lib.platform.state reset [NAME] --yes           # archive first, then empty both trees
    python -m trading_crab_lib.platform.state restore NAME                 # into EMPTY trees, sha256-checked
    python -m trading_crab_lib.platform.state promote NAME                 # commit + tag model/NAME (never pushes)

An archive is ``archives/NAME/state.tar.gz`` plus ``manifest.json`` (name, note, times, git commit, registry
trial count, and ``{path, bytes, sha256}`` per file). Pickles (``.pkl``, ``.pickle``, ``.joblib``) are never
archived (P27: no pickle arrives through git); ``reset`` deletes them and prints their paths. ``registry/`` and
``archives/`` are never touched by ``reset``. Stdlib only; this module loads no model and runs no refit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import shutil
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from zoneinfo import ZoneInfo

from trading_crab_lib.platform.checkpoints import PLATFORM_DATA_DIR, PLATFORM_OUTPUT_DIR, PLATFORM_ROOT_DIR
from trading_crab_lib.platform.honesty.registry import DEFAULT_REGISTRY_PATH, total_trial_count

log = logging.getLogger(__name__)

# Read at call time, so tests point them at tmp_path.
DATA_DIR = PLATFORM_DATA_DIR
OUTPUT_DIR = PLATFORM_OUTPUT_DIR
REPO_ROOT = PLATFORM_ROOT_DIR
ARCHIVES_DIR = REPO_ROOT / "archives"
REGISTRY_PATH = DEFAULT_REGISTRY_PATH

PICKLE_SUFFIXES = (".pkl", ".pickle", ".joblib")
_NAME_RE = re.compile(r"[a-z0-9][a-z0-9._-]*")


class StateError(RuntimeError):
    """A refusal. The message says what happened and what to do next."""


# ── Helpers ──


def _check_name(name: str) -> None:
    if not _NAME_RE.fullmatch(name) or name == "promoted":
        raise StateError(f"bad archive name {name!r}: use lowercase letters, digits, '.', '_', '-' ('promoted' is reserved)")


def _trees() -> tuple[tuple[str, Path], tuple[str, Path]]:
    return (("data", DATA_DIR), ("outputs", OUTPUT_DIR))


def _walk(root: Path) -> list[Path]:
    """Every file or symlink under *root*, sorted, except .gitkeep."""
    if not root.is_dir():
        return []
    return sorted(p for p in root.rglob("*") if (p.is_file() or p.is_symlink()) and p.name != ".gitkeep")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    try:
        done = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise StateError("git is not installed") from exc
    if done.returncode:
        raise StateError(f"git {args[0]} failed: {done.stderr.strip()}")
    return done.stdout.rstrip()


def _git_state() -> tuple[str, bool | None]:
    """(HEAD commit, dirty tracked files?) or ("unknown", None) when git fails."""
    try:
        return _git("rev-parse", "HEAD"), bool(_git("status", "--porcelain", "--untracked-files=no"))
    except StateError:
        return "unknown", None


def _check_member_path(path: str) -> None:
    parts = PurePosixPath(path).parts
    if len(parts) < 2 or parts[0] not in ("data", "outputs") or ".." in parts or "\\" in path:
        raise StateError(f"refusing archive path {path!r}: not a relative path under data/ or outputs/")
    if path.lower().endswith(PICKLE_SUFFIXES):
        raise StateError(f"refusing archive path {path!r}: pickles are never archived (P27)")


def _find(name: str) -> Path:
    _check_name(name)
    for base in (ARCHIVES_DIR, ARCHIVES_DIR / "promoted"):
        if (base / name / "manifest.json").is_file():
            return base / name
    raise StateError(f"no archive named {name!r} (see: list)")


def _check_archive(folder: Path) -> tuple[dict, Path]:
    """Read the manifest and check the tar's members equal it: regular files, safe names, same sizes."""
    tar_path = folder / "state.tar.gz"
    try:
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        want = sorted((f["path"], int(f["bytes"])) for f in manifest["files"])
        with tarfile.open(tar_path, "r:gz") as tar:
            members = tar.getmembers()
    except (OSError, ValueError, KeyError, TypeError, tarfile.TarError) as exc:
        raise StateError(f"archive {folder.name!r} cannot be read: {exc}") from exc
    for path, _ in want:
        _check_member_path(path)
    for member in members:
        if not member.isreg():
            raise StateError(f"refusing archive member {member.name!r}: not a regular file")
        _check_member_path(member.name)
    if sorted((m.name, m.size) for m in members) != want:
        raise StateError(f"archive {folder.name!r}: tar members differ from manifest.json (names or sizes)")
    return manifest, tar_path


# ── archive / restore ──


def archive(name: str, note: str = "") -> Path:
    """Write archives/NAME/{state.tar.gz, manifest.json} from data/ and outputs/ (no pickles); return the folder."""
    _check_name(name)
    final = ARCHIVES_DIR / name
    if final.exists():
        raise StateError(f"archive {name!r} already exists; archives are immutable, pick another name")
    commit, dirty = _git_state()
    now = datetime.now(timezone.utc)
    tmp = ARCHIVES_DIR / f".tmp-{name}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    try:
        files = []
        with tarfile.open(tmp / "state.tar.gz", "w:gz") as tar:
            for prefix, root in _trees():
                for path in _walk(root):
                    if path.name.lower().endswith(PICKLE_SUFFIXES):
                        continue
                    if path.is_symlink():
                        log.warning("archive %s: skipping symlink %s", name, path)
                        continue
                    arcname = f"{prefix}/{path.relative_to(root).as_posix()}"
                    tar.add(path, arcname=arcname, recursive=False)
                    files.append({"path": arcname, "bytes": path.stat().st_size, "sha256": _sha256(path)})
        manifest = {
            "name": name,
            "note": note,
            "created_utc": now.isoformat(timespec="seconds"),
            "created_chicago": now.astimezone(ZoneInfo("America/Chicago")).isoformat(timespec="seconds"),
            "git_commit": commit,
            "git_dirty": dirty,
            "registry_trials": total_trial_count(REGISTRY_PATH),
            "files": files,
        }
        (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        _check_archive(tmp)  # reopen what was written before anything depends on it
        tmp.rename(final)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return final


def restore(name: str) -> dict:
    """Restore archive NAME (local, else promoted) into EMPTY data/ and outputs/; return {name, files, warning}."""
    if not hasattr(tarfile, "data_filter"):
        raise StateError("this Python's tarfile has no data_filter; update to 3.10.12+ / 3.11.4+ before restoring")
    folder = _find(name)
    leftovers = [p for _, root in _trees() for p in _walk(root)]
    if leftovers:
        raise StateError(f"data/ and outputs/ must be empty to restore ({len(leftovers)} files, e.g. {leftovers[0]}); "
                         "run reset first")
    manifest, tar_path = _check_archive(folder)
    staging = ARCHIVES_DIR / f".restore-{name}"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    try:
        try:
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(staging, filter="data")
        except (OSError, tarfile.TarError) as exc:
            raise StateError(f"archive {name!r} refused during extraction: {exc}") from exc
        for entry in manifest["files"]:  # every hash first; nothing moves until all match
            staged = staging / entry["path"]
            if staged.is_symlink() or not staged.is_file() or _sha256(staged) != entry["sha256"]:
                raise StateError(f"sha256 mismatch for {entry['path']} in archive {name!r}; nothing was restored")
        roots = dict(_trees())
        for entry in manifest["files"]:
            prefix, _, rest = entry["path"].partition("/")
            dest = roots[prefix] / rest
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(staging / entry["path"]), str(dest))
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    commit, _ = _git_state()
    warning = None
    if manifest.get("git_commit") != commit:
        warning = f"archive was made at commit {manifest.get('git_commit')}, HEAD is {commit}"
        log.warning("restore %s: %s", name, warning)
    return {"name": name, "files": len(manifest["files"]), "warning": warning}


# ── Command line ──


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Archive, list, reset, restore and promote the platform's generated state")
    sub = parser.add_subparsers(dest="command", required=True)
    p_archive = sub.add_parser("archive", help="write archives/NAME from data/ and outputs/")
    p_archive.add_argument("name")
    p_archive.add_argument("--note", default="")
    sub.add_parser("restore", help="restore NAME into empty data/ and outputs/").add_argument("name")
    args = parser.parse_args(argv)
    try:
        if args.command == "archive":
            folder = archive(args.name, args.note)
            print(f"archived {args.name} -> {folder} ({(folder / 'state.tar.gz').stat().st_size / 1e6:.1f} MB)")
        elif args.command == "restore":
            result = restore(args.name)
            print(f"restored {result['files']} files from {args.name}")
            if result["warning"]:
                print(f"warning: {result['warning']}")
            from trading_crab_lib.platform.report.serving import SERVING_BUILD_COMMAND

            print(f"regime_tilt users only: no model is restored; refit with `{SERVING_BUILD_COMMAND}`")
    except StateError as exc:
        print(f"refused: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
