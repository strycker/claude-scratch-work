"""
Start over at will: archive, list, reset, restore and promote the platform's generated state (08.4, K-1..K-8).

    python -m trading_crab_lib.platform.state archive NAME [--note TEXT]   # data/ + outputs/ -> archives/NAME/
    python -m trading_crab_lib.platform.state list | reset [NAME] --yes | restore NAME | promote NAME

``archives/NAME/`` holds ``state.tar.gz`` and ``manifest.json`` (name, note, times, git commit, registry trial
count, ``{path, bytes, sha256}`` per file). Pickles are never archived (P27); ``reset`` deletes them and prints
their paths. ``registry/`` and ``archives/`` are never touched by ``reset``. ``promote`` commits a copy without G-11
live state and tags ``model/NAME``; it never pushes. Stdlib only; no model is loaded and nothing is refit.
"""

from __future__ import annotations

import argparse
import hashlib
import io
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
MAX_PROMOTE_BYTES = 50 * 1024 * 1024
#: G-11 live book / per-machine state (the .gitignore list); a promoted archive never carries them.
LIVE_STATE_STEMS = (
    "allocation_mode", "asset_returns", "executed_weights", "hysteresis_state",
    "nowcaster_class_prior", "regime_belief", "returns_by_regime",
)
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


# ── list / reset / promote ──


def list_archives() -> list[dict]:
    """Local and promoted archives, read from their manifests, sorted by name."""
    found: dict[str, dict] = {}
    for base, promoted in ((ARCHIVES_DIR, False), (ARCHIVES_DIR / "promoted", True)):
        folders = sorted(base.iterdir()) if base.is_dir() else []
        for folder in folders:
            if folder.name.startswith(".") or folder.name == "promoted" or not (folder / "manifest.json").is_file():
                continue
            manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
            row = found.setdefault(folder.name, {"name": folder.name, "promoted": False})
            row["promoted"] = row["promoted"] or promoted
            row.update(
                created=manifest.get("created_utc", ""), size=(folder / "state.tar.gz").stat().st_size,
                commit=str(manifest.get("git_commit", ""))[:7], trials=manifest.get("registry_trials"),
                note=manifest.get("note", ""),
            )
    return [found[k] for k in sorted(found)]


def reset(name: str | None = None) -> tuple[str, list[str]]:
    """Archive first (NAME or auto-<UTC stamp>), then empty data/ and outputs/; return (name, deleted pickle paths)."""
    for protected in (ARCHIVES_DIR, REGISTRY_PATH):
        for _, root in _trees():
            if protected.resolve().is_relative_to(root.resolve()):
                raise StateError(f"refusing to reset: {protected} lies inside {root}")
    name = name or datetime.now(timezone.utc).strftime("auto-%Y%m%d-%H%M%S")
    folder = archive(name, note="made by reset")
    if not (folder / "manifest.json").is_file():
        raise StateError(f"archive {name!r} was not written; nothing was deleted")
    pickles: list[str] = []
    for _, root in _trees():
        for path in _walk(root):
            if path.name.lower().endswith(PICKLE_SUFFIXES):
                pickles.append(str(path))
            path.unlink()
        for sub in sorted((p for p in root.rglob("*") if p.is_dir()), reverse=True):
            if not any(sub.iterdir()):
                sub.rmdir()
    return name, pickles


def promote(name: str) -> str:
    """Re-pack archive NAME without live book state into archives/promoted/NAME, commit and tag it; return the tag."""
    _check_name(name)
    dirty = _git("status", "--porcelain", "--untracked-files=no")
    if dirty:
        raise StateError("refusing to promote: tracked files are modified (commit or revert them first):\n" + dirty)
    src = ARCHIVES_DIR / name
    dest = ARCHIVES_DIR / "promoted" / name
    if not (src / "manifest.json").is_file():
        raise StateError(f"no local archive named {name!r} (see: list)")
    if dest.exists():
        raise StateError(f"{name!r} is already promoted")
    manifest, tar_path = _check_archive(src)
    rel = dest.relative_to(REPO_ROOT).as_posix()
    dest.mkdir(parents=True)
    try:
        kept, excluded = [], []
        with tarfile.open(tar_path, "r:gz") as old, tarfile.open(dest / "state.tar.gz", "w:gz") as new:
            for entry in manifest["files"]:
                base = Path(entry["path"]).name
                if base.split(".")[0] in LIVE_STATE_STEMS or base == "weekly_report.md":
                    excluded.append(entry["path"])
                    continue
                data = old.extractfile(entry["path"]).read()
                if hashlib.sha256(data).hexdigest() != entry["sha256"]:
                    raise StateError(f"sha256 mismatch for {entry['path']} in archive {name!r}")
                new.addfile(old.getmember(entry["path"]), io.BytesIO(data))
                kept.append(entry)
        if (dest / "state.tar.gz").stat().st_size > MAX_PROMOTE_BYTES:
            raise StateError(f"refusing to promote: the archive is over {MAX_PROMOTE_BYTES // 2**20} MB")
        promoted = {**manifest, "files": kept, "excluded": excluded}
        (dest / "manifest.json").write_text(json.dumps(promoted, indent=2) + "\n", encoding="utf-8")
        _git("add", "--", rel)
        _git("commit", "-m", f"chore(archive): promote {name}", "--", rel)
    except (StateError, OSError, KeyError, tarfile.TarError):
        subprocess.run(["git", "reset", "-q", "--", rel], cwd=REPO_ROOT, capture_output=True)
        shutil.rmtree(dest, ignore_errors=True)
        raise
    tag = f"model/{name}"
    summary = f"{manifest['git_commit'][:7]}, {manifest['registry_trials']} trials, {manifest.get('note', '')}"
    _git("tag", "-a", tag, "-m", f"{name}: {summary}")
    return tag


# ── Command line ──


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Archive, list, reset, restore and promote the platform's generated state")
    sub = parser.add_subparsers(dest="command", required=True)
    p_archive = sub.add_parser("archive", help="write archives/NAME from data/ and outputs/")
    p_archive.add_argument("name")
    p_archive.add_argument("--note", default="")
    sub.add_parser("list", help="show local and promoted archives")
    p_reset = sub.add_parser("reset", help="archive, then empty data/ and outputs/ (needs --yes)")
    p_reset.add_argument("name", nargs="?")
    p_reset.add_argument("--yes", action="store_true")
    sub.add_parser("restore", help="restore NAME into empty data/ and outputs/").add_argument("name")
    sub.add_parser("promote", help="commit NAME (no live book state) and tag model/NAME; never pushes").add_argument("name")
    args = parser.parse_args(argv)
    try:
        if args.command == "archive":
            folder = archive(args.name, args.note)
            print(f"archived {args.name} -> {folder} ({(folder / 'state.tar.gz').stat().st_size / 1e6:.1f} MB)")
        elif args.command == "list":
            for row in list_archives():
                promoted = "yes" if row["promoted"] else "no"
                print(f"{row['name']:<28} {row['created'][:19]} {row['size'] / 1e6:6.1f} MB promoted={promoted:<3} "
                      f"{row['commit']} trials={row['trials']} {row['note']}")
        elif args.command == "reset":
            if not args.yes:
                print(f"reset would empty {DATA_DIR} and {OUTPUT_DIR} (after archiving them); re-run with --yes")
                return 1
            archived, pickles = reset(args.name)
            for path in pickles:
                print(f"deleted pickle (never archived): {path}")
            print(f"reset done; everything else is in archive {archived}. Undo: python -m trading_crab_lib.platform.state "
                  f"restore {archived}")
        elif args.command == "promote":
            tag = promote(args.name)
            print(f"promoted {args.name}: committed and tagged {tag}. Not pushed; to publish: git push origin {tag}")
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
