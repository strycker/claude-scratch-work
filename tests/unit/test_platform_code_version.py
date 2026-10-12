"""code_version(): the first line every platform CLI prints (Glenn, 2026-10-12)."""

from __future__ import annotations

import shutil
import subprocess

import pytest

from trading_crab_lib.platform import code_version as cv


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """An isolated git repo whose src/ holds a tracked copy of the module (no global config, no signing)."""
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "/dev/null")
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")

    def git(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q", "-b", "claude/try-me")
    for key, value in (("user.name", "t"), ("user.email", "t@example.com"), ("commit.gpgsign", "false")):
        git("config", key, value)
    pkg = tmp_path / "src"
    pkg.mkdir()
    (pkg / "code_version.py").write_text("one\n")
    git("add", "src/code_version.py")
    git("commit", "-q", "-m", "first")
    monkeypatch.setattr(cv, "_HERE", pkg)
    git.root = tmp_path
    return git


def _short_sha(root):
    done = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root, check=True, capture_output=True, text=True)
    return done.stdout.strip()


def test_names_branch_short_sha_and_checkout(repo):
    assert cv.code_version() == f"Code: claude/try-me @ {_short_sha(repo.root)} — {repo.root.resolve()}"


def test_flags_edits_and_new_files_under_the_code_folders_only(repo):
    (repo.root / "notes.txt").write_text("scratch at the top level\n")
    assert "+ uncommitted changes" not in cv.code_version()
    (repo.root / "src" / "new_module.py").write_text("x = 1\n")
    assert f"@ {_short_sha(repo.root)} + uncommitted changes —" in cv.code_version()
    (repo.root / "src" / "new_module.py").unlink()
    (repo.root / "src" / "code_version.py").write_text("two\n")
    assert "+ uncommitted changes" in cv.code_version()


def test_detached_head(repo):
    repo("checkout", "-q", "--detach")
    assert cv.code_version().startswith(f"Code: (detached) @ {_short_sha(repo.root)} —")


def test_an_untracked_installed_copy_inside_a_checkout_does_not_borrow_its_commit(repo, monkeypatch):
    site = repo.root / ".venv" / "site-packages"  # untracked, like a non-editable install in the repo's venv
    site.mkdir(parents=True)
    (site / "code_version.py").write_text("one\n")
    monkeypatch.setattr(cv, "_HERE", site)
    assert cv.code_version() == f"Code: an installed copy, not tracked by the checkout around it — {site}"


def test_unknown_status_is_not_reported_as_clean(repo, monkeypatch):
    real = cv._git
    monkeypatch.setattr(cv, "_git", lambda *args, cwd=None: None if args[0] == "status" else real(*args, cwd=cwd))
    assert cv.code_version().startswith(f"Code: claude/try-me @ {_short_sha(repo.root)} (uncommitted changes unknown) —")


def test_outside_a_git_checkout(tmp_path, monkeypatch):
    monkeypatch.setattr(cv, "_HERE", tmp_path)
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    assert cv.code_version() == f"Code: not a git checkout (or git unavailable) — {tmp_path}"


@pytest.mark.parametrize("error", [FileNotFoundError("git"), subprocess.TimeoutExpired("git", 5)])
def test_missing_or_hung_git_never_raises(monkeypatch, error):
    def boom(*args, **kwargs):
        raise error

    monkeypatch.setattr(cv.subprocess, "run", boom)
    assert cv.code_version().startswith("Code: not a git checkout (or git unavailable) — ")
