"""code_version(): the first line every platform CLI prints (Glenn, 2026-10-12)."""

from __future__ import annotations

import subprocess

import pytest

from trading_crab_lib.platform import code_version as cv


def _git(repo, *args):
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    _git(tmp_path, "init", "-q", "-b", "claude/try-me")
    _git(tmp_path, "config", "user.email", "t@example.com")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "f.txt").write_text("one\n")
    _git(tmp_path, "add", "f.txt")
    _git(tmp_path, "commit", "-q", "-m", "first")
    pkg = tmp_path / "src"
    pkg.mkdir()
    monkeypatch.setattr(cv, "_HERE", pkg)
    return tmp_path


def _short_sha(repo):
    done = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo, check=True, capture_output=True, text=True)
    return done.stdout.strip()


def test_names_branch_short_sha_and_checkout(repo):
    assert cv.code_version() == f"Code: claude/try-me @ {_short_sha(repo)} — {repo.resolve()}"


def test_flags_uncommitted_changes_to_tracked_files_only(repo):
    (repo / "scratch.txt").write_text("untracked\n")
    assert "+ uncommitted changes" not in cv.code_version()
    (repo / "f.txt").write_text("two\n")
    assert f"@ {_short_sha(repo)} + uncommitted changes —" in cv.code_version()


def test_detached_head(repo):
    _git(repo, "checkout", "-q", "--detach")
    assert cv.code_version().startswith(f"Code: (detached) @ {_short_sha(repo)} —")


def test_outside_a_git_checkout(tmp_path, monkeypatch):
    monkeypatch.setattr(cv, "_HERE", tmp_path)
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    assert cv.code_version() == f"Code: not a git checkout — {tmp_path}"
