"""archive / list / reset / restore / promote of the platform's generated state (phase 08.4, K-1..K-8).

Everything runs under tmp_path: the state constants are monkeypatched, promote uses a throwaway
`git init` repo with isolated git config, nothing is pushed, and the real data/, outputs/,
registry/ and git history are never touched. 0 registry rows.
"""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import subprocess
import tarfile

import pytest

from trading_crab_lib.platform import state

WORLD = {
    "data/checkpoints/platform/monthly_raw.parquet": b"parquet-bytes",
    "data/checkpoints/platform/monthly_raw.meta.json": b'{"rows": 3}',
    "data/checkpoints/platform/nowcaster.pkl": b"PICKLE",
    "data/legacy_labels.pickle": b"OLD PICKLE",
    "data/checkpoints/platform/executed_weights.parquet": b"the live book",
    "data/checkpoints/platform/executed_weights.meta.json": b"{}",
    "outputs/reports/platform/weekly_report.md": b"# the weekly page\n",
    "outputs/reports/platform/backtest.csv": b"a,b\n1,2\n",
}


def _write_world(root):
    for rel, data in WORLD.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    (root / "data/snapshots/platform").mkdir(parents=True, exist_ok=True)
    (root / "data/snapshots/platform/.gitkeep").write_bytes(b"")
    reg = root / "registry"
    reg.mkdir(exist_ok=True)
    (reg / "trials.jsonl").write_text('{"trial_id": "t1"}\n{"trial_id": "t2"}\n', encoding="utf-8")


def _point_state_at(monkeypatch, root):
    monkeypatch.setattr(state, "REPO_ROOT", root)
    monkeypatch.setattr(state, "DATA_DIR", root / "data")
    monkeypatch.setattr(state, "OUTPUT_DIR", root / "outputs")
    monkeypatch.setattr(state, "ARCHIVES_DIR", root / "archives")
    monkeypatch.setattr(state, "REGISTRY_PATH", root / "registry" / "trials.jsonl")


@pytest.fixture
def state_env(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    _write_world(root)
    _point_state_at(monkeypatch, root)
    assert state.DATA_DIR.is_relative_to(tmp_path) and state.ARCHIVES_DIR.is_relative_to(tmp_path)
    assert state.OUTPUT_DIR.is_relative_to(tmp_path) and state.REGISTRY_PATH.is_relative_to(tmp_path)
    assert state.REPO_ROOT.is_relative_to(tmp_path)
    return root


def _snapshot(root):
    """{relative path: bytes} of every file under data/ and outputs/ (the .gitkeep included)."""
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for tree in ("data", "outputs")
        for p in (root / tree).rglob("*")
        if p.is_file()
    }


def _empty_trees(root):
    for tree in ("data", "outputs"):
        for p in (root / tree).rglob("*"):
            if p.is_file() and p.name != ".gitkeep":
                p.unlink()


def _is_pickle(path):
    return path.endswith(state.PICKLE_SUFFIXES)


def test_cli_round_trip_is_byte_identical_minus_pickles(state_env, capsys):
    before = _snapshot(state_env)

    assert state.main(["archive", "first", "--note", "round trip"]) == 0
    manifest = json.loads((state_env / "archives/first/manifest.json").read_text())
    assert set(manifest) == {
        "name", "note", "created_utc", "created_chicago", "git_commit", "git_dirty", "registry_trials", "files",
    }
    assert manifest["name"] == "first" and manifest["note"] == "round trip" and manifest["registry_trials"] == 2
    assert manifest["created_utc"].endswith("+00:00")
    assert manifest["created_chicago"][-6:] in ("-05:00", "-06:00")
    paths = [f["path"] for f in manifest["files"]]
    assert paths == sorted(paths)
    assert not any(_is_pickle(p) for p in paths) and not any(p.endswith(".gitkeep") for p in paths)
    assert all(set(f) == {"path", "bytes", "sha256"} for f in manifest["files"])
    with tarfile.open(state_env / "archives/first/state.tar.gz") as tar:
        assert tar.getnames() == paths

    _empty_trees(state_env)
    assert state.main(["restore", "first"]) == 0
    out = capsys.readouterr().out
    assert "python -m trading_crab_lib.platform.report.serving" in out

    expected = {k: v for k, v in before.items() if not _is_pickle(k)}
    assert _snapshot(state_env) == expected
    assert not (state.ARCHIVES_DIR / ".restore-first").exists()


@pytest.mark.parametrize("bad", ["", "Upper", "-lead", "has space", "a/b", "../x", "promoted", ".hidden"])
def test_bad_names_are_refused(state_env, bad):
    with pytest.raises(state.StateError):
        state.archive(bad)
    assert not state.ARCHIVES_DIR.exists()


def test_an_existing_name_is_refused_and_the_archive_untouched(state_env):
    state.archive("once")
    tar_bytes = (state.ARCHIVES_DIR / "once/state.tar.gz").read_bytes()
    with pytest.raises(state.StateError, match="already exists"):
        state.archive("once", note="again")
    assert (state.ARCHIVES_DIR / "once/state.tar.gz").read_bytes() == tar_bytes


def test_restore_refuses_a_non_empty_tree(state_env):
    state.archive("full")
    with pytest.raises(state.StateError, match="reset"):
        state.restore("full")  # the world is still there


def test_a_tampered_manifest_hash_is_refused_and_nothing_moves(state_env):
    state.archive("tamper")
    manifest_path = state.ARCHIVES_DIR / "tamper/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][0]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    _empty_trees(state_env)

    with pytest.raises(state.StateError, match="sha256 mismatch"):
        state.restore("tamper")

    assert _snapshot(state_env) == {"data/snapshots/platform/.gitkeep": b""}
    assert not (state.ARCHIVES_DIR / ".restore-tamper").exists()


def _handmade_archive(root, name, members, manifest_files):
    folder = root / "archives" / name
    folder.mkdir(parents=True)
    with tarfile.open(folder / "state.tar.gz", "w:gz") as tar:
        for member_name, data in members:
            info = tarfile.TarInfo(member_name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    (folder / "manifest.json").write_text(json.dumps({"name": name, "files": manifest_files}))


def test_a_path_traversal_member_is_refused_and_nothing_is_written_outside(state_env, tmp_path):
    _handmade_archive(
        state_env, "evil", [("../evil", b"x")], [{"path": "../evil", "bytes": 1, "sha256": "0" * 64}]
    )
    _empty_trees(state_env)

    with pytest.raises(state.StateError, match="refusing"):
        state.restore("evil")

    assert not (state_env.parent / "evil").exists() and not (tmp_path / "evil").exists()
    assert _snapshot(state_env) == {"data/snapshots/platform/.gitkeep": b""}


def test_members_missing_from_the_manifest_or_pickles_or_links_are_refused(state_env):
    good = [("data/a.csv", b"1")]
    entry = {"path": "data/a.csv", "bytes": 1, "sha256": "0" * 64}
    _handmade_archive(state_env, "extra", good + [("data/b.csv", b"2")], [entry])
    _handmade_archive(state_env, "pickle", [("data/m.pkl", b"2")], [{**entry, "path": "data/m.pkl"}])
    link = state_env / "archives/link"
    link.mkdir(parents=True)
    with tarfile.open(link / "state.tar.gz", "w:gz") as tar:
        info = tarfile.TarInfo("data/a.csv")
        info.type, info.linkname = tarfile.SYMTYPE, "/etc/passwd"
        tar.addfile(info)
    (link / "manifest.json").write_text(json.dumps({"name": "link", "files": [{**entry, "bytes": 0}]}))
    _empty_trees(state_env)

    for name in ("extra", "pickle", "link"):
        with pytest.raises(state.StateError):
            state.restore(name)
    assert _snapshot(state_env) == {"data/snapshots/platform/.gitkeep": b""}


def test_restore_is_refused_when_tarfile_has_no_data_filter(state_env, monkeypatch):
    state.archive("nofilter")
    _empty_trees(state_env)
    monkeypatch.delattr(tarfile, "data_filter")

    with pytest.raises(state.StateError, match="data_filter"):
        state.restore("nofilter")
    assert _snapshot(state_env) == {"data/snapshots/platform/.gitkeep": b""}


# ── list and reset ──


def _tree_hash(folder):
    return {p.relative_to(folder).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(folder.rglob("*")) if p.is_file()}


def test_reset_without_yes_changes_nothing(state_env, capsys):
    before = _snapshot(state_env)

    assert state.main(["reset"]) == 1

    assert "--yes" in capsys.readouterr().out
    assert _snapshot(state_env) == before and not state.ARCHIVES_DIR.exists()


def test_reset_archives_first_empties_the_trees_and_restore_brings_it_back(state_env, capsys):
    before = _snapshot(state_env)
    registry_before = _tree_hash(state_env / "registry")
    state.archive("keep-me")
    archives_before = _tree_hash(state.ARCHIVES_DIR)

    assert state.main(["reset", "--yes"]) == 0

    out = capsys.readouterr().out
    assert "nowcaster.pkl" in out and "legacy_labels.pickle" in out  # deleted pickles are named
    assert _snapshot(state_env) == {"data/snapshots/platform/.gitkeep": b""}
    assert not (state_env / "outputs/reports").exists()  # emptied sub-directories are removed
    assert _tree_hash(state_env / "registry") == registry_before
    after = _tree_hash(state.ARCHIVES_DIR)
    assert {k: v for k, v in after.items() if k.startswith("keep-me/")} == archives_before  # untouched
    auto = [row["name"] for row in state.list_archives() if row["name"].startswith("auto-")]
    assert len(auto) == 1 and f"restore {auto[0]}" in out

    state.restore(auto[0])
    assert _snapshot(state_env) == {k: v for k, v in before.items() if not _is_pickle(k)}


def test_reset_with_a_name_uses_it_and_refuses_when_registry_or_archives_sit_in_a_tree(state_env, monkeypatch):
    assert state.reset("named")[0] == "named"
    assert (state.ARCHIVES_DIR / "named/manifest.json").is_file()

    _write_world(state_env)
    monkeypatch.setattr(state, "ARCHIVES_DIR", state.DATA_DIR / "archives")
    with pytest.raises(state.StateError, match="inside"):
        state.reset("inside")
    monkeypatch.setattr(state, "ARCHIVES_DIR", state_env / "archives")
    monkeypatch.setattr(state, "REGISTRY_PATH", state.OUTPUT_DIR / "trials.jsonl")
    with pytest.raises(state.StateError, match="inside"):
        state.reset("inside")
    assert (state_env / "data/checkpoints/platform/monthly_raw.parquet").exists()


def test_list_shows_name_size_promoted_commit_trials_and_note(state_env, capsys):
    state.archive("alpha", note="first one")
    assert state.main(["list"]) == 0
    line = capsys.readouterr().out.strip()
    assert line.startswith("alpha") and "promoted=no" in line and "trials=2" in line and "first one" in line
    assert [row["name"] for row in state.list_archives()] == ["alpha"]


# ── promote, in a throwaway git repo ──


@pytest.fixture
def git_env(state_env, monkeypatch):
    """state_env turned into an isolated git repo; no global config, no signing, nothing pushed."""
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "/dev/null")
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    root = state_env

    def git(*args):
        done = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True)
        return done.stdout.strip()

    git("init", "-q")
    for key, value in (("user.name", "Test"), ("user.email", "test@example.com"),
                       ("commit.gpgsign", "false"), ("tag.gpgsign", "false")):
        git("config", key, value)
    (root / ".gitignore").write_text("archives/*\n!archives/promoted/\n", encoding="utf-8")
    (root / "tracked.txt").write_text("one\n", encoding="utf-8")
    git("add", ".gitignore", "tracked.txt")
    git("commit", "-q", "-m", "initial")
    git.root = root
    return git


def test_promote_commits_only_the_promoted_folder_tags_it_and_drops_live_state(git_env):
    root = git_env.root
    state.archive("model-a", note="for the tag")
    head_before = git_env("rev-parse", "HEAD")

    assert state.promote("model-a") == "model/model-a"

    assert git_env("rev-parse", "HEAD~1") == head_before
    assert set(git_env("show", "--name-only", "--format=", "HEAD").splitlines()) == {
        "archives/promoted/model-a/manifest.json", "archives/promoted/model-a/state.tar.gz"}
    assert git_env("cat-file", "-t", "model/model-a") == "tag"
    assert "for the tag" in git_env("tag", "-l", "-n5", "model/model-a")
    with tarfile.open(root / "archives/promoted/model-a/state.tar.gz") as tar:
        names = tar.getnames()
    assert not any("executed_weights" in n or n.endswith("weekly_report.md") or _is_pickle(n) for n in names)
    assert "data/checkpoints/platform/monthly_raw.parquet" in names
    promoted = json.loads((root / "archives/promoted/model-a/manifest.json").read_text())
    assert sorted(promoted["excluded"]) == ["data/checkpoints/platform/executed_weights.meta.json",
                                            "data/checkpoints/platform/executed_weights.parquet",
                                            "outputs/reports/platform/weekly_report.md"]
    assert [f["path"] for f in promoted["files"]] == names
    assert subprocess.run(["git", "check-ignore", "-q", "archives/model-a/state.tar.gz"], cwd=root).returncode == 0
    assert subprocess.run(["git", "check-ignore", "-q", "archives/promoted/model-a/state.tar.gz"], cwd=root).returncode == 1
    assert [row["promoted"] for row in state.list_archives()] == [True]

    _empty_trees(root)
    shutil.rmtree(root / "archives/model-a")  # only the promoted copy is left, as on a fresh clone
    assert state.restore("model-a")["files"] == len(names)


def test_promote_refuses_a_dirty_tracked_file_a_missing_or_promoted_name_and_an_oversized_archive(git_env, monkeypatch):
    state.archive("model-b")
    head = git_env("rev-parse", "HEAD")
    (git_env.root / "tracked.txt").write_text("changed\n", encoding="utf-8")
    with pytest.raises(state.StateError, match="tracked.txt"):
        state.promote("model-b")
    git_env("checkout", "--", "tracked.txt")

    with pytest.raises(state.StateError, match="no local archive"):
        state.promote("nothing-here")

    with monkeypatch.context() as m:  # scoped, so the fixture's path patches stay in force
        m.setattr(state, "MAX_PROMOTE_BYTES", 10)
        with pytest.raises(state.StateError, match="MB|over"):
            state.promote("model-b")
    assert not (state.ARCHIVES_DIR / "promoted/model-b").exists()

    assert git_env("rev-parse", "HEAD") == head and git_env("tag", "-l", "model/*") == ""
    assert git_env("status", "--porcelain", "--untracked-files=no") == ""


def test_a_failed_commit_unstages_and_removes_the_promoted_folder(git_env):
    state.archive("model-c")
    hook = git_env.root / ".git/hooks/pre-commit"
    hook.write_text("#!/bin/sh\necho nope >&2\nexit 1\n", encoding="utf-8")
    hook.chmod(0o755)
    head = git_env("rev-parse", "HEAD")

    with pytest.raises(state.StateError, match="nope"):
        state.promote("model-c")

    assert git_env("rev-parse", "HEAD") == head and git_env("tag", "-l", "model/*") == ""
    assert not (state.ARCHIVES_DIR / "promoted/model-c").exists()
    assert git_env("diff", "--cached", "--name-only") == ""


# ── review fixes (2026-10-07) ──


@pytest.mark.parametrize("bad", ["a..b", "x.lock", "x.", "Upper", "promoted", ".hidden"])
def test_names_that_are_not_valid_git_tags_are_refused_before_anything_is_written(state_env, bad):
    with pytest.raises(state.StateError, match="bad archive name"):
        state.archive(bad)
    assert not state.ARCHIVES_DIR.exists()


def test_promote_leaves_notebook_scratch_out(git_env):
    scratch = state.DATA_DIR / "checkpoints/platform_notebook/scratch.parquet"
    scratch.parent.mkdir(parents=True, exist_ok=True)
    scratch.write_bytes(b"scratch")
    state.archive("model-d")
    state.promote("model-d")
    manifest = json.loads((state.ARCHIVES_DIR / "promoted/model-d/manifest.json").read_text(encoding="utf-8"))
    assert "data/checkpoints/platform_notebook/scratch.parquet" in manifest["excluded"]
    assert all(not f["path"].startswith("data/checkpoints/platform_notebook/") for f in manifest["files"])


def test_list_survives_an_unreadable_archive(state_env):
    state.archive("good")
    broken = state.ARCHIVES_DIR / "broken"
    broken.mkdir()
    (broken / "manifest.json").write_text("{not json", encoding="utf-8")
    rows = {row["name"]: row for row in state.list_archives()}
    assert rows["good"]["trials"] is not None
    assert rows["broken"]["note"].startswith("UNREADABLE")
