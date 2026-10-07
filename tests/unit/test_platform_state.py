"""archive / list / reset / restore / promote of the platform's generated state (phase 08.4, K-1..K-8).

Everything runs under tmp_path: the state constants are monkeypatched, promote uses a throwaway
`git init` repo with isolated git config, nothing is pushed, and the real data/, outputs/,
registry/ and git history are never touched. 0 registry rows.
"""

from __future__ import annotations

import io
import json
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
