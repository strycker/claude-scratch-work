"""Unit tests for trading_crab_lib.platform.snapshots (Task 3, 260806-u89).

Every test uses tmp_path for both the snapshot dir and the CheckpointManager
— no network, no reads of the real data/ tree.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
import pytest

from trading_crab_lib.checkpoints import CheckpointManager
from trading_crab_lib.platform.snapshots import (
    DEFAULT_SNAPSHOT_NAMES,
    export_snapshot,
    is_snapshot_backed,
    load_snapshot,
    read_manifest,
    restore_snapshot,
    snapshots_enabled,
)


def _make_df(periods: int = 6, cols: tuple[str, ...] = ("a", "b")) -> pd.DataFrame:
    idx = pd.date_range("2020-01-31", periods=periods, freq="ME")
    rng = np.random.default_rng(0)
    return pd.DataFrame({c: rng.uniform(0, 100, periods) for c in cols}, index=idx)


@pytest.fixture
def cm(tmp_path) -> CheckpointManager:
    return CheckpointManager(checkpoint_dir=tmp_path / "platform")


@pytest.fixture
def snap_dir(tmp_path):
    return tmp_path / "snapshot"


class TestSnapshotsEnabled:
    def test_disabled_when_unset(self, monkeypatch):
        monkeypatch.delenv("TC_USE_SNAPSHOTS", raising=False)
        assert snapshots_enabled() is False

    def test_enabled_when_truthy_1(self, monkeypatch):
        monkeypatch.setenv("TC_USE_SNAPSHOTS", "1")
        assert snapshots_enabled() is True

    def test_enabled_when_truthy_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("TC_USE_SNAPSHOTS", "True")
        assert snapshots_enabled() is True

    def test_disabled_for_other_values(self, monkeypatch):
        monkeypatch.setenv("TC_USE_SNAPSHOTS", "0")
        assert snapshots_enabled() is False


class TestExportSnapshot:
    def test_round_trips_via_load_snapshot(self, cm, snap_dir):
        df = _make_df()
        cm.save(df, "daily_raw")

        export_snapshot(names=["daily_raw"], cm=cm, snapshot_dir=snap_dir)
        reloaded = load_snapshot("daily_raw", snapshot_dir=snap_dir)

        pd.testing.assert_frame_equal(reloaded, df, check_freq=False)

    def test_writes_manifest_with_required_fields(self, cm, snap_dir):
        df = _make_df()
        cm.save(df, "daily_raw")

        export_snapshot(names=["daily_raw"], cm=cm, snapshot_dir=snap_dir)
        manifest = read_manifest(snapshot_dir=snap_dir)

        assert "captured_at" in manifest
        assert "git_commit" in manifest
        assert manifest["names"]["daily_raw"]["rows"] == len(df)
        assert manifest["names"]["daily_raw"]["columns"] == len(df.columns)
        assert manifest["names"]["daily_raw"]["col_names"] == list(df.columns)
        assert manifest["names"]["daily_raw"]["index_start"] == str(df.index[0])
        assert manifest["names"]["daily_raw"]["index_end"] == str(df.index[-1])

    def test_includes_splice_provenance_when_present(self, cm, snap_dir):
        df = _make_df()
        cm.save(df, "monthly_raw")
        provenance = {"gold": {"status": "primary"}}
        (cm.dir / "splice_provenance.json").write_text(
            json.dumps({"captured_at": "2020-01-01T00:00:00", "provenance": provenance})
        )

        export_snapshot(names=["monthly_raw"], cm=cm, snapshot_dir=snap_dir)
        manifest = read_manifest(snapshot_dir=snap_dir)

        assert manifest["source_provenance"]["provenance"] == provenance

    def test_no_provenance_key_when_absent(self, cm, snap_dir):
        cm.save(_make_df(), "monthly_raw")

        export_snapshot(names=["monthly_raw"], cm=cm, snapshot_dir=snap_dir)
        manifest = read_manifest(snapshot_dir=snap_dir)

        assert "source_provenance" not in manifest

    def test_missing_checkpoint_skipped_with_warning_others_still_export(self, cm, snap_dir, caplog):
        cm.save(_make_df(), "monthly_raw")  # only monthly_raw exists; daily_raw does not

        with caplog.at_level(logging.WARNING):
            export_snapshot(names=["daily_raw", "monthly_raw"], cm=cm, snapshot_dir=snap_dir)

        assert not (snap_dir / "daily_raw.parquet").exists()
        assert (snap_dir / "monthly_raw.parquet").exists()
        assert any("daily_raw" in r.message for r in caplog.records)

    def test_default_names_used_when_none_given(self, cm, snap_dir):
        for name in DEFAULT_SNAPSHOT_NAMES:
            cm.save(_make_df(), name)

        export_snapshot(cm=cm, snapshot_dir=snap_dir)

        for name in DEFAULT_SNAPSHOT_NAMES:
            assert (snap_dir / f"{name}.parquet").exists()

    def test_daily_start_truncates_daily_raw_only(self, cm, snap_dir):
        idx = pd.date_range("2000-01-31", periods=12, freq="ME")
        daily = pd.DataFrame({"a": np.arange(12.0)}, index=idx)
        cm.save(daily, "daily_raw")
        cm.save(daily.copy(), "monthly_raw")

        export_snapshot(
            names=["daily_raw", "monthly_raw"], cm=cm, snapshot_dir=snap_dir, daily_start="2000-07-01",
        )

        daily_reloaded = load_snapshot("daily_raw", snapshot_dir=snap_dir)
        monthly_reloaded = load_snapshot("monthly_raw", snapshot_dir=snap_dir)
        assert daily_reloaded.index.min() >= pd.Timestamp("2000-07-01")
        assert monthly_reloaded.index.min() == idx[0]  # not truncated

    def test_refuses_to_exceed_max_mb_and_names_largest_file(self, cm, snap_dir):
        big_idx = pd.date_range("2000-01-01", periods=5000, freq="D")
        big = pd.DataFrame(
            {f"col_{i}": np.random.default_rng(i).uniform(0, 1, len(big_idx)) for i in range(20)},
            index=big_idx,
        )
        cm.save(big, "daily_raw")

        with pytest.raises(ValueError, match="exceeds"):
            export_snapshot(names=["daily_raw"], cm=cm, snapshot_dir=snap_dir, max_mb=0.001)

        # Nothing left behind — the oversized export is fully rolled back.
        assert not (snap_dir / "daily_raw.parquet").exists()
        assert not (snap_dir / "manifest.json").exists()

    def test_returns_snapshot_dir(self, cm, snap_dir):
        cm.save(_make_df(), "daily_raw")
        result = export_snapshot(names=["daily_raw"], cm=cm, snapshot_dir=snap_dir)
        assert result == snap_dir


class TestLoadSnapshot:
    def test_missing_snapshot_raises(self, snap_dir):
        snap_dir.mkdir(parents=True)
        with pytest.raises(FileNotFoundError):
            load_snapshot("daily_raw", snapshot_dir=snap_dir)

    def test_every_call_logs_warning_with_capture_timestamp(self, cm, snap_dir, caplog):
        cm.save(_make_df(), "daily_raw")
        export_snapshot(names=["daily_raw"], cm=cm, snapshot_dir=snap_dir)
        manifest = read_manifest(snapshot_dir=snap_dir)

        with caplog.at_level(logging.WARNING):
            load_snapshot("daily_raw", snapshot_dir=snap_dir)

        warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any(manifest["captured_at"] in m for m in warnings)


class TestRestoreSnapshot:
    def test_restores_parquet_and_meta_with_marker(self, cm, snap_dir):
        df = _make_df()
        cm.save(df, "daily_raw")
        export_snapshot(names=["daily_raw"], cm=cm, snapshot_dir=snap_dir)
        cm.clear("daily_raw")  # simulate a fresh checkout with no live checkpoint

        restored_names = restore_snapshot(names=["daily_raw"], cm=cm, snapshot_dir=snap_dir)

        assert restored_names == ["daily_raw"]
        reloaded = cm.load("daily_raw")
        pd.testing.assert_frame_equal(reloaded, df, check_freq=False)
        meta = json.loads((cm.dir / "daily_raw.meta.json").read_text())
        assert meta["source"] == "offline-snapshot"
        assert "snapshot_captured_at" in meta

    def test_logs_one_warning_banner_naming_every_checkpoint_replaced(self, cm, snap_dir, caplog):
        cm.save(_make_df(), "daily_raw")
        cm.save(_make_df(), "monthly_raw")
        export_snapshot(names=["daily_raw", "monthly_raw"], cm=cm, snapshot_dir=snap_dir)

        with caplog.at_level(logging.WARNING):
            restore_snapshot(names=["daily_raw", "monthly_raw"], cm=cm, snapshot_dir=snap_dir)

        banner_logs = [r for r in caplog.records if "replaced checkpoint" in r.message]
        assert len(banner_logs) == 1
        assert "daily_raw" in banner_logs[0].message
        assert "monthly_raw" in banner_logs[0].message

    def test_is_snapshot_backed_true_after_restore_false_for_ordinary_save(self, cm, snap_dir):
        df = _make_df()
        cm.save(df, "daily_raw")
        export_snapshot(names=["daily_raw"], cm=cm, snapshot_dir=snap_dir)

        assert is_snapshot_backed(cm, "daily_raw") is False  # ordinary save, no marker yet

        restore_snapshot(names=["daily_raw"], cm=cm, snapshot_dir=snap_dir)
        assert is_snapshot_backed(cm, "daily_raw") is True

        cm.save(df, "daily_raw")  # a fresh live save always writes a fresh meta — marker clears
        assert is_snapshot_backed(cm, "daily_raw") is False

    def test_missing_name_skipped_not_error(self, cm, snap_dir):
        snap_dir.mkdir(parents=True)
        (snap_dir / "manifest.json").write_text(json.dumps({"captured_at": "x", "names": {}}))

        restored = restore_snapshot(names=["daily_raw"], cm=cm, snapshot_dir=snap_dir)

        assert restored == []

    def test_is_snapshot_backed_false_for_missing_or_unparseable_meta(self, cm):
        assert is_snapshot_backed(cm, "does_not_exist") is False
        (cm.dir / "broken.meta.json").write_text("not json")
        assert is_snapshot_backed(cm, "broken") is False
