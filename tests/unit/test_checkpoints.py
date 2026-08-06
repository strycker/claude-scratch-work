"""Unit tests for src/trading_crab_lib/io/checkpoints.py"""

import json
import logging

import numpy as np
import pandas as pd
import pytest

from trading_crab_lib.checkpoints import (
    MERGE_ON_SAVE_CHECKPOINT_NAMES,
    PRESERVATION_CHECKPOINT_NAMES,
    CheckpointManager,
    merge_preserving,
    preservation_checkpoint_should_write,
)


@pytest.fixture
def cm(tmp_path):
    """A CheckpointManager backed by a temp directory."""
    return CheckpointManager(checkpoint_dir=tmp_path)


@pytest.fixture
def sample_df(quarterly_index):
    return pd.DataFrame(
        {"a": np.arange(20, dtype=float), "b": np.ones(20)},
        index=quarterly_index,
    )


# ── save / load round-trip ─────────────────────────────────────────────────

class TestSaveLoad:
    def test_save_creates_parquet(self, cm, sample_df):
        cm.save(sample_df, "test")
        assert (cm.dir / "test.parquet").exists()

    def test_save_creates_meta(self, cm, sample_df):
        cm.save(sample_df, "test")
        assert (cm.dir / "test.meta.json").exists()

    def test_load_round_trip(self, cm, sample_df):
        cm.save(sample_df, "test")
        loaded = cm.load("test")
        # Parquet does not preserve DatetimeIndex frequency; compare values only
        pd.testing.assert_frame_equal(loaded, sample_df, check_freq=False)

    def test_load_missing_raises(self, cm):
        with pytest.raises(FileNotFoundError):
            cm.load("does_not_exist")

    def test_save_returns_path(self, cm, sample_df):
        path = cm.save(sample_df, "test")
        assert path.exists()
        assert path.suffix == ".parquet"


# ── is_fresh ───────────────────────────────────────────────────────────────

class TestIsFresh:
    def test_fresh_after_save(self, cm, sample_df):
        cm.save(sample_df, "test")
        assert cm.is_fresh("test", max_age_days=1.0)

    def test_missing_checkpoint_not_fresh(self, cm):
        assert not cm.is_fresh("nonexistent")

    def test_stale_by_age(self, cm, sample_df):
        cm.save(sample_df, "test")
        # max_age_days=0 means any age is stale
        assert not cm.is_fresh("test", max_age_days=0.0)


# ── clear ──────────────────────────────────────────────────────────────────

class TestClear:
    def test_clear_removes_files(self, cm, sample_df):
        cm.save(sample_df, "test")
        cm.clear("test")
        assert not (cm.dir / "test.parquet").exists()
        assert not (cm.dir / "test.meta.json").exists()

    def test_clear_nonexistent_does_not_raise(self, cm):
        cm.clear("nonexistent")  # should not raise

    def test_clear_all_removes_non_preservation(self, cm, sample_df):
        cm.save(sample_df, "a")
        cm.save(sample_df, "b")
        cm.clear_all()
        assert not list(cm.dir.iterdir())

    def test_clear_all_preserves_secondary(self, cm, sample_df):
        cm.save(sample_df, "a")
        cm.save(sample_df, "macro_raw_secondary")
        cm.clear_all()
        remaining = {f.name for f in cm.dir.iterdir()}
        assert "macro_raw_secondary.parquet" in remaining
        assert "macro_raw_secondary.meta.json" in remaining
        assert "a.parquet" not in remaining

    def test_clear_all_with_include_preservation(self, cm, sample_df):
        cm.save(sample_df, "a")
        cm.save(sample_df, "macro_raw_secondary")
        cm.clear_all(include_preservation=True)
        assert not list(cm.dir.iterdir())


# ── list / summary ─────────────────────────────────────────────────────────

class TestList:
    def test_list_empty_when_no_checkpoints(self, cm):
        assert not cm.list()

    def test_list_returns_metadata(self, cm, sample_df):
        cm.save(sample_df, "test")
        entries = cm.list()
        assert len(entries) == 1
        assert entries[0]["name"] == "test"
        assert entries[0]["rows"] == len(sample_df)
        assert entries[0]["columns"] == len(sample_df.columns)

    def test_list_sorted_by_creation(self, cm, sample_df):
        cm.save(sample_df, "first")
        cm.save(sample_df, "second")
        entries = cm.list()
        names = [e["name"] for e in entries]
        assert names.index("first") < names.index("second")

    def test_summary_string(self, cm, sample_df):
        cm.save(sample_df, "test")
        summary = cm.summary()
        assert "test" in summary
        assert "20" in summary  # row count


# ── model checkpoints ──────────────────────────────────────────────────────

class TestModelCheckpoints:
    def test_save_load_model(self, cm):
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=2, random_state=0)
        cm.save_model(model, "rf")
        loaded = cm.load_model("rf")
        assert hasattr(loaded, "predict")

    def test_load_missing_model_raises(self, cm):
        with pytest.raises(FileNotFoundError):
            cm.load_model("nonexistent")

    def test_model_exists(self, cm):
        from sklearn.ensemble import RandomForestClassifier
        assert not cm.model_exists("rf")
        cm.save_model(RandomForestClassifier(), "rf")
        assert cm.model_exists("rf")


# ── preservation checkpoints ──────────────────────────────────────────────

class TestPreservationCheckpoints:
    def test_names_frozenset(self):
        assert isinstance(PRESERVATION_CHECKPOINT_NAMES, frozenset)
        assert "macro_raw_secondary" in PRESERVATION_CHECKPOINT_NAMES
        assert "features_secondary" in PRESERVATION_CHECKPOINT_NAMES
        assert "features_supervised_secondary" in PRESERVATION_CHECKPOINT_NAMES

    def test_should_write_when_missing(self, cm):
        assert preservation_checkpoint_should_write("macro_raw_secondary", cm)

    def test_should_not_write_when_exists(self, cm, sample_df):
        cm.save(sample_df, "macro_raw_secondary")
        assert not preservation_checkpoint_should_write("macro_raw_secondary", cm)

    def test_should_write_when_force(self, cm, sample_df):
        cm.save(sample_df, "macro_raw_secondary")
        assert preservation_checkpoint_should_write(
            "macro_raw_secondary", cm, force=True,
        )

    def test_non_preservation_name_returns_false(self, cm):
        assert not preservation_checkpoint_should_write("macro_raw", cm)

    def test_all_three_preservation_names_accepted(self, cm):
        for name in PRESERVATION_CHECKPOINT_NAMES:
            assert preservation_checkpoint_should_write(name, cm)

    def test_clear_all_keeps_all_preservation_types(self, cm, sample_df):
        for name in PRESERVATION_CHECKPOINT_NAMES:
            cm.save(sample_df, name)
        cm.save(sample_df, "regular_checkpoint")
        cm.clear_all()
        remaining_stems = set()
        for f in cm.dir.iterdir():
            stem = f.stem
            if stem.endswith(".meta"):
                stem = stem[: -len(".meta")]
            remaining_stems.add(stem)
        assert PRESERVATION_CHECKPOINT_NAMES == remaining_stems
        assert "regular_checkpoint" not in remaining_stems


# ── merge-on-save: never-lose-coverage checkpoint writes ───────────────────

class TestMergeOnSaveNames:
    def test_names_frozenset(self):
        assert isinstance(MERGE_ON_SAVE_CHECKPOINT_NAMES, frozenset)
        for name in ("daily_raw", "monthly_raw", "macro_raw", "asset_prices"):
            assert name in MERGE_ON_SAVE_CHECKPOINT_NAMES

    def test_derived_frames_excluded(self):
        for name in (
            "monthly_features", "features", "features_supervised",
            "features_causal", "features_noncausal",
        ):
            assert name not in MERGE_ON_SAVE_CHECKPOINT_NAMES


class TestMergePreserving:
    """Pure-function tests for merge_preserving(existing, new)."""

    def test_overlapping_cell_prefers_new_when_non_nan(self):
        idx = pd.date_range("2020-01-31", periods=3, freq="ME")
        existing = pd.DataFrame({"a": [1.0, 2.0, 3.0]}, index=idx)
        new = pd.DataFrame({"a": [10.0, 20.0, 30.0]}, index=idx)

        merged, _stats = merge_preserving(existing, new)

        assert list(merged["a"]) == [10.0, 20.0, 30.0]

    def test_overlapping_cell_keeps_existing_when_new_is_nan(self):
        idx = pd.date_range("2020-01-31", periods=3, freq="ME")
        existing = pd.DataFrame({"a": [1.0, 2.0, 3.0]}, index=idx)
        new = pd.DataFrame({"a": [10.0, np.nan, 30.0]}, index=idx)

        merged, _stats = merge_preserving(existing, new)

        assert list(merged["a"]) == [10.0, 2.0, 30.0]

    def test_column_only_in_existing_is_kept(self):
        idx = pd.date_range("2020-01-31", periods=3, freq="ME")
        existing = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]}, index=idx)
        new = pd.DataFrame({"a": [10.0, 20.0, 30.0]}, index=idx)

        merged, stats = merge_preserving(existing, new)

        assert list(merged["b"]) == [4.0, 5.0, 6.0]
        assert stats["cols_kept_from_disk"] == ["b"]

    def test_column_only_in_new_is_added(self):
        idx = pd.date_range("2020-01-31", periods=3, freq="ME")
        existing = pd.DataFrame({"a": [1.0, 2.0, 3.0]}, index=idx)
        new = pd.DataFrame({"a": [10.0, 20.0, 30.0], "c": [7.0, 8.0, 9.0]}, index=idx)

        merged, stats = merge_preserving(existing, new)

        assert list(merged["c"]) == [7.0, 8.0, 9.0]
        assert stats["cols_added"] == ["c"]

    def test_column_order_new_first_then_existing_only(self):
        idx = pd.date_range("2020-01-31", periods=2, freq="ME")
        existing = pd.DataFrame({"z": [1.0, 2.0], "a": [1.0, 2.0]}, index=idx)
        new = pd.DataFrame({"m": [1.0, 2.0], "a": [10.0, 20.0]}, index=idx)

        merged, _stats = merge_preserving(existing, new)

        assert list(merged.columns) == ["m", "a", "z"]

    def test_index_is_union_and_sorted_ascending(self):
        idx_existing = pd.date_range("2020-01-31", periods=3, freq="ME")
        idx_new = pd.date_range("2020-03-31", periods=3, freq="ME")  # overlaps last existing month
        existing = pd.DataFrame({"a": [1.0, 2.0, 3.0]}, index=idx_existing)
        new = pd.DataFrame({"a": [30.0, 40.0, 50.0]}, index=idx_new)

        merged, _stats = merge_preserving(existing, new)

        assert list(merged.index) == sorted(set(idx_existing) | set(idx_new))
        assert merged.loc[idx_existing[0], "a"] == 1.0  # existing-only row kept
        assert merged.loc[idx_new[0], "a"] == 30.0  # overlapping index -> new wins

    def test_row_count_stats(self):
        idx_existing = pd.date_range("2020-01-31", periods=3, freq="ME")
        idx_new = pd.date_range("2020-03-31", periods=3, freq="ME")
        existing = pd.DataFrame({"a": [1.0, 2.0, 3.0]}, index=idx_existing)
        new = pd.DataFrame({"a": [30.0, 40.0, 50.0]}, index=idx_new)

        _merged, stats = merge_preserving(existing, new)

        assert stats["rows_kept_from_disk"] == 2  # existing-only rows
        assert stats["rows_added"] == 2  # new-only rows

    def test_cells_filled_from_disk_counts_recovered_cells(self):
        idx = pd.date_range("2020-01-31", periods=3, freq="ME")
        existing = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]}, index=idx)
        new = pd.DataFrame({"a": [10.0, np.nan, 30.0]}, index=idx)

        _merged, stats = merge_preserving(existing, new)

        # 1 cell recovered in "a" (the NaN at idx[1]) + 3 cells recovered in "b"
        # (entirely existing-only) = 4.
        assert stats["cells_filled_from_disk"] == 4

    def test_duplicate_columns_returns_new_unchanged_with_degraded_flag(self, caplog):
        idx = pd.date_range("2020-01-31", periods=2, freq="ME")
        existing = pd.DataFrame({"a": [1.0, 2.0]}, index=idx)
        new = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], index=idx, columns=["a", "a"])

        with caplog.at_level(logging.WARNING):
            merged, stats = merge_preserving(existing, new)

        assert merged is new
        assert stats["degraded"] == "duplicate column labels"
        assert any("duplicate" in r.message.lower() for r in caplog.records)


class TestSaveMergeSemantics:
    """Integration tests for CheckpointManager.save()'s merge-on-save behavior."""

    def test_A_empty_write_refused_over_non_empty_checkpoint(self, tmp_path, caplog):
        cm = CheckpointManager(checkpoint_dir=tmp_path)
        idx = pd.date_range("2020-01-31", periods=3, freq="ME")
        non_empty = pd.DataFrame({"a": [1.0, 2.0, 3.0]}, index=idx)
        cm.save(non_empty, "daily_raw")
        before_bytes = (tmp_path / "daily_raw.parquet").read_bytes()
        before_meta = (tmp_path / "daily_raw.meta.json").read_text()

        with caplog.at_level(logging.WARNING):
            cm.save(pd.DataFrame(), "daily_raw", source="test-fixture-empty-fetch")

        after_bytes = (tmp_path / "daily_raw.parquet").read_bytes()
        after_meta = (tmp_path / "daily_raw.meta.json").read_text()
        assert after_bytes == before_bytes
        assert after_meta == before_meta
        warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("refusing" in m.lower() for m in warnings)
        assert any("daily_raw" in m and "test-fixture-empty-fetch" in m for m in warnings)

    def test_B_column_only_in_existing_survives_reload(self, tmp_path):
        cm = CheckpointManager(checkpoint_dir=tmp_path)
        idx = pd.date_range("2020-01-31", periods=3, freq="ME")
        stored = pd.DataFrame(
            {"SPY": [1.0, 2.0, 3.0], "IAU": [10.0, 20.0, 30.0], "TLT": [5.0, 6.0, 7.0]}, index=idx
        )
        cm.save(stored, "daily_raw")
        new = pd.DataFrame({"SPY": [100.0, 200.0, 300.0], "TLT": [50.0, 60.0, 70.0]}, index=idx)

        cm.save(new, "daily_raw")
        reloaded = cm.load("daily_raw")

        assert "IAU" in reloaded.columns
        assert list(reloaded["IAU"]) == [10.0, 20.0, 30.0]

    def test_C_index_span_extended_not_truncated(self, tmp_path):
        cm = CheckpointManager(checkpoint_dir=tmp_path)
        idx_full = pd.date_range("2000-01-31", periods=6, freq="ME")
        stored = pd.DataFrame({"a": np.arange(6.0)}, index=idx_full)
        cm.save(stored, "daily_raw")
        idx_partial = idx_full[-3:]
        new = pd.DataFrame({"a": [100.0, 200.0, 300.0]}, index=idx_partial)

        cm.save(new, "daily_raw")
        reloaded = cm.load("daily_raw")

        assert reloaded.index.min() == idx_full[0]
        assert reloaded.loc[idx_full[0], "a"] == 0.0  # pre-window value preserved

    def test_D_nan_cell_does_not_overwrite_stored_value(self, tmp_path):
        cm = CheckpointManager(checkpoint_dir=tmp_path)
        idx = pd.date_range("2020-01-31", periods=1, freq="ME")
        stored = pd.DataFrame({"SPY": [100.0]}, index=idx)
        cm.save(stored, "daily_raw")
        new = pd.DataFrame({"SPY": [np.nan]}, index=idx)

        cm.save(new, "daily_raw")
        reloaded = cm.load("daily_raw")

        assert reloaded.loc[idx[0], "SPY"] == 100.0

    def test_E_force_replace_kwarg_restores_plain_replace(self, tmp_path):
        cm = CheckpointManager(checkpoint_dir=tmp_path)
        idx = pd.date_range("2020-01-31", periods=3, freq="ME")
        stored = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]}, index=idx)
        cm.save(stored, "daily_raw")
        new = pd.DataFrame({"a": [10.0]}, index=idx[:1])

        cm.save(new, "daily_raw", force_replace=True)
        reloaded = cm.load("daily_raw")

        pd.testing.assert_frame_equal(reloaded, new, check_freq=False)

    def test_E_force_replace_env_var_restores_plain_replace(self, tmp_path, monkeypatch):
        cm = CheckpointManager(checkpoint_dir=tmp_path)
        idx = pd.date_range("2020-01-31", periods=3, freq="ME")
        stored = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]}, index=idx)
        cm.save(stored, "daily_raw")
        new = pd.DataFrame({"a": [10.0]}, index=idx[:1])
        monkeypatch.setenv("TC_CHECKPOINT_FORCE_REPLACE", "1")

        cm.save(new, "daily_raw")
        reloaded = cm.load("daily_raw")

        pd.testing.assert_frame_equal(reloaded, new, check_freq=False)

    def test_F_non_merge_eligible_name_replaces_wholesale(self, tmp_path):
        cm = CheckpointManager(checkpoint_dir=tmp_path)
        idx = pd.date_range("2020-01-31", periods=3, freq="ME")
        stored = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]}, index=idx)
        cm.save(stored, "monthly_features")
        smaller = pd.DataFrame({"a": [10.0]}, index=idx[:1])

        cm.save(smaller, "monthly_features")
        reloaded = cm.load("monthly_features")

        pd.testing.assert_frame_equal(reloaded, smaller, check_freq=False)

    def test_G_merge_true_forces_merge_on_arbitrary_name(self, tmp_path):
        cm = CheckpointManager(checkpoint_dir=tmp_path)
        idx = pd.date_range("2020-01-31", periods=3, freq="ME")
        stored = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]}, index=idx)
        cm.save(stored, "arbitrary_name")
        smaller = pd.DataFrame({"a": [10.0]}, index=idx[:1])

        cm.save(smaller, "arbitrary_name", merge=True)
        reloaded = cm.load("arbitrary_name")

        assert "b" in reloaded.columns
        assert len(reloaded) == 3

    def test_G_merge_false_forces_replace_on_daily_raw(self, tmp_path):
        cm = CheckpointManager(checkpoint_dir=tmp_path)
        idx = pd.date_range("2020-01-31", periods=3, freq="ME")
        stored = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]}, index=idx)
        cm.save(stored, "daily_raw")
        smaller = pd.DataFrame({"a": [10.0]}, index=idx[:1])

        cm.save(smaller, "daily_raw", merge=False)
        reloaded = cm.load("daily_raw")

        pd.testing.assert_frame_equal(reloaded, smaller, check_freq=False)

    def test_H_merge_emits_info_log_with_stats(self, tmp_path, caplog):
        cm = CheckpointManager(checkpoint_dir=tmp_path)
        idx = pd.date_range("2020-01-31", periods=3, freq="ME")
        stored = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]}, index=idx)
        cm.save(stored, "daily_raw")
        new = pd.DataFrame({"a": [10.0, 20.0, 30.0]}, index=idx)

        with caplog.at_level(logging.INFO):
            cm.save(new, "daily_raw")

        merged_logs = [r.message for r in caplog.records if "merged" in r.message.lower()]
        assert merged_logs
        assert "kept" in merged_logs[0].lower()

    def test_I_meta_json_carries_merge_block_with_stats(self, tmp_path):
        cm = CheckpointManager(checkpoint_dir=tmp_path)
        idx = pd.date_range("2020-01-31", periods=3, freq="ME")
        stored = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]}, index=idx)
        cm.save(stored, "daily_raw")
        new = pd.DataFrame({"a": [10.0, 20.0, 30.0]}, index=idx)

        cm.save(new, "daily_raw")
        meta = json.loads((tmp_path / "daily_raw.meta.json").read_text())

        assert "merge" in meta
        assert meta["merge"]["cols_kept_from_disk"] == ["b"]
        assert meta["merge"]["pre_merge_shape"] == {"rows": 3, "columns": 2}

    def test_first_save_of_merge_eligible_name_has_no_merge_block(self, tmp_path):
        cm = CheckpointManager(checkpoint_dir=tmp_path)
        idx = pd.date_range("2020-01-31", periods=3, freq="ME")
        first = pd.DataFrame({"a": [1.0, 2.0, 3.0]}, index=idx)

        cm.save(first, "daily_raw")
        meta = json.loads((tmp_path / "daily_raw.meta.json").read_text())

        assert "merge" not in meta

    def test_unreadable_existing_parquet_falls_back_to_plain_replace(self, tmp_path, caplog):
        cm = CheckpointManager(checkpoint_dir=tmp_path)
        (tmp_path / "daily_raw.parquet").write_bytes(b"not a real parquet file")
        idx = pd.date_range("2020-01-31", periods=2, freq="ME")
        new = pd.DataFrame({"a": [1.0, 2.0]}, index=idx)

        with caplog.at_level(logging.WARNING):
            cm.save(new, "daily_raw")
        reloaded = cm.load("daily_raw")

        pd.testing.assert_frame_equal(reloaded, new, check_freq=False)
        assert any("unreadable" in r.message.lower() for r in caplog.records)
