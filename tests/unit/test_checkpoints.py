"""Unit tests for src/market_regime/io/checkpoints.py"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from market_regime.checkpoints import CheckpointManager


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

    def test_clear_all(self, cm, sample_df):
        cm.save(sample_df, "a")
        cm.save(sample_df, "b")
        cm.clear_all()
        assert list(cm.dir.iterdir()) == []


# ── list / summary ─────────────────────────────────────────────────────────

class TestList:
    def test_list_empty_when_no_checkpoints(self, cm):
        assert cm.list() == []

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
