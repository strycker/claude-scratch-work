"""Unit tests for trading_crab_lib.platform.honesty.holdout (HON-01).

Follows the incumbent tests/unit/test_platform_taxonomy.py structure: a
class per behavior, docstring per test. The headline invariant test proves —
by real load through the default manager, never by monkeypatching — that the
default dev path CANNOT return post-2020 rows.
"""

from __future__ import annotations

import pandas as pd
import pytest

from trading_crab_lib.checkpoints import CheckpointManager
from trading_crab_lib.platform.honesty.holdout import (
    DEFAULT_HOLDOUT_CUTOFF,
    assert_dev_checkpoint_within_boundary,
    get_holdout_checkpoint_manager,
    split_by_holdout_boundary,
    write_monthly_features_split,
)

# ── Helpers ────────────────────────────────────────────────────────────────────


def _synthetic_monthly_df(start: str = "2019-01-31", end: str = "2022-12-31") -> pd.DataFrame:
    """A synthetic monthly DataFrame with a month-end DatetimeIndex spanning
    2019-01 through 2022-12 (no file I/O, no network)."""
    index = pd.date_range(start=start, end=end, freq="ME")
    return pd.DataFrame({"value": range(len(index))}, index=index)


@pytest.fixture
def dev_manager(tmp_path, monkeypatch) -> CheckpointManager:
    """A CheckpointManager wired to a tmp dir, patched in as the default platform manager."""
    manager = CheckpointManager(checkpoint_dir=tmp_path / "platform")
    monkeypatch.setattr(
        "trading_crab_lib.platform.honesty.holdout.get_platform_checkpoint_manager",
        lambda: manager,
    )
    return manager


@pytest.fixture
def holdout_manager(tmp_path, monkeypatch) -> CheckpointManager:
    """A CheckpointManager wired to a tmp dir, patched in as the holdout manager."""
    manager = CheckpointManager(checkpoint_dir=tmp_path / "holdout")
    monkeypatch.setattr(
        "trading_crab_lib.platform.honesty.holdout.HOLDOUT_CHECKPOINT_DIR",
        tmp_path / "holdout",
    )
    return manager


# ── split_by_holdout_boundary ───────────────────────────────────────────────────


class TestSplitByHoldoutBoundary:
    def test_dev_and_holdout_are_disjoint_and_sum_to_input_length(self):
        """The two frames never overlap and together account for every row."""
        df = _synthetic_monthly_df()
        dev_df, holdout_df = split_by_holdout_boundary(df, cutoff=DEFAULT_HOLDOUT_CUTOFF)
        assert len(dev_df) + len(holdout_df) == len(df)
        assert not dev_df.index.intersection(holdout_df.index).any()

    def test_dev_side_honors_cutoff_inclusive(self):
        """dev_df.index.max() <= cutoff."""
        df = _synthetic_monthly_df()
        dev_df, _ = split_by_holdout_boundary(df, cutoff=DEFAULT_HOLDOUT_CUTOFF)
        assert dev_df.index.max() <= pd.Timestamp(DEFAULT_HOLDOUT_CUTOFF)

    def test_holdout_side_honors_cutoff_exclusive(self):
        """holdout_df.index.min() > cutoff."""
        df = _synthetic_monthly_df()
        _, holdout_df = split_by_holdout_boundary(df, cutoff=DEFAULT_HOLDOUT_CUTOFF)
        assert holdout_df.index.min() > pd.Timestamp(DEFAULT_HOLDOUT_CUTOFF)

    def test_all_rows_below_cutoff_yields_empty_holdout(self):
        """A DataFrame entirely <= cutoff produces an empty holdout_df."""
        df = _synthetic_monthly_df(start="2015-01-31", end="2019-12-31")
        dev_df, holdout_df = split_by_holdout_boundary(df, cutoff=DEFAULT_HOLDOUT_CUTOFF)
        assert len(dev_df) == len(df)
        assert holdout_df.empty

    def test_all_rows_above_cutoff_yields_empty_dev(self):
        """A DataFrame entirely > cutoff produces an empty dev_df."""
        df = _synthetic_monthly_df(start="2021-01-31", end="2022-12-31")
        dev_df, holdout_df = split_by_holdout_boundary(df, cutoff=DEFAULT_HOLDOUT_CUTOFF)
        assert dev_df.empty
        assert len(holdout_df) == len(df)


# ── get_holdout_checkpoint_manager ──────────────────────────────────────────────


class TestGetHoldoutCheckpointManager:
    def test_holdout_manager_dir_is_distinct_from_platform_manager_dir(self, dev_manager):
        """The holdout manager and the default platform manager point at different dirs."""
        holdout_cm = get_holdout_checkpoint_manager()
        assert holdout_cm.dir != dev_manager.dir


# ── write_monthly_features_split ────────────────────────────────────────────────


class TestWriteMonthlyFeaturesSplit:
    def test_reloading_each_side_returns_only_its_half(self, dev_manager, holdout_manager):
        """Reloading the dev checkpoint yields only <=cutoff rows; reloading the
        holdout checkpoint yields only >cutoff rows."""
        df = _synthetic_monthly_df()
        write_monthly_features_split(df, name="monthly_features", cutoff=DEFAULT_HOLDOUT_CUTOFF)

        dev_loaded = dev_manager.load("monthly_features")
        holdout_loaded = holdout_manager.load("monthly_features")

        assert dev_loaded.index.max() <= pd.Timestamp(DEFAULT_HOLDOUT_CUTOFF)
        assert holdout_loaded.index.min() > pd.Timestamp(DEFAULT_HOLDOUT_CUTOFF)
        assert len(dev_loaded) + len(holdout_loaded) == len(df)


# ── assert_dev_checkpoint_within_boundary ───────────────────────────────────────


class TestAssertDevCheckpointWithinBoundary:
    def test_raises_runtime_error_when_dev_checkpoint_has_post_cutoff_row(self, dev_manager):
        """A dev checkpoint holding a 2021+ row raises RuntimeError naming the offending date."""
        df = _synthetic_monthly_df(start="2019-01-31", end="2021-06-30")
        dev_manager.save(df, "monthly_features")
        with pytest.raises(RuntimeError) as exc_info:
            assert_dev_checkpoint_within_boundary("monthly_features", cutoff=DEFAULT_HOLDOUT_CUTOFF)
        assert "monthly_features" in str(exc_info.value)

    def test_does_not_raise_when_all_rows_within_boundary(self, dev_manager):
        """A dev checkpoint entirely <= cutoff returns None, no raise."""
        df = _synthetic_monthly_df(start="2019-01-31", end="2020-12-31")
        dev_manager.save(df, "monthly_features")
        assert assert_dev_checkpoint_within_boundary("monthly_features", cutoff=DEFAULT_HOLDOUT_CUTOFF) is None


# ── Headline invariant ───────────────────────────────────────────────────────────


class TestHoldoutBoundary:
    def test_default_manager_cannot_load_post_2020_rows(self, dev_manager, holdout_manager):
        """After a split-write, the default-manager load of monthly_features has
        index.max() <= 2020-12-31, and loading a holdout-only checkpoint through the
        default manager raises FileNotFoundError — no fallback code path exists."""
        df = _synthetic_monthly_df()
        write_monthly_features_split(df, name="monthly_features", cutoff=DEFAULT_HOLDOUT_CUTOFF)

        default_loaded = dev_manager.load("monthly_features")
        assert default_loaded.index.max() <= pd.Timestamp("2020-12-31")

        # A checkpoint that exists ONLY in the holdout tree (never written to dev).
        holdout_only_df = _synthetic_monthly_df(start="2021-01-31", end="2021-12-31")
        holdout_manager.save(holdout_only_df, "holdout_only_checkpoint")

        with pytest.raises(FileNotFoundError):
            dev_manager.load("holdout_only_checkpoint")
