"""Unit tests for trading_crab_lib.platform.honesty.cv (HON-04).

Follows the incumbent tests/unit/test_platform_transforms.py synthetic-frame
style: no network, no file I/O, no fixtures beyond plain numpy/pandas
construction. Every behavior bullet from the plan is encoded as an explicit
property test, including a parametrized leakage-window sweep proving no
train index ever falls within the purge-or-embargo window of any test fold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import BaseCrossValidator

from trading_crab_lib.platform.honesty.cv import PurgedEmbargoedKFold

# ── Helpers ────────────────────────────────────────────────────────────────────


def _synthetic_monthly_frame(n: int) -> pd.DataFrame:
    """A synthetic monthly DataFrame with n rows (no network, no file I/O)."""
    index = pd.date_range(start="2000-01-31", periods=n, freq="ME")
    return pd.DataFrame({"value": range(n)}, index=index)


def _fold_bounds(test_idx: np.ndarray) -> tuple[int, int]:
    return int(test_idx[0]), int(test_idx[-1])


# ── Leakage-window property sweep ───────────────────────────────────────────────


LEAKAGE_SWEEP_PARAMS = [
    # (n, n_splits, label_horizon, embargo)
    (60, 5, 3, 1),
    (120, 5, 12, 1),
    (100, 4, 6, 0),
]


class TestNoLeakageAcrossPurgeEmbargoWindow:
    """No train index may fall within the purge (before) or embargo (after) window
    of any test fold, for every parametrization in the sweep."""

    @pytest.mark.parametrize("n, n_splits, label_horizon, embargo", LEAKAGE_SWEEP_PARAMS)
    def test_no_train_index_in_purge_or_embargo_window(self, n, n_splits, label_horizon, embargo):
        X = _synthetic_monthly_frame(n)
        cv = PurgedEmbargoedKFold(n_splits=n_splits, label_horizon=label_horizon, embargo=embargo)
        for train_idx, test_idx in cv.split(X):
            test_start, test_end = _fold_bounds(test_idx)
            purge_start = max(0, test_start - label_horizon)
            embargo_end = min(n, test_end + 1 + embargo)
            forbidden = set(range(purge_start, embargo_end))
            assert set(train_idx.tolist()).isdisjoint(forbidden)


class TestTrainTestDisjoint:
    def test_train_test_disjoint(self):
        X = _synthetic_monthly_frame(100)
        cv = PurgedEmbargoedKFold(n_splits=5, label_horizon=6, embargo=1)
        for train_idx, test_idx in cv.split(X):
            assert set(train_idx.tolist()) & set(test_idx.tolist()) == set()


class TestFoldsPartitionAllIndices:
    def test_test_folds_partition_all_indices(self):
        n = 100
        X = _synthetic_monthly_frame(n)
        cv = PurgedEmbargoedKFold(n_splits=5, label_horizon=6, embargo=1)
        all_test = np.concatenate([test_idx for _, test_idx in cv.split(X)])
        assert sorted(all_test.tolist()) == list(range(n))


class TestTinyWindowNoEmptyFold:
    @pytest.mark.parametrize("n", [1, 2, 3, 4])
    def test_no_indexerror_and_no_empty_test_fold_when_n_below_n_splits(self, n):
        # Tiny early walk-forward windows (n < n_splits) made np.array_split emit
        # empty folds → IndexError on test_idx[0]. The splitter must skip them.
        X = _synthetic_monthly_frame(n)
        cv = PurgedEmbargoedKFold(n_splits=5, label_horizon=6, embargo=1)
        folds = list(cv.split(X))  # must not raise IndexError
        assert all(len(test_idx) > 0 for _, test_idx in folds)
        # every yielded test fold still covers real rows within [0, n)
        for _, test_idx in folds:
            assert test_idx.min() >= 0 and test_idx.max() < n


class TestGetNSplitsMatches:
    def test_get_n_splits_matches(self):
        n = 100
        X = _synthetic_monthly_frame(n)
        cv = PurgedEmbargoedKFold(n_splits=5, label_horizon=6, embargo=1)
        assert cv.get_n_splits(X) == 5
        assert sum(1 for _ in cv.split(X)) == 5


class TestRequiredArgsNoSilentDefault:
    def test_missing_both_label_horizon_and_embargo_raises(self):
        with pytest.raises(TypeError):
            PurgedEmbargoedKFold(n_splits=5)

    def test_missing_embargo_raises(self):
        with pytest.raises(TypeError):
            PurgedEmbargoedKFold(n_splits=5, label_horizon=12)

    def test_missing_label_horizon_raises(self):
        with pytest.raises(TypeError):
            PurgedEmbargoedKFold(n_splits=5, embargo=1)


class TestNontrivialPurgeAtMonthlyHorizon:
    def test_nontrivial_purge_at_monthly_horizon(self):
        n = 120
        X = _synthetic_monthly_frame(n)
        cv = PurgedEmbargoedKFold(n_splits=5, label_horizon=12, embargo=1)
        found_nontrivial = False
        for train_idx, test_idx in cv.split(X):
            purged_or_embargoed = n - len(train_idx) - len(test_idx)
            if purged_or_embargoed > 0:
                found_nontrivial = True
        assert found_nontrivial


class TestBaseCrossValidatorContract:
    def test_isinstance_basecrossvalidator(self):
        cv = PurgedEmbargoedKFold(3, label_horizon=1, embargo=0)
        assert isinstance(cv, BaseCrossValidator)

    def test_split_yields_numpy_integer_arrays(self):
        X = _synthetic_monthly_frame(30)
        cv = PurgedEmbargoedKFold(3, label_horizon=1, embargo=0)
        for train_idx, test_idx in cv.split(X):
            assert isinstance(train_idx, np.ndarray)
            assert isinstance(test_idx, np.ndarray)
            assert np.issubdtype(train_idx.dtype, np.integer)
            assert np.issubdtype(test_idx.dtype, np.integer)
