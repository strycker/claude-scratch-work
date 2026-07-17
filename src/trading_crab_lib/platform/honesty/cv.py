"""
Purged + embargoed K-fold cross-validator (HON-04, design §6.5 / R7).

Implements López de Prado, *Advances in Financial Machine Learning*, ch. 7:
overlapping-label time series need more than a contiguous train/test split —
training rows whose label window overlaps a test fold must be *purged*, and a
further window immediately after the test fold must be *embargoed* to block
leakage through slow-moving/autocorrelated features.

Purge and embargo windows are sized in index POSITIONS, not calendar time.
Callers on this project's monthly modeling spine pass purge/embargo widths in
MONTHS (one position == one month); a daily-bar convention transplanted
verbatim would no-op the guard at monthly cadence.

There is no silent default for ``label_horizon`` or ``embargo`` — both are
required keyword-only constructor arguments. This is deliberate (RESEARCH.md
Open Question 2): every Phase 3+ caller must reason about its own target's
label horizon rather than inheriting a hidden global default.

Usage::

    from trading_crab_lib.platform.honesty.cv import PurgedEmbargoedKFold

    cv = PurgedEmbargoedKFold(n_splits=5, label_horizon=12, embargo=1)
    for train_idx, test_idx in cv.split(X):
        ...
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import BaseCrossValidator


class PurgedEmbargoedKFold(BaseCrossValidator):
    """K-fold CV with purging (drop train rows whose label window overlaps the
    test fold) and embargo (drop a further window of train rows immediately
    after the test fold).

    Args:
        n_splits: number of folds.
        label_horizon: purge window, in index positions — rows whose label
            "resolves" up to ``label_horizon`` steps after their own index are
            purged if that resolution window overlaps the test fold. Required,
            no default (see module docstring).
        embargo: embargo window, in index positions, applied immediately after
            each test fold. Required, no default.
    """

    def __init__(self, n_splits: int = 5, *, label_horizon: int, embargo: int) -> None:
        self.n_splits = n_splits
        self.label_horizon = label_horizon
        self.embargo = embargo

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def split(self, X, y=None, groups=None):
        n = len(X)
        indices = np.arange(n)
        fold_bounds = np.array_split(indices, self.n_splits)
        for test_idx in fold_bounds:
            test_start, test_end = test_idx[0], test_idx[-1]
            purge_start = max(0, test_start - self.label_horizon)
            embargo_end = min(n, test_end + 1 + self.embargo)
            train_mask = np.ones(n, dtype=bool)
            train_mask[purge_start:embargo_end] = False
            yield indices[train_mask], test_idx
