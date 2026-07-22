"""Unit tests for trading_crab_lib.platform.honesty.walkforward (HON-03).

Synthetic monthly DataFrame, no network, no file I/O beyond a tmp_path
registry — mirrors tests/integration/test_mini_pipeline.py's synthetic-frame
convention. Headline test is the leakage invariant: train_index[-1] <
test_index[0] at every step.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_crab_lib.platform.honesty.registry import read_trials
from trading_crab_lib.platform.honesty.walkforward import expanding_steps, run_walkforward

N_MONTHS = 120
MIN_TRAIN = 60


def _make_synthetic_monthly(n_months: int = N_MONTHS, seed: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    """Return (features_df, target_series) on a month-end DatetimeIndex.

    Target is sign-of-next-month synthetic return (binary 0/1), a stand-in
    for the real jump-model regime label that arrives in Phase 3.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2010-01-31", periods=n_months, freq="ME")
    features_df = pd.DataFrame(
        {
            "equities_tr": rng.normal(0, 1, n_months),
            "term_spread": rng.normal(0, 1, n_months),
        },
        index=idx,
    )
    returns = rng.normal(0, 1, n_months)
    target_series = pd.Series((returns > 0).astype(int), index=idx, name="target")
    return features_df, target_series


# ── expanding_steps: leakage invariant ──────────────────────────────────────


class TestExpandingStepsLeakageInvariant:
    def test_train_window_strictly_before_test(self):
        """Headline invariant: train_index[-1] < test_index[0] at every step,
        and the test row is never in the train set."""
        idx = pd.date_range("2010-01-31", periods=N_MONTHS, freq="ME")
        steps = list(expanding_steps(idx, min_train=MIN_TRAIN))
        assert steps, "expanding_steps produced no steps"
        for t, train_index, test_index in steps:
            assert train_index[-1] < test_index[0]
            assert test_index[0] not in set(train_index)
            assert test_index[0] == t


# ── run_walkforward: one decision per step ──────────────────────────────────


class TestOneDecisionPerStep:
    def test_one_decision_per_step(self, tmp_path):
        """len(decisions) == len(index) - min_train; every decision has the full key set."""
        features_df, target_series = _make_synthetic_monthly()
        decisions = run_walkforward(
            features_df, target_series, min_train=MIN_TRAIN, registry_path=tmp_path / "trials.jsonl"
        )
        assert len(decisions) == len(features_df.index) - MIN_TRAIN
        for decision in decisions:
            assert set(decision.keys()) == {"t", "prediction", "params"}


# ── run_walkforward: end-to-end, no network ─────────────────────────────────


class TestEndToEndNoNetwork:
    def test_end_to_end_no_network(self, tmp_path):
        """run_walkforward completes on synthetic data with the default
        DummyClassifier and returns a non-empty decisions list."""
        features_df, target_series = _make_synthetic_monthly()
        decisions = run_walkforward(
            features_df, target_series, min_train=MIN_TRAIN, registry_path=tmp_path / "trials.jsonl"
        )
        assert len(decisions) > 0


# ── run_walkforward: automatic registry logging ─────────────────────────────


class TestExactlyOneRegistryTrial:
    def test_exactly_one_registry_trial(self, tmp_path):
        """After a run, read_trials(tmp_path) returns exactly 1 row whose
        features list equals the feature columns and whose config records
        the model class and min_train."""
        features_df, target_series = _make_synthetic_monthly()
        registry_path = tmp_path / "trials.jsonl"
        run_walkforward(features_df, target_series, min_train=MIN_TRAIN, registry_path=registry_path)

        trials = read_trials(path=registry_path)
        assert len(trials) == 1
        row = trials.iloc[0]
        assert row["features"] == list(features_df.columns)
        assert row["config"]["model_class"] == "DummyClassifier"
        assert row["config"]["min_train"] == MIN_TRAIN


class TestRegistryLoggingIsAutomatic:
    def test_registry_logging_is_automatic(self, tmp_path):
        """No test-side manual append_trial call is needed — the runner logs
        it. The row exists purely from calling run_walkforward."""
        features_df, target_series = _make_synthetic_monthly()
        registry_path = tmp_path / "trials.jsonl"
        assert not registry_path.exists()

        run_walkforward(features_df, target_series, min_train=MIN_TRAIN, registry_path=registry_path)

        assert registry_path.exists()
        trials = read_trials(path=registry_path)
        assert len(trials) == 1


# ── run_walkforward: multiple runs each log their own trial ────────────────


class TestMultipleRunsLogMultipleTrials:
    def test_two_runs_produce_two_trials(self, tmp_path):
        """D-02: every evaluated configuration is logged, so two separate
        runs against the same registry path append two rows, not one."""
        features_df, target_series = _make_synthetic_monthly()
        registry_path = tmp_path / "trials.jsonl"
        run_walkforward(features_df, target_series, min_train=MIN_TRAIN, registry_path=registry_path)
        run_walkforward(features_df, target_series, min_train=MIN_TRAIN, registry_path=registry_path)

        trials = read_trials(path=registry_path)
        assert len(trials) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-q"])
