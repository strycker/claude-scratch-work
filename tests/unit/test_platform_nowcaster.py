"""Unit tests for trading_crab_lib.platform.prediction.nowcaster (L2-01, L1-02).

Synthetic monthly DataFrame, no network — mirrors
tests/unit/test_platform_walkforward.py's synthetic-frame convention.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_crab_lib.platform.honesty.registry import read_trials
from trading_crab_lib.platform.prediction.nowcaster import (
    build_nowcaster_training_set,
    evaluate_nowcaster,
    fit_nowcaster,
    transition_window_accuracy,
)

N_MONTHS = 180
N_FEATURES = 13
N_STATES = 5


@pytest.fixture(autouse=True)
def _redirect_platform_checkpoints(tmp_path, monkeypatch):
    """Route all platform checkpoint I/O to a temp dir — production
    data/checkpoints/platform/ is never written to during this test run."""
    from trading_crab_lib.platform import checkpoints as platform_checkpoints

    monkeypatch.setattr(platform_checkpoints, "PLATFORM_CHECKPOINT_DIR", tmp_path / "platform")


def _make_synthetic_monthly(n_months: int = N_MONTHS, seed: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    """Return (features_df, labels) on a month-end DatetimeIndex.

    features_df: 13-column causal feature frame (stand-in for the lean
    monthly_features taxonomy pull). labels: integer regime state 0..4, a
    stand-in for the L1 jump-model labeler's output.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2000-01-31", periods=n_months, freq="ME")
    feature_cols = [f"feat_{i}" for i in range(N_FEATURES)]
    features_df = pd.DataFrame(
        rng.normal(0, 1, size=(n_months, N_FEATURES)),
        index=idx,
        columns=feature_cols,
    )
    labels = pd.Series(rng.integers(0, N_STATES, size=n_months), index=idx, name="state")
    return features_df, labels


class TestEmbargoInvariant:
    def test_trailing_12_months_excluded_on_real_output(self):
        """Headline invariant (T-03-04): a real assertion on real output, not
        a mock of the embargo boundary."""
        features_df, labels = _make_synthetic_monthly()
        _, y = build_nowcaster_training_set(features_df, labels, embargo_months=12)
        assert y.index.max() <= labels.index.max() - pd.DateOffset(months=12)

    def test_index_alignment(self):
        features_df, labels = _make_synthetic_monthly()
        X, y = build_nowcaster_training_set(features_df, labels)
        assert list(X.index) == list(y.index)

    def test_embargo_months_zero_is_degenerate_no_embargo(self):
        features_df, labels = _make_synthetic_monthly()
        X, y = build_nowcaster_training_set(features_df, labels, embargo_months=0)
        common = features_df.index.intersection(labels.index)
        assert list(X.index) == list(common)
        assert list(y.index) == list(common)

    def test_negative_embargo_raises_before_slicing(self):
        features_df, labels = _make_synthetic_monthly()
        with pytest.raises(ValueError):
            build_nowcaster_training_set(features_df, labels, embargo_months=-1)


class TestNowcasterCalibratedOutput:
    def test_predict_proba_shape_and_rows_sum_to_one(self):
        features_df, labels = _make_synthetic_monthly()
        X, y = build_nowcaster_training_set(features_df, labels)
        model = fit_nowcaster(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (len(X), len(y.unique()))
        assert proba.sum(axis=1) == pytest.approx(np.ones(len(X)))

    def test_predict_proba_is_a_distribution_never_argmax(self):
        """Callers get predict_proba (a distribution) — not predict()/argmax."""
        features_df, labels = _make_synthetic_monthly()
        X, y = build_nowcaster_training_set(features_df, labels)
        model = fit_nowcaster(X, y)
        assert hasattr(model, "predict_proba")
        proba = model.predict_proba(X)
        assert proba.ndim == 2
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_nowcaster_does_not_pass_multi_class_kwarg(self):
        """Our code must never pass ``multi_class`` — it is deprecated (and removed in
        sklearn>=1.7); multinomial is automatic for the lbfgs solver. Guarding our own
        source is version-independent, unlike asserting on sklearn's signature (which
        varies across the CI matrix: older sklearn still exposes multi_class='deprecated')."""
        import inspect

        import trading_crab_lib.platform.prediction.nowcaster as nowcaster_module

        assert "multi_class=" not in inspect.getsource(nowcaster_module)


class TestRegistryLoggingOne:
    def test_exactly_one_new_trial_per_evaluate_call(self, tmp_path):
        features_df, labels = _make_synthetic_monthly()
        registry_path = tmp_path / "trials.jsonl"

        evaluate_nowcaster(features_df, labels, registry_path=registry_path)
        assert len(read_trials(path=registry_path)) == 1

        evaluate_nowcaster(features_df, labels, registry_path=registry_path)
        assert len(read_trials(path=registry_path)) == 2

    def test_model_persisted_via_joblib_never_raw_pickle(self):
        """P27: never raw pickle for the model artifact."""
        import inspect

        import trading_crab_lib.platform.prediction.nowcaster as nowcaster_module

        source = inspect.getsource(nowcaster_module)
        assert "pickle.dump" not in source


class TestTransitionWindowAccuracy:
    def test_returns_exactly_three_keys(self):
        idx = pd.date_range("2020-01-31", periods=10, freq="ME")
        y_true = pd.Series([0, 0, 0, 0, 1, 1, 1, 1, 2, 2], index=idx)
        result = transition_window_accuracy(y_true, y_true, window_months=1)
        assert set(result.keys()) == {"overall_accuracy", "transition_accuracy", "steady_state_accuracy"}

    def test_hand_computed_transition_vs_steady_state(self):
        """Two transitions at positions 4 (0->1) and 8 (1->2), window=1 month.
        near-transition positions: {3,4,5,7,8}; steady-state: {0,1,2,6,9}.
        Errors deliberately placed at positions 1 (steady), 4 and 8 (transition).
        """
        idx = pd.date_range("2020-01-31", periods=10, freq="ME")
        y_true = pd.Series([0, 0, 0, 0, 1, 1, 1, 1, 2, 2], index=idx)
        y_pred = pd.Series([0, 1, 0, 0, 0, 1, 1, 1, 1, 2], index=idx)

        result = transition_window_accuracy(y_true, y_pred, window_months=1)

        assert result["overall_accuracy"] == pytest.approx(0.7)
        assert result["transition_accuracy"] == pytest.approx(0.6)
        assert result["steady_state_accuracy"] == pytest.approx(0.8)

    def test_no_transitions_returns_nan_transition_accuracy_without_raising(self):
        idx = pd.date_range("2020-01-31", periods=10, freq="ME")
        y_true = pd.Series([1] * 10, index=idx)
        y_pred = pd.Series([1] * 10, index=idx)

        result = transition_window_accuracy(y_true, y_pred, window_months=1)

        assert np.isnan(result["transition_accuracy"])
        assert not np.isnan(result["steady_state_accuracy"])
        assert result["overall_accuracy"] == pytest.approx(1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-q"])
