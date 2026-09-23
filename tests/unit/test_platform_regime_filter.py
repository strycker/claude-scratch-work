"""Unit tests for platform.prediction.regime_filter (plan 08-06 Task 1).

Every test here is written so that a specific wrong implementation fails it; the
wrong implementation is named in each docstring. Synthetic fixtures only — no
network, no checkpoint.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from trading_crab_lib.platform.prediction.regime_filter import (
    filter_step,
    likelihood_ratio,
    predict_only_step,
    transition_matrix_for,
    unconditional_belief,
)

IDX3 = [0, 1, 2]

#: A doubly-stochastic, non-identity A whose row 1 sends most mass to state 2.
#: Chosen so that predict-then-update and update-then-predict give DIFFERENT
#: argmaxes on the sharp-likelihood fixture below.
A_ROTATE = pd.DataFrame(
    [[0.8, 0.1, 0.1], [0.1, 0.1, 0.8], [0.1, 0.8, 0.1]], index=IDX3, columns=IDX3
)
UNIFORM3 = pd.Series([1 / 3] * 3, index=IDX3)


# ── unconditional_belief: cold start AND class prior ─────────────────────────────


class TestUnconditionalBelief:
    def test_is_the_in_window_frequency_zero_filled(self):
        states = pd.Series([0, 0, 0, 1, 1, 0])
        belief = unconditional_belief(states, state_index=[0, 1, 2, 3])
        assert list(belief.index) == [0, 1, 2, 3]
        assert belief.to_dict() == pytest.approx({0: 4 / 6, 1: 2 / 6, 2: 0.0, 3: 0.0})
        assert belief.sum() == pytest.approx(1.0, abs=1e-15)

    def test_label_outside_the_state_index_raises(self):
        with pytest.raises(ValueError, match=r"\[5\]"):
            unconditional_belief(pd.Series([0, 5]), state_index=IDX3)

    def test_nan_rows_are_ignored_not_counted(self):
        belief = unconditional_belief(pd.Series([0, np.nan, 1, 1]), state_index=[0, 1])
        assert belief.to_dict() == pytest.approx({0: 1 / 3, 1: 2 / 3})


# ── filter_step ──────────────────────────────────────────────────────────────────


class TestSumToOne:
    def test_belief_sums_to_one_over_a_20_step_sequence(self):
        """Fails on a missing normalization: the likelihood ratios below are far
        from summing to 1, so an unnormalized product drifts off 1.0 at step 1."""
        rng = np.random.default_rng(0)
        prior = pd.Series([0.5, 0.3, 0.2], index=IDX3)
        belief = prior.copy()
        for _ in range(20):
            raw = rng.uniform(0.05, 1.0, size=3)
            posterior = pd.Series(raw / raw.sum(), index=IDX3)
            belief = filter_step(belief, A_ROTATE, posterior, prior)
            assert abs(belief.sum() - 1.0) < 1e-12
            assert (belief >= 0).all()


class TestPredictAndUpdateOrder:
    def test_sharp_likelihood_dominates_a_diffuse_prior(self):
        """Uniform prior, sharp posterior on state 1. Expected argmax: 1.

        - Update dropped: belief = uniform A = uniform -> idxmax 0. Fails.
        - Transposed (update then predict): sharp-on-1 pushed through row 1 of
          A_ROTATE -> mass on state 2 -> argmax 2. Fails.
        The exact vector is [0.05, 0.9, 0.05] because uniform @ A_ROTATE is uniform.
        """
        posterior = pd.Series([0.05, 0.9, 0.05], index=IDX3)
        belief = filter_step(UNIFORM3, A_ROTATE, posterior, UNIFORM3)
        assert int(belief.idxmax()) == 1
        np.testing.assert_allclose(belief.to_numpy(), [0.05, 0.9, 0.05], atol=1e-12)

    def test_diffuse_likelihood_does_not_move_a_sharp_prior(self):
        """Sharp prior on state 1, near-flat posterior. Expected argmax: 2 —
        A_ROTATE sends state 1 to state 2, and the flat evidence cannot undo that.

        - Predict dropped: belief = prior x flat L -> argmax 1. Fails.
        Exact: prior @ A = [0.135, 0.135, 0.73]; x L = [0.9, 1.2, 0.9];
        -> [0.1215, 0.162, 0.657] / 0.9405.
        """
        prior_belief = pd.Series([0.05, 0.9, 0.05], index=IDX3)
        posterior = pd.Series([0.3, 0.4, 0.3], index=IDX3)
        belief = filter_step(prior_belief, A_ROTATE, posterior, UNIFORM3)
        assert int(belief.idxmax()) == 2
        expected = np.array([0.1215, 0.162, 0.657]) / 0.9405
        np.testing.assert_allclose(belief.to_numpy(), expected, atol=1e-12)


class TestAbsentClassRule:
    def test_absent_state_keeps_its_prediction_share_not_zero(self):
        """Posterior over {0, 1, 2} only; state 3 is absent (model.classes_ subset).

        Every row of A equals the class prior c, so the prediction step gives
        c = [0.3, 0.2, 0.1, 0.4] regardless of the prior belief. Update:
        [0.3*0.5/0.3, 0.2*0.3/0.2, 0.1*0.2/0.1, 0.4*1.0] = [0.5, 0.3, 0.2, 0.4],
        mass 1.4 -> state 3's belief is 0.4 / 1.4 = 2/7: its pure prediction value
        times a likelihood ratio of exactly 1.0, under the common normalization.
        Zero-filling the absent class would give 0.0.
        """
        idx = [0, 1, 2, 3]
        c = pd.Series([0.3, 0.2, 0.1, 0.4], index=idx)
        a = pd.DataFrame([c.to_numpy()] * 4, index=idx, columns=idx)
        posterior = pd.Series({0: 0.5, 1: 0.3, 2: 0.2})

        assert likelihood_ratio(posterior, c, state_index=idx).loc[3] == 1.0
        belief = filter_step(pd.Series([0.25] * 4, index=idx), a, posterior, c)
        assert belief.loc[3] == pytest.approx(2 / 7, abs=1e-12)
        np.testing.assert_allclose(belief.to_numpy(), np.array([0.5, 0.3, 0.2, 0.4]) / 1.4, atol=1e-12)


class TestZeroPriorWithPresentPosteriorRaises:
    def test_raises_naming_the_state(self):
        prior = pd.Series([0.5, 0.5, 0.0], index=IDX3)
        posterior = pd.Series([0.4, 0.4, 0.2], index=IDX3)
        with pytest.raises(ValueError, match="state 2 has class prior 0.0"):
            likelihood_ratio(posterior, prior, state_index=IDX3)
        with pytest.raises(ValueError, match="state 2"):
            filter_step(UNIFORM3, A_ROTATE, posterior, prior)

    def test_zero_prior_with_the_state_ABSENT_is_fine(self):
        """The raise must be about the combination, not about a zero prior alone."""
        prior = pd.Series([0.5, 0.5, 0.0], index=IDX3)
        ratio = likelihood_ratio(pd.Series({0: 0.6, 1: 0.4}), prior, state_index=IDX3)
        assert ratio.to_dict() == pytest.approx({0: 1.2, 1: 0.8, 2: 1.0})


class TestZeroMassRaises:
    def test_all_mass_on_a_zero_likelihood_state_raises_instead_of_nan(self):
        identity = pd.DataFrame(np.eye(3), index=IDX3, columns=IDX3)
        with pytest.raises(ValueError, match="pre-normalization mass"):
            filter_step(pd.Series([1.0, 0.0, 0.0], index=IDX3), identity, pd.Series([0.0, 0.5, 0.5], index=IDX3), UNIFORM3)


# ── predict_only_step: the missing-observation rule ─────────────────────────────


class TestPredictOnlyStepMovesTheBelief:
    def test_one_step_moves_and_repeated_steps_reach_the_stationary_distribution(self):
        """Fails if the missing-observation rule is implemented as a hold."""
        a = pd.DataFrame([[0.9, 0.1, 0.0], [0.0, 0.7, 0.3], [0.2, 0.0, 0.8]], index=IDX3, columns=IDX3)
        start = pd.Series([1.0, 0.0, 0.0], index=IDX3)

        one = predict_only_step(start, a)
        assert (one - start).abs().max() > 1e-6
        np.testing.assert_allclose(one.to_numpy(), [0.9, 0.1, 0.0], atol=1e-15)

        # Stationary distribution: left eigenvector of A for eigenvalue 1.
        vals, vecs = np.linalg.eig(a.to_numpy().T)
        stat = np.real(vecs[:, np.argmin(np.abs(vals - 1.0))])
        stat = stat / stat.sum()

        belief = start
        dist_first = float(np.abs(one.to_numpy() - stat).sum())
        for _ in range(500):
            belief = predict_only_step(belief, a)
        assert float(np.abs(belief.to_numpy() - stat).sum()) < 1e-9 < dist_first
        assert abs(belief.sum() - 1.0) < 1e-12


# ── purity ───────────────────────────────────────────────────────────────────────


class TestDeterminismAndPurity:
    def test_no_argument_is_mutated_and_repeated_calls_are_equal(self):
        prior_belief = pd.Series([0.2, 0.5, 0.3], index=IDX3)
        a = A_ROTATE.copy()
        posterior = pd.Series({0: 0.1, 2: 0.9})
        class_prior = pd.Series([0.4, 0.4, 0.2], index=IDX3)
        snapshots = [x.copy() for x in (prior_belief, a, posterior, class_prior)]

        first = filter_step(prior_belief, a, posterior, class_prior)
        second = filter_step(prior_belief, a, posterior, class_prior)

        pd.testing.assert_series_equal(first, second)
        pd.testing.assert_series_equal(prior_belief, snapshots[0])
        pd.testing.assert_frame_equal(a, snapshots[1])
        pd.testing.assert_series_equal(posterior, snapshots[2])
        pd.testing.assert_series_equal(class_prior, snapshots[3])


# ── transition_matrix_for ────────────────────────────────────────────────────────


class TestTransitionMatrixFor:
    def test_absent_from_row_is_the_unconditional_belief_and_warns(self, caplog):
        """State 2 is only the final label, so empirical_transition_matrix has no
        row for it. The row must be the window's unconditional distribution
        [3/6, 2/6, 1/6] — not uniform, not NaN — and a WARNING must name state 2."""
        states = pd.Series([0, 0, 1, 1, 0, 2])
        with caplog.at_level(logging.WARNING, logger="trading_crab_lib.platform.prediction.regime_filter"):
            a = transition_matrix_for(states, state_index=IDX3)
        assert list(a.index) == IDX3 and list(a.columns) == IDX3
        assert not a.isna().any().any()
        np.testing.assert_allclose(a.loc[2].to_numpy(), [3 / 6, 2 / 6, 1 / 6], atol=1e-15)
        assert not np.allclose(a.loc[2].to_numpy(), 1 / 3)
        np.testing.assert_allclose(a.sum(axis=1).to_numpy(), 1.0, atol=1e-15)
        assert any("state 2" in r.getMessage() and r.levelno == logging.WARNING for r in caplog.records)

    def test_observed_rows_equal_the_empirical_matrix(self):
        states = pd.Series([0, 1, 0, 1, 1, 0, 2, 2, 1])
        a = transition_matrix_for(states, state_index=IDX3)
        # from 0: ->1, ->1, ->2   from 1: ->0, ->1, ->0   from 2: ->2, ->1
        np.testing.assert_allclose(a.loc[0].to_numpy(), [0.0, 2 / 3, 1 / 3], atol=1e-15)
        np.testing.assert_allclose(a.loc[1].to_numpy(), [2 / 3, 1 / 3, 0.0], atol=1e-15)
        np.testing.assert_allclose(a.loc[2].to_numpy(), [0.0, 0.5, 0.5], atol=1e-15)

    def test_a_never_visited_state_gets_a_full_row_and_a_zero_column(self):
        a = transition_matrix_for(pd.Series([0, 1, 1, 0]), state_index=[0, 1, 2])
        assert a.loc[:, 2].tolist() == [0.0, 0.0, 0.0]
        np.testing.assert_allclose(a.loc[2].to_numpy(), [0.5, 0.5, 0.0], atol=1e-15)

    def test_output_plugs_straight_into_filter_step(self):
        states = pd.Series([0, 0, 1, 1, 0, 2])
        prior = unconditional_belief(states, state_index=IDX3)
        belief = filter_step(prior, transition_matrix_for(states, state_index=IDX3), pd.Series({0: 0.2, 1: 0.8}), prior)
        assert abs(belief.sum() - 1.0) < 1e-12
