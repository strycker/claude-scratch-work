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


# ── unconditional_belief: the cold start π_0 (and transition_matrix_for's fallback row) ──


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

        Every row of A equals c = [0.3, 0.2, 0.1, 0.4], so the prediction step gives
        c regardless of the prior belief. The likelihood divides by the nowcaster's
        TRAINING prior over {0, 1, 2}: c restricted to them and renormalized,
        [0.5, 1/3, 1/6]. Update: [0.3*0.5/0.5, 0.2*0.3/(1/3), 0.1*0.2/(1/6), 0.4*1.0]
        = [0.3, 0.18, 0.12, 0.4], mass 1.0 -> state 3's belief is exactly its
        prediction share, 0.4. Zero-filling the absent class would give 0.0.

        Re-derived in plan 08-17. The old expected value, 2/7 against a prediction
        share of 0.4, was CR-01's inflation artefact (08-REVIEW.md): dividing by the
        whole 4-state prior inflated every present state by 1/0.6, so a posterior
        carrying no information still pushed mass off the absent state.
        """
        idx = [0, 1, 2, 3]
        c = pd.Series([0.3, 0.2, 0.1, 0.4], index=idx)
        a = pd.DataFrame([c.to_numpy()] * 4, index=idx, columns=idx)
        posterior = pd.Series({0: 0.5, 1: 0.3, 2: 0.2})
        training_prior = c.loc[[0, 1, 2]] / c.loc[[0, 1, 2]].sum()

        assert likelihood_ratio(posterior, training_prior, state_index=idx).loc[3] == 1.0
        belief = filter_step(pd.Series([0.25] * 4, index=idx), a, posterior, training_prior)
        assert belief.loc[3] == pytest.approx(0.4, abs=1e-12)
        np.testing.assert_allclose(belief.to_numpy(), np.array([0.3, 0.18, 0.12, 0.4]), atol=1e-12)


class TestTheLikelihoodPriorIsTheTrainingPrior:
    """CR-01 (plan 08-17): the class prior must be the nowcaster's training prior over
    the posterior's own states. The old whole-window shape is refused, not repaired."""

    IDX4 = [0, 1, 2, 3]
    #: A sticky, non-uniform A so normalize(π A) is not π.
    A4 = pd.DataFrame(
        [[0.7, 0.1, 0.1, 0.1], [0.1, 0.6, 0.2, 0.1], [0.05, 0.15, 0.7, 0.1], [0.2, 0.2, 0.2, 0.4]],
        index=IDX4, columns=IDX4,
    )

    def test_a_posterior_equal_to_the_training_prior_is_no_evidence(self):
        """A calibrated posterior that just repeats the training prior carries no
        information: every ratio is 1.0 (present and absent states alike) and the
        belief is the prediction step alone. Fails if absent states are treated
        differently from present ones (CR-01's second paragraph)."""
        training_prior = pd.Series({0: 0.5, 1: 0.3, 2: 0.2})
        posterior = training_prior.copy()
        pi = pd.Series([0.1, 0.2, 0.3, 0.4], index=self.IDX4)

        ratio = likelihood_ratio(posterior, training_prior, state_index=self.IDX4)
        assert ratio.to_dict() == {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0}
        predicted = pi.to_numpy() @ self.A4.to_numpy()
        belief = filter_step(pi, self.A4, posterior, training_prior)
        np.testing.assert_allclose(belief.to_numpy(), predicted / predicted.sum(), rtol=0, atol=1e-15)

        # The old call shape (a whole-window prior with mass on state 3, which the
        # posterior lacks) is refused rather than inflating states 0-2 by 1/0.6.
        window_prior = pd.Series([0.3, 0.18, 0.12, 0.4], index=self.IDX4)
        with pytest.raises(ValueError):
            likelihood_ratio(posterior, window_prior, state_index=self.IDX4)

    def test_a_prior_with_mass_on_a_state_the_posterior_lacks_is_refused(self):
        """The whole-window shape. Fails if it is accepted (the pre-08-17 behaviour) or
        silently renormalized over the posterior's states."""
        posterior = pd.Series({0: 0.5, 1: 0.3, 2: 0.2})
        window_prior = pd.Series([0.3, 0.2, 0.1, 0.4], index=self.IDX4)
        msg = r"state\(s\) \[3\].*training prior.*fit_l2_nowcaster"
        with pytest.raises(ValueError, match=msg):
            likelihood_ratio(posterior, window_prior, state_index=self.IDX4)
        with pytest.raises(ValueError, match=msg):
            filter_step(pd.Series([0.25] * 4, index=self.IDX4), self.A4, posterior, window_prior)

    def test_a_prior_that_does_not_sum_to_one_is_refused(self):
        """Support right, mass wrong: e.g. counts divided by the wrong total. Fails if
        accepted or renormalized."""
        posterior = pd.Series({0: 0.5, 1: 0.3, 2: 0.2})
        for bad in (pd.Series({0: 0.3, 1: 0.2, 2: 0.1}), pd.Series({0: 0.5, 1: 0.3, 2: 0.2 + 1e-6})):
            with pytest.raises(ValueError, match="sums to"):
                likelihood_ratio(posterior, bad, state_index=self.IDX4)
        # Within 1e-9 of 1 is a float-rounding difference, not a wrong prior.
        ok = pd.Series({0: 0.5, 1: 0.3, 2: 0.2 + 1e-12})
        assert likelihood_ratio(posterior, ok, state_index=self.IDX4).loc[3] == 1.0


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
        # The training prior over the posterior's classes_ {0, 2} (plan 08-17; was the
        # 3-state window shape [0.4, 0.4, 0.2], which likelihood_ratio now refuses).
        class_prior = pd.Series({0: 2 / 3, 2: 1 / 3})
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
        start = unconditional_belief(states, state_index=IDX3)
        # The likelihood's prior is the nowcaster's training prior over its classes_ {0, 1}
        # (plan 08-17), not the window distribution over all three states.
        training_prior = pd.Series({0: 0.6, 1: 0.4})
        transition = transition_matrix_for(states, state_index=IDX3)
        belief = filter_step(start, transition, pd.Series({0: 0.2, 1: 0.8}), training_prior)
        assert abs(belief.sum() - 1.0) < 1e-12
