"""
Criterion 7 — the joint (classifier #1 x classifier #2) allocation-lift harness.

**Criterion 7, in its own words:** does adding a second L1 labeler help, measured
walk-forward, on the same harness and the same window as the classifier-#1-alone
leg it is compared against — and does that help survive deflation for the number
of configurations this project has actually searched?

The measurement is structured as a **one-parameter ablation**. ``run_joint_backtest``
at ``blend_weight_1 = 1.0`` IS the #1-alone baseline leg: same loop, same steps,
same refits, same cost model, same ADR-0001 condition-(iv) shrinkage path. Only
the blend weight moves. That is what guarantees the two legs cannot silently
diverge in window or cost model — the failure ``.planning/UAT-AUDIT-2026-09-09.md``
documents and threat T-07-26 names.

**Routing (ADR-0002 decision (e), pinned before any run).** The criterion-7 lift
is routed **L1-only and decision-bearing**: each classifier's per-step probability
vector is a degenerate one-hot on its own LAST FILTERED STATE — the state it
assigns to the most recent month of ``train_index``, which is strictly before the
decision date. No L2 nowcaster is consulted on this path
(:data:`ROUTING_L1_ONLY`). A second routing (:data:`ROUTING_L2_NOWCAST`) runs each
classifier's labels through ``driver.py::_refit_l2``; ADR-0002 makes that leg
**observational and firewalled** — it is appended with
``honesty.registry.NO_REGISTRY`` so it contributes zero rows and counts toward
neither D-16's deflated-Sharpe denominator nor D-17's ceiling, and **nothing
downstream in phase 7 may change on the basis of it**.

**What the routing means for the measurement window.** The L1-only path has no
L2 refit, so the dominant degradation mechanism in wave 1 — an early small
post-embargo window starving a K-fold, which degraded 232 of 588 steps and
narrowed the comparison window to 356 months — is absent. The L1-only legs
therefore run over the FULL decision range. The direct consequence, stated here
rather than footnoted: **these deltas are NOT comparable to wave 1's
``wealth_delta`` +0.377847 and ``dd_delta`` -0.066124**, which were measured
through L2 on 356 steps ending 2017-05. The #1-alone baseline is re-run through
this harness; it is never reused from wave 1.

**Degradation policy, and why it is symmetric.** A step degrades — holds the
previous weights, is marked ``degraded`` and is excluded from the per-step
metrics — if EITHER classifier fails to produce a usable labeling. Neither refit
depends on ``blend_weight_1``, so the degraded-step SET is identical across the
two legs by construction, not merely their index. Degrading only the failing
classifier would have made a #2-failure invisible at weight 1.0 and material at
weight 0.5 — two changes, not one. Per-classifier degrade counts are still
reported (``n_degraded_classifier_1`` / ``_2``) so the cause is not hidden.

**Exactly one registry row per call.** ``run_full_backtest_evaluation``'s factor
of two comes from its own two ``append_trial`` sites (strategy + ablation) and
does not apply here: this harness traverses one site per run.

**``driver.py`` is imported, never modified.** ``_refit_l1`` and ``_refit_l2`` are
reused as-is so the frozen-features policy stays single (criterion 1). One seam
is unavoidable: ``_refit_l1`` calls ``canonicalize_states`` with its default
``sort_column='trailing_return_1m'``, a column classifier #2's disjoint feature
set (D-10) does not contain, so classifier #2 needs a refit entry point that
passes its own ordering column. :func:`_refit_classifier2` is that entry point;
it composes the same public helpers, and
``TestRefitParity::test_refit_classifier2_reproduces_refit_l1_on_classifier_ones_inputs``
pins that it reproduces ``_refit_l1`` exactly on classifier #1's own inputs — so
it is a seam, not a fork.

**The Bayes filter (plan 08-08, ROADMAP criterion 1) applies to the l2 routing
ONLY.** Under :data:`ROUTING_L2_NOWCAST` (and ``use_regime_filter=True``) each
classifier's raw nowcaster posterior is filtered into a belief —
``prediction/regime_filter.py``'s ``π_t ∝ [π_{t−1} A] · L_t`` — carried across
steps as a LOOP VARIABLE (``prev_belief_1`` / ``prev_belief_2``), and the
**belief**, never the raw posterior, is what ``update_active_regime`` and
``blend_regime_tilts`` consume. ``A`` and the class prior come from the step's own
in-window labels (``states_N``, ``train_index`` only); the cold start is
``unconditional_belief`` on those same labels — the one rule ``driver.py`` and
``report/weekly.py`` also use. A degraded step has no observation: the belief
advances by ``predict_only_step`` (``π A``) when that step's labels exist, and is
held with a WARNING when they do not. The raw posterior is still accumulated into
``per_step_metrics_N`` (so raw-posterior churn stays measurable as a control); the
belief goes into ``per_step_belief_N``. Under the decision-bearing
:data:`ROUTING_L1_ONLY` the filter is **not applied**: there ``probs`` is
``_last_state_one_hot(states)`` — a label, not a likelihood — and filtering it
would fabricate a posterior the routing declares does not exist and would move the
decision-bearing leg for a wiring reason. The gate is a literal
``routing == ROUTING_L2_NOWCAST`` test in code, not a property of the data; the
l1only curve is pinned bit-for-bit with the filter on and off.

A known approximation, stated rather than absorbed (found in plan 08-06): the
class prior that inverts the posterior into a likelihood is the distribution of
the WHOLE in-window label series, whereas the nowcaster trains on the D-01
embargoed subset (trailing ``embargo_months`` dropped, non-finite rows dropped).
The superset cannot fire the zero-prior raise on a state the nowcaster saw; the
two priors differ slightly all the same.

A third inconsistency, recorded here and left for plan 08-09 (which owns the §5.3
wiring): ``update_active_regime`` receives classifier #1's probabilities ALONE
while ``blend_regime_tilts`` trades BOTH classifiers — the hysteresis tracks one
classifier while the tilt blends two. It is left unchanged here so this plan's
before/after comparison stays clean.

**Plausibility bands.** The band constants below are ``07-BANDS.md`` §8's
confirmed dispositions (Glenn, 2026-09-18), recorded before any joint-lift number
existed. Two tiers, and only one governs a verdict: a **universal/arithmetic**
breach means the MEASUREMENT is broken (halt; do not report a lift); a
**domain/advisory** breach is a recorded note and criterion 7 still reports.
``DD_DELTA_UNIVERSAL`` is the REVISED ``[-1, 1]``, not the retired ``[-2, 2]`` —
the old bound was wider than the quantity's own arithmetic range
(``max_drawdown`` in ``[-1, 0]`` per leg) and could therefore only confirm.

Usage::

    from trading_crab_lib.platform.backtest.joint_driver import (
        joint_lift_table, run_joint_backtest,
    )

    joint, joint_meta = run_joint_backtest(
        features_1, asset_returns, cfg, blend_weight_1=0.50,
        features_2=features_2, frozen_features_1=f1, frozen_features_2=f2,
        cash_returns=cash, trial_tag="07-11-joint",
    )
    base, base_meta = run_joint_backtest(
        features_1, asset_returns, cfg, blend_weight_1=1.00, ...
    )
    lift = joint_lift_table(joint, base)   # deltas AND their window, one mapping
"""

from __future__ import annotations

import dataclasses
import logging
import time
from typing import Any

import pandas as pd

from trading_crab_lib.platform.allocation.hysteresis import update_active_regime
from trading_crab_lib.platform.allocation.joint_tilt import blend_regime_tilts
from trading_crab_lib.platform.assets.returns import returns_by_regime_stats
from trading_crab_lib.platform.backtest.costs import apply_transaction_cost, compute_turnover
from trading_crab_lib.platform.backtest.driver import (
    _L2_DEGRADE_EXCEPTIONS,
    _realized_return,
    _refit_l1,
    _refit_l2,
)
from trading_crab_lib.platform.evaluation.kpis import max_drawdown_and_duration, terminal_log_wealth
from trading_crab_lib.platform.honesty import registry
from trading_crab_lib.platform.honesty.holdout import DEFAULT_HOLDOUT_CUTOFF, split_by_holdout_boundary
from trading_crab_lib.platform.honesty.walkforward import expanding_steps
from trading_crab_lib.platform.labeling.classifier2 import classifier2_config
from trading_crab_lib.platform.labeling.jump_model import (
    canonicalize_states,
    fit_jump_model,
    standardize_features,
)
from trading_crab_lib.platform.prediction.regime_filter import (
    filter_step,
    predict_only_step,
    transition_matrix_for,
    unconditional_belief,
)

log = logging.getLogger(__name__)

_PROGRESS_EVERY_STEPS = 24

#: ADR-0002 decision (e): the DECISION-BEARING routing. Each classifier's
#: per-step probability vector is a degenerate one-hot on its own last filtered
#: state (the state assigned to the most recent month of ``train_index``, which
#: is strictly before the decision date). No L2 nowcaster is consulted.
ROUTING_L1_ONLY: str = "L1_ONLY_LAST_FILTERED_STATE"

#: ADR-0002 decision (e): the OBSERVATIONAL, FIREWALLED routing. Each
#: classifier's labels are run through ``driver.py::_refit_l2``'s calibrated
#: nowcaster. Must be appended with ``registry.NO_REGISTRY``; contributes zero
#: rows and counts toward neither D-16 nor D-17. Nothing downstream in phase 7
#: may change on the basis of a number produced under this routing.
ROUTING_L2_NOWCAST: str = "L2_NOWCAST"

_ROUTINGS = (ROUTING_L1_ONLY, ROUTING_L2_NOWCAST)

# ── 07-BANDS.md §8 — Glenn's confirmed dispositions, 2026-09-18 ─────────────

#: Universal / GOVERNING bound on ``wealth_delta`` (nats). A breach means the
#: measurement is broken — halt, do not report a lift.
WEALTH_DELTA_UNIVERSAL: float = 15.0

#: Domain / ADVISORY trigger on ``wealth_delta``. A breach is a recorded note;
#: criterion 7 still reports (D-07; A11 stays open by design).
WEALTH_DELTA_DOMAIN: float = 5.0

#: Universal / GOVERNING bound on ``dd_delta`` (fraction of peak). REVISED from
#: the retired ``[-2, 2]``: each leg's ``max_drawdown`` lies in ``[-1, 0]``, so
#: their difference is arithmetically confined to ``[-1, 1]`` and a wider bound
#: could only ever confirm. A breach here PROVES a leg's ``max_drawdown`` is not
#: a fraction in ``[-1, 0]``, i.e. the KPI is broken.
DD_DELTA_UNIVERSAL: tuple[float, float] = (-1.0, 1.0)

#: Domain / ADVISORY trigger on ``dd_delta``.
DD_DELTA_DOMAIN: float = 0.5


@dataclasses.dataclass
class JointStepRecord:
    """One walk-forward step of the joint harness.

    ``degraded`` steps are still RECORDED (with the previous weights held) so
    the equity-curve index never silently shrinks — two legs covering different
    months is exactly the failure this harness exists to prevent.
    """

    date: pd.Timestamp
    ret: float
    turnover: float
    cost: float
    active_regime: Any
    scale: float
    degraded: bool
    state_1: Any = None
    state_2: Any = None

    def as_row(self) -> dict[str, Any]:
        """The equity-curve row this record contributes (``driver.py``'s schema)."""
        return {
            "date": self.date,
            "return": self.ret,
            "turnover": self.turnover,
            "cost": self.cost,
            "active_regime": self.active_regime,
            "scale": self.scale,
            "degraded": self.degraded,
            "state_1": self.state_1,
            "state_2": self.state_2,
        }


def _refit_classifier2(
    train_features: pd.DataFrame,
    *,
    frozen_features: list[str],
    K: int,
    lam: float,
    n_restarts: int,
    sort_column: str,
) -> pd.Series:
    """``driver.py::_refit_l1``'s frozen path plus an explicit ``sort_column``.

    Exists ONLY because ``_refit_l1`` hardcodes ``canonicalize_states``' default
    ordering column (``trailing_return_1m``), which classifier #2's disjoint
    feature set (D-10) does not contain — so ``_refit_l1`` cannot fit classifier
    #2 at all, and ``driver.py`` must stay unmodified. Every other step (frozen
    column resolution in declaration order, the two loud boundary failures,
    dropna, standardize, fit, canonicalize) is the same composition of the same
    public helpers. Parity with ``_refit_l1`` on classifier #1's own inputs is
    pinned by ``TestRefitParity`` — this is a seam, not a fork.

    Raises:
        ValueError: if ``frozen_features`` resolves to 0 usable columns, or to
            fewer than ``K``, or if ``sort_column`` is absent from the resolved
            columns (propagated from ``canonicalize_states``, which has had no
            fallback since plan 07-05).
    """
    active = [c for c in frozen_features if c in train_features.columns]
    if len(active) == 0:
        raise ValueError(
            f"frozen_features resolved to 0 usable columns (of {len(frozen_features)} "
            "requested) — none are present in train_features.columns."
        )
    if len(active) < K:
        raise ValueError(
            f"frozen_features resolved to only {len(active)} usable column(s), fewer "
            f"than K={K} — cannot fit a {K}-state jump model on fewer feature columns "
            "than states."
        )

    X_df = train_features[active].dropna(axis=0, how="any")
    used_cols = list(X_df.columns)
    X = standardize_features(X_df)
    fit = fit_jump_model(X, K=K, lam=lam, n_restarts=n_restarts)
    states, _centroids = canonicalize_states(
        fit["states"], fit["centroids"], used_cols, sort_column=sort_column
    )
    return pd.Series(states, index=X_df.index, name="state")


def _last_state_one_hot(states: pd.Series) -> pd.Series:
    """The degenerate probability vector the L1-only routing feeds the tilt.

    One-hot on the labeler's most recent filtered state. Causal: ``states`` comes
    from a fit on ``train_index`` only, which ``expanding_steps`` guarantees is
    strictly before the decision date, and the month selected is the LAST of
    that window.
    """
    if states is None or states.empty:
        return pd.Series(dtype=float)
    return pd.Series({states.iloc[-1]: 1.0}, dtype=float)


def _occupancy(states: pd.Series) -> pd.Series:
    """Per-state share of the training window — ADR-0001 condition (iv)'s input.

    The driver holds the actual state Series, so it passes measured occupancy to
    ``blend_regime_tilts`` rather than letting it fall back to
    ``regime_occupancy``'s ``n_obs``-based estimate.
    """
    if states is None or states.empty:
        return pd.Series(dtype=float)
    return states.value_counts(normalize=True).astype(float)


def _filtered_belief(
    prev_belief: pd.Series | None,
    states: pd.Series,
    posterior: pd.Series,
    *,
    state_index: list[int],
) -> pd.Series:
    """One l2 filter step for one classifier, from this step's own in-window labels.

    ``A`` and the class prior are built from ``states`` (``train_index`` only). The
    cold start — no previous belief — is ``unconditional_belief`` on the same labels,
    i.e. the class prior itself: the one rule shared with ``driver.py`` and
    ``report/weekly.py``.
    """
    prior = unconditional_belief(states, state_index=state_index)
    transition = transition_matrix_for(states, state_index=state_index)
    start = prior if prev_belief is None else prev_belief
    return filter_step(start, transition, posterior, prior)


def _advance_without_observation(
    prev_belief: pd.Series | None,
    states: pd.Series,
    *,
    state_index: list[int],
    t: Any,
    which: int,
) -> pd.Series | None:
    """The degraded-step rule: ``predict_only_step`` when labels exist, else hold.

    Weights are held either way; this only sets the NEXT step's prior. ``None``
    stays ``None`` — there is nothing to advance before the first observation.
    """
    if prev_belief is None:
        return None
    if states is None or len(states.dropna()) < 2:
        log.warning(
            "Step %s: classifier #%d has no in-window labels on this degraded step — "
            "holding its filtered belief unchanged (no A to advance by)", t, which,
        )
        return prev_belief
    return predict_only_step(prev_belief, transition_matrix_for(states, state_index=state_index))


def _classifier2_params(cfg: dict[str, Any]) -> dict[str, Any]:
    """``classifier2_config`` with a defensive fallback for synthetic test configs."""
    return classifier2_config(cfg)


def run_joint_backtest(
    monthly_features: pd.DataFrame,
    asset_returns: pd.DataFrame,
    cfg: dict[str, Any],
    *,
    blend_weight_1: float,
    features_2: pd.DataFrame | None = None,
    frozen_features_1: list[str] | None = None,
    frozen_features_2: list[str] | None = None,
    routing: str = ROUTING_L1_ONLY,
    use_regime_filter: bool = True,
    min_train: int | None = None,
    cash_returns: pd.Series | None = None,
    registry_path: Any = None,
    trial_tag: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """One expanding-window loop producing EITHER leg of criterion 7's ablation.

    At ``blend_weight_1 = 1.0`` the result is the classifier-#1-alone baseline
    leg; strictly between 0 and 1 it is the joint leg. Both legs visit the
    identical step sequence and share an identical degraded-step set, so their
    equity-curve indexes are equal element-wise, not merely in length.

    Args:
        monthly_features: classifier #1's causal feature frame (may physically
            extend past the holdout cutoff — it is split first).
        asset_returns: monthly simple returns per tradable asset.
        cfg: platform config.
        blend_weight_1: weight on classifier #1's tilt; its complement goes to
            classifier #2. Must lie in ``[0, 1]`` (validated inside
            ``blend_regime_tilts``).
        features_2: classifier #2's causal feature frame, indexed IDENTICALLY to
            ``monthly_features``. ``None`` means "classifier #2's columns live in
            ``monthly_features`` too" (combined-frame mode), in which case both
            classifiers' L2 nowcasters, under :data:`ROUTING_L2_NOWCAST`, see the
            same frame.
        frozen_features_1: classifier #1's frozen column list (ADR-0001's ten),
            threaded to ``_refit_l1`` unchanged. ``None`` falls back to
            ``_refit_l1``'s own per-window admission rule.
        frozen_features_2: classifier #2's frozen column list
            (``freeze_classifier2_columns``' output). ``None`` falls back to
            ``labeling_2.features``.
        routing: :data:`ROUTING_L1_ONLY` (decision-bearing) or
            :data:`ROUTING_L2_NOWCAST` (observational; MUST be paired with
            ``registry_path=registry.NO_REGISTRY``).
        use_regime_filter: apply the Bayes filter under :data:`ROUTING_L2_NOWCAST`
            (plan 08-08). Has no effect under :data:`ROUTING_L1_ONLY`, where the
            filter is never applied. ``False`` reproduces the pre-08-08 l2 leg.
        min_train: overrides ``cfg["backtest"]["min_train_months"]``.
        cash_returns: the cash sleeve's own return series (review F4).
        registry_path: ledger path, or ``registry.NO_REGISTRY`` for zero rows.
        trial_tag: names what was evaluated. ``None`` surfaces
            ``append_trial``'s refusal, which is the intended behavior — an
            untagged row still counts toward D-16.

    Returns:
        ``(equity_curve, metadata)``. ``equity_curve`` is indexed by decision
        date with ``driver.py``'s columns plus ``state_1``/``state_2``.
        ``metadata`` carries ``routing``, ``blend_weight_1``, ``n_steps``,
        ``n_degraded``, ``n_degraded_classifier_1``, ``n_degraded_classifier_2``,
        ``first_date``, ``last_date``, ``records``, each classifier's
        ``per_step_metrics`` (the RAW posterior) and ``per_step_belief`` (the
        filtered belief; empty unless the filter ran).
    """
    if routing not in _ROUTINGS:
        raise ValueError(
            f"unknown routing {routing!r} — must be one of {_ROUTINGS}. The "
            "criterion-7 routing is pinned by ADR-0002 decision (e); a routing "
            "this module does not implement cannot be silently approximated."
        )

    backtest_cfg = cfg.get("backtest", {})
    if min_train is None:
        min_train = backtest_cfg.get("min_train_months", 120)
    cost_bps = backtest_cfg.get("cost_bps", 10)

    allocation_cfg = cfg.get("allocation", {})
    target_vol_annual = allocation_cfg.get("target_vol_annual", 0.10)
    halflife = allocation_cfg.get("ewma_halflife_months", 6)
    portfolio_vol_min_obs = allocation_cfg.get("portfolio_vol_min_obs", 12)
    hysteresis_cfg = allocation_cfg.get("hysteresis", {})
    act_threshold = hysteresis_cfg.get("act_threshold", 0.70)
    unwind_threshold = hysteresis_cfg.get("unwind_threshold", 0.40)

    c2 = _classifier2_params(cfg)
    if frozen_features_2 is None:
        frozen_features_2 = list(c2["features"])

    if features_2 is None:
        features_2 = monthly_features
    elif not features_2.index.equals(monthly_features.index):
        raise ValueError(
            "features_2 must share monthly_features' index exactly — the two "
            f"classifiers must be labeled on the same months. Got "
            f"{len(features_2.index)} vs {len(monthly_features.index)} rows, "
            f"identical={features_2.index.equals(monthly_features.index)}."
        )

    # T-07-30 / T-05-03: the holdout boundary is applied BEFORE expanding_steps
    # is constructed, so the visited index never exceeds the cutoff whatever the
    # input frames physically contain.
    dev_features_1, _ = split_by_holdout_boundary(monthly_features, cutoff=DEFAULT_HOLDOUT_CUTOFF)
    dev_features_2, _ = split_by_holdout_boundary(features_2, cutoff=DEFAULT_HOLDOUT_CUTOFF)
    dev_asset_returns, _ = split_by_holdout_boundary(asset_returns, cutoff=DEFAULT_HOLDOUT_CUTOFF)

    records: list[JointStepRecord] = []
    per_step_1: dict[str, list] = {"dates": [], "proba": [], "classes": []}
    per_step_2: dict[str, list] = {"dates": [], "proba": [], "classes": []}
    per_step_belief_1: dict[str, list] = {"dates": [], "proba": [], "classes": []}
    per_step_belief_2: dict[str, list] = {"dates": [], "proba": [], "classes": []}
    # The filter's recursion state — a LOOP VARIABLE, not a feature (plan 08-08).
    prev_belief_1: pd.Series | None = None
    prev_belief_2: pd.Series | None = None
    state_index_1 = list(range(int(cfg.get("labeling", {}).get("K", 5))))
    state_index_2 = list(range(int(c2["K"])))

    prev_weights: pd.Series = pd.Series(dtype=float)
    prev_active_regime: int | None = None
    prev_cash: float = 1.0

    n_degraded_1 = 0
    n_degraded_2 = 0

    steps = list(expanding_steps(dev_features_1.index, min_train=min_train))
    total_steps = len(steps)
    started = time.monotonic()
    log.info(
        "Joint backtest [%s, blend_weight_1=%.2f]: %d monthly steps from %s to %s",
        routing, blend_weight_1, total_steps,
        steps[0][0].date() if steps else "n/a",
        steps[-1][0].date() if steps else "n/a",
    )

    for step_no, (t, train_index, test_index) in enumerate(steps, start=1):
        if step_no % _PROGRESS_EVERY_STEPS == 0 or step_no == total_steps:
            elapsed = time.monotonic() - started
            rate = step_no / elapsed if elapsed > 0 else 0.0
            log.info(
                "Joint backtest progress: %d/%d (%.0f%%) — %s — %.1fs elapsed, ~%.0fs remaining",
                step_no, total_steps, 100.0 * step_no / total_steps, t.date(), elapsed,
                (total_steps - step_no) / rate if rate > 0 else 0.0,
            )

        train_1 = dev_features_1.loc[train_index]
        train_2 = dev_features_2.loc[train_index]
        degraded = False
        states_1 = pd.Series(dtype=float)
        states_2 = pd.Series(dtype=float)
        probs_1 = pd.Series(dtype=float)
        probs_2 = pd.Series(dtype=float)

        try:
            states_1 = _refit_l1(train_1, cfg, frozen_features=frozen_features_1)
        except _L2_DEGRADE_EXCEPTIONS as exc:
            log.warning("Step %s: classifier #1 refit degraded — holding previous weights: %s", t, exc)
            degraded = True
            n_degraded_1 += 1

        if not degraded:
            try:
                states_2 = _refit_classifier2(
                    train_2,
                    frozen_features=frozen_features_2,
                    K=c2["K"], lam=c2["lam"], n_restarts=c2["n_restarts"],
                    sort_column=c2["sort_column"],
                )
            except _L2_DEGRADE_EXCEPTIONS as exc:
                log.warning("Step %s: classifier #2 refit degraded — holding previous weights: %s", t, exc)
                degraded = True
                n_degraded_2 += 1

        if not degraded:
            if routing == ROUTING_L1_ONLY:
                probs_1 = _last_state_one_hot(states_1)
                probs_2 = _last_state_one_hot(states_2)
            else:
                # Each classifier's nowcaster is caught SEPARATELY so the degrade
                # count attributes the failure to the labeler that actually
                # starved the K-fold. A single try around both would report every
                # L2 failure against one classifier — a count that reads as
                # evidence about classifier #2 while measuring something else.
                try:
                    probs_1 = _refit_l2(train_1, states_1, dev_features_1.loc[[t]], cfg)
                except _L2_DEGRADE_EXCEPTIONS as exc:
                    log.warning(
                        "Step %s: classifier #1 L2 refit degraded (RESEARCH Pitfall 2) "
                        "— holding previous weights: %s", t, exc,
                    )
                    degraded = True
                    n_degraded_1 += 1
                if not degraded:
                    try:
                        probs_2 = _refit_l2(train_2, states_2, dev_features_2.loc[[t]], cfg)
                    except _L2_DEGRADE_EXCEPTIONS as exc:
                        log.warning(
                            "Step %s: classifier #2 L2 refit degraded (RESEARCH Pitfall 2) "
                            "— holding previous weights: %s", t, exc,
                        )
                        degraded = True
                        n_degraded_2 += 1

        # What the allocator consumes. Under l1only this is the one-hot, untouched.
        belief_1, belief_2 = probs_1, probs_2
        filtered = False
        if routing == ROUTING_L2_NOWCAST and use_regime_filter:
            if degraded:
                prev_belief_1 = _advance_without_observation(
                    prev_belief_1, states_1, state_index=state_index_1, t=t, which=1
                )
                prev_belief_2 = _advance_without_observation(
                    prev_belief_2, states_2, state_index=state_index_2, t=t, which=2
                )
            else:
                belief_1 = _filtered_belief(prev_belief_1, states_1, probs_1, state_index=state_index_1)
                belief_2 = _filtered_belief(prev_belief_2, states_2, probs_2, state_index=state_index_2)
                prev_belief_1, prev_belief_2 = belief_1, belief_2
                filtered = True

        if degraded:
            new_weights = prev_weights
            new_active_regime = prev_active_regime
            new_cash = prev_cash
        else:
            train_returns = dev_asset_returns.loc[train_index]
            stats_1 = returns_by_regime_stats(train_returns, states_1)
            stats_2 = returns_by_regime_stats(train_returns, states_2)
            new_active_regime = update_active_regime(
                belief_1, prev_active_regime,
                act_threshold=act_threshold, unwind_threshold=unwind_threshold,
            )
            tilt = blend_regime_tilts(
                belief_1, stats_1,
                belief_2, stats_2,
                train_returns,
                weight_1=blend_weight_1,
                target_vol_annual=target_vol_annual,
                halflife=halflife,
                min_obs=portfolio_vol_min_obs,
                occupancy_1=_occupancy(states_1),
                occupancy_2=_occupancy(states_2),
            )
            new_weights = tilt["weights"]
            new_cash = tilt["cash"]

        turnover = compute_turnover(prev_weights, new_weights)
        test_date = test_index[0]
        asset_return_row = dev_asset_returns.loc[test_date]
        cash_ret = float(cash_returns.loc[test_date]) if cash_returns is not None else 0.0
        gross = _realized_return(new_weights, new_cash, asset_return_row, cash_return=cash_ret)
        net = apply_transaction_cost(gross, turnover, cost_bps)

        records.append(
            JointStepRecord(
                date=test_date,
                ret=net,
                turnover=turnover,
                cost=gross - net,
                active_regime=new_active_regime,
                scale=float(new_weights.sum()) if len(new_weights) else 0.0,
                degraded=degraded,
                state_1=(None if states_1.empty else states_1.iloc[-1]),
                state_2=(None if states_2.empty else states_2.iloc[-1]),
            )
        )

        if not degraded:
            for bucket, probs in ((per_step_1, probs_1), (per_step_2, probs_2)):
                bucket["dates"].append(t)
                bucket["proba"].append(probs.values)
                bucket["classes"].append(list(probs.index))
            if filtered:
                for bucket, belief in ((per_step_belief_1, belief_1), (per_step_belief_2, belief_2)):
                    bucket["dates"].append(t)
                    bucket["proba"].append(belief.values)
                    bucket["classes"].append(list(belief.index))

        prev_weights = new_weights
        prev_active_regime = new_active_regime
        prev_cash = new_cash

    columns = ["return", "turnover", "cost", "active_regime", "scale", "degraded", "state_1", "state_2"]
    equity_curve = (
        pd.DataFrame([r.as_row() for r in records]).set_index("date")
        if records
        else pd.DataFrame(columns=columns)
    )

    tlw = terminal_log_wealth(equity_curve["return"]) if not equity_curve.empty else 0.0
    n_degraded = int(sum(1 for r in records if r.degraded))

    trial_config = {
        "phase": "07-regime-representation",
        "plan": "07-11",
        "criterion": 7,
        "routing": routing,
        "blend_weight_1": float(blend_weight_1),
        "min_train": min_train,
        "cost_bps": cost_bps,
        "K_1": cfg.get("labeling", {}).get("K"),
        "lambda_1": cfg.get("labeling", {}).get("lambda"),
        "K_2": c2["K"],
        "lambda_2": c2["lam"],
        "features_1": list(frozen_features_1) if frozen_features_1 else [],
        "features_2": list(frozen_features_2),
    }
    if trial_tag is not None:
        trial_config["trial_tag"] = trial_tag
    # EXACTLY ONE append_trial site in this module — run_full_backtest_evaluation's
    # factor of two comes from its own two sites and does not apply here.
    registry.append_trial(
        config=trial_config,
        features=list(frozen_features_1 or []) + list(frozen_features_2),
        metrics={
            "n_steps": int(len(equity_curve)),
            "terminal_log_wealth": tlw,
            "n_degraded": n_degraded,
        },
        path=registry_path,
    )

    metadata: dict[str, Any] = {
        "routing": routing,
        "blend_weight_1": float(blend_weight_1),
        "n_steps": int(len(equity_curve)),
        "n_degraded": n_degraded,
        "n_degraded_classifier_1": n_degraded_1,
        "n_degraded_classifier_2": n_degraded_2,
        "first_date": (equity_curve.index.min() if not equity_curve.empty else None),
        "last_date": (equity_curve.index.max() if not equity_curve.empty else None),
        "terminal_log_wealth": tlw,
        "records": records,
        "per_step_metrics_1": per_step_1,
        "per_step_metrics_2": per_step_2,
        "per_step_belief_1": per_step_belief_1,
        "per_step_belief_2": per_step_belief_2,
        "use_regime_filter": bool(use_regime_filter and routing == ROUTING_L2_NOWCAST),
        "frozen_features_1": list(frozen_features_1 or []),
        "frozen_features_2": list(frozen_features_2),
        "registry_row_written": registry_path != registry.NO_REGISTRY,
    }
    return equity_curve, metadata


def joint_lift_table(joint_curve: pd.DataFrame, baseline_curve: pd.DataFrame) -> dict[str, Any]:
    """Both criterion-7 axes, each inseparable from the window it was measured on.

    The deltas are computed over the INTERSECTION of the two curves' indexes, and
    that intersection's size and endpoints are returned in the SAME mapping as
    the numbers. A caller physically cannot obtain a delta without its window —
    which is the binding condition wave 1's UAT attached to criterion 3 and the
    orchestrator extended to criterion 7.

    Under this harness's design the two indexes should be identical. A difference
    means one leg degraded steps the other did not, so it is logged at WARNING
    naming BOTH sizes and reported as ``indexes_identical=False`` rather than
    silently truncated.

    Returns:
        dict with ``wealth_delta``, ``dd_delta``, each leg's own
        ``terminal_log_wealth`` / ``max_drawdown`` / ``duration_months``, the
        window (``n_steps``, ``first_date``, ``last_date``, ``n_steps_joint``,
        ``n_steps_baseline``, ``indexes_identical``), and the ``07-BANDS.md`` §8
        verdicts (``*_universal_ok`` governs; ``*_domain_note`` is advisory).
    """
    j_index = joint_curve.index
    b_index = baseline_curve.index
    identical = bool(j_index.equals(b_index))
    if not identical:
        log.warning(
            "joint_lift_table: the two legs' equity-curve indexes are NOT identical "
            "— joint has %d steps, baseline has %d steps. The deltas below are "
            "computed over their intersection (%d steps) and are NOT a clean "
            "single-parameter ablation: one leg degraded steps the other did not.",
            len(j_index), len(b_index), len(j_index.intersection(b_index)),
        )

    common = j_index.intersection(b_index)
    j_ret = joint_curve.loc[common, "return"].dropna()
    b_ret = baseline_curve.loc[common, "return"].dropna()

    j_tlw = terminal_log_wealth(j_ret) if len(j_ret) else 0.0
    b_tlw = terminal_log_wealth(b_ret) if len(b_ret) else 0.0
    j_dd = max_drawdown_and_duration(j_ret) if len(j_ret) else {"max_drawdown": 0.0, "duration_months": 0}
    b_dd = max_drawdown_and_duration(b_ret) if len(b_ret) else {"max_drawdown": 0.0, "duration_months": 0}

    wealth_delta = float(j_tlw - b_tlw)
    dd_delta = float(j_dd["max_drawdown"] - b_dd["max_drawdown"])

    return {
        # the numbers
        "wealth_delta": wealth_delta,
        "dd_delta": dd_delta,
        "joint_terminal_log_wealth": float(j_tlw),
        "baseline_terminal_log_wealth": float(b_tlw),
        "joint_max_drawdown": float(j_dd["max_drawdown"]),
        "baseline_max_drawdown": float(b_dd["max_drawdown"]),
        "joint_duration_months": int(j_dd["duration_months"]),
        "baseline_duration_months": int(b_dd["duration_months"]),
        # ...and the window they were measured on, in the SAME mapping
        "n_steps": int(len(common)),
        "first_date": (common.min() if len(common) else None),
        "last_date": (common.max() if len(common) else None),
        "n_steps_joint": int(len(j_index)),
        "n_steps_baseline": int(len(b_index)),
        "indexes_identical": identical,
        # 07-BANDS.md §8 verdicts: universal governs, domain is advisory
        "wealth_delta_universal_ok": bool(abs(wealth_delta) < WEALTH_DELTA_UNIVERSAL),
        "wealth_delta_domain_note": bool(abs(wealth_delta) >= WEALTH_DELTA_DOMAIN),
        "dd_delta_universal_ok": bool(DD_DELTA_UNIVERSAL[0] <= dd_delta <= DD_DELTA_UNIVERSAL[1]),
        "dd_delta_domain_note": bool(abs(dd_delta) >= DD_DELTA_DOMAIN),
    }
