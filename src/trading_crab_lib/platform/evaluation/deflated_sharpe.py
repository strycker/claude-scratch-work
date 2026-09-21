"""Deflated Sharpe ratio (DSR) — corrects for selection bias and non-Normality.

Bailey, D.H. and López de Prado, M. (2014), "The Deflated Sharpe Ratio: Correcting
for Selection Bias, Backtest Overfitting and Non-Normality," Journal of Portfolio
Management (SSRN 2460551). Design §8.4/§22 name DSR as the headline-performance
correction for the whole-registry trial count (D-16). The estimator choice this
module implements against — including the degenerate-case fallback and its named
limitation — is recorded in
``.planning/phases/07-regime-representation/07-DSR-ESTIMATOR-NOTE.md``; that note
is the specification, not this docstring.

- ``expected_max_sharpe(n_trials, sharpe_variance)``: the expected maximum Sharpe
  ratio of ``n_trials`` skill-less (SR=0) trials under Extreme Value Theory (paper
  Eq. 5/6, Appendix 1). Zero for one or fewer trials (no selection occurred) or
  zero variance (a degenerate, single-point null has no dispersion to select from).
- ``deflated_sharpe_ratio(observed_sharpe, n_trials, sharpe_variance, skew,
  kurtosis, n_obs)``: the Probabilistic Sharpe Ratio of ``observed_sharpe``
  evaluated against ``expected_max_sharpe``'s threshold, correcting for the
  trial's own skew/kurtosis and track length. Raises ``ValueError`` on a
  non-positive non-normality denominator rather than returning a silent NaN.
- ``registry_sharpe_variance(path=None)``: reads the trial registry
  (``platform.honesty.registry``) and returns the sample variance of its
  Sharpe-bearing trials, per the estimator note's chosen approach; falls back to
  ``DEGENERATE_SHARPE_VARIANCE`` when fewer than two usable observations exist.
- ``format_dsr_verdict(dsr)``: a one-line, unsoftened verdict string. Per
  ``07-VALIDATION.md``, there is no pass/fail target — a DSR at or below 0.5 is
  reported plainly as not clearing the multiple-testing hurdle.

Usage::

    from trading_crab_lib.platform.evaluation.deflated_sharpe import (
        deflated_sharpe_ratio, format_dsr_verdict, registry_sharpe_variance,
    )
    from trading_crab_lib.platform.honesty.registry import total_trial_count

    variance = registry_sharpe_variance()
    dsr = deflated_sharpe_ratio(
        observed_sharpe=0.9, n_trials=total_trial_count(), sharpe_variance=variance,
        skew=-0.2, kurtosis=4.1, n_obs=356,
    )
    print(format_dsr_verdict(dsr))
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

import numpy as np
from scipy.stats import norm

from trading_crab_lib.platform.honesty.registry import PROVENANCE_RECORD_TYPE, read_trials

log = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────────

#: Euler-Mascheroni constant (paper Appendix 1, Eq. 5/6).
_EULER_MASCHERONI: Final[float] = 0.5772156649

#: Degenerate-case sharpe_variance fallback (07-DSR-ESTIMATOR-NOTE.md §4). Deliberately
#: NOT 0.0: expected_max_sharpe(n, 0.0) == 0.0 for every n, which would silently disable
#: the entire multiple-testing correction — the systematic under-penalization T-07-05
#: names as this project's closest analog to a security defect. 1.0 keeps the
#: correction mechanically active while the registry lacks real Sharpe-bearing history;
#: it is a documented assumption, never a measured quantity.
DEGENERATE_SHARPE_VARIANCE: Final[float] = 1.0

#: Minimum number of usable Sharpe observations before the registry population's own
#: sample variance is trusted over the degenerate-case fallback (07-DSR-ESTIMATOR-NOTE.md §3/4).
# RAISED 2 -> 20 on 2026-09-21 (Glenn's decision). At 2, any two sharpe-bearing
# rows switched this estimator off its conservative placeholder and onto a
# computed value. Measured with plan 07-11's two rows (Sharpe 0.917073 and
# 0.914903): sample variance 2.354450e-06, and expected_max_sharpe(42, .) falls
# from 2.208694 to 0.003389 — a 99.85% collapse of the multiple-testing hurdle,
# after which essentially any strategy clears DSR. A variance estimate from
# n=2 is unusable regardless of which two rows they are.
_MIN_USABLE_SHARPE_OBSERVATIONS: Final[int] = 20

# Rows whose config carries ``independent_trial: False`` are excluded from the
# variance entirely. In Bailey-Lopez de Prado, sharpe_variance is the dispersion
# of Sharpe ratios ACROSS INDEPENDENTLY-TRIED CONFIGURATIONS — it estimates how
# good the best of N trials looks by luck. Two arms of ONE ablation (a baseline
# and its single-parameter variant) are deliberately near-identical, so counting
# them answers "how different are the two arms of one comparison" (~0 by
# construction) rather than "how much do different strategies vary". That is a
# category error, not a tuning problem: raising the minimum alone would only
# defer it until enough ablation rows accumulated.
_INDEPENDENT_TRIAL_KEY: Final[str] = "independent_trial"

#: format_dsr_verdict's reporting threshold (07-VALIDATION.md: no target, but a value at
#: or below this must be reported plainly as not clearing the hurdle).
_VERDICT_HURDLE: Final[float] = 0.5


# ── expected_max_sharpe ──────────────────────────────────────────────────────────


def expected_max_sharpe(n_trials: int, sharpe_variance: float) -> float:
    """Expected maximum Sharpe ratio of ``n_trials`` skill-less (SR=0) trials.

    Bailey & López de Prado (2014) Eq. 5/6 (Appendix 1), reproduced verbatim from the
    paper's own Python snippet (``07-DSR-ESTIMATOR-NOTE.md`` §1): under the null that
    all trials share SR=0 and a common ``sharpe_variance`` (the dispersion of SR
    estimates within the search that produced them), the expected maximum grows with
    the number of independent trials attempted — this is the reason a large search
    "discovers" impressive-looking Sharpe ratios even with zero true skill.

    Args:
        n_trials: number of independent trials searched over. ``<= 1`` returns
            ``0.0`` — with one trial (or none) there is no selection to correct for.
        sharpe_variance: variance of the SR estimates across the trial population.
            ``<= 0.0`` returns ``0.0`` — a degenerate, single-point null distribution
            has no dispersion to select a maximum from.

    Returns:
        float: the expected maximum Sharpe ratio under the null, in the same units
        as ``sharpe_variance``'s square root (i.e. Sharpe-ratio units).
    """
    if n_trials <= 1 or sharpe_variance <= 0.0:
        return 0.0

    max_z = (1 - _EULER_MASCHERONI) * norm.ppf(1 - 1.0 / n_trials) + _EULER_MASCHERONI * norm.ppf(
        1 - 1.0 / (n_trials * np.e)
    )
    return float(np.sqrt(sharpe_variance) * max_z)


# ── deflated_sharpe_ratio ────────────────────────────────────────────────────────


def deflated_sharpe_ratio(
    observed_sharpe: float,
    n_trials: int,
    sharpe_variance: float,
    skew: float,
    kurtosis: float,
    n_obs: int,
) -> float:
    """Probabilistic Sharpe Ratio of ``observed_sharpe`` against the expected-maximum-
    under-the-null threshold from ``n_trials`` skill-less trials.

    DSR is a PSR (Bailey & López de Prado 2012) whose rejection threshold is
    :func:`expected_max_sharpe` rather than a user-chosen constant — this is what
    makes it a *deflated* Sharpe ratio: the more trials searched, or the more
    dispersed those trials' outcomes, the higher the bar ``observed_sharpe`` must
    clear. The non-normality denominator (Mertens 2002's delta-method variance
    approximation, the standard PSR/DSR convention — ``kurtosis`` is the RAW,
    non-excess kurtosis, i.e. 3.0 for a Normal distribution) corrects for skew and
    kurtosis in the SELECTED trial's own return distribution — separate from
    ``sharpe_variance``, which describes the dispersion ACROSS trials.

    Args:
        observed_sharpe: the selected trial's own estimated Sharpe ratio.
        n_trials: total number of trials searched over (D-16's whole-registry
            count — see ``platform.honesty.registry.total_trial_count``).
        sharpe_variance: variance of SR estimates across the trial population (see
            :func:`registry_sharpe_variance`).
        skew: third standardized moment of the selected trial's own returns.
        kurtosis: fourth standardized moment (RAW, not excess — 3.0 for Normal) of
            the selected trial's own returns.
        n_obs: number of return observations backing ``observed_sharpe``.

    Returns:
        float: DSR in the open unit interval ``(0, 1)`` — a ``norm.cdf`` output, not
        a bounded-by-construction heuristic. There is no pass/fail target
        (``07-VALIDATION.md``); see :func:`format_dsr_verdict` for plain reporting.

    Raises:
        ValueError: if the non-normality denominator (before the square root) is
            non-positive — an invalid combination of ``skew``/``kurtosis``/
            ``observed_sharpe`` that would otherwise silently produce a NaN DSR
            reported as a number (the exact evidence-shape failure this project's
            honesty framework exists to prevent).
    """
    sr0 = expected_max_sharpe(n_trials, sharpe_variance)

    denom_sq = 1.0 - skew * observed_sharpe + ((kurtosis - 1.0) / 4.0) * observed_sharpe**2
    if denom_sq <= 0.0:
        raise ValueError(
            f"non-normality denominator is non-positive ({denom_sq!r}) for "
            f"skew={skew!r}, kurtosis={kurtosis!r}, observed_sharpe={observed_sharpe!r} "
            "— this combination of moments cannot produce a valid PSR/DSR "
            "denominator; a NaN DSR reported as a number is not acceptable."
        )
    denom = np.sqrt(denom_sq)

    z = (observed_sharpe - sr0) * np.sqrt(n_obs - 1) / denom
    return float(norm.cdf(z))


# ── registry_sharpe_variance ─────────────────────────────────────────────────────


def registry_sharpe_variance(path: Path | str | None = None) -> float:
    """Sample variance of the trial registry's own Sharpe-bearing trials.

    Per ``07-DSR-ESTIMATOR-NOTE.md`` §3: the chosen ``sharpe_variance`` estimator is
    the observed population of this project's own trial history, read live via
    :func:`trading_crab_lib.platform.honesty.registry.read_trials` (excluding
    provenance-header rows, discriminated the same way
    :func:`trading_crab_lib.platform.honesty.registry.total_trial_count` does — the
    constant is imported, not re-declared). Each row's ``metrics["sharpe"]``, when
    present, is one usable observation.

    Applies the note's §4 degenerate-case policy when fewer than
    :data:`_MIN_USABLE_SHARPE_OBSERVATIONS` usable observations exist: returns
    :data:`DEGENERATE_SHARPE_VARIANCE` (``1.0``, never ``0.0`` — see that constant's
    docstring) and logs at WARNING exactly how many usable observations were found,
    so the fallback is never mistaken for a computed value. Never raises.

    Args:
        path: registry path, or ``None`` for the default live ledger.

    Returns:
        float: sample variance (``ddof=1``) of usable Sharpe observations, or
        :data:`DEGENERATE_SHARPE_VARIANCE` if fewer than two exist.
    """
    df = read_trials(path)
    sharpe_values: list[float] = []

    if not df.empty and "config" in df.columns and "metrics" in df.columns:
        is_header = df["config"].apply(
            lambda cfg: isinstance(cfg, dict) and cfg.get("record_type") == PROVENANCE_RECORD_TYPE
        )
        n_excluded = 0
        for cfg, metrics in zip(df.loc[~is_header, "config"], df.loc[~is_header, "metrics"], strict=False):
            if not (isinstance(metrics, dict) and "sharpe" in metrics):
                continue
            if isinstance(cfg, dict) and cfg.get(_INDEPENDENT_TRIAL_KEY) is False:
                n_excluded += 1
                continue
            try:
                sharpe_values.append(float(metrics["sharpe"]))
            except (TypeError, ValueError):
                log.warning("registry_sharpe_variance: unparseable sharpe metric %r skipped", metrics.get("sharpe"))
        if n_excluded:
            log.info(
                "registry_sharpe_variance: excluded %d sharpe-bearing row(s) marked "
                "%s=False — arms of one ablation are not independent trials and must "
                "not enter the across-trials variance.",
                n_excluded, _INDEPENDENT_TRIAL_KEY,
            )

    if len(sharpe_values) < _MIN_USABLE_SHARPE_OBSERVATIONS:
        log.warning(
            "registry_sharpe_variance: only %d usable Sharpe observation(s) found "
            "(need >= %d) — falling back to the degenerate-case placeholder variance "
            "%.1f per 07-DSR-ESTIMATOR-NOTE.md §4. This is an assumption, not a "
            "measured quantity.",
            len(sharpe_values),
            _MIN_USABLE_SHARPE_OBSERVATIONS,
            DEGENERATE_SHARPE_VARIANCE,
        )
        return DEGENERATE_SHARPE_VARIANCE

    return float(np.var(sharpe_values, ddof=1))


# ── format_dsr_verdict ───────────────────────────────────────────────────────────


def format_dsr_verdict(dsr: float) -> str:
    """One-line, unsoftened verdict string for a computed DSR.

    Per ``07-VALIDATION.md``: there is no pass/fail target for DSR. A value at or
    below 0.5 is reported PLAINLY as not clearing the multiple-testing hurdle — this
    function must never soften that into a near-miss ("close to significant",
    "borderline", "nearly clears") regardless of how close to 0.5 the value is.

    Args:
        dsr: the deflated Sharpe ratio, in ``(0, 1)``.

    Returns:
        str: a single-line verdict. Contains the phrase "does not clear the
        multiple-testing hurdle" at or below 0.5; a plainly different statement
        above it.
    """
    if dsr <= _VERDICT_HURDLE:
        return (
            f"Deflated Sharpe ratio {dsr:.4f} does not clear the multiple-testing "
            "hurdle — statistically indistinguishable from a skill-less discovery "
            "given the number of trials searched."
        )
    return (
        f"Deflated Sharpe ratio {dsr:.4f} clears the multiple-testing hurdle — "
        "statistically distinguishable from a skill-less discovery given the "
        "number of trials searched. No pass/fail target is set (07-VALIDATION.md); "
        "this reports, it does not gate."
    )
