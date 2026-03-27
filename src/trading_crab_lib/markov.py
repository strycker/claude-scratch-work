"""
Markov regime-switching model — 2-state diagnostic for recession/expansion.

Uses `statsmodels.tsa.regime_switching.MarkovRegression` to fit a switching-mean
model on GDP growth (or other univariate macro series). This produces a principled
2-state (recession/expansion) classification that can be compared against the
5-regime KMeans labels as a sanity check.

This is NOT a replacement for KMeans clustering. It serves two purposes:
1. **Diagnostic**: do the KMeans regimes align with the fundamental recession/expansion
   cycle? If regime X is "always recession" according to the Markov model, that's
   a useful label anchor.
2. **Feature generator**: the smoothed recession probability P(recession|data) can be
   added as an input feature for supervised models.

Usage
------
    from trading_crab_lib.markov import fit_markov_switching, markov_labels, markov_probabilities

    result = fit_markov_switching(gdp_growth_series, k_regimes=2)
    labels = markov_labels(result)
    probs  = markov_probabilities(result)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_STATSMODELS_AVAILABLE = False
try:
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
    _STATSMODELS_AVAILABLE = True
except ImportError:
    pass


def _require_statsmodels() -> None:
    """Raise ImportError with install instructions if statsmodels is missing."""
    if not _STATSMODELS_AVAILABLE:
        raise ImportError(
            "statsmodels is required for Markov regime-switching models. "
            "Install it with: pip install statsmodels"
        )


def fit_markov_switching(
    series: pd.Series,
    k_regimes: int = 2,
    order: int = 0,
    switching_variance: bool = False,
) -> dict:
    """
    Fit a Markov regime-switching model on a univariate time series.

    The default is a 2-regime switching-mean model with no AR terms, which
    captures the simplest recession/expansion dichotomy in GDP growth.

    Args:
        series             — Univariate time series (e.g., gdp_growth). Must have
                             a DatetimeIndex or PeriodIndex. NaN rows are dropped.
        k_regimes          — Number of regimes (default 2: recession/expansion)
        order              — AR order (default 0: switching-mean only, no AR terms)
        switching_variance — If True, variance also switches between regimes.
                             Default False (only the mean switches).

    Returns:
        dict with keys:
            "model"       — the fitted MarkovRegression results object
            "series_name" — name of the input series
            "k_regimes"   — number of regimes
            "regime_means"— dict mapping regime_id → estimated mean
            "transition_matrix" — DataFrame of transition probabilities
    """
    _require_statsmodels()

    clean = series.dropna()
    if len(clean) < 20:
        raise ValueError(
            f"Series has only {len(clean)} non-NaN observations — need ≥20 "
            "for Markov regime-switching estimation"
        )

    mod = MarkovRegression(
        clean,
        k_regimes=k_regimes,
        order=order,
        switching_variance=switching_variance,
    )
    res = mod.fit(disp=False)

    # Extract regime means
    regime_means = {}
    for i in range(k_regimes):
        param_name = f"const[{i}]"
        if param_name in res.params.index:
            regime_means[i] = float(res.params[param_name])
        else:
            regime_means[i] = float("nan")

    # Transition matrix — squeeze extra trailing dimension from statsmodels
    trans = np.squeeze(res.regime_transition)
    labels = [f"regime_{i}" for i in range(k_regimes)]
    trans_df = pd.DataFrame(trans, index=labels, columns=labels)

    log.info(
        "Markov switching fit: k=%d, series=%s, regime_means=%s",
        k_regimes, series.name, regime_means,
    )

    return {
        "model": res,
        "series_name": series.name or "unnamed",
        "k_regimes": k_regimes,
        "regime_means": regime_means,
        "transition_matrix": trans_df,
    }


def markov_labels(result: dict) -> pd.Series:
    """
    Extract hard regime labels (argmax of smoothed probabilities).

    Labels are canonicalized so that regime 0 has the LOWER mean value.
    For GDP growth this means regime 0 = recession (lower mean growth).

    Args:
        result — dict returned by fit_markov_switching

    Returns:
        Series indexed like the input, name="markov_regime".
    """
    res = result["model"]
    means = result["regime_means"]

    # Smoothed probabilities: (n_obs, k_regimes)
    probs = res.smoothed_marginal_probabilities

    raw_labels = probs.values.argmax(axis=1)
    raw_series = pd.Series(raw_labels, index=probs.index, name="markov_regime")

    # Canonicalize: regime 0 = lowest mean
    sorted_regimes = sorted(means.keys(), key=lambda r: means[r])
    label_map = {old: new for new, old in enumerate(sorted_regimes)}
    return raw_series.map(label_map).rename("markov_regime")


def markov_probabilities(result: dict) -> pd.DataFrame:
    """
    Extract smoothed marginal regime probabilities.

    Returns:
        DataFrame indexed like the input, columns = markov_prob_0, markov_prob_1, …
        Each row sums to 1.
    """
    res = result["model"]
    probs = res.smoothed_marginal_probabilities

    # Canonicalize column order to match label ordering (regime 0 = lowest mean)
    means = result["regime_means"]
    sorted_regimes = sorted(means.keys(), key=lambda r: means[r])

    out = pd.DataFrame(index=probs.index)
    for new_idx, old_idx in enumerate(sorted_regimes):
        col_name = f"markov_prob_{new_idx}"
        if old_idx < probs.shape[1]:
            out[col_name] = probs.iloc[:, old_idx].values
        else:
            out[col_name] = np.nan

    return out


def compare_markov_kmeans(
    markov_result: dict,
    kmeans_labels: pd.Series,
) -> pd.DataFrame:
    """
    Cross-tabulate Markov recession/expansion against KMeans regimes.

    Useful for answering: "Which KMeans regimes correspond to recessions?"

    Args:
        markov_result — dict returned by fit_markov_switching
        kmeans_labels — Series of KMeans cluster labels (e.g., balanced_cluster)

    Returns:
        DataFrame cross-tabulation (Markov regime × KMeans regime), values = counts.
    """
    m_labels = markov_labels(markov_result)

    common = m_labels.index.intersection(kmeans_labels.index)
    if len(common) == 0:
        raise ValueError("No overlapping index between Markov and KMeans labels")

    return pd.crosstab(
        m_labels.loc[common].rename("markov_regime"),
        kmeans_labels.loc[common].rename("kmeans_regime"),
    )
