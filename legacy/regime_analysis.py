"""
Regime Analysis: interpretation, naming, and transition probabilities.

After clustering, this module gives each regime a human-readable profile
and computes forward-looking transition probabilities.
"""

import logging

import numpy as np
import pandas as pd

from trading_crab.config import REGIME_PROFILE_FEATURES

log = logging.getLogger(__name__)


# ===================================================================
# Regime profiling
# ===================================================================

def profile_regimes(
    quarterly_df: pd.DataFrame,
    cluster_labels: np.ndarray,
    features: list[str] | None = None,
) -> pd.DataFrame:
    """
    Compute mean and std of key features within each cluster.

    Args:
        quarterly_df: prepared DataFrame (must contain the profile features)
        cluster_labels: integer array of cluster assignments, aligned with
            quarterly_df after dropna
        features: columns to profile (defaults to REGIME_PROFILE_FEATURES)

    Returns:
        DataFrame with multi-level columns: (feature, 'mean'|'std')
    """
    features = features or REGIME_PROFILE_FEATURES
    log.info("Profiling regimes across %d features...", len(features))

    # Align: use only rows that survived the dropna in clustering
    available = [f for f in features if f in quarterly_df.columns]
    if len(available) < len(features):
        missing = set(features) - set(available)
        log.warning("  Missing profile features (skipped): %s", missing)

    clean = quarterly_df[available].dropna()
    # Trim to match cluster_labels length (clustering may have dropped rows)
    clean = clean.iloc[:len(cluster_labels)].copy()
    clean["regime"] = cluster_labels

    profile = clean.groupby("regime")[available].agg(["mean", "std"])
    log.info("  Profiled %d regimes", profile.index.nunique())
    return profile


def name_regimes(profile: pd.DataFrame) -> dict[int, str]:
    """
    Assign human-readable names to regimes based on their statistical profile.

    Heuristic rules:
    - High inflation + low GDP growth → "Stagflation"
    - High GDP growth + rising S&P → "Growth Boom"
    - Wide credit spread + falling S&P → "Credit Crisis"
    - Low inflation + moderate growth → "Goldilocks"
    - Everything else → "Transition"
    """
    log.info("Naming regimes...")
    names = {}

    for regime in profile.index:
        row = profile.loc[regime]

        # Extract means (handle multi-level columns gracefully)
        def _get(feat):
            try:
                return row[(feat, "mean")]
            except KeyError:
                return np.nan

        gdp_g = _get("gdp_growth")
        infl = _get("us_infl")
        spread = _get("credit_spread")
        sp_d1 = _get("log_sp500_d1")

        # Classification heuristics based on relative thresholds
        if not np.isnan(infl) and infl > 0.04 and not np.isnan(gdp_g) and gdp_g < 0.02:
            names[regime] = "Stagflation"
        elif not np.isnan(gdp_g) and gdp_g > 0.03 and not np.isnan(sp_d1) and sp_d1 > 0:
            names[regime] = "Growth Boom"
        elif not np.isnan(spread) and spread > 0.015 and not np.isnan(sp_d1) and sp_d1 < 0:
            names[regime] = "Credit Crisis"
        elif not np.isnan(infl) and infl < 0.03 and not np.isnan(gdp_g) and 0.01 < gdp_g < 0.04:
            names[regime] = "Goldilocks"
        else:
            names[regime] = "Transition"

    # Deduplicate names by appending regime number if needed
    seen = {}
    for regime, name in names.items():
        if name in seen.values():
            names[regime] = f"{name} ({regime})"
        seen[regime] = names[regime]

    for regime, name in names.items():
        log.info("  Regime %d → %s", regime, name)

    return names


# ===================================================================
# Transition probabilities
# ===================================================================

def compute_transition_matrix(labels: np.ndarray) -> pd.DataFrame:
    """
    Compute the empirical regime transition matrix P(j at t+1 | i at t).

    Args:
        labels: 1-D array of regime labels in chronological order

    Returns:
        DataFrame where entry [i, j] = P(transition from regime i to regime j)
    """
    log.info("Computing regime transition matrix...")
    unique = sorted(set(labels))
    n_regimes = len(unique)
    counts = np.zeros((n_regimes, n_regimes), dtype=int)

    regime_to_idx = {r: i for i, r in enumerate(unique)}
    for t in range(len(labels) - 1):
        i = regime_to_idx[labels[t]]
        j = regime_to_idx[labels[t + 1]]
        counts[i, j] += 1

    # Normalize rows to probabilities
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # avoid division by zero
    probs = counts / row_sums

    result = pd.DataFrame(probs, index=unique, columns=unique)
    result.index.name = "from_regime"
    result.columns.name = "to_regime"
    log.info("  %d×%d transition matrix computed", n_regimes, n_regimes)
    return result


def compute_forward_probabilities(
    labels: np.ndarray, horizons: list[int] = None
) -> dict[int, pd.DataFrame]:
    """
    For each horizon N, compute P(regime_j within next N quarters | regime_i now).

    Unlike the transition matrix (which is one-step), this asks: "if we're
    in regime i now, what's the probability of seeing regime j at least
    once in the next N quarters?"

    Args:
        labels: chronological regime labels
        horizons: list of forward horizons in quarters (default: [1, 4, 8])

    Returns:
        dict mapping horizon → probability DataFrame
    """
    horizons = horizons or [1, 4, 8]
    log.info("Computing forward probabilities for horizons %s...", horizons)

    unique = sorted(set(labels))
    results = {}

    for h in horizons:
        counts = {r: {r2: 0 for r2 in unique} for r in unique}
        totals = {r: 0 for r in unique}

        for t in range(len(labels) - h):
            current = labels[t]
            future_window = labels[t + 1: t + 1 + h]
            totals[current] += 1
            for future_regime in unique:
                if future_regime in future_window:
                    counts[current][future_regime] += 1

        probs = pd.DataFrame(
            {to_r: {from_r: (counts[from_r][to_r] / totals[from_r]
                              if totals[from_r] > 0 else 0)
                     for from_r in unique}
             for to_r in unique}
        )
        probs.index.name = "from_regime"
        probs.columns.name = "to_regime"
        results[h] = probs
        log.info("  Horizon %d: computed", h)

    return results
