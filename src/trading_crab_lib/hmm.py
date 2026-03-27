"""
Hidden Markov Model (HMM) regime detection — temporal alternative to KMeans.

Why HMM instead of KMeans?
- Temporal structure: HMM models transition probabilities between regimes, so
  the assignment for quarter *t* depends on the assignment for quarter *t−1*.
  KMeans treats each quarter independently.
- Soft assignments: like GMM, HMM produces posterior probability vectors over
  regimes for each quarter (via the forward-backward algorithm).
- Transition matrix: HMM directly estimates how likely each regime→regime
  transition is per quarter, which is exactly what the prediction step tries
  to learn from KMeans labels.

Trade-offs vs KMeans
---------------------
- EM sensitivity: GaussianHMM uses EM which is sensitive to initialization,
  especially with N≈300 and k≥4. Multiple restarts (n_iter, n_init via
  best-of-N) mitigate this.
- No balanced-size guarantee: HMM may assign most quarters to one state.
  Compare with balanced KMeans to assess severity.
- Slower fitting: O(N·k²) per EM step vs O(N·k) for KMeans.

Usage
------
    from trading_crab_lib.hmm import fit_hmm, select_hmm_k, hmm_labels, hmm_probabilities

    scores_df, models, scaler = fit_hmm(pca_df, k_range=range(2, 8))
    best_k = select_hmm_k(scores_df)
    labels = hmm_labels(pca_df, models[best_k], scaler=scaler)
    probs  = hmm_probabilities(pca_df, models[best_k], scaler=scaler)
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)

_HMM_AVAILABLE = False
try:
    from hmmlearn.hmm import GaussianHMM
    _HMM_AVAILABLE = True
except ImportError:
    pass


def _require_hmmlearn() -> None:
    """Raise ImportError with install instructions if hmmlearn is missing."""
    if not _HMM_AVAILABLE:
        raise ImportError(
            "hmmlearn is required for HMM regime detection. "
            "Install it with: pip install hmmlearn"
        )


def fit_hmm(
    pca_df: pd.DataFrame,
    k_range: range | None = None,
    covariance_type: str = "diag",
    n_iter: int = 200,
    n_restarts: int = 10,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[int, "GaussianHMM"], StandardScaler]:
    """
    Fit GaussianHMM for each k in k_range, picking the best restart per k.

    Args:
        pca_df          — PCA-reduced feature matrix (output of reduce_pca)
        k_range         — number of hidden states to sweep (default range(2, 8))
        covariance_type — "diag", "full", "tied", or "spherical"
        n_iter          — max EM iterations per fit
        n_restarts      — number of random initializations per k (best-of-N)
        random_state    — base seed (each restart uses random_state + i)

    Returns:
        scores_df — DataFrame: k, log_likelihood, bic, aic, converged
        models    — dict mapping k → best-fitted GaussianHMM (highest LL)
        scaler    — fitted StandardScaler (pass to hmm_labels / hmm_probabilities)
    """
    _require_hmmlearn()

    if pca_df.empty:
        raise ValueError("pca_df is empty — cannot fit HMM on zero rows")
    if k_range is None:
        k_range = range(2, 8)

    scaler = StandardScaler()
    X = scaler.fit_transform(pca_df.values)
    n_samples, n_features = X.shape

    rows: list[dict] = []
    models: dict[int, GaussianHMM] = {}

    for k in k_range:
        best_ll = -np.inf
        best_model = None
        converged = False

        for restart in range(n_restarts):
            seed = random_state + restart
            try:
                hmm = GaussianHMM(
                    n_components=k,
                    covariance_type=covariance_type,
                    n_iter=n_iter,
                    random_state=seed,
                )
                with warnings.catch_warnings(record=True):
                    warnings.simplefilter("always")
                    hmm.fit(X)

                ll = float(hmm.score(X))
                if ll > best_ll:
                    best_ll = ll
                    best_model = hmm
                    converged = bool(hmm.monitor_.converged)

            except Exception as exc:
                log.debug("HMM k=%d restart=%d failed: %s", k, restart, exc)
                continue

        if best_model is not None:
            # BIC and AIC computation
            # Number of free parameters for GaussianHMM:
            # - Transition matrix: k*(k-1) (rows sum to 1)
            # - Start probabilities: k-1
            # - Means: k * n_features
            # - Covariances: depends on covariance_type
            n_params = k * (k - 1) + (k - 1) + k * n_features
            if covariance_type == "diag":
                n_params += k * n_features
            elif covariance_type == "full":
                n_params += k * n_features * (n_features + 1) // 2
            elif covariance_type == "tied":
                n_params += n_features * (n_features + 1) // 2
            elif covariance_type == "spherical":
                n_params += k

            bic = -2 * best_ll * n_samples + n_params * np.log(n_samples)
            aic = -2 * best_ll * n_samples + 2 * n_params

            rows.append({
                "k": k,
                "log_likelihood": best_ll,
                "bic": bic,
                "aic": aic,
                "converged": converged,
                "n_params": n_params,
            })
            models[k] = best_model
            log.info(
                "HMM k=%d  LL=%.4f  BIC=%.1f  AIC=%.1f  converged=%s",
                k, best_ll, bic, aic, converged,
            )
        else:
            log.warning("HMM k=%d — all %d restarts failed", k, n_restarts)

    return pd.DataFrame(rows), models, scaler


def select_hmm_k(scores_df: pd.DataFrame) -> int:
    """
    Select best k minimizing BIC.

    Raises:
        ValueError if scores_df is empty.
    """
    if scores_df.empty:
        raise ValueError("scores_df is empty — no HMM fits succeeded")
    best_row = scores_df.loc[scores_df["bic"].idxmin()]
    best_k = int(best_row["k"])
    log.info("Best HMM: k=%d, BIC=%.1f", best_k, float(best_row["bic"]))
    return best_k


def hmm_labels(
    pca_df: pd.DataFrame,
    model: "GaussianHMM",
    scaler: StandardScaler | None = None,
) -> pd.Series:
    """
    Return hard state assignments via Viterbi decoding.

    Labels are canonicalized so that state 0 has the smallest mean PC1 value
    (consistent with KMeans and GMM canonicalization).

    Args:
        pca_df — PCA-reduced feature matrix
        model  — fitted GaussianHMM
        scaler — fitted StandardScaler from fit_hmm

    Returns:
        Series indexed by quarter, name="hmm_state".
    """
    _require_hmmlearn()
    if scaler is None:
        log.warning("hmm_labels called without scaler — fitting new scaler on pca_df")
        scaler = StandardScaler().fit(pca_df.values)
    X = scaler.transform(pca_df.values)
    raw_labels = pd.Series(model.predict(X), index=pca_df.index, name="hmm_state")

    # Canonicalize: state 0 = smallest mean PC1
    pc1 = pca_df.iloc[:, 0]
    mean_pc1 = pc1.groupby(raw_labels).mean()
    label_map = {old: new for new, old in enumerate(mean_pc1.sort_values().index)}
    return raw_labels.map(label_map).rename("hmm_state")


def hmm_probabilities(
    pca_df: pd.DataFrame,
    model: "GaussianHMM",
    scaler: StandardScaler | None = None,
) -> pd.DataFrame:
    """
    Return posterior state probabilities via forward-backward algorithm.

    Args:
        pca_df — PCA-reduced feature matrix
        model  — fitted GaussianHMM
        scaler — fitted StandardScaler from fit_hmm

    Returns:
        DataFrame indexed by quarter, columns = hmm_prob_0 … hmm_prob_{k-1},
        each row sums to 1.
    """
    _require_hmmlearn()
    if scaler is None:
        log.warning("hmm_probabilities called without scaler — fitting new scaler")
        scaler = StandardScaler().fit(pca_df.values)
    X = scaler.transform(pca_df.values)
    probs = model.predict_proba(X)
    k = probs.shape[1]
    cols = [f"hmm_prob_{i}" for i in range(k)]
    return pd.DataFrame(probs, index=pca_df.index, columns=cols)


def hmm_transition_matrix(model: "GaussianHMM") -> pd.DataFrame:
    """
    Extract the learned transition matrix as a labeled DataFrame.

    Returns:
        Square DataFrame where entry (i, j) is P(state_j | state_i).
        Rows sum to 1.
    """
    _require_hmmlearn()
    k = model.n_components
    labels = [f"state_{i}" for i in range(k)]
    return pd.DataFrame(model.transmat_, index=labels, columns=labels)
