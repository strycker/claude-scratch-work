"""
Gaussian Mixture Model (GMM) clustering — alternative to KMeans.

Why GMM instead of KMeans?
- Soft assignments: each quarter gets a probability vector over regimes rather than
  a hard label.  Useful as richer input to supervised classifiers.
- Elliptical clusters: KMeans assumes spherical equal-variance clusters (Voronoi);
  GMM models full covariance, handling elongated or correlated regime shapes.
- BIC/AIC for k selection: GMM provides a principled likelihood-based criterion,
  removing the need to eyeball silhouette plots.

Covariance types (sklearn convention)
--------------------------------------
- "diag"   — diagonal covariance per component (recommended for N≈300, D≈5)
- "tied"   — all components share one covariance matrix
- "full"   — each component has its own full covariance (overfit risk at small N)
- "spherical" — each component has a scalar variance (most restrictive)

Usage
------
    from market_regime.gmm import fit_gmm, select_gmm_k, gmm_labels, gmm_probabilities

    bic_df, models = fit_gmm(pca_df, k_range=range(2, 10))
    best_k, best_cov = select_gmm_k(bic_df)
    labels = gmm_labels(pca_df, models[(best_k, best_cov)])
    probs  = gmm_probabilities(pca_df, models[(best_k, best_cov)])
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)

_DEFAULT_COV_TYPES = ("diag", "tied", "full")


def fit_gmm(
    pca_df: pd.DataFrame,
    k_range: range | None = None,
    covariance_types: tuple[str, ...] = _DEFAULT_COV_TYPES,
    n_init: int = 10,
    max_iter: int = 300,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[tuple[int, str], GaussianMixture]]:
    """
    Fit GaussianMixture for all (k, covariance_type) combinations.

    Args:
        pca_df           — PCA-reduced feature matrix (output of reduce_pca)
        k_range          — number of components to sweep (default range(2, 10))
        covariance_types — sklearn GMM covariance types to try
        n_init           — restarts per (k, cov_type) pair
        max_iter         — EM iteration limit
        random_state

    Returns:
        bic_df  — DataFrame with columns: k, covariance_type, bic, aic, log_likelihood
        models  — dict mapping (k, covariance_type) → fitted GaussianMixture
    """
    if k_range is None:
        k_range = range(2, 10)

    X = StandardScaler().fit_transform(pca_df.values)
    rows: list[dict] = []
    models: dict[tuple[int, str], GaussianMixture] = {}

    for cov_type in covariance_types:
        for k in k_range:
            try:
                gm = GaussianMixture(
                    n_components=k,
                    covariance_type=cov_type,
                    n_init=n_init,
                    max_iter=max_iter,
                    random_state=random_state,
                )
                gm.fit(X)
                bic = float(gm.bic(X))
                aic = float(gm.aic(X))
                ll  = float(gm.score(X))  # mean log-likelihood per sample
                rows.append({"k": k, "covariance_type": cov_type, "bic": bic, "aic": aic, "log_likelihood": ll})
                models[(k, cov_type)] = gm
                log.info("GMM k=%d cov=%s  BIC=%.1f  AIC=%.1f  LL=%.4f", k, cov_type, bic, aic, ll)
            except Exception as exc:
                log.warning("GMM k=%d cov=%s failed: %s", k, cov_type, exc)

    return pd.DataFrame(rows), models


def select_gmm_k(bic_df: pd.DataFrame) -> tuple[int, str]:
    """
    Return (best_k, best_covariance_type) minimizing BIC.

    BIC balances log-likelihood against model complexity (penalises more parameters),
    making it suitable for small N where AIC would overfit.
    """
    if bic_df.empty:
        raise ValueError("bic_df is empty — no GMM fits succeeded")
    best_row = bic_df.loc[bic_df["bic"].idxmin()]
    best_k = int(best_row["k"])
    best_cov = str(best_row["covariance_type"])
    log.info("Best GMM: k=%d, cov=%s, BIC=%.1f", best_k, best_cov, float(best_row["bic"]))
    return best_k, best_cov


def gmm_labels(pca_df: pd.DataFrame, model: GaussianMixture) -> pd.Series:
    """
    Return hard cluster labels (argmax of component responsibilities).

    Labels are sorted so that cluster 0 has the smallest mean PC1 value
    (consistent with the KMeans canonicalization in clustering.py).
    """
    X = StandardScaler().fit_transform(pca_df.values)
    raw_labels = pd.Series(model.predict(X), index=pca_df.index, name="gmm_cluster")

    # Canonicalize: sort by mean PC1
    pc1 = pca_df.iloc[:, 0]
    mean_pc1 = raw_labels.groupby(raw_labels).apply(lambda grp: pc1.loc[grp.index].mean())
    label_map = {old: new for new, old in enumerate(mean_pc1.sort_values().index)}
    return raw_labels.map(label_map).rename("gmm_cluster")


def gmm_probabilities(pca_df: pd.DataFrame, model: GaussianMixture) -> pd.DataFrame:
    """
    Return soft cluster probability matrix (responsibilities).

    Returns:
        DataFrame indexed by quarter, columns = gmm_prob_0 … gmm_prob_{k-1},
        where each row sums to 1.
    """
    X = StandardScaler().fit_transform(pca_df.values)
    probs = model.predict_proba(X)
    k = probs.shape[1]
    cols = [f"gmm_prob_{i}" for i in range(k)]
    return pd.DataFrame(probs, index=pca_df.index, columns=cols)
