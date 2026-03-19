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

Scaler consistency
-------------------
fit_gmm() returns the fitted StandardScaler alongside the models dict.
Always pass this scaler to gmm_labels() and gmm_probabilities() to ensure
the data is transformed identically to how it was during training.
Omitting the scaler causes a new scaler to be fit on whatever data is passed,
which will produce wrong assignments if the data distribution differs at all.

Usage
------
    from trading_crab_lib.gmm import fit_gmm, select_gmm_k, gmm_labels, gmm_probabilities

    bic_df, models, scaler = fit_gmm(pca_df, k_range=range(2, 10))
    best_k, best_cov = select_gmm_k(bic_df)
    labels = gmm_labels(pca_df, models[(best_k, best_cov)], scaler=scaler)
    probs  = gmm_probabilities(pca_df, models[(best_k, best_cov)], scaler=scaler)
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
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
) -> tuple[pd.DataFrame, dict[tuple[int, str], GaussianMixture], StandardScaler]:
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
        scaler  — the fitted StandardScaler used to transform pca_df.
                  Pass this to gmm_labels() and gmm_probabilities() to guarantee
                  consistent scaling.
    """
    if pca_df.empty:
        raise ValueError("pca_df is empty — cannot fit GMM on zero rows")
    if k_range is None:
        k_range = range(2, 10)

    scaler = StandardScaler()
    X = scaler.fit_transform(pca_df.values)
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
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", ConvergenceWarning)
                    gm.fit(X)

                if any(issubclass(w.category, ConvergenceWarning) for w in caught):
                    log.warning(
                        "GMM k=%d cov=%s did not converge after %d iterations "
                        "— BIC may be unreliable; increase max_iter or n_init",
                        k, cov_type, max_iter,
                    )

                bic = float(gm.bic(X))
                aic = float(gm.aic(X))
                ll  = float(gm.score(X))  # mean log-likelihood per sample
                rows.append({"k": k, "covariance_type": cov_type, "bic": bic, "aic": aic, "log_likelihood": ll})
                models[(k, cov_type)] = gm
                log.info("GMM k=%d cov=%s  BIC=%.1f  AIC=%.1f  LL=%.4f", k, cov_type, bic, aic, ll)
            except Exception as exc:
                log.warning("GMM k=%d cov=%s failed: %s", k, cov_type, exc)

    return pd.DataFrame(rows), models, scaler


def select_gmm_k(bic_df: pd.DataFrame) -> tuple[int, str]:
    """
    Return (best_k, best_covariance_type) minimizing BIC.

    BIC balances log-likelihood against model complexity (penalises more parameters),
    making it suitable for small N where AIC would overfit.

    Raises:
        ValueError if bic_df is empty or all BIC values are NaN/missing.
    """
    if bic_df.empty:
        raise ValueError("bic_df is empty — no GMM fits succeeded")
    if bic_df["bic"].isna().all():
        raise ValueError("bic_df has no valid BIC values — all GMM fits failed or produced NaN")
    best_row = bic_df.loc[bic_df["bic"].idxmin()]
    best_k = int(best_row["k"])
    best_cov = str(best_row["covariance_type"])
    log.info("Best GMM: k=%d, cov=%s, BIC=%.1f", best_k, best_cov, float(best_row["bic"]))
    return best_k, best_cov


def gmm_labels(
    pca_df: pd.DataFrame,
    model: GaussianMixture,
    scaler: StandardScaler | None = None,
) -> pd.Series:
    """
    Return hard cluster labels (argmax of component responsibilities).

    Labels are sorted so that cluster 0 has the smallest mean PC1 value
    (consistent with the KMeans canonicalization in clustering.py).

    Args:
        pca_df  — PCA-reduced feature matrix (same as used for fit_gmm)
        model   — fitted GaussianMixture (from fit_gmm models dict)
        scaler  — fitted StandardScaler returned by fit_gmm.  If None, a new
                  scaler is fitted on pca_df which may produce inconsistent
                  label assignments if the distribution differs from training.

    Returns:
        Series indexed by quarter, name="gmm_cluster".
    """
    if scaler is None:
        log.warning(
            "gmm_labels called without a fitted scaler — fitting a new scaler on "
            "pca_df.  For correct assignments pass the scaler returned by fit_gmm()."
        )
        scaler = StandardScaler().fit(pca_df.values)
    X = scaler.transform(pca_df.values)
    raw_labels = pd.Series(model.predict(X), index=pca_df.index, name="gmm_cluster")

    # Canonicalize: cluster 0 = smallest mean PC1 (consistent with KMeans canonicalization)
    pc1 = pca_df.iloc[:, 0]
    mean_pc1 = pc1.groupby(raw_labels).mean()
    label_map = {old: new for new, old in enumerate(mean_pc1.sort_values().index)}
    return raw_labels.map(label_map).rename("gmm_cluster")


def gmm_probabilities(
    pca_df: pd.DataFrame,
    model: GaussianMixture,
    scaler: StandardScaler | None = None,
) -> pd.DataFrame:
    """
    Return soft cluster probability matrix (responsibilities).

    Args:
        pca_df  — PCA-reduced feature matrix (same as used for fit_gmm)
        model   — fitted GaussianMixture (from fit_gmm models dict)
        scaler  — fitted StandardScaler returned by fit_gmm.  If None, a new
                  scaler is fitted on pca_df (see gmm_labels warning).

    Returns:
        DataFrame indexed by quarter, columns = gmm_prob_0 … gmm_prob_{k-1},
        where each row sums to 1.
    """
    if scaler is None:
        log.warning(
            "gmm_probabilities called without a fitted scaler — fitting a new scaler on "
            "pca_df.  For correct probabilities pass the scaler returned by fit_gmm()."
        )
        scaler = StandardScaler().fit(pca_df.values)
    X = scaler.transform(pca_df.values)
    probs = model.predict_proba(X)
    k = probs.shape[1]
    cols = [f"gmm_prob_{i}" for i in range(k)]
    return pd.DataFrame(probs, index=pca_df.index, columns=cols)
