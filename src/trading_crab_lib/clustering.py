"""
PCA dimensionality reduction, KMeans cluster evaluation, and clustering.

Core pipeline functions (called by pipelines/03_cluster.py):
  1. reduce_pca()          — StandardScale + PCA to N fixed components
  2. evaluate_kmeans()     — sweep k, score with silhouette/CH/DB, pick best k
  3. fit_clusters()        — standard KMeans + size-constrained KMeans

Exploration / investigation functions (used by notebooks/03_clustering.ipynb):
  4. optimize_n_components() — sweep PCA n to find optimal dimensionality
  5. compare_svd_pca()      — TruncatedSVD vs PCA component loadings comparison
  6. compute_gap_statistic() — gap statistic for k selection (Tibshirani 2001)
  7. find_knee_k()          — elbow detection on inertia curve
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)


def _load_constrained_kmeans():
    try:
        from k_means_constrained import KMeansConstrained
        return KMeansConstrained
    except ImportError:
        log.warning(
            "k-means-constrained not installed — balanced clustering unavailable. "
            "Run: pip install k-means-constrained"
        )
        return None


# ── 1. PCA ─────────────────────────────────────────────────────────────────

def reduce_pca(
    df: pd.DataFrame,
    n_components: int = 5,
    random_state: int = 42,
) -> tuple[pd.DataFrame, PCA, StandardScaler]:
    """
    StandardScale the features then reduce to exactly n_components PCA axes.

    Returns:
        pca_df   — DataFrame of PC columns (PC1…PCn), same index as df
        pca_obj  — fitted PCA (kept for scoring new data later)
        scaler   — fitted StandardScaler (kept for the same reason)
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df.values)

    pca = PCA(n_components=n_components, random_state=random_state)
    X_reduced = pca.fit_transform(X_scaled)

    ratios = np.round(pca.explained_variance_ratio_, 3)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    log.info(
        "\nRunning PCA... done.\n"
        "PCA: %d components explain %.1f%% of variance\n"
        "PCA explained variance ratios: %s\n",
        n_components,
        cumvar[-1] * 100,
        ratios,
    )

    col_names = [f"PC{i + 1}" for i in range(n_components)]
    pca_df = pd.DataFrame(X_reduced, index=df.index, columns=col_names)
    return pca_df, pca, scaler


# ── 2. K evaluation ────────────────────────────────────────────────────────

def evaluate_kmeans(
    X: np.ndarray,
    k_range: range,
    n_init: int = 50,
    random_state: int = 0,
) -> pd.DataFrame:
    """
    Run KMeans for each k in k_range and return a DataFrame of quality scores:
      inertia, silhouette, calinski, davies_bouldin.

    Args:
        X            — scaled feature matrix (output of StandardScaler)
        k_range      — range of k values to evaluate, e.g. range(2, 13)
        n_init       — KMeans restarts per k (higher = more stable)
        random_state

    Returns:
        DataFrame with one row per k, columns: k, inertia, silhouette,
        calinski, davies_bouldin.  Rows are in k order (not sorted).
    """
    results = []
    for k in k_range:
        model = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
        labels = model.fit_predict(X)
        results.append({
            "k":              k,
            "inertia":        model.inertia_,
            "silhouette":     silhouette_score(X, labels),
            "calinski":       calinski_harabasz_score(X, labels),
            "davies_bouldin": davies_bouldin_score(X, labels),
        })
        log.debug("k=%d  sil=%.4f  CH=%.1f  DB=%.4f",
                  k, results[-1]["silhouette"],
                  results[-1]["calinski"],
                  results[-1]["davies_bouldin"])

    scores = pd.DataFrame(results)

    sorted_scores = scores.sort_values("silhouette", ascending=False)
    table = sorted_scores.to_string(float_format=lambda x: f"{x:.6f}")
    best_k = int(scores.loc[scores["silhouette"].idxmax(), "k"])
    log.info(
        "\nEvaluating cluster counts... done.\n"
        "Silhouette scores:\n%s\n"
        "\nBest k by silhouette: %d  (score=%.4f)\n",
        table,
        best_k,
        scores["silhouette"].max(),
    )
    return scores


def pick_best_k(scores: pd.DataFrame, k_cap: int = 5) -> int:
    """Return the k with the highest silhouette score, capped at k_cap."""
    best = int(scores.loc[scores["silhouette"].idxmax(), "k"])
    return min(best, k_cap)


# ── 3. Clustering ──────────────────────────────────────────────────────────

def fit_clusters(
    pca_df: pd.DataFrame,
    best_k: int,
    balanced_k: int,
    random_state: int = 42,
    use_constrained: bool = True,
) -> pd.DataFrame:
    """
    Fit two clusterings on the PCA-reduced data:
      - "cluster"          — standard KMeans at best_k
      - "balanced_cluster" — size-constrained KMeans at balanced_k

    Args:
        pca_df           — output of reduce_pca()
        best_k           — k chosen by silhouette search (via pick_best_k)
        balanced_k       — k for equal-size clustering (from config)
        random_state
        use_constrained  — if False, fall back to plain KMeans for balanced_cluster
                           (use when k-means-constrained is not installed)

    Returns:
        pca_df with two new columns: cluster, balanced_cluster.
    """
    # Re-scale the PCA components before clustering
    X = StandardScaler().fit_transform(pca_df.values)
    result = pca_df.copy()

    # Standard KMeans
    result["cluster"] = KMeans(
        n_clusters=best_k, n_init=100, random_state=random_state
    ).fit_predict(X)
    log.info("Standard KMeans (k=%d): %s", best_k, _size_summary(result["cluster"]))

    # Size-constrained KMeans
    KMC = _load_constrained_kmeans() if use_constrained else None
    if KMC is not None:
        n = len(X)
        bucket = n // balanced_k
        model = KMC(
            n_clusters=balanced_k,
            size_min=bucket - 2,
            size_max=bucket + 2,
            random_state=random_state,
        )
        result["balanced_cluster"] = model.fit_predict(X)
        log.info(
            "Balanced KMeans (k=%d): %s",
            balanced_k, _size_summary(result["balanced_cluster"]),
        )
    else:
        # Fall back to plain KMeans so the column always exists
        result["balanced_cluster"] = KMeans(
            n_clusters=balanced_k, n_init=100, random_state=random_state
        ).fit_predict(X)
        log.warning("balanced_cluster uses plain KMeans (k-means-constrained unavailable)")

    # Canonicalize label IDs so cluster 0 always has the smallest mean PC1 value.
    # This makes label assignments deterministic across different k-means random seeds
    # or sklearn versions, as long as the PCA projection is the same.
    result = _canonicalize_cluster_col(result, "cluster")
    result = _canonicalize_cluster_col(result, "balanced_cluster")

    log.info(
        "\n%d quarters clustered into %d regimes (balanced into %d).\n",
        len(result),
        best_k,
        balanced_k,
    )
    return result


def _canonicalize_cluster_col(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Relabel cluster IDs so they are ordered by ascending mean PC1 value of the cluster.
    Cluster 0 → lowest mean PC1, cluster 1 → next, etc.

    This removes the arbitrary label permutation that k-means produces, making
    regime IDs stable across runs with different random seeds.
    """
    pc1_col = next((c for c in df.columns if c.startswith("PC")), None)
    if pc1_col is None or col not in df.columns:
        return df
    mean_pc1 = df.groupby(df[col])[pc1_col].mean().sort_values()
    label_map = {old: new for new, old in enumerate(mean_pc1.index)}
    df = df.copy()
    df[col] = df[col].map(label_map)
    return df


def _size_summary(labels: pd.Series) -> str:
    counts = labels.value_counts().sort_index()
    return ", ".join(f"{k}:{v}" for k, v in counts.items())


# ── Exploration / investigation helpers ────────────────────────────────────────

def optimize_n_components(
    df: pd.DataFrame,
    n_range: range | None = None,
    balanced_k: int = 5,
    n_init: int = 50,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Sweep PCA n_components over n_range and score each with balanced KMeans.

    For each n, fits PCA(n) → StandardScaler → KMeans(balanced_k) and records
    silhouette, Davies-Bouldin, Calinski-Harabasz, and cumulative explained variance.

    Args:
        df         — feature matrix (before PCA; all numeric, no NaNs)
        n_range    — n_components values to sweep (default range(3, 11))
        balanced_k — k for KMeans at each n
        n_init     — KMeans restarts per n
        random_state

    Returns:
        DataFrame with one row per n, columns:
        n_components, explained_variance_pct, silhouette, davies_bouldin, calinski

    Raises:
        ValueError if df is empty, n_range is empty, or balanced_k < 2.
    """
    if df.empty:
        raise ValueError("df is empty — cannot optimize PCA components")
    if balanced_k < 2:
        raise ValueError(f"balanced_k must be >= 2, got {balanced_k}")

    max_components = min(df.shape)
    if n_range is None:
        n_range = range(3, min(11, max_components))

    valid_n = [n for n in n_range if n < max_components]
    skipped = [n for n in n_range if n >= max_components]
    if skipped:
        log.warning(
            "optimize_n_components: skipping n=%s — exceeds min(n_samples, n_features)=%d",
            skipped, max_components,
        )
    if not valid_n:
        raise ValueError(
            f"No valid n values in n_range after applying min(n_samples, n_features)={max_components} limit"
        )

    scaler_outer = StandardScaler()
    X_raw = scaler_outer.fit_transform(df.values)

    rows = []
    for n in valid_n:
        pca = PCA(n_components=n, random_state=random_state)
        X_pca = pca.fit_transform(X_raw)
        cumvar = float(np.sum(pca.explained_variance_ratio_))

        X_scaled = StandardScaler().fit_transform(X_pca)
        labels = KMeans(n_clusters=balanced_k, n_init=n_init, random_state=random_state).fit_predict(X_scaled)

        rows.append({
            "n_components": n,
            "explained_variance_pct": round(cumvar * 100, 2),
            "silhouette": silhouette_score(X_scaled, labels),
            "davies_bouldin": davies_bouldin_score(X_scaled, labels),
            "calinski": calinski_harabasz_score(X_scaled, labels),
        })
        log.info(
            "PCA n=%d  var=%.1f%%  sil=%.4f  DB=%.4f  CH=%.1f",
            n, cumvar * 100, rows[-1]["silhouette"], rows[-1]["davies_bouldin"], rows[-1]["calinski"],
        )

    return pd.DataFrame(rows)


def compare_svd_pca(
    df: pd.DataFrame,
    n_components: int = 5,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run PCA and TruncatedSVD on the same StandardScaled data and compare.

    Since StandardScaler (with_mean=True, the default) already zero-centres the data,
    TruncatedSVD and PCA are mathematically equivalent here — both decompose the
    same zero-mean matrix.  Any differences in loadings are sign flips or numerical
    rounding, not substantive structure.  The comparison is useful for sanity-checking
    that both decompositions agree, and for building intuition about SVD for cases
    where centering is intentionally disabled (e.g. sparse data contexts).

    Returns:
        pca_df      — PC1…PCn components (same index as df)
        svd_df      — SV1…SVn components (same index as df)
        loadings_df — feature × component absolute loadings for both methods side-by-side

    Raises:
        ValueError if n_components >= min(n_samples, n_features).
    """
    if df.empty:
        raise ValueError("df is empty — cannot run SVD/PCA comparison")
    max_components = min(df.shape)
    if n_components >= max_components:
        raise ValueError(
            f"n_components={n_components} must be < min(n_samples, n_features)={max_components}"
        )

    feature_names = list(df.columns)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df.values)

    # PCA
    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)
    pca_cols = [f"PC{i+1}" for i in range(n_components)]
    pca_df = pd.DataFrame(X_pca, index=df.index, columns=pca_cols)

    # TruncatedSVD (no additional centering — operates on X_scaled directly)
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    X_svd = svd.fit_transform(X_scaled)
    svd_cols = [f"SV{i+1}" for i in range(n_components)]
    svd_df = pd.DataFrame(X_svd, index=df.index, columns=svd_cols)

    # Loadings comparison: absolute value of component weights per feature
    pca_loadings = pd.DataFrame(
        np.abs(pca.components_).T, index=feature_names, columns=[f"PCA_{c}" for c in pca_cols]
    )
    svd_loadings = pd.DataFrame(
        np.abs(svd.components_).T, index=feature_names, columns=[f"SVD_{c}" for c in svd_cols]
    )
    loadings_df = pd.concat([pca_loadings, svd_loadings], axis=1)

    pca_var = float(np.sum(pca.explained_variance_ratio_))
    svd_var = float(np.sum(svd.explained_variance_ratio_))
    log.info(
        "PCA: %d components, %.1f%% variance.  SVD: %d components, %.1f%% variance.",
        n_components, pca_var * 100, n_components, svd_var * 100,
    )
    return pca_df, svd_df, loadings_df


def compute_gap_statistic(
    X: np.ndarray,
    k_range: range | None = None,
    n_boots: int = 10,
    n_init: int = 20,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Compute the gap statistic (Tibshirani, Walther & Hastie, 2001) for each k.

    The optimal k is the smallest k such that:
        Gap(k) >= Gap(k+1) - s_{k+1}
    where s_k = sd_k * sqrt(1 + 1/B) is the simulation error.

    Args:
        X        — scaled feature matrix (rows = samples, cols = features)
        k_range  — k values to evaluate (default range(2, 12))
        n_boots  — bootstrap reference datasets (higher = more accurate, slower)
        n_init   — KMeans restarts per k

    Returns:
        DataFrame with columns:
          k        — number of clusters
          gap      — Gap(k) = E*[log W_k^ref] - log W_k
          gap_std  — raw bootstrap standard deviation sd_k = std(log W_k^ref)
          gap_sk   — simulation error s_k = sd_k * sqrt(1 + 1/B)
          optimal  — True for the first k satisfying the Tibshirani criterion

    Raises:
        ValueError if X has fewer than 2 samples or k_range is empty.
    """
    if len(X) < 2:
        raise ValueError(f"X must have at least 2 samples, got {len(X)}")
    if k_range is None:
        k_range = range(2, 12)

    ks = list(k_range)
    if not ks:
        raise ValueError("k_range is empty — provide at least one k value")

    rng = np.random.default_rng(random_state)

    # Bounding box for uniform reference sampling
    mins = X.min(axis=0)
    maxs = X.max(axis=0)

    def _log_wk(X_data: np.ndarray, k: int, seed: int) -> float:
        model = KMeans(n_clusters=k, n_init=n_init, random_state=seed)
        model.fit(X_data)
        return float(np.log(model.inertia_ + 1e-12))

    log_wks: list[float] = []
    boot_log_wks: list[list[float]] = []

    for k in ks:
        log.info("Gap statistic: k=%d ...", k)
        log_wks.append(_log_wk(X, k, seed=random_state))

        boot_vals = []
        for _ in range(n_boots):
            X_ref = rng.uniform(mins, maxs, size=X.shape)
            # Use independent seeds for each bootstrap fit to avoid correlated KMeans init
            boot_seed = int(rng.integers(0, 2**31))
            boot_vals.append(_log_wk(X_ref, k, seed=boot_seed))
        boot_log_wks.append(boot_vals)

    gaps = [float(np.mean(boot)) - obs for boot, obs in zip(boot_log_wks, log_wks)]
    # gap_std: raw bootstrap standard deviation sd_k
    raw_sds = [float(np.std(boot, ddof=1)) if len(boot) > 1 else 0.0 for boot in boot_log_wks]
    # gap_sk: Tibshirani simulation error = sd_k * sqrt(1 + 1/B)
    gap_sks = [s * float(np.sqrt(1 + 1 / n_boots)) for s in raw_sds]

    # Optimal k: smallest k where gap(k) >= gap(k+1) - s(k+1)
    optimal = [False] * len(ks)
    for i in range(len(ks) - 1):
        if gaps[i] >= gaps[i + 1] - gap_sks[i + 1]:
            optimal[i] = True
            break
    if not any(optimal):
        optimal[-1] = True  # fallback to last k
        log.warning(
            "Gap statistic: Tibshirani criterion not satisfied for any k in %s — "
            "defaulting to last k=%d",
            list(k_range), ks[-1],
        )

    return pd.DataFrame({
        "k":       ks,
        "gap":     gaps,
        "gap_std": raw_sds,
        "gap_sk":  gap_sks,
        "optimal": optimal,
    })


def find_knee_k(scores: pd.DataFrame) -> int:
    """
    Find the elbow/knee in the inertia curve via second-derivative (gradient of gradient).

    Uses the `kneed` library if available for a more robust estimate;
    falls back to the gradient-of-gradient method.

    Args:
        scores — DataFrame from evaluate_kmeans() with 'k' and 'inertia' columns.
                 Must have at least 3 rows for meaningful elbow detection.

    Returns:
        k value at the elbow.

    Raises:
        ValueError if required columns are missing or DataFrame has fewer than 2 rows.
    """
    missing_cols = [c for c in ("k", "inertia") if c not in scores.columns]
    if missing_cols:
        raise ValueError(
            f"find_knee_k: scores DataFrame is missing required columns: {missing_cols}. "
            "Pass the output of evaluate_kmeans()."
        )
    if len(scores) < 2:
        raise ValueError(
            f"find_knee_k: scores DataFrame has only {len(scores)} row(s) — "
            "need at least 2 k values for elbow detection"
        )

    ks = scores["k"].values
    inertia = scores["inertia"].values

    # Attempt kneed first
    try:
        from kneed import KneeLocator  # type: ignore[import]
        kl = KneeLocator(ks, inertia, curve="convex", direction="decreasing")
        if kl.knee is not None:
            log.info("kneed: knee at k=%d", kl.knee)
            return int(kl.knee)
    except ImportError:
        log.debug("kneed not installed — using gradient method for elbow detection")

    # Gradient-of-gradient method
    d2 = np.gradient(np.gradient(inertia))
    knee_idx = int(np.argmax(d2))
    knee_k = int(ks[knee_idx])
    log.info("Elbow (gradient method): k=%d", knee_k)
    return knee_k
