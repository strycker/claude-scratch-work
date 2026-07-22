"""
Labeling persistence, churn monitoring, and §4.4 report-only diagnostics
(L1-02/L1-03, design §5.4).

- ``occupancy_and_sojourns(states)``: per-state occupancy fraction and
  run-length (sojourn) statistics — the raw inputs to the §4.4 acceptance
  criteria this module reports on but never gates (D-02).
- ``label_churn(prev_states, new_states, trailing_months=24)``: fraction of
  the trailing ``trailing_months`` labels that were revised between two
  labeler runs (design §5.4). Operates on ALREADY-canonicalized labels —
  un-canonicalized labels read as false churn purely from arbitrary
  cluster-index relabeling (Pitfall 1; see
  ``jump_model.canonicalize_states``).
- ``auto_profile(centroids, feature_names)``: one-line economic profile per
  state from the sign/magnitude of its standardized centroid coordinates
  (D-04) — numeric state ids, no human-pinned names (Phase 4).
- ``report_labeling_diagnostics(metrics, output_dir=None)``: persists the
  schema-stable diagnostics artifact and prints a human-readable summary;
  logs a WARNING on a §4.4 occupancy violation but never raises (D-02:
  report-only in v1, hard-gating deferred to v2/L1-V2-01).
- ``label_regimes(monthly_features, cfg, checkpoint_dir=None)``: the
  orchestration wiring Plan 03-01's jump-model core to persistence and
  monitoring — pull lean cols -> standardize -> fit -> canonicalize ->
  confidences -> churn (load BEFORE save, Pitfall 3) -> persist
  regime_labels/regime_confidences/regime_profiles -> report diagnostics.

Usage::

    from trading_crab_lib.platform.config import load_platform_config
    from trading_crab_lib.platform.checkpoints import get_platform_checkpoint_manager
    from trading_crab_lib.platform.labeling.diagnostics import label_regimes

    cfg = load_platform_config()
    monthly_features = get_platform_checkpoint_manager().load("monthly_features")
    result = label_regimes(monthly_features, cfg)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from trading_crab_lib import OUTPUT_DIR
from trading_crab_lib.checkpoints import CheckpointManager
from trading_crab_lib.platform.checkpoints import get_platform_checkpoint_manager
from trading_crab_lib.platform.labeling.jump_model import (
    canonicalize_states,
    fit_jump_model,
    soft_confidences,
    standardize_features,
)
from trading_crab_lib.platform.taxonomy import lean_feature_set

log = logging.getLogger(__name__)

# Schema-stable columns for the persisted diagnostics artifact (D-05 pattern,
# ported from gap_lag.py) — always present in the written parquet, even when
# metrics is empty.
_ARTIFACT_COLUMNS = ["state", "occupancy_pct", "median_sojourn_months", "profile"]

# §4.4 report-only sanity threshold: a state occupying less than 5% of
# months is worth a loud WARNING (degenerate/near-empty regime) but must
# never block the labeler from completing (D-02).
_MIN_OCCUPANCY_THRESHOLD = 0.05


def occupancy_and_sojourns(states, n_states: int | None = None) -> dict:
    """Per-state occupancy fraction and run-length (sojourn) statistics.

    Args:
        states: array-like of canonicalized state labels (int), length T.
        n_states: total number of possible states (0..n_states-1). Defaults
            to ``max(states) + 1``. Pass K explicitly to surface never-
            occupied states (e.g. K=5 requested but only 3 states occupied).

    Returns:
        dict with keys:
            - ``"occupancy_pct"``: dict[int, float] — fraction of months in
              each state; sums to 1.0 over all states (0.0 for never-
              occupied states, never a ZeroDivisionError).
            - ``"sojourns"``: dict[int, dict] — per-state
              ``{"median_months": float, "mean_months": float, "n_runs": int}``
              from a run-length scan of consecutive same-state positions;
              ``nan`` median/mean for a never-occupied state.
            - ``"overall_median_sojourn_months"`` / ``"overall_mean_sojourn_months"``:
              float, pooled across every run of every state.
    """
    arr = np.asarray(states)
    if arr.size == 0:
        raise ValueError("states must be non-empty")
    total_states = n_states if n_states is not None else int(arr.max()) + 1

    # Run-length scan: consecutive same-state positions form one "sojourn".
    runs: list[tuple[int, int]] = []
    run_state = int(arr[0])
    run_len = 1
    for s in arr[1:]:
        s = int(s)
        if s == run_state:
            run_len += 1
        else:
            runs.append((run_state, run_len))
            run_state, run_len = s, 1
    runs.append((run_state, run_len))

    n_total = len(arr)
    occupancy_pct: dict[int, float] = {}
    sojourns: dict[int, dict] = {}
    all_run_lengths: list[int] = []
    for state in range(total_states):
        occupancy_pct[state] = int((arr == state).sum()) / n_total
        state_runs = [length for s, length in runs if s == state]
        if state_runs:
            sojourns[state] = {
                "median_months": float(np.median(state_runs)),
                "mean_months": float(np.mean(state_runs)),
                "n_runs": len(state_runs),
            }
            all_run_lengths.extend(state_runs)
        else:
            sojourns[state] = {"median_months": float("nan"), "mean_months": float("nan"), "n_runs": 0}

    return {
        "occupancy_pct": occupancy_pct,
        "sojourns": sojourns,
        "overall_median_sojourn_months": float(np.median(all_run_lengths)) if all_run_lengths else float("nan"),
        "overall_mean_sojourn_months": float(np.mean(all_run_lengths)) if all_run_lengths else float("nan"),
    }


def label_churn(prev_states, new_states, *, trailing_months: int = 24) -> float:
    """Fraction of the trailing ``trailing_months`` labels that differ (design §5.4).

    Operates on ALREADY-canonicalized labels — un-canonicalized labels
    produce false churn purely from arbitrary cluster-index relabeling
    (Pitfall 1). Compares the trailing ``trailing_months`` positions of
    ``prev_states`` against the trailing ``trailing_months`` positions of
    ``new_states`` (aligned by position from the end of each array).

    Args:
        prev_states: array-like of previous canonicalized states.
        new_states: array-like of new canonicalized states.
        trailing_months: window size (design default: 24).

    Returns:
        float in [0.0, 1.0], or ``nan`` if either input is empty.
    """
    prev_arr = np.asarray(prev_states)
    new_arr = np.asarray(new_states)
    if prev_arr.size == 0 or new_arr.size == 0:
        return float("nan")
    n = min(len(prev_arr), len(new_arr), trailing_months)
    return float(np.mean(prev_arr[-n:] != new_arr[-n:]))


def _get_checkpoint_manager(checkpoint_dir: Path | None) -> CheckpointManager:
    """Return the platform checkpoint manager, or one scoped to *checkpoint_dir* for tests."""
    if checkpoint_dir is None:
        return get_platform_checkpoint_manager()
    return CheckpointManager(checkpoint_dir=checkpoint_dir)


def _churn_against_previous(new_states, cm: CheckpointManager, *, trailing_months: int = 24) -> float:
    """Load-before-save churn helper (Pitfall 3): MUST be called BEFORE the
    new "regime_labels" checkpoint is saved, or the "previous" checkpoint
    read here would already be the new one, producing a false 0.0 churn
    every run instead of the correct NaN-on-first-run / real-churn-after.
    """
    try:
        previous = cm.load("regime_labels")
    except FileNotFoundError:
        log.info("no previous regime_labels checkpoint — churn metric unavailable on first run")
        return float("nan")
    return label_churn(previous["state"].to_numpy(), new_states, trailing_months=trailing_months)


def auto_profile(centroids: np.ndarray, feature_names: list[str], *, top_k: int = 3) -> dict[int, str]:
    """One-line economic profile per state from standardized centroid signs (D-04).

    Args:
        centroids: (K, d) standardized centroid array (canonicalized order).
        feature_names: column names matching centroids' second axis, in order.
        top_k: number of most-salient features to name per state.

    Returns:
        dict[int, str] — one deterministic profile string per state, e.g.
        "state 0: low trailing_return_1m, high realized_vol_3m, wide credit_spread".
    """
    profiles: dict[int, str] = {}
    for state_idx, row in enumerate(centroids):
        top_idx = np.argsort(-np.abs(row))[:top_k]
        parts = [f"{'high' if row[i] >= 0 else 'low'} {feature_names[i]}" for i in top_idx]
        profiles[state_idx] = f"state {state_idx}: " + ", ".join(parts)
    return profiles


def report_labeling_diagnostics(metrics: dict, *, output_dir: Path | None = None) -> Path:
    """Print + persist the §4.4 report-only diagnostics artifact (D-02, D-05 pattern).

    A §4.4 occupancy violation (a state below ``_MIN_OCCUPANCY_THRESHOLD``)
    is logged loudly at WARNING but the artifact is still written and this
    function still returns normally — the labeler always completes.

    Args:
        metrics: dict, any subset of:
            - ``"occupancy"``: dict[int, float] (state -> occupancy_pct)
            - ``"sojourns"``: dict[int, dict] (state -> {"median_months": ...})
            - ``"profiles"``: dict[int, str] (state -> profile)
        output_dir: overrides the default artifact directory (for tests).

    Returns:
        Path: the written parquet artifact path.
    """
    target_dir = Path(output_dir) if output_dir is not None else OUTPUT_DIR / "reports" / "model_metrics"
    target_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = target_dir / "labeling_diagnostics.parquet"

    occupancy = metrics.get("occupancy") or {}
    sojourns = metrics.get("sojourns") or {}
    profiles = metrics.get("profiles") or {}

    rows = []
    for state in sorted(set(occupancy) | set(sojourns) | set(profiles)):
        occ = occupancy.get(state, float("nan"))
        rows.append({
            "state": state,
            "occupancy_pct": occ,
            "median_sojourn_months": sojourns.get(state, {}).get("median_months", float("nan")),
            "profile": profiles.get(state, ""),
        })
        if not np.isnan(occ) and occ < _MIN_OCCUPANCY_THRESHOLD:
            log.warning(
                "State %d occupancy %.1f%% below §4.4 sanity threshold — report-only, not blocking (D-02)",
                state, occ * 100,
            )

    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=_ARTIFACT_COLUMNS)
    df.to_parquet(artifact_path, index=False)

    summary_lines = ["Labeling Diagnostics (design §4.4, report-only D-02)"]
    for row in rows:
        summary_lines.append(
            f"  state {row['state']}: occupancy={row['occupancy_pct']:.1%}  "
            f"median_sojourn={row['median_sojourn_months']}mo  {row['profile']}"
        )
    summary_lines.append(f"  artifact: {artifact_path}")
    summary = "\n".join(summary_lines)
    log.info(summary)
    print(summary)  # noqa: T201 — first-class CLI run output (D-05 pattern), not debug noise

    return artifact_path


def label_regimes(monthly_features: pd.DataFrame, cfg: dict, *, checkpoint_dir: Path | None = None) -> dict:
    """Full L1 labeler orchestration (L1-01..03, D-02/D-03/D-04).

    Pull lean cols -> standardize -> fit_jump_model -> canonicalize ->
    soft_confidences -> compute churn against the previous regime_labels
    checkpoint (load BEFORE save, Pitfall 3) -> persist regime_labels,
    regime_confidences, regime_profiles -> compute+report §4.4 diagnostics
    (report-only, D-02 — this function always completes).

    Args:
        monthly_features: lean-tagged monthly features DataFrame (Phase 1
            ``monthly_features`` checkpoint shape).
        cfg: platform config (``load_platform_config()`` output); reads the
            optional ``labeling:`` section (K, lambda, n_restarts).
        checkpoint_dir: overrides the platform checkpoint directory (for tests).

    Returns:
        dict with keys ``states``, ``confidences``, ``churn``, ``diagnostics_path``.
    """
    labeling_cfg = cfg.get("labeling", {})
    K = labeling_cfg.get("K", 5)
    lam = labeling_cfg.get("lambda", 52.0)
    n_restarts = labeling_cfg.get("n_restarts", 10)

    lean_cols = sorted(lean_feature_set(cfg) & set(monthly_features.columns))
    X_df = monthly_features[lean_cols].dropna()
    X = standardize_features(X_df)

    fit = fit_jump_model(X, K=K, lam=lam, n_restarts=n_restarts)
    states, centroids = canonicalize_states(fit["states"], fit["centroids"], lean_cols)

    d = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    confidences = soft_confidences(d)

    cm = _get_checkpoint_manager(checkpoint_dir)
    churn = _churn_against_previous(states, cm)  # MUST run before the save below (Pitfall 3)

    labels_df = pd.DataFrame({"state": states}, index=X_df.index)
    confidences_df = pd.DataFrame(confidences, index=X_df.index, columns=[f"state_{k}" for k in range(K)])
    cm.save(labels_df, "regime_labels")
    cm.save(confidences_df, "regime_confidences")

    profiles = auto_profile(centroids, lean_cols)
    profiles_df = pd.DataFrame({"state": list(profiles.keys()), "profile": list(profiles.values())})
    cm.save(profiles_df, "regime_profiles")

    occ_sojourn = occupancy_and_sojourns(states, n_states=K)
    diagnostics_path = report_labeling_diagnostics(
        {
            "occupancy": occ_sojourn["occupancy_pct"],
            "sojourns": occ_sojourn["sojourns"],
            "profiles": profiles,
        },
        # Route the diagnostics artifact alongside the checkpoint dir override
        # so tests never touch the real OUTPUT_DIR (checkpoint_dir is only
        # non-None in tests / the __main__ self-check).
        output_dir=checkpoint_dir,
    )

    return {"states": states, "confidences": confidences, "churn": churn, "diagnostics_path": diagnostics_path}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Synthetic self-check — no network, no Phase-1 real data (mirrors
    # gap_lag.py's __main__ footer, per ponytail's "non-trivial logic
    # leaves one runnable check" rule).
    rng = np.random.default_rng(42)
    n_months = 96
    idx = pd.date_range("2010-01-31", periods=n_months, freq="ME")
    columns = [
        "curve_10y3m", "curve_10y2y", "credit_spread_baa_aaa", "fred_vix", "gold", "oil",
        "trailing_return_1m", "trailing_return_3m", "realized_vol_1m", "realized_vol_3m",
        "cape_shiller", "div_yield", "real_rate_level",
    ]
    synthetic_features = pd.DataFrame({col: rng.normal(0, 1, n_months) for col in columns}, index=idx)
    synthetic_cfg = {
        "labeling": {"K": 3, "lambda": 20.0, "n_restarts": 3},
        "taxonomy": {"fast": columns[:10], "slow": columns[10:], "agency": []},
    }

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        result = label_regimes(synthetic_features, synthetic_cfg, checkpoint_dir=Path(tmp))
        print(f"self-check: {len(result['states'])} months labeled, churn={result['churn']}")  # noqa: T201
