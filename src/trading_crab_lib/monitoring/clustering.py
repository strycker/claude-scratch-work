"""Monitoring helpers for clustering and regime stability (C2.3–C2.4)."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

log = logging.getLogger(__name__)


# ── C2.3: Clustering method comparison summary ───────────────────────────


def format_method_comparison(comparison_df: pd.DataFrame) -> str:
    """Format a clustering comparison DataFrame for logging.

    Parameters
    ----------
    comparison_df :
        Output of ``cluster_comparison.compare_all_methods()`` with columns:
        method, n_clusters, silhouette, davies_bouldin, calinski.
    """
    if comparison_df.empty:
        return "  (no methods to compare)"

    lines = ["  Clustering method comparison:"]
    header = f"    {'Method':<20} {'k':>3}  {'Silhouette':>10}  {'DB':>8}  {'CH':>10}"
    lines.append(header)
    lines.append("    " + "-" * (len(header) - 4))
    for _, row in comparison_df.iterrows():
        lines.append(
            f"    {row.get('method', '?'):<20} {int(row.get('n_clusters', 0)):>3}"
            f"  {row.get('silhouette', float('nan')):>10.4f}"
            f"  {row.get('davies_bouldin', float('nan')):>8.3f}"
            f"  {row.get('calinski', float('nan')):>10.1f}"
        )
    # Identify best method by silhouette
    best_idx = comparison_df["silhouette"].idxmax()
    best = comparison_df.loc[best_idx]
    lines.append(f"  Best by silhouette: {best['method']} ({best['silhouette']:.4f})")
    return "\n".join(lines)


# ── C2.4: Regime stability summary ───────────────────────────────────────


@dataclass
class RegimeStabilityReport:
    """Result of a regime stability analysis from the transition matrix."""

    persistence: dict[int, float] = field(default_factory=dict)
    most_stable: tuple[int, float] | None = None
    least_stable: tuple[int, float] | None = None
    avg_duration: dict[int, float] = field(default_factory=dict)

    def summary(self) -> str:
        """Return a formatted summary of regime persistence and run lengths."""
        lines = ["  Regime stability (transition matrix diagonal):"]
        if not self.persistence:
            return "  (no persistence data)"
        max_id_len = max(len(str(r)) for r in self.persistence)
        for rid, pval in sorted(self.persistence.items()):
            bar = "#" * int(pval * 20)
            dur = self.avg_duration.get(rid, 0)
            lines.append(
                f"    Regime {rid:<{max_id_len}}:  persist={pval:5.1%}  "
                f"avg_run={dur:.1f}Q  {bar}"
            )
        if self.most_stable:
            lines.append(
                f"  Most stable:  regime {self.most_stable[0]} "
                f"({self.most_stable[1]:.1%} self-transition)"
            )
        if self.least_stable:
            lines.append(
                f"  Least stable: regime {self.least_stable[0]} "
                f"({self.least_stable[1]:.1%} self-transition)"
            )
        return "\n".join(lines)


def compute_regime_stability(
    transition_matrix: pd.DataFrame,
    labels: pd.Series,
) -> RegimeStabilityReport:
    """Compute regime stability metrics from a transition matrix and labels.

    Parameters
    ----------
    transition_matrix :
        Square DataFrame where entry (i, j) is P(regime j at t+1 | regime i at t).
        Rows and columns are regime IDs.
    labels :
        Time-ordered Series of regime assignments (one per quarter).

    Returns
    -------
    RegimeStabilityReport with persistence probabilities and average durations.
    """
    report = RegimeStabilityReport()

    # Diagonal = persistence probability
    for rid in transition_matrix.index:
        if rid in transition_matrix.columns:
            report.persistence[rid] = float(transition_matrix.loc[rid, rid])

    if report.persistence:
        best = max(report.persistence.items(), key=lambda x: x[1])
        worst = min(report.persistence.items(), key=lambda x: x[1])
        report.most_stable = best
        report.least_stable = worst

    # Average consecutive run length per regime
    if not labels.empty:
        labels_arr = labels.values
        for rid in sorted(set(labels_arr)):
            runs: list[int] = []
            current_run = 0
            for val in labels_arr:
                if val == rid:
                    current_run += 1
                else:
                    if current_run > 0:
                        runs.append(current_run)
                    current_run = 0
            if current_run > 0:
                runs.append(current_run)
            report.avg_duration[rid] = sum(runs) / len(runs) if runs else 0.0

    log.info("Regime stability:\n%s", report.summary())
    return report
