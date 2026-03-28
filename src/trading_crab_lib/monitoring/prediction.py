"""Monitoring helpers for prediction and dashboard QA (C3.1, C3.5)."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ── C3.1: Per-fold CV accuracy summary ───────────────────────────────────


@dataclass
class CVFoldReport:
    """Per-fold cross-validation results for one or more models."""

    model_scores: dict[str, list[float]] = field(default_factory=dict)

    def add(self, model_name: str, fold_scores: list[float]) -> None:
        """Record per-fold accuracy scores for *model_name*."""
        self.model_scores[model_name] = fold_scores

    def summary(self) -> str:
        """Return a formatted table of per-model fold accuracies with mean ± std."""
        lines = ["  Cross-validation fold accuracy:"]
        for name, scores in self.model_scores.items():
            mean = np.mean(scores) if scores else 0.0
            std = np.std(scores) if scores else 0.0
            fold_str = "  ".join(f"{s:.1%}" for s in scores)
            lines.append(f"    {name:<10}  folds: [{fold_str}]  mean={mean:.1%} ± {std:.1%}")
        return "\n".join(lines)


def compute_cv_fold_scores(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_splits: int = 5,
) -> list[float]:
    """Run TimeSeriesSplit CV on a fitted model's class and return per-fold accuracies.

    Clones the model via sklearn's clone() to avoid mutating the original.
    """
    from sklearn.base import clone
    from sklearn.model_selection import TimeSeriesSplit

    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores: list[float] = []
    for train_idx, test_idx in tscv.split(X):
        m = clone(model)
        m.fit(X.iloc[train_idx], y.iloc[train_idx])
        scores.append(float(m.score(X.iloc[test_idx], y.iloc[test_idx])))
    return scores


# ── C3.5: Dashboard QA gate ──────────────────────────────────────────────


def check_regime_probabilities(
    probabilities: dict[int, float],
    *,
    min_threshold: float = 0.05,
) -> list[str]:
    """Check if any regime has suspiciously low probability.

    Returns a list of warning messages (empty if all OK).
    """
    warnings: list[str] = []
    for regime_id, prob in sorted(probabilities.items()):
        if prob < min_threshold:
            warnings.append(
                f"Regime {regime_id} has very low probability ({prob:.1%} < {min_threshold:.0%})"
            )
    return warnings
