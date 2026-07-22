"""
Walk-forward runner — expanding-window refit loop (HON-03/D-02).

Design §8.1 ("walk-forward everything": refit on data <= t, record every
decision, step forward) and §14's Phase 0 exit criterion ("walk-forward
runner executes a trivial model end-to-end"). This module proves the
refit -> record -> step-forward cycle with a TRIVIAL model
(``DummyClassifier(strategy="prior")`` — ponytail rung 5, sklearn already
implements "predict the class prior", no hand-rolled persistence classifier
needed). Phase 3 swaps in real models (jump-model labeler, nowcaster)
against this exact same interface.

Per D-02 and RESEARCH.md Pitfall 3, the runner logs the full evaluated
configuration to the trial registry automatically, exactly once per run —
callers never have to remember to call ``append_trial`` themselves.

Usage:
    from trading_crab_lib.platform.honesty.walkforward import run_walkforward

    decisions = run_walkforward(features_df, target_series, min_train=60,
                                 registry_path=tmp_path / "trials.jsonl")
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier

from trading_crab_lib.platform.honesty import registry

log = logging.getLogger(__name__)


def expanding_steps(index: pd.Index, *, min_train: int, step: int = 1):
    """Yield ``(t, train_index, test_index)`` for an expanding window.

    ``train_index = index[:i]`` (STRICTLY before ``t`` — the row being
    predicted is never in its own training window) and
    ``test_index = index[i:i+1]`` for ``i`` from ``min_train`` to
    ``len(index) - 1``, stepping by ``step``. Requires a monotonically
    increasing ``index`` (a sorted DatetimeIndex, e.g.) so that position
    order matches value order.
    """
    for i in range(min_train, len(index), step):
        yield index[i], index[:i], index[i : i + 1]


def run_walkforward(
    features_df: pd.DataFrame,
    target_series: pd.Series,
    *,
    model: object = None,
    min_train: int = 60,
    registry_path: object = None,
    config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Run the expanding-window walk-forward loop and log one trial.

    Clones *model* (default ``DummyClassifier(strategy="prior")``) at every
    step via :func:`sklearn.base.clone` (never refits one shared estimator —
    mirrors the clone-per-fold idiom in
    ``monitoring/prediction.py::compute_cv_fold_scores``), fits on the train
    slice, predicts the single test row, and records a decision dict
    ``{t, prediction, params}``.

    On completion, calls :func:`registry.append_trial` exactly once (D-02: a
    trial is one evaluated configuration's full walk-forward result, not one
    step) with the model class + params + ``min_train``, the feature list,
    and aggregate metrics. Returns the list of per-step decisions.
    """
    model = model if model is not None else DummyClassifier(strategy="prior")

    decisions: list[dict[str, Any]] = []
    hits = 0
    for t, train_index, test_index in expanding_steps(features_df.index, min_train=min_train):
        m = clone(model)
        m.fit(features_df.loc[train_index], target_series.loc[train_index])
        pred = m.predict(features_df.loc[test_index])
        decisions.append({"t": t, "prediction": pred.tolist(), "params": m.get_params()})
        if pred[0] == target_series.loc[test_index].iloc[0]:
            hits += 1

    n_steps = len(decisions)
    hit_rate = hits / n_steps if n_steps else 0.0

    trial_config = dict(config or {})
    trial_config.update(
        {
            "model_class": type(model).__name__,
            "model_params": model.get_params(),
            "min_train": min_train,
            "step": 1,
        }
    )
    registry.append_trial(
        config=trial_config,
        features=list(features_df.columns),
        metrics={"n_steps": n_steps, "hit_rate": hit_rate},
        path=registry_path,
    )

    return decisions
