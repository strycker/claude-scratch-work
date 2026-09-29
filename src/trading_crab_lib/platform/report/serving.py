"""
Serving builder: the artifacts the weekly report reads, built by the evaluated recipe
(plan 08-13, gap G-08-2).

``report/weekly.py`` loads three platform checkpoints that no supported command produced
before this module existed:

- ``nowcaster`` (``nowcaster.pkl``, joblib): the L2 model the report scores;
- ``returns_by_regime``: the per-(regime, asset) table the tilt conditions on;
- ``asset_returns``: the tradable monthly returns the tilt's live volatility estimate uses.

This module builds all three. The nowcaster is fit by ``backtest/driver.py::fit_l2_nowcaster``,
the SAME function the backtest's ``_refit_l2`` calls at every walk-forward step: same
embargo, same ``_cv_safe_active_features`` rule, same calibrated LR, same label series. One
function with two callers is the train/serve-skew guarantee. ``asset_returns`` uses
``assets/returns.py::tradable_asset_returns``, the same research-to-tradable mapping the
evaluated backtest's universe uses.

**The ruling (Glenn, 2026-09-28): "the serving fit is NOT a registry trial".** It refits an
already-evaluated configuration on full history and selects nothing. ``registry.append_trial``
is called exactly once, with ``path=registry.NO_REGISTRY`` hard-coded; no parameter can change
that. ``total_trial_count()`` is unchanged and ``registry/trials.jsonl`` stays byte-identical.

**The data window.**

- Fitting reads the DEV ``monthly_features`` and ``regime_labels`` only, both ending
  2020-12-31 (the holdout boundary), through the default platform checkpoint manager, and the
  features pass through ``split_by_holdout_boundary`` as a belt-and-braces fence. Training
  targets end 2019-12-31 under the 12-month label embargo. No holdout path is opened: not
  ``load_full_span``, not ``get_holdout_checkpoint_manager``, not ``HOLDOUT_CHECKPOINT_DIR``.
- ``asset_returns`` spans the full ``monthly_raw`` (which runs to the latest month), because
  the tilt's live volatility estimate needs current months. That is looking, not fitting,
  and it selects nothing. ``returns_by_regime`` conditions only the dev side of those returns
  on the dev labels.

**Run order.**

1. ``python scripts/build_platform_data.py`` (the data; FRED key plus network).
2. ``python -m trading_crab_lib.platform.report.serving`` (this module).
3. ``python -m trading_crab_lib.platform.report.weekly [--send-email]``.

**What it deliberately does not do.** No registry row. No accuracy, in-sample or otherwise:
the only model statistic it computes is the count of distinct posteriors over the dev rows,
an input-sensitivity diagnostic, not a metric of merit. No holdout path. No call to
``prediction/nowcaster.py::evaluate_nowcaster`` (which appends a trial).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_crab_lib.platform.assets.returns import (
    compute_monthly_returns,
    report_returns_by_regime,
    tradable_asset_returns,
)
from trading_crab_lib.platform.backtest.driver import fit_l2_nowcaster
from trading_crab_lib.platform.checkpoints import get_platform_checkpoint_manager
from trading_crab_lib.platform.config import load_platform_config
from trading_crab_lib.platform.honesty import registry
from trading_crab_lib.platform.honesty.holdout import DEFAULT_HOLDOUT_CUTOFF, split_by_holdout_boundary
from trading_crab_lib.platform.prediction.nowcaster import build_nowcaster_training_set
from trading_crab_lib.platform.splice import build_core_research_series

log = logging.getLogger(__name__)

#: The supported command that builds every serving artifact. ``weekly.py`` names it in its
#: missing-artifact errors; a test asserts it resolves to this module.
SERVING_BUILD_COMMAND = "python -m trading_crab_lib.platform.report.serving"

#: Tag carried by the (never written) NO_REGISTRY row, so the row says what it was.
SERVING_TRIAL_TAG = "serving-nowcaster-refit"

_WEEKLY_COMMAND = "python -m trading_crab_lib.platform.report.weekly"


def _training_block(
    dev_features: pd.DataFrame, labels: pd.Series, columns: list[str], cfg: dict[str, Any]
) -> pd.Series:
    """The labels of the rows the fit actually trained on, for the record only.

    Rebuilds the embargoed training set with the configured embargo and keeps the rows
    where every model column is finite — ``fit_nowcaster``'s own row rule.
    """
    embargo_months = cfg.get("labeling", {}).get("embargo_months", 12)
    X, y = build_nowcaster_training_set(dev_features, labels, embargo_months=embargo_months)
    finite = np.isfinite(X[columns].to_numpy(dtype=float)).all(axis=1)
    return y.loc[finite]


def build_serving_artifacts(
    cfg: dict[str, Any],
    *,
    cm: Any = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Fit the serving nowcaster and write the three artifacts the weekly report reads.

    Args:
        cfg: platform config (``load_platform_config()``).
        cm: checkpoint manager for the DEV platform namespace (default
            ``get_platform_checkpoint_manager()``). Loosely typed on purpose: annotating it
            ``CheckpointManager`` would import a legacy module into this one.
        output_dir: overrides where ``report_returns_by_regime`` writes its parquet
            artifact (default ``OUTPUT_DIR/reports/platform``).

    Returns:
        dict: the facts of what was built — ``columns``, ``classes``, ``class_counts``,
        ``n_train_rows``, ``train_first``, ``train_last``, ``n_distinct_posteriors_dev``,
        ``asset_returns_first`` / ``asset_returns_last`` / ``asset_returns_columns``,
        ``returns_by_regime_rows`` and ``registry_row_written`` (always False).
    """
    cm = cm or get_platform_checkpoint_manager()

    # 1-2. DEV features and labels only. The fence is on fitting: no holdout path here.
    dev_features, _ = split_by_holdout_boundary(cm.load("monthly_features"), cutoff=DEFAULT_HOLDOUT_CUTOFF)
    labels = cm.load("regime_labels")["state"]

    # 3. The evaluated recipe, by its own function.
    model, columns = fit_l2_nowcaster(dev_features, labels, cfg)

    # 4. For the record only: the block the fit trained on.
    train_y = _training_block(dev_features, labels, columns, cfg)
    class_counts = {int(k): int(v) for k, v in train_y.value_counts().sort_index().items()}
    train_first = pd.Timestamp(train_y.index.min())
    train_last = pd.Timestamp(train_y.index.max())

    # 5. Input sensitivity: how many distinct posteriors the model emits over the complete
    # dev rows. Exact float comparison, no rounding.
    complete_dev = dev_features[columns].dropna()
    n_distinct = int(np.unique(model.predict_proba(complete_dev), axis=0).shape[0])
    log.info(
        "serving nowcaster: %d distinct posterior vector(s) across %d complete dev months "
        "(input-sensitivity diagnostic, not a metric of merit)",
        n_distinct, len(complete_dev),
    )

    # 6. The ruling as code: NOT a registry trial. Hard-coded, not a parameter.
    registry.append_trial(
        config={
            "trial_tag": SERVING_TRIAL_TAG,
            "recipe": "backtest.driver.fit_l2_nowcaster",
            "labels_source": "regime_labels",
            "independent_trial": False,
            "ruling": "2026-09-28: serving fit is not a registry trial",
        },
        features=columns,
        metrics={
            "n_train_rows": len(train_y),
            "train_first": train_first.date().isoformat(),
            "train_last": train_last.date().isoformat(),
            "class_counts": {str(k): v for k, v in class_counts.items()},
        },
        path=registry.NO_REGISTRY,
    )

    # 7. The model (joblib via CheckpointManager.save_model, never raw pickle — P27).
    cm.save_model(model, "nowcaster")
    log.info(
        "wrote nowcaster: %d columns, classes %s, trained on %d rows %s -> %s",
        len(columns), [int(c) for c in model.classes_], len(train_y), train_first.date(), train_last.date(),
    )

    # 8. Tradable returns over the full monthly_raw span (looking, not fitting).
    returns = compute_monthly_returns(build_core_research_series(cm.load("monthly_raw"), cfg))
    asset_returns = tradable_asset_returns(returns, cfg["splice"])
    excluded = [
        params["research_name"]
        for name, params in cfg["splice"].items()
        if name != "cash" and params["research_name"] not in returns.columns
    ]
    if excluded:
        log.warning("serving asset universe EXCLUDES unavailable research classes: %s", excluded)
    cm.save(asset_returns, "asset_returns")
    log.info(
        "wrote asset_returns: %s, %s -> %s (%d months)",
        list(asset_returns.columns), asset_returns.index.min().date(), asset_returns.index.max().date(),
        len(asset_returns),
    )

    # 9. Returns conditioned on the dev labels (both sides of the join end at the boundary).
    dev_returns, _ = split_by_holdout_boundary(asset_returns, cutoff=DEFAULT_HOLDOUT_CUTOFF)
    report_returns_by_regime(dev_returns, labels, cm=cm, output_dir=output_dir)
    rbr_rows = len(cm.load("returns_by_regime"))
    log.info(
        "wrote returns_by_regime: %d (regime, asset) rows, labels %s -> %s",
        rbr_rows, labels.index.min().date(), labels.index.max().date(),
    )

    return {
        "columns": list(columns),
        "classes": [int(c) for c in model.classes_],
        "class_counts": class_counts,
        "n_train_rows": len(train_y),
        "train_first": train_first,
        "train_last": train_last,
        "n_distinct_posteriors_dev": n_distinct,
        "asset_returns_first": pd.Timestamp(asset_returns.index.min()),
        "asset_returns_last": pd.Timestamp(asset_returns.index.max()),
        "asset_returns_columns": list(asset_returns.columns),
        "returns_by_regime_rows": rbr_rows,
        "registry_row_written": False,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: build the serving artifacts, then name the next command."""
    parser = argparse.ArgumentParser(
        description=(
            "Build the serving artifacts the platform weekly report reads (nowcaster, "
            "returns_by_regime, asset_returns) by the evaluated recipe. Not a registry trial."
        )
    )
    parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)

    cfg = load_platform_config()
    facts = build_serving_artifacts(cfg)
    log.info(
        "serving artifacts built (%d model columns, classes %s, no registry row). Next: %s",
        len(facts["columns"]), facts["classes"], _WEEKLY_COMMAND,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
