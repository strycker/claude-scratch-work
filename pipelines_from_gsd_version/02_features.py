"""
Pipeline step 2 — Feature Engineering

Thin wrapper around run_pipeline.step2_features so there is a single source
of truth for how features are engineered and checkpointed.

Run:
    python pipelines/02_features.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from market_regime.config import load, setup_logging
from market_regime.runtime import RunConfig

from run_pipeline import step2_features


def main() -> None:
    """Entry point for standalone step-02 execution."""
    setup_logging()
    cfg = load()

    # For a direct step-02 run we always recompute features from the latest
    # macro_raw checkpoint and skip plotting. Other flags use RunConfig defaults.
    run_cfg = RunConfig(
        recompute_derived_datasets=True,
        generate_plots=False,
    )

    step2_features(cfg, run_cfg)


if __name__ == "__main__":
    main()
