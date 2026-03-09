"""
RunConfig — global runtime flags for the pipeline.

Mirrors the top-of-script flags in legacy/unified_script.py:
  VERBOSE, GENERATE_PLOTS, GENERATE_OPTIONAL_SNS_PAIRPLOT,
  GENERATE_OPTIONAL_SCATTER_MATRIX_PLOT, REFRESH_SOURCE_DATASETS,
  RECOMPUTE_DERIVED_DATASETS

Construct one RunConfig at the entry point (run_pipeline.py or a pipeline
script) and pass it through to every module that needs it.

Usage:
    from market_regime.runtime import RunConfig
    run_cfg = RunConfig(generate_plots=True, verbose=True)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field


@dataclass
class RunConfig:
    # ── verbosity ─────────────────────────────────────────────────────────
    verbose: bool = False                   # DEBUG logging if True

    # ── plotting ──────────────────────────────────────────────────────────
    generate_plots: bool = False            # produce matplotlib figures
    generate_pairplot: bool = False         # seaborn pairplot (slow, optional)
    generate_scatter_matrix: bool = False   # pandas scatter_matrix (slow, optional)
    save_plots: bool = True                 # write PNGs to outputs/plots/
    show_plots: bool = False                # plt.show() — False for headless/CI

    # ── data refresh ──────────────────────────────────────────────────────
    refresh_source_datasets: bool = False   # re-scrape multpl + re-hit FRED
    recompute_derived_datasets: bool = False  # recompute features from cached raw

    # ── misc ──────────────────────────────────────────────────────────────
    use_constrained_kmeans: bool = True     # attempt k-means-constrained

    @classmethod
    def from_args(cls, args) -> "RunConfig":
        """
        Build a RunConfig from a parsed argparse.Namespace.

        Designed to work with the argparse setup in run_pipeline.py —
        attribute names match the argparse dest names exactly.
        """
        return cls(
            verbose=getattr(args, "verbose", False),
            generate_plots=getattr(args, "plots", False),
            generate_pairplot=getattr(args, "pairplot", False),
            generate_scatter_matrix=getattr(args, "scatter_matrix", False),
            save_plots=not getattr(args, "no_save_plots", False),
            show_plots=getattr(args, "show_plots", False),
            refresh_source_datasets=getattr(args, "refresh", False),
            recompute_derived_datasets=getattr(args, "recompute", False),
            use_constrained_kmeans=not getattr(args, "no_constrained", False),
        )

    def apply_logging(self) -> None:
        """Set root logger to DEBUG if verbose, else leave at INFO."""
        if self.verbose:
            logging.getLogger().setLevel(logging.DEBUG)

    def __str__(self) -> str:
        flags = []
        if self.verbose:
            flags.append("verbose")
        if self.generate_plots:
            flags.append("plots")
        if self.generate_pairplot:
            flags.append("pairplot")
        if self.refresh_source_datasets:
            flags.append("refresh")
        if self.recompute_derived_datasets:
            flags.append("recompute")
        return f"RunConfig({', '.join(flags) or 'defaults'})"
