"""
run_pipeline.py — Backward-compatible entry point.

The pipeline logic now lives in ``trading_crab.pipeline``.
Use the ``tradingcrab`` CLI command (installed via pip) for the same functionality.

Usage:
    python run_pipeline.py --refresh --recompute --plots
    python run_pipeline.py --steps 3,4,5,6,7 --plots
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from repo root without `pip install -e .`
sys.path.insert(0, str(Path(__file__).parent / "src"))

from trading_crab.pipeline import main  # noqa: E402

if __name__ == "__main__":
    main()
