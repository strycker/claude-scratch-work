"""
trading_crab_lib.platform — Monthly data layer & long-histories subpackage.

This subpackage is the Phase 6 migration unit (D-02, platform_design.md v1.7):
it is built as a self-contained package so it can lift cleanly into
`strycker/trading-crab` once validated. It imports the incumbent's existing
fetchers (fred.py, multpl.py, macrotrends.py, assets.py) for reuse, but never
edits them (D-01) — the quarterly incumbent pipeline (9 steps, 769 tests) is
FROZEN.

No re-exports are defined here on purpose: downstream modules import directly
from their owning submodule (``trading_crab_lib.platform.config``,
``trading_crab_lib.platform.checkpoints``, ``trading_crab_lib.platform.taxonomy``)
to keep import graphs explicit and avoid accidental circular imports as the
subpackage grows across Phase 1's later plans.
"""

from __future__ import annotations
