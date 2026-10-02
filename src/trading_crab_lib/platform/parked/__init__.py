"""
Parked research code: moved here in Phase 8.2 (08.2-02), not deleted.

What is parked
    ``classifier2``   the leadership / relative classifier #2 (DECISIONS L1-04: DEFER, added no lift).
    ``joint_driver``  the two-classifier joint backtest behind criterion 7 (E-02 / E-03: no lift).
    ``stability``     the subsample-stability suite for criterion 3 (G-07 scope cut).

Why
    The weekly product (MVP-1) does not use any of them, and G-07 asks that the active tree show only
    what the weekly page runs. Git history is kept (``git mv``), and there are no import shims.

Rule
    Active ``src`` must never import this package, at module level or inside a function. That is
    enforced by ``tests/unit/test_platform_parked_boundary.py`` (a fresh-interpreter ``sys.modules``
    check over ``report/`` and ``tripwire/``, plus an AST scan of all other modules).

How to un-park
    ``git mv`` the module back to its old directory (``labeling/`` for classifier2 and stability,
    ``backtest/`` for joint_driver), fix the importers in ``scripts/`` and ``tests/`` listed in
    ``platform_design/MODULE-MAP.md``, and drop the module from the expectations in the boundary test.
"""

from __future__ import annotations
