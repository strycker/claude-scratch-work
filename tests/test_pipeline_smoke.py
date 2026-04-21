"""Smoke tests for trading_crab.pipeline.

Covers: build_parser(), STEPS registry, step function signatures, and
main() dispatch (all step functions mocked — no data I/O).
"""

from __future__ import annotations

import argparse
import sys
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# build_parser()
# ---------------------------------------------------------------------------

def test_build_parser_returns_argument_parser():
    from trading_crab.pipeline import build_parser
    assert isinstance(build_parser(), argparse.ArgumentParser)


def test_build_parser_default_args():
    from trading_crab.pipeline import build_parser
    args = build_parser().parse_args([])
    assert args.refresh is False
    assert args.recompute is False
    assert args.plots is False
    assert args.verbose is False
    assert args.steps is None
    assert args.market_code is None


def test_build_parser_steps_flag():
    from trading_crab.pipeline import build_parser
    args = build_parser().parse_args(["--steps", "1,3,5"])
    assert args.steps == "1,3,5"


def test_build_parser_boolean_flags():
    from trading_crab.pipeline import build_parser
    args = build_parser().parse_args(["--refresh", "--recompute", "--plots", "--verbose"])
    assert args.refresh is True
    assert args.recompute is True
    assert args.plots is True
    assert args.verbose is True


def test_build_parser_market_code_flag():
    from trading_crab.pipeline import build_parser
    args = build_parser().parse_args(["--market-code", "grok"])
    assert args.market_code == "grok"


def test_build_parser_is_deterministic():
    """Calling build_parser() twice registers the same set of arguments."""
    from trading_crab.pipeline import build_parser
    dests1 = {a.dest for a in build_parser()._actions}
    dests2 = {a.dest for a in build_parser()._actions}
    assert dests1 == dests2


# ---------------------------------------------------------------------------
# STEPS registry
# ---------------------------------------------------------------------------

def test_steps_registry_has_nine_entries():
    from trading_crab.pipeline import STEPS
    assert set(STEPS.keys()) == {1, 2, 3, 4, 5, 6, 7, 8, 9}


def test_steps_registry_entries_are_callable():
    from trading_crab.pipeline import STEPS
    for step_num, (description, fn) in STEPS.items():
        assert isinstance(description, str), f"Step {step_num} description is not a str"
        assert callable(fn), f"Step {step_num} function is not callable"


# ---------------------------------------------------------------------------
# Individual step function signatures
# ---------------------------------------------------------------------------

def test_step3_cluster_save_market_code_defaults_false():
    import inspect

    from trading_crab.pipeline import step3_cluster
    sig = inspect.signature(step3_cluster)
    assert "save_market_code" in sig.parameters
    assert sig.parameters["save_market_code"].default is False


# ---------------------------------------------------------------------------
# main() dispatch — all step functions mocked
# ---------------------------------------------------------------------------

def _make_step_mocks() -> dict[int, MagicMock]:
    """Return a STEPS-shaped dict mapping step number → MagicMock."""
    import trading_crab.pipeline as _pl
    return {
        k: MagicMock(wraps=None)
        for k in _pl.STEPS
    }


def _patched_steps(mocks: dict[int, MagicMock]):
    """Context manager: replace STEPS values with mocks for the duration."""
    from contextlib import contextmanager

    import trading_crab.pipeline as _pl

    @contextmanager
    def _ctx():
        original = dict(_pl.STEPS)
        for k, m in mocks.items():
            desc, _ = original[k]
            _pl.STEPS[k] = (desc, m)
        try:
            yield
        finally:
            _pl.STEPS.update(original)

    return _ctx()


def test_main_runs_all_steps_by_default(monkeypatch):
    """main() with no --steps flag dispatches all 9 steps exactly once."""
    monkeypatch.setattr(sys, "argv", ["tradingcrab"])
    mocks = _make_step_mocks()
    with _patched_steps(mocks):
        from trading_crab.pipeline import main
        main()
    for step_num, m in mocks.items():
        assert m.called, f"step {step_num} was not called"


def test_main_runs_only_requested_steps(monkeypatch):
    """main() with --steps 3 only calls step3_cluster."""
    monkeypatch.setattr(sys, "argv", ["tradingcrab", "--steps", "3"])
    mocks = _make_step_mocks()
    with _patched_steps(mocks):
        from trading_crab.pipeline import main
        main()
    assert mocks[3].called
    for k, m in mocks.items():
        if k != 3:
            assert not m.called, f"step {k} should not have been called"


def test_main_runs_multiple_requested_steps(monkeypatch):
    """main() with --steps 1,3 calls only step1 and step3."""
    monkeypatch.setattr(sys, "argv", ["tradingcrab", "--steps", "1,3"])
    mocks = _make_step_mocks()
    with _patched_steps(mocks):
        from trading_crab.pipeline import main
        main()
    assert mocks[1].called
    assert mocks[3].called
    for k, m in mocks.items():
        if k not in (1, 3):
            assert not m.called, f"step {k} should not have been called"
