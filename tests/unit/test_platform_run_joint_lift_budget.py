"""CR-06: ``scripts/run_joint_lift.py`` checks the declared ADR-0004 ceiling BEFORE any append.

The registry is append-only (ADR-0004 §2): a row written past the ceiling cannot be removed,
so the check must precede every write, must read a DECLARED ceiling rather than a hard-coded
one, and must survive ``python -O``.

**No test here touches the real ledger.** Every test fakes ``total_trial_count``,
``run_joint_backtest`` and ``build_inputs``; a module-scoped fixture asserts the real
``registry/trials.jsonl`` sha256 is unchanged across the whole module.
"""

from __future__ import annotations

import ast
import hashlib
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _ROOT / "scripts"
_SCRIPT = _SCRIPTS / "run_joint_lift.py"
_LEDGER = _ROOT / "registry" / "trials.jsonl"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import run_joint_lift as R  # noqa: E402 — needs the sys.path line above


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module", autouse=True)
def _real_ledger_untouched():
    """The real ledger is byte-identical before and after every test in this module."""
    before = _sha(_LEDGER)
    yield
    assert _sha(_LEDGER) == before, "a run_joint_lift test touched registry/trials.jsonl"


class _Sentinel(Exception):
    """Raised by the fake first leg: reaching it proves the pre-flight passed."""


class _Fakes:
    def __init__(self, count: int, *, bump: bool = True, sentinel: bool = False) -> None:
        self.count = count
        self.bump = bump
        self.sentinel = sentinel
        self.legs: list[dict[str, Any]] = []
        self.builds = 0

    def total_trial_count(self) -> int:
        return self.count

    def build_inputs(self, cfg: dict[str, Any]) -> dict[str, Any]:
        self.builds += 1
        return {
            key: None
            for key in ("features_1", "features_2", "asset_returns", "cash_returns", "frozen_1", "frozen_2")
        }

    def run_joint_backtest(self, *args: Any, **kwargs: Any):
        self.legs.append(kwargs)
        if self.sentinel:
            raise _Sentinel("first leg reached")
        if self.bump and kwargs.get("registry_path") is not R.NO_REGISTRY:
            self.count += 1  # one append_trial per leg
        return None, {}


@pytest.fixture
def fake(monkeypatch):
    def _install(count: int, **kw: Any) -> _Fakes:
        f = _Fakes(count, **kw)
        monkeypatch.setattr(R, "total_trial_count", f.total_trial_count)
        monkeypatch.setattr(R, "build_inputs", f.build_inputs)
        monkeypatch.setattr(R, "run_joint_backtest", f.run_joint_backtest)
        monkeypatch.setattr(R, "load_platform_config", lambda: {})
        monkeypatch.setattr(R, "classifier2_config", lambda cfg: {"K": 3, "lam": 1.0})
        monkeypatch.setattr(R, "registry_sharpe_variance", lambda: 1.0)
        return f

    return _install


def test_a_run_that_would_exceed_the_ceiling_is_refused_before_any_append(fake):
    f = fake(44)
    with pytest.raises(RuntimeError, match="ADR-0004"):
        R.run("l1", dry_run=False, declared_ceiling=44)
    assert len(f.legs) == 0
    assert f.builds == 0


def test_the_head_signature_shape_is_refused_before_any_append(fake):
    """HEAD's call shape (no ceiling) at count 44 — the exact CR-06 sequence on HEAD."""
    f = fake(44)
    with pytest.raises(Exception) as exc:
        R.run("l1", dry_run=False)
    assert len(f.legs) == 0 and f.builds == 0, (
        f"CR-06 sequence: {f.builds} input build(s) and {len(f.legs)} leg call(s) "
        f"(fake count now {f.count}) BEFORE {exc.type.__name__}: {exc.value}"
    )
    assert exc.type in (ValueError, RuntimeError)


def test_a_decision_bearing_run_without_a_declared_ceiling_is_refused_before_any_work(fake):
    f = fake(10)
    with pytest.raises(ValueError, match="ADR-0004") as exc:
        R.run("l1", dry_run=False)
    assert "--declared-ceiling" in str(exc.value)
    assert len(f.legs) == 0
    assert f.builds == 0


def test_exactly_at_the_ceiling_proceeds(fake):
    f = fake(42, sentinel=True)
    with pytest.raises(_Sentinel):
        R.run("l1", dry_run=False, declared_ceiling=44)  # 42 + 2 == 44: inclusive
    assert f.builds == 1
    assert len(f.legs) == 1
    assert f.legs[0]["registry_path"] is None  # decision-bearing: the real (faked) ledger


@pytest.mark.parametrize(("routing", "dry_run"), [("l2", False), ("l1", True)])
def test_observational_and_dry_runs_plan_zero_rows_and_need_no_ceiling(fake, routing, dry_run):
    f = fake(44, sentinel=True)
    with pytest.raises(_Sentinel):
        R.run(routing, dry_run=dry_run)
    assert len(f.legs) == 1
    assert f.legs[0]["registry_path"] is R.NO_REGISTRY


def test_a_rows_added_mismatch_raises_explicitly(fake):
    fake(10, bump=False)
    with pytest.raises(RuntimeError, match="accounting mismatch"):
        R.run("l1", dry_run=False, declared_ceiling=50)


def test_no_assert_statement_guards_the_script():
    tree = ast.parse(_SCRIPT.read_text(encoding="utf-8"))
    asserts = [node.lineno for node in ast.walk(tree) if isinstance(node, ast.Assert)]
    assert asserts == [], f"assert statements (stripped by python -O) at lines {asserts}"


def test_the_preflight_guard_survives_python_O():
    code = (
        "import sys; sys.path.insert(0, 'scripts'); import run_joint_lift as R; "
        "R.preflight_trial_budget(count_before=44, planned_rows=2, declared_ceiling=44)"
    )
    proc = subprocess.run(
        [sys.executable, "-O", "-c", code], cwd=_ROOT, capture_output=True, text=True, timeout=300
    )
    assert proc.returncode != 0, "the pre-flight passed under python -O"
    assert "RuntimeError" in proc.stderr, proc.stderr[-2000:]


def test_the_stale_constant_is_gone_and_the_record_names_its_ceiling():
    assert not hasattr(R, "ADR_0002_CEILING")
    block = R._registry_block(
        count_before=40, read_before_at="t0", count_after=42, read_after_at="t1",
        rows_added=2, planned_rows=2, declared_ceiling=46, decision_bearing=True,
    )
    assert block["declared_ceiling"] == 46
    assert "ADR-0004" in block["declared_ceiling_source"]
    assert block["ceiling_respected"] is True
    assert "adr_0002_ceiling" not in block

    observational = R._registry_block(
        count_before=44, read_before_at="t0", count_after=44, read_after_at="t1",
        rows_added=0, planned_rows=0, declared_ceiling=None, decision_bearing=False,
    )
    assert observational["declared_ceiling"] is None
    assert observational["declared_ceiling_source"] is None
    assert observational["ceiling_respected"] is None
